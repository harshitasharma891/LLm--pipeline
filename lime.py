import os
# IMPORTANT: set env BEFORE importing transformers/sentence_transformers
os.environ["CUDA_VISIBLE_DEVICES"] = ""                 # force CPU
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from sentence_transformers import SentenceTransformer, util
from src.logging import logger


class LIMEHighlighter:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

        # Force a NON-meta load on CPU (avoids accelerate/meta init issues)
        # model_kwargs/tokenizer_kwargs are supported in recent sentence-transformers.
        try:
            self.model = SentenceTransformer(
                model_name,
                device="cpu",
                model_kwargs={"device_map": None, "low_cpu_mem_usage": False},
                tokenizer_kwargs={"use_fast": True},
            )
        except TypeError:
            # Fallback for older sentence-transformers versions that don't accept kwargs
            self.model = SentenceTransformer(model_name, device="cpu")

        self.model.eval()

        # Guard: if anything is still on meta, fail early with a clear message
        try:
            any_meta = any(getattr(p, "is_meta", False) for p in self.model.parameters())
            if any_meta:
                raise RuntimeError(
                    "Model parameters are on 'meta' device after load. "
                    "Ensure device_map is disabled and low_cpu_mem_usage=False."
                )
        except Exception:
            # Some SentenceTransformer wrappers may not expose parameters cleanly; ignore.
            pass

        self.explainer = LimeTextExplainer(class_names=["NotSimilar", "Similar"])
        # logger.info(f"LIMEHighlighter initialized on CPU with model: {model_name}")

    def _similarity_classifier(self, query_embedding):
        """
        LIME expects classifier(texts) -> np.array shape (n, 2) with class probs.
        We compute cosine similarity(query, text) and return [1-sim, sim].
        """
        def classifier(texts):
            logger.debug(f"Running similarity classifier on {len(texts)} texts")

            # Batch encode for speed
            text_embs = self.model.encode(
                texts,
                convert_to_tensor=True,
                batch_size=32,
                show_progress_bar=False,
            )

            sims = util.cos_sim(query_embedding, text_embs).squeeze(0)  # shape: (n,)
            sims = torch.clamp(sims, 0.001, 0.999).detach().cpu().numpy()

            probs = np.stack([1.0 - sims, sims], axis=1)  # (n,2)
            return probs

        return classifier

    def explain(self, query: str, response: str, num_features: int = 6, num_samples: int = 500):
        logger.info(
            f"Generating LIME explanation for query (len={len(query)}) and response (len={len(response)})"
        )
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        explanation = self.explainer.explain_instance(
            text_instance=response,
            classifier_fn=self._similarity_classifier(query_embedding),
            num_features=num_features,
            num_samples=num_samples,
        )
        logger.info(f"LIME explanation generated with {num_features} features")
        return explanation

    def get_highlighted_html(self, response: str, explanation):
        logger.debug("Generating highlighted HTML for LIME explanation")
        word_weights = dict(explanation.as_list())

        highlighted_response = response
        for word, weight in word_weights.items():
            color = "#FFA500" if weight > 0 else "#1E90FF"
            highlighted_response = re.sub(
                fr"\b{re.escape(word)}\b",
                f"<span style='background-color:{color};color:black;padding:1px;'>{word}</span>",
                highlighted_response,
                flags=re.IGNORECASE,
            )

        logger.debug("Highlighted HTML generated")
        return (
            "<div style='background-color:#222;color:white;padding:10px;border-radius:10px;'>"
            "<strong>Text with highlighted words</strong><br><br>"
            f"{highlighted_response}</div>"
        )

    def get_feature_weights(self, explanation):
        logger.debug("Extracting feature weights from LIME explanation")
        seen = {}
        for word, weight in explanation.as_list():
            key = word.lower()
            if key not in seen:
                seen[key] = {
                    "word": word,
                    "weight": round(float(weight), 4),
                    "polarity": "positive" if weight > 0 else "negative",
                }
        features = list(seen.values())
        logger.debug(f"{len(features)} unique features extracted")
        return features

    def get_word_positions(self, response: str, explanation):
        logger.debug("Extracting word positions from response for LIME explanation")
        coordinates = []
        for word, weight in explanation.as_list():
            for match in re.finditer(rf"\b{re.escape(word)}\b", response, re.IGNORECASE):
                coordinates.append(
                    {
                        "word": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "weight": round(float(weight), 4),
                        "polarity": "positive" if weight > 0 else "negative",
                    }
                )
        logger.debug(f"{len(coordinates)} word positions extracted")
        return coordinates

    def get_bar_chart_data(self, explanation):
        logger.debug("Generating bar chart data from LIME explanation")
        features = []
        seen = set()
        for word, weight in explanation.as_list():
            key = word.lower()
            if key not in seen:
                features.append(
                    {
                        "feature": word,
                        "weight": round(float(weight), 4),
                        "color": "#FFA500" if weight > 0 else "#1E90FF",
                    }
                )
                seen.add(key)
        logger.debug(f"{len(features)} bar chart features prepared")
        return features


