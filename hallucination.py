# --------- MUST BE FIRST (before torch / transformers imports) ----------
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import re
import json
from typing import List, Union, Tuple, Dict, Any
import torch
from sentence_transformers import SentenceTransformer, util

from src.logging import logger


class HallucinationEvaluator:
    """
    COMPREHENSIVE MEANING-AWARE EVALUATION:
    
    1. Removes query-context anchors so concise answers aren't penalized.
    2. Sanitizes date parts (March 31, 2025) and scale units ('000s, millions) from numeric checks.
    3. Grants 100% groundedness score (1.0) for concise direct extractions and valid detailed paraphrases.
    4. Rescales dense embedding noise floors to avoid domain similarity leakage.
    5. Triggers hard failure (0.0) when core factual values are wrong or absent.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info("Initializing HallucinationEvaluator...")

        self.embedding_model = SentenceTransformer(
            model_name,
            device="cpu"
        )

        self.embedding_model.eval()
        logger.info("HallucinationEvaluator initialized successfully.")

    # ---------------- INTERNAL HELPERS ---------------- #

    def _clean_text(self, s: str) -> str:
        s = "" if s is None else str(s)
        return re.sub(r"\s+", " ", s).strip()

    def _normalize_input(self, text: Union[str, List[str]]) -> List[str]:
        """Ensure input is always a list of non-empty strings."""
        if isinstance(text, list):
            out = [self._clean_text(t) for t in text]
            return [t if t else " " for t in out]
        t = self._clean_text(text)
        return [t if t else " "]

    def _extract_core_concept(self, text: str) -> str:
        text_clean = self._clean_text(text)
        text_lower = text_clean.lower()
        
        if not text_clean:
            return ""

        non_answer_patterns = [
            "not found", "not provided", "not available", "cannot find",
            "i don't know", "unclear", "not mentioned", "no information",
            "not specified", "cannot determine", "not clear", "not stated"
        ]
        for pattern in non_answer_patterns:
            if pattern in text_lower:
                return ""

        filler_words = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "of", "in", "on", "at", "to", "for", "with", "by", "from",
            "what", "which", "who", "when", "where", "why", "how", "as",
            "single", "enterprise", "level", "customer", "relationship",
            "management", "platform", "called", "named", "used", "name", "total", "amount"
        }

        tokens = re.findall(r"\d[\d,]*\.?\d*|[a-zA-Z]+|\+|\-|\*|\/|\=", text_clean)
        meaningful = [t for t in tokens if t.lower() not in filler_words]

        if meaningful:
            return " ".join(meaningful)

        return text_clean

    
    def _extract_key_entities(self, text: str) -> set:
        text_lower = self._clean_text(text).lower()
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they",
            "what", "which", "who", "when", "where", "why", "how", "as", "if",
            # Generic procedural/linking verbs & adverbs to exclude
            "determine", "determined", "determines", "use", "used", "uses", "using",
            "ultimately", "ensure", "ensuring", "ensures", "help", "helps", "helping",
            "make", "makes", "making", "provide", "provides", "providing", "describe",
            "described", "describes", "significantly", "crucial", "various", "different"
        }
        words = re.findall(r"\b[a-z0-9]+\b", text_lower)
        return {w for w in words if w not in stop_words and len(w) > 1}

    def _extract_value_numbers(self, s: str) -> List[float]:
        """Extract true factual values while removing scale units, dates, fiscal years, and structural references."""
        s = "" if s is None else str(s)
        
        # 1. Remove units like '000s, in thousands, millions, etc.
        s_clean = re.sub(r"\b000s?\b", " ", s, flags=re.IGNORECASE)
        s_clean = re.sub(r"\b(?:in\s+)?(?:thousands?|millions?|billions?|lakhs?|crores?)\b", " ", s_clean, flags=re.IGNORECASE)

        # 2. Remove fiscal year ranges (e.g., 2024-25, 2024-2025, FY2024-25)
        s_clean = re.sub(r"\b(?:FY\s*)?(?:19|20)\d{2}\s*[-/]\s*\d{2,4}\b", " ", s_clean, flags=re.IGNORECASE)
        s_clean = re.sub(r"\bFY\s*\d{2,4}\b", " ", s_clean, flags=re.IGNORECASE)

        # 3. Remove document structural references (e.g., Schedule 16, Note 3, Item 4)
        s_clean = re.sub(r"\b(?:schedule|note|item|clause|page|section|table)\s+\d+\b", " ", s_clean, flags=re.IGNORECASE)

        # 4. Remove date structures (e.g., March 31, 2025 or 31 March 2025)
        months = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        s_clean = re.sub(rf"\b{months}\s+\d{{1,2}},?\s+\d{{4}}\b", " ", s_clean, flags=re.IGNORECASE)
        s_clean = re.sub(rf"\b\d{{1,2}}\s+{months},?\s+\d{{4}}\b", " ", s_clean, flags=re.IGNORECASE)
        s_clean = re.sub(rf"\b{months}\s+\d{{1,2}}\b", " ", s_clean, flags=re.IGNORECASE)

        # 5. Remove standalone 4-digit years (e.g., 1900-2099)
        s_clean = re.sub(r"\b(19|20)\d{2}\b", " ", s_clean)

        # 6. Strip currency symbols, backticks, and punctuation noise
        s_clean = s_clean.replace("`", " ").replace("₹", " ").replace("$", " ").replace("€", " ").replace("£", " ")
        nums = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b", s_clean)
        
        out: List[float] = []
        for n in nums:
            try:
                out.append(float(n.replace(",", "")))
            except Exception:
                continue
        return out

    def _extract_factual_anchors(self, text: str) -> Dict[str, Any]:
        """Extract key factual anchors: monetary values, date months, and calendar days."""
        s = "" if text is None else str(text)
        s_lower = s.lower()

        month_pattern = r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b"
        months_found = set(re.findall(month_pattern, s_lower))

        nums = self._extract_value_numbers(s)

        day_nums = set(re.findall(rf"{month_pattern}\s+(\d{{1,2}})\b", s_lower))
        day_nums_rev = set(re.findall(rf"\b(\d{{1,2}})\s+{month_pattern}", s_lower))
        all_days = day_nums.union(day_nums_rev)

        return {
            "months": months_found,
            "numbers": nums,
            "days": all_days
        }

    def _compute_anchor_match_ratio(self, answer_text: str, gt_text: str, query_text: str = "") -> Tuple[float, bool]:
        """Calculates factual accuracy across numerical figures, date months, and days (excluding query context)."""
        gt_anchors = self._extract_factual_anchors(gt_text)
        gen_anchors = self._extract_factual_anchors(answer_text)
        query_anchors = self._extract_factual_anchors(query_text) if query_text else {"months": set(), "days": set(), "numbers": []}

        target_months = gt_anchors["months"] - query_anchors["months"]
        target_days = gt_anchors["days"] - query_anchors["days"]
        
        target_numbers = []
        tol = 1e-4
        for g in gt_anchors["numbers"]:
            is_query_num = any(abs(g - q) <= tol for q in query_anchors["numbers"])
            if not is_query_num:
                target_numbers.append(g)

        has_anchors = False
        anchor_scores = []

        if target_months:
            has_anchors = True
            matched_months = len(target_months & gen_anchors["months"])
            anchor_scores.append(matched_months / len(target_months))

        if target_days:
            has_anchors = True
            matched_days = len(target_days & gen_anchors["days"])
            anchor_scores.append(matched_days / len(target_days))

        a_nums = gen_anchors["numbers"]
        g_nums = target_numbers if target_numbers else gt_anchors["numbers"]
        if g_nums:
            has_anchors = True
            if not a_nums:
                num_score = 0.0
            else:
                matched_count = 0
                a_nums_copy = list(a_nums)
                for g in g_nums:
                    for i, a in enumerate(a_nums_copy):
                        if abs(a - g) <= tol:
                            matched_count += 1
                            a_nums_copy.pop(i)
                            break
                num_score = matched_count / len(g_nums)
            anchor_scores.append(num_score)

        if not has_anchors:
            return 1.0, False

        overall_ratio = sum(anchor_scores) / len(anchor_scores)
        return overall_ratio, True

    def _rescale_similarity(self, sim: float, floor: float = 0.20) -> float:
        """Removes baseline domain embedding noise while preserving high semantic match scores."""
        if sim <= floor:
            return 0.0
        return max(0.0, min(1.0, (sim - floor) / (1.0 - floor)))

    def _is_non_answer(self, text: str) -> bool:
        text_lower = text.lower().strip()
        non_answer_patterns = [
            "not found", "not provided", "not available", "cannot find",
            "i don't know", "unclear", "not mentioned", "no information",
            "not specified", "cannot determine", "not clear", "not stated"
        ]
        return any(pattern in text_lower for pattern in non_answer_patterns)

    @torch.inference_mode()
    def _encode(self, text: Union[str, List[str]]):
        text_list = self._normalize_input(text)
        return self.embedding_model.encode(
            text_list,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        
        
    
    def compute_groundedness_and_hallucination(
        self,
        rag_response: str,
        rag_context: Union[str, List[str]] = None,
        ground_truth: Union[str, List[str]] = None,
        query: str = "",
        temperature: float = 0.0,
        top_k_groundedness: int = 3,
    ) -> Tuple[float, float]:

        logger.info("=" * 80)
        logger.info("COMPUTING GROUNDEDNESS WITH NUMERICAL & DOMAIN GUARDRAILS")
        logger.info("=" * 80)

        gen_text = self._clean_text(rag_response)
        gt_text = " ".join(self._normalize_input(ground_truth)).strip()
        query_text = self._clean_text(query)

        if not gen_text or not gt_text:
            return 0.0, 1.0

        # ===== GUARDRAIL 1: NON-ANSWER DETECTION =====
        if self._is_non_answer(gen_text) and gt_text.strip() and not self._is_non_answer(gt_text):
            logger.warning("[CRITICAL] Non-answer detected while Ground Truth has valid information!")
            return 0.0, 1.0

        # ===== STEP 1: EXTRACT ANCHORS & NUMERIC TARGETS =====
        gt_anchors = self._extract_factual_anchors(gt_text)
        gen_anchors = self._extract_factual_anchors(gen_text)
        query_anchors = self._extract_factual_anchors(query_text) if query_text else {"numbers": []}

        tol = 1e-4
        target_numbers = [
            g for g in gt_anchors["numbers"] 
            if not any(abs(g - q) <= tol for q in query_anchors["numbers"])
        ]
        gen_target_numbers = [
            a for a in gen_anchors["numbers"] 
            if not any(abs(a - q) <= tol for q in query_anchors["numbers"])
        ]

        # ===== GUARDRAIL 2: DIRECT NUMERICAL CONTRADICTION HARD FAIL =====
        if target_numbers and gen_target_numbers:
            has_num_match = any(
                any(abs(a - g) <= tol for g in target_numbers) 
                for a in gen_target_numbers
            )
            if not has_num_match:
                logger.warning(
                    f"[CRITICAL CONTRADICTION] Generated number {gen_target_numbers} "
                    f"does not match Ground Truth number {target_numbers}! Hard failure."
                )
                return 0.0, 1.0

        anchor_match_ratio, has_gt_anchors = self._compute_anchor_match_ratio(gen_text, gt_text, query_text)

        # ===== GUARDRAIL 3: MISSING FACTUAL ANCHORS CHECK =====
        if has_gt_anchors and anchor_match_ratio == 0.0:
            logger.warning("[CRITICAL] Target factual anchor is wrong or missing! Hard failure.")
            return 0.0, 1.0

        # ===== GUARDRAIL 4: DIRECT EXTRACTION SUBSTRING OVERRIDE =====
        gen_norm = re.sub(r"[^\w\s]", "", gen_text.lower()).strip()
        gt_norm = re.sub(r"[^\w\s]", "", gt_text.lower()).strip()
        if gen_norm and gt_norm and not self._is_non_answer(gen_text):
            if gen_norm in gt_norm or gt_norm in gen_norm:
                a_nums = self._extract_value_numbers(gen_text)
                g_nums = self._extract_value_numbers(gt_text)
                all_match = all(any(abs(a - g) <= tol for g in g_nums) for a in a_nums) if a_nums else True
                if all_match:
                    logger.info("[PERFECT MATCH] Direct text extraction match detected.")
                    return 1.0, 0.0

        # ===== GUARDRAIL 5: CORE DOMAIN ENTITY RECALL OVERRIDE =====
        gen_entities = self._extract_key_entities(gen_text)
        gt_entities = self._extract_key_entities(gt_text)
        gt_entity_recall = (
            len(gen_entities & gt_entities) / len(gt_entities)
            if gt_entities else 1.0
        )

        # If 100% of Ground Truth domain entities are present in response and anchors match
        if gt_entity_recall == 1.0 and (not has_gt_anchors or anchor_match_ratio == 1.0):
            logger.info("[PERFECT MATCH] All Ground Truth domain entities fully covered.")
            return 1.0, 0.0

        # ===== STEP 2: EMBEDDING SIMILARITY CALCULATIONS =====
        gen_embed = self._encode(gen_text)
        gt_embed = self._encode(gt_text)
        raw_full_sim = float(util.pytorch_cos_sim(gen_embed, gt_embed).item())

        gen_core = self._extract_core_concept(gen_text)
        gt_core = self._extract_core_concept(gt_text)

        full_similarity = self._rescale_similarity(raw_full_sim, floor=0.20)

        if gen_core and gt_core:
            gen_core_embed = self._encode(gen_core)
            gt_core_embed = self._encode(gt_core)
            raw_core_sim = float(util.pytorch_cos_sim(gen_core_embed, gt_core_embed).item())
            core_similarity = self._rescale_similarity(raw_core_sim, floor=0.20)
        else:
            core_similarity = full_similarity

        # ===== STEP 3: COMBINE METRICS =====
        if has_gt_anchors:
            combined_similarity = (
                0.15 * full_similarity +
                0.35 * core_similarity +
                0.50 * anchor_match_ratio
            )
        else:
            combined_similarity = (
                0.40 * full_similarity +
                0.40 * core_similarity +
                0.20 * gt_entity_recall
            )

        groundedness_score = round(max(0.0, min(combined_similarity, 1.0)), 4)
        hallucination_rate = round(max(0.0, min(1.0 - groundedness_score, 1.0)), 4)

        return groundedness_score, hallucination_rate
    
    
    # -------- EXPORT (UI JSON UNCHANGED) -------- #

    def plot_results(
        self,
        groundedness_scores: List[float],
        hallucination_rates: List[float],
        filename: str = "results/hallucination_combined.json",
    ):
        logger.info("Saving groundedness vs hallucination rate results...")

        if len(groundedness_scores) != len(hallucination_rates):
            raise ValueError("Mismatched groundedness and hallucination lengths")

        scatter_plot = [
            {"query_index": i + 1, "groundedness_score": float(g), "hallucination_rate": float(h)}
            for i, (g, h) in enumerate(zip(groundedness_scores, hallucination_rates))
        ]

        bar_chart = {
            "labels": [f"Query {i + 1}" for i in range(len(groundedness_scores))],
            "groundedness_values": [float(x) for x in groundedness_scores],
            "hallucination_values": [float(x) for x in hallucination_rates],
        }

        data = {"scatter_plot": scatter_plot, "bar_chart": bar_chart}

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logger.info(f"Results saved to {filename}")
        return data

    def visualize_bar_chart(self, bar_chart: dict):
        """Keep matplotlib import local so FastAPI workers avoid GUI imports."""
        import matplotlib.pyplot as plt

        labels = bar_chart["labels"]
        g_scores = bar_chart["groundedness_values"]
        h_scores = bar_chart["hallucination_values"]

        x = range(len(labels))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x, g_scores, width, label="Groundedness")
        plt.bar([i + width for i in x], h_scores, width, label="Hallucination Rate")

        plt.xlabel("Queries")
        plt.ylabel("Scores")
        plt.title("Groundedness vs Hallucination Rate")
        plt.xticks([i + width / 2 for i in x], labels, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
