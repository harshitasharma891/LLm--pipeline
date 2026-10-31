import os
import numpy as np
import pandas as pd
import shap
import matplotlib

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from typing import List, Dict, Optional
import json
from pathlib import Path
import re
import base64
from io import BytesIO

from docx import Document as DocxDocument
from odf import teletype
from odf.opendocument import load as load_odt
import html2text
from src.logging import logger

import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

class ShapRetrieverExplainer:
    def __init__(
        self,
        chain,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
        output_dir: str = "results",
        relevance_threshold: float = 0.3,
    ):
        self.chain = chain
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.output_dir = output_dir
        self.relevance_threshold = relevance_threshold

        os.makedirs(self.output_dir, exist_ok=True)

        self.documents = {}
        self.embeddings = {}
        self.chunk_metadata = {}
        self.explainer = None

    # =====================================================
    # SAFE EMBEDDING WRAPPERS
    # =====================================================
    def _encode_query(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        try:
            return self.model.encode_query(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as e:
            logger.warning(f"encode_query failed, falling back to encode(): {e}")
            return self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

    def _encode_document(self, texts, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]

        try:
            return self.model.encode_document(
                texts,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
            )
        except Exception as e:
            logger.warning(f"encode_document failed, falling back to encode(): {e}")
            return self.model.encode(
                texts,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
            )

    # =====================================================
    # MULTI-FORMAT TEXT EXTRACTION
    # =====================================================
    def extract_text(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()

        try:
            if ext == ".pdf":
                return self.extract_text_from_pdf(file_path)
            elif ext in [".txt", ".md"]:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            elif ext == ".docx":
                return self.extract_text_from_docx(file_path)
            elif ext == ".odt":
                return self.extract_text_from_odt(file_path)
            elif ext == ".rtf":
                return self.extract_text_from_rtf(file_path)
            elif ext in [".html", ".htm"]:
                return self.extract_text_from_html(file_path)
            else:
                logger.warning(f"Unsupported format: {ext}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""

    # def extract_text_from_pdf(self, pdf_path: str) -> str:
    #     try:
    #         with open(pdf_path, "rb") as file:
    #             pdf_reader = PyPDF2.PdfReader(file)
    #             text = ""
    #             for page in pdf_reader.pages:
    #                 page_text = page.extract_text() or ""
    #                 text += page_text + "\n"
    #             return text
    #     except Exception as e:
    #         logger.error(f"Error reading PDF {pdf_path}: {e}")
    #         return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            doc = DocxDocument(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""

    def extract_text_from_odt(self, file_path: str) -> str:
        try:
            odt_doc = load_odt(file_path)
            return teletype.extractText(odt_doc)
        except Exception as e:
            logger.error(f"Error reading ODT {file_path}: {e}")
            return ""

    def extract_text_from_rtf(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read()
            data = re.sub(r"{\\[^}]+}", "", data)
            data = re.sub(r"\\[a-z]+\d* ?", "", data)
            return data
        except Exception as e:
            logger.error(f"Error reading RTF {file_path}: {e}")
            return ""

    def extract_text_from_html(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_data = f.read()
            return html2text.html2text(html_data)
        except Exception as e:
            logger.error(f"Error reading HTML {file_path}: {e}")
            return ""

    # =====================================================
    # CHUNKING
    # =====================================================
    def chunk_text(self, text: str, source: str) -> List[Dict]:
        text = re.sub(r"\s+", " ", text).strip()

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                sentence_end = text.rfind(".", start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1

            current_chunk = text[start:end].strip()

            if current_chunk:
                chunks.append(
                    {
                        "text": current_chunk,
                        "source": source,
                        "chunk_id": chunk_id,
                        "start_pos": start,
                        "end_pos": end,
                        "length": len(current_chunk),
                    }
                )
                chunk_id += 1

            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks

    # =====================================================
    # PROCESS DOCUMENTS
    # =====================================================
    def process_documents(
        self,
        dataset_folder: Optional[str] = None,
        document_path: Optional[str] = None,
        dataset_df: Optional[pd.DataFrame] = None,
    ):
        inputs_provided = sum(
            x is not None for x in [dataset_folder, document_path, dataset_df]
        )
        if inputs_provided != 1:
            raise ValueError(
                "Provide exactly ONE of: dataset_folder, document_path, or dataset_df"
            )

        supported_ext = {
            ".pdf", ".txt", ".md", ".docx", ".odt", ".rtf", ".html", ".htm"
        }

        if dataset_df is not None:
            if dataset_df.empty:
                raise ValueError("Dataset API DataFrame is empty")
            if "chunk" not in dataset_df.columns:
                raise ValueError("Dataset API DataFrame must contain a 'chunk' column")

            logger.info(f"Processing {len(dataset_df)} chunks from Dataset API")

            file_name = "api_dataset"
            texts = dataset_df["chunk"].fillna("").astype(str).tolist()

            chunks = []
            for i, text in enumerate(texts):
                chunks.append(
                    {
                        "text": text,
                        "source": dataset_df.iloc[i].get("source", "api_dataset"),
                        "chunk_id": i,
                        "start_pos": 0,
                        "end_pos": len(text),
                        "length": len(text),
                    }
                )

            self.documents[file_name] = texts
            self.chunk_metadata[file_name] = chunks

            logger.info("Generating embeddings for Dataset API chunks")
            self.embeddings[file_name] = self._encode_document(
                texts,
                show_progress_bar=False,
            )

            logger.info("Processed Dataset API")
            return

        if document_path:
            if not os.path.isfile(document_path):
                raise FileNotFoundError(f"Document not found: {document_path}")

            ext = Path(document_path).suffix.lower()
            if ext not in supported_ext:
                raise ValueError(f"Unsupported file type: {ext}")

            files = [document_path]
        else:
            if not os.path.isdir(dataset_folder):
                raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")

            files = [
                os.path.join(dataset_folder, f)
                for f in os.listdir(dataset_folder)
                if Path(f).suffix.lower() in supported_ext
            ]

        logger.info(f"Processing {len(files)} document(s) from filesystem")

        for file_path in files:
            file = os.path.basename(file_path)
            file_name = os.path.splitext(file)[0]

            logger.info(f"Processing: {file}")

            text = self.extract_text(file_path)
            if not text.strip():
                logger.warning(f"No text extracted from: {file}")
                continue

            chunks = self.chunk_text(text, file_name)
            if not chunks:
                logger.warning(f"No chunks created for: {file}")
                continue

            chunk_texts = [c["text"] for c in chunks]

            self.documents[file_name] = chunk_texts
            self.chunk_metadata[file_name] = chunks

            logger.info(f"Generating embeddings ({len(chunk_texts)} chunks)...")
            self.embeddings[file_name] = self._encode_document(
                chunk_texts,
                show_progress_bar=True,
            )

            logger.info(f"Processed {file}: {len(chunks)} chunks")

    def ask_question(self, query):
        answer = self.chain.invoke({"input": query})
        full_answer = answer.get("answer", "")
        response = full_answer.split("Answer:")[-1].strip()

        query_response = {
            "query": query,
            "response": response,
        }

        logger.info("Query response received.")
        return query_response

    def _check_query_relevance(self, query: str, top_scores: List[float]) -> bool:
        if not top_scores:
            return False

        max_score = max(top_scores)
        avg_score = sum(top_scores) / len(top_scores)
        is_relevant = max_score >= self.relevance_threshold

        logger.info("Relevance Check:")
        logger.info(f"Max similarity score: {max_score:.4f}")
        logger.info(f"Average similarity score: {avg_score:.4f}")
        logger.info(f"Threshold: {self.relevance_threshold}")
        logger.info(f"Query relevant: {'YES' if is_relevant else 'NO'}")

        return is_relevant

    def retrieve_and_explain(self, query: str, use_tfidf_features: bool = False) -> Dict:
        logger.info(f"Query: {query}")

        query_embedding = self._encode_query(query)[0]

        all_chunks = []
        all_scores = []
        all_embeddings = []
        all_metadata = []

        for pdf_name, chunks in self.documents.items():
            embeddings = self.embeddings[pdf_name]
            metadata = self.chunk_metadata[pdf_name]

            similarities = cosine_similarity([query_embedding], embeddings)[0]

            for chunk, score, embed, meta in zip(chunks, similarities, embeddings, metadata):
                all_chunks.append(chunk)
                all_scores.append(float(score))
                all_embeddings.append(embed)
                all_metadata.append(
                    {
                        **meta,
                        "pdf_name": pdf_name,
                        "similarity_score": float(score),
                    }
                )

        top_indices = np.argsort(all_scores)[-self.top_k:][::-1]
        top_chunks = [all_chunks[i] for i in top_indices]
        top_scores = [all_scores[i] for i in top_indices]
        top_embeddings = np.array([all_embeddings[i] for i in top_indices])
        top_metadata = [all_metadata[i] for i in top_indices]

        is_relevant = self._check_query_relevance(query, top_scores)

        results = {
            "query": str(query),
            "is_relevant": bool(is_relevant),
            "max_similarity_score": float(max(top_scores)) if top_scores else 0.0,
            "top_chunks": [],
        }

        for chunk, score, meta in zip(top_chunks, top_scores, top_metadata):
            results["top_chunks"].append(
                {
                    "text": str(chunk),
                    "score": float(score),
                    "metadata": {
                        "pdf_name": str(meta["pdf_name"]),
                        "chunk_id": int(meta["chunk_id"]),
                        "source": str(meta["source"]),
                        "start_pos": int(meta["start_pos"]),
                        "end_pos": int(meta["end_pos"]),
                        "length": int(meta["length"]),
                        "similarity_score": float(meta["similarity_score"]),
                    },
                }
            )

        if is_relevant:
            try:
                shap_results = self._generate_shap_explanations(
                    query_embedding,
                    top_embeddings,
                    top_chunks,
                    top_metadata,
                    use_tfidf_features,
                )
                results["shap_analysis"] = shap_results
            except Exception as e:
                logger.exception("SHAP explanation failed")
                results["shap_analysis"] = {
                    "method": "failed",
                    "reason": str(e),
                    "plots": {},
                }
        else:
            results["shap_analysis"] = {
                "method": "skipped",
                "reason": f"Score {max(top_scores):.4f} < threshold {self.relevance_threshold}",
                "plots": {},
            }

        self._save_results(results)
        return results

    def _generate_shap_explanations(
        self, query_embedding, top_embeddings, top_chunks, top_metadata, use_tfidf_features
    ):
        if use_tfidf_features:
            return self._shap_with_tfidf(query_embedding, top_chunks, top_metadata)
        return self._shap_with_embeddings(query_embedding, top_embeddings, top_chunks, top_metadata)

    def _shap_with_embeddings(self, query_embedding, top_embeddings, top_chunks, top_metadata):
        def predict_fn(embeddings):
            embeddings = np.asarray(embeddings)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            return cosine_similarity(embeddings, query_embedding.reshape(1, -1)).flatten()

        self.explainer = shap.KernelExplainer(predict_fn, top_embeddings)
        shap_values = self.explainer.shap_values(top_embeddings, nsamples=100)

        feature_names = [f"dim_{i}" for i in range(top_embeddings.shape[1])]
        plots = self._create_shap_plots(shap_values, top_embeddings, feature_names, top_metadata)

        return {
            "method": "embedding_dimensions",
            "shap_values": np.asarray(shap_values).tolist(),
            "feature_names": feature_names,
            "plots": plots,
            "explainer_info": "KernelExplainer with embeddings",
        }

    def _shap_with_tfidf(self, query_embedding, top_chunks, top_metadata):
        all_texts = top_chunks + [" ".join(top_chunks)]
        vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        chunk_vectors = tfidf_matrix[:-1].toarray()
        feature_names = vectorizer.get_feature_names_out().tolist()

        def predict_fn(tfidf_vectors):
            tfidf_vectors = np.asarray(tfidf_vectors)
            if tfidf_vectors.ndim == 1:
                tfidf_vectors = tfidf_vectors.reshape(1, -1)
            return np.array([np.sum(v) for v in tfidf_vectors])

        self.explainer = shap.KernelExplainer(predict_fn, chunk_vectors)
        shap_values = self.explainer.shap_values(chunk_vectors, nsamples=100)

        plots = self._create_shap_plots(shap_values, chunk_vectors, feature_names, top_metadata)

        return {
            "method": "tfidf_features",
            "shap_values": np.asarray(shap_values).tolist(),
            "feature_names": feature_names,
            # "plots": plots,
            "explainer_info": "KernelExplainer with TF-IDF",
        }

    def _create_shap_plots(self, shap_values, feature_matrix, feature_names, metadata):
        plots = {}

        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                feature_matrix,
                feature_names=feature_names,
                show=False,
            )
            plt.title("SHAP Summary Plot (Retrieval Attribution)")
            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
            plt.close("all")
            plots["summary"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error creating summary plot: {e}")

        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = np.array(expected_value).flatten()[0]

        shap_values_arr = np.asarray(shap_values)

        for i in range(len(shap_values_arr)):
            try:
                plt.figure(figsize=(15, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values_arr[i],
                        base_values=expected_value,
                        data=feature_matrix[i],
                        feature_names=feature_names,
                    ),
                    show=False,
                )
                buffer = BytesIO()
                plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                plt.close("all")
                plots[f"waterfall_chunk_{i+1}"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            except Exception as e:
                logger.error(f"Error creating waterfall plot for chunk {i+1}: {e}")

        return plots

    def _save_results(self, results: Dict):
        results_path = os.path.join(self.output_dir, "retrieval_results.json")

        try:
            plots = results.get("shap_analysis", {}).get("plots", {})
            filtered_plots = {}

            if "summary" in plots:
                filtered_plots["summary"] = plots["summary"]

            for k, v in plots.items():
                if k.startswith("waterfall_chunk_"):
                    filtered_plots[k] = v

            filtered_results = {
                "response": results.get("response", ""),
                "plots": filtered_plots,
            }

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(filtered_results, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved results to {results_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def print_results(self, results: Dict):
        print("\n" + "=" * 80)
        print("RETRIEVAL ANALYSIS RESULTS")
        print("=" * 80)

        print(f"\nQuery: {results['query']}")
        print(f"\nRelevance:")
        print(f"Relevant: {'YES' if results['is_relevant'] else 'NO'}")
        print(f"Max Similarity: {results['max_similarity_score']:.4f}")

        print(f"\nTop {len(results['top_chunks'])} Chunks:")
        for i, chunk in enumerate(results["top_chunks"], 1):
            print(f"\nRank {i}")
            print(f"Score: {chunk['score']:.4f}")
            print(f"Source: {chunk['metadata']['pdf_name']}")
            print(f"Text: {chunk['text'][:200]}...")

        shap_info = results.get("shap_analysis", {})
        print(f"\nSHAP Analysis:")
        print(f"Method: {shap_info.get('method', 'N/A')}")
        print(f"Reason: {shap_info.get('reason', 'N/A')}")
        print(f"Plots: {len(shap_info.get('plots', {}))}")

        print("\n" + "=" * 80)


# import os
# import numpy as np
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt
# from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import PyPDF2
# from typing import List, Dict, Tuple, Optional
# import logging
# import json
# from pathlib import Path
# import re
# import base64
# from io import BytesIO

# # ---- NEW IMPORTS FOR MULTI-FORMAT SUPPORT ----
# from docx import Document as DocxDocument
# from odf import text, teletype
# from odf.opendocument import load as load_odt
# import html2text
# from src.logging import logger

# import torch

# # =====================================================
# # MAIN CLASS
# # =====================================================
# class ShapRetrieverExplainer:
#     def __init__(self, 
#                  chain: str,
#                  model_name: str = "all-MiniLM-L6-v2",
#                  chunk_size: int = 500,
#                  chunk_overlap: int = 50,
#                  top_k: int = 3,
#                  output_dir: str = "results",
#                  relevance_threshold: float = 0.3):

#         self.chain = chain
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = SentenceTransformer(model_name, device=device)
#         # self.model = SentenceTransformer(model_name, device="auto")
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.top_k = top_k
#         self.output_dir = output_dir
#         self.relevance_threshold = relevance_threshold

#         os.makedirs(self.output_dir, exist_ok=True)

#         self.documents = {}
#         self.embeddings = {}
#         self.chunk_metadata = {}

#         self.explainer = None

#         # logger.info(f" Initialized PDFRetrievalExplainer with model: {model_name}")
#         # logger.info(f" Relevance threshold set to: {relevance_threshold}")

#     # =====================================================================
#     # MULTI-FORMAT TEXT EXTRACTION
#     # =====================================================================
#     def extract_text(self, file_path: str) -> str:
#         """Extract text from any supported file format."""
#         ext = Path(file_path).suffix.lower()

#         try:
#             if ext == ".pdf":
#                 return self.extract_text_from_pdf(file_path)

#             elif ext in [".txt", ".md"]:
#                 with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#                     return f.read()

#             elif ext == ".docx":
#                 return self.extract_text_from_docx(file_path)

#             elif ext == ".odt":
#                 return self.extract_text_from_odt(file_path)

#             elif ext == ".rtf":
#                 return self.extract_text_from_rtf(file_path)

#             elif ext in [".html", ".htm"]:
#                 return self.extract_text_from_html(file_path)

#             else:
#                 logger.warning(f" Unsupported format: {ext}")
#                 return ""

#         except Exception as e:
#             logger.error(f"❌ Error extracting text from {file_path}: {e}")
#             return ""

#     # =====================================================================
#     # INDIVIDUAL EXTRACTORS
#     # =====================================================================
#     def extract_text_from_pdf(self, pdf_path: str) -> str:
#         try:
#             with open(pdf_path, 'rb') as file:
#                 pdf_reader = PyPDF2.PdfReader(file)
#                 text = ""
#                 for page in pdf_reader.pages:
#                     text += page.extract_text() + "\n"
#                 return text
#         except Exception as e:
#             logger.error(f"Error reading PDF {pdf_path}: {e}")
#             return ""

#     def extract_text_from_docx(self, file_path: str) -> str:
#         try:
#             doc = DocxDocument(file_path)
#             return "\n".join([p.text for p in doc.paragraphs])
#         except:
#             return ""

#     def extract_text_from_odt(self, file_path: str) -> str:
#         try:
#             odt_doc = load_odt(file_path)
#             return teletype.extractText(odt_doc)
#         except:
#             return ""

#     def extract_text_from_rtf(self, file_path: str) -> str:
#         try:
#             with open(file_path, 'r', encoding='utf-8', errors="ignore") as f:
#                 data = f.read()
#             data = re.sub(r'{\\[^}]+}', '', data)     
#             data = re.sub(r'\\[a-z]+\d* ?', '', data) 
#             return data
#         except:
#             return ""

#     def extract_text_from_html(self, file_path: str) -> str:
#         try:
#             with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#                 html_data = f.read()
#             return html2text.html2text(html_data)
#         except:
#             return ""

#     # =====================================================================
#     # ORIGINAL CHUNKING FUNCTION (UNCHANGED)
#     # =====================================================================
#     def chunk_text(self, text: str, source: str) -> List[Dict]:
#         text = re.sub(r'\s+', ' ', text).strip()

#         chunks = []
#         start = 0
#         chunk_id = 0

#         while start < len(text):
#             end = start + self.chunk_size

#             if end < len(text):
#                 sentence_end = text.rfind('.', start, end)
#                 if sentence_end > start + self.chunk_size // 2:
#                     end = sentence_end + 1

#             chunk_text = text[start:end].strip()

#             if chunk_text:
#                 chunks.append({
#                     'text': chunk_text,
#                     'source': source,
#                     'chunk_id': chunk_id,
#                     'start_pos': start,
#                     'end_pos': end,
#                     'length': len(chunk_text)
#                 })
#                 chunk_id += 1

#             start = end - self.chunk_overlap

#         return chunks

#     # =====================================================================
#     # NEW: PROCESS ALL DOCUMENT TYPES (REPLACES process_pdfs)
#     # =====================================================================
#     def process_documents(
#         self,
#         dataset_folder: Optional[str] = None,
#         document_path: Optional[str] = None,
#         dataset_df: Optional[pd.DataFrame] = None,
#     ):
#         """
#         Process documents from:
#         - Dataset API DataFrame (dataset_df)
#         - a directory (dataset_folder)
#         - a single file (document_path)
#         """

#         # --------------------------------------------------
#         # VALIDATION
#         # --------------------------------------------------
#         inputs_provided = sum(
#             x is not None for x in [dataset_folder, document_path, dataset_df]
#         )

#         if inputs_provided != 1:
#             raise ValueError(
#                 "Provide exactly ONE of: dataset_folder, document_path, or dataset_df"
#             )

#         supported_ext = {
#             ".pdf", ".txt", ".md", ".docx", ".odt", ".rtf", ".html", ".htm"
#         }

#         # ==================================================
#         # CASE 1: DATASET API (DataFrame)
#         # ==================================================
#         if dataset_df is not None:
#             if dataset_df.empty:
#                 raise ValueError("Dataset API DataFrame is empty")

#             logger.info(f"Processing {len(dataset_df)} chunks from Dataset API")

#             file_name = "api_dataset"
#             texts = dataset_df["chunk"].astype(str).tolist()

#             chunks = []
#             for i, text in enumerate(texts):
#                 chunks.append({
#                     "text": text,
#                     "source": dataset_df.iloc[i].get("source", "api_dataset"),
#                     "chunk_id": i,
#                     "start_pos": 0,
#                     "end_pos": len(text),
#                     "length": len(text),
#                 })

#             self.documents[file_name] = texts
#             self.chunk_metadata[file_name] = chunks

#             logger.info("Generating embeddings for Dataset API chunks")
#             self.embeddings[file_name] = self.model.encode(
#                 texts, show_progress_bar=False
#             )

#             logger.info("✅ Processed Dataset API")
#             return

#         # ==================================================
#         # CASE 2: SINGLE DOCUMENT
#         # ==================================================
#         if document_path:
#             if not os.path.isfile(document_path):
#                 raise FileNotFoundError(f"Document not found: {document_path}")

#             ext = Path(document_path).suffix.lower()
#             if ext not in supported_ext:
#                 raise ValueError(f"Unsupported file type: {ext}")

#             files = [document_path]

#         # ==================================================
#         # CASE 3: DIRECTORY
#         # ==================================================
#         else:
#             if not os.path.isdir(dataset_folder):
#                 raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")

#             files = [
#                 os.path.join(dataset_folder, f)
#                 for f in os.listdir(dataset_folder)
#                 if Path(f).suffix.lower() in supported_ext
#             ]

#         logger.info(f"Processing {len(files)} document(s) from filesystem")

#         # --------------------------------------------------
#         # FILESYSTEM PROCESSING (UNCHANGED LOGIC)
#         # --------------------------------------------------
#         for file_path in files:
#             file = os.path.basename(file_path)
#             file_name = os.path.splitext(file)[0]

#             logger.info(f"Processing: {file}")

#             text = self.extract_text(file_path)
#             if not text.strip():
#                 logger.warning(f"No text extracted from: {file}")
#                 continue

#             chunks = self.chunk_text(text, file_name)
#             if not chunks:
#                 continue

#             chunk_texts = [c["text"] for c in chunks]

#             self.documents[file_name] = chunk_texts
#             self.chunk_metadata[file_name] = chunks

#             logger.info(f"Generating embeddings ({len(chunk_texts)} chunks)...")
#             self.embeddings[file_name] = self.model.encode(
#                 chunk_texts, show_progress_bar=True
#             )

#             logger.info(f"✅ Processed {file}: {len(chunks)} chunks")


#     # =====================================================================
#     # (ALL REMAINING FUNCTIONS ARE *IDENTICAL* TO YOUR ORIGINAL CODE)
#     # =====================================================================

#     def ask_question(self, query):
#         answer = self.chain.invoke({"input": query})
#         full_answer = answer.get("answer", "")
#         response = full_answer.split("Answer:")[-1].strip()

#         query_response = {
#             "query": query,
#             "response": response
#         }

#         logger.info(" Query response received.")
#         return query_response

#     def _check_query_relevance(self, query: str, top_scores: List[float]) -> bool:
#         if not top_scores:
#             return False

#         max_score = max(top_scores)
#         avg_score = sum(top_scores) / len(top_scores)
#         is_relevant = max_score >= self.relevance_threshold

#         logger.info(f"  Relevance Check:")
#         logger.info(f"   Max similarity score: {max_score:.4f}")
#         logger.info(f"   Average similarity score: {avg_score:.4f}")
#         logger.info(f"   Threshold: {self.relevance_threshold}")
#         logger.info(f"   Query relevant: {'YES' if is_relevant else 'NO'}")

#         return is_relevant

#     # -------------------- RETRIEVAL + SHAP --------------------
#     # (NO MODIFICATIONS – EXACTLY YOUR ORIGINAL CODE)
#     # --------------------

#     def retrieve_and_explain(self, query: str, use_tfidf_features: bool = False) -> Dict:
#         logger.info(f"🔍 Query: {query}")

#         query_embedding = self.model.encode([query])[0]

#         all_chunks = []
#         all_scores = []
#         all_embeddings = []
#         all_metadata = []

#         for pdf_name, chunks in self.documents.items():
#             embeddings = self.embeddings[pdf_name]
#             metadata = self.chunk_metadata[pdf_name]

#             similarities = cosine_similarity([query_embedding], embeddings)[0]

#             for chunk, score, embed, meta in zip(chunks, similarities, embeddings, metadata):
#                 all_chunks.append(chunk)
#                 all_scores.append(float(score))
#                 all_embeddings.append(embed)
#                 all_metadata.append({**meta, "pdf_name": pdf_name, "similarity_score": float(score)})

#         top_indices = np.argsort(all_scores)[-self.top_k:][::-1]
#         top_chunks = [all_chunks[i] for i in top_indices]
#         top_scores = [all_scores[i] for i in top_indices]
#         top_embeddings = np.array([all_embeddings[i] for i in top_indices])
#         top_metadata = [all_metadata[i] for i in top_indices]

#         is_relevant = self._check_query_relevance(query, top_scores)

#         results = {
#             'query': str(query),
#             'is_relevant': bool(is_relevant),
#             'max_similarity_score': float(max(top_scores)) if top_scores else 0.0,
#             'top_chunks': []
#         }

#         for chunk, score, meta in zip(top_chunks, top_scores, top_metadata):
#             results['top_chunks'].append({
#                 'text': str(chunk),
#                 'score': float(score),
#                 'metadata': {
#                     'pdf_name': str(meta['pdf_name']),
#                     'chunk_id': int(meta['chunk_id']),
#                     'source': str(meta['source']),
#                     'start_pos': int(meta['start_pos']),
#                     'end_pos': int(meta['end_pos']),
#                     'length': int(meta['length']),
#                     'similarity_score': float(meta['similarity_score'])
#                 }
#             })

#         if is_relevant:
#             try:
#                 shap_results = self._generate_shap_explanations(
#                     query_embedding,
#                     top_embeddings,
#                     top_chunks,
#                     top_metadata,
#                     use_tfidf_features
#                 )
#                 results['shap_analysis'] = shap_results
#             except Exception as e:
#                 results['shap_analysis'] = {
#                     'method': 'failed',
#                     'reason': str(e),
#                     'plots': {}
#                 }
#         else:
#             results['shap_analysis'] = {
#                 'method': 'skipped',
#                 'reason': f"Score {max(top_scores):.4f} < threshold {self.relevance_threshold}",
#                 'plots': {}
#             }

#         self._save_results(results)
#         return results

#     # -------------------- SHAP FUNCTIONS (UNTOUCHED) --------------------

#     def _generate_shap_explanations(self, query_embedding, top_embeddings, top_chunks, top_metadata, use_tfidf_features):
#         if use_tfidf_features:
#             return self._shap_with_tfidf(query_embedding, top_chunks, top_metadata)
#         else:
#             return self._shap_with_embeddings(query_embedding, top_embeddings, top_chunks, top_metadata)

#     def _shap_with_embeddings(self, query_embedding, top_embeddings, top_chunks, top_metadata):
#         def predict_fn(embeddings):
#             if len(embeddings.shape) == 1:
#                 embeddings = embeddings.reshape(1, -1)
#             return cosine_similarity(embeddings, query_embedding.reshape(1, -1)).flatten()

#         self.explainer = shap.KernelExplainer(predict_fn, top_embeddings)
#         shap_values = self.explainer.shap_values(top_embeddings, nsamples=100)

#         feature_names = [f"dim_{i}" for i in range(top_embeddings.shape[1])]

#         plots = self._create_shap_plots(shap_values, top_embeddings, feature_names, top_metadata)

#         return {
#             'method': 'embedding_dimensions',
#             'shap_values': shap_values.tolist(),
#             'feature_names': feature_names,
#             'plots': plots,
#             'explainer_info': 'KernelExplainer with embeddings'
#         }

#     def _shap_with_tfidf(self, query_embedding, top_chunks, top_metadata):
#         all_texts = top_chunks + [' '.join(top_chunks)]
#         vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
#         tfidf_matrix = vectorizer.fit_transform(all_texts)

#         chunk_vectors = tfidf_matrix[:-1].toarray()
#         feature_names = vectorizer.get_feature_names_out().tolist()

#         def predict_fn(tfidf_vectors):
#             if len(tfidf_vectors.shape) == 1:
#                 tfidf_vectors = tfidf_vectors.reshape(1, -1)
#             return np.array([np.sum(v) for v in tfidf_vectors])

#         self.explainer = shap.KernelExplainer(predict_fn, chunk_vectors)
#         shap_values = self.explainer.shap_values(chunk_vectors, nsamples=100)

#         plots = self._create_shap_plots(shap_values, chunk_vectors, feature_names, top_metadata)

#         return {
#             'method': 'tfidf_features',
#             'shap_values': shap_values.tolist(),
#             'feature_names': feature_names,
#             # 'plots': plots,
#             'explainer_info': 'KernelExplainer with TF-IDF'
#         }

#     # -------------------- SHAP Plotting (UNCHANGED) --------------------

#     def _create_shap_plots(self, shap_values, feature_matrix, feature_names, metadata):
#         plots = {}

#         try:
#             plt.figure(figsize=(10, 8))
#             shap.summary_plot(shap_values, feature_matrix, feature_names=feature_names, show=False)
#             plt.title("SHAP Summary Plot (Retrieval Attribution)")
#             plt.tight_layout()

#             buffer = BytesIO()
#             plt.savefig(buffer, format="png", dpi=300, bbox_inches='tight')
#             plt.close()
#             plots['summary'] = base64.b64encode(buffer.getvalue()).decode("utf-8")

#         except Exception as e:
#             logger.error(f"Error creating summary plot: {e}")

#         for i in range(len(shap_values)):
#             try:
#                 plt.figure(figsize=(15, 6))
#                 shap.waterfall_plot(
#                     shap.Explanation(
#                         values=shap_values[i],
#                         base_values=self.explainer.expected_value,
#                         data=feature_matrix[i],
#                         feature_names=feature_names
#                     ),
#                     show=False
#                 )
#                 buffer = BytesIO()
#                 plt.savefig(buffer, format="png", dpi=300, bbox_inches='tight')
#                 plt.close()
#                 plots[f'waterfall_chunk_{i+1}'] = base64.b64encode(buffer.getvalue()).decode("utf-8")

#             except Exception as e:
#                 logger.error(f"Error creating waterfall plot for chunk {i+1}: {e}")

#         return plots

#     # -------------------- RESULTS SAVING (UNCHANGED) --------------------

#     def _save_results(self, results: Dict):
#         results_path = os.path.join(self.output_dir, "retrieval_results.json")

#         try:
#             plots = results.get("shap_analysis", {}).get("plots", {})
#             filtered_plots = {}

#             if "summary" in plots:
#                 filtered_plots["summary"] = plots["summary"]

#             for k, v in plots.items():
#                 if k.startswith("waterfall_chunk_"):
#                     filtered_plots[k] = v

#             filtered_results = {
#                 "response": results.get("response", ""),
#                 "plots": filtered_plots
#             }

#             with open(results_path, "w", encoding="utf-8") as f:
#                 json.dump(filtered_results, f, indent=2, ensure_ascii=False)

#             logger.info(f" Saved results to {results_path}")

#         except Exception as e:
#             logger.error(f"Error saving results: {e}")

#     # -------------------- PRINT RESULTS (UNCHANGED) --------------------

#     def print_results(self, results: Dict):
#         print("\n" + "="*80)
#         print("🔍 RETRIEVAL ANALYSIS RESULTS")
#         print("="*80)

#         print(f"\n Query: {results['query']}")

#         print(f"\n Relevance:")
#         print(f"   Relevant: {'YES' if results['is_relevant'] else 'NO'}")
#         print(f"   Max Similarity: {results['max_similarity_score']:.4f}")

#         print(f"\n Top {len(results['top_chunks'])} Chunks:")
#         for i, chunk in enumerate(results['top_chunks'], 1):
#             print(f"\n Rank {i}")
#             print(f"Score: {chunk['score']:.4f}")
#             print(f"Source: {chunk['metadata']['pdf_name']}")
#             print(f"Text: {chunk['text'][:200]}...")

#         shap_info = results.get("shap_analysis", {})
#         print(f"\n SHAP Analysis:")
#         print(f"   Method: {shap_info.get('method', 'N/A')}")
#         print(f"   Reason: {shap_info.get('reason', 'N/A')}")
#         print(f"   Plots: {len(shap_info.get('plots', {}))}")

#         print("\n" + "="*80)
