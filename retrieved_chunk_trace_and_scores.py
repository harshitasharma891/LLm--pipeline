from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

try:
    from src.logging import logger
except Exception:  # keeps this module usable in notebooks/local tests too
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class TemporalReasoningHelper:
    """Handles temporal extraction and reasoning (dates, fiscal years, ranges)."""

    FISCAL_YEAR_PATTERN = re.compile(r"\bFY\s*(\d{2,4})(?:\s*[-–—/]\s*(\d{2,4}))?\b", re.IGNORECASE)
    YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
    DATE_PATTERNS = [
        re.compile(r"\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2})\b"),
        re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-]((?:19|20)\d{2})\b"),
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{1,2}),?\s+((?:19|20)\d{2})\b",
            re.IGNORECASE,
        ),
    ]

    @staticmethod
    def _expand_year(year_str: str, base_year: Optional[int] = None) -> int:
        """Expand 2-digit years safely: 24 -> 2024, FY2024-25 -> 2025."""
        year_str = str(year_str).strip()
        if len(year_str) == 2:
            year = 2000 + int(year_str)
            if base_year is not None:
                century = (base_year // 100) * 100
                candidate = century + int(year_str)
                if candidate < base_year:
                    candidate += 100
                year = candidate
            return year
        return int(year_str)

    @staticmethod
    def normalize_fiscal_year(fy_label: str) -> Optional[Dict[str, str]]:
        """Convert FY label to actual date range, e.g. FY2024-25 -> 2024-04-01 to 2025-03-31."""
        if not fy_label:
            return None

        match = TemporalReasoningHelper.FISCAL_YEAR_PATTERN.search(str(fy_label))
        if not match:
            return None

        start_year = TemporalReasoningHelper._expand_year(match.group(1))
        end_part = match.group(2)
        end_year = (
            TemporalReasoningHelper._expand_year(end_part, base_year=start_year)
            if end_part
            else start_year + 1
        )

        if end_year <= start_year:
            end_year = start_year + 1

        return {
            "fiscal_year": fy_label,
            "start_year": str(start_year),
            "end_year": str(end_year),
            "date_range": f"{start_year}-04-01 to {end_year}-03-31",
        }

    @staticmethod
    def extract_temporal_info(text: str) -> Dict[str, Any]:
        """Extract fiscal years, date ranges, and calendar years from text."""
        text = str(text or "")
        temporal_info = {
            "fiscal_years": [],
            "date_ranges": [],
            "years": [],
        }

        for match in TemporalReasoningHelper.FISCAL_YEAR_PATTERN.finditer(text):
            fy_info = TemporalReasoningHelper.normalize_fiscal_year(match.group(0))
            if fy_info:
                temporal_info["fiscal_years"].append(fy_info)

        for pattern in TemporalReasoningHelper.DATE_PATTERNS:
            for match in pattern.finditer(text):
                temporal_info["date_ranges"].append(match.group(0))

        temporal_info["years"] = sorted(set(TemporalReasoningHelper.YEAR_PATTERN.findall(text)))
        return temporal_info


class ExtractionValidator:
    """Validates answers against retrieved chunks to detect hallucinations/unsupported facts."""

    YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

    COMMON_WORDS = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "of", "is", "are", "was", "were",
        "be", "been", "being", "it", "its", "for", "with", "by", "from", "that", "this", "these",
        "those", "as", "has", "have", "had", "not", "no", "but", "about", "what", "which", "who",
        "whom", "whose", "when", "where", "why", "how", "does", "do", "did", "can", "could", "would",
        "should", "may", "might", "must", "tell", "explain", "criteria", "applying", "apply",
        "answer", "based", "states", "stated", "provided", "context", "requires", "requirement",
    }

    @staticmethod
    def normalize_text(text: Any) -> str:
        """Normalize text so matching survives OCR/PDF spaces, punctuation, and newlines."""
        text = str(text or "").lower()
        text = text.replace("₹", " rs ")
        text = re.sub(r"[\u2010-\u2015]", "-", text)
        text = re.sub(r"[^a-z0-9.%+-]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """Extract key entities: full years, numbers, quoted phrases, and proper nouns."""
        text = str(text or "")
        return {
            "years": ExtractionValidator.YEAR_PATTERN.findall(text),
            "numbers": re.findall(r"\b\d+(?:\.\d+)?\b", text),
            "quoted_phrases": re.findall(r"\"([^\"]+)\"", text),
            "proper_nouns": re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text),
        }

    @staticmethod
    def _content_words(text: Any) -> List[str]:
        norm = ExtractionValidator.normalize_text(text)
        words = []
        for word in norm.split():
            if len(word) <= 3:
                continue
            if word in ExtractionValidator.COMMON_WORDS:
                continue
            words.append(word)
        return words

    @staticmethod
    def validate_answer_against_chunks(
        answer: str,
        chunks: List[Dict[str, Any]],
        question: str = "",
    ) -> Dict[str, Any]:
        """
        Validate if answer is supported by retrieved chunks.
        Returns validation report with hallucination/unsupported-fact detection.
        """
        validation_report = {
            "is_valid": True,
            "confidence": 1.0,
            "issues": [],
            "supported_facts": [],
            "unsupported_facts": [],
        }

        if not answer or not chunks:
            return validation_report

        answer = str(answer or "")
        chunk_text_raw = " ".join(
            str(c.get("text") or c.get("text_preview") or "") for c in chunks if isinstance(c, dict)
        )
        chunk_text_norm = ExtractionValidator.normalize_text(chunk_text_raw)

        answer_entities = ExtractionValidator.extract_entities(answer)

        # Full-year detection: previous code returned only '19'/'20' because of a capturing-group regex.
        answer_years = set(answer_entities["years"])
        chunk_years = set(ExtractionValidator.YEAR_PATTERN.findall(chunk_text_raw))
        unsupported_years = sorted(answer_years - chunk_years)
        if unsupported_years:
            validation_report["issues"].append({
                "type": "temporal_hallucination",
                "detail": f"Years in answer not found in chunks: {unsupported_years}",
                "severity": "high",
            })
            validation_report["unsupported_facts"].extend([f"Year: {y}" for y in unsupported_years])
            validation_report["confidence"] *= 0.5
            validation_report["is_valid"] = False

        # Numeric facts are usually important in BFSI answers.
        answer_numbers = set(answer_entities["numbers"])
        chunk_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", chunk_text_raw))
        unsupported_numbers = sorted(answer_numbers - chunk_numbers)
        if unsupported_numbers:
            validation_report["issues"].append({
                "type": "hallucination",
                "detail": f"Numbers in answer not found in chunks: {unsupported_numbers}",
                "severity": "high",
            })
            validation_report["unsupported_facts"].extend([f"Number: {n}" for n in unsupported_numbers])
            validation_report["confidence"] *= 0.65
            validation_report["is_valid"] = False

        # Quoted phrases must appear nearly exactly after normalization.
        for phrase in answer_entities["quoted_phrases"]:
            phrase_norm = ExtractionValidator.normalize_text(phrase)
            if phrase_norm and phrase_norm not in chunk_text_norm:
                validation_report["issues"].append({
                    "type": "hallucination",
                    "detail": f"Quoted phrase not found in chunks: {phrase}",
                    "severity": "high",
                })
                validation_report["unsupported_facts"].append(f"Quoted phrase: {phrase}")
                validation_report["confidence"] *= 0.75
                validation_report["is_valid"] = False

        key_words = set(ExtractionValidator._content_words(answer))
        missing_support = [word for word in key_words if word not in chunk_text_norm]

        if key_words and len(missing_support) > len(key_words) * 0.35:
            validation_report["issues"].append({
                "type": "key_phrase_extraction_failure",
                "detail": f"{len(missing_support)}/{len(key_words)} key words not in chunks",
                "severity": "medium",
            })
            validation_report["is_valid"] = False
            validation_report["confidence"] *= 0.7

        # Generic over-generation check. Kept broad, not domain-hardcoded.
        if len(answer) > max(700, len(chunk_text_raw) * 0.75):
            validation_report["issues"].append({
                "type": "over_generation",
                "detail": "Answer is unusually long compared with retrieved support",
                "severity": "medium",
            })
            validation_report["is_valid"] = False
            validation_report["confidence"] *= 0.85

        validation_report["confidence"] = round(max(0.0, min(1.0, validation_report["confidence"])), 4)
        return validation_report


class RerankerIntegration:
    """Handles reranking score prioritization and chunk ranking optimization."""

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _query_terms(query: str) -> List[str]:
        return ExtractionValidator._content_words(query)

    @staticmethod
    def _has_exact_match(query: str, text: str) -> bool:
        """
        Robust exact/key-match check.
        Keeps the existing output field name `has_exact_match`, but avoids false negatives caused by PDF spacing.
        """
        query_norm = ExtractionValidator.normalize_text(query)
        text_norm = ExtractionValidator.normalize_text(text)

        if not query_norm or not text_norm:
            return False

        if len(query_norm) >= 8 and query_norm in text_norm:
            return True

        terms = RerankerIntegration._query_terms(query_norm)
        if not terms:
            return False

        matched_terms = sum(1 for term in set(terms) if term in text_norm)
        coverage = matched_terms / max(1, len(set(terms)))

        # Try phrase-level matching from important adjacent terms, e.g. "maximum loan amount".
        phrase_hit = False
        for n in (4, 3, 2):
            if len(terms) < n:
                continue
            for i in range(0, len(terms) - n + 1):
                phrase = " ".join(terms[i:i + n])
                if phrase in text_norm:
                    phrase_hit = True
                    break
            if phrase_hit:
                break

        return phrase_hit and coverage >= 0.5

    @staticmethod
    def compute_final_score(
        base_score: float,
        rerank_score: Optional[float],
        position: int = 0,
        has_exact_match: bool = False,
    ) -> float:
        """
        Compute final score combining base retrieval + reranking.
        Existing response field `final_score` is preserved.
        """
        base_score = RerankerIntegration._as_float(base_score, 0.0)
        rerank_score = None if rerank_score is None else RerankerIntegration._as_float(rerank_score, 0.0)

        if rerank_score is not None:
            final_score = (rerank_score * 0.7) + (base_score * 0.3)
        else:
            final_score = base_score

        if has_exact_match:
            final_score *= 1.1

        # Fixed bug: previous max(1.0, 1.0 - penalty) always returned 1.0.
        position_penalty = max(0.50, 1.0 - (max(0, position) * 0.02))
        final_score *= position_penalty

        return round(max(0.0, min(1.0, final_score)), 6)

    @staticmethod
    def rank_chunks_by_quality(
        chunks: List[Dict[str, Any]],
        query: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks by final_score. Output chunk structure is unchanged.
        """
        scored_chunks: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue

            base_score = RerankerIntegration._as_float(chunk.get("score", 0.0), 0.0)
            rerank_score = chunk.get("rerank_score")
            full_text = chunk.get("text") or chunk.get("text_preview") or ""

            has_exact_match = RerankerIntegration._has_exact_match(query, full_text)
            final_score = RerankerIntegration.compute_final_score(
                base_score=base_score,
                rerank_score=rerank_score,
                position=i,
                has_exact_match=has_exact_match,
            )

            chunk_copy = chunk.copy()
            chunk_copy["score"] = base_score
            chunk_copy["final_score"] = final_score
            chunk_copy["has_exact_match"] = has_exact_match
            scored_chunks.append(chunk_copy)

        scored_chunks.sort(
            key=lambda x: (
                x.get("final_score", 0.0),
                x.get("rerank_score") if x.get("rerank_score") is not None else -1.0,
                x.get("score", 0.0),
            ),
            reverse=True,
        )
        return scored_chunks


class AdvancedMetricsCollector:
    """Collects advanced diagnostics and quality metrics."""

    def __init__(self):
        self.metrics = {
            "total_rows": 0,
            "total_chunks": 0,
            "avg_score": 0.0,
            "low_score_queries": [],
            "extraction_failures": [],
            "hallucinations": [],
            "temporal_issues": [],
            "duplicate_rows": [],
            "rerank_rows_without_effect": [],
            "chunks_with_rerank_score": 0,
            "chunks_missing_rerank_score": 0,
            "quality_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0,
            },
        }

    @staticmethod
    def _preview(value: Any, limit: int = 100) -> str:
        return str(value or "")[:limit]

    def record_low_score(self, row_index: int, query: str, score: float):
        self.metrics["low_score_queries"].append({
            "row_index": row_index,
            "query": self._preview(query),
            "score": score,
        })

    def record_extraction_failure(self, row_index: int, query: str, issue: str):
        self.metrics["extraction_failures"].append({
            "row_index": row_index,
            "query": self._preview(query),
            "issue": str(issue),
        })

    def record_hallucination(self, row_index: int, query: str, answer: str, issue: str):
        self.metrics["hallucinations"].append({
            "row_index": row_index,
            "query": self._preview(query),
            "answer_preview": self._preview(answer),
            "issue": str(issue),
        })

    def record_temporal_issue(self, row_index: int, query: str, issue: str):
        self.metrics["temporal_issues"].append({
            "row_index": row_index,
            "query": self._preview(query),
            "issue": str(issue),
        })

    def record_duplicate_row(self, row_index: int, first_seen_row_index: int, query: str):
        self.metrics["duplicate_rows"].append({
            "row_index": row_index,
            "duplicate_of_row_index": first_seen_row_index,
            "query": self._preview(query),
        })

    def record_rerank_row_without_effect(self, row_index: int, query: str):
        self.metrics["rerank_rows_without_effect"].append({
            "row_index": row_index,
            "query": self._preview(query),
            "issue": "reranking_enabled is true, but rerank_score is missing or identical to score for all chunks",
        })

    def update_quality_distribution(self, score: float):
        score = RerankerIntegration._as_float(score, 0.0)
        if score >= 0.75:
            self.metrics["quality_distribution"]["high"] += 1
        elif score >= 0.6:
            self.metrics["quality_distribution"]["medium"] += 1
        else:
            self.metrics["quality_distribution"]["low"] += 1

    def update_rerank_stats(self, chunks: List[Dict[str, Any]]):
        for chunk in chunks:
            if chunk.get("rerank_score") is None:
                self.metrics["chunks_missing_rerank_score"] += 1
            else:
                self.metrics["chunks_with_rerank_score"] += 1

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_rows": self.metrics["total_rows"],
            "total_chunks": self.metrics["total_chunks"],
            "avg_score": self.metrics.get("avg_score", 0.0),
            "low_score_count": len(self.metrics["low_score_queries"]),
            "extraction_failure_count": len(self.metrics["extraction_failures"]),
            "hallucination_count": len(self.metrics["hallucinations"]),
            "temporal_issue_count": len(self.metrics["temporal_issues"]),
            "duplicate_row_count": len(self.metrics["duplicate_rows"]),
            "rerank_rows_without_effect_count": len(self.metrics["rerank_rows_without_effect"]),
            "chunks_with_rerank_score": self.metrics["chunks_with_rerank_score"],
            "chunks_missing_rerank_score": self.metrics["chunks_missing_rerank_score"],
            "quality_distribution": self.metrics["quality_distribution"],
        }


class RetrievedChunkTraceBatch:
    """
    Produces: per-query retrieved chunk trace + scores
    Inputs: JSON list with rows containing retrieved_chunks
    Output:
      - JSON: detailed trace
      - CSV: flat view (easy to open in Excel)

    Main response/output structure is preserved:
      payload keys: status, rows, total_chunks, zero_chunk_rows, rerank_rows, output_json, output_csv
      trace row keys: row_index, query, answer, ground_truth, reranking_enabled, chunks_sorted, chunks_total
    """

    def __init__(self):
        self.validator = ExtractionValidator()
        self.temporal_helper = TemporalReasoningHelper()
        self.reranker = RerankerIntegration()
        self.metrics = AdvancedMetricsCollector()

    @staticmethod
    def _first_present(*values: Any, default: Any = None) -> Any:
        for value in values:
            if value is not None:
                return value
        return default

    @staticmethod
    def _safe_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "enabled"}
        return bool(value)

    @staticmethod
    def _row_signature(row: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
        chunk_ids = [str(c.get("chunk_id", "")) for c in chunks[:20]]
        signature_payload = {
            "query": str(row.get("query") or row.get("question") or "").strip(),
            "answer": str(row.get("answer") or "").strip(),
            "ground_truth": str(row.get("ground_truth") or "").strip(),
            "chunk_ids": chunk_ids,
        }
        return json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)

    def _load_rows(self, input_json_path: str) -> List[Dict[str, Any]]:
        p = Path(input_json_path)

        if not p.exists():
            logger.error("Input JSON not found: %s", str(p))
            return []

        try:
            raw = p.read_text(encoding="utf-8")
        except Exception:
            logger.exception("Failed to read input JSON: %s", str(p))
            return []

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in %s | %s", str(p), str(e))
            return []
        except Exception:
            logger.exception("Unexpected error parsing JSON: %s", str(p))
            return []

        if isinstance(data, dict) and "rows" in data:
            data = data["rows"]

        if not isinstance(data, list):
            logger.warning(
                "Input JSON root is not a list (or dict with rows). type=%s | path=%s",
                type(data),
                str(p),
            )
            return []

        logger.info("Loaded %d rows from %s", len(data), str(p))
        return data

    def _get_raw_chunks(self, row: Dict[str, Any]) -> List[Any]:
        """Accept both pre-trace input rows and already-generated trace rows."""
        raw_chunks = self._first_present(
            row.get("retrieved_chunks"),
            row.get("chunks_sorted"),
            row.get("chunks"),
            row.get("source_documents"),
            row.get("contexts"),
            default=[],
        )

        if isinstance(raw_chunks, list):
            return raw_chunks

        if isinstance(raw_chunks, str):
            return [{"text": raw_chunks}]

        return []

    def _get_reranking_enabled(self, row: Dict[str, Any], chunks: List[Dict[str, Any]]) -> bool:
        reranking_obj = row.get("reranking") if isinstance(row.get("reranking"), dict) else {}
        explicit_flag = self._first_present(
            row.get("reranking_enabled"),
            reranking_obj.get("enabled"),
            row.get("rerank_enabled"),
            default=None,
        )
        if explicit_flag is not None:
            return self._safe_bool(explicit_flag)
        return any(c.get("rerank_score") is not None for c in chunks)

    def _norm_chunks(self, chunks: Any) -> List[Dict[str, Any]]:
        if not isinstance(chunks, list):
            return []

        out: List[Dict[str, Any]] = []
        skipped = 0

        for idx, c in enumerate(chunks):
            if isinstance(c, str):
                c = {"text": c, "chunk_id": f"string_chunk_{idx}"}

            if not isinstance(c, dict):
                skipped += 1
                continue

            meta = c.get("meta") if isinstance(c.get("meta"), dict) else c.get("metadata")
            if not isinstance(meta, dict):
                meta = {}

            chunk_id = self._first_present(
                c.get("chunk_id"),
                c.get("id"),
                meta.get("chunk_id"),
                default=f"unknown_{idx}",
            )
            text = self._first_present(
                c.get("text"),
                c.get("page_content"),
                c.get("content"),
                c.get("text_preview"),
                default="",
            )
            score = self._first_present(
                c.get("score"),
                c.get("similarity_score"),
                c.get("retrieval_score"),
                c.get("base_score"),
                default=0.0,
            )
            rerank_score = self._first_present(
                c.get("rerank_score"),
                c.get("re_rank_score"),
                c.get("cross_encoder_score"),
                c.get("rank_score"),
                default=None,
            )

            score = RerankerIntegration._as_float(score, 0.0)
            if rerank_score is not None:
                rerank_score = RerankerIntegration._as_float(rerank_score, 0.0)

            text = str(text or "")

            out.append({
                "chunk_id": str(chunk_id),
                "text": text,
                "score": score,
                "rerank_score": rerank_score,
                "source": self._first_present(c.get("source"), meta.get("source"), meta.get("file"), default="unknown"),
                "page": self._first_present(c.get("page"), meta.get("page"), meta.get("page_idx"), meta.get("page_number"), default=None),
                "text_preview": str(self._first_present(c.get("text_preview"), text[:220], default=""))[:220],
            })

        if skipped:
            logger.debug("Skipped %d non-dict chunks while normalizing", skipped)

        return out

    def _to_csv(self, rows_trace: List[Dict[str, Any]]) -> str:
        # minimal CSV writer (no pandas dependency)
        import csv
        import io

        buf = io.StringIO()
        w = csv.writer(buf)

        # CSV structure preserved.
        w.writerow([
            "row_index",
            "query",
            "chunk_rank",
            "chunk_id",
            "score",
            "rerank_score",
            "source",
            "page",
            "text_preview",
        ])

        for r in rows_trace:
            row_index = r.get("row_index")
            query = r.get("query", "")
            chunks = r.get("chunks_sorted", [])
            for rank, c in enumerate(chunks, start=1):
                w.writerow([
                    row_index,
                    query,
                    rank,
                    c.get("chunk_id", ""),
                    c.get("score", 0.0),
                    c.get("rerank_score", ""),
                    c.get("source", ""),
                    c.get("page", ""),
                    c.get("text_preview", ""),
                ])

        return buf.getvalue()

    def run_and_save(self, input_json_path: str, output_dir: str = "results/observability") -> Tuple[Dict[str, Any], Path, Path]:
        logger.info("RetrievedChunkTraceBatch started | in=%s | out_dir=%s", input_json_path, output_dir)

        try:
            rows = self._load_rows(input_json_path)
            self.metrics.metrics["total_rows"] = len(rows)

            trace_rows: List[Dict[str, Any]] = []
            total_chunks = 0
            total_score = 0.0
            rows_with_zero_chunks = 0
            rows_with_rerank = 0
            seen_signatures: Dict[str, int] = {}

            for i, row in enumerate(rows):
                if not isinstance(row, dict):
                    self.metrics.record_extraction_failure(i, "", "Row is not a JSON object")
                    row = {}

                query = row.get("query") or row.get("question") or ""
                answer = row.get("answer") or ""
                ground_truth = row.get("ground_truth") or ""

                raw_chunks = self._get_raw_chunks(row)
                chunks = self._norm_chunks(raw_chunks)
                reranking_enabled = self._get_reranking_enabled(row, chunks)

                if reranking_enabled:
                    rows_with_rerank += 1

                signature = self._row_signature(row, chunks)
                if signature in seen_signatures:
                    self.metrics.record_duplicate_row(i, seen_signatures[signature], str(query))
                else:
                    seen_signatures[signature] = i

                chunks = self.reranker.rank_chunks_by_quality(chunks, str(query))

                if reranking_enabled and chunks:
                    has_distinct_rerank = any(
                        c.get("rerank_score") is not None
                        and abs(RerankerIntegration._as_float(c.get("rerank_score")) - RerankerIntegration._as_float(c.get("score"))) > 1e-9
                        for c in chunks
                    )
                    if not has_distinct_rerank:
                        self.metrics.record_rerank_row_without_effect(i, str(query))

                total_chunks += len(chunks)
                self.metrics.update_rerank_stats(chunks)

                for chunk in chunks:
                    final_score = RerankerIntegration._as_float(chunk.get("final_score", chunk.get("score", 0.0)), 0.0)
                    self.metrics.update_quality_distribution(final_score)
                    total_score += final_score

                validation_report = self.validator.validate_answer_against_chunks(
                    str(answer),
                    chunks,
                    str(query),
                )

                if len(chunks) == 0:
                    rows_with_zero_chunks += 1
                    logger.warning(
                        "No retrieved chunks found | row_index=%d | query_preview=%s | reranking_enabled=%s",
                        i,
                        str(query)[:120],
                        reranking_enabled,
                    )
                    self.metrics.record_extraction_failure(i, str(query), "No chunks retrieved")
                else:
                    top_score = RerankerIntegration._as_float(chunks[0].get("final_score", chunks[0].get("score", 0.0)), 0.0)
                    if top_score < 0.6:
                        self.metrics.record_low_score(i, str(query), top_score)

                    if not validation_report["is_valid"]:
                        for issue in validation_report["issues"]:
                            issue_type = issue.get("type", "")
                            detail = issue.get("detail", "")
                            if issue_type == "hallucination":
                                self.metrics.record_hallucination(i, str(query), str(answer), detail)
                            elif issue_type == "temporal_hallucination":
                                self.metrics.record_temporal_issue(i, str(query), detail)
                            else:
                                self.metrics.record_extraction_failure(i, str(query), detail)

                # Main response structure preserved exactly.
                trace_rows.append({
                    "row_index": i,
                    "query": query,
                    "answer": str(row.get("answer", ""))[:],
                    "ground_truth": ground_truth,
                    "reranking_enabled": reranking_enabled,
                    "chunks_sorted": chunks,
                    "chunks_total": len(chunks),
                })

                if (i + 1) % 200 == 0:
                    logger.info("Processed %d/%d rows...", i + 1, len(rows))

            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            out_json = out_dir / "overall_retrieved_chunk_trace.json"
            out_csv = out_dir / "overall_retrieved_chunk_trace.csv"
            out_metrics = out_dir / "advanced_diagnostics_metrics.json"

            out_json.write_text(json.dumps(trace_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            out_csv.write_text(self._to_csv(trace_rows), encoding="utf-8")

            self.metrics.metrics["total_chunks"] = total_chunks
            self.metrics.metrics["avg_score"] = round(total_score / max(1, total_chunks), 4)

            diagnostics = {
                "summary": self.metrics.get_summary(),
                "low_score_queries": self.metrics.metrics["low_score_queries"][:50],
                "extraction_failures": self.metrics.metrics["extraction_failures"][:50],
                "hallucinations": self.metrics.metrics["hallucinations"][:50],
                "temporal_issues": self.metrics.metrics["temporal_issues"][:50],
                "duplicate_rows": self.metrics.metrics["duplicate_rows"][:50],
                "rerank_rows_without_effect": self.metrics.metrics["rerank_rows_without_effect"][:50],
                "recommendations": self._generate_recommendations(),
            }
            out_metrics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

            logger.info(
                "RetrievedChunkTraceBatch completed | rows=%d | total_chunks=%d | avg_chunks_per_row=%.2f | zero_chunk_rows=%d | rerank_rows=%d | output_json=%s | output_csv=%s",
                len(trace_rows),
                total_chunks,
                (total_chunks / max(1, len(trace_rows))),
                rows_with_zero_chunks,
                rows_with_rerank,
                str(out_json),
                str(out_csv),
            )

            # Payload structure preserved exactly.
            payload = {
                "status": "ok",
                "rows": len(trace_rows),
                "total_chunks": total_chunks,
                "zero_chunk_rows": rows_with_zero_chunks,
                "rerank_rows": rows_with_rerank,
                "output_json": str(out_json),
                "output_csv": str(out_csv),
            }
            return payload, out_json, out_csv

        except Exception:
            logger.exception(
                "RetrievedChunkTraceBatch failed | in=%s | out_dir=%s",
                input_json_path,
                output_dir,
            )
            raise

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on collected metrics."""
        recommendations: List[Dict[str, Any]] = []

        low_score_count = len(self.metrics.metrics["low_score_queries"])
        if low_score_count > 0:
            recommendations.append({
                "priority": "high",
                "category": "retrieval",
                "issue": f"{low_score_count} queries with low chunk scores (< 0.6)",
                "action": "Improve embedding model, query expansion, reranker quality, or chunking strategy",
                "affected_queries": self.metrics.metrics["low_score_queries"][:5],
            })

        duplicate_count = len(self.metrics.metrics["duplicate_rows"])
        if duplicate_count > 0:
            recommendations.append({
                "priority": "medium",
                "category": "observability",
                "issue": f"{duplicate_count} duplicate trace rows detected",
                "action": "Deduplicate upstream inputs or attach run_id/repeat_id/timestamp so repeated runs are intentional",
                "affected_queries": self.metrics.metrics["duplicate_rows"][:5],
            })

        rerank_no_effect_count = len(self.metrics.metrics["rerank_rows_without_effect"])
        if rerank_no_effect_count > 0:
            recommendations.append({
                "priority": "high",
                "category": "reranking",
                "issue": f"{rerank_no_effect_count} rows have reranking enabled but no distinct rerank signal",
                "action": "Ensure reranker output is stored separately from retriever score and sort by final_score",
                "affected_queries": self.metrics.metrics["rerank_rows_without_effect"][:5],
            })

        hallucination_count = len(self.metrics.metrics["hallucinations"])
        if hallucination_count > 0:
            recommendations.append({
                "priority": "high",
                "category": "extraction",
                "issue": f"{hallucination_count} hallucinations/unsupported facts detected",
                "action": "Add extraction validation and enforce context-only answer prompts",
                "affected_queries": self.metrics.metrics["hallucinations"][:5],
            })

        temporal_count = len(self.metrics.metrics["temporal_issues"])
        if temporal_count > 0:
            recommendations.append({
                "priority": "high",
                "category": "temporal_reasoning",
                "issue": f"{temporal_count} temporal extraction failures",
                "action": "Use full-year regex, normalize fiscal years, and verify answer years against retrieved chunks",
                "affected_queries": self.metrics.metrics["temporal_issues"][:5],
            })

        extraction_count = len(self.metrics.metrics["extraction_failures"])
        if extraction_count > 0:
            recommendations.append({
                "priority": "medium",
                "category": "extraction",
                "issue": f"{extraction_count} extraction/support issues",
                "action": "Improve chunk boundaries, reduce noisy top-k chunks, and add sentence-level support checks",
                "affected_queries": self.metrics.metrics["extraction_failures"][:5],
            })

        return recommendations
