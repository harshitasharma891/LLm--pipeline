# src/observability/sentence_level_citation_coverage.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from src.logging import logger
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")  # catches 30000, 30,000, 1.2 etc.

_STOP_WORDS = {
    "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "from", "by", 
    "with", "in", "into", "of", "is", "are", "was", "were", "be", "been", "being", "it", 
    "its", "that", "this", "these", "those", "as", "have", "has", "had", "not", "no", 
    "they", "their", "them", "he", "she", "his", "her", "we", "us", "our", "you", "your", 
    "i", "my", "me", "which", "who", "whom", "whose", "what", "where", "when", "why", 
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
    "than", "too", "very", "can", "will", "would", "could", "should", "just", "now", "about"
}

def _tokens(text: str) -> List[str]:
    words = _WORD_RE.findall((text or "").lower())
    return [w for w in words if w not in _STOP_WORDS]
def _token_overlap_score(a: str, b: str) -> float:
    at = set(_tokens(a))
    bt = set(_tokens(b))
    if not at or not bt:
        return 0.0
    return len(at & bt) / max(1, len(at))
def _extract_numbers(text: str) -> List[str]:
    nums = []
    for n in _NUM_RE.findall(text or ""):
        nums.append(n.replace(",", ""))  # "30,000" -> "30000"
    return nums
def _is_abstain_answer(ans: str) -> bool:
    a = (ans or "").strip().lower()
    if not a:
        return True
    patterns = [
        "not found in provided documents",
        "information not available in the provided documents",
        "not available in the provided documents",
        "not found in the provided documents",
    ]
    return any(p in a for p in patterns)
class BatchSentenceCitationCoverage:
    """
    Sentence-level citation coverage:
      - Splits answer into sentences
      - For each sentence, finds supporting chunks
      - Numbers in sentence must be supportable (if present)
    Input rows need:
      - answer OR generated_answer OR answer_preview
      - retrieved_chunks: [{chunk_id, text, meta, score}]
    """
    def __init__(
        self,
        sentence_support_threshold: float = 0.40,
        max_evidence_chunks: int = 3,
        min_sentence_chars: int = 12,
        require_number_match: bool = True,
    ):
        self.sentence_support_threshold = float(sentence_support_threshold)
        self.max_evidence_chunks = int(max_evidence_chunks)
        self.min_sentence_chars = int(min_sentence_chars)
        self.require_number_match = bool(require_number_match)
        logger.info(
            "BatchSentenceCitationCoverage initialized | threshold=%.3f | max_evidence=%d | min_sentence_chars=%d | require_number_match=%s",
            self.sentence_support_threshold,
            self.max_evidence_chunks,
            self.min_sentence_chars,
            self.require_number_match,
        )
    def _safe_list(self, x: Any) -> List[Any]:
        return x if isinstance(x, list) else []
    def _split_sentences(self, answer: str) -> List[str]:
        s = (answer or "").strip()
        if not s:
            return []
        parts = _SENT_SPLIT.split(s)
        if len(parts) <= 1 and "\n" in s:
            parts = [p.strip() for p in s.splitlines() if p.strip()]
        cleaned = []
        for p in parts:
            p2 = p.strip()
            if len(p2) >= self.min_sentence_chars:
                cleaned.append(p2)
        if not cleaned and s:
            logger.debug("Answer produced 0 eligible sentences after filtering | answer_len=%d", len(s))
        return cleaned
    def _norm_chunks(self, chunks: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        skipped = 0
        for c in self._safe_list(chunks):
            if not isinstance(c, dict):
                skipped += 1
                continue
            meta = c.get("meta") if isinstance(c.get("meta"), dict) else (c.get("metadata") or {})
            if not isinstance(meta, dict):
                meta = {}
            chunk_id = c.get("chunk_id") or c.get("id") or meta.get("chunk_id") or "unknown"
            text = c.get("text") or c.get("page_content") or c.get("content") or ""
            score = c.get("score", 0.0)
            try:
                score = float(score or 0.0)
            except Exception:
                logger.debug("Non-numeric chunk score, defaulting | chunk_id=%s | score=%r", chunk_id, score)
                score = 0.0
            out.append(
                {
                    "chunk_id": str(chunk_id),
                    "text": str(text),  # keep full text internally for number checks
                    "score": float(score),
                    "source": meta.get("source", meta.get("file", "unknown")),
                    "page": meta.get("page", meta.get("page_idx", meta.get("page_number"))),
                }
            )
        if skipped:
            logger.debug("Skipped %d non-dict chunks while normalizing", skipped)
        return out
    def _rank_evidence(self, sentence: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Returns top evidence chunks with:
          - support_score (overlap + retrieval)
          - numbers_supported computed using FULL chunk text (not preview)
        """
        nums = _extract_numbers(sentence)
        scored: List[Tuple[float, Dict[str, Any], float, bool]] = []
        for ch in chunks:
            chunk_text_full = (ch.get("text") or "").replace(",", "")
            overlap = _token_overlap_score(sentence, chunk_text_full)
            retr = float(ch.get("score", 0.0))
            combined = (0.85 * overlap) + (0.15 * retr)
            numbers_supported = True
            if self.require_number_match and nums:
                numbers_supported = all(n in chunk_text_full for n in nums)
            scored.append((combined, ch, overlap, numbers_supported))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for combined, ch, overlap, numbers_supported in scored[: max(1, self.max_evidence_chunks)]:
            out.append(
                {
                    "chunk_id": ch["chunk_id"],
                    "support_score": round(float(combined), 6),
                    "overlap_score": round(float(overlap), 6),
                    "retrieval_score": round(float(ch.get("score", 0.0)), 6),
                    "numbers_required": _extract_numbers(sentence),
                    "numbers_supported": bool(numbers_supported),
                    "source": ch.get("source", "unknown"),
                    "page": ch.get("page", None),
                    "text_preview": (ch.get("text", "") or "")[:220],
                }
            )
        return out
    def _passes_number_rule(self, sentence: str, evidence: List[Dict[str, Any]]) -> bool:
        if not self.require_number_match:
            return True
        nums = _extract_numbers(sentence)
        if not nums:
            return True
        return any(bool(ev.get("numbers_supported")) for ev in evidence)
    def _is_supported(self, sentence: str, evidence: List[Dict[str, Any]]) -> Tuple[bool, str]:
        if not evidence:
            return False, "No evidence candidates"
        best = float(evidence[0].get("support_score", 0.0))
        if best < self.sentence_support_threshold:
            return False, "No chunk meets support threshold"
        if not self._passes_number_rule(sentence, evidence):
            return False, "Numbers in sentence not supported by any evidence chunk"
        return True, ""
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
    def run_and_save(self, input_json_path: str, output_dir: str = "results/observability") -> Tuple[Dict[str, Any], Path]:
        logger.info(
            "Sentence citation coverage started | in=%s | out_dir=%s | threshold=%.3f",
            input_json_path,
            output_dir,
            self.sentence_support_threshold,
        )
        try:
            rows = self._load_rows(input_json_path)
            out_rows: List[Dict[str, Any]] = []
            abstained_rows = 0
            total_sentences = 0
            supported_sentences_total = 0
            zero_chunk_rows = 0
            for i, row in enumerate(rows):
                query = row.get("query") or row.get("question") or ""
                answer = (
                    row.get("answer")
                    or row.get("generated_answer")
                    or row.get("answer_preview")
                    or ""
                )
                answer = str(answer)
                ground_truth = row.get("ground_truth") or ""
                chunks = self._norm_chunks(row.get("retrieved_chunks") or [])
                if not chunks:
                    zero_chunk_rows += 1
                    logger.warning(
                        "No retrieved_chunks for row | row_index=%d | query_preview=%s",
                        i,
                        (query[:120] if isinstance(query, str) else ""),
                    )
                if _is_abstain_answer(answer):
                    abstained_rows += 1
                    out_rows.append(
                        {
                            "row_index": i,
                            "query": query,
                            "answer": answer,
                            "ground_truth": str(ground_truth),
                            "abstained": True,
                            "sentences_total": 0,
                            "sentences_supported": 0,
                            "sentences_unsupported": 0,
                            "coverage_percent": None,
                            "threshold": self.sentence_support_threshold,
                            "supported_sentences": [],
                            "unsupported_sentences": [],
                            "note": "Abstained answer (Not found / not available). Coverage not applicable.",
                        }
                    )
                    continue
                sentences = self._split_sentences(answer)
                supported: List[Dict[str, Any]] = []
                unsupported: List[Dict[str, Any]] = []
                if not sentences:
                    logger.warning(
                        "No eligible sentences after split/filter | row_index=%d | answer_len=%d | query_preview=%s",
                        i,
                        len(answer),
                        (query[:120] if isinstance(query, str) else ""),
                    )
                for s in sentences:
                    evidence = self._rank_evidence(s, chunks)
                    ok, reason = self._is_supported(s, evidence)
                    if ok:
                        supported.append({"sentence": s, "evidence": evidence})
                    else:
                        unsupported.append(
                            {
                                "sentence": s,
                                "reason": reason,
                                "top_candidates": evidence,
                            }
                        )
                total = len(sentences)
                supported_n = len(supported)
                unsupported_n = len(unsupported)
                coverage = round((supported_n / total) * 100, 2) if total else 0.0
                total_sentences += total
                supported_sentences_total += supported_n
                out_rows.append(
                    {
                        "row_index": i,
                        "query": query,
                        "answer": answer,
                        "ground_truth": str(ground_truth),
                        "abstained": False,
                        "sentences_total": total,
                        "sentences_supported": supported_n,
                        "sentences_unsupported": unsupported_n,
                        "coverage_percent": coverage,
                        "threshold": self.sentence_support_threshold,
                        "supported_sentences": supported,
                        "unsupported_sentences": unsupported,
                    }
                )
                # Optional: low-noise progress log for big files
                if (i + 1) % 200 == 0:
                    logger.info("Processed %d/%d rows...", i + 1, len(rows))
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "overall_sentence_citation_coverage.json"
            out_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            overall_coverage = (
                round((supported_sentences_total / total_sentences) * 100, 2)
                if total_sentences
                else None
            )
            logger.info(
                "Sentence citation coverage completed | rows=%d | abstained_rows=%d | zero_chunk_rows=%d | total_sentences=%d | supported_sentences=%d | overall_coverage_percent=%s | output=%s",
                len(out_rows),
                abstained_rows,
                zero_chunk_rows,
                total_sentences,
                supported_sentences_total,
                str(overall_coverage) if overall_coverage is not None else "n/a",
                str(out_path),
            )
            payload = {
                "status": "ok",
                "rows": len(out_rows),
                "abstained_rows": abstained_rows,
                "zero_chunk_rows": zero_chunk_rows,
                "total_sentences": total_sentences,
                "supported_sentences": supported_sentences_total,
                "overall_coverage_percent": overall_coverage,
                "output_json": str(out_path),
            }
            return payload, out_path
        except Exception:
            logger.exception(
                "Sentence citation coverage failed | in=%s | out_dir=%s",
                input_json_path,
                output_dir,
            )
            raise
