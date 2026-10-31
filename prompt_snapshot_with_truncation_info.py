# src/observability/prompt_snapshot_with_truncation_info.py
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from src.logging import logger


class PromptSnapshotMinimalJob:
    """
    Produces: prompt snapshot + truncation info (per query)
    Input: rows with prompt_info + prompt_truncation
    Output: minimal JSON for auditing / observability

    Uses:
      - row["prompt_info"]["prompt_text"]
      - row["prompt_truncation"]  (recommended from server)
    """

    def __init__(
        self,
        default_input_json_path: str = "results/overall_eval_with_ret_chunks.json",
        default_output_json_path: str = "results/observability/overall_minimal_output.json",
        include_final_prompt: bool = True,
    ):
        self.default_input_json_path = default_input_json_path
        self.default_output_json_path = default_output_json_path
        self.include_final_prompt = bool(include_final_prompt)

        logger.info(
            "PromptSnapshotMinimalJob initialized | input=%s | output=%s | include_final_prompt=%s",
            self.default_input_json_path,
            self.default_output_json_path,
            self.include_final_prompt,
        )

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
            logger.warning("Input JSON root is not a list (or dict with rows). type=%s | path=%s", type(data), str(p))
            return []

        logger.info("Loaded %d rows from %s", len(data), str(p))
        return data

    def _safe_dict(self, x: Any) -> Dict[str, Any]:
        return x if isinstance(x, dict) else {}

    def _hash(self, s: str) -> str:
        return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

    def _normalize_trunc(self, pt: Any) -> Dict[str, Any]:
        d = self._safe_dict(pt)
        # stable defaults
        d.setdefault("enabled", False)
        d.setdefault("was_truncated", False)
        d.setdefault("reason", "unknown")
        d.setdefault("policy", "unknown")
        d.setdefault("max_prompt_chars", None)
        d.setdefault("template_overhead_chars", None)
        d.setdefault("available_context_chars", None)
        d.setdefault("context_chars_before", None)
        d.setdefault("context_chars_after", None)
        d.setdefault("truncated_chars", 0)
        return d

    def run_and_save(
        self,
        input_json_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Path]:
        in_path = input_json_path or self.default_input_json_path
        out_path = output_json_path or self.default_output_json_path

        logger.info("Prompt snapshot job started | in=%s | out=%s", in_path, out_path)

        try:
            rows = self._load_rows(in_path)
            out_rows: List[Dict[str, Any]] = []

            truncated_count = 0
            missing_prompt_text = 0

            for i, row in enumerate(rows):
                query = row.get("query") or row.get("question") or ""
                answer = str(row.get("answer") or "")
                ground_truth = row.get("ground_truth") or ""

                prompt_info = self._safe_dict(row.get("prompt_info"))
                prompt_text = str(prompt_info.get("prompt_text") or "")

                trunc = self._normalize_trunc(row.get("prompt_truncation"))
                if trunc.get("was_truncated") is True:
                    truncated_count += 1

                if not prompt_text:
                    missing_prompt_text += 1
                    # keep this as warning (common integration issue)
                    logger.warning(
                        "Missing prompt_text | row_index=%d | query_preview=%s",
                        i,
                        (query[:120] if isinstance(query, str) else ""),
                    )

                item: Dict[str, Any] = {
                    "row_index": i,
                    "query": query,
                    "answer_preview": answer,
                    "ground_truth": ground_truth,
                    "model_name": prompt_info.get("model_name", ""),
                    "prompt_version": prompt_info.get("prompt_version", ""),
                    "prompt_template": prompt_info.get("prompt_template", ""),
                    "system_prompt_preview": str(prompt_info.get("system_prompt", ""))[:240],
                    "prompt_chars": len(prompt_text),
                    "prompt_hash": self._hash(prompt_text),
                    "prompt_truncation": trunc,
                }

                if self.include_final_prompt:
                    item["prompt_text"] = prompt_text  # full prompt (audit)

                out_rows.append(item)

                # Optional: low-noise progress log for big files
                if (i + 1) % 200 == 0:
                    logger.info("Processed %d/%d rows...", i + 1, len(rows))

            p_out = Path(out_path)
            p_out.parent.mkdir(parents=True, exist_ok=True)

            p_out.write_text(
                json.dumps(out_rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            logger.info(
                "Prompt snapshot job completed | rows=%d | truncated=%d | missing_prompt_text=%d | output=%s",
                len(out_rows),
                truncated_count,
                missing_prompt_text,
                str(p_out),
            )

            payload = {
                "status": "ok",
                "rows": len(out_rows),
                "output_json": str(p_out),
                "truncated_rows": truncated_count,
                "missing_prompt_text_rows": missing_prompt_text,
            }
            return payload, p_out

        except Exception:
            logger.exception("Prompt snapshot job failed | in=%s | out=%s", in_path, out_path)
            raise


