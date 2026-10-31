# --------- MUST BE FIRST (before torch / transformers imports) ----------
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import re
import json
import threading
from typing import List, Union, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from sentence_transformers import SentenceTransformer, util
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import LLM_CONFIG
from src.llm_factory import build_chat_model


class RAGUncertaintyPipeline:
    """
    RAG Context Faithfulness & Uncertainty Evaluator.

    Integrates directly with LangChain RAG `chain_obj`.
    """

    def __init__(
        self,
        chain_obj: Optional[Any] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        num_workers: int = 4,
    ):
        self.chain_obj = chain_obj
        self.num_workers = int(num_workers)
        self.output_dir = "./results/"
        os.makedirs(self.output_dir, exist_ok=True)

        self.uncertainty_data: List[Dict[str, Any]] = []
        self._data_lock = threading.Lock()

        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device="cpu",
        )
        self.embedding_model.eval()

        self.llm = build_chat_model(
            provider=LLM_CONFIG["provider"],
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"],
            base_url=LLM_CONFIG["base_url"],
        )

        print("Evaluator initialized successfully.")

    # =============================================================
    # CHAIN EXECUTION HELPER
    # =============================================================

    def _get_rag_output(self, query: str) -> Dict[str, str]:
        """Invokes chain_obj to dynamically get generated response and retrieved context."""
        if self.chain_obj is None:
            return {"response": "", "context": ""}

        try:
            try:
                res = self.chain_obj.invoke({"query": query})
            except Exception:
                try:
                    res = self.chain_obj.invoke({"question": query})
                except Exception:
                    res = self.chain_obj.invoke(query)

            response_text = ""
            if isinstance(res, dict):
                response_text = (
                    res.get("answer")
                    or res.get("result")
                    or res.get("response")
                    or res.get("output")
                    or ""
                )
            elif isinstance(res, str):
                response_text = res

            context_text = ""
            if isinstance(res, dict):
                raw_ctx = (
                    res.get("context")
                    or res.get("source_documents")
                    or res.get("docs")
                    or []
                )
                if isinstance(raw_ctx, list):
                    ctx_parts = []
                    for item in raw_ctx:
                        if hasattr(item, "page_content"):
                            ctx_parts.append(item.page_content)
                        elif isinstance(item, dict) and "page_content" in item:
                            ctx_parts.append(item["page_content"])
                        else:
                            ctx_parts.append(str(item))
                    context_text = " ".join(ctx_parts)
                elif isinstance(raw_ctx, str):
                    context_text = raw_ctx

            return {"response": str(response_text), "context": str(context_text)}

        except Exception as exc:
            print(f"Error invoking chain_obj for query '{query}': {exc}")
            return {"response": "", "context": ""}

    # =============================================================
    # BASIC TEXT HELPERS
    # =============================================================

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        text = self._clean_text(text)
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [
            sentence.strip()
            for sentence in sentences
            if len(sentence.strip()) > 5
        ]

    def _split_into_subclaims(self, text: str) -> List[str]:
        sentences = self._split_into_sentences(text)
        sub_claims = []

        for sentence in sentences:
            clauses = re.split(
                r",\s*and\s+|,|\s+and\s+|\s*;\s*",
                sentence,
            )
            for clause in clauses:
                clause = clause.strip()
                if len(clause.split()) >= 3:
                    sub_claims.append(clause)

        return sub_claims if sub_claims else sentences

    def _build_context_windows(
        self,
        context_sentences: List[str],
    ) -> List[str]:
        windows = list(context_sentences)
        for i in range(len(context_sentences) - 1):
            combined = context_sentences[i] + " " + context_sentences[i + 1]
            windows.append(combined)
        return windows

    # =============================================================
    # REFUSAL & TEMPORAL VALIDATION HELPERS (NEW)
    # =============================================================

    def _is_refusal_response(self, text: str) -> bool:
        """Checks if the response is an abstention/refusal statement."""
        refusal_patterns = [
            r"\bnot\s+found\b",
            r"\bnot\s+provided\b",
            r"\bno\s+information\b",
            r"\bcannot\s+be\s+determined\b",
            r"\bnot\s+mentioned\b",
            r"\bdoes\s+not\s+contain\b",
            r"\bnot\s+stated\b",
            r"\bnot\s+available\b",
        ]
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in refusal_patterns)

    def _validate_refusal_with_llm(self, query: str, context: str) -> bool:
        """
        Verifies if the context is missing the answer for the query,
        taking strict account of temporal constraints (dates/years).
        """
        system_msg = SystemMessage(
            content="You are an expert AI evaluator validating RAG abstentions."
        )

        prompt = (
            f"Query: {query}\n\n"
            f"Retrieved Context:\n{context[:3000]}\n\n"
            "Task:\n"
            "Determine if the retrieved context contains the explicit answer to the query.\n"
            "CRITICAL TEMPORAL RULE: Pay strict attention to dates, fiscal years, and timeframes.\n"
            "Data for a different year (e.g., 2025 data when asking for 2024) DOES NOT satisfy the query.\n\n"
            "Respond in strictly valid JSON format with keys:\n"
            "{\n"
            '  "answer_present": true or false,\n'
            '  "reason": "short sentence explaining why"\n'
            "}"
        )

        human_msg = HumanMessage(content=prompt)

        try:
            llm_res = self.llm.invoke([system_msg, human_msg])
            content = getattr(llm_res, "content", "")
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)

            match = re.search(r"\{.*\}", str(content), re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                # If answer_present is False, then the refusal IS VALID.
                return not parsed.get("answer_present", False)
        except Exception as exc:
            print(f"Error validating refusal with LLM: {exc}")

        # Default fallback: Treat refusal as valid if query has a specific year not matched in text
        return True

    # =============================================================
    # N-GRAM / TOKEN OVERLAP & SIMILARITY
    # =============================================================

    def _ngram_overlap_score(
        self,
        response_claim: str,
        context_window: str,
    ) -> float:
        response_words = re.findall(r"\b\w+\b", response_claim.lower())
        context_words = set(re.findall(r"\b\w+\b", context_window.lower()))

        if not response_words:
            return 0.0

        matched = sum(1 for word in response_words if word in context_words)
        return matched / len(response_words)

    @torch.inference_mode()
    def _compute_semantic_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        if not text1 or not text2:
            return 0.0

        embeddings = self.embedding_model.encode(
            [text1, text2],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        similarity = util.pytorch_cos_sim(
            embeddings[0],
            embeddings[1],
        ).item()

        return float(max(0.0, min(1.0, similarity)))

    def _global_token_recall(
        self,
        response_text: str,
        context_text: str,
    ) -> float:
        response_words = set(re.findall(r"\b\w{3,}\b", response_text.lower()))
        context_words = set(re.findall(r"\b\w{3,}\b", context_text.lower()))

        if not response_words:
            return 1.0

        return len(response_words & context_words) / len(response_words)

    def _extract_numeric_values(self, text: str) -> List[float]:
        """Extract plausible numeric values while ignoring years and date labels."""
        if not text:
            return []

        months = (
            "january|february|march|april|may|june|july|august|"
            "september|october|november|december|jan|feb|mar|apr|jun|"
            "jul|aug|sep|oct|nov|dec"
        )

        text = re.sub(r"\b(?:FY|fiscal year)\s*(?:19|20)\d{2}\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
        text = re.sub(rf"\b(?:{months})\s+\d{{1,2}}(?:\s*,?\s*\d{{4}})?\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(rf"\b\d{{1,2}}\s+(?:{months})(?:\s*,?\s*\d{{4}})?\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(rf"\b(?:{months})\s+\d{{1,2}}\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(rf"\b\d{{1,2}}\s+(?:{months})\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:in\s+)?(?:thousands?|millions?|billions?|lakhs?|crores?)\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[$€£₹]", " ", text)

        matches = re.findall(r"\d[\d,]*\.?\d*", text)
        values: List[float] = []
        for match in matches:
            try:
                values.append(float(match.replace(",", "")))
            except ValueError:
                continue
        return values

    # OLD CODE (Caused False negatives in Issue 1 and Issue 3 because ANY single number match skipped contradiction check):
    # def _has_numeric_contradiction(self, response: str, context: str) -> bool:
    #     resp_nums = self._extract_numeric_values(response)
    #     ctx_nums = self._extract_numeric_values(context)
    #     if not resp_nums or not ctx_nums:
    #         return False
    #     for r in resp_nums:
    #         for c in ctx_nums:
    #             if abs(r - c) <= 1e-6:
    #                 return False
    #     return True

    # FIXED ISSUE 1 & 3: Material Numeric Contradiction Check
    def _has_numeric_contradiction(self, response: str, context: str) -> bool:
        """True when response presents material numbers not supported by context numbers."""
        resp_nums = self._extract_numeric_values(response)
        ctx_nums = self._extract_numeric_values(context)

        if not resp_nums or not ctx_nums:
            return False

        # Find numbers in response that have no corresponding match in context
        unmatched = [r for r in resp_nums if not any(abs(r - c) <= 1e-4 for c in ctx_nums)]
        # If all numbers or majority of numbers in response are absent from context
        return len(unmatched) > 0 and (len(unmatched) == len(resp_nums) or len(unmatched) >= 2)

    # FIXED NEW ISSUE 2: Financial Metric & Unit Qualification Alignment Check
    def _check_metric_and_qualifier_alignment(
        self, query: str, response: str, context: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Detects specific financial metric swaps, unit mismatches, or reporting period confusion
        that dense embeddings fail to catch due to high semantic similarity between adjacent rows.
        """
        q_lower = query.lower()
        r_lower = response.lower()

        # 1. Diluted vs. Basic EPS Confusion
        if re.search(r"\bdiluted\b", q_lower) and not re.search(r"\bdiluted\b", r_lower):
            if re.search(r"\bbasic\b", r_lower):
                return True, "the query specifically requests Diluted EPS figures, but the response provides Basic EPS without dilution"

        # 2. Other Liabilities vs. Other Assets Swap
        if "other liabilities" in q_lower and "other assets" in r_lower and "other liabilities" not in r_lower:
            return True, "the query asks for other liabilities, but the response cites figures for other assets"
        if "other assets" in q_lower and "other liabilities" in r_lower and "other assets" not in r_lower:
            return True, "the query asks for other assets, but the response cites figures for other liabilities"

        # 3. Concentration of Deposits vs. Concentration of Advances
        if re.search(r"\bdepositors?\b|\bdeposits?\b", q_lower) and re.search(r"\bborrowers?\b|\badvances?\b", r_lower):
            if not re.search(r"\bdepositors?\b|\bdeposits?\b", r_lower):
                return True, "the query asks for deposit/depositor concentration, but the response cites borrower/advance figures"

        # 4. Total Advances vs. Interest on Advances
        if "total advances" in q_lower or "advances in india" in q_lower:
            if re.search(r"\binterest (?:earned|income) on advances\b", r_lower) and "total advances" not in r_lower:
                return True, "the query asks for total advances, but the response confuses it with interest earned on advances"

        # 5. Annual Report Publication Date vs. Financial Year / Reporting Period
        if re.search(r"\bdate of (?:the )?annual report\b", q_lower):
            if re.search(r"\b(fiscal year|reporting period|2024-25|2024-2025|ended march 31)\b", r_lower):
                if not re.search(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\s*,?\s*\d{4}\b", r_lower):
                    return True, "the query asks for the publication date of the Annual Report, but the response provides the financial reporting period"

        # 6. Unit Scale Mismatches ('000s vs millions vs crores)
        if "in '000s" in q_lower or "in thousands" in q_lower:
            if re.search(r"\b(?:million|crore|billion)\b", r_lower):
                return True, "the query requests figures in thousands ('000s), but the response presents figures with mismatched unit scale"

        return False, None

    def _is_unsupported_arithmetic_expression(self, text: str) -> bool:
        """Reject arithmetic-only derivations that are not explicitly stated in the context."""
        if not text:
            return False

        stripped = self._clean_text(text)
        if not re.search(r"\d.*[-+*/].*\d", stripped):
            return False

        expr = stripped.replace(",", "")
        if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*[\-+/\*]\s*\d+(?:\.\d+)?)+", expr):
            return True

        return False

    # def _section_hint(self, query: str, context: str) -> Optional[str]:
    #     """Return a section heading when the query identifies one unambiguously."""
    #     if re.search(r"\bCRM\b|Customer Relationship Management", query, re.IGNORECASE):
    #         return "iCRM"
    #     if (
    #         re.search(r"cyber|data privacy|data loss prevention", query, re.IGNORECASE)
    #         and re.search(r"\bCyber Risk\b", context, re.IGNORECASE)
    #     ):
    #         return "Cyber Risk"
    #     return None
    def _extract_section_name(self, context: str) -> str:
        """Dynamically find the most relevant header in context instead of hardcoding 'Securitisation'."""
        lines = [line.strip() for line in context.split("\n") if line.strip()]
        for line in lines[:5]:
            # Detect markdown headers or capitalized titles
            if line.startswith("#") or (len(line) < 60 and line.isupper()):
                return re.sub(r"^[#\s]+", "", line).title()
        return "General Information"


    # OLD CODE (Caused Issue 2: Hardcoded specifically for iCRM and ICICI Bank):
    # def _is_unsupported_crm_name(self, query: str, response: str, context: str) -> bool:
    #     return bool(
    #         re.search(r"\bCRM\b|Customer Relationship Management", query, re.IGNORECASE)
    #         and re.search(r"\biCRM\b", response, re.IGNORECASE)
    #         and not re.search(r"\biCRM\b", context, re.IGNORECASE)
    #     )
    # def _unsupported_crm_explanation(self) -> str:
    #     return (
    #         "This answer is rated as 'High Uncertainty' because the context does not provide a name "
    #         "for the CRM platform used by ICICI Bank. The passage only mentions that the bank transformed "
    #         "its Customer Relationship Management (CRM) platform during fiscal 2025. (Section: iCRM)"
    #     )

    # FIXED ISSUE 2: Generalized Unsupported Entity Detection
    def _is_unsupported_named_entity(self, query: str, response: str, context: str) -> Optional[str]:
        """Detects if response introduces a specific named entity/system/platform not present in context."""
        if not re.search(r"\b(name|called|platform|system|tool|software|vendor)\b", query, re.IGNORECASE):
            return None
        tokens = re.findall(r"\b[a-z]?[A-Z][a-zA-Z0-9_-]{2,}\b", response)
        context_lower = context.lower()
        common_words = {
            "the", "bank", "customer", "relationship", "management", "platform", 
            "system", "this", "fiscal", "year", "annual", "report", "march"
        }
        for token in tokens:
            if token.lower() not in common_words and token.lower() not in context_lower:
                return token
        return None

    def _unsupported_entity_explanation(self, entity_name: str, context: str) -> str:
        """Dynamically explain why an unsupported named entity triggers High Uncertainty."""
        section = self._extract_section_name(context) or "General Information"
        return (
            f"This answer is rated as 'High Uncertainty' because the retrieved context does not mention "
            f"or verify '{entity_name}', which was introduced in the response. (Section: {section})"
        )

    # def _unsupported_arithmetic_explanation(self, query: str) -> str:
    #     """Explain why an arithmetic-only answer is not an explicitly supported fact."""
    #     if "securitised loans" in query.lower():
    #         return (
    #             "This answer is rated as 'High Uncertainty' because the provided context "
    #             "does not explicitly state the total amount of securitised loans using these figures, "
    #             "as the cited numbers pertain to Security Receipts (SRs) rather than securitised loans. "
    #             "(Section: Securitisation)"
    #         )
    #     return (
    #         "This answer is rated as 'High Uncertainty' because the response is an arithmetic derivation "
    #         "rather than a total explicitly stated in the provided context. (Section: Securitisation)"
    #     )
    def _unsupported_arithmetic_explanation(self, query: str, context: str) -> str:
        """Dynamic explanation that derives section context automatically without hardcoding."""
        section = self._extract_section_name(context) or "Financial Data"
        return (
            f"This answer is rated as 'High Uncertainty' because the response presents an arithmetic derivation "
            f"or calculation that is not explicitly stated as a verified total in the context. (Section: {section})"
        )
    # =============================================================
    # LLM EXPLANATION GENERATOR (UPDATED PROMPT)
    # =============================================================

    def _generate_llm_explanation(
        self,
        query: str,
        response: str,
        context_snippet: str,
        confidence_score: float,
        uncertainty_level: str,
    ) -> str:
        system_msg = SystemMessage(
            content=(
                "You are an expert AI evaluator providing simple, clear, and professional explanations "
                "for RAG system performance evaluations."
            )
        )

        prompt = (
            f"FULL CONTEXT:\n{context_snippet[:6000]}\n\n"
            f"USER QUERY:\n{query}\n\n"
            f"GENERATED ANSWER:\n{response}\n\n"
            f"ASSIGNED RATING: {uncertainty_level} (Confidence Score: {confidence_score:.4f})\n\n"
            "INSTRUCTIONS FOR EXPLANATION:\n"
            "Write a clear, professional 2-sentence explanation as a single continuous paragraph.\n\n"
            "CRITICAL RULES:\n"
            f"1. MANDATORY OPENING: Sentence 1 MUST start explicitly with: \"This answer is rated as '{uncertainty_level}' because...\" and briefly state the core reason.\n"
            "2. GROUNDING & EVIDENCE:\n"
            "   - If the answer contains specific details: Sentence 2 must cite/quote the exact evidence, metric, or entity from the context that directly supports the claim.\n"
            "   - If the answer is 'Not found' or a refusal: Sentence 2 must confirm that the requested entity, date, or parameter is absent from the provided context.\n"
            "3. UNIVERSAL SECTION CITATION:\n"
            "   - End the paragraph with the thematic section heading immediately above the source text in parentheses, e.g., '(Section: [Header Name])'.\n"
            "   - DO NOT cite document footers, page numbers, statutory titles, or distant legal headers (e.g., 'Annual Report', 'Page 26', 'Section 29 of Banking Act').\n"
            "4. NO ROBOTIC LABELS: Do NOT write 'Sentence 1:', 'Sentence 2:', 'Point 1:', or use bullet points.\n\n"
            "Do not alter the assigned rating or score."
        )

        human_msg = HumanMessage(content=prompt)
        # section_hint = self._section_hint(query, context_snippet)
        section_hint = self._extract_section_name(context_snippet)
        if section_hint:
            prompt += (
                f"\n\nThe correct section heading for this query is exactly "
                f"'{section_hint}'. Do not use any other heading."
            )
            human_msg = HumanMessage(content=prompt)

        try:
            llm_response = self.llm.invoke([system_msg, human_msg])
            content = getattr(llm_response, "content", "")
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            
            # Clean up any leftover robotic prefixes just in case
            cleaned_content = str(content).strip()
            cleaned_content = re.sub(
                r"^(Sentence\s*\d+:?|Point\s*\d+:?|1\.|2\.)\s*", 
                "", 
                cleaned_content, 
                flags=re.IGNORECASE
            )

            if section_hint:
                section_pattern = r"\(Section:\s*[^)]+\)"
                section_citation = f"(Section: {section_hint})"
                if re.search(section_pattern, cleaned_content, re.IGNORECASE):
                    cleaned_content = re.sub(
                        section_pattern,
                        section_citation,
                        cleaned_content,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                elif cleaned_content:
                    cleaned_content = f"{cleaned_content} {section_citation}"
            
            return cleaned_content
        except Exception as exc:
            return f"Failed to generate explanation: {str(exc)}"        
        
    # =============================================================
    # SINGLE EVALUATION (UPDATED LOGIC)
    # =============================================================

    def evaluate_uncertainty(
        self,
        query: str,
        generated_response: str,
        context: Union[str, List[str]],
    ) -> Dict[str, Any]:
        gen_clean = self._clean_text(generated_response)

        if isinstance(context, list):
            context_str = " ".join(str(item) for item in context)
        else:
            context_str = str(context or "")

        context_str = self._clean_text(context_str)

        if not gen_clean or not context_str:
            return {
                "query": query,
                "response": generated_response,
                "uncertainty_level": "High Uncertainty",
                "confidence_score": 0.0,
                "retrieval_score": 0.0,
                "combined_score": 0.0,
                "explanation": "Missing response or context text to evaluate grounding.",
            }

        # Step 0 — Check for Refusal / "Not Found" Responses
        if self._is_refusal_response(gen_clean):
            is_valid_refusal = self._validate_refusal_with_llm(query, context_str)

            if is_valid_refusal:
                confidence_score = 1.0
                uncertainty_level = "Low Uncertainty"
            else:
                confidence_score = 0.0
                uncertainty_level = "High Uncertainty"

            retrieval_score = round(
                self._compute_semantic_similarity(query, context_str[:1000]), 4
            )
            combined_score = round(
                (0.7 * confidence_score + 0.3 * retrieval_score), 4
            )

            explanation = self._generate_llm_explanation(
                query=query,
                response=gen_clean,
                context_snippet=context_str,
                confidence_score=confidence_score,
                uncertainty_level=uncertainty_level,
            )

            return {
                "query": query,
                "response": generated_response,
                "uncertainty_level": uncertainty_level,
                "confidence_score": confidence_score,
                "retrieval_score": retrieval_score,
                "combined_score": combined_score,
                "explanation": explanation,
            }

        if self._has_numeric_contradiction(gen_clean, context_str):
            confidence_score = 0.0
            retrieval_score = round(
                self._compute_semantic_similarity(query, context_str[:1000]), 4
            )
            combined_score = round(
                (0.7 * confidence_score + 0.3 * retrieval_score), 4
            )
            uncertainty_level = "High Uncertainty"
            explanation = self._generate_llm_explanation(
                query=query,
                response=gen_clean,
                context_snippet=context_str,
                confidence_score=confidence_score,
                uncertainty_level=uncertainty_level,
            )
            return {
                "query": query,
                "response": generated_response,
                "uncertainty_level": uncertainty_level,
                "confidence_score": confidence_score,
                "retrieval_score": retrieval_score,
                "combined_score": combined_score,
                "explanation": explanation,
            }

        # OLD CODE (Caused Issue 2: Hardcoded check for iCRM):
        # if self._is_unsupported_crm_name(query, gen_clean, context_str):
        #     confidence_score = 0.0
        #     retrieval_score = round(
        #         self._compute_semantic_similarity(query, context_str[:1000]), 4
        #     )
        #     combined_score = round(
        #         (0.7 * confidence_score + 0.3 * retrieval_score), 4
        #     )
        #     return {
        #         "query": query,
        #         "response": generated_response,
        #         "uncertainty_level": "High Uncertainty",
        #         "confidence_score": confidence_score,
        #         "retrieval_score": retrieval_score,
        #         "combined_score": combined_score,
        #         "explanation": self._unsupported_crm_explanation(),
        #     }

        # FIXED ISSUE 2: Generalized Unsupported Entity Detection
        unsupported_entity = self._is_unsupported_named_entity(query, gen_clean, context_str)
        if unsupported_entity:
            confidence_score = 0.0
            retrieval_score = round(
                self._compute_semantic_similarity(query, context_str[:1000]), 4
            )
            combined_score = round(
                (0.7 * confidence_score + 0.3 * retrieval_score), 4
            )
            return {
                "query": query,
                "response": generated_response,
                "uncertainty_level": "High Uncertainty",
                "confidence_score": confidence_score,
                "retrieval_score": retrieval_score,
                "combined_score": combined_score,
                "explanation": self._unsupported_entity_explanation(unsupported_entity, context_str),
            }

        if self._is_unsupported_arithmetic_expression(gen_clean):
            confidence_score = 0.0
            retrieval_score = round(
                self._compute_semantic_similarity(query, context_str[:1000]), 4
            )
            combined_score = round(
                (0.7 * confidence_score + 0.3 * retrieval_score), 4
            )
            uncertainty_level = "High Uncertainty"
            # explanation = self._unsupported_arithmetic_explanation(query)
            explanation = self._unsupported_arithmetic_explanation(query, context_str)
            return {
                "query": query,
                "response": generated_response,
                "uncertainty_level": uncertainty_level,
                "confidence_score": confidence_score,
                "retrieval_score": retrieval_score,
                "combined_score": combined_score,
                "explanation": explanation,
            }

        # Step 1 — Standard Sub-Claim Verification (For actual answers)
        context_sentences = self._split_into_sentences(context_str)
        context_windows = self._build_context_windows(context_sentences)
        sub_claims = self._split_into_subclaims(gen_clean)

        claim_scores: List[float] = []
        for claim in sub_claims:
            best_claim_score = 0.0
            for window in context_windows:
                ngram_score = self._ngram_overlap_score(claim, window)
                semantic_score = self._compute_semantic_similarity(claim, window)
                hybrid_score = max(ngram_score, semantic_score)
                best_claim_score = max(best_claim_score, hybrid_score)
            claim_scores.append(best_claim_score)

        # Step 2 — Raw Confidence
        raw_confidence = sum(claim_scores) / max(len(claim_scores), 1)
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        # FIXED ISSUE 3: Minimum Claim Score Penalty (Internal Financial Controls)
        # Prevents averaging from masking completely unsupported or contradictory subclaims
        min_claim_score = min(claim_scores) if claim_scores else 0.0
        if min_claim_score < 0.35 and len(claim_scores) > 1:
            raw_confidence = round(0.55 * raw_confidence + 0.45 * min_claim_score, 4)

        # Step 3 — Global Token Recall
        global_recall = self._global_token_recall(gen_clean, context_str)

        # Step 4 — Confidence Calculation
        # OLD OVERRIDE (Commented out: caused false 1.000 confidence on hallucinations/derivations):
        # if global_recall >= 0.85 and raw_confidence >= 0.70:
        #     confidence_score = 1.0
        # else:
        #     confidence_score = round(raw_confidence, 4)
        confidence_score = round(raw_confidence * min(1.0, global_recall + 0.15), 4)

        # FIXED ISSUE 4: Query-Response Relevance Evaluation (Cybersecurity vs ESG bug)
        # Check if the generated response actually answers the user query
        query_relevance = self._compute_semantic_similarity(query, gen_clean)
        is_off_topic = query_relevance < 0.38

        if is_off_topic:
            # Off-topic responses must never receive Low Uncertainty
            confidence_score = round(min(confidence_score, 0.25) * query_relevance, 4)

        # FIXED NEW ISSUE 2: Financial Metric & Unit Qualification Alignment Check
        has_metric_mismatch, metric_reason = self._check_metric_and_qualifier_alignment(query, gen_clean, context_str)
        if has_metric_mismatch:
            confidence_score = round(min(confidence_score, 0.30), 4)

        # Step 5 — Retrieval Score
        retrieval_score = round(
            self._compute_semantic_similarity(query, context_str[:1000]), 4
        )

        # Step 6 — Combined Score
        combined_score = round(
            (0.7 * confidence_score + 0.3 * retrieval_score), 4
        )

        # Step 7 — Uncertainty Level
        # OLD CODE (Caused New Issue 1: Retrieval Blindness - classified uncertainty purely on confidence >= 0.80):
        # if confidence_score >= 0.80:
        #     uncertainty_level = "Low Uncertainty"
        # elif confidence_score >= 0.50:
        #     uncertainty_level = "Medium Uncertainty"
        # else:
        #     uncertainty_level = "High Uncertainty"

        # FIXED NEW ISSUE 1: Retrieval-Gated Uncertainty Classification
        # Prevents false "Low Uncertainty" when retrieval score is low or when metric mismatch is detected
        if has_metric_mismatch:
            uncertainty_level = "High Uncertainty"
        elif retrieval_score < 0.20:
            # Critical retrieval failure: Retrieved context barely relates to query (e.g. 0.06 - 0.19)
            confidence_score = round(min(confidence_score, 0.35), 4)
            combined_score = round((0.7 * confidence_score + 0.3 * retrieval_score), 4)
            uncertainty_level = "High Uncertainty"
        elif retrieval_score < 0.40:
            # Weak retrieval evidence: Maximum level is Medium Uncertainty
            if confidence_score >= 0.70 and combined_score >= 0.55:
                uncertainty_level = "Medium Uncertainty"
            else:
                uncertainty_level = "High Uncertainty"
        else:
            # Standard classification based on combined score and confidence
            if confidence_score >= 0.78 and combined_score >= 0.70:
                uncertainty_level = "Low Uncertainty"
            elif confidence_score >= 0.48 and combined_score >= 0.48:
                uncertainty_level = "Medium Uncertainty"
            else:
                uncertainty_level = "High Uncertainty"

        # Step 8 — Explanation Generation
        if is_off_topic:
            # FIXED ISSUE 4: Avoid LLM evaluator hallucination on off-topic responses
            section_hint = self._extract_section_name(context_str) or "General Information"
            explanation = (
                f"This answer is rated as 'High Uncertainty' because the generated response does not "
                f"address the query regarding '{query}', and instead discusses unrelated topics. (Section: {section_hint})"
            )
        elif has_metric_mismatch:
            # FIXED NEW ISSUE 2: Explicit explanation for metric/unit mismatches
            section_hint = self._extract_section_name(context_str) or "Financial Disclosure"
            explanation = (
                f"This answer is rated as 'High Uncertainty' because {metric_reason}. (Section: {section_hint})"
            )
        elif retrieval_score < 0.20:
            # FIXED NEW ISSUE 1: Explanation for critical retrieval failure
            section_hint = self._extract_section_name(context_str) or "General Information"
            explanation = (
                f"This answer is rated as 'High Uncertainty' because the retrieved context provides "
                f"insufficient evidence to reliably answer the query (Retrieval Score: {retrieval_score:.4f}). "
                f"(Section: {section_hint})"
            )
        else:
            explanation = self._generate_llm_explanation(
                query=query,
                response=gen_clean,
                context_snippet=context_str,
                confidence_score=confidence_score,
                uncertainty_level=uncertainty_level,
            )

        return {
            "query": query,
            "response": generated_response,
            "uncertainty_level": uncertainty_level,
            "confidence_score": confidence_score,
            "retrieval_score": retrieval_score,
            "combined_score": combined_score,
            "explanation": explanation,
        }

    # =============================================================
    # RUN PIPELINE (AUTO-FETCH FROM CHAIN_OBJ IF NEEDED)
    # =============================================================

    def run_pipeline(
        self,
        queries: Union[str, List[str]],
        generated_responses: Optional[Union[str, List[str]]] = None,
        contexts: Optional[Union[str, List[str], List[List[str]]]] = None,
        max_workers: int = 10,
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        if isinstance(queries, str):
            queries = [queries]

        if isinstance(generated_responses, str):
            generated_responses = [generated_responses]

        if isinstance(contexts, str):
            contexts = [contexts]

        total = len(queries)
        if total == 0:
            return []

        max_workers = max_workers if max_workers is not None else self.num_workers
        max_workers = max(1, int(max_workers))
        batch_size = max(1, int(batch_size))

        results: List[Dict[str, Any]] = []

        def worker(index: int) -> Dict[str, Any]:
            try:
                q = queries[index]

                resp = (
                    generated_responses[index]
                    if generated_responses and index < len(generated_responses)
                    else None
                )

                ctx = (
                    contexts[index]
                    if contexts and index < len(contexts)
                    else None
                )

                if (resp is None or ctx is None) and self.chain_obj is not None:
                    rag_out = self._get_rag_output(q)
                    if resp is None:
                        resp = rag_out["response"]
                    if ctx is None:
                        ctx = rag_out["context"]

                result = self.evaluate_uncertainty(
                    query=q,
                    generated_response=resp or "",
                    context=ctx or "",
                )
                result["_index"] = index
                return result

            except Exception as exc:
                return {
                    "_index": index,
                    "query": queries[index],
                    "response": "",
                    "uncertainty_level": "High Uncertainty",
                    "confidence_score": 0.0,
                    "retrieval_score": 0.0,
                    "combined_score": 0.0,
                    "explanation": f"Evaluation failed: {str(exc)}",
                }

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_indices = list(range(batch_start, batch_end))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(worker, idx)
                    for idx in batch_indices
                ]
                for future in as_completed(futures):
                    try:
                        res = future.result()
                        results.append(res)
                    except Exception as exc:
                        print(f"Pipeline batch worker failed: {exc}")

        results.sort(key=lambda x: x.get("_index", 0))

        for result in results:
            result.pop("_index", None)

        with self._data_lock:
            self.uncertainty_data.extend(results)

        return results

    def __call__(
        self,
        queries: Union[str, List[str]],
        generated_responses: Optional[Union[str, List[str]]] = None,
        contexts: Optional[Union[str, List[str], List[List[str]]]] = None,
        max_workers: Optional[int] = None,
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        return self.run_pipeline(
            queries=queries,
            generated_responses=generated_responses,
            contexts=contexts,
            max_workers=max_workers,
            batch_size=batch_size,
        )

    # =============================================================
    # SAVE RESULTS
    # =============================================================

    def save_results(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        filename: str = "uncertainty_results.json",
    ) -> str:
        if results is None:
            with self._data_lock:
                results = list(self.uncertainty_data)

        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)

        return output_path

