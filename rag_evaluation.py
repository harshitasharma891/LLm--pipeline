# src/rag_evaluation.py

import os
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from src.evaluation_metrics import EvaluationMetrics
from src.llm_factory import build_chat_model
from src.logging import logger


class EvaluationRAG:
    def __init__(
        self,
        chain,
        model=None,
    ):
        """
        chain:
            RAG chain supporting:
            chain.invoke({"input": query})

        model:
            LLM used for metric evaluation.
        """
        logger.info(
            "Initializing EvaluationRAG..."
        )

        self.chain = chain

        if (
            model is not None
            and isinstance(
                model,
                type(chain),
            )
        ):
            model = None

        self.model = (
            model
            or build_chat_model(
                provider=os.getenv(
                    "EVAL_LLM_PROVIDER",
                    "openai",
                ),
                model=os.getenv(
                    "EVAL_LLM_MODEL",
                    "gpt-4o-mini",
                ),
                temperature=float(
                    os.getenv(
                        "EVAL_LLM_TEMPERATURE",
                        "0.0",
                    )
                ),
                base_url=os.getenv(
                    "EVAL_LLM_BASE_URL"
                ),
            )
        )

        self.metrics = EvaluationMetrics(
            self.model
        )

        logger.info(
            "EvaluationRAG initialized successfully."
        )

    # ======================================================
    # Extract Context
    # ======================================================
    def _extract_context_texts(
        self,
        response: dict,
    ):
        """
        Supports:
        - response["context"] as Document list
        - response["context"] as string
        - response["retrieved_context"] as string
        """
        docs = response.get(
            "context"
        )

        if (
            isinstance(docs, list)
            and docs
        ):
            texts = []

            for document in docs:
                if hasattr(
                    document,
                    "page_content",
                ):
                    texts.append(
                        document.page_content
                    )
                else:
                    texts.append(
                        str(document)
                    )

            return texts

        if (
            isinstance(docs, str)
            and docs.strip()
        ):
            return [
                docs.strip()
            ]

        retrieved_context = response.get(
            "retrieved_context"
        )

        if (
            isinstance(
                retrieved_context,
                str,
            )
            and retrieved_context.strip()
        ):
            return [
                retrieved_context.strip()
            ]

        return []


    def answer_correctness(self, f1, asim, w=0.7):
        return round((w * f1) + ((1 - w) * asim), 3)


    # ======================================================
    # Overall Score for Dataset Evaluation
    # ======================================================
    def _calculate_overall_score(
        self,
        row,
    ):
        """
        Calculate an overall score using only the retained
        output metrics.

        Answer quality receives the highest weight so that
        relevant context cannot rescue a factually wrong
        answer.

        Penalties for abstention when context provides partial
        or full answers.
        """
        try:
            def value(
                metric_name,
                default=0.0,
            ):
                try:
                    metric_value = row.get(
                        metric_name,
                        default,
                    )

                    return max(
                        0.0,
                        min(
                            float(metric_value),
                            1.0,
                        ),
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    return default

            support_scores = (
                row.get(
                    "chunk_support_scores",
                    [],
                )
                or []
            )

            valid_support_scores = [
                float(score)
                for score in support_scores
                if isinstance(
                    score,
                    (int, float),
                )
            ]

            average_support = (
                sum(valid_support_scores)
                / len(valid_support_scores)
                if valid_support_scores
                else 0.0
            )

            answer_quality = (
                0.25 * value("answer_correctness")
                + 0.10 * value("answer_similarity")
                + 0.10 * value("f1_score")
                + 0.07 * value("precision")
                + 0.07 * value("recall")
                + 0.10 * value("answer_relevancy")
                + 0.08 * value("completeness")
                + 0.06 * value("factual_support")
                + 0.04 * value("custom_factcc_style")
                + 0.04 * value("custom_summac_style")
                + 0.03 * value("custom_qags_style")
                + 0.02 * value("custom_dae_style")
                + 0.04 * value("custom_bleurt_style")
            )

            retrieval_quality = (
                0.35 * value(
                    "context_precision"
                )
                + 0.30 * value(
                    "context_relevancy"
                )
                + 0.20 * max(
                    0.0,
                    min(
                        average_support,
                        1.0,
                    ),
                )
                + 0.15 * value(
                    "chunk_utilization"
                )
            )

            base_score = (
                0.80 * answer_quality
                + 0.20 * retrieval_quality
            )

            correctness = value(
                "answer_correctness"
            )

            pass_threshold = (
                self.metrics.pass_threshold
            )

            # PENALTY: If correctness is below pass threshold,
            # cap overall score below it too
            if correctness < pass_threshold:
                base_score = min(
                    base_score,
                    max(
                        0.0,
                        pass_threshold - 0.001,
                    ),
                )

            # SEVERE PENALTY: Very low correctness
            if correctness <= 0.10:
                base_score = min(
                    base_score,
                    0.20,
                )

            elif correctness <= 0.25:
                base_score = min(
                    base_score,
                    0.35,
                )

            # PENALTY: Toxicity
            if (
                value("toxicity_score")
                > self.metrics.toxicity_max
            ):
                base_score *= 0.50

            # NEW: PENALTY for unsafe abstention
            # If answer is abstention but context provides partial coverage,
            # this reduces the score significantly
            intent_coverage = row.get(
                "intent_coverage",
                1.0,
            )

            is_abstention = self.metrics.is_abstention(
                str(row.get("answer", ""))
            )

            if (
                is_abstention
                and intent_coverage
                >= self.metrics
                .abstention_coverage_threshold
            ):
                # Context CAN answer the query,
                # but model refused: UNSAFE ABSTENTION
                base_score = min(
                    base_score,
                    0.15,
                )

            elif (
                is_abstention
                and intent_coverage
                < self.metrics
                .abstention_coverage_threshold
            ):
                # Context CANNOT fully answer query,
                # and model refused: SAFE ABSTENTION
                # This is rewarded in answer_relevancy already,
                # so reduce penalty here
                pass

            return round(
                max(
                    0.0,
                    min(
                        base_score,
                        1.0,
                    ),
                ),
                3,
            )

        except Exception as exc:
            logger.error(
                (
                    "Overall score calculation "
                    f"failed: {exc}"
                ),
                exc_info=True,
            )
            return 0.0

    # ======================================================
    # Single Batch Evaluation
    # ======================================================
    def single_batch_evaluation(
        self,
        query,
        true_answer,
        context_texts,
        generated_answer,
    ):
        """
        Evaluate one row having:
        - query
        - ground truth
        - context
        - generated answer

        Now includes multi-intent coverage.
        """
        try:
            chunks = self.metrics.split_chunks(
                context_texts
            )

            chunk_impacts = (
                self.metrics.chunk_impact(
                    query=query,
                    answer=generated_answer,
                    contexts=chunks,
                    # reference=true_answer,
                )
            )

            relevance_scores = (
                chunk_impacts.get(
                    "chunk_relevance",
                    [],
                )
            )

            utilization_scores = (
                chunk_impacts.get(
                    "chunk_utilization",
                    [],
                )
            )

            support_scores = (
                chunk_impacts.get(
                    "chunk_support",
                    [],
                )
            )

            raw_context_precision = (
                # self.metrics.context_precision(
                #     relevance_scores,
                #     contexts=chunks,
                # )
                self.metrics.context_precision(relevance_scores, top_k=3)
            )

            raw_context_relevancy = (
                self.metrics.context_relevancy(relevance_scores, top_k=3)
            )

            # Calculate multi-intent coverage
            intent_coverage = (
                self.metrics.query_intent_coverage(
                    query=query,
                    contexts=chunks,
                )
            )

            # Apply intent coverage to context metrics
            context_precision = round(
                raw_context_precision
                * intent_coverage,
                3,
            )

            context_relevancy = round(
                raw_context_relevancy
                * intent_coverage,
                3,
            )

            # Check if answer is abstention
            is_abstention = self.metrics.is_abstention(
                generated_answer
            )

            # Calculate answer relevancy with abstention awareness
            answer_relevancy = (
                self.metrics.answer_relevancy(
                    query=query,
                    response=generated_answer,
                )
            )

            # LOGIC FIX: If answer is unsafe abstention,
            # reduce answer_relevancy
            if (
                is_abstention
                and intent_coverage
                >= self.metrics
                .abstention_coverage_threshold
            ):
                # Context CAN answer, but model refused
                answer_relevancy = round(
                    answer_relevancy * 0.3,
                    3,
                )

            elif (
                is_abstention
                and intent_coverage
                < self.metrics
                .abstention_coverage_threshold
            ):
                # Context CANNOT fully answer,
                # and model safely refused: REWARD
                answer_relevancy = (
                    self.metrics
                    .safe_abstention_relevancy(
                        base_relevancy=(
                            answer_relevancy
                        ),
                        intent_coverage=(
                            intent_coverage
                        ),
                    )
                )

            answer_similarity = (
                self.metrics.answer_similarity(
                    gt=true_answer,
                    answer=generated_answer,
                )
            )

            (
                precision,
                recall,
                f1_score,
            ) = self.metrics.bert_score([generated_answer], [true_answer])


            answer_correctness = (
                # self.answer_correctness(
                #     true_answer=true_answer,
                #     generated_answer=(
                #         generated_answer
                #     ),
                #     context_texts=context_texts,
                #     query=query,
                # )
                self.answer_correctness(
                    f1=f1_score,
                    asim=answer_similarity,
                    w=0.7
                )
            )

            (
                toxicity_score,
                toxicity_reason,
            ) = self.metrics.toxicity(
                generated_answer
            )

            # POC-aligned additional evaluation metrics
            completeness = self.metrics.completeness(
                ground_truth, generated_answer, context_texts
            )
            factual_support = self.metrics.factual_support(
                generated_answer, context_texts
            )
            custom_factcc_style = self.metrics.custom_factcc_style(
                generated_answer, context_texts
            )
            custom_summac_style = self.metrics.custom_summac_style(
                generated_answer, context_texts
            )
            custom_qags_style = self.metrics.custom_qags_style(
                generated_answer, context_texts
            )
            custom_dae_style = self.metrics.custom_dae_style(
                generated_answer, context_texts
            )
            custom_bleurt_style = self.metrics.custom_bleurt_style(
                ground_truth, generated_answer
            )

            average_chunk_utilization = (
                round(
                    (
                        sum(utilization_scores)
                        / len(
                            utilization_scores
                        )
                    ),
                    3,
                )
                if utilization_scores
                else 0.0
            )

            return {
                "chunk_utilization": (
                    average_chunk_utilization
                ),
                "chunk_relevance_scores": (
                    relevance_scores
                ),
                "chunk_utilization_scores": (
                    utilization_scores
                ),
                "chunk_support_scores": (
                    support_scores
                ),
                "context_relevancy": (
                    context_relevancy
                ),
                "context_precision": (
                    context_precision
                ),
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "answer_relevancy": (
                    answer_relevancy
                ),
                "answer_similarity": (
                    answer_similarity
                ),
                "answer_correctness": (
                    answer_correctness
                ),
                "completeness": completeness,
                "factual_support": factual_support,
                "custom_factcc_style": custom_factcc_style,
                "custom_summac_style": custom_summac_style,
                "custom_qags_style": custom_qags_style,
                "custom_dae_style": custom_dae_style,
                "custom_bleurt_style": custom_bleurt_style,
                "toxicity_score": (
                    toxicity_score
                ),
                "toxicity_reason": (
                    toxicity_reason
                ),
                "intent_coverage": (
                    intent_coverage
                ),
            }

        except Exception as exc:
            logger.exception(
                (
                    "single_batch_evaluation "
                    f"failed: {exc}"
                )
            )

            return {
                "chunk_utilization": 0.0,
                "chunk_relevance_scores": [],
                "chunk_utilization_scores": [],
                "chunk_support_scores": [],
                "context_relevancy": 0.0,
                "context_precision": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "answer_relevancy": 0.0,
                "answer_similarity": 0.0,
                "answer_correctness": 0.0,
                "completeness": 0.0,
                "factual_support": 0.0,
                "custom_factcc_style": 0.0,
                "custom_summac_style": 0.0,
                "custom_qags_style": 0.0,
                "custom_dae_style": 0.0,
                "custom_bleurt_style": 0.0,
                "toxicity_score": 0,
                "toxicity_reason": (
                    f"Error: {exc}"
                ),
                "intent_coverage": 0.0,
            }

    # ======================================================
    # Document Evaluation
    # ======================================================
    def document_evaluation(
        self,
        testset,
        batch_size=500,
        max_workers=25,
    ):
        """
        Required columns:
        - question
        - ground_truth
        - rag_response
        - rag_context
        """
        logger.info(
            (
                "Starting Document RAG Evaluation "
                "with Parallel Processing..."
            )
        )

        try:
            required_columns = [
                "question",
                "ground_truth",
                "rag_response",
                "rag_context",
            ]

            for column in required_columns:
                if column not in testset.columns:
                    raise ValueError(
                        (
                            "Missing required column: "
                            f"{column}"
                        )
                    )

            test_questions = (
                testset["question"].tolist()
            )

            ground_truths = (
                testset["ground_truth"].tolist()
            )

            responses = (
                testset["rag_response"].tolist()
            )

            retrieved_contexts = (
                testset["rag_context"].tolist()
            )

            total = len(
                test_questions
            )

            results = [
                None
            ] * total

            for start in range(
                0,
                total,
                batch_size,
            ):
                end = min(
                    start + batch_size,
                    total,
                )

                logger.info(
                    (
                        f"Processing batch {start} "
                        f"to {end - 1}"
                    )
                )

                with ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    future_to_index = {}

                    for index in range(
                        start,
                        end,
                    ):
                        future = executor.submit(
                            self.single_batch_evaluation,
                            test_questions[index],
                            ground_truths[index],
                            retrieved_contexts[index],
                            responses[index],
                        )

                        future_to_index[
                            future
                        ] = index

                    for future in as_completed(
                        future_to_index
                    ):
                        index = (
                            future_to_index[
                                future
                            ]
                        )

                        results[index] = (
                            future.result()
                        )

            result_df = testset.copy()

            if (
                not results
                or results[0] is None
            ):
                logger.warning(
                    "No evaluation results produced."
                )
                return result_df

            for key in results[0].keys():
                result_df[key] = [
                    result[key]
                    for result in results
                ]

            # Add the answer from rag_response for abstention detection
            result_df["answer"] = result_df[
                "rag_response"
            ]

            result_df[
                "overall_score"
            ] = result_df.apply(
                self._calculate_overall_score,
                axis=1,
            )

            os.makedirs(
                "results",
                exist_ok=True,
            )

            result_df.to_csv(
                "results/overall_results1.csv",
                index=False,
            )

            result_df.to_json(
                "results/overall_results1.json",
                orient="records",
                lines=True,
            )

            logger.info(
                (
                    "Saved results to "
                    "results/overall_results1.csv"
                )
            )

            logger.info(
                (
                    "Saved results to "
                    "results/overall_results1.json"
                )
            )

            return result_df

        except Exception as exc:
            logger.exception(
                (
                    "Document evaluation "
                    f"failed: {exc}"
                )
            )
            return testset

    # ======================================================
    # Single Live Evaluation
    # ======================================================
    def single_evaluation(
        self,
        query,
        response,
    ):
        """
        Evaluate one live RAG response without ground truth.

        Behaviour:
        1. Context scores measure coverage of the complete query.
        2. Missing query parts reduce context metrics.
        3. Safe abstention is rewarded when the complete query
           cannot be answered.
        4. An unnecessary abstention is not rewarded when the
           context fully answers the query.

        No new fields are added to the returned result.
        """
        logger.info(
            "Starting single response evaluation..."
        )

        try:
            context_texts = (
                self._extract_context_texts(
                    response
                )
            )

            if not context_texts:
                logger.warning(
                    (
                        "No context provided "
                        "for evaluation"
                    )
                )

                return {
                    "error": (
                        "No context provided "
                        "for evaluation"
                    )
                }

            answer = str(
                response.get(
                    "answer",
                    "",
                )
                or ""
            ).strip()

            chunks = self.metrics.split_chunks(
                context_texts
            )

            chunk_scores = (
                self.metrics.chunk_impact(
                    query=query,
                    answer=answer,
                    contexts=chunks,
                )
            )

            relevance_scores = (
                chunk_scores.get(
                    "chunk_relevance",
                    [],
                )
            )

            utilization_scores = (
                chunk_scores.get(
                    "chunk_utilization",
                    [],
                )
            )

            support_scores = (
                chunk_scores.get(
                    "chunk_support",
                    [],
                )
            )

            average_chunk_utilization = (
                round(
                    (
                        sum(utilization_scores)
                        / len(
                            utilization_scores
                        )
                    ),
                    3,
                )
                if utilization_scores
                else 0.0
            )

            raw_context_relevancy = (
                self.metrics.context_relevancy(
                    relevance_scores
                )
            )

            raw_context_precision = (
                self.metrics.context_precision(
                    relevance_scores,
                    # contexts=chunks,
                )
            )

            intent_coverage = (
                self.metrics.query_intent_coverage(
                    query=query,
                    contexts=chunks,
                )
            )

            context_relevancy = round(
                raw_context_relevancy
                * intent_coverage,
                3,
            )

            context_precision = round(
                raw_context_precision
                * intent_coverage,
                3,
            )

            answer_relevancy = (
                self.metrics.answer_relevancy(
                    query=query,
                    response=answer,
                )
            )

            is_abstention = (
                self.metrics.is_abstention(
                    answer
                )
            )

            safe_abstention = (
                self.metrics.allow_safe_abstention
                and is_abstention
                and (
                    intent_coverage
                    < self.metrics
                    .abstention_coverage_threshold
                )
            )

            if safe_abstention:
                answer_relevancy = (
                    self.metrics
                    .safe_abstention_relevancy(
                        base_relevancy=(
                            answer_relevancy
                        ),
                        intent_coverage=(
                            intent_coverage
                        ),
                    )
                )

            diversity_score = (
                self.metrics.diversity_score(
                    answer
                )
            )

            (
                toxicity_score,
                toxicity_reason,
            ) = self.metrics.toxicity(
                answer
            )

            score = {
                "chunk_utilization": (
                    average_chunk_utilization
                ),
                "chunk_relevance_scores": (
                    relevance_scores
                ),
                "chunk_utilization_scores": (
                    utilization_scores
                ),
                "chunk_support_scores": (
                    support_scores
                ),
                "context_relevancy": (
                    context_relevancy
                ),
                "context_precision": (
                    context_precision
                ),
                "answer_relevancy": (
                    answer_relevancy
                ),
                "diversity_score": (
                    diversity_score
                ),
                "toxicity_score": (
                    toxicity_score
                ),
                "toxicity_reason": (
                    toxicity_reason
                ),
                "intent_coverage": (
                    intent_coverage
                ),
            }

            if safe_abstention:
                missing_coverage = (
                    1.0
                    - intent_coverage
                )

                base_score = (
                    0.75 * answer_relevancy
                    + 0.25 * missing_coverage
                )

            else:
                base_score = (
                    0.40 * context_precision
                    + 0.35 * context_relevancy
                    + 0.15 * answer_relevancy
                    + 0.10
                    * average_chunk_utilization
                )

            if (
                toxicity_score
                > self.metrics.toxicity_max
            ):
                base_score *= 0.50

            score["overall_score"] = round(
                max(
                    0.0,
                    min(
                        base_score,
                        1.0,
                    ),
                ),
                3,
            )

            return score

        except Exception as exc:
            logger.exception(
                (
                    "Single evaluation "
                    f"failed: {exc}"
                )
            )

            return {
                "error": str(exc)
            }




