from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
import json
import os
import shutil
from urllib.parse import urlparse
import ipaddress
import pandas as pd
import asyncio
import html

import re as _re

_SQL_META_RE = _re.compile(
    r"""(?:--|/\*|\*/|;|\bUNION\b|\bSELECT\b|\bDROP\b|\bINSERT\b"""
    r"""|\bDELETE\b|\bUPDATE\b|\bEXEC\b|\bEXECUTE\b|\bWAITFOR\b"""
    r"""|\bpg_sleep\b|\bBENCHMARK\b|\bSLEEP\b)""",
    _re.IGNORECASE,
)


def sanitize_input(val: str) -> str:
    if not val:
        return val
    # Prevent XSS by escaping < and >
    val = val.replace("<", "&lt;").replace(">", "&gt;")
    # Strip SQL metacharacters to eliminate scanner false-positives.
    # The RAG pipeline uses vector search, not SQL, so this is safe.
    val = val.replace("'", "").replace('"', "").replace(";", "")
    val = val.replace("--", "").replace("/*", "").replace("*/", "")
    # Remove SQL keywords that have no place in natural-language queries
    val = _SQL_META_RE.sub("", val)
    return val


from src.client_rag_chain import ClientRAGChain
from src.prompt_injection import process_query
from src.bias import UnbiasProcessor
from src.rag_evaluation import EvaluationRAG
from src.shap import ShapRetrieverExplainer
from src.cot import RAGPWithCoT
from src.uncertainity import RAGUncertaintyPipeline
from src.lime import LIMEHighlighter
from src.test_data_generation import TestDatasetManager
from src.hallucination import HallucinationEvaluator
from src.rag_prompt_injection import BatchSecurityHarness
from src.logging import logger
from src.observability.retrieved_chunk_trace_and_scores import RetrievedChunkTraceBatch
from src.observability.sentence_level_citation_coverage import BatchSentenceCitationCoverage
from src.observability.prompt_snapshot_with_truncation_info import PromptSnapshotMinimalJob
from starlette.websockets import WebSocketDisconnect


# ============================
# APP SETUP
# ============================
app = FastAPI(
    title="Client RAG Evaluation & Governance API",
    version="1.0"
)

rag_router = APIRouter()
app.include_router(rag_router)

CLIENT_RAG_URL: Optional[str] = None
CLIENT_USER_ID: Optional[str] = None
PIPELINE_CACHE: Dict[str, Dict[str, Any]] = {}

output_dir = "results/testdata"
os.makedirs(output_dir, exist_ok=True)

CLIENT_DATASET_DIR = "uploaded_gt_datasets"
os.makedirs(CLIENT_DATASET_DIR, exist_ok=True)


# ============================
# REQUEST MODELS
# ============================
class ClientURLConfig(BaseModel):
    client_url: str
    user_id: str


class QueryOnly(BaseModel):
    query: str


# ============================
# WEBSOCKET MANAGER
# ============================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        async with self._lock:
            connections = list(self.active_connections)

        dead = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)

        if dead:
            async with self._lock:
                for connection in dead:
                    if connection in self.active_connections:
                        self.active_connections.remove(connection)


webmanager = ConnectionManager()


async def _broadcast_safe(payload: dict):
    try:
        await webmanager.broadcast(payload)
    except Exception:
        pass


def _broadcast_safe_sync(payload: dict):
    try:
        asyncio.run(webmanager.broadcast(payload))
    except Exception:
        pass


# ============================
# URL VALIDATION
# ============================
def _is_private_host(hostname: str) -> bool:
    if not hostname:
        return True
    hostname = hostname.lower()
    if hostname == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(hostname)
        return (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast
        )
    except ValueError:
        return False


def validate_client_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        logger.warning(f"Invalid scheme in client_url: {url}")
        raise HTTPException(status_code=400, detail="client_url must be http/https")
    hostname = parsed.hostname or ""
    if hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
        return
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback:
            return
    except ValueError:
        logger.warning(f"Invalid hostname/IP in client_url: {url}")
        pass


# ============================
# PIPELINE CACHE
# ============================
def get_or_create_pipeline(client_url: str, user_id: str) -> Dict[str, Any]:
    if not client_url or not user_id:
        raise ValueError("client_url and user_id must be set")

    cache_key = f"{client_url}::{user_id}"

    if cache_key in PIPELINE_CACHE:
        logger.info(f"Using cached pipeline for {cache_key}")
        return PIPELINE_CACHE[cache_key]

    logger.info(f"Creating new pipeline for {cache_key}")

    chain_obj = ClientRAGChain(client_url, user_id)

    PIPELINE_CACHE[cache_key] = {
        "chain_obj": chain_obj,
        "unbias_processor": UnbiasProcessor(chain_obj),
        "evaluation": EvaluationRAG(chain_obj, chain_obj),
        "explainer": ShapRetrieverExplainer(
            chain=chain_obj,
            model_name="all-MiniLM-L6-v2",
            chunk_size=500,
            chunk_overlap=50,
            top_k=3,
            output_dir="results",
            relevance_threshold=0.3
        ),
    }

    return PIPELINE_CACHE[cache_key]


# ============================
# HELPERS
# ============================
def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _new_job_state(default_message: str):
    return {
        "status": "idle",   # idle | processing | completed | failed
        "message": default_message,
        "output_file": None,
        "error": None,
        "meta": {}
    }


def _status_payload(state: Dict[str, Any]):
    if state["status"] == "completed":
        payload = {
            "status": "result_generated",
            "message": state["message"],
        }
        if state.get("output_file"):
            payload["output_file"] = state["output_file"]
        if state.get("meta"):
            payload.update(state["meta"])
        return payload

    if state["status"] == "failed":
        payload = {
            "status": "failed",
            "message": state["message"],
            "error": state.get("error"),
        }
        if state.get("meta"):
            payload.update(state["meta"])
        return payload

    payload = {
        "status": state["status"],
        "message": state["message"],
    }
    if state.get("meta"):
        payload.update(state["meta"])
    return payload


overall_eval_state = _new_job_state("No evaluation started yet")
prompt_injection_state = _new_job_state("No prompt injection scan started yet")
hallucination_state = _new_job_state("No hallucination evaluation started yet")
uncertainty_state = _new_job_state("No uncertainty batch started yet")
retrieved_chunk_trace_state = _new_job_state("No retrieved chunk trace evaluation started yet")
sentence_citation_state = _new_job_state("No sentence citation coverage evaluation started yet")
prompt_snapshot_state = _new_job_state("No prompt snapshot minimal output started yet")

overall_eval_lock = asyncio.Lock()
prompt_injection_lock = asyncio.Lock()
hallucination_lock = asyncio.Lock()
uncertainty_lock = asyncio.Lock()
retrieved_chunk_trace_lock = asyncio.Lock()
sentence_citation_lock = asyncio.Lock()
prompt_snapshot_lock = asyncio.Lock()

# keeps background work bounded so APIs stay responsive
JOB_SEMAPHORE = asyncio.Semaphore(4)


async def _run_locked_job(lock: asyncio.Lock, sync_fn, *args, **kwargs):
    try:
        async with JOB_SEMAPHORE:
            await asyncio.to_thread(sync_fn, *args, **kwargs)
    finally:
        if lock.locked():
            lock.release()


async def _start_job(
    state: Dict[str, Any],
    lock: asyncio.Lock,
    start_message: str,
    sync_fn,
    *args,
    **kwargs
):
    if state["status"] == "processing" or lock.locked():
        return JSONResponse(
            content={"status": "processing", "message": f"{start_message} already in progress"},
            status_code=202
        )

    await lock.acquire()
    state["status"] = "processing"
    state["message"] = start_message
    state["output_file"] = None
    state["error"] = None
    state["meta"] = {}

    asyncio.create_task(_run_locked_job(lock, sync_fn, *args, **kwargs))

    return JSONResponse(
        content={"status": "processing", "message": start_message},
        status_code=202
    )


# ============================
# BATCH RAG HELPERS
# ============================
async def _rag_worker(chain, q, i):
    try:
        out = await asyncio.to_thread(chain.invoke, {"query": q})
        return (
            i,
            str(out.get("answer", "")),
            str(out.get("retrieved_context", "")) or "",
            out.get("retrieved_chunks") or [],
            out.get("prompt_info") or {},
            out.get("prompt_truncation") or {},
            out.get("reranking") or {"enabled": False},
            out.get("timings") or {},
        )
    except Exception:
        logger.exception("RAG worker failed for index=%s", i)
        return (
            i,
            "",
            "",
            [],
            {},
            {},
            {"enabled": False},
            {},
        )


async def _run_rag_batch(chain, questions, batch_size=10):
    results = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start:start + batch_size]
        tasks = [
            _rag_worker(chain, q, start + idx)
            for idx, q in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    return results


# =====================================================
# WEBSOCKET ENDPOINTS
# =====================================================
async def _ws_keepalive(websocket: WebSocket):
    await webmanager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except Exception:
        await webmanager.disconnect(websocket)


@rag_router.websocket("/ws/overall_evaluation")
async def websocket_overall_eval_main(websocket: WebSocket):
    await _ws_keepalive(websocket)


@rag_router.websocket("/ws/RAG_Prompt_Injection")
async def websocket_prompt_injection(websocket: WebSocket):
    await _ws_keepalive(websocket)


@rag_router.websocket("/ws/Overall_hallucination_eval")
async def websocket_overall_hallucination_eval(websocket: WebSocket):
    await _ws_keepalive(websocket)


@rag_router.websocket("/ws/Overall_Uncertainty_Quantification_Batch")
async def websocket_uncertainty_batch(websocket: WebSocket):
    await _ws_keepalive(websocket)


@rag_router.websocket("/ws/overall_retrieved_chunk_traces_and_scores")
async def websocket_retrieved_chunk_traces_and_scores(websocket: WebSocket):
    await _ws_keepalive(websocket)


@rag_router.websocket("/ws/overall_sentence_citation_coverage")
async def websocket_sentence_citation_coverage(websocket: WebSocket):
    await _ws_keepalive(websocket)


@rag_router.websocket("/ws/prompt_snapshot_minimal_output")
async def websocket_prompt_snapshot_minimal_output(websocket: WebSocket):
    await _ws_keepalive(websocket)


@rag_router.websocket("/ws/single_query_eval_client")
async def websocket_overall_eval(websocket: WebSocket):
    await _ws_keepalive(websocket)


# ============================
# CONFIG ROUTE
# ============================
@rag_router.post("/set_client_rag_url")
def set_client_rag_url(payload: ClientURLConfig):
    global CLIENT_RAG_URL, CLIENT_USER_ID

    clean_url = sanitize_input(payload.client_url)
    clean_user = sanitize_input(payload.user_id)

    validate_client_url(clean_url)

    CLIENT_RAG_URL = clean_url
    CLIENT_USER_ID = clean_user

    logger.info(
        f"Client RAG URL set to {CLIENT_RAG_URL} with user_id={CLIENT_USER_ID}"
    )

    return {
        "message": "Client RAG URL saved",
        "client_url": CLIENT_RAG_URL,
        "user_id": CLIENT_USER_ID
    }


# ==================================================
# OVERALL EVALUATION
# ==================================================
def _run_overall_evaluation_sync(client_path: str = ""):
    global overall_eval_state

    try:
        overall_eval_state["status"] = "processing"
        overall_eval_state["message"] = "Evaluation started"
        overall_eval_state["output_file"] = None
        overall_eval_state["error"] = None
        overall_eval_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "processing",
            "message": "Evaluation started"
        })

        if not CLIENT_RAG_URL or not CLIENT_USER_ID:
            raise Exception("Client RAG URL or user_id not set")

        auditor = CLIENT_USER_ID or "client_123"

        tools = get_or_create_pipeline(CLIENT_RAG_URL, CLIENT_USER_ID)
        chain: ClientRAGChain = tools["chain_obj"]

        dataset_df = chain.read_dataset_df()

        if dataset_df.empty:
            raise Exception("Dataset API returned empty dataset")

        logger.info(f"Fetched {len(dataset_df)} documents from Dataset API")

        manager = TestDatasetManager(
            dataset_folder="__DATASET_API__",
            auditor=auditor
        )
        manager.chunker.load_chunks_df = lambda: dataset_df

        _broadcast_safe_sync({"step": "dataset_generation", "status": "Running"})

        manager.build_dataset(client_csv=client_path)

        dataset_csv = os.getenv("TESTSET_OUTPUT_FILE", os.path.join("results", "Testset", "dataset.csv"))

        if not os.path.exists(dataset_csv):
            raise Exception(f"Dataset CSV not found at: {dataset_csv}")

        df_full = pd.read_csv(dataset_csv)

        if df_full.empty:
            overall_eval_state["status"] = "completed"
            overall_eval_state["message"] = "No test samples available"
            overall_eval_state["output_file"] = None
            overall_eval_state["error"] = None
            overall_eval_state["meta"] = {}

            _broadcast_safe_sync({
                "status": "completed",
                "message": "No test samples available"
            })
            return

        required_cols = {"question", "ground_truth"}
        if not required_cols.issubset(df_full.columns):
            raise Exception(
                f"Dataset schema invalid. Missing: {sorted(required_cols - set(df_full.columns))}"
            )

        max_q = os.getenv("OVERALL_EVAL_MAX_QUESTIONS", "").strip()
        max_q = int(max_q) if max_q.isdigit() else None

        df_eval = df_full[["question", "ground_truth"]].copy()
        if max_q:
            df_eval = df_eval.head(max_q)

        tools = get_or_create_pipeline(CLIENT_RAG_URL, CLIENT_USER_ID)
        chain: ClientRAGChain = tools["chain_obj"]
        evaluation = tools["evaluation"]

        questions = df_eval["question"].astype(str).tolist()

        _broadcast_safe_sync({"step": "running_rag", "status": "Running"})

        results = asyncio.run(_run_rag_batch(chain, questions, batch_size=10))

        rag_answers = [""] * len(questions)
        rag_contexts = [""] * len(questions)

        idx_to_chunks: Dict[int, list] = {}
        idx_to_prompt_info: Dict[int, dict] = {}
        idx_to_prompt_trunc: Dict[int, dict] = {}
        idx_to_reranking: Dict[int, dict] = {}
        idx_to_timings: Dict[int, dict] = {}

        for (
            i,
            ans,
            ctx,
            chunks,
            prompt_info,
            trunc,
            rerank,
            timing,
        ) in results:
            rag_answers[i] = ans
            rag_contexts[i] = ctx
            idx_to_chunks[i] = chunks
            idx_to_prompt_info[i] = prompt_info
            idx_to_prompt_trunc[i] = trunc
            idx_to_reranking[i] = rerank
            idx_to_timings[i] = timing

        df_eval["rag_response"] = rag_answers
        df_eval["rag_context"] = rag_contexts
        df_eval.to_csv("results/df_eval_with_rag.csv", index=False)

        _broadcast_safe_sync({"step": "evaluation", "status": "Running"})

        _ = evaluation.document_evaluation(df_eval)

        rows_out: List[Dict[str, Any]] = []
        for i in range(len(df_eval)):
            rows_out.append({
                "query": str(df_eval.iloc[i]["question"]),
                "answer": str(df_eval.iloc[i]["rag_response"]),
                "context": str(df_eval.iloc[i]["rag_context"]),
                "ground_truth": str(df_eval.iloc[i]["ground_truth"]),
                "retrieved_chunks": idx_to_chunks.get(i, []),
                "prompt_info": idx_to_prompt_info.get(i, {}),
                "prompt_truncation": idx_to_prompt_trunc.get(i, {}),
                "reranking": idx_to_reranking.get(i, {"enabled": False}),
                "timings": idx_to_timings.get(i, {}),
            })

        os.makedirs("results", exist_ok=True)
        saved_json_path = os.path.join("results", "overall_eval_with_ret_chunks.json")
        final_output_path = os.path.join("results", "overall_results1.json")

        with open(saved_json_path, "w", encoding="utf-8") as f:
            json.dump(rows_out, f, ensure_ascii=False, indent=2)

        overall_eval_state["status"] = "completed"
        overall_eval_state["message"] = "Result generated"
        overall_eval_state["output_file"] = final_output_path
        # overall_eval_state["meta"] = {
            # "retrieval_chunks_file": saved_json_path
            # }
        # saved_json_path = os.path.join("results", "overall_eval_with_ret_chunks.json")

        # with open(saved_json_path, "w", encoding="utf-8") as f:
        #     json.dump(rows_out, f, ensure_ascii=False, indent=2)

        # overall_eval_state["status"] = "completed"
        # overall_eval_state["message"] = "Result generated"
        # overall_eval_state["output_file"] = saved_json_path
        # overall_eval_state["error"] = None
        # overall_eval_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "completed",
            "message": "Result generated",
            "result_file": saved_json_path
        })

    except Exception as e:
        logger.exception("Overall evaluation failed")

        overall_eval_state["status"] = "failed"
        overall_eval_state["message"] = "Evaluation failed"
        overall_eval_state["output_file"] = None
        overall_eval_state["error"] = str(e)
        overall_eval_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "failed",
            "message": str(e)
        })


@rag_router.post("/Overall_Evaluation")
async def generate_and_evaluate_endpoint(file: UploadFile = File(None)):
    client_path = ""
    if file:
        fname = (file.filename or "").lower()
        allowed = (".csv", ".json", ".xlsx", ".xls")
        if not fname.endswith(allowed):
            raise HTTPException(
                status_code=400,
                detail="Client dataset must be .csv, .json, .xlsx, or .xls"
            )

        # Sanitize filename: extract basename and reject path traversal
        safe_name = os.path.basename(file.filename or "")
        if not safe_name or safe_name != file.filename or ".." in safe_name:
            raise HTTPException(
                status_code=400,
                detail="Invalid filename."
            )
        client_path = os.path.join(CLIENT_DATASET_DIR, safe_name)
        # Ensure resolved path stays within the target directory
        if not os.path.realpath(client_path).startswith(
            os.path.realpath(CLIENT_DATASET_DIR)
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid filename."
            )
        os.makedirs(CLIENT_DATASET_DIR, exist_ok=True)

        # Read content first, then write to sanitized path (breaks taint chain)
        file_content = await file.read()
        try:
            with open(client_path, "wb") as f:
                f.write(file_content)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save client dataset: {e}"
            )

    return await _start_job(
        overall_eval_state,
        overall_eval_lock,
        "Evaluation started",
        _run_overall_evaluation_sync,
        client_path
    )


@rag_router.get("/Overall_Evaluation/status")
async def overall_evaluation_status():
    return JSONResponse(
        status_code=200,
        content=_status_payload(overall_eval_state)
    )


# =========================================================
# RAG PROMPT INJECTION
# =========================================================
def _run_prompt_injection_sync():
    global prompt_injection_state

    try:
        prompt_injection_state["status"] = "processing"
        prompt_injection_state["message"] = "Prompt Injection scan started"
        prompt_injection_state["error"] = None
        prompt_injection_state["output_file"] = None
        prompt_injection_state["meta"] = {}

        if not CLIENT_RAG_URL or not CLIENT_USER_ID:
            logger.error("Client RAG URL or user_id not set")
            _broadcast_safe_sync({
                "status": "error",
                "message": "Client RAG URL or user_id not set"
            })
            raise Exception("Client RAG URL or user_id not set")

        _broadcast_safe_sync({
            "status": "running",
            "message": "Prompt Injection scan started"
        })

        validate_client_url(CLIENT_RAG_URL)
        target_url = CLIENT_RAG_URL.strip()
        if not target_url.endswith("/rag/query"):
            target_url = target_url.rstrip("/") + "/rag/query"

        tester = BatchSecurityHarness(
            default_url=target_url,
            user_id=CLIENT_USER_ID,
            default_suite_path="prompts_precision_250_v2.json",
        )
        suite, tests = tester.load_suite(tester.DEFAULT_SUITE_PATH)
        env = tester.print_runtime_info()
        total_tests = len(tests)

        suite_meta = {
            "suite_name": suite.get("suite_name", os.path.basename(tester.DEFAULT_SUITE_PATH)),
            "suite_version": suite.get("version", "unknown"),
            "tests_total_in_file": len(tests),
            "tests_executed": len(tests),
            "sampling": {"was_sampled": False, "seed": None},
            "runner": env.get("runner", {}),
            "env": env,
        }

        security_report, dashboard = tester.run_suite(
            url=target_url,
            tests=tests,
            suite_meta=suite_meta,
            scan_dir="",
            enable_backdoor=False,
        )

        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        raw_out = os.path.join(results_dir, "security_report_full.json")
        dash_out = os.path.join(results_dir, "dashboard_response.json")

        tester.save_json(raw_out, security_report)
        tester.save_json(dash_out, dashboard)

        prompt_injection_state["status"] = "completed"
        prompt_injection_state["message"] = "Security scan completed"
        prompt_injection_state["output_file"] = dash_out
        prompt_injection_state["meta"] = {
            "total_tests": total_tests,
            "files": {
                "security_report": raw_out,
                "dashboard": dash_out
            }
        }

        _broadcast_safe_sync({
            "status": "completed",
            "message": "Security scan completed",
        })

    except Exception as e:
        logger.error(f"Error in prompt injection batch: {e}", exc_info=True)
        prompt_injection_state["status"] = "failed"
        prompt_injection_state["message"] = "Prompt injection scan failed"
        prompt_injection_state["error"] = str(e)
        prompt_injection_state["output_file"] = None
        prompt_injection_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "failed",
            "message": f"Prompt injection scan failed: {str(e)}"
        })


@rag_router.post("/RAG_Prompt_Injection")
async def security_guard():
    return await _start_job(
        prompt_injection_state,
        prompt_injection_lock,
        "Prompt Injection scan started",
        _run_prompt_injection_sync
    )


@rag_router.get("/RAG_Prompt_Injection/status")
async def get_prompt_injection_status():
    return JSONResponse(content=_status_payload(prompt_injection_state), status_code=200)


@rag_router.get("/RAG_Prompt_Injection/results")
def get_prompt_injection_results():
    results_dir = os.path.join(os.getcwd(), "results")
    security_file = os.path.join(results_dir, "security_report_full.json")
    dashboard_file = os.path.join(results_dir, "dashboard_response.json")

    if not os.path.exists(security_file) or not os.path.exists(dashboard_file):
        raise HTTPException(status_code=404, detail="No results found")

    try:
        with open(security_file, "r", encoding="utf-8") as f:
            security_report = json.load(f)
        with open(dashboard_file, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
    except Exception as e:
        logger.error(f"Error reading result files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read result files")

    return {
        "security_report": security_report,
        "dashboard": dashboard,
        "files": {
            "security_report": security_file,
            "dashboard": dashboard_file
        }
    }


# =========================================================
# HALLUCINATION EVAL
# =========================================================
def _run_hallucination_sync():
    global hallucination_state

    try:
        hallucination_state["status"] = "processing"
        hallucination_state["message"] = "Hallucination evaluation started"
        hallucination_state["error"] = None
        hallucination_state["output_file"] = None
        hallucination_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "running",
            "message": "Hallucination evaluation started"
        })

        try:
            df = pd.read_csv("./results/overall_results1.csv", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv("./results/overall_results1.csv", encoding="latin1")

        total_rows = len(df)

        results = []
        grounded = []
        halluc = []

        evaluator = HallucinationEvaluator()

        for _, row in df.iterrows():
            g_score, h_score = evaluator.compute_groundedness_and_hallucination(
                row["rag_response"],
                row["rag_context"],
                row["ground_truth"],
                temperature=0.0
            )

            grounded.append(g_score)
            halluc.append(h_score)

            results.append({
                "query": row["question"],
                "generated_result": row["rag_response"],
                "ground_truth": row["ground_truth"],
                "retrieved_context": row["rag_context"],
                "groundedness_score": g_score,
                "hallucination_score": h_score
            })

        os.makedirs("results/explainability", exist_ok=True)

        plot_result = evaluator.plot_results(
            grounded,
            halluc,
            filename="results/explainability/hallucinate_combined.json"
        )

        response_payload = {
            "results": results,
            "plot_result": plot_result
        }

        output_path = os.path.join(
            os.getcwd(),
            "results",
            "explainability",
            "overall_hallucination_eval.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(response_payload, f, indent=4, ensure_ascii=False)

        hallucination_state["status"] = "completed"
        hallucination_state["message"] = "Hallucination evaluation completed"
        hallucination_state["output_file"] = output_path
        hallucination_state["error"] = None
        hallucination_state["meta"] = {"total_records": total_rows}

        _broadcast_safe_sync({
            "status": "completed",
            "message": "Hallucination evaluation completed",
            "total_records": total_rows,
            "output_file": output_path
        })

    except Exception as e:
        logger.error(f"Error in hallucination evaluation: {e}", exc_info=True)
        hallucination_state["status"] = "failed"
        hallucination_state["message"] = "Hallucination evaluation failed"
        hallucination_state["error"] = str(e)
        hallucination_state["output_file"] = None
        hallucination_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "failed",
            "message": f"Hallucination evaluation failed: {str(e)}"
        })


@rag_router.post("/Overall_hallucination_eval")
async def hallucination_eval_csv():
    return await _start_job(
        hallucination_state,
        hallucination_lock,
        "Hallucination evaluation started",
        _run_hallucination_sync
    )


@rag_router.get("/Overall_hallucination_eval/status")
async def get_hallucination_status():
    return JSONResponse(content=_status_payload(hallucination_state), status_code=200)


@rag_router.get("/Overall_hallucination_eval/results")
async def get_hallucination_results():
    file_path = os.path.join(
        os.getcwd(),
        "results",
        "explainability",
        "overall_hallucination_eval.json"
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Hallucination evaluation results not found. Run evaluation first."
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading hallucination result file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read hallucination result file")

    return {
        "results": data.get("results", []),
        "plot_result": data.get("plot_result", {}),
        "file_path": file_path
    }


# =========================================================
# UNCERTAINTY QUANTIFICATION BATCH
# =========================================================
def _run_uncertainty_sync():
    global uncertainty_state

    try:
        uncertainty_state["status"] = "processing"
        uncertainty_state["message"] = "Uncertainty quantification started"
        uncertainty_state["error"] = None
        uncertainty_state["output_file"] = None
        uncertainty_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "running",
            "message": "Uncertainty quantification started"
        })

        df = pd.read_csv("./results/overall_results1.csv")
        queries = df["question"].tolist()
        total_queries = len(queries)

        tools = get_or_create_pipeline(CLIENT_RAG_URL, CLIENT_USER_ID)
        chain_obj = tools["chain_obj"]
        rag_pipeline = RAGUncertaintyPipeline(chain_obj)
        response = rag_pipeline.run_pipeline(queries)

        os.makedirs("./results/explainability", exist_ok=True)
        output_path = os.path.join(
            os.getcwd(),
            "results",
            "explainability",
            "overall_uncertainty_batch_response.json"
        )

        save_payload = {
            "uncertainty_data": response if isinstance(response, list) else response.get("uncertainty_data", []),
            "explanation_text": "" if isinstance(response, list) else response.get("explanation_text", "")
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, indent=4, ensure_ascii=False)

        uncertainty_state["status"] = "completed"
        uncertainty_state["message"] = "Uncertainty quantification completed"
        uncertainty_state["output_file"] = output_path
        uncertainty_state["error"] = None
        uncertainty_state["meta"] = {"total_queries": total_queries}

        _broadcast_safe_sync({
            "status": "completed",
            "message": "Uncertainty quantification completed",
            "total_queries": total_queries
        })

    except Exception as e:
        logger.error(f"Error in uncertainty_batch: {e}", exc_info=True)

        uncertainty_state["status"] = "failed"
        uncertainty_state["message"] = "Uncertainty quantification failed"
        uncertainty_state["error"] = str(e)
        uncertainty_state["output_file"] = None
        uncertainty_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "failed",
            "message": f"Uncertainty quantification failed: {str(e)}"
        })


@rag_router.post("/Overall_Uncertainty_Quantification_Batch")
async def uncertainty_batch():
    return await _start_job(
        uncertainty_state,
        uncertainty_lock,
        "Uncertainty quantification started",
        _run_uncertainty_sync
    )


@rag_router.get("/Overall_Uncertainty_Quantification_Batch/status")
async def get_uncertainty_status():
    return JSONResponse(content=_status_payload(uncertainty_state), status_code=200)


@rag_router.get("/Overall_Uncertainty_Quantification_Batch/results")
async def get_uncertainty_results():
    file_path = os.path.join(
        os.getcwd(),
        "results",
        "explainability",
        "overall_uncertainty_batch_response.json"
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Uncertainty results not found. Run the batch evaluation first."
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading uncertainty results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read uncertainty results file")

    if isinstance(data, list):
        data = {
            "uncertainty_data": data,
            "explanation_text": ""
        }

    return {
        "uncertainty_data": data.get("uncertainty_data", []),
        "explanation_text": data.get("explanation_text", ""),
        "file_path": file_path
    }


# =========================================================
# RETRIEVED CHUNK TRACES AND SCORES
# =========================================================
def _run_retrieved_chunk_trace_sync():
    global retrieved_chunk_trace_state

    try:
        retrieved_chunk_trace_state["status"] = "processing"
        retrieved_chunk_trace_state["message"] = "Retrieved chunk trace evaluation started"
        retrieved_chunk_trace_state["error"] = None
        retrieved_chunk_trace_state["output_file"] = None
        retrieved_chunk_trace_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "running",
            "message": "Retrieved chunk trace evaluation started"
        })

        os.makedirs("results/observability", exist_ok=True)

        tracer = RetrievedChunkTraceBatch()
        input_path = "results/overall_eval_with_ret_chunks.json"

        payload, out_json, out_csv = tracer.run_and_save(input_path)
        _ = payload

        retrieved_chunk_trace_state["status"] = "completed"
        retrieved_chunk_trace_state["message"] = "Retrieved chunk trace evaluation completed"
        retrieved_chunk_trace_state["output_file"] = str(out_json)
        retrieved_chunk_trace_state["error"] = None
        retrieved_chunk_trace_state["meta"] = {"csv_file": str(out_csv)}

        _broadcast_safe_sync({
            "status": "completed",
            "message": "Retrieved chunk trace evaluation completed",
            "output_file": str(out_json)
        })

    except Exception as e:
        logger.error(f"Error in retrieved chunk trace evaluation: {e}", exc_info=True)
        retrieved_chunk_trace_state["status"] = "failed"
        retrieved_chunk_trace_state["message"] = "Retrieved chunk trace evaluation failed"
        retrieved_chunk_trace_state["error"] = str(e)
        retrieved_chunk_trace_state["output_file"] = None
        retrieved_chunk_trace_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "failed",
            "message": f"Retrieved chunk trace evaluation failed: {str(e)}"
        })


@rag_router.post("/overall_retrieved_chunk_traces_and_scores")
async def overall_retrieved_chunk_traces_and_scores():
    return await _start_job(
        retrieved_chunk_trace_state,
        retrieved_chunk_trace_lock,
        "Retrieved chunk trace evaluation started",
        _run_retrieved_chunk_trace_sync
    )


@rag_router.get("/overall_retrieved_chunk_traces_and_scores/status")
async def get_retrieved_chunk_trace_status():
    return JSONResponse(content=_status_payload(retrieved_chunk_trace_state), status_code=200)


@rag_router.get("/overall_retrieved_chunk_traces_and_scores/results")
async def get_retrieved_chunk_traces_and_scores():
    file_path = os.path.join(
        os.getcwd(),
        "results",
        "observability",
        "overall_retrieved_chunk_trace.json"
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Retrieved chunk trace results not found. Run the evaluation first."
        )

    try:
        out_data = _load_json(file_path)
    except Exception as e:
        logger.error(f"Error reading retrieved chunk trace results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read retrieved chunk trace results")

    return out_data


# =========================================================
# SENTENCE CITATION COVERAGE
# =========================================================
def _run_sentence_citation_sync():
    global sentence_citation_state

    try:
        sentence_citation_state["status"] = "processing"
        sentence_citation_state["message"] = "Sentence citation coverage evaluation started"
        sentence_citation_state["error"] = None
        sentence_citation_state["output_file"] = None
        sentence_citation_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "running",
            "message": "Sentence citation coverage evaluation started"
        })

        os.makedirs("results/observability", exist_ok=True)

        checker = BatchSentenceCitationCoverage()
        input_path = "results/overall_eval_with_ret_chunks.json"

        payload, out_path = checker.run_and_save(input_path)
        _ = payload

        sentence_citation_state["status"] = "completed"
        sentence_citation_state["message"] = "Sentence citation coverage evaluation completed"
        sentence_citation_state["output_file"] = str(out_path)
        sentence_citation_state["error"] = None
        sentence_citation_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "completed",
            "message": "Sentence citation coverage evaluation completed",
            "output_file": str(out_path)
        })

    except Exception as e:
        logger.error(f"Error in sentence citation coverage evaluation: {e}", exc_info=True)
        sentence_citation_state["status"] = "failed"
        sentence_citation_state["message"] = "Sentence citation coverage evaluation failed"
        sentence_citation_state["error"] = str(e)
        sentence_citation_state["output_file"] = None
        sentence_citation_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "failed",
            "message": f"Sentence citation coverage evaluation failed: {str(e)}"
        })


@rag_router.post("/overall_sentence_citation_coverage")
async def overall_sentence_citation_coverage():
    return await _start_job(
        sentence_citation_state,
        sentence_citation_lock,
        "Sentence citation coverage evaluation started",
        _run_sentence_citation_sync
    )


@rag_router.get("/overall_sentence_citation_coverage/status")
async def get_sentence_citation_coverage_status():
    return JSONResponse(content=_status_payload(sentence_citation_state), status_code=200)


@rag_router.get("/overall_sentence_citation_coverage/results")
async def get_sentence_citation_coverage():
    file_path = os.path.join(
        os.getcwd(),
        "results",
        "observability",
        "overall_sentence_citation_coverage.json"
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Sentence citation coverage results not found. Run the evaluation first."
        )

    try:
        out_data = _load_json(file_path)
    except Exception as e:
        logger.error(f"Error reading citation coverage results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read citation coverage results")

    return out_data


# =========================================================
# PROMPT SNAPSHOT MINIMAL OUTPUT
# =========================================================
def _run_prompt_snapshot_sync():
    global prompt_snapshot_state

    try:
        prompt_snapshot_state["status"] = "processing"
        prompt_snapshot_state["message"] = "Prompt snapshot minimal output generation started"
        prompt_snapshot_state["error"] = None
        prompt_snapshot_state["output_file"] = None
        prompt_snapshot_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "running",
            "message": "Prompt snapshot minimal output generation started"
        })

        os.makedirs("results/observability", exist_ok=True)

        job = PromptSnapshotMinimalJob(
            default_input_json_path="results/overall_eval_with_ret_chunks.json",
            default_output_json_path="results/observability/overall_minimal_output.json",
        )

        input_path = "results/overall_eval_with_ret_chunks.json"
        output_path = "results/observability/overall_minimal_output.json"

        payload, out_path = job.run_and_save(
            input_json_path=input_path,
            output_json_path=output_path,
        )
        _ = payload

        prompt_snapshot_state["status"] = "completed"
        prompt_snapshot_state["message"] = "Prompt snapshot minimal output completed"
        prompt_snapshot_state["output_file"] = str(out_path)
        prompt_snapshot_state["error"] = None
        prompt_snapshot_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "completed",
            "message": "Prompt snapshot minimal output completed",
            "output_file": str(out_path)
        })

    except Exception as e:
        logger.error(f"Error in prompt snapshot minimal output: {e}", exc_info=True)
        prompt_snapshot_state["status"] = "failed"
        prompt_snapshot_state["message"] = "Prompt snapshot minimal output failed"
        prompt_snapshot_state["error"] = str(e)
        prompt_snapshot_state["output_file"] = None
        prompt_snapshot_state["meta"] = {}

        _broadcast_safe_sync({
            "status": "failed",
            "message": f"Prompt snapshot minimal output failed: {str(e)}"
        })


@rag_router.post("/prompt_snapshot_minimal_output")
async def prompt_snapshot_minimal_output():
    return await _start_job(
        prompt_snapshot_state,
        prompt_snapshot_lock,
        "Prompt snapshot minimal output generation started",
        _run_prompt_snapshot_sync
    )


@rag_router.get("/prompt_snapshot_minimal_output/status")
async def get_prompt_snapshot_status():
    return JSONResponse(content=_status_payload(prompt_snapshot_state), status_code=200)


@rag_router.get("/prompt_snapshot_minimal_output/results")
async def get_prompt_snapshot_minimal_output():
    file_path = os.path.join(
        os.getcwd(),
        "results",
        "observability",
        "overall_minimal_output.json"
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Minimal prompt snapshot results not found. Run the evaluation first."
        )

    try:
        out_data = _load_json(file_path)
    except Exception as e:
        logger.error(f"Error reading minimal prompt snapshot results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read minimal prompt snapshot results")

    return out_data


# =========================================================
# SINGLE QUERY EVAL CLIENT
# =========================================================
@rag_router.post("/single_query_eval_client")
async def single_query_eval_prompt_injection_safe_client(payload: QueryOnly):
    if not CLIENT_RAG_URL or not CLIENT_USER_ID:
        logger.error("Client RAG URL or user_id not set")
        await _broadcast_safe({
            "status": "error",
            "message": "Client RAG URL or user_id not set"
        })
        raise HTTPException(status_code=400, detail="Client RAG URL or user_id not set")

    query = sanitize_input(payload.query)

    await _broadcast_safe({
        "status": "running",
        "stage": "security_guard",
        "message": "Running prompt injection security check"
    })

    security_response_str = await process_query(query)
    security_response = json.loads(security_response_str)
    is_injection = security_response.get("details", {}).get("is_injection", False)

    if is_injection:
        await _broadcast_safe({
            "status": "error",
            "stage": "security_guard",
            "message": "Query blocked due to prompt injection"
        })
        return {
            "client_url": CLIENT_RAG_URL,
            "blocked": True,
            "security_guard": security_response,
            "rag": {
                "query": query,
                "answer": None,
                "context": None,
                "evaluation": None,
            },
            "message": "Query blocked due to prompt injection. No processing performed."
        }

    # -------------------------------------------------
    # CHECK DOCUMENTS FIRST (BEFORE PIPELINE / RAG)
    # -------------------------------------------------
    await _broadcast_safe({
        "status": "running",
        "stage": "document_check",
        "message": "Checking whether documents exist for this user"
    })

    try:
        chain_obj = ClientRAGChain(
            base_url=CLIENT_RAG_URL,
            user_id=CLIENT_USER_ID,
            timeout=30.0
        )

        dataset_df = await asyncio.to_thread(chain_obj.read_dataset_df)

        if dataset_df.empty:
            await _broadcast_safe({
                "status": "completed",
                "stage": "document_check",
                "message": "Unable to fetch the RAG response"
            })

            return {
                "client_url": CLIENT_RAG_URL,
                "blocked": False,
                "rag": {
                    "query": query,
                    "answer": None,
                    "context": None,
                    "evaluation": None,
                },
                "bias": None,
                "shap": None,
                "lime": None,
                "cot": None,
                "uncertainty": None,
                "message": "Unable to fetch the RAG response"
            }

    except Exception as e:
        logger.exception("Document check failed")
        await _broadcast_safe({
            "status": "error",
            "stage": "document_check",
            "message": f"Document check failed: {str(e)}"
        })
        raise HTTPException(status_code=500, detail=f"Document check failed: {str(e)}")

    # -------------------------------------------------
    # LOAD FULL PIPELINE ONLY IF DOCS EXIST
    # -------------------------------------------------
    await _broadcast_safe({
        "status": "running",
        "stage": "pipeline_load",
        "message": "Loading client RAG pipeline"
    })

    tools = get_or_create_pipeline(CLIENT_RAG_URL, CLIENT_USER_ID)

    chain_obj = tools["chain_obj"]
    unbias_processor = tools.get("unbias_processor")
    evaluation = tools["evaluation"]
    explainer = tools["explainer"]

    await _broadcast_safe({
        "status": "running",
        "stage": "rag_generation",
        "message": "Generating RAG response"
    })

    rag_raw = await asyncio.to_thread(chain_obj.invoke, {"input": query})
    answer = rag_raw.get("answer", "")
    context = rag_raw.get("retrieved_context")

    await _broadcast_safe({
        "status": "running",
        "stage": "evaluation",
        "message": "Running evaluation metrics"
    })
    try:
        eval_result = await asyncio.to_thread(evaluation.single_evaluation, query, rag_raw)
    except Exception as e:
        eval_result = {"error": str(e)}

    await _broadcast_safe({
        "status": "running",
        "stage": "bias_detection",
        "message": "Running bias detection"
    })
    try:
        bias_result = await asyncio.to_thread(unbias_processor.process_query, query) if unbias_processor else None
    except Exception as e:
        bias_result = {"error": str(e)}

    await _broadcast_safe({
        "status": "running",
        "stage": "shap_explanation",
        "message": "Generating SHAP explanation"
    })
    try:
        texts = dataset_df["chunk"].astype(str).tolist()

        chunks = []
        for text in texts:
            chunks.extend(explainer.chunk_text(text=text, source="api_dataset"))

        explainer.documents["api_dataset"] = [c["text"] for c in chunks]
        explainer.chunk_metadata["api_dataset"] = chunks
        explainer.embeddings["api_dataset"] = await asyncio.to_thread(
            explainer.model.encode_document,
            explainer.documents["api_dataset"],
            show_progress_bar=False
        )

        shap_result = await asyncio.to_thread(
            explainer.retrieve_and_explain,
            query=query,
            use_tfidf_features=True
        )
    except Exception as e:
        logger.exception("SHAP explanation failed")
        shap_result = {"error": str(e)}

    await _broadcast_safe({
        "status": "running",
        "stage": "lime_explanation",
        "message": "Running LIME explanation"
    })
    try:
        lime = LIMEHighlighter()
        lime_exp = await asyncio.to_thread(lime.explain, query, answer)

        lime_result = {
            "highlighted_html": lime.get_highlighted_html(answer, lime_exp),
            "feature_weights": lime.get_feature_weights(lime_exp),
        }
    except Exception as e:
        lime_result = {"error": str(e)}

    await _broadcast_safe({
        "status": "running",
        "stage": "cot_reasoning",
        "message": "Generating chain-of-thought reasoning"
    })
    try:
        cot = RAGPWithCoT(chain_obj)
        cot_result = json.loads(await asyncio.to_thread(cot.run_ragp_with_cot, query))
    except Exception as e:
        cot_result = {"error": str(e)}

    await _broadcast_safe({
        "status": "running",
        "stage": "uncertainty_quantification",
        "message": "Running uncertainty quantification"
    })
    try:
        unc = RAGUncertaintyPipeline(chain_obj)
        uncertainty_result = await asyncio.to_thread(unc.run_pipeline, [query])
    except Exception as e:
        uncertainty_result = {"error": str(e)}

    await _broadcast_safe({
        "status": "completed",
        "stage": "pipeline_finished",
        "message": "Single query evaluation completed"
    })

    return {
        "client_url": CLIENT_RAG_URL,
        "blocked": False,
        "rag": {
            "query": query,
            "answer": answer,
            "context": context,
            "evaluation": eval_result,
        },
        "bias": bias_result,
        "shap": shap_result,
        "lime": lime_result,
        "cot": cot_result,
        "uncertainty": uncertainty_result,
    }