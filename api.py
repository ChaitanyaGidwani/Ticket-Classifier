"""FastAPI application for Support Ticket Classifier.

Provides high-throughput REST endpoints alongside the Streamlit UI:
  - Single ticket classification with multi-issue decomposition
  - Batch ticket classification
  - Semantic duplicate pair detection
  - Dynamic threshold configuration
  - Persistent duplicate store history management
  - Performance evaluation endpoint
  - Hugging Face model metadata & health checks

Note on Rate Limiting:
  Recommended reverse-proxy rate limiting (e.g., via Nginx or Cloudflare):
  100 requests/minute per client IP for /classify; 20 requests/minute for /batch.
  No external or paid LLM APIs are used — inference is 100% self-hosted and CPU-friendly.
"""
from contextlib import asynccontextmanager
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import uvicorn

from classifier import HUGGINGFACE_MODEL_NAME, LABELS, TicketClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticket_classifier_api")


# ---------- Pydantic Schemas ----------
class IssueDetail(BaseModel):
    segment: str = Field(..., description="Segment text of the extracted sub-issue")
    label: str = Field(..., description="Predicted label for the segment")
    score: float = Field(..., description="Cosine similarity score for this segment")


class DuplicateDetail(BaseModel):
    ticket: str = Field(..., description="Previously seen ticket text")
    similarity: float = Field(..., description="Cosine similarity with the current ticket")


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000, description="Customer message or support ticket text (max 20,000 characters)")
    check_duplicates: bool = Field(True, description="Whether to check and record this ticket in the persistent duplicate store")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "I was charged twice this month. Also, the app keeps crashing on launch.",
                "check_duplicates": True,
            }
        }
    }


class ClassifyResponse(BaseModel):
    text: str
    primary: str
    labels: List[str]
    scores: Dict[str, float]
    confidence: Dict[str, float]
    calibrated_confidence: Optional[float] = Field(None, description="Platt-scaled empirical accuracy estimate based on margin and score")
    margin: float
    uncertain: bool
    uncertain_reason: str
    issues: List[IssueDetail]
    urgent: bool
    urgent_reason: str
    duplicates: List[DuplicateDetail]


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=1000, description="List of ticket texts to classify (max 1,000 per batch)")
    check_duplicates: bool = Field(False, description="Whether to match/store tickets against the persistent duplicate store")
    detect_batch_duplicates: bool = Field(True, description="Whether to discover duplicate pairs within this submitted batch")

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "I was charged twice this month.",
                    "The app crashes every time I open it.",
                    "My card was charged two times for my subscription.",
                ],
                "check_duplicates": False,
                "detect_batch_duplicates": True,
            }
        }
    }


class BatchDuplicatePair(BaseModel):
    index_a: int
    index_b: int
    ticket_a: str
    ticket_b: str
    similarity: float


class BatchClassifyResponse(BaseModel):
    total: int
    results: List[ClassifyResponse]
    duplicate_pairs: List[BatchDuplicatePair]


class DuplicateSearchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=2, max_length=1000, description="List of tickets to compare for duplicates")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional custom duplicate similarity threshold (defaults to classifier dup_threshold)")


class DuplicateSearchResponse(BaseModel):
    total_candidates: int
    threshold_used: float
    duplicate_pairs: List[BatchDuplicatePair]


class ThresholdsConfig(BaseModel):
    label_threshold: float = Field(..., ge=0.0, le=1.0, description="Min similarity to count a label")
    uncertain_below: float = Field(..., ge=0.0, le=1.0, description="Best score below this flags uncertainty")
    margin_threshold: float = Field(..., ge=0.0, le=1.0, description="Top1-top2 score gap below this flags uncertainty")
    urgent_threshold: float = Field(..., ge=0.0, le=1.0, description="Similarity to urgent prototypes to trigger urgent flag")
    dup_threshold: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity threshold for duplicates")
    temperature: float = Field(..., gt=0.0, description="Softmax temperature for confidence calibration")


class ModelInfoResponse(BaseModel):
    status: str
    model_name: str
    huggingface_repo: str
    architecture: str
    backend: str
    hidden_dimension: int
    total_parameters: int
    device: str
    supported_labels: List[str]
    stored_tickets_count: int
    store_capacity: int
    thresholds: ThresholdsConfig


# ---------- Lifespan & App Setup ----------
classifier_instance: Optional[TicketClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier_instance
    logger.info("Initializing TicketClassifier singleton...")
    classifier_instance = TicketClassifier()
    app.state.classifier = classifier_instance
    logger.info("TicketClassifier initialized successfully.")
    yield
    classifier_instance = None


app = FastAPI(
    title="Support Ticket Classifier API",
    description=(
        "Production-ready, lightweight support ticket classifier powered directly by "
        f"Hugging Face transformers (`{HUGGINGFACE_MODEL_NAME}`). "
        "Provides fast CPU inference, multi-issue segmentation, urgency detection, "
        "persistent semantic duplicate store, and confidence threshold tuning."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.2f}ms)")
    return response


def get_clf() -> TicketClassifier:
    global classifier_instance
    if classifier_instance is None:
        classifier_instance = TicketClassifier()
    return classifier_instance


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy and pandas types to native python primitives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, pd.DataFrame):
        return sanitize_for_json(obj.to_dict(orient="records"))
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


# ---------- Endpoints ----------
@app.get("/", tags=["General"])
def root():
    clf = get_clf()
    return {
        "service": "Support Ticket Classifier API",
        "huggingface_model": clf.hf_model_repo,
        "docs_url": "/docs",
        "health_url": "/health",
        "labels": clf.labels,
        "rate_limiting": "Recommended: 100 req/min per IP at reverse proxy level",
    }


@app.get("/health", response_model=ModelInfoResponse, tags=["General"])
def health():
    clf = get_clf()
    info = clf.get_model_info()
    return ModelInfoResponse(
        status="healthy",
        model_name=clf.model_name,
        huggingface_repo=info["huggingface_repo"],
        architecture=info["architecture"],
        backend=info.get("backend", "PyTorch"),
        hidden_dimension=info["hidden_dimension"],
        total_parameters=info["total_parameters"],
        device=info["device"],
        supported_labels=clf.labels,
        stored_tickets_count=len(clf.store_text),
        store_capacity=info.get("store_capacity", 10000),
        thresholds=ThresholdsConfig(
            label_threshold=clf.label_threshold,
            uncertain_below=clf.uncertain_below,
            margin_threshold=clf.margin_threshold,
            urgent_threshold=clf.urgent_threshold,
            dup_threshold=clf.dup_threshold,
            temperature=clf.temperature,
        ),
    )


@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
def classify_ticket(req: ClassifyRequest):
    clf = get_clf()
    cleaned = req.text.strip()
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ticket text cannot be empty or blank.",
        )
    result = clf.classify(cleaned, check_duplicates=req.check_duplicates)
    return result


@app.post("/batch", response_model=BatchClassifyResponse, tags=["Classification"])
def batch_classify(req: BatchClassifyRequest):
    clf = get_clf()
    filtered_texts = [t.strip() for t in req.texts if t and t.strip()]
    if not filtered_texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid non-empty ticket texts provided in batch.",
        )

    results = [clf.classify(t, check_duplicates=req.check_duplicates) for t in filtered_texts]

    dup_pairs: List[BatchDuplicatePair] = []
    if req.detect_batch_duplicates and len(filtered_texts) >= 2:
        pairs = clf.find_duplicate_pairs(filtered_texts)
        for i, j, sim in pairs:
            dup_pairs.append(
                BatchDuplicatePair(
                    index_a=i,
                    index_b=j,
                    ticket_a=filtered_texts[i],
                    ticket_b=filtered_texts[j],
                    similarity=round(sim, 4),
                )
            )

    return BatchClassifyResponse(
        total=len(results),
        results=results,
        duplicate_pairs=dup_pairs,
    )


@app.post("/duplicates", response_model=DuplicateSearchResponse, tags=["Duplicates"])
def find_duplicates(req: DuplicateSearchRequest):
    clf = get_clf()
    filtered_texts = [t.strip() for t in req.texts if t and t.strip()]
    if len(filtered_texts) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 non-empty tickets are required to search for duplicates.",
        )

    orig_threshold = clf.dup_threshold
    threshold_used = req.threshold if req.threshold is not None else orig_threshold
    try:
        clf.dup_threshold = threshold_used
        pairs = clf.find_duplicate_pairs(filtered_texts)
    finally:
        clf.dup_threshold = orig_threshold

    dup_pairs = [
        BatchDuplicatePair(
            index_a=i,
            index_b=j,
            ticket_a=filtered_texts[i],
            ticket_b=filtered_texts[j],
            similarity=round(sim, 4),
        )
        for i, j, sim in pairs
    ]
    return DuplicateSearchResponse(
        total_candidates=len(filtered_texts),
        threshold_used=threshold_used,
        duplicate_pairs=dup_pairs,
    )


@app.get("/thresholds", response_model=ThresholdsConfig, tags=["Configuration"])
def get_thresholds():
    clf = get_clf()
    return ThresholdsConfig(
        label_threshold=clf.label_threshold,
        uncertain_below=clf.uncertain_below,
        margin_threshold=clf.margin_threshold,
        urgent_threshold=clf.urgent_threshold,
        dup_threshold=clf.dup_threshold,
        temperature=clf.temperature,
    )


@app.put("/thresholds", response_model=ThresholdsConfig, tags=["Configuration"])
def update_thresholds(cfg: ThresholdsConfig):
    clf = get_clf()
    clf.label_threshold = cfg.label_threshold
    clf.uncertain_below = cfg.uncertain_below
    clf.margin_threshold = cfg.margin_threshold
    clf.urgent_threshold = cfg.urgent_threshold
    clf.dup_threshold = cfg.dup_threshold
    clf.temperature = cfg.temperature
    return cfg


@app.post("/history/reset", tags=["Configuration"])
def reset_history():
    clf = get_clf()
    clf.reset_store()
    return {"message": "Ticket duplicate history cleared successfully.", "stored_tickets_count": 0}


@app.get("/evaluate", tags=["Evaluation"])
def evaluate():
    from evaluate import run_eval
    clf = get_clf()
    eval_result = run_eval(clf)
    if "report" in eval_result and isinstance(eval_result["report"], pd.DataFrame):
        eval_result["report"] = eval_result["report"].to_dict(orient="index")
    return sanitize_for_json(eval_result)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
