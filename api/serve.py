"""
api/serve.py — FastAPI inference endpoint for Social Sentiment Tracker.

Exposes three endpoints:
  GET  /health          — liveness check
  POST /predict         — single text prediction
  POST /predict/batch   — batch prediction (up to 128 texts)

The baseline model is always loaded (no GPU required).
BERT is loaded opportunistically if the checkpoint exists.

Usage:
    pip install fastapi uvicorn
    uvicorn api.serve:app --host 0.0.0.0 --port 8000

    # Single prediction
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"text": "I absolutely love this product!"}'

    # Batch prediction
    curl -X POST http://localhost:8000/predict/batch \\
         -H "Content-Type: application/json" \\
         -d '{"texts": ["Great!", "Terrible.", "It is okay."], "model": "baseline"}'
"""

import sys
import time
from pathlib import Path
from typing import List

# Make project root importable when called via `uvicorn api.serve:app`
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from config import get_logger, set_seed

logger = get_logger(__name__)
set_seed()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Social Sentiment Tracker API",
    description=(
        "REST API for three-class sentiment analysis (Negative / Positive / Neutral).\n\n"
        "Two models are available:\n"
        "- **baseline** — TF-IDF + Logistic Regression (always available, CPU-only)\n"
        "- **bert**     — Fine-tuned bert-base-uncased (requires trained checkpoint)"
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

LABEL_MAP = {0: "Negative", 1: "Positive", 2: "Neutral"}
MAX_BATCH = 128


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048,
                      description="Input text to classify.")
    model: str = Field("baseline",
                       description="Model to use: 'baseline' or 'bert'.")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in ("baseline", "bert"):
            raise ValueError("model must be 'baseline' or 'bert'")
        return v


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    label: int
    confidence: float
    model_used: str
    latency_ms: float


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH,
                             description=f"List of texts (max {MAX_BATCH}).")
    model: str = Field("baseline",
                       description="Model to use: 'baseline' or 'bert'.")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in ("baseline", "bert"):
            raise ValueError("model must be 'baseline' or 'bert'")
        return v

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        if len(v) > MAX_BATCH:
            raise ValueError(f"Maximum {MAX_BATCH} texts per request")
        return [t.strip() for t in v if t.strip()]


class BatchResponse(BaseModel):
    results: List[PredictResponse]
    model_used: str
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    version: str


# ---------------------------------------------------------------------------
# Model state (loaded once at startup)
# ---------------------------------------------------------------------------

_baseline_pipeline = None
_bert_model = None
_bert_tokenizer = None


@app.on_event("startup")
async def load_models() -> None:
    global _baseline_pipeline, _bert_model, _bert_tokenizer

    # Load baseline (required)
    try:
        from src.baseline_model import load_baseline_model
        _baseline_pipeline = load_baseline_model()
        logger.info("Baseline model loaded.")
    except FileNotFoundError:
        logger.warning(
            "Baseline model checkpoint not found. "
            "Run: python scripts/train_full.py --model baseline"
        )

    # Load BERT (optional)
    try:
        from src.bert_model import load_bert_model
        _bert_model, _bert_tokenizer = load_bert_model()
        logger.info("BERT model loaded.")
    except (FileNotFoundError, RuntimeError) as exc:
        logger.info("BERT model not available (%s). Baseline-only mode.", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    from src.data_loader import clean_text
    return clean_text(text)


def _run_baseline(texts: List[str]):
    if _baseline_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Baseline model not loaded. Run scripts/train_full.py --model baseline first.",
        )
    from src.baseline_model import predict
    labels, probs = predict(_baseline_pipeline, texts)
    confidences = probs.max(axis=1)
    return labels, confidences


def _run_bert(texts: List[str]):
    if _bert_model is None:
        raise HTTPException(
            status_code=503,
            detail="BERT model not loaded. Run scripts/train_full.py --model bert first.",
        )
    from src.bert_model import predict_bert
    labels, probs = predict_bert(_bert_model, _bert_tokenizer, texts)
    confidences = probs.max(axis=1)
    return labels, confidences


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Liveness check — returns which models are ready."""
    loaded = []
    if _baseline_pipeline is not None:
        loaded.append("baseline")
    if _bert_model is not None:
        loaded.append("bert")
    return HealthResponse(
        status="ok" if loaded else "degraded",
        models_loaded=loaded,
        version=app.version,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_single(req: PredictRequest) -> PredictResponse:
    """Classify a single text as Negative / Positive / Neutral.

    Returns the predicted sentiment, its integer label (0/1/2),
    confidence score, and inference latency.
    """
    t0 = time.perf_counter()

    cleaned = _clean(req.text)
    if not cleaned:
        raise HTTPException(status_code=422, detail="Text is empty after cleaning.")

    if req.model == "baseline":
        labels, confs = _run_baseline([cleaned])
    else:
        labels, confs = _run_bert([cleaned])

    latency_ms = (time.perf_counter() - t0) * 1000
    label = int(labels[0])
    return PredictResponse(
        text=req.text,
        sentiment=LABEL_MAP[label],
        label=label,
        confidence=round(float(confs[0]), 4),
        model_used=req.model,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(req: BatchRequest) -> BatchResponse:
    """Classify up to 128 texts in a single request.

    Useful for offline scoring or pipeline integration.
    Returns per-text results plus total batch latency.
    """
    if not req.texts:
        raise HTTPException(status_code=422, detail="texts list is empty.")

    t0 = time.perf_counter()

    cleaned = [_clean(t) for t in req.texts]
    # Replace empty-after-cleaning texts with a placeholder to preserve index alignment
    cleaned_safe = [c if c else "empty" for c in cleaned]

    if req.model == "baseline":
        labels, confs = _run_baseline(cleaned_safe)
    else:
        labels, confs = _run_bert(cleaned_safe)

    total_ms = (time.perf_counter() - t0) * 1000
    per_ms = total_ms / len(req.texts)

    results = [
        PredictResponse(
            text=orig,
            sentiment=LABEL_MAP[int(lbl)],
            label=int(lbl),
            confidence=round(float(conf), 4),
            model_used=req.model,
            latency_ms=round(per_ms, 2),
        )
        for orig, lbl, conf in zip(req.texts, labels, confs)
    ]

    return BatchResponse(
        results=results,
        model_used=req.model,
        total_latency_ms=round(total_ms, 2),
    )
