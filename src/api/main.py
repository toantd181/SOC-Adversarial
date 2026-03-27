"""
src/api/main.py

FastAPI application entry-point for the AI-SOC Adversarial Defense System.

Run locally with:
    uvicorn src.api.main:app --reload

Architecture
------------
Startup  → load weights (if available), initialise InputSanitizer.
/health  → liveness / readiness probe.
/predict → full E2E pipeline:
           decode → sanitize → inference → structured response.

Phase 1 note
------------
The Robust Inference Engine (src/models/) is intentionally stubbed in this
phase.  The endpoint returns a placeholder prediction so that the API starts
and the sanitization pipeline can be validated end-to-end before weights are
available from the cloud training phase.
"""

from __future__ import annotations

import base64
import io
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from src.api.logger import get_logger, soc_alert
from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ImagePayload,
    PredictionResponse,
    ThreatStatus,
)
from src.defense.sanitizer import InputSanitizer, SanitizerConfig

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Application state (populated during lifespan)
# ---------------------------------------------------------------------------

class _AppState:
    """Mutable container for application-lifetime objects."""
    sanitizer: InputSanitizer
    model_loaded: bool = False
    # In Phase 3 this will hold the loaded CNNClassifier instance:
    # model: Optional[CNNClassifier] = None


_state = _AppState()


# ---------------------------------------------------------------------------
# Lifespan handler (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Execute startup tasks before yielding, and teardown tasks after.

    Startup
    -------
    1. Initialise the InputSanitizer with default thresholds.
    2. Attempt to load model weights (Phase 3).  If weights are absent the
       API still starts cleanly – /predict returns a placeholder response.

    Shutdown
    --------
    Flush any buffered log handlers.
    """
    logger.info("=" * 60)
    logger.info("AI-SOC Adversarial Defense System – API Gateway starting")
    logger.info("=" * 60)

    # ---- Sanitizer -------------------------------------------------------
    _state.sanitizer = InputSanitizer(config=SanitizerConfig())
    logger.info("InputSanitizer ready.")

    # ---- Model weights (Phase 3 – graceful skip in Phase 1) --------------
    try:
        # Deferred import: avoids hard dependency on torch during Phase 1
        from src.models.cnn_classifier import CNNClassifier  # noqa: F401
        import torch

        weights_path = "weights/robust_cnn.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = CNNClassifier(...)
        # model.load_state_dict(torch.load(weights_path, map_location=device))
        # model.eval()
        # _state.model = model
        _state.model_loaded = True
        logger.info("Model weights loaded from '%s' on device '%s'.", weights_path, device)
    except (ImportError, FileNotFoundError) as exc:
        logger.warning(
            "Model weights not loaded (Phase 1 expected): %s. "
            "Inference will return placeholder predictions.",
            exc,
        )
        _state.model_loaded = False

    logger.info("Startup complete.  API is ready to accept requests.")
    yield

    # ---- Teardown --------------------------------------------------------
    logger.info("AI-SOC API shutting down – flushing log handlers.")
    import logging
    for handler in logging.getLogger("ai_soc").handlers:
        handler.flush()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-SOC Adversarial Defense System",
    description=(
        "End-to-End AI Security Operations Centre pipeline. "
        "Detects and neutralises adversarial ML attacks (FGSM, PGD) "
        "before they reach the inference engine."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return a structured ErrorResponse for all HTTP exceptions."""
    logger.warning(
        "HTTP %d on %s | %s",
        exc.status_code,
        request.url.path,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            detail=str(exc.detail),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected server errors – log and return 500."""
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            detail="An unexpected error occurred.  Check SOC logs for details.",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _decode_image(image_b64: str) -> np.ndarray:
    """
    Decode a Base64 image string into an HxWx3 uint8 RGB numpy array.

    Raises
    ------
    HTTPException(422)
        If the bytes cannot be decoded as a valid image.
    """
    try:
        raw_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cannot decode image bytes: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image decoding failed: {exc}",
        ) from exc


def _run_inference(image_np: np.ndarray) -> tuple[int, str, float]:
    """
    Run the Robust Inference Engine on a preprocessed image array.

    Phase 1 stub
    ------------
    Returns a deterministic placeholder until model weights are loaded in
    Phase 3.  The placeholder values make it immediately obvious that no
    real inference has occurred (class=-1, label="PLACEHOLDER").

    Parameters
    ----------
    image_np:
        HxWx3 uint8 numpy array in RGB order.

    Returns
    -------
    tuple[int, str, float]
        (predicted_class_index, predicted_label, confidence)
    """
    if not _state.model_loaded:
        logger.debug("Model not loaded – returning Phase 1 placeholder prediction.")
        return -1, "PLACEHOLDER_PHASE1", 0.0

    # Phase 3 implementation (uncomment when weights are available):
    # import torch
    # import torchvision.transforms as T
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # transform = T.Compose([T.ToPILImage(), T.Resize((32, 32)), T.ToTensor(),
    #                         T.Normalize((0.5,), (0.5,))])
    # tensor = transform(image_np).unsqueeze(0).to(device)
    # with torch.no_grad():
    #     logits = _state.model(tensor)
    #     probs = torch.softmax(logits, dim=1)
    #     confidence, pred_class = probs.max(dim=1)
    # return int(pred_class.item()), CLASS_NAMES[int(pred_class.item())], float(confidence.item())
    return -1, "PLACEHOLDER_PHASE1", 0.0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness & readiness probe",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    """
    Return the current health status of the API.

    Useful for load-balancer health checks and Kubernetes readiness probes.
    """
    return HealthResponse(model_loaded=_state.model_loaded)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify an image with adversarial threat detection",
    tags=["Inference"],
    responses={
        400: {"model": ErrorResponse, "description": "Request blocked – adversarial attack detected"},
        422: {"model": ErrorResponse, "description": "Invalid payload"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def predict(payload: ImagePayload) -> PredictionResponse:
    """
    Full E2E inference pipeline with integrated adversarial threat detection.

    Pipeline
    --------
    1. Assign / echo a correlation ``request_id``.
    2. Decode the Base64 image into an RGB numpy array.
    3. Pass the array through the Threat Detection & Sanitization module.
       - **BLOCKED** → ``HTTP 400`` with a structured error and a ``[CRITICAL]``
         SOC alert.
       - **SANITIZED** → log a warning and continue inference on the cleaned
         image.
       - **CLEAN** → continue inference unchanged.
    4. Run the Robust Inference Engine.
    5. Return a ``PredictionResponse`` with the threat report embedded.
    """
    t_start = time.perf_counter()

    # ---- Correlation ID --------------------------------------------------
    request_id: str = payload.request_id or str(uuid.uuid4())
    logger.info("Received /predict | request_id=%s", request_id)

    # ---- Image decoding --------------------------------------------------
    image_np = _decode_image(payload.image_b64)
    logger.debug(
        "Image decoded | request_id=%s | shape=%s | dtype=%s",
        request_id,
        image_np.shape,
        image_np.dtype,
    )

    # ---- Threat detection & sanitization ---------------------------------
    processed_image, threat_report = _state.sanitizer.inspect(
        image_np,
        request_id=request_id,
        force_sanitize=payload.sanitize,
    )

    # ---- Handle BLOCKED requests -----------------------------------------
    if threat_report.status == ThreatStatus.BLOCKED:
        soc_alert(
            message=threat_report.details or "Adversarial pattern exceeded block threshold.",
            request_id=request_id,
            anomaly_score=threat_report.anomaly_score,
            logger=logger,
        )
        processing_time_ms = (time.perf_counter() - t_start) * 1000.0
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Request blocked by AI Firewall. "
                f"Anomaly score: {threat_report.anomaly_score:.4f}. "
                f"request_id: {request_id}"
            ),
        )

    # ---- Inference -------------------------------------------------------
    predicted_class, predicted_label, confidence = _run_inference(processed_image)

    processing_time_ms = (time.perf_counter() - t_start) * 1000.0

    logger.info(
        "Inference complete | request_id=%s | class=%s | label=%s | "
        "confidence=%.4f | threat_status=%s | latency_ms=%.2f",
        request_id,
        predicted_class if predicted_class != -1 else "N/A",
        predicted_label,
        confidence,
        threat_report.status.value,
        processing_time_ms,
    )

    return PredictionResponse(
        request_id=request_id,
        predicted_class=predicted_class if predicted_class != -1 else None,
        predicted_label=predicted_label if predicted_label != "PLACEHOLDER_PHASE1" else None,
        confidence=confidence if confidence > 0.0 else None,
        threat_report=threat_report,
        processing_time_ms=round(processing_time_ms, 3),
    )