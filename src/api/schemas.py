"""
src/api/schemas.py

Pydantic models for AI-SOC API request and response validation.
All fields are strictly typed and documented for production use.
"""

from __future__ import annotations

import base64
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ThreatStatus(str, Enum):
    """Describes the threat assessment result for an incoming request."""
    CLEAN = "CLEAN"
    SANITIZED = "SANITIZED"
    BLOCKED = "BLOCKED"


class SanitizationMethod(str, Enum):
    """Enumeration of supported sanitization techniques."""
    GAUSSIAN_BLUR = "gaussian_blur"
    MEDIAN_FILTER = "median_filter"
    JPEG_COMPRESSION = "jpeg_compression"
    NONE = "none"


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class ImagePayload(BaseModel):
    """
    Incoming image classification request.

    The image must be supplied as a Base64-encoded string.  The optional
    `request_id` field enables end-to-end tracing across SOC logs.
    """

    image_b64: str = Field(
        ...,
        description=(
            "Base64-encoded image bytes (JPEG / PNG). "
            "Must decode to a valid image buffer."
        ),
        min_length=4,
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Caller-supplied correlation ID for distributed tracing.",
        max_length=128,
    )
    sanitize: bool = Field(
        default=True,
        description=(
            "When True the Threat Detection module will attempt to sanitize "
            "suspicious inputs before inference.  Set to False to run raw "
            "inference (red-team / evaluation mode only)."
        ),
    )

    @field_validator("image_b64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Ensure the payload is valid Base64 before it reaches the pipeline."""
        try:
            decoded = base64.b64decode(v, validate=True)
        except Exception as exc:
            raise ValueError(
                "image_b64 is not valid Base64-encoded data."
            ) from exc

        # Minimal magic-byte check: JPEG (FF D8) or PNG (89 50 4E 47)
        jpeg_magic = b"\xff\xd8"
        png_magic = b"\x89PNG"
        if not (decoded[:2] == jpeg_magic or decoded[:4] == png_magic):
            raise ValueError(
                "Decoded bytes do not match a supported image format "
                "(expected JPEG or PNG magic bytes)."
            )
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "image_b64": "<base64-encoded-image-string>",
            "request_id": "req-2024-001",
            "sanitize": True,
        }
    }}


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class ThreatReport(BaseModel):
    """
    Structured output of the Threat Detection & Sanitization module.
    Embedded inside every PredictionResponse for full audit traceability.
    """

    status: ThreatStatus = Field(
        ...,
        description="Overall threat assessment outcome.",
    )
    anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Normalised anomaly score in [0, 1].  "
            "Higher values indicate stronger adversarial signal."
        ),
    )
    sanitization_applied: SanitizationMethod = Field(
        default=SanitizationMethod.NONE,
        description="Which sanitization technique was applied, if any.",
    )
    details: Optional[str] = Field(
        default=None,
        description="Human-readable description of the threat finding.",
    )


class PredictionResponse(BaseModel):
    """
    Full response envelope returned by the /predict endpoint.

    Contains the model's classification result *and* the SOC threat report
    so that callers and downstream SIEM systems receive a single, auditable
    payload.
    """

    request_id: Optional[str] = Field(
        default=None,
        description="Echo of the caller-supplied correlation ID.",
    )
    predicted_class: Optional[int] = Field(
        default=None,
        description=(
            "Zero-indexed class label predicted by the Robust Inference Engine. "
            "None when the request was BLOCKED."
        ),
    )
    predicted_label: Optional[str] = Field(
        default=None,
        description="Human-readable class name (e.g. 'Speed limit 50 km/h').",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Softmax confidence of the top prediction.",
    )
    threat_report: ThreatReport = Field(
        ...,
        description="Structured output from the Threat Detection module.",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total server-side processing time in milliseconds.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "request_id": "req-2024-001",
            "predicted_class": 14,
            "predicted_label": "Stop",
            "confidence": 0.9821,
            "threat_report": {
                "status": "SANITIZED",
                "anomaly_score": 0.73,
                "sanitization_applied": "gaussian_blur",
                "details": "High-frequency perturbation pattern detected; Gaussian blur applied.",
            },
            "processing_time_ms": 18.4,
        }
    }}


class HealthResponse(BaseModel):
    """Lightweight liveness/readiness probe response."""

    status: str = Field(default="ok")
    model_loaded: bool = Field(
        ...,
        description="True when the Robust Inference Engine has weights loaded.",
    )
    version: str = Field(
        default="1.0.0",
        description="API version string.",
    )


class ErrorResponse(BaseModel):
    """Standardised error envelope for 4xx / 5xx responses."""

    error: str = Field(..., description="Short error code or type.")
    detail: str = Field(..., description="Human-readable error description.")
    request_id: Optional[str] = Field(default=None)