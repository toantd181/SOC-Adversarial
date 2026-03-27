"""
src/defense/sanitizer.py

Threat Detection & Sanitization module for the AI-SOC pipeline.

Threat detection strategy
--------------------------
Adversarial perturbations (FGSM / PGD) concentrate energy in high-frequency
pixel-space components that are largely invisible to the human eye but
statistically anomalous.  This module characterises that signal through three
complementary detectors:

1. **High-frequency energy ratio (FFT-based)**
   The discrete 2-D Fourier transform separates low- and high-frequency
   components.  Clean natural images have most energy in low frequencies;
   adversarial noise raises the high-frequency ratio measurably.

2. **Local gradient magnitude statistics**
   Adversarial perturbations produce atypically uniform, small-magnitude
   gradients across the *entire* image.  The Sobel operator estimates pixel
   gradients; their 90th-percentile / mean ratio serves as a texture-
   regularity score.

3. **Pixel-value distribution kurtosis**
   FGSM-perturbed images exhibit excess kurtosis in the pixel-value histogram
   because perturbation clips values towards {0, 255}.

The three sub-scores are linearly combined into a single ``anomaly_score``
∈ [0, 1].  Configurable thresholds in ``src/config.py`` (or the inline
defaults below) govern the SANITIZE vs BLOCK decision.

Sanitization methods
---------------------
* **Gaussian blur**       – attenuates high-frequency adversarial noise.
* **Median filter**       – robust to salt-and-pepper / pixel-flip attacks.
* **JPEG re-compression** – lossy codec naturally removes imperceptible noise.

All three preserve semantic content while disrupting the precise pixel values
needed to fool the classifier.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.api.logger import get_logger
from src.api.schemas import SanitizationMethod, ThreatReport, ThreatStatus

logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds (override via src/config.py in production)
# ---------------------------------------------------------------------------

_DEFAULT_SANITIZE_THRESHOLD: float = 0.45   # anomaly_score ≥ this → sanitize
_DEFAULT_BLOCK_THRESHOLD: float = 0.80      # anomaly_score ≥ this → block
_DEFAULT_JPEG_QUALITY: int = 75             # JPEG re-compression quality
_DEFAULT_GAUSSIAN_KERNEL: int = 5           # Gaussian blur kernel size (odd)
_DEFAULT_MEDIAN_KERNEL: int = 3             # Median filter kernel size (odd)

# Weights for the three sub-detectors (must sum to 1.0)
_W_FFT: float = 0.45
_W_GRADIENT: float = 0.35
_W_KURTOSIS: float = 0.20


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _DetectorScores:
    """Intermediate per-detector scores before aggregation."""
    fft_score: float = 0.0
    gradient_score: float = 0.0
    kurtosis_score: float = 0.0

    @property
    def combined(self) -> float:
        return (
            _W_FFT * self.fft_score
            + _W_GRADIENT * self.gradient_score
            + _W_KURTOSIS * self.kurtosis_score
        )


# ---------------------------------------------------------------------------
# Detector implementations
# ---------------------------------------------------------------------------

def _compute_fft_score(gray: np.ndarray) -> float:
    """
    Compute a normalised high-frequency energy ratio via 2-D FFT.

    Parameters
    ----------
    gray:
        Single-channel uint8 image of shape (H, W).

    Returns
    -------
    float
        Score in [0, 1]; higher means more high-frequency energy (more
        anomalous).
    """
    # Float conversion and mean-centering for cleaner FFT
    f = np.fft.fft2(gray.astype(np.float64))
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Define a circular low-frequency mask with radius = min(H,W) // 8
    radius = min(h, w) // 8
    y_idx, x_idx = np.ogrid[:h, :w]
    low_freq_mask = (y_idx - cy) ** 2 + (x_idx - cx) ** 2 <= radius ** 2

    total_energy = np.sum(magnitude) + 1e-9
    low_energy = np.sum(magnitude[low_freq_mask])
    high_energy = total_energy - low_energy

    # Natural images: high_energy / total_energy ≈ 0.05 – 0.25
    # Adversarial:                                 ≈ 0.35 – 0.65
    raw_ratio = high_energy / total_energy
    # Sigmoid-like normalisation: map [0, 0.6] → [0, 1]
    score = float(np.clip(raw_ratio / 0.6, 0.0, 1.0))
    return score


def _compute_gradient_score(gray: np.ndarray) -> float:
    """
    Detect anomalously *uniform* gradient fields via Sobel magnitude statistics.

    Adversarial perturbations add structured, nearly-uniform noise across the
    image, reducing the variability of local gradients.  We measure this as:

        score = 1 - (std(|∇|) / (mean(|∇|) + ε))

    A clean image has high gradient variability (score ≈ low).
    An adversarial image has low gradient variability (score ≈ high).

    Parameters
    ----------
    gray:
        Single-channel uint8 image of shape (H, W).

    Returns
    -------
    float
        Score in [0, 1].
    """
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    mean_mag = float(np.mean(magnitude))
    std_mag = float(np.std(magnitude))

    # Coefficient of variation (CV) – high for natural images, low for adversarial
    cv = std_mag / (mean_mag + 1e-9)

    # Map: CV ∈ [0, 5] → score ∈ [1, 0]  (inverted – lower CV = higher score)
    score = float(np.clip(1.0 - cv / 5.0, 0.0, 1.0))
    return score


def _compute_kurtosis_score(gray: np.ndarray) -> float:
    """
    Measure excess kurtosis of the pixel-value distribution.

    FGSM clips perturbed pixel values to [0, 255], creating a bimodal
    concentration at the extremes that raises histogram kurtosis.

    Parameters
    ----------
    gray:
        Single-channel uint8 image of shape (H, W).

    Returns
    -------
    float
        Score in [0, 1].
    """
    pixels = gray.flatten().astype(np.float64)
    mean = np.mean(pixels)
    std = np.std(pixels) + 1e-9
    # Fisher kurtosis (excess kurtosis; normal distribution ≈ 0)
    kurtosis = float(np.mean(((pixels - mean) / std) ** 4) - 3.0)

    # Natural images: kurtosis ≈ -1 to 2
    # Adversarial:    kurtosis ≈ 3  to 10+
    # Map [0, 10] → [0, 1], clamp negatives to 0
    score = float(np.clip(max(kurtosis, 0.0) / 10.0, 0.0, 1.0))
    return score


# ---------------------------------------------------------------------------
# Sanitization implementations
# ---------------------------------------------------------------------------

def _sanitize_gaussian(image_np: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply Gaussian blur to suppress high-frequency adversarial noise.

    Parameters
    ----------
    image_np:
        HxWxC uint8 numpy array (BGR or RGB; channel order is irrelevant here).
    kernel_size:
        Odd integer kernel size, e.g. 5 for a 5×5 kernel.

    Returns
    -------
    np.ndarray
        Blurred image as uint8 HxWxC array.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # enforce odd kernel
    # sigma=0 → OpenCV auto-computes sigma from kernel_size
    return cv2.GaussianBlur(image_np, (kernel_size, kernel_size), sigmaX=0)


def _sanitize_median(image_np: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply a median filter to suppress salt-and-pepper / pixel-flip attacks.

    Parameters
    ----------
    image_np:
        HxWxC uint8 numpy array.
    kernel_size:
        Odd integer aperture size for the median filter.

    Returns
    -------
    np.ndarray
        Filtered image as uint8 HxWxC array.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image_np, kernel_size)


def _sanitize_jpeg(image_np: np.ndarray, quality: int) -> np.ndarray:
    """
    Re-compress the image as JPEG to destroy adversarial perturbations via
    lossy quantisation.

    The JPEG codec's DCT-based quantisation naturally rounds small
    perturbation values to zero, acting as a structured low-pass filter.

    Parameters
    ----------
    image_np:
        HxWxC uint8 numpy array (RGB).
    quality:
        JPEG quality factor ∈ [1, 95].  Lower = more aggressive destruction
        of high-frequency content.  Recommended range: [60, 80].

    Returns
    -------
    np.ndarray
        Decoded JPEG image as uint8 HxWxC RGB array.
    """
    pil_img = Image.fromarray(image_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class SanitizerConfig:
    """
    Runtime configuration for the InputSanitizer.

    Attributes
    ----------
    sanitize_threshold:
        Anomaly scores at or above this value trigger sanitization.
    block_threshold:
        Anomaly scores at or above this value cause the request to be blocked.
    jpeg_quality:
        Quality factor for JPEG re-compression sanitization.
    gaussian_kernel:
        Kernel size for Gaussian blur sanitization.
    median_kernel:
        Kernel size for median filter sanitization.
    preferred_method:
        Which sanitization technique to apply when sanitization is triggered.
    """
    sanitize_threshold: float = _DEFAULT_SANITIZE_THRESHOLD
    block_threshold: float = _DEFAULT_BLOCK_THRESHOLD
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY
    gaussian_kernel: int = _DEFAULT_GAUSSIAN_KERNEL
    median_kernel: int = _DEFAULT_MEDIAN_KERNEL
    preferred_method: SanitizationMethod = SanitizationMethod.GAUSSIAN_BLUR


class InputSanitizer:
    """
    AI Firewall: inspects incoming image tensors for adversarial perturbations
    and either sanitizes or blocks the request.

    Usage::

        sanitizer = InputSanitizer(config=SanitizerConfig())
        clean_array, report = sanitizer.inspect(image_np)
        if report.status == ThreatStatus.BLOCKED:
            raise HTTPException(status_code=400, ...)
    """

    def __init__(self, config: Optional[SanitizerConfig] = None) -> None:
        self._cfg = config or SanitizerConfig()
        logger.info(
            "InputSanitizer initialised | sanitize_threshold=%.2f | "
            "block_threshold=%.2f | method=%s",
            self._cfg.sanitize_threshold,
            self._cfg.block_threshold,
            self._cfg.preferred_method.value,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_grayscale(self, image_np: np.ndarray) -> np.ndarray:
        """Convert HxWxC uint8 RGB array to single-channel grayscale."""
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    def _detect(self, image_np: np.ndarray) -> Tuple[float, _DetectorScores]:
        """
        Run all three sub-detectors and return the combined anomaly_score
        alongside the individual component scores.
        """
        gray = self._to_grayscale(image_np)

        scores = _DetectorScores(
            fft_score=_compute_fft_score(gray),
            gradient_score=_compute_gradient_score(gray),
            kurtosis_score=_compute_kurtosis_score(gray),
        )

        logger.debug(
            "Detector sub-scores | fft=%.4f | gradient=%.4f | kurtosis=%.4f | combined=%.4f",
            scores.fft_score,
            scores.gradient_score,
            scores.kurtosis_score,
            scores.combined,
        )
        return scores.combined, scores

    def _apply_sanitization(
        self, image_np: np.ndarray, method: SanitizationMethod
    ) -> np.ndarray:
        """Dispatch to the appropriate sanitization function."""
        if method == SanitizationMethod.GAUSSIAN_BLUR:
            return _sanitize_gaussian(image_np, self._cfg.gaussian_kernel)
        elif method == SanitizationMethod.MEDIAN_FILTER:
            return _sanitize_median(image_np, self._cfg.median_kernel)
        elif method == SanitizationMethod.JPEG_COMPRESSION:
            return _sanitize_jpeg(image_np, self._cfg.jpeg_quality)
        else:
            return image_np  # SanitizationMethod.NONE – pass-through

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def inspect(
        self,
        image_np: np.ndarray,
        *,
        request_id: Optional[str] = None,
        force_sanitize: bool = True,
    ) -> Tuple[np.ndarray, ThreatReport]:
        """
        Inspect an image array for adversarial perturbations.

        Parameters
        ----------
        image_np:
            HxWxC uint8 numpy array in RGB channel order.
        request_id:
            Correlation ID for log traceability.
        force_sanitize:
            When False the sanitizer will only *detect* anomalies and report
            them; no sanitization is applied.  Used for evaluation / red-team
            mode.

        Returns
        -------
        Tuple[np.ndarray, ThreatReport]
            * The (possibly sanitized) image array ready for inference.
            * A ThreatReport capturing the full threat assessment.
        """
        anomaly_score, sub_scores = self._detect(image_np)

        # ---- BLOCK -------------------------------------------------------
        if anomaly_score >= self._cfg.block_threshold:
            report = ThreatReport(
                status=ThreatStatus.BLOCKED,
                anomaly_score=round(anomaly_score, 4),
                sanitization_applied=SanitizationMethod.NONE,
                details=(
                    f"Anomaly score {anomaly_score:.4f} exceeds block threshold "
                    f"{self._cfg.block_threshold:.2f}. "
                    f"Sub-scores: FFT={sub_scores.fft_score:.3f}, "
                    f"Gradient={sub_scores.gradient_score:.3f}, "
                    f"Kurtosis={sub_scores.kurtosis_score:.3f}."
                ),
            )
            logger.error(
                "THREAT BLOCKED | request_id=%s | anomaly_score=%.4f",
                request_id or "N/A",
                anomaly_score,
            )
            return image_np, report

        # ---- SANITIZE ----------------------------------------------------
        if force_sanitize and anomaly_score >= self._cfg.sanitize_threshold:
            method = self._cfg.preferred_method
            sanitized = self._apply_sanitization(image_np, method)
            report = ThreatReport(
                status=ThreatStatus.SANITIZED,
                anomaly_score=round(anomaly_score, 4),
                sanitization_applied=method,
                details=(
                    f"Anomaly score {anomaly_score:.4f} exceeded sanitize "
                    f"threshold {self._cfg.sanitize_threshold:.2f}. "
                    f"Applied {method.value}."
                ),
            )
            logger.warning(
                "INPUT SANITIZED | request_id=%s | anomaly_score=%.4f | method=%s",
                request_id or "N/A",
                anomaly_score,
                method.value,
            )
            return sanitized, report

        # ---- CLEAN -------------------------------------------------------
        report = ThreatReport(
            status=ThreatStatus.CLEAN,
            anomaly_score=round(anomaly_score, 4),
            sanitization_applied=SanitizationMethod.NONE,
            details=f"Anomaly score {anomaly_score:.4f} is within normal bounds.",
        )
        logger.info(
            "INPUT CLEAN | request_id=%s | anomaly_score=%.4f",
            request_id or "N/A",
            anomaly_score,
        )
        return image_np, report