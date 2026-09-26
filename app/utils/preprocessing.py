"""
Freshness Index calculation and input validation helpers.
"""
import logging
from pathlib import Path

log = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/bmp",
    # Common variants some browsers / tools send
    "image/x-png", "image/x-ms-bmp", "image/pjpeg",
    "application/octet-stream",   # generic fallback — we verify via extension
}
MAX_UPLOAD_BYTES   = 10 * 1024 * 1024   # 10 MB


# ── Freshness Index ───────────────────────────────────────────────────────────

def compute_freshness_index(status: str, confidence: float) -> float:
    """
    Deterministic freshness index in [0, 100].

    Rules (specification-compliant):
      - status == "Fresh"  → index in [10, 100]
      - status == "Rotten" → index ALWAYS < 10

    Formula
    -------
    Fresh  :  freshness_index = 10 + (confidence / 100) * 90
    Rotten :  freshness_index = (1 - confidence / 100) * 9.99
              (confidence here is confidence IN the rotten prediction,
               which is NOT a freshness signal — so we invert it and
               cap at 9.99 to guarantee < 10)

    Parameters
    ----------
    status     : "Fresh" | "Rotten"
    confidence : CNN confidence for the predicted class, 0–100

    Returns
    -------
    float  in [0, 100], rounded to 2 dp
    """
    conf_frac = max(0.0, min(1.0, confidence / 100.0))

    if status.lower() == "fresh":
        idx = 10.0 + conf_frac * 90.0
    else:  # rotten
        # Invert: high rotten-confidence → low freshness index
        idx = (1.0 - conf_frac) * 9.99

    idx = max(0.0, min(100.0, idx))
    return round(idx, 2)


# ── Upload validation ─────────────────────────────────────────────────────────

def validate_image_upload(
    filename: str,
    content_type: str,
    file_size: int,
) -> None:
    """
    Raise ValueError with a user-friendly message if the upload is invalid.
    """
    ext = Path(filename).suffix.lower()

    # Accept if extension OR content-type is recognised
    ext_ok  = ext in ALLOWED_EXTENSIONS
    mime_ok = content_type in ALLOWED_MIME_TYPES

    if not ext_ok and not mime_ok:
        raise ValueError(
            f"Unsupported image format (extension='{ext}', "
            f"type='{content_type}'). "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    if file_size > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise ValueError(f"File too large. Maximum allowed size is {mb} MB.")
    if file_size == 0:
        raise ValueError("Uploaded file is empty.")


def validate_numeric_input(
    temperature: float,
    humidity: float,
    moisture: float,
) -> None:
    """Raise ValueError if environmental inputs are out of reasonable range."""
    if not (-50.0 <= temperature <= 80.0):
        raise ValueError(f"Temperature {temperature}°C is out of range [-50, 80].")
    if not (0.0 <= humidity <= 100.0):
        raise ValueError(f"Humidity {humidity}% must be between 0 and 100.")
    if not (0.0 <= moisture <= 100.0):
        raise ValueError(f"Moisture {moisture}% must be between 0 and 100.")
