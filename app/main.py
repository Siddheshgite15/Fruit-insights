"""
Fruit Freshness & Remaining Shelf-Life Prediction API
=====================================================
FastAPI application entry-point.

Models are loaded ONCE at startup via the lifespan context and stored on
app.state. Every request reads from app.state — no per-request loading.
"""

import json
import logging
import logging.config
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import (
    DEVICE,
    LSTM_CONFIG_PATH,
    LSTM_MODEL_PATH,
    RESNET_MODEL_PATH,
    SCALER_PATH,
    STATIC_DIR,
    TEMPLATE_DIR,
    UPLOAD_DIR,
    DEFAULT_DAY_NO,
    LOG_FILE,
    LOG_LEVEL,
)
from app.models.resnet_model import load_resnet_model
from app.models.lstm_model import load_lstm_model, load_scaler
from app.services.prediction_service import run_prediction
from app.utils.preprocessing import (
    validate_image_upload,
    validate_numeric_input,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Ensure upload dir exists ──────────────────────────────────────────────────
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Lifespan (load models once at startup) ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("Starting Fruit Freshness & Shelf-Life API")
    log.info("Device: %s", DEVICE)

    # ResNet50
    resnet_model, class_names = load_resnet_model(RESNET_MODEL_PATH, DEVICE)
    app.state.resnet_model  = resnet_model
    app.state.class_names   = class_names

    # LSTM
    lstm_model, lstm_ckpt = load_lstm_model(LSTM_MODEL_PATH, DEVICE)
    app.state.lstm_model  = lstm_model
    app.state.lstm_config = lstm_ckpt   # full dict with feature metadata

    # Scaler
    scaler = load_scaler(SCALER_PATH)
    app.state.scaler = scaler

    # Validate dimensions at startup — fail fast rather than at first request
    expected_input = lstm_ckpt["input_size"]   # 55
    n_numeric      = len(lstm_ckpt["numeric_feature_columns"])   # 52
    n_fruits       = len(lstm_ckpt["fruit_types"])               # 3
    assert n_numeric + n_fruits == expected_input, (
        f"LSTM input mismatch: {n_numeric} numeric + {n_fruits} one-hot "
        f"= {n_numeric + n_fruits}, expected {expected_input}"
    )
    assert scaler.n_features_in_ == n_numeric, (
        f"Scaler expects {scaler.n_features_in_} features, "
        f"but numeric_feature_columns has {n_numeric}"
    )
    log.info("✓ LSTM input size validated: %d", expected_input)
    log.info("✓ Scaler features validated: %d", scaler.n_features_in_)
    log.info("✓ CNN classes: %s", class_names)
    log.info("All models loaded and validated. Accepting requests.")
    log.info("=" * 60)

    yield   # ← server is running

    log.info("Shutting down — releasing model resources.")
    del app.state.resnet_model
    del app.state.lstm_model
    del app.state.scaler


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fruit Freshness & Shelf-Life API",
    description=(
        "ResNet50 CNN + LSTM pipeline that predicts fruit freshness "
        "and remaining shelf life from a single image."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# ── Helper ────────────────────────────────────────────────────────────────────

def _safe_float(val, name: str) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail=f"'{name}' must be a valid number, got: {val!r}",
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def index(request: Request):
    """Render the upload form."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/health", tags=["Health"])
async def health(request: Request):
    """Liveness check — returns model load status."""
    return {
        "status":      "ok",
        "cnn_model":   "loaded" if hasattr(request.app.state, "resnet_model") else "not loaded",
        "lstm_model":  "loaded" if hasattr(request.app.state, "lstm_model")   else "not loaded",
        "device":      str(DEVICE),
    }


@app.post("/predict", tags=["Prediction"])
async def predict(
    request: Request,
    file:        UploadFile = File(...,  description="Fruit image (JPEG/PNG/WEBP/BMP)"),
    temperature: str        = Form(...,  description="Ambient temperature °C"),
    humidity:    str        = Form(...,  description="Relative humidity %"),
    moisture:    str        = Form(...,  description="Fruit moisture %"),
):
    """
    Run the full ResNet50 → Freshness Index → LSTM pipeline.

    Returns an HTML result page rendered by Jinja2.
    For raw JSON, call POST /api/predict with the same fields.
    """
    # ── Parse & validate env inputs ───────────────────────────────────────────
    temp_f = _safe_float(temperature, "temperature")
    hum_f  = _safe_float(humidity,    "humidity")
    mois_f = _safe_float(moisture,    "moisture")

    try:
        validate_numeric_input(temp_f, hum_f, mois_f)
    except ValueError as exc:
        return templates.TemplateResponse(
            request, "index.html",
            {"error": str(exc)},
            status_code=422,
        )

    # ── Read & validate image ─────────────────────────────────────────────────
    image_bytes = await file.read()
    try:
        validate_image_upload(file.filename or "upload", file.content_type or "", len(image_bytes))
    except ValueError as exc:
        return templates.TemplateResponse(
            request, "index.html",
            {"error": str(exc)},
            status_code=415,
        )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        result = run_prediction(
            image_bytes=image_bytes,
            temperature=temp_f,
            humidity=hum_f,
            moisture=mois_f,
            day_no=DEFAULT_DAY_NO,
            resnet_model=request.app.state.resnet_model,
            resnet_classes=request.app.state.class_names,
            lstm_model=request.app.state.lstm_model,
            lstm_config=request.app.state.lstm_config,
            scaler=request.app.state.scaler,
            device=DEVICE,
        )
    except ValueError as exc:
        log.warning("Prediction input error: %s", exc)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(exc)},
            status_code=422,
        )
    except Exception as exc:
        log.exception("Unexpected pipeline error")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "An internal error occurred. Please check server logs.",
            },
            status_code=500,
        )

    # ── Render result page ────────────────────────────────────────────────────
    import base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime      = file.content_type or "image/jpeg"

    return templates.TemplateResponse(
        "result.html",
        {
            "request":    request,
            "result":     result,
            "image_b64":  image_b64,
            "image_mime": mime,
        },
    )


@app.post("/api/predict", tags=["API"])
async def api_predict(
    request: Request,
    file:        UploadFile = File(...),
    temperature: str        = Form(...),
    humidity:    str        = Form(...),
    moisture:    str        = Form(...),
):
    """
    Same pipeline as POST /predict but returns raw JSON instead of HTML.
    Useful for programmatic access / curl.
    """
    temp_f = _safe_float(temperature, "temperature")
    hum_f  = _safe_float(humidity,    "humidity")
    mois_f = _safe_float(moisture,    "moisture")

    try:
        validate_numeric_input(temp_f, hum_f, mois_f)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    image_bytes = await file.read()
    try:
        validate_image_upload(file.filename or "upload", file.content_type or "", len(image_bytes))
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    try:
        result = run_prediction(
            image_bytes=image_bytes,
            temperature=temp_f,
            humidity=hum_f,
            moisture=mois_f,
            day_no=DEFAULT_DAY_NO,
            resnet_model=request.app.state.resnet_model,
            resnet_classes=request.app.state.class_names,
            lstm_model=request.app.state.lstm_model,
            lstm_config=request.app.state.lstm_config,
            scaler=request.app.state.scaler,
            device=DEVICE,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("API pipeline error")
        raise HTTPException(status_code=500, detail="Internal inference error.")

    return {
        "fruit_type":               result.fruit_type,
        "expiry_status":            result.expiry_status,
        "cnn_confidence":           result.cnn_confidence,
        "freshness_index":          result.freshness_index,
        "remaining_shelf_life_days": result.remaining_shelf_life_days,
        "temperature":              result.temperature,
        "humidity":                 result.humidity,
        "moisture":                 result.moisture,
        "day_no":                   result.day_no,
        "processing_time_ms":       result.processing_time_ms,
    }
