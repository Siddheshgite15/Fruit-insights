"""
Central configuration.
All paths and tunable defaults live here.
Override via environment variables or edit this file directly.
"""
import os
from pathlib import Path

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent   # fruit_shelf_life/
MODEL_DIR      = BASE_DIR / "models"
UPLOAD_DIR     = BASE_DIR / "uploads"
LOG_DIR        = BASE_DIR / "logs"
TEMPLATE_DIR   = BASE_DIR / "templates"
STATIC_DIR     = BASE_DIR / "static"

# ── Model file paths ──────────────────────────────────────────────────────────
RESNET_MODEL_PATH  = os.getenv("RESNET_MODEL_PATH",  str(MODEL_DIR / "resnet50_best.pth"))
LSTM_MODEL_PATH    = os.getenv("LSTM_MODEL_PATH",    str(MODEL_DIR / "lstm_shelf_life_final.pth"))
SCALER_PATH        = os.getenv("SCALER_PATH",        str(MODEL_DIR / "lstm_feature_scaler.pkl"))
LSTM_CONFIG_PATH   = os.getenv("LSTM_CONFIG_PATH",   str(MODEL_DIR / "lstm_config.json"))

# ── Device ────────────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── ResNet50 inference ────────────────────────────────────────────────────────
IMAGE_SIZE     = 224
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# ── LSTM deployment defaults ──────────────────────────────────────────────────
# day_no = 1 means "first observation of this fruit".
# If your system tracks the same fruit across multiple days, supply the real
# day number from your database/tracking layer.
DEFAULT_DAY_NO = int(os.getenv("DEFAULT_DAY_NO", "1"))

# ── Upload security ───────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES      = 10 * 1024 * 1024          # 10 MB
ALLOWED_IMAGE_TYPES   = {
    "image/jpeg", "image/png", "image/webp", "image/bmp",
    "image/x-png", "image/x-ms-bmp", "image/pjpeg",
    "application/octet-stream",
}
ALLOWED_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = str(LOG_DIR / "app.log")
