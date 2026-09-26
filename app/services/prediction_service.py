"""
End-to-end prediction pipeline orchestrator.

Flow
----
image_bytes + temperature + humidity + moisture
  │
  ├─► ResNet50 CNN
  │     └─► fruit_type, status, cnn_confidence, fresh_prob
  │
  ├─► Freshness Index (deterministic, not ML)
  │     └─► 10–100 if Fresh, 0–9.99 if Rotten
  │
  ├─► Image feature extraction (48 features)
  │     + temperature, humidity, moisture, day_no
  │     └─► 52 numeric features dict
  │
  ├─► LSTM input assembly
  │     ├─► scaler.transform(numeric_52) → scaled_52
  │     ├─► one_hot(fruit_type, 3)
  │     └─► [scaled_52 | one_hot_3] = feature_vector_55
  │
  ├─► LSTM (sequence_len=1, single-observation)
  │     └─► remaining_shelf_life_days
  │
  └─► PredictionResult
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

from app.features.image_features import build_numeric_feature_dict
from app.models.lstm_model import build_feature_vector, predict_shelf_life
from app.models.resnet_model import predict_resnet
from app.utils.preprocessing import compute_freshness_index

log = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    fruit_type:               str
    expiry_status:            str
    cnn_confidence:           float
    freshness_index:          float
    remaining_shelf_life_days: float
    temperature:              float
    humidity:                 float
    moisture:                 float
    day_no:                   int
    processing_time_ms:       float


def run_prediction(
    image_bytes: bytes,
    temperature: float,
    humidity: float,
    moisture: float,
    day_no: int,
    *,
    resnet_model,
    resnet_classes: list,
    lstm_model,
    lstm_config: dict,
    scaler,
    device,
) -> PredictionResult:
    """
    Execute the full pipeline for one request.

    Parameters
    ----------
    image_bytes    : raw image bytes from upload
    temperature    : float °C
    humidity       : float %
    moisture       : float %
    day_no         : int   (1 = first observation)
    resnet_model   : loaded nn.Module
    resnet_classes : list of class name strings
    lstm_model     : loaded ShelfLifeLSTM
    lstm_config    : dict loaded from lstm_shelf_life_final.pth
    scaler         : fitted StandardScaler
    device         : torch.device
    """
    t0 = time.perf_counter()

    # ── 1. CNN classification ─────────────────────────────────────────────────
    cnn_result = predict_resnet(resnet_model, resnet_classes, image_bytes, device)
    fruit_type    = cnn_result["fruit_type"]    # e.g. "Apple"
    status        = cnn_result["status"]        # "Fresh" | "Rotten"
    cnn_confidence = cnn_result["confidence"]   # 0–100
    log.info("CNN → %s %s (%.2f%%)", status, fruit_type, cnn_confidence)

    # ── 2. Freshness Index ────────────────────────────────────────────────────
    freshness_index = compute_freshness_index(status, cnn_confidence)
    log.info("Freshness Index → %.2f", freshness_index)

    # ── 3. Handcrafted feature extraction (48) + env (4) = 52 ────────────────
    numeric_dict = build_numeric_feature_dict(
        image_bytes=image_bytes,
        temperature=temperature,
        humidity=humidity,
        moisture=moisture,
        day_no=day_no,
    )

    # ── 4. Build LSTM feature vector [scaled_52 | one_hot_3] = 55 ────────────
    feature_vector = build_feature_vector(
        numeric_features=numeric_dict,
        fruit_type_str=fruit_type.lower(),
        numeric_feature_columns=lstm_config["numeric_feature_columns"],
        fruit_to_index=lstm_config["fruit_to_index"],
        scaler=scaler,
    )
    log.info("LSTM feature vector shape: %s", feature_vector.shape)

    # ── 5. LSTM → remaining shelf life ───────────────────────────────────────
    shelf_life = predict_shelf_life(lstm_model, feature_vector, device)
    log.info("Remaining shelf life → %.2f days", shelf_life)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return PredictionResult(
        fruit_type=fruit_type,
        expiry_status=status,
        cnn_confidence=round(cnn_confidence, 2),
        freshness_index=freshness_index,
        remaining_shelf_life_days=shelf_life,
        temperature=temperature,
        humidity=humidity,
        moisture=moisture,
        day_no=day_no,
        processing_time_ms=round(elapsed_ms, 1),
    )
