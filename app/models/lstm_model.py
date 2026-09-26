"""
LSTM shelf-life inference module.

Architecture confirmed from lstm_shelf_life_final.pth:
  - input_size  = 55  (52 numeric + 3 one-hot fruit)
  - hidden_size = 64
  - num_layers  = 1
  - dropout     = 0.2  (applied only when num_layers > 1, so effectively 0 here)
  - bidirectional = False
  - regressor:
        Linear(64 → 32) → ReLU → Dropout(0.2) → Linear(32 → 1)
  - sequence_type = many_to_many (returns prediction at every timestep)
  - For deployment (single image): sequence length = 1

Feature vector layout (per timestep, shape = 55):
  [ scaled_numeric_52 | one_hot_fruit_3 ]

Scaler: lstm_feature_scaler.pkl
  - StandardScaler fitted on 52 numeric features ONLY
  - One-hot fruit features are appended AFTER scaling (not scaled)

NOTE — mismatch between prompt and actual checkpoint:
  The prompt says "2-layer LSTM, hidden_size=128".
  The actual .pth has num_layers=1, hidden_size=64.
  Deployment uses the ACTUAL checkpoint values. This is documented here.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ── Architecture ──────────────────────────────────────────────────────────────

class ShelfLifeLSTM(nn.Module):
    """
    Exact architecture from LSTM_3.ipynb (Cell 32) matched to the
    actual checkpoint dimensions (hidden=64, layers=1).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Regressor confirmed from state_dict:
        #   regressor.0 : Linear(64, 32)
        #   regressor.3 : Linear(32, 1)
        #   (indices 0,3 because Sequential has ReLU@1 and Dropout@2)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """
        x       : (batch, seq_len, input_size)
        lengths : optional LongTensor of valid lengths per sample
        Returns : (batch, seq_len)  — prediction at every timestep
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            output, _     = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            output, _ = self.lstm(x)

        predictions = self.regressor(output)   # (batch, seq, 1)
        return predictions.squeeze(-1)         # (batch, seq)


# ── Loader ────────────────────────────────────────────────────────────────────

def load_lstm_model(model_path: str, device: torch.device):
    """
    Load LSTM model, scaler, and configuration from .pth checkpoint.

    Returns
    -------
    model   : ShelfLifeLSTM (eval, frozen, on device)
    config  : dict  (all metadata from checkpoint)
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found: {path}")

    log.info("Loading LSTM from %s …", path)
    ckpt = torch.load(str(path), map_location=device)

    model = ShelfLifeLSTM(
        input_size  = ckpt["input_size"],
        hidden_size = ckpt["hidden_size"],
        num_layers  = ckpt["num_layers"],
        dropout     = ckpt["dropout"],
        bidirectional = ckpt.get("bidirectional", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    log.info(
        "LSTM loaded. input=%d  hidden=%d  layers=%d  device=%s",
        ckpt["input_size"], ckpt["hidden_size"], ckpt["num_layers"], device,
    )
    return model, ckpt


def load_scaler(scaler_path: str):
    """Load the fitted StandardScaler from .pkl file."""
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}")
    scaler = joblib.load(str(path))
    log.info("Scaler loaded. n_features_in_=%d", scaler.n_features_in_)
    return scaler


# ── Feature vector assembly ───────────────────────────────────────────────────

def build_feature_vector(
    numeric_features: dict,
    fruit_type_str: str,
    numeric_feature_columns: list,
    fruit_to_index: dict,
    scaler,
) -> np.ndarray:
    """
    Assemble one timestep input vector of shape (55,).

    Layout (matches LSTM_3.ipynb Cell 22):
        [ scaled_numeric_52 | one_hot_fruit_3 ]

    Parameters
    ----------
    numeric_features        : dict mapping feature name → float value
    fruit_type_str          : e.g. "apple", "banana", "mango"  (lowercase)
    numeric_feature_columns : ordered list of 52 numeric column names
    fruit_to_index          : {"apple": 0, "banana": 1, "mango": 2}
    scaler                  : fitted StandardScaler (52 features)

    Returns
    -------
    np.ndarray of shape (55,) = float32
    """
    # ── 1. Build numeric array in exact column order ──────────────────────────
    n_numeric = len(numeric_feature_columns)
    numeric_arr = np.zeros(n_numeric, dtype=np.float32)
    missing = []
    for i, col in enumerate(numeric_feature_columns):
        if col in numeric_features:
            numeric_arr[i] = float(numeric_features[col])
        else:
            missing.append(col)

    if missing:
        log.warning("Missing features (will be 0.0): %s", missing)

    # ── 2. Scale numeric features ─────────────────────────────────────────────
    scaled = scaler.transform(numeric_arr.reshape(1, -1))[0].astype(np.float32)

    # ── 3. One-hot encode fruit type ──────────────────────────────────────────
    n_fruits   = len(fruit_to_index)
    one_hot    = np.zeros(n_fruits, dtype=np.float32)
    fruit_key  = fruit_type_str.lower().strip()
    if fruit_key in fruit_to_index:
        one_hot[fruit_to_index[fruit_key]] = 1.0
    else:
        log.warning("Unknown fruit type '%s'. One-hot will be all-zero.", fruit_type_str)

    # ── 4. Concatenate [scaled_numeric | one_hot] ─────────────────────────────
    feature_vector = np.concatenate([scaled, one_hot])  # (55,)
    assert feature_vector.shape[0] == n_numeric + n_fruits, (
        f"Feature vector length mismatch: expected {n_numeric + n_fruits}, "
        f"got {feature_vector.shape[0]}"
    )
    return feature_vector


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_shelf_life(
    model: ShelfLifeLSTM,
    feature_vector: np.ndarray,
    device: torch.device,
) -> float:
    """
    Predict remaining shelf life from a single-timestep feature vector.

    Single-image deployment strategy:
      - The LSTM was trained on multi-day sequences (many-to-many).
      - At deployment we have ONE observation (sequence_length = 1).
      - We feed a sequence of shape (1, 1, 55) — batch=1, seq=1, features=55.
      - The model returns a prediction at timestep 0, which is the current
        remaining shelf-life estimate.
      - This is an incomplete temporal sequence. Accuracy will be lower than
        a full multi-day sequence because the LSTM has no prior context.
      - Document this clearly to the user.

    Returns
    -------
    float — predicted remaining days, clipped to >= 0, rounded to 2 dp
    """
    assert feature_vector.shape == (55,), (
        f"Expected feature vector of length 55, got {feature_vector.shape}"
    )

    x      = torch.tensor(feature_vector, dtype=torch.float32)
    x      = x.unsqueeze(0).unsqueeze(0)          # (1, 1, 55)
    x      = x.to(device)

    with torch.no_grad():
        out = model(x)                             # (1, 1) → squeezed to (1,) or scalar

    pred = float(out.squeeze())
    pred = max(0.0, round(pred, 2))
    return pred
