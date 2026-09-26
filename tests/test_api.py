"""
Integration tests for the FastAPI endpoints.

All heavy dependencies (torch, torchvision, app.models.*) are stubbed at
the sys.modules level before app.main is ever imported, so these tests run
on any machine regardless of whether torch/torchvision are version-compatible.

The 14 freshness-logic tests (test_freshness.py) are the pure-Python suite
that runs unconditionally.
"""
import io
import sys
import types
import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


# ── Stub torch & torchvision before anything else touches them ────────────────

def _stub_torch_torchvision():
    """
    Insert lightweight stubs so that importing app.main never triggers
    the torchvision binary (which crashes on a torch version mismatch).
    """
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.device       = lambda *a, **k: MagicMock()
        torch_stub.cuda         = MagicMock()
        torch_stub.cuda.is_available = lambda: False
        torch_stub.no_grad      = MagicMock(return_value=MagicMock(__enter__=lambda s,*a: s, __exit__=lambda s,*a: None))
        torch_stub.load         = MagicMock(return_value={})
        torch_stub.tensor       = MagicMock()
        torch_stub.nn           = types.ModuleType("torch.nn")
        torch_stub.nn.Module    = object
        torch_stub.nn.Linear    = MagicMock()
        torch_stub.nn.ReLU      = MagicMock()
        torch_stub.nn.Dropout   = MagicMock()
        torch_stub.nn.Sequential= MagicMock()
        torch_stub.nn.LSTM      = MagicMock()
        torch_stub.nn.utils     = MagicMock()
        sys.modules["torch"]    = torch_stub
        sys.modules["torch.nn"] = torch_stub.nn

    for mod in ["torchvision", "torchvision.models", "torchvision.transforms"]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

_stub_torch_torchvision()


# ── Now safe to import app ─────────────────────────────────────────────────────

# Stub the model loader modules too, so app.main doesn't run load_resnet_model etc.
_resnet_stub = types.ModuleType("app.models.resnet_model")
_resnet_stub.load_resnet_model = MagicMock()
_resnet_stub.predict_resnet    = MagicMock()
sys.modules["app.models.resnet_model"] = _resnet_stub

_lstm_stub = types.ModuleType("app.models.lstm_model")
_lstm_stub.load_lstm_model  = MagicMock()
_lstm_stub.load_scaler      = MagicMock()
_lstm_stub.ShelfLifeLSTM    = MagicMock()
_lstm_stub.build_feature_vector = MagicMock()
_lstm_stub.predict_shelf_life   = MagicMock()
sys.modules["app.models.lstm_model"] = _lstm_stub


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_png(r: int = 200, g: int = 180, b: int = 0) -> bytes:
    img = Image.fromarray(
        np.full((64, 64, 3), [r, g, b], dtype=np.uint8), "RGB"
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


MOCK_RESULT = MagicMock(
    fruit_type="Apple",
    expiry_status="Fresh",
    cnn_confidence=94.25,
    freshness_index=94.83,
    remaining_shelf_life_days=4.72,
    temperature=25.0,
    humidity=60.0,
    moisture=45.0,
    day_no=1,
    processing_time_ms=123.4,
)


# ── Client fixture ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Build a TestClient with lifespan model-loading fully mocked out.
    Uses sys.modules stubs above so torchvision is never imported.
    """
    # Patch the loader functions that lifespan calls
    _resnet_stub.load_resnet_model.return_value = (
        MagicMock(),
        ["fresh_apple", "fresh_banana", "fresh_mango",
         "rotten_apple", "rotten_banana", "rotten_mango"],
    )

    mock_ckpt = {
        "input_size": 55,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.2,
        "numeric_feature_columns": [f"feat_{i}" for i in range(52)],
        "fruit_types": ["apple", "banana", "mango"],
        "fruit_to_index": {"apple": 0, "banana": 1, "mango": 2},
        "target_column": "Remaining Shelf Life",
        "sequence_type": "many_to_many",
        "bidirectional": False,
        "seed": 42,
        "test_mae": 0.94,
        "test_rmse": 1.61,
        "test_r2": 0.31,
    }
    _lstm_stub.load_lstm_model.return_value = (MagicMock(), mock_ckpt)

    scaler_mock = MagicMock()
    scaler_mock.n_features_in_ = 52
    scaler_mock.transform.side_effect = lambda x: x
    _lstm_stub.load_scaler.return_value = scaler_mock

    from app.main import app
    from fastapi.testclient import TestClient

    # starlette 0.27 signature: TestClient(app, base_url, raise_server_exceptions)
    # newer versions use keyword arg; pass positionally for broadest compatibility
    with TestClient(app) as c:
        yield c


# ── Health ─────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_has_model_fields(self, client):
        data = client.get("/health").json()
        assert "cnn_model"  in data
        assert "lstm_model" in data
        assert "device"     in data


# ── Index page ─────────────────────────────────────────────────────────────────

class TestIndex:
    def test_200(self, client):
        assert client.get("/").status_code == 200

    def test_html_content(self, client):
        r = client.get("/")
        assert "text/html" in r.headers["content-type"]

    def test_contains_form(self, client):
        r = client.get("/")
        assert b"predict-form" in r.content or b"/predict" in r.content


# ── POST /api/predict (JSON) ───────────────────────────────────────────────────

class TestApiPredict:

    def _post(self, client, img=None, temperature="25", humidity="60",
              moisture="45", mime="image/png"):
        if img is None:
            img = _make_png()
        files = {"file": ("fruit.png", img, mime)}
        data  = {"temperature": temperature, "humidity": humidity, "moisture": moisture}
        return client.post("/api/predict", files=files, data=data)

    def test_200_with_mock(self, client):
        with patch("app.main.run_prediction", return_value=MOCK_RESULT):
            r = self._post(client)
        assert r.status_code == 200

    def test_response_fields(self, client):
        with patch("app.main.run_prediction", return_value=MOCK_RESULT):
            data = self._post(client).json()
        required = {
            "fruit_type", "expiry_status", "cnn_confidence",
            "freshness_index", "remaining_shelf_life_days",
            "temperature", "humidity", "moisture", "day_no",
        }
        assert required.issubset(data.keys())

    def test_response_values(self, client):
        with patch("app.main.run_prediction", return_value=MOCK_RESULT):
            data = self._post(client).json()
        assert data["fruit_type"]     == "Apple"
        assert data["expiry_status"]  == "Fresh"
        assert data["cnn_confidence"] == 94.25

    def test_unsupported_mime_415(self, client):
        r = self._post(client, mime="image/gif")
        assert r.status_code == 415

    def test_empty_file_rejected(self, client):
        files = {"file": ("empty.png", b"", "image/png")}
        data  = {"temperature": "25", "humidity": "60", "moisture": "45"}
        r = client.post("/api/predict", files=files, data=data)
        # empty file triggers validate_image_upload → 415
        assert r.status_code in (400, 415)

    def test_missing_temperature_422(self, client):
        files = {"file": ("f.png", _make_png(), "image/png")}
        data  = {"humidity": "60", "moisture": "45"}   # no temperature
        r = client.post("/api/predict", files=files, data=data)
        assert r.status_code == 422

    def test_invalid_humidity_422(self, client):
        with patch("app.main.run_prediction", return_value=MOCK_RESULT):
            r = self._post(client, humidity="150")
        assert r.status_code == 422

    def test_invalid_temperature_422(self, client):
        with patch("app.main.run_prediction", return_value=MOCK_RESULT):
            r = self._post(client, temperature="999")
        assert r.status_code == 422

    def test_negative_moisture_422(self, client):
        with patch("app.main.run_prediction", return_value=MOCK_RESULT):
            r = self._post(client, moisture="-5")
        assert r.status_code == 422


# ── POST /predict (HTML result page) ──────────────────────────────────────────

class TestPredictHTML:

    def test_returns_html(self, client):
        files = {"file": ("f.png", _make_png(), "image/png")}
        data  = {"temperature": "25", "humidity": "60", "moisture": "45"}
        with patch("app.main.run_prediction", return_value=MOCK_RESULT):
            r = client.post("/predict", files=files, data=data)
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_result_contains_fruit_type(self, client):
        files = {"file": ("f.png", _make_png(), "image/png")}
        data  = {"temperature": "25", "humidity": "60", "moisture": "45"}
        with patch("app.main.run_prediction", return_value=MOCK_RESULT):
            r = client.post("/predict", files=files, data=data)
        assert b"Apple" in r.content

    def test_result_contains_status(self, client):
        files = {"file": ("f.png", _make_png(), "image/png")}
        data  = {"temperature": "25", "humidity": "60", "moisture": "45"}
        with patch("app.main.run_prediction", return_value=MOCK_RESULT):
            r = client.post("/predict", files=files, data=data)
        assert b"FRESH" in r.content or b"Fresh" in r.content
