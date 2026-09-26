# 🍎 Fruit Freshness & Remaining Shelf-Life Prediction

Production-ready FastAPI application that classifies fruit freshness
and predicts remaining shelf life using a **ResNet50 CNN + LSTM** pipeline.

---

## 1. Project Overview

| Item | Detail |
|---|---|
| CNN | ResNet50, 6 classes, val_accuracy = 96.15 % |
| Classes | `fresh_apple`, `fresh_banana`, `fresh_mango`, `rotten_apple`, `rotten_banana`, `rotten_mango` |
| LSTM | ShelfLifeLSTM, input=55, hidden=64, layers=1, many-to-many |
| LSTM input | 52 numeric features (scaled) + 3 one-hot fruit = 55 |
| Scaler | StandardScaler, 52 features, from training data |
| Freshness Index | Deterministic heuristic (not ML output) |

---

## 2. Architecture

```
USER INPUT: image + temperature + humidity + moisture
      │
      ▼
ResNet50 CNN  (resnet50_best.pth)
      │
      ├─► fruit_type  (Apple / Banana / Mango)
      ├─► expiry_status  (Fresh / Rotten)
      └─► cnn_confidence  (0–100 %)
      │
      ▼
Freshness Index  [deterministic heuristic]
      Fresh  → 10 + (confidence/100) × 90       range: [10, 100]
      Rotten → (1 − confidence/100) × 9.99      range: [0,  9.99)
      │
      ▼
Image Feature Extraction  (48 features)
      RGB stats · HSV stats · Lab stats
      Color ratios · Pixel-color ratios
      GLCM texture · Surface deterioration
      Edge/spot · Shape
      + temperature + humidity + moisture + day_no = 52 numeric features
      │
      ▼
StandardScaler.transform(52 numeric)
      + one_hot(fruit_type, 3)
      = feature_vector (55,)
      │
      ▼
LSTM  [sequence_len=1, single-observation]  (lstm_shelf_life_final.pth)
      │
      ▼
remaining_shelf_life_days  (≥ 0)
      │
      ▼
JSON / HTML RESULT
```

### Important deployment note

The LSTM was trained on multi-day temporal sequences (many-to-many).
At deployment only **one image** is available, so the sequence length is 1.
The model still produces a valid prediction at that single timestep,
but without prior temporal context accuracy is reduced compared to a full
multi-day series. This is clearly labelled in the UI.

---

## 3. Installation

```bash
cd fruit_shelf_life

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

> **NumPy note:** `numpy==1.26.4` is pinned because PyTorch 2.4 and
> scikit-image are not yet compatible with NumPy 2.x.

---

## 4. Model Placement

Place all model files in the `models/` folder:

```
models/
├── resnet50_best.pth           ← ResNet50 checkpoint (dict with model_state_dict)
├── lstm_shelf_life_final.pth   ← LSTM checkpoint (dict with all metadata)
├── lstm_feature_scaler.pkl     ← fitted StandardScaler (52 features)
└── lstm_config.json            ← generated from checkpoint at setup
```

`lstm_config.json` is generated automatically if missing. To regenerate:

```bash
python -c "
import torch, json
ckpt = torch.load('models/lstm_shelf_life_final.pth', map_location='cpu')
cfg  = {k: ckpt[k] for k in ['input_size','hidden_size','num_layers','dropout',
        'numeric_feature_columns','fruit_types','fruit_to_index','target_column',
        'sequence_type','bidirectional','seed']}
json.dump(cfg, open('models/lstm_config.json','w'), indent=4)
print('Done')
"
```

---

## 5. Running Locally

```bash
python run.py
```

- Web UI:      http://localhost:8000/
- Health:      http://localhost:8000/health
- Swagger UI:  http://localhost:8000/docs
- JSON API:    POST http://localhost:8000/api/predict

---

## 6. API Endpoints

### `GET /`
Renders the upload form (HTML).

### `GET /health`
```json
{
  "status": "ok",
  "cnn_model": "loaded",
  "lstm_model": "loaded",
  "device": "cpu"
}
```

### `POST /predict`
Accepts `multipart/form-data`, returns HTML result page.

### `POST /api/predict`
Accepts `multipart/form-data`, returns JSON.

---

## 7. Input Format

| Field | Type | Description |
|---|---|---|
| `file` | image file | JPEG / PNG / WEBP / BMP, max 10 MB |
| `temperature` | float | Ambient temperature °C (−50 to 80) |
| `humidity` | float | Relative humidity % (0–100) |
| `moisture` | float | Fruit moisture % (0–100) |

---

## 8. Output Format

```json
{
  "fruit_type": "Apple",
  "expiry_status": "Fresh",
  "cnn_confidence": 94.25,
  "freshness_index": 94.83,
  "remaining_shelf_life_days": 4.72,
  "temperature": 25.0,
  "humidity": 60.0,
  "moisture": 45.0,
  "day_no": 1,
  "processing_time_ms": 142.3
}
```

---

## 9. CNN Inference

- Architecture: ResNet50 with custom head
  `Linear(2048→512) → ReLU → Dropout(0.4) → Linear(512→6)`
- Inference transform: `Resize(256,256) → CenterCrop(224) → ToTensor → ImageNet Normalize`
- Checkpoint key: `model_state_dict`
- Class names from checkpoint: `['fresh_apple','fresh_banana','fresh_mango','rotten_apple','rotten_banana','rotten_mango']`

---

## 10. Freshness Index Calculation

**Not an ML output.** A deterministic formula driven by CNN confidence:

```
Fresh  → freshness_index = 10 + (confidence / 100) × 90    → range [10, 100]
Rotten → freshness_index = (1 − confidence / 100) × 9.99   → range [0,  9.99)
```

Key guarantees:
- Fresh always ≥ 10
- Rotten always < 10
- Result clipped to [0, 100] and rounded to 2 dp
- CNN confidence and freshness index are **different** values

---

## 11. LSTM Inference

- Architecture: `LSTM(55→64) → Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→1)`
- Feature vector: `[scaled_52_numeric | one_hot_fruit_3]` = 55 dimensions
- Scaler applied to numeric 52 only; one-hot is appended unscaled
- Single-observation sequence: shape `(1, 1, 55)` — sequence_length=1
- Output: predicted remaining days at the current timestep

---

## 12. Feature Extraction (52 numeric)

| Group | Features |
|---|---|
| RGB stats | mean_R/G/B, std_R/G/B |
| HSV stats | mean_H/S/V, std_H/S/V |
| Lab stats | mean_L, mean_A, mean_Lab_B |
| Color ratios | R_G_ratio, G_B_ratio |
| Pixel ratios | yellow, green, red, orange, dark_spot |
| Color variation | color_variation |
| GLCM texture | contrast, dissimilarity, homogeneity, energy, correlation, ASM, entropy |
| Deterioration | dark_pixel_ratio, brown_pixel_ratio, abnormal_light_ratio, abnormal_surface_ratio |
| Edge/spot | edge_density, abnormal_region_count, largest_abnormal_region_ratio, spot_density, spot_size_mean, spot_size_std |
| Surface | surface_uniformity, surface_roughness |
| Shape | fruit_area_ratio, fruit_perimeter, aspect_ratio, circularity, extent, solidity |
| Environmental | temperature, humidity, moisture, day_no |

---

## 13. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: resnet50_best.pth` | Model not in `models/` | Copy `.pth` to `models/` |
| `FileNotFoundError: lstm_feature_scaler.pkl` | Scaler missing | Copy `.pkl` to `models/` |
| `LSTM input mismatch` | Checkpoint corrupted | Re-copy `lstm_shelf_life_final.pth` |
| `415 Unsupported` | Wrong image format | Use JPEG / PNG / WEBP / BMP |
| `422 Unprocessable` | Invalid numeric input | Check temperature/humidity/moisture ranges |
| `numpy` version error | NumPy 2.x incompatibility | `pip install "numpy<2"` |
| `torchvision` crash | torch/torchvision version mismatch | Install matching versions from requirements.txt |

---

## 14. Docker Deployment

```bash
# Build and start
docker-compose up --build

# Or run manually
docker build -t fruit-shelf-life-api .
docker run -p 8000:8000 fruit-shelf-life-api
```

---

## 15. curl Example

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@/path/to/apple.jpg" \
  -F "temperature=25" \
  -F "humidity=60" \
  -F "moisture=45"
```

Expected response:
```json
{
  "fruit_type": "Apple",
  "expiry_status": "Fresh",
  "cnn_confidence": 94.25,
  "freshness_index": 94.83,
  "remaining_shelf_life_days": 4.72,
  "temperature": 25.0,
  "humidity": 60.0,
  "moisture": 45.0,
  "day_no": 1,
  "processing_time_ms": 142.3
}
```

---

## Pipeline Summary

```
IMAGE + TEMPERATURE + HUMIDITY + MOISTURE
        │
        ▼  ResNet50 CNN
        ├─ fruit_type: Apple / Banana / Mango
        ├─ expiry_status: Fresh / Rotten
        └─ cnn_confidence: 0–100 %
        │
        ▼  Freshness Index (heuristic)
        └─ freshness_index: 0–100
           (Fresh ≥10, Rotten <10)
        │
        ▼  Image feature extraction (48) + env (4) = 52
        ▼  StandardScaler.transform(52)
        ▼  Append one_hot(fruit, 3)
        ▼  feature_vector: (55,)
        │
        ▼  LSTM — single timestep, seq=(1,1,55)
        └─ remaining_shelf_life_days: ≥ 0
        │
        ▼
        JSON RESPONSE / HTML RESULT PAGE
```
