"""Quick diagnostic: load the LSTM and inspect what it outputs for sample inputs."""
import sys, torch, joblib, json, numpy as np
from pathlib import Path

MODEL_DIR = Path("models")
device = torch.device("cpu")

# --- Load checkpoint ---
ckpt = torch.load(str(MODEL_DIR / "lstm_shelf_life_final.pth"), map_location=device, weights_only=False)
print("=== Checkpoint keys ===")
for k, v in ckpt.items():
    if k != "model_state_dict":
        print(f"  {k}: {v}")

# --- Rebuild model ---
sys.path.insert(0, ".")
from app.models.lstm_model import ShelfLifeLSTM

model = ShelfLifeLSTM(
    input_size=ckpt["input_size"],
    hidden_size=ckpt["hidden_size"],
    num_layers=ckpt["num_layers"],
    dropout=ckpt["dropout"],
    bidirectional=ckpt.get("bidirectional", False),
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# --- Load scaler ---
scaler = joblib.load(str(MODEL_DIR / "lstm_feature_scaler.pkl"))
print(f"\n=== Scaler ===")
print(f"  n_features: {scaler.n_features_in_}")
print(f"  means (first 5): {scaler.mean_[:5]}")
print(f"  scales (first 5): {scaler.scale_[:5]}")

# --- Test 1: all-zeros input ---
with torch.no_grad():
    x_zeros = torch.zeros(1, 1, 55)
    out_zeros = model(x_zeros)
    print(f"\n=== Test 1: all-zeros input ===")
    print(f"  Raw output: {float(out_zeros.squeeze()):.6f}")

# --- Test 2: random normal input ---
with torch.no_grad():
    x_rand = torch.randn(1, 1, 55)
    out_rand = model(x_rand)
    print(f"\n=== Test 2: random-normal input ===")
    print(f"  Raw output: {float(out_rand.squeeze()):.6f}")

# --- Test 3: simulate a "typical" fresh fruit ---
# Build a plausible feature vector: scale zeros (mean fruit), set fruit = apple
numeric_cols = ckpt.get("numeric_feature_columns", [])
fruit_to_idx = ckpt.get("fruit_to_index", {})
n_numeric = len(numeric_cols)
n_fruits  = len(fruit_to_idx)

# Use scaler mean as "average" numeric values, then scale → zeros
avg_scaled = np.zeros(n_numeric, dtype=np.float32)   # scaler(mean) = 0
one_hot = np.zeros(n_fruits, dtype=np.float32)
one_hot[0] = 1.0   # apple
fv = np.concatenate([avg_scaled, one_hot])

with torch.no_grad():
    x_avg = torch.tensor(fv, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    out_avg = model(x_avg)
    print(f"\n=== Test 3: average-fruit (scaled=0, apple one-hot) ===")
    print(f"  Raw output: {float(out_avg.squeeze()):.6f}")
    print(f"  After max(0, round): {max(0.0, round(float(out_avg.squeeze()), 2))}")

# --- Test 4: try several day_no values ---
print(f"\n=== Test 4: varying day_no (index of day_no in columns) ===")
if "day_no" in numeric_cols:
    day_idx = numeric_cols.index("day_no")
    print(f"  day_no is at index {day_idx} in numeric_feature_columns")
    print(f"  scaler mean for day_no: {scaler.mean_[day_idx]:.4f}")
    print(f"  scaler scale for day_no: {scaler.scale_[day_idx]:.4f}")
    
    for day in [1, 3, 5, 7, 10]:
        scaled_day = (day - scaler.mean_[day_idx]) / scaler.scale_[day_idx]
        test_fv = fv.copy()
        test_fv[day_idx] = scaled_day
        with torch.no_grad():
            x = torch.tensor(test_fv, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            out = model(x)
            raw = float(out.squeeze())
            print(f"  day_no={day:2d} -> scaled={scaled_day:+.4f} -> raw_output={raw:+.6f} -> clipped={max(0.0, round(raw, 2))}")

# --- Test 5: longer sequence ---
print(f"\n=== Test 5: multi-step sequence (seq_len=5) ===")
with torch.no_grad():
    x_seq = torch.zeros(1, 5, 55)
    out_seq = model(x_seq)
    print(f"  Output shape: {out_seq.shape}")
    print(f"  Raw outputs per step: {[round(float(v), 6) for v in out_seq.squeeze()]}")
