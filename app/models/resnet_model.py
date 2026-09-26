"""
ResNet50 inference module.

Architecture confirmed from checkpoint (resnet50_best.pth):
  - Standard ResNet50 backbone (pretrained ImageNet)
  - Custom head:
      model.fc = nn.Sequential(
          nn.Linear(2048, 512),   # fc.0
          nn.ReLU(),              # fc.1
          nn.Dropout(p=0.4),     # fc.2
          nn.Linear(512, 6),     # fc.3
      )
  - 6 classes (from checkpoint["class_names"]):
      ['fresh_apple', 'fresh_banana', 'fresh_mango',
       'rotten_apple', 'rotten_banana', 'rotten_mango']
  - Checkpoint format: dict with key "model_state_dict"
  - val_accuracy = 96.15 %

Inference transform matches val_test_transform from CNN_3.ipynb:
  Resize(256,256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)
"""

import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Inference transform = val/test transform from training notebook
_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Architecture ──────────────────────────────────────────────────────────────

def _build_resnet50(num_classes: int) -> nn.Module:
    """
    Rebuild the exact architecture used in training (CNN_3.ipynb Cell 22).

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes),
    )
    """
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features          # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes),
    )
    return model


# ── Loader ────────────────────────────────────────────────────────────────────

def load_resnet_model(checkpoint_path: str, device: torch.device):
    """
    Load the trained ResNet50 from checkpoint.

    Returns
    -------
    model        : nn.Module  (eval mode, frozen, on device)
    class_names  : list[str]  e.g. ['fresh_apple', ...]
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"ResNet50 checkpoint not found: {path}")

    log.info("Loading ResNet50 from %s …", path)
    ckpt = torch.load(str(path), map_location=device)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            "Unexpected checkpoint format — expected dict with 'model_state_dict'."
        )

    class_names: list = ckpt["class_names"]
    num_classes        = len(class_names)
    log.info("Classes (%d): %s", num_classes, class_names)

    model = _build_resnet50(num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    log.info(
        "ResNet50 loaded. val_accuracy=%.4f  device=%s",
        ckpt.get("val_accuracy", float("nan")),
        device,
    )
    return model, class_names


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _load_image(image_bytes: bytes) -> Image.Image:
    """Convert raw bytes to a PIL RGB image."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return img
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_resnet(
    model: nn.Module,
    class_names: list,
    image_bytes: bytes,
    device: torch.device,
) -> Dict:
    """
    Run ResNet50 classification on one image.

    Returns
    -------
    {
        "raw_class"   : str    e.g. "fresh_apple"
        "fruit_type"  : str    e.g. "Apple"
        "status"      : str    "Fresh" | "Rotten"
        "confidence"  : float  0–100  (probability of predicted class × 100)
        "probabilities": dict  {class_name: prob}
        "fresh_prob"  : float  0–1  (probability of the fresh-status class)
    }
    """
    img    = _load_image(image_bytes)
    tensor = _INFERENCE_TRANSFORM(img).unsqueeze(0).to(device)  # (1,3,224,224)

    with torch.no_grad():
        logits = model(tensor)                         # (1, 6)
        probs  = torch.softmax(logits, dim=1)[0]       # (6,)

    pred_idx    = int(probs.argmax())
    raw_class   = class_names[pred_idx]
    confidence  = float(probs[pred_idx]) * 100.0

    # Parse "fresh_apple" → fruit_type="Apple", status="Fresh"
    parts      = raw_class.split("_", 1)               # ["fresh", "apple"]
    status_raw = parts[0]                              # "fresh" | "rotten"
    fruit_raw  = parts[1] if len(parts) > 1 else raw_class
    fruit_type = fruit_raw.capitalize()
    status     = status_raw.capitalize()               # "Fresh" | "Rotten"

    # fresh_prob = probability of the corresponding fresh_{fruit} class
    fresh_class = f"fresh_{fruit_raw}"
    fresh_idx   = class_names.index(fresh_class) if fresh_class in class_names else None
    fresh_prob  = float(probs[fresh_idx]) if fresh_idx is not None else (
        float(probs[pred_idx]) if status == "Fresh" else 1.0 - float(probs[pred_idx])
    )

    prob_dict = {cn: round(float(p), 6) for cn, p in zip(class_names, probs)}

    return {
        "raw_class":     raw_class,
        "fruit_type":    fruit_type,
        "status":        status,
        "confidence":    round(confidence, 4),
        "probabilities": prob_dict,
        "fresh_prob":    round(fresh_prob, 6),
    }
