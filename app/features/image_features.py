"""
Handcrafted image feature extraction.

Reproduces every feature used in the LSTM training CSV (LSTM_3.ipynb).
The NUMERIC_FEATURE_COLUMNS (52 features) are extracted here, then
temperature, humidity, moisture, and day_no are appended by the caller.

Feature groups
--------------
RGB statistics        : mean_R/G/B,  std_R/G/B
HSV statistics        : mean_H/S/V,  std_H/S/V
Lab statistics        : mean_L/A/Lab_B
Color ratios          : R_G_ratio, G_B_ratio
Pixel-color ratios    : yellow, green, red, orange, dark_spot
Color variation       : color_variation
GLCM texture          : contrast, dissimilarity, homogeneity, energy,
                        correlation, ASM, entropy
Surface deterioration : dark_pixel_ratio, brown_pixel_ratio,
                        abnormal_light_ratio, abnormal_surface_ratio
Edge / spot           : edge_density, abnormal_region_count,
                        largest_abnormal_region_ratio,
                        spot_density, spot_size_mean, spot_size_std
Surface stats         : surface_uniformity, surface_roughness
Shape                 : fruit_area_ratio, fruit_perimeter, aspect_ratio,
                        circularity, extent, solidity
"""

import io
import logging

import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

log = logging.getLogger(__name__)

# ── HSV thresholds ────────────────────────────────────────────────────────────
# These are tuned for common fruits under normal lighting.
# Do not change — they must match the training notebook.
_YELLOW_LOW  = np.array([20,  100,  100])
_YELLOW_HIGH = np.array([35,  255,  255])
_GREEN_LOW   = np.array([36,   50,   50])
_GREEN_HIGH  = np.array([89,  255,  255])
_RED_LOW1    = np.array([0,   100,  100])
_RED_HIGH1   = np.array([10,  255,  255])
_RED_LOW2    = np.array([160, 100,  100])
_RED_HIGH2   = np.array([180, 255,  255])
_ORANGE_LOW  = np.array([10,  100,  100])
_ORANGE_HIGH = np.array([20,  255,  255])
_DARK_LOW    = np.array([0,     0,    0])
_DARK_HIGH   = np.array([180, 255,   50])
_BROWN_LOW   = np.array([5,    50,   20])
_BROWN_HIGH  = np.array([20,  200,  200])
_ALIGHT_LOW  = np.array([0,     0,  230])
_ALIGHT_HIGH = np.array([180,  30,  255])
_ASURF_LOW   = np.array([0,     0,    0])
_ASURF_HIGH  = np.array([180, 255,   30])

# GLCM settings
_GLCM_DIST   = [1]
_GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bytes_to_arrays(image_bytes: bytes):
    """
    Returns
    -------
    img_rgb  : H×W×3  uint8  RGB
    img_bgr  : H×W×3  uint8  BGR  (for OpenCV)
    img_gray : H×W    uint8
    img_hsv  : H×W×3  uint8  HSV
    img_lab  : H×W×3  float32  Lab (opencv L*a*b*)
    """
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil = pil.resize((256, 256), Image.LANCZOS)     # consistent spatial size
    img_rgb  = np.array(pil, dtype=np.uint8)        # H×W×3
    img_bgr  = cv2.cvtColor(img_rgb,  cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr,  cv2.COLOR_BGR2GRAY)
    img_hsv  = cv2.cvtColor(img_bgr,  cv2.COLOR_BGR2HSV)
    img_lab  = cv2.cvtColor(img_bgr,  cv2.COLOR_BGR2LAB).astype(np.float32)
    return img_rgb, img_bgr, img_gray, img_hsv, img_lab


def _mask_ratio(mask: np.ndarray, total: int) -> float:
    return float(np.sum(mask > 0)) / total


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_image_features(image_bytes: bytes) -> dict:
    """
    Extract all 48 image-derived features.
    Temperature, humidity, moisture, and day_no are NOT added here —
    the caller appends them to produce the full 52-feature numeric vector.

    Returns
    -------
    dict with exactly 48 keys (matching the LSTM training CSV minus
    temperature, humidity, moisture, day_no).
    """
    img_rgb, img_bgr, img_gray, img_hsv, img_lab = _bytes_to_arrays(image_bytes)
    h, w   = img_gray.shape
    total  = h * w

    feats: dict = {}

    # ── RGB statistics ────────────────────────────────────────────────────────
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    feats["mean_R"] = float(np.mean(R))
    feats["mean_G"] = float(np.mean(G))
    feats["mean_B"] = float(np.mean(B))
    feats["std_R"]  = float(np.std(R))
    feats["std_G"]  = float(np.std(G))
    feats["std_B"]  = float(np.std(B))

    # ── HSV statistics ────────────────────────────────────────────────────────
    H, S, V = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    feats["mean_H"] = float(np.mean(H))
    feats["mean_S"] = float(np.mean(S))
    feats["mean_V"] = float(np.mean(V))
    feats["std_H"]  = float(np.std(H))
    feats["std_S"]  = float(np.std(S))
    feats["std_V"]  = float(np.std(V))

    # ── Lab statistics ────────────────────────────────────────────────────────
    feats["mean_L"]     = float(np.mean(img_lab[:, :, 0]))
    feats["mean_A"]     = float(np.mean(img_lab[:, :, 1]))
    feats["mean_Lab_B"] = float(np.mean(img_lab[:, :, 2]))

    # ── Color ratios ──────────────────────────────────────────────────────────
    eps = 1e-6
    feats["R_G_ratio"] = float(np.mean(R) / (np.mean(G) + eps))
    feats["G_B_ratio"] = float(np.mean(G) / (np.mean(B) + eps))

    # ── Pixel-color ratios (HSV-based) ────────────────────────────────────────
    yellow_mask = cv2.inRange(img_hsv, _YELLOW_LOW, _YELLOW_HIGH)
    green_mask  = cv2.inRange(img_hsv, _GREEN_LOW,  _GREEN_HIGH)
    red_mask1   = cv2.inRange(img_hsv, _RED_LOW1,   _RED_HIGH1)
    red_mask2   = cv2.inRange(img_hsv, _RED_LOW2,   _RED_HIGH2)
    red_mask    = cv2.bitwise_or(red_mask1, red_mask2)
    orange_mask = cv2.inRange(img_hsv, _ORANGE_LOW, _ORANGE_HIGH)
    dark_mask   = cv2.inRange(img_hsv, _DARK_LOW,   _DARK_HIGH)

    feats["yellow_pixel_ratio"] = _mask_ratio(yellow_mask, total)
    feats["green_pixel_ratio"]  = _mask_ratio(green_mask,  total)
    feats["red_pixel_ratio"]    = _mask_ratio(red_mask,    total)
    feats["orange_pixel_ratio"] = _mask_ratio(orange_mask, total)
    feats["dark_spot_ratio"]    = _mask_ratio(dark_mask,   total)

    # ── Color variation ───────────────────────────────────────────────────────
    feats["color_variation"] = float(np.std(img_rgb.astype(np.float32)))

    # ── GLCM texture ──────────────────────────────────────────────────────────
    # Resize gray to 128×128 before GLCM (faster, still informative)
    gray_small = cv2.resize(img_gray, (128, 128))
    glcm = graycomatrix(
        gray_small,
        distances=_GLCM_DIST,
        angles=_GLCM_ANGLES,
        levels=256,
        symmetric=True,
        normed=True,
    )
    feats["texture_contrast"]      = float(graycoprops(glcm, "contrast").mean())
    feats["texture_dissimilarity"] = float(graycoprops(glcm, "dissimilarity").mean())
    feats["texture_homogeneity"]   = float(graycoprops(glcm, "homogeneity").mean())
    feats["texture_energy"]        = float(graycoprops(glcm, "energy").mean())
    feats["texture_correlation"]   = float(graycoprops(glcm, "correlation").mean())
    feats["texture_ASM"]           = float(graycoprops(glcm, "ASM").mean())

    # Entropy from normalised histogram
    hist = cv2.calcHist([gray_small], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    hist_nz = hist[hist > 0]
    feats["texture_entropy"] = float(-np.sum(hist_nz * np.log2(hist_nz)))

    # ── Surface deterioration ─────────────────────────────────────────────────
    dark_px     = cv2.inRange(img_hsv, _DARK_LOW,   _DARK_HIGH)
    brown_px    = cv2.inRange(img_hsv, _BROWN_LOW,  _BROWN_HIGH)
    alight_px   = cv2.inRange(img_hsv, _ALIGHT_LOW, _ALIGHT_HIGH)
    asurf_px    = cv2.inRange(img_hsv, _ASURF_LOW,  _ASURF_HIGH)

    feats["dark_pixel_ratio"]       = _mask_ratio(dark_px,   total)
    feats["brown_pixel_ratio"]      = _mask_ratio(brown_px,  total)
    feats["abnormal_light_ratio"]   = _mask_ratio(alight_px, total)
    feats["abnormal_surface_ratio"] = _mask_ratio(asurf_px,  total)

    # ── Edge density ──────────────────────────────────────────────────────────
    edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)
    feats["edge_density"] = float(np.sum(edges > 0)) / total

    # ── Abnormal regions (contour analysis on dark/brown mask) ────────────────
    abnormal_combined = cv2.bitwise_or(dark_px, brown_px)
    kernel            = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    abnormal_clean    = cv2.morphologyEx(abnormal_combined, cv2.MORPH_CLOSE, kernel)
    contours, _       = cv2.findContours(
        abnormal_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    feats["abnormal_region_count"]       = float(len(contours))
    areas = [cv2.contourArea(c) for c in contours]
    feats["largest_abnormal_region_ratio"] = (
        float(max(areas)) / total if areas else 0.0
    )

    # ── Spot statistics ───────────────────────────────────────────────────────
    spot_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spot_mask   = cv2.morphologyEx(dark_px, cv2.MORPH_OPEN, spot_kernel)
    s_contours, _ = cv2.findContours(
        spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    spot_areas = [cv2.contourArea(c) for c in s_contours if cv2.contourArea(c) > 4]

    feats["spot_density"]   = float(len(spot_areas)) / total
    feats["spot_size_mean"] = float(np.mean(spot_areas))  if spot_areas else 0.0
    feats["spot_size_std"]  = float(np.std(spot_areas))   if spot_areas else 0.0

    # ── Surface statistics ────────────────────────────────────────────────────
    # surface_uniformity : inverse of local variance → high = smooth
    blur        = cv2.GaussianBlur(img_gray, (5, 5), 0)
    local_var   = cv2.Laplacian(blur, cv2.CV_64F).var()
    feats["surface_uniformity"] = float(1.0 / (1.0 + local_var))
    feats["surface_roughness"]  = float(local_var)

    # ── Shape features ────────────────────────────────────────────────────────
    # Threshold to get fruit silhouette
    _, fruit_thresh = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    f_contours, _   = cv2.findContours(
        fruit_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if f_contours:
        largest = max(f_contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        perim   = cv2.arcLength(largest, closed=True)
        x, y, fw, fh = cv2.boundingRect(largest)
        hull    = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        rect_area = fw * fh

        feats["fruit_area_ratio"] = float(area) / total
        feats["fruit_perimeter"]  = float(perim)
        feats["aspect_ratio"]     = float(fw) / float(fh + 1e-6)
        feats["circularity"]      = (4 * np.pi * area) / (perim ** 2 + 1e-6)
        feats["extent"]           = float(area) / float(rect_area + 1e-6)
        feats["solidity"]         = float(area) / float(hull_area + 1e-6)
    else:
        feats["fruit_area_ratio"] = 0.0
        feats["fruit_perimeter"]  = 0.0
        feats["aspect_ratio"]     = 1.0
        feats["circularity"]      = 0.0
        feats["extent"]           = 0.0
        feats["solidity"]         = 0.0

    # ── Sanity check ──────────────────────────────────────────────────────────
    # Replace any NaN/Inf with 0.0
    for k, v in feats.items():
        if not np.isfinite(v):
            log.warning("Feature %s is non-finite (%s) — replacing with 0.0", k, v)
            feats[k] = 0.0

    return feats


def build_numeric_feature_dict(
    image_bytes: bytes,
    temperature: float,
    humidity: float,
    moisture: float,
    day_no: int,
) -> dict:
    """
    Extract all 52 numeric features required by the LSTM scaler.

    Returns a dict in the exact order of NUMERIC_FEATURE_COLUMNS
    (confirmed from the checkpoint / scaler feature_names_in_).
    """
    feats = extract_image_features(image_bytes)
    feats["temperature"] = float(temperature)
    feats["humidity"]    = float(humidity)
    feats["moisture"]    = float(moisture)
    feats["day_no"]      = float(day_no)
    return feats
