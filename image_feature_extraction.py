"""
image_feature_extraction.py

Extracts NUMERIC features from each of the captured Windy layer
screenshots (satellite, wind, solarpower, clouds, rain) -- using TWO
complementary methods, both deterministic (no LLM involved):

1. COLOR/BRIGHTNESS STATS (original method) -- average brightness, color
   saturation/hue, and % of "bright" pixels in a region centered on the
   plant. These are proxies: a color/intensity that CORRELATES with the
   real value, but isn't the real value itself.

2. OCR (NEW) -- Windy prints EXACT numbers directly on the screenshot:
   the marker popup near the plant's location (e.g. "98 %" cloud cover,
   or a temperature like "-14 degC"), and the bottom hourly-forecast
   panel (Temperature / Rain / Wind / Wind gusts rows). Instead of
   guessing these from color, we read the actual printed text with
   Tesseract OCR -- this is exact, not a proxy.

WHY BOTH: OCR gives exact values when it can read the text cleanly, but
can fail (font rendering, overlapping popups, layer-specific panel
differences) -- so we keep the color-based stats as a reliable fallback
signal that always produces SOME number even if OCR comes back empty.

SETUP FOR OCR (one-time, on your Windows PC):
    pip install pytesseract
    Install the Tesseract-OCR engine itself (OCR needs the actual program,
    not just the Python wrapper): https://github.com/UB-Mannheim/tesseract/wiki
    After installing, if "tesseract" isn't on your PATH, set the path
    explicitly near the top of this file (see TESSERACT_CMD below).

CALIBRATION NOTE: The marker-popup region and bottom-panel region below
are estimated from VIEWPORT_WIDTH x VIEWPORT_HEIGHT (1600x1000) and
Windy's typical layout. If OCR keeps returning None/empty for your
screenshots, open one screenshot and check the ACTUAL pixel position of
the popup/panel, then adjust MARKER_REGION_FRACTION / BOTTOM_PANEL_FRACTION
below to match.

Usage:
    from image_feature_extraction import extract_image_features
    features = extract_image_features(image_map)   # image_map: {filepath: description}
"""

import re
from pathlib import Path

import cv2
import numpy as np

try:
    import pytesseract
    # If tesseract isn't on your system PATH, uncomment and set the exact
    # install path, e.g.:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False
    print("[WARN] pytesseract not installed -- OCR features will be skipped "
          "(color/brightness features still work). Run: pip install pytesseract")

# ---- Color/brightness stats settings (unchanged from before) ----
ROI_FRACTION = 0.4
BRIGHT_PIXEL_THRESHOLD = 150

# ---- OCR region settings (CALIBRATE THESE if OCR returns nothing useful) ----
# The marker popup (e.g. "98 %") appears near the plant's location, which
# is near the center of the frame but Windy renders the popup slightly
# ABOVE the actual marker point -- this box covers roughly the upper-
# middle area where it tends to appear.
MARKER_REGION_FRACTION = (0.35, 0.30, 0.65, 0.50)  # (x1, y1, x2, y2) as fractions of width/height

# The bottom hourly-forecast panel (Temperature/Rain/Wind/Wind gusts rows)
# sits in roughly the bottom 22% of the screenshot.
BOTTOM_PANEL_FRACTION = 0.22

# Row labels to search for in the bottom panel, and what feature name each
# maps to. Add/adjust text if your Windy UI language/wording differs.
FORECAST_ROW_LABELS = {
    "temperature": "ocr_temp_c",
    "rain": "ocr_rain_mm",
    "wind gusts": "ocr_wind_gusts",   # check this BEFORE plain "wind" (more specific first)
    "wind": "ocr_wind_speed",
}

# Windy renders those row labels as ICONS, not text, so label matching alone
# finds nothing. Rows are therefore also identified by the shape of their
# values (degree signs / decimals / bare integers). A row must contain at least
# this many same-shaped numbers before it is accepted, so one stray OCR token
# cannot be mistaken for a whole row.
MIN_ROW_VALUE_MATCHES = 3

# Human-readable names used when warning about a row that could not be read.
OCR_WARN_LABELS = {
    "Temperature": "ocr_temp_c",
    "Rain": "ocr_rain_mm",
    "Wind": "ocr_wind_speed",
    "Wind gusts": "ocr_wind_gusts",
}


def _get_roi_box(width, height, fraction=ROI_FRACTION):
    box_w = int(width * fraction)
    box_h = int(height * fraction)
    x1 = (width - box_w) // 2
    y1 = (height - box_h) // 2
    return x1, y1, x1 + box_w, y1 + box_h


def _layer_name_from_path(filepath: str) -> str:
    """windy_screenshots/.../satellite.png -> 'satellite'"""
    return Path(filepath).stem


def _extract_single_image_stats(filepath: str) -> dict:
    """
    Returns {avg_brightness, avg_saturation, avg_hue_deg, bright_pixel_pct}
    computed over the centered ROI of one screenshot. Returns None-filled
    dict if the image can't be read.
    """
    img = cv2.imread(filepath)  # BGR
    if img is None:
        return {
            "avg_brightness": None,
            "avg_saturation": None,
            "avg_hue_deg": None,
            "bright_pixel_pct": None,
        }

    height, width = img.shape[:2]
    x1, y1, x2, y2 = _get_roi_box(width, height)
    roi = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    avg_brightness = float(np.mean(gray))
    avg_saturation = float(np.mean(hsv[..., 1]))

    # Hue is circular (0-179 in OpenCV, representing 0-360 degrees) --
    # average it as a circular mean (via unit vectors) instead of a
    # plain arithmetic mean, otherwise e.g. 350 deg and 10 deg would
    # wrongly average to 180 deg instead of 0 deg.
    hue_rad = hsv[..., 0].astype(np.float64) * (2 * np.pi / 180.0)
    mean_sin = float(np.mean(np.sin(hue_rad)))
    mean_cos = float(np.mean(np.cos(hue_rad)))
    avg_hue_deg = float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360)

    bright_pixels = int(np.count_nonzero(gray > BRIGHT_PIXEL_THRESHOLD))
    bright_pixel_pct = 100.0 * bright_pixels / gray.size

    return {
        "avg_brightness": round(avg_brightness, 2),
        "avg_saturation": round(avg_saturation, 2),
        "avg_hue_deg": round(avg_hue_deg, 2),
        "bright_pixel_pct": round(bright_pixel_pct, 2),
    }


# ---------------------------------------------------------------------
# OCR-based exact-number extraction (NEW)
# ---------------------------------------------------------------------

_NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


def _crop_fraction(img, x1f, y1f, x2f, y2f):
    """Crops img using fractional coordinates (0-1) of its width/height."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(w * x1f), int(h * y1f), int(w * x2f), int(h * y2f)
    return img[y1:y2, x1:x2]


def _ocr_preprocess_variants(crop):
    """
    Tesseract is tuned for dark text on a light background, but Windy's
    popups/panels are often light text on a dark badge (or vice versa),
    which makes plain OCR misread numbers badly (e.g. "98" -> "jos").
    This generates a few variants to try in order:
        1. The raw crop as-is (works fine for many real screenshots)
        2. Upscaled grayscale (helps when the source text is tiny)
        3/4. Upscaled + binary threshold, normal and inverted (helps
             with badge-style light-on-dark or dark-on-light popups)
    The caller tries OCR on each variant in turn and keeps the first one
    that yields a usable result, so whichever style matches your actual
    screenshots will be picked up automatically.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    _, thresh_normal = cv2.threshold(gray_upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh_inverted = cv2.threshold(gray_upscaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return [crop, gray_upscaled, thresh_normal, thresh_inverted]


def _extract_marker_value(img) -> dict:
    """
    Reads the small popup box near the plant's marker (e.g. "98 %" for
    cloud cover, or a temperature like "-14 degC") using OCR on a cropped
    region. Tries a few preprocessing variants (see _ocr_preprocess_variants)
    since these popups are often light-text-on-dark-badge, which plain
    OCR misreads. Returns {"marker_value": float or None, "marker_is_percent": bool}.
    """
    if not _OCR_AVAILABLE:
        return {"marker_value": None, "marker_is_percent": None}

    x1f, y1f, x2f, y2f = MARKER_REGION_FRACTION
    crop = _crop_fraction(img, x1f, y1f, x2f, y2f)

    for variant in _ocr_preprocess_variants(crop):
        try:
            text = pytesseract.image_to_string(variant, config="--psm 7")
        except Exception as e:
            print(f"[WARN] OCR failed on marker region: {e}")
            continue

        match = _NUMBER_PATTERN.search(text)
        if match:
            value = float(match.group())
            is_percent = "%" in text
            return {"marker_value": value, "marker_is_percent": is_percent}

    return {"marker_value": None, "marker_is_percent": None}


def _extract_forecast_row_values(img) -> dict:
    """
    Reads the bottom hourly-forecast panel using OCR word-level data
    (positions, not just plain text), groups words into rows by their
    y-coordinate, matches each row to a known label (Temperature/Rain/
    Wind/Wind gusts), and returns the FIRST (leftmost = soonest) numeric
    value found in that row -- i.e. the closest-to-now forecast reading.

    Returns a dict like {"ocr_temp_c": 27.0, "ocr_rain_mm": 0.02, ...},
    with None for any row that couldn't be confidently read.
    """
    result = {name: None for name in FORECAST_ROW_LABELS.values()}
    if not _OCR_AVAILABLE:
        return result

    h, w = img.shape[:2]
    panel_y_start = int(h * (1 - BOTTOM_PANEL_FRACTION))
    panel = img[panel_y_start:h, 0:w]

    # Upscale 2x -- the panel's small forecast numbers OCR more reliably
    # enlarged, same reasoning as the marker-popup preprocessing above.
    panel_gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    panel_gray = cv2.resize(panel_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    scale_x, scale_y = 2, 2

    try:
        data = pytesseract.image_to_data(panel_gray, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"[WARN] OCR failed on forecast panel: {e}")
        return result

    n = len(data.get("text", []))
    words = []
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        words.append({
            "text": text,
            "x": data["left"][i] / scale_x,
            "y": data["top"][i] / scale_y,
        })

    if not words:
        print("[WARN] OCR read no text at all from the forecast panel -- "
              "all forecast row values will be empty.")
        for label_text, feature_name in OCR_WARN_LABELS.items():
            print(f"[WARN] {label_text} row not detected")
        return result

    # ---- Group words into rows by y-coordinate (words within ~12px of
    # each other vertically are considered the same row/line) ----
    words.sort(key=lambda wd: wd["y"])
    rows = []
    current_row, current_y = [], None
    ROW_Y_TOLERANCE = 12
    for wd in words:
        if current_y is None or abs(wd["y"] - current_y) <= ROW_Y_TOLERANCE:
            current_row.append(wd)
            current_y = wd["y"] if current_y is None else current_y
        else:
            rows.append(current_row)
            current_row, current_y = [wd], wd["y"]
    if current_row:
        rows.append(current_row)

    # ---- Identify each row by the SHAPE OF ITS VALUES, not by a text label.
    # Windy renders the row labels (Temperature/Rain/Wind/Gusts) as ICONS, so
    # the old `if label in row_text_lower` check never matched and every value
    # silently stayed None. The numbers themselves OCR fine, and each row has a
    # distinctive value signature:
    #     temperature -> tokens carry a degree sign:  30°  25°  24°
    #     rain        -> decimal values:              0.01  0.05  0.11
    #     wind/gusts  -> bare small integers:         11  3  3  4
    # Wind and gusts share the same signature, so they are assigned in vertical
    # order: the first integer row is wind speed, the second is gusts (this is
    # the order Windy lays them out in the panel).
    # Label matching is still attempted FIRST, so if a Windy build (or another
    # UI language) does render real text labels, that still wins.
    for row in rows:
        row_sorted = sorted(row, key=lambda wd: wd["x"])
        row_text_lower = " ".join(wd["text"] for wd in row_sorted).lower()

        matched_feature = None
        for label, feature_name in FORECAST_ROW_LABELS.items():
            if label in row_text_lower and result[feature_name] is None:
                matched_feature = feature_name
                break

        if matched_feature is not None:
            for wd in row_sorted:
                m = _NUMBER_PATTERN.fullmatch(wd["text"].replace("°", "").replace("%", ""))
                if m:
                    result[matched_feature] = float(m.group())
                    break

    # ---- Unit/shape-based pass for whatever the label pass could not fill ----
    integer_rows = []
    for row in rows:
        row_sorted = sorted(row, key=lambda wd: wd["x"])

        degree_vals, decimal_vals, integer_vals = [], [], []
        for wd in row_sorted:
            token = wd["text"].strip()
            cleaned = token.replace("°", "").replace("%", "").replace(",", ".")
            m = _NUMBER_PATTERN.fullmatch(cleaned)
            if not m:
                continue
            value = float(m.group())
            if "°" in token:
                degree_vals.append(value)
            elif "." in cleaned:
                decimal_vals.append(value)
            else:
                integer_vals.append(value)

        # Require a few same-shaped values so a stray number can't hijack a row.
        if len(degree_vals) >= MIN_ROW_VALUE_MATCHES:
            if result["ocr_temp_c"] is None:
                result["ocr_temp_c"] = degree_vals[0]
        elif len(decimal_vals) >= MIN_ROW_VALUE_MATCHES:
            if result["ocr_rain_mm"] is None:
                result["ocr_rain_mm"] = decimal_vals[0]
        elif len(integer_vals) >= MIN_ROW_VALUE_MATCHES:
            integer_rows.append((row_sorted[0]["y"], integer_vals[0]))

    # Wind speed then gusts, in the order the rows appear down the panel.
    integer_rows.sort(key=lambda item: item[0])
    for feature_name in ("ocr_wind_speed", "ocr_wind_gusts"):
        if result[feature_name] is None and integer_rows:
            result[feature_name] = integer_rows.pop(0)[1]

    # ---- Fail loudly: never return silently-empty OCR features ----
    for label_text, feature_name in OCR_WARN_LABELS.items():
        if result[feature_name] is None:
            print(f"[WARN] {label_text} row not detected")

    return result


def extract_image_features(image_map: dict) -> dict:
    """
    image_map: {filepath: description} as produced by capture_all_layers()
    in test_multi_image.py.

    Returns a FLAT dict combining, per layer:
      - the 4 color/brightness stats (unchanged from before)
      - OCR-derived exact numbers: marker popup value + bottom-panel
        Temperature/Rain/Wind/Wind-gusts readings

    e.g.:
        {
          "satellite_avg_brightness": 132.4,
          "satellite_avg_saturation": 18.2,
          "satellite_avg_hue_deg": 95.3,
          "satellite_bright_pixel_pct": 41.7,
          "satellite_marker_value": 98.0,
          "satellite_marker_is_percent": True,
          "satellite_ocr_temp_c": 27.0,
          "satellite_ocr_rain_mm": 0.02,
          "satellite_ocr_wind_speed": 5.0,
          "satellite_ocr_wind_gusts": 8.0,
          "clouds_avg_brightness": ...,
          ...
        }

    NOTE: the bottom forecast panel is generally the SAME across all 5
    layer screenshots (it's Windy's point-forecast panel, not layer-
    specific), so ocr_temp_c/ocr_rain_mm/etc. will likely repeat across
    layers -- that's expected, not a bug. Only the marker popup value
    (cloud %, wind speed color, etc.) actually differs per layer.
    """
    features = {}
    for filepath in image_map:
        layer = _layer_name_from_path(filepath)
        stats = _extract_single_image_stats(filepath)

        img = cv2.imread(filepath)
        if img is not None:
            marker = _extract_marker_value(img)
            forecast_row_values = _extract_forecast_row_values(img)
        else:
            marker = {"marker_value": None, "marker_is_percent": None}
            forecast_row_values = {name: None for name in FORECAST_ROW_LABELS.values()}

        combined = {**stats, **marker, **forecast_row_values}
        for stat_name, value in combined.items():
            features[f"{layer}_{stat_name}"] = value

    return features


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python image_feature_extraction.py <screenshot_dir>")
        sys.exit(1)

    shot_dir = Path(sys.argv[1])
    fake_map = {str(p): p.stem for p in shot_dir.glob("*.png")}
    result = extract_image_features(fake_map)
    for k, v in result.items():
        print(f"{k}: {v}")