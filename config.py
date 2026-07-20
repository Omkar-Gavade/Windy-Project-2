"""
config.py

Single source of truth for plant details, file paths, and pipeline
settings -- imported by every other file in this project. Centralizing
these here avoids circular imports between test_multi_image.py and the
new feature/ML pipeline files, and means you only ever update plant
details in ONE place.
"""

from pathlib import Path

# ---- Plant details ----
PLANT_NAME = "SIRMOUR"

# State the plant sits in. Sent as the `state` form field when uploading a
# recording; the backend uses it to build the S3 key
# (videos/<state>/<plant>/<date>/...). No spaces -- it becomes a path segment.
PLANT_STATE = "MadhyaPradesh"
PLANT_LAT = 24.56253056
PLANT_LON = 75.09140278

# Rated (nameplate) capacity in MW.
PLANT_CAPACITY_MW = 5.1

# Performance Ratio -- accounts for real-world losses (panel temperature,
# inverter, wiring, soiling, shading, mismatch etc.). 0.75-0.85 is typical
# for a well-maintained plant. Update this once you have your plant's
# actual historical PR.
PERFORMANCE_RATIO = 0.78

# ---- Windy capture settings ----
ZOOM_LEVEL = 11  # calibrated so the screenshot covers ~100km x 100km
VIEWPORT_WIDTH = 1600
VIEWPORT_HEIGHT = 1000

LAYERS = {
    "satellite": "Satellite cloud imagery -- cloud position, density, and movement around the plant",
    "wind": "Wind speed, direction, and gusts",
    "solarpower": "Solar power / solar irradiance layer -- expected solar radiation intensity reaching the ground around the plant",
    "clouds": "Cloud cover layer -- overall cloud coverage and thickness around the plant",
    "rain": "Rain / precipitation layer -- rainfall intensity and coverage around the plant",
}

# ---- Animation video settings ----
RECORD_ANIMATION_VIDEO = True
ANIMATION_LAYER = "satellite"
ANIMATION_RECORD_SECONDS = 20

# ---- Forecast settings ----
NUM_FORECAST_BLOCKS = 8       # 8 x 15 min = next 2 hours
BLOCK_MINUTES = 15
RUN_INTERVAL_SECONDS = 20 * 60

# ---- Paths ----
STORAGE_STATE_PATH = Path("windy_login.json")
SCREENSHOT_DIR = Path("windy_screenshots") / f"{PLANT_LAT}_{PLANT_LON}"
VIDEO_DIR = Path("windy_videos")
PREDICTIONS_DIR = Path("energy_predictions")
FEATURES_LOG_DIR = Path("features_log")
MODELS_DIR = Path("models")
ACCURACY_REPORTS_DIR = Path("accuracy_reports")

for _dir in (SCREENSHOT_DIR, VIDEO_DIR, PREDICTIONS_DIR, FEATURES_LOG_DIR, MODELS_DIR, ACCURACY_REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "generation_model.pkl"
