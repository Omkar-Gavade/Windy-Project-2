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

# How long to keep recording after Play is clicked. Windy's nowcast buffers for
# a long, variable time before it actually animates, so we record generously and
# then trim to the moving window (see _trim_to_motion) rather than a fixed
# offset. Must comfortably exceed ANIMATION_RECORD_SECONDS plus that warm-up.
POST_PLAY_CAPTURE_SECONDS = 55

# A recording is retried up to this many times if the captured clip is frozen /
# static. If all attempts fail the run uploads nothing.
RECORD_MAX_ATTEMPTS = 2

# ---- Capture schedule (Asia/Kolkata) ----
# The pipeline runs ONLY at these local times, every day -- never on a fixed
# interval. Timezone-aware scheduling; no cron, no external scheduler.
TIMEZONE = "Asia/Kolkata"
CAPTURE_TIMES = [
    (6, 45),
    (8, 15),
    (9, 45),
    (11, 15),
    (12, 45),
    (14, 15),
    (15, 45),
]

# ---- Forecast settings ----
NUM_FORECAST_BLOCKS = 8       # 8 x 15 min = next 2 hours
BLOCK_MINUTES = 15
RUN_INTERVAL_SECONDS = 20 * 60  # retained for reference; scheduler uses CAPTURE_TIMES

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

# Liveness heartbeat: the main loop refreshes this file's mtime continuously
# (including while waiting for the next scheduled capture). The Docker
# healthcheck reads it, so container health reflects "the loop is ticking"
# rather than "a prediction happened recently" -- correct given long gaps
# between scheduled captures (and the ~15h overnight gap).
HEARTBEAT_PATH = Path(".heartbeat")
