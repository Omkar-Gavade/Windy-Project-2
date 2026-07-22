"""
batch/video_validation.py

Video integrity + motion checks used to guarantee that only a real, moving
satellite animation is ever uploaded.

Why this exists (root cause it addresses):
    Windy's satellite nowcast does not begin animating the moment Play is
    clicked -- it spends a long, variable warm-up buffering frames, during which
    the map is completely static. The recorder used to trim a FIXED 24s offset
    out of the full recording, which landed squarely in that frozen warm-up and
    produced frozen clips. Measured motion in a real recording: seconds 0-43
    were static, the animation only moved from ~44s onward.

    So instead of a fixed offset we locate the window that actually moves
    (`best_motion_window`) and trim there, then prove it moves (`is_moving`)
    before uploading. `validate_playable` adds basic integrity checks.

Pure OpenCV/NumPy -- no new dependencies.
"""

from pathlib import Path

import cv2
import numpy as np

# Downscale every frame before diffing: motion detection needs shape, not
# resolution, and small frames make this fast.
_DIFF_SIZE = (320, 200)

# A single second whose mean inter-frame pixel difference is at/above this is
# considered "moving". Calibrated from real recordings: frozen seconds measured
# < 0.15, genuinely animating seconds measured 0.2 - 0.7.
MOVING_SECOND_DIFF = 0.10

# A trimmed clip is accepted as a real animation only if the mean per-second
# diff over the whole clip is at least this, AND at least this fraction of its
# seconds are individually "moving".
#
# Windy's nowcast animates DISCRETELY (advance a frame, hold it, advance again),
# so only ~half the seconds show a big jump while the rest hold -- yet the clip
# is unmistakably animating. Measured separation is huge: frozen clips ~0.05
# mean, moving clips ~0.68 mean. So mean_diff carries the decision; the fraction
# is a loose guard against a single spike (e.g. one click frame) masquerading as
# motion.
MIN_CLIP_MEAN_DIFF = 0.12
MIN_MOVING_FRACTION = 0.3


def _motion_by_second(path) -> tuple[float, float, int, dict[int, float]]:
    """Returns (fps, duration_s, frame_count, {second: mean_inter_frame_diff}).

    The per-second value is the mean absolute difference between consecutive
    (downscaled, grayscale) frames in that second -- 0 means a frozen image.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0, 0.0, 0, {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    prev = None
    idx = 0
    buckets: dict[int, list[float]] = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, _DIFF_SIZE), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            second = int((idx - 1) / fps)
            buckets.setdefault(second, []).append(
                float(np.mean(cv2.absdiff(prev, gray)))
            )
        prev = gray
        idx += 1
    cap.release()

    per_second = {s: float(np.mean(v)) for s, v in buckets.items()}
    duration = idx / fps if fps else 0.0
    return fps, duration, idx, per_second


def validate_playable(path, expected_seconds: float, tolerance: float = 0.5) -> tuple[bool, str]:
    """Basic integrity gate: file exists, is non-empty, decodes, and its
    duration is roughly the expected length.

    tolerance is a fraction (0.5 => duration must be within +/-50% of expected).
    Returns (ok, reason). reason is "" when ok.
    """
    path = Path(path)
    if not path.exists():
        return False, f"file does not exist: {path}"
    size = path.stat().st_size
    if size <= 0:
        return False, f"file is empty (0 bytes): {path}"

    fps, duration, frames, _ = _motion_by_second(path)
    if frames <= 0:
        return False, f"video has no decodable frames: {path}"

    lo = expected_seconds * (1 - tolerance)
    hi = expected_seconds * (1 + tolerance)
    if not (lo <= duration <= hi):
        return False, (
            f"duration {duration:.1f}s outside expected "
            f"{expected_seconds:.0f}s +/-{int(tolerance * 100)}% "
            f"([{lo:.1f}, {hi:.1f}]s)"
        )
    return True, ""


def best_motion_window(path, window_seconds: int) -> tuple[int, float]:
    """Finds the start second of the contiguous `window_seconds` window with the
    highest average motion in the recording.

    Returns (start_second, mean_diff). If the recording is shorter than the
    window (or unreadable) returns (0, mean_of_whatever_exists).
    """
    _, _, _, per_second = _motion_by_second(path)
    if not per_second:
        return 0, 0.0

    last = max(per_second)
    if last + 1 <= window_seconds:
        vals = list(per_second.values())
        return 0, float(np.mean(vals)) if vals else 0.0

    best_start, best_mean = 0, -1.0
    for start in range(0, last - window_seconds + 2):
        vals = [per_second.get(s, 0.0) for s in range(start, start + window_seconds)]
        mean = float(np.mean(vals))
        if mean > best_mean:
            best_start, best_mean = start, mean
    return best_start, best_mean


def is_moving(path, expected_seconds: int) -> tuple[bool, float, float]:
    """Decides whether a (trimmed) clip contains a genuinely moving animation.

    Returns (ok, mean_diff, moving_fraction).
    """
    _, _, _, per_second = _motion_by_second(path)
    if not per_second:
        return False, 0.0, 0.0

    vals = list(per_second.values())
    mean_diff = float(np.mean(vals))
    moving_fraction = sum(1 for v in vals if v >= MOVING_SECOND_DIFF) / len(vals)

    ok = mean_diff >= MIN_CLIP_MEAN_DIFF and moving_fraction >= MIN_MOVING_FRACTION
    return ok, mean_diff, moving_fraction
