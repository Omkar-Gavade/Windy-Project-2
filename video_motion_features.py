"""
video_motion_features.py

STANDALONE script/module that takes the recorded Windy satellite/cloud
animation video (the "_clean.mp4" file produced by test_multi_image.py)
and extracts NUMERIC cloud-motion + cloud-coverage features using
classical computer vision (optical flow + brightness thresholding) --
instead of asking an LLM to "watch" the raw video.

WHY: Vision-language models like Gemini are not reliable at precisely
quantifying motion (exact direction/speed) from video frames -- they can
describe things qualitatively ("clouds seem to be moving north-east") but
struggle to give consistent, reproducible numbers. Classical optical flow
gives deterministic, math-based numbers every time. The idea is:
    1. This script processes the video and produces a short text summary
       with real numbers (direction, speed, cloud-coverage trend).
    2. That summary text gets sent to Gemini ALONGSIDE the screenshots
       (instead of, or in addition to, the raw video) -- so Gemini is
       reasoning over solid numeric facts rather than raw pixels.

Requirements:
    pip install opencv-python numpy

Usage (standalone, from the command line):
    python video_motion_features.py path/to/SIRMOUR_satellite_..._clean.mp4

Usage (as a module, e.g. import this from test_multi_image.py later):
    from video_motion_features import analyze_video

    features = analyze_video(video_path)
    if features:
        print(features["summary_text"])
        # features["summary_text"] can be added into the Gemini prompt,
        # e.g. as an extra line before the screenshots list.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Calibration constants -- adjust these if you change the main script's
# ZOOM_LEVEL / VIEWPORT_WIDTH, since they affect how many real km one
# frame's width covers.
# ---------------------------------------------------------------------

# The Windy screenshots/video are calibrated (via ZOOM_LEVEL=11 in the
# main script) to cover roughly a 100km x 100km area around the plant.
# This is an approximation -- for a precise value, measure two known
# points' pixel distance in a screenshot and compare to their real-world
# distance, then update this constant.
APPROX_KM_ACROSS_FRAME = 100.0

# Region of interest (ROI) around the plant. Since the map is always
# centered on PLANT_LAT/PLANT_LON, the plant sits at the exact center of
# every frame -- so the ROI is just a centered box. Expressed as a
# fraction of the frame's width/height (0.4 = a centered box covering 40%
# of the frame in each dimension). Keeping the ROI smaller than the full
# frame focuses the analysis on what actually matters for the plant, and
# avoids being thrown off by clouds far away that will never reach it.
ROI_FRACTION = 0.4

# Sample every Nth video frame instead of every single one. Consecutive
# frames in a screen recording are often near-identical (same underlying
# Windy animation frame, or just antialiasing/UI noise), so sampling
# keeps the optical-flow computation meaningful and fast.
FRAME_SAMPLE_STEP = 3

# Brightness threshold (0-255, grayscale) above which a pixel is counted
# as "cloud". Satellite/IR imagery: clouds/precipitation are usually
# bright/white, clear sky and ground are darker. If your coverage %
# numbers look clearly wrong (e.g. always ~0% or ~100%), tune this value
# -- print a sample grayscale histogram to help pick a better threshold.
CLOUD_BRIGHTNESS_THRESHOLD = 150

# Minimum average flow magnitude (in pixels per sampled frame pair) that counts
# as real motion rather than sub-pixel noise.
#
# This matters more than it looks: one pixel per sample scales up to ~1875 km/h
# at the current zoom/sample-rate, so a meaningless 0.017px average still
# produced "~32.8 km/h" while the direction check separately reported
# "negligible / stationary" -- the same vector described two contradictory ways.
# Direction AND speed now both derive from this one threshold.
MIN_MOTION_PX = 0.05


def _get_roi_box(width, height, fraction=ROI_FRACTION):
    """Returns (x1, y1, x2, y2) pixel coordinates for a box of the given
    size fraction, centered in a frame of the given width/height."""
    box_w = int(width * fraction)
    box_h = int(height * fraction)
    x1 = (width - box_w) // 2
    y1 = (height - box_h) // 2
    return x1, y1, x1 + box_w, y1 + box_h


def _direction_from_vector(dx, dy):
    """
    Converts an average optical-flow pixel vector (dx, dy) into a
    compass direction string. Note: image/video y-coordinates increase
    DOWNWARD, so a positive dy means motion toward the bottom of the
    frame (south) -- this is corrected for below.
    """
    # Use the vector's magnitude (not each axis separately) so this agrees
    # exactly with the speed calculation in analyze_video().
    if float(np.hypot(dx, dy)) < MIN_MOTION_PX:
        return "negligible / stationary"

    # Flip dy so that "up" on screen (north) is treated as positive,
    # matching normal compass-angle math.
    angle = np.degrees(np.arctan2(-dy, dx)) % 360

    directions = [
        "East", "Northeast", "North", "Northwest",
        "West", "Southwest", "South", "Southeast",
    ]
    idx = int(((angle + 22.5) % 360) // 45)
    return directions[idx]


def analyze_video(video_path, roi_fraction=ROI_FRACTION, sample_step=FRAME_SAMPLE_STEP):
    """
    Processes the given video file and returns a dict:
        {
            "avg_direction":     e.g. "Northeast"
            "avg_speed_kmh":     e.g. 14.3   (rough estimate)
            "coverage_start_pct": e.g. 42.1  (cloud % in ROI, first sampled frame)
            "coverage_end_pct":   e.g. 55.8  (cloud % in ROI, last sampled frame)
            "coverage_trend":    "increasing" / "decreasing" / "stable"
            "frame_count":       how many frames were sampled/used
            "summary_text":      ready-to-paste text for the Gemini prompt
        }
    Returns None if the video couldn't be opened or had too few usable
    frames (need at least 2 sampled frames for optical flow to work).
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_step == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()

    if len(frames) < 2:
        print(f"[ERROR] Only {len(frames)} usable frame(s) found in the video "
              f"-- it may be too short, corrupted, or unreadable.")
        return None

    height, width = frames[0].shape[:2]
    x1, y1, x2, y2 = _get_roi_box(width, height, roi_fraction)
    km_per_px = APPROX_KM_ACROSS_FRAME / width
    seconds_between_samples = sample_step / fps  # accounts for frame skipping

    coverage_pcts = []
    flow_vectors = []

    prev_gray_roi = None
    for frame in frames:
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # ---- Cloud coverage %: fraction of ROI pixels above the
        # brightness threshold ----
        cloud_pixels = int(np.count_nonzero(gray_roi > CLOUD_BRIGHTNESS_THRESHOLD))
        total_pixels = int(gray_roi.size)
        coverage_pct = 100.0 * cloud_pixels / total_pixels
        coverage_pcts.append(coverage_pct)

        # ---- Optical flow between this frame and the previous one ----
        if prev_gray_roi is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray_roi, gray_roi, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mean_dx = float(np.mean(flow[..., 0]))
            mean_dy = float(np.mean(flow[..., 1]))
            flow_vectors.append((mean_dx, mean_dy))

        prev_gray_roi = gray_roi

    # ---- Aggregate motion across the whole clip ----
    if flow_vectors:
        avg_dx = float(np.mean([v[0] for v in flow_vectors]))
        avg_dy = float(np.mean([v[1] for v in flow_vectors]))
    else:
        avg_dx, avg_dy = 0.0, 0.0

    avg_direction = _direction_from_vector(avg_dx, avg_dy)

    pixel_speed_per_sample = float(np.hypot(avg_dx, avg_dy))

    # Sub-pixel averages are noise, not cloud movement. Report them as zero
    # instead of amplifying them by ~1875x into a plausible-looking km/h figure
    # that contradicts the "negligible / stationary" direction above.
    if pixel_speed_per_sample < MIN_MOTION_PX:
        speed_kmh = 0.0
    else:
        km_per_sample = pixel_speed_per_sample * km_per_px
        speed_kmh = (
            (km_per_sample / seconds_between_samples) * 3600.0
            if seconds_between_samples > 0 else 0.0
        )

    # ---- Cloud coverage trend (start of clip vs end of clip) ----
    coverage_start = coverage_pcts[0]
    coverage_end = coverage_pcts[-1]
    coverage_delta = coverage_end - coverage_start

    if abs(coverage_delta) < 3:
        coverage_trend = "stable"
    elif coverage_delta > 0:
        coverage_trend = "increasing"
    else:
        coverage_trend = "decreasing"

    summary_text = (
        "Cloud motion analysis (computed via optical flow on the recorded "
        "video, NOT LLM-estimated -- treat these as ground-truth numbers):\n"
        f"- Dominant cloud motion direction over the plant's area: {avg_direction}\n"
        f"- Estimated cloud movement speed: ~{speed_kmh:.1f} km/h\n"
        f"- Cloud coverage directly over the plant: {coverage_start:.1f}% at "
        f"the start of the clip -> {coverage_end:.1f}% at the end "
        f"({coverage_trend})\n"
        f"- Based on {len(frames)} sampled video frames."
    )

    return {
        "avg_direction": avg_direction,
        "avg_speed_kmh": round(speed_kmh, 2),
        "coverage_start_pct": round(coverage_start, 2),
        "coverage_end_pct": round(coverage_end, 2),
        "coverage_trend": coverage_trend,
        "frame_count": len(frames),
        "summary_text": summary_text,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_motion_features.py <path_to_video.mp4>")
        sys.exit(1)

    video_file = Path(sys.argv[1])
    if not video_file.exists():
        print(f"[ERROR] File not found: {video_file}")
        sys.exit(1)

    result = analyze_video(video_file)
    if result is None:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("VIDEO MOTION FEATURE EXTRACTION RESULT")
    print("=" * 60)
    for key, value in result.items():
        if key == "summary_text":
            continue
        print(f"{key}: {value}")

    print("\n--- Summary text (ready to paste into your Gemini prompt) ---")
    print(result["summary_text"])
