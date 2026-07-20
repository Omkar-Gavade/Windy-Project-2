"""
batch/uploader.py

Uploads a generated clean MP4 to the existing FastAPI backend.

The backend owns S3 entirely: it builds the object key from the metadata
fields below, so nothing here constructs an S3 path.

    POST {BACKEND_API_URL}/api/videos/upload      (multipart/form-data)
        file            -> the *_clean.mp4
        state           -> e.g. "MadhyaPradesh"
        plant           -> e.g. "SIRMOUR"
        recording_date  -> "YYYY-MM-DD"
        recording_time  -> "HH:MM:SS"

The backend URL always comes from the BACKEND_API_URL environment variable
(loaded from .env) -- it is never hardcoded.
"""

import os
from pathlib import Path

import requests

import config

UPLOAD_PATH = "/api/videos/upload"
UPLOAD_TIMEOUT_SECONDS = 120

_ENV_FILE = Path(".env")


def _load_backend_url_from_env_file() -> str | None:
    """Fallback for running outside Docker.

    Under docker compose the `env_file:` directive already puts BACKEND_API_URL
    into the real environment, so this is never needed there. When the script is
    run directly with `python test_multi_image.py`, nothing loads .env, so read
    that one key here rather than adding a dependency just for it.
    """
    if not _ENV_FILE.exists():
        return None

    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "BACKEND_API_URL":
            return value.strip().strip('"').strip("'")
    return None


def get_backend_url() -> str:
    """Returns the backend base URL, without a trailing slash.

    Raises RuntimeError if it is not configured, so a misconfigured deployment
    fails with a clear message instead of posting to nowhere.
    """
    url = os.environ.get("BACKEND_API_URL") or _load_backend_url_from_env_file()
    if not url:
        raise RuntimeError(
            "BACKEND_API_URL is not set. Add it to .env "
            "(e.g. BACKEND_API_URL=http://13.206.205.164:8000)."
        )
    return url.rstrip("/")


def upload_video(video_path, plant_name: str, recording_datetime) -> dict:
    """Uploads one clean MP4 to the backend and returns the parsed JSON body.

    video_path        : path to the *_clean.mp4 produced by the recorder
    plant_name        : e.g. "SIRMOUR"
    recording_datetime: datetime the recording was made; split into the
                        recording_date / recording_time form fields

    Raises RuntimeError with a clear message on any failure (missing file,
    missing config, network error, non-2xx response, unparseable body). The
    caller is responsible for catching it so the pipeline can continue.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise RuntimeError(f"Video file not found: {video_path}")

    # Guard the contract: only the trimmed clip is uploaded, never the raw .webm.
    if not video_path.name.endswith("_clean.mp4"):
        raise RuntimeError(
            f"Refusing to upload '{video_path.name}' -- only *_clean.mp4 files "
            f"are uploaded."
        )

    url = get_backend_url() + UPLOAD_PATH

    data = {
        "state": config.PLANT_STATE,
        "plant": plant_name,
        "recording_date": recording_datetime.strftime("%Y-%m-%d"),
        "recording_time": recording_datetime.strftime("%H:%M:%S"),
    }

    print(f"POST {url}")

    try:
        with open(video_path, "rb") as fh:
            files = {"file": (video_path.name, fh, "video/mp4")}
            response = requests.post(
                url, files=files, data=data, timeout=UPLOAD_TIMEOUT_SECONDS
            )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"upload timed out after {UPLOAD_TIMEOUT_SECONDS}s ({url})"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"could not reach backend at {url}: {exc}") from exc

    if not response.ok:
        # Include a trimmed body -- FastAPI validation errors explain the cause.
        raise RuntimeError(
            f"backend returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"backend returned a non-JSON response: {response.text[:300]}"
        ) from exc
