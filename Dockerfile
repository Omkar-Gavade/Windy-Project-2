# Production image for the Windy solar-generation forecast pipeline.
#
# The application is a single long-running Python loop (test_multi_image.py):
#   capture Windy screenshots + animation -> OCR/colour features + optical flow
#   -> ML/physics forecast -> CSV output, once at each of the fixed daily times
#   in CAPTURE_TIMES (config.py).
# There is no web server, no database and no exposed port.
#
# Pinned to bookworm (Debian 12) for a stable Playwright/Chromium dependency set.
FROM python:3.12-slim-bookworm

# Match the default Ubuntu login user (uid 1000) on EC2 so bind-mounted output
# directories are writable by the non-root container user without chown steps.
ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Shared browser location so the root-time install is usable by the app user.
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright

# System packages:
#   ffmpeg       - trims the recorded animation (subprocess call in test_multi_image.py)
#   tesseract-ocr- OCR engine behind pytesseract (image_feature_extraction.py)
#   tzdata       - lets the TZ env var work; forecast blocks use local wall-clock time
#   procps       - pgrep, used by the healthcheck (and ps/top for debugging)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        tzdata \
        procps \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Chromium + its OS-level dependencies for Playwright.
RUN playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

# Health probe (kept before the source copy so it stays cached).
COPY docker-healthcheck.sh /usr/local/bin/docker-healthcheck
RUN chmod +x /usr/local/bin/docker-healthcheck

# Application source last — the layer that changes most often.
COPY . .

# Non-root runtime user owning the app dir and the browser bundle.
RUN groupadd --gid "${APP_GID}" appuser \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home appuser \
    && chown -R appuser:appuser /app /opt/playwright
USER appuser

# The loop writes a fresh predictions CSV every cycle; the probe checks that
# heartbeat. start-period covers the first (slow) capture + prediction run.
HEALTHCHECK --interval=2m --timeout=15s --start-period=15m --retries=3 \
    CMD ["docker-healthcheck"]

# SIGTERM (docker stop) terminates the loop; compose runs tini as PID 1 via
# `init: true` so signals are delivered and zombies reaped correctly.
STOPSIGNAL SIGTERM

CMD ["python", "test_multi_image.py"]
