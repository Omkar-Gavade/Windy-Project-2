# Deploying Windy-Project-2 on AWS EC2 (Ubuntu) with Docker

Production deployment guide for the Windy solar-generation forecast pipeline.

---

## 1. Architecture

### What this application actually is

A **single long-running Python process**. It is not a web application:

- no frontend
- no backend/HTTP API
- no database
- no listening ports (nothing is published)

`test_multi_image.py` runs an infinite loop. Each cycle:

```
                  ┌──────────────────────────────────────────┐
                  │  loop every RUN_INTERVAL_SECONDS (20 min) │
                  └──────────────────────────────────────────┘
                                     │
   1. Playwright + Chromium ─────────┤ log in via windy_login.json (session reuse)
      capture 5 Windy layers         │ satellite, wind, solarpower, clouds, rain
                                     │ → windy_screenshots/
                                     │
   2. Record 20s satellite animation │ → windy_videos/*_full.webm
      ffmpeg trims it                │ → windy_videos/*_clean.mp4
                                     │
   3. image_feature_extraction.py    │ colour/brightness + tesseract OCR
      video_motion_features.py       │ OpenCV optical flow (cloud drift)
      time_features.py               │ solar geometry per 15-min block
                                     │
   4. feature_builder.py             │ combine into one feature row per block
      ml_forecast_model.py           │ trained model if models/generation_model.pkl
                                     │ exists, else physics-based fallback
                                     │
   5. prediction_store.py            │ → energy_predictions/<PLANT>_energy_generation.csv
                                     │ → features_log/<PLANT>_features_log.csv
                                     ▼
                                  sleep 20 min
```

Outbound network: **windy.com only**. No inbound traffic.

### Container diagram

```
┌─────────────────────── EC2 instance (Ubuntu) ────────────────────────┐
│                                                                       │
│  docker network: windy-net (bridge)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ container: windy-forecast   (image windy-forecast:latest)       │ │
│  │ PID 1: tini (init: true) → python test_multi_image.py           │ │
│  │ user:  appuser (uid 1000, non-root)                             │ │
│  │ deps:  Chromium (Playwright) · ffmpeg · tesseract-ocr           │ │
│  │ health: docker-healthcheck (predictions CSV freshness)          │ │
│  └───────────────┬─────────────────────────────────────────────────┘ │
│                  │ bind mounts (host ./ ↔ /app/)                      │
│   windy_login.json:ro · energy_predictions · features_log             │
│   models · accuracy_reports · windy_screenshots · windy_videos        │
│                  │                                                     │
└──────────────────┼─────────────────────────────────────────────────────┘
                   ▼  outbound HTTPS
              windy.com
```

### Ports

**None.** No `ports:` mapping exists, because the application does not listen on
any socket. Your EC2 security group needs **no inbound rules** for this service
(keep SSH/22 for administration only). Outbound HTTPS (443) must be allowed.

### Networks

| Name | Driver | Purpose |
|---|---|---|
| `windy-net` | bridge | Named network for the single service. Present so future services attach cleanly; no cross-container traffic today. |

### Volumes

All bind mounts, so results are readable directly on the EC2 filesystem.

| Host path | Container path | Mode | Contents |
|---|---|---|---|
| `./windy_login.json` | `/app/windy_login.json` | **ro** | Windy Premium session state |
| `./energy_predictions` | `/app/energy_predictions` | rw | Final forecast CSV (the deliverable) |
| `./features_log` | `/app/features_log` | rw | Feature rows — training data for `train_model.py` |
| `./models` | `/app/models` | rw | `generation_model.pkl` once trained |
| `./accuracy_reports` | `/app/accuracy_reports` | rw | `accuracy_tracker.py` output |
| `./windy_screenshots` | `/app/windy_screenshots` | rw | Captured layer screenshots |
| `./windy_videos` | `/app/windy_videos` | rw | Recorded + trimmed animations |

These directories are committed with `.gitkeep`, so they exist after `git clone`
and are owned by your login user. The container runs as uid 1000 to match
Ubuntu's default `ubuntu` user, so writes succeed without any `chown` step.

### Environment variables

The **Python code reads no environment variables at all** — there are no API
keys, tokens, passwords or AWS credentials for this service. Plant settings live
in `config.py`; Windy access comes from the mounted session file.

`.env` still exists because Docker consumes it:

| Variable | Default | Consumed by | Why it matters |
|---|---|---|---|
| `TZ` | `Asia/Kolkata` | container OS | **Correctness-critical.** Forecast blocks come from local wall-clock `datetime.now()`. A fresh EC2 is UTC, which would shift every 15-minute block and misalign the solar-elevation model. Set to the plant's real timezone. |
| `HEALTH_MAX_AGE_SECONDS` | `2700` | healthcheck | Max age of the predictions CSV before the container is marked unhealthy (~2 cycles). Raise if you increase `RUN_INTERVAL_SECONDS`. |

---

## 2. Deployment

### Step 1 — Install Docker and Git on a fresh Ubuntu EC2

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git ca-certificates curl

# Docker Engine + Compose plugin (official repository)
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Run docker without sudo
sudo usermod -aG docker $USER
newgrp docker

docker --version && docker compose version
```

### Step 2 — Clone

```bash
git clone https://github.com/Kushal70-51/Windy-Project-2.git
cd Windy-Project-2
```

### Step 3 — Configure

```bash
cp .env.example .env
nano .env      # set TZ to the plant's timezone
```

No secrets to add. See the environment table above.

### Step 4 — Windy session file

`windy_login.json` is required. It is currently committed to the repository, so
a clone already includes it — **but read the Security section below, because
that is a problem you should fix.**

If you remove it from the repo (recommended), copy it in from a workstation:

```bash
# from your local machine
scp -i <key.pem> windy_login.json ubuntu@<EC2_IP>:~/Windy-Project-2/windy_login.json
```

### Step 5 — Build

```bash
docker compose build
```

First build takes ~5–10 minutes (Chromium + system dependencies).

### Step 6 — Run

```bash
docker compose up -d
```

---

## 3. Operations

### Verify health

```bash
docker compose ps                      # STATUS should reach "Up (healthy)"
docker inspect -f '{{.State.Health.Status}}' windy-forecast
docker inspect -f '{{range .State.Health.Log}}{{.ExitCode}} {{.Output}}{{end}}' windy-forecast
```

Expected when healthy:

```
healthy: SIRMOUR_energy_generation.csv updated 664s ago
```

The container reports `health: starting` for up to 15 minutes
(`start_period`) — the first capture + prediction cycle must finish before a
CSV is fresh. This is expected, not a failure.

### Logs

```bash
docker compose logs -f                 # follow
docker compose logs --tail=100         # recent
docker compose logs --since 1h         # time-boxed
```

Log rotation is configured (10 MB × 5 files) so logs cannot fill the disk.

### Results

```bash
cat energy_predictions/SIRMOUR_energy_generation.csv
tail -5 features_log/SIRMOUR_features_log.csv
```

### Restart / stop / update

```bash
docker compose restart                 # restart service
docker compose stop                    # stop, keep container
docker compose down                    # stop and remove container + network
docker compose up -d                   # start again

# update to latest code
git pull
docker compose build
docker compose up -d
```

### Training a model (optional)

The pipeline uses a physics-based fallback until a model exists. Once you have
accumulated `features_log/` rows and real meter data:

```bash
docker compose exec windy-forecast python train_model.py <actual_meter.csv>
```

It writes `models/generation_model.pkl`, which is picked up automatically on the
next cycle — no code change and no rebuild needed.

---

## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Stuck at `health: starting` >15 min | First cycle not finished, or the loop is failing | `docker compose logs --tail=50` |
| `UNHEALTHY: test_multi_image.py is not running` | Process crashed | Check logs; container will auto-restart (`unless-stopped`) |
| `UNHEALTHY: ... is Ns old — loop appears stalled` | Cycle exceeding ~45 min | Usually Windy timeouts or an under-sized instance. Check logs, consider a larger instance |
| Login/capture returns empty or logged-out pages | Windy session expired | Regenerate `windy_login.json` (see Security) |
| `Permission denied` writing CSVs | Host dirs owned by root | `sudo chown -R 1000:1000 energy_predictions features_log models accuracy_reports windy_screenshots windy_videos` |
| Container OOM-killed | Chromium exceeded `mem_limit: 2g` | Use ≥ t3.medium; raise `mem_limit` in `docker-compose.yml` |
| Build fails pulling base image | DNS/network | Retry; verify outbound 443 is open |
| `[WARN] pytesseract not installed` | Should not occur in the image | Rebuild without cache: `docker compose build --no-cache` |
| Predictions are `0.0 MW` | Correct at night — the solar model returns zero when the sun is down | Compare during daylight hours |

---

## 5. Security

> **Action required: `windy_login.json` is committed to this public repository
> and contains live Windy session cookies.** Anyone who clones the repo can
> reuse that Premium session. Fix before treating this as production:

```bash
# 1. Stop tracking it (it is now in .gitignore)
git rm --cached windy_login.json
git commit -m "Remove session state from version control"
git push

# 2. Rotate: log out of that Windy session in a browser, log in again,
#    and regenerate the file locally by running the app once on a desktop
#    (ensure_login() opens a visible browser when the file is absent).

# 3. Purge it from git history as well — the old cookies stay reachable
#    in previous commits until you do. Use git-filter-repo or BFG.
```

Also note: the session file cannot be regenerated inside the container.
`ensure_login()` launches a **non-headless** browser and waits on `input()`,
which cannot work in a detached container. Always generate it on a workstation
and copy it to EC2.

Other hardening already applied:

- Container runs as non-root (`appuser`, uid 1000)
- Session file mounted read-only
- No inbound ports published
- No secrets baked into the image (`.dockerignore` excludes `.env` and the session file)
- Memory/CPU caps prevent one runaway Chromium from taking down the box
- Log rotation prevents disk exhaustion

---

## 6. Production recommendations

1. **Instance size** — use at least **t3.medium** (2 vCPU / 4 GB). Chromium plus
   OpenCV in a 2 GB cap is tight on t3.small; t3.micro will OOM.
2. **Disk** — 30 GB+ gp3. Screenshots and videos accumulate every 20 minutes.
3. **Prune old media** — nothing deletes old captures. Add a cron job:
   ```bash
   0 3 * * * find ~/Windy-Project-2/windy_videos ~/Windy-Project-2/windy_screenshots \
             -type f -mtime +7 -delete
   ```
   Do **not** prune `features_log/` — it is your future training set.
4. **Back up the deliverables** — sync `energy_predictions/` and `features_log/`
   to S3 daily (`aws s3 sync`). They are the only irreplaceable outputs.
5. **Set `TZ` correctly.** Re-stated because it silently corrupts every forecast
   block if wrong.
6. **Monitor the session file.** An expired Windy login produces captures that
   look fine but contain logged-out content. Consider alerting if predicted MW
   is flat/zero across a full daylight cycle.
7. **Start Docker on boot** — `sudo systemctl enable docker`. With
   `restart: unless-stopped`, the service then survives instance reboots.
8. **Image size (4.32 GB)** — dominated by Chromium and its system libraries. If
   size matters, drop `scikit-learn` from `requirements.txt` on the forecast host
   and run `train_model.py` elsewhere.
