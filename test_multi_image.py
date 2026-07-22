"""
FULL AUTOMATION (PREMIUM): Playwright captures Windy screenshots using your
logged-in Premium account, records a short animated video of the cloud
movement, then runs a LOCAL feature-extraction + ML pipeline (NO LLM) to
predict 15-minute-block solar generation.

Pipeline (see run_pipeline.py for the orchestration):
    Screenshots --> image_feature_extraction.py --\\
                                                     >-- feature_builder.py -> ml_forecast_model.py -> prediction_store.py
    Video       --> video_motion_features.py ------/

Setup:
1. pip install -r requirements.txt
2. playwright install chromium
3. Update PLANT_NAME / PLANT_LAT / PLANT_LON / PLANT_CAPACITY_MW in
   config.py with your actual plant details.

Run:
    python test_multi_image.py
"""

import time
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from playwright.sync_api import sync_playwright

from config import (
    PLANT_NAME, PLANT_LAT, PLANT_LON, ZOOM_LEVEL, VIEWPORT_WIDTH, VIEWPORT_HEIGHT,
    LAYERS, RECORD_ANIMATION_VIDEO, ANIMATION_LAYER, ANIMATION_RECORD_SECONDS,
    VIDEO_DIR, STORAGE_STATE_PATH, SCREENSHOT_DIR, RUN_INTERVAL_SECONDS,
    POST_PLAY_CAPTURE_SECONDS, RECORD_MAX_ATTEMPTS, TIMEZONE, CAPTURE_TIMES,
    HEARTBEAT_PATH,
)
from run_pipeline import run_prediction_pipeline
from batch import uploader
from batch import video_validation


def ensure_login():
    """First run only: opens a VISIBLE browser so you can log in to your
    Windy Premium account. Saves the session so future runs are headless
    and still use your premium access."""
    if STORAGE_STATE_PATH.exists():
        return

    print("No saved login found.")
    print("A browser window will open -- please log in to your Windy")
    print("PREMIUM account there, then come back here and press Enter.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.windy.com/", wait_until="domcontentloaded", timeout=60000)
        input("Press Enter here once you are logged in on the browser window... ")
        context.storage_state(path=str(STORAGE_STATE_PATH))
        browser.close()

    print("Login saved to windy_login.json. Future runs will be automatic.\n")


def dismiss_popups(page):
    """Tries to close any cookie-consent / install-app / promo popups that
    Windy sometimes shows, since these can sit on top of the map/panel in
    the screenshot."""
    possible_texts = ["Accept", "I agree", "Got it", "Agree", "Close", "OK", "Allow all"]
    for text in possible_texts:
        try:
            btn = page.get_by_text(text, exact=False).first
            if btn.is_visible(timeout=1000):
                btn.click(timeout=1000)
                page.wait_for_timeout(500)
        except Exception:
            pass

    try:
        page.keyboard.press("Escape")
    except Exception:
        pass


def set_weather_picker_point(page):
    """
    Right-clicks the plant's location on the map (map center, since the
    page is already centered on PLANT_LAT/PLANT_LON) to open Windy's
    right-click context menu, then clicks "Show weather picker" from that
    menu. This drops Windy's own weather-picker point exactly on the
    plant's coordinates -- this is what should be visible in the
    screenshot, instead of a leftover/older picked point.
    """
    try:
        center_x = VIEWPORT_WIDTH // 2
        center_y = VIEWPORT_HEIGHT // 2

        page.mouse.click(center_x, center_y, button="right")
        print(f"  Right-clicked map center ({center_x}, {center_y}) to open context menu.")
        page.wait_for_timeout(1000)

        picker_option = page.get_by_text("Show weather picker", exact=False).first
        if picker_option.is_visible(timeout=3000):
            picker_option.click(timeout=3000)
            print("  Clicked 'Show weather picker' from context menu.")
            page.wait_for_timeout(1500)  # let the weather-picker point render
        else:
            print("  [WARN] 'Show weather picker' option not visible in context menu.")
    except Exception as e:
        print(f"  [WARN] Could not open weather picker via right-click: {e}")


def capture_all_layers() -> dict:
    """
    Opens Windy.com for each layer in LAYERS (using the saved premium
    session). Putting lat/lon in the URL PATH (not just the query string)
    makes Windy treat it as a searched/picked location -- the same as
    manually typing coordinates into the search box -- which auto-opens
    the bottom wide hourly-forecast panel. No manual click is needed.
    Returns a dict of {filepath: description}.

    Each run's screenshots are saved into their OWN timestamped subfolder
    (windy_screenshots/<lat>_<lon>/<YYYY-MM-DD_HH-MM-SS>/<layer>.png)
    instead of a fixed filename, so a new run no longer overwrites and
    permanently deletes the previous run's screenshots. This doesn't
    change what goes into features_log.csv -- feature extraction already
    happened before the old file would've been overwritten -- it only
    means past screenshots are now kept around for debugging/reference.
    """
    captured = {}
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = SCREENSHOT_DIR / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            storage_state=str(STORAGE_STATE_PATH),
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
        )
        page = context.new_page()

        for overlay, description in LAYERS.items():
            url = (
                f"https://www.windy.com/{PLANT_LAT}/{PLANT_LON}"
                f"?{overlay},{PLANT_LAT},{PLANT_LON},{ZOOM_LEVEL},p:cities"
            )
            print(f"Opening layer '{overlay}' -> {url}")

            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(6000)  # let map tiles fully render

            dismiss_popups(page)
            page.wait_for_timeout(2000)  # let bottom forecast panel finish animating in

            # Right-click the plant's location and select "Show weather
            # picker" so the correct point is set on the map before the
            # screenshot is taken (instead of an old/leftover picked point).
            set_weather_picker_point(page)

            out_path = run_dir / f"{overlay}.png"
            page.screenshot(path=str(out_path))
            captured[str(out_path)] = description
            print(f"  [OK] Saved {out_path}")

        browser.close()

    return captured


def dismiss_timeline_overlay(page):
    """Some layers (e.g. satellite nowcast) show a white info box
    on top of the timeline -- things like '6:43 PM - 5h ago', '24h ago /
    6h ago / 1h ago / Next 1h', 'Overlay with radar', 'Blue / Visible /
    Infra', 'More options...'. This box can sit on top of (or hide) the
    play button, so it needs to be dismissed first. Since the exact class
    name can vary by layer, this tries several likely selectors in order
    and moves on quietly if none match (nothing to close = fine)."""
    candidates = [
        "div.closing-x",
        "[aria-label='Close']",
        "[title='Close']",
        "div.timeline-info .close",
        "div.nowcast-info .close",
        "svg[class*='close']",
        "div[class*='closing']",
    ]
    for sel in candidates:
        try:
            el = page.locator(sel).first
            if el.is_visible(timeout=1000):
                el.click(timeout=1000)
                print(f"  Closed overlay using selector: {sel}")
                page.wait_for_timeout(500)
                return True
        except Exception:
            continue
    print("  [INFO] No closable overlay found (may already be closed) -- continuing.")
    return False


def seek_timeline_to_one_hour_ago(page) -> bool:
    """
    Clicks the '1h ago' label/tick on the nowcast timeline so the playhead
    jumps back to an hour in the past BEFORE play is pressed. Combined
    with clicking play afterwards and "Play with forecast" being enabled,
    this makes the recorded animation move from ~1 hour ago, through
    'now', and on to the forecast frames -- instead of only showing
    forecast frames starting from wherever 'now' happened to be.
    """
    # Windy's satellite nowcast renders the time-range choices as a row of
    # segment buttons: "24h ago" | "6h ago" | "1h ago" | "Next 1h".
    # These are real, stable, semantic class names (not hashed), so target the
    # segment element directly. The previous get_by_text(...).first approach
    # resolved to an outer wrapper node that was not clickable, which is why it
    # always fell through to the warning.
    try:
        # Wait for the segment row to actually exist rather than relying on the
        # caller's fixed sleep -- on slower loads the timeline renders after it,
        # which made this seek fail intermittently.
        page.wait_for_selector(".radsat__segment", state="visible", timeout=15000)
        segments = page.locator(".radsat__segment")
        count = segments.count()
        for i in range(count):
            seg = segments.nth(i)
            if seg.inner_text(timeout=2000).strip().lower() == "1h ago":
                # force=True: the segment is visible and hit-testable, but the
                # timeline animates continuously, so Playwright's "element is
                # stable" actionability check never settles and the click times
                # out. Visibility is confirmed above, so skip that check.
                seg.scroll_into_view_if_needed(timeout=2000)
                seg.click(timeout=3000, force=True)
                print("  Seeked timeline back using segment button: '1h ago'")
                page.wait_for_timeout(800)
                return True
    except Exception as e:
        print(f"  [DEBUG] Segment-based timeline seek failed: {e}")

    # Fallback: exact-text match anywhere on the page (covers other layers /
    # future markup that does not use .radsat__segment).
    for text in ["1h ago", "1 h ago", "-1h", "1hr ago"]:
        try:
            el = page.get_by_text(text, exact=True).first
            if el.is_visible(timeout=2000):
                el.click(timeout=2000)
                print(f"  Seeked timeline back using label: '{text}'")
                page.wait_for_timeout(800)
                return True
        except Exception:
            continue

    print("  [WARN] Could not seek timeline to '1h ago' automatically -- "
          "the animation may start from 'now' instead of an hour in the "
          "past. Inspect the timeline's '1h ago' tick/label (right-click "
          "-> Inspect) and add its exact selector to "
          "seek_timeline_to_one_hour_ago().")
    return False


def click_play_button(page, attempts: int = 3, per_try_timeout: int = 4000) -> bool:
    """Tries several likely selectors for the play (>) button, since
    different Windy layers (clouds vs satellite nowcast) use
    different play controls. Confirmed working selector for the Clouds
    layer is 'div.play-pause' -- the others are fallbacks for layers
    (like Satellite) that use a different widget.

    Retries the whole candidate list up to `attempts` times with a short
    pause in between, since the button can simply not be rendered yet on
    slower page loads -- a longer per-try timeout and a couple of retries
    fixes most "could not find play button" flakiness without needing an
    exact selector."""
    candidates = [
        "div.play-pause",
        "[title='Play']",
        "[aria-label='Play']",
        "svg[class*='play']",
        "div[class*='play']",
        ".ecmwf-timeline .play",
        "button[class*='play']",
    ]
    for attempt in range(1, attempts + 1):
        for sel in candidates:
            try:
                el = page.locator(sel).first
                if el.is_visible(timeout=per_try_timeout):
                    el.click(timeout=per_try_timeout)
                    print(f"  Clicked play using selector: {sel} (attempt {attempt})")
                    return True
            except Exception:
                continue
        if attempt < attempts:
            print(f"  [INFO] Play button not found yet (attempt {attempt}/{attempts}) -- waiting and retrying...")
            page.wait_for_timeout(2500)
    print("  [WARN] Could not find any play button automatically -- "
          "the video will still record, but the map may stay static. "
          "Inspect the real play button (right-click -> Inspect) and "
          "add its exact selector to the 'candidates' list above.")
    return False


def click_play_with_forecast(page) -> bool:
    """
    Enables the 'Play with forecast' toggle that sits next to the
    play/pause button on Satellite/nowcast layers (visible in your
    screenshot, to the right of the timeline). Since the exact class name
    isn't confirmed via inspect yet, this tries a few reasonable ways to
    find and click it, in order:
      1. Click directly on the "Play with forecast" text label -- on most
         sites clicking a toggle's label also flips the toggle.
      2. If that doesn't work, look for a toggle/checkbox/switch element
         that sits immediately next to that text and click it directly.
    """
    # Windy builds these toggles as `div.checkbox` whose ON/OFF state is carried
    # in a `checkbox--off` / `checkbox--on` class. Read that state first: the
    # previous version clicked unconditionally, which silently turned the toggle
    # back OFF whenever it happened to already be ON.
    try:
        toggle = page.locator("div.checkbox", has_text="Play with forecast").first
        if toggle.is_visible(timeout=3000):
            classes = toggle.get_attribute("class") or ""
            if "checkbox--off" not in classes:
                print("  'Play with forecast' is already enabled -- leaving it on.")
                return True
            toggle.click(timeout=3000)
            page.wait_for_timeout(500)
            print("  Enabled 'Play with forecast' via its checkbox element.")
            return True
    except Exception as e:
        print(f"  [DEBUG] Checkbox-based 'Play with forecast' toggle failed: {e}")

    try:
        label = page.get_by_text("Play with forecast", exact=False).first
        if label.is_visible(timeout=3000):
            label.click(timeout=3000)
            print("  Clicked 'Play with forecast' label to enable it.")
            page.wait_for_timeout(500)
            return True
    except Exception as e:
        print(f"  [DEBUG] Clicking 'Play with forecast' label failed: {e}")

    # Fallback: try clicking a toggle/switch/checkbox element sitting right
    # next to the label (covers the case where the label itself isn't
    # clickable and the actual switch is a separate sibling element).
    # NOTE: the previous fallback list also tried
    #   //*[contains(text(),'Play with forecast')]/parent::*//*[contains(@class,'switch')]
    # and the equivalent 'toggle' variant. Those are DANGEROUS on the current
    # Windy UI: the anchor text is no longer rendered, and the broad
    # class-contains match can resolve to an unrelated control -- in particular
    # the satellite display-style switch (Blue / Visible / Infra). Clicking that
    # silently changes how the satellite imagery is coloured, which corrupts the
    # brightness/hue features this pipeline extracts. They are removed; only the
    # checkbox-scoped locator (the actual component type Windy uses for these
    # toggles) is kept.
    fallback_selectors = [
        "xpath=//*[contains(text(),'Play with forecast')]/parent::*//*[contains(@class,'checkbox')]",
    ]
    for sel in fallback_selectors:
        try:
            el = page.locator(sel).first
            if el.is_visible(timeout=1500):
                el.click(timeout=1500)
                print(f"  Clicked 'Play with forecast' toggle using fallback selector: {sel}")
                page.wait_for_timeout(500)
                return True
        except Exception:
            continue

    # Verified against the live satellite nowcast page: the only toggles Windy
    # now renders there are 'Overlay with radar', 'pressure' and
    # 'particles animation' -- there is no 'Play with forecast' control at all.
    # This is therefore an expected no-op on the current UI, not a broken
    # selector. Recording still proceeds normally.
    print("  [INFO] 'Play with forecast' control is not present in the current "
          "Windy satellite UI -- skipping. Recording continues; the animation "
          "plays over the nowcast range only.")
    return False


def click_slow_animation_speed(page) -> bool:
    """
    Selects the SLOWEST playback speed. Windy shows this as a row of
    THREE ICON-ONLY buttons (turtle, rabbit, llama -- no visible text
    label like "Speed"), sitting immediately to the LEFT of the
    "Play with forecast" toggle on the same row. Because there's no text
    to search for, this locates "Play with forecast" first (which IS
    text, and already works reliably elsewhere in this script), then
    clicks at a pixel offset to its left where the turtle (slowest, i.e.
    first/leftmost) icon sits.
    """
    # NOTE: the previous implementation measured the "Play with forecast" label's
    # bounding box and then issued a raw page.mouse.click() at a computed offset
    # to its left. That offset is not over any speed control on the current
    # Windy UI -- it lands on the MAP itself, which can drop a pin or shift the
    # view in the middle of a recording. It is removed rather than retuned:
    # a blind coordinate click is worse than not setting the speed at all.
    #
    # Windy exposes its icon-only button groups as `.switch__item` elements
    # inside a `.switch` container, with the container's purpose in a
    # `data-tooltip` attribute. Try those semantic locators instead.
    speed_group_selectors = [
        ".switch[data-tooltip*='speed' i] .switch__item",
        "[data-tooltip*='animation speed' i] .switch__item",
        "#animation-speed .switch__item",
    ]
    for group_sel in speed_group_selectors:
        try:
            items = page.locator(group_sel)
            if items.count() > 0:
                # The slowest option is the first item in the group.
                items.first.click(timeout=2000)
                print(f"  Selected slowest animation speed via: {group_sel}")
                page.wait_for_timeout(500)
                return True
        except Exception:
            continue

    # Fallback: in case some Windy layer/version DOES expose usable
    # title/aria-label attributes on these icons.
    fallback_selectors = [
        "[title*='slow' i]",
        "[aria-label*='slow' i]",
        "[title*='turtle' i]",
        "[aria-label*='turtle' i]",
    ]
    for sel in fallback_selectors:
        try:
            el = page.locator(sel).first
            if el.is_visible(timeout=1500):
                el.click(timeout=1500)
                print(f"  Selected slow animation speed using fallback selector: {sel}")
                page.wait_for_timeout(500)
                return True
        except Exception:
            continue

    # Verified against the live satellite nowcast page: Windy no longer exposes
    # the turtle/rabbit/llama speed selector there, so there is nothing to
    # click. Expected no-op on the current UI rather than a broken selector.
    print("  [INFO] Animation speed selector is not present in the current "
          "Windy satellite UI -- skipping. The animation plays at Windy's "
          "default speed (recording duration is unchanged).")
    return False


def click_plant_marker(page):

    """
    Clicks the center of the map viewport once, so Windy drops a pin/
    pointer marker exactly on the plant's coordinates. Since the page URL
    already centers the map on (PLANT_LAT, PLANT_LON), the center of the
    viewport IS the plant's location -- so a plain center-click places
    the marker there without needing to search/type coordinates again.
    """
    try:
        center_x = VIEWPORT_WIDTH // 2
        center_y = VIEWPORT_HEIGHT // 2
        page.mouse.click(center_x, center_y)
        print(f"  Clicked map center ({center_x}, {center_y}) to drop a pointer on the plant location.")
        page.wait_for_timeout(1000)  # let the marker/pin render
    except Exception as e:
        print(f"  [WARN] Could not click map to place pointer: {e}")


def _capture_full_session() -> Path | None:
    """Runs one full browser lifecycle and returns the raw recorded .webm.

    Every Playwright resource created here (browser, context, page) is released
    in a finally block before returning -- even on error -- so no context, page
    or Chromium process is ever leaked between runs.
    """
    DEDICATED_NOWCAST_URLS = {
        # Satellite only gets a real animated timeline + play button on Windy's
        # DEDICATED nowcast page; the generic "/{lat}/{lon}?..." URL opens a
        # static forecast page with no play button.
        "satellite": "https://www.windy.com/-Satellite-satellite?satellite,{lat},{lon},{zoom},p:cities",
    }
    if ANIMATION_LAYER in DEDICATED_NOWCAST_URLS:
        url = DEDICATED_NOWCAST_URLS[ANIMATION_LAYER].format(
            lat=PLANT_LAT, lon=PLANT_LON, zoom=ZOOM_LEVEL
        )
    else:
        url = (
            f"https://www.windy.com/{PLANT_LAT}/{PLANT_LON}"
            f"?{ANIMATION_LAYER},{PLANT_LAT},{PLANT_LON},{ZOOM_LEVEL},p:cities"
        )

    browser = context = page = video_obj = None
    raw_path_str = None
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                storage_state=str(STORAGE_STATE_PATH),
                viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
                record_video_dir=str(VIDEO_DIR),
                record_video_size={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            )
            # Cap every operation so a hung page can never wedge the run forever.
            context.set_default_timeout(60000)
            context.set_default_navigation_timeout(60000)
            page = context.new_page()

            page.goto(url, wait_until="domcontentloaded", timeout=60000)

            # Let map tiles + timeline frames load before pressing play.
            page.wait_for_timeout(15000)
            dismiss_popups(page)
            page.wait_for_timeout(1500)

            dismiss_timeline_overlay(page)
            seek_timeline_to_one_hour_ago(page)

            # Marker BEFORE play so the click is not injected into the captured
            # animation window (it drops a pin; harmless, but keeping it out of
            # the recording keeps the motion check clean).
            click_plant_marker(page)

            click_play_button(page)
            click_slow_animation_speed(page)
            click_play_with_forecast(page)

            # Keep recording long enough that the moving animation is guaranteed
            # to be somewhere in the clip. Windy's nowcast buffers for a long,
            # variable time after play before it actually animates, so we record
            # generously and pick the moving window afterwards (see
            # _trim_to_motion) instead of trusting a fixed offset.
            print(f"  Capturing {POST_PLAY_CAPTURE_SECONDS}s of animation "
                  f"(moving window is selected afterwards)...")
            # Use time.sleep, NOT page.wait_for_timeout, for the long capture:
            # a multi-tens-of-seconds driver-side wait trips Playwright with
            # "Event loop is closed". time.sleep blocks Python while the browser
            # keeps recording untouched -- the approach the original used.
            time.sleep(POST_PLAY_CAPTURE_SECONDS)

            video_obj = page.video
        finally:
            # Order matters: close context first (finalizes the video file),
            # then the browser. Guarded so a failure in one still attempts the
            # other -- no leaked Chromium.
            if context is not None:
                try:
                    context.close()
                except Exception as exc:
                    print(f"  [WARN] context.close() failed: {exc}")
            if browser is not None:
                try:
                    browser.close()
                except Exception as exc:
                    print(f"  [WARN] browser.close() failed: {exc}")

        # video_obj.path() must be read while Playwright is still running (i.e.
        # still inside this `with sync_playwright()` block); calling it after the
        # block exits raises "Event loop is closed". The video file is already
        # finalized by context.close() above.
        raw_path_str = video_obj.path() if video_obj is not None else None

    if raw_path_str is None:
        return None

    raw_path = Path(raw_path_str)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = VIDEO_DIR / f"{PLANT_NAME}_{ANIMATION_LAYER}_{timestamp}_full.webm"
    try:
        raw_path.rename(full_path)
    except Exception:
        full_path = raw_path
    print(f"  [OK] Full video saved: {full_path.resolve()}")
    return full_path


def _trim_to_motion(full_path: Path) -> Path | None:
    """Trims the ANIMATION_RECORD_SECONDS window with the MOST motion out of the
    full recording (not a fixed offset), so the clean clip contains the actual
    moving animation rather than the frozen warm-up. Returns the clean mp4 path,
    or None if ffmpeg is unavailable.
    """
    start, mean_diff = video_validation.best_motion_window(
        full_path, ANIMATION_RECORD_SECONDS
    )
    print(f"  Most-motion window starts at {start}s "
          f"(mean inter-frame diff {mean_diff:.3f}); trimming "
          f"{ANIMATION_RECORD_SECONDS}s from there.")

    clean_path = full_path.with_name(full_path.name.replace("_full.webm", "_clean.mp4"))
    try:
        import subprocess
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(full_path),
                "-t", str(ANIMATION_RECORD_SECONDS),
                str(clean_path),
            ],
            check=True,
            capture_output=True,
        )
    except Exception as e:
        print(f"  [WARN] Could not trim video with ffmpeg ({e}).")
        return None

    print(f"  [OK] Clean trimmed clip saved: {clean_path.resolve()}")
    return clean_path


def record_cloud_animation() -> Path | None:
    """
    Records the animated satellite layer around the plant and returns the path
    to a clean mp4 that is VERIFIED to contain a moving animation -- or None if
    every attempt failed.

    Reliability contract (do not weaken):
      * Each attempt is a full, isolated browser lifecycle with guaranteed
        cleanup (_capture_full_session).
      * The clip is trimmed to the actually-moving window, then checked with
        video_validation.is_moving. A frozen clip is never returned.
      * Up to RECORD_MAX_ATTEMPTS tries; if none move, returns None so the
        caller fails the run and uploads nothing.
    """
    if not RECORD_ANIMATION_VIDEO:
        return None

    print(f"\nRecording the '{ANIMATION_LAYER}' animation "
          f"(want {ANIMATION_RECORD_SECONDS}s of real motion)...")

    for attempt in range(1, RECORD_MAX_ATTEMPTS + 1):
        print(f"\n  --- Recording attempt {attempt}/{RECORD_MAX_ATTEMPTS} ---")
        try:
            full_path = _capture_full_session()
        except Exception as exc:
            print(f"  [WARN] Recording attempt {attempt} raised: {exc}")
            continue

        if full_path is None:
            print(f"  [WARN] Attempt {attempt} produced no video file.")
            continue

        clean_path = _trim_to_motion(full_path)
        if clean_path is None:
            continue

        moving, mean_diff, moving_frac = video_validation.is_moving(
            clean_path, ANIMATION_RECORD_SECONDS
        )
        if moving:
            print(f"  [OK] Motion verified (mean diff {mean_diff:.3f}, "
                  f"{moving_frac*100:.0f}% of seconds moving). Accepting clip.")
            return clean_path

        print(f"  [WARN] Attempt {attempt} clip is frozen/static "
              f"(mean diff {mean_diff:.3f}, only {moving_frac*100:.0f}% moving) "
              f"-- discarding and retrying.")
        try:
            clean_path.unlink(missing_ok=True)
        except Exception:
            pass

    print(f"  [ERROR] Recording failed after {RECORD_MAX_ATTEMPTS} attempts -- "
          f"no moving animation captured. This run will NOT upload a video.")
    return None


RUN_INTERVAL = RUN_INTERVAL_SECONDS


def _recording_datetime_from_path(video_path) -> datetime.datetime:
    """Recovers the moment a recording was made from its filename.

    Files are named "<PLANT>_<LAYER>_<YYYY-MM-DD>_<HH-MM-SS>_clean.mp4", so the
    timestamp is read back from there rather than using "now" (the upload runs
    after trimming, which takes a while). Falls back to the file's mtime if the
    name does not parse.
    """
    stem = Path(video_path).stem  # drops ".mp4"
    if stem.endswith("_clean"):
        stem = stem[: -len("_clean")]

    parts = stem.split("_")
    if len(parts) >= 2:
        try:
            return datetime.datetime.strptime(
                f"{parts[-2]}_{parts[-1]}", "%Y-%m-%d_%H-%M-%S"
            )
        except ValueError:
            pass

    return datetime.datetime.fromtimestamp(Path(video_path).stat().st_mtime)


def upload_recording(video_path) -> None:
    """Step 3: send the clean MP4 to the backend.

    Never raises: an upload problem (backend down, bad URL, timeout) must not
    stop feature extraction and prediction, which work purely from local files.
    """
    print("\nStep 3: Uploading video...")

    if video_path is None:
        print("Upload skipped: no clean MP4 was produced by this run.")
        return

    # Final gate before the network call: never upload a corrupt or wrong-length
    # recording. (Motion was already verified inside record_cloud_animation;
    # this re-checks file integrity and duration as defence in depth.)
    playable, reason = video_validation.validate_playable(
        video_path, ANIMATION_RECORD_SECONDS
    )
    if not playable:
        print(f"Upload skipped: recording failed validation ({reason}).")
        return

    try:
        recording_dt = _recording_datetime_from_path(video_path)
        result = uploader.upload_video(video_path, PLANT_NAME, recording_dt)
    except Exception as exc:
        # Deliberately broad: nothing about uploading may abort the pipeline.
        print(f"Upload failed: {exc}")
        return

    # The backend replies {"success": true, "data": {... "key": "videos/..."}},
    # so look inside "data" first and fall back to a top-level key (and finally
    # to the whole body) if that response shape ever changes.
    payload = result.get("data") if isinstance(result.get("data"), dict) else result
    s3_key = payload.get("key") or payload.get("s3_key") or payload.get("Key")
    print("Upload successful.")
    print(f"S3 Key: {s3_key if s3_key else result}")


def run_once():
    """Captures screenshots, records a cloud-animation video, and runs
    the LOCAL feature-extraction + ML prediction pipeline (no LLM)."""
    print("Step 1: Capturing screenshots (satellite + wind + solarpower + clouds + rain)...\n")
    image_map = capture_all_layers()
    _touch_heartbeat()

    print("\nStep 2: Recording cloud movement animation...")
    video_path = record_cloud_animation()
    _touch_heartbeat()

    upload_recording(video_path)
    _touch_heartbeat()

    print("\nStep 4: Running local feature-extraction + ML prediction pipeline (no LLM)...")
    run_prediction_pipeline(image_map, video_path)
    _touch_heartbeat()


# ---------------------------------------------------------------------
# Fixed-time scheduler (Asia/Kolkata) -- no cron, no external scheduler.
# ---------------------------------------------------------------------

_TZ = ZoneInfo(TIMEZONE)

# How often the loop wakes while waiting: refreshes the liveness heartbeat and
# keeps SIGTERM shutdown responsive even across the long overnight gap.
HEARTBEAT_INTERVAL_SECONDS = 60


def _now() -> datetime.datetime:
    return datetime.datetime.now(_TZ)


def _touch_heartbeat() -> None:
    """Refreshes the heartbeat file the Docker healthcheck watches. Best-effort;
    a write failure must never crash the loop."""
    try:
        HEARTBEAT_PATH.write_text(_now().isoformat(), encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] could not update heartbeat: {exc}")


def next_capture_datetime(now: datetime.datetime) -> datetime.datetime:
    """Returns the next scheduled capture time strictly after `now`.

    Computed purely from the wall-clock schedule (today, else tomorrow's first
    slot), so it never drifts with processing time and a scheduled slot missed
    because a previous run overran is simply skipped -- captures only ever
    happen at CAPTURE_TIMES.
    """
    for day_offset in (0, 1):
        day = (now + datetime.timedelta(days=day_offset)).date()
        for hour, minute in sorted(CAPTURE_TIMES):
            candidate = datetime.datetime.combine(
                day, datetime.time(hour, minute)
            ).replace(tzinfo=_TZ)
            if candidate > now:
                return candidate
    # CAPTURE_TIMES is non-empty, so this is unreachable.
    raise RuntimeError("no capture times configured")


def _is_scheduled_now(now: datetime.datetime) -> bool:
    """True if `now` falls exactly on a scheduled minute (startup fast-path)."""
    return (now.hour, now.minute) in CAPTURE_TIMES


def _sleep_until(target: datetime.datetime) -> None:
    """Sleeps until `target`, waking every HEARTBEAT_INTERVAL_SECONDS to refresh
    the heartbeat (so health stays green through long waits)."""
    while True:
        remaining = (target - _now()).total_seconds()
        if remaining <= 0:
            return
        _touch_heartbeat()
        time.sleep(min(remaining, HEARTBEAT_INTERVAL_SECONDS))


def _do_run(run_count: int) -> None:
    start = _now().strftime("%Y-%m-%d %H:%M:%S %Z")
    print("\n" + "#" * 60)
    print(f"# RUN {run_count} -- started at {start}")
    print("#" * 60)
    _touch_heartbeat()
    try:
        run_once()
    except Exception as e:
        print(f"\n[ERROR] Run {run_count} failed: {e}")
    _touch_heartbeat()


def main():
    ensure_login()
    _touch_heartbeat()

    schedule = ", ".join(f"{h:02d}:{m:02d}" for h, m in sorted(CAPTURE_TIMES))
    print(f"Scheduler active ({TIMEZONE}). Capture times: {schedule}")

    run_count = 0

    # Starting exactly on a scheduled minute runs immediately.
    if _is_scheduled_now(_now()):
        run_count += 1
        _do_run(run_count)

    while True:
        target = next_capture_datetime(_now())
        print(f"\nNext capture scheduled for {target.strftime('%Y-%m-%d %H:%M:%S %Z')} "
              f"(waiting; Ctrl+C to stop).")
        _sleep_until(target)
        run_count += 1
        _do_run(run_count)


if __name__ == "__main__":
    main()