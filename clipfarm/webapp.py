"""The clipfarm app: a local web UI over the pipeline.

Run `clipfarm app` and open http://127.0.0.1:8000. It only listens on this
machine. API keys typed into the Setup panel are written to .env and are never
sent back to the browser.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

from .config import Settings
from .discover import find_trending
from .source import BROWSERS

STATIC = Path(__file__).parent / "static"
KEYS = ("YOUTUBE_API_KEY", "ELEVENLABS_API_KEY", "ANTHROPIC_API_KEY")
ENV_FILE = Path(".env")

app = Flask(__name__, static_folder=None)
jobs: dict[str, dict] = {}       # id -> {id, label, status, log, clip, created}
job_queue: "queue.Queue[tuple[str, str | dict]]" = queue.Queue()
TEST_FOOTAGE = "test-gameplay.mp4"
auto = {"on": False, "interval": 15, "window": 60, "region": "US", "query": "",
        "last_scan": None, "message": ""}
_auto_wake = threading.Event()


# ------------------------------------------------------------------ jobs

class _JobLog:
    """Sends print() output from the worker thread into the running job's log."""

    def __init__(self, real):
        self.real = real
        self.job_id: str | None = None
        self.thread: threading.Thread | None = None

    def write(self, text):
        if self.job_id and threading.current_thread() is self.thread:
            jobs[self.job_id]["log"] += text
        return self.real.write(text)

    def flush(self):
        self.real.flush()

    def __getattr__(self, name):  # isatty, encoding, fileno, ... from the real stream
        return getattr(self.real, name)


_log = _JobLog(sys.stdout)
_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()


def _worker():
    from . import pipeline  # already imported by serve(); cheap here

    _log.thread = threading.current_thread()
    print("[worker] ready for jobs")
    while True:
        job_id, target = job_queue.get()
        job = jobs[job_id]
        job["status"] = "running"
        _log.job_id = job_id
        try:
            s = Settings.from_env()
            if isinstance(target, dict):  # GTA shorts from uploaded footage
                from . import gameplay

                path = s.footage_dir / target["file"]
                if target["file"] == TEST_FOOTAGE and not path.exists():
                    print("Creating test footage...")
                    gameplay.make_test_footage(path)
                finals = gameplay.make_shorts(path, s, target["count"], target["style"],
                                              target["layout"])
                final = finals[-1]
            elif target == "demo":
                final = pipeline.demo(s)
            else:
                final = pipeline.make_from_url(target, s)
            job["clip"] = final.parent.name
            job["status"] = "done"
        except BaseException as e:  # never let one bad job kill the worker
            detail = str(e) or type(e).__name__
            if not isinstance(e, RuntimeError):
                detail = traceback.format_exc()
            job["log"] += f"\nFailed: {detail}\n"
            job["status"] = "failed"
        finally:
            _log.job_id = None


def _ensure_worker() -> None:
    """Start the worker, or restart it if it has died."""
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_worker, daemon=True, name="clipfarm-worker")
            _worker_thread.start()


def enqueue(target: str | dict, label: str) -> dict:
    job = {"id": uuid.uuid4().hex[:8], "label": label, "status": "queued", "log": "",
           "clip": None, "created": time.time()}
    jobs[job["id"]] = job
    job_queue.put((job["id"], target))
    _ensure_worker()
    return job


def _seen(s: Settings) -> set[str]:
    p = s.output_dir / "seen.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


def _mark_seen(s: Settings, video_id: str) -> None:
    seen = _seen(s) | {video_id}
    s.output_dir.mkdir(parents=True, exist_ok=True)
    (s.output_dir / "seen.json").write_text(json.dumps(sorted(seen)))


def _auto_loop():
    while True:
        if auto["on"]:
            s = Settings.from_env()
            try:
                found = find_trending(s.youtube_api_key, auto["window"], auto["region"],
                                      auto["query"] or None)
                fresh = [c for c in found if c.video_id not in _seen(s)]
                if fresh:
                    top = fresh[0]
                    _mark_seen(s, top.video_id)
                    enqueue(top.url, top.title)
                    auto["message"] = f"Queued: {top.title}"
                else:
                    auto["message"] = "Nothing new in the last scan."
            except Exception as e:
                auto["message"] = f"Scan failed: {e}"
            auto["last_scan"] = time.time()
        _auto_wake.wait(auto["interval"] * 60 if auto["on"] else None)
        _auto_wake.clear()


# ------------------------------------------------------------------- API

def _save_env(values: dict[str, str]) -> None:
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    for key, value in values.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
        for i, line in enumerate(lines):
            if line.split("=", 1)[0].strip() == key:
                lines[i] = f"{key}={value}"
                break
        else:
            lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n")
    ENV_FILE.chmod(0o600)


def _clips(s: Settings) -> list[dict]:
    out = []
    if s.output_dir.is_dir():
        for meta in s.output_dir.glob("*/meta.json"):
            final = meta.parent / "final.mp4"
            if final.exists():
                out.append({**json.loads(meta.read_text()), "id": meta.parent.name,
                            "made": final.stat().st_mtime})
    return sorted(out, key=lambda c: c["made"], reverse=True)


def _footage(s: Settings) -> list[dict]:
    from .gameplay import VIDEO_EXTS

    if not s.footage_dir.is_dir():
        return []
    files = [p for p in s.footage_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS]
    return [{"name": p.name, "mb": round(p.stat().st_size / 1e6, 1)}
            for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)]


@app.post("/api/footage")
def upload_footage():
    from werkzeug.utils import secure_filename

    from .gameplay import VIDEO_EXTS

    s = Settings.from_env()
    s.footage_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in request.files.getlist("files"):
        name = secure_filename(f.filename or "")
        if not name or Path(name).suffix.lower() not in VIDEO_EXTS:
            return jsonify(error=f"{f.filename}: upload a video file (mp4, mov, mkv, webm)."), 400
        f.save(s.footage_dir / name)
        saved.append(name)
    if not saved:
        return jsonify(error="Choose at least one video file."), 400
    return jsonify(saved=saved)


@app.post("/api/gta")
def make_gta():
    from .gameplay import LAYOUTS, STYLES

    body = request.get_json(force=True)
    s = Settings.from_env()
    name = str(body.get("file", ""))
    style, layout = body.get("style", "hype"), body.get("layout", "fullscreen")
    count = int(body.get("count", 3))
    if style not in STYLES or layout not in LAYOUTS or not 1 <= count <= 10:
        return jsonify(error="Pick a style, layout and count."), 400
    if name != TEST_FOOTAGE and name not in {f["name"] for f in _footage(s)}:
        return jsonify(error="That recording is not in the footage folder."), 400
    label = f"{count} {style} short{'s' if count > 1 else ''} from {name}"
    return jsonify(enqueue({"file": name, "count": count, "style": style, "layout": layout}, label))


@app.get("/")
def index():
    return send_file(STATIC / "index.html")


@app.get("/api/state")
def state():
    s = Settings.from_env()
    return jsonify(
        keys={k: bool(os.environ.get(k)) for k in KEYS},
        cookies_browser=s.cookies_browser,
        worker_alive=bool(_worker_thread and _worker_thread.is_alive()),
        footage=_footage(s),
        jobs=sorted(jobs.values(), key=lambda j: j["created"], reverse=True)[:20],
        clips=_clips(s),
        auto=auto,
    )


@app.post("/api/settings")
def settings():
    body = request.get_json(force=True)
    values = {k: body[k].strip() for k in KEYS if isinstance(body.get(k), str) and body[k].strip()}
    if any("\n" in v or "\r" in v for v in values.values()):
        abort(400)
    if "cookies_browser" in body:
        browser = str(body["cookies_browser"]).strip().lower()
        if browser and browser not in BROWSERS:
            abort(400)
        values["CLIPFARM_COOKIES_BROWSER"] = browser
    if values:
        _save_env(values)
    return jsonify(ok=True)


@app.post("/api/discover")
def discover():
    body = request.get_json(force=True)
    s = Settings.from_env()
    try:
        found = find_trending(
            s.youtube_api_key, int(body.get("window", 60)), body.get("region") or "US",
            body.get("query") or None, bool(body.get("creative_commons")),
        )
    except RuntimeError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=f"Could not reach YouTube: {e}"), 502
    seen = _seen(s)
    return jsonify(videos=[{**c.to_dict(), "seen": c.video_id in seen} for c in found[:25]])


@app.post("/api/make")
def make():
    body = request.get_json(force=True)
    url = (body.get("url") or "").strip()
    if url == "demo":
        return jsonify(enqueue("demo", "Offline demo clip"))
    if not url.startswith(("https://www.youtube.com/", "https://youtube.com/",
                           "https://youtu.be/", "https://m.youtube.com/")):
        return jsonify(error="Paste a YouTube link (https://www.youtube.com/watch?v=...)."), 400
    if body.get("video_id"):
        _mark_seen(Settings.from_env(), body["video_id"])
    return jsonify(enqueue(url, body.get("title") or url))


@app.post("/api/auto")
def set_auto():
    body = request.get_json(force=True)
    auto["on"] = bool(body.get("on"))
    auto["interval"] = max(5, int(body.get("interval", auto["interval"])))
    auto["window"] = int(body.get("window", auto["window"]))
    auto["region"] = body.get("region") or "US"
    auto["query"] = body.get("query") or ""
    auto["message"] = "Auto-farm on." if auto["on"] else "Auto-farm off."
    _auto_wake.set()
    return jsonify(auto)


@app.get("/media/<clip_id>/final.mp4")
def media(clip_id: str):
    path = (Settings.from_env().output_dir / clip_id / "final.mp4").resolve()
    if not clip_id.replace("-", "").replace("_", "").isalnum() or not path.exists():
        abort(404)
    return send_file(path, mimetype="video/mp4", conditional=True,
                     as_attachment="download" in request.args,
                     download_name=f"clipfarm-{clip_id}.mp4")


def serve(port: int = 8000) -> None:
    # Import the whole pipeline here, on the main thread, so a missing
    # dependency stops the app with a clear message instead of silently
    # killing the background worker.
    from . import pipeline  # noqa: F401

    sys.stdout = _log
    _ensure_worker()
    threading.Thread(target=_auto_loop, daemon=True).start()
    print(f"clipfarm is running: open http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)
