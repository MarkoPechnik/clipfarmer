"""Talk to YouTube through yt-dlp: metadata, transcript, and the clip itself."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Line:
    """One caption line, in seconds from the start of the video."""
    start: float
    end: float
    text: str


@dataclass
class SourceInfo:
    video_id: str
    title: str
    channel: str
    url: str
    duration: float
    transcript: list[Line] = field(default_factory=list)
    # yt-dlp "most replayed" data: [{start_time, end_time, value}], often
    # missing on videos that are only minutes old.
    heatmap: list[dict] = field(default_factory=list)


def parse_json3(data: dict) -> list[Line]:
    """YouTube 'json3' caption format -> lines."""
    lines = []
    for ev in data.get("events", []):
        text = "".join(seg.get("utf8", "") for seg in ev.get("segs", []))
        text = " ".join(text.split())
        if not text:
            continue
        start = ev.get("tStartMs", 0) / 1000
        lines.append(Line(start, start + ev.get("dDurationMs", 0) / 1000, text))
    return lines


def _pick_caption_track(info: dict) -> str | None:
    for group in ("subtitles", "automatic_captions"):
        tracks = info.get(group) or {}
        langs = [l for l in tracks if l == "en"] + [l for l in tracks if l.startswith("en")]
        langs += [l for l in tracks if l not in langs]
        for lang in langs:
            for fmt in tracks[lang]:
                if fmt.get("ext") == "json3" and fmt.get("url"):
                    return fmt["url"]
    return None


def fetch_info(url: str) -> SourceInfo:
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        transcript: list[Line] = []
        track = _pick_caption_track(info)
        if track:
            try:
                transcript = parse_json3(json.loads(ydl.urlopen(track).read()))
            except Exception as e:  # captions are a nice-to-have
                print(f"  ! could not fetch captions: {e}")
    return SourceInfo(
        video_id=info["id"],
        title=info.get("title", ""),
        channel=info.get("channel") or info.get("uploader") or "",
        url=info.get("webpage_url", url),
        duration=float(info.get("duration") or 0),
        transcript=transcript,
        heatmap=info.get("heatmap") or [],
    )


def download_section(url: str, start: float, end: float, out: Path) -> Path:
    """Download only [start, end] of the video as an mp4."""
    import yt_dlp
    from yt_dlp.utils import download_range_func

    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "merge_output_format": "mp4",
        "outtmpl": str(out.with_suffix("")) + ".%(ext)s",
        "overwrites": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    if not out.exists():
        raise RuntimeError(f"yt-dlp did not produce {out}")
    return out
