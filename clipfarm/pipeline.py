"""End to end: one source video -> one finished reaction clip."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from . import voice
from .character import Character, render_panel
from .compose import compose, probe
from .config import Settings
from .moments import make_plan
from .source import Line, SourceInfo, download_section, fetch_info


def produce(info: SourceInfo, s: Settings, local_clip: Path | None = None) -> Path:
    """Build the clip for `info`. `local_clip` skips the download (demo mode)."""
    work = s.output_dir / info.video_id
    work.mkdir(parents=True, exist_ok=True)

    plan = make_plan(info, s.clip_seconds, s.use_claude)
    print(f"  plan ({plan.planner}): {plan.start:.1f}s-{plan.end:.1f}s, "
          f"{len(plan.reactions)} reactions")

    clip = local_clip or download_section(info.url, plan.start, plan.end, work / "source.mp4")
    duration, has_audio = probe(clip)

    lines = voice.synthesize(plan.reactions, work, s)
    scheduled = voice.schedule(plan.reactions, lines, duration)
    track = voice.mix_track(scheduled, duration, work / "voice.wav")
    for r, _, seconds in scheduled:
        print(f"    {r.at:5.1f}s  {r.emotion:<6} {r.text!r} ({seconds:.1f}s)")

    credit = f"via {info.channel}" if info.channel else ""
    panel = render_panel(Character(s.character_name, credit), scheduled, track, duration,
                         work / "panel.mp4")
    final = compose(clip, panel, work / "voice.wav", duration, has_audio, work / "final.mp4")

    (work / "meta.json").write_text(json.dumps({
        "title": plan.title,
        "source_url": info.url,
        "source_title": info.title,
        "source_channel": info.channel,
        "source_start": plan.start,
        "source_end": plan.end,
        "planner": plan.planner,
        "voice": "elevenlabs" if s.elevenlabs_api_key else "placeholder",
        "reactions": [asdict(r) for r, _, _ in scheduled],
        "description": f"Original video: {info.url} (credit: {info.channel})",
    }, indent=2))
    return final


def make_from_url(url: str, s: Settings) -> Path:
    print(f"> {url}")
    info = fetch_info(url)
    print(f"  {info.title!r} by {info.channel} "
          f"({info.duration:.0f}s, {len(info.transcript)} caption lines)")
    return produce(info, s)


def demo(s: Settings) -> Path:
    """Offline run on a generated test video: no YouTube, no API keys needed."""
    work = s.output_dir / "demo"
    work.mkdir(parents=True, exist_ok=True)
    src = work / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30",
         "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100", "-t", "20",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)],
        check=True,
    )
    info = SourceInfo(
        video_id="demo", title="Demo source video", channel="Test Pattern TV", url="local",
        duration=20.0,
        transcript=[
            Line(0.5, 4.0, "So today I have a big announcement to make."),
            Line(4.0, 7.5, "Honestly, I quit my job and sold the house. No way back!"),
            Line(8.0, 11.5, "My mom thought it was a joke, she was laughing so hard."),
            Line(12.0, 15.0, "Then I told her the price and it got really awkward."),
            Line(15.5, 19.0, "Anyway, thanks for watching, see you next time."),
        ],
    )
    demo_settings = Settings(**{**s.__dict__, "clip_seconds": 20.0, "use_claude": False})
    return produce(info, demo_settings, local_clip=src)
