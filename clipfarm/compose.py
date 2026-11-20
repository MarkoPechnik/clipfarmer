"""Stack the character panel on top of the clip and mix the audio."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import CLIP_HEIGHT, FPS, HEIGHT, WIDTH

GAME_BAND = 960


def probe(path: Path) -> tuple[float, bool]:
    """(duration in seconds, has audio stream)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    return float(data["format"]["duration"]), has_audio


def _audio_graph(has_audio: bool, voice_input: int) -> str:
    """Duck the source audio under the voice and mix both into [au]."""
    if not has_audio:
        return f";[{voice_input}:a]aformat=channel_layouts=stereo[au]"
    return (
        ";[0:a]aresample=44100,aformat=channel_layouts=stereo[src];"
        f"[{voice_input}:a]aformat=channel_layouts=stereo,asplit[key][vo];"
        "[src][key]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=350[duck];"
        "[duck][vo]amix=inputs=2:normalize=0:duration=first[au]"
    )


def compose(clip: Path, panel: Path, voice: Path, duration: float, has_audio: bool,
            out: Path) -> Path:
    # The clip is fitted inside its area over a blurred, zoomed copy of itself,
    # so both landscape videos and vertical Shorts fill the frame.
    video = (
        f"[0:v]fps={FPS},split[a][b];"
        f"[a]scale={WIDTH}:{CLIP_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{CLIP_HEIGHT},boxblur=22:2,eq=brightness=-0.12[bg];"
        f"[b]scale={WIDTH}:{CLIP_HEIGHT}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[bot];"
        f"[1:v][bot]vstack=inputs=2,format=yuv420p[v]"
    )
    audio = _audio_graph(has_audio, voice_input=2)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-i", str(panel), "-i", str(voice),
         "-filter_complex", video + audio, "-map", "[v]", "-map", "[au]",
         "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)],
        check=True,
    )
    return out


def compose_fullscreen(clip: Path, voice: Path, captions: Path, duration: float,
                       has_audio: bool, out: Path) -> Path:
    """Zoomed gameplay across the middle of a 9:16 frame over a blurred copy,
    with the title and captions burned in from an .ass subtitle file."""
    video = (
        f"[0:v]fps={FPS},split[a][b];"
        f"[a]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},boxblur=24:2,eq=brightness=-0.18[bg];"
        # Zoom into the centre of the game: a 1080x960 band fills half the
        # screen, leaving room for the hook above and captions below.
        f"[b]scale={WIDTH}:{GAME_BAND}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{GAME_BAND}[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,ass={captions.name},format=yuv420p[v]"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(clip.resolve()), "-i", str(voice.resolve()),
         "-filter_complex", video + _audio_graph(has_audio, voice_input=1),
         "-map", "[v]", "-map", "[au]", "-t", f"{duration:.3f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out.resolve())],
        check=True, cwd=captions.parent,  # the ass filter reads the file by name
    )
    return out
