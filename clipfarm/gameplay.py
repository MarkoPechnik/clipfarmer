"""Your own gameplay -> commentated vertical shorts.

1. Find highlights: loud bursts and fast on-screen action in a recording.
2. Write commentary: Claude looks at frames from each highlight and writes
   lines in the chosen style (falls back to simple templates without a key).
3. Voice it with ElevenLabs, add captions, and render a 1080x1920 short.

Only use footage you recorded yourself: that is what makes the result your own
content for YouTube monetization.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import voice
from .character import Character, render_panel
from .compose import compose, compose_fullscreen, probe
from .config import Settings
from .moments import Reaction

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
STYLES = ("hype", "funny", "story")
LAYOUTS = ("fullscreen", "character")
LINE_EMOTIONS = ("talk", "hype", "laugh", "shock", "gasp", "cringe")
HOP = 0.5  # seconds per analysis bin
MODEL = "claude-opus-5-5"


@dataclass
class Highlight:
    start: float
    end: float
    peak: float  # seconds from the start of the recording
    score: float


@dataclass
class Script:
    hook: str  # on-screen title, a few words
    title: str  # video title for the upload
    lines: list[Reaction] = field(default_factory=list)
    writer: str = "template"


# ------------------------------------------------------------- highlights

def _audio_loudness(path: Path, bins: int) -> np.ndarray:
    """Loudness in dB per HOP-second bin (zeros if the file has no audio)."""
    rate = 4000
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(rate),
         "-f", "f32le", "-"],
        capture_output=True,
    )
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    hop = int(rate * HOP)
    n = min(bins, len(samples) // hop)
    out = np.full(bins, -80.0, dtype=np.float32)
    if n:
        rms = np.sqrt((samples[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
        out[:n] = 20 * np.log10(np.maximum(rms, 1e-4))
    return out


def _motion(path: Path, bins: int) -> np.ndarray:
    """How much the picture changes per HOP-second bin (tiny grayscale frames)."""
    w, h = 64, 36
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-an",
         "-vf", f"fps={1 / HOP},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
        capture_output=True,
    )
    frames = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = len(frames) // (w * h)
    out = np.zeros(bins, dtype=np.float32)
    if n > 1:
        f = frames[: n * w * h].reshape(n, h, w).astype(np.float32)
        diff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
        m = min(bins - 1, len(diff))
        out[1:m + 1] = diff[:m]
    return out


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-6 else np.zeros_like(x)


def excitement(loud_db: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Per-bin excitement: loudness, sudden jumps in loudness, and motion."""
    jump = np.maximum(np.diff(loud_db, prepend=loud_db[:1]), 0)
    raw = _z(loud_db) + 1.2 * _z(jump) + 0.8 * _z(motion)
    kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    return np.convolve(raw, kernel / kernel.sum(), mode="same")


def pick_highlights(score: np.ndarray, length: float, count: int,
                    duration: float) -> list[Highlight]:
    """Best non-overlapping windows of `length` seconds.

    Each window is placed so its peak sits about two thirds of the way in:
    enough build-up to follow, and a beat after the payoff for a reaction.
    """
    win = max(1, int(round(length / HOP)))
    if duration <= length or len(score) <= win:
        peak = float(np.argmax(score)) * HOP if len(score) else 0.0
        return [Highlight(0.0, duration, peak, float(score.max()) if len(score) else 0.0)]
    s = score.copy()
    picks: list[Highlight] = []
    for _ in range(count):
        i = int(np.argmax(s))
        if s[i] == -np.inf:
            break
        start_bin = int(np.clip(i - round(win * 0.65), 0, len(score) - win))
        segment = score[start_bin:start_bin + win]
        picks.append(Highlight(
            start=round(start_bin * HOP, 2),
            end=round(min(start_bin * HOP + length, duration), 2),
            peak=round(i * HOP, 2),
            score=round(float(segment.mean() + segment.max()), 3),
        ))
        s[max(0, start_bin - win // 4): start_bin + win + win // 4] = -np.inf
    return sorted(picks, key=lambda h: h.score, reverse=True)


def find_highlights(path: Path, length: float, count: int) -> list[Highlight]:
    duration, _ = probe(path)
    bins = max(1, int(duration / HOP))
    score = excitement(_audio_loudness(path, bins), _motion(path, bins))
    return pick_highlights(score, length, count, duration)


# ------------------------------------------------------------- commentary

_STYLE_GUIDE = {
    "hype": "High-energy play-by-play, like a sports caster calling the action as it happens.",
    "funny": "Dry, sarcastic jokes about what the player is doing, like a friend roasting them.",
    "story": ("Narrate it as a dramatic story about the player's character, a little "
              "over the top, like a storytime video. Keep talking through the clip."),
}

_SYSTEM = """You write the voice-over for short vertical gameplay videos (YouTube \
Shorts, TikTok). The footage is the creator's own Grand Theft Auto gameplay and \
your commentary is what makes the video original, so it has to be about what \
actually happens on screen in these frames, not generic filler.

Rules:
- Start with a hook in the first second that makes people stay.
- Time each line to the moment it describes. Lines may not overlap: allow about \
2.5 words per second of speaking time, plus a short pause.
- Keep each line under 14 words. The lines are read by text-to-speech, so write \
only words to be spoken: no emojis, hashtags, or stage directions.
- Only describe what the frames show. Do not invent names for real people.
- The final line should land the payoff or a punchline before the clip ends."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "hook": {"type": "string", "description": "On-screen title, at most 6 words"},
        "title": {"type": "string", "description": "Upload title, at most 70 characters"},
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "at": {"type": "number", "description": "Seconds from the start of the clip"},
                    "text": {"type": "string"},
                    "emotion": {"type": "string", "enum": list(LINE_EMOTIONS)},
                },
                "required": ["at", "text", "emotion"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["hook", "title", "lines"],
    "additionalProperties": False,
}


def _frames(path: Path, h: Highlight, n: int = 8) -> list[tuple[float, bytes]]:
    """JPEG stills spread through the highlight, plus one at the peak."""
    length = h.end - h.start
    times = sorted({round(length * (i + 0.5) / n, 1) for i in range(n)} | {round(h.peak - h.start, 1)})
    out = []
    for t in times:
        jpg = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{h.start + t:.2f}", "-i", str(path),
             "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "5", "-f", "image2pipe",
             "-vcodec", "mjpeg", "-"],
            capture_output=True,
        ).stdout
        if jpg:
            out.append((t, jpg))
    return out


def claude_script(path: Path, h: Highlight, style: str) -> Script:
    import anthropic

    length = h.end - h.start
    content: list[dict] = []
    for t, jpg in _frames(path, h):
        content.append({"type": "text", "text": f"Frame at {t:.1f}s:"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(jpg).decode()}})
    content.append({"type": "text", "text": (
        f"The clip is {length:.0f} seconds long. The loudest, most intense moment is "
        f"at about {h.peak - h.start:.1f}s.\nStyle: {_STYLE_GUIDE[style]}\n"
        "Write the hook, an upload title, and the timed voice-over lines.")})

    response = anthropic.Anthropic().messages.create(
        model=MODEL,
        max_tokens=16000,
        system=_SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"Claude stopped with {response.stop_reason}")
    data = json.loads(next(b.text for b in response.content if b.type == "text"))
    lines = [
        Reaction(round(float(l["at"]), 2), l["text"].strip(), l["emotion"])
        for l in sorted(data["lines"], key=lambda l: l["at"])
        if 0 <= l["at"] < length - 1 and l["text"].strip()
    ]
    return Script(data["hook"].strip(), data["title"].strip(), lines, "claude")


_TEMPLATES = {
    "hype": (["Okay, watch this. This is going to be good.", "Here we go, here we go!",
              "NO WAY! Did you see that?!", "Absolutely unreal. Run it back."],
             "Watch till the end"),
    "funny": (["So, this was supposed to be a simple plan.", "Totally under control. Definitely.",
               "And there it is. Of course.", "Ten out of ten. No notes."],
              "It went perfectly. Mostly"),
    "story": (["It was supposed to be a normal day in Los Santos.",
               "But something felt off, and he knew it.",
               "Then everything went wrong at once.",
               "And that is how the legend was born."],
              "He never saw it coming"),
}


def template_script(h: Highlight, style: str) -> Script:
    """Generic lines placed around the peak. Add an Anthropic key for real commentary."""
    length = h.end - h.start
    peak = min(max(h.peak - h.start, length * 0.4), length - 4)
    lines, hook = _TEMPLATES[style]
    times = [0.3, max(2.5, peak - 6), peak, min(peak + 3.5, length - 3)]
    emotions = {"hype": ["hype", "talk", "shock", "hype"],
                "funny": ["talk", "talk", "laugh", "talk"],
                "story": ["talk", "talk", "gasp", "talk"]}[style]
    return Script(hook, hook, [Reaction(round(t, 2), text, emo)
                               for t, text, emo in zip(times, lines, emotions)], "template")


def make_script(path: Path, h: Highlight, style: str, use_claude: bool) -> Script:
    if use_claude:
        try:
            script = claude_script(path, h, style)
            if script.lines:
                return script
            print("  ! Claude wrote no usable lines; using template commentary")
        except Exception as e:
            print(f"  ! Claude commentary failed ({e}); using template commentary")
    return template_script(h, style)


# --------------------------------------------------------------- captions

def _ass_text(text: str) -> str:
    return re.sub(r"[{}\\]", "", text).replace("\n", " ")


def _ass_time(t: float) -> str:
    t = max(t, 0.0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def write_captions(hook: str, scheduled: list[tuple[Reaction, np.ndarray, float]],
                   duration: float, out: Path) -> Path:
    """Shorts-style captions: the hook up top, spoken words in chunks of 3."""
    events = [f"Dialogue: 0,{_ass_time(0)},{_ass_time(duration)},Hook,,0,0,0,,"
              f"{_ass_text(hook).upper()}"]
    for r, _, seconds in scheduled:
        words = r.text.split()
        chunks = [words[i:i + 3] for i in range(0, len(words), 3)] or [[""]]
        weights = [sum(len(w) + 1 for w in c) for c in chunks]
        t = r.at
        for chunk, weight in zip(chunks, weights):
            span = seconds * weight / sum(weights)
            events.append(f"Dialogue: 1,{_ass_time(t)},{_ass_time(t + span)},Caption,,0,0,0,,"
                          f"{_ass_text(' '.join(chunk)).upper()}")
            t += span
    out.write_text("\n".join([
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920",
        "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Hook,Arial Black,78,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,"
        "-1,0,0,0,100,100,0,0,1,7,3,8,70,70,300,1",
        "Style: Caption,Arial Black,92,&H0037E0FF,&H00FFFFFF,&H00000000,&H64000000,"
        "-1,0,0,0,100,100,0,0,1,8,3,2,60,60,290,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *events, "",
    ]))
    return out


# ----------------------------------------------------------------- render

def _cut(path: Path, h: Highlight, out: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{h.start:.2f}", "-i", str(path),
         "-t", f"{h.end - h.start:.2f}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "16", "-c:a", "aac", "-b:a", "192k", str(out)],
        check=True,
    )
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:40] or "clip"


def make_short(path: Path, h: Highlight, s: Settings, style: str, layout: str) -> Path:
    clip_id = f"gta-{_slug(path.stem)}-{int(h.start)}-{style}-{layout}"
    work = s.output_dir / clip_id
    work.mkdir(parents=True, exist_ok=True)
    print(f"  highlight {h.start:.0f}s-{h.end:.0f}s (peak {h.peak:.0f}s)")

    clip = _cut(path, h, work / "source.mp4")
    duration, has_audio = probe(clip)
    script = make_script(path, h, style, s.use_claude)
    print(f"  commentary ({script.writer}): {script.hook!r}")

    audio = voice.synthesize(script.lines, work, s)
    scheduled = voice.schedule(script.lines, audio, duration, gap=0.3)
    track = voice.mix_track(scheduled, duration, work / "voice.wav")
    for r, _, seconds in scheduled:
        print(f"    {r.at:5.1f}s  {r.text!r}")

    if layout == "character":
        panel = render_panel(Character(s.character_name, script.hook), scheduled, track,
                             duration, work / "panel.mp4")
        final = compose(clip, panel, work / "voice.wav", duration, has_audio, work / "final.mp4")
    else:
        captions = write_captions(script.hook, scheduled, duration, work / "captions.ass")
        final = compose_fullscreen(clip, work / "voice.wav", captions, duration, has_audio,
                                   work / "final.mp4")

    (work / "meta.json").write_text(json.dumps({
        "title": script.title or script.hook,
        "source_title": path.name,
        "source_channel": "Your gameplay",
        "source_start": h.start,
        "source_end": h.end,
        "planner": script.writer,
        "style": style,
        "layout": layout,
        "voice": "elevenlabs" if s.elevenlabs_api_key else "placeholder",
        "reactions": [asdict(r) for r, _, _ in scheduled],
        "description": f"{script.title or script.hook}\n\n#gta #gta5 #gaming #shorts",
    }, indent=2))
    return final


def make_shorts(path: Path, s: Settings, count: int = 3, style: str = "hype",
                layout: str = "fullscreen") -> list[Path]:
    if style not in STYLES or layout not in LAYOUTS:
        raise RuntimeError(f"style must be one of {STYLES}, layout one of {LAYOUTS}")
    print(f"> {path.name}: finding the best {count} moments")
    highlights = find_highlights(path, s.clip_seconds, count)
    return [make_short(path, h, s, style, layout) for h in highlights]


def make_test_footage(out: Path, seconds: int = 75) -> Path:
    """Fake 'gameplay' with a quiet stretch and a loud, busy burst at 50s."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=s=1280x720:r=30:d={seconds}",
         "-f", "lavfi", "-i", f"anoisesrc=d={seconds}:c=pink:a=0.02:r=44100",
         "-filter_complex",
         "[0:v]hue=s='if(between(t,48,54),3,1)':H='if(between(t,48,54),t*6,0)',"
         "eq=brightness='if(between(t,48,54),0.25*sin(t*40),0)'[v];"
         "[1:a]volume='if(between(t,48,54),25,1)':eval=frame[a]",
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)],
        check=True,
    )
    return out
