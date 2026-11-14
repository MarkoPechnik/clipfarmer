"""Reaction voice lines: ElevenLabs text-to-speech, mixed into one track."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
import requests

from .config import SAMPLE_RATE, Settings
from .moments import Reaction

# eleven_v3 understands inline audio tags, which is what makes a gasp sound
# like a gasp instead of the word "gasp". Other models would read the tag
# aloud, so tags are only added for v3 models.
_V3_TAGS = {
    "gasp": "[gasps]",
    "shock": "[shocked]",
    "laugh": "[laughs]",
    "cringe": "[sighs]",
}


def speech_text(reaction: Reaction, model: str) -> str:
    if "v3" in model and reaction.emotion in _V3_TAGS:
        return f"{_V3_TAGS[reaction.emotion]} {reaction.text}"
    return reaction.text


def _elevenlabs(text: str, out: Path, s: Settings) -> None:
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{s.elevenlabs_voice_id}",
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": s.elevenlabs_api_key},
        json={"text": text, "model_id": s.elevenlabs_model},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs TTS failed ({r.status_code}): {r.text[:300]}")
    out.write_bytes(r.content)


def _placeholder(reaction: Reaction) -> np.ndarray:
    """A wobbly blip standing in for a voice when no ElevenLabs key is set."""
    dur = min(0.35 + 0.07 * len(reaction.text.split()), 1.6)
    t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
    base = {"gasp": 520, "shock": 620, "laugh": 440, "cringe": 300}.get(reaction.emotion, 480)
    freq = base * (1 + 0.25 * np.sin(2 * np.pi * 6 * t)) * (1 + 0.4 * t / dur)
    tone = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
    syllables = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t - np.pi / 2)
    fade = np.minimum(1, np.minimum(t, dur - t) / 0.03)
    return (0.35 * tone * syllables * fade).astype(np.float32)


def decode(path: Path) -> np.ndarray:
    """Any audio file -> mono float32 at SAMPLE_RATE."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def synthesize(reactions: list[Reaction], workdir: Path, s: Settings) -> list[np.ndarray]:
    """One mono float32 array per reaction."""
    clips = []
    for i, r in enumerate(reactions):
        if s.elevenlabs_api_key:
            mp3 = workdir / f"voice_{i}.mp3"
            _elevenlabs(speech_text(r, s.elevenlabs_model), mp3, s)
            clips.append(decode(mp3))
        else:
            clips.append(_placeholder(r))
    return clips


def schedule(reactions: list[Reaction], clips: list[np.ndarray], duration: float,
             gap: float = 0.6) -> list[tuple[Reaction, np.ndarray, float]]:
    """Fix timing now that voice lengths are known.

    Pushes a reaction later if it would talk over the previous one and drops
    any that no longer fit before the clip ends. Returns (reaction, audio,
    seconds) triples with `reaction.at` updated.
    """
    out = []
    free_at = 0.0
    for r, audio in sorted(zip(reactions, clips), key=lambda p: p[0].at):
        seconds = len(audio) / SAMPLE_RATE
        at = max(r.at, free_at)
        if at + seconds > duration - 0.2:
            continue
        r.at = round(at, 2)
        out.append((r, audio, seconds))
        free_at = at + seconds + gap
    return out


def mix_track(scheduled: list[tuple[Reaction, np.ndarray, float]], duration: float,
              out: Path) -> np.ndarray:
    """Place every voice line on a silent track as long as the clip; save as WAV."""
    track = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
    for r, audio, _ in scheduled:
        i = int(r.at * SAMPLE_RATE)
        chunk = audio[: len(track) - i]
        track[i:i + len(chunk)] += chunk
    np.clip(track, -1, 1, out=track)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((track * 32767).astype(np.int16).tobytes())
    return track
