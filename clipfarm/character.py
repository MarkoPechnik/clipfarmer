"""The reaction character: an original cartoon blob drawn frame by frame.

Each frame is drawn at 2x and reduced for smooth edges. The mouth follows the
loudness of the voice track, so it lip-flaps in time with the ElevenLabs audio.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FPS, PANEL_HEIGHT, SAMPLE_RATE, WIDTH
from .moments import Reaction

S = 2  # supersampling factor
BODY = (124, 92, 255)
BODY_DARK = (92, 64, 214)
INK = (28, 22, 48)
WHITE = (255, 255, 255)
BG = {
    None: ((30, 27, 58), (58, 44, 110)),
    "gasp": ((24, 52, 96), (40, 110, 170)),
    "shock": ((92, 22, 40), (176, 48, 60)),
    "laugh": ((96, 66, 10), (214, 150, 30)),
    "cringe": ((28, 62, 52), (60, 120, 96)),
}
_FONTS = [
    "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _font(size: int) -> ImageFont.ImageFont:
    for path in _FONTS:
        try:
            return ImageFont.truetype(path, size * S)
        except OSError:
            continue
    return ImageFont.load_default(size * S)


def _gradient(top, bottom) -> Image.Image:
    ramp = np.linspace(0, 1, PANEL_HEIGHT * S)[:, None, None]
    col = np.array(top) * (1 - ramp) + np.array(bottom) * ramp
    return Image.fromarray(np.broadcast_to(col, (PANEL_HEIGHT * S, WIDTH * S, 3)).astype(np.uint8))


def mouth_envelope(track: np.ndarray, frames: int) -> np.ndarray:
    """Per-frame mouth openness in [0, 1] from the voice track's loudness."""
    hop = SAMPLE_RATE // FPS
    padded = np.zeros(frames * hop, dtype=np.float32)
    n = min(len(track), len(padded))
    padded[:n] = track[:n]
    rms = np.sqrt((padded.reshape(frames, hop) ** 2).mean(axis=1))
    peak = np.percentile(rms[rms > 1e-4], 90) if (rms > 1e-4).any() else 1.0
    env = np.clip(rms / peak, 0, 1)
    return np.convolve(env, [0.25, 0.5, 0.25], mode="same")  # soften jitter


class Character:
    def __init__(self, name: str, credit: str = ""):
        self.name = name
        self.credit = credit
        self.backgrounds = {k: _gradient(*v) for k, v in BG.items()}
        self.caption_font = _font(50)
        self.small_font = _font(26)

    # -- helpers; all coordinates are in output pixels, scaled by S here
    @staticmethod
    def _ellipse(d, cx, cy, rx, ry, **kw):
        d.ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S], **kw)

    @staticmethod
    def _line(d, pts, width, fill=INK):
        d.line([(x * S, y * S) for x, y in pts], fill=fill, width=int(width * S), joint="curve")

    def frame(self, t: float, emotion: str | None, k: float, mouth: float, text: str) -> Image.Image:
        """k: how far into the reaction pose we are (0 idle .. 1 full)."""
        img = self.backgrounds[None].copy()
        if emotion and k > 0:
            img = Image.blend(img, self.backgrounds[emotion], k)
        d = ImageDraw.Draw(img)
        e = emotion if k > 0 else None

        cx = WIDTH / 2
        cy = 330 + 7 * math.sin(2 * math.pi * 0.45 * t)  # idle breathing
        if e == "shock":
            cx += k * 9 * math.sin(2 * math.pi * 13 * t)
        elif e == "laugh":
            cy -= k * 20 * abs(math.sin(2 * math.pi * 4.5 * t))
        elif e == "gasp":
            cy -= k * 14
        elif e == "cringe":
            cy += k * 10

        # body
        self._ellipse(d, WIDTH / 2, 566, 210, 22, fill=(0, 0, 0))
        squash = 1 + (0.05 * k if e == "gasp" else -0.04 * k if e == "cringe" else 0)
        self._ellipse(d, cx, cy + 20, 225 / squash, 225 * squash, fill=BODY_DARK)
        self._ellipse(d, cx - 6, cy + 8, 212 / squash, 212 * squash, fill=BODY)
        self._ellipse(d, cx - 95, cy - 120, 46, 26, fill=(160, 134, 255))  # highlight

        # eyes
        blink = (t % 3.7) < 0.12 and not e
        for side in (-1, 1):
            ex, ey = cx + side * 84, cy - 40
            if e == "laugh" and k > 0.5:
                self._line(d, [(ex - 40, ey + 12), (ex, ey - 22), (ex + 40, ey + 12)], 11)
            elif blink:
                self._line(d, [(ex - 40, ey), (ex + 40, ey)], 10)
            else:
                wide = {"gasp": 0.3, "shock": 0.42, "cringe": -0.5}.get(e, 0) * k
                rx, ry = 46 * (1 + wide * 0.4), 56 * (1 + wide)
                self._ellipse(d, ex, ey, rx, ry, fill=WHITE, outline=INK, width=5 * S)
                pupil = 22 * (1 - 0.5 * k) if e in ("gasp", "shock") else 22
                look = 5 * math.sin(2 * math.pi * 0.2 * t) if not e else 0
                self._ellipse(d, ex + look, ey + (6 if e == "cringe" else 0), pupil, min(pupil, ry - 6), fill=INK)
                self._ellipse(d, ex + look - 7, ey - 8, pupil * 0.3, pupil * 0.3, fill=WHITE)
            # brows
            by = ey - 78 - (34 * k if e in ("gasp", "shock") else 0)
            tilt = {"cringe": 20, "laugh": -8}.get(e, 0) * k
            self._line(d, [(ex - side * 44, by - tilt * 0.2), (ex + side * 44, by + tilt * -1 + (0 if e else 4))], 12)

        # mouth
        mx, my = cx, cy + 92
        if e in ("gasp", "shock"):
            rx = (34 if e == "gasp" else 56) * (0.6 + 0.4 * k)
            ry = (30 + 46 * mouth if e == "gasp" else 46 + 60 * mouth) * k + 8
            self._ellipse(d, mx, my + ry * 0.4, rx, ry, fill=INK)
            if ry > 40:
                self._ellipse(d, mx, my + ry * 1.05, rx * 0.62, ry * 0.3, fill=(255, 110, 130))
        elif e == "laugh":
            h = (50 + 60 * mouth) * k + 10
            d.pieslice([(mx - 92) * S, (my - h) * S, (mx + 92) * S, (my + h) * S], 0, 180, fill=INK)
            d.pieslice([(mx - 50) * S, (my + h * 0.35) * S, (mx + 50) * S, (my + h * 1.25) * S], 180, 360, fill=(255, 110, 130))
        elif e == "cringe":
            amp = 10 + 10 * mouth
            pts = [(mx - 80 + i * 20, my + 14 + (amp if i % 2 else -amp) * k) for i in range(9)]
            self._line(d, pts, 10)
            self._ellipse(d, cx + 186, cy - 96 + 30 * ((t * 1.4) % 1), 13, 20, fill=(140, 215, 255))
        else:
            d.arc([(mx - 60) * S, (my - 50) * S, (mx + 60) * S, (my + 26) * S], 20, 160, fill=INK, width=10 * S)

        if e == "shock":  # hands on cheeks
            for side in (-1, 1):
                self._ellipse(d, cx + side * (250 - 60 * k), cy + 70, 46, 54, fill=BODY, outline=BODY_DARK, width=6 * S)

        # labels
        d.text((30 * S, 26 * S), self.name.upper(), font=self.small_font, fill=(255, 255, 255))
        if self.credit:
            w = d.textlength(self.credit, font=self.small_font)
            d.text((WIDTH * S - w - 30 * S, 26 * S), self.credit, font=self.small_font, fill=(215, 210, 240))
        if text and k > 0.3:
            self._caption(d, text)
        return img.reduce(S)

    def _caption(self, d: ImageDraw.ImageDraw, text: str) -> None:
        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if d.textlength(trial, font=self.caption_font) > (WIDTH - 120) * S and cur:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        lines.append(cur)
        lines = lines[-2:]
        y = (PANEL_HEIGHT - 34 - 62 * len(lines)) * S
        for line in lines:
            w = d.textlength(line, font=self.caption_font)
            d.text(((WIDTH * S - w) / 2, y), line, font=self.caption_font, fill=WHITE,
                   stroke_width=7 * S, stroke_fill=INK)
            y += 62 * S


def render_panel(character: Character, scheduled: list[tuple[Reaction, np.ndarray, float]],
                 track: np.ndarray, duration: float, out: Path) -> Path:
    """Render the character panel for the whole clip to an mp4."""
    frames = int(round(duration * FPS))
    env = mouth_envelope(track, frames)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{WIDTH}x{PANEL_HEIGHT}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE,
    )
    try:
        for i in range(frames):
            t = i / FPS
            emotion, k, text = None, 0.0, ""
            for r, _, seconds in scheduled:
                # ease in just before the voice starts, hold briefly after it ends
                t0, t1 = r.at - 0.15, r.at + seconds + 0.45
                if t0 <= t < t1:
                    k = min(1.0, (t - t0) / 0.15, (t1 - t) / 0.3)
                    emotion, text = r.emotion, r.text
                    break
            proc.stdin.write(character.frame(t, emotion, max(k, 0.0), float(env[i]), text).tobytes())
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding the character panel")
    return out
