"""Settings, read from the environment (and a local .env file if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Output canvas: a 9:16 short. The character panel sits on top of the clip.
WIDTH = 1080
HEIGHT = 1920
PANEL_HEIGHT = 720
CLIP_HEIGHT = HEIGHT - PANEL_HEIGHT
FPS = 30
SAMPLE_RATE = 44100


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines, existing env vars win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


@dataclass
class Settings:
    youtube_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model: str = "eleven_v3"
    use_claude: bool = False
    output_dir: Path = Path("output")
    clip_seconds: float = 30.0
    character_name: str = "Gaspy"
    # Browser to borrow YouTube sign-in cookies from (chrome, safari, firefox,
    # edge, brave). Needed when YouTube says "Sign in to confirm you're not a bot".
    cookies_browser: str = ""
    footage_dir: Path = Path("footage")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        env = os.environ.get
        return cls(
            youtube_api_key=env("YOUTUBE_API_KEY", ""),
            elevenlabs_api_key=env("ELEVENLABS_API_KEY", ""),
            elevenlabs_voice_id=env("ELEVENLABS_VOICE_ID", cls.elevenlabs_voice_id),
            elevenlabs_model=env("ELEVENLABS_MODEL", cls.elevenlabs_model),
            use_claude=bool(env("ANTHROPIC_API_KEY") or env("ANTHROPIC_AUTH_TOKEN")),
            output_dir=Path(env("CLIPFARM_OUTPUT", "output")),
            clip_seconds=float(env("CLIPFARM_CLIP_SECONDS", "30")),
            character_name=env("CLIPFARM_CHARACTER_NAME", cls.character_name),
            cookies_browser=env("CLIPFARM_COOKIES_BROWSER", "").strip().lower(),
            footage_dir=Path(env("CLIPFARM_FOOTAGE", "footage")),
        )
