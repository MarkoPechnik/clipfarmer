# clipfarm

Finds YouTube videos that were posted in the last few minutes and are gaining
views fastest, cuts the best ~30 seconds, and turns it into a vertical
(1080x1920) reaction clip: an animated cartoon character sits above the video
and reacts out loud, voiced by ElevenLabs.

```
discover ──> plan ──> download ──> voice ──> animate ──> compose
YouTube API  Claude   yt-dlp       ElevenLabs Pillow     ffmpeg
```

## Setup

Needs Python 3.10+ and `ffmpeg` on your PATH.

```bash
pip install -e .
cp .env.example .env    # then fill in the keys
clipfarm demo           # offline test, no keys needed -> output/demo/final.mp4
```

| Key | Used for | Without it |
| --- | --- | --- |
| `YOUTUBE_API_KEY` | finding new, fast-rising videos | `discover`, `run`, `watch` do not work; `make <url>` still does |
| `ELEVENLABS_API_KEY` | the reaction voice | a placeholder blip is used instead of speech |
| `ANTHROPIC_API_KEY` | Claude reads the transcript, picks the moment and writes reactions to what was said | keyword heuristic and canned lines ("Oh my god!", "Wait, WHAT?!") |

## Usage

```bash
clipfarm discover --window 60          # fastest risers posted in the last hour
clipfarm discover --window 10 --query "podcast"
clipfarm make https://www.youtube.com/watch?v=VIDEO_ID
clipfarm run --window 60 --count 3     # discover + make the top 3 new ones
clipfarm watch --interval 15           # keep scanning every 15 minutes
```

Each clip lands in `output/<video id>/`:

- `final.mp4` — the finished short
- `meta.json` — suggested title, source link and credit line, reaction timings

`output/seen.json` remembers which videos were already processed.

## How it works

**Discovery** (`discover.py`). One `search.list` call for videos published
inside the window, one `videos.list` call for statistics. Videos are ranked by
engagement per minute since upload: `(views + 20·likes + 50·comments) / age`.
Each scan costs about 101 quota units out of YouTube's default 10,000 per day,
so scanning every 15 minutes is fine and scanning every minute is not.
YouTube's search index also lags by a few minutes, so a `--window` under about
10 usually returns very little.

**Planning** (`moments.py`). With Anthropic credentials, Claude gets the
timestamped captions and returns the clip range plus 2–4 reactions, each tied
to the line it reacts to. Otherwise the heuristic uses YouTube's
"most replayed" data when it exists, or the stretch of captions with the most
charged words, and picks canned lines by emotion.

**Voice** (`voice.py`). Each reaction goes to ElevenLabs text-to-speech. With
the default `eleven_v3` model the line is prefixed with an audio tag
(`[gasps]`, `[laughs]`, ...) so it is performed rather than read. Reactions are
rescheduled so they never talk over each other, and the source audio ducks
under the voice.

**Character** (`character.py`). An original blob drawn with Pillow, with four
reaction poses (gasp, shock, laugh, cringe) plus idle breathing and blinking.
The mouth follows the loudness of the voice track, so it moves in time with
the speech. Rename it with `CLIPFARM_CHARACTER_NAME`.

## Before you post

Clips made from other people's videos are their content. Short commentary
clips are common, but they can still draw copyright claims or strikes, and
downloading from YouTube is against its terms of service. To stay on the safe
side: use `--creative-commons` to only pick CC-licensed videos, clip channels
that allow it, and keep the credit (it is drawn on the clip and written to
`meta.json`).

## Tests

```bash
pip install -e ".[dev]"
pytest
```
