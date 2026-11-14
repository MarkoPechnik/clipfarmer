"""Decide which part of the video to clip and how the character reacts.

Two planners:
  * Claude reads the timestamped transcript and writes reactions to what was
    actually said (used when Anthropic credentials are available).
  * A heuristic picks the busiest stretch (or the most-replayed one) and uses
    canned lines. Also the fallback if the Claude call fails.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .source import Line, SourceInfo

EMOTIONS = ("gasp", "shock", "laugh", "cringe")
MODEL = "claude-opus-5-5"


@dataclass
class Reaction:
    at: float  # seconds from the start of the clip
    text: str
    emotion: str


@dataclass
class Plan:
    start: float  # seconds from the start of the source video
    end: float
    title: str
    reactions: list[Reaction] = field(default_factory=list)
    planner: str = "heuristic"


# ---------------------------------------------------------------- heuristic

_HOT = {
    "shock": r"\b(no way|insane|crazy|what|unbelievable|shocking|never|worst|fired|exposed|lied|secret|million|billion|dead|wild)\b",
    "laugh": r"\b(funny|hilarious|lol|laugh(ing|ed)?|joke|joking|bro|stupid|dumb)\b",
    "cringe": r"\b(awkward|cringe|embarrass\w*|yikes|sorry|apolog\w*|mistake|oops|fail(ed)?)\b",
    "gasp": r"\b(oh my|omg|wow|whoa|wait|actually|honestly|literally|announce\w*|reveal\w*|breaking)\b",
}
_HOT = {k: re.compile(v, re.I) for k, v in _HOT.items()}

CANNED = {
    "gasp": ["Oh my god!", "Wait, hold on!", "Oh! He did not!"],
    "shock": ["No way. I can't believe he said that!", "Wait, WHAT?!", "Excuse me?!"],
    "laugh": ["Ha ha ha, stop!", "I'm crying, that's so good!", "Ha! No he didn't!"],
    "cringe": ["Ooh... that's rough.", "Yikes. Oh no.", "Oof, I felt that."],
}


def _heat(line: Line) -> tuple[float, str]:
    """(score, best emotion) for a caption line."""
    best, emotion = 0, "gasp"
    total = 0
    for emo, rx in _HOT.items():
        n = len(rx.findall(line.text))
        total += n
        if n > best:
            best, emotion = n, emo
    score = len(line.text.split()) * 0.2 + 5 * total + 3 * line.text.count("!") + 2 * line.text.count("?")
    return score, emotion


def _window_from_heatmap(heatmap: list[dict], length: float, duration: float) -> float | None:
    if not heatmap:
        return None
    peak = max(heatmap, key=lambda h: h.get("value", 0))
    # Start a little before the most-replayed point so the setup is included.
    return max(0.0, min(peak["start_time"] - length * 0.3, max(duration - length, 0.0)))


def heuristic_plan(info: SourceInfo, clip_seconds: float) -> Plan:
    duration = info.duration or (info.transcript[-1].end if info.transcript else clip_seconds)
    length = min(clip_seconds, duration)
    last_start = max(duration - length, 0.0)

    start = _window_from_heatmap(info.heatmap, length, duration)
    if start is None and info.transcript:
        best = -1.0
        t = 0.0
        while t <= last_start + 1e-6:
            # Lines too close to either edge cannot be reacted to, so they
            # count for less: this keeps the hot line inside the clip.
            score = sum(
                _heat(l)[0] * (1.0 if t + 2 <= l.end <= t + length - 3 else 0.3)
                for l in info.transcript if t <= l.start < t + length
            )
            if score > best:
                best, start = score, t
            t += 5.0
    if start is None:
        start = 0.0
    end = start + length

    inside = [l for l in info.transcript if start + 2 <= l.end <= end - 3]
    picks: list[tuple[float, str]] = []  # (clip-relative time, emotion)
    for line in sorted(inside, key=lambda l: _heat(l)[0], reverse=True):
        at = line.end - start
        if all(abs(at - p[0]) >= 7 for p in picks):
            picks.append((at, _heat(line)[1]))
        if len(picks) == 3:
            break
    if not picks:  # no captions: space reactions evenly
        n = max(1, min(3, int(length // 10)))
        picks = [(length * (i + 1) / (n + 1), EMOTIONS[i % 3]) for i in range(n)]
    picks.sort()

    used: dict[str, int] = {}
    reactions = []
    for at, emo in picks:
        i = used.get(emo, 0)
        used[emo] = i + 1
        reactions.append(Reaction(round(at, 2), CANNED[emo][i % len(CANNED[emo])], emo))
    return Plan(start, end, info.title, reactions, "heuristic")


# ------------------------------------------------------------------- Claude

_SCHEMA = {
    "type": "object",
    "properties": {
        "start": {"type": "number", "description": "Clip start, seconds into the video"},
        "end": {"type": "number", "description": "Clip end, seconds into the video"},
        "title": {"type": "string", "description": "Short, punchy title for the clip"},
        "reactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "at": {"type": "number", "description": "Seconds into the video, right after the line being reacted to"},
                    "text": {"type": "string", "description": "What the character says, at most 10 words"},
                    "emotion": {"type": "string", "enum": list(EMOTIONS)},
                },
                "required": ["at", "text", "emotion"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["start", "end", "title", "reactions"],
    "additionalProperties": False,
}

_SYSTEM = """You plan short vertical reaction clips. A cartoon character sits above the \
video and reacts out loud to what the people in it say.

You get a video's title and its timestamped transcript. Pick the single most \
clip-worthy stretch: the moment a viewer would send to a friend. It should \
make sense on its own, so start where the setup begins, not mid-sentence.

Then write the character's reactions. They are quick spoken interjections \
(gasps, disbelief, laughing, wincing) that respond to the specific thing just \
said, the way a friend watching next to you would. Place each one right after \
the line it reacts to, leave at least 6 seconds between reactions, and keep \
the first 2 and last 3 seconds of the clip free. Two to four reactions is \
right for a 30 second clip. The reactions are voiced by text-to-speech, so \
write only words that should be spoken."""


def claude_plan(info: SourceInfo, clip_seconds: float) -> Plan:
    import anthropic

    transcript = "\n".join(f"[{l.start:7.1f}] {l.text}" for l in info.transcript)
    prompt = (
        f"Title: {info.title}\nChannel: {info.channel}\n"
        f"Video length: {info.duration:.0f} seconds\n"
        f"Target clip length: about {clip_seconds:.0f} seconds\n\n"
        f"Transcript ([seconds] text):\n{transcript}"
    )
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"Claude stopped with {response.stop_reason}")
    data = json.loads(next(b.text for b in response.content if b.type == "text"))

    duration = info.duration or data["end"]
    start = max(0.0, min(float(data["start"]), duration))
    end = max(start + 5.0, min(float(data["end"]), duration, start + clip_seconds * 1.5))
    reactions = [
        Reaction(round(r["at"] - start, 2), r["text"].strip(), r["emotion"])
        for r in sorted(data["reactions"], key=lambda r: r["at"])
        if 0.5 <= r["at"] - start <= end - start - 1.5 and r["text"].strip()
    ]
    return Plan(start, end, data["title"].strip() or info.title, reactions, "claude")


def make_plan(info: SourceInfo, clip_seconds: float, use_claude: bool) -> Plan:
    if use_claude and info.transcript:
        try:
            plan = claude_plan(info, clip_seconds)
            if plan.reactions:
                return plan
            print("  ! Claude returned no usable reactions; using heuristic plan")
        except Exception as e:
            print(f"  ! Claude planning failed ({e}); using heuristic plan")
    return heuristic_plan(info, clip_seconds)
