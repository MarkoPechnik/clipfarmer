"""Find videos that were just posted and are gaining views fastest.

Uses the YouTube Data API v3: one search.list call (100 quota units) for videos
published inside the window, then one videos.list call (1 unit) for statistics.
The default quota is 10,000 units/day, so roughly 99 scans per day.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import requests

API = "https://www.googleapis.com/youtube/v3"
_DURATION = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$")


@dataclass
class Candidate:
    video_id: str
    title: str
    channel: str
    published_at: str
    age_minutes: float
    duration_seconds: int
    views: int
    likes: int
    comments: int
    score: float

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    def to_dict(self) -> dict:
        return {**asdict(self), "url": self.url}


def parse_duration(iso: str) -> int:
    """'PT1H2M3S' -> 3723. Unknown formats (e.g. live 'P0D') -> 0."""
    m = _DURATION.match(iso or "")
    if not m:
        return 0
    h, mnt, s = (int(g or 0) for g in m.groups())
    return h * 3600 + mnt * 60 + s


def velocity_score(views: int, likes: int, comments: int, age_minutes: float) -> float:
    """Engagement per minute since upload.

    Likes and comments are weighted up because they are a stronger signal than
    a raw view on a video that is only minutes old. Age is floored at 5 minutes:
    the search index lags, and a 1-minute-old video with 40 views should not
    outrank a 30-minute-old video with 50,000.
    """
    engagement = views + 20 * likes + 50 * comments
    return engagement / max(age_minutes, 5.0)


def _get(path: str, key: str, **params) -> dict:
    r = requests.get(f"{API}/{path}", params={**params, "key": key}, timeout=30)
    if r.status_code != 200:
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = r.text[:200]
        raise RuntimeError(f"YouTube API {path} failed ({r.status_code}): {msg}")
    return r.json()


def find_trending(
    api_key: str,
    window_minutes: int = 60,
    region: str = "US",
    query: str | None = None,
    creative_commons_only: bool = False,
    min_seconds: int = 20,
    max_seconds: int = 4 * 3600,
    now: datetime | None = None,
) -> list[Candidate]:
    """Videos published in the last `window_minutes`, fastest-rising first."""
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY is not set (see .env.example).")
    now = now or datetime.now(timezone.utc)
    after = (now - timedelta(minutes=window_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = dict(
        part="id", type="video", order="viewCount", maxResults=50,
        publishedAfter=after, regionCode=region,
    )
    if query:
        params["q"] = query
    if creative_commons_only:
        params["videoLicense"] = "creativeCommon"
    ids = [it["id"]["videoId"] for it in _get("search", api_key, **params).get("items", [])]
    if not ids:
        return []

    items = _get(
        "videos", api_key, part="snippet,statistics,contentDetails", id=",".join(ids)
    ).get("items", [])
    out = []
    for it in items:
        sn, st = it["snippet"], it.get("statistics", {})
        if sn.get("liveBroadcastContent", "none") != "none":
            continue  # live or upcoming: nothing to clip yet
        seconds = parse_duration(it["contentDetails"].get("duration", ""))
        if not min_seconds <= seconds <= max_seconds:
            continue
        published = datetime.fromisoformat(sn["publishedAt"].replace("Z", "+00:00"))
        age = max((now - published).total_seconds() / 60, 0.0)
        views, likes, comments = (
            int(st.get(k, 0)) for k in ("viewCount", "likeCount", "commentCount")
        )
        out.append(Candidate(
            video_id=it["id"], title=sn["title"], channel=sn["channelTitle"],
            published_at=sn["publishedAt"], age_minutes=round(age, 1),
            duration_seconds=seconds, views=views, likes=likes, comments=comments,
            score=round(velocity_score(views, likes, comments, age), 1),
        ))
    return sorted(out, key=lambda c: c.score, reverse=True)
