import numpy as np

from clipfarm.config import FPS, SAMPLE_RATE
from clipfarm.character import Character, mouth_envelope
from clipfarm.discover import parse_duration, velocity_score
from clipfarm.moments import Reaction, heuristic_plan
from clipfarm.source import Line, SourceInfo, parse_json3
from clipfarm.voice import schedule, speech_text


def test_parse_duration():
    assert parse_duration("PT1H2M3S") == 3723
    assert parse_duration("PT45S") == 45
    assert parse_duration("P0D") == 0


def test_velocity_prefers_fast_risers_and_floors_age():
    assert velocity_score(50_000, 0, 0, 30) > velocity_score(50_000, 0, 0, 55)
    # a 1-minute-old video is scored as if 5 minutes old
    assert velocity_score(100, 0, 0, 1) == velocity_score(100, 0, 0, 5)


def test_parse_json3_skips_empty_events():
    data = {"events": [
        {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "hello "}, {"utf8": "world"}]},
        {"tStartMs": 3000, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 4000},
    ]}
    assert parse_json3(data) == [Line(1.0, 3.0, "hello world")]


def _info(**kw):
    lines = [Line(i * 5.0, i * 5.0 + 4, "just some filler talking here") for i in range(24)]
    lines[14] = Line(70.0, 74.0, "No way, that is insane! He lied?!")
    return SourceInfo("id", "Title", "Chan", "url", 120.0, lines, **kw)


def test_heuristic_plan_finds_the_hot_stretch():
    plan = heuristic_plan(_info(), 30)
    assert plan.end - plan.start == 30
    assert plan.start <= 70 and plan.end >= 74
    assert plan.reactions and all(0 < r.at < 30 for r in plan.reactions)
    hot = min(plan.reactions, key=lambda r: abs(r.at - (74 - plan.start)))
    assert hot.emotion == "shock"


def test_heuristic_plan_uses_heatmap_and_works_without_captions():
    heat = [{"start_time": t, "end_time": t + 10, "value": 1.0 if t == 90 else 0.1}
            for t in range(0, 120, 10)]
    plan = heuristic_plan(SourceInfo("id", "T", "C", "u", 120.0, [], heat), 30)
    assert plan.start <= 90 < plan.end
    assert len(plan.reactions) == 3


def test_schedule_pushes_overlaps_and_drops_overflow():
    two_sec = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    rs = [Reaction(1.0, "a", "gasp"), Reaction(2.0, "b", "shock"), Reaction(9.0, "c", "laugh")]
    out = schedule(rs, [two_sec] * 3, duration=10.0)
    assert [r.text for r, _, _ in out] == ["a", "b"]
    assert out[1][0].at >= 3.6  # after the first line plus the gap


def test_v3_tags_only_for_v3_models():
    r = Reaction(1.0, "Oh my god!", "gasp")
    assert speech_text(r, "eleven_v3") == "[gasps] Oh my god!"
    assert speech_text(r, "eleven_turbo_v2_5") == "Oh my god!"


def test_mouth_follows_voice_and_frames_render():
    track = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    track[SAMPLE_RATE:] = 0.5
    env = mouth_envelope(track, 2 * FPS)
    assert env[:FPS - 2].max() == 0 and env[FPS + 2:].min() > 0.5
    c = Character("Gaspy", "via Chan")
    for emotion in (None, "gasp", "shock", "laugh", "cringe"):
        img = c.frame(1.0, emotion, 1.0 if emotion else 0.0, 0.7, "Wait, WHAT?!")
        assert img.size == (1080, 720)


# ------------------------------------------------------------- gameplay

from clipfarm.gameplay import HOP, Highlight, pick_highlights, template_script, write_captions


def test_pick_highlights_finds_separate_peaks():
    score = np.zeros(int(300 / HOP))
    score[int(100 / HOP)] = 10   # 100s
    score[int(220 / HOP)] = 8    # 220s
    hs = pick_highlights(score, 30, 3, 300.0)
    peaks = sorted(h.peak for h in hs[:2])
    assert peaks == [100.0, 220.0]
    for h in hs:
        assert h.end - h.start == 30 and h.start <= h.peak <= h.end
    top = hs[0]
    assert top.peak == 100.0 and 15 < top.peak - top.start < 25  # build-up before the payoff


def test_pick_highlights_short_recording_uses_whole_file():
    hs = pick_highlights(np.array([0.0, 1.0, 0.0]), 30, 3, 1.5)
    assert len(hs) == 1 and hs[0].start == 0 and hs[0].end == 1.5


def test_template_script_fits_the_clip():
    for style in ("hype", "funny", "story"):
        sc = template_script(Highlight(0, 30, 25, 1), style)
        assert sc.hook and len(sc.lines) == 4
        assert all(0 <= l.at < 30 for l in sc.lines)


def test_captions_are_chunked_and_escaped(tmp_path):
    r = Reaction(2.0, "one two three four {bad}", "talk")
    out = write_captions("Hook {x}", [(r, np.zeros(10, dtype=np.float32), 2.0)], 30, tmp_path / "c.ass")
    text = out.read_text()
    assert "Hook,,0,0,0,,HOOK X" in text
    assert "ONE TWO THREE" in text and "FOUR BAD" in text and "{" not in text.split("[Events]")[1]
