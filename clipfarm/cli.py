"""Command line entry point: `clipfarm <command>` or `python -m clipfarm`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Settings
from .discover import Candidate, find_trending


def _seen_path(s: Settings) -> Path:
    return s.output_dir / "seen.json"


def _load_seen(s: Settings) -> set[str]:
    p = _seen_path(s)
    return set(json.loads(p.read_text())) if p.exists() else set()


def _save_seen(s: Settings, seen: set[str]) -> None:
    s.output_dir.mkdir(parents=True, exist_ok=True)
    _seen_path(s).write_text(json.dumps(sorted(seen)))


def _discover(args, s: Settings) -> list[Candidate]:
    return find_trending(
        s.youtube_api_key, window_minutes=args.window, region=args.region,
        query=args.query, creative_commons_only=args.creative_commons,
    )


def _print_table(cands: list[Candidate]) -> None:
    print(f"{'score':>9} {'views':>8} {'age':>5}  title")
    for c in cands:
        print(f"{c.score:>9.0f} {c.views:>8} {c.age_minutes:>4.0f}m  {c.title[:58]}  [{c.url}]")


def cmd_discover(args, s: Settings) -> int:
    cands = _discover(args, s)[: args.top]
    if args.json:
        print(json.dumps([c.to_dict() for c in cands], indent=2))
    elif cands:
        _print_table(cands)
    else:
        print("No videos found in that window.")
    return 0


def _run_once(args, s: Settings) -> int:
    from .pipeline import make_from_url

    seen = _load_seen(s)
    fresh = [c for c in _discover(args, s) if c.video_id not in seen][: args.count]
    if not fresh:
        print("Nothing new.")
        return 0
    failures = 0
    for c in fresh:
        try:
            print(f"  -> {make_from_url(c.url, s)}")
        except Exception as e:
            failures += 1
            print(f"  ! failed on {c.url}: {e}")
        seen.add(c.video_id)  # failed ones too: do not retry them every scan
        _save_seen(s, seen)
    return 1 if failures == len(fresh) else 0


def cmd_run(args, s: Settings) -> int:
    return _run_once(args, s)


def cmd_watch(args, s: Settings) -> int:
    print(f"Scanning every {args.interval} min (each scan costs ~101 YouTube quota units). Ctrl-C to stop.")
    try:
        while True:
            print(time.strftime("[%H:%M:%S] scanning"))
            try:
                _run_once(args, s)
            except Exception as e:
                print(f"  ! scan failed: {e}")
            time.sleep(args.interval * 60)
    except KeyboardInterrupt:
        return 0


def cmd_make(args, s: Settings) -> int:
    from .pipeline import make_from_url

    for url in args.urls:
        print(f"  -> {make_from_url(url, s)}")
    return 0


def cmd_demo(args, s: Settings) -> int:
    from .pipeline import demo

    print(f"  -> {demo(s)}")
    return 0


def cmd_app(args, s: Settings) -> int:
    from .webapp import serve

    serve(args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="clipfarm", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add_discovery(sp):
        sp.add_argument("--window", type=int, default=60,
                        help="only videos published in the last N minutes (default 60)")
        sp.add_argument("--region", default="US", help="ISO country code (default US)")
        sp.add_argument("--query", help="optional search terms to narrow the niche")
        sp.add_argument("--creative-commons", action="store_true",
                        help="only videos under a Creative Commons licence")

    sp = sub.add_parser("app", help="open the web app (start here)")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=cmd_app)

    sp = sub.add_parser("discover", help="list the fastest-rising new videos")
    add_discovery(sp)
    sp.add_argument("--top", type=int, default=15)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_discover)

    sp = sub.add_parser("run", help="discover, then make clips from the top new videos")
    add_discovery(sp)
    sp.add_argument("--count", type=int, default=3, help="clips to make per scan (default 3)")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("watch", help="repeat `run` on a timer")
    add_discovery(sp)
    sp.add_argument("--count", type=int, default=1)
    sp.add_argument("--interval", type=int, default=15, help="minutes between scans (default 15)")
    sp.set_defaults(func=cmd_watch)

    sp = sub.add_parser("make", help="make a clip from specific video URLs")
    sp.add_argument("urls", nargs="+")
    sp.set_defaults(func=cmd_make)

    sp = sub.add_parser("demo", help="offline test run on a generated video (no keys needed)")
    sp.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    try:
        return args.func(args, Settings.from_env())
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
