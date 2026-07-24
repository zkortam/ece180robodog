#!/usr/bin/env python3
"""Render the voice UI as a static page for Vercel.

The hosted page is only a shell. Every request it makes goes to the agent running
on the UNO Q, reached through the laptop's `adb forward` port. Nothing about the
model, the biometric database, or the camera pipeline leaves the board, and
Vercel never needs Python.

Browsers treat http://127.0.0.1 as a trustworthy origin, so an HTTPS page is
allowed to call it. The board supplies the matching CORS and Private Network
Access headers (see voice_agent/web.py).

    python scripts/build_vercel.py            # -> public/index.html
    python scripts/build_vercel.py --api http://127.0.0.1:9000
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice_agent import config          # noqa: E402
from voice_agent.web import PAGE        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8100",
                    help="where the board is reachable from the browser")
    ap.add_argument("--out", default=str(ROOT.parent / "public" / "index.html"))
    args = ap.parse_args()

    page = (PAGE.replace("__ENDPOINT_MS__", str(config.VAD_ENDPOINT_MS))
                .replace("__API_BASE__", json.dumps(args.api.rstrip("/"))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out}  (api base: {args.api})")


if __name__ == "__main__":
    main()
