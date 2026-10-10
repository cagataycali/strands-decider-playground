"""Keep the Space card's front matter in step with the facts it describes.

    python docs/check_front_matter.py          # exit 1 when README.md's front matter is stale
    python docs/check_front_matter.py --write  # rewrite the front matter block in place
    python docs/check_front_matter.py --live   # also compare `models:` with the org on the Hub

The generated keys are `models` (the public deciders the vendored baseline was captured on,
from bench/baseline/meta.json -- the same list the registry reads from the org at runtime),
`thumbnail` (the share card, from seo.py) and `short_description` length (<=60, HF's limit).
Hand-maintained keys (title, emoji, sdk, tags, license, suggested_hardware, ...) are kept as
they are; the block is rewritten key-for-key so a diff shows exactly what moved.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import seo  # noqa: E402

README = ROOT / "README.md"
META = ROOT / "bench" / "baseline" / "meta.json"
ORG = "StrandsAgents"
SHORT_MAX = 60

# Order HF renders them in; anything not listed keeps its position after these.
ORDER = ["title", "emoji", "colorFrom", "colorTo", "sdk", "sdk_version", "app_file", "pinned", "header",
         "license", "short_description", "suggested_hardware", "thumbnail", "tags", "models", "datasets"]

HAND = {  # defaults for keys a human owns; only written when absent
    "license": "mit",
    "suggested_hardware": "a100-large",
    "tags": ["decision-making", "calibration", "probabilistic-decisions", "jevbench", "benchmark",
             "lora", "gemma", "qwen", "strands", "agents", "playground"],
    "datasets": ["JevBench/jevbench"],
}


def parse(text: str) -> tuple[dict, str]:
    """Front matter -> ordered dict of str | list[str]; also returns the body after it."""
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise SystemExit("README.md has no front matter")
    out: dict = {}
    key = None
    for line in m.group(1).splitlines():
        if re.match(r"^\s*-\s+", line) and key is not None:
            if not isinstance(out[key], list):
                out[key] = []
            out[key].append(line.split("-", 1)[1].strip())
        elif ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            out[key] = val if val != "" else []
    return out, text[m.end():]


def dump(fm: dict) -> str:
    keys = [k for k in ORDER if k in fm] + [k for k in fm if k not in ORDER]
    lines = ["---"]
    for k in keys:
        v = fm[k]
        if isinstance(v, list):
            lines.append(f"{k}:")
            lines.extend(f"  - {x}" for x in v)
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def expected(fm: dict) -> dict:
    meta = json.loads(META.read_text(encoding="utf-8"))
    repos = sorted(f"{ORG}/{m}" for m in meta["models"])
    want = dict(fm)
    for k, v in HAND.items():
        want.setdefault(k, v)
    want["models"] = repos
    # preload_from_hub is deliberately ABSENT: measured 2026-10-10 07:48Z on this Space, the build-time
    # preload left /home/user/.cache/huggingface owned by another user, so every model load died with
    # "I/O error: Permission denied (os error 13)" (safetensors reconstruct + xet log). Never add it back
    # without re-testing a model call on the deployed Space.
    want.pop("preload_from_hub", None)
    want["thumbnail"] = seo.OG_IMAGE
    return want


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--live", action="store_true", help="compare models: with the public org on the Hub")
    a = ap.parse_args()
    text = README.read_text(encoding="utf-8")
    fm, body = parse(text)
    want = expected(fm)
    problems = []
    sd = str(want.get("short_description", ""))
    if not sd or len(sd) > SHORT_MAX:
        problems.append(f"short_description {len(sd)} chars (max {SHORT_MAX})")
    if "preload_from_hub" in fm:
        problems.append("preload_from_hub present: it breaks model loads on this Space (cache ownership), remove it")
    for k in ("models", "thumbnail", "license", "tags", "datasets", "suggested_hardware"):
        if fm.get(k) != want.get(k):
            problems.append(f"{k}: README has {fm.get(k)!r}, facts say {want.get(k)!r}")
    if a.live:
        from huggingface_hub import HfApi
        live = sorted(m.id for m in HfApi().list_models(author=ORG) if m.id.split("/")[-1].startswith("strands-decider-") and not m.private)
        if live != want["models"]:
            problems.append(f"org has {len(live)} public deciders, baseline/meta.json lists {len(want['models'])}: "
                            f"{sorted(set(live) ^ set(want['models']))}")
    if a.write:
        README.write_text(dump(want) + body, encoding="utf-8")
        print("front matter written:", ", ".join(k for k in want))
        problems = [p for p in problems if p.startswith("short_description") or p.startswith("org has")]
    for p in problems:
        print("STALE:", p)
    print("front matter OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
