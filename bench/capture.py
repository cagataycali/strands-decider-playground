"""Drive a JevBench run on the live Space over the network and fetch its results.

This is how the vendored baseline under ``bench/baseline/`` was captured, and the
tool for anyone who wants fresh numbers without touching the UI: it never loads a
model locally — every forward pass happens on the Space (a100, bf16) through the
same ``/_bench_start`` and ``/_bench_tick`` handlers the Run button uses.

    python -m bench.capture start   [--scope "full JevBench (231)"] [--model "all public models"]
    python -m bench.capture wait    <run_id>            # poll until finished + pushed
    python -m bench.capture fetch   <run_id> [--out bench/baseline]   # download results.jsonl + meta.json
    python -m bench.capture run     ...                 # start + wait + fetch in one go
    python -m bench.capture vendor  <run_id> [--out bench/baseline]   # fetch + write the baseline meta
    python -m bench.capture baseline [--out bench/baseline]   # the full 231 x 7: ONE run PER MODEL, then merged

Why one run per model: the board's own Run button is task-major (a task column fills across all
models), and the 26B base is "exclusive" on an 80 GB card — it evicts every other base — so a
task-major 7-model run reloads weights on every task (measured 9 s/cell, ~4 h). Model-major loads
each base once: minutes. The merged file is identical in shape to a single run's results.jsonl.

Needs a Hugging Face token (``huggingface-cli login`` or HF_TOKEN) with read access to
the Space and to the results dataset; ``--space`` / ``--dataset`` override the repos.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import time
from pathlib import Path

SPACE = "cagataydev/strands-decider-playground"
DATASET = "cagataydev/strands-decider-playground-bench"
FULL_SCOPE = "full JevBench (231)"
ALL_MODELS = "all public models"

_RUN_RE = re.compile(r"· run ([0-9]{8}T[0-9]{6}Z-[0-9a-f]{4})")
_DONE_RE = re.compile(r"(\d+)/(\d+) cells")
_STATE_RE = re.compile(r'class="sdb-status-t">(.*?)<span class="sdb-dim">', re.S)


def _client(space: str):
    from gradio_client import Client
    from huggingface_hub import get_token

    return Client(space, token=get_token(), verbose=False)


def _status(board_html: str) -> dict:
    """Parse the board's status line into {run_id, done, total, live, text}."""
    text = ""
    m = _STATE_RE.search(board_html or "")
    if m:
        text = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
    rid = _RUN_RE.search(board_html or "")
    dn = _DONE_RE.search(text)
    return {
        "run_id": rid.group(1) if rid else None,
        "done": int(dn.group(1)) if dn else None,
        "total": int(dn.group(2)) if dn else None,
        "live": 'data-live="1"' in (board_html or ""),
        "pushed": "pushed to private dataset" in text,
        "text": text,
    }


def model_names(space: str) -> list[str]:
    """The exact model names the Space's Models dropdown offers (everything but "all public models")."""
    c = _client(space)
    info = c.view_api(return_format="dict", print_info=False)
    params = info["named_endpoints"]["/_bench_start"]["parameters"]
    enum = next(p for p in params if p["parameter_name"] == "model_v")["type"]["enum"]
    return [m for m in enum if m != ALL_MODELS]


def start(space: str, scope: str, model: str) -> dict:
    c = _client(space)
    board, est = c.predict(scope, model, api_name="/_bench_start")
    st = _status(board)
    st["estimate"] = html.unescape(re.sub(r"<[^>]+>", "", est or "")).strip()
    return st


def tick(space: str) -> dict:
    c = _client(space)
    board, _errors = c.predict(False, api_name="/_bench_tick")
    return _status(board)


def _log(*a):
    print(*a, flush=True)


def wait(space: str, run_id: str, every: float = 15.0, log=_log) -> dict:
    """Poll /_bench_tick until the run is no longer live and its push is reported."""
    t0 = time.time()
    last = None
    while True:
        st = tick(space)
        if st["run_id"] and st["run_id"] != run_id:
            log(f"[wait] Space's latest run is {st['run_id']}, not {run_id} — someone started a newer run")
        key = (st["done"], st["live"], st["pushed"])
        if key != last:
            log(f"[wait] {time.time() - t0:7.0f}s  {st['text'][:160]}")
            last = key
        if not st["live"] and (st["pushed"] or "not pushed" in st["text"] or "push failed" in st["text"]
                               or "push refused" in st["text"] or "error:" in st["text"]):
            st["wall_s"] = time.time() - t0
            return st
        time.sleep(every)


def fetch(dataset: str, run_id: str, out: Path) -> tuple[Path, Path]:
    """Download runs/<run_id>/{results.jsonl,meta.json} from the private dataset (read-only)."""
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("results.jsonl", "meta.json"):
        p = hf_hub_download(dataset, f"runs/{run_id}/{name}", repo_type="dataset", local_dir=str(out / "_dl"))
        dst = out / name
        dst.write_bytes(Path(p).read_bytes())
        paths.append(dst)
    return paths[0], paths[1]


def baseline_capture(space: str, dataset: str, out: Path, every: float, log=_log) -> dict:
    """Full 231 x every public model as sequential single-model runs on the Space, merged into out/."""
    out.mkdir(parents=True, exist_ok=True)
    models = model_names(space)
    log(f"[baseline] {len(models)} models: {models}")
    t0 = time.time()
    runs, metas, rows = [], [], []
    for m in models:
        st = start(space, FULL_SCOPE, m)
        if not st["run_id"]:
            raise SystemExit(f"could not start {m}: {st['text'][:200]}")
        log(f"[baseline] {m}: run {st['run_id']} started")
        fin = wait(space, st["run_id"], every, log)
        tmp = out / "_runs" / st["run_id"]
        fetch(dataset, st["run_id"], tmp)
        meta = json.loads((tmp / "meta.json").read_text())
        metas.append(meta)
        runs.append(st["run_id"])
        n = 0
        for line in (tmp / "results.jsonl").open(encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
                n += 1
        log(f"[baseline] {m}: {n} cells, {meta.get('finished', 0) - meta.get('started', 0):.0f} s on the Space, push {meta.get('push', {}).get('status')}")
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    merged = {
        "run_id": "+".join(runs), "run_ids": runs, "label": f"baseline · {len(models)} single-model runs merged",
        "models": models, "n_tasks": metas[0].get("n_tasks"), "task_ids": metas[0].get("task_ids"),
        "started": min(x.get("started", 0) for x in metas), "finished": max(x.get("finished", 0) for x in metas),
        "done": len(rows), "total": sum(x.get("total", 0) for x in metas),
        "cancelled": any(x.get("cancelled") for x in metas), "error": "; ".join(x["error"] for x in metas if x.get("error")) or None,
        "push": {"status": "pushed", "repo": dataset, "paths": [f"runs/{r}" for r in runs]},
        "device": metas[0].get("device"), "totals": {k: v for x in metas for k, v in (x.get("totals") or {}).items()},
        "per_run": [{"run_id": x["run_id"], "models": x["models"], "started": x.get("started"), "finished": x.get("finished"),
                     "done": x.get("done"), "total": x.get("total"), "push": x.get("push")} for x in metas],
        "order": "model-major (one run per model); see bench/capture.py",
    }
    (out / "meta.json").write_text(json.dumps(merged, indent=1))
    return vendor_meta(space, dataset, merged["run_id"], out, time.time() - t0)


def vendor_meta(space: str, dataset: str, run_id: str, out: Path, wall_s: float | None = None) -> dict:
    """Rewrite meta.json into the small, self-describing baseline meta the UI reads."""
    from huggingface_hub import HfApi

    meta = json.loads((out / "meta.json").read_text())
    rows = [json.loads(l) for l in (out / "results.jsonl").open(encoding="utf-8") if l.strip()]
    api = HfApi()
    info = api.space_info(space)
    lat = {}
    for r in rows:
        if isinstance(r.get("latency_ms"), (int, float)) and not r.get("cold"):
            lat.setdefault(r["model"], []).append(float(r["latency_ms"]))
    import statistics

    per_model = {}
    for m, tot in (meta.get("totals") or {}).items():
        v = sorted(lat.get(m, []))
        per_model[m] = dict(tot)
        per_model[m]["latency_ms_median"] = round(statistics.median(v), 1) if v else None
        per_model[m]["latency_ms_p90"] = round(v[int(0.9 * (len(v) - 1))], 1) if v else None
    env = meta.get("env") or {}
    base = {
        "kind": "baseline",
        "run_id": run_id,
        "label": meta.get("label"),
        "captured_utc": dt.datetime.fromtimestamp(meta.get("finished") or time.time(), dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ"),
        "date": dt.datetime.fromtimestamp(meta.get("finished") or time.time(), dt.timezone.utc).strftime("%Y-%m-%d"),
        "space": space,
        "space_sha": getattr(info, "sha", None),
        "hardware": "a100-large",
        "hardware_detail": "Nvidia A100 large (80 GB), bf16",
        "dtype": "bf16",
        "models": meta.get("models"),
        "n_tasks": meta.get("n_tasks"),
        "n_cells": len(rows),
        "total": meta.get("total"),
        "wall_s": round(sum((x.get("finished") or 0) - (x.get("started") or 0) for x in meta["per_run"]), 1) if meta.get("per_run")
        else round((meta.get("finished") or 0) - (meta.get("started") or 0), 1),
        "wall_s_note": "sum of the Space-side wall time of every run (model load + 231 forwards + scoring); latency_ms per cell is the forward alone",
        "run_ids": meta.get("run_ids") or [run_id],
        "order": meta.get("order", "task-major"),
        "wall_s_observed_by_client": round(wall_s, 1) if wall_s else None,
        "cancelled": bool(meta.get("cancelled")),
        "error": meta.get("error"),
        "dataset": dataset,
        "dataset_paths": [f"runs/{r}" for r in (meta.get("run_ids") or [run_id])],
        "versions": env,
        "per_model": per_model,
        "how": "python -m bench.capture run — every cell computed on the Space through answer(), scored with JevBench's score_task",
    }
    (out / "meta.json").write_text(json.dumps(base, indent=1) + "\n")
    return base


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["start", "tick", "wait", "fetch", "run", "vendor", "baseline"])
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--space", default=SPACE)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--scope", default=FULL_SCOPE)
    ap.add_argument("--model", default=ALL_MODELS)
    ap.add_argument("--out", default="bench/baseline")
    ap.add_argument("--every", type=float, default=15.0, help="poll interval in seconds")
    a = ap.parse_args(argv)
    out = Path(a.out)

    if a.cmd == "start":
        st = start(a.space, a.scope, a.model)
        print(json.dumps(st, indent=1))
        return 0 if st["run_id"] else 1
    if a.cmd == "tick":
        print(json.dumps(tick(a.space), indent=1))
        return 0
    if a.cmd == "wait":
        if not a.run_id:
            ap.error("wait needs a run_id")
        st = wait(a.space, a.run_id, a.every)
        print(json.dumps(st, indent=1))
        return 0
    if a.cmd == "fetch":
        if not a.run_id:
            ap.error("fetch needs a run_id")
        r, m = fetch(a.dataset, a.run_id, out)
        print(r, m)
        return 0
    if a.cmd == "vendor":
        if not a.run_id:
            ap.error("vendor needs a run_id")
        fetch(a.dataset, a.run_id, out)
        print(json.dumps(vendor_meta(a.space, a.dataset, a.run_id, out), indent=1))
        return 0
    if a.cmd == "baseline":
        print(json.dumps(baseline_capture(a.space, a.dataset, out, a.every), indent=1))
        return 0
    # run = start + wait + fetch + vendor
    st = start(a.space, a.scope, a.model)
    print(json.dumps(st, indent=1), flush=True)
    if not st["run_id"]:
        print("could not start a run (another run live?)", file=sys.stderr)
        return 1
    fin = wait(a.space, st["run_id"], a.every)
    print(json.dumps(fin, indent=1), flush=True)
    fetch(a.dataset, st["run_id"], out)
    print(json.dumps(vendor_meta(a.space, a.dataset, st["run_id"], out, fin.get("wall_s")), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
