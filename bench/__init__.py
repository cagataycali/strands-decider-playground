"""JevBench inside the playground: browse the 231 public tasks, run the bench on every listed
model from a button (a persistent background job whose rows are pushed to a Hugging Face dataset,
`BENCH_DATASET_REPO`), and the activity board (task x model, contribution-graph style) with the
common-error panels.

Scoring is JevBench's own (bench/jev.py delegates to the pinned checkout); the paired test is
strands-decider's evaluation/jevbench/paired.py copied verbatim (bench/paired_upstream.py).
Nothing here edits app.answer(): the job calls it exactly like the Decide button does.
"""
from __future__ import annotations

import csv
import html
import json
import math
import os
import statistics
import threading
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from . import jev
from .paired_upstream import mcnemar_exact

HERE = Path(__file__).resolve().parent.parent  # space root (this file is bench/__init__.py)
RUNS = HERE / "bench" / "runs"
PUBLISHED = HERE / "bench" / "published"
BASELINE = HERE / "bench" / "baseline"  # the full 231 x 7 run captured once on the Space (bench/capture.py)
STATIC = HERE / "static"
DATASET_REPO = os.environ.get("BENCH_DATASET_REPO", "cagataydev/strands-decider-playground-bench")
TIERS = ("easy", "standard", "hard")
TIER_OF_FILE = {"original": "standard", "easy": "easy", "hard": "hard"}

_TASKS: dict | None = None
_ORDER: list[str] | None = None
_LOCK = threading.RLock()  # re-entrant: start_run holds it while task_ids() -> tasks() takes it again (a plain Lock deadlocked the Run button)


def E(s: Any) -> str:
    """HTML-escape anything (None -> empty string)."""
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------------
def tasks() -> dict:
    """{task_id: Task} for the 231 vendored tasks, loaded once."""
    global _TASKS, _ORDER
    with _LOCK:
        if _TASKS is None:
            _TASKS = jev.load_tasks()
            _ORDER = jev.ordered_ids(_TASKS)
    return _TASKS


def order() -> list[str]:
    """Task ids in benchmark order (original, easy, hard)."""
    tasks()
    return list(_ORDER or [])


def tier(t) -> str:
    """A task's tier (easy | standard | hard)."""
    return (t.provenance or {}).get("tier", "standard")


def task_ids(tier_filter: str | None = None) -> list[str]:
    """Ordered task ids, optionally one tier only."""
    ts = tasks()
    return [i for i in order() if not tier_filter or tier(ts[i]) == tier_filter]


def state_text(t) -> str:
    """JevBench states are str or dict; strands-decider's prompting.render_content emits a dict as
    json.dumps(indent=2, ensure_ascii=False) and strips a str -- so this string renders IDENTICALLY
    to what the typesafe adapter's raw dict would have (prefix-cache-stable, parity-safe)."""
    return t.state if isinstance(t.state, str) else json.dumps(t.state, indent=2, ensure_ascii=False)


def fill_values(task_id: str) -> dict | None:
    """What the ask bar needs for one task: state + ONE question (type, instructions, options).
    The gold answer is NOT part of this -- the UI only shows it after the models have answered."""
    t = tasks().get(task_id)
    if t is None:
        return None
    spec = jev.to_spec(t)
    return {"id": t.id, "state": state_text(t), "type": spec["type"], "instructions": spec["instructions"],
            "options": list(spec.get("options") or []), "spec": spec}


def groups() -> list[tuple[str, str, list[str]]]:
    """Board column groups: (tier, family, [task ids]) in board order."""
    ts = tasks()
    out: list[tuple[str, str, list[str]]] = []
    for tid in order():
        key = (tier(ts[tid]), ts[tid].family)
        if out and (out[-1][0], out[-1][1]) == key:
            out[-1][2].append(tid)
        else:
            out.append((key[0], key[1], [tid]))
    return out


def task_meta(tid: str) -> dict:
    """Small record for the picker and the board: id, tier, family, type, title."""
    t = tasks()[tid]
    return {"id": tid, "tier": tier(t), "family": t.family, "type": t.question["type"],
            "labels": list(t.labels), "expected": str(t.expected)}


def picker_payload() -> list[dict]:
    """Compact task list embedded in the page for the client-side search/filter."""
    ts = tasks()
    out = []
    for tid in order():
        t = ts[tid]
        out.append({"id": tid, "tier": tier(t), "family": t.family, "type": t.question["type"],
                    "q": str(t.question["instructions"])[:140], "state": state_text(t)[:160], "n": len(t.labels)})
    return out


# ---------------------------------------------------------------------------------
# scoring one card (the dict app.answer() returns for one model)
# ---------------------------------------------------------------------------------
def score_card(card: dict, tid: str) -> dict:
    """Score one model card against task `tid` with JevBench's own scorer -> one results row."""
    t = tasks()[tid]
    ans = None
    for v in (card.get("answers") or {}).values():
        ans = v
        break
    probs = jev.probs_from_card_answer(ans or {}, t) if ans else None
    sc = jev.score_probs(probs, t)
    m = card.get("metrics") or {}
    return {
        "task_id": tid, "tier": tier(t), "family": t.family, "type": t.question["type"],
        "model": card.get("name"), "repo": card.get("repo"),
        "predicted": sc.get("predicted"), "expected": str(t.expected),
        "correct": bool(sc.get("correct")), "valid": bool(sc.get("valid")), "strict_valid": bool(sc.get("strict_valid")),
        "p_gold": round(float(sc.get("p_gold", 0.0)), 4), "top_p": round(float(sc.get("top_p", 0.0)), 4),
        "brier": round(float(sc.get("brier", 1.0)), 4), "probs": sc.get("probs") or probs,
        "latency_ms": m.get("latency_ms"), "cold": card.get("cold"),
        "error": card.get("error") or sc.get("error"), "ts": time.time(),
    }


# ---------------------------------------------------------------------------------
# published runs (board rows on day one, before any GPU run)
# ---------------------------------------------------------------------------------
def published_rows() -> list[dict]:
    """7 published per-task result files (with probabilities) + the correct-only history CSV."""
    rows: list[dict] = []
    for d in sorted(PUBLISHED.iterdir()):
        f = d / "results.jsonl"
        if not d.is_dir() or not f.exists():
            continue
        res: dict[str, dict] = {}
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            tid = r.get("task_id")
            if tid not in tasks():
                continue
            t = tasks()[tid]
            probs = r.get("probs") or None
            sc = jev.score_probs(probs, t) if probs else {}
            res[tid] = {
                "task_id": tid, "correct": bool(r.get("correct")), "valid": bool(r.get("valid", True)),
                "strict_valid": bool(r.get("strict_valid", r.get("valid", True))),
                "predicted": r.get("predicted"), "expected": str(t.expected),
                "p_gold": round(float(sc.get("p_gold", 1.0 if r.get("correct") else 0.0)), 4),
                "brier": round(float(sc.get("brier", 1.0)), 4) if probs else None,
                "latency_ms": round(float(r["latency_s"]) * 1000, 1) if r.get("latency_s") is not None else None,
                "probs": probs,
            }
        src = {}
        if (d / "SOURCE.json").exists():
            src = json.loads((d / "SOURCE.json").read_text())
        rows.append({"key": "pub:" + d.name, "label": d.name.replace("strands-decider-", ""), "kind": "published",
                     "sub": "published by strands-decider · " + (src.get("eval_dir") or "strands-decider repo"), "results": res,
                     "repo": src.get("repo")})
    # correct-only history
    hist: dict[str, dict[str, dict]] = defaultdict(dict)
    csvf = PUBLISHED / "jevbench_results.csv"
    if csvf.exists():
        for r in csv.DictReader(csvf.open(encoding="utf-8")):
            tid = r["task_id"]
            if tid in tasks():
                ok = r["correct"] in ("1", "True", "true")
                hist[r["run"]][tid] = {"task_id": tid, "correct": ok, "valid": True, "strict_valid": True,
                                       "predicted": None, "expected": str(tasks()[tid].expected),
                                       "p_gold": 1.0 if ok else 0.0, "brier": None, "latency_ms": None, "probs": None}
    for run in sorted(hist, key=lambda k: -sum(1 for v in hist[k].values() if v["correct"])):
        rows.append({"key": "hist:" + run, "label": run, "kind": "history",
                     "sub": "published · correct-only · research/data/jevbench_results.csv", "results": hist[run]})
    return rows


# ---------------------------------------------------------------------------------
# the job: one background thread, persistent on disk, pushed to a private HF dataset
# ---------------------------------------------------------------------------------
class Job:
    """One bench run: a background thread that calls answer() per task x model, appends rows to
    bench/runs/<run_id>/results.jsonl, and pushes the folder to the dataset when it ends."""

    def __init__(self, run_id: str, models: list[str], tids: list[str], label: str):
        self.run_id, self.models, self.tids, self.label = run_id, models, tids, label
        self.dir = RUNS / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.dir / "results.jsonl"
        self.meta_path = self.dir / "meta.json"
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.started = time.time()
        self.finished: float | None = None
        self.done = 0
        self.total = len(models) * len(tids)
        self.results: dict[str, dict[str, dict]] = {m: {} for m in models}  # model -> tid -> row
        self.lat: dict[str, list[float]] = defaultdict(list)
        self.error: str | None = None
        self.push: dict = {"status": "not pushed"}
        self.current: tuple[str, str] | None = None
        self.kind = "run"  # "run" = started on this Space, "baseline" = vendored from bench/baseline/
        self.info: dict = {}  # baseline meta (date, hardware, versions) when kind == "baseline"

    # -- persistence ------------------------------------------------------------
    def meta(self) -> dict:
        """The run's meta.json contents."""
        return {
            "run_id": self.run_id, "label": self.label, "models": self.models, "n_tasks": len(self.tids),
            "task_ids": self.tids, "started": self.started, "finished": self.finished, "done": self.done,
            "total": self.total, "cancelled": self.cancel.is_set(), "error": self.error, "push": self.push,
            "device": os.environ.get("BENCH_DEVICE_HINT"), "totals": self.totals(),
        }

    def save_meta(self) -> None:
        """Write meta.json."""
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.meta(), indent=1))
        tmp.replace(self.meta_path)

    def append(self, row: dict) -> None:
        """Append one results row (JSON line)."""
        with self.results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.results[row["model"]][row["task_id"]] = row
        if isinstance(row.get("latency_ms"), (int, float)) and not row.get("cold"):
            self.lat[row["model"]].append(float(row["latency_ms"]))
        self.done += 1

    @classmethod
    def load(cls, run_dir: Path) -> "Job | None":
        """Re-read a finished run from disk; None when the folder is not a run."""
        try:
            meta = json.loads((run_dir / "meta.json").read_text())
        except Exception:  # noqa: BLE001
            return None
        job = cls(meta["run_id"], meta["models"], meta.get("task_ids") or [], meta.get("label", ""))
        job.started, job.finished, job.error, job.push = meta.get("started", 0), meta.get("finished"), meta.get("error"), meta.get("push", {})
        if meta.get("cancelled"):
            job.cancel.set()
        job.done = 0
        if job.results_path.exists():
            for line in job.results_path.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    job.results.setdefault(r["model"], {})[r["task_id"]] = r
                    job.done += 1
        if job.finished is None and job.thread is None:
            job.error = job.error or "interrupted (process restarted while running)"
            job.finished = job.finished or time.time()
        return job

    @classmethod
    def load_baseline(cls, base_dir: Path | None = None) -> "Job | None":
        """The vendored baseline (bench/baseline/results.jsonl + meta.json written by bench/capture.py vendor):
        a finished Job whose rows were all computed on the Space. None when the files are missing."""
        d = base_dir or BASELINE
        try:
            meta = json.loads((d / "meta.json").read_text())
        except Exception:  # noqa: BLE001
            return None
        f = d / "results.jsonl"
        if not f.exists():
            return None
        job = cls.__new__(cls)
        job.run_id, job.models, job.tids = f"baseline-{meta.get('date', '')}", list(meta.get("models") or []), order()
        job.label = f"baseline · {len(job.models)} models × {meta.get('n_tasks', len(job.tids))} tasks"
        job.dir, job.results_path, job.meta_path = d, f, d / "meta.json"
        job.cancel, job.thread = threading.Event(), None
        job.started, job.finished = 0.0, 0.0
        job.done, job.total = 0, int(meta.get("total") or len(job.models) * len(job.tids))
        job.results, job.lat, job.error, job.current = {m: {} for m in job.models}, defaultdict(list), None, None
        job.push = {"status": "vendored", "repo": meta.get("dataset"), "paths": meta.get("dataset_paths")}
        job.kind, job.info = "baseline", meta
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                job.results.setdefault(r["model"], {})[r["task_id"]] = r
                job.done += 1
                if isinstance(r.get("latency_ms"), (int, float)) and not r.get("cold"):
                    job.lat[r["model"]].append(float(r["latency_ms"]))
        job.started, job.finished = 0.0, float(meta.get("wall_s") or 0.0)
        return job

    # -- state -----------------------------------------------------------------
    def running(self) -> bool:
        """True while the worker thread is alive."""
        return self.thread is not None and self.thread.is_alive()

    def totals(self) -> dict[str, dict]:
        """Per model: done, correct, p50 latency, Brier -- from the rows so far."""
        out = {}
        n = len(self.tids)
        for m in self.models:
            rows = list(self.results.get(m, {}).values())
            out[m] = _totals(rows, n)
        return out

    def eta_s(self) -> float | None:
        """Seconds left at the measured pace, while running."""
        if not self.running():
            return None
        per = [statistics.median(v) for v in self.lat.values() if v]
        if not per:
            return None
        rem = self.total - self.done
        return rem * (statistics.mean(per) / 1000.0 + OVERHEAD_S)

    def elapsed(self) -> float:
        """Seconds since the run started."""
        return (self.finished or time.time()) - self.started

    # -- work ------------------------------------------------------------------
    def start(self, answer_fn: Callable[..., dict]) -> None:
        """Start the worker thread; `answer_fn` is app.answer."""
        self.thread = threading.Thread(target=self._loop, args=(answer_fn,), name=f"bench-{self.run_id}", daemon=True)
        self.save_meta()
        self.thread.start()

    def _loop(self, answer_fn: Callable[..., dict]) -> None:
        """The worker: one answer() per task x model, scored and appended; pushes at the end."""
        try:
            # task-major: a column fills across all models, like a day on the activity board.
            # The runner's cache keeps every public model resident on the Space's GPU
            # (BENCH_ORDER=model makes it model-major instead).
            pairs = [(tid, m) for tid in self.tids for m in self.models]
            if os.environ.get("BENCH_ORDER") == "model":
                pairs = [(tid, m) for m in self.models for tid in self.tids]
            last_save = 0.0
            for tid, m in pairs:
                if self.cancel.is_set():
                    break
                if tid in self.results.get(m, {}):
                    continue
                self.current = (tid, m)
                fv = fill_values(tid)
                try:
                    res = answer_fn(fv["state"], None, [fv["spec"]], [m])
                    cards = res.get("cards") or []
                    if not cards:
                        row = _error_row(tid, m, res.get("error") or "no card")
                    else:
                        row = score_card(cards[0], tid)
                except Exception as e:  # noqa: BLE001
                    row = _error_row(tid, m, f"{type(e).__name__}: {e}")
                self.append(row)
                if time.time() - last_save > 2:
                    self.save_meta()
                    last_save = time.time()
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.current = None
            self.finished = time.time()
            self.save_meta()
            self._push()

    def _push(self) -> None:
        """Upload the run folder to BENCH_DATASET_REPO (skipped without a token); records the outcome."""
        tok = hf_token()
        if not tok:
            self.push = {"status": "skipped", "why": "no HF token in this environment"}
            self.save_meta()
            return
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=tok)
            api.create_repo(DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)
            info = api.repo_info(DATASET_REPO, repo_type="dataset")
            if not info.private:
                self.push = {"status": "refused", "why": "dataset repo is not private"}
                self.save_meta()
                return
            self.push = {"status": "pushing"}
            self.save_meta()
            api.upload_folder(folder_path=str(self.dir), path_in_repo=f"runs/{self.run_id}", repo_id=DATASET_REPO,
                              repo_type="dataset", commit_message=f"bench run {self.run_id}: {self.label}")
            # readback: the file list must contain our results
            files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
            ok = f"runs/{self.run_id}/results.jsonl" in files
            self.push = {"status": "pushed" if ok else "push unverified", "repo": DATASET_REPO,
                         "private": True, "path": f"runs/{self.run_id}", "verified_readback": ok, "at": time.time()}
        except Exception as e:  # noqa: BLE001
            self.push = {"status": "failed", "why": f"{type(e).__name__}: {e}"}
        self.save_meta()


OVERHEAD_S = 0.05  # request build + scoring per call (<50 ms)


def hf_token() -> str | None:
    """HF_TOKEN env (the Space) or the local huggingface-cli login (a developer's machine)."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:  # noqa: BLE001
        return None


def _error_row(tid: str, m: str, err: str) -> dict:
    """A results row for a call that failed (counted wrong, Brier 1.0)."""
    t = tasks()[tid]
    return {"task_id": tid, "tier": tier(t), "family": t.family, "type": t.question["type"], "model": m,
            "predicted": None, "expected": str(t.expected), "correct": False, "valid": False, "strict_valid": False,
            "p_gold": 0.0, "top_p": 0.0, "brier": 1.0, "probs": None, "latency_ms": None, "error": err, "ts": time.time()}


def _totals(rows: list[dict], n: int) -> dict:
    """Aggregate one model's rows: done, correct, p50 latency, mean Brier."""
    done = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    briers = [r["brier"] for r in rows if isinstance(r.get("brier"), (int, float))]
    lat = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    return {"n": n, "done": done, "correct": correct,
            "brier": round(sum(briers) / len(briers), 4) if briers else None,
            "strict_valid": sum(1 for r in rows if r.get("strict_valid")),
            "valid": sum(1 for r in rows if r.get("valid")),
            "p50_ms": round(statistics.median(lat), 1) if lat else None}


# ---- registry of jobs (one live at a time: the runner lock serialises models anyway) ----
_JOBS: dict[str, Job] = {}
_LIVE: Job | None = None


def jobs() -> dict[str, Job]:
    """Every run on disk plus the live one, by run id."""
    if not _JOBS and RUNS.exists():
        for d in sorted(RUNS.iterdir()):
            j = Job.load(d)
            if j:
                _JOBS[j.run_id] = j
    return _JOBS


def live() -> Job | None:
    """The run in progress, if any."""
    return _LIVE if _LIVE and _LIVE.running() else None


_BASELINE: Job | None = None


def baseline() -> Job | None:
    """The vendored baseline job (cached), or None when bench/baseline/ is absent."""
    global _BASELINE
    if _BASELINE is None:
        _BASELINE = Job.load_baseline()
    return _BASELINE


def latest() -> Job | None:
    """What the board shows when nothing is selected: the live run, else the newest run started on this
    Space, else the vendored baseline."""
    js = jobs()
    if _LIVE:
        return _LIVE
    if js:
        return max(js.values(), key=lambda j: j.started)
    return baseline()


def history_choices() -> list[tuple[str, str]]:
    """(label, key) pairs for the history dropdown: the baseline first, then runs on this Space, newest first."""
    out: list[tuple[str, str]] = []
    b = baseline()
    if b:
        out.append((f"baseline · {b.info.get('date', '')} · {b.done}/{b.total} cells", "baseline"))
    for j in sorted(jobs().values(), key=lambda j: -j.started):
        when = time.strftime("%Y-%m-%d %H:%MZ", time.gmtime(j.started))
        out.append((f"{when} · {j.label} · {j.done}/{j.total}", "run:" + j.run_id))
    return out


def job_for(key: str | None) -> Job | None:
    """Resolve a history key ("baseline" | "run:<id>" | ""/None = latest())."""
    if not key:
        return latest()
    if key == "baseline":
        return baseline() or latest()
    return jobs().get(key.split(":", 1)[1]) if key.startswith("run:") else latest()


def start_run(answer_fn: Callable[..., dict], models: list[str], tier_filter: str | None, label: str = "") -> Job | str:
    """Start a run over `models` x the tasks of `tier_filter` (None = all). Returns the Job, or a
    message string when one is already running."""
    global _LIVE
    with _LOCK:
        if live():
            return f"a run is already live ({_LIVE.run_id}); cancel it first"
        tids = task_ids(tier_filter)
        if not models or not tids:
            return "nothing to run"
        rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:4]
        job = Job(rid, list(models), tids, label or (f"tier {tier_filter}" if tier_filter else "full JevBench"))
        jobs()[rid] = job
        _LIVE = job
        job.start(answer_fn)
        return job


def cancel_live() -> bool:
    """Ask the live run to stop after its current call; the partial run is still pushed."""
    j = live()
    if j:
        j.cancel.set()
        return True
    return False


def estimate_s(models: list[str], n_tasks: int) -> tuple[float, str]:
    """Wall-time estimate BEFORE a run, from measured per-model medians only: the last finished run on this
    Space when there is one, else the vendored baseline (captured on the Space). Models with no measurement
    are counted at the median of the measured ones; nothing is assumed from another machine."""
    per: dict[str, float] = {}
    b = baseline()
    if b:
        for m, t in b.totals().items():
            if t.get("p50_ms"):
                per[m] = t["p50_ms"]
    src = "medians measured on this Space"
    for j in sorted(jobs().values(), key=lambda j: j.started):
        for m, t in j.totals().items():
            if t.get("p50_ms"):
                per[m] = t["p50_ms"]
    known = [per[m] for m in models if m in per]
    missing = [m for m in models if m not in per]
    if not known:
        return 0.0, "no measured latency yet"
    fill = statistics.median(known)
    for m in missing:
        per[m] = fill
    if missing:
        src += f" ({len(missing)} model(s) at the measured median)"
    total = sum(per[m] / 1000.0 + OVERHEAD_S for m in models) * n_tasks
    return total, src


# ---------------------------------------------------------------------------------
# the board (SVG) -- Strands drawing rules: 1px strokes, small radius, accent only for correct
# ---------------------------------------------------------------------------------
CELL, GAP, GGAP, LABEL_W, TOTALS_W, HEAD_H, FOOT_H = 11, 2, 9, 310, 230, 68, 44


def _rows_from_job(job: Job | None, labels: dict[str, str], all_models: list[str] | None = None) -> list[dict]:
    """One row per PUBLIC model (the board always shows all of them; a model outside the run has empty results ->
    1px-outline 'not attempted' cells). Without a job the rows are empty outlines so the board reads the same on day one."""
    names = list(all_models or [])
    for m in (job.models if job else []):
        if m not in names:
            names.append(m)
    kind = "live" if job and job.running() else ("baseline" if job and job.kind == "baseline" else "run")
    return [{"key": "run:" + m, "label": labels.get(m, m), "kind": kind, "sub": job.label if job else "",
             "results": (job.results.get(m, {}) if job else {}), "model": m} for m in names]


def board_svg(rows: list[dict], n_models_for_foot: int | None = None) -> str:
    """The task x model activity board as one SVG: accent = correct, ink = wrong, opacity = confidence."""
    gs = groups()
    cols: list[tuple[str, int]] = []  # (tid, x)
    x = LABEL_W
    sep: list[tuple[int, str, str]] = []
    for gi, (tr, fam, ids) in enumerate(gs):
        if gi:
            x += GGAP
        sep.append((x, tr, fam))
        for tid in ids:
            cols.append((tid, x))
            x += CELL + GAP
    grid_w = x - GAP
    W = grid_w + 12 + TOTALS_W
    H = HEAD_H + len(rows) * (CELL + GAP) + FOOT_H
    out = [f'<svg class="sdb-board" xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="group" '
           f'aria-label="JevBench board: {len(rows)} rows x {len(cols)} tasks">']
    # tier brackets + family labels
    tier_x: dict[str, list[int]] = defaultdict(list)
    for (sx, tr, fam), (_, _, ids) in zip(sep, gs):
        tier_x[tr] += [sx, sx + len(ids) * (CELL + GAP) - GAP]
        if len(ids) * (CELL + GAP) > 24:
            out.append(f'<text class="sdb-fam" x="{sx}" y="{HEAD_H - 22}">{E(fam)}</text>')
    for tr, xs in tier_x.items():
        x0, x1 = min(xs), max(xs)
        out.append(f'<path class="sdb-tier" d="M{x0} {HEAD_H - 40} v-6 H{x1} v6" />'
                   f'<text class="sdb-tier-l" x="{(x0 + x1) / 2:.0f}" y="{HEAD_H - 50}" text-anchor="middle">{E(tr)}</text>')
    # column hit targets (click -> task detail)
    for i, (tid, cx) in enumerate(cols):
        out.append(f'<rect class="sdb-col" data-task="{E(tid)}" x="{cx}" y="{HEAD_H - 8}" width="{CELL}" height="{H - HEAD_H - FOOT_H + 8}" '
                   f'rx="2" tabindex="0" role="button" aria-label="task {E(tid)}"/>')
    # rows
    per_task = Counter()
    per_task_n = Counter()
    for ri, row in enumerate(rows):
        y = HEAD_H + ri * (CELL + GAP)
        res = row["results"]
        klass = f"sdb-row sdb-{row['kind']}"
        out.append(f'<g class="{klass}" data-row="{E(row["key"])}">')
        out.append(f'<text class="sdb-lab" x="{LABEL_W - 10}" y="{y + CELL - 2}" text-anchor="end">'
                   f'<title>{E(row.get("sub", ""))}</title>{E(row["label"])}</text>')
        for tid, cx in cols:
            r = res.get(tid)
            if r is None:
                out.append(f'<rect class="sdb-c sdb-none" x="{cx}" y="{y}" width="{CELL}" height="{CELL}" rx="2" data-task="{E(tid)}" data-row="{E(row["key"])}"/>')
                continue
            p = r.get("p_gold")
            p = float(p) if isinstance(p, (int, float)) else (1.0 if r.get("correct") else 0.0)
            if r.get("correct"):
                cls, op = "sdb-ok", 0.3 + 0.7 * p
                per_task[tid] += 1
            elif not r.get("valid", True) or r.get("error"):
                cls, op = "sdb-bad sdb-inv", 1.0
            else:
                cls, op = "sdb-bad", 0.3 + 0.7 * (1.0 - p)
            if row["kind"] in ("live", "run", "baseline"):
                per_task_n[tid] += 1
            pred = r.get("predicted")
            tip = (f'{tid}|{row["label"]}|{E(pred if pred is not None else "—")}|{E(r.get("expected", ""))}|'
                   f'{p:.3f}|{round(r["latency_ms"]) if r.get("latency_ms") is not None else ""}|{E(r.get("error") or "")}')
            out.append(f'<rect class="sdb-c {cls}" x="{cx}" y="{y}" width="{CELL}" height="{CELL}" rx="2" fill-opacity="{op:.3f}" '
                       f'data-task="{E(tid)}" data-row="{E(row["key"])}" data-tip="{tip}"/>')
        # right margin: totals
        tot = _totals(list(res.values()), len(cols))
        tx = grid_w + 12
        bs = f'{tot["brier"]:.3f}' if tot["brier"] is not None else "—"
        out.append(f'<text class="sdb-tot" x="{tx}" y="{y + CELL - 2}">'
                   f'<tspan class="sdb-num" data-count="{tot["correct"]}">{tot["correct"]}</tspan>'
                   f'<tspan class="sdb-dim">/{len(cols)}</tspan>'
                   f'<tspan class="sdb-dim" dx="10">brier </tspan><tspan>{bs}</tspan>'
                   f'<tspan class="sdb-dim" dx="10">strict </tspan><tspan>{tot["strict_valid"]}</tspan></text>')
        out.append("</g>")
    # bottom margin: how many models got each task (live/run rows only; falls back to all rows)
    src_n = per_task_n if per_task_n else Counter({tid: len(rows) for tid, _ in cols})
    src_c = Counter()
    for row in rows:
        if per_task_n and row["kind"] not in ("live", "run", "baseline"):
            continue
        for tid, r in row["results"].items():
            if r.get("correct"):
                src_c[tid] += 1
    fy = HEAD_H + len(rows) * (CELL + GAP) + 8
    out.append(f'<text class="sdb-fam" x="{LABEL_W - 10}" y="{fy + 22}" text-anchor="end">models right / task</text>')
    for tid, cx in cols:
        n = src_n.get(tid, 0)
        c = src_c.get(tid, 0)
        hgt = 0 if not n else max(1, round(28 * c / n))
        out.append(f'<rect class="sdb-foot" x="{cx}" y="{fy + 30 - hgt}" width="{CELL}" height="{hgt}" rx="1" data-task="{E(tid)}" data-tip="{E(tid)}|right for {c}/{n} models"/>')
    out.append("</svg>")
    return "".join(out)


def board_html(job: Job | None, labels: dict[str, str], show_history: bool = False, all_models: list[str] | None = None) -> str:
    """Board + status line for a run, with the published runs as italic rows (history optional)."""
    rows = _rows_from_job(job, labels, all_models or list(labels))
    pub = published_rows()
    rows += [r for r in pub if r["kind"] == "published"]
    hist = [r for r in pub if r["kind"] == "history"]
    if show_history:
        rows += hist
    status = _status_html(job)
    legend = ('<div class="sdb-legend"><span class="sdb-sw sdb-ok"></span>correct · shade = p(gold) '
              '<span class="sdb-sw sdb-bad"></span>wrong '
              '<span class="sdb-sw sdb-bad sdb-inv"></span>invalid / error '
              '<span class="sdb-sw sdb-none"></span>not attempted · '
              f'<span class="sdb-dim">{len(hist)} history runs hidden</span></div>')
    return (f'<div class="sdp sdb-wrap" data-live="{1 if job and job.running() else 0}">{status}'
            f'<div class="sdb-scroll">{board_svg(rows)}</div>{legend}'
            '<div class="sdb-tip" role="tooltip" hidden></div></div>')


def _status_html(job: Job | None) -> str:
    """Run status: progress, ETA, push outcome."""
    if not job:
        return '<div class="sdb-status"><span class="sdb-dim">No run on this Space yet; rows are the published runs.</span></div>'
    pct = 100.0 * job.done / max(1, job.total)
    if job.kind == "baseline":
        i = job.info
        txt = (f'baseline: {len(job.models)} models × {i.get("n_tasks", len(job.tids))} tasks · {E(i.get("date", "?"))} · '
               f'{E(i.get("hardware", "a100"))} {E(i.get("dtype", "bf16"))}'
               + (f' · {_fmt_s(float(i["wall_s"]))}' if i.get("wall_s") else ""))
        return (f'<div class="sdb-status"><div class="sdb-bar"><span style="width:{pct:.1f}%"></span></div>'
                f'<div class="sdb-status-t">{txt} <span class="sdb-dim">· run {E(job.run_id)} · {job.done}/{job.total} cells</span></div></div>')
    if job.running():
        eta = job.eta_s()
        eta_s = (" · all cells done, pushing to the dataset…" if job.done >= job.total
                 else (f" · about {_fmt_s(eta)} left" if eta else " · measuring latency…"))
        cur = f" · now {job.current[1]} on {job.current[0]}" if job.current else ""
        txt = f'<span class="sdb-live">live</span> {job.label}: {job.done}/{job.total} cells · {_fmt_s(job.elapsed())} elapsed{eta_s}{cur}'
    else:
        why = "cancelled" if job.cancel.is_set() else ("error: " + job.error if job.error else "finished")
        push = job.push.get("status", "")
        pushtxt = {"pushed": f'pushed to private dataset {E(job.push.get("repo"))} (readback ok)',
                   "skipped": f'not pushed ({E(job.push.get("why"))})',
                   "failed": f'push failed: {E(job.push.get("why"))}',
                   "refused": f'push refused: {E(job.push.get("why"))}'}.get(push, E(push))
        txt = f'{job.label}: {why} · {job.done}/{job.total} cells in {_fmt_s(job.elapsed())} · {pushtxt}'
    return (f'<div class="sdb-status"><div class="sdb-bar"><span style="width:{pct:.1f}%"></span></div>'
            f'<div class="sdb-status-t">{txt} <span class="sdb-dim">· run {E(job.run_id)}</span></div></div>')


def _fmt_s(s: float | None) -> str:
    """Seconds -> '42 s' / '3 min 10 s' / '1.5 h'."""
    if s is None:
        return "—"
    s = int(s)
    if s < 90:
        return f"{s} s"
    if s < 5400:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s / 3600:.1f} h"


# ---------------------------------------------------------------------------------
# task detail (click a column) + the gold card shown AFTER the models answered
# ---------------------------------------------------------------------------------
# The words the question-type control uses; the API names (noul/choice/score) stay in the data.
TYPE_WORDS = {"noul": "yes / no", "choice": "pick one", "score": "rate"}


def task_detail_html(tid: str, job: Job | None, labels: dict[str, str]) -> str:
    """One task in full: state, question, gold, every model's answer side by side."""
    ts = tasks()
    if tid not in ts:
        return ""
    t = ts[tid]
    rows = _rows_from_job(job, labels) + [r for r in published_rows() if r["kind"] == "published"]
    spec = jev.to_spec(t)
    opts = spec.get("options") or (["true", "false"] if spec["type"] == "noul" else [])
    head = (f'<div class="sdb-detail"><div class="sdb-dhead"><span class="sdb-chip">{E(tier(t))}</span>'
            f'<span class="sdb-chip">{E(t.family)}</span><span class="sdb-chip">{E(TYPE_WORDS.get(spec["type"], spec["type"]))}</span>'
            f'<code>{E(tid)}</code></div>'
            f'<p class="sdb-state">{E(state_text(t))}</p><p class="sdb-q"><b>{E(spec["instructions"])}</b>'
            + (f' <span class="sdb-dim">options: {E(", ".join(opts))}</span>' if opts else "") + "</p>"
            f'<p class="sdb-gold">gold: <b>{E(t.expected)}</b></p>')
    trs = []
    for r in rows:
        x = r["results"].get(tid)
        if not x:
            trs.append(f'<tr><td>{E(r["label"])}</td><td class="sdb-dim" colspan="4">not attempted</td></tr>')
            continue
        cls = "ok" if x.get("correct") else "bad"
        probs = x.get("probs") or {}
        ptxt = " ".join(f'{E(k)} {float(v):.2f}' for k, v in sorted(probs.items(), key=lambda kv: -float(kv[1]))[:5]) if probs else "—"
        trs.append(f'<tr class="sdb-{cls}"><td>{E(r["label"])} <span class="sdb-dim">{E({"published": "· published", "baseline": "· baseline"}.get(r["kind"], ""))}</span></td>'
                   f'<td><b>{E(x.get("predicted") if x.get("predicted") is not None else (x.get("error") or "—"))}</b></td>'
                   f'<td>{float(x.get("p_gold") or 0):.3f}</td><td class="sdb-dim">{ptxt}</td>'
                   f'<td>{round(x["latency_ms"]) if x.get("latency_ms") is not None else "—"}</td></tr>')
    table = ('<table class="sdb-table"><thead><tr><th>model</th><th>answered</th><th>p(gold)</th><th>distribution</th><th>latency ms</th></tr></thead>'
             f'<tbody>{"".join(trs)}</tbody></table></div>')
    return head + table


def gold_html(tid: str | None, result: dict | None) -> str:
    """Rendered under the playground result ONLY after the models answered a JevBench task."""
    if not tid or not result or not result.get("cards"):
        return ""
    t = tasks().get(tid)
    if not t:
        return ""
    rows = []
    for c in result["cards"]:
        r = score_card(c, tid)
        cls = "ok" if r["correct"] else "bad"
        rows.append(f'<tr class="sdb-{cls}"><td>{E(c.get("name"))}</td><td><b>{E(r["predicted"] if r["predicted"] is not None else (r["error"] or "—"))}</b></td>'
                    f'<td>{r["p_gold"]:.3f}</td><td>{r["brier"]:.3f}</td><td>{"yes" if r["strict_valid"] else "no"}</td></tr>')
    n_ok = sum(1 for c in result["cards"] if score_card(c, tid)["correct"])
    return (f'<div class="sdp sdb-goldcard"><div class="sdb-dhead"><span class="sdb-chip">JevBench {E(tid)}</span>'
            f'<span class="sdb-chip">{E(tier(t))}</span><span class="sdb-chip">{E(t.family)}</span>'
            f'<span class="sdb-gold">gold: <b>{E(t.expected)}</b> · {n_ok}/{len(result["cards"])} models right</span></div>'
            '<table class="sdb-table"><thead><tr><th>model</th><th>answered</th><th>p(gold)</th><th>Brier</th><th>strict valid</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table><div class="sdb-dim">scored by JevBench\'s own score_task (pinned 1bcc55eb); the gold answer was never sent to the models.</div></div>')


# ---------------------------------------------------------------------------------
# common errors: only patterns we can COUNT
# ---------------------------------------------------------------------------------
def _tlines(lines: list[str], show: int = 12) -> str:
    """A task list that stays short: the first `show` lines, the rest behind a 'show all N' disclosure."""
    if len(lines) <= show:
        return "".join(lines)
    return ("".join(lines[:show])
            + f'<details class="sdb-moretasks"><summary>show all {len(lines)}</summary>{"".join(lines[show:])}</details>')


def errors_html(job: Job | None, labels: dict[str, str], include_published: bool = True) -> str:
    """Common-error panels: per model, per task, models x tiers, and the paired McNemar test."""
    rows = _rows_from_job(job, labels)
    if not rows:
        rows = [r for r in published_rows() if r["kind"] == "published"]
        note = "from the 7 published runs"
    elif job and job.kind == "baseline":
        note = f"from run {job.run_id}"
    else:
        note = f"from run {job.run_id if job else ''}"
    rows = [r for r in rows if r["results"]]
    if not rows:
        return '<div class="sdp sdb-dim">No finished results yet.</div>'
    ts = tasks()
    ids = [tid for tid in order() if all(tid in r["results"] for r in rows)]  # tasks every row attempted
    out = [f'<div class="sdp sdb-errors"><div class="sdb-dim">{E(note)} · {len(ids)} tasks</div>']

    # (a) per model
    out.append('<div class="sdb-grid2">')
    for r in rows:
        res = r["results"]
        missed = [tid for tid in ids if not res[tid].get("correct")]
        by_fam = Counter(ts[tid].family for tid in missed)
        pat = Counter()
        for tid in missed:
            t, x = ts[tid], res[tid]
            pred = x.get("predicted")
            if not x.get("valid", True) or x.get("error"):
                pat["invalid distribution / error"] += 1
            elif t.question["type"] == "noul":
                pat[f'says {pred} when gold is {t.expected}'] += 1
            elif t.question["type"] == "choice" and pred in t.labels:
                pat[f'picks option #{t.labels.index(pred) + 1} of {len(t.labels)}'] += 1
            elif t.question["type"] == "score" and pred is not None:
                try:
                    d = int(str(pred)) - int(str(t.expected))
                    pat[f'{"over" if d > 0 else "under"}-scores by {abs(d)}'] += 1
                except ValueError:
                    pat["off-scale answer"] += 1
        fam_txt = " · ".join(f'{E(f)} <b>{n}</b>' for f, n in by_fam.most_common(6)) or "—"
        pat_txt = "".join(f'<li>{E(k)} <b>{n}</b></li>' for k, n in pat.most_common(4)) or "<li>—</li>"
        out.append(f'<div class="sdb-card"><div class="sdb-ct">{E(r["label"])} <span class="sdb-dim">{len(missed)} missed</span></div>'
                   f'<div class="sdb-dim">by category</div><div>{fam_txt}</div>'
                   f'<div class="sdb-dim">most frequent wrong pattern</div><ul>{pat_txt}</ul></div>')
    out.append("</div>")

    # (b) per task — only meaningful across models: with < 3 rows "hard for everyone" and "one model knows" degenerate
    # into "this model missed it" / "this model got it", so they are not computed (counted facts only, no padding).
    n = len(rows)
    hard = [(tid, sum(1 for r in rows if not r["results"][tid].get("correct"))) for tid in ids]
    thr = max(1, math.ceil(n * 5 / 7))
    if n >= 3:
        hard_all = [(tid, k) for tid, k in hard if k >= thr]
        one = [(tid, [r["label"] for r in rows if r["results"][tid].get("correct")][0]) for tid, k in hard if k == n - 1]
    else:
        hard_all, one = [], []
        out.append(f'<div class="sdb-dim">per-task panels need ≥ 3 models in the run (this one has {n}); the per-model panel above still counts.</div>')
    out.append(f'<div class="sdb-grid2"><div class="sdb-card"><div class="sdb-ct">hard for everyone <span class="sdb-dim">missed by ≥ {thr} of {n}</span></div>'
               + _tlines([f'<div class="sdb-tline" data-task="{E(tid)}"><code>{E(tid)}</code> <span class="sdb-dim">{E(ts[tid].family)} · {k}/{n} wrong</span></div>'
                          for tid, k in sorted(hard_all, key=lambda x: -x[1])[:40]])
               + (f'<div class="sdb-dim">{len(hard_all)} tasks</div>' if hard_all else '<div class="sdb-dim">none</div>') + "</div>")
    out.append(f'<div class="sdb-card"><div class="sdb-ct">one model knows this <span class="sdb-dim">exactly 1 of {n} right</span></div>'
               + _tlines([f'<div class="sdb-tline" data-task="{E(tid)}"><code>{E(tid)}</code> <span class="sdb-dim">{E(ts[tid].family)} ·</span> <b>{E(m)}</b></div>'
                          for tid, m in one[:40]])
               + (f'<div class="sdb-dim">{len(one)} tasks</div>' if one else '<div class="sdb-dim">none</div>') + "</div></div>")

    # models x tiers
    out.append('<div class="sdb-ct">right answers per difficulty tier <span class="sdb-dim">· Brier: 0 is perfect, 1 is worst · '
               'strict: answers whose probabilities summed to 1 within 0.001 (JevBench\'s pre-registered validity rule)</span></div>')
    out.append('<table class="sdb-table"><thead><tr><th>model</th>' + "".join(f'<th>{t}</th>' for t in TIERS) + '<th>total</th><th>Brier</th><th>strict</th></tr></thead><tbody>')
    for r in rows:
        res = r["results"]
        cells = []
        for tr in TIERS:
            tids = [tid for tid in ids if tier(ts[tid]) == tr]
            c = sum(1 for tid in tids if res[tid].get("correct"))
            cells.append(f'<td>{c}/{len(tids)}</td>')
        tot = _totals([res[tid] for tid in ids], len(ids))
        bs = f'{tot["brier"]:.3f}' if tot["brier"] is not None else "—"
        out.append(f'<tr><td>{E(r["label"])}</td>{"".join(cells)}<td><b>{tot["correct"]}/{len(ids)}</b></td><td>{bs}</td><td>{tot["strict_valid"]}/{len(ids)}</td></tr>')
    out.append("</tbody></table>")

    # pairwise McNemar (b = A right & B wrong, c = A wrong & B right; exact two-sided p from paired_upstream)
    if n < 2:
        out.append('<div class="sdb-dim">pairwise McNemar needs ≥ 2 models in the run.</div></div>')
        return "".join(out)
    out.append('<div class="sdb-ct">pairwise: row right, column wrong (b) / row wrong, column right (c) · exact McNemar p</div>')
    out.append('<table class="sdb-table sdb-pair"><thead><tr><th></th>' + "".join(f'<th>{E(r["label"])}</th>' for r in rows) + '</tr></thead><tbody>')
    for a in rows:
        cells = []
        for b in rows:
            if a is b:
                cells.append('<td class="sdb-dim">·</td>')
                continue
            bb = sum(1 for tid in ids if a["results"][tid].get("correct") and not b["results"][tid].get("correct"))
            cc = sum(1 for tid in ids if not a["results"][tid].get("correct") and b["results"][tid].get("correct"))
            p = mcnemar_exact(bb, cc)
            sig = " sdb-sig" if p < 0.05 else ""
            cells.append(f'<td class="{sig.strip()}"><b>{bb}</b>/{cc}<br><span class="sdb-dim">p={p:.3g}</span></td>')
        out.append(f'<tr><td>{E(a["label"])}</td>{"".join(cells)}</tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


# ---------------------------------------------------------------------------------
# picker + static
# ---------------------------------------------------------------------------------
def picker_html() -> str:
    """The task picker shell; the task list rides along as escaped JSON for static/bench.js."""
    # HTML-escaped JSON in a hidden div (textContent unescapes it); a <script type=json> would be
    # flagged/stripped by gr.HTML's sanitiser.
    data = E(json.dumps(picker_payload(), ensure_ascii=False))
    tiers = "".join(f'<button type="button" class="sdb-f" data-tier="{t}">{t}</button>' for t in TIERS)
    fams = sorted({t.family for t in tasks().values()})
    fam_opts = "".join(f'<option value="{E(f)}">{E(f)}</option>' for f in fams)
    return (f'<div class="sdp sdb-picker" id="sdb-picker"><div id="sdb-tasks" hidden>{data}</div>'
            '<div class="sdb-prow"><input class="sdb-search" type="search" placeholder="search tasks" aria-label="search tasks">'
            f'<div class="sdb-filters"><button type="button" class="sdb-f on" data-tier="">all</button>{tiers}</div>'
            f'<select class="sdb-fam" aria-label="category"><option value="">every category</option>{fam_opts}</select>'
            '<span class="sdb-count sdb-dim"></span></div>'
            '<div class="sdb-list" role="listbox" aria-label="JevBench tasks"></div>'
            '<div class="sdb-dim sdb-note">JevBench · 231 tasks · MIT · pinned 1bcc55eb</div></div>')


def css() -> str:
    """static/bench.css for app.py's launch(css=...)."""
    return (STATIC / "bench.css").read_text(encoding="utf-8")


def js() -> str:
    """static/bench.js for app.py's launch(js=...)."""
    return (STATIC / "bench.js").read_text(encoding="utf-8")
