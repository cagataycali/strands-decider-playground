"""JevBench tasks + scoring for the playground. Scoring is DELEGATED to the pinned JevBench
checkout (pip install --no-deps -e; see requirements) -- never re-implemented.

Task -> app.answer()-shaped request: ONE question with id "decision", the task's exact
`instructions` and `criteria` (noul criteria {true,false}; choice criteria {option: desc};
score criteria = ordered level descriptions). Labels and the gold answer never enter the request.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from jevbench.metrics import brier_score, ece_top_label  # pinned checkout
from jevbench.scoring import score_task
from jevbench.tasks import Task

HERE = Path(__file__).resolve().parent
TASK_FILES = ("original", "easy", "hard")
TIER_LABEL = {"standard": "standard", "easy": "easy", "hard": "hard"}


def load_tasks() -> dict[str, Task]:
    """231 public tasks at the pinned commit, with `tier` attached from strands-decider's CSV."""
    pin = json.loads((HERE / "tasks" / "PIN.json").read_text())
    tasks: dict[str, Task] = {}
    for f in TASK_FILES:
        for line in (HERE / "tasks" / f"{f}.jsonl").open(encoding="utf-8"):
            if line.strip():
                t = Task.from_dict(json.loads(line))
                t.validate()
                tasks[t.id] = t
    if len(tasks) != pin["count"]:
        raise RuntimeError(f"JevBench task count {len(tasks)} != pinned {pin['count']}")
    tiers = {r["task_id"]: r["tier"] for r in csv.DictReader((HERE / "published" / "jevbench_tasks.csv").open())}
    if set(tiers) != set(tasks):
        raise RuntimeError("task ids differ from strands-decider research/data/jevbench_tasks.csv")
    for tid, t in tasks.items():
        t.provenance = dict(t.provenance or {})
        t.provenance["tier"] = tiers[tid]
    return tasks


def ordered_ids(tasks: dict[str, Task]) -> list[str]:
    """Board column order: tier (easy, standard, hard) then family then id."""
    rank = {"easy": 0, "standard": 1, "hard": 2}
    return sorted(tasks, key=lambda i: (rank[tasks[i].provenance["tier"]], tasks[i].family, i))


def to_spec(t: Task) -> dict:
    """The question spec app.answer() takes -- mirrors JevBench's typesafe adapter build_question()."""
    q = t.question
    spec: dict[str, Any] = {"id": "decision", "type": q["type"], "instructions": q["instructions"]}
    crit = q.get("criteria")
    if q["type"] == "choice":
        # criteria map option -> description; keep the task's label order.
        if isinstance(crit, dict):
            spec["criteria"] = {k: str(crit.get(k, "")) for k in t.labels}
        else:
            spec["criteria"] = {k: "" for k in t.labels}
        spec["options"] = list(t.labels)
    elif q["type"] == "score":
        # labels are level indices "0".."L-1"; criteria is the ordered level description list.
        if isinstance(crit, list):
            spec["criteria"] = [str(c) for c in crit]
        elif isinstance(crit, dict):
            spec["criteria"] = [str(crit.get(k, k)) for k in t.labels]
        else:
            spec["criteria"] = [str(k) for k in t.labels]
        spec["options"] = list(spec["criteria"])
    else:  # noul
        if isinstance(crit, dict):
            spec["criteria"] = {"true": str(crit.get("true", "")), "false": str(crit.get("false", ""))}
    return spec


def probs_from_card_answer(ans: dict, t: Task) -> dict | None:
    """runner.normalise_answer() output -> JevBench exact-label distribution (typesafe adapter rules):
    noul value P(true) -> {yes, no}; choice probs as-is; score probs re-keyed to level indices."""
    if not ans:
        return None
    typ = ans.get("type")
    if typ != t.question["type"]:
        return None
    if typ == "noul":
        p = ans.get("value")
        if not isinstance(p, (int, float)):
            return None
        return {"yes": float(p), "no": 1.0 - float(p)}
    if typ == "choice":
        return dict(ans.get("probs") or {})
    if typ == "score":
        legend = ans.get("legend") or {}
        inv = {v: k for k, v in legend.items()}
        out = {}
        for k, v in (ans.get("probs") or {}).items():
            out[str(inv.get(k, k))] = v
        return out
    return None


def score_probs(probs: dict | None, t: Task) -> dict:
    """JevBench's score_task + Brier (1.0 for an invalid distribution, as JevBench's summarize does)
    + the probability put on the gold label (board opacity)."""
    sc = score_task(probs or {}, t)
    exp = str(t.expected)
    if sc.get("valid"):
        sc["brier"] = brier_score(sc["probs"], exp, t.labels)
        sc["p_gold"] = float(sc["probs"].get(exp, 0.0))
        sc["top_p"] = max(sc["probs"].values())
    else:
        sc["brier"] = 1.0
        sc["p_gold"] = 0.0
        sc["top_p"] = 0.0
    sc["expected"] = exp
    return sc


def ece(pairs: list[tuple[float, bool]]) -> float:
    return float(ece_top_label(pairs)["ece"]) if pairs else 0.0
