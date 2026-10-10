#!/usr/bin/env python3
"""GATE: reproduce published JevBench totals from the repos' own per-task records using the
PINNED JevBench scoring code (jevbench.scoring.score_task) -- before any model is run.

Measured 2026-10-10 on the pinned checkout 1bcc55eb: the published results.jsonl files store
choice distributions with ONE option key scrubbed on 13 five-option tasks (e.g. billing_question,
get_stock_price, ledger_reconciliation) and one record fully empty (hard-opus-c-temporal_numeric-04).
A decider's distribution sums to 1, so the single missing mass is reconstructed as 1 - sum(rest);
the empty record is kept as JevBench scored it (invalid -> wrong, Brier 1.0). With that, every
model reproduces its README/summary.json n_correct EXACTLY with 0 per-task mismatches.

    python bench/parity.py            -> prints one line per published run, exits 1 on any mismatch
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bench.jev import load_tasks, score_probs  # noqa: E402

PUBLISHED = {
    "strands-decider-2B-qwen3.5-v1-2610": 180,
    "strands-decider-2B-hobson-v19": 167,
    "strands-decider-26B-A4B-gemma4-v1-2610": 207,
}


def reconstruct(probs: dict, labels: list) -> tuple[dict, str]:
    miss = [k for k in labels if k not in (probs or {})]
    if not miss:
        return probs, "complete"
    if len(miss) == 1 and probs:
        p = dict(probs)
        p[miss[0]] = round(max(0.0, 1.0 - sum(probs.values())), 4)
        return p, "reconstructed_1_key"
    return probs, f"unrecoverable_{len(miss)}_missing"


def main() -> int:
    tasks = load_tasks()
    assert len(tasks) == 231, len(tasks)
    bad = 0
    for name, expect in PUBLISHED.items():
        d = HERE / "published" / name
        recs = [json.loads(l) for l in (d / "results.jsonl").open() if l.strip()]
        summ = json.loads((d / "summary.json").read_text())
        n_ok = mism = 0
        briers = []
        how_count: dict[str, int] = {}
        for r in recs:
            t = tasks[r["task_id"]]
            p, how = reconstruct(r["probs_as_returned"], t.labels)
            how_count[how] = how_count.get(how, 0) + 1
            sc = score_probs(p, t)
            mism += int(bool(sc["correct"]) != bool(r["correct"]))
            n_ok += int(bool(sc["correct"]))
            briers.append(sc["brier"])
        brier = sum(briers) / len(briers)
        # Brier is NOT gated to 1e-3: 13 reconstructed keys are rounded to 4 dp, so it moves by <=0.0015.
        ok = n_ok == expect == summ["n_correct"] and mism == 0 and abs(brier - summ["brier_mean"]) < 2e-3
        bad += int(not ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}: rescored {n_ok}/231 readme {expect} summary {summ['n_correct']} "
              f"per-task mismatches {mism} brier {brier:.4f} (summary {summ['brier_mean']:.4f}, delta {brier - summ['brier_mean']:+.4f}) {how_count}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
