"""runner.py -- loads deciders, answers questions, measures the cost.

OWNED BY LANE APP. Imports `metrics.py` (owned by LANE METRICS) and never edits it.

Two hard constraints from the BRIEF, both enforced here:

  * MPS CRASHES ON CONCURRENT REQUESTS (Metal command encoder). Every model call --
    load AND forward -- goes through the single module-level `_LOCK`. Gradio serves
    requests on a thread pool, so without this the second user crashes the Space.

  * 26B-A4B bf16 CANNOT CO-RESIDE with the others -- RESEARCH measured that it cannot be
    resident on 48 GB AT ALL (48.067 GiB of weights vs 48 GiB of raw VRAM). So admission and
    eviction are BASE-KEYED and live in cache.py (DESIGN section 4); load time is reported in
    its own field, never added to the forward latency, so a cold card does not look slow.
"""

from __future__ import annotations

import base64
import gc
import io
import os
import threading
import time
from typing import Any

import torch

import metrics as M
from registry import ModelInfo

# ---- the one lock -----------------------------------------------------------------
# DESIGN 4e: held for the whole load+forward. Coarse on purpose -- a per-model lock would
# still let two different models drive two Metal encoders at once, which is the crash.
from cache import INFER_LOCK as _LOCK
from cache import ModelCache, ModelTooLargeError

_CACHE_OBJ: ModelCache | None = None


def cache() -> ModelCache:
    """The one process-wide base-keyed LRU (DESIGN section 4)."""
    global _CACHE_OBJ
    if _CACHE_OBJ is None:
        _CACHE_OBJ = ModelCache(device=default_device())
    return _CACHE_OBJ


def default_device() -> str:
    if os.environ.get("DECIDER_DEVICE"):
        return os.environ["DECIDER_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cache_state() -> list[str]:
    return cache().resident()


def evict_all() -> None:
    with _LOCK:
        cache().clear()


# ---- questions --------------------------------------------------------------------


def build_questions(specs: list[dict]) -> dict[str, Any]:
    """specs: [{id?, type, instructions, options}] -> the SystemOneRequest questions dict.

    The real schema field is `criteria`, NOT `options` (schema.py; COORDINATION 01:18Z):
      noul   -> criteria optional {"true"|"false": desc}
      choice -> criteria {name: desc}  (>=2)
      score  -> criteria [level, ...]  (2..10, ascending)
    """
    from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

    out: dict[str, Any] = {}
    counts = {"noul": 0, "choice": 0, "score": 0}
    for spec in specs:
        qtype = (spec.get("type") or "noul").strip().lower()
        instr = (spec.get("instructions") or "").strip()
        if not instr:
            continue
        raw_opts = spec.get("options")
        if isinstance(raw_opts, str):
            opts = [o.strip() for o in raw_opts.split(",") if o.strip()]
        else:
            opts = [str(o).strip() for o in (raw_opts or []) if str(o).strip()]
        qid = (spec.get("id") or "").strip() or f"{qtype}_{counts.get(qtype, 0)}"
        counts[qtype] = counts.get(qtype, 0) + 1
        if qtype == "noul":
            out[qid] = NoulQuestion(instructions=instr)
        elif qtype == "choice":
            if len(opts) < 2:
                raise ValueError(f"question {qid!r}: choice needs at least 2 options, got {opts}")
            out[qid] = ChoiceQuestion(instructions=instr, criteria={o: "" for o in opts})
        elif qtype == "score":
            if not (2 <= len(opts) <= 10):
                raise ValueError(
                    f"question {qid!r}: score needs 2..10 ascending levels, got {opts}"
                )
            out[qid] = ScoreQuestion(instructions=instr, criteria=opts)
        else:
            raise ValueError(f"question {qid!r}: unknown type {qtype!r}")
    if not out:
        raise ValueError("add at least one question with instructions")
    return out


def _encode_image(image: Any) -> str | None:
    """PIL image / path / bytes -> base64 ascii, as SystemOneRequest.images wants."""
    if image is None:
        return None
    if isinstance(image, (bytes, bytearray)):
        return base64.b64encode(bytes(image)).decode("ascii")
    if isinstance(image, str):
        with open(image, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---- answer normalisation ---------------------------------------------------------


def normalise_answer(ans: dict) -> dict:
    """One shape for the table/chart regardless of question type.

    `value` is the headline, `probs` is {label: p} for the bar chart, `confidence` may be
    None -- a noul answer HAS no confidence field by design (schema.py: with two outcomes
    the probability IS the uncertainty), so we must not invent one.
    """
    t = ans.get("type")
    if t == "noul":
        p = ans.get("noul")
        return {
            "type": "noul",
            "display": f"{p:.4f}" if isinstance(p, (int, float)) else str(p),
            "value": p,
            "probs": {"true": p, "false": (1.0 - p) if isinstance(p, (int, float)) else None},
            "confidence": None,  # by design, not missing
            "vote": "true" if isinstance(p, (int, float)) and p >= 0.5 else "false",
        }
    if t == "choice":
        return {
            "type": "choice",
            "display": str(ans.get("choice")),
            "value": ans.get("choice"),
            "probs": dict(ans.get("probabilities") or {}),
            "confidence": ans.get("confidence"),
            "vote": str(ans.get("choice")),
        }
    if t == "score":
        legend = ans.get("legend") or {}
        probs = {legend.get(k, k): v for k, v in (ans.get("probabilities") or {}).items()}
        sc = ans.get("score")
        top = max((ans.get("probabilities") or {}).items(), key=lambda kv: kv[1], default=(None, 0))
        return {
            "type": "score",
            "display": f"{sc:.4f}" if isinstance(sc, (int, float)) else str(sc),
            "value": sc,
            "probs": probs,
            "confidence": ans.get("confidence"),
            "legend": legend,
            "vote": legend.get(top[0], top[0]) if top[0] is not None else None,
        }
    return {"type": str(t), "display": str(ans), "value": None, "probs": {},
            "confidence": None, "vote": None}


# ---- the one call that does everything -------------------------------------------


def run_one(
    info: ModelInfo,
    state: str,
    questions: dict[str, Any],
    *,
    image: Any = None,
    device: str | None = None,
    usd_per_hour: Any = None,
    active_params: Any = None,
) -> dict:
    """Load (or reuse) one decider, answer `questions`, and measure the cost.

    Serialized behind _LOCK. Returns a card dict -- never raises for a model-level
    failure: the card carries `error` so one bad model cannot blank the whole page.
    """
    device = device or default_device()
    want_vision = image is not None
    card: dict[str, Any] = {
        "repo": info.repo,
        "name": info.name,
        "device": device,
        "vision_requested": want_vision,
        "vision_capable": info.vision,
        "base_model": info.base_model,
        "answers": {},
        "metrics": {},
        "error": None,
        "skipped": None,
        "cold": False,
    }

    if want_vision and not info.vision:
        card["skipped"] = (
            f"image ignored: {info.name} reads text only (base {info.base_model}; Gemma 4 "
            "checkpoints drop the base vision encoder -- docs/vision.md). Re-run without the "
            "image, or pick a Qwen3.5 decider."
        )
        card["error"] = card["skipped"]
        return card

    b64 = _encode_image(image) if want_vision else None

    t_wait = time.perf_counter()
    with _LOCK:
        card["lock_wait_s"] = round(time.perf_counter() - t_wait, 3)
        try:
            mode = "vision" if want_vision else "text"
            c = cache()
            engine, rep = c.get_detailed(info.repo, mode)
            card["cold"] = not rep.was_cached
            card["load"] = {
                "load_seconds": rep.load_seconds,
                "was_cached": rep.was_cached,
                "evicted": list(rep.evicted),
                "exclusive": rep.exclusive,
                "base_model": rep.base_model,
                "base_gib": round(rep.base_bytes / 2**30, 3),
                "resident_gib_after": round(rep.resident_bytes_after / 2**30, 3),
                "budget_gib": round(rep.budget_bytes / 2**30, 3),
                "shared_base_note": rep.shared_base_note,
            }
            entry = {"baseline": c.baseline_of(info.repo, mode)}

            def _call() -> Any:
                if want_vision:
                    return engine.ask_images(state, questions, [b64])
                return engine.ask(state, questions)

            text_for_tokens = [state] + [
                str(getattr(q, "instructions", "")) for q in questions.values()
            ]
            m = M.measure(
                _call,
                device=device,
                dtype=info.torch_dtype,
                repo=info.repo,
                baseline=entry["baseline"],
                model=getattr(engine, "model", None),
                tokenizer=getattr(engine, "tok", None),
                text=text_for_tokens,
                usd_per_hour=usd_per_hour,
                active_params=active_params,
            )
            resp = m.pop("result", None)
            card["metrics"] = m
            if resp is not None:
                payload = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
                card["usage"] = payload.get("usage")
                card["answers"] = {
                    qid: normalise_answer(a if isinstance(a, dict) else a.model_dump())
                    for qid, a in (payload.get("answers") or {}).items()
                }
            if not m.get("ok", False) and m.get("error"):
                card["error"] = str(m["error"])
        except Exception as e:  # noqa: BLE001 -- one model must not blank the page
            card["error"] = f"{type(e).__name__}: {e}"
    return card


# ---- cross-model comparison -------------------------------------------------------


def comparison(cards: list[dict], question_ids: list[str]) -> dict:
    """Rows = models, cols = questions, plus the AGREEMENT row.

    `spread` is max-min of the WINNING option's probability across models, and
    `dissenters` are the models whose vote differs from the majority vote. A question
    where every model failed reports nothing rather than a fake consensus.
    """
    rows = []
    for c in cards:
        row = {"model": c["name"], "error": c.get("error")}
        for qid in question_ids:
            a = (c.get("answers") or {}).get(qid)
            row[qid] = a["display"] if a else ("ERROR" if c.get("error") else "-")
        rows.append(row)

    agreement: dict[str, Any] = {}
    for qid in question_ids:
        votes: dict[str, list[str]] = {}
        winning_p: list[float] = []
        for c in cards:
            a = (c.get("answers") or {}).get(qid)
            if not a:
                continue
            v = a.get("vote")
            if v is None:
                continue
            votes.setdefault(str(v), []).append(c["name"])
            probs = {k: p for k, p in (a.get("probs") or {}).items()
                     if isinstance(p, (int, float))}
            if probs:
                winning_p.append(max(probs.values()))
        n = sum(len(v) for v in votes.values())
        if not n:
            agreement[qid] = {"n": 0, "note": "no model answered"}
            continue
        majority, holders = max(votes.items(), key=lambda kv: len(kv[1]))
        dissenters = [m for v, ms in votes.items() if v != majority for m in ms]
        agreement[qid] = {
            "n": n,
            "majority": majority,
            "majority_n": len(holders),
            "unanimous": len(votes) == 1,
            "dissenters": dissenters,
            "distinct_votes": len(votes),
            "votes": {v: ms for v, ms in votes.items()},
            "prob_spread": (round(max(winning_p) - min(winning_p), 4) if len(winning_p) > 1 else 0.0),
            "prob_min": round(min(winning_p), 4) if winning_p else None,
            "prob_max": round(max(winning_p), 4) if winning_p else None,
        }
    return {"rows": rows, "agreement": agreement}
