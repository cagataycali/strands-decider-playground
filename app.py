"""Strands Decider Playground -- the Gradio app.

One decision question goes to every public StrandsAgents decider; each answers side by side
with a probability per option, a calibrated confidence, the model's own published evals and
the measured compute cost of the call.

How the pieces fit:
  * `registry.py` reads the model list live from the Hub at start (public repos under the org
    prefix -- there is no hardcoded list), plus each repo's config and `eval/**/summary.json`.
  * `runner.py` loads a model through `cache.py`, answers the questions and measures the call
    with `metrics.py`.
  * `answer()` below is the headless entry point: the same dict the UI renders, no browser needed.
  * Everything after the "rendering" line turns that dict into HTML; `static/app.js` animates
    it after insertion. `landing.py`, `examples.py` and `viz.py` render their own sections
    from the same dict.

Headless use:
    from app import answer
    answer(state, image, question_specs, model_names) -> dict
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any

# Set BEFORE metrics is imported: its spin-sampler at the default interval inflates CPU
# latency several-fold (measured 3.71x). 0.01 s removes the inflation; the cost is peak-memory
# resolution, so the poll interval is printed on every card.
os.environ.setdefault("METRICS_SAMPLE_S", "0.01")

import gradio as gr  # noqa: E402

import metrics as M  # noqa: E402
import runner  # noqa: E402
from registry import ModelInfo, active_params, load_registry  # noqa: E402

# Hourly prices from huggingface.co/pricing, used only to price a measured latency.
HARDWARE: dict[str, float] = {
    "CPU Basic — free": 0.0,
    "ZeroGPU (RTX PRO 6000 Blackwell, ≤96 GB) — free": 0.0,
    "CPU Upgrade (8 vCPU/32 GB) — $0.03/h": 0.03,
    "Nvidia T4 small (16 GB) — $0.40/h": 0.40,
    "1x Nvidia L4 (24 GB) — $0.80/h": 0.80,
    "1x Nvidia L40S (48 GB) — $1.80/h": 1.80,
    "Nvidia A100 large (80 GB) — $2.50/h": 2.50,
    "4x Nvidia L40S (192 GB) — $8.30/h": 8.30,
}

# Cached per base: one Hub round trip each, not per request.
_ACTIVE: dict[str, dict] = {}


def active_for(info: ModelInfo) -> dict:
    """Active-parameter record for a model's base, fetched once per base."""
    if info.base_model not in _ACTIVE:
        _ACTIVE[info.base_model] = active_params(info.base_model)
    return _ACTIVE[info.base_model]

MAX_QUESTIONS = 8
GOLDEN_STATE = "Help! My payouts have been failing for 3 days!"

# ---------------------------------------------------------------------------------
# registry, read live at import
# ---------------------------------------------------------------------------------

REGISTRY: list[ModelInfo] = []
REGISTRY_ERROR: str | None = None


def load_models() -> list[ModelInfo]:
    """Read the registry (public repos + evals) into REGISTRY; a failure lands in REGISTRY_ERROR."""
    global REGISTRY, REGISTRY_ERROR
    try:
        REGISTRY = load_registry(with_evals=True)
        REGISTRY_ERROR = None
    except Exception as e:  # noqa: BLE001
        REGISTRY, REGISTRY_ERROR = [], f"{type(e).__name__}: {e}"
    return REGISTRY


def by_name(name: str) -> ModelInfo | None:
    """Look a model up by short name or full repo id."""
    return next((m for m in REGISTRY if m.name == name or m.repo == name), None)


def model_choices() -> list[str]:
    """Short names of every listed model, in registry order."""
    return [m.name for m in REGISTRY]


def vision_models() -> list[str]:
    """Short names of the models that can read an image."""
    return [m.name for m in REGISTRY if m.vision]


# ---------------------------------------------------------------------------------
# the core callable (headless-testable)
# ---------------------------------------------------------------------------------


def answer(
    state: str,
    image: Any,
    question_specs: list[dict],
    model_names: list[str],
    *,
    usd_per_hour: float | None = None,
) -> dict:
    """Run every selected model on the same state+questions. Never raises for one bad model."""
    if not REGISTRY:
        load_models()
    specs = [s for s in (question_specs or []) if (s.get("instructions") or "").strip()]
    if not specs:
        return {"error": "Add at least one question (an instruction is required).", "cards": []}
    try:
        questions = runner.build_questions(specs)
    except ValueError as e:
        return {"error": str(e), "cards": []}

    qids = list(questions.keys())
    chosen = [by_name(n) for n in (model_names or [])]
    chosen = [m for m in chosen if m is not None]
    if not chosen:
        return {"error": "Select at least one model.", "cards": []}

    # A choice question cannot exceed the checkpoint's pointer slots.
    for m in chosen:
        if m.num_slots:
            for qid, q in questions.items():
                crit = getattr(q, "criteria", None)
                if isinstance(crit, dict) and len(crit) > m.num_slots:
                    return {
                        "error": f"question {qid!r} has {len(crit)} options but "
                                 f"{m.name} has only {m.num_slots} pointer slots.",
                        "cards": [],
                    }

    cards: list[dict] = []
    t0 = time.perf_counter()
    for m in chosen:
        try:
            ap = active_for(m)
            card = runner.run_one(
                m, state or "", questions, image=image,
                usd_per_hour=usd_per_hour,
                active_params=ap.get("active"),
            )
            card["active_params"] = ap
            cards.append(card)
        except Exception as e:  # noqa: BLE001
            cards.append({
                "repo": m.repo, "name": m.name, "error": f"{type(e).__name__}: {e}",
                "answers": {}, "metrics": {}, "traceback": traceback.format_exc()[-1500:],
            })
    wall = time.perf_counter() - t0
    cmp = runner.comparison(cards, qids)
    return {
        "error": None,
        "cards": cards,
        "question_ids": qids,
        "question_specs": specs,
        "comparison": cmp,
        "wall_seconds": round(wall, 3),
        "device": runner.default_device(),
        "sample_interval_s": os.environ.get("METRICS_SAMPLE_S"),
        "budget": runner.cache().describe_budget(),
    }


# ---------------------------------------------------------------------------------
# rendering. The server renders static HTML; static/app.js animates it after insertion.
# Every number shown is read from the result dict that `answer()` returned -- nothing is
# recomputed here.
# ---------------------------------------------------------------------------------

import html as _html  # noqa: E402
import re  # noqa: E402
import base64  # noqa: E402
import platform  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import transformers  # noqa: E402

from registry import HIDDEN_PRIVATE, PUBLIC_ONLY  # noqa: E402

# Section renderers: each is pure (dict -> HTML) and ships its own static/*.css + *.js.
import landing  # noqa: E402  hero, proof row, how-to, model row, footer
import examples as EX  # noqa: E402  the clickable example gallery
import viz  # noqa: E402  animated SVG decision diagrams
import bench  # noqa: E402  JevBench: picker, runner, board, errors (bench/ package)

STATIC = Path(__file__).parent / "static"


def _static(name: str) -> str:
    """Text of a file under static/."""
    return (STATIC / name).read_text(encoding="utf-8")


def _svg(name: str, cls: str, extra: str = "") -> str:
    """Inline an SVG from static/ with a class and aria-hidden (the frog mark and the wordmark
    come from the strands-labs/robots docs)."""
    raw = _static(name)
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S).strip()
    return re.sub(r"<svg\b", f'<svg class="{cls}" aria-hidden="true" {extra}', raw, count=1)


MARK_SVG = _svg("mark.svg", "mark")
WORDMARK_SVG = _svg("wordmark.svg", "wordmark")


TIPS = {
    "activation": "Peak memory during the forward pass minus the memory right after the model loaded: what the call itself needed on top of the weights.",
    "weight": "Bytes of model weights resident on the device after loading (base torso + LoRA adapter + head).",
    "baseline": "Device memory in use right after load and before the first forward: weights plus allocator overhead.",
    "peak": "Highest device memory observed during the forward pass.",
    "latency": "Wall time of the forward pass alone. The load time is reported separately on purpose: a cold load is tens of seconds, a forward is under a second.",
    "load": "Time to bring the weights onto the device. Warm means the base was already resident and only the adapter was swapped.",
    "flops": "Estimated floating-point operations for this request: 2 x active parameters x tokens. For a mixture-of-experts base only the active experts count.",
    "usd": "Latency priced at the selected hardware's hourly rate. Only an estimate for that card: measured here, priced there.",
    "tokens": "Input tokens the tokenizer produced for state + questions.",
    "confidence": "Calibrated confidence the decider attaches to its pick. Yes/no questions carry none: the probability itself is the uncertainty.",
}


def E(s: Any) -> str:
    """HTML-escape anything (None -> empty string)."""
    return _html.escape(str(s if s is not None else ""), quote=True)


def _tip(label: str, key: str) -> str:
    """A label carrying its one-sentence tooltip from TIPS."""
    return f'<span class="sdp-tip" tabindex="0" data-tip="{E(TIPS.get(key, ""))}">{E(label)}</span>'


def _fmt(v: Any, nd: int = 4) -> str:
    """Format a value for display: ints with separators, floats to `nd` places, None as a dash."""
    if isinstance(v, bool) or v is None:
        return "—" if v is None else str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _num(v: Any) -> bool:
    """True for a real number (bool excluded)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _count(v: Any, dec: int, unit: str = "", cls: str = "") -> str:
    """A number that counts up on reveal (static/app.js). Non-numbers render as-is, italic."""
    if not _num(v):
        return f'<span class="v na">{E(v if v is not None else "not measured")}</span>'
    return (f'<span class="v {cls}" data-sdp="count" data-count="{float(v)}" data-dec="{dec}" '
            f'data-unit="{E(unit)}">{v:,.{dec}f}{E(unit)}</span>')


# ---------------------------------------------------------------------------------
# model label: base + lora_r, never the digits in the repo name
# ---------------------------------------------------------------------------------

_LORA_R: dict[str, Any] = {}


def lora_r(info: ModelInfo) -> Any:
    """LoRA rank from the decider config (a local Hub-cache hit after the registry read it)."""
    if info.repo in _LORA_R:
        return _LORA_R[info.repo]
    val = None
    try:
        if info.config_file:
            from huggingface_hub import hf_hub_download
            cfg = json.load(open(hf_hub_download(info.repo, info.config_file,
                                                 token=os.environ.get("HF_TOKEN") or None)))
            val = cfg.get("lora_r") or cfg.get("r")
    except Exception:  # noqa: BLE001 -- a label must never block a card
        val = None
    _LORA_R[info.repo] = val
    return val


def base_short(info: ModelInfo | None) -> str:
    """Base model name without the org prefix."""
    b = (info.base_model if info else "") or ""
    return b.split("/")[-1] or "unknown base"


def model_label(info: ModelInfo | None) -> str:
    """Card label: base + LoRA rank. The digits in a repo name are not a size."""
    if not info:
        return "unknown"
    r = lora_r(info)
    return f"{base_short(info)}" + (f" · r {r}" if r else "")


def _short_name(info: ModelInfo | None, card_name: str) -> str:
    """Compact label for strip headers: base family + rank."""
    if not info:
        return card_name
    b = base_short(info).replace("gemma-4-", "g4-").replace("Qwen3.5-", "q3.5-").replace("-Base", "").replace("-it", "")
    r = lora_r(info)
    ver = re.search(r"(v\d+(?:-\d+)?)$", info.name)  # a version tag, not a size
    return b + (f" r{r}" if r else "") + (f" {ver.group(1)}" if ver else "")


# ---------------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------------


def ring_html(value: Any, label: str) -> str:
    """Confidence ring: SVG stroke-dashoffset animates from empty to value."""
    if not _num(value):
        return ('<div class="sdp-ring" aria-hidden="true"><svg viewBox="0 0 56 56"><circle class="track" cx="28" cy="28" r="24"/></svg>'
                '<div class="val muted">—</div><div class="lbl">' + E(label) + '</div></div>')
    v = max(0.0, min(1.0, float(value)))
    circ = 150.8  # 2*pi*24
    off = circ * (1 - v)
    return (f'<div class="sdp-ring" role="img" aria-label="{E(label)} {v:.4f}"><svg viewBox="0 0 56 56">'
            f'<circle class="track" cx="28" cy="28" r="24"/>'
            f'<circle class="arc" cx="28" cy="28" r="24" data-sdp="ring" data-off="{off:.2f}"/></svg>'
            f'<div class="val">{v:.2f}</div><div class="lbl">{E(label)}</div></div>')


def bars_html(probs: dict) -> str:
    """Per-option probability bars, sorted by probability; the argmax is the accent."""
    ps = [(k, float(v)) for k, v in (probs or {}).items() if _num(v)]
    if not ps:
        return ""
    ps.sort(key=lambda kv: -kv[1])
    top = ps[0][1]
    out = ['<div class="sdp-bars" role="list">']
    for k, v in ps:
        mx = " max" if v == top else ""
        out.append(
            f'<div class="opt{mx}" title="{E(k)}">{E(k)}</div>'
            f'<div class="track"><i class="fill{mx}" data-sdp="bar" data-w="{v * 100:.2f}"></i></div>'
            f'<div class="p{mx}">{v:.4f}</div>'
        )
    out.append("</div>")
    return "".join(out)


def question_text(specs: list[dict], qids: list[str], qid: str) -> str:
    """The instruction text of question `qid`, or an empty string."""
    try:
        i = qids.index(qid)
        return (specs[i].get("instructions") or "").strip()
    except (ValueError, IndexError):
        return ""


def answers_html(card: dict, specs: list[dict], qids: list[str]) -> str:
    """One block per question: id, text, headline answer, confidence (or p(true)), bars."""
    out = []
    for qid, a in (card.get("answers") or {}).items():
        conf = a.get("confidence")
        confs = (f'<span class="conf">conf {conf:.4f}</span>' if _num(conf)
                 else ('<span class="conf">p(true)</span>' if a.get("type") == "noul" else ""))
        out.append(
            '<div class="sdp-q">'
            f'<div class="qhead"><span class="qid">{E(qid)}</span>'
            f'<span class="qtxt" title="{E(question_text(specs, qids, qid))}">{E(question_text(specs, qids, qid))}</span>'
            f'<span class="ans">{E(a.get("display"))}</span>{confs}</div>'
            + bars_html(a.get("probs") or {}) + "</div>"
        )
    return "".join(out)


def memory_stack_svg(m: dict) -> str:
    """Horizontal stacked bands: weights (within baseline) -> baseline -> peak; activation = peak - baseline.
    Drawn in a 320-wide viewBox with the aspect ratio kept, so labels stay legible at card width."""
    w, b, p = m.get("weight_mb"), m.get("baseline_mb"), m.get("peak_mb")
    if not all(_num(x) for x in (w, b, p)) or p <= 0:
        return '<div class="sdp-foot">memory stack not drawn: ' + E(m.get("activation_mb") or "no peak measured") + "</div>"
    W, H = 320.0, 64.0
    scale = (W - 4) / max(p, w, b, 1e-9)
    xw, xb, xp = w * scale, b * scale, p * scale
    act = max(0.0, p - b)
    xa = max(xp - xb, 0.0)
    xa_draw = max(xa, 3.0) if act > 0 else 0.0  # the activation band is the story: never below 3px
    return (
        f'<svg class="sdp-stack" viewBox="0 0 {W:.0f} {H:.0f}" role="img" '
        f'aria-label="weights {w:,.0f} MB, baseline {b:,.0f} MB, peak {p:,.0f} MB, activation {act:,.1f} MB">'
        f'<text x="2" y="10">weights {w:,.0f} MB</text>'
        f'<text x="{W - 2:.0f}" y="10" text-anchor="end">peak {p:,.0f} MB</text>'
        f'<rect class="b" x="2" y="16" height="20" rx="3" width="0" data-sdp="rect" data-w="{xb:.1f}"/>'
        f'<rect class="w" x="2" y="20" height="12" rx="2" width="0" data-sdp="rect" data-w="{xw:.1f}"/>'
        f'<rect class="a" x="{2 + xb:.1f}" y="13" height="26" rx="3" width="0" data-sdp="rect" data-w="{xa_draw:.1f}"/>'
        f'<line class="guide" x1="2" y1="44" x2="{2 + xp:.1f}" y2="44"/>'
        f'<text x="2" y="58" class="hi">activation {act:,.1f} MB</text>'
        f'<text x="{W - 2:.0f}" y="58" text-anchor="end">baseline {b:,.0f} MB</text>'
        "</svg>"
    )


def timeline_svg(load_s: Any, latency_ms: Any, warm: bool) -> str:
    """Load time and inference latency on ONE linear axis, so a cold load is never mistaken for
    a slow forward pass."""
    if not _num(latency_ms):
        return ""
    lat_s = float(latency_ms) / 1000.0
    ld = float(load_s) if _num(load_s) else 0.0
    total = max(ld + lat_s, 1e-9)
    W, H = 320.0, 30.0
    xl = (W - 4) * (ld / total)
    xi = max((W - 4) * (lat_s / total), 2.0)
    ratio = (ld / lat_s) if lat_s > 0 and ld > 0 else None
    label_l = f"load {ld:.2f} s {'warm' if warm else 'cold'}" if ld > 0 else "load 0 s warm"
    rt = f" ({ratio:,.0f}x)" if ratio and ratio >= 2 else ""
    return (
        f'<svg class="sdp-timeline" viewBox="0 0 {W:.0f} {H:.0f}" role="img" '
        f'aria-label="{E(label_l)}, inference {lat_s * 1000:,.0f} ms">'
        f'<rect class="load" x="2" y="2" height="10" rx="2" width="0" data-sdp="rect" data-w="{xl:.1f}"/>'
        f'<rect class="infer" x="{2 + xl:.1f}" y="2" height="10" rx="2" width="0" data-sdp="rect" data-w="{xi:.1f}"/>'
        f'<text x="2" y="26">{E(label_l)}</text>'
        f'<text x="{W - 2:.0f}" y="26" text-anchor="end" class="hi">inference {lat_s * 1000:,.0f} ms{E(rt)}</text>'
        "</svg>"
    )


def compute_html(card: dict) -> str:
    """The compute block: memory stack, load-vs-inference timeline, metric rows, method footnote."""
    m = card.get("metrics") or {}
    load = card.get("load") or {}
    ap = card.get("active_params") or {}
    if not m:
        return '<div class="sdp-compute"><div class="sdp-foot">Compute not measured: the call did not run.</div></div>'
    warm = bool(load.get("was_cached"))
    moe = bool(ap.get("moe"))
    flops_note = (" (active params, MoE)" if moe else "")
    rows = [
        (_tip("activation", "activation"), _count(m.get("activation_mb"), 1, " MB", "hi")),
        (_tip("inference latency", "latency"), _count(m.get("latency_ms"), 1, " ms", "hi")),
        (_tip("load time", "load"), '<span class="vv">' + _count(load.get("load_seconds"), 2, " s") + f' <span class="k">{"warm" if warm else "cold"}</span></span>'),
        (_tip("input tokens", "tokens"), _count(m.get("input_tokens"), 0)),
        (_tip("est. FLOPs" + flops_note, "flops"), _count(m.get("est_tflops"), 3, " TFLOPs")),
        (_tip("$/h · est. $ this call", "usd"),
         (f'<span class="vv"><span class="v">{_fmt(m.get("usd_per_hour"), 2)}</span> <span class="k">·</span> ' + _count(m.get("est_usd"), 6, "") + "</span>")
         if _num(m.get("est_usd")) else f'<span class="v na">{E(m.get("est_usd"))}</span>'),
    ]
    legend = ('<div class="sdp-legend"><span><i class="w"></i>' + _tip("weights", "weight") + "</span>"
              '<span><i class="b"></i>' + _tip("baseline after load", "baseline") + "</span>"
              '<span><i class="a"></i>' + _tip("activation = peak − baseline", "peak") + "</span></div>")
    foot = (f"{E(m.get('backend'))} / {E(m.get('device'))} · {E(m.get('dtype'))} · memory via {E(m.get('mem_method'))}; "
            f"peak via {E(m.get('peak_method'))}; latency via {E(m.get('latency_method'))}; sampler poll "
            f"{E(os.environ.get('METRICS_SAMPLE_S'))} s.")
    if ap.get("active"):
        foot += f" Active params {int(ap['active']):,}" + (f" — {E(ap.get('method'))}" if moe else "") + "."
    if load.get("evicted"):
        foot += " Evicted to make room: " + E(", ".join(load["evicted"])) + "."
    if load.get("shared_base_note"):
        foot += " " + E(load["shared_base_note"])
    return (
        '<div class="sdp-compute">'
        + memory_stack_svg(m) + legend
        + timeline_svg(load.get("load_seconds"), m.get("latency_ms"), warm)
        + "".join(f'<div class="row"><span class="k">{k}</span>{v}</div>' for k, v in rows)
        + f'<div class="sdp-foot">{foot}</div></div>'
    )


def evals_html(info: ModelInfo | None) -> str:
    """Published-eval badges, each linking to the exact summary.json it was read from."""
    ev = (info.evals if info else None) or {}
    hl = ev.get("headline") or []
    if not hl:
        msg = ev.get("error") or "no eval/**/summary.json in this repo"
        return f'<div class="sdp-evals"><span class="sdp-foot">Published evals: {E(msg)}</span></div>'
    out = ['<div class="sdp-evals">']
    for h in hl:
        acc, nc, n = h.get("accuracy"), h.get("n_correct"), h.get("n")
        accs = f"{acc:.3f}" if _num(acc) else "—"
        frac = f" {nc}/{n}" if nc is not None and n else ""
        href = f"https://huggingface.co/{info.repo}/blob/main/{h.get('source')}" if info else "#"
        out.append(
            f'<a class="sdp-badge" href="{E(href)}" target="_blank" rel="noopener" title="source: {E(h.get("source"))}">'
            f'<span class="s">{E(h.get("suite"))}</span><b>{accs}</b><span class="m">{E(frac)}</span>'
            + (f'<span class="m">Brier</span><b>{h["brier"]:.3f}</b>' if _num(h.get("brier")) else "")
            + (f'<span class="m">ECE</span><b>{h["ece"]:.3f}</b>' if _num(h.get("ece")) else "")
            + "</a>"
        )
    out.append("</div>")
    return "".join(out)


def card_html(card: dict, info: ModelInfo | None, specs: list[dict], qids: list[str]) -> str:
    """One finished model card: header with pills and ring, answers, compute, evals (or the error)."""
    err = card.get("error")
    ap = card.get("active_params") or {}
    pills = []
    if info:
        pills.append(f'<span class="sdp-pill {"on" if info.vision else ""}">{"reads images" if info.vision else "text only"}</span>')
        if info.num_slots:
            pills.append(f'<span class="sdp-pill">{info.num_slots} slots</span>')
        if info.max_length:
            pills.append(f'<span class="sdp-pill">ctx {info.max_length:,}</span>')
        if info.torch_dtype:
            pills.append(f'<span class="sdp-pill">{E(info.torch_dtype)}</span>')
    if ap.get("moe"):
        pills.append(f'<span class="sdp-pill moe">MoE · {ap["active"] / 1e9:.2f}B active</span>')
    elif _num(ap.get("active")):
        pills.append(f'<span class="sdp-pill">{ap["active"] / 1e9:.2f}B params</span>')
    # the ring shows the first calibrated confidence; a yes/no-only run shows p(true)
    ring_v, ring_l = None, "confidence"
    for a in (card.get("answers") or {}).values():
        if _num(a.get("confidence")):
            ring_v = a["confidence"]
            break
    if ring_v is None:
        for a in (card.get("answers") or {}).values():
            if a.get("type") == "noul" and _num(a.get("value")):
                ring_v, ring_l = a["value"], "p(true)"
                break
    repo_url = f"https://huggingface.co/{card.get('repo')}"
    body = (f'<div class="sdp-foot" style="color:var(--warn)">{E(err)}</div>' if err else
            answers_html(card, specs, qids) + compute_html(card))
    return (
        f'<article class="sdp-card{" error" if err else ""}" data-sdp="card">'
        '<div class="head"><div class="title">'
        f'<div class="base">{E(base_short(info))}' + (f' <span class="r">· r {lora_r(info)}</span>' if info and lora_r(info) else "") + "</div>"
        f'<div class="repo"><a href="{E(repo_url)}" target="_blank" rel="noopener">{E(card.get("name"))}</a></div>'
        f'<div class="caps">{"".join(pills)}</div></div>'
        + ring_html(ring_v, ring_l) + "</div>"
        + body + evals_html(info) + "</article>"
    )


def skeleton_html(info: ModelInfo | None, name: str, state_text: str) -> str:
    """Placeholder card shown while a model is queued or running."""
    return (
        '<article class="sdp-card skel in" aria-busy="true">'
        '<div class="head"><div class="title">'
        f'<div class="base">{E(base_short(info))}' + (f' <span class="r">· r {lora_r(info)}</span>' if info and lora_r(info) else "") + "</div>"
        f'<div class="repo">{E(name)}</div><div class="sk s"></div></div><div class="sk ring"></div></div>'
        '<div class="sk bar"></div><div class="sk bar" style="width:72%"></div><div class="sk bar" style="width:48%"></div>'
        f'<div class="state">{E(state_text)}</div></article>'
    )


def grid_html(done: list[dict], pending: list[ModelInfo], specs: list[dict], qids: list[str], active: str | None) -> str:
    """The card grid: finished cards first, then a skeleton per pending model (`active` = running now)."""
    parts = [card_html(c, by_name(c["name"]), specs, qids) for c in done]
    for m in pending:
        parts.append(skeleton_html(m, m.name, "loading and answering…" if m.name == active else "queued"))
    return f'<div class="sdp sdp-grid">{"".join(parts)}</div>'


def section_html(title: str, why: str, elem_id: str = "", kick: str = "") -> str:
    """A section header in one pattern: optional step kicker (mono, muted), title, one-sentence why."""
    kicker = f'<span class="kick">{E(kick)}</span>' if kick else ""
    eid = f' id="{E(elem_id)}"' if elem_id else ""
    return f'<div class="sdp sdp-section"{eid}>{kicker}<h2>{E(title)}</h2><span class="sub">{E(why)}</span></div>'


def empty_html(text: str, elem_id: str = "") -> str:
    """What will appear in an empty results slot, in the dim style (replaced by the real surface after Decide)."""
    eid = f' id="{E(elem_id)}"' if elem_id else ""
    return f'<div class="sdp sdp-empty"{eid}>{E(text)}</div>'


def status_html(kind: str, text: str, done: int = 0, total: int = 0) -> str:
    """Status line with a dot (kind = live | ok | err | "") and an optional progress bar."""
    bar = (f'<span class="progress"><i data-sdp="progress" data-w="{100 * done / total:.1f}"></i></span>' if total else "")
    return f'<div class="sdp sdp-status"><span class="dot {kind}"></span><span>{E(text)}</span>{bar}</div>'


def comparison_html(result: dict) -> str:
    """Cross-model view: agreement strip (argmax per model, majority filled), small multiples of
    the full distribution per question, and a footer with device, wall time and cache budget."""
    cards = result.get("cards") or []
    qids = result.get("question_ids") or []
    specs = result.get("question_specs") or []
    agr = (result.get("comparison") or {}).get("agreement") or {}
    if len(cards) < 1 or not qids:
        return ""
    infos = [by_name(c["name"]) for c in cards]
    heads = "".join(f'<span title="{E(c["name"])}">{E(_short_name(i, c["name"]))}</span>' for c, i in zip(cards, infos))
    rows = []
    for qid in qids:
        a = agr.get(qid) or {}
        maj = a.get("majority")
        cells = []
        for c in cards:
            ans = (c.get("answers") or {}).get(qid)
            if not ans:
                cells.append('<span class="cell err" data-sdp="cell">error</span>')
                continue
            vote = str(ans.get("vote")) if ans.get("vote") is not None else ans.get("display")
            cls = " maj" if maj is not None and vote == str(maj) else ""
            shown = vote if vote == ans.get("display") else f'{vote} <span class="muted">{ans.get("display")}</span>'
            cells.append(f'<span class="cell{cls}" data-sdp="cell" title="{E(c["name"])}: {E(vote)} ({E(ans.get("display"))})">{shown}</span>')
        if a.get("n"):
            verdict = (f'<span class="verdict{" unanimous" if a.get("unanimous") else ""}"><b>{a["majority_n"]}/{a["n"]}</b> '
                       f'{"unanimous" if a.get("unanimous") else "agree"} · spread {a.get("prob_spread", 0):.4f}</span>')
        else:
            verdict = f'<span class="verdict">{E(a.get("note", ""))}</span>'
        rows.append(f'<div class="srow"><span class="q"><span class="qid">{E(qid)}</span>{E(question_text(specs, qids, qid))}</span>'
                    f'<span class="cells">{"".join(cells)}</span>{verdict}</div>')
    strip = (f'<div class="sdp-strip"><div class="heads"><span>argmax per model</span><span class="cells">{heads}</span><span></span></div>'
             + "".join(rows) + "</div>")

    # small multiples: one chart per question; x = options, one thin bar per model; the model's argmax in accent
    mults = []
    for qid in qids:
        opts: list[str] = []
        for c in cards:
            for k, v in (((c.get("answers") or {}).get(qid) or {}).get("probs") or {}).items():
                if _num(v) and k not in opts:
                    opts.append(k)
        if not opts:
            continue
        W, H, pad = 220.0, 110.0, 16.0
        n_m = max(len(cards), 1)
        group_w = (W - 2 * pad) / len(opts)
        bar_w = max(min(group_w * 0.7 / n_m, 14.0), 2.0)
        svg = [f'<svg viewBox="0 0 {W:.0f} {H:.0f}" role="img" aria-label="probabilities per option for {E(qid)}">']
        svg.append(f'<line x1="{pad}" y1="{H - 18}" x2="{W - pad}" y2="{H - 18}"/>')
        for i, o in enumerate(opts):
            x0 = pad + i * group_w + (group_w - bar_w * n_m) / 2
            for j, c in enumerate(cards):
                ans = ((c.get("answers") or {}).get(qid) or {})
                v = (ans.get("probs") or {}).get(o)
                if not _num(v):
                    continue
                h = max(float(v) * (H - 30), 0.0)
                vote = str(ans.get("vote")) if ans.get("vote") is not None else None
                is_max = vote == o or (vote is None and ans.get("display") == o)
                svg.append(f'<rect class="bar{" max" if is_max else ""}" x="{x0 + j * bar_w:.1f}" width="{bar_w - 1:.1f}" '
                           f'y="{H - 18:.1f}" height="0" data-sdp="rect" data-h="{h:.1f}" data-y="{H - 18 - h:.1f}">'
                           f'<title>{E(c["name"])} · {E(o)} {float(v):.4f}</title></rect>')
            label = o if len(o) <= 10 else o[:9] + "…"
            svg.append(f'<text x="{pad + i * group_w + group_w / 2:.1f}" y="{H - 5}" text-anchor="middle">{E(label)}</text>')
        svg.append("</svg>")
        a = agr.get(qid) or {}
        mn, mx = a.get("prob_min"), a.get("prob_max")
        mults.append(f'<div class="sdp-mult"><div class="mt"><span>{E(question_text(specs, qids, qid)) or E(qid)}</span><span class="qid">{E(qid)}</span></div>'
                     + "".join(svg)
                     + (f'<div class="ml"><span>winning p min {mn:.4f}</span><span>max {mx:.4f}</span></div>' if _num(mn) and _num(mx) else "")
                     + "</div>")
    b = result.get("budget") or {}
    foot = (f"device {E(result.get('device'))} · wall {result.get('wall_seconds')} s for {len(cards)} model(s), serialized behind one lock · "
            f"cache budget {b.get('max_resident_gib')} GiB · resident {b.get('resident_gib')} GiB {E(b.get('resident_bases'))}")
    return (f'<div class="sdp sdp-compare">{strip}'
            f'<div class="sdp-multiples">{"".join(mults)}</div>'
            f'<div class="sdp-foot">{foot}</div></div>')


def header_html() -> str:
    """Header row (mark, wordmark, project tag) followed by the landing hero and proof row."""
    n, v = len(REGISTRY), len(vision_models())
    count = (f"{n} public models read live from the org at start · {v} read images · "
             "<kbd>Enter</kbd> runs · <kbd>Esc</kbd> clears" if not REGISTRY_ERROR
             else f"could not read the org: {E(REGISTRY_ERROR)}")
    return (
        '<div class="sdp"><header class="sdp-header">' + MARK_SVG
        + '<a href="https://strandsagents.com" target="_blank" rel="noopener" title="strandsagents.com">' + WORDMARK_SVG + "</a>"
        '<span class="project">/decider-playground</span><span class="tag" style="margin-left:auto">public checkpoints only</span></header></div>'
        + landing.hero_html(n) + landing.proof_html(n)
    )


def footer_html() -> str:
    """Status footer: public model count, the hardware this process runs on, library versions."""
    dev = runner.default_device()
    hw = {"cuda": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cuda"),
          "mps": f"Apple {platform.machine()} (MPS)", "cpu": f"{platform.machine()} CPU"}.get(dev, dev)
    n = len(REGISTRY)
    return (
        '<div class="sdp"><footer class="sdp-footer" id="sdp-footer" data-public-count="' + str(n) + '">'
        + MARK_SVG + f'<span><b>{n} public model{"s" if n != 1 else ""}</b>'
        + '</span><span class="sep">·</span>'
        f"<span>hardware <b>{E(hw)}</b></span><span class=\"sep\">·</span>"
        f"<span>torch <b>{E(torch.__version__)}</b> · transformers <b>{E(transformers.__version__)}</b> · gradio <b>{E(gr.__version__)}</b></span>"
        '<span class="sep">·</span><span>built by <a href="https://strandsagents.com" target="_blank" rel="noopener">strands</a></span>'
        "</footer></div>"
    )


# ---------------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------------

def _css_with_fonts() -> str:
    """tokens.css refers to fonts/*.woff2; inline them as data URIs (no static route to configure)."""
    css = _static("tokens.css")
    for f in sorted((STATIC / "fonts").glob("*.woff2")):
        b64 = base64.b64encode(f.read_bytes()).decode("ascii")
        css = css.replace(f'url("fonts/{f.name}")', f'url("data:font/woff2;base64,{b64}")')
    return css + "\n" + _static("app.css")


CSS = _css_with_fonts()
JS = _static("app.js")
# gr.HTML cannot run <script>, so every section's CSS/JS rides launch(css=CSS, js=JS).
CSS += "\n" + landing.LANDING_CSS
JS += "\n" + landing.LANDING_JS
CSS += "\n" + EX.css()
JS += "\n" + EX.js()
CSS += "\n" + _static("viz.css")
JS += "\n" + _static("viz.js")
CSS += "\n" + bench.css()
JS += "\n" + bench.js()

GOLDEN_Q = [("noul", "Does this convey urgency?", []),
            ("choice", "Which team should handle this?", ["billing", "sales", "retail"]),
            ("score", "How frustrated is the writer?", ["calm", "frustrated", "depressed"])]


# Bench section helpers (used inside build_ui).
_BENCH_LABELS: dict[str, str] = {}


def bench_estimate_html(names: list[str], scope_v: str, model_v: str, started: bool = False) -> str:
    """Wall-time estimate shown before a run, from medians measured on this Space only (bench.estimate_s)."""
    if not _BENCH_LABELS:
        _BENCH_LABELS.update({m.name: f"{m.name.replace('strands-decider-', '')} · {model_label(m)}" for m in REGISTRY})
    ms = names if model_v == "all public models" else [model_v]
    tier_v = scope_v.split(":", 1)[1].strip() if scope_v.startswith("tier:") else None
    n = len(bench.task_ids(tier_v))
    secs, src = bench.estimate_s(ms, n)
    lead = "started · " if started else ""
    when = f"about {bench._fmt_s(secs)} ({E(src)})" if secs else E(src)
    return (f'<div class="sdp sdb-est">{lead}{n} tasks × {len(ms)} model{"s" if len(ms) != 1 else ""} = {n * len(ms)} calls · '
            f'{when} · one model call at a time</div>')


def make_theme() -> gr.themes.Base:
    """Gradio theme pinned to the robots tokens; app.css does the rest with the CSS variables."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(c50="#e6fff1", c100="#ccffe4", c200="#99ffc9", c300="#66ffad", c400="#33ff92",
                                     c500="#00FF77", c600="#00cc5f", c700="#009947", c800="#006630", c900="#003318", c950="#001a0c"),
        neutral_hue=gr.themes.Color(c50="#fafafa", c100="#f5f5f5", c200="#eeeeee", c300="#c8c8c8", c400="#989898",
                                     c500="#666666", c600="#444444", c700="#333333", c800="#1a1a1a", c900="#111111", c950="#0a0a0a"),
        font=[gr.themes.GoogleFont("Inter"), "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica", "Arial", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        body_background_fill="#fafafa", body_background_fill_dark="#0a0a0a",
        body_text_color="#333333", body_text_color_dark="#c8c8c8",
        block_background_fill="rgba(245,245,245,0.7)", block_background_fill_dark="rgba(255,255,255,0.03)",
        block_border_color="rgba(0,0,0,0.06)", block_border_color_dark="rgba(255,255,255,0.06)",
        button_primary_background_fill="#555555", button_primary_background_fill_dark="#00FF77",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#0a0a0a",
        input_background_fill="#f5f5f5", input_background_fill_dark="rgba(255,255,255,0.04)",
        shadow_drop="none", shadow_drop_lg="none", block_shadow="none",
    )


def build_ui() -> gr.Blocks:
    """Assemble the page and wire the handlers. Reads the registry first."""
    load_models()
    names = model_choices()
    vnames = vision_models()

    with gr.Blocks(title="Strands Decider Playground", fill_width=True) as demo:
        gr.HTML(header_html(), elem_classes=["sdp-plain"])

        # ---- ask bar -------------------------------------------------------------
        gr.HTML('<div class="sdp sdp-kick" id="sdp-step-ask"><span class="kick">01 · ask</span>'
                '<span class="why">Type a piece of state and a question, or pick an example or a benchmark task below — every public decider answers it.</span></div>',
                elem_classes=["sdp-plain"])
        with gr.Group(elem_id="sdp-ask"):
            state = gr.Textbox(label="State — what is being decided about", lines=3, value=GOLDEN_STATE,
                               placeholder="Paste the ticket, message, document, board position…")
            with gr.Row():
                q1type = gr.Radio(["noul", "choice", "score"], value="noul", label="Question type", scale=3,
                                  info="noul yes/no · choice pick one · score rubric")
                q1ins = gr.Textbox(label="Question", value="Does this convey urgency?", scale=6,
                                   placeholder="Which team should handle this?")
            q1opts = gr.Dropdown(choices=[], value=[], multiselect=True, allow_custom_value=True,
                                 label="Options", info="type one, press Enter · choice needs 2 or more · score 2–10 in ascending order · none for yes/no")
            with gr.Accordion("More questions", open=False):
                nq = gr.State(1)
                qrows, qtypes, qinstr, qopts = [], [], [], []
                for i in range(1, MAX_QUESTIONS):
                    with gr.Group(visible=False) as row:
                        with gr.Row():
                            t = gr.Radio(["noul", "choice", "score"], value="noul", label=f"Q{i + 1} type", scale=2)
                            ins = gr.Textbox(label=f"Q{i + 1} question", scale=5)
                        op = gr.Dropdown(choices=[], value=[], multiselect=True, allow_custom_value=True, label=f"Q{i + 1} options")
                    qrows.append(row); qtypes.append(t); qinstr.append(ins); qopts.append(op)
                with gr.Row():
                    add = gr.Button("Add question", size="sm")
                    rm = gr.Button("Remove last", size="sm")
            with gr.Accordion("Image, models, pricing", open=False):
                image = gr.Image(label=f"Image (optional) — read by {', '.join(vnames) or 'no model'}; Gemma 4 deciders drop "
                                       "their base vision encoder and read text only", type="pil", height=160)
                models = gr.CheckboxGroup(choices=names, value=names, label="Models (public only, default all)", elem_id="sdp-models")
                hardware = gr.Dropdown(choices=list(HARDWARE.keys()), value="Nvidia A100 large (80 GB) — $2.50/h",
                                       label="Price the call as this hardware ($/h from huggingface.co/pricing)")
            with gr.Row():
                run = gr.Button("Decide", variant="primary", elem_id="sdp-run", scale=3)
                golden = gr.Button("Load golden question", scale=2)
                clear = gr.Button("Clear", elem_id="sdp-clear", scale=1, variant="secondary", size="sm", elem_classes=["sdp-quiet"])
                unload = gr.Button("Unload models", scale=1, variant="secondary", size="sm", elem_classes=["sdp-quiet"])

        # Example gallery. #sdp-examples is the landing hero's "See the examples" target; a card click
        # writes "<example-id>#<nonce>" into the hidden textbox (1x1 px via CSS), which fills and runs.
        gr.HTML(EX.gallery_html(), elem_classes=["sdp-plain"], elem_id="sdp-examples")
        ex_pick = gr.Textbox(value="", visible=True, elem_id="sdx-pick", container=False, show_label=False,
                             elem_classes=["sdx-hidden"])
        # JevBench picker: click a task -> fills the ask bar + runs; the gold answer is shown only after the answers.
        gr.HTML(section_html("Try a benchmark task", "231 public JevBench tasks — click one to run it like an example; the gold answer appears after the models answer", "sdb-head"),
                elem_classes=["sdp-plain"])
        gr.HTML(bench.picker_html(), elem_classes=["sdp-plain"], elem_id="sdb-picker-host")
        bench_pick = gr.Textbox(value="", visible=True, elem_id="sdb-pick", container=False, show_label=False, elem_classes=["sdb-hidden"])
        bench_tid = gr.State("")  # the JevBench task the ask bar currently holds ("" = a free question)


        # ---- results band: everything a Decide fills, on one surface ------------
        EMPTY_COMPARE = empty_html("appears after Decide — every model's answer in one table, so agreement and the odd one out are visible at a glance")
        EMPTY_VIZ = empty_html("appears after Decide — the consensus diagram: where the deciders agree, and how sure each one is")

        def idle_grid_html() -> str:
            """The card grid before any run: one static placeholder per model (no shimmer — nothing is loading)."""
            return grid_html([], REGISTRY, [], [], None).replace("queued", "idle").replace('class="sdp sdp-grid"', 'class="sdp sdp-grid idle"')

        with gr.Column(elem_id="sdp-results", elem_classes=["sdp-band"]):
            gr.HTML('<div class="sdp sdp-kick"><span class="kick">02 · see</span>'
                    '<span class="why">The answers land here: one table, one card per model, one diagram.</span></div>', elem_classes=["sdp-plain"])
            status = gr.HTML(status_html("", "Ready."), elem_classes=["sdp-plain"])
            gr.HTML(section_html("Comparison", "the option every model picked, and the full distribution behind it"), elem_classes=["sdp-plain"])
            compare = gr.HTML(EMPTY_COMPARE, elem_classes=["sdp-plain"])
            gr.HTML(section_html("Models", "one card per decider: name · base · LoRA rank · the answer, its confidence, memory and latency"), elem_classes=["sdp-plain"])
            cards = gr.HTML(idle_grid_html(), elem_classes=["sdp-plain"])
            # Gold answer + JevBench score per model, rendered from `raw` after the models answered.
            bench_gold = gr.HTML("", elem_classes=["sdp-plain"], elem_id="sdb-gold")
            # Decision diagrams, re-drawn from the result dict whenever `raw` changes (see raw.change below).
            gr.HTML(section_html("Decisions", "every decider on one diagram — same choice, different certainty", "sdp-viz-head"), elem_classes=["sdp-plain"])
            viz_out = gr.HTML(EMPTY_VIZ, elem_classes=["sdp-plain"], elem_id="sdp-viz")
        # Run the bench + the activity board + common errors (see bench/__init__.py).
        _BENCH_LABELS.update({m.name: f"{m.name.replace('strands-decider-', '')} · {model_label(m)}" for m in REGISTRY})
        # ---- bench band: the board, its controls, the error panels -------------
        with gr.Column(elem_id="sdp-bench", elem_classes=["sdp-band"]):
            gr.HTML('<div class="sdp sdp-kick"><span class="kick">03 · bench</span>'
                    '<span class="why">All 231 JevBench tasks against every public model — read the vendored baseline, or run it again here.</span></div>',
                    elem_classes=["sdp-plain"])
            gr.HTML(section_html("JevBench board", "one row per model, one column per task; green = right, dark = wrong, the shade = how sure it was", "sdb-board-head"),
                    elem_classes=["sdp-plain"])
            with gr.Row(elem_classes=["sdb-controls"]):
                _hc = bench.history_choices()
                bench_history = gr.Dropdown(choices=_hc, value=(_hc[0][1] if _hc else None), label="Showing", scale=3, elem_id="sdb-history",
                                            info="the vendored baseline first, then runs started on this Space")
                bench_scope = gr.Dropdown(choices=["full JevBench (231)", "tier: easy", "tier: standard", "tier: hard"], value="full JevBench (231)",
                                          label="Scope", scale=2, elem_id="sdb-scope")
                bench_model = gr.Dropdown(choices=["all public models"] + names, value="all public models", label="Models", scale=2, elem_id="sdb-model")
                bench_run = gr.Button("Run JevBench", variant="primary", elem_id="sdb-run", scale=1)
                bench_cancel = gr.Button("Cancel", scale=1)
            with gr.Accordion("Daily schedule · older history rows", open=False, elem_id="sdb-more"):
                with gr.Row(elem_classes=["sdb-controls"]):
                    bench_hist = gr.Checkbox(value=False, label="show the correct-only history runs (32 rows)", scale=2)
                    bench_sched_on = gr.Checkbox(value=False, label="repeat daily (UTC)", scale=1)
                    bench_sched_at = gr.Textbox(value="03:00", label="at HH:MM UTC", max_lines=1, scale=1)
            bench_est = gr.HTML(bench_estimate_html(names, "full JevBench (231)", "all public models"), elem_classes=["sdp-plain"])
            bench_board = gr.HTML(bench.board_html(bench.latest(), _BENCH_LABELS), elem_classes=["sdp-plain"], elem_id="sdb-board")
            bench_detail_pick = gr.Textbox(value="", visible=True, elem_id="sdb-detail", container=False, show_label=False, elem_classes=["sdb-hidden"])
            bench_detail = gr.HTML(empty_html("click a column of the board (or a task in the error panels) for the task detail — state, question, every model's answer side by side"),
                                   elem_classes=["sdp-plain"], elem_id="sdb-detail-out")
            gr.HTML(section_html("Common errors", "only what can be counted — which tasks each model misses, which tasks everyone misses, and the paired McNemar test strands-decider itself uses"),
                    elem_classes=["sdp-plain"])
            bench_errors = gr.HTML(bench.errors_html(bench.latest(), _BENCH_LABELS), elem_classes=["sdp-plain"], elem_id="sdb-errors")
            bench_sig = gr.State("")
            bench_timer = gr.Timer(3.0, active=True)
            with gr.Accordion("Raw result — every number on this page comes from here", open=False):
                raw = gr.JSON(value=None)
        # How-to-use + model row, then the status footer (the one <footer>, carrying data-public-count),
        # then the landing foot: CTA panel, install line, link columns, wordmark.
        gr.HTML(landing.howto_html() + landing.models_html(REGISTRY) + footer_html() + landing.footer_html(), elem_classes=["sdp-plain"])

        # ---- question add/remove -------------------------------------------------
        def _set_n(n: int, delta: int):
            """Show the first n question rows."""
            n = max(1, min(MAX_QUESTIONS, n + delta))
            return [n] + [gr.update(visible=(i + 1 < n)) for i in range(MAX_QUESTIONS - 1)]

        add.click(lambda n: _set_n(n, +1), [nq], [nq] + qrows)
        rm.click(lambda n: _set_n(n, -1), [nq], [nq] + qrows)

        def _golden():
            """The 3-question golden set from the strands-decider README."""
            out = [GOLDEN_STATE, GOLDEN_Q[0][0], GOLDEN_Q[0][1], gr.update(choices=[], value=[]), 3]
            out += [gr.update(visible=(i + 1 < 3)) for i in range(MAX_QUESTIONS - 1)]
            for i in range(1, MAX_QUESTIONS):
                t, ins, op = GOLDEN_Q[i] if i < 3 else ("noul", "", [])
                out += [t, ins, gr.update(choices=op, value=op)]
            return out

        golden.click(_golden, None, [state, q1type, q1ins, q1opts, nq] + qrows
                     + [x for trio in zip(qtypes, qinstr, qopts) for x in trio])

        def _clear():
            """Reset the ask bar, the status, the comparison and the cards."""
            out = ["", "noul", "", gr.update(choices=[], value=[]), 1]
            out += [gr.update(visible=False) for _ in range(MAX_QUESTIONS - 1)]
            for _ in range(1, MAX_QUESTIONS):
                out += ["noul", "", gr.update(choices=[], value=[])]
            out += [status_html("", "Cleared."), EMPTY_COMPARE, idle_grid_html(), None]
            return out

        clear.click(_clear, None, [state, q1type, q1ins, q1opts, nq] + qrows
                    + [x for trio in zip(qtypes, qinstr, qopts) for x in trio] + [status, compare, cards, raw])
        unload.click(lambda: (runner.evict_all(), status_html("", "Models unloaded; the next call is a cold load."))[1], None, status)

        # ---- run: one model at a time, cards stream in as each finishes ----------
        def _run_stream(state_v, image_v, models_v, hardware_v, t1, i1, o1, n, *flat):
            """Run the selected models one at a time, yielding (status, comparison, cards, raw) so
            each card streams in as its model finishes. The final `raw` has `answer()`'s shape."""
            k = MAX_QUESTIONS - 1
            ts, ins, ops = flat[:k], flat[k:2 * k], flat[2 * k:3 * k]
            specs = [{"type": t1, "instructions": i1, "options": o1 or []}]
            specs += [{"type": ts[i], "instructions": ins[i], "options": ops[i] or []} for i in range(int(n) - 1)]
            chosen = [m for m in (by_name(x) for x in (models_v or [])) if m is not None]
            usd = HARDWARE.get(hardware_v, 0.0)
            if not chosen:
                yield status_html("err", "Select at least one model."), "", grid_html([], [], [], [], None), None
                return
            yield (status_html("live", f"0/{len(chosen)} answered…", 0, len(chosen)), "",
                   grid_html([], chosen, [], [], chosen[0].name), None)
            done: list[dict] = []
            qids: list[str] = []
            clean_specs: list[dict] = []
            last: dict = {}
            t0 = time.perf_counter()
            for i, m in enumerate(chosen):
                res = answer(state_v, image_v, specs, [m.name], usd_per_hour=usd)
                if res.get("error") and not res.get("cards"):
                    yield status_html("err", res["error"]), "", grid_html(done, [], clean_specs, qids, None), res
                    return
                done += res["cards"]
                qids, clean_specs, last = res["question_ids"], res["question_specs"], res
                nxt = chosen[i + 1].name if i + 1 < len(chosen) else None
                yield (status_html("live", f"{len(done)}/{len(chosen)} answered · {time.perf_counter() - t0:.1f} s", len(done), len(chosen)),
                       "", grid_html(done, chosen[i + 1:], clean_specs, qids, nxt), None)
            result = {
                "error": None, "cards": done, "question_ids": qids, "question_specs": clean_specs,
                "comparison": runner.comparison(done, qids), "wall_seconds": round(time.perf_counter() - t0, 3),
                "device": last.get("device"), "sample_interval_s": last.get("sample_interval_s"),
                "budget": last.get("budget"), "public_models": len(REGISTRY), "hidden_private": len(HIDDEN_PRIVATE),
            }
            ok = sum(1 for c in done if not c.get("error"))
            yield (status_html("ok" if ok == len(done) else "err",
                               f"{ok}/{len(done)} models answered in {result['wall_seconds']} s on {result['device']}", len(done), len(done)),
                   comparison_html(result), grid_html(done, [], clean_specs, qids, None), result)

        run.click(_run_stream, [state, image, models, hardware, q1type, q1ins, q1opts, nq] + qtypes + qinstr + qopts,
                  [status, compare, cards, raw])

        def _fill_example(raw_id: str):
            """Fill the ask bar from a clicked example (the output list mirrors `_ex_outputs`)."""
            ex_id = (raw_id or "").split("#", 1)[0].strip()
            ex = EX.example_by_id(ex_id)
            if ex is None:
                return [gr.update()] * (5 + (MAX_QUESTIONS - 1) * 4 + 2)
            f = EX.fill_values(ex, REGISTRY)
            qs = f["questions"]
            n = max(1, min(MAX_QUESTIONS, len(qs)))
            q0 = qs[0]
            out = [f["state"], q0["type"], q0["instructions"], gr.update(choices=q0.get("options") or [], value=q0.get("options") or []), n]
            out += [gr.update(visible=(i + 1 < n)) for i in range(MAX_QUESTIONS - 1)]
            for i in range(1, MAX_QUESTIONS):
                if i < len(qs):
                    q = qs[i]; op = q.get("options") or []
                    out += [q["type"], q["instructions"], gr.update(choices=op, value=op)]
                else:
                    out += ["noul", "", gr.update(choices=[], value=[])]
            out += [f["image"], gr.update(value=f["models"])]
            return out

        _ex_outputs = ([state, q1type, q1ins, q1opts, nq] + qrows
                       + [x for trio in zip(qtypes, qinstr, qopts) for x in trio] + [image, models])
        ex_pick.input(_fill_example, [ex_pick], _ex_outputs).then(
            _run_stream, [state, image, models, hardware, q1type, q1ins, q1opts, nq] + qtypes + qinstr + qopts,
            [status, compare, cards, raw])

        # `raw` holds the exact dict _run_stream produced; any path that sets it re-draws the diagrams.
        _LORA = {m.name: lora_r(m) for m in REGISTRY}

        def _viz(result):
            """Diagrams from the result dict; a drawing bug must never hide the cards."""
            try:
                return viz.render(result, _LORA) if result else EMPTY_VIZ
            except Exception as e:  # noqa: BLE001 -- a diagram bug must never hide the cards
                return f'<div class="sdp sdp-foot">viz error: {E(f"{type(e).__name__}: {e}")}</div>'

        raw.change(_viz, [raw], [viz_out])

        # Bench handlers; every model call goes through answer() exactly like the Decide button.
        def _fill_task(raw_id: str):
            tid = (raw_id or "").split("#", 1)[0].strip()
            f = bench.fill_values(tid)
            if f is None:
                return [gr.update()] * (5 + (MAX_QUESTIONS - 1) * 4 + 2) + [""]
            out = [f["state"], f["type"], f["instructions"], gr.update(choices=f["options"], value=f["options"]), 1]
            out += [gr.update(visible=False) for _ in range(MAX_QUESTIONS - 1)]
            for _ in range(1, MAX_QUESTIONS):
                out += ["noul", "", gr.update(choices=[], value=[])]
            out += [None, gr.update(value=names)]  # no image; every public model, like the bench itself
            return out + [tid]

        bench_pick.input(_fill_task, [bench_pick], _ex_outputs + [bench_tid]).then(
            _run_stream, [state, image, models, hardware, q1type, q1ins, q1opts, nq] + qtypes + qinstr + qopts,
            [status, compare, cards, raw])
        # a free question (Decide / example / golden) clears the task id so no gold card is shown for it
        run.click(lambda: "", None, bench_tid)
        ex_pick.input(lambda: "", None, bench_tid)
        golden.click(lambda: "", None, bench_tid)
        clear.click(lambda: "", None, bench_tid)

        def _gold(result, tid):
            try:
                return bench.gold_html(tid, result) if result and tid else ""
            except Exception as e:  # noqa: BLE001
                return f'<div class="sdp sdb-dim">bench scoring error: {E(f"{type(e).__name__}: {e}")}</div>'

        raw.change(_gold, [raw, bench_tid], [bench_gold])

        def _bench_models(model_v):
            return names if model_v == "all public models" else [model_v]

        def _bench_tier(scope_v):
            return scope_v.split(":", 1)[1].strip() if scope_v.startswith("tier:") else None

        def _bench_start(scope_v, model_v):
            j = bench.start_run(answer, _bench_models(model_v), _bench_tier(scope_v), label=f"{scope_v} · {model_v}")
            if isinstance(j, str):
                return gr.update(), f'<div class="sdp sdb-est">{E(j)}</div>', gr.update()
            return (bench.board_html(j, _BENCH_LABELS), bench_estimate_html(names, scope_v, model_v, started=True),
                    gr.update(choices=bench.history_choices(), value="run:" + j.run_id))

        # Bench handlers are pure-Python renders (the model calls happen in the Job thread, which takes the runner lock itself),
        # so they get their own queue: a board poll or a column click must never wait behind a 7-model Decide.
        bench_run.click(_bench_start, [bench_scope, bench_model], [bench_board, bench_est, bench_history], concurrency_id="bench-ui", concurrency_limit=8)
        bench_cancel.click(lambda: (bench.cancel_live(), bench.board_html(bench.latest(), _BENCH_LABELS))[1], None, bench_board, concurrency_id="bench-ui", concurrency_limit=8)
        bench_history.change(lambda k, h: (bench.board_html(bench.job_for(k), _BENCH_LABELS, show_history=bool(h)), bench.errors_html(bench.job_for(k), _BENCH_LABELS)),
                             [bench_history, bench_hist], [bench_board, bench_errors], concurrency_id="bench-ui", concurrency_limit=8)
        bench_scope.change(lambda sc, mo: bench_estimate_html(names, sc, mo), [bench_scope, bench_model], bench_est, concurrency_id="bench-ui", concurrency_limit=8)
        bench_model.change(lambda sc, mo: bench_estimate_html(names, sc, mo), [bench_scope, bench_model], bench_est, concurrency_id="bench-ui", concurrency_limit=8)

        def _bench_tick(sig, hist, sel):
            j = bench.job_for(sel)  # the selected history entry (the baseline by default); a started run selects itself
            new = f"{j.run_id if j else ''}:{j.done if j else 0}:{int(bool(j and j.running()))}:{int(bool(hist))}:{(j.push.get('status') if j else '')}"
            if new == sig:
                return gr.update(), gr.update(), sig
            errs = gr.update() if (j and j.running() and j.done % 7) else bench.errors_html(j, _BENCH_LABELS)
            return bench.board_html(j, _BENCH_LABELS, show_history=bool(hist)), errs, new

        bench_timer.tick(_bench_tick, [bench_sig, bench_hist, bench_history], [bench_board, bench_errors, bench_sig], concurrency_id="bench-ui", concurrency_limit=8)
        bench_hist.change(lambda h, k: bench.board_html(bench.job_for(k), _BENCH_LABELS, show_history=bool(h)), [bench_hist, bench_history], bench_board, concurrency_id="bench-ui", concurrency_limit=8)

        def _bench_detail(raw_id: str, sel):
            tid = (raw_id or "").split("#", 1)[0].strip()
            return bench.task_detail_html(tid, bench.job_for(sel), _BENCH_LABELS) or gr.update()

        bench_detail_pick.input(_bench_detail, [bench_detail_pick, bench_history], [bench_detail], concurrency_id="bench-ui", concurrency_limit=8)

        def _bench_sched(on, at):
            try:
                st = bench.set_schedule(bool(on), at or "03:00", answer, names)
            except Exception as e:  # noqa: BLE001
                return f'<div class="sdp sdb-est">schedule error: {E(e)}</div>'
            nxt = time.strftime("%Y-%m-%d %H:%MZ", time.gmtime(st["next"])) if st.get("next") else "off"
            return (f'<div class="sdp sdb-est">daily run: {"on" if st["enabled"] else "off"} · next {nxt} · '
                    f'persisted: {E(st.get("persisted", "no HF_TOKEN"))} · honest note: a thread timer fires only while this process is up — '
                    f'a sleeping Space will not fire it (this a100 Space does not sleep unless configured to)</div>')

        bench_sched_on.change(_bench_sched, [bench_sched_on, bench_sched_at], bench_est, concurrency_id="bench-ui", concurrency_limit=8)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # default_concurrency_limit=1: Gradio serves on a thread pool, and model calls must not
    # overlap (unsafe on MPS, unattributable everywhere). The lock in cache.py is the real
    # guard; this keeps the queue from piling requests up behind it.
    # Gradio 6: css/js/theme belong to launch(), not Blocks().
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=make_theme(),
        css=CSS,
        js=JS,
    )
