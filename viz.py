"""Animated SVG decision diagrams: every decider's answer on one picture.

Pure functions: the dict `app.answer()` returns -> SVG/HTML strings. No Gradio, no torch, no
model code, no I/O except the `--demo` CLI at the bottom. `app.py` calls `render()` whenever
the raw result changes; `static/viz.css` + `static/viz.js` animate the markup.

Input shape (app.answer(), runner.normalise_answer()):
    result = {"cards": [card...], "question_ids": [qid...], "question_specs": [...], ...}
    card   = {"name", "repo", "base_model", "error", "active_params": {"active": int|None, ...},
              "answers": {qid: {"type": "noul"|"choice"|"score", "value", "probs": {label: p},
                                "confidence": float|None, "vote": str|None, "legend"?: {...}}}}

Every number drawn is read from that dict -- nothing is recomputed except sums and means,
which are the point of the "sum of decisions" diagram and are labelled as such.

Diagrams (all inline SVG, animated by static/viz.css + static/viz.js, reduced-motion safe):
  1. consensus bubble-up   -- each model's argmax rises to its option; columns show vote count
                              AND probability mass (sum of p(option) across models); score
                              questions land bubbles on an ordered axis at the expected value.
  2. distribution multiples -- one tiny bar chart per option, bars = models in registry order.
  3. confidence vs size     -- scatter, x = active params (log), y = confidence; 0.9 reference.
  4. agreement matrix       -- NxN grid, filled when two models share an argmax (>= 3 models).

Strands rules: 1px strokes, 8px rx, accent for the signal, fg for structure, muted labels,
dashed groups, JetBrains Mono, NO gradients / glow / shadows. Colours come from tokens.css
variables through CSS classes, so the same markup is correct in paper and dark.
"""

from __future__ import annotations

import html
import math
import re
from typing import Any, Iterable

__all__ = [
    "render",
    "consensus_svg",
    "multiples_html",
    "scatter_svg",
    "agreement_svg",
    "model_label",
    "demo_results",
]

# The ONE external number we quote. From the strands-decider README (evaluation/results.md):
# a confidence of 0.9 corresponds to ~95% accuracy on unseen short tasks. Do not add others.
CONF_REF = 0.9
CONF_REF_LABEL = "0.9: ~95% right on unseen short tasks (evaluation/results.md)"

E = html.escape


# ---------------------------------------------------------------------------------
# labels (string work only; mirrors app._short_name without importing app)
# ---------------------------------------------------------------------------------


def base_short(base_model: str | None) -> str:
    """Compact base family: gemma-4-E4B-it -> g4-E4B, Qwen3.5-2B-Base -> q3.5-2B."""
    b = (base_model or "").split("/")[-1]
    return (b.replace("gemma-4-", "g4-").replace("Qwen3.5-", "q3.5-")
             .replace("-Base", "").replace("-it", "")) or "unknown"


def model_label(card: dict, lora: dict[str, Any] | None = None) -> str:
    """base + lora_r (+ version tag), never the digits in the repo name."""
    name = str(card.get("name") or card.get("repo") or "?")
    r = (lora or {}).get(name)
    ver = re.search(r"(v\d+(?:-\d+)?)$", name)
    out = base_short(card.get("base_model"))
    if r:
        out += f" r{r}"
    if ver:
        out += f" {ver.group(1)}"
    return out


def _f(x: Any, dec: int = 4) -> str:
    """Fixed-point text, or a dash for anything that is not a finite number."""
    return f"{x:.{dec}f}" if isinstance(x, (int, float)) and math.isfinite(x) else "—"


def _count(value: Any, dec: int, cls: str = "", unit: str = "") -> str:
    """A number that counts up via static/app.js (data-sdp=count). Static text if JS is off."""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return f'<tspan class="{cls}">—</tspan>'
    return (f'<tspan class="{cls}" data-sdp="count" data-count="{value}" data-dec="{dec}"'
            f'{f" data-unit={E(unit)!r}" if unit else ""}>{value:.{dec}f}{E(unit)}</tspan>')


def _answers(result: dict, qid: str) -> list[tuple[dict, dict | None]]:
    """[(card, answer_or_None)] in registry (card) order."""
    out = []
    for c in result.get("cards") or []:
        a = (c.get("answers") or {}).get(qid)
        out.append((c, a if isinstance(a, dict) else None))
    return out


def _options(answers: Iterable[dict | None], qtype: str) -> list[str]:
    """Option order: noul fixed (true, false); score = legend order; choice = first seen."""
    if qtype == "noul":
        return ["true", "false"]
    seen: list[str] = []
    for a in answers:
        if not a:
            continue
        if qtype == "score" and a.get("legend"):
            leg = a["legend"]
            order = [leg[k] for k in sorted(leg, key=lambda k: float(k) if str(k).lstrip("-").isdigit() else k)]
            for o in order:
                if o not in seen:
                    seen.append(o)
        for k in (a.get("probs") or {}):
            if k not in seen:
                seen.append(str(k))
    return seen


def _conf(a: dict | None) -> float | None:
    """Confidence for a bubble radius. noul has NO confidence field by design (schema.py), so
    the radius uses P(argmax) = max(p, 1-p) and the caption says so."""
    if not a:
        return None
    c = a.get("confidence")
    if isinstance(c, (int, float)):
        return float(c)
    if a.get("type") == "noul":
        probs = a.get("probs") or {}
        vals = [p for p in probs.values() if isinstance(p, (int, float))]
        return max(vals) if vals else None
    return None


def _radius(conf: float | None) -> float:
    """Bubble radius in px from a confidence (5..16); 6 when there is none."""
    if conf is None:
        return 6.0
    return 5.0 + 11.0 * max(0.0, min(1.0, conf))


def _title(s: str) -> str:
    """An SVG <title> (native tooltip)."""
    return f"<title>{E(s)}</title>"


# ---------------------------------------------------------------------------------
# 1. consensus bubble-up
# ---------------------------------------------------------------------------------


def consensus_svg(result: dict, qid: str, lora: dict[str, Any] | None = None) -> str:
    """Diagram 1: each model's argmax rises to its option (vote count + probability mass per
    column); score questions place bubbles on the ordered axis at the expected value."""
    rows = _answers(result, qid)
    answered = [(c, a) for c, a in rows if a and a.get("vote") is not None]
    qtype = next((a.get("type") for _, a in answered), None) or "choice"
    if qtype == "score":
        return _consensus_score(rows, answered, qid, lora)
    return _consensus_vote(rows, answered, qid, qtype, lora)


def _label_lines(card: dict, lora) -> tuple[str, str]:
    """('q3.5-2B', 'r16 v19') -- two short lines so seven nodes fit side by side."""
    lab = model_label(card, lora)
    parts = lab.split(" ", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _node_row(rows, lora, W, y, pad=24):
    """Bottom row of model nodes: numbered squares (the number is reused on bubbles and in the
    agreement matrix), labelled base + lora_r on two lines."""
    n = max(1, len(rows))
    step = (W - 2 * pad) / n
    xs = [pad + step * (i + 0.5) for i in range(n)]
    parts = []
    for i, (c, a) in enumerate(rows):
        x = xs[i]
        l1, l2 = _label_lines(c, lora)
        dead = a is None or a.get("vote") is None
        cls = "viz-node" + (" dead" if dead else "")
        tip = f"#{i + 1} {c.get('name')}\nbase {c.get('base_model')}"
        if dead:
            tip += f"\n{c.get('error') or 'no answer'}"
        parts.append(
            f'<g class="{cls}" transform="translate({x:.1f},{y:.1f})">{_title(tip)}'
            f'<rect x="-8" y="-8" width="16" height="16" rx="3"/>'
            f'<text class="viz-lab tiny" y="3" text-anchor="middle">{i + 1}</text>'
            f'<text class="viz-lab" y="24" text-anchor="middle">{E(l1)}</text>'
            f'<text class="viz-lab mute" y="36" text-anchor="middle">{E(l2)}</text>'
            + ('<text class="viz-lab mute" y="48" text-anchor="middle">no answer</text>' if dead else "")
            + "</g>"
        )
    return xs, "".join(parts)


def _consensus_vote(rows, answered, qid, qtype, lora):
    W, H = 720, 372
    pad = 24
    opts = _options([a for _, a in rows], qtype) or ["—"]
    K = len(opts)
    N = len(answered)
    col_w = (W - 2 * pad) / K
    col_x = [pad + col_w * (k + 0.5) for k in range(K)]
    y_nodes = H - 64
    y_land = 160          # where bubbles rest (second row at y_land + 30 when crowded)
    y_head = 16           # column header top
    head_h = 92

    votes = {o: [] for o in opts}
    mass = {o: 0.0 for o in opts}
    for c, a in answered:
        v = str(a.get("vote"))
        votes.setdefault(v, []).append((c, a))
        for o in opts:
            p = (a.get("probs") or {}).get(o)
            if isinstance(p, (int, float)):
                mass[o] += float(p)
    winner = max(opts, key=lambda o: (len(votes.get(o, [])), mass.get(o, 0.0))) if N else None

    out = [f'<svg class="viz viz-consensus" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="consensus bubble-up for {E(qid)}">']
    # columns
    for k, o in enumerate(opts):
        x0 = pad + col_w * k + 4
        w = col_w - 8
        win = o == winner
        nv = len(votes.get(o, []))
        tip = f"{o}: {nv} of {N} votes · probability mass Σp = {mass[o]:.4f} of {N}"
        # Fit the name to its column (12 px mono ≈ 7.4 px/char), leaving room for the "winner" label;
        # the full option always stays in the column's <title> tooltip.
        fit = max(4, int((w - 20 - (50 if win and N else 0)) / 7.4))
        shown = o if len(o) <= fit else o[: fit - 1] + "…"
        out.append(f'<g class="viz-col{" win" if win else ""}">{_title(tip)}'
                   f'<rect class="viz-colbox" x="{x0:.1f}" y="{y_head}" width="{w:.1f}" height="{head_h}" rx="8"/>'
                   f'<text class="viz-opt" x="{x0 + 10:.1f}" y="{y_head + 20}">{E(shown)}</text>')
        if win and N:
            out.append(f'<text class="viz-lab acc" x="{x0 + w - 10:.1f}" y="{y_head + 20}" text-anchor="end">winner</text>')
        # two stacked bars: votes (N of total) and mass (Σp of total)
        bw = w - 20
        for j, (lab, val, dec) in enumerate((("votes", nv, 0), ("Σp", mass[o], 2))):
            by = y_head + 34 + j * 26
            frac = (val / N) if N else 0.0
            out.append(
                f'<text class="viz-lab mute" x="{x0 + 10:.1f}" y="{by + 9}">{lab}</text>'
                f'<rect class="viz-track" x="{x0 + 10:.1f}" y="{by + 12}" width="{bw:.1f}" height="4" rx="2"/>'
                f'<rect class="viz-fill{" acc" if win else ""}" data-viz="bar" x="{x0 + 10:.1f}" y="{by + 12}" '
                f'width="0" data-w="{bw * frac:.1f}" height="4" rx="2"/>'
                f'<text class="viz-num" x="{x0 + w - 10:.1f}" y="{by + 9}" text-anchor="end">'
                f'{_count(val, dec)}<tspan class="mute"> / {N}</tspan></text>'
            )
        out.append("</g>")
    # dashed group around the landing zone
    out.append(f'<rect class="viz-group" x="{pad}" y="{y_head + head_h + 10}" width="{W - 2 * pad}" '
               f'height="{y_land - y_head - head_h + 52}" rx="8"/>')
    out.append(f'<text class="viz-lab mute" x="{pad + 8}" y="{y_head + head_h + 24}">argmax per model · bubble radius = confidence</text>')

    xs, nodes = _node_row(rows, lora, W, y_nodes, pad)
    # tracks + bubbles
    land_i = {o: 0 for o in opts}
    bubbles = []
    order = 0
    for i, (c, a) in enumerate(rows):
        if a is None or a.get("vote") is None:
            continue
        v = str(a.get("vote"))
        k = opts.index(v) if v in opts else 0
        n_in = len(votes.get(v, []))
        r = _radius(_conf(a))
        crowded = n_in * 34 > col_w - 16
        per_row = math.ceil(n_in / 2) if crowded else n_in
        slot = land_i[v]
        row_i, col_i = (slot % 2, slot // 2) if crowded else (0, slot)
        spacing = min(40.0, max(14.0, (col_w - 24) / max(1, per_row)))
        lx = col_x[k] - spacing * (per_row - 1) / 2 + spacing * col_i
        ly = y_land + row_i * 30
        land_i[v] += 1
        x0, y0 = xs[i], y_nodes
        out.append(f'<line class="viz-trackline" x1="{x0:.1f}" y1="{y0 - 10:.1f}" x2="{lx:.1f}" y2="{ly + r:.1f}"/>')
        conf = _conf(a)
        p = (a.get("probs") or {}).get(v)
        tip = (f"{c.get('name')} → {v}\np({v}) = {_f(p)}"
               + (f"\nconfidence = {_f(conf)}" if isinstance(a.get("confidence"), (int, float))
                  else f"\nP(argmax) = {_f(conf)} (noul has no confidence field)"))
        bubbles.append(
            f'<g class="viz-bubble{" win" if v == winner else ""}" data-viz="bubble" data-delay="{order * 80}" '
            f'style="--x0:{x0:.1f}px;--y0:{y0:.1f}px;--x1:{lx:.1f}px;--y1:{ly:.1f}px">'
            f'{_title(tip)}<circle r="{r:.1f}"/>'
            f'<text class="viz-lab tiny" y="3" text-anchor="middle">{i + 1}</text></g>'
        )
        order += 1
    out.append(nodes)
    out.extend(bubbles)
    out.append("</svg>")
    if N == 0:
        cap = "No model answered this question, so there is nothing to sum."
    else:
        cap = (f"{N} model{'s' if N != 1 else ''} · {winner}: {len(votes.get(winner, []))} of {N} votes, "
               f"Σp = {mass[winner]:.2f} of {N}"
               + (" · noul bubbles use P(argmax) because a yes/no answer has no confidence field." if qtype == "noul" else "."))
    return _figure("consensus", f"{qid} · sum of decisions", out, cap)


def _consensus_score(rows, answered, qid, lora):
    W, H = 720, 392
    pad = 24
    levels = _options([a for _, a in rows], "score") or ["—"]
    L = len(levels)
    N = len(answered)
    y_axis = 186
    y_nodes = H - 64
    y_head = 16
    head_h = 92
    ax0, ax1 = pad + 40, W - pad - 40
    def xof(v: float) -> float:
        return ax0 + (ax1 - ax0) * (v / max(1, L - 1))

    mass = {o: 0.0 for o in levels}
    evs: list[float] = []
    for c, a in answered:
        if isinstance(a.get("value"), (int, float)):
            evs.append(float(a["value"]))
        for o in levels:
            p = (a.get("probs") or {}).get(o)
            if isinstance(p, (int, float)):
                mass[o] += float(p)
    mean_ev = sum(evs) / len(evs) if evs else None
    top = max(levels, key=lambda o: mass[o]) if N else None

    out = [f'<svg class="viz viz-consensus" viewBox="0 0 {W} {H}" role="img" aria-label="expected-value axis for {E(qid)}">']
    # per-level mass columns
    col_w = (W - 2 * pad) / L
    for k, o in enumerate(levels):
        x0 = pad + col_w * k + 4
        w = col_w - 8
        win = o == top
        frac = (mass[o] / N) if N else 0.0
        tip = f"level {k}: {o} · probability mass Σp = {mass[o]:.4f} of {N}"
        out.append(f'<g class="viz-col{" win" if win else ""}">{_title(tip)}'
                   f'<rect class="viz-colbox" x="{x0:.1f}" y="{y_head}" width="{w:.1f}" height="{head_h - 30}" rx="8"/>'
                   f'<text class="viz-opt" x="{x0 + 10:.1f}" y="{y_head + 20}"><tspan class="mute">{k} · </tspan>{E(o)}</text>'
                   f'<text class="viz-lab mute" x="{x0 + 10:.1f}" y="{y_head + 43}">Σp</text>'
                   f'<rect class="viz-track" x="{x0 + 10:.1f}" y="{y_head + 46}" width="{w - 20:.1f}" height="4" rx="2"/>'
                   f'<rect class="viz-fill{" acc" if win else ""}" data-viz="bar" x="{x0 + 10:.1f}" y="{y_head + 46}" width="0" '
                   f'data-w="{(w - 20) * frac:.1f}" height="4" rx="2"/>'
                   f'<text class="viz-num" x="{x0 + w - 10:.1f}" y="{y_head + 43}" text-anchor="end">{_count(mass[o], 2)}'
                   f'<tspan class="mute"> / {N}</tspan></text></g>')
    # ordered axis
    out.append(f'<rect class="viz-group" x="{pad}" y="{y_head + head_h - 10}" width="{W - 2 * pad}" height="{y_axis - y_head - head_h + 88}" rx="8"/>')
    out.append(f'<line class="viz-axis" x1="{ax0}" y1="{y_axis + 48}" x2="{ax1}" y2="{y_axis + 48}"/>')
    for k, o in enumerate(levels):
        x = xof(k)
        anchor = "start" if k == 0 else ("end" if k == L - 1 else "middle")
        out.append(f'<line class="viz-axis" x1="{x:.1f}" y1="{y_axis + 44}" x2="{x:.1f}" y2="{y_axis + 52}"/>'
                   f'<text class="viz-lab mute" x="{x:.1f}" y="{y_axis + 64}" text-anchor="{anchor}">{k} · {E(o)}</text>')
    if mean_ev is not None:
        mx = xof(mean_ev)
        out.append(f'<g class="viz-mean" data-viz="mark">{_title(f"mean expected value across {N} models = {mean_ev:.4f}")}'
                   f'<line x1="{mx:.1f}" y1="{y_axis - 100}" x2="{mx:.1f}" y2="{y_axis + 48}"/>'
                   f'<text class="viz-num acc" x="{mx + 6:.1f}" y="{y_axis + 42}">mean {_count(mean_ev, 4)}</text></g>')

    xs, nodes = _node_row(rows, lora, W, y_nodes, pad)
    bubbles = []
    order = 0
    placed: list[tuple[float, float, float]] = []  # (x, y, r)
    for i, (c, a) in enumerate(rows):
        if a is None or not isinstance(a.get("value"), (int, float)):
            continue
        ev = float(a["value"])
        r = _radius(_conf(a))
        tx = xof(ev)                      # the TRUE expected value on the axis
        lx, ly = tx, y_axis
        # beeswarm: keep x when possible; when models cluster, step up, then sideways, and draw
        # a stem from the bubble to its exact position on the axis so nothing is misread.
        for dx, dy in ((0, 0), (0, -32), (30, 0), (-30, 0), (30, -32), (-30, -32), (0, -64), (30, -64), (-30, -64),
                       (60, 0), (-60, 0), (60, -32), (-60, -32), (60, -64), (-60, -64)):
            lx, ly = tx + dx, y_axis + dy
            if ax0 - 10 <= lx <= ax1 + 10 and not any((lx - px) ** 2 + (ly - py) ** 2 < (r + pr + 3) ** 2 for px, py, pr in placed):
                break
        placed.append((lx, ly, r))
        x0 = xs[i]
        out.append(f'<line class="viz-trackline" x1="{x0:.1f}" y1="{y_nodes - 10}" x2="{lx:.1f}" y2="{ly + r:.1f}"/>')
        out.append(f'<line class="viz-stem" x1="{lx:.1f}" y1="{ly + r:.1f}" x2="{tx:.1f}" y2="{y_axis + 48}"/>'
                   f'<line class="viz-stem tick" x1="{tx:.1f}" y1="{y_axis + 44}" x2="{tx:.1f}" y2="{y_axis + 52}"/>')
        tip = (f"{c.get('name')}\nexpected value = {ev:.4f}\nconfidence = {_f(a.get('confidence'))}\n"
               + " · ".join(f"p({o}) = {_f((a.get('probs') or {}).get(o))}" for o in levels))
        bubbles.append(
            f'<g class="viz-bubble" data-viz="bubble" data-delay="{order * 80}" '
            f'style="--x0:{x0:.1f}px;--y0:{y_nodes:.1f}px;--x1:{lx:.1f}px;--y1:{ly:.1f}px">'
            f'{_title(tip)}<circle r="{r:.1f}"/>'
            f'<text class="viz-lab tiny" y="3" text-anchor="middle">{i + 1}</text></g>')
        order += 1
    out.append(nodes)
    out.extend(bubbles)
    out.append("</svg>")
    if N == 0:
        cap = "No model answered this question, so there is no expected value to place."
    else:
        cap = (f"{N} model{'s' if N != 1 else ''}: each bubble sits at that model's expected value (radius = confidence; "
               f"a stem marks the exact value when bubbles are moved apart) · mean {mean_ev:.4f} on levels 0…{L - 1} · "
               f"most probability mass on {top} (Σp = {mass[top]:.2f} of {N}).")
    return _figure("consensus", f"{qid} · expected value", out, cap)


# ---------------------------------------------------------------------------------
# 2. distribution small-multiples
# ---------------------------------------------------------------------------------


def multiples_html(result: dict, qid: str, lora: dict[str, Any] | None = None) -> str:
    """Diagram 2: one tiny bar chart per option, one bar per model in registry order."""
    rows = _answers(result, qid)
    qtype = next((a.get("type") for _, a in rows if a), None) or "choice"
    opts = _options([a for _, a in rows], qtype)
    if not opts:
        return ""
    n = len(rows)
    W, H = 220, 120
    pad_l, pad_r, pad_t, pad_b = 10, 10, 24, 22
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    step = plot_w / max(1, n)
    bw = max(4.0, min(18.0, step * 0.6))
    cells = []
    for o in opts:
        vals = []
        for c, a in rows:
            p = (a.get("probs") or {}).get(o) if a else None
            vals.append(float(p) if isinstance(p, (int, float)) else None)
        present = [v for v in vals if v is not None]
        vmax = max(present) if present else None
        # Fit the name beside the right-aligned "max 0.0000" (≈74 px at 11 px mono); a truncated
        # name gets a <title> with the full option (the aria-label always carries it).
        fit = max(4, int((W - pad_l - pad_r - (74 if present else 0)) / 7.4))
        cut = len(o) > fit
        svg = [f'<svg class="viz viz-multiple" viewBox="0 0 {W} {H}" role="img" aria-label="p({E(o)}) per model">'
               + (_title(o) if cut else "")]
        svg.append(f'<text class="viz-opt" x="{pad_l}" y="15">{E(o[: fit - 1] + "…" if cut else o)}</text>')
        if present:
            svg.append(f'<text class="viz-num" x="{W - pad_r}" y="15" text-anchor="end">max {_count(vmax, 4)}</text>')
        # shared axis: 0, 0.5, 1
        for t in (0.0, 0.5, 1.0):
            y = pad_t + plot_h * (1 - t)
            svg.append(f'<line class="viz-grid" x1="{pad_l}" y1="{y:.1f}" x2="{W - pad_r}" y2="{y:.1f}"/>')
        svg.append(f'<text class="viz-lab tiny mute" x="{pad_l}" y="{pad_t + plot_h + 12}">models #1…{n} in registry order · y 0–1</text>')
        for i, ((c, a), v) in enumerate(zip(rows, vals)):
            x = pad_l + step * (i + 0.5) - bw / 2
            if v is None:
                tip = f"{c.get('name')}: no answer"
                svg.append(f'<g class="viz-barg dead">{_title(tip)}'
                           f'<rect class="viz-track" x="{x:.1f}" y="{pad_t + plot_h - 2}" width="{bw:.1f}" height="2"/></g>')
                continue
            h = plot_h * max(0.0, min(1.0, v))
            is_max = vmax is not None and v == vmax
            tip = f"{c.get('name')} · {model_label(c, lora)}\np({o}) = {v:.4f}"
            svg.append(f'<g class="viz-barg{" acc" if is_max else ""}">{_title(tip)}'
                       f'<rect class="viz-fill{" acc" if is_max else ""}" data-viz="rect" x="{x:.1f}" y="{pad_t + plot_h:.1f}" width="{bw:.1f}" '
                       f'height="0" data-h="{h:.1f}" data-y="{pad_t + plot_h - h:.1f}" rx="2"/></g>')
        svg.append("</svg>")
        cells.append(f'<div class="viz-cell">{"".join(svg)}</div>')
    n_ok = sum(1 for _, a in rows if a)
    cap = (f"p(option) for {n_ok} of {n} models (same order and numbers as the nodes above), one panel per option, "
           f"shared 0–1 axis; the tallest bar in each panel is the accent.")
    return _figure("multiples", f"{qid} · distributions", [f'<div class="viz-grid-cells">{"".join(cells)}</div>'], cap)


# ---------------------------------------------------------------------------------
# 3. confidence vs size
# ---------------------------------------------------------------------------------


def scatter_svg(result: dict, qid: str, lora: dict[str, Any] | None = None) -> str:
    """Diagram 3: confidence against active parameters (log x), with the 0.9 reference line."""
    rows = _answers(result, qid)
    qtype = next((a.get("type") for _, a in rows if a), None) or "choice"
    W, H = 720, 310
    pl, pr, pt, pb = 64, 24, 30, 48
    pw, ph = W - pl - pr, H - pt - pb
    pts = []
    for c, a in rows:
        ap = (c.get("active_params") or {}).get("active") if isinstance(c.get("active_params"), dict) else None
        conf = _conf(a)
        if isinstance(ap, (int, float)) and ap > 0 and conf is not None:
            pts.append((c, a, float(ap), conf))
    lo = min([p[2] for p in pts] + [1e9])
    hi = max([p[2] for p in pts] + [3e10])
    lo, hi = 10 ** math.floor(math.log10(lo)), 10 ** math.ceil(math.log10(hi))
    def xof(v: float) -> float:
        return pl + pw * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    def yof(v: float) -> float:
        return pt + ph * (1 - max(0.0, min(1.0, v)))
    ylab = "confidence" if qtype != "noul" else "P(argmax)"
    out = [f'<svg class="viz viz-scatter" viewBox="0 0 {W} {H}" role="img" aria-label="{ylab} vs active parameters">']
    # grid: y
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = yof(t)
        out.append(f'<line class="viz-grid" x1="{pl}" y1="{y:.1f}" x2="{W - pr}" y2="{y:.1f}"/>'
                   f'<text class="viz-lab mute" x="{pl - 8}" y="{y + 3:.1f}" text-anchor="end">{t:.2f}</text>')
    # grid: x decades + 2/5 minors
    d = int(math.log10(lo))
    while 10 ** d <= hi:
        for m in (1, 2, 5):
            v = m * 10 ** d
            if lo <= v <= hi:
                x = xof(v)
                out.append(f'<line class="viz-grid{" minor" if m != 1 else ""}" x1="{x:.1f}" y1="{pt}" x2="{x:.1f}" y2="{pt + ph}"/>')
                out.append(f'<text class="viz-lab mute" x="{x:.1f}" y="{pt + ph + 16}" text-anchor="middle">{_params(v)}</text>')
        d += 1
    out.append(f'<text class="viz-lab mute" x="{pl + pw / 2:.1f}" y="{H - 6}" text-anchor="middle">active parameters (log) — MoE counts only the experts that run</text>')
    out.append(f'<text class="viz-lab mute" transform="translate(14,{pt + ph / 2:.1f}) rotate(-90)" text-anchor="middle">{ylab}</text>')
    # the one reference line
    yr = yof(CONF_REF)
    out.append(f'<g class="viz-ref" data-viz="mark">{_title(CONF_REF_LABEL)}'
               f'<line x1="{pl}" y1="{yr:.1f}" x2="{W - pr}" y2="{yr:.1f}"/>'
               f'<line x1="{W - pr - 24}" y1="12" x2="{W - pr}" y2="12"/>'
               f'<text class="viz-lab acc" x="{W - pr - 30}" y="16" text-anchor="end">{E(CONF_REF_LABEL)}</text></g>')
    # points carry their node NUMBER only (full labels live in the legend below), nudged apart
    # vertically when two models share a spot -- the three Qwen3.5-2B deciders have one x.
    placed: list[tuple[float, float]] = []
    for i, (c, a, ap, conf) in enumerate(pts):
        x, y = xof(ap), yof(conf)
        idx = next(j for j, (cc, _a) in enumerate(rows) if cc is c) + 1
        lx, ly = x + 9, y + 4
        for _ in range(10):
            if any(abs(lx - px) < 16 and abs(ly - py) < 11 for px, py in placed):
                ly += 11
            else:
                break
        placed.append((lx, ly))
        tip = (f"#{idx} {c.get('name')} · {model_label(c, lora)}\nactive params = {ap:,.0f}\n{ylab} = {conf:.4f}\nargmax = {a.get('vote')}")
        out.append(f'<g class="viz-point" data-viz="point" data-delay="{i * 60}" style="--x:{x:.1f}px;--y:{y:.1f}px">{_title(tip)}'
                   f'<circle r="5"/><text class="viz-lab" x="{lx - x:.1f}" y="{ly - y:.1f}">{idx}</text></g>')
    if not pts:
        out.append(f'<text class="viz-lab mute" x="{pl + pw / 2:.1f}" y="{pt + ph / 2:.1f}" text-anchor="middle">no (active params, {ylab}) pair to plot</text>')
    out.append("</svg>")
    legend = " · ".join(f'<b>{j + 1}</b> {E(model_label(c, lora))}' for j, (c, _a) in enumerate(rows))
    out.append(f'<p class="viz-legend">{legend}</p>')
    cap = (f"{len(pts)} point{'s' if len(pts) != 1 else ''}: {ylab} against active parameters; the dashed line is the one "
           f"calibration figure the strands-decider README publishes (evaluation/results.md)"
           + (" · noul answers carry no confidence field, so y is P(argmax)." if qtype == "noul" else "."))
    return _figure("scatter", f"{qid} · {ylab} vs size", out, cap)


def _params(v: float) -> str:
    if v >= 1e9:
        s = f"{v / 1e9:g}B"
    elif v >= 1e6:
        s = f"{v / 1e6:g}M"
    else:
        s = f"{v:g}"
    return s


# ---------------------------------------------------------------------------------
# 4. agreement matrix
# ---------------------------------------------------------------------------------


def agreement_svg(result: dict, qid: str, lora: dict[str, Any] | None = None) -> str:
    """Diagram 4: NxN grid, a cell filled when two models share an argmax (3+ models only)."""
    rows = _answers(result, qid)
    answered = [(i + 1, c, a) for i, (c, a) in enumerate(rows) if a and a.get("vote") is not None]
    n = len(answered)
    if n < 3:
        return ""
    cell = 30
    lab_w, lab_h = 290, 26
    W = lab_w + cell * n + 12
    H = lab_h + cell * n + 12
    out = [f'<svg class="viz viz-matrix" viewBox="0 0 {W} {H}" role="img" aria-label="agreement matrix for {E(qid)}">']
    pairs = 0
    agree = 0
    for r, (ii, ci, ai) in enumerate(answered):
        li = model_label(ci, lora)
        out.append(f'<text class="viz-lab" x="{lab_w - 10}" y="{lab_h + cell * r + cell / 2 + 4:.1f}" text-anchor="end">'
                   f'<tspan class="mute">{ii} · </tspan>{E(li)}<tspan class="acc"> → {E(str(ai.get("vote")))}</tspan></text>')
        out.append(f'<text class="viz-lab mute" x="{lab_w + cell * r + cell / 2:.1f}" y="{lab_h - 9}" text-anchor="middle">{ii}</text>')
        for q, (jj, cj, aj) in enumerate(answered):
            same = str(ai.get("vote")) == str(aj.get("vote"))
            x, y = lab_w + cell * q, lab_h + cell * r
            if q > r:
                pairs += 1
                agree += int(same)
            cls = "diag" if r == q else ("same" if same else "diff")
            tip = (f"#{ii} {ci.get('name')} → {ai.get('vote')}\n#{jj} {cj.get('name')} → {aj.get('vote')}\n"
                   + ("same model" if r == q else ("same argmax" if same else "different argmax")))
            out.append(f'<g class="viz-mcell {cls}" data-viz="cell" data-delay="{(r + q) * 40}">{_title(tip)}'
                       f'<rect x="{x + 2}" y="{y + 2}" width="{cell - 4}" height="{cell - 4}" rx="3"/>'
                       + "</g>")
    out.append(f'<rect class="viz-group" x="{lab_w - 2}" y="{lab_h - 2}" width="{cell * n + 4}" height="{cell * n + 4}" rx="8"/>')
    out.append("</svg>")
    cap = (f"{agree} of {pairs} model pairs share an argmax; a filled cell means the row and column models voted the same; "
           f"each row label ends with that model's vote; numbers are the node numbers above.")
    return _figure("matrix", f"{qid} · agreement", out, cap)


# ---------------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------------


def _figure(kind: str, title: str, body: list[str], caption: str) -> str:
    """<figure> wrapper with title and caption around one diagram."""
    return (f'<figure class="viz-fig viz-{kind}-fig" data-viz="fig">'
            f'<figcaption class="viz-title">{E(title)}</figcaption>'
            + "".join(body)
            + f'<p class="viz-cap">{E(caption)}</p></figure>')


def render(result: dict, lora: dict[str, Any] | None = None, *, heading: bool = False) -> str:
    """Every diagram for every question in `result`. Empty string when there is nothing to draw."""
    if not isinstance(result, dict) or result.get("error") or not result.get("cards"):
        return ""
    qids = list(result.get("question_ids") or [])
    if not qids:
        seen: list[str] = []
        for c in result["cards"]:
            for q in (c.get("answers") or {}):
                if q not in seen:
                    seen.append(q)
        qids = seen
    blocks = []
    for qid in qids:
        parts = [consensus_svg(result, qid, lora), multiples_html(result, qid, lora),
                 scatter_svg(result, qid, lora), agreement_svg(result, qid, lora)]
        parts = [p for p in parts if p]
        if parts:
            specs = list(result.get("question_specs") or [])
            spec = next((s for s in specs if (s.get("id") or "") == qid), None)
            if spec is None and len(specs) == len(qids):  # specs without ids: positional (runner.build_questions order)
                spec = specs[qids.index(qid)]
            instr = (spec or {}).get("instructions") or ""
            blocks.append(f'<section class="viz-q" data-qid="{E(qid)}">'
                          f'<h3 class="viz-qh"><span class="mono">{E(qid)}</span>'
                          + (f' <span class="viz-instr">{E(instr)}</span>' if instr else "")
                          + "</h3>" + "".join(parts) + "</section>")
    if not blocks:
        return ""
    head = ('<div class="sdp sdp-section"><h2>Decisions, summed</h2>'
            '<span class="sub">every model answers the same question; its argmax rises to the option it chose</span></div>'
            if heading else "")
    return f'<div class="viz-root" data-viz="root">{head}{"".join(blocks)}</div>'


# ---------------------------------------------------------------------------------
# demo data (the real golden + one synthetic 7-model result) and the --demo CLI
# ---------------------------------------------------------------------------------

# The 7 public StrandsAgents deciders (base, active params, lora_r) at the time of writing.
# DEMO ONLY: the app reads all of this live from the registry; here it supplies labels and
# x-positions for the synthetic layout preview.
PUBLIC_MODELS = [
    ("strands-decider-2B-hobson-v19", "Qwen/Qwen3.5-2B-Base", 2_274_069_824, 16),
    ("strands-decider-2B-hobson-v21", "Qwen/Qwen3.5-2B-Base", 2_274_069_824, 16),
    ("strands-decider-2B-qwen3.5-v1-2610", "Qwen/Qwen3.5-2B-Base", 2_274_069_824, 48),
    ("strands-decider-E2B-gemma4-v1-2610", "google/gemma-4-E2B-it", 5_120_000_000, 48),
    ("strands-decider-E4B-gemma4-v1-2610", "google/gemma-4-E4B-it", 8_000_000_000, 48),
    ("strands-decider-12B-gemma4-v1-2610", "google/gemma-4-12B-it", 11_960_000_000, 48),
    ("strands-decider-26B-A4B-gemma4-v1-2610", "google/gemma-4-26B-A4B-it", 4_395_325_006, 48),
]


def _norm(ans: dict) -> dict:
    """A copy of runner.normalise_answer's shape so the demo needs no import of model code."""
    t = ans.get("type")
    if t == "noul":
        p = ans["noul"]
        return {"type": "noul", "display": f"{p:.4f}", "value": p, "probs": {"true": p, "false": round(1 - p, 4)},
                "confidence": None, "vote": "true" if p >= 0.5 else "false"}
    if t == "choice":
        return {"type": "choice", "display": ans["choice"], "value": ans["choice"], "probs": dict(ans["probabilities"]),
                "confidence": ans.get("confidence"), "vote": ans["choice"]}
    legend = ans.get("legend") or {}
    probs = {legend.get(k, k): v for k, v in ans["probabilities"].items()}
    top = max(ans["probabilities"].items(), key=lambda kv: kv[1])[0]
    return {"type": "score", "display": f"{ans['score']:.4f}", "value": ans["score"], "probs": probs,
            "confidence": ans.get("confidence"), "legend": legend, "vote": legend.get(top, top)}


def golden_result(golden: dict) -> dict:
    """A golden.json recording (one real model, three real answers) -> the answer() dict shape."""
    var = golden["variants"]["brief_no_trailing_space"]
    repo = golden["repo"]
    name = repo.split("/")[-1]
    card = {"repo": repo, "name": name, "base_model": "Qwen/Qwen3.5-2B-Base", "error": None,
            "active_params": {"active": 2_274_069_824, "total": 2_274_069_824, "moe": False},
            "answers": {qid: _norm(a) for qid, a in var["answers"].items()}}
    return {"error": None, "cards": [card], "question_ids": list(var["answers"].keys()),
            "question_specs": [{"id": "noul_0", "type": "noul", "instructions": "Does this convey urgency?"},
                               {"id": "choice_0", "type": "choice", "instructions": "Which team should handle it?", "options": ["billing", "sales", "retail"]},
                               {"id": "score_0", "type": "score", "instructions": "How upset is the writer?", "options": ["calm", "frustrated", "depressed"]}]}


def synthetic_result() -> dict:
    """SEVEN public models, SYNTHETIC numbers (deterministic, labelled synthetic in the demo page).
    Exists only so the layout can be screenshotted without loading 99 GiB of weights."""
    cards = []
    nouls = [0.8287, 0.8753, 0.8581, 0.6120, 0.4410, 0.9120, 0.9530]
    choices = [("billing", (0.7728, 0.1213, 0.1060), 0.6591), ("billing", (0.6410, 0.2210, 0.1380), 0.5520),
               ("billing", (0.7110, 0.1700, 0.1190), 0.6020), ("sales", (0.3810, 0.4420, 0.1770), 0.3310),
               ("billing", (0.5230, 0.3110, 0.1660), 0.4120), ("billing", (0.8910, 0.0610, 0.0480), 0.8420),
               ("billing", (0.9210, 0.0450, 0.0340), 0.9050)]
    scores = [((0.1958, 0.5738, 0.2303), 0.5091), ((0.2500, 0.6000, 0.1500), 0.5400), ((0.1200, 0.6400, 0.2400), 0.5900),
              ((0.4100, 0.4500, 0.1400), 0.3800), ((0.3000, 0.5200, 0.1800), 0.4600), ((0.0800, 0.7200, 0.2000), 0.7000),
              ((0.0500, 0.6900, 0.2600), 0.7400)]
    legend = {"0": "calm", "1": "frustrated", "2": "depressed"}
    for i, (name, base, active, _r) in enumerate(PUBLIC_MODELS):
        ch, (pb, ps, pr), cc = choices[i]
        sp, sc = scores[i]
        ev = sum(k * p for k, p in enumerate(sp))
        cards.append({
            "repo": f"StrandsAgents/{name}", "name": name, "base_model": base, "error": None,
            "active_params": {"active": active, "moe": "A4B" in name},
            "answers": {
                "noul_0": _norm({"type": "noul", "noul": nouls[i]}),
                "choice_0": _norm({"type": "choice", "choice": ch, "probabilities": {"billing": pb, "sales": ps, "retail": pr}, "confidence": cc}),
                "score_0": _norm({"type": "score", "score": round(ev, 4), "legend": legend,
                                  "probabilities": {"0": sp[0], "1": sp[1], "2": sp[2]}, "confidence": sc}),
            },
        })
    # one model that failed, so the "no answer" path is visible
    cards.append({"repo": "StrandsAgents/strands-decider-E2B-gemma4-v1-2610", "name": "strands-decider-E2B-gemma4-v1-2610 (image run)",
                  "base_model": "google/gemma-4-E2B-it", "error": "image ignored: reads text only", "answers": {},
                  "active_params": {"active": 5_120_000_000}})
    return {"error": None, "cards": cards, "question_ids": ["noul_0", "choice_0", "score_0"],
            "question_specs": [{"id": "noul_0", "type": "noul", "instructions": "Does this convey urgency?"},
                               {"id": "choice_0", "type": "choice", "instructions": "Which team should handle it?", "options": ["billing", "sales", "retail"]},
                               {"id": "score_0", "type": "score", "instructions": "How upset is the writer?", "options": ["calm", "frustrated", "depressed"]}]}


def demo_lora() -> dict[str, int]:
    """{name: lora_r} for the demo models."""
    return {name: r for name, _b, _a, r in PUBLIC_MODELS}


def merge_results(results: list[dict]) -> dict | None:
    """Several answer() dicts (one question each, same models) -> one dict with every question.
    Cards are matched by name; the first result's card order wins. Used for live captures in the demo."""
    results = [r for r in results if isinstance(r, dict) and r.get("cards")]
    if not results:
        return None
    base = results[0]
    cards = {c["name"]: {**c, "answers": dict(c.get("answers") or {})} for c in base["cards"]}
    qids = list(base.get("question_ids") or [])
    specs = list(base.get("question_specs") or [])
    for r in results[1:]:
        rq = list(r.get("question_ids") or [])
        rs = list(r.get("question_specs") or [])
        qids += [q for q in rq if q not in qids]
        specs += [{**sp, "id": sp.get("id") or (rq[i] if i < len(rq) else "")} for i, sp in enumerate(rs)]
        for c in r["cards"]:
            if c["name"] in cards:
                cards[c["name"]]["answers"].update(c.get("answers") or {})
    for i, sp in enumerate(specs):
        if not sp.get("id") and i < len(qids):
            sp["id"] = qids[i]
    return {**base, "cards": list(cards.values()), "question_ids": qids, "question_specs": specs}


def demo_results(golden: dict | None, live: list[dict] | None = None) -> list[tuple[str, dict]]:
    """(title, result) pairs for the demo page: golden recording, live captures, synthetic layout."""
    out = []
    if golden:
        out.append(("golden — the REAL result of a golden.json recording (1 model)", golden_result(golden)))
    merged = merge_results(live or [])
    if merged:
        out.append((f"live — REAL answers of all {len(merged['cards'])} public models captured from the Space "
                    f"on {merged.get('device', '?')}", merged))
    out.append(("synthetic — 7 public models + one failed card, INVENTED numbers for layout only (not a measurement)", synthetic_result()))
    return out


def demo_page(golden: dict | None, static_dir: str = "static", live: list[dict] | None = None) -> str:
    """Standalone HTML: tokens.css + viz.css/js + app.js (the shared counter)."""
    lora = demo_lora()
    sections = []
    for title, res in demo_results(golden, live):
        sections.append(f'<section class="viz-demo-block"><h2 class="viz-demo-h">{E(title)}</h2>{render(res, lora, heading=False)}</section>')
    s = static_dir.rstrip("/")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>viz demo · strands decider playground</title>
<link rel="stylesheet" href="{s}/tokens.css"><link rel="stylesheet" href="{s}/viz.css">
<style>body{{margin:0;background:var(--bg);color:var(--text);font-family:var(--sr-text)}}
.viz-demo{{max-width:1120px;margin:0 auto;padding:24px}}
.viz-demo-h{{font:600 .95rem/1.3 var(--sr-mono);color:var(--fg);letter-spacing:.01em;margin:32px 0 8px}}
.viz-toggle{{position:fixed;top:12px;right:12px;font:500 .72rem var(--sr-mono);background:var(--chip-bg);color:var(--text);border:1px solid var(--hair);border-radius:999px;padding:6px 12px;cursor:pointer}}</style>
</head><body><button class="viz-toggle" onclick="document.documentElement.classList.toggle('dark');document.body.classList.toggle('dark')">paper / dark</button>
<main class="viz-demo">{"".join(sections)}</main>
<script src="{s}/app.js"></script><script src="{s}/viz.js"></script>
<script>if(location.search.indexOf('dark')>=0){{document.documentElement.classList.add('dark');document.body.classList.add('dark')}}</script>
</body></html>"""


def _main(argv: list[str]) -> int:  # pragma: no cover -- CLI only
    """python viz.py --demo [--golden golden.json] [--live DIR] [--static static] > demo.html

    --golden  a golden.json recording (variants.brief_no_trailing_space.answers), optional
    --live    a directory of captured result dicts (*.json, either the dict itself or {"raw": dict})
    --static  path to static/ as the demo page should reference it (default "static")
    """
    import glob
    import json
    import os
    import sys
    if "--demo" not in argv:
        print(_main.__doc__, file=sys.stderr)
        return 2
    golden_path = argv[argv.index("--golden") + 1] if "--golden" in argv else None
    static = argv[argv.index("--static") + 1] if "--static" in argv else "static"
    golden = json.load(open(golden_path)) if golden_path and os.path.exists(golden_path) else None
    live = []
    if "--live" in argv:
        for fp in sorted(glob.glob(os.path.join(argv[argv.index("--live") + 1], "*.json"))):
            raw = json.load(open(fp))
            live.append(raw.get("raw") if isinstance(raw, dict) and "raw" in raw else raw)
    sys.stdout.write(demo_page(golden, static, live))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(_main(sys.argv[1:]))
