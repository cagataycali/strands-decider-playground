"""The landing sections: hero, how-to-use, model row, footer.

Pure HTML strings in the strandsagents.com voice; `app.py` drops them into gr.HTML above and
below the ask bar. Nothing here recomputes a model answer.

Every number on the page is either read from the registry at build time (model count, base
model, LoRA rank, vision) or CITED to the strands-decider README with a link next to it. A model the README table does not know gets "no README figure".

Standalone preview:  python landing.py --demo [--dark] > demo.html
"""
from __future__ import annotations

import base64
import html as _html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

STATIC = Path(__file__).parent / "static"
REPO_URL = "https://github.com/strands-labs/strands-decider"
HF_ORG_URL = "https://huggingface.co/StrandsAgents"
DISCORD_URL = "https://discord.gg/Wa4CQrxsP"  # README "Community"
ARCH_URL = f"{REPO_URL}/blob/main/docs/architecture.md#the-architecture"
AGENT_EXAMPLE_URL = f"{REPO_URL}/blob/main/examples/strands/README.md"
DOCS_URL = f"{REPO_URL}/tree/main/docs"
PYPI_URL = "https://pypi.org/project/strands-decider/"

PIP_PYPI = "pip install strands-decider"
PIP_MAIN = 'pip install "strands-decider @ git+https://github.com/strands-labs/strands-decider"'
GOLDEN_REPO = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
ASK_CMD = (f"strands-decider ask {GOLDEN_REPO} \\\n"
           '  --state "Help! My payouts have been failing for 3 days!" \\\n'
           '  --choice "Which team should handle this?=billing,sales,retail" \\\n'
           '  --noul "Does this convey urgency?" \\\n'
           '  --score "How frustrated is the writer?=calm,frustrated,depressed"')
ASK_OUT = ("noul_0 noul = 0.858\n"
           "choice_0 -> billing (confidence 0.659)\n"
           "  billing                  0.773\n"
           "  sales                    0.121\n"
           "  retail                   0.106\n"
           "score_0 score = 1.03 (confidence 0.509)\n"
           "  0: calm                                     0.196\n"
           "  1: frustrated                               0.574\n"
           "  2: depressed                                0.230")
SERVE_CMD = f"strands-decider serve {GOLDEN_REPO} --port 8000"
CURL_CMD = ("curl -s localhost:8000/v1/systemone \\\n"
            "  -H 'content-type: application/json' \\\n"
            "  -d '{\"state\": \"Help! My payouts have been failing for 3 days!\",\n"
            "       \"questions\": {\"is_urgent\": {\"type\": \"noul\", \"instructions\": \"Does this convey urgency?\"}}}'")

# README tables (strands-labs/strands-decider, main, read 2026-10-10): JevBench public set, 231 tasks.
# "Gemma 4 models" table + "Performance" table. Keyed by the short repo name. CITED, never recomputed.
README_EVALS: dict[str, dict[str, Any]] = {
    "strands-decider-2B-qwen3.5-v1-2610": {"jevbench": 180, "brier": 0.280, "anchor": "#performance"},
    "strands-decider-2B-hobson-v21": {"jevbench": 176, "brier": 0.323, "anchor": "#performance"},
    "strands-decider-2B-hobson-v19": {"jevbench": 167, "brier": 0.342, "anchor": "#performance"},
    "strands-decider-E2B-gemma4-v1-2610": {"jevbench": 180, "brier": 0.313, "anchor": "#gemma-4-models"},
    "strands-decider-E4B-gemma4-v1-2610": {"jevbench": 187, "brier": 0.243, "anchor": "#gemma-4-models"},
    "strands-decider-12B-gemma4-v1-2610": {"jevbench": 200, "brier": 0.173, "anchor": "#gemma-4-models"},
    "strands-decider-26B-A4B-gemma4-v1-2610": {"jevbench": 207, "brier": 0.162, "anchor": "#gemma-4-models"},
}
JEVBENCH_N = 231


def E(s: Any) -> str:
    """HTML-escape anything (None -> empty string)."""
    return _html.escape("" if s is None else str(s), quote=True)


def _static(name: str) -> str:
    """Text of a file under static/."""
    return (STATIC / name).read_text(encoding="utf-8")


def _svg(name: str, cls: str, extra: str = "") -> str:
    """Inline an SVG from static/ with a class and aria-hidden."""
    raw = re.sub(r"<!--.*?-->", "", _static(name), flags=re.S).strip()
    return re.sub(r"<svg\b", f'<svg class="{cls}" aria-hidden="true" {extra}', raw, count=1)


LANDING_CSS = _static("landing.css")
LANDING_JS = _static("landing.js")
MARK = _svg("img/mark.svg", "ld-mark")
WORDMARK = _svg("img/wordmark.svg", "ld-wordmark")


# ---------------------------------------------------------------------------------
# registry helpers (small duplicates of app.py's, on purpose: this module must not import app.py)
# ---------------------------------------------------------------------------------

_LORA_R: dict[str, Any] = {}


def lora_r(info: Any) -> Any:
    """LoRA rank from the decider config (a local Hub-cache hit after the registry read it)."""
    repo = getattr(info, "repo", None)
    if repo in _LORA_R:
        return _LORA_R[repo]
    val = None
    try:
        cfg_file = getattr(info, "config_file", None)
        if cfg_file:
            from huggingface_hub import hf_hub_download
            cfg = json.load(open(hf_hub_download(repo, cfg_file, token=os.environ.get("HF_TOKEN") or None)))
            val = cfg.get("lora_r") or cfg.get("r")
    except Exception:  # noqa: BLE001 -- a label must never block the page
        val = None
    _LORA_R[repo] = val
    return val


def base_short(info: Any) -> str:
    """Base model name without the org prefix."""
    return ((getattr(info, "base_model", "") or "").split("/")[-1]) or "unknown base"


def model_label(info: Any) -> str:
    """Card label: base + LoRA rank."""
    r = lora_r(info)
    return base_short(info) + (f" · LoRA r {r}" if r else "")


# ---------------------------------------------------------------------------------
# 1. hero
# ---------------------------------------------------------------------------------

def _copy_btn(text: str, cls: str = "sr-copy") -> str:
    """A copy-to-clipboard button; static/landing.js handles the click."""
    return f'<button type="button" class="{cls}" data-ld-copy="{E(text)}" aria-label="copy to clipboard">copy</button>'


# the concept figure --------------------------------------------------------------------
FIG_W, FIG_H = 620, 386
FIG_STATE = ["ticket: charged twice", "plan: pro · 2 years"]
# the three question shapes: (chip label, question, options, probabilities) -- illustrative numbers
FIG_SHAPES: list[tuple[str, str, list[str], list[float]]] = [
    ("yes / no", "refund the 2nd charge?", ["yes", "no"], [0.86, 0.14]),
    ("pick one", "which team takes it?", ["billing", "sales", "retail"], [0.66, 0.21, 0.13]),
    ("rate 0-4", "how urgent, 0-4?", ["0", "1", "2", "3", "4"], [0.31, 0.42, 0.19, 0.06, 0.02]),
]
_SIZE_RE = re.compile(r"(E\dB|\d+(?:\.\d+)?B(?:-A\d+B)?)", re.I)
_FAMILY_RE = re.compile(r"^(gemma|qwen|llama|mistral|phi|smol)[-_]?(\d+(?:\.\d+)?)?", re.I)


def base_family(base_model: str) -> str:
    """'google/gemma-4-E2B-it' -> 'Gemma 4', 'Qwen/Qwen3.5-2B-Base' -> 'Qwen3.5'; else the first name token."""
    short = (base_model or "").split("/")[-1]
    m = _FAMILY_RE.match(short)
    if not m:
        return short.split("-")[0] or "unknown base"
    fam = m.group(1).capitalize()
    ver = m.group(2) or ""
    sep = " " if short[len(m.group(1)):len(m.group(1)) + 1] in "-_" else ""
    return f"{fam}{sep}{ver}".strip()


def _size_key(size: str) -> float:
    m = re.match(r"E?(\d+(?:\.\d+)?)", size)
    return float(m.group(1)) if m else 999.0


def family_tiles(registry: list[Any]) -> list[dict[str, Any]]:
    """One tile per listed model, read from the registry (base, repo) and the vendored baseline (correct/n).

    size = the parameter token of the base ('E2B', '12B', '26B-A4B', '2B'); variant = what is left of the
    repo name once the prefix, the size and the base tokens are removed ('v1', 'hobson v19'); ordered by
    base family, then size, then the plain 'v1' release before named variants."""
    tiles: list[dict[str, Any]] = []
    for info in registry:
        name = getattr(info, "name", "") or getattr(info, "repo", "").split("/")[-1]
        base = getattr(info, "base_model", "") or ""
        short = base.split("/")[-1]
        m = _SIZE_RE.search(short) or _SIZE_RE.search(name)
        size = m.group(1) if m else "?"
        rest = name.replace("strands-decider-", "")
        if size != "?":
            rest = rest.replace(size, "")
        toks = [t for t in rest.split("-") if t and not _FAMILY_RE.match(t) and not re.fullmatch(r"\d{4}", t)]
        variant = " ".join(toks) or "v1"
        ev = BASELINE_EVALS.get(name) or {}
        tiles.append({"name": name, "repo": f"https://huggingface.co/{getattr(info, 'repo', '') or name}",
                      "base": base_family(base), "size": size, "variant": variant,
                      "correct": ev.get("jevbench"), "n": ev.get("n")})
    tiles.sort(key=lambda t: (t["base"], _size_key(t["size"]), 0 if t["variant"] == "v1" else 1, t["variant"]))
    return tiles


def _mark_symbol() -> str:
    """The frog mark as a <symbol> in currentColor, used once per decider tile."""
    body = re.sub(r"<svg[^>]*>|</svg>", "", MARK_SVG_RAW).replace('fill="#00cc60"', 'fill="currentColor"')
    return f'<symbol id="ld-frog" viewBox="0 0 32 32">{body}</symbol>'


MARK_SVG_RAW = (STATIC / "img" / "mark.svg").read_text() if (STATIC / "img" / "mark.svg").exists() else ""


def figure_svg(registry: list[Any] | None = None) -> str:
    """The concept, for anyone: state + question -> decider -> a probability per option, in three question
    shapes; then the family of public deciders (from the registry, bars = JevBench correct/231 from the
    vendored baseline) and the playground's promise as the last frame.

    Pure viewBox geometry; every colour is a token (static/landing.css .ld-fig rules), so it inherits either
    palette. Motion is CSS keyframes in landing.css: four 1.5 s slots, paused on hover, static under
    prefers-reduced-motion -- the static frame is what docs/img/decider-concept.svg exports."""
    tiles = family_tiles(registry or [])
    g: list[str] = []
    g.append(f'<svg class="ld-fig" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {FIG_W} {FIG_H}" role="img" aria-labelledby="ld-fig-t ld-fig-d">')
    g.append('<title id="ld-fig-t">What a Strands decider does</title>')
    g.append('<desc id="ld-fig-d">A decider reads a piece of state and a question and answers with one probability per option '
             'instead of generating text: yes or no, pick one, or rate on a scale. The public StrandsAgents deciders, '
             'grouped by base model, all answer the same question in the playground.</desc>')
    g.append("<defs>" + _mark_symbol() + "</defs>")
    # --- row 1: the concept
    M = 16
    LX, LW = M, 196
    CX, CW = 252, 116
    RX, RW = 404, FIG_W - M - 404
    TOP, CARD_H = 30, 112
    g.append(f'<text x="{LX}" y="{TOP - 12}" class="cap">state + question</text>')
    g.append(f'<text x="{RX}" y="{TOP - 12}" class="cap">a probability per option</text>')
    g.append(f'<rect x="{LX}" y="{TOP}" width="{LW}" height="{CARD_H}" rx="8" class="box"/>')
    y = TOP + 22
    for line in FIG_STATE:
        g.append(f'<text x="{LX + 12}" y="{y}" class="mono muted">{E(line)}</text>')
        y += 17
    qy = y + 10
    for i, (_chip, q, opts, ps) in enumerate(FIG_SHAPES):
        g.append(f'<g class="frame f{i}"><text x="{LX + 12}" y="{qy}" class="mono fg">{E(q)}</text>'
                 f'<text x="{LX + 12}" y="{qy + 20}" class="mono">{E(" · ".join(opts))}</text>')
        n, bh = len(opts), 12
        gap = min(14, (CARD_H - 16 - n * bh) / max(n - 1, 1))
        y0 = TOP + (CARD_H - (n * bh + (n - 1) * gap)) / 2
        best = max(range(n), key=lambda k: ps[k])
        lab_w = 56
        track = RW - lab_w - 44
        for k, (o, p) in enumerate(zip(opts, ps)):
            yy = y0 + k * (bh + gap)
            cls = " best" if k == best else ""
            g.append(f'<text x="{RX + lab_w - 8}" y="{yy + bh - 2}" class="mono lab{cls}">{E(o)}</text>'
                     f'<rect x="{RX + lab_w}" y="{yy}" width="{track}" height="{bh}" rx="3" class="track"/>'
                     f'<rect x="{RX + lab_w}" y="{yy}" width="{track * p:.1f}" height="{bh}" rx="3" class="fill{cls}"/>'
                     f'<text x="{RX + lab_w + track + 8}" y="{yy + bh - 2}" class="mono val{cls}">{p:.2f}</text>')
        g.append("</g>")
    cy = TOP + CARD_H + 20
    x = LX
    for i, (chip, *_rest) in enumerate(FIG_SHAPES):
        w = round(16 + len(chip) * 6.8)
        g.append(f'<g class="chip c{i}"><rect x="{x}" y="{cy - 11}" width="{w}" height="18" rx="9" class="chipbox"/>'
                 f'<text x="{x + w / 2:.0f}" y="{cy + 2}" class="mono chiplab">{E(chip)}</text></g>')
        x += w + 6
    DY, DH = TOP + 16, 80
    g.append(f'<rect x="{CX}" y="{DY}" width="{CW}" height="{DH}" rx="8" class="box"/>'
             f'<use href="#ld-frog" x="{CX + CW / 2 - 14}" y="{DY + 12}" width="28" height="28" class="frog"/>'
             f'<text x="{CX + CW / 2}" y="{DY + 62}" class="mono name">decider</text>')
    my = DY + DH / 2
    g.append(f'<path d="M{LX + LW} {my} H{CX - 2}" class="edge"/><path d="M{CX - 6} {my - 3} l4 3 -4 3" class="edge"/>'
             f'<path d="M{CX + CW} {my} H{RX - 2}" class="edge sig"/><path d="M{RX - 6} {my - 3} l4 3 -4 3" class="edge sig"/>')
    # --- row 2: the family
    CAPY, BUSY, FY = 200, 218, 232
    g.append(f'<text x="{M}" y="{CAPY}" class="cap">the family today · {len(tiles)} public deciders</text>'
             f'<text x="{FIG_W - M}" y="{CAPY}" class="cap right promise">one question · every decider</text>')
    dcx = CX + CW / 2
    g.append(f'<path d="M{dcx} {DY + DH} V{BUSY}" class="edge bus"/>')
    tw, th, tg, grp_gap, hdr = 76, 66, 6, 14, 20
    groups: dict[str, list[dict[str, Any]]] = {}
    for t in tiles:
        groups.setdefault(t["base"], []).append(t)
    total_w = sum(len(v) * tw + (len(v) - 1) * tg + 12 for v in groups.values()) + grp_gap * (len(groups) - 1)
    x = M + ((FIG_W - 2 * M) - total_w) / 2 + 6
    first_x = last_x = dcx
    for base, ms in groups.items():
        gw = len(ms) * tw + (len(ms) - 1) * tg
        g.append(f'<rect x="{x - 6}" y="{FY}" width="{gw + 12}" height="{th + hdr + 12}" rx="8" class="cluster"/>'
                 f'<text x="{x}" y="{FY + 14}" class="mono muted">{E(base)}</text>')
        for t in ms:
            cx = x + tw / 2
            first_x, last_x = min(first_x, cx), max(last_x, cx)
            ty = FY + hdr
            g.append(f'<path d="M{cx} {BUSY} V{FY}" class="edge bus"/>'
                     f'<a class="ld-tile" href="{E(t["repo"])}" target="_blank" rel="noopener"><g class="tile">'
                     f'<rect x="{x}" y="{ty}" width="{tw}" height="{th}" rx="8" class="box tilebox"/>'
                     f'<use href="#ld-frog" x="{x + 7}" y="{ty + 8}" width="11" height="11" class="frog"/>'
                     f'<text x="{x + tw - 8}" y="{ty + 18}" class="mono size">{E(t["size"])}</text>'
                     f'<text x="{x + tw / 2}" y="{ty + 36}" class="mono var">{E(t["variant"])}</text>')
            bw = tw - 16
            if t["correct"] is not None and t["n"]:
                g.append(f'<rect x="{x + 8}" y="{ty + 45}" width="{bw}" height="4" rx="2" class="track"/>'
                         f'<rect x="{x + 8}" y="{ty + 45}" width="{bw * t["correct"] / t["n"]:.1f}" height="4" rx="2" class="fill tfill"/>'
                         f'<text x="{x + tw / 2}" y="{ty + 60}" class="mono tiny">{t["correct"]}/{t["n"]}</text>')
            else:
                g.append(f'<text x="{x + tw / 2}" y="{ty + 60}" class="mono tiny">not benched</text>')
            g.append("</g></a>")
            x += tw + tg
        x += grp_gap + 12
    g.append(f'<path d="M{first_x} {BUSY} H{last_x}" class="edge bus"/>')
    fy = FIG_H - 24
    g.append(f'<text x="{M}" y="{fy}" class="mono muted">how: a small LoRA on a frozen base + a 1M-param head · one softmax over the options</text>'
             f'<text x="{M}" y="{fy + 16}" class="mono muted">bars: JevBench correct of {JEVBENCH_N}, measured on this Space</text>')
    g.append("</svg>")
    return "".join(g)


def hero_html(n_models: int, registry: list[Any] | None = None) -> str:
    """Title, lead, the two CTAs, the install line, and the concept figure (family from `registry`)."""
    return (
        '<div class="ld" id="ld-hero">'
        '<section class="ld-hero">'
        '<div class="ld-hero__copy">'
        '<h1 class="ld-hero__title">One question. <em>Every decider.</em></h1>'
        '<p class="ld-hero__lead">No generated text: one forward pass returns a calibrated probability for every '
        'option, faster than an LLM, and you can see how sure it is.</p>'
        '<div class="ld-hero__actions">'
        '<a class="sr-btn sr-btn--primary" href="#sdp-ask" data-ld-scroll="#sdp-ask textarea">Ask a question now</a>'
        '<a class="sr-btn" href="#sdx-gallery-host" data-ld-scroll="#sdx-gallery-host, #sdp-examples, #sdp-ask">See the examples</a>'
        f'<span class="sr-install"><code>{E(PIP_PYPI)}</code>{_copy_btn(PIP_PYPI)}</span>'
        '</div>'
        f'<p class="ld-hero__note">{n_models} public <a href="{HF_ORG_URL}" target="_blank" rel="noopener">StrandsAgents</a> deciders answer below: pick an example or write your own.</p>'
        '</div>'
        f'<div class="ld-hero__stage">{figure_svg(registry)}'
        f'<p class="ld-fig__cap">hover to pause · the mechanism in <a href="{ARCH_URL}" target="_blank" rel="noopener">docs/architecture.md</a></p></div>'
        '</section></div>'
    )


# ---------------------------------------------------------------------------------
# 2. how to use now
# ---------------------------------------------------------------------------------

def _term(cmd: str, title: str = "", out: str | None = None) -> str:
    """A terminal block with a copy button and an optional output pane."""
    head = f'<div class="ld-term__head"><span class="dots"><i></i><i></i><i></i></span><span class="t">{E(title)}</span>{_copy_btn(cmd)}</div>'
    body = f'<pre class="ld-term__body"><code>{E(cmd)}</code></pre>'
    if out:
        body += f'<pre class="ld-term__out"><span class="o">output</span><code>{E(out)}</code></pre>'
    return f'<div class="ld-term">{head}{body}</div>'


def howto_html() -> str:
    """Three ways in: ask here, install (PyPI or main) with the README commands, use it in an agent."""
    return (
        '<div class="ld" id="ld-howto"><section class="ld-howto">'
        '<header class="ld-sec"><h2>How to use it now</h2></header>'
        '<div class="ld-steps">'
        # (1) ask here
        '<article class="ld-step"><div class="ld-step__n mono">1</div><h3>Ask here</h3>'
        '<p>Every public decider answers the bar below.</p>'
        '<a class="sr-btn sr-btn--primary" href="#sdp-ask" data-ld-scroll="#sdp-ask textarea">Ask a question now</a></article>'
        # (2) install
        '<article class="ld-step ld-step--wide"><div class="ld-step__n mono">2</div><h3>Install</h3>'
        '<p>Gemma 4 models need <code>main</code>; PyPI has Qwen3.5.</p>'
        '<div class="ld-tabs" data-ld-tabs>'
        '<div class="ld-tabs__bar" role="tablist">'
        '<button type="button" role="tab" class="is-on" aria-selected="true" data-ld-tab="pypi">PyPI · Qwen3.5 models</button>'
        '<button type="button" role="tab" aria-selected="false" data-ld-tab="main">main · Gemma 4 models</button></div>'
        f'<div class="ld-tabs__pane is-on" data-ld-pane="pypi"><span class="sr-install"><code>{E(PIP_PYPI)}</code>{_copy_btn(PIP_PYPI)}</span></div>'
        f'<div class="ld-tabs__pane" data-ld-pane="main"><span class="sr-install"><code>{E(PIP_MAIN)}</code>{_copy_btn(PIP_MAIN)}</span></div>'
        '</div>'
        + _term(ASK_CMD, "ask", ASK_OUT)
        + '<p class="ld-step__sub">Or over HTTP:</p>'
        + _term(SERVE_CMD, "serve")
        + _term(CURL_CMD, "POST /v1/systemone")
        + f'<p class="ld-step__cite">From the <a href="{REPO_URL}#getting-started" target="_blank" rel="noopener">README</a>.</p></article>'
        # (3) in an agent
        '<article class="ld-step"><div class="ld-step__n mono">3</div><h3>Use it in an agent</h3>'
        '<p>A <code>before_tool_call</code> hook gates a tool call on two yes/no decisions (Strands Agents SDK).</p>'
        f'<a class="sr-btn" href="{AGENT_EXAMPLE_URL}" target="_blank" rel="noopener">examples/strands</a></article>'
        '</div></section></div>'
    )


# ---------------------------------------------------------------------------------
# 3. model row
# ---------------------------------------------------------------------------------

def _baseline_evals() -> dict[str, dict[str, Any]]:
    """Per-model JevBench correct + Brier measured on this Space (bench/baseline/meta.json), keyed by short name.

    Empty when the vendored baseline is missing, so the README figures take over."""
    try:
        meta = json.loads((Path(__file__).parent / "bench" / "baseline" / "meta.json").read_text())
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, row in (meta.get("per_model") or {}).items():
        if isinstance(row.get("correct"), int) and isinstance(row.get("brier"), (int, float)):
            out[name] = {"jevbench": row["correct"], "brier": float(row["brier"]), "n": row.get("n") or JEVBENCH_N,
                         "anchor": "#sdb-board", "src": f"measured here, {meta.get('date', '')} {meta.get('dtype', '')}".strip()}
    return out


BASELINE_EVALS: dict[str, dict[str, Any]] = _baseline_evals()


def _eval_badges(info: Any) -> str:
    """JevBench + Brier badges: the baseline measured on this Space when it has the model, else the README table."""
    name = getattr(info, "name", "") or getattr(info, "repo", "").split("/")[-1]
    ev = BASELINE_EVALS.get(name)
    if ev:
        href, src = "#sdb-board", ev["src"]
    else:
        ev = README_EVALS.get(name)
        if not ev:
            return f'<span class="ld-badge muted">no figure yet</span>'
        href, src = f"{REPO_URL}{ev['anchor']}", "README table"
    target = '' if href.startswith('#') else ' target="_blank" rel="noopener"'
    n = ev.get("n", JEVBENCH_N)
    return (f'<a class="ld-badge" href="{href}"{target} title="{src}: JevBench public set, {n} tasks">'
            f'<span class="k">JevBench</span><b>{ev["jevbench"]}</b><span class="k">/{n}</span></a>'
            f'<a class="ld-badge" href="{href}"{target} title="{src}: Brier score, lower is better">'
            f'<span class="k">Brier</span><b>{ev["brier"]:.3f}</b></a>')


def _cite() -> str:
    """One line naming where the card numbers come from."""
    if BASELINE_EVALS:
        return "JevBench and Brier as measured on the board above; gaps under ~10 tasks are noise (retrain SD 3.2)."
    return "JevBench and Brier from the README; gaps under ~10 tasks are noise (retrain SD 3.2)."


def models_html(registry: list[Any]) -> str:
    """One card per listed model: label, Hub link, README badges, vision tag."""
    cards = []
    for info in registry:
        repo = getattr(info, "repo", "")
        name = getattr(info, "name", repo.split("/")[-1])
        vision = bool(getattr(info, "vision", False))
        tag = '<span class="ld-tag on">vision</span>' if vision else '<span class="ld-tag">text only</span>'
        cards.append(
            '<div class="ld-model">'
            f'<span class="ld-model__base">{E(model_label(info))}</span>'
            f'<a class="ld-model__name mono" href="https://huggingface.co/{E(repo)}" target="_blank" rel="noopener" title="{E(repo)} on Hugging Face">{E(name)} ↗</a>'
            f'<span class="ld-model__row">{_eval_badges(info)}</span>'
            f'<span class="ld-model__row">{tag}</span></div>'
        )
    n = len(registry)
    return (
        '<div class="ld" id="ld-models"><section class="ld-models">'
        f'<header class="ld-sec"><h2>{n} public model{"s" if n != 1 else ""}</h2>'
        '<span class="sub">repo-name digits are not a size</span></header>'
        f'<div class="ld-models__grid">{"".join(cards)}</div>'
        f'<p class="ld-step__cite">{_cite()}</p>'
        '</section></div>'
    )


# ---------------------------------------------------------------------------------
# 4. footer
# ---------------------------------------------------------------------------------

def footer_html() -> str:
    """CTA panel, install line, link columns, wordmark and the legal line."""
    cols = [
        ("Use", [("Install", f"{REPO_URL}#getting-started"), ("Serve", f"{REPO_URL}#running-as-a-server"),
                 ("Images", f"{REPO_URL}#images"), ("In an agent", AGENT_EXAMPLE_URL)]),
        ("Read", [("Architecture", ARCH_URL), ("Evaluation", f"{REPO_URL}/blob/main/evaluation/results.md"),
                  ("Research", f"{REPO_URL}/blob/main/research/README.md"), ("Docs", DOCS_URL)]),
        ("Community", [("GitHub", REPO_URL), ("Hugging Face", HF_ORG_URL), ("PyPI", PYPI_URL),
                       ("Discord", DISCORD_URL), ("Strands Agents", "https://strandsagents.com/")]),
    ]
    col_html = "".join(
        f'<div><p class="ld-footer__heading">{E(t)}</p><ul>' +
        "".join(f'<li><a href="{E(h)}" target="_blank" rel="noopener">{E(l)}</a></li>' for l, h in links) + "</ul></div>"
        for t, links in cols
    )
    return (
        '<div class="ld" id="ld-footer">'
        '<section class="ld-cta"><div class="ld-cta__panel"><p class="ld-cta__title">Decide the rote parts. Keep the LLM for the hard ones.</p>'
        '<p class="ld-cta__links"><a href="#sdp-ask" data-ld-scroll="#sdp-ask textarea">Ask a question &rarr;</a>'
        f'<a href="{REPO_URL}#getting-started" target="_blank" rel="noopener">Install &rarr;</a>'
        f'<a href="{AGENT_EXAMPLE_URL}" target="_blank" rel="noopener">Agent example &rarr;</a></p></div></section>'
        '<section class="ld-footer" aria-label="links"><div class="ld-footer__inner">'
        f'<p class="sr-install sr-install--footer"><code>{E(PIP_PYPI)}</code>{_copy_btn(PIP_PYPI)}</p>'
        f'<div class="ld-footer__columns">{col_html}</div></div>'
        f'<div class="ld-footmark" aria-hidden="true">{WORDMARK}</div>'
        f'<div class="ld-footer__legal"><span class="ld-footer__brand">{MARK}<span class="ld-wordmark-wrap">{WORDMARK}</span><span class="proj mono">/decider</span></span>'
        f'<span class="muted">this page: MIT · <a href="{REPO_URL}/blob/main/LICENSE" target="_blank" rel="noopener">strands-decider: Apache-2.0</a> · '
        'models: StrandsAgents on Hugging Face · design: strands</span></div>'
        '</section></div>'
    )


# ---------------------------------------------------------------------------------
# standalone preview
# ---------------------------------------------------------------------------------

def _tokens_css_inlined() -> str:
    """tokens.css with the woff2 fonts inlined as data URIs (for the standalone preview)."""
    css = _static("tokens.css")
    for f in sorted((STATIC / "fonts").glob("*.woff2")):
        b64 = base64.b64encode(f.read_bytes()).decode("ascii")
        css = css.replace(f'url("fonts/{f.name}")', f'url("data:font/woff2;base64,{b64}")')
    return css


class _Demo:
    """Minimal stand-in for registry.ModelInfo in the standalone preview."""

    def __init__(self, name: str, base: str, vision: bool, r: Any):
        self.repo, self.name, self.base_model, self.vision, self.config_file = f"StrandsAgents/{name}", name, base, vision, None
        _LORA_R[self.repo] = r


def _demo_registry() -> list[Any]:
    """Live registry when the Hub answers; otherwise the 7 public repos as of writing (preview only)."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import registry  # noqa: PLC0415
        reg = registry.load_registry(with_evals=False)
        if reg:
            return reg
    except Exception as e:  # noqa: BLE001
        print(f"registry unavailable ({e}); using the measured fallback", file=sys.stderr)
    return [
        _Demo("strands-decider-2B-hobson-v19", "Qwen/Qwen3.5-2B-Base", True, 16),
        _Demo("strands-decider-2B-hobson-v21", "Qwen/Qwen3.5-2B-Base", True, 16),
        _Demo("strands-decider-2B-qwen3.5-v1-2610", "Qwen/Qwen3.5-2B-Base", True, 48),
        _Demo("strands-decider-E2B-gemma4-v1-2610", "google/gemma-4-E2B-it", False, 48),
        _Demo("strands-decider-E4B-gemma4-v1-2610", "google/gemma-4-E4B-it", False, 48),
        _Demo("strands-decider-12B-gemma4-v1-2610", "google/gemma-4-12B-it", False, 48),
        _Demo("strands-decider-26B-A4B-gemma4-v1-2610", "google/gemma-4-26B-A4B-it", False, 48),
    ]


def demo_page(dark: bool = False) -> str:
    """Standalone HTML of every landing section with placeholders for the ask bar and gallery."""
    reg = _demo_registry()
    n = len(reg)
    body = ('<div class="gradio-container">' + hero_html(n, reg)
            + '<div id="sdp-ask" class="ld-demo-ask"><label>State — what is being decided about'
              '<textarea rows="3" placeholder="the ask bar of the running app sits here"></textarea></label></div>'
              '<div id="sdx-gallery-host" class="ld-demo-ask"><em>the example gallery of the running app sits here</em></div>'
            + howto_html() + models_html(reg) + footer_html() + '</div>')
    return (f'<!doctype html><html class="{"dark" if dark else ""}" lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Strands Decider Playground — landing preview</title>'
            f'<style>{_tokens_css_inlined()}</style><style>{LANDING_CSS}</style>'
            '<style>body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sr-text)}'
            '.gradio-container{max-width:1280px;margin:0 auto;padding:0 24px}'
            '.ld-demo-ask{border:1px dashed var(--hair);border-radius:8px;padding:14px;margin:18px 0;color:var(--muted);font-size:.8rem}'
            '.ld-demo-ask textarea{display:block;width:100%;margin-top:6px;background:var(--chip-bg);color:var(--text);border:1px solid var(--hair);border-radius:8px;padding:8px;font:inherit}'
            '.ld-demo-ask textarea:focus{outline:none;border-color:var(--accent)}</style></head>'
            f'<body>{body}<script>{LANDING_JS}</script></body></html>')


if __name__ == "__main__":
    if "--demo" in sys.argv:
        sys.stdout.write(demo_page(dark="--dark" in sys.argv))
    else:
        print(__doc__)
