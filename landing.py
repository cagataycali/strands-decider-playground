"""landing.py -- LANE LANDING owns this module.

Hero / proof row / how-to-use CTA / model row / footer for the Strands Decider Playground,
in the strandsagents.com / strands-robots voice. Pure HTML strings: `app.py` drops them into
gr.HTML inside its `# --- LANE LANDING ---` region. Nothing here recomputes a model answer.

Every number on the page is either read from the registry at build time (model count,
base model, LoRA rank, vision) or CITED to the strands-decider README / evaluation/results.md
with a link next to it. A model the README table does not know gets "no README figure".

Standalone preview:  python -m space.landing --demo > lanes/LANDING/demo.html
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
RESULTS_URL = f"{REPO_URL}/blob/main/evaluation/results.md#summary"
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
    return _html.escape("" if s is None else str(s), quote=True)


def _static(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _svg(name: str, cls: str, extra: str = "") -> str:
    raw = re.sub(r"<!--.*?-->", "", _static(name), flags=re.S).strip()
    return re.sub(r"<svg\b", f'<svg class="{cls}" aria-hidden="true" {extra}', raw, count=1)


LANDING_CSS = _static("landing.css")
LANDING_JS = _static("landing.js")
MARK = _svg("img/mark.svg", "ld-mark")
WORDMARK = _svg("img/wordmark.svg", "ld-wordmark")


# ---------------------------------------------------------------------------------
# registry helpers (duplicated on purpose: landing.py must not import app.py)
# ---------------------------------------------------------------------------------

_LORA_R: dict[str, Any] = {}


def lora_r(info: Any) -> Any:
    """`lora_r` from the decider config (local HF cache hit after registry read it)."""
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
    return ((getattr(info, "base_model", "") or "").split("/")[-1]) or "unknown base"


def model_label(info: Any) -> str:
    r = lora_r(info)
    return base_short(info) + (f" · LoRA r {r}" if r else "")


# ---------------------------------------------------------------------------------
# 1. hero
# ---------------------------------------------------------------------------------

def _copy_btn(text: str, cls: str = "sr-copy") -> str:
    return f'<button type="button" class="{cls}" data-ld-copy="{E(text)}" aria-label="copy to clipboard">copy</button>'


def figure_svg() -> str:
    """The architecture from the README, drawn to the strands rules: 1px strokes, rx 8, dashed
    cluster, accent only on the signal. The pointer comparison sweeps <answer> -> each option's
    last token on a 1.2 s loop (CSS keyframes; paused on hover; static under reduced motion)."""
    # column x positions
    xs = {"in": 22, "torso": 150, "hid": 300, "soft": 392, "out": 470}
    opts = [("billing", 92), ("sales", 150), ("retail", 208)]
    g = []
    g.append('<svg class="ld-fig" viewBox="0 0 560 320" role="img" aria-labelledby="ld-fig-t ld-fig-d">')
    g.append('<title id="ld-fig-t">Strands decider architecture</title>')
    g.append('<desc id="ld-fig-d">State, question and options enter a decoder torso whose language-modelling head was '
             'discarded. A pointer head compares the hidden state at the answer position against the hidden state at '
             'each option\'s last token. One masked softmax is read back three ways: noul, choice, score.</desc>')
    # inputs
    g.append(f'<g class="ld-f-in" transform="translate({xs["in"]} 56)">')
    for i, (lab, cls) in enumerate([("state", ""), ("question", ""), ("billing", "opt"), ("sales", "opt"), ("retail", "opt")]):
        y = i * 38
        g.append(f'<rect x="0" y="{y}" width="96" height="26" rx="8" class="box {cls}"/>'
                 f'<text x="48" y="{y + 17}" class="lab mono">{lab}</text>')
        g.append(f'<path d="M96 {y + 13} H{xs["torso"] - xs["in"] - 2}" class="edge"/>')
    g.append('<text x="48" y="-12" class="cap">request</text></g>')
    # torso cluster
    tx, ty, tw, th = xs["torso"], 40, 118, 220
    g.append(f'<g transform="translate({tx} {ty})"><rect x="0" y="0" width="{tw}" height="{th}" rx="8" class="cluster"/>'
             f'<text x="{tw / 2}" y="-12" class="cap">decoder torso</text>'
             f'<text x="{tw / 2}" y="{th + 18}" class="cap muted">LM head discarded · LoRA on the torso</text>')
    for j in range(6):
        g.append(f'<rect x="12" y="{14 + j * 32}" width="{tw - 24}" height="20" rx="8" class="box layer"/>')
    g.append(f'<text x="{tw / 2}" y="{th / 2 + 4}" class="lab mono layerlab">transformer blocks</text></g>')
    # hidden states column
    hx = xs["hid"]
    g.append(f'<g transform="translate({hx} 0)"><text x="26" y="44" class="cap">last hidden state</text>')
    ay = 60  # <answer>
    g.append(f'<path d="M{tx + tw - hx} {ty + th / 2} C {-30} {ty + th / 2}, {-30} {ay + 11}, 0 {ay + 11}" class="edge"/>')
    g.append(f'<rect x="0" y="{ay}" width="52" height="22" rx="8" class="box answer"/><text x="26" y="{ay + 15}" class="lab mono answerlab">&lt;answer&gt;</text>')
    for k, (lab, _) in enumerate(opts):
        oy = 128 + k * 50
        g.append(f'<path d="M{tx + tw - hx} {ty + th / 2} C {-30} {ty + th / 2}, {-30} {oy + 11}, 0 {oy + 11}" class="edge"/>')
        g.append(f'<rect x="0" y="{oy}" width="52" height="22" rx="8" class="box opt o{k}"/>'
                 f'<text x="26" y="{oy + 15}" class="lab mono">{lab}</text>'
                 f'<text x="26" y="{oy + 34}" class="cap muted tiny">last token</text>')
        # pointer comparison: a 1px line from <answer> to this option, drawn in its third of the loop
        g.append(f'<line x1="26" y1="{ay + 22}" x2="26" y2="{oy}" class="ptr p{k}" pathLength="100"/>')
        g.append(f'<text x="34" y="{oy - 6}" class="cap score s{k}">dot · score</text>')
    g.append('<text x="26" y="300" class="cap">pointer head · ~1M params · fp32</text></g>')
    # softmax
    sx = xs["soft"]
    g.append(f'<g transform="translate({sx} 0)"><rect x="0" y="118" width="58" height="92" rx="8" class="box soft"/>'
             '<text x="29" y="150" class="lab mono">one</text><text x="29" y="166" class="lab mono">masked</text>'
             '<text x="29" y="182" class="lab mono">softmax</text>')
    for k in range(3):
        oy = 128 + k * 50 + 11
        g.append(f'<path d="M{hx + 52 - sx} {oy} H0" class="edge sig"/>')
    g.append("</g>")
    # readouts
    ox = xs["out"]
    g.append(f'<g transform="translate({ox} 0)"><text x="36" y="44" class="cap">read back</text>')
    outs = [("noul", "P(yes) 0.86", 92), ("choice", "billing · 0.66", 150), ("score", "E[level] 1.03", 208)]
    for k, (lab, val, oy) in enumerate(outs):
        g.append(f'<path d="M{sx + 58 - ox} {164} C -14 164, -14 {oy + 11}, 0 {oy + 11}" class="edge"/>')
        g.append(f'<rect x="0" y="{oy - 4}" width="78" height="30" rx="8" class="box out"/>'
                 f'<text x="39" y="{oy + 8}" class="lab mono">{lab}</text>'
                 f'<text x="39" y="{oy + 20}" class="cap accent">{val}</text>')
    g.append("</g></svg>")
    return "".join(g)


def hero_html(n_models: int) -> str:
    return (
        f'<div class="ld" id="ld-hero"><style>{LANDING_CSS}</style>'
        '<section class="ld-hero">'
        '<div class="ld-hero__copy">'
        '<h1 class="ld-hero__title">One question. <em>Every decider.</em></h1>'
        '<p class="ld-hero__lead">A decision model picks between sets of options and rates things on a scale instead of '
        'generating text. It answers faster than an LLM, needs no classifier to be trained, and every answer carries a '
        'calibrated confidence.</p>'
        '<div class="ld-hero__actions">'
        '<a class="sr-btn sr-btn--primary" href="#sdp-ask" data-ld-scroll="#sdp-ask textarea">Ask a question now</a>'
        '<a class="sr-btn" href="#sdp-examples" data-ld-scroll="#sdp-examples, #sdp-ask">See the examples</a>'
        f'<span class="sr-install"><code>{E(PIP_PYPI)}</code>{_copy_btn(PIP_PYPI)}</span>'
        '</div>'
        f'<p class="ld-hero__note">{n_models} public checkpoints from <a href="{HF_ORG_URL}" target="_blank" rel="noopener">StrandsAgents</a> '
        'answer the same question below, side by side, with the memory and latency each call needed.</p>'
        '</div>'
        f'<div class="ld-hero__stage">{figure_svg()}'
        f'<p class="ld-fig__cap">Torso, <code>&lt;answer&gt;</code> position, pointer head scoring each option\'s last token, one masked softmax, '
        f'three readouts. <a href="{ARCH_URL}" target="_blank" rel="noopener">docs/architecture.md</a></p></div>'
        '</section></div>'
    )


# ---------------------------------------------------------------------------------
# 2. proof row
# ---------------------------------------------------------------------------------

def proof_html(n_models: int) -> str:
    return (
        '<div class="ld"><section class="sr-proof ld-proof">'
        f'<div><strong data-ld-count="{n_models}">{n_models}</strong><span>public models, read from the org when this page was built</span></div>'
        '<div><strong data-ld-count="3">3</strong><span>question types from one masked softmax: noul, choice, score</span></div>'
        '<div><strong><span data-ld-count="95" data-ld-suffix="%">95%</span></strong>'
        f'<span>of answers at confidence 0.9 or more are right on short tasks the model has never seen · '
        f'<a href="{RESULTS_URL}" target="_blank" rel="noopener">evaluation/results.md</a></span></div>'
        '</section></div>'
    )


# ---------------------------------------------------------------------------------
# 3. how to use now
# ---------------------------------------------------------------------------------

def _term(cmd: str, title: str = "", out: str | None = None) -> str:
    head = f'<div class="ld-term__head"><span class="dots"><i></i><i></i><i></i></span><span class="t">{E(title)}</span>{_copy_btn(cmd)}</div>'
    body = f'<pre class="ld-term__body"><code>{E(cmd)}</code></pre>'
    if out:
        body += f'<pre class="ld-term__out"><span class="o">output</span><code>{E(out)}</code></pre>'
    return f'<div class="ld-term">{head}{body}</div>'


def howto_html() -> str:
    return (
        '<div class="ld" id="ld-howto"><section class="ld-howto">'
        '<header class="ld-sec"><h2>How to use it now</h2><span class="sub">three ways in, shortest first</span></header>'
        '<div class="ld-steps">'
        # (1) ask here
        '<article class="ld-step"><div class="ld-step__n mono">1</div><h3>Ask here</h3>'
        '<p>Type a piece of state and a question in the bar below. Every public decider answers it; the cards show the '
        'probability of each option, the confidence, and what the call cost.</p>'
        '<a class="sr-btn sr-btn--primary" href="#sdp-ask" data-ld-scroll="#sdp-ask textarea">Ask a question now</a></article>'
        # (2) install
        '<article class="ld-step ld-step--wide"><div class="ld-step__n mono">2</div><h3>Install</h3>'
        '<p>One package, two sources. PyPI carries the Qwen3.5 models; Gemma 4 support is on <code>main</code> and not yet in the PyPI release.</p>'
        '<div class="ld-tabs" data-ld-tabs>'
        '<div class="ld-tabs__bar" role="tablist">'
        '<button type="button" role="tab" class="is-on" aria-selected="true" data-ld-tab="pypi">PyPI · Qwen3.5 models</button>'
        '<button type="button" role="tab" aria-selected="false" data-ld-tab="main">main · Gemma 4 models</button></div>'
        f'<div class="ld-tabs__pane is-on" data-ld-pane="pypi"><span class="sr-install"><code>{E(PIP_PYPI)}</code>{_copy_btn(PIP_PYPI)}</span></div>'
        f'<div class="ld-tabs__pane" data-ld-pane="main"><span class="sr-install"><code>{E(PIP_MAIN)}</code>{_copy_btn(PIP_MAIN)}</span></div>'
        '</div>'
        + _term(ASK_CMD, "ask — three question types in one call, the state is read once", ASK_OUT)
        + '<p class="ld-step__sub">Or serve it and ask over HTTP:</p>'
        + _term(SERVE_CMD, "serve")
        + _term(CURL_CMD, "POST /v1/systemone")
        + f'<p class="ld-step__cite">Commands and outputs from the <a href="{REPO_URL}#getting-started" target="_blank" rel="noopener">README</a>, '
        f'<code>--device cpu</code>; other devices differ in the last digits.</p></article>'
        # (3) in an agent
        '<article class="ld-step"><div class="ld-step__n mono">3</div><h3>Use it in an agent</h3>'
        '<p>A <code>before_tool_call</code> hook gates a weather-tool call on two yes/no decisions, so the agent asks which '
        'city instead of guessing. The worked example runs with the Strands Agents SDK.</p>'
        f'<a class="sr-btn" href="{AGENT_EXAMPLE_URL}" target="_blank" rel="noopener">examples/strands</a></article>'
        '</div></section></div>'
    )


# ---------------------------------------------------------------------------------
# 4. model row
# ---------------------------------------------------------------------------------

def _eval_badges(info: Any) -> str:
    name = getattr(info, "name", "") or getattr(info, "repo", "").split("/")[-1]
    ev = README_EVALS.get(name)
    if not ev:
        return f'<span class="ld-badge muted">no README figure</span>'
    href = f"{REPO_URL}{ev['anchor']}"
    return (f'<a class="ld-badge" href="{href}" target="_blank" rel="noopener" title="README table, JevBench public set, {JEVBENCH_N} tasks">'
            f'<span class="k">JevBench</span><b>{ev["jevbench"]}</b><span class="k">/{JEVBENCH_N}</span></a>'
            f'<a class="ld-badge" href="{href}" target="_blank" rel="noopener" title="README table, Brier score (lower is better)">'
            f'<span class="k">Brier</span><b>{ev["brier"]:.3f}</b></a>')


def models_html(registry: list[Any]) -> str:
    cards = []
    for info in registry:
        repo = getattr(info, "repo", "")
        name = getattr(info, "name", repo.split("/")[-1])
        vision = bool(getattr(info, "vision", False))
        tag = '<span class="ld-tag on">vision · reads images</span>' if vision else '<span class="ld-tag">text only</span>'
        cards.append(
            f'<a class="ld-model" href="https://huggingface.co/{E(repo)}" target="_blank" rel="noopener">'
            f'<span class="ld-model__base">{E(model_label(info))}</span>'
            f'<span class="ld-model__name mono">{E(name)}</span>'
            f'<span class="ld-model__row">{_eval_badges(info)}</span>'
            f'<span class="ld-model__row">{tag}</span></a>'
        )
    n = len(registry)
    return (
        '<div class="ld" id="ld-models"><section class="ld-models">'
        f'<header class="ld-sec"><h2>{n} public model{"s" if n != 1 else ""}</h2>'
        '<span class="sub">labelled by base and LoRA rank · the digits in a repo name are not a size</span></header>'
        f'<div class="ld-models__grid">{"".join(cards)}</div>'
        f'<p class="ld-step__cite">JevBench and Brier from the README tables (231 public tasks; six retrains of one recipe had SD 3.2 tasks, '
        f'so treat a gap under about 10 tasks as unresolved). Each card links to its Hugging Face page.</p>'
        '</section></div>'
    )


# ---------------------------------------------------------------------------------
# 5. footer
# ---------------------------------------------------------------------------------

def footer_html() -> str:
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
        '<footer class="ld-footer"><div class="ld-footer__inner">'
        '<p class="ld-footer__kicker">Start deciding in seconds</p>'
        f'<p class="sr-install sr-install--footer"><code>{E(PIP_PYPI)}</code>{_copy_btn(PIP_PYPI)}</p>'
        f'<div class="ld-footer__columns">{col_html}</div></div>'
        f'<div class="ld-footmark" aria-hidden="true">{WORDMARK}</div>'
        f'<div class="ld-footer__legal"><span class="ld-footer__brand">{MARK}<span class="ld-wordmark-wrap">{WORDMARK}</span><span class="proj mono">/decider</span></span>'
        f'<a href="{REPO_URL}/blob/main/LICENSE" target="_blank" rel="noopener">Apache-2.0</a>'
        '<span class="muted">models: StrandsAgents on Hugging Face · page: strands design system</span></div>'
        '</footer></div>'
    )


# ---------------------------------------------------------------------------------
# standalone preview
# ---------------------------------------------------------------------------------

def _tokens_css_inlined() -> str:
    css = _static("tokens.css")
    for f in sorted((STATIC / "fonts").glob("*.woff2")):
        b64 = base64.b64encode(f.read_bytes()).decode("ascii")
        css = css.replace(f'url("fonts/{f.name}")', f'url("data:font/woff2;base64,{b64}")')
    return css


class _Demo:
    def __init__(self, name: str, base: str, vision: bool, r: Any):
        self.repo, self.name, self.base_model, self.vision, self.config_file = f"StrandsAgents/{name}", name, base, vision, None
        _LORA_R[self.repo] = r


def _demo_registry() -> list[Any]:
    """Live registry when the Hub answers; the 7 public repos with their measured bases otherwise."""
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
        _Demo("strands-decider-2B-qwen3.5-v1-2610", "Qwen/Qwen3.5-2B-Base", True, 16),
        _Demo("strands-decider-E2B-gemma4-v1-2610", "google/gemma-4-E2B-it", False, 48),
        _Demo("strands-decider-E4B-gemma4-v1-2610", "google/gemma-4-E4B-it", False, 48),
        _Demo("strands-decider-12B-gemma4-v1-2610", "google/gemma-4-12B-it", False, 48),
        _Demo("strands-decider-26B-A4B-gemma4-v1-2610", "google/gemma-4-26B-A4B-it", False, 48),
    ]


def demo_page(dark: bool = False) -> str:
    reg = _demo_registry()
    n = len(reg)
    body = (hero_html(n) + proof_html(n)
            + '<div class="gradio-container"><div id="sdp-ask" class="ld-demo-ask"><label>State — what is being decided about'
              '<textarea rows="3" placeholder="the ask bar of the running app sits here"></textarea></label></div>'
              '<div id="sdp-examples" class="ld-demo-ask"><em>examples region (LANE EXAMPLES)</em></div></div>'
            + howto_html() + models_html(reg) + footer_html())
    return (f'<!doctype html><html class="{"dark" if dark else ""}" lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Strands Decider Playground — landing preview</title>'
            f'<style>{_tokens_css_inlined()}</style>'
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
