"""Generate the two standalone figures under docs/ from the same code that draws the page.

* docs/img/decider-concept.svg -- the hero figure's static frame (landing.figure_svg): state +
  question -> decider -> a probability per option, and the family of public deciders. The family
  is READ from the Hub registry and bench/baseline/meta.json, never typed, so the generator refuses
  to run when the Hub does not answer.
* docs/architecture.svg -- the technical figure: what happens inside a decider (frozen base, pointer
  head, one masked softmax, three readouts) and how one request moves through this playground.

Both carry their own <style> with the strands tokens for both palettes (prefers-color-scheme) and a
system-mono fallback, so they render the same in a README, a doc page or a slide.

    python docs/gen_figures.py            # rewrite both files
    python docs/gen_figures.py --check    # exit 1 when a committed figure is stale
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import landing  # noqa: E402

CONCEPT = os.path.join(HERE, "img", "decider-concept.svg")
ARCH = os.path.join(HERE, "architecture.svg")

# the strands tokens (static/tokens.css) for a file that lives outside the page
TOKENS = (
    ":root{--bg:#fff;--fg:#000;--text:#3d3c3c;--muted:#767373;--hair:rgba(0,0,0,.15);--accent:#007a3d;"
    "--accent-soft:rgba(2,164,53,.12);--chip-bg:#f4f4f4;"
    '--sr-mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}'
    "@media (prefers-color-scheme: dark){:root{--bg:#000;--fg:#fff;--text:#b6b6b6;--muted:#999696;--hair:#28292a;"
    "--accent:#00cc60;--accent-soft:rgba(0,204,96,.14);--chip-bg:#28292a}}"
    "svg{background:var(--bg)}"
)


def _fig_rules() -> str:
    """The .ld-fig rules of static/landing.css, minus every animation: the static frame."""
    css = open(os.path.join(ROOT, "static", "landing.css"), encoding="utf-8").read()
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)       # drop keyframe blocks
    css = re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)           # and media blocks
    keep = []
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        sel = sel.strip()
        if sel.startswith(".ld-fig") and "__cap" not in sel and "animation" not in body:
            keep.append(f"{sel}{{{body.strip()}}}")
    static = (".ld-fig .frame{opacity:0}.ld-fig .frame.f1{opacity:1}"
              ".ld-fig .chip.c1 .chipbox{stroke:var(--accent);fill:var(--accent-soft)}.ld-fig .chip.c1 .chiplab{fill:var(--fg)}"
              ".ld-fig .edge.bus{stroke:var(--accent)}.ld-fig .tilebox{stroke:var(--fg)}")
    return "".join(keep) + static


def concept_svg() -> str:
    """landing.figure_svg on the live registry, with embedded style and no motion."""
    import registry  # noqa: PLC0415

    reg = registry.load_registry(with_evals=False)
    if not reg:
        raise SystemExit("the Hub returned no public decider: refusing to write a figure from typed data")
    svg = landing.figure_svg(reg)
    style = "<style>" + TOKENS + _fig_rules() + "</style>"
    return svg.replace("<defs>", style + "<defs>", 1) + "\n"


# ---------------------------------------------------------------------------------
# the technical figure
# ---------------------------------------------------------------------------------
CH = 6.6  # px per character, 11 px mono
LH = 15   # line height


def E(s: str) -> str:
    return landing.E(s)


class Draw:
    """A tiny drawing language: boxes sized by their text, straight edges with arrowheads, captions."""

    def __init__(self) -> None:
        self.g: list[str] = []

    def box(self, x: float, y: float, title: str, lines: list[str], cls: str = "", w: float | None = None) -> tuple[float, float, float, float]:
        n = max([len(title)] + [len(li) for li in lines])
        w = w or round(n * CH + 24)
        h = 26 + LH * len(lines) + (6 if lines else 0)
        self.g.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="box {cls}"/>')
        self.g.append(f'<text x="{x + 12}" y="{y + 17}" class="t title">{E(title)}</text>')
        for i, li in enumerate(lines):
            self.g.append(f'<text x="{x + 12}" y="{y + 32 + LH * i}" class="t">{E(li)}</text>')
        return x, y, w, h

    def cluster(self, x: float, y: float, w: float, h: float, cap: str) -> None:
        self.g.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="cluster"/>')
        self.g.append(f'<text x="{x}" y="{y - 8}" class="cap">{E(cap)}</text>')

    def arrow(self, x1: float, y1: float, x2: float, y2: float, cls: str = "") -> None:
        """Horizontal (same y), vertical (same x), or a vertical-horizontal-vertical elbow; the head points along the last leg."""
        if y1 == y2:
            d, head = f"M{x1} {y1} L{x2} {y2}", f"M{x2 - 5} {y2 - 3} l5 3 -5 3"
        elif x1 == x2:
            d, head = f"M{x1} {y1} L{x2} {y2}", f"M{x2 - 3} {y2 - 5} l3 5 3 -5"
        else:
            ym = (y1 + y2) / 2
            d, head = f"M{x1} {y1} V{ym} H{x2} V{y2}", f"M{x2 - 3} {y2 - 5} l3 5 3 -5"
        self.g.append(f'<path d="{d}" class="edge {cls}"/><path d="{head}" class="edge {cls}"/>')

    def text(self, x: float, y: float, s: str, cls: str = "t") -> None:
        self.g.append(f'<text x="{x}" y="{y}" class="{cls}">{E(s)}</text>')


ARCH_W, ARCH_H = 960, 692


def architecture_svg() -> str:
    d = Draw()
    GAP = 40
    # ---- band A: inside a decider ------------------------------------------------------
    AY = 42
    d.cluster(16, AY, ARCH_W - 32, 236, "inside a decider · one forward pass, no generated text")
    rx, ry, rw, rh = d.box(32, AY + 22, "request", ["state", "question", "options:", "billing · sales · retail"], "in")
    bx, by, bw, bh = d.box(rx + rw + GAP, AY + 22, "frozen base (decoder)", ["LoRA of rank r on the torso", "LM head discarded", "trained: LoRA + head only"], "")
    hx, hy, hw, hh = d.box(bx + bw + GAP, AY + 22, "last hidden states", ["h(<answer>)", "h(option k), last token", "for every option k"], "")
    px, py, pw, ph = d.box(hx + hw + GAP, AY + 22, "pointer head", ["~1M params, fp32", "score_k = h(<answer>)·W·h(k)"], "accent")
    assert px + pw <= ARCH_W - 32, px + pw
    d.arrow(rx + rw, ry + 20, bx, ry + 20)
    d.arrow(bx + bw, ry + 20, hx, ry + 20)
    d.arrow(hx + hw, ry + 20, px, ry + 20, "sig")
    # softmax under the pointer head, readouts under the softmax (all right-aligned to the head)
    sw = round(len("p = softmax(score) over options") * CH + 24)
    sx, sy, sw, sh = d.box(px + pw - sw, py + ph + 24, "one masked softmax", ["p = softmax(score) over options"], "", w=sw)
    d.arrow(px + pw / 2, py + ph, px + pw / 2, sy, "sig")
    outs = [("noul", "P(yes)"), ("choice", "argmax p"), ("score", "E[level] = Σ k·p_k")]
    widths = [round(max(len(n), len(v)) * CH + 24) for n, v in outs]
    oy = sy + sh + 24
    x = px + pw - (sum(widths) + 12 * (len(outs) - 1))
    for (name, val), w in zip(outs, widths):
        d.box(x, oy, name, [val], "out", w=w)
        d.arrow(x + w / 2, sy + sh, x + w / 2, oy, "sig")
        x += w + 12
    ty = ry + rh + 30
    for i, line in enumerate(["an answer costs one forward pass of the base plus a dot product per option;",
                              "nothing is generated, so there is nothing to parse and no hallucinated option.",
                              "the same p is read back three ways: yes/no (noul) · pick one (choice) · a rating (score)."]):
        d.text(32, ty + 16 * i, line, "t muted")

    # ---- band B: this playground ---------------------------------------------------------
    BY = AY + 236 + 46
    d.cluster(16, BY, ARCH_W - 32, ARCH_H - BY - 16, "this playground · one request, every listed model, one result dict")
    G2 = 24
    hx2, hy2, hw2, hh2 = d.box(32, BY + 22, "Hugging Face", ["StrandsAgents org", "public decider repos", "config + evals per repo"], "")
    rgx, rgy, rgw, rgh = d.box(hx2 + hw2 + G2, BY + 22, "registry.py · at start", ["lists the org live", "public repos only", "reads config + evals"], "")
    ax, ay, aw, ah = d.box(rgx + rgw + G2, BY + 22, "app.answer()", ["state · question · image", "per model: load, ask,", "measure -> one card"], "")
    rnx, rny, rnw, rnh = d.box(ax + aw + G2, BY + 22, "runner.py", ["one lock per model"], "")
    cx, cy, cw, chh = d.box(rnx, rny + rnh + 12, "cache.py", ["load · evict by base"], "")
    mx, my, mw, mh = d.box(rnx, cy + chh + 12, "metrics.py", ["memory · latency"], "")
    sdx, sdy, sdw, sdh = d.box(rnx + rnw + G2, BY + 22, "strands_decider", ["engine.ask(...)"], "accent")
    assert sdx + sdw <= ARCH_W - 32, sdx + sdw
    d.arrow(hx2 + hw2, hy2 + 20, rgx, hy2 + 20)
    d.arrow(rgx + rgw, rgy + 20, ax, rgy + 20)
    d.arrow(ax + aw, ay + 20, rnx, ay + 20)
    d.arrow(rnx + rnw, rny + 20, sdx, rny + 20, "sig")
    # card dict pill + the renderers
    cdy = my + mh + 30
    d.g.append(f'<rect x="{ax + aw / 2 - 40}" y="{cdy - 12}" width="80" height="18" rx="9" class="pill"/>')
    d.text(ax + aw / 2, cdy + 1, "card dict", "t mid onaccent")
    d.arrow(ax + aw / 2, ay + ah, ax + aw / 2, cdy - 12, "sig")
    rend = [("app.py cards", ["probabilities · confidence", "published evals · compute"]),
            ("viz.py", ["sum of decisions · multiples", "confidence vs size"]),
            ("bench/", ["231 JevBench tasks · board", "common errors"]),
            ("landing.py · examples.py", ["concept figure · model row", "clickable examples"])]
    widths = [round(max([len(n)] + [len(li) for li in ls]) * CH + 24) for n, ls in rend]
    ry2 = cdy + 30
    gap = ((ARCH_W - 64) - sum(widths)) / (len(rend) - 1)
    x = 32
    for (name, lines), w in zip(rend, widths):
        d.box(x, ry2, name, lines, "")
        d.arrow(ax + aw / 2, cdy + 6, x + w / 2, ry2, "sig")
        x += w + gap
    d.text(32, ARCH_H - 28, "every number on the page is read from that one card dict; nothing is recomputed, and a number that cannot be measured says so.", "t muted")

    style = TOKENS + (
        "text{font-family:var(--sr-mono);font-size:11px;fill:var(--text)}"
        ".title{font-weight:600;fill:var(--fg)}.muted{fill:var(--muted)}.mid{text-anchor:middle}.end{text-anchor:end}"
        ".cap{font-size:11px;letter-spacing:.05em;text-transform:uppercase;fill:var(--fg)}"
        ".box{fill:var(--bg);stroke:var(--fg);stroke-width:1}.box.accent{stroke:var(--accent)}.box.out{stroke:var(--hair)}.box.in{stroke:var(--hair)}"
        ".cluster{fill:none;stroke:var(--hair);stroke-width:1;stroke-dasharray:4 3}"
        ".edge{fill:none;stroke:var(--muted);stroke-width:1}.edge.sig{stroke:var(--accent)}"
        ".pill{fill:var(--accent)}.onaccent{fill:var(--bg);font-weight:600}"
    )
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ARCH_W} {ARCH_H}" role="img" aria-labelledby="arch-t arch-d">'
            '<title id="arch-t">Strands decider: inside the model, and through this playground</title>'
            '<desc id="arch-d">Top: a request (state, question, options) runs once through a frozen decoder base with a small LoRA; '
            'a 1M-parameter pointer head scores the hidden state at the answer position against each option\'s last token, one masked '
            'softmax turns the scores into a probability per option, read back as yes/no, pick one, or a rating. Bottom: registry.py '
            'lists the public StrandsAgents repos at start; one request goes through app.answer(), runner.py, cache.py and metrics.py '
            'into strands_decider, and every card, diagram and bench cell is rendered from the same result dict.</desc>'
            f"<style>{style}</style>")
    return head + "".join(d.g) + "</svg>\n"


def main(argv: list[str]) -> int:
    outputs = {CONCEPT: concept_svg(), ARCH: architecture_svg()}
    stale = []
    for path, text in outputs.items():
        cur = open(path, encoding="utf-8").read() if os.path.exists(path) else None
        if cur != text:
            stale.append(os.path.relpath(path, ROOT))
            if "--check" not in argv:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "w", encoding="utf-8").write(text)
    if "--check" in argv:
        print("stale: " + ", ".join(stale) if stale else "figures up to date")
        return 1 if stale else 0
    print("wrote " + ", ".join(os.path.relpath(p, ROOT) + f" ({len(t):,} bytes)" for p, t in outputs.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
