"""Render the share card and the icons with a real browser.

    python docs/og/render.py [--fig path/to/figure.svg]

Writes docs/og/og-1200x630.png (the Open Graph / Twitter card), favicon-32.png and
apple-touch-icon-180.png, plus og-preview-600.png -- the card downscaled to the width Slack
and X show it at, so legibility is checked by looking, not assumed.

The card is an HTML page in the strands system (static/tokens.css, dark palette): frog mark
and pixel wordmark, the title, the one-line promise, and the hero's concept figure in its
static frame (reduced-motion). The figure is the hero lane's `docs/img/decider-concept.svg`
when it exists, else landing.figure_svg(); --fig overrides. Fonts are the self-hosted woff2.
Nothing here loads a model.
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

TITLE = "Strands Decider Playground"
LINE = "One question, every decider."
SUB = "7 public StrandsAgents deciders answer the same question side by side — with a probability, a confidence and a cost."


def _font_face(name: str, file: str, weight: str) -> str:
    b64 = base64.b64encode((STATIC / "fonts" / file).read_bytes()).decode("ascii")
    return (f'@font-face{{font-family:"{name}";font-weight:{weight};font-style:normal;'
            f'src:url("data:font/woff2;base64,{b64}") format("woff2")}}')


def figure_svg(path: Path | None) -> str:
    if path is None:
        cand = ROOT / "docs" / "img" / "decider-concept.svg"
        if cand.exists():
            path = cand
    if path is not None:
        return path.read_text(encoding="utf-8")
    sys.path.insert(0, str(ROOT))
    import landing  # noqa: E402
    return landing.figure_svg() if hasattr(landing, "figure_svg") else ""


def card_html(fig: str, fig_css: str, n_models: int) -> str:
    tokens = (STATIC / "tokens.css").read_text(encoding="utf-8")
    mark = (STATIC / "mark.svg").read_text(encoding="utf-8").replace("<svg ", '<svg class="mark" ', 1)
    word = (STATIC / "wordmark.svg").read_text(encoding="utf-8").replace("<svg ", '<svg class="wordmark" ', 1)
    sub = SUB.replace("7 public", f"{n_models} public") if n_models else SUB
    return f"""<!doctype html><html lang="en" class="dark"><head><meta charset="utf-8"><style>
{_font_face("JetBrains Mono", "JetBrainsMono-500.woff2", "500 600")}
{_font_face("Space Grotesk", "SpaceGrotesk-400.woff2", "400")}
{tokens}
{fig_css}
html,body{{margin:0;width:1200px;height:630px;overflow:hidden;background:#000;color:var(--fg)}}
.card{{position:relative;box-sizing:border-box;width:1200px;height:630px;padding:56px 64px;display:grid;grid-template-columns:560px 1fr;column-gap:40px;align-items:center}}
.card::before{{content:"";position:absolute;inset:0 0 40%;pointer-events:none;background-image:radial-gradient(var(--dot) 1.3px,transparent 1.9px);background-size:14px 14px;-webkit-mask-image:linear-gradient(to bottom,rgba(0,0,0,.5),transparent);mask-image:linear-gradient(to bottom,rgba(0,0,0,.5),transparent)}}
.brand{{position:absolute;left:64px;top:48px;display:flex;align-items:center;gap:16px}}
.brand .mark{{width:36px;height:36px;image-rendering:pixelated}} .brand .mark path{{fill:var(--accent)}}
.brand .wordmark{{height:26px;width:auto;display:block;color:var(--accent)}}
.brand .project{{font-family:var(--sr-mono);font-weight:600;font-size:22px;color:var(--fg);letter-spacing:.01em}}
.copy{{position:relative;align-self:end;padding-bottom:64px}}
h1{{font-family:var(--sr-mono);font-weight:600;font-size:60px;line-height:1.08;letter-spacing:.01em;margin:0 0 22px;color:var(--fg)}}
h1 em{{font-style:normal;color:var(--accent)}}
.sub{{font-family:var(--sr-text);font-size:23px;line-height:1.35;color:var(--text);margin:0;max-width:600px}}
.url{{position:absolute;left:64px;bottom:40px;font-family:var(--sr-mono);font-size:19px;letter-spacing:.04em;color:var(--muted)}}
.url b{{color:var(--fg);font-weight:500}}
.fig{{position:relative;align-self:center;border:1px solid var(--hair);border-radius:var(--sr-radius);padding:18px 18px 14px;background:#000}}
.fig svg{{display:block;width:100%;height:auto}}
</style></head><body><div class="card">
<div class="brand">{mark}{word}<span class="project">/decider-playground</span></div>
<div class="copy"><h1>One question.<br><em>Every decider.</em></h1><p class="sub">{sub}</p></div>
<div class="fig">{fig}</div>
<div class="url"><b>huggingface.co</b>/spaces/cagataydev/strands-decider-playground</div>
</div></body></html>"""


def icons_html() -> str:
    mark = (STATIC / "mark.svg").read_text(encoding="utf-8")
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:#000}}'
            f'#a{{width:32px;height:32px}} #b{{width:180px;height:180px;box-sizing:border-box;padding:22px;background:#000}}'
            f'svg{{display:block;width:100%;height:100%;shape-rendering:crispEdges}}</style></head>'
            f'<body><div id="a">{mark}</div><div id="b">{mark}</div></body></html>')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=Path, default=None, help="svg to place on the card (default: docs/img/decider-concept.svg or landing.figure_svg())")
    ap.add_argument("--fig-css", type=Path, default=None, help="extra css for the figure (default: static/landing.css)")
    ap.add_argument("--models", type=int, default=0, help="public model count for the subtitle (0 = keep the text as is)")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    fig = figure_svg(a.fig)
    css_path = a.fig_css or (STATIC / "landing.css")
    fig_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    (OUT / "card.html").write_text(card_html(fig, fig_css, a.models), encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1200, "height": 630}, device_scale_factor=1,
                            color_scheme="dark", reduced_motion="reduce")
        pg = ctx.new_page()
        pg.goto((OUT / "card.html").as_uri()); pg.wait_for_timeout(600)
        pg.screenshot(path=str(OUT / "og-1200x630.png"), clip={"x": 0, "y": 0, "width": 1200, "height": 630})
        # the size Slack/X show it at
        ctx2 = b.new_context(viewport={"width": 1200, "height": 630}, device_scale_factor=0.5,
                             color_scheme="dark", reduced_motion="reduce")
        pg2 = ctx2.new_page(); pg2.goto((OUT / "card.html").as_uri()); pg2.wait_for_timeout(400)
        pg2.screenshot(path=str(OUT / "og-preview-600.png"))
        pg3 = b.new_page(viewport={"width": 300, "height": 300}); pg3.set_content(icons_html()); pg3.wait_for_timeout(200)
        pg3.locator("#a").screenshot(path=str(OUT / "favicon-32.png"))
        pg3.locator("#b").screenshot(path=str(OUT / "apple-touch-icon-180.png"))
        b.close()
    (OUT / "card.html").unlink()
    for f in ("og-1200x630.png", "og-preview-600.png", "favicon-32.png", "apple-touch-icon-180.png"):
        print(f"{f}: {(OUT / f).stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
