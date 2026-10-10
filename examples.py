"""The example gallery: one card per ability of the decider, click to fill the ask bar and run.

Data lives in examples/examples.json (+ images); the look in static/examples.css; the click
handling in static/examples.js. `app.py` renders `gallery_html()` above the status line and
listens on a hidden textbox for the clicked id. Nothing here touches answer().

Public API
    load_examples()            -> dict (the parsed examples.json, measured blocks included)
    gallery_html()             -> str  (server-rendered gallery; static/examples.js makes it clickable)
    example_by_id(id)          -> dict | None
    fill_values(ex, registry)  -> dict with everything the ask bar needs to be filled from one example:
                                  state, questions[{type, instructions, options}], image (str path|None -- a path, not PIL: see fill_values),
                                  models (list[str] -- ONLY the vision models for vision_only examples)
    CSS / JS                   -> file contents for app.py to append to its launch(css=, js=)

Standalone preview:  python examples.py --demo [--dark|--light] > demo.html
"""
from __future__ import annotations

import base64
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EX_DIR = HERE / "examples"
EX_JSON = EX_DIR / "examples.json"
STATIC = HERE / "static"

_CACHE: dict | None = None


def load_examples() -> dict:
    """Parsed examples.json (cached)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(EX_JSON.read_text())
    return _CACHE


def example_by_id(ex_id: str) -> dict | None:
    """The example record with this id, or None."""
    return next((e for e in load_examples()["examples"] if e["id"] == ex_id), None)


def image_path(ex: dict) -> Path | None:
    """Absolute path of the example's image, if it has one."""
    return (EX_DIR / ex["image"]) if ex.get("image") else None


def fill_values(ex: dict, registry: list[Any] | None = None) -> dict:
    """Everything the ask bar needs. `registry` = app.REGISTRY (ModelInfo with .name/.vision)."""
    # The image goes to the gr.Image component as a FILE PATH, never as a PIL object: gr.Image's
    # default output format is lossy WebP (gradio 6.30: Image().format == "webp"), and a PIL value
    # is re-encoded through it before the model sees it. MEASURED: form_signed.png as PIL -> model
    # read P(true) 0.5510; the precomputed answer on the original bytes is 0.6868. A path is served as-is.
    img: str | None = None
    p = image_path(ex)
    if p is not None:
        img = str(p)
    models: list[str] | None = None
    if registry is not None:
        if ex.get("vision_only"):
            models = [m.name for m in registry if getattr(m, "vision", False)]
        else:
            models = [m.name for m in registry]
    return {
        "id": ex["id"],
        "state": ex.get("state", ""),
        "questions": [dict(q) for q in ex["questions"]],
        "image": img,
        "models": models,
        "vision_only": bool(ex.get("vision_only")),
    }


# ---------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------

E = html.escape
_TYPE_LABEL = {"noul": "yes/no", "choice": "choice", "score": "score"}


def _thumb(ex: dict) -> str:
    """Inline 96 px thumbnail of the example image (data URI), or an empty string."""
    p = image_path(ex)
    if p is None or not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return (f'<img class="sdx-thumb" src="data:image/png;base64,{b64}" alt="{E(ex["title"])} example image" '
            f'loading="lazy" width="96" height="96">')


def _fmt(v: Any, nd: int = 3) -> str:
    """Fixed-point text; non-numbers are escaped as-is."""
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return E(str(v))


def _answer_html(qid: str, a: dict, q: dict | None) -> str:
    """One measured answer -> value line + mini distribution. Numbers are read, never computed."""
    t = a.get("type") or (q or {}).get("type") or ""
    conf = a.get("confidence")
    probs = a.get("probs") or {}
    display = a.get("display", a.get("value"))
    if t == "noul":
        p = probs.get("true", probs.get("yes"))
        if p is None:
            try:
                p = float(display)
            except (TypeError, ValueError):
                p = None
        pct = f"{p * 100:.1f}" if isinstance(p, float) else "0"
        head = f'<span class="k">P(true)</span><span class="v">{_fmt(p, 4) if p is not None else E(str(display))}</span>'
        bar = (f'<div class="sdx-noul" role="img" aria-label="P(true) {_fmt(p, 3)}"><span style="--p:{pct}%"></span></div>'
               if p is not None else "")
        return f'<div class="sdx-ans"><div class="line">{head}</div>{bar}</div>'
    if t == "choice":
        head = (f'<span class="k">&rarr;</span><span class="v">{E(str(display))}</span>'
                + (f'<span class="c">confidence {_fmt(conf)}</span>' if conf is not None else ""))
        rows = "".join(
            f'<div class="row{" top" if k == display else ""}"><span class="l">{E(str(k))}</span>'
            f'<span class="b"><i style="--p:{float(v) * 100:.1f}%"></i></span><span class="n">{_fmt(v)}</span></div>'
            for k, v in probs.items()
        )
        return f'<div class="sdx-ans"><div class="line">{head}</div><div class="sdx-bars">{rows}</div></div>'
    if t == "score":
        head = (f'<span class="k">score</span><span class="v">{E(str(display))}</span>'
                + (f'<span class="c">confidence {_fmt(conf)}</span>' if conf is not None else ""))
        rows = "".join(
            f'<div class="row"><span class="l"><em>{i}</em> {E(str(k))}</span>'
            f'<span class="b"><i style="--p:{float(v) * 100:.1f}%"></i></span><span class="n">{_fmt(v)}</span></div>'
            for i, (k, v) in enumerate(probs.items())
        )
        return f'<div class="sdx-ans"><div class="line">{head}</div><div class="sdx-bars">{rows}</div></div>'
    return f'<div class="sdx-ans"><div class="line"><span class="v">{E(json.dumps(a))}</span></div></div>'


def _card(ex: dict) -> str:
    """One gallery card: title, type chips, request, the measured answer (if any), what it teaches."""
    qs = ex["questions"]
    chips = "".join(f'<span class="sdx-chip t-{E(q["type"])}">{E(_TYPE_LABEL.get(q["type"], q["type"]))}</span>' for q in qs)
    if ex.get("golden"):
        chips += '<span class="sdx-chip golden" title="the README regression check -- its answer is published">golden</span>'
    if ex.get("vision_only"):
        chips += '<span class="sdx-chip vision" title="image input exists only on the Qwen3.5 deciders; Gemma 4 checkpoints drop the vision encoder">Qwen3.5 models only</span>'
    if len(qs) > 1:
        chips += f'<span class="sdx-chip">{len(qs)} questions</span>'

    req = []
    if ex.get("state"):
        req.append(f'<div class="sdx-state"><span class="lbl">state</span><pre>{E(ex["state"])}</pre></div>')
    elif ex.get("image"):
        req.append('<div class="sdx-state"><span class="lbl">state</span><pre class="empty">(empty — the image is the state)</pre></div>')
    for q in qs:
        opts = q.get("options") or []
        req.append(
            f'<div class="sdx-q"><span class="lbl">{E(_TYPE_LABEL.get(q["type"], q["type"]))}</span>'
            f'<span class="ins">{E(q["instructions"])}</span>'
            + (f'<span class="opts">{" · ".join(E(str(o)) for o in opts)}</span>' if opts else "")
            + "</div>"
        )

    m = ex.get("measured") or {}
    if m.get("answers"):
        qmap = {}
        for i, q in enumerate(qs):
            qmap[f'{q["type"]}_{sum(1 for p in qs[:i] if p["type"] == q["type"])}'] = q
        ans = "".join(_answer_html(qid, a, qmap.get(qid)) for qid, a in m["answers"].items())
        # Caption: model + date + token count. Latency and device are left out on purpose: they
        # describe the machine the recording ran on, not the example.
        parts = ["measured", str(m.get("model") or "").replace("strands-decider-", "") or "decider"]
        day = re.search(r"\d{4}-\d{2}-\d{2}", str(m.get("label", "")))
        if day:
            parts.append(day.group(0))
        if m.get("input_tokens") is not None:
            parts.append(f'{m["input_tokens"]} input tokens')
        title = f'recorded through app.answer() on {m.get("model", "")}'
        measured = (f'<div class="sdx-measured"><div class="cap" title="{E(title)}">{E(" · ".join(parts))}'
                    f'</div>{ans}</div>')
    elif m.get("error"):
        measured = f'<div class="sdx-measured err"><div class="cap">not measured</div><div class="line">{E(m["error"])}</div></div>'
    else:
        measured = '<div class="sdx-measured none"><div class="cap">not yet measured</div></div>'

    thumb = _thumb(ex)
    return (
        f'<article class="sdx-card{" has-img" if thumb else ""}" data-example="{E(ex["id"])}" tabindex="0" role="button" '
        f'aria-label="Load example: {E(ex["title"])}">'
        f'<div class="sdx-head">{thumb}<div class="sdx-title-wrap"><h3 class="sdx-title">{E(ex["title"])}</h3>'
        f'<div class="sdx-chips">{chips}</div></div></div>'
        f'<p class="sdx-shows">{E(ex["shows"])}</p>'
        f'<div class="sdx-req">{"".join(req)}</div>'
        f'{measured}'
        f'<p class="sdx-teaches"><span class="lbl">teaches</span>{E(ex["teaches"])}</p>'
        f'<div class="sdx-cta"><span class="sdx-run">run this example <span aria-hidden="true">&rarr;</span></span></div>'
        f'</article>'
    )


def gallery_html() -> str:
    """The whole gallery grouped by family, as server-rendered HTML."""
    data = load_examples()
    by_fam: dict[str, list[dict]] = {}
    for ex in data["examples"]:
        by_fam.setdefault(ex["family"], []).append(ex)
    n = len(data["examples"])
    n_measured = sum(1 for e in data["examples"] if (e.get("measured") or {}).get("answers"))
    out = [f'<a id="sdx-gallery-host" class="sdx-anchor" aria-hidden="true"></a><section class="sdx" id="sdx-gallery" aria-label="Examples">'
           f'<div class="sdx-section"><h2>Examples</h2><span class="sub">{n} abilities · {n_measured} with a measured answer · '
           f'click one to fill the ask bar and run it</span></div>']
    for fam in data["families"]:
        items = by_fam.get(fam["id"], [])
        if not items:
            continue
        out.append(f'<div class="sdx-family" data-family="{E(fam["id"])}"><div class="sdx-fam-head">'
                   f'<h3>{E(fam["label"])}</h3><span class="blurb">{E(fam["blurb"])}</span></div>'
                   f'<div class="sdx-grid">{"".join(_card(e) for e in items)}</div></div>')
    out.append("</section>")
    return "".join(out)


def _read(name: str) -> str:
    """Text of a file under static/, empty if missing."""
    p = STATIC / name
    return p.read_text() if p.exists() else ""


def css() -> str:
    """static/examples.css for app.py's launch(css=...)."""
    return _read("examples.css")


def js() -> str:
    """static/examples.js for app.py's launch(js=...)."""
    return _read("examples.js")


# ---------------------------------------------------------------------------------
# standalone preview
# ---------------------------------------------------------------------------------


def demo_html(scheme: str = "auto") -> str:
    """Standalone HTML of the gallery (scheme = auto | dark | light)."""
    tokens = _read("tokens.css")
    cls = {"dark": ' class="dark"', "light": "", "auto": ""}[scheme]
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>examples — strands decider playground</title>"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<style>{tokens}\n{css()}\nbody{{margin:0;background:var(--bg);color:var(--text);font-family:var(--sr-text);}}'
        ".wrap{max-width:66rem;margin:0 auto;padding:24px 16px;}</style></head>"
        f'<body{cls}><div class="wrap">{gallery_html()}</div>'
        f'<script>{js()}</script></body></html>'
    )


if __name__ == "__main__":
    if "--demo" in sys.argv:
        scheme = "dark" if "--dark" in sys.argv else ("light" if "--light" in sys.argv else "auto")
        sys.stdout.write(demo_html(scheme))
    else:
        d = load_examples()
        for e in d["examples"]:
            m = e.get("measured") or {}
            flag = "measured" if m.get("answers") else ("ERROR" if m.get("error") else "unmeasured")
            print(f'{e["id"]:22s} {e["family"]:8s} {len(e["questions"])}q {"img " if e.get("image") else "    "}{flag}')
