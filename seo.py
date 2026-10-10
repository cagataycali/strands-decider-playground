"""Discoverability: the <head> a crawler sees, the share card, robots.txt and the sitemap.

Why this module exists (measured 2026-10-10 on gradio 6.30.0):

* Gradio's page template hardcodes its OWN share card -- ``og:title "Gradio"``,
  ``og:description "Click to try out the app!"``, the Gradio header image and a literal
  ``og:url "{url}"`` -- before anything an app can add.
* ``launch(head=...)`` is injected by the browser-side bundle, never by the server. Under
  server-side rendering (the Spaces default) the HTML a crawler downloads carries no trace of
  it; under client-side rendering it sits inside a JSON string. Link unfurlers (Slack, X,
  LinkedIn, Facebook, iMessage) do not run JavaScript, so ``head=`` alone cannot change what
  they show.

So the page is served through a small FastAPI app (``mount()``): an ASGI middleware rewrites
the HTML of ``/`` -- Gradio's default ``og:*``/``twitter:*`` block is removed and ours is
written right after ``<meta charset>`` -- and the same app serves ``/robots.txt``,
``/sitemap.xml``, ``/og.png`` and the icons. Server-side rendering is switched off because
in that mode the page is served by Gradio's Node proxy, which this middleware never sees.
Everything else -- the queue, ``answer()``, the handlers -- is the same ``Blocks`` object.

Every fact in the tags is read from the registry, the README front matter or the vendored
baseline's ``meta.json``; nothing is typed twice. ``python seo.py`` prints the head block and
validates the JSON-LD.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).parent

SPACE_ID = "cagataydev/strands-decider-playground"
SPACE_URL = f"https://huggingface.co/spaces/{SPACE_ID}"
APP_URL = "https://cagataydev-strands-decider-playground.hf.space/"
REPO_URL = "https://github.com/cagataycali/strands-decider-playground"
RAW_URL = f"https://raw.githubusercontent.com/cagataycali/strands-decider-playground/main"
DECIDER_REPO_URL = "https://github.com/strands-labs/strands-decider"
JEVBENCH_URL = "https://github.com/fstandhartinger/jevbench"
ORG_URL = "https://huggingface.co/StrandsAgents"
SITE_NAME = "Strands"

OG_PATH = "docs/og/og-1200x630.png"
OG_IMAGE = f"{SPACE_URL}/resolve/main/{OG_PATH}"  # CDN-hosted; resolves once the Space is public
OG_W, OG_H = 1200, 630
OG_ALT = ("Strands Decider Playground: the frog mark and the line "
          "'One question, every decider' on black.")

THEME_DARK, THEME_LIGHT = "#000000", "#ffffff"

DESCRIPTION_MAX = 155


# ---------------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------------

def front_matter(path: Path = HERE / "README.md") -> dict[str, Any]:
    """The Space card's YAML front matter as a flat dict (strings and simple lists only)."""
    text = path.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    out: dict[str, Any] = {}
    if not m:
        return out
    key = None
    for line in m.group(1).splitlines():
        if re.match(r"^\s*-\s+", line) and key:
            out.setdefault(key, [])
            if isinstance(out[key], list):
                out[key].append(line.split("-", 1)[1].strip().strip("'\""))
            continue
        if ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip().strip("'\"")
            out[key] = val if val else []
    return out


def baseline_meta(path: Path = HERE / "bench" / "baseline" / "meta.json") -> dict[str, Any]:
    """The vendored JevBench baseline's provenance record (models, task count, versions)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def description(n_models: int, n_tasks: int) -> str:
    """One plain sentence, <=155 characters, for <meta description> and the share card."""
    who = f"{n_models} public" if n_models else "every public"
    s = (f"Ask one decision question and watch {who} Strands decider models answer "
         f"side by side: probabilities, confidence, cost, and the {n_tasks}-task JevBench board.")
    if len(s) > DESCRIPTION_MAX:
        s = s[:DESCRIPTION_MAX - 1].rsplit(" ", 1)[0] + "…"
    return s


# ---------------------------------------------------------------------------------
# head
# ---------------------------------------------------------------------------------

def _e(s: Any) -> str:
    return html.escape(str(s), quote=True)


def json_ld(title: str, desc: str, model_repos: Iterable[str], meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Two schema.org objects: the app, and the vendored baseline as a Dataset."""
    repos = list(model_repos)
    app = {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": title,
        "description": desc,
        "url": SPACE_URL,
        "sameAs": [APP_URL, REPO_URL],
        "applicationCategory": "DeveloperApplication",
        "applicationSubCategory": "Machine learning playground",
        "operatingSystem": "Web",
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
        "author": {"@type": "Organization", "name": SITE_NAME, "url": "https://strandsagents.com"},
        "license": "https://opensource.org/licenses/MIT",
        "isBasedOn": DECIDER_REPO_URL,
        "screenshot": OG_IMAGE,
        "image": OG_IMAGE,
        "keywords": "decision model, calibration, LoRA, JevBench, Gemma, Qwen, Strands",
    }
    if repos:
        app["mentions"] = [{"@type": "SoftwareSourceCode", "name": r.split("/")[-1],
                            "codeRepository": f"https://huggingface.co/{r}"} for r in repos]
    n_tasks = meta.get("n_tasks")
    n_cells = meta.get("n_cells")
    dataset = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": "Strands Decider Playground — JevBench baseline",
        "description": (f"Every public Strands decider scored on the {n_tasks or 'public'} JevBench tasks "
                        f"({n_cells or 'all'} model×task cells) with JevBench's own scorer, captured on "
                        f"{meta.get('hardware_detail') or 'a Hugging Face Space'} on {meta.get('date') or 'the capture date'}."),
        "url": f"{REPO_URL}/tree/main/bench/baseline",
        "license": "https://opensource.org/licenses/MIT",
        "creator": {"@type": "Organization", "name": SITE_NAME, "url": "https://strandsagents.com"},
        "isBasedOn": JEVBENCH_URL,
        "distribution": [
            {"@type": "DataDownload", "encodingFormat": "application/jsonl",
             "contentUrl": f"{RAW_URL}/bench/baseline/results.jsonl"},
            {"@type": "DataDownload", "encodingFormat": "application/json",
             "contentUrl": f"{RAW_URL}/bench/baseline/meta.json"},
        ],
        "variableMeasured": [
            {"@type": "PropertyValue", "name": "correct", "description": "JevBench score_task verdict per cell"},
            {"@type": "PropertyValue", "name": "p_gold", "description": "probability the model put on the gold answer"},
            {"@type": "PropertyValue", "name": "brier", "description": "Brier score of the answer distribution"},
            {"@type": "PropertyValue", "name": "latency_ms", "description": "forward pass alone, GPU, bf16"},
        ],
    }
    if meta.get("date"):
        dataset["dateCreated"] = meta["date"]
    if meta.get("space_sha"):
        dataset["version"] = str(meta["space_sha"])[:12]
    return [app, dataset]


def head_html(title: str, desc: str, model_repos: Iterable[str], meta: dict[str, Any]) -> str:
    """The tags a crawler needs. Plain string; the middleware writes it into <head>."""
    tags = [
        f"<title>{_e(title)}</title>",  # the client-side template has none until JS runs
        f'<meta name="description" content="{_e(desc)}">',
        '<meta name="robots" content="index,follow,max-image-preview:large">',
        f'<link rel="canonical" href="{SPACE_URL}">',
        '<meta property="og:type" content="website">',
        f'<meta property="og:site_name" content="{SITE_NAME}">',
        f'<meta property="og:title" content="{_e(title)}">',
        f'<meta property="og:description" content="{_e(desc)}">',
        f'<meta property="og:url" content="{SPACE_URL}">',
        f'<meta property="og:image" content="{OG_IMAGE}">',
        '<meta property="og:image:type" content="image/png">',
        f'<meta property="og:image:width" content="{OG_W}">',
        f'<meta property="og:image:height" content="{OG_H}">',
        f'<meta property="og:image:alt" content="{_e(OG_ALT)}">',
        '<meta property="og:locale" content="en_US">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{_e(title)}">',
        f'<meta name="twitter:description" content="{_e(desc)}">',
        f'<meta name="twitter:image" content="{OG_IMAGE}">',
        f'<meta name="twitter:image:alt" content="{_e(OG_ALT)}">',
        f'<meta name="theme-color" media="(prefers-color-scheme: dark)" content="{THEME_DARK}">',
        f'<meta name="theme-color" media="(prefers-color-scheme: light)" content="{THEME_LIGHT}">',
        '<meta name="color-scheme" content="dark light">',
        '<link rel="icon" href="/icon.svg" type="image/svg+xml">',
        '<link rel="icon" href="/favicon.ico" sizes="32x32">',
        '<link rel="apple-touch-icon" href="/apple-touch-icon.png" sizes="180x180">',
    ]
    for obj in json_ld(title, desc, model_repos, meta):
        tags.append('<script type="application/ld+json">'
                    + json.dumps(obj, ensure_ascii=False).replace("</", "<\\/") + "</script>")
    return "\n".join(tags)


def build(model_repos: Iterable[str]) -> dict[str, Any]:
    """Everything the server needs, from the facts on disk + the registry's repo list."""
    fm = front_matter()
    meta = baseline_meta()
    repos = list(model_repos)
    title = fm.get("title") or "Strands Decider Playground"
    desc = description(len(repos), int(meta.get("n_tasks") or 231))
    return {"title": title, "description": desc, "head": head_html(title, desc, repos, meta),
            "repos": repos, "meta": meta}


# ---------------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------------

# Gradio's own share-card block: every og:/twitter: meta in the template. Removed as a whole,
# including the multi-line ones, so a crawler sees exactly one card.
_GRADIO_SOCIAL = re.compile(r'\s*<meta\s+(?:property|name)="(?:og|twitter):[^"]*"[^>]*?/?>', re.S)
_CHARSET = re.compile(r'(<meta\s+charset="utf-8"\s*/?>)', re.I)
_HTML_OPEN = re.compile(r"<html(\s[^>]*)?>", re.I)


def rewrite_html(doc: str, head: str) -> str:
    """Drop Gradio's default card, write ours after <meta charset>, make sure <html lang="en">."""
    doc = _GRADIO_SOCIAL.sub("", doc)
    doc = _CHARSET.sub(lambda m: m.group(1) + "\n\t\t" + head.replace("\n", "\n\t\t"), doc, count=1)
    m = _HTML_OPEN.search(doc)
    if m and "lang=" not in (m.group(1) or ""):
        doc = doc[:m.start()] + '<html lang="en"' + (m.group(1) or "") + ">" + doc[m.end():]
    return doc


class HeadRewriteMiddleware:
    """Pure ASGI: buffer the text/html body of GET / (and /?...) and rewrite its <head>.

    Only the root document is touched; every other route (the queue's SSE stream, file
    uploads, /gradio_api/...) passes through untouched and unbuffered.
    """

    def __init__(self, app, head: str):
        self.app = app
        self.head = head

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path", "/") not in ("", "/") or scope.get("method") not in ("GET", "HEAD"):
            await self.app(scope, receive, send)
            return
        start: dict | None = None
        chunks: list[bytes] = []
        passthrough = False

        async def _send(message):
            nonlocal start, passthrough
            if message["type"] == "http.response.start":
                headers = {k.lower(): v for k, v in message.get("headers", [])}
                if b"text/html" in headers.get(b"content-type", b""):
                    start = message
                    return
                passthrough = True
                await send(message)
            elif message["type"] == "http.response.body":
                if passthrough or start is None:
                    await send(message)
                    return
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    body = rewrite_html(b"".join(chunks).decode("utf-8", "replace"), self.head).encode("utf-8")
                    hdrs = [(k, v) for k, v in start["headers"] if k.lower() != b"content-length"]
                    hdrs.append((b"content-length", str(len(body)).encode()))
                    await send({"type": "http.response.start", "status": start["status"], "headers": hdrs})
                    await send({"type": "http.response.body", "body": body, "more_body": False})
            else:
                await send(message)

        await self.app(scope, receive, _send)


ROBOTS_TXT = ("User-agent: *\nAllow: /\nDisallow: /gradio_api/\n"
              f"Sitemap: {APP_URL}sitemap.xml\n")


def sitemap_xml(lastmod: str | None) -> str:
    """One URL: the canonical Space page. The app URL is the same document."""
    lm = f"<lastmod>{_e(lastmod)}</lastmod>" if lastmod else ""
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            f"  <url><loc>{SPACE_URL}</loc>{lm}<changefreq>weekly</changefreq></url>\n"
            "</urlset>\n")


def mount(demo, *, model_repos: Iterable[str], static_dir: Path, og_dir: Path, **launch_kwargs):
    """Wrap the Blocks in a FastAPI app with the crawler-facing routes; returns (app, facts).

    ``launch_kwargs`` are what ``demo.launch`` would have received (theme, css, js, ...).
    Server-side rendering is forced off -- see the module docstring for the measured reason.
    """
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, PlainTextResponse, Response

    facts = build(model_repos)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/robots.txt", response_class=PlainTextResponse)
    def robots() -> str:
        return ROBOTS_TXT

    @app.get("/sitemap.xml")
    def sitemap() -> Response:
        return Response(sitemap_xml(facts["meta"].get("date")), media_type="application/xml")

    @app.get("/icon.svg")
    def icon_svg() -> FileResponse:
        return FileResponse(static_dir / "mark.svg", media_type="image/svg+xml")

    @app.get("/og.png")
    def og_png() -> FileResponse:
        return FileResponse(og_dir / "og-1200x630.png", media_type="image/png")

    @app.get("/apple-touch-icon.png")
    def apple_icon() -> FileResponse:
        return FileResponse(og_dir / "apple-touch-icon-180.png", media_type="image/png")

    launch_kwargs.setdefault("favicon_path", str(og_dir / "favicon-32.png"))
    app = gr.mount_gradio_app(app, demo, path="/", ssr_mode=False, **launch_kwargs)
    app.add_middleware(HeadRewriteMiddleware, head=facts["head"])
    return app, facts


# ---------------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------------

def validate(head: str) -> list[str]:
    """Structural checks a reviewer would do by eye; returns problems (empty = fine)."""
    problems: list[str] = []
    m = re.search(r'name="description" content="([^"]*)"', head)
    if not m:
        problems.append("no description")
    elif len(html.unescape(m.group(1))) > DESCRIPTION_MAX:
        problems.append(f"description {len(m.group(1))} chars > {DESCRIPTION_MAX}")
    for prop in ("og:title", "og:description", "og:image", "og:url", "twitter:card", "twitter:image"):
        if head.count(f'"{prop}"') != 1:
            problems.append(f"{prop} appears {head.count(chr(34) + prop + chr(34))} times")
    scripts = re.findall(r'<script type="application/ld\+json">(.*?)</script>', head, re.S)
    if len(scripts) != 2:
        problems.append(f"{len(scripts)} JSON-LD blocks, expected 2")
    for s in scripts:
        try:
            obj = json.loads(s.replace("<\\/", "</"))
            for k in ("@context", "@type", "name", "description", "url"):
                if k not in obj:
                    problems.append(f"JSON-LD {obj.get('@type')} lacks {k}")
        except ValueError as e:
            problems.append(f"JSON-LD does not parse: {e}")
    return problems


if __name__ == "__main__":
    repos = sys.argv[1:] or [f"StrandsAgents/{m}" for m in baseline_meta().get("models", [])]
    facts = build(repos)
    print(facts["head"])
    probs = validate(facts["head"])
    print("\n# description:", len(facts["description"]), "chars")
    print("# problems:", probs or "none")
    sys.exit(1 if probs else 0)
