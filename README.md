---
title: Strands Decider Playground
emoji: 🐸
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
short_description: One question, every public Strands decider, side by side
---

# Strands Decider Playground

One question. Every decider. Ask once and every **public** StrandsAgents decider answers side
by side — a probability for each option, a calibrated confidence, the model's own published
evals, and the measured cost of the call.

A decider is a small post-tuned language model that answers a typed question about a piece of
state — yes/no, pick one, or rate on a rubric — with a probability for every option instead of
free text.

> This Space is **private**. Visibility is set through the API by lane DEPLOY, not by this card.
> Only repos whose Hub record says `private: false` are listed; the footer count is that filter's
> result, read live at start (7 at the time of writing).

## The page

Design tokens come from the strands-labs/robots docs (`docs/stylesheets/extra.css`): flat black
or white surfaces, one green (`#00cc60` dark, `#007a3d` light), 1 px ink outlines, 8 px radius,
JetBrains Mono headings over Space Grotesk text, the pixel frog mark. Everything lives in
`static/` (`tokens.css`, `app.css`, `app.js`, `mark.svg`, `wordmark.svg`, self-hosted fonts).

1. **Ask bar** — state, one question with its type, options as chips (type, Enter adds). Up to
   eight questions. `Enter` runs, `Esc` clears.
2. **Cards stream in one model at a time**: skeletons with a quiet shimmer are replaced as each
   model finishes. Cards are labelled by **base + LoRA rank** — the digits in a repo name are not
   a size (`2B-gemma4` is an E2B, `4b-gemma4` an E4B). Probability bars grow from zero, the
   confidence ring draws itself, the argmax is the one green element.
3. **Comparison** is a diagram: an agreement strip with each model's argmax (majority filled),
   small multiples of the full distribution per option, and the winning-probability spread.
4. **Compute** per card: a memory stack (weights → baseline after load → peak, the activation
   band = peak − baseline in green), load vs inference on one linear axis — load is ~52× a
   forward, and the picture says so — FLOPs with the active-parameter footnote for the MoE,
   and $/h priced at the selected hardware. Numbers count up on reveal; every metric has a
   one-sentence tooltip.
5. **Published evals** as badges that link to the exact `eval/**/summary.json` in the repo.
6. **Footer**: public model count · hardware · torch / transformers / gradio · built by strands.

Motion is 120–600 ms and lands instantly under `prefers-reduced-motion`. No emoji in the UI.

## What it shows, per model

- **The answer** for each question you build — `noul` (yes/no), `choice` (pick one of N), or
  `score` (rate against an ascending rubric).
- **A per-option probability bar** for every question.
- **Calibrated confidence** — except on `noul`, which has no confidence field *by design*: with
  two outcomes the probability **is** the uncertainty, so the app shows that sentence instead of
  inventing a number.
- **The model's published evals**, read live from its own `eval/**/summary.json`. Paths are
  globbed, not named: `2B-hobson-v19` carries both `jevbench-w3072` and `-w4096`, and
  `2B-gemma4` also ships `jf100` and `typed-decisions`.
- **The compute block** from `metrics.py` — weights, post-load baseline, peak, **activation
  memory (peak − baseline)**, real tokenised input length, latency, estimated FLOPs and $.

Plus a **comparison table** (rows = models, columns = questions) and an **agreement row**:
the majority answer, which models dissent, and the spread of the winning probability.

## Things this app refuses to get wrong

- **The model list is read live from the org at start**, then filtered to **public** repos
  (`model_info().private == False`). There is no hardcoded list; the two private retrains
  (`2B-gemma4`, `4b-gemma4`) are dropped by the Hub flag, not by name.
- **Images only go to models that can read them.** The rule is not a config flag — no decider
  config has one, and *every* base model carries a `vision_config`. Per `docs/vision.md`, images
  work on **Qwen3.5 bases with a pointer head**; the Gemma 4 checkpoints load their base's text
  decoder and drop its vision encoder. Vision-capable: `2B-hobson-v19`, `2B-hobson-v21`,
  `2B-qwen3.5-v1-2610`. Picking an image with a Gemma 4 model gives you a clear refusal, not a
  silently text-only answer.
- **Load time is never added to inference latency.** A cold model must not look slow, so the
  card shows them as two numbers and marks the call COLD or warm.
- **Every model call is serialized behind one lock.** Concurrent requests crash the Metal
  command encoder, and the lock is what makes memory and latency attributable to one request.
- **Big models evict, they don't thrash.** Admission is base-keyed (9 models → 5 distinct
  bases). `26B-A4B` is marked *exclusive*: it cannot be resident on 48 GB at all (48.07 GiB of
  weights), so selecting it unloads the rest, and the card says what was evicted.
- **A number that cannot be reproduced is not shown.** Unmeasurable fields arrive as
  `UNAVAILABLE: <reason>`, never as a plausible zero.

## Files

| file | owner | what |
|---|---|---|
| `app.py` | lane APP (`answer()`), lane DESIGN (layout + rendering) | Gradio UI + the headless `answer()` entry point |
| `registry.py` | lane APP (+ DESIGN: public-only filter) | live org read, public filter, capability detection, eval fetch |
| `static/` | lane DESIGN | tokens, stylesheet, motion, mark, wordmark, fonts |
| `runner.py` | lane APP | one locked call: load → answer → measure |
| `cache.py` | lane APP | base-keyed LRU with eviction and `plan()` |
| `metrics.py` | lane METRICS | the compute measurement (imported, never edited) |

## Golden regression check

`lanes/APP/GOLDEN.md` records `2B-qwen3.5-v1-2610` on CPU reproducing every published digit of
the README example (noul 0.8581 vs 0.858 · choice `billing` 0.7728/conf 0.6591 vs 0.773/0.659 ·
score 1.0345/conf 0.5091, `frustrated` 0.5738 vs 1.03/0.509/0.574). Max absolute delta on any
probability: **0.0003**. Re-run through `app.answer()` after the DESIGN layout rewrite
(`lanes/DESIGN/golden_via_app.json`): max |delta| vs that recording **0.0**. Click **Load golden
question** in the app to re-run it yourself.
