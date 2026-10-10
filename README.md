---
title: Strands Decider Playground
emoji: 🎛️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
short_description: Every Strands decider answers one question, side by side
---

# Strands Decider Playground

Type one decision question and watch **every post-tuned StrandsAgents decider** answer it side
by side — with per-option probabilities, calibrated confidence, each model's own published
evals, and the measured compute cost of the call.

> This Space is **private**. Visibility is set through the API by lane DEPLOY, not by this card.

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

- **The model list is read live from the org at start.** There is no hardcoded list, which is
  how two models nobody had written down (`2B-gemma4`, `4b-gemma4`) appear on their own.
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
| `app.py` | lane APP | Gradio UI + the headless `answer()` entry point |
| `registry.py` | lane APP | live org read, capability detection, eval fetch |
| `runner.py` | lane APP | one locked call: load → answer → measure |
| `cache.py` | lane APP | base-keyed LRU with eviction and `plan()` |
| `metrics.py` | lane METRICS | the compute measurement (imported, never edited) |

## Golden regression check

`lanes/APP/GOLDEN.md` records `2B-qwen3.5-v1-2610` on CPU reproducing every published digit of
the README example (noul 0.8581 vs 0.858 · choice `billing` 0.7728/conf 0.6591 vs 0.773/0.659 ·
score 1.0345/conf 0.5091, `frustrated` 0.5738 vs 1.03/0.509/0.574). Max absolute delta on any
probability: **0.0003**. Click **Load golden question** in the app to re-run it.
