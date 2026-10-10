# How it works

The page is a thin Gradio front on [strands-decider](https://github.com/strands-labs/strands-decider).
Every number you see is read from one result dict per model; nothing is recomputed in the renderer.

![architecture](architecture.svg)

| file | what |
|---|---|
| `app.py` | the Gradio page, the headless `answer()` entry point, and the card/comparison renderers |
| `registry.py` | live org read, public filter, capability detection, published evals, active-parameter arithmetic |
| `runner.py` | one locked call per model: load → answer → measure; question building; cross-model comparison |
| `cache.py` | base-keyed model cache with eviction and a dry-run `plan()` |
| `metrics.py` | the measurement: memory, latency, tokens, clearly-labelled estimates; same keys on cuda/mps/cpu |
| `landing.py` · `examples.py` · `viz.py` | hero + how-to + model row · clickable example gallery · decision diagrams |
| `bench/` | vendored JevBench tasks, pinned scoring, runner, board, error panels, parity gate |
| `bench/baseline/` · `bench/capture.py` | the shipped 231 × 7 baseline captured on the Space, and the CLI that captures it over the network |
| `static/` | design tokens, stylesheets, motion, the frog mark and wordmark, self-hosted fonts |

## A decider, in one paragraph

A *decider* is a small post-tuned language model that answers a typed question about a piece of
state — yes/no (`noul`), pick one (`choice`), or rate on a rubric (`score`) — with a probability for
every option instead of free text. The LM head is discarded; a pointer head reads the last hidden
state through one masked softmax, so the answer is a distribution, not a sentence. Models are
labelled on the page by **base + LoRA rank** — the digits in a repo name are not a size.

## Things this app refuses to get wrong

- **The model list is read live from the org at start**, then filtered to public repos by the Hub's
  own `private` flag. There is no hardcoded list; `registry.py` is the whole rule.
- **Images only go to models that can read them.** Per the library's `docs/vision.md`, image input
  works on Qwen3.5 bases with a pointer head; Gemma 4 checkpoints load their base's text decoder
  and drop the vision encoder. Picking an image with a text-only model gives a clear refusal, not a
  silently text-only answer.
- **Load time is never added to inference latency.** A cold model must not look slow: the card
  shows them as two numbers and marks the call COLD or warm.
- **Every model call is serialized behind one lock**, so memory and latency are attributable to one
  request. Big models evict by base model rather than thrash; a model that cannot fit the device at
  all says so.
- **A number that cannot be reproduced is not shown.** Unmeasurable fields arrive as
  `UNAVAILABLE: <reason>`, never as a plausible zero.

## The golden question

`strands-decider-2B-qwen3.5-v1-2610` on the library README example (*"Help! My payouts have been
failing for 3 days!"*) reproduces every published digit: `noul` 0.858 · `choice` billing 0.773 with
confidence 0.659 · `score` 1.03 with confidence 0.509. On the Space's A100 (bf16) the same call
gives 0.856 / billing / 1.034 on the `flash-linear-attention` kernel (see *Kernels* below): the
fourth decimal moves between Space restarts — 0.8562 / 1.0343 on one boot, 0.8561 / 1.0336 on the
next — because Triton picks kernel configurations by timing them; within one running process every
repeat is identical. The reference PyTorch path gave 0.8553 / billing / 1.0354 on every boot. If you
change anything on the answer path, press **Golden question** and compare at three decimals.

## The bench

`bench/` carries the 231 public JevBench tasks, vendored at a pinned commit
(`bench/tasks/PIN.json` has the commit and the sha256 of every file; `bench/tasks/LICENSE-jevbench`
is their MIT licence). Scoring is JevBench's own code at that same commit — `bench/jev.py` imports
it, it is never re-implemented. `bench/parity.py` re-scores the three public model repos' own
published per-task records and reproduces their README totals exactly (180 / 167 / 207 of 231);
run it after touching `bench/`.

**The baseline** on the board is `bench/baseline/results.jsonl`: every public model × all 231 tasks
(1617 cells), captured once on the Space (A100 80 GB, bf16) and committed as data. Nothing is
computed on page load; the per-cell latency is the forward pass the Space measured.
`bench/baseline/meta.json` records date, hardware, dtype, library versions, run ids and per-model
totals. `bench/capture.py` made it and can redo it from any machine with a token that can read the
Space — no model is loaded locally:

```bash
python -m bench.capture baseline          # one run per model on the Space, merged into bench/baseline/
python -m bench.capture run --scope "tier: easy" --model strands-decider-2B-qwen3.5-v1-2610 --out /tmp/easy
```

One run per model is deliberate: the 26B base is exclusive on an 80 GB card, so a task-major run
over all 7 models reloads weights on every task (measured 9 s/cell); model-major is ~50 s per model.

**Run again** from the page: pick a scope and a model (or all) and press **Run JevBench**. The run
calls `app.answer()` exactly as the Decide button does, appends one row per task × model to
`bench/runs/<run_id>/results.jsonl`, and shows progress on the board. When it ends the folder is
uploaded to a Hugging Face dataset (`BENCH_DATASET_REPO`, default
`cagataydev/strands-decider-playground-bench`; skipped without a token). The History dropdown lists
the baseline first, then the runs made on this Space; italic rows are the published runs from the
strands-decider repo.

**Kernels.** The Space runs torch 2.14.1+cu130 on Python 3.13 with `flash-linear-attention` 0.5.2
(Triton, pure Python — nothing compiles). It matters only for the three Qwen3.5-based models, whose
18 of 24 layers are gated-delta-net; the four Gemma 4 models never touch that code. Measured on
2026-10-10 with 20 warm requests per model (`torch.cuda.Event` around the forward, 86-token input),
the kernel makes the Qwen3.5 forward 12–17 % faster than transformers' reference PyTorch path
(102.5 → 85.4 ms, 96.5 → 83.3, 94.6 → 83.0 at p50) and moves probabilities in the third to fourth
decimal — a different accumulation order in bf16, not a different answer: the golden question's
0.8553 became 0.856x and its score 1.0354 became 1.034x (the fourth decimal differs per boot, see
above), every label stayed the same. On the 231-task bench the re-capture changed 3 of 693
predictions, all at p ≈ 0.49–0.52, two of them from wrong to right. The baseline
rows for those three models were re-captured on the kernel the same day; `bench/baseline/meta.json`
records it under `kernels`. `causal_conv1d` stays on the reference path: no build exists for this
torch (wheels stop at 2.10, Hub kernels at 2.12) and it is the small part of the cost.

## Environment

`HF_TOKEN` is read from the environment only (models are public; the token is for pushing bench
runs). `DECIDER_DEVICE=cpu|mps|cuda` overrides the device pick; `DECIDER_ORG` / `DECIDER_PREFIX`
point the registry at another org or name prefix.

## The look

Design tokens come from the strands-labs/robots documentation: flat black or white surfaces, one
green (`#00cc60` dark, `#007a3d` light), 1 px ink outlines, 8 px radius, JetBrains Mono headings over
Space Grotesk text. Motion is 120–600 ms and lands instantly under `prefers-reduced-motion`.
