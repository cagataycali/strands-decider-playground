# Contributing

Thanks for looking. This is a small Gradio app; the rules below keep it honest.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python app.py            # http://127.0.0.1:7860
```

The model list is read from the Hub at start. On a laptop, untick everything but
`strands-decider-2B-qwen3.5-v1-2610` before pressing Decide: the 2B runs on CPU in well under a
minute; the other six want a GPU. `DECIDER_DEVICE=cpu|mps|cuda` overrides the device pick.

## Before you open a pull request

1. `python -m py_compile *.py bench/*.py`
2. Render check without loading a model: `python -c "import app"` must succeed, and every
   `*_html()` in `app.py`, `landing.py`, `examples.py`, `viz.py` must render a captured result
   dict (the Raw result accordion in the app gives you one).
3. If you touched anything on the answer path (`runner.py`, `cache.py`, `metrics.py`), re-run
   the golden question (the **Golden question** button, 2B model) and compare with the
   numbers in the README. They must match to four decimals.
4. `python metrics_selftest.py --device cpu` when `metrics.py` changed.
5. `python bench/parity.py` when anything under `bench/` changed: the published JevBench
   totals must still reproduce exactly.

## Rules of the house

* Nothing user-visible is typed in: model lists come from the Hub, eval numbers from each
  repo's own `summary.json`, latencies from measurement. A number that cannot be reproduced
  is shown as `UNAVAILABLE: <reason>`, never as a plausible zero.
* Load time and inference latency are never added together.
* No secrets in the tree or in history. The app reads `HF_TOKEN` from the environment only.
* Keep the design system: one green, 1 px outlines, 8 px radius, JetBrains Mono over Space
  Grotesk, motion under 600 ms and off under `prefers-reduced-motion`. No emoji in the UI.
