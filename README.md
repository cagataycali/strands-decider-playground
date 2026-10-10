---
title: Strands Decider Playground
emoji: 🐸
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: One question, every public Strands decider, side by side
suggested_hardware: a100-large
thumbnail: https://huggingface.co/spaces/cagataydev/strands-decider-playground/resolve/main/docs/og/og-1200x630.png
tags:
  - decision-making
  - calibration
  - probabilistic-decisions
  - jevbench
  - benchmark
  - lora
  - gemma
  - qwen
  - strands
  - agents
  - playground
models:
  - StrandsAgents/strands-decider-12B-gemma4-v1-2610
  - StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610
  - StrandsAgents/strands-decider-2B-hobson-v19
  - StrandsAgents/strands-decider-2B-hobson-v21
  - StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
  - StrandsAgents/strands-decider-E2B-gemma4-v1-2610
  - StrandsAgents/strands-decider-E4B-gemma4-v1-2610
datasets:
  - JevBench/jevbench
---

# Strands Decider Playground

One question. Every decider. Paste a piece of state, ask yes/no, pick-one or rate-on-a-scale, and
every public [StrandsAgents](https://huggingface.co/StrandsAgents) decider answers side by side —
a probability per option, a calibrated confidence, and the measured cost of the call. The whole
public [JevBench](https://github.com/fstandhartinger/jevbench) set can be run against every model
from one button.

![The playground hero: "One question. Every decider." beside the concept figure](docs/img/hero.png)

**Run it**

```bash
git clone https://github.com/cagataycali/strands-decider-playground && cd strands-decider-playground
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python app.py   # http://127.0.0.1:7860
```

Source, screenshots, the model table and how it works:
[github.com/cagataycali/strands-decider-playground](https://github.com/cagataycali/strands-decider-playground) · MIT.
