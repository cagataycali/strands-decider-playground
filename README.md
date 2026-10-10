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

One question. Every decider. Paste a piece of state, ask yes/no, pick-one or rate-on-a-scale, and
every public [StrandsAgents](https://huggingface.co/StrandsAgents) decider answers side by side —
a probability per option, a calibrated confidence, and the measured cost of the call. The whole
public [JevBench](https://github.com/fstandhartinger/jevbench) set can be run against every model
from one button.

![The playground hero: "One question. Every decider." beside the pointer-head figure](docs/img/hero.png)

**Run it**

```bash
git clone https://github.com/cagataycali/strands-decider-playground && cd strands-decider-playground
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python app.py   # http://127.0.0.1:7860
```

Source, screenshots, the model table and how it works:
[github.com/cagataycali/strands-decider-playground](https://github.com/cagataycali/strands-decider-playground) · MIT.
