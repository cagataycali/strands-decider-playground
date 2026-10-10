#!/usr/bin/env python
"""Self-test for metrics.py: proves it measures something real, on every backend.

    python metrics_selftest.py                 # mps (or cuda) AND cpu, synthetic module
    python metrics_selftest.py --device cpu    # one backend only
    python metrics_selftest.py --repo StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
                                               # same assertions against a REAL decider load

Asserts (these are the claims, and they are checked, not asserted in prose):
  A1  activation_bytes  > 0                       (a forward really allocates)
  A2  peak_bytes       >= baseline_bytes          (peak is not below the post-load floor)
  A3  input_tokens      > 0  and is an int        (tokenised, not estimated)
  A4  the SAME KEYS come back on every backend    (the app's table can never have holes)
  A5  latency_ms        > 0
  A6  weight_bytes      > 0                       (the load was actually measured)
  A7  reproducible: two runs agree on tokens, and activation within 25%
  A8  per_layer=True yields rows; per_layer=False yields an UNAVAILABLE string
  A9  est_flops == 2 * active_params * input_tokens
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from metrics import KEYS, backend_info, baseline_after_load, measure  # noqa: E402

FAILS: list[str] = []
CHECKS = 0


def check(cond: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILS.append(label)


class Tiny(torch.nn.Module):
    """Small stack whose forward allocates clearly more than its weights, so an
    activation measurement that is silently zero cannot pass."""

    def __init__(self, vocab: int = 4096, d: int = 256, layers: int = 4):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, d)
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))
             for _ in range(layers)]
        )
        self.head = torch.nn.Linear(d, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # kwarg name matters: the hook reads it
        h = self.emb(input_ids)
        for b in self.blocks:
            h = h + b(h)
        return self.head(h)


def run_synthetic(device: str, per_layer: bool = False) -> dict:
    dev = torch.device(device)
    with baseline_after_load.around("synthetic/Tiny", device=device, dtype="float32") as box:
        model = Tiny().to(dev).eval()
        box.model = model
    base = box.result

    ids = torch.randint(0, 4096, (2, 512), device=dev)
    with torch.inference_mode():
        model(input_ids=ids)  # warm the allocator so the measured run is not a first-touch outlier

    return measure(
        lambda: model(input_ids=ids),
        device=device,
        dtype="float32",
        active_params=sum(p.numel() for p in model.parameters()),
        repo="synthetic/Tiny",
        baseline=base,
        usd_per_hour=1.0,
        per_layer=per_layer,
        model=model,
    )


def run_real(repo: str, device: str) -> dict:
    """Same assertions, real decider checkpoint. Needs strands-decider + HF access."""
    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.modeling import StrandsDeciderModel

    state = "Help! My payouts have been failing for 3 days!"
    # choice/score take `criteria`, not `options` (the schema rejects `options`).
    questions = {
        "noul_0": {"type": "noul", "instructions": "Does this convey urgency?"},
        "choice_0": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "", "sales": "", "retail": ""},
        },
    }

    # The engine does model.to(device) and the CPU fp32 upcast, so bracket it too --
    # otherwise the weights are still on the host and the device delta reads near zero.
    with baseline_after_load.around(repo, device=device) as box:
        model = StrandsDeciderModel.load(repo)
        engine = SystemOneEngine(
            model, EngineConfig(device=device, model_name=repo.split("/")[-1])
        )
        box.model = engine.model
    base = box.result

    engine.ask(state, questions)  # warm-up: not the measured run

    return measure(
        lambda: engine.ask(state, questions),
        device=device,
        dtype=base.get("dtype"),
        active_params=sum(p.numel() for p in engine.model.parameters()),
        repo=repo,
        baseline=base,
        usd_per_hour=1.0,
        model=engine.model,
    )


def assert_core(m: dict, tag: str) -> None:
    print(f"\n[{tag}] backend={m['backend']} dtype={m['dtype']}")
    print(f"       weight {m['weight_mb']} MB · baseline {m['baseline_mb']} MB · peak {m['peak_mb']} MB")
    print(f"       ACTIVATION {m['activation_mb']} MB · tokens {m['input_tokens']} · {m['latency_ms']} ms")
    print(f"       peak_method: {m['peak_method']}")
    check(m["ok"] is True and m["error"] is None, f"{tag}: measure() ran without error ({m['error']})")
    check(isinstance(m["activation_bytes"], int) and m["activation_bytes"] > 0,
          f"A1 {tag}: activation_bytes > 0 (got {m['activation_bytes']})")
    check(isinstance(m["peak_bytes"], int) and isinstance(m["baseline_bytes"], int)
          and m["peak_bytes"] >= m["baseline_bytes"],
          f"A2 {tag}: peak >= post-load baseline ({m['peak_bytes']} >= {m['baseline_bytes']})")
    check(isinstance(m["input_tokens"], int) and m["input_tokens"] > 0,
          f"A3 {tag}: input_tokens int > 0 (got {m['input_tokens']!r})")
    check(isinstance(m["latency_ms"], float) and m["latency_ms"] > 0,
          f"A5 {tag}: latency_ms > 0 (got {m['latency_ms']})")
    check(isinstance(m["weight_bytes"], int) and m["weight_bytes"] > 0,
          f"A6 {tag}: weight_bytes > 0 (got {m['weight_bytes']!r})")
    basis = m["compute_positions"] if isinstance(m["compute_positions"], int) else m["input_tokens"]
    exp = 2.0 * float(m["active_params"]) * float(basis)
    check(abs(float(m["est_flops"]) - exp) < 1.0,
          f"A9 {tag}: est_flops == 2*active_params*compute_positions ({m['est_flops']:.3e})")
    check(isinstance(m["compute_positions"], int) and m["compute_positions"] >= m["input_tokens"],
          f"A10 {tag}: padded positions >= real tokens "
          f"({m['compute_positions']} >= {m['input_tokens']})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="one backend only (cuda|mps|cpu)")
    ap.add_argument("--repo", default=None, help="measure a real decider checkpoint too")
    args = ap.parse_args()

    devices = [args.device] if args.device else []
    if not devices:
        if torch.cuda.is_available():
            devices.append("cuda")
        elif torch.backends.mps.is_available():
            devices.append("mps")
        devices.append("cpu")

    print(f"torch {torch.__version__} · devices under test: {devices}")
    results: dict[str, dict] = {}

    for dev in devices:
        print(f"\n=== {dev} ===")
        print(f"  backend_info: {backend_info(dev)}")
        m = run_synthetic(dev)
        assert_core(m, f"synthetic/{dev}")
        results[dev] = m

        # A7 reproducibility -- a number we cannot get twice is not a metric
        m2 = run_synthetic(dev)
        same_tok = m2["input_tokens"] == m["input_tokens"]
        a1, a2 = m["activation_bytes"], m2["activation_bytes"]
        close = isinstance(a1, int) and isinstance(a2, int) and a1 > 0 and abs(a1 - a2) / a1 <= 0.25
        check(same_tok and close,
              f"A7 {dev}: reproducible (tokens {m['input_tokens']}=={m2['input_tokens']}, "
              f"activation {a1} vs {a2}, within 25%)")

        # A8 per-layer toggle
        off = m["per_layer"]
        on = run_synthetic(dev, per_layer=True)["per_layer"]
        check(isinstance(off, str) and off.startswith("UNAVAILABLE"),
              f"A8a {dev}: per_layer=False -> UNAVAILABLE string")
        check(isinstance(on, list) and len(on) > 0 and on[0]["bytes"] > 0,
              f"A8b {dev}: per_layer=True -> {len(on) if isinstance(on, list) else on} rows, "
              f"top={on[0]['layer'] if isinstance(on, list) and on else '-'}")

    # A4 key parity across backends
    for dev, m in results.items():
        missing = [k for k in KEYS if k not in m]
        check(not missing, f"A4 {dev}: all {len(KEYS)} canonical keys present (missing {missing})")
    if len(results) > 1:
        sets = {dev: set(m) for dev, m in results.items()}
        first = next(iter(sets.values()))
        check(all(s == first for s in sets.values()),
              f"A4 parity: identical key sets across {list(results)}")

    if args.repo:
        dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"\n=== REAL MODEL {args.repo} on {dev} ===")
        rm = run_real(args.repo, dev)
        assert_core(rm, f"real/{dev}")
        missing = [k for k in KEYS if k not in rm]
        check(not missing, f"A4 real: all canonical keys present (missing {missing})")

    print(f"\n{'PASS' if not FAILS else 'FAIL'}: {CHECKS - len(FAILS)}/{CHECKS} checks")
    for f in FAILS:
        print(f"  - {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
