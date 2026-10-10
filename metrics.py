"""Per-request compute measurement: memory, latency, tokens, and clearly-labelled estimates.

`runner.py` wraps every model load in `baseline_after_load.around()` and every forward in
`measure()`; the dict that comes back is what the compute block on each card shows.

Design rules this file obeys:
  * Every backend (cuda | mps | cpu) returns THE SAME KEYS, so the app's table never has holes.
  * A field that cannot be measured is the STRING "UNAVAILABLE: <reason>" -- never missing, never None,
    never a silently-wrong zero.
  * The headline number is ACTIVATION memory = peak-during-forward minus post-load baseline.
  * Nothing here is estimated unless its key says `est_`; `$/h` and `active_params` are INPUTS
    passed by the caller, not constants invented in this file.

Public API:

    baseline_after_load(repo=None, *, device=None, model=None, load_s=None,
                        weight_bytes=None, dtype=None) -> dict
        Call immediately AFTER the model load and BEFORE any forward.
        Pass it to measure(baseline=...). Also usable as a context manager around the load:

            with baseline_after_load.around(repo, device="mps") as b:
                model = load(...)
            b.result  ->  the dict (weight_bytes + load_s measured for you)

    measure(fn, *, device, dtype=None, active_params=None, repo=None, baseline=None,
            tokens=None, tokenizer=None, text=None, usd_per_hour=None,
            per_layer=False, model=None) -> dict
        Runs fn() exactly ONCE and returns the metrics dict (plus ["result"] = fn()'s return).

    count_tokens(tokenizer, text) -> int          # real tokenisation, no estimate
    device_label(obj) -> str                      # "cuda" | "mps" | "cpu"
    KEYS                                          # canonical key order for the app's table
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable

import torch

__all__ = [
    "measure",
    "baseline_after_load",
    "count_tokens",
    "device_label",
    "backend_info",
    "KEYS",
    "UNAVAILABLE",
]

# Poll interval for the sampling peak-tracker (MPS and CPU have no native peak counter).
# 0 = yielding spin (time.sleep(0)), which is the only way to get real resolution on macOS:
# MEASURED on an Apple-silicon machine: time.sleep(tiny) has ~1.7 ms granularity so a 12 ms forward got
# 1-2 samples, while sleep(0) gives ~16,500 reads/ms AND leaves the forward's own time
# unchanged (12.00 ms). A sampler that never yields at all is NOT the answer: it starved
# the main thread and the same forward took 444 ms.
_SAMPLE_S = float(os.environ.get("METRICS_SAMPLE_S", "0"))


def UNAVAILABLE(reason: str) -> str:  # noqa: N802  (it is a value constructor, not a class)
    return f"UNAVAILABLE: {reason}"


#: Canonical key order. Every backend returns exactly these keys.
KEYS: tuple[str, ...] = (
    "repo",
    "backend",
    "device",
    "dtype",
    # --- memory ---
    "weight_bytes",
    "weight_mb",
    "baseline_bytes",
    "baseline_mb",
    "peak_bytes",
    "peak_mb",
    "activation_bytes",
    "activation_mb",
    "driver_bytes",
    "driver_mb",
    "peak_method",
    "mem_method",
    # --- work ---
    "input_tokens",
    "compute_positions",
    "token_method",
    "forward_calls",
    # --- time ---
    "load_s",
    "latency_ms",
    "latency_method",
    # --- derived (clearly marked estimates) ---
    "active_params",
    "est_flops",
    "est_tflops",
    "est_flops_basis",
    "usd_per_hour",
    "est_usd",
    # --- optional detail ---
    "per_layer",
    "per_layer_total_bytes",
    # --- bookkeeping ---
    "ok",
    "error",
)

_MB = 1024.0 * 1024.0


def _mb(v: Any) -> Any:
    """Bytes -> MiB to 3 places; non-numbers (UNAVAILABLE strings) pass through."""
    return round(v / _MB, 3) if isinstance(v, (int, float)) else v


# --------------------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------------------


def device_label(obj: Any = None) -> str:
    """'cuda' | 'mps' | 'cpu' from a str, torch.device, tensor or nn.Module."""
    if obj is None:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if isinstance(obj, str):
        return obj.split(":")[0].strip().lower()
    if isinstance(obj, torch.device):
        return obj.type
    dev = getattr(obj, "device", None)
    if dev is not None:
        return device_label(dev)
    try:  # nn.Module: ask its first parameter
        return device_label(next(obj.parameters()).device)
    except Exception:
        return device_label(None)


class _Backend:
    """One memory/clock interface over cuda, mps and cpu.

    `allocated()` is the number activation memory is computed from: it must mean
    'bytes of tensor storage this process holds right now' on every backend.
    """

    name = "cpu"
    native_peak = False
    mem_method = "unknown"

    def sync(self) -> None:
        pass

    def allocated(self) -> int | str:
        return UNAVAILABLE("no allocator counter on this backend")

    def driver(self) -> int | str:
        return UNAVAILABLE("no driver counter on this backend")

    def reset_peak(self) -> None:
        pass

    def peak(self) -> int | str:
        return UNAVAILABLE("no native peak counter on this backend")

    def empty_cache(self) -> None:
        pass


class _Cuda(_Backend):
    name = "cuda"
    native_peak = True
    mem_method = "torch.cuda.memory_allocated"

    def sync(self) -> None:
        torch.cuda.synchronize()

    def allocated(self) -> int:
        return int(torch.cuda.memory_allocated())

    def driver(self) -> int:
        return int(torch.cuda.memory_reserved())

    def reset_peak(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def peak(self) -> int:
        return int(torch.cuda.max_memory_allocated())

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()


class _Mps(_Backend):
    name = "mps"
    native_peak = False  # torch.mps exposes no max_memory_allocated
    mem_method = "torch.mps.current_allocated_memory"

    def sync(self) -> None:
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def allocated(self) -> int | str:
        if hasattr(torch.mps, "current_allocated_memory"):
            return int(torch.mps.current_allocated_memory())
        return UNAVAILABLE("torch.mps.current_allocated_memory missing in torch %s" % torch.__version__)

    def driver(self) -> int | str:
        if hasattr(torch.mps, "driver_allocated_memory"):
            return int(torch.mps.driver_allocated_memory())
        return UNAVAILABLE("torch.mps.driver_allocated_memory missing in torch %s" % torch.__version__)

    def empty_cache(self) -> None:
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


class _Cpu(_Backend):
    name = "cpu"
    native_peak = False
    mem_method = "psutil RSS"

    def __init__(self) -> None:
        try:
            import psutil

            self._p = psutil.Process(os.getpid())
            self._why = None
        except Exception as exc:  # pragma: no cover
            self._p = None
            self._why = f"psutil unavailable ({exc})"

    def allocated(self) -> int | str:
        if self._p is None:
            return UNAVAILABLE(self._why or "psutil unavailable")
        return int(self._p.memory_info().rss)

    def driver(self) -> int | str:
        # On CPU the process RSS *is* the driver-level footprint; report vms so the
        # key still carries a real, different measurement rather than a duplicate.
        if self._p is None:
            return UNAVAILABLE(self._why or "psutil unavailable")
        return int(self._p.memory_info().vms)


def _backend(device: Any) -> _Backend:
    """The backend object for a device label."""
    label = device_label(device)
    if label == "cuda":
        return _Cuda()
    if label == "mps":
        return _Mps()
    return _Cpu()


def backend_info(device: Any = None) -> dict:
    """What this machine can actually measure -- printed by the selftest."""
    be = _backend(device)
    return {
        "backend": be.name,
        "mem_method": be.mem_method,
        "native_peak": be.native_peak,
        "torch": torch.__version__,
        "allocated_now": be.allocated(),
        "driver_now": be.driver(),
    }


# --------------------------------------------------------------------------------------
# peak tracking
# --------------------------------------------------------------------------------------


class _Sampler:
    """Background max-tracker for backends with no native peak counter.

    Honest about itself: the returned peak is a SAMPLED maximum, and the method string
    says so, so nobody reads it as an exact allocator high-water mark.
    """

    #: MEASURED with a real 2B decider on CPU: a yielding spin costs the
    #: forward +2.1 to +4.2 s on a ~11.4 s CPU inference (18-30%), because on CPU the
    #: sampler competes with torch's own 10 compute threads. On MPS the same spin is free
    #: (12.00 ms with and without). So the sampler SPINS only where the main thread is
    #: merely enqueueing kernels, and sleeps on CPU -- where RSS moves slowly anyway.
    _CPU_INTERVAL = 0.002

    def __init__(self, be: _Backend, interval: float | None = None):
        self._be = be
        if interval is None:
            interval = self._CPU_INTERVAL if be.name == "cpu" else _SAMPLE_S
        self._interval = interval
        self._stop = threading.Event()
        self._max = 0
        self._n = 0
        self._t: threading.Thread | None = None

    def _run(self) -> None:
        # time.sleep (not Event.wait) and a short interval: Event.wait under GIL contention
        # with a busy main thread was measured taking only ONE sample across a 25 ms forward,
        # which would make the peak a coin flip rather than a measurement.
        while not self._stop.is_set():
            v = self._be.allocated()
            if isinstance(v, int):
                self._n += 1
                if v > self._max:
                    self._max = v
            time.sleep(self._interval)  # 0 => yield the GIL without the ~1.7 ms OS timer

    def start(self) -> None:
        v = self._be.allocated()
        self._max = v if isinstance(v, int) else 0
        # The sampler is a Python thread competing for the GIL with a main thread that is
        # dispatching kernels. At the default 5 ms switch interval it was MEASURED taking
        # only 2 samples across a 26 ms forward -- so lower the interval for the window and
        # put it back afterwards. Measured effect: 2 samples -> hundreds.
        self._switch = sys.getswitchinterval()
        try:
            sys.setswitchinterval(min(self._switch, max(self._interval / 4.0, 1e-5)))
        except Exception:
            pass
        self._t = threading.Thread(target=self._run, daemon=True, name="metrics-peak")
        self._t.start()

    def stop(self) -> tuple[int, int]:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)
        try:
            sys.setswitchinterval(self._switch)
        except Exception:
            pass
        v = self._be.allocated()  # one last read after sync, in case the peak is still live
        if isinstance(v, int) and v > self._max:
            self._max = v
        return self._max, self._n


# --------------------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------------------


def count_tokens(tokenizer: Any, text: Any) -> int | str:
    """Real tokenisation (never len(text)/4). `text` may be a str or a list of str."""
    if tokenizer is None:
        return UNAVAILABLE("no tokenizer passed")
    try:
        items: Iterable[str] = [text] if isinstance(text, str) else list(text)
        total = 0
        for t in items:
            enc = tokenizer(t, add_special_tokens=True)
            ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "__getitem__") else enc
            if ids and isinstance(ids[0], (list, tuple)):
                total += sum(len(x) for x in ids)
            else:
                total += len(ids)
        return int(total)
    except Exception as exc:
        return UNAVAILABLE(f"tokenizer failed ({type(exc).__name__}: {exc})")


class _TokenHook:
    """Counts the token positions actually fed to the torso, by watching forward().

    This is the tokenised truth for FLOPs: it sees every forward the engine makes and
    sums the real input_ids / inputs_embeds positions, padding included, because padding
    costs FLOPs too.

    Why it hooks more than the top module: strands_decider's engine has TWO paths -- the
    batched one calls `model(...)`, but the DEFAULT shared-prefix one calls
    `model.torso(...)` directly, so a hook on the top module alone would count ZERO
    tokens on the path that actually runs. Every plausible entry module is hooked and a
    depth counter keeps only the OUTERMOST call, so nesting cannot double-count.
    """

    #: attribute names that are the real entry point on a wrapper model
    _INNER = ("torso", "model", "backbone", "transformer", "base_model")

    def __init__(self, model: Any):
        self.model = model
        self.tokens = 0      # REAL tokens (attention_mask sum): what the user was charged for
        self.positions = 0   # PADDED positions the torso actually ran: what the FLOPs cost
        self.calls = 0
        self._hs: list[Any] = []
        self._depth = 0
        self.why: str | None = None

    def _targets(self) -> list[Any]:
        out: list[Any] = []
        seen: set[int] = set()
        for m in [self.model] + [getattr(self.model, n, None) for n in self._INNER]:
            if isinstance(m, torch.nn.Module) and id(m) not in seen:
                seen.add(id(m))
                out.append(m)
        return out

    def __enter__(self) -> "_TokenHook":
        if self.model is None:
            self.why = "no model= passed"
            return self
        try:
            for m in self._targets():
                self._hs.append(m.register_forward_pre_hook(self._pre, with_kwargs=True))
                self._hs.append(m.register_forward_hook(self._post))
        except Exception as exc:
            self.why = f"forward hook failed ({type(exc).__name__}: {exc})"
        return self

    @staticmethod
    def _positions(args: tuple, kwargs: dict) -> tuple[int, int]:
        """(padded positions, real tokens) for one forward.

        The two differ and BOTH are true. MEASURED on the golden request, 2B decider:
        the engine runs forward(1, 21) for the shared prefix and forward(3, 65) for the
        three questions right-padded to the longest -> 216 padded positions, while the
        attention masks sum to 21 + (65+47+63) = 196, which is EXACTLY the number the
        engine itself reports as usage.input_tokens. Padding costs FLOPs, so FLOPs use
        the padded count; the user-facing token count is the real one.
        """
        t = kwargs.get("input_ids")
        if t is None:
            t = kwargs.get("inputs_embeds")
        if t is None:
            for a in args:
                if torch.is_tensor(a):
                    t = a
                    break
        if not torch.is_tensor(t):
            return 0, 0
        # [batch, seq] of ids, or [batch, seq, hidden] of embeds
        padded = int(t.shape[0] * t.shape[1]) if t.dim() >= 2 else int(t.numel())
        real = padded
        mask = kwargs.get("attention_mask")
        if torch.is_tensor(mask) and mask.dim() == 2 and t.dim() >= 2:
            # The mask spans CACHE + CURRENT when a KV cache is in play, so a bare
            # mask.sum() counts the shared prefix once per row. MEASURED: it returned 259
            # for a request whose real size is 196 (prefix 21 counted 3 extra times, plus
            # 175). Slicing to the current segment's own columns gives exactly 196, which
            # is the number the engine itself reports -- an independent confirmation.
            seq = int(t.shape[1])
            try:
                real = int(mask[:, -seq:].sum())
            except Exception:
                real = padded
            if real <= 0 or real > padded:
                real = padded  # a 4-D/additive/float mask: do not pretend to read it
        return padded, real

    def _pre(self, _mod: Any, args: tuple, kwargs: dict) -> None:
        if self._depth == 0:  # outermost call only
            padded, real = self._positions(args, kwargs)
            if padded:
                self.calls += 1
                self.positions += padded
                self.tokens += real
        self._depth += 1

    def _post(self, _mod: Any, _args: Any, _out: Any) -> None:
        self._depth = max(0, self._depth - 1)

    def __exit__(self, *exc: Any) -> None:
        for h in self._hs:
            try:
                h.remove()
            except Exception:
                pass
        self._hs.clear()


# --------------------------------------------------------------------------------------
# per-layer activations (opt-in: hooks are slow)
# --------------------------------------------------------------------------------------


class _PerLayer:
    """Output-tensor bytes per leaf module, via forward hooks. Default OFF."""

    def __init__(self, model: Any, enabled: bool, top: int = 40):
        self.enabled = bool(enabled)
        self.model = model
        self.top = top
        self._hs: list[Any] = []
        self._by: dict[str, int] = {}
        self._cnt: dict[str, int] = {}
        self.why: str | None = None

    @staticmethod
    def _bytes(out: Any) -> int:
        if torch.is_tensor(out):
            return int(out.numel() * out.element_size())
        if isinstance(out, (list, tuple)):
            return sum(_PerLayer._bytes(o) for o in out)
        if isinstance(out, dict):
            return sum(_PerLayer._bytes(o) for o in out.values())
        return 0

    def __enter__(self) -> "_PerLayer":
        if not self.enabled:
            self.why = "per_layer=False (forward hooks are slow; pass per_layer=True)"
            return self
        if self.model is None:
            self.why = "per_layer=True but no model= passed"
            return self
        try:
            for name, mod in self.model.named_modules():
                if name and not list(mod.children()):  # leaves only
                    self._hs.append(mod.register_forward_hook(self._mk(name)))
        except Exception as exc:
            self.why = f"hook registration failed ({type(exc).__name__}: {exc})"
        return self

    def _mk(self, name: str) -> Callable:
        def hook(_m: Any, _i: Any, out: Any) -> None:
            self._by[name] = self._by.get(name, 0) + self._bytes(out)
            self._cnt[name] = self._cnt.get(name, 0) + 1

        return hook

    def __exit__(self, *exc: Any) -> None:
        for h in self._hs:
            try:
                h.remove()
            except Exception:
                pass
        self._hs.clear()

    def report(self) -> tuple[Any, Any]:
        if self.why:
            return UNAVAILABLE(self.why), UNAVAILABLE(self.why)
        rows = sorted(self._by.items(), key=lambda kv: -kv[1])
        total = sum(self._by.values())
        out = [
            {
                "layer": n,
                "bytes": b,
                "mb": _mb(b),
                "calls": self._cnt.get(n, 0),
                "pct_of_total": round(100.0 * b / total, 2) if total else 0.0,
            }
            for n, b in rows[: self.top]
        ]
        return out, total


# --------------------------------------------------------------------------------------
# baseline_after_load
# --------------------------------------------------------------------------------------


def baseline_after_load(
    repo: str | None = None,
    *,
    device: Any = None,
    model: Any = None,
    load_s: Any = None,
    weight_bytes: Any = None,
    dtype: Any = None,
) -> dict:
    """The post-load baseline the activation number is measured against.

    Call it right after the model is on the device and before any forward pass.
    If `device` is omitted it is read off `model`.
    """
    dev = device if device is not None else (model if model is not None else None)
    be = _backend(dev)
    be.sync()
    alloc = be.allocated()

    if weight_bytes is None and model is not None:
        try:
            seen: set[int] = set()
            tot = 0
            for t in list(model.parameters()) + list(model.buffers()):
                if id(t) not in seen:
                    seen.add(id(t))
                    tot += int(t.numel() * t.element_size())
            weight_bytes = tot
        except Exception as exc:
            weight_bytes = UNAVAILABLE(f"param walk failed ({type(exc).__name__}: {exc})")
    if weight_bytes is None:
        weight_bytes = UNAVAILABLE("not measured: pass weight_bytes= or model=")

    if dtype is None and model is not None:
        try:
            dtype = str(next(model.parameters()).dtype).replace("torch.", "")
        except Exception:
            dtype = None

    return {
        "repo": repo,
        "backend": be.name,
        "device": device_label(dev),
        "dtype": dtype or UNAVAILABLE("not measured: pass dtype= or model="),
        "baseline_bytes": alloc,
        "baseline_mb": _mb(alloc),
        "driver_bytes": be.driver(),
        "weight_bytes": weight_bytes,
        "weight_mb": _mb(weight_bytes),
        "load_s": load_s if load_s is not None else UNAVAILABLE("not timed: pass load_s= or use baseline_after_load.around()"),
        "mem_method": be.mem_method,
    }


@contextmanager
def _around(repo: str | None = None, *, device: Any = None, dtype: Any = None):
    """`with baseline_after_load.around(repo, device=...) as b:` -- times the load and
    measures the allocator delta across it, then fills b.result with the baseline dict."""

    class _Box:
        result: dict | None = None

    box = _Box()
    be = _backend(device)
    gc.collect()
    be.empty_cache()
    be.sync()
    before = be.allocated()
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        be.sync()
        load_s = round(time.perf_counter() - t0, 4)
        after = be.allocated()
        delta = (after - before) if isinstance(before, int) and isinstance(after, int) else None
        # On CUDA/MPS the allocator delta is exact. On CPU it is an RSS delta, and that was
        # MEASURED swinging 5204-7509 MB for the SAME 2B model (file cache, the engine's
        # fp32 torso upcast, malloc arenas), so the deterministic parameter walk is used
        # there instead and the RSS delta is reported beside it, not in place of it.
        use_delta = delta is not None and delta > 0 and be.name != "cpu"
        box.result = baseline_after_load(
            repo,
            device=device,
            model=getattr(box, "model", None),
            load_s=load_s,
            weight_bytes=delta if use_delta else None,
            dtype=dtype,
        )
        box.result["weight_method"] = (
            f"{be.mem_method} delta around the load"
            if use_delta
            else ("parameter+buffer walk (CPU RSS delta is too noisy to publish)"
                  if be.name == "cpu" else "parameter+buffer walk")
        )
        box.result["weight_rss_delta_bytes"] = delta if delta is not None else UNAVAILABLE(
            "allocator counter unavailable on this backend"
        )


baseline_after_load.around = _around  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# measure
# --------------------------------------------------------------------------------------


def measure(
    fn: Callable[[], Any],
    *,
    device: Any,
    dtype: Any = None,
    active_params: Any = None,
    repo: str | None = None,
    baseline: dict | None = None,
    tokens: Any = None,
    tokenizer: Any = None,
    text: Any = None,
    usd_per_hour: Any = None,
    per_layer: bool = False,
    model: Any = None,
) -> dict:
    """Run fn() ONCE and report what it cost. Returns every key in KEYS.

    device        : "cuda" | "mps" | "cpu" (or anything device_label understands)
    baseline      : the dict from baseline_after_load(); supplies weight/baseline/load_s
    active_params : ACTIVE parameter count (MoE: active, not total), supplied by the caller
    usd_per_hour  : hardware price, supplied by the caller -- never invented here
    model         : the nn.Module, so input tokens and per-layer bytes can be hooked
    per_layer     : forward hooks per leaf module. Slow. Default off.
    """
    be = _backend(device)
    out: dict[str, Any] = {k: UNAVAILABLE("not measured") for k in KEYS}
    out.update(
        {
            "repo": repo,
            "backend": be.name,
            "device": device_label(device),
            "dtype": dtype or UNAVAILABLE("not measured: pass dtype="),
            "mem_method": be.mem_method,
            "ok": False,
            "error": None,
            "forward_calls": 0,
        }
    )

    # ---- what the load already told us -------------------------------------------
    base_alloc: Any = UNAVAILABLE("no baseline= passed; call baseline_after_load() after the load")
    if baseline:
        for k in ("weight_bytes", "weight_mb", "load_s", "baseline_bytes", "baseline_mb"):
            if k in baseline:
                out[k] = baseline[k]
        if baseline.get("dtype") and not dtype:
            out["dtype"] = baseline["dtype"]
        base_alloc = baseline.get("baseline_bytes", base_alloc)
    out["baseline_bytes"] = base_alloc
    out["baseline_mb"] = _mb(base_alloc)
    if not isinstance(out.get("weight_bytes"), (int, float)):
        out["weight_bytes"] = out.get("weight_bytes") or UNAVAILABLE("not measured: no baseline=")
        out["weight_mb"] = out["weight_bytes"]

    # If no baseline was handed in, fall back to the allocator reading right now --
    # correct as long as nothing else ran since the load, and the method string says so.
    fallback_base = False
    if not isinstance(base_alloc, (int, float)):
        be.sync()
        now = be.allocated()
        if isinstance(now, int):
            base_alloc = now
            fallback_base = True

    # ---- peak + latency + hooks ---------------------------------------------------
    sampler: _Sampler | None = None
    if be.native_peak:
        be.reset_peak()
        out["peak_method"] = "torch.cuda.max_memory_allocated (exact high-water mark)"
    else:
        sampler = _Sampler(be)
        iv = sampler._interval
        out["peak_method"] = (
            f"sampled max of {be.mem_method}, yielding spin (sleep(0))"
            if iv == 0
            else f"sampled max of {be.mem_method} every {iv * 1000:.2f} ms"
        )
        if be.name == "cpu":
            # Say the loud part: RSS is a high-water mark the OS does not give back, so a
            # SECOND measurement in the same process sees no new growth and under-reports
            # activation (measured: 110-197 MB on a fresh load, 0.19 MB on a reused one).
            out["peak_method"] += (
                " | CAVEAT: RSS does not shrink, so activation is only valid for the FIRST"
                " measurement after this model's load in this process"
            )

    use_events = be.name == "cuda"
    ev0 = ev1 = None
    if use_events:
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        out["latency_method"] = "torch.cuda.Event elapsed_time"
    else:
        out["latency_method"] = f"perf_counter around fn() with {be.name} synchronize"

    result: Any = None
    pl = _PerLayer(model, per_layer)
    th = _TokenHook(model)
    be.sync()
    try:
        with pl, th:
            if sampler:
                sampler.start()
            if use_events:
                ev0.record()  # type: ignore[union-attr]
            t0 = time.perf_counter()
            result = fn()
            if use_events:
                ev1.record()  # type: ignore[union-attr]
            be.sync()
            t1 = time.perf_counter()
            if sampler:
                peak_sampled, n_samples = sampler.stop()
        out["ok"] = True
    except Exception as exc:
        if sampler:
            try:
                sampler.stop()
            except Exception:
                pass
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["ok"] = False
        return out

    # ---- latency -------------------------------------------------------------------
    if use_events:
        try:
            out["latency_ms"] = round(float(ev0.elapsed_time(ev1)), 3)  # type: ignore[union-attr]
        except Exception as exc:
            out["latency_ms"] = round((t1 - t0) * 1000.0, 3)
            out["latency_method"] = f"perf_counter (cuda events failed: {exc})"
    else:
        out["latency_ms"] = round((t1 - t0) * 1000.0, 3)

    # ---- memory --------------------------------------------------------------------
    if be.native_peak:
        peak = be.peak()
    else:
        peak = peak_sampled
        out["peak_method"] += f" ({n_samples} samples)"
    out["peak_bytes"] = peak
    out["peak_mb"] = _mb(peak)
    out["driver_bytes"] = be.driver()
    out["driver_mb"] = _mb(out["driver_bytes"])

    if isinstance(peak, (int, float)) and isinstance(base_alloc, (int, float)):
        act = int(peak) - int(base_alloc)
        out["activation_bytes"] = act
        out["activation_mb"] = _mb(act)
        if fallback_base:
            out["baseline_bytes"] = int(base_alloc)
            out["baseline_mb"] = _mb(base_alloc)
            out["peak_method"] += " | baseline read at measure() time, not from baseline_after_load()"
    else:
        why = "peak unavailable" if not isinstance(peak, (int, float)) else "post-load baseline unavailable"
        out["activation_bytes"] = UNAVAILABLE(why)
        out["activation_mb"] = out["activation_bytes"]

    # ---- tokens ---------------------------------------------------------------------
    out["forward_calls"] = th.calls
    out["compute_positions"] = th.positions or UNAVAILABLE(
        str(th.why or "no model= hook, so the padded position count was not observed")
    )
    if isinstance(tokens, int) and tokens > 0:
        out["input_tokens"] = tokens
        out["token_method"] = "tokens= passed in by the caller"
    elif th.tokens > 0:
        out["input_tokens"] = th.tokens
        out["token_method"] = (
            f"forward hook over {th.calls} forward call(s): attention_mask sum = "
            f"{th.tokens} real tokens; {th.positions} padded positions actually computed"
        )
    else:
        tc = count_tokens(tokenizer, text) if (tokenizer is not None and text is not None) else None
        if isinstance(tc, int) and tc > 0:
            out["input_tokens"] = tc
            out["token_method"] = "tokenizer(text) -- real tokenisation of the prompt text"
        else:
            why = th.why or (tc if isinstance(tc, str) else "no tokens=, no model= hook, no tokenizer+text")
            out["input_tokens"] = UNAVAILABLE(str(why).replace("UNAVAILABLE: ", ""))
            out["token_method"] = out["input_tokens"]

    # ---- per layer ------------------------------------------------------------------
    rows, total = pl.report()
    out["per_layer"] = rows
    out["per_layer_total_bytes"] = total

    # ---- estimates (and they say so) ------------------------------------------------
    out["active_params"] = active_params if active_params is not None else UNAVAILABLE(
        "not measured: pass active_params= (MoE: ACTIVE, not total)"
    )
    basis = out["compute_positions"] if isinstance(out["compute_positions"], int) else out["input_tokens"]
    if isinstance(active_params, (int, float)) and isinstance(basis, int):
        flops = 2.0 * float(active_params) * float(basis)
        out["est_flops"] = flops
        out["est_tflops"] = round(flops / 1e12, 4)
        out["est_flops_basis"] = (
            "2 x active_params x PADDED compute positions (padding costs FLOPs)"
            if isinstance(out["compute_positions"], int)
            else "2 x active_params x input_tokens (padded count unobserved)"
        )
    else:
        why = "needs active_params= and a token count"
        out["est_flops"] = UNAVAILABLE(why)
        out["est_tflops"] = UNAVAILABLE(why)
        out["est_flops_basis"] = UNAVAILABLE(why)

    out["usd_per_hour"] = usd_per_hour if usd_per_hour is not None else UNAVAILABLE(
        "not measured: pass usd_per_hour="
    )
    if isinstance(usd_per_hour, (int, float)) and isinstance(out["latency_ms"], (int, float)):
        out["est_usd"] = out["latency_ms"] / 1000.0 / 3600.0 * float(usd_per_hour)
    else:
        out["est_usd"] = UNAVAILABLE("needs usd_per_hour= and a latency")

    out["result"] = result
    return out
