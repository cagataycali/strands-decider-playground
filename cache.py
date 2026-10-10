"""cache.py -- the base-keyed LRU model cache from RESEARCH/DESIGN.md section 4.

OWNED BY LANE APP. Implements the signatures DESIGN.md 4d publishes, with two honest
deviations that are REPORTED, not hidden (program LAW 4 -- never publish a number or a
behaviour you cannot reproduce):

  * `base_hit` / adapter hot-swap. DESIGN 4a wants one resident torso with LoRA adapters
    hot-swapped on it. `strands_decider.infer.load_engine(repo)` builds the torso, the LoRA
    and the pointer head together and exposes no "attach a different adapter to this torso"
    entry point, so a true shared-torso swap would mean reaching into library internals.
    This cache therefore caches ENGINES per (repo, device, mode) and does its ADMISSION and
    EVICTION accounting per BASE, which is what the 48 GB budget actually needs. When two
    repos share a base, `base_hit` is reported **False** and `shared_base_note` says so.
    Nothing claims a saving that did not happen.

  * `max_resident_bytes` / reserve. DESIGN 4c flags its 3 GiB reserve as this program's one
    unmeasured number. On CUDA we replace it with `torch.cuda.mem_get_info()` at construction
    (DESIGN 4c explicitly asks DEPLOY to do this); the fallback is recorded in `budget_source`.

The LOAD/LATENCY RULE from DESIGN 4d is enforced by construction: `LoadReport.load_seconds`
is wall time inside `get()` only, and no method here ever returns a number that adds it to a
forward latency.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

# One process-wide lock: DESIGN 4e. Concurrent requests crash the Metal encoder, and on CUDA
# it is what makes load_seconds / peak memory / latency attributable to ONE request.
INFER_LOCK = threading.Lock()

_GIB = 2 ** 30


class ModelTooLargeError(RuntimeError):
    def __init__(self, repo: str, needed: int, budget: int):
        self.repo, self.needed, self.budget = repo, needed, budget
        super().__init__(
            f"{repo} needs {needed / _GIB:.3f} GiB of base weights but the budget is "
            f"{budget / _GIB:.3f} GiB -- it cannot be resident on this device at all. "
            "DESIGN 3c: 26B-A4B does not fit 48 GB under either VRAM convention."
        )


@dataclass(frozen=True)
class LoadReport:
    repo: str
    base_model: str
    was_cached: bool
    base_hit: bool
    adapter_hit: bool
    load_seconds: float
    base_load_seconds: float
    adapter_load_seconds: float
    activate_seconds: float
    evicted: tuple = ()
    resident_bytes_after: int = 0
    device: str = "cuda"
    # --- additions, so the deviations above are visible in the data, not just the docstring
    shared_base_note: str | None = None
    base_bytes: int = 0
    exclusive: bool = False
    budget_bytes: int = 0


def _dtype_width(name: str) -> int:
    n = name.upper()
    if "BF16" in n or "F16" in n or "FP16" in n or "I16" in n:
        return 2
    if "F32" in n or "FP32" in n or "I32" in n:
        return 4
    if "F64" in n or "I64" in n:
        return 8
    return 1


class ModelCache:
    """Base-keyed admission/eviction around per-repo engines. See module docstring."""

    def __init__(
        self,
        max_resident_bytes: int | None = None,
        device: str = "cuda",
        reserve_bytes: int = 3 * _GIB,
        dtype: str = "bfloat16",
        adapter_cache_bytes: int = 4 * _GIB,
        token: str | None = None,
        vram_bytes: int | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.reserve_bytes = reserve_bytes
        self.adapter_cache_bytes = adapter_cache_bytes
        self.token = token or os.environ.get("HF_TOKEN")
        self.budget_source = "explicit max_resident_bytes"

        if max_resident_bytes is None:
            total = vram_bytes
            if total is None and device == "cuda" and torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()  # DESIGN 4c asks for exactly this
                self.budget_source = f"torch.cuda.mem_get_info total={total}"
            elif total is None:
                # MPS/CPU: unified memory. psutil total is the honest ceiling here.
                try:
                    import psutil

                    total = int(psutil.virtual_memory().total)
                    self.budget_source = f"psutil.virtual_memory().total={total} (unified memory)"
                except Exception:  # noqa: BLE001
                    total = 48 * 10 ** 9
                    self.budget_source = "FALLBACK 48e9 (no cuda, no psutil)"
            max_resident_bytes = int(total * 0.92) - reserve_bytes
        self.max_resident_bytes = int(max_resident_bytes)

        self._engines: dict[str, Any] = {}        # (repo|device|mode) -> entry
        self._base_lru: list[str] = []            # base ids, oldest first
        self._base_keys: dict[str, set[str]] = {} # base -> engine keys resident on it
        self._pinned: set[str] = set()
        self._bytes: dict[str, int] = {}          # base -> measured weight bytes
        self._base_of: dict[str, str] = {}        # repo -> base
        self.last_report: LoadReport | None = None

    # ---- cheap introspection ------------------------------------------------------

    def base_of(self, repo: str) -> str:
        if repo not in self._base_of:
            from registry import describe_model

            self._base_of[repo] = describe_model(repo).base_model
        return self._base_of[repo]

    def bytes_of(self, repo_or_base: str) -> int:
        """Measured bf16 RESIDENT weight bytes of the base, computed live.

        params x dtype width from model_info().safetensors -- not the on-disk file size,
        which also carries the safetensors header (+0.0026%, COORDINATION 01:40Z), and not
        a flat params*2, which would misprice Qwen3.5-2B-Base (mixed F32 + BF16).
        """
        base = repo_or_base if "/" not in repo_or_base or repo_or_base.startswith(
            ("Qwen/", "google/", "meta-llama/")
        ) else self.base_of(repo_or_base)
        if base in self._bytes:
            return self._bytes[base]
        from huggingface_hub import HfApi

        st = getattr(HfApi(token=self.token).model_info(base), "safetensors", None)
        params = dict(getattr(st, "parameters", {}) or {}) if st else {}
        self._bytes[base] = sum(n * _dtype_width(k) for k, n in params.items())
        return self._bytes[base]

    def is_exclusive(self, repo: str) -> bool:
        """DESIGN 4b: a base over half the budget evicts everything else."""
        return self.bytes_of(self.base_of(repo)) > self.max_resident_bytes / 2

    def resident(self) -> list[str]:
        return list(self._base_lru)

    def resident_bytes(self) -> int:
        return sum(self.bytes_of(b) for b in self._base_lru)

    def pin(self, repo: str) -> None:
        self._pinned.add(self.base_of(repo))

    def unpin(self, repo: str) -> None:
        self._pinned.discard(self.base_of(repo))

    def _key(self, repo: str, mode: str) -> str:
        return f"{repo}|{self.device}|{mode}"

    def plan(self, repo: str, mode: str = "text") -> LoadReport:
        """Dry run -- what WOULD happen. No loading, no eviction, load_seconds 0."""
        base = self.base_of(repo)
        bb = self.bytes_of(base)
        key = self._key(repo, mode)
        hit = key in self._engines
        excl = self.is_exclusive(repo)
        evicted: list[str] = []
        if not hit:
            if bb + self.reserve_bytes > self.max_resident_bytes:
                # Report rather than raise: plan() is what the UI calls to WARN.
                evicted = [b for b in self._base_lru if b not in self._pinned]
            else:
                resident = list(self._base_lru)
                total = sum(self.bytes_of(b) for b in resident)
                if excl:
                    evicted = [b for b in resident if b != base and b not in self._pinned]
                else:
                    for b in resident:  # oldest first
                        if base not in resident and total + bb <= self.max_resident_bytes:
                            break
                        if b == base or b in self._pinned:
                            continue
                        evicted.append(b)
                        total -= self.bytes_of(b)
                        if total + bb <= self.max_resident_bytes:
                            break
        others = [k for k in self._base_keys.get(base, set()) if k != key]
        return LoadReport(
            repo=repo, base_model=base, was_cached=hit, base_hit=False,
            adapter_hit=hit, load_seconds=0.0, base_load_seconds=0.0,
            adapter_load_seconds=0.0, activate_seconds=0.0, evicted=tuple(evicted),
            resident_bytes_after=self.resident_bytes() + (0 if hit else bb),
            device=self.device, base_bytes=bb, exclusive=excl,
            budget_bytes=self.max_resident_bytes,
            shared_base_note=(
                f"{len(others)} other loaded engine(s) share base {base}; this build does not "
                "hot-swap adapters on a shared torso, so the base is loaded again (see cache.py)"
            ) if others and not hit else None,
        )

    # ---- the real call ------------------------------------------------------------

    def get(self, repo: str, mode: str = "text") -> tuple[Any, float, bool]:
        engine, rep = self.get_detailed(repo, mode)
        return engine, rep.load_seconds, rep.was_cached

    def get_detailed(self, repo: str, mode: str = "text") -> tuple[Any, LoadReport]:
        """Caller MUST hold INFER_LOCK (DESIGN 4e)."""
        base = self.base_of(repo)
        bb = self.bytes_of(base)
        key = self._key(repo, mode)
        t0 = time.perf_counter()

        if key in self._engines:
            self._touch(base)
            rep = LoadReport(
                repo=repo, base_model=base, was_cached=True, base_hit=True, adapter_hit=True,
                load_seconds=0.0, base_load_seconds=0.0, adapter_load_seconds=0.0,
                activate_seconds=0.0, resident_bytes_after=self.resident_bytes(),
                device=self.device, base_bytes=bb, exclusive=self.is_exclusive(repo),
                budget_bytes=self.max_resident_bytes,
            )
            self.last_report = rep
            return self._engines[key]["engine"], rep

        if bb + self.reserve_bytes > self.max_resident_bytes:
            raise ModelTooLargeError(repo, bb + self.reserve_bytes, self.max_resident_bytes)

        evicted = self._admit(base, bb, exclusive=self.is_exclusive(repo))
        note = self.plan(repo, mode).shared_base_note

        import metrics as M
        from registry import describe_model

        info = describe_model(repo)
        with M.baseline_after_load.around(repo, device=self.device, dtype=info.torch_dtype) as b:
            if mode == "vision":
                from strands_decider.infer import EngineConfig
                from strands_decider.vision import load_vision_engine

                engine = load_vision_engine(repo, EngineConfig(device=self.device))
            else:
                from strands_decider.infer import load_engine

                engine = load_engine(repo, device=self.device)
            # metrics._around() reads `box.model` to do its parameter+buffer walk. On CPU it
            # MUST have it: METRICS measured the RSS delta swinging 5204-7509 MB for the same
            # 2B model, so without this `weight_bytes` is UNAVAILABLE rather than wrong.
            b.model = getattr(engine, "model", None)
        load_s = time.perf_counter() - t0
        self._engines[key] = {"engine": engine, "baseline": b.result, "base": base, "repo": repo}
        self._base_keys.setdefault(base, set()).add(key)
        self._touch(base)
        rep = LoadReport(
            repo=repo, base_model=base, was_cached=False, base_hit=False, adapter_hit=False,
            load_seconds=round(load_s, 4),
            base_load_seconds=round(b.result.get("load_s") or load_s, 4),
            adapter_load_seconds=0.0, activate_seconds=0.0, evicted=tuple(evicted),
            resident_bytes_after=self.resident_bytes(), device=self.device,
            base_bytes=bb, exclusive=self.is_exclusive(repo),
            budget_bytes=self.max_resident_bytes, shared_base_note=note,
        )
        self.last_report = rep
        return engine, rep

    def baseline_of(self, repo: str, mode: str = "text") -> dict | None:
        e = self._engines.get(self._key(repo, mode))
        return e["baseline"] if e else None

    # ---- eviction ----------------------------------------------------------------

    def _touch(self, base: str) -> None:
        if base in self._base_lru:
            self._base_lru.remove(base)
        self._base_lru.append(base)

    def _admit(self, base: str, bb: int, *, exclusive: bool) -> list[str]:
        evicted: list[str] = []
        if exclusive:
            for b in list(self._base_lru):
                if b != base and b not in self._pinned:
                    self.evict(b)
                    evicted.append(b)
            return evicted
        # strict LRU, ties by largest bytes first (DESIGN 4b)
        while self._base_lru and self.resident_bytes() + bb > self.max_resident_bytes:
            cand = [b for b in self._base_lru if b not in self._pinned and b != base]
            if not cand:
                break
            oldest = cand[0]
            same = [b for b in cand if self._base_lru.index(b) == self._base_lru.index(oldest)]
            victim = max(same, key=self.bytes_of) if len(same) > 1 else oldest
            self.evict(victim)
            evicted.append(victim)
        return evicted

    def evict(self, base_model: str) -> int:
        freed = 0
        for key in list(self._base_keys.get(base_model, set())):
            self._engines.pop(key, None)
            freed += self.bytes_of(base_model)
        self._base_keys.pop(base_model, None)
        if base_model in self._base_lru:
            self._base_lru.remove(base_model)
        self._gc()
        return freed

    def clear(self) -> int:
        freed = sum(self.bytes_of(b) for b in list(self._base_lru))
        self._engines.clear()
        self._base_keys.clear()
        self._base_lru.clear()
        self._gc()
        return freed

    @staticmethod
    def _gc() -> None:
        import gc

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def describe_budget(self) -> dict:
        return {
            "max_resident_bytes": self.max_resident_bytes,
            "max_resident_gib": round(self.max_resident_bytes / _GIB, 3),
            "reserve_gib": round(self.reserve_bytes / _GIB, 3),
            "budget_source": self.budget_source,
            "resident_bases": self.resident(),
            "resident_gib": round(self.resident_bytes() / _GIB, 3),
            "device": self.device,
        }
