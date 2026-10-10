"""registry.py -- the LIVE model registry for the decider playground.

OWNED BY LANE APP.

Nothing in this file is a hardcoded model list. The org is read at app start
(`list_decider_models()`); every eval number is pulled from the repo's own files.

Two rules learned by measurement (see COORDINATION.md 01:32Z), both load-bearing:

1. VISION. No `strands_decider_config.json` has a `vision` key, and EVERY base model
   (including the text-only Qwen3.5-2B-Base) carries a `vision_config`, so neither can
   decide it. The authority is `docs/vision.md`:
     "Images work with the Qwen3.5 models only: the Gemma 4 models load the text decoder
      of their base model and drop its vision encoder, so they read text only."
   plus `vision.py` raising ValueError unless `head_type == "pointer"`. So the rule is
   (Qwen3.5 base) AND (pointer head) -- see `supports_vision()`.

2. EVAL PATHS VARY. `eval/jevbench-w4096/summary.json` is NOT a fixed location:
   v19 also has `jevbench-w3072`, and `2B-gemma4` has `jf100` and `typed-decisions`.
   We glob every `eval/**/summary.json` instead of naming one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

ORG = os.environ.get("DECIDER_ORG", "StrandsAgents")
# Only repos that are actually deciders. A prefix filter, not a list of names.
PREFIX = os.environ.get("DECIDER_PREFIX", "strands-decider-")

CONFIG_NAMES = ("strands_decider_config.json", "hobson_config.json")


def _api() -> HfApi:
    return HfApi(token=os.environ.get("HF_TOKEN") or True)


# ---------------------------------------------------------------------------------
# capability
# ---------------------------------------------------------------------------------


def supports_vision(base_model: str, head_type: str | None) -> bool:
    """See rule 1 in the module docstring. Measured over all 9 repos at 01:32Z."""
    base = (base_model or "").lower().replace("-", "").replace("_", "").replace(".", "")
    is_qwen35 = "qwen35" in base
    return bool(is_qwen35 and head_type == "pointer")


@dataclass
class ModelInfo:
    repo: str  # "StrandsAgents/strands-decider-..."
    name: str  # short name
    config_file: str | None
    base_model: str
    head_type: str | None
    num_slots: int | None
    max_length: int | None
    torch_dtype: str | None
    temperature: float | None
    vision: bool
    files: list[str] = field(default_factory=list)
    evals: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def max_options(self) -> int | None:
        """A choice question may not exceed the checkpoint's pointer slots."""
        return self.num_slots


def _dl(repo: str, path: str) -> Any:
    return json.load(open(hf_hub_download(repo, path, token=os.environ.get("HF_TOKEN") or None)))


def describe_model(repo: str, files: list[str] | None = None) -> ModelInfo:
    """Read one repo's real config. Never guesses: a failure is recorded in `.error`."""
    api = _api()
    try:
        files = files if files is not None else list(api.list_repo_files(repo))
    except HfHubHTTPError as e:  # private/gated/offline
        return ModelInfo(repo, repo.split("/")[-1], None, "", None, None, None, None, None,
                         False, [], {}, f"list_repo_files failed: {e}")
    cfg_name = next((c for c in CONFIG_NAMES if c in files), None)
    cfg: dict = {}
    err = None
    if cfg_name:
        try:
            cfg = _dl(repo, cfg_name)
        except Exception as e:  # noqa: BLE001
            err = f"{cfg_name}: {e}"
    else:
        err = f"no decider config (looked for {', '.join(CONFIG_NAMES)})"
    base = cfg.get("base_model", "")
    head = cfg.get("head_type")
    return ModelInfo(
        repo=repo,
        name=repo.split("/")[-1],
        config_file=cfg_name,
        base_model=base,
        head_type=head,
        num_slots=cfg.get("num_slots"),
        max_length=cfg.get("max_length"),
        torch_dtype=cfg.get("torch_dtype"),
        temperature=cfg.get("temperature"),
        vision=supports_vision(base, head),
        files=files,
        error=err,
    )


# PUBLIC-ONLY FILTER (owner decision 2026-10-10 01:4xZ, edited by LANE DESIGN -- the one logic
# edit that lane is allowed in this file). The Space carries an HF_TOKEN so private repos WOULD
# resolve; this filter is what keeps them off the page. `private` comes from the Hub, never from
# a name list. DECIDER_PUBLIC_ONLY=0 shows everything the token can see (local debugging only).
PUBLIC_ONLY = os.environ.get("DECIDER_PUBLIC_ONLY", "1") != "0"
HIDDEN_PRIVATE: list[str] = []  # repos the filter removed in the last list_decider_models()


def _is_private(api: HfApi, m: Any) -> bool:
    """`list_models` already carries `.private`; fall back to model_info() when it is None."""
    priv = getattr(m, "private", None)
    if priv is None:
        priv = bool(getattr(api.model_info(m.id), "private", False))
    return bool(priv)


def list_decider_models(org: str = ORG) -> list[ModelInfo]:
    """LIVE read of the org. The app calls this at start -- never a literal list.

    With PUBLIC_ONLY (default) a repo whose Hub record says private=True is dropped and its
    id is kept in HIDDEN_PRIVATE so the page footer can assert the public count.
    """
    api = _api()
    HIDDEN_PRIVATE.clear()
    repos: list[str] = []
    for m in api.list_models(author=org):
        if not m.id.split("/")[-1].startswith(PREFIX):
            continue
        if PUBLIC_ONLY and _is_private(api, m):
            HIDDEN_PRIVATE.append(m.id)
            continue
        repos.append(m.id)
    out = [describe_model(r) for r in sorted(repos)]
    return out


# ---------------------------------------------------------------------------------
# published evals -- pulled live, never typed
# ---------------------------------------------------------------------------------


def _ece_scalar(v: Any) -> Any:
    """`summary["ece"]` is a dict {ece, n, bins}; older files may hold a float."""
    if isinstance(v, dict):
        return v.get("ece")
    return v


def fetch_evals(info: ModelInfo) -> dict[str, Any]:
    """Every `eval/**/summary.json` in the repo, plus a flattened headline table.

    Returns {"raw": {path: json}, "headline": [rows], "arms": [...], "error": ...}
    """
    paths = [f for f in info.files if f.startswith("eval/") and f.endswith("summary.json")]
    raw: dict[str, Any] = {}
    errors: list[str] = []
    for p in sorted(paths):
        try:
            raw[p] = _dl(info.repo, p)
        except (EntryNotFoundError, HfHubHTTPError, ValueError) as e:
            errors.append(f"{p}: {e}")

    headline: list[dict[str, Any]] = []

    # eval/summary.json carries a ready-made jevbench list of arms.
    top = raw.get("eval/summary.json") or {}
    for arm in top.get("jevbench") or []:
        if isinstance(arm, dict):
            headline.append(
                {
                    "suite": f"jevbench {arm.get('arm') or arm.get('window') or ''}".strip(),
                    "source": "eval/summary.json",
                    "accuracy": arm.get("accuracy"),
                    "n_correct": arm.get("n_correct"),
                    "n": arm.get("n"),
                    "brier": arm.get("brier"),
                    "ece": _ece_scalar(arm.get("ece")),
                }
            )

    # Every other summary.json (jevbench-wNNNN, jf100, typed-decisions, ...).
    seen = {h["suite"] for h in headline}
    for p, s in raw.items():
        if p == "eval/summary.json" or not isinstance(s, dict):
            continue
        suite = p[len("eval/"):-len("/summary.json")]
        if "accuracy" not in s:
            continue
        row = {
            "suite": suite,
            "source": p,
            "accuracy": s.get("accuracy"),
            "n_correct": s.get("n_correct"),
            "n": s.get("n_planned") or s.get("calibration_n"),
            "brier": s.get("brier_mean"),
            "ece": _ece_scalar(s.get("ece")),
        }
        lat = s.get("latency")
        if isinstance(lat, dict):
            row["p50_s"] = lat.get("p50_s")
            row["p95_s"] = lat.get("p95_s")
        row["macro_accuracy"] = s.get("macro_accuracy")
        row["coverage"] = s.get("coverage")
        # Prefer the richer per-suite file over the eval/summary.json stub.
        headline = [h for h in headline if not (suite.startswith("jevbench") and h["suite"].replace(" ", "-") in (suite, f"jevbench-{suite}"))]
        headline.append(row)
        seen.add(suite)

    # Internal (recipe-held-out) task accuracies, if present.
    internal = top.get("internal") if isinstance(top.get("internal"), dict) else {}

    return {
        "raw": raw,
        "headline": sorted(headline, key=lambda r: str(r["suite"])),
        "internal": internal,
        "name": top.get("name"),
        "role": top.get("role"),
        "run_id": top.get("run_id"),
        "paths": sorted(paths),
        "error": "; ".join(errors) if errors else None,
    }


def load_registry(org: str = ORG, with_evals: bool = True) -> list[ModelInfo]:
    models = list_decider_models(org)
    if with_evals:
        for m in models:
            try:
                m.evals = fetch_evals(m)
            except Exception as e:  # noqa: BLE001  -- a card must still render
                m.evals = {"error": str(e), "headline": [], "raw": {}, "paths": []}
    return models


if __name__ == "__main__":  # python registry.py -> a quick live check
    ms = load_registry()
    print(f"{len(ms)} decider models in {ORG}")
    for m in ms:
        hl = ", ".join(
            f"{h['suite']} acc={h['accuracy']:.4f}" for h in m.evals.get("headline", [])
            if isinstance(h.get("accuracy"), (int, float))
        )
        print(f"  {m.name:<40} base={m.base_model:<26} vision={m.vision!s:<5} slots={m.num_slots}")
        print(f"      evals: {hl or m.evals.get('error')}")


# ---------------------------------------------------------------------------------
# active parameters -- the FLOPs input (RESEARCH/DESIGN.md section 2)
# ---------------------------------------------------------------------------------


def base_param_total(base_model: str, token: str | None = None) -> int | None:
    """Total parameters of the base, live from the Hub's safetensors metadata."""
    from huggingface_hub import HfApi

    st = getattr(HfApi(token=token or os.environ.get("HF_TOKEN")).model_info(base_model),
                 "safetensors", None)
    if st is None:
        return None
    tot = getattr(st, "total", None)
    if isinstance(tot, int) and tot > 0:
        return tot
    params = dict(getattr(st, "parameters", {}) or {})
    return sum(params.values()) or None


def active_params(base_model: str, token: str | None = None) -> dict:
    """ACTIVE (not total) parameters -- what latency and FLOPs scale with.

    For a dense model active == total. For the MoE base `gemma-4-26B-A4B-it` only top_k of
    num_experts run per token, so the two differ by ~6x and using TOTAL would overstate the
    FLOPs of the biggest model by that factor. The arithmetic is DESIGN.md section 2's,
    recomputed here from the base `config.json` rather than copied:

        per_expert    = 3 * hidden_size * moe_intermediate_size
        all_experts   = num_experts * per_expert * num_layers
        active_expert = top_k       * per_expert * num_layers
        active_total  = (total - all_experts) + active_expert
    """
    out: dict[str, Any] = {"base_model": base_model, "total": None, "active": None,
                           "moe": False, "method": None, "error": None}
    try:
        total = base_param_total(base_model, token)
        out["total"] = total
        cfg = json.load(open(hf_hub_download(base_model, "config.json",
                                             token=token or os.environ.get("HF_TOKEN") or None)))
        tc = cfg.get("text_config") or cfg
        H = tc.get("hidden_size")
        L = tc.get("num_hidden_layers")
        E = tc.get("num_experts") or tc.get("num_local_experts")
        K = tc.get("top_k_experts") or tc.get("num_experts_per_tok")
        MI = tc.get("moe_intermediate_size")
        if total and all(isinstance(x, int) and x > 0 for x in (H, L, E, K, MI)):
            per = 3 * H * MI
            all_e = E * per * L
            act_e = K * per * L
            out.update(
                moe=True, active=(total - all_e) + act_e,
                method=(f"MoE: total {total:,} - all_experts {all_e:,} + active_experts "
                        f"{act_e:,} (H={H}, L={L}, E={E}, K={K}, MI={MI})"),
            )
        else:
            out.update(active=total, method="dense: active == total (safetensors.total)")
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out
