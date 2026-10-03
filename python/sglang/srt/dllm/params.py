"""Dependency-free dLLM parameter resolution.

Kept out of ``dllm/config.py`` so the request-admission path can read the same
numbers without importing the model stack.
"""

from __future__ import annotations

from typing import Optional, Tuple

DEFAULT_STEPS_PER_BLOCK = 8

# Algorithms whose CUDA-graph paths (whole-block graph, graphed prompt prefill) are
# validated, so they run by default; the env var still overrides either way.
GRAPHED_BY_DEFAULT = frozenset({"DuoBlock"})


class DllmContractError(RuntimeError):
    """A violated dLLM decode-contract invariant.

    Not an AssertionError, so `python -O` cannot strip these guards.
    """


def steps_per_block_bounds(algorithm_config: Optional[dict]) -> Tuple[int, int]:
    """(server default NFE, per-request ceiling) for a dLLM algorithm config;
    shared by the algorithm's loop bound and the request-admission check."""
    c = algorithm_config or {}
    default = int(c.get("steps_per_block", DEFAULT_STEPS_PER_BLOCK))
    ceiling = int(c.get("max_steps_per_block", max(default, 64)))
    return default, ceiling


def steps_per_block_bounds_from_server_args(server_args) -> Optional[Tuple[int, int]]:
    """Bounds without constructing a ModelConfig (runs in the tokenizer process)."""
    if getattr(server_args, "dllm_algorithm", None) is None:
        return None
    algorithm_config = {}
    path = getattr(server_args, "dllm_algorithm_config", None)
    if path is not None:
        import yaml

        with open(path, "r") as f:
            algorithm_config = yaml.safe_load(f) or {}
    return steps_per_block_bounds(algorithm_config)


def dllm_page_size_refusal(
    algorithm_config: Optional[dict],
    configured_page_size: int,
    allocator_page_size: int,
) -> Optional[str]:
    """The reason this dLLM config cannot serve at this page size, or None.

    Uses max(configured, allocator): under attention DCP the allocator's page
    size is the configured one times dcp_size.
    """
    c = algorithm_config or {}
    wants = [k for k in ("commit_fusion", "ar_verify", "ar_stop") if c.get(k, False)]
    if not wants:
        return None
    effective = max(int(configured_page_size), int(allocator_page_size))
    if effective == 1:
        return None
    why = {
        "commit_fusion": (
            "the fused window reuses the carried block's KV in place, and "
            "partial reuse is only expressible when one page is one token"
        ),
        "ar_stop": (
            "a block cut short at the head's stop decision gives back a "
            "token-aligned tail, which paged allocation cannot express"
        ),
        "ar_verify": (
            "a partially accepted block gives back a token-aligned tail, "
            "which paged allocation cannot express without either freeing "
            "pages that still hold committed tokens or leaking the remainder"
        ),
    }
    return (
        f"{' and '.join(wants)} "
        f"{'require' if len(wants) > 1 else 'requires'} an effective "
        f"page_size of 1, got "
        f"{effective} (configured {configured_page_size}, allocator "
        f"{allocator_page_size}). "
        + "; ".join(why[k] for k in wants)
        + ". Run --page-size 1 without attention DCP, or disable "
        + " / ".join(wants)
        + "."
    )


def request_sampling_overrides(sampling_params) -> Tuple[Optional[float], ...]:
    """(temperature, top_p, top_k) a request set away from SGLang's defaults.

    None where the request left the default, since SamplingParams cannot tell an
    explicit default from an unset one. A greedy request (top_k == 1, which is
    what temperature 0 becomes) overrides nothing: greedy ignores all three.
    """
    from sglang.srt.sampling.sampling_params import TOP_K_ALL

    sp = sampling_params
    if sp.top_k == 1:
        return (None, None, None)
    return (
        sp.temperature if sp.temperature != 1.0 else None,
        sp.top_p if sp.top_p < 1.0 else None,
        sp.top_k if sp.top_k < TOP_K_ALL else None,
    )


def dllm_graph_flag(flag, algorithm: Optional[str]) -> bool:
    """`flag` (an EnvBool) where the user set it, else whether `algorithm` graphs
    by default."""
    return flag.get() if flag.is_set() else algorithm in GRAPHED_BY_DEFAULT
