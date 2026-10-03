"""Decide whether a fused dLLM kernel should replace the platform one.

Install only where the owning module has no fused provider (it would fall
through to `forward_native`); decided from what it imported, never by `is_hip`.
"""

from __future__ import annotations

from sglang.srt.environ import envs

# Not None, so callers treat an unresolvable registry as forced.
_UNKNOWN_BACKEND = object()


def forced_backend():
    """The process-wide forced fused-op backend, or None; unknown counts as forced.

    Applied per call without touching `_forward_method`, so liveness checks need it.
    """
    try:
        from sglang.kernels.fused_op import get_fused_op_backend

        return get_fused_op_backend()
    except Exception:
        return _UNKNOWN_BACKEND


# Not the resolved method name (forward_hip may delegate to forward_native):
# flags the owning module sets after each provider import try/except.
_PROVIDER_FLAGS = (
    "_use_aiter",  # ROCm: aiter kernels
    "_has_vllm_rms_norm",  # ROCm: vllm _custom_ops
    "_has_aiter_layer_norm",  # ROCm: aiter layer_norm
    "_is_flashinfer_available",  # CUDA: flashinfer
)

# layernorm.py imports sgl_kernel's rmsnorm without a flag, so also check
# whether a provider symbol got bound (covers sgl_kernel, vllm and aiter).
_PROVIDER_SYMBOLS = (
    "rmsnorm",  # sgl_kernel
    "fused_add_rmsnorm",  # sgl_kernel
    "rms_norm",  # vllm _custom_ops / aiter
    "fused_add_rms_norm",  # vllm _custom_ops / aiter
)


def module_has_provider(module_name: str) -> bool:
    """True if the module that defines the op reports a real fused provider."""
    import importlib

    try:
        m = importlib.import_module(module_name)
    except Exception:
        return False
    if any(bool(getattr(m, f, False)) for f in _PROVIDER_FLAGS):
        return True
    return any(callable(getattr(m, sym, None)) for sym in _PROVIDER_SYMBOLS)


def resolved_name(mod) -> str:
    """Name of the forward BaseFusedOp would use, or '' if not resolvable."""
    m = getattr(mod, "_forward_method", None)
    if m is not None:
        return getattr(m, "__name__", "") or ""
    # not yet resolved: replicate the lookup without calling the op
    for attr in ("forward_hip", "forward_cuda"):
        if hasattr(mod, attr):
            # Present is not usable: only a resolved _forward_method is
            # authoritative, so report unknown.
            return ""
    return ""


def should_install(
    mod,
    module_name: str = "sglang.srt.layers.layernorm",
    *,
    required_by: str = "",
) -> tuple[bool, bool, str]:
    """Install our Triton kernel here? Returns (install, consumers_only, reason).

    `required_by` (a consumer fusing into our kernel) grants install even where a
    provider exists, with `consumers_only` scoping it to that consumer's modules.
    """
    if envs.SGLANG_DLLM_FORCE_FUSED.get():
        return True, False, "SGLANG_DLLM_FORCE_FUSED=1"
    # _PROVIDER_SYMBOLS only knows CUDA/ROCm providers; elsewhere (e.g. NPU)
    # we would bypass a working platform kernel.
    if not _is_cuda_like():
        return False, False, "not a CUDA/ROCm device"
    if not module_has_provider(module_name):
        return True, False, f"{module_name} reports no fused provider"
    if required_by:
        return (
            True,
            True,
            (
                f"{module_name} has a fused provider, but {required_by} fuses into "
                "this kernel and cannot consume the platform one; scoped to the "
                "modules it consumes"
            ),
        )
    return False, False, f"{module_name} already has a fused provider"


def _is_cuda_like() -> bool:
    """True only on real CUDA or ROCm, where _PROVIDER_SYMBOLS is meaningful."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        from sglang.srt.utils import is_hip

        return bool(is_hip()) or torch.version.cuda is not None
    except Exception:
        return False
