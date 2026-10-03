"""Fused RMSNorm (with optional residual add) in one launch, vs ~11 for forward_native.

Not bit-exact: within 1 ULP of forward_native for bf16/fp16, 4-5 ULP for fp32,
so it is restricted to bf16/fp16.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.dllm.kernels._dispatch import forced_backend
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        w_ptr,
        res_ptr,
        out_ptr,
        res_out_ptr,
        N,
        eps,
        HAS_RESIDUAL: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        p = row * N + offs

        x = tl.load(x_ptr + p, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            r = tl.load(res_ptr + p, mask=mask, other=0.0).to(tl.float32)
            x = x + r
            tl.store(res_out_ptr + p, x.to(res_out_ptr.dtype.element_ty), mask=mask)

        # variance = mean(x^2) over the row, in fp32 (matches forward_native)
        var = tl.sum(x * x, axis=0) / N
        x = x * tl.math.rsqrt(var + eps)

        if HAS_WEIGHT:
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            x = x * w
        tl.store(out_ptr + p, x.to(out_ptr.dtype.element_ty), mask=mask)


# Storage dtypes the kernel is restricted to (within 1 ULP of forward_native,
# not bit-exact despite the name); fp32 falls back.
EXACT_DTYPES = (torch.bfloat16, torch.float16)


# Installed is not executed: the first call of each path writes
# "<SGLANG_DLLM_FUSED_MARKER>.rmsnorm"; counts are not end-of-run totals.
FUSED_CALLS = 0
NATIVE_CALLS = 0


def _write_sidecar() -> None:
    mk = envs.SGLANG_DLLM_FUSED_MARKER.get()
    if not mk:
        return
    try:
        with open(str(mk) + ".rmsnorm", "w") as f:
            f.write(
                f"rmsnorm fused_calls={FUSED_CALLS} " f"native_calls={NATIVE_CALLS}\n"
            )
    except OSError:  # pragma: no cover
        pass


def _note_call(fused: bool) -> None:
    global FUSED_CALLS, NATIVE_CALLS
    if fused:
        FUSED_CALLS += 1
        first = FUSED_CALLS == 1
    else:
        NATIVE_CALLS += 1
        first = NATIVE_CALLS == 1
    if first:
        _write_sidecar()


def _ref_rmsnorm(x, weight, eps: float, residual=None):
    """RMSNorm.forward_native's arithmetic, op for op, with no module needed."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual_out = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    if weight is not None:
        x = x * weight
    out = x.to(orig_dtype)
    return out if residual is None else (out, residual_out)


def fused_rmsnorm(x: torch.Tensor, weight, eps: float, residual=None):
    """forward_native-equivalent RMSNorm in one launch.

    Returns `out` or `(out, residual_out)` to match RMSNorm's contract.
    """
    if not _HAS_TRITON or x.device.type == "cpu":
        # Never launch Triton on CPU tensors.
        return _ref_rmsnorm(x, weight, eps, residual)
    orig_shape = x.shape
    N = orig_shape[-1]
    x2 = x.contiguous().view(-1, N)
    M = x2.shape[0]
    if M == 0:  # idle rank: empty forward, see adaln.py
        out = torch.empty_like(x2).view(orig_shape)
        return out if residual is None else (out, torch.empty_like(x2).view(orig_shape))
    out = torch.empty_like(x2)
    res_in = res_out = None
    if residual is not None:
        res_in = residual.contiguous().view(-1, N)
        res_out = torch.empty_like(res_in)
    w = weight.contiguous() if weight is not None else None

    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 1024 else 8
    _rmsnorm_kernel[(M,)](
        x2,
        w if w is not None else x2,
        res_in if res_in is not None else x2,
        out,
        res_out if res_out is not None else out,
        N,
        eps,
        HAS_RESIDUAL=residual is not None,
        HAS_WEIGHT=w is not None,
        BLOCK=BLOCK,
        num_warps=num_warps,
        # match torch's op-at-a-time rounding (no mul+add FMA contraction)
        enable_fp_fusion=False,
    )
    out = out.view(orig_shape)
    if residual is None:
        return out
    return out, res_out.view(orig_shape)


def triton_is_active(mod) -> bool:
    """Whether our Triton kernel is what `mod` would dispatch right now.

    enter_torch_compile can reassign `_forward_method`; a forced backend bypasses it.
    """
    fwd = getattr(mod, "_dllm_triton_fwd", None)
    if fwd is None or getattr(mod, "_forward_method", None) is not fwd:
        return False
    return forced_backend() is None


def can_fuse(mod) -> bool:
    """Only the plain configuration is replicated; anything else falls back."""
    if not _HAS_TRITON:
        return False
    w = getattr(mod, "weight", None)
    if w is not None and w.dtype not in EXACT_DTYPES:
        return False
    # CPU-resident modules are never installed (also checked per call).
    if w is not None and w.device.type == "cpu":
        return False
    if getattr(mod, "variance_size_override", None) is not None:
        return False
    if getattr(mod, "cast_x_before_out_mul", False):
        return False
    if getattr(mod, "fp32_residual", False):
        return False
    if getattr(mod, "override_orig_dtype", None) is not None:
        return False
    if getattr(mod, "x_pad_to_multiple", 0):
        return False
    n = int(getattr(mod, "hidden_size", 0) or 0)
    # one row per program: the row must fit a single Triton block
    return 0 < n <= 8192


def install(
    model, enabled: bool, *, required_by: str = "", consumer_modules=None
) -> int:
    """Swap eligible RMSNorms under `model` onto the fused path.

    For a `required_by` grant, `consumer_modules` bounds it (see _dispatch.should_install).
    """
    if not enabled:
        return 0
    from sglang.srt.dllm.kernels._dispatch import should_install
    from sglang.srt.layers.layernorm import RMSNorm

    # Do not displace a real platform kernel (e.g. flashinfer on CUDA); decided
    # once from the owning module's provider flags (see _dispatch.py).
    install_here, consumers_only, why = should_install(
        None, "sglang.srt.layers.layernorm", required_by=required_by
    )
    if not install_here:
        model._dllm_norm_forced = False
        logger.warning(
            "[dllm-fused] rmsnorm: not installing (%s); "
            "SGLANG_DLLM_FORCE_FUSED=1 to override",
            why,
        )
        return 0
    # Every decline must happen before the "installing" log.
    scope = None
    if consumers_only:
        if consumer_modules is None:
            # An unbounded grant would displace the platform kernel tree-wide.
            logger.warning(
                "[dllm-fused] rmsnorm: %s licensed the install but named no "
                "consumer modules; declining rather than installing tree-wide",
                required_by,
            )
            return 0
        scope = {id(m) for m in consumer_modules}
        if not scope:
            logger.warning(
                "[dllm-fused] rmsnorm: %s licensed the install but consumes no "
                "module in this model; leaving the platform kernel in place",
                required_by,
            )
            return 0
    # Record whether the install was forced; the module count cannot tell.
    model._dllm_norm_forced = why == "SGLANG_DLLM_FORCE_FUSED=1"
    logger.warning("[dllm-fused] rmsnorm: installing (%s)", why)

    n = 0
    for m in model.modules():
        if scope is not None and id(m) not in scope:
            continue
        if isinstance(m, RMSNorm) and can_fuse(m):

            def _fwd(x, residual=None, _m=m, **kw):
                # anything this kernel does not replicate -> stock path
                if kw.get("post_residual_addition") is not None or kw.get(
                    "quant_linear"
                ):
                    _note_call(False)
                    return RMSNorm.forward_native(_m, x, residual, **kw)
                if (
                    x.device.type == "cpu"
                    or x.dtype not in EXACT_DTYPES
                    or (residual is not None and residual.dtype not in EXACT_DTYPES)
                ):
                    _note_call(False)
                    return RMSNorm.forward_native(_m, x, residual, **kw)
                _note_call(True)
                w = _m.weight if _m.has_weight else None
                return fused_rmsnorm(x, w, _m.variance_epsilon, residual)

            m._forward_method = _fwd
            # Keep the function, not a flag: _forward_method may be reassigned
            # later, and triton_is_active compares identity.
            m._dllm_triton_fwd = _fwd
            n += 1
    return n
