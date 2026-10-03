"""Fused `F.silu(gate) * up` in one launch; bit-exact for bf16 only (see EXACT_DTYPES)."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _silu_mul_kernel(x_ptr, out_ptr, D, x_row_stride, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        pid = tl.program_id(1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D
        xb = x_ptr + row * x_row_stride
        g = tl.load(xb + offs, mask=mask, other=0.0)
        u = tl.load(xb + D + offs, mask=mask, other=0.0)
        # F.silu computes in fp32 for a bf16/fp16 input and rounds the result
        # to the storage dtype; the multiply then rounds again.
        s = (g.to(tl.float32) * tl.sigmoid(g.to(tl.float32))).to(
            out_ptr.dtype.element_ty
        )
        o = (s * u).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * D + offs, o, mask=mask)


# tl.sigmoid differs from torch's silu approximation, so the kernel is 1 ULP
# off for fp16/fp32 storage; bf16's 8-bit mantissa rounds the difference away.
EXACT_DTYPES = (torch.bfloat16,)


if _HAS_TRITON:

    @triton.jit
    def _silu_mul_pair_kernel(g_ptr, u_ptr, out_ptr, D, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        pid = tl.program_id(1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D
        p = row * D + offs
        g = tl.load(g_ptr + p, mask=mask, other=0.0)
        u = tl.load(u_ptr + p, mask=mask, other=0.0)
        s = (g.to(tl.float32) * tl.sigmoid(g.to(tl.float32))).to(
            out_ptr.dtype.element_ty
        )
        tl.store(out_ptr + p, (s * u).to(out_ptr.dtype.element_ty), mask=mask)


def _ref(x):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def fused_silu_mul(x: torch.Tensor) -> torch.Tensor:
    if not _HAS_TRITON or x.device.type == "cpu" or x.dtype not in EXACT_DTYPES:
        return _ref(x)
    d = x.shape[-1] // 2
    x2 = x.contiguous().view(-1, x.shape[-1])
    M = x2.shape[0]
    if M == 0:  # idle rank: empty forward, see adaln.py
        return torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
    out = torch.empty((M, d), dtype=x.dtype, device=x.device)
    BLOCK = 1024 if d >= 1024 else triton.next_power_of_2(d)
    _silu_mul_kernel[(M, triton.cdiv(d, BLOCK))](
        x2,
        out,
        d,
        x2.stride(0),
        BLOCK=BLOCK,
        enable_fp_fusion=False,
    )
    return out.view(*x.shape[:-1], d)


def fused_silu_mul_pair(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up for separate tensors, without concatenating them."""
    if not _HAS_TRITON or gate.device.type == "cpu" or gate.dtype not in EXACT_DTYPES:
        return F.silu(gate) * up
    d = gate.shape[-1]
    g2 = gate.contiguous().view(-1, d)
    u2 = up.contiguous().view(-1, d)
    M = g2.shape[0]
    if M == 0:
        return torch.empty_like(gate)
    out = torch.empty_like(g2)
    BLOCK = 1024 if d >= 1024 else triton.next_power_of_2(d)
    _silu_mul_pair_kernel[(M, triton.cdiv(d, BLOCK))](
        g2,
        u2,
        out,
        d,
        BLOCK=BLOCK,
        enable_fp_fusion=False,
    )
    return out.view(*gate.shape)


def install(model, enabled: bool, merge_gemm: bool = False) -> int:
    """Replace Lfm2MLP.forward with fused silu*mul, optionally merging w1/w3 into one GEMM.

    merge_gemm is opt-in: the cached w13 goes stale on any w1/w3 write outside load_weights.
    """
    if not enabled or not _HAS_TRITON:
        return 0
    # Unlike RMSNorm there is no platform kernel to displace (the inline
    # silu*mul is 2 launches on CUDA and ROCm alike), so no dispatch check.
    from sglang.srt.models.lfm2 import Lfm2MLP

    n = 0
    seen = skipped = 0
    for m in model.modules():
        if not isinstance(m, Lfm2MLP):
            continue
        seen += 1
        w1 = getattr(m, "w1", None)
        w3 = getattr(m, "w3", None)
        if w1 is None or w3 is None:
            skipped += 1
            continue
        # A quantized linear exposes a temporary bf16 `weight` during load_weights,
        # and the replacement bypasses quant_method.apply; only unquantized qualify.
        qm = getattr(w1, "quant_method", None)
        qm3 = getattr(w3, "quant_method", None)

        def _unquantized(q):
            return q is None or type(q).__name__ == "UnquantizedLinearMethod"

        if not (_unquantized(qm) and _unquantized(qm3)):
            skipped += 1
            continue
        if w1.weight.dtype not in EXACT_DTYPES:
            skipped += 1
            continue

        if not merge_gemm:
            # activation-only fusion: no weight copy, no lifecycle hazard
            def _fwd_nomerge(x, _m=m):
                gate, _ = _m.w1(x)
                up, _ = _m.w3(x)
                return _m.w2(fused_silu_mul_pair(gate, up))[0]

            m.forward = _fwd_nomerge
            m._dllm_silu_fwd = _fwd_nomerge
            m._dllm_silu_merged = False
            n += 1
            continue

        try:
            # Address-stable: allocate once, copy_ on every later refresh, so a
            # reload cannot move the buffer out from under a captured graph.
            existing = getattr(m, "_fused_w13", None)
            wcat = torch.cat([w1.weight.data, w3.weight.data], dim=0).contiguous()
            if existing is not None and existing.shape == wcat.shape:
                existing.copy_(wcat)
                wcat = existing
        except Exception as e:
            logger.warning(f"[dllm-fused] w1/w3 cat failed: {e}")
            skipped += 1
            continue
        # CPU offload only moves state_dict() entries, so a plain tensor
        # attribute would stay on CPU; refuse the merge.
        if wcat.device.type != "cuda":
            logger.warning(
                "[dllm-fused] w1/w3 merge skipped: weights are not on CUDA "
                "(CPU offload); the derived tensor would not be moved with "
                "the module",
            )
            skipped += 1
            continue
        # Captured graphs reference _fused_w13's address, which only moves on a
        # shape change; in-place w1/w3 writes are not detectable here.
        _prev_w13 = getattr(m, "_fused_w13_ptr", None)
        m._fused_w13 = wcat
        m._fused_d = w1.weight.shape[0]
        m._fused_w13_ptr = wcat.data_ptr()
        m._fused_src_ptrs = (w1.weight.data_ptr(), w3.weight.data_ptr())
        if _prev_w13 is not None and _prev_w13 != m._fused_w13_ptr:
            logger.warning(
                "[dllm-fused] w1/w3 merged buffer MOVED (%s -> %s), which "
                "means the merged shape changed. Any CUDA graph captured "
                "before this reload replays the old address and must be "
                "re-captured.",
                _prev_w13,
                m._fused_w13_ptr,
            )

        def _fwd(x, _m=m):
            # The merged GEMM yields the packed [.., 2D] layout _silu_mul_kernel
            # reads; splitting it into slices would add two .contiguous() copies.
            gu = torch.nn.functional.linear(x, _m._fused_w13)
            return _m.w2(fused_silu_mul(gu))[0]

        m.forward = _fwd
        m._dllm_silu_fwd = _fwd
        m._dllm_silu_merged = True
        n += 1
    if seen and not n:
        logger.warning(
            f"[dllm-fused] found {seen} Lfm2MLP but fused 0 " f"({skipped} skipped)",
        )
    return n


def _is_live(m) -> bool:
    """Patched by this installer and not overwritten since."""
    fwd = getattr(m, "_dllm_silu_fwd", None)
    return fwd is not None and getattr(m, "forward", None) is fwd


def active_modules(model) -> int:
    """MLPs actually running this installer's fused forward."""
    return sum(1 for m in model.modules() if _is_live(m))


def total_modules(model) -> int:
    """Every Lfm2MLP in the tree: the denominator for `mlp`."""
    from sglang.srt.models.lfm2 import Lfm2MLP

    return sum(1 for m in model.modules() if isinstance(m, Lfm2MLP))


def merged_modules(model) -> int:
    """MLPs actually running the w1/w3 GEMM merge.

    SGLANG_DLLM_FUSE_MLP_GEMM is a request; the merge can be skipped per module.
    """
    return sum(
        1
        for m in model.modules()
        if getattr(m, "_dllm_silu_merged", False) and _is_live(m)
    )
