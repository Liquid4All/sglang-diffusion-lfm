"""Fused adaLN-single modulation (_ref_pre / _ref_post): 2 launches per layer, not ~11.

Bit-exact with eager: casts back to the tensor dtype after every op, in torch's order.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton always present on the target
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _adaln_pre_kernel(
        h_ptr,
        base_ptr,
        emb_ptr,
        hmod_ptr,
        gate_ptr,
        n_tok,
        D,
        base_row_stride,
        BLOCK: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs = pid_d * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D

        b = base_ptr + pid_t * base_row_stride
        # base = adaln_base + emb, then chunk(3) -> [scale | shift | gate]
        s_raw = tl.load(b + offs, mask=mask, other=0.0) + tl.load(
            emb_ptr + offs, mask=mask, other=0.0
        )
        sh_raw = tl.load(b + D + offs, mask=mask, other=0.0) + tl.load(
            emb_ptr + D + offs, mask=mask, other=0.0
        )
        g_raw = tl.load(b + 2 * D + offs, mask=mask, other=0.0) + tl.load(
            emb_ptr + 2 * D + offs, mask=mask, other=0.0
        )

        # torch rounds to the storage dtype after the add that forms `base`
        s_raw = s_raw.to(hmod_ptr.dtype.element_ty)
        sh = sh_raw.to(hmod_ptr.dtype.element_ty)
        g_raw = g_raw.to(hmod_ptr.dtype.element_ty)

        # scale = (1.0 + scale), gate = (1.0 + gate) -- each its own rounding
        scale = (s_raw + 1.0).to(hmod_ptr.dtype.element_ty)
        gate = (g_raw + 1.0).to(hmod_ptr.dtype.element_ty)

        h = tl.load(h_ptr + pid_t * D + offs, mask=mask, other=0.0)
        # h_mod = h * scale + shift, rounding after the mul AND after the add
        prod = (h * scale).to(hmod_ptr.dtype.element_ty)
        hmod = (prod + sh).to(hmod_ptr.dtype.element_ty)

        tl.store(hmod_ptr + pid_t * D + offs, hmod, mask=mask)
        tl.store(gate_ptr + pid_t * D + offs, gate, mask=mask)

    @triton.jit
    def _adaln_post_kernel(
        h_ptr,
        gate_ptr,
        y_ptr,
        hmod_ptr,
        out_ptr,
        n_tok,
        D,
        BLOCK: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs = pid_d * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D
        base = pid_t * D + offs

        h = tl.load(h_ptr + base, mask=mask, other=0.0)
        g = tl.load(gate_ptr + base, mask=mask, other=0.0)
        y = tl.load(y_ptr + base, mask=mask, other=0.0)
        hm = tl.load(hmod_ptr + base, mask=mask, other=0.0)

        # h + gate * (y - h_mod), rounding after each of the three ops
        d = (y - hm).to(out_ptr.dtype.element_ty)
        m = (g * d).to(out_ptr.dtype.element_ty)
        o = (h + m).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + base, o, mask=mask)


def _ref_pre(h, adaln_base, emb):
    base = adaln_base + emb
    scale, shift, gate = base.chunk(3, dim=-1)
    scale = (1.0 + scale).to(h.dtype)
    shift = shift.to(h.dtype)
    gate = (1.0 + gate).to(h.dtype)
    return h * scale + shift, gate


def _ref_post(h, gate, y, h_mod):
    return h + gate * (y - h_mod)


def adaln_pre(h: torch.Tensor, adaln_base: torch.Tensor, emb: torch.Tensor):
    """-> (h_mod, gate). Fused equivalent of the pre-layer modulation."""
    if not _HAS_TRITON or h.device.type == "cpu":
        return _ref_pre(h, adaln_base, emb)
    T, D = h.shape
    # Idle DP-attention ranks forward empty batches; a zero grid is rejected.
    if T == 0:
        return torch.empty_like(h), torch.empty_like(h)
    if adaln_base.shape[-1] != 3 * D or emb.numel() != 3 * D:
        return _ref_pre(h, adaln_base, emb)
    # dtype uniformity is what makes the round-after-every-op replication exact
    if not (h.dtype == adaln_base.dtype == emb.dtype):
        return _ref_pre(h, adaln_base, emb)
    h = h.contiguous()
    adaln_base = adaln_base.contiguous()
    emb = emb.contiguous()
    h_mod = torch.empty_like(h)
    gate = torch.empty_like(h)
    BLOCK = 1024 if D >= 1024 else triton.next_power_of_2(D)
    grid = (T, triton.cdiv(D, BLOCK))
    # enable_fp_fusion=False is required for bit-exactness: an FMA would round
    # `h*scale + shift` once, torch rounds after the mul and after the add.
    _adaln_pre_kernel[grid](
        h,
        adaln_base,
        emb,
        h_mod,
        gate,
        T,
        D,
        adaln_base.stride(0),
        BLOCK=BLOCK,
        enable_fp_fusion=False,
    )
    return h_mod, gate


def adaln_post(
    h: torch.Tensor, gate: torch.Tensor, y: torch.Tensor, h_mod: torch.Tensor
) -> torch.Tensor:
    """-> h + gate * (y - h_mod), fused."""
    if not _HAS_TRITON or h.device.type == "cpu":
        return _ref_post(h, gate, y, h_mod)
    T, D = h.shape
    if T == 0:
        return torch.empty_like(h)
    if not (h.dtype == gate.dtype == y.dtype == h_mod.dtype):
        return _ref_post(h, gate, y, h_mod)
    h = h.contiguous()
    gate = gate.contiguous()
    y = y.contiguous()
    h_mod = h_mod.contiguous()
    out = torch.empty_like(h)
    BLOCK = 1024 if D >= 1024 else triton.next_power_of_2(D)
    grid = (T, triton.cdiv(D, BLOCK))
    _adaln_post_kernel[grid](
        h, gate, y, h_mod, out, T, D, BLOCK=BLOCK, enable_fp_fusion=False
    )
    return out
