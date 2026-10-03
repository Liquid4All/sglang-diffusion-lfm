"""Gated short conv in one launch: B*x -> depthwise causal conv (with state) -> C*y.

Rounds where the five-kernel path rounds; not guaranteed bit-identical to the CUDA
kernel (FMA contraction). conv_states[slot, :, 0] is the older tap, [.., 1] the newer.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    _HAS_TRITON = False

BLOCK_N = 128
# Tokens per program; each recomputes its two left neighbours (bitwise equal to serial).
# 1 measured fastest on H100, D=1024: 1.76 us vs 7.76 us serial.
BLOCK_T = 1


if _HAS_TRITON:

    @triton.jit
    def _gated_sconv_kernel(
        proj_ptr,
        w_ptr,
        out_ptr,
        state_ptr,
        cache_idx_ptr,
        has_init_ptr,
        qsl_ptr,
        D: tl.constexpr,
        stride_proj_tok,
        stride_w_dim,
        stride_state_slot,
        stride_state_dim,
        stride_out_tok,
        BLOCK_N: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        seq = tl.program_id(0)
        feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        t0 = tl.program_id(2).to(tl.int64) * BLOCK_T
        fmask = feats < D
        start = tl.load(qsl_ptr + seq).to(tl.int64)
        end = tl.load(qsl_ptr + seq + 1).to(tl.int64)
        L = end - start
        slot = tl.load(cache_idx_ptr + seq).to(tl.int64)
        # Padded rows (sentinel slot or no tokens) and programs past the row's
        # length (grid is sized by the packed total) do nothing.
        if slot < 0:
            return
        if L <= 0:
            return
        if t0 >= L:
            return
        init = tl.load(has_init_ptr + seq).to(tl.int1)
        act_dtype = proj_ptr.dtype.element_ty
        sbase = state_ptr + slot * stride_state_slot + feats * stride_state_dim
        # state[.., 0] = u(-2), state[.., 1] = u(-1); zeros without a state.
        s0 = tl.load(sbase + 0, mask=fmask & init, other=0.0).to(tl.float32)
        s1 = tl.load(sbase + 1, mask=fmask & init, other=0.0).to(tl.float32)
        wb = w_ptr + feats * stride_w_dim
        w0 = tl.load(wb + 0, mask=fmask, other=0.0).to(tl.float32)
        w1 = tl.load(wb + 1, mask=fmask, other=0.0).to(tl.float32)
        w2 = tl.load(wb + 2, mask=fmask, other=0.0).to(tl.float32)
        # Left context u(t0-2), u(t0-1): from input rows, else state (or zero).
        tm2 = t0 - 2
        tm1 = t0 - 1
        rm2 = proj_ptr + (start + tl.maximum(tm2, 0)) * stride_proj_tok
        rm1 = proj_ptr + (start + tl.maximum(tm1, 0)) * stride_proj_tok
        bm2 = tl.load(rm2 + feats, mask=fmask & (tm2 >= 0), other=0.0)
        xm2 = tl.load(rm2 + 2 * D + feats, mask=fmask & (tm2 >= 0), other=0.0)
        bm1 = tl.load(rm1 + feats, mask=fmask & (tm1 >= 0), other=0.0)
        xm1 = tl.load(rm1 + 2 * D + feats, mask=fmask & (tm1 >= 0), other=0.0)
        um2 = (bm2.to(tl.float32) * xm2.to(tl.float32)).to(act_dtype).to(tl.float32)
        um1 = (bm1.to(tl.float32) * xm1.to(tl.float32)).to(act_dtype).to(tl.float32)
        p2 = tl.where(tm2 >= 0, um2, tl.where(tm2 == -1, s1, s0))
        p1 = tl.where(tm1 >= 0, um1, s1)
        t_end = tl.minimum(t0 + BLOCK_T, L)
        for t in range(t0, t_end):
            row = proj_ptr + (start + t) * stride_proj_tok
            b = tl.load(row + feats, mask=fmask, other=0.0)
            c = tl.load(row + D + feats, mask=fmask, other=0.0)
            x = tl.load(row + 2 * D + feats, mask=fmask, other=0.0)
            u = (b.to(tl.float32) * x.to(tl.float32)).to(act_dtype).to(tl.float32)
            y = w0 * p2 + w1 * p1 + w2 * u
            y = y.to(act_dtype).to(tl.float32)
            o = (c.to(tl.float32) * y).to(act_dtype)
            tl.store(out_ptr + (start + t) * stride_out_tok + feats, o, mask=fmask)
            p2 = p1
            p1 = u

    @triton.jit
    def _write_conv_state_kernel(
        proj_ptr,
        state_ptr,
        cache_idx_ptr,
        has_init_ptr,
        qsl_ptr,
        at_ptr,
        D: tl.constexpr,
        stride_proj_tok,
        stride_state_slot,
        stride_state_dim,
        STATE_AT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Write each row's conv state after the conv, one program per row.

        Writing from inside the conv would race with the token programs reading it.
        """
        seq = tl.program_id(0)
        feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        fmask = feats < D
        slot = tl.load(cache_idx_ptr + seq).to(tl.int64)
        if slot < 0:
            return
        start = tl.load(qsl_ptr + seq).to(tl.int64)
        L = tl.load(qsl_ptr + seq + 1).to(tl.int64) - start
        if L <= 0:
            return
        n = L
        if STATE_AT:
            n = tl.load(at_ptr + seq).to(tl.int64)
            # Out of range leaves the state untouched; the host validates.
            if n < 1:
                return
            if n > L:
                return
        act_dtype = proj_ptr.dtype.element_ty
        sbase = state_ptr + slot * stride_state_slot + feats * stride_state_dim
        r1 = proj_ptr + (start + n - 1) * stride_proj_tok
        b1 = tl.load(r1 + feats, mask=fmask, other=0.0)
        x1 = tl.load(r1 + 2 * D + feats, mask=fmask, other=0.0)
        u1 = (b1.to(tl.float32) * x1.to(tl.float32)).to(act_dtype)
        # n == 1: the older tap is the previous window's newer tap, or zero
        # for a row with no history.
        init = tl.load(has_init_ptr + seq).to(tl.int1)
        # Round through the activation dtype so a cache wider than the
        # activations (fp32 cache, bf16 proj) stores the same tap as the conv.
        prev = (
            tl.load(sbase + 1, mask=fmask & init, other=0.0)
            .to(tl.float32)
            .to(act_dtype)
        )
        r0 = proj_ptr + (start + tl.maximum(n - 2, 0)) * stride_proj_tok
        b0 = tl.load(r0 + feats, mask=fmask, other=0.0)
        x0 = tl.load(r0 + 2 * D + feats, mask=fmask, other=0.0)
        u0 = (b0.to(tl.float32) * x0.to(tl.float32)).to(act_dtype)
        u0 = tl.where(n >= 2, u0, prev)
        tl.store(sbase + 0, u0, mask=fmask)
        tl.store(sbase + 1, u1, mask=fmask)


def gated_sconv(
    proj: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    write_state: bool = True,
    block_t: Optional[int] = None,
    state_at: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """out[T, D] = C * conv(B * x) over packed `proj` [T, 3D] (chunks B | C | x).

    Advances conv_state[cache_indices] in place. The token grid is sized by T,
    so no host sync is needed."""
    T, threeD = proj.shape
    D = threeD // 3
    assert weight.shape == (
        D,
        3,
    ), f"width-3 depthwise conv expected, got {tuple(weight.shape)}"
    assert conv_state.stride(2) == 1 and conv_state.shape[2] == 2
    bs = cache_indices.shape[0]
    if has_initial_state is None:
        has_initial_state = torch.zeros(bs, dtype=torch.bool, device=proj.device)
    out = torch.empty(T, D, dtype=proj.dtype, device=proj.device)
    bt = BLOCK_T if block_t is None else int(block_t)
    if state_at is not None:
        if not write_state:
            raise ValueError(
                "gated_sconv: state_at with write_state=False writes nothing; "
                "the caller asked for a state point on a forward that keeps no "
                "state, which is a contradiction rather than a no-op."
            )
        if state_at.shape != (bs,):
            raise ValueError(
                f"gated_sconv: state_at must be one index per sequence "
                f"({bs},), got {tuple(state_at.shape)}"
            )
        state_at = state_at.to(device=proj.device, dtype=torch.int32)
    grid = (bs, triton.cdiv(D, BLOCK_N), triton.cdiv(max(T, 1), bt))
    _gated_sconv_kernel[grid](
        proj,
        weight,
        out,
        conv_state,
        cache_indices,
        has_initial_state,
        query_start_loc,
        D,
        proj.stride(0),
        weight.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        out.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_T=bt,
    )
    if write_state:
        # One extra small launch on state-writing forwards only (commit,
        # prefill); denoise forwards skip it.
        _write_conv_state_kernel[(bs, triton.cdiv(D, BLOCK_N))](
            proj,
            conv_state,
            cache_indices,
            has_initial_state,
            query_start_loc,
            state_at,
            D,
            proj.stride(0),
            conv_state.stride(0),
            conv_state.stride(1),
            STATE_AT=state_at is not None,
            BLOCK_N=BLOCK_N,
        )
    return out


def enabled() -> bool:
    return bool(envs.SGLANG_DLLM_FUSE_SCONV.get()) and _HAS_TRITON


def _can_fuse(m) -> bool:
    w = getattr(m, "conv_weight", None)
    return (
        w is not None
        and w.dim() == 2
        and w.shape[1] == 3
        and getattr(m, "conv_bias", None) is None
        and w.device.type in ("cuda",)
        and w.dtype in (torch.bfloat16, torch.float16)
    )


# Bumped by install(); keys the cached short-conv list so the hot covers_all()
# re-walks model.modules() only after a reinstall.
_INSTALL_GEN = 0


def _short_convs(model, refresh: bool = False) -> list:
    from sglang.srt.models.lfm2 import Lfm2ShortConv

    cached = model.__dict__.get("_dllm_sconv_modules")
    if not refresh and cached is not None and cached[0] == _INSTALL_GEN:
        return cached[1]
    mods = [m for m in model.modules() if isinstance(m, Lfm2ShortConv)]
    # object.__setattr__: a plain attribute, not a registered submodule list.
    object.__setattr__(model, "_dllm_sconv_modules", (_INSTALL_GEN, mods))
    return mods


def install(model, enabled_flag: bool) -> int:
    """Route every Lfm2ShortConv's extend forwards through the fused kernel.

    Decode keeps causal_conv1d_update. Returns the number patched this call;
    active_modules() reports current state."""
    from sglang.srt.models.lfm2 import Lfm2ShortConv

    global _INSTALL_GEN
    _INSTALL_GEN += 1
    n = 0
    for m in model.modules():
        if not isinstance(m, Lfm2ShortConv):
            continue
        if not enabled_flag:
            orig = getattr(m, "_dllm_sconv_orig", None)
            if orig is not None:
                m.forward = orig
                m._dllm_sconv_fwd = None
            continue
        if (
            getattr(m, "_dllm_sconv_fwd", None) is not None
            and m.forward is m._dllm_sconv_fwd
        ):
            continue
        if not _can_fuse(m):
            continue
        orig = m.forward

        def _fwd(hidden_states, forward_batch, _m=m, _orig=orig):
            if (
                forward_batch.forward_mode.is_idle()
                or forward_batch.forward_mode.is_decode()
            ):
                return _orig(hidden_states, forward_batch)
            from sglang.srt.model_executor.forward_context import get_attn_backend

            meta = get_attn_backend().conv_state_metadata(_m.layer_idx, forward_batch)
            proj, _ = _m.in_proj(hidden_states)
            y = gated_sconv(
                proj,
                _m.conv_weight,
                meta.layer_cache.conv[0],
                meta.cache_indices,
                meta.has_initial_state,
                meta.query_start_loc,
                write_state=forward_batch.dllm_save_kv,
                state_at=forward_batch.dllm_conv_state_at,
            )
            output, _ = _m.out_proj(y)
            return output

        m._dllm_sconv_orig = orig
        m._dllm_sconv_fwd = _fwd
        m.forward = _fwd
        n += 1
    return n


def covers_all(model, refresh: bool = False) -> bool:
    """Whether every Lfm2ShortConv runs the fused kernel.

    If so, denoising forwards leave conv state untouched and the block loop
    needs no snapshot/restore. refresh=True re-walks the module tree."""
    mods = _short_convs(model, refresh=refresh)
    return len(mods) > 0 and active_modules(model) == len(mods)


def active_modules(model) -> int:
    """Lfm2ShortConv modules whose live forward is the fused kernel."""
    return sum(
        1
        for m in _short_convs(model)
        if getattr(m, "_dllm_sconv_fwd", None) is not None
        and m.forward is m._dllm_sconv_fwd
    )
