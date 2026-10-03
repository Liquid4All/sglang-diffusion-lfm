"""Attention prologue in one launch: per-head q/k RMSNorm, neox RoPE, contiguous q/k/v.

The norm mirrors rmsnorm._rmsnorm_kernel op for op; RoPE mirrors sgl_kernel's CUDA
(fp32 cache) or ROCm (activation-dtype cache) rotation. See static_refusal for eligibility.
"""

import logging
from typing import Optional

import torch

from sglang.srt.dllm.kernels._dispatch import forced_backend as _forced_backend
from sglang.srt.dllm.kernels.rmsnorm import triton_is_active as _triton_is_active
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _rope_pair(lo, hi, cos, sin, VARIANT: tl.constexpr, dt: tl.constexpr):
        # nvcc contracts one product of each term into an FMA; the pattern is
        # a constexpr chosen by the parity test (see ROPE_VARIANT).
        if VARIANT == 0:
            out_lo = lo * cos - hi * sin
            out_hi = lo * sin + hi * cos
        elif VARIANT == 10:
            # ROCm: with an activation-dtype cache, sgl_kernel.rotary_embedding
            # rounds each product to that dtype before the sum/difference.
            out_lo = (lo * cos).to(dt).to(tl.float32) - (hi * sin).to(dt).to(tl.float32)
            out_hi = (lo * sin).to(dt).to(tl.float32) + (hi * cos).to(dt).to(tl.float32)
        elif VARIANT == 1:
            out_lo = tl.fma(lo, cos, -(hi * sin))
            out_hi = tl.fma(lo, sin, hi * cos)
        elif VARIANT == 2:
            out_lo = tl.fma(-hi, sin, lo * cos)
            out_hi = tl.fma(hi, cos, lo * sin)
        elif VARIANT == 3:
            out_lo = tl.fma(lo, cos, -(hi * sin))
            out_hi = tl.fma(hi, cos, lo * sin)
        else:
            out_lo = tl.fma(-hi, sin, lo * cos)
            out_hi = tl.fma(lo, sin, hi * cos)
        return out_lo, out_hi

    @triton.jit
    def _norm_rope_row(
        x_ptr,
        w_ptr,
        cos,
        sin,
        out_ptr,
        eps,
        D: tl.constexpr,
        HALF: tl.constexpr,
        VARIANT: tl.constexpr,
        dt: tl.constexpr,
    ):
        # One row: RMSNorm with _rmsnorm_kernel's reduction tree (same shape
        # and num_warps, bitwise), then RoPE.
        offs_d = tl.arange(0, D)
        offs_r = tl.arange(0, HALF)
        x = tl.load(x_ptr + offs_d).to(tl.float32)
        var = tl.sum(x * x, axis=0) / D
        x = x * tl.math.rsqrt(var + eps)
        w = tl.load(w_ptr + offs_d).to(tl.float32)
        xn = (x * w).to(dt).to(tl.float32)
        x2 = tl.permute(tl.reshape(xn, [2, HALF]), (1, 0))
        lo, hi = tl.split(x2)
        out_lo, out_hi = _rope_pair(lo, hi, cos, sin, VARIANT, dt)
        tl.store(out_ptr + offs_r, out_lo.to(dt))
        tl.store(out_ptr + HALF + offs_r, out_hi.to(dt))

    @triton.jit
    def _qkv_prologue_kernel(
        qkv_ptr,
        pos_ptr,
        wq_ptr,
        wk_ptr,
        cs_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        stride_qkv_tok,
        eps,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        HALF: tl.constexpr,
        VARIANT: tl.constexpr,
    ):
        # No branch yields a value: ROCm Triton rejects pointer-typed scf.if
        # results and pointer selects, and narrows large offsets between allocations.
        t = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1)
        row = qkv_ptr + t * stride_qkv_tok
        dt = q_out_ptr.dtype.element_ty
        pos = tl.load(pos_ptr + t).to(tl.int64)
        offs_r = tl.arange(0, HALF)
        cos = tl.load(cs_ptr + pos * D + offs_r).to(tl.float32)
        sin = tl.load(cs_ptr + pos * D + HALF + offs_r).to(tl.float32)
        _norm_rope_row(
            row + h * D,
            wq_ptr,
            cos,
            sin,
            q_out_ptr + t * (HQ * D) + h * D,
            eps,
            D,
            HALF,
            VARIANT,
            dt,
        )
        if h < HK:
            _norm_rope_row(
                row + HQ * D + h * D,
                wk_ptr,
                cos,
                sin,
                k_out_ptr + t * (HK * D) + h * D,
                eps,
                D,
                HALF,
                VARIANT,
                dt,
            )
            offs_d = tl.arange(0, D)
            v = tl.load(row + (HQ + HK) * D + h * D + offs_d)
            tl.store(v_out_ptr + t * (HK * D) + h * D + offs_d, v)


# nvcc's FMA placement for the CUDA rotation, the only pattern with zero
# mismatches in an H100 sweep; explicit, with contraction disabled.
ROPE_VARIANT = 3


def qkv_prologue(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """(q, k, v), each [T, heads*head_dim] contiguous, from qkv [T, (HQ+2HK)*D]."""
    T = qkv.shape[0]
    HQ, HK, D = int(num_q_heads), int(num_kv_heads), int(head_dim)
    assert qkv.shape[1] == (HQ + 2 * HK) * D, (tuple(qkv.shape), HQ, HK, D)
    assert cos_sin_cache.shape[1] == D
    # fp32 cache -> CUDA arithmetic; activation-dtype cache -> ROCm (mode 10)
    if cos_sin_cache.dtype == torch.float32:
        variant = ROPE_VARIANT
    else:
        assert cos_sin_cache.dtype == qkv.dtype, (cos_sin_cache.dtype, qkv.dtype)
        variant = 10
    assert positions.shape[0] == T
    q = torch.empty(T, HQ * D, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty(T, HK * D, dtype=qkv.dtype, device=qkv.device)
    v = torch.empty(T, HK * D, dtype=qkv.dtype, device=qkv.device)
    if T == 0:
        return q, k, v
    assert wq.dtype == wk.dtype and wq.is_contiguous() and wk.is_contiguous()
    assert HK <= HQ, (HQ, HK)
    _qkv_prologue_kernel[(T, HQ)](
        qkv,
        positions,
        wq,
        wk,
        cos_sin_cache,
        q,
        k,
        v,
        qkv.stride(0),
        eps,
        HQ=HQ,
        HK=HK,
        D=D,
        HALF=D // 2,
        VARIANT=variant,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q, k, v


def enabled() -> bool:
    return bool(envs.SGLANG_DLLM_FUSE_QKV.get()) and _HAS_TRITON


def cache_mode_refusal(cache: torch.Tensor, act_dtype: torch.dtype) -> Optional[str]:
    """Refuse unless the cache dtype selects a verified rotation mode.

    CUDA needs an fp32 cache; ROCm needs a bf16 cache and bf16 activations."""
    if torch.version.hip:
        if act_dtype != torch.bfloat16:
            return "ROCm rotation mirrored for bf16 only"
        if cache.dtype != torch.bfloat16:
            return "ROCm cos/sin cache not bf16 (SGLANG_ROPE_CACHE_FP32?)"
    else:
        if cache.dtype != torch.float32:
            return "CUDA cos/sin cache is not fp32"
    return None


def static_refusal(m) -> Optional[str]:
    """Why this module can never take the fused prologue, ignoring norm dispatch.

    Also gates the FUSED_NORM consumer grant for its q/k norms: a refused
    module keeps its platform norm."""
    rot = getattr(m, "rotary_emb", None)
    qn, kn = getattr(m, "q_layernorm", None), getattr(m, "k_layernorm", None)
    if rot is None or qn is None or kn is None:
        return "no rotary_emb / q_layernorm / k_layernorm"
    D = int(m.head_dim)
    if D & (D - 1) or D > 256 or D < 2:
        return f"head_dim {D} not a power of two <= 256"
    if int(getattr(rot, "rotary_dim", D)) != D:
        return "rotary_dim != head_dim"
    if not getattr(rot, "is_neox_style", False):
        return "not neox-style RoPE"
    # Only the two verified rotation paths are accepted; anything else the
    # layer might dispatch would need another mirror.
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    if type(rot) is not RotaryEmbedding:
        return f"{type(rot).__name__} is not the plain RotaryEmbedding"
    cache = getattr(rot, "cos_sin_cache", None)
    if cache is None or cache.shape[-1] != D or cache.device.type != "cuda":
        return "cos/sin cache missing, wrong width, or not on the GPU"
    if torch.version.hip:
        # ROCm rotates with sgl_kernel.rotary_embedding on a model-dtype cache.
        if not getattr(rot, "use_fallback_kernel", False):
            return "ROCm rotary not on the sgl_kernel fallback path"
    else:
        if getattr(rot, "use_fallback_kernel", True):
            return "RotaryEmbedding uses its torch fallback for this head size"
    why = cache_mode_refusal(cache, qn.weight.dtype)
    if why is not None:
        return why
    if (
        qn.weight.dtype not in (torch.bfloat16, torch.float16)
        or kn.weight.dtype != qn.weight.dtype
    ):
        return "norm weights not bf16/fp16 or mismatched"
    if abs(float(qn.variance_epsilon) - float(kn.variance_epsilon)) > 0:
        return "q/k norm eps differ"
    if (
        getattr(m.attn, "k_scale", None) is not None
        or getattr(m.attn, "v_scale", None) is not None
    ):
        return "k/v scales present"
    # Rotary dispatch is fixed by process-level settings, so it is known at
    # grant time.
    if not _rope_is_cuda_kernel(m.rotary_emb):
        return "RotaryEmbedding does not dispatch forward_cuda (deterministic/native/forced backend)"
    return None


def _rope_is_cuda_kernel(rot) -> bool:
    """Whether RotaryEmbedding would dispatch forward_cuda right now, with no forced backend."""
    method = getattr(rot, "_forward_method", None)
    if method is None:
        resolve = getattr(rot, "_resolve_forward_method", None)
        if resolve is None:
            return False
        method = resolve()
    if getattr(method, "__func__", None) is not type(rot).forward_cuda:
        return False
    return _forced_backend() is None


def _live(m) -> bool:
    """Whether the norms and rotary currently dispatch the kernels this mirrors."""
    return (
        _triton_is_active(m.q_layernorm)
        and _triton_is_active(m.k_layernorm)
        and _rope_is_cuda_kernel(m.rotary_emb)
        and cache_mode_refusal(m.rotary_emb.cos_sin_cache, m.q_layernorm.weight.dtype)
        is None
    )


def _refusal(m) -> Optional[str]:
    why = static_refusal(m)
    if why is not None:
        return why
    if not (_triton_is_active(m.q_layernorm) and _triton_is_active(m.k_layernorm)):
        return "q/k norms are not the Triton kernel (set SGLANG_DLLM_FUSED_NORM=1)"
    return None


def install(model, enabled_flag: bool) -> int:
    """Route every Lfm2Attention's extend forwards through the fused prologue.

    Decode and idle keep the stock path. Returns the number patched this call;
    active_modules() reports current state."""
    from sglang.srt.models.lfm2 import Lfm2Attention

    n = 0
    refused = {}
    for m in model.modules():
        if not isinstance(m, Lfm2Attention):
            continue
        if not enabled_flag:
            orig = getattr(m, "_dllm_qkv_orig", None)
            if orig is not None:
                m.forward = orig
                m._dllm_qkv_fwd = None
            continue
        if (
            getattr(m, "_dllm_qkv_fwd", None) is not None
            and m.forward is m._dllm_qkv_fwd
        ):
            continue
        why = _refusal(m)
        if why is not None:
            refused[why] = refused.get(why, 0) + 1
            continue
        orig = m.forward

        def _fwd(positions, hidden_states, forward_batch, _m=m, _orig=orig):
            if (
                forward_batch.forward_mode.is_idle()
                or forward_batch.forward_mode.is_decode()
            ):
                return _orig(positions, hidden_states, forward_batch)
            # Decided before the QKV GEMM so the projection never runs twice;
            # the fallback still uses the granted Triton norms.
            _pos_ok = (
                (torch.int64,) if torch.version.hip else (torch.int32, torch.int64)
            )
            if (
                not _live(_m)
                or hidden_states.dtype != _m.q_layernorm.weight.dtype
                or positions.dtype not in _pos_ok
            ):
                return _orig(positions, hidden_states, forward_batch)
            qkv, _ = _m.qkv_proj(hidden_states)
            if qkv.dtype != hidden_states.dtype:
                # Dtype-changing (quantized) projection: finish with the stock
                # ops on this qkv rather than recomputing it.
                T = hidden_states.shape[0]
                q_size = _m.num_local_q_heads * _m.head_dim
                kv_size = _m.num_local_kv_heads * _m.head_dim
                q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
                q = _m.q_layernorm(q.reshape(-1, _m.head_dim)).reshape(
                    T, _m.num_local_q_heads, _m.head_dim
                )
                k = _m.k_layernorm(k.reshape(-1, _m.head_dim)).reshape(
                    T, _m.num_local_kv_heads, _m.head_dim
                )
                q, k = _m.rotary_emb(positions, q, k)
                attn_out = _m.attn(
                    q.reshape(T, -1),
                    k.reshape(T, -1),
                    v,
                    forward_batch,
                    save_kv_cache=forward_batch.dllm_save_kv,
                )
                out, _ = _m.out_proj(attn_out)
                return out
            q, k, v = qkv_prologue(
                qkv,
                positions,
                _m.q_layernorm.weight,
                _m.k_layernorm.weight,
                _m.q_layernorm.variance_epsilon,
                _m.rotary_emb.cos_sin_cache,
                _m.num_local_q_heads,
                _m.num_local_kv_heads,
                _m.head_dim,
            )
            attn_out = _m.attn(
                q,
                k,
                v,
                forward_batch,
                save_kv_cache=forward_batch.dllm_save_kv,
            )
            out, _ = _m.out_proj(attn_out)
            return out

        m._dllm_qkv_orig = orig
        m._dllm_qkv_fwd = _fwd
        m.forward = _fwd
        n += 1
    if refused:
        logger.warning("[dllm-fused] qkv prologue refused: %s", refused)
    return n


def active_modules(model) -> int:
    """Lfm2Attention modules whose live forward is the fused prologue."""
    from sglang.srt.models.lfm2 import Lfm2Attention

    return sum(
        1
        for m in model.modules()
        if isinstance(m, Lfm2Attention)
        and getattr(m, "_dllm_qkv_fwd", None) is not None
        and m.forward is m._dllm_qkv_fwd
        # Live, not merely installed: the wrapper falls back on re-dispatch.
        and _live(m)
    )


def total_modules(model) -> int:
    from sglang.srt.models.lfm2 import Lfm2Attention

    return sum(1 for m in model.modules() if isinstance(m, Lfm2Attention))
