"""adaln_pre fused with the following operator_norm: emits h_mod, gate, and normed.

Bit-identical to _adaln_pre_kernel + _rmsnorm_kernel composed. D must fit one
BLOCK: the RMS reduction needs the whole row in one program.
"""

import logging

import torch

from sglang.srt.dllm.kernels.rmsnorm import triton_is_active as _triton_is_active
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _flag() -> bool:
    """Read at call time so later toggles apply; envs owns the default."""
    return envs.SGLANG_DLLM_FUSE_ADALN_NORM.get()


try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False

if _HAVE_TRITON:

    @triton.jit
    def _adaln_pre_norm_kernel(
        h_ptr,
        base_ptr,
        emb_ptr,
        w_ptr,
        hmod_ptr,
        gate_ptr,
        normed_ptr,
        base_row_stride,
        D: tl.constexpr,
        eps,
        HAS_WEIGHT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        b = base_ptr + pid_t * base_row_stride
        dt = hmod_ptr.dtype.element_ty

        # --- adaLN pre, rounding where _adaln_pre_kernel rounds ---
        s_raw = tl.load(b + offs, mask=mask, other=0.0) + tl.load(
            emb_ptr + offs, mask=mask, other=0.0
        )
        sh_raw = tl.load(b + D + offs, mask=mask, other=0.0) + tl.load(
            emb_ptr + D + offs, mask=mask, other=0.0
        )
        g_raw = tl.load(b + 2 * D + offs, mask=mask, other=0.0) + tl.load(
            emb_ptr + 2 * D + offs, mask=mask, other=0.0
        )
        s_raw = s_raw.to(dt)
        sh = sh_raw.to(dt)
        g_raw = g_raw.to(dt)
        scale = (s_raw + 1.0).to(dt)
        gate = (g_raw + 1.0).to(dt)

        h = tl.load(h_ptr + pid_t * D + offs, mask=mask, other=0.0)
        prod = (h * scale).to(dt)
        hmod = (prod + sh).to(dt)
        tl.store(hmod_ptr + pid_t * D + offs, hmod, mask=mask)
        tl.store(gate_ptr + pid_t * D + offs, gate, mask=mask)

        # --- RMSNorm on hmod, fp32 accumulation as _rmsnorm_kernel does ---
        x = hmod.to(tl.float32)
        var = tl.sum(x * x, axis=0) / D
        # rsqrt, not 1.0/sqrt, to match _rmsnorm_kernel's single rounding.
        x = x * tl.math.rsqrt(var + eps)
        if HAS_WEIGHT:
            wv = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            x = x * wv
        tl.store(normed_ptr + pid_t * D + offs, x.to(dt), mask=mask)

    @triton.jit
    def _adaln_post_pre_norm_kernel(
        hprev_ptr,
        gprev_ptr,
        y_ptr,
        y2_ptr,
        hmprev_ptr,
        base_ptr,
        emb_ptr,
        w_ptr,
        hout_ptr,
        hmod_ptr,
        gate_ptr,
        normed_ptr,
        base_row_stride,
        D: tl.constexpr,
        eps,
        HAS_WEIGHT: tl.constexpr,
        HAS_Y2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """adaln_post(layer i) + adaln_pre(layer i+1) + operator_norm, fused.

        HAS_Y2: y + y2 in fp32 with one rounding, matching torch's `a + b`.
        """
        pid_t = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        dt = hout_ptr.dtype.element_ty
        p = pid_t * D + offs

        # --- adaln_post: h + gate*(y - h_mod), rounding after each op ---
        hp = tl.load(hprev_ptr + p, mask=mask, other=0.0)
        gp = tl.load(gprev_ptr + p, mask=mask, other=0.0)
        yv = tl.load(y_ptr + p, mask=mask, other=0.0)
        if HAS_Y2:
            y2 = tl.load(y2_ptr + p, mask=mask, other=0.0)
            yv = (yv.to(tl.float32) + y2.to(tl.float32)).to(dt)
        hm = tl.load(hmprev_ptr + p, mask=mask, other=0.0)
        d = (yv - hm).to(dt)
        m = (gp * d).to(dt)
        h = (hp + m).to(dt)
        tl.store(hout_ptr + p, h, mask=mask)

        # --- adaln_pre for the next layer ---
        b = base_ptr + pid_t * base_row_stride
        s_raw = tl.load(b + offs, mask=mask, other=0.0) + tl.load(
            emb_ptr + offs, mask=mask, other=0.0
        )
        sh_raw = tl.load(b + D + offs, mask=mask, other=0.0) + tl.load(
            emb_ptr + D + offs, mask=mask, other=0.0
        )
        g_raw = tl.load(b + 2 * D + offs, mask=mask, other=0.0) + tl.load(
            emb_ptr + 2 * D + offs, mask=mask, other=0.0
        )
        s_raw = s_raw.to(dt)
        sh = sh_raw.to(dt)
        g_raw = g_raw.to(dt)
        scale = (s_raw + 1.0).to(dt)
        gate = (g_raw + 1.0).to(dt)
        prod = (h * scale).to(dt)
        hmod = (prod + sh).to(dt)
        tl.store(hmod_ptr + p, hmod, mask=mask)
        tl.store(gate_ptr + p, gate, mask=mask)

        # --- operator_norm on hmod ---
        x = hmod.to(tl.float32)
        var = tl.sum(x * x, axis=0) / D
        x = x * tl.math.rsqrt(var + eps)
        if HAS_WEIGHT:
            wv = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            x = x * wv
        tl.store(normed_ptr + p, x.to(dt), mask=mask)


# Must match the dtypes rmsnorm.py's Triton path accepts.
EXACT_DTYPES = (torch.bfloat16, torch.float16)


def enabled() -> bool:
    return _flag() and _HAVE_TRITON


def eligible(*tensors: torch.Tensor) -> bool:
    """Runtime guard over every operand: CUDA, a verified dtype, and one shared dtype.

    Mixed dtypes are not exact: eager rounds `base` in the conditioning dtype.
    """
    ts = [t for t in tensors if t is not None]
    if not ts:
        return False
    first = ts[0]
    return (
        all(t.is_cuda for t in ts)
        and first.dtype in EXACT_DTYPES
        and all(t.dtype == first.dtype for t in ts)
    )


def adaln_pre_norm(h, adaln_base, emb, weight, eps: float):
    """(h_mod, gate, normed) in ONE launch. Shapes follow adaln_pre's."""
    orig = h.shape
    D = orig[-1]
    # The kernel indexes rows as `pid_t * D + offs`, so strided views must be copied.
    hf = h.reshape(-1, D).contiguous()
    T = hf.shape[0]
    BLOCK = triton.next_power_of_2(D)
    hmod = torch.empty_like(hf)
    gate = torch.empty_like(hf)
    normed = torch.empty_like(hf)
    if T == 0:
        return (hmod.view(orig), gate.view(orig), normed.view(orig))
    bf = adaln_base.reshape(-1, adaln_base.shape[-1]).contiguous()
    _adaln_pre_norm_kernel[(T,)](
        hf,
        bf,
        emb.reshape(-1).contiguous(),
        weight if weight is not None else hf,
        hmod,
        gate,
        normed,
        bf.stride(0),
        D=D,
        eps=eps,
        HAS_WEIGHT=weight is not None,
        BLOCK=BLOCK,
        # Must match _rmsnorm_kernel: warp count sets tl.sum's reduction order.
        num_warps=4 if BLOCK <= 1024 else 8,
        # Required: the .to(dt) casts do not stop LLVM contracting
        # `h*scale + shift` into a single-rounding FMA.
        enable_fp_fusion=False,
    )
    return (hmod.view(orig), gate.view(orig), normed.view(orig))


def adaln_post_pre_norm(
    h_prev, gate_prev, y, hmod_prev, adaln_base, emb, weight, eps: float, y2=None
):
    """(h, h_mod, gate, normed) in ONE launch; `y2` is an optional second addend of y."""
    orig = h_prev.shape
    D = orig[-1]
    hp = h_prev.reshape(-1, D).contiguous()
    T = hp.shape[0]
    BLOCK = triton.next_power_of_2(D)
    hout = torch.empty_like(hp)
    hmod = torch.empty_like(hp)
    gate = torch.empty_like(hp)
    normed = torch.empty_like(hp)
    if T == 0:
        return (hout.view(orig), hmod.view(orig), gate.view(orig), normed.view(orig))
    bf = adaln_base.reshape(-1, adaln_base.shape[-1]).contiguous()
    _adaln_post_pre_norm_kernel[(T,)](
        hp,
        gate_prev.reshape(-1, D).contiguous(),
        y.reshape(-1, D).contiguous(),
        y2.reshape(-1, D).contiguous() if y2 is not None else hp,
        hmod_prev.reshape(-1, D).contiguous(),
        bf,
        emb.reshape(-1).contiguous(),
        weight if weight is not None else hp,
        hout,
        hmod,
        gate,
        normed,
        bf.stride(0),
        D=D,
        eps=eps,
        HAS_WEIGHT=weight is not None,
        HAS_Y2=y2 is not None,
        BLOCK=BLOCK,
        num_warps=4 if BLOCK <= 1024 else 8,
        enable_fp_fusion=False,
    )
    return (hout.view(orig), hmod.view(orig), gate.view(orig), normed.view(orig))


def _hookable(layer):
    """The operator_norm this fusion would hook on `layer`, or None if already hooked.

    Omits the _dllm_triton_fwd check: consumer_norms runs before the norm installs.
    """
    norm = getattr(layer, "operator_norm", None)
    if norm is None or getattr(layer, "_dllm_norm_hooked", False):
        return None
    return norm


def active_layers(model) -> int:
    """Layers where this fusion is installed AND would actually dispatch.

    Mirrors the bail conditions in the installed `_norm_forward` wrapper.
    """
    return sum(
        1
        for layer in getattr(model, "layers", [])
        if getattr(layer, "_dllm_norm_hooked", False)
        and _triton_is_active(getattr(layer, "operator_norm", None))
    )


def consumer_norms(model) -> list:
    """The norm modules this fusion would hook, in install order.

    The norm installer scopes a consumer-licensed install to exactly these.
    """
    out = []
    for layer in getattr(model, "layers", []):
        norm = _hookable(layer)
        if norm is not None:
            out.append(norm)
    return out


def install(model) -> int:
    """Hook `operator_norm` to return, once, the normed tensor stashed on the layer.

    A call with no stash computes a real norm.
    """
    n = 0
    skipped = 0
    for layer in getattr(model, "layers", []):
        norm = _hookable(layer)
        if norm is None:
            continue
        # Only where operator_norm is already our Triton kernel; otherwise the
        # fusion would substitute different arithmetic for the platform norm.
        if getattr(norm, "_dllm_triton_fwd", None) is None:
            skipped += 1
            continue
        layer._dllm_normed = None
        _orig = norm.forward

        def _norm_forward(x, *a, _l=layer, _o=_orig, _n=norm, **kw):
            v = _l._dllm_normed
            _l._dllm_normed = None
            if v is None:
                return _o(x, *a, **kw)
            # torch-compile or a forced fused-op backend may have replaced the
            # Triton kernel since install; the stash is then wrong.
            if not _triton_is_active(_n):
                return _o(x, *a, **kw)
            return v

        norm.forward = _norm_forward
        layer._dllm_norm_hooked = True
        n += 1
    if skipped:
        logger.warning(
            f"[dllm-fused] adaln+norm skipped on {skipped} layers: their "
            "operator_norm is not the Triton kernel (set "
            "SGLANG_DLLM_FUSED_NORM=1 to install it). Fusing would change "
            "the norm's arithmetic.",
        )
    return n
