"""Fused denoise step: log-softmax, argmax/Gumbel-max into the canvas, and masked probs.

Not bitwise: the online softmax over BLOCK_V chunks can move log(sum) by the last
fp32 bit, so the argmax differs only on that close a tie.
"""

import logging

import torch

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
    def _greedy_readout_kernel(
        logits_ptr,
        stride_row,
        x_ptr,  # canvas rows to update in place: x[r] (int64), stride_x
        stride_x,
        upd_ptr,  # bool per row
        probs_ptr,  # [rows, V] activation dtype, contiguous
        temp,
        V,
        BLOCK_V: tl.constexpr,
    ):
        r = tl.program_id(0).to(tl.int64)
        row = logits_ptr + r * stride_row
        offs = tl.arange(0, BLOCK_V)
        # pass 1: online max / sum-exp; while the running max is -inf both
        # factors are forced to 0, since exp(-inf - -inf) is NaN.
        m = tl.zeros((), tl.float32) - float("inf")
        s = tl.zeros((), tl.float32)
        nan_seen = tl.zeros((), tl.int32)
        for start in range(0, V, BLOCK_V):
            idx = start + offs
            mask = idx < V
            y = tl.load(row + idx, mask=mask, other=float("-inf")).to(tl.float32) / temp
            # NaN or +inf anywhere: torch's log_softmax row is all NaN, argmax 0
            nan_seen += tl.sum(
                tl.where(mask & ((y != y) | (y == float("inf"))), 1, 0), axis=0
            )
            m_new = tl.maximum(m, tl.max(y, axis=0))
            alive = m_new > float("-inf")
            alpha = tl.where(alive, tl.exp(m - m_new), 0.0)
            pe = tl.where(alive, tl.exp(y - m_new), 0.0)
            s = s * alpha + tl.sum(pe, axis=0)
            m = m_new
        log_s = tl.log(s)
        # all -inf row: torch's log_softmax is all NaN and its argmax is 0
        dead = (m == float("-inf")) | (nan_seen > 0)
        # pass 2: argmax is the first index of max(log_x0), taken on log_x0
        # rather than y since subtracting the constant can create ties.
        upd = tl.load(upd_ptr + r).to(tl.int1)
        best_v = tl.zeros((), tl.float32) - float("inf")
        best = tl.zeros((), tl.int32) + 0x7FFFFFFF
        for start in range(0, V, BLOCK_V):
            idx = start + offs
            mask = idx < V
            y = tl.load(row + idx, mask=mask, other=float("-inf")).to(tl.float32) / temp
            lx = (y - m) - log_s
            cm = tl.max(lx, axis=0)
            cand = tl.min(tl.where(lx == cm, idx, 0x7FFFFFFF), axis=0)
            best = tl.where(cm > best_v, cand, best)
            best_v = tl.maximum(best_v, cm)
            p = tl.exp(lx)
            p = tl.where(upd, p, 0.0)
            tl.store(
                probs_ptr + r * V + idx, p.to(probs_ptr.dtype.element_ty), mask=mask
            )
        # canvas: argmax where the mask is set, else keep; a row with no
        # candidate is torch's all-NaN row, whose argmax is 0.
        best = tl.where(dead | (best == 0x7FFFFFFF), 0, best)
        xv = tl.load(x_ptr + r * stride_x)
        nxt = tl.where(upd, best.to(tl.int64), xv)
        tl.store(x_ptr + r * stride_x, nxt)


# One program streams a row twice, so it is bound by single-CTA bandwidth;
# largest block measured fastest on H100 (18.0 us vs 39.5 us at 2048/8).
BLOCK_V = 16384
# HIP caps a block at 1024 threads = 16 wavefronts of 64; 32 warps fails to launch.
NUM_WARPS = 16 if torch.version.hip else 32


def greedy_readout(
    logits: torch.Tensor,  # [rows, Vpad] (any row stride), activation dtype
    canvas: torch.Tensor,  # [rows] int64 view into the canvas; updated IN PLACE
    upd: torch.Tensor,  # [rows] bool
    vocab_size: int,
    temperature: float,
    probs_out: torch.Tensor,  # [rows, V] activation dtype, contiguous
) -> torch.Tensor:
    rows = logits.shape[0]
    assert canvas.dim() == 1 and canvas.shape[0] == rows and canvas.dtype == torch.int64
    assert upd.shape[0] == rows and upd.dtype == torch.bool
    assert probs_out.shape == (rows, vocab_size) and probs_out.is_contiguous()
    assert logits.stride(1) == 1 and logits.shape[1] >= vocab_size
    assert (
        temperature > 0.0 and temperature == temperature and temperature != float("inf")
    ), temperature
    if rows == 0:
        return probs_out
    _greedy_readout_kernel[(rows,)](
        logits,
        logits.stride(0),
        canvas,
        canvas.stride(0),
        upd,
        probs_out,
        float(temperature),
        int(vocab_size),
        BLOCK_V=BLOCK_V,
        num_warps=NUM_WARPS,
    )
    _note_launch()
    return probs_out


if _HAS_TRITON:

    @triton.jit
    def _ancestral_step_kernel(
        logits_ptr,
        stride_row,
        x_ptr,  # canvas rows, int64, updated in place where upd
        stride_x,
        upd_ptr,
        u_ptr,  # [rows, V] fp32 uniforms from torch.rand(generator) -- the ONLY randomness
        probs_ptr,  # [rows, V] probs in E's dtype (for self-conditioning)
        alpha_ptr,  # fp32 [2]: alpha_t, alpha_s for this step (the fp32 schedule values)
        temp,
        V,
        BLOCK_V: tl.constexpr,
    ):
        """duo_reverse_from_logits' ancestral branch (kappa=1, no truncation, use_float64).

        Only the online softmax reduction is non-bitwise, as in greedy."""
        r = tl.program_id(0).to(tl.int64)
        row = logits_ptr + r * stride_row
        offs = tl.arange(0, BLOCK_V)
        m = tl.zeros((), tl.float32) - float("inf")
        s = tl.zeros((), tl.float32)
        nan_seen = tl.zeros((), tl.int32)
        for start in range(0, V, BLOCK_V):
            idx = start + offs
            mask = idx < V
            y = tl.load(row + idx, mask=mask, other=float("-inf")).to(tl.float32) / temp
            # NaN or +inf anywhere: torch's log_softmax row is all NaN, argmax 0
            nan_seen += tl.sum(
                tl.where(mask & ((y != y) | (y == float("inf"))), 1, 0), axis=0
            )
            m_new = tl.maximum(m, tl.max(y, axis=0))
            alive = m_new > float("-inf")
            alpha = tl.where(alive, tl.exp(m - m_new), 0.0)
            pe = tl.where(alive, tl.exp(y - m_new), 0.0)
            s = s * alpha + tl.sum(pe, axis=0)
            m = m_new
        log_s = tl.log(s)
        dead = (m == float("-inf")) | (nan_seen > 0)
        # posterior constants, in fp64 exactly as the eager path widens them
        a_t = tl.load(alpha_ptr).to(tl.float64)
        a_s = tl.load(alpha_ptr + 1).to(tl.float64)
        a_ts = a_t / a_s
        d_alpha = a_s - a_t
        Vf = (V * 1.0).to(tl.float64)  # exact for any real vocabulary size
        c0 = (1.0 - a_ts) * (1.0 - a_s) / Vf
        xt = tl.load(x_ptr + r * stride_x)
        y_xt = tl.load(row + xt).to(tl.float32) / temp
        p_xt = tl.exp((y_xt - m) - log_s).to(tl.float64)
        denom = a_t * Vf * p_xt + (1.0 - a_t)
        upd = tl.load(upd_ptr + r).to(tl.int1)
        best_v = tl.zeros((), tl.float32) - float("inf")
        best = tl.zeros((), tl.int32) + 0x7FFFFFFF
        for start in range(0, V, BLOCK_V):
            idx = start + offs
            mask = idx < V
            y = tl.load(row + idx, mask=mask, other=float("-inf")).to(tl.float32) / temp
            lx = (y - m) - log_s
            p = tl.exp(lx)
            tl.store(
                probs_ptr + r * V + idx,
                tl.where(upd, p, 0.0).to(probs_ptr.dtype.element_ty),
                mask=mask,
            )
            p64 = p.to(tl.float64)
            onehot = (idx == xt).to(tl.float64)
            numerator = (
                a_t * Vf * p64 * onehot + (a_ts - a_t) * onehot + d_alpha * p64 + c0
            )
            q = numerator / denom
            q = tl.maximum(q, 1e-30)
            lq = tl.log(q).to(tl.float32)
            u = tl.load(u_ptr + r * V + idx, mask=mask, other=1.0)
            u = tl.maximum(u, 1e-20)
            g = -tl.log(-tl.log(u))
            score = tl.where(mask, lq + g, float("-inf"))
            cm = tl.max(score, axis=0)
            cand = tl.min(tl.where(score == cm, idx, 0x7FFFFFFF), axis=0)
            best = tl.where(cm > best_v, cand, best)
            best_v = tl.maximum(best_v, cm)
        best = tl.where(dead | (best == 0x7FFFFFFF), 0, best)
        nxt = tl.where(upd, best.to(tl.int64), xt)
        tl.store(x_ptr + r * stride_x, nxt)


def ancestral_step(
    logits: torch.Tensor,  # [rows, Vpad] float
    canvas: torch.Tensor,  # [rows] int64, updated in place
    upd: torch.Tensor,  # [rows] bool
    u: torch.Tensor,  # [rows, V] fp32 uniforms (torch.rand with the request generator)
    alphas: torch.Tensor,  # fp32 [2]: alpha_t, alpha_s
    vocab_size: int,
    temperature: float,
    probs_out: torch.Tensor,  # [rows, V] in E's dtype
) -> torch.Tensor:
    rows = logits.shape[0]
    assert (
        canvas.shape == (rows,) and canvas.dtype == torch.int64 and upd.shape == (rows,)
    )
    assert (
        u.shape == (rows, vocab_size) and u.dtype == torch.float32 and u.is_contiguous()
    )
    assert probs_out.shape == (rows, vocab_size) and probs_out.is_contiguous()
    assert alphas.dtype == torch.float32 and alphas.numel() == 2
    assert logits.stride(1) == 1 and logits.shape[1] >= vocab_size
    assert (
        temperature > 0.0 and temperature == temperature and temperature != float("inf")
    )
    if rows == 0:
        return probs_out
    _ancestral_step_kernel[(rows,)](
        logits,
        logits.stride(0),
        canvas,
        canvas.stride(0),
        upd,
        u,
        probs_out,
        alphas,
        float(temperature),
        int(vocab_size),
        BLOCK_V=BLOCK_V,
        num_warps=NUM_WARPS,
    )
    _note_launch()
    return probs_out


# The flag alone does not prove the kernel ran (DuoBlock._fused_readout_ok has
# further checks), so the first launch writes "<SGLANG_DLLM_FUSED_MARKER>.denoise_step".
LAUNCHES = 0


def _note_launch() -> None:
    global LAUNCHES
    LAUNCHES += 1
    if LAUNCHES != 1:
        return
    mk = envs.SGLANG_DLLM_FUSED_MARKER.get()
    if not mk:
        return
    try:
        with open(str(mk) + ".denoise_step", "w") as f:
            f.write("denoise_step launches=1\n")
    except OSError:  # pragma: no cover
        pass


def ancestral_reference(
    logits, x, upd, u, a_t, a_s, vocab_size, temperature, out_dtype
):
    """Eager ancestral step with a given uniform draw u, for the parity test."""
    import torch.nn.functional as F

    from sglang.srt.dllm import duo_math

    lg = logits[..., :vocab_size].float()
    log_x0 = F.log_softmax(lg / temperature, dim=-1)
    x0 = log_x0.exp().double()
    q = duo_math.duo_posterior_from_x0(
        x0[None],
        x[None],
        a_s.double().reshape(1, 1, 1),
        a_t.double().reshape(1, 1, 1),
        vocab_size,
    )[0]
    q = q.clamp_min(1e-30)
    g = -(-u.clamp_min(1e-20).log()).log()
    x_next = (q.clamp_min(1e-30).log().float() + g).argmax(-1)
    canvas = torch.where(upd, x_next, x)
    probs = log_x0.exp().to(out_dtype) * upd.to(out_dtype).unsqueeze(-1)
    return canvas, probs


def enabled() -> bool:
    return bool(envs.SGLANG_DLLM_FUSE_DENOISE_STEP.get()) and _HAS_TRITON
