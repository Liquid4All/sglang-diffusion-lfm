"""Split-prefix (flash-decode style) attention for the dLLM denoise shape.

A denoise forward attends a window of ``block_size`` fresh queries against
the committed prefix KV and, bidirectionally, the window itself.
``extend_attention_fwd`` uses a ``(bs, heads, cdiv(W, BLOCK_M))`` grid in which
each program streams the whole prefix serially, which underfills the device at
small bs. Ported from ``verify_splitkv.py`` with a bidirectional ``tl.dot``
in-window stage, page-aware prefix addressing, and the whole window as one
query tile.

  * ``_dllm_prefix_stage1``: grid ``(bs, H_Q, N_SPLITS)``; online softmax over
    one prefix chunk, writing a normalised partial output and its LSE. Empty
    splits write an ``-inf`` LSE sentinel.
  * ``_dllm_combine_stage2``: grid ``(bs, H_Q)``; LSE-merges the partials, runs
    the in-window attention, and LSE-merges the two.

Not bitwise against ``extend_attention_fwd`` (split reduction and LSE merge
change fp32 association order), so it is opt-in via
``SGLANG_DLLM_USE_SPLIT_PREFIX_ATTN``. The commit forward never takes this
path, so committed KV is unchanged by it.

``dllm_splitkv_fwd`` takes the same positional args as ``extend_attention_fwd``
and returns False if the case is outside its contract. Every gate is value-free
(static shapes and python scalars) because it runs under captured graphs.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import _extract_kv_strides
from sglang.srt.utils import is_hip

_IS_HIP = is_hip()
_MIN_BLOCK_KV = 32

# BLOCK_N=64 matches extend_attention_fwd's head_dim-64 tile order. ROCm Triton
# does not pipeline this loop, so num_stages is 1 there.
BLOCK_N = 64
NUM_WARPS_STAGE1 = 4
NUM_STAGES_STAGE1 = 1 if _IS_HIP else 3
NUM_WARPS_STAGE2 = 4
_AMD_LAUNCH_KWARGS = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16} if _IS_HIP else {}

# Split count (tuned on H100 and MI325X): one split
# per ~256 prefix tokens, capped at ~4 programs per SM/CU because each split
# adds a fixed stage-2 cost. Python-int arithmetic only, so graph-capture safe.
MIN_N_SPLITS = 4
MAX_N_SPLITS = 16
TOKENS_PER_SPLIT = 256
PROGRAMS_PER_CORE = 4
# Below this average prefix the single-launch kernel is faster on both
# platforms (0.75x at 100, ~0.95x at 350, 1.1-1.2x at 512, growing after).
SPLIT_MIN_PREFIX = 512

_CORE_COUNT = None


def _min_prefix() -> int:
    """The env knob when set, else the module constant (import kept lazy so
    the kernel module stays importable without the srt package)."""
    try:
        from sglang.srt.environ import envs

        return int(envs.SGLANG_DLLM_SPLIT_PREFIX_ATTN_MIN_PREFIX.get())
    except Exception:  # pragma: no cover
        return SPLIT_MIN_PREFIX


def _device_core_count(device):
    global _CORE_COUNT
    if _CORE_COUNT is None:
        try:
            _CORE_COUNT = torch.cuda.get_device_properties(device).multi_processor_count
        except Exception:  # pragma: no cover
            _CORE_COUNT = 128
    return _CORE_COUNT


def _pow2_floor(n):
    n = int(n)
    return 1 << (n.bit_length() - 1) if n >= 1 else 1


def choose_n_splits(avg_prefix, bs, h_q, device):
    """Power-of-two split count in [MIN_N_SPLITS, MAX_N_SPLITS] from static
    shapes only."""
    by_len = _pow2_floor(int(avg_prefix) // TOKENS_PER_SPLIT)
    by_occ = _pow2_floor(
        PROGRAMS_PER_CORE * _device_core_count(device) // max(1, bs * h_q)
    )
    n = min(by_len, by_occ)
    return max(MIN_N_SPLITS, min(MAX_N_SPLITS, n))


@triton.jit
def _dllm_prefix_stage1(
    Q,  # [tokens, H_Q, D]
    K_Buffer,  # pool, 3-D [slots, H_KV, D] or 4-D [pages, PAGE_SIZE, H_KV, D]
    V_Buffer,
    sm_scale,
    k_scale,
    v_scale,
    qo_indptr,  # [BS+1]
    kv_indptr,  # [BS+1]
    kv_indices,  # [sum prefix]
    Att_Out,  # [max_bs, H_Q, N_SPLITS, L_PAD, Dv] fp32
    Att_Lse,  # [max_bs, H_Q, N_SPLITS, L_PAD]     fp32
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vbs,
    stride_buf_vh,
    stride_buf_vpage,
    stride_buf_vtok,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    kv_group_num: tl.constexpr,
    N_SPLITS: tl.constexpr,
    L_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_id = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = tl.arange(0, L_PAD)
    mask_d = offs_d < HEAD_DIM
    mask_dv = offs_dv < V_HEAD_DIM

    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext

    kv_start_idx = tl.load(kv_indptr + cur_batch)
    prefix_len = tl.load(kv_indptr + cur_batch + 1) - kv_start_idx

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(prefix_len, N_SPLITS), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_start = kv_len_per_split * split_id
    split_end = tl.minimum(split_start + kv_len_per_split, prefix_len)

    offs_lse = (
        cur_batch * stride_lb + cur_head * stride_lh + split_id * stride_ls + offs_l
    )

    if split_end > split_start:
        offs_q = (
            (cur_q_start + offs_l)[:, None] * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :]
        )
        q = tl.load(Q + offs_q, mask=mask_l[:, None] & mask_d[None, :], other=0.0)
        q_k = q.to(K_Buffer.dtype.element_ty)

        e_max = tl.zeros([L_PAD], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([L_PAD], dtype=tl.float32)
        acc = tl.zeros([L_PAD, BLOCK_DV], dtype=tl.float32)

        for start_n in tl.range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_end
            kv_loc = tl.load(kv_indices + kv_start_idx + offs_n, mask=n_mask, other=0)
            if PAGE_SIZE == 1:
                offs_k = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_d[:, None]
                )
                offs_v = (
                    kv_loc[:, None] * stride_buf_vbs
                    + cur_kv_head * stride_buf_vh
                    + offs_dv[None, :]
                )
            else:
                page_id = kv_loc // PAGE_SIZE
                tok_in_p = kv_loc % PAGE_SIZE
                offs_k = (
                    page_id[None, :] * stride_buf_kpage
                    + tok_in_p[None, :] * stride_buf_ktok
                    + cur_kv_head * stride_buf_kh
                    + offs_d[:, None]
                )
                offs_v = (
                    page_id[:, None] * stride_buf_vpage
                    + tok_in_p[:, None] * stride_buf_vtok
                    + cur_kv_head * stride_buf_vh
                    + offs_dv[None, :]
                )
            k = tl.load(
                K_Buffer + offs_k, mask=mask_d[:, None] & n_mask[None, :], other=0.0
            )
            qk = tl.dot(q_k, k)  # [L_PAD, BLOCK_N]
            qk *= sm_scale * k_scale
            # No prefix mask: every committed token is visible to every row.
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            v = tl.load(
                V_Buffer + offs_v, mask=n_mask[:, None] & mask_dv[None, :], other=0.0
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        acc *= v_scale
        offs_o = (
            cur_batch * stride_ob
            + cur_head * stride_oh
            + split_id * stride_os
            + offs_l[:, None] * stride_ol
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_o,
            acc / e_sum[:, None],
            mask=mask_l[:, None] & mask_dv[None, :],
        )
        tl.store(Att_Lse + offs_lse, e_max + tl.log(e_sum), mask=mask_l)
    else:
        # Empty split: sentinel LSE so stage 2 weights it exp(-inf) = 0; its
        # Att_Out rows are left unwritten and never read with nonzero weight.
        tl.store(
            Att_Lse + offs_lse,
            tl.zeros([L_PAD], tl.float32) - float("inf"),
            mask=mask_l,
        )


@triton.jit
def _dllm_combine_stage2(
    Att_Out,
    Att_Lse,
    Q,  # [tokens, H_Q, D]
    K_Extend,  # [tokens, H_KV, D]
    V_Extend,  # [tokens, H_KV, Dv]
    O_Out,  # [tokens, H_Q, Dv]
    sm_scale,
    qo_indptr,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_qbs,
    stride_qh,
    stride_kebs,
    stride_keh,
    stride_vebs,
    stride_veh,
    stride_oobs,
    stride_ooh,
    kv_group_num: tl.constexpr,
    N_SPLITS: tl.constexpr,
    L_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = tl.arange(0, L_PAD)
    offs_s = tl.arange(0, N_SPLITS)
    mask_d = offs_d < HEAD_DIM
    mask_dv = offs_dv < V_HEAD_DIM

    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext

    # (a) merge the prefix splits. An empty-prefix row (all LSE -inf) gets a
    # guarded max (no NaN), zero output and lse_prefix = -inf.
    offs_lse = (
        cur_batch * stride_lb
        + cur_head * stride_lh
        + offs_s[:, None] * stride_ls
        + offs_l[None, :]
    )
    lse = tl.load(Att_Lse + offs_lse, mask=mask_l[None, :], other=float("-inf"))
    m_p = tl.max(lse, 0)
    m_p_fixed = tl.where(m_p == float("-inf"), 0.0, m_p)
    w = tl.exp(lse - m_p_fixed[None, :])
    denom_p = tl.sum(w, 0)
    offs_ao = (
        cur_batch * stride_ob
        + cur_head * stride_oh
        + offs_s[:, None, None] * stride_os
        + offs_l[None, :, None] * stride_ol
        + offs_dv[None, None, :]
    )
    ao = tl.load(
        Att_Out + offs_ao,
        mask=mask_l[None, :, None]
        & mask_dv[None, None, :]
        & (lse != float("-inf"))[:, :, None],
        other=0.0,
    )
    o_prefix = tl.sum(ao * w[:, :, None], 0)
    has_prefix = denom_p > 0
    o_prefix = tl.where(
        has_prefix[:, None], o_prefix / tl.where(has_prefix, denom_p, 1.0)[:, None], 0.0
    )
    lse_prefix = tl.where(
        has_prefix,
        m_p_fixed + tl.log(tl.where(has_prefix, denom_p, 1.0)),
        float("-inf"),
    )

    # (b) bidirectional in-window attention, same arithmetic as the reference
    # extend loop (sm_scale only: fresh activations, not the fp8 pool).
    offs_q = (
        (cur_q_start + offs_l)[:, None] * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=mask_l[:, None] & mask_d[None, :], other=0.0)
    offs_ke = (
        (cur_q_start + offs_l)[:, None] * stride_kebs
        + cur_kv_head * stride_keh
        + offs_d[None, :]
    )
    ke = tl.load(K_Extend + offs_ke, mask=mask_l[:, None] & mask_d[None, :], other=0.0)
    offs_ve = (
        (cur_q_start + offs_l)[:, None] * stride_vebs
        + cur_kv_head * stride_veh
        + offs_dv[None, :]
    )
    ve = tl.load(V_Extend + offs_ve, mask=mask_l[:, None] & mask_dv[None, :], other=0.0)

    qk = tl.dot(q.to(ke.dtype), tl.trans(ke)) * sm_scale  # [L_PAD, L_PAD]
    valid = mask_l[:, None] & mask_l[None, :]
    qk = tl.where(valid, qk, float("-inf"))
    m_d = tl.max(qk, 1)
    m_d_fixed = tl.where(m_d == float("-inf"), 0.0, m_d)
    pd = tl.exp(qk - m_d_fixed[:, None])
    denom_d = tl.sum(pd, 1)
    o_win = tl.dot(pd.to(ve.dtype), ve)
    has_win = denom_d > 0
    o_win = o_win / tl.where(has_win, denom_d, 1.0)[:, None]
    lse_win = tl.where(
        has_win, m_d_fixed + tl.log(tl.where(has_win, denom_d, 1.0)), float("-inf")
    )

    # (c) merge prefix and window.
    m = tl.maximum(lse_prefix, lse_win)
    m_fixed = tl.where(m == float("-inf"), 0.0, m)
    wp = tl.exp(lse_prefix - m_fixed)
    wd = tl.exp(lse_win - m_fixed)
    den = wp + wd
    den = tl.where(den > 0, den, 1.0)
    o = (o_prefix * wp[:, None] + o_win * wd[:, None]) / den[:, None]

    offs_oo = (
        (cur_q_start + offs_l)[:, None] * stride_oobs
        + cur_head * stride_ooh
        + offs_dv[None, :]
    )
    tl.store(
        O_Out + offs_oo,
        o.to(O_Out.dtype.element_ty),
        mask=mask_l[:, None] & mask_dv[None, :],
    )


class DllmSplitKV:
    """Scratch buffers for one problem shape, sized by a stable ``max_bs`` so
    their addresses never move (graph-safe); the grid uses the call's bs."""

    def __init__(
        self, max_bs, h_q, h_kv, head_dim, v_head_dim, l_ext, n_splits, device
    ):
        self.h_q = h_q
        self.h_kv = h_kv
        self.group = h_q // h_kv
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.l_pad = triton.next_power_of_2(l_ext)
        self.n_splits = n_splits
        self.device = device
        self._alloc(max_bs)

    def _alloc(self, max_bs):
        self.max_bs = max_bs
        self.att_out = torch.empty(
            (max_bs, self.h_q, self.n_splits, self.l_pad, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.att_lse = torch.empty(
            (max_bs, self.h_q, self.n_splits, self.l_pad),
            dtype=torch.float32,
            device=self.device,
        )

    def grow_buffers(self, max_bs):
        if max_bs > self.max_bs:
            self._alloc(max_bs)

    def __call__(
        self,
        q_extend,
        k_extend,
        v_extend,
        o_out,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        k_scale,
        v_scale,
        page_size,
    ):
        bs = qo_indptr.shape[0] - 1
        k_bs, k_h, k_page, k_tok = _extract_kv_strides(k_buffer, page_size)
        v_bs, v_h, v_page, v_tok = _extract_kv_strides(v_buffer, page_size)
        _dllm_prefix_stage1[(bs, self.h_q, self.n_splits)](
            q_extend,
            k_buffer,
            v_buffer,
            sm_scale,
            k_scale,
            v_scale,
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.att_out,
            self.att_lse,
            q_extend.stride(0),
            q_extend.stride(1),
            k_bs,
            k_h,
            k_page,
            k_tok,
            v_bs,
            v_h,
            v_page,
            v_tok,
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            kv_group_num=self.group,
            N_SPLITS=self.n_splits,
            L_PAD=self.l_pad,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=BLOCK_N,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            PAGE_SIZE=page_size,
            num_warps=NUM_WARPS_STAGE1,
            num_stages=NUM_STAGES_STAGE1,
            **_AMD_LAUNCH_KWARGS,
        )
        _dllm_combine_stage2[(bs, self.h_q)](
            self.att_out,
            self.att_lse,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            sm_scale,
            qo_indptr,
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            q_extend.stride(0),
            q_extend.stride(1),
            k_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(0),
            v_extend.stride(1),
            o_out.stride(0),
            o_out.stride(1),
            kv_group_num=self.group,
            N_SPLITS=self.n_splits,
            L_PAD=self.l_pad,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            num_warps=NUM_WARPS_STAGE2,
            num_stages=1,
        )
        return o_out


_CACHE = {}


def _get(max_bs, h_q, h_kv, head_dim, v_head_dim, l_ext, n_splits, device):
    key = (h_q, h_kv, head_dim, v_head_dim, l_ext, n_splits, str(device))
    inst = _CACHE.get(key)
    if inst is None:
        inst = DllmSplitKV(
            max_bs, h_q, h_kv, head_dim, v_head_dim, l_ext, n_splits, device
        )
        _CACHE[key] = inst
    else:
        inst.grow_buffers(max_bs)
    return inst


def can_handle(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    *,
    sliding_window_size=-1,
    sinks=None,
    logit_cap=0.0,
    xai_temperature_len=-1,
    clean_upto=0,
    causal_rows=None,
    score_mod=None,
    lse_extend=None,
    skip_prefix=False,
    skip_extend=False,
    min_prefix=None,
    avg_prefix=None,
):
    """Value-free contract check; False means fall back to extend_attention_fwd.

    Only the plain bidirectional window over a bf16, non-MLA pool is accepted.
    """
    # Only an explicit bidirectional request; None means the layer default,
    # which is causal for this model.
    if is_causal is not False:
        return False
    if custom_mask is not None or mask_indptr is not None:
        return False
    if causal_rows is not None or (clean_upto is not None and int(clean_upto) > 0):
        return False
    if sinks is not None:
        return False
    if sliding_window_size is not None and sliding_window_size > 0:
        return False
    if logit_cap and logit_cap > 0:
        return False
    if xai_temperature_len is not None and xai_temperature_len > 0:
        return False
    if score_mod is not None:
        return False
    if lse_extend is not None or skip_prefix or skip_extend:
        return False
    if q_extend.dim() != 3 or k_extend.dim() != 3 or v_extend.dim() != 3:
        return False
    if q_extend.dtype != torch.bfloat16 or k_buffer.dtype != torch.bfloat16:
        return False
    if k_buffer.dtype != v_buffer.dtype or k_extend.dtype != q_extend.dtype:
        return False
    h_q, h_kv = q_extend.shape[1], k_extend.shape[1]
    if h_kv == 0 or h_q % h_kv != 0:
        return False
    if k_buffer.shape[-2] != h_kv or v_buffer.shape[-2] != h_kv:
        return False
    d, dv = q_extend.shape[2], v_extend.shape[2]
    if d != k_extend.shape[2] or d != k_buffer.shape[-1] or dv != v_buffer.shape[-1]:
        return False
    if d != dv or d > 128:
        return False
    bs = qo_indptr.shape[0] - 1
    if bs < 1:
        return False
    try:
        mle = int(max_len_extend)
    except (TypeError, ValueError):
        return False
    if mle < 1 or mle > 64:
        return False
    if q_extend.shape[0] != bs * mle:
        return False
    if avg_prefix is None:
        avg_prefix = kv_indices.shape[0] / bs
    thr = _min_prefix() if min_prefix is None else min_prefix
    if avg_prefix < thr:
        return False
    return True


def dllm_splitkv_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale,
    v_scale,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    clean_upto=0,
    causal_rows=None,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    lse_extend=None,
    skip_prefix=False,
    skip_extend=False,
    page_size=1,
    score_mod=None,
    aux_tensors=None,
    extend_seq_lens_cpu=None,
    max_bs=None,
    n_splits=None,
    min_prefix=None,
    avg_prefix=None,
):
    """Drop-in for ``extend_attention_fwd`` on the dLLM denoise shape.

    Returns True if it ran, False if nothing was done. Pass ``avg_prefix`` under
    CUDA graphs, where ``kv_indices.shape[0] / bs`` is the pool capacity.
    """
    if not can_handle(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        is_causal,
        mask_indptr,
        max_len_extend,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        clean_upto=clean_upto,
        causal_rows=causal_rows,
        score_mod=score_mod,
        lse_extend=lse_extend,
        skip_prefix=skip_prefix,
        skip_extend=skip_extend,
        min_prefix=min_prefix,
        avg_prefix=avg_prefix,
    ):
        return False

    bs = qo_indptr.shape[0] - 1
    h_q, h_kv = q_extend.shape[1], k_extend.shape[1]
    head_dim, v_head_dim = q_extend.shape[2], v_extend.shape[2]
    l_ext = int(max_len_extend)
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)
    try:
        k_scale = float(k_scale)
    except (TypeError, ValueError):
        k_scale = 1.0
    try:
        v_scale = float(v_scale)
    except (TypeError, ValueError):
        v_scale = 1.0
    if n_splits is None:
        n_splits = choose_n_splits(
            kv_indices.shape[0] / bs if avg_prefix is None else avg_prefix,
            bs,
            h_q,
            q_extend.device,
        )
    if max_bs is None or max_bs < bs:
        max_bs = bs
    inst = _get(
        max_bs, h_q, h_kv, head_dim, v_head_dim, l_ext, n_splits, q_extend.device
    )
    inst(
        q_extend,
        k_extend.contiguous(),
        v_extend.contiguous(),
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        k_scale,
        v_scale,
        int(page_size),
    )
    _note_launch()
    return True


# Launch evidence: a flag proves intent, not a launch. The first launch writes
# "<SGLANG_DLLM_FUSED_MARKER>.splitattn"; LAUNCHES counts in memory for tests.
LAUNCHES = 0


def _note_launch() -> None:
    global LAUNCHES
    LAUNCHES += 1
    if LAUNCHES != 1:
        return
    try:
        from sglang.srt.environ import envs

        mk = envs.SGLANG_DLLM_FUSED_MARKER.get()
    except Exception:  # pragma: no cover
        mk = None
    if not mk:
        return
    try:
        with open(str(mk) + ".splitattn", "w") as f:
            f.write(f"splitattn launches={LAUNCHES}\n")
    except OSError:  # pragma: no cover
        pass
