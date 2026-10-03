"""DuoBlock: uniform-state (DUO) block-diffusion sampler for LFM2-diffusion.

Port of the reference DUO block sampler onto SGLang's dLLM loop. Per block:

    [pre]     replace canvas placeholders with the DUO prior (uniform random
              tokens; zero-init under greedy parity mode), snapshot conv state
    fwd 0..N  denoise: bidirectional-in-block forwards at sigma_i, ancestral
              posterior reverse step (temp annealed, kappa, nucleus), optional
              self-conditioning fed to the next step, adaptive early stop
    fwd N+1   readout: forward at sigma = max(terminal sigma, floor), terminal
              reverse step (alpha_s = 1) -> final tokens
    fwd N+2   commit: forward with the final tokens at sigma = 0; causal under
              --dllm-prefix-attention causal, bidirectional under
              "bidirectional". Its KV is what later blocks attend. It does not change tokens, except that
              ar_verify / ar_stop may replace draft tokens and run a
              correction forward that re-persists the block.

Under commit_fusion there is no separate commit forward. A request's first
generation block runs narrow and persists nothing; every later block k's first
forward is 2*block_size wide and persists block k-1's final tokens (KV, and
the conv state stopped at the carried boundary); its other forwards run narrow.

All requests in a dLLM batch share one phase/schedule (batch rows stop together).

LFM2's short-conv cache is a sequential accumulator: only the forward that
persists the block (the commit forward, or under fusion the next block's wide
first forward) may advance it. With the fused short conv (default) the other
forwards leave it untouched; with the unfused conv every forward after the first
restores the pre-block snapshot.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from typing import Any, List, Optional

import torch

from sglang.srt.dllm import duo_math
from sglang.srt.dllm.algorithm import duo_debug
from sglang.srt.dllm.algorithm.base import (
    DllmAlgorithm,
    dllm_device_synchronize,
    dllm_sync_enabled,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.params import DllmContractError, dllm_graph_flag
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# SGLANG_DUO_TRACE=<path>: one record per reverse step (phase, step, sigma, canvas),
# dumped as JSON when the block completes.
_TRACE_PATH = envs.SGLANG_DUO_TRACE.get()
_TRACE: List[dict] = []


# One-shot per process. Module scope keeps _selfcond_embed callable with a
# minimal stub from the TP gate.
_logged_shard = False


# AR self-verification, measurement only: the commit forward (sigma 0, causal
# under tcp) gives AR next-token logits, so logits at p-1 verify the token at p.
_VERIFY_AR = envs.SGLANG_DLLM_VERIFY_AR.get()

# Read once at import: the graph slot for this tensor is registered at engine
# start, so it cannot change per forward.
_TENSOR_CAUSALITY = envs.SGLANG_DLLM_ENABLE_TENSOR_CAUSALITY.get()
_AR_STATS: List[dict] = []


logger = logging.getLogger(__name__)

# Teacher forcing for parity: force the canvas to the reference trace's after
# each step so every step is comparable despite early bf16 tie-flips.
_TF_PATH = envs.SGLANG_DLLM_TEACHER_CANVAS.get()
_TF_CANVAS: Optional[dict] = None
if _TF_PATH:
    with open(_TF_PATH) as _f:
        _TF_CANVAS = {
            (int(r["block"]), int(r["i"])): r["canvas"]
            for r in json.load(_f)
            if "canvas" in r and r.get("canvas") is not None
        }
    logger.warning(
        "dllm-teacher-forcing: %d (block, step) canvases loaded from %s. This "
        "OVERRIDES the sampler's own output every step -- a run in this mode "
        "measures per-step agreement, NOT what the engine would generate.",
        len(_TF_CANVAS),
        _TF_PATH,
    )


class _Buffers:
    """Per-(batch, block) tensors reused across forwards; a captured graph reads
    ``dllm_sigma``/``dllm_selfcond`` by address, so they are written in place."""

    __slots__ = (
        "sigma",
        "sc",
        "upd_f",
        "x_in",
        "b",
        "lb",
        "hidden",
        "sc_pos",
        "causal_rows",
        "upd_idx",
        "adaln_base",
        "upd",
        "state_at",
        "fusion_at",
    )

    def __init__(self, b: int, lb: int, device, dtype=torch.float32):
        self.b, self.lb, self.hidden = b, lb, None
        self.sigma = torch.zeros(b * lb, dtype=dtype, device=device)
        self.upd_f = torch.zeros(b, lb, dtype=dtype, device=device)
        # int view of upd for index_select into the per-step adaLN table
        self.upd_idx = torch.zeros(b * lb, dtype=torch.long, device=device)
        # fixed-address: the block graph reads st.upd by pointer
        self.upd = torch.zeros(b, lb, dtype=torch.bool, device=device)
        self.adaln_base = None  # [B*Lb, 3*hidden], on first use
        self.x_in = torch.zeros(b, lb, dtype=torch.long, device=device)
        self.sc = None  # [B, Lb, hidden], on first self-cond use
        self.sc_pos = None  # [B*W] selfcond position gate (fusion)
        # [B] per-request causality. Same fixed-address rule as sigma: a
        # captured graph reads it by pointer, so it is written in place.
        self.causal_rows = torch.zeros(b, dtype=torch.int32, device=device)
        # [B] per-request committed length for AR-verify's correction forward;
        # allocated on first use.
        self.state_at = None
        # [B] commit fusion's conv state point: every row stops at the carried
        # half's end. Constant per shape, so filled once.
        self.fusion_at = None

    def fusion_state_at(self, b: int, c: int, device) -> torch.Tensor:
        if self.fusion_at is None:
            self.fusion_at = torch.full((b,), int(c), dtype=torch.int32, device=device)
        return self.fusion_at

    def ar_state_at(self, b: int, device) -> torch.Tensor:
        if self.state_at is None:
            self.state_at = torch.zeros(b, dtype=torch.int32, device=device)
        return self.state_at

    def matches(self, b: int, lb: int, device) -> bool:
        return self.b == b and self.lb == lb and self.sigma.device == device

    def sc_buffer(self, hidden: int) -> torch.Tensor:
        if self.sc is None or self.hidden != hidden:
            self.hidden = hidden
            self.sc = torch.zeros(
                self.b,
                self.lb,
                hidden,
                dtype=self.sigma.dtype,
                device=self.sigma.device,
            )
        return self.sc


class _BlockCtx:
    """Per-batch scratch for the block in flight, carried on the batch's rows.

    Under FDFO another batch can run between two forwards of this one, so every
    hook re-binds the algorithm's mirrors from this ctx (``_bind``)."""

    __slots__ = ("buf", "ts", "start_list", "blk_idx")

    def __init__(self, buf, ts, start_list, blk_idx):
        self.buf = buf
        self.ts = ts
        self.start_list = start_list
        self.blk_idx = blk_idx


class _BatchState:
    """Per-request state, one object per row. Rows advance in lockstep: row 0 is
    read, writes fan out (``_fanout``); mixed phases are refused."""

    __slots__ = (
        "ctx",
        "row",
        "i",
        "phase",
        "sc",
        "x_in",
        "steps_run",
        "conv_snapshot",
        "cache_indices",
        "generator",
        "upd",
        "spb",
        "greedy",
        # Effective sampler for this block: the request's value, else the YAML's.
        "temperature",
        "top_p",
        "top_k",
        # AR best-of-N candidates, their AR scores, and the one being scored.
        # Shared, not per-row: every row is scored on the same pass.
        "bon",
        "bon_scores",
        "bon_i",
        # ar_bon_mode="traj": the running trajectory and the block-entry canvas,
        # restored before each restart so every trajectory starts from one x_T.
        "bon_t",
        "bon_x0",
        "bon_steps",
        # ar_redo: rounds left, and the block's original step count (a redo leg
        # runs a shorter schedule, so spb is rewritten).
        "redo_left",
        "spb0",
    )

    # the bookkeeping every row of a batch must agree on (compared by value)
    _UNIFORM_FIELDS = (
        "phase",
        "i",
        "spb",
        "greedy",
        "temperature",
        "top_p",
        "top_k",
        "steps_run",
        "bon_i",
        "bon_t",
        "bon_steps",
        "redo_left",
        "spb0",
    )
    # the batch-shaped tensors / containers every row must SHARE (by identity)
    _SHARED_FIELDS = (
        "ctx",
        "sc",
        "x_in",
        "upd",
        "conv_snapshot",
        "bon",
        "bon_scores",
        "bon_x0",
    )
    # per-row identity, never fanned out
    _ROW_FIELDS = ("row",)

    def __init__(self, row: int = 0):
        self.ctx = None  # _BlockCtx of the batch this row is in
        # Block scratch is positional; prepare_forward refuses a batch whose rows
        # moved, or every row would get another request's mask.
        self.row = row
        self.i = 0  # next denoise step index
        self.spb = None  # effective NFE for THIS block
        self.greedy = None  # effective argmax-vs-ancestral for THIS block
        self.temperature = None  # static temperature (unused when annealing)
        self.top_p = None
        self.top_k = None
        self.phase = "denoise"  # denoise -> readout -> commit -> done
        self.sc = None  # [B, Lb, hidden] self-cond for next fwd
        self.x_in = None  # canvas entering the last fwd (adaptive stop)
        self.steps_run = 0
        self.conv_snapshot = None  # list[(layer_idx, idx, tensor)] pre-block
        self.cache_indices = None
        self.generator = None
        # [B, Lb] bool: positions being denoised. False = clean prompt tail inside
        # the aligned window: held at sigma 0, never rewritten, no self-cond.
        self.upd = None
        self.bon = None  # list of [B, Lb] candidate blocks
        self.bon_scores = None  # list of [B] mean AR logprob per candidate
        self.bon_i = 0  # candidate currently being scored
        self.bon_t = 0  # trajectory currently being denoised (traj mode)
        self.bon_x0 = None  # [B, W] canvas at block entry (traj mode)
        # Reverse steps banked from finished trajectories (st.i restarts at 0
        # per trajectory).
        self.bon_steps = 0
        self.redo_left = 0  # ar_redo rounds remaining for this block
        self.spb0 = None  # the block's step count before any redo leg


# Every key DuoBlock (or the shared bounds/config readers) consults; an added
# `c.get("x")` with no entry fails the round-trip check in test_algorithm_registry.
_DUOBLOCK_KEYS = frozenset(
    {
        # Not "algorithm": that is server_args.dllm_algorithm, never read from
        # the YAML. block_size stays: config.py reads it from algorithm_config.
        "adaptive_stop",
        "block_size",
        "commit_fusion",
        "eps",
        "greedy",
        "kappa",
        "max_steps_per_block",
        "noise_removal",
        "rho",
        "schedule",
        "self_cond",
        "steps_per_block",
        "stop_entropy",
        "temp_anneal",
        "temp_end",
        "temp_schedule",
        "temp_start",
        "temperature",
        "terminal_sigma_floor",
        "top_k",
        "top_p",
        "top_p_site",
        "use_float64",
        # confidence-targeted posterior edits: all off at their defaults
        # AR self-verification, enforcing: off at its default
        "ar_verify",
        "ar_verify_tau",
        "ar_repair",
        "ar_repair_width",
        "ar_stop",
        # AR best-of-N at the readout: off at its default
        "ar_bon",
        "ar_bon_temp",
        "ar_bon_mode",
        "ar_bon_resample_x0",
        "ar_bon_temp_spread",
        # AR-triggered partial redo: off at its default
        "ar_redo",
        "ar_redo_tau",
        "ar_redo_frac",
        "ar_redo_steps",
        # what temperature=0 means: "argmax" at its default
        "greedy_mode",
        # AR-allocated NFE: off at its default
        "ar_nfe_adapt",
        "ar_nfe_lo",
        "ar_nfe_hi",
        "ar_nfe_tau",
        "ar_nfe_signal",
    }
)


# Keys that are real but not the algorithm config's to set, mapped to where each
# comes from. test_algorithm_registry pins this group against DLLM_PARAMS.
_FROM_DLLM_PARAMS = ("anchored", "init_mode", "mask_id", "real_vocab_size")
_ELSEWHERE = {
    k: "a per-architecture constant in DLLM_PARAMS (dllm/config.py), keyed on "
    "the model's architecture -- not settable per run"
    for k in _FROM_DLLM_PARAMS
}
# CLI-backed DllmConfig fields (fed by DllmConfig.from_server_args), with the
# flag that sets each.
_FROM_SERVER_ARGS = {
    "prefix_attention": "--dllm-prefix-attention {causal,bidirectional}",
    "cuda_graph": "--dllm-cuda-graph",
    "fdfo": "--no-dllm-fdfo",
    "first_done_first_out_mode": "--no-dllm-fdfo",
    "max_running_requests": "--max-running-requests",
    "radix_cache": "--disable-radix-cache / --dllm-experimental-prefix-cache",
    "algorithm": "--dllm-algorithm",
}
_ELSEWHERE.update(
    {k: f"set by the server flag {v}" for k, v in _FROM_SERVER_ARGS.items()}
)


# Removed keys, so a stale config gets a specific error instead of a fuzzy match.
_REMOVED_KEYS = {
    "prefix_mode": (
        "removed: clean-context attention follows the server flag "
        "--dllm-prefix-attention, which also governs prompt prefill"
    ),
    **{
        k: "removed: the confidence-targeted commit/renoise edits were never used"
        for k in (
            "commit_conf",
            "commit_conf_end",
            "renoise_conf",
            "renoise_conf_end",
            "renoise_prob",
        )
    },
    "nfe_counts_readout": (
        "removed: NFE always counts the readout (steps_per_block=8 is 7 "
        "denoise steps + 1 readout). The flag only ever renamed the budget -- "
        "false at steps_per_block=N was exactly steps_per_block=N+1. Drop the "
        "key; if it was false, add 1 to steps_per_block (and to "
        "max_steps_per_block if that was equal to it)"
    ),
}


def _reject_unknown_keys(c) -> None:
    """Raise on unknown algorithm-config keys, suggesting the nearest real key."""
    if not c:
        return
    unknown = sorted(k for k in c if k not in _DUOBLOCK_KEYS)
    if not unknown:
        return
    import difflib

    hints = []
    for k in unknown:
        if k in _REMOVED_KEYS:
            hints.append(f"{k!r} -- {_REMOVED_KEYS[k]}")
            continue
        if k in _ELSEWHERE:
            hints.append(f"{k!r} -- {_ELSEWHERE[k]}")
            continue
        near = difflib.get_close_matches(k, sorted(_DUOBLOCK_KEYS), n=1)
        hints.append(f"{k!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
    raise DllmContractError(
        "unknown key(s) in the DuoBlock algorithm config: "
        + ", ".join(hints)
        + ". Unrecognised keys are ignored by every reader, so a typo here "
        "silently changes the decode instead of failing. Known keys: "
        + ", ".join(sorted(_DUOBLOCK_KEYS))
    )


class DuoBlock(DllmAlgorithm):
    def __init__(self, config: DllmConfig):
        super().__init__(config)
        if config.init_mode != "uniform_random":
            raise DllmContractError(
                "DuoBlock is a uniform-state sampler; register the arch with "
                "init_mode='uniform_random' in DLLM_PARAMS."
            )
        self.real_vocab_size = config.real_vocab_size
        c = config.algorithm_config
        _reject_unknown_keys(c)
        # Default NFE and the ceiling for the per-request override, shared with
        # the admission-time validator.
        from sglang.srt.dllm.params import steps_per_block_bounds

        self.steps_per_block, self.max_steps_per_block = steps_per_block_bounds(c)
        if self.max_steps_per_block < self.steps_per_block:
            raise DllmContractError(
                f"max_steps_per_block={self.max_steps_per_block} < default "
                f"steps_per_block={self.steps_per_block}"
            )
        self.eps = float(c.get("eps", 1e-3))
        self.schedule = c.get("schedule", "linear")
        self.rho = float(c.get("rho", 1.0))
        self.temperature = float(c.get("temperature", 1.0))
        # Bounds given means anneal between them and the static temperature is
        # unused; no bounds means the static temperature. An explicit
        # temp_anneal still wins.
        self.temp_anneal = bool(
            c.get("temp_anneal", "temp_start" in c or "temp_end" in c)
        )
        self.temp_start = float(c.get("temp_start", 0.8))
        self.temp_end = float(c.get("temp_end", 0.4))
        self.temp_schedule = c.get("temp_schedule", "linear")
        # 0.0 = the schedule's natural endpoint, as in the DUO reference.
        self.terminal_sigma_floor = float(c.get("terminal_sigma_floor", 0.0))
        self.kappa = float(c.get("kappa", 1.0))
        for _name, _ok, _want in (
            ("steps_per_block", self.steps_per_block >= 1, ">= 1"),
            ("eps", 0.0 < self.eps < 1.0, "in (0, 1)"),
            ("rho", self.rho > 0.0, "> 0"),
            ("kappa", 0.0 <= self.kappa <= 1.0, "in [0, 1]"),
            ("terminal_sigma_floor", self.terminal_sigma_floor >= 0.0, ">= 0"),
        ):
            if not _ok:
                raise DllmContractError(
                    f"{_name}={getattr(self, _name)!r} must be {_want}"
                )
        # top_p=None is reference-legal and means no nucleus truncation.
        _tp = c.get("top_p", 1.0)
        self.top_p = 1.0 if _tp is None else float(_tp)
        _tk = c.get("top_k", 0)
        self.top_k = 0 if _tk is None else int(_tk)
        self.top_p_site = c.get("top_p_site", "belief")
        self.self_cond = bool(c.get("self_cond", True))
        self.greedy = bool(c.get("greedy", False))
        # NFE counts the readout (arXiv 2506.10892): steps_per_block=8 is 7
        # ancestral steps + 1 readout, plus the commit.
        # Commit fusion folds the commit into block k's first forward, so a block
        # costs NFE forwards; not bitwise equal to unfused (bf16 at 2x width).
        self.commit_fusion = bool(c.get("commit_fusion", False))
        if self.commit_fusion:
            # Every forward is [block k-1 clean | block k canvas]; the generation
            # region is the last block_size positions.
            self.window_size = 2 * self.block_size
            # A request's first generation block has no predecessor, so it runs
            # narrow; both widths must be accepted, or it looks like a prefill.
            self._widths = (self.block_size, 2 * self.block_size)
            if self.prefix_bidirectional:
                # Supporting it needs a second kernel constexpr and capture
                # variant per mode.
                raise DllmContractError(
                    "commit_fusion does not support --dllm-prefix-attention "
                    "bidirectional: the fused clean half is masked causally, "
                    "so the KV it caches would differ from the standalone "
                    "commit forward it replaces. Run causal prefix attention, "
                    "or disable commit_fusion."
                )
            # Unset defers to the scheduler's cap, so it counts as unbounded.
            _mrr = getattr(config, "max_running_requests", None)
            if _mrr is None or int(_mrr) > 1:
                # _fusion_first_start is one algorithm-wide value; a second
                # request's prefill would overwrite it.
                raise DllmContractError(
                    "commit_fusion requires --max-running-requests 1: the fused "
                    "window view and readout are single-request, and the "
                    "narrow-first-block guard tracks one request's prompt end."
                )
        # Under graphs, capture keys each graph on clean_upto in {0, block_size};
        # a key mismatch silently falls back to eager.

        # ---- AR self-verification, enforcing -------------------------------
        # Keep the accepted prefix of the draft and replace the first rejected
        # token with the AR head's choice. tau=0 accepts while the draft is the
        # AR argmax; tau>0 while p_AR(draft) >= tau (not AR-distribution-exact).
        self.ar_verify = bool(c.get("ar_verify", False))
        _tau = c.get("ar_verify_tau", 0.0)
        self.ar_verify_tau = 0.0 if _tau is None else float(_tau)
        if not 0.0 <= self.ar_verify_tau < 1.0:
            raise DllmContractError(
                f"ar_verify_tau={self.ar_verify_tau} outside [0, 1). 0 selects "
                "the argmax rule; a value at or above 1 accepts nothing, so "
                "every block would commit one AR token and the diffusion draft "
                "would be pure overhead."
            )
        # 0 keeps the accept-prefix rule; k > 0 allows up to k substitute-and-
        # recheck passes, which under tau=0 converge to the AR greedy continuation.
        self.ar_repair = int(c.get("ar_repair", 0) or 0)
        if self.ar_repair < 0:
            raise DllmContractError(f"ar_repair={self.ar_repair} must be >= 0")
        if self.ar_repair > self.block_size:
            # The prefix grows by at least one per pass.
            raise DllmContractError(
                f"ar_repair={self.ar_repair} exceeds block_size="
                f"{self.block_size}. Each pass advances the accepted prefix by "
                "at least one position, so more passes than positions cannot "
                "run."
            )
        # "first" substitutes only at the first miss; "all" also at later misses
        # (re-judged next pass). Only the first miss is ever forced.
        self.ar_repair_width = c.get("ar_repair_width", "all")
        if self.ar_repair_width not in ("first", "all"):
            raise DllmContractError(
                f"ar_repair_width={self.ar_repair_width!r}; expected 'first' "
                "or 'all'."
            )
        if self.ar_repair and not self.ar_verify:
            raise DllmContractError(
                "ar_repair needs ar_verify: repair is a rule for what the "
                "verifier does after a rejection, and with verification off "
                "there are no rejections."
            )
        # AR best-of-N: N candidate blocks, each scored by the AR head on a clean
        # causal pass that writes no KV; the winner is committed whole.
        # "readout" draws N from one final belief (near-copies); "traj" runs N
        # trajectories from the block-entry canvas: N*NFE + N + 1 forwards.
        self.ar_bon_mode = str(c.get("ar_bon_mode", "traj") or "traj")
        if self.ar_bon_mode not in ("traj", "readout"):
            raise DllmContractError(
                f"ar_bon_mode={self.ar_bon_mode!r} must be 'traj' or 'readout'"
            )
        # resample_x0: a fresh x_T per trajectory (False shares one draw).
        # temp_spread: trajectory t's temperature scales by 1 + spread*(2t/(N-1) - 1).
        self.ar_bon_resample_x0 = bool(c.get("ar_bon_resample_x0", True))
        self.ar_bon_temp_spread = float(c.get("ar_bon_temp_spread", 0.0) or 0.0)
        if not 0.0 <= self.ar_bon_temp_spread < 1.0:
            raise DllmContractError(
                f"ar_bon_temp_spread={self.ar_bon_temp_spread} must be in "
                "[0, 1): at 1 the coldest trajectory samples at temperature 0"
            )
        self.ar_bon = int(c.get("ar_bon", 0) or 0)
        # Temperature for the candidate draws; 0 means "use the readout temperature".
        self.ar_bon_temp = float(c.get("ar_bon_temp", 0.0) or 0.0)
        if self.ar_bon_temp < 0.0:
            raise DllmContractError(
                f"ar_bon_temp={self.ar_bon_temp} must be >= 0 "
                "(0 = use the readout temperature)"
            )
        if self.ar_bon == 1:
            self.ar_bon = 0  # best-of-one is the ordinary path
        if self.ar_bon < 0:
            raise DllmContractError(f"ar_bon={self.ar_bon} must be >= 0")
        if self.ar_bon and self.greedy:
            # every candidate would be the same draw
            raise DllmContractError(
                "ar_bon needs a sampled readout: under greedy every candidate "
                "is the same block and best-of-N is N copies of one answer."
            )
        # AR-triggered partial redo: a block whose mean AR probability is below
        # ar_redo_tau keeps its liked positions at sigma 0 and re-noises the rest.
        self.ar_redo = int(c.get("ar_redo", 0) or 0)
        if self.ar_redo < 0:
            raise DllmContractError(f"ar_redo={self.ar_redo} must be >= 0")
        # 0 would never redo, so it is refused.
        self.ar_redo_tau = float(c.get("ar_redo_tau", 0.0) or 0.0)
        if self.ar_redo and not 0.0 < self.ar_redo_tau < 1.0:
            raise DllmContractError(
                f"ar_redo needs 0 < ar_redo_tau < 1, got {self.ar_redo_tau}. "
                "tau IS the latency knob: it sets what fraction of blocks pay "
                "for a redo, and 0 would never trigger one."
            )
        # Fraction of the block re-noised on a redo: the lowest-scoring
        # positions. The rest keep their committed tokens.
        _frac = c.get("ar_redo_frac", 0.25)
        self.ar_redo_frac = 0.25 if _frac is None else float(_frac)
        if self.ar_redo and not 0.0 < self.ar_redo_frac <= 1.0:
            raise DllmContractError(
                f"ar_redo_frac={self.ar_redo_frac} must be in (0, 1]"
            )
        # Denoise steps for a redo leg; 0 scales with ar_redo_frac.
        self.ar_redo_steps = int(c.get("ar_redo_steps", 0) or 0)
        if self.ar_redo_steps < 0:
            raise DllmContractError(
                f"ar_redo_steps={self.ar_redo_steps} must be >= 0 (0 = scale "
                "with ar_redo_frac)"
            )
        # Adaptive NFE: block k-1's commit row for block k's position 0 (_ar_carry)
        # picks ar_nfe_lo if confidence >= ar_nfe_tau, else ar_nfe_hi; batch min decides.
        self.ar_nfe_adapt = bool(c.get("ar_nfe_adapt", False))
        self.ar_nfe_lo = int(c.get("ar_nfe_lo", 0) or 0)
        self.ar_nfe_hi = int(c.get("ar_nfe_hi", 0) or 0)
        self.ar_nfe_tau = float(c.get("ar_nfe_tau", 0.0) or 0.0)
        # Confidence in [0, 1]: pmax, margin (p1 - p2), entropy (1 - H/log V), or
        # prevscore (exp mean log-prob of the previous block's committed tokens).
        self.ar_nfe_signal = str(c.get("ar_nfe_signal", "pmax") or "pmax")
        if self.ar_nfe_signal not in ("pmax", "margin", "entropy", "prevscore"):
            raise DllmContractError(
                f"ar_nfe_signal={self.ar_nfe_signal!r} must be one of "
                "pmax, margin, entropy, prevscore"
            )
        if self.ar_nfe_adapt:
            if not 0 < self.ar_nfe_lo <= self.ar_nfe_hi:
                raise DllmContractError(
                    f"ar_nfe_adapt needs 0 < ar_nfe_lo <= ar_nfe_hi, got "
                    f"lo={self.ar_nfe_lo} hi={self.ar_nfe_hi}"
                )
            if not 0.0 < self.ar_nfe_tau < 1.0:
                raise DllmContractError(
                    f"ar_nfe_adapt needs 0 < ar_nfe_tau < 1, got "
                    f"{self.ar_nfe_tau}: tau sets what fraction of blocks get "
                    "the cheap schedule, and therefore the average cost."
                )
            if self.ar_nfe_hi > self.max_steps_per_block:
                raise DllmContractError(
                    f"ar_nfe_hi={self.ar_nfe_hi} exceeds max_steps_per_block="
                    f"{self.max_steps_per_block}; raise the ceiling or lower hi"
                )
        if self.ar_nfe_adapt:
            self._ar_verify_refuse(config, feature="ar_nfe_adapt")
        # temperature=0: "argmax" at every step, or "tail" (ancestral steps, argmax
        # readout). "tail" draws batch-dependent noise, so it is not deterministic.
        self.greedy_mode = str(c.get("greedy_mode", "argmax") or "argmax")
        if self.greedy_mode not in ("argmax", "tail"):
            raise DllmContractError(
                f"greedy_mode={self.greedy_mode!r} must be 'argmax' or 'tail'"
            )
        # ar_stop: end the request where the AR head would have stopped; a detected
        # stop runs one correction forward to persist the truncated block.
        self.ar_stop = bool(c.get("ar_stop", False))
        if self.ar_stop and self.ar_verify:
            raise DllmContractError(
                "ar_stop does not compose with ar_verify: verification keeps "
                "a drafted token its rule accepts even when the head's argmax "
                "is a stop token, so termination would depend on the "
                "threshold rather than on the head. Enable one."
            )
        if self.ar_stop:
            # The verdict is read off a clean causal commit forward, and acting
            # on it commits a partial block.
            self._ar_verify_refuse(config, feature="ar_stop")
        if self.ar_bon:
            # Also: commit fusion has no standalone commit pass to score on,
            # and FDFO reports one accept length per row rather than a block.
            self._ar_verify_refuse(config, feature="ar_bon")
            if self.ar_verify or self.ar_stop:
                raise DllmContractError(
                    "ar_bon does not compose with ar_verify or ar_stop: those "
                    "rewrite or truncate the committed block, and best-of-N "
                    "commits a drafted block whole. Enable one."
                )
        if self.ar_nfe_adapt and (self.ar_verify or self.ar_stop):
            # Must sit below the assignment of self.ar_stop.
            raise DllmContractError(
                "ar_nfe_adapt does not compose with ar_verify or ar_stop: "
                "those commit a PARTIAL block, so the per-block AR score and "
                "the boundary row this reads are taken at the accepted "
                "length rather than the block end. Enable one."
            )
        if self.ar_redo:
            # The verdict is read off a clean causal commit forward, and acting
            # on it needs the pre-block conv state back.
            self._ar_verify_refuse(config, feature="ar_redo")
            if self.ar_verify or self.ar_stop:
                raise DllmContractError(
                    "ar_redo does not compose with ar_verify or ar_stop: all "
                    "three act on the commit forward's verdict and would "
                    "rewrite the same block. Enable one."
                )
            if self.ar_bon:
                raise DllmContractError(
                    "ar_redo does not compose with ar_bon: best-of-N commits "
                    "one of N whole drafts and a redo rewrites a subset of "
                    "the committed one, so the redo leg would re-enter the "
                    "candidate draw with a partial mask. Enable one."
                )
        if self.ar_verify:
            self._ar_verify_refuse(config)
        # conv slot -> [V] AR logit row at the previous block's last committed
        # position, verifying the next block's position 0.
        self._ar_carry: dict = {}
        # slot -> exp(mean AR log-prob) over the last block's committed tokens
        # (ar_nfe_signal="prevscore").
        self._ar_prevscore: dict = {}
        # Per-row committed length of the last block, for base._block_tail.
        # None means the whole window.
        self._commit_lens: Optional[List[int]] = None
        # The correction forward's output, so the block publishes the logits and
        # graph flag of the forward that ran last.
        self._corrected_out = None
        # Running accept statistics, reported by the engine on request.
        self._ar_totals = dict(blocks=0, drafted=0, accepted=0, bonus=0, passes=0)
        # Set by after_clean_prefill: the one window start at which a narrow
        # window is legitimate under fusion. None until a prefill has been seen.
        self._fusion_first_start = None
        # commit fusion's narrow view of the current block, rebuilt per block
        self._narrow_view = None
        # fixed-address inputs of the padded prompt prefill, built on first use
        self._pp_bufs = None
        # "ancestral" (sampled readout) | "greedy" (argmax on the readout only,
        # unlike greedy=True, which makes every step argmax).
        self.noise_removal = c.get("noise_removal", "ancestral")
        if self.noise_removal not in ("ancestral", "greedy"):
            raise DllmContractError(
                f"noise_removal={self.noise_removal!r} is not supported; use "
                "'ancestral' or 'greedy'. 'none' skips the readout, which the "
                "commit forward needs: it doubles as KV persistence and must see "
                "final tokens produced by a readout."
            )
        # top_p=0.0 is legal ("keep only the argmax"), so the bound is >= 0.
        if self.top_p is not None and not 0.0 <= float(self.top_p) <= 1.0:
            raise DllmContractError(
                f"top_p={self.top_p!r} is outside [0, 1]. The reference asserts "
                "this range; accepting it here decodes unfiltered (top_p>1) or "
                "kills every token (top_p<0) while the config claims nucleus "
                "sampling."
            )
        if self.top_k is not None and int(self.top_k) < 0:
            raise DllmContractError(
                f"top_k={self.top_k!r} is negative. The reference asserts "
                "top_k >= 0; here it silently disabled truncation, so a run "
                "logged as top-k sampled from the full distribution."
            )
        # Reject here: timestep_grid raises only mid-block, after admission.
        if self.schedule not in ("linear", "rho", "cosine", "geometric"):
            raise DllmContractError(
                f"schedule={self.schedule!r} is not a timestep schedule; use "
                "'linear', 'rho', 'cosine' or 'geometric'."
            )
        # Logits are divided by the temperature: zero gives NaN beliefs that the
        # argmax turns into arbitrary tokens, with no error.
        for _name in (
            ("temp_start", "temp_end") if self.temp_anneal else ("temperature",)
        ):
            if not 0.0 < getattr(self, _name) < math.inf:
                raise DllmContractError(
                    f"{_name}={getattr(self, _name)!r} must be finite and > 0; set "
                    "greedy: true for argmax decoding."
                )
        # Reject here: anneal_scalar raises only mid-block, after admission.
        if self.temp_schedule not in ("linear", "rev_log"):
            raise DllmContractError(
                f"temp_schedule={self.temp_schedule!r} is not implemented. The "
                "reference silently treats unrecognised schedules as LINEAR, so "
                "computing anything else would disagree with it without saying "
                "so. Use 'linear' or 'rev_log'."
            )
        if self.top_p_site not in ("belief", "posterior"):
            raise DllmContractError(
                f"top_p_site={self.top_p_site!r} is not recognised; the "
                "reference accepts only 'belief' or 'posterior'."
            )
        if _TF_CANVAS is not None and (not self.greedy or self.self_cond):
            raise DllmContractError(
                "teacher forcing is only meaningful with greedy=True and "
                f"self_cond=False (got greedy={self.greedy}, "
                f"self_cond={self.self_cond}). Forcing replaces the token canvas "
                "only: under sampling the pre-step-0 draw differs, and with "
                "self-conditioning each side feeds back its own belief, so the "
                "inputs are NOT identical and a per-step comparison would "
                "attribute feedback amplification to the port. Call the greedy "
                "case CANVAS-FORCED, not identical-inputs."
            )
        self.adaptive_stop = bool(c.get("adaptive_stop", False))
        self.stop_entropy = float(c.get("stop_entropy", 0.0))
        if self.stop_entropy < 0.0:
            raise DllmContractError(f"stop_entropy={self.stop_entropy!r} must be >= 0")
        self.use_float64 = bool(c.get("use_float64", True))
        if self.fdfo and self.commit_fusion:
            raise DllmContractError(
                "DuoBlock: commit fusion is single-request and synchronous "
                "(its carried half and clean_upto are baked per block); it "
                "cannot run under FDFO. Launch with --no-dllm-fdfo."
            )
        self._warned_temperature_ignored = False
        self._timer = duo_debug.BlockTimer()
        # Index of the block in flight, for traces and records; -1 before the first.
        self._blk_counter = -1
        self._blk_idx = -1
        self._model_runner = None  # captured in run()
        # (spb, phase, step) -> [2, 3*hidden] adaLN base rows for sigma levels
        # (0, -log a_t); fixed per instance.
        self._adaln_tables: dict = {}
        self._ts_cache: dict = {}  # (spb, device) -> timestep grid, fixed address
        # Keyed by (batch_size, block_size, device); never replaced, since a
        # captured block graph records their addresses.
        self._buffers: dict = {}
        # FDFO: per-shape free pool of _Buffers, one per in-flight block; an
        # aborted block never returns its buffer (reuse only).
        self._buf_pool: dict = {}
        self._buf: Optional[_Buffers] = None
        self._start_list: Optional[List[int]] = None

    def _ar_verify_refuse(self, config, feature: str = "ar_verify") -> None:
        """Refuse configs where an AR-head verdict would be read off the wrong forward."""
        if self.commit_fusion:
            raise DllmContractError(
                f"{feature} does not compose with commit_fusion: fusion has no "
                "standalone commit forward (step() finishes the block straight "
                "after readout), so there is no clean causal pass whose logits "
                "are AR next-token predictions. The verdict would be read off "
                "a bidirectional forward and mean nothing."
            )
        if self.prefix_bidirectional:
            raise DllmContractError(
                f"{feature} requires --dllm-prefix-attention causal. Under "
                "bidirectional prefix attention the commit forward lets a "
                "clean position attend its own successors, so its logits are "
                "not next-token predictions and every acceptance would be "
                "read from a distribution that has already seen the answer."
            )
        if not getattr(config, "anchored", False):
            raise DllmContractError(
                f"{feature} requires the anchored block grid: a partially "
                "accepted block leaves the next window starting at an "
                "arbitrary offset, and the legacy absolute grid asserts "
                "prefix_len %% block_size == 0 "
                "(ReqDllmMixin._update_block_offset_for_dllm)."
            )
        if getattr(config, "first_done_first_out_mode", False):
            raise DllmContractError(
                f"{feature} does not compose with FDFO: that path reports one "
                "accept length per row and a fixed block_size token list, so a "
                "per-row partial commit cannot be expressed in its result "
                "contract. Run the synchronous block loop (--no-dllm-fdfo)."
            )

    def override_block_out(self, out):
        """The correction forward's output, when step() ran one after the commit."""
        got, self._corrected_out = self._corrected_out, None
        return got if got is not None else out

    def commit_lengths(self):
        """Per-row committed length of the last block; None keeps the full slice."""
        return self._commit_lens

    def ar_verify_stats(self) -> dict:
        """Accept statistics since server start; `accepted` counts the first
        verdict's prefix, before any repair."""
        t = dict(self._ar_totals)
        b, d = t["blocks"], t["drafted"]
        t["accept_rate"] = (t["accepted"] / d) if d else 0.0
        t["passes_per_block"] = (t["passes"] / b) if b else 0.0
        return t

    def _block_sampler_mode(self, state, forward_batch) -> str:
        """step()'s sampler: the per-block greedy override, else the config."""
        greedy = state.greedy if state.greedy is not None else self.greedy
        if not greedy:
            return "sampled"
        return "greedy-tail" if self.greedy_mode == "tail" else "greedy"

    def _block_start_phase(self, st) -> str:
        # NFE 1 has no denoise steps: the block starts at its readout.
        return "denoise" if st.spb > 0 else "readout"

    def _block_baked_tag(self, states) -> tuple:
        # Everything a capture bakes in beyond the batch shape: the step count
        # and the sampler; the static temperature only counts when it is used.
        return tuple(
            (
                int(st.spb),
                None if self.temp_anneal else st.temperature,
                st.top_p,
                st.top_k,
            )
            for st in states
        )

    def _capture_unsafe_reasons(self) -> list:
        """Host syncs/reads inside step()/prepare_forward(first=False) that block
        CUDA-graph capture; first=True gates run before capture and are omitted."""
        reasons = []
        if _TRACE_PATH:
            reasons.append(
                "SGLANG_DUO_TRACE (.tolist() of canvas and beliefs per step)"
            )
        if _VERIFY_AR:
            reasons.append("SGLANG_DLLM_VERIFY_AR (.tolist() of run stats on commit)")
        if self.ar_bon:
            # _block_graph_key admits only all-greedy blocks, which ar_bon refuses.
            _why = "candidates are sampled; a replay would repeat one block's draws"
            if duo_debug.T_OUT:
                _why += ", and SGLANG_DLLM_INSTR_OUT reads the winner to the host"
            reasons.append(f"ar_bon={self.ar_bon} ({_why})")
        if self.ar_nfe_adapt:
            reasons.append(
                f"ar_nfe_adapt lo={self.ar_nfe_lo} hi={self.ar_nfe_hi} "
                "(the step count is chosen per block from a host read, so the "
                "forward sequence is not fixed at capture time)"
            )
        if self.ar_redo:
            reasons.append(
                f"ar_redo tau={self.ar_redo_tau} (host read of the reject "
                "verdict, and a data-dependent number of redo legs)"
            )
        if self.ar_stop:
            reasons.append(
                "ar_stop (host read of the stop position, plus a "
                "data-dependent correction forward)"
            )
        if self.ar_verify:
            reasons.append(
                "ar_verify (host read of the accepted length, plus a "
                "data-dependent correction forward)"
            )
        if duo_debug.STATE_DUMP:
            reasons.append("SGLANG_DLLM_STATE_DUMP (D2H state write in _finish_block)")
        if duo_debug.INSTR:
            reasons.append(
                "SGLANG_DLLM_TIMING/PROFILE (the block timer synchronizes inside "
                "step() on the final step once past SGLANG_DLLM_TIMING_SKIP)"
            )
        if _TF_PATH:
            reasons.append(
                "SGLANG_DLLM_TEACHER_CANVAS (pageable H2D of the teacher canvas)"
            )
        return reasons

    def run(self, model_runner, forward_batch, algo_states=None):
        if self._model_runner is None:
            self._assert_causality_aware_backend(model_runner)
        self._model_runner = model_runner
        return super().run(model_runner, forward_batch, algo_states)

    @staticmethod
    def _assert_causality_aware_backend(model_runner) -> None:
        """Gate on the resolved backend (ROCm defaults to aiter): only triton honors
        ``dllm_causal_override`` for every DuoBlock mask."""
        if envs.SGLANG_DLLM_DEBUG_ALLOW_NONTRITON_ATTENTION.get():
            logger.warning(
                "DuoBlock backend guard lifted by "
                "SGLANG_DLLM_DEBUG_ALLOW_NONTRITON_ATTENTION; resolved "
                "backends: prefill=%r decode=%r",
                getattr(model_runner, "prefill_attention_backend_str", None),
                getattr(model_runner, "decode_attention_backend_str", None),
            )
            return
        for attr in ("prefill_attention_backend_str", "decode_attention_backend_str"):
            be = getattr(model_runner, attr, None)
            if be not in (None, "triton"):
                raise DllmContractError(
                    f"DuoBlock requires the triton attention backend; the "
                    f"engine resolved {attr}={be!r}. That backend ignores the "
                    "per-forward causality override and would denoise causally."
                )

    def max_steps(self, block_size: int) -> int:
        # steps + readout + commit + the final done-on-entry step() call. A hard
        # stop (exhaustion commits no KV), so it counts every best-of-N and redo leg.
        _legs = self.ar_bon if self.ar_bon_mode == "traj" else 1
        # ...plus the redo legs, each a shortened denoise run + readout + commit.
        _redo = self.ar_redo * ((self.ar_redo_steps or self.max_steps_per_block) + 2)
        return (self.max_steps_per_block + 1) * max(1, _legs) + 2 + self.ar_bon + _redo

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        return [_BatchState(row=k) for k in range(forward_batch.batch_size)]

    @staticmethod
    def _uniform_state(states: List[Any]) -> _BatchState:
        """Row 0, after checking that every row agrees with it."""
        st = states[0]
        for k, s in enumerate(states):
            if k == 0:
                continue
            for f in _BatchState._UNIFORM_FIELDS:
                if getattr(s, f) != getattr(st, f):
                    raise DllmContractError(
                        "DuoBlock: rows of one batch disagree on their state "
                        f"({f}: row 0 = {getattr(st, f)!r}, row {k} = "
                        f"{getattr(s, f)!r}). Mixed-phase batches need per-row "
                        "conditioning, which DuoBlock does not implement; the "
                        "scheduler's _dllm_batch_uniform should never have "
                        "assembled this batch."
                    )
            for f in _BatchState._SHARED_FIELDS:
                if getattr(s, f) is not getattr(st, f):
                    raise DllmContractError(
                        "DuoBlock: rows of one batch do not share their "
                        f"block scratch ({f} differs between row 0 and row "
                        f"{k}); they were prepared as different batches."
                    )
        return st

    @staticmethod
    def _fanout(states: List[Any]) -> None:
        """Copy row 0's state onto every other row (lockstep batch)."""
        st = states[0]
        for s in states[1:]:
            if s is st:
                continue
            for f in _BatchState.__slots__:
                if f in _BatchState._ROW_FIELDS:
                    continue
                setattr(s, f, getattr(st, f))

    def _broadcast_state(self, states: List[Any]) -> None:
        # base-loop hook: the block-graph loop writes states[0] directly
        self._fanout(states)

    def _prepare_is_uniform(self) -> bool:
        return True

    def _bind(self, st: _BatchState) -> None:
        """Point the instance mirrors at THIS batch's block context."""
        ctx = st.ctx
        if ctx is None:
            return
        self._buf = ctx.buf
        self._ts = ctx.ts
        self._start_list = ctx.start_list
        self._blk_idx = ctx.blk_idx

    def _relayout(self, forward_batch: ForwardBatch, states: List[Any]) -> None:
        """Gather the block scratch for a cohort that shrank or was reordered
        mid-block (FDFO only); rows are independent, so each continues as it was."""
        st = states[0]
        old = st.ctx
        B = forward_batch.batch_size
        rows = [s.row for s in states]
        if st.bon is not None or st.bon_x0 is not None:
            # Unreachable (ar_bon refuses FDFO); bon tensors are not gathered.
            raise DllmContractError(
                "DuoBlock: cannot re-lay out a block with best-of-N state in "
                f"flight (candidates={0 if st.bon is None else len(st.bon)}, "
                f"trajectory={st.bon_t}). ar_bon refuses FDFO precisely so "
                "this cohort change cannot happen mid-block; gather bon/"
                "bon_scores/bon_x0 by row here before lifting that."
            )
        if old.buf is None or len(set(rows)) != len(rows) or max(rows) >= old.buf.b:
            raise DllmContractError(
                "DuoBlock: cannot re-lay out the block scratch -- the rows in "
                f"front of us ({rows}) are not a subset of the cohort that "
                f"started the block (size {old.buf.b if old.buf else None})."
            )
        dev = old.buf.sigma.device
        W = old.buf.lb
        idx = torch.tensor(rows, dtype=torch.long, device=dev)
        key = (B, W, str(dev))
        pool = self._buf_pool.setdefault(key, [])
        buf = pool.pop() if pool else _Buffers(B, W, dev)
        duo_debug.sync("relayout")
        buf.upd.copy_(old.buf.upd.index_select(0, idx))
        buf.upd_f.copy_(buf.upd.to(buf.upd_f.dtype))
        buf.upd_idx.copy_(buf.upd.reshape(-1).to(torch.long))
        if st.sc is not None:
            sc = buf.sc_buffer(st.sc.shape[-1])
            sc.copy_(st.sc.index_select(0, idx))
            st.sc = sc
        buf.x_in.copy_(old.buf.x_in.index_select(0, idx))
        if old.buf.sc_pos is not None:
            buf.sc_pos = (
                old.buf.sc_pos.view(old.buf.b, W).index_select(0, idx).reshape(-1)
            )
        if st.conv_snapshot:
            # (pool tensor, slot indices [B], saved [B, dim, k-1] or
            # [layers, B, dim, k-1]) -> keep the survivors'
            snap = []
            for conv_state, sidx, saved in st.conv_snapshot:
                ridx = idx.to(sidx.device)
                row_dim = 1 if saved.dim() == 4 else 0
                snap.append(
                    (
                        conv_state,
                        sidx.index_select(0, ridx),
                        saved.index_select(row_dim, ridx.to(saved.device)),
                    )
                )
            st.conv_snapshot = snap
        st.upd = buf.upd
        st.x_in = buf.x_in
        st.ctx = _BlockCtx(
            buf,
            old.ts,
            [old.start_list[r] for r in rows] if old.start_list is not None else None,
            old.blk_idx,
        )
        # the old scratch is not returned to the pool: dropped rows may be
        # aborted, and a buffer shared by two cohorts is the bug it prevents
        for k, s in enumerate(states):
            s.row = k
            s.ctx = st.ctx
            s.upd = st.upd
            s.x_in = st.x_in
            s.sc = st.sc
            s.conv_snapshot = st.conv_snapshot

    # ---- hooks ---------------------------------------------------------------

    def prepare_forward(
        self, forward_batch: ForwardBatch, states: List[Any], first: bool
    ) -> None:
        """Before every forward: canvas init, conv snapshot/restore, and the
        per-forward conditioning (sigma / selfcond / causality)."""
        st: _BatchState = self._uniform_state(states)
        if first:
            # a new block: fresh scratch, bound once buffers and schedule resolve
            st.ctx = None
        else:
            if st.ctx is None:
                raise DllmContractError(
                    "DuoBlock: prepare_forward(first=False) on rows that never "
                    "started a block (no block context). Under FDFO this is a "
                    "carried state whose first forward ran as a different batch."
                )
            if st.ctx.buf.b != forward_batch.batch_size or any(
                s.row != k for k, s in enumerate(states)
            ):
                # The cohort changed mid-block (abort, or reordered survivors).
                # Scratch is positional, so re-lay it out rather than refuse.
                self._relayout(forward_batch, states)
            self._bind(st)
        try:
            self._prepare_forward_uniform(forward_batch, st, first)
        finally:
            self._fanout(states)

    def _prepare_forward_uniform(
        self, forward_batch: ForwardBatch, st: _BatchState, first: bool
    ) -> None:
        B = forward_batch.batch_size
        Lb = self.block_size
        W = self.forward_width(forward_batch) or self.window_size
        # Clean carry-over half under commit fusion; 0 otherwise. The canvas is
        # always the trailing Lb positions.
        C = W - Lb
        dev = forward_batch.input_ids.device
        # One ForwardBatch serves every forward of a block, so reset per forward.
        forward_batch.dllm_graph_replayed = False
        # The clean-prompt path below sets it True.
        forward_batch.dllm_prompt_prefill = False
        # scheduler short-row hazard (chunked/partial staging): fail with a
        # named error instead of view()'s shape complaint or silent corruption
        if forward_batch.input_ids.numel() != B * W:
            raise DllmContractError(
                f"DuoBlock requires full {W}-token windows per request; got "
                f"{forward_batch.input_ids.numel()} tokens for batch {B}. A "
                "partial dLLM staging chunk reached the algorithm."
            )

        if first:
            ids = forward_batch.input_ids.view(B, W)
            # Mask spans the window (drives upd); committed counts use the canvas.
            ph = ids == self.mask_id
            ph_canvas = ph[:, -Lb:]
            # The block's one D2H sync. Canvas-relative (base slices [-Lb:]); over
            # the full window a fused [clean | masks] row would read as complete.
            self._start_list = (~ph_canvas).sum(dim=1).tolist()
            # A completed row in a fresh batch would be re-noised by the reverse step.
            row_fresh = [c < Lb for c in self._start_list]
            if any(row_fresh) and not all(row_fresh):
                raise DllmContractError(
                    "DuoBlock v1 requires a batch that is uniformly fresh or "
                    "uniformly complete; got a mix (committed counts: "
                    f"{self._start_list}). The uniform-state reverse step "
                    "would corrupt the completed rows."
                )
            if not any(row_fresh):
                # Block complete on entry (a clean prompt): a prompt prefill at
                # sigma 0, never a block-graph replay.
                st.phase = "commit"
                forward_batch.dllm_prompt_prefill = True
                # Bypasses base._run_sync's prefill branch; a recycled slot keeps a stale carry.
                self.on_clean_prefill(forward_batch)
                forward_batch.dllm_sigma = torch.zeros(
                    B * W, dtype=torch.float32, device=dev
                )
                forward_batch.dllm_selfcond = None
                # Same clean-context rule as base._run_sync.
                forward_batch.dllm_causal_override = (
                    False if self.prefix_bidirectional else None
                )
                return
            st.upd = ph.clone()
            self._commit_lens = None
            # temperature=0 makes the canvas init deterministic too.
            if self._resolve_greedy(forward_batch):
                # oracle _block_zero_init: deterministic canvas for parity runs
                init = torch.zeros_like(ids)
            else:
                # Draw for the canvas only, so the RNG stream matches the unfused path.
                init = ids.clone()
                init[:, W - Lb :] = torch.randint(
                    0, self.real_vocab_size, (B, Lb), device=dev, generator=st.generator
                )
            forward_batch.input_ids.copy_(torch.where(ph, init, ids).view(-1))
            # The snapshot is taken during the first forward into this list; runners
            # pass the model a shallow replace()-copy, so only a shared container survives.
            if not self._conv_restore_each_forward() and not self._ar_may_correct():
                # the fused conv kernel does not write state on denoise forwards
                forward_batch.dllm_conv_snapshot = None
                forward_batch.dllm_conv_capture = False
            else:
                forward_batch.dllm_conv_snapshot = []
                forward_batch.dllm_conv_capture = True
            # Advanced unconditionally; per-(block, step) traces need it.
            self._blk_counter += 1
            self._blk_idx = self._blk_counter
            if duo_debug.INSTR:
                self._timer.begin()
            st.spb = self._resolve_steps_per_block(forward_batch)
            # A request that names its NFE keeps it; adaptation only picks the
            # server default's replacement.
            if self.ar_nfe_adapt and not any(forward_batch.dllm_steps_per_block or ()):
                st.spb = self._ar_adaptive_spb(forward_batch, st.spb)
            st.greedy = self._resolve_greedy(forward_batch)
            st.temperature, st.top_p, st.top_k = self._resolve_sampling(forward_batch)
            if st.spb <= 0:
                # nfe=1 is readout-only; the denoise arm would index _ts[i + 1]
                # on a length-1 grid.
                st.phase = "readout"
            # One tensor per step count: the block graph records its address.
            _tsk = (int(st.spb), str(dev))
            _ts = self._ts_cache.get(_tsk)
            if _ts is None:
                _ts = duo_math.timestep_grid(
                    st.spb, self.eps, self.schedule, self.rho, dev
                )
                self._ts_cache[_tsk] = _ts
            self._ts = _ts
            key = (B, W, str(dev))
            if self.fdfo:
                pool = self._buf_pool.setdefault(key, [])
                buf = pool.pop() if pool else _Buffers(B, W, dev)
            else:
                buf = self._buffers.get(key)
                if buf is None:
                    buf = _Buffers(B, W, dev)
                    self._buffers[key] = buf
            self._buf = buf
            # the block's scratch travels with the rows from here on
            st.ctx = _BlockCtx(buf, _ts, self._start_list, self._blk_idx)
            st.bon = st.bon_scores = st.bon_x0 = None
            st.bon_i = st.bon_t = st.bon_steps = 0
            st.redo_left, st.spb0 = self.ar_redo, None
            if (
                self.ar_bon
                and self.ar_bon_mode == "traj"
                and not self.commit_fusion
                and not st.greedy
                and st.phase in ("denoise", "readout")
            ):
                # x_T for every trajectory; each restart copies it back.
                st.bon = []
                st.bon_x0 = forward_batch.input_ids.view(B, W).clone()
            # upd is constant for the block; materialize its float form once
            duo_debug.sync("upd")
            self._buf.upd.copy_(st.upd)
            st.upd = self._buf.upd
            self._buf.upd_f.copy_(st.upd.to(self._buf.upd_f.dtype))
            self._buf.upd_idx.copy_(st.upd.reshape(-1).to(torch.long))
            # A wide fused window (C > 0) needs no conv capture or restore:
            # nothing advances the carried conv state until the readout.
            if self.commit_fusion and C == 0:
                # Narrow is legitimate only for a request's first generation
                # block; later, it means block k-1's clean KV was never written.
                _ws = self._window_start(forward_batch)
                _first = getattr(self, "_fusion_first_start", None)
                if _ws is None or _first is None or _ws != _first:
                    # A missing window start refuses too; "allow" is a silent wrong answer.
                    raise DllmContractError(
                        "commit fusion: window is block_size at window_start="
                        f"{_ws}, but the only narrow window fusion permits is "
                        f"the first generation block at {_first}. The fused "
                        "2*block_size window never materialised, so the commit "
                        "was skipped WITHOUT carrying the previous block and "
                        "its clean KV was never written -- measured effect: "
                        "52% of tokens changed, top-2 margins to 9.25, i.e. "
                        "nothing to do with numerics. Scheduler staging must "
                        "hold the prefix back one block under fusion."
                    )
        else:
            forward_batch.dllm_conv_capture = False
            if self._conv_restore_each_forward():
                self._adopt_conv_snapshot(forward_batch, st)
                self._restore_conv(st)
        # per-forward conditioning, written in place into the block buffers.
        # sigma is nonzero only on denoised positions (oracle _block_sigma_tok).
        buf = self._buf
        sigma_2d = buf.sigma.view(B, W)
        if st.phase == "denoise":
            a_t = duo_math.loglinear_alpha(self._ts[st.i], self.eps)
            duo_debug.sync("sigma")
            sigma_2d.copy_(buf.upd_f).mul_(-a_t.clamp_min(1e-30).log())
        elif st.phase == "readout":
            a_t = duo_math.readout_alpha(
                self._ts[-1], self.eps, self.terminal_sigma_floor
            )
            duo_debug.sync("sigma")
            sigma_2d.copy_(buf.upd_f).mul_(-a_t.clamp_min(1e-30).log())
        else:  # commit: clean re-encode
            a_t = None
            buf.sigma.zero_()
        forward_batch.dllm_sigma = buf.sigma
        self._set_adaln_base(forward_batch, st, buf, a_t, B * W)
        if self.commit_fusion:
            # No self-conditioning before the block start; block k-1 sits at
            # [0, C) inside the window, so gate it (fixed address).
            if buf.sc_pos is None:
                g = torch.zeros(B, W, dtype=torch.float32, device=dev)
                g[:, C:] = 1.0
                buf.sc_pos = g.view(B * W)
            forward_batch.dllm_selfcond_pos_mask = buf.sc_pos
        else:
            forward_batch.dllm_selfcond_pos_mask = None
        forward_batch.dllm_selfcond = (
            st.sc.view(B * W, -1)
            if (st.sc is not None and st.phase in ("denoise", "readout"))
            else None
        )
        # commit: causal under tcp. A best-of-N scoring pass is a commit-like
        # forward that persists nothing.
        _scoring = st.phase == "bon_score"
        _causal = (st.phase == "commit" or _scoring) and not self.prefix_bidirectional
        forward_batch.dllm_causal_override = _causal
        # Only the commit (and ar_verify correction) forward's K/V are read later.
        forward_batch.dllm_save_kv = st.phase == "commit"
        if self.commit_fusion:
            # Only the wide first forward persists: conv stops at C; block k's
            # init-canvas K/V is overwritten by block k+1's first forward before any read.
            _fwd0 = first or self._graph_forward0
            _carry = C > 0 and _fwd0
            forward_batch.dllm_save_kv = _carry
            forward_batch.dllm_conv_state_at = (
                buf.fusion_state_at(B, C, dev) if _carry else None
            )
            if _fwd0:
                self._narrow_view = None
        if self.commit_fusion and first and not self._conv_state_static():
            # Checked here, not in __init__: _conv_state_static() needs _model_runner.
            raise DllmContractError(
                "commit fusion requires a static conv state (fused conv "
                "kernels). Here the conv is unfused, so the per-forward "
                "restore undoes the conv advance that the carried half's KV "
                "write depends on, and the carry would be half-written."
            )
        if _scoring:
            # Causal variants are captured only conditionally; do not inherit a plan.
            forward_batch.attn_metadata_ready = False
            forward_batch.dllm_graph_replayed = False
            # The unfused conv writes conv_states regardless of dllm_save_kv.
            if self._conv_restore_each_forward():
                self._adopt_conv_snapshot(forward_batch, st)
                self._restore_conv(st)
            # write the candidate under test into the canvas; the carried head
            # is untouched.
            _cand = st.bon[st.bon_i]
            _Cr = W - _cand.shape[1]
            if _Cr:
                forward_batch.input_ids.view(B, W)[:, _Cr:].copy_(_cand)
            else:
                forward_batch.input_ids.copy_(_cand.view(-1))
        if _TENSOR_CAUSALITY:
            # Per-request causality as graph-slot data; batch-uniform for now, so
            # it must stay token-identical to the bool.
            buf.causal_rows.fill_(int(_causal))
            forward_batch.dllm_causal_rows = buf.causal_rows
        else:
            forward_batch.dllm_causal_rows = None
        # Rows below clean_upto are causal (the carried block), rows at or above
        # are bidirectional. C is 0 whenever the window is narrow.
        forward_batch.dllm_clean_upto = C
        forward_batch.dllm_phase = st.phase
        duo_debug.sync("x_in")
        buf.x_in.copy_(forward_batch.input_ids.view(B, W))
        st.x_in = buf.x_in
        if duo_debug.INSTR:
            self._timer.split("prep")

    def _denoise_steps(self, nfe: int) -> int:
        """Partial ancestral steps, given an NFE budget that includes readout."""
        # nfe=1 is a legitimate readout-only 1-NFE sampler.
        return max(nfe - 1, 0)

    def _resolve_steps_per_block(self, forward_batch: ForwardBatch) -> int:
        """Effective NFE: the per-request override, else the server default."""
        per_req = forward_batch.dllm_steps_per_block
        if not per_req:
            return self._denoise_steps(self.steps_per_block)
        vals = {v if v is not None else self.steps_per_block for v in per_req}
        if len(vals) != 1:
            raise DllmContractError(
                "a dLLM batch mixes per-request dllm_steps_per_block "
                f"({sorted(vals)}); one block runs one shared noise schedule. "
                "Send requests with the same NFE together, or serve with "
                "--max-running-requests 1."
            )
        spb = vals.pop()
        if not 1 <= spb <= self.max_steps_per_block:
            raise DllmContractError(
                f"dllm_steps_per_block={spb} outside [1, "
                f"{self.max_steps_per_block}]; raise max_steps_per_block in "
                "the algorithm config to allow more."
            )
        return self._denoise_steps(spb)

    def _resolve_greedy(self, forward_batch: ForwardBatch) -> bool:
        """Argmax for this block (honours `temperature: 0`), else the server default."""
        per_req = forward_batch.dllm_greedy
        if not per_req:
            return self.greedy
        vals = set(per_req)
        if len(vals) != 1:
            raise DllmContractError(
                "a dLLM batch mixes greedy and sampled requests "
                "(temperature=0 / top_k=1 alongside temperature>0); one block "
                "runs one shared reverse step. Send them separately, or serve "
                "with --max-running-requests 1."
            )
        # SamplingParams defaults temperature to 1.0; reading that as "sample"
        # would override the configured anneal for every client.
        return vals.pop() or self.greedy

    def _resolve_sampling(self, forward_batch: ForwardBatch) -> tuple:
        """(temperature, top_p, top_k) for this block: the request's value where it
        set one, else the server config's. A request temperature applies only to a
        static-temperature config; an annealing one keeps its bounds."""
        per_req = forward_batch.dllm_sampling
        if not per_req:
            return self.temperature, self.top_p, self.top_k
        if len(set(per_req)) != 1:
            raise DllmContractError(
                "a dLLM batch mixes per-request temperature/top_p/top_k "
                f"({sorted(set(per_req), key=str)}); one block runs one sampler. "
                "Send requests with the same settings together, or serve with "
                "--max-running-requests 1."
            )
        temperature, top_p, top_k = per_req[0]
        if temperature is not None and self.temp_anneal:
            if not self._warned_temperature_ignored:
                self._warned_temperature_ignored = True
                logger.warning(
                    "dLLM: a request temperature is ignored because the "
                    "algorithm config anneals between temp_start and temp_end; "
                    "drop the bounds from the config to use a static "
                    "temperature."
                )
            temperature = None
        return (
            self.temperature if temperature is None else temperature,
            self.top_p if top_p is None else top_p,
            self.top_k if top_k is None else top_k,
        )

    def block_start_list(self, forward_batch: ForwardBatch) -> List[int]:
        # measured in prepare_forward BEFORE the canvas was overwritten with
        # the DUO prior -- probing now would read random tokens
        if self._start_list is None:
            raise DllmContractError(
                "block_start_list called before prepare_forward measured the " "canvas"
            )
        return self._start_list

    # ---- the reverse-step state machine ---------------------------------------

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        st: _BatchState = self._uniform_state(states)
        if st.phase == "done":
            return [True] * forward_batch.batch_size  # safety: already reported
        if st.ctx is None:
            raise DllmContractError(
                "DuoBlock: step() on rows without a block context; "
                "prepare_forward(first=True) never ran for this batch."
            )
        self._bind(st)
        try:
            return self._step_uniform(forward_batch, full_logits, st)
        finally:
            self._fanout(states)

    def _step_uniform(
        self, forward_batch: ForwardBatch, full_logits: torch.Tensor, st: _BatchState
    ) -> List[bool]:
        B = forward_batch.batch_size
        Lb = self.block_size
        V = self.real_vocab_size

        # Adopt on the same ForwardBatch: under FDFO the next prepare_forward
        # sees a rebuilt batch whose snapshot list is empty.
        if self._conv_restore_each_forward() and st.conv_snapshot is None:
            self._adopt_conv_snapshot(forward_batch, st)

        if duo_debug.INSTR:
            self._timer.split("fwd")
        # Views span the whole window; under fusion the leading W - Lb positions
        # are block k-1's clean tokens.
        W = self.forward_width(forward_batch) or self.window_size
        logits = full_logits.view(B, -1, full_logits.shape[-1])
        # A narrow forward inside a fused block returns logits for the current
        # half only; offset by C so `logits[:, C:]` names the same positions.
        _lo = W - logits.shape[1]
        if _lo not in (0, W - Lb):
            raise DllmContractError(
                f"logits cover {logits.shape[1]} positions of a {W}-wide window; "
                f"expected {W} or the {Lb}-wide current half"
            )
        x = forward_batch.input_ids.view(B, W)

        if _TRACE_PATH and st.phase != "done":
            # top-2 logits per block position, to judge whether a divergence
            # exceeds bf16 noise.
            t2 = torch.topk(logits.float(), 2, dim=-1)
            _TRACE.append(
                dict(
                    phase=st.phase,
                    i=st.i,
                    sigma=float(forward_batch.dllm_sigma.max()),
                    sc=forward_batch.dllm_selfcond is not None,
                    graph=bool(forward_batch.dllm_graph_replayed),
                    x_in=x.tolist(),
                    top2_ids=t2.indices.tolist(),
                    top2_vals=t2.values.tolist(),
                )
            )

        if st.phase == "commit" and _VERIFY_AR:
            # logits[:, i] predicts x[:, i+1]; position 0's verifier is in the
            # previous window.
            with torch.no_grad():
                ar_pred = logits[:, :-1, :V].argmax(-1)  # predicts x[:, 1:]
                agree = ar_pred == x[:, 1:]
                # Per-row: the accepted prefix is inherently per-request.
                ai = agree.long()
                # run: longest agreeing prefix; total: agreeing positions anywhere.
                run_pr = ai.cumprod(dim=1).sum(dim=1)  # [B]
                tot_pr = ai.sum(dim=1)  # [B]
                runs = run_pr.tolist()
                totals = tot_pr.tolist()
                run = float(run_pr.float().mean())
                total = float(tot_pr.float().mean())
                nverif = int(agree.shape[1])
                # Also p_AR of the drafted token, its AR rank, and top-k hit rates.
                lp = torch.log_softmax(logits[:, :-1, :V].float(), dim=-1)
                tgt = x[:, 1:].unsqueeze(-1)
                p_tok = lp.gather(-1, tgt).squeeze(-1).exp()
                # rank = how many tokens the AR head scores strictly higher
                rank = (lp > lp.gather(-1, tgt)).sum(-1)
                # Threshold rule, not min(1, p_AR/q_draft): the readout logits
                # are not a proposal density speculative sampling could use.
                taus = (0.1, 0.3, 0.5)
                tau_pass, tau_run = [], []
                for _t in taus:
                    ok = (p_tok >= _t).long()
                    # pass rate is a batch average over tokens; the accepted
                    # prefix is per-row, then averaged across rows
                    tau_pass.append(float(ok.float().mean()))
                    tau_run.append(float(ok.cumprod(dim=1).sum(dim=1).float().mean()))
                # Within-block p_AR profile per position (mean over rows).
                p_by_pos = p_tok.mean(dim=0).tolist()
                p_mean = float(p_tok.mean())
                p_med = float(p_tok.median())
                r_med = float(rank.float().median())
                top1 = float((rank == 0).float().mean())
                top5 = float((rank < 5).float().mean())
                top10 = float((rank < 10).float().mean())
                # Degeneracy probe (repetition is trivially AR-predictable), per row.
                dis_pr, run_tok_pr = [], []
                for _b in range(int(x.shape[0])):
                    _r = x[_b]
                    dis_pr.append(int(_r.unique().numel()))
                    _best = _cur = 1
                    for _k in range(1, int(_r.numel())):
                        _cur = _cur + 1 if bool(_r[_k] == _r[_k - 1]) else 1
                        _best = max(_best, _cur)
                    run_tok_pr.append(_best)
                distinct = float(sum(dis_pr)) / len(dis_pr)
                _best = float(sum(run_tok_pr)) / len(run_tok_pr)
            _ar_rec = dict(
                kind="ar",
                block=len(_AR_STATS),
                accepted=run,
                agree_total=total,
                verifiable=nverif,
                nfe=int(st.spb or 0),
                distinct=distinct,
                max_run=_best,
                p_ar_mean=p_mean,
                p_ar_med=p_med,
                rank_med=r_med,
                top1=top1,
                top5=top5,
                top10=top10,
                tau_pass=tau_pass,
                tau_run=tau_run,
                taus=list(taus),
                p_by_pos=p_by_pos,
                batch=int(x.shape[0]),
                # Per-row carried head: tells a fresh window from a carried one.
                starts=list(self._start_list or [0] * int(x.shape[0])),
                runs_per_row=runs,
                totals_per_row=totals,
                distinct_per_row=dis_pr,
                prefill=bool(forward_batch.dllm_prompt_prefill),
            )
            _AR_STATS.append(_ar_rec)
            duo_debug.instr_emit(_ar_rec)
            # A block_size clean prompt reaches here as a prefill; label it apart.
            logger.warning(
                "dllm-ar-verify: kind=%s B=%d accepted_run=%.1f/%d "
                "agree_total=%.1f/%d distinct=%.1f/%d max_tok_run=%.1f p_ar=%.3f rank_med=%.0f "
                "top1=%.2f top5=%.2f top10=%.2f pass@0.3=%.2f run@0.3=%.1f "
                "(block_size=%d, nfe=%s)",
                ("prefill" if forward_batch.dllm_prompt_prefill else "draft"),
                int(x.shape[0]),
                run,
                nverif,
                total,
                nverif,
                distinct,
                Lb,
                _best,
                p_mean,
                r_med,
                top1,
                top5,
                top10,
                tau_pass[1],
                tau_run[1],
                Lb,
                st.spb,
            )

        if st.phase == "bon_score":
            # Logit at p scores the token at p+1, so positions 1..Lb-1 are scored.
            cand = st.bon[st.bon_i]
            Cr = W - cand.shape[1]
            lg = logits[:, Cr:, :V]
            with torch.no_grad():
                lp = torch.log_softmax(lg[:, :-1].float(), dim=-1)
                tgt = cand[:, 1:].unsqueeze(-1)
                # Mean, not sum, so a stop-shortened candidate is not favoured.
                st.bon_scores.append(lp.gather(-1, tgt).squeeze(-1).mean(dim=1).clone())
            st.bon_i += 1
            if st.bon_i < len(st.bon):
                return [False] * B  # score the next candidate
            # every candidate scored: commit the winner per row
            sc = torch.stack(st.bon_scores, dim=0)  # [N, B]
            win = sc.argmax(dim=0)  # [B]
            cands = torch.stack(st.bon, dim=0)  # [N, B, Lb]
            best = cands[win, torch.arange(B, device=cands.device)]
            if Cr:
                x[:, Cr:] = best
                forward_batch.input_ids.copy_(x.view(-1))
            else:
                forward_batch.input_ids.copy_(best.view(-1))
            # The record syncs, so it is built only when a sink is open.
            if duo_debug.T_OUT:
                duo_debug.instr_emit(
                    dict(
                        kind="ar_bon",
                        block=self._blk_idx,
                        n=len(st.bon),
                        winner=[int(v) for v in win.tolist()],
                        mode=self.ar_bon_mode,
                        spread=float((sc.max(0).values - sc.min(0).values).mean()),
                        nfe=int(st.spb or 0),
                    )
                )
            st.bon = st.bon_scores = st.bon_x0 = None
            st.bon_i = st.bon_t = 0
            st.phase = "commit"
            if duo_debug.INSTR:
                self._timer.split("samp")
            return [False] * B

        if st.phase == "commit":
            # The commit pass persisted the block's KV: report done now.
            if self.ar_stop:
                # Refused together with ar_verify at construction.
                self._ar_stop_only(forward_batch, st, logits, x, B, W, Lb, V)
            if self.ar_verify:
                self._ar_enforce(forward_batch, st, logits, x, B, W, Lb, V)
            if self.ar_nfe_adapt:
                # ar_verify and ar_stop are refused with this, so nothing else
                # maintains the carry or the prevscore.
                _lgb = logits[:, :, :V]
                _slx = getattr(forward_batch, "req_pool_indices", None)
                if _slx is not None:
                    with torch.no_grad():
                        # the block's own AR score (prevscore), same left shift
                        # as bon_score and _ar_redo_maybe
                        _cr = W - Lb
                        _lp = torch.log_softmax(_lgb[:, _cr:][:, :-1], dim=-1)
                        _tg = (x[:, _cr:] if _cr else x)[:, 1:].unsqueeze(-1)
                        _ms = _lp.gather(-1, _tg).squeeze(-1).mean(dim=1).exp()
                    for _b, _s in enumerate(_slx.tolist()):
                        self._ar_carry[int(_s)] = _lgb[_b, W - 1].detach().clone()
                        self._ar_prevscore[int(_s)] = float(_ms[_b])
            if self.ar_redo and st.redo_left > 0 and not st.greedy:
                # The redo's commit overwrites this KV (block k+1 has not run);
                # greedy blocks would re-draw identical tokens.
                if self._ar_redo_maybe(forward_batch, st, logits, x, B, W, Lb, V):
                    return [False] * B
            self._finish_block(forward_batch, st, x)
            return [True] * B

        if st.phase == "readout":
            a_t = duo_math.readout_alpha(
                self._ts[-1], self.eps, self.terminal_sigma_floor
            )
            a_t = a_t.reshape(1, 1, 1).expand(B, 1, 1)
            # st.greedy: a request's temperature=0 wins over an ancestral config.
            _greedy = st.greedy if st.greedy is not None else self.greedy
            temp = self.temp_end if self.temp_anneal else st.temperature
            # The temperature fan reaches the readout too: at nfe=1 it is the only draw.
            temp = temp * self._bon_temp_mul(st)
            # Canvas only, so the RNG stream matches the unfused path.
            Cr = W - Lb
            if Cr:
                lgr, xr, updr = logits[:, Cr - _lo :], x[:, Cr:], st.upd[:, Cr:]
            else:
                lgr, xr, updr = logits, x, st.upd
            # bind the belief: the readout's log_x0 is what the trace compares
            x_next, log_x0_r = duo_math.duo_reverse_from_logits(
                lgr,
                xr,
                a_t,
                torch.ones_like(a_t),
                V,
                temp,
                self.use_float64,
                greedy=(_greedy or self.noise_removal == "greedy"),
                kappa=self.kappa,
                top_p=st.top_p,
                top_k=st.top_k,
                top_p_site=self.top_p_site,
                generator=st.generator,
            )
            _post_r = (
                torch.where(updr, x_next, xr) if Cr else torch.where(st.upd, x_next, x)
            )
            # Draw candidates before the canvas write: `xr` views input_ids, so
            # later draws would condition on the readout result instead of x_t.
            _bon_cands = None
            if (
                self.ar_bon
                and self.ar_bon_mode == "readout"
                and not self.commit_fusion
                and not _greedy
            ):
                _bon_cands = [_post_r.clone()]
                # candidate 0 is the ordinary readout draw
                _btemp = self.ar_bon_temp or temp
                for _ in range(self.ar_bon - 1):
                    _alt, _ = duo_math.duo_reverse_from_logits(
                        lgr,
                        xr,
                        a_t,
                        torch.ones_like(a_t),
                        V,
                        _btemp,
                        self.use_float64,
                        greedy=(_greedy or self.noise_removal == "greedy"),
                        kappa=self.kappa,
                        top_p=st.top_p,
                        top_k=st.top_k,
                        top_p_site=self.top_p_site,
                        generator=st.generator,
                    )
                    _bon_cands.append(
                        (
                            torch.where(updr, _alt, xr)
                            if Cr
                            else torch.where(st.upd, _alt, x)
                        ).clone()
                    )
            if _TRACE_PATH:
                _b2r = torch.topk(log_x0_r.float(), 2, dim=-1)
                _TRACE.append(
                    dict(
                        phase="belief",
                        readout=True,
                        i=int(st.i),
                        block=self._blk_idx,
                        temp=float(temp),
                        b2_ids=_b2r.indices.tolist(),
                        b2_vals=_b2r.values.tolist(),
                        canvas=_post_r.tolist(),
                        forced=_TF_CANVAS is not None,
                    )
                )
            _forced_r = None
            if _TF_CANVAS is not None:
                _blk_r = self._blk_idx
                _key_r = (_blk_r, int(st.i))
                if _key_r not in _TF_CANVAS:
                    raise DllmContractError(
                        f"teacher forcing has no canvas for the READOUT at block "
                        f"{_blk_r} step {int(st.i)}. Leaving it unforced would "
                        "un-align the committed prefix that every later block "
                        "attends, so later blocks would be compared against a "
                        "different history."
                    )
                _forced_r = torch.as_tensor(
                    _TF_CANVAS[_key_r], device=x.device, dtype=x.dtype
                )
            if Cr:
                nxt = x.clone()
                nxt[:, Cr:] = _forced_r if _forced_r is not None else _post_r
                forward_batch.input_ids.copy_(nxt.view(-1))
            else:
                forward_batch.input_ids.copy_(
                    (_forced_r if _forced_r is not None else _post_r).view(-1)
                )
            st.sc = None
            # Teacher forcing wins: best-of-N stands down when the canvas is forced.
            if st.bon_x0 is not None and _forced_r is None:
                # --- traj mode: this trajectory is done, bank it ------------
                st.bon.append(_post_r.clone())
                if st.bon_t + 1 < self.ar_bon:
                    # Restart from x_T; st.generator is deliberately not reset.
                    if self.ar_bon_resample_x0:
                        # Fresh x_T on the denoised positions only.
                        _x0 = st.bon_x0.clone()
                        _nz = torch.randint(
                            0,
                            self.real_vocab_size,
                            (B, Lb),
                            device=_x0.device,
                            generator=st.generator,
                        )
                        _x0[:, W - Lb :] = torch.where(st.upd, _nz, _x0[:, W - Lb :])
                        forward_batch.input_ids.copy_(_x0.view(-1))
                    else:
                        forward_batch.input_ids.copy_(st.bon_x0.view(-1))
                    # bank this trajectory's reverse steps before st.i resets
                    st.bon_steps = st.steps_run
                    st.i = 0
                    st.bon_t += 1
                    # Back to the start phase: an nfe=1 block starts in "readout".
                    st.phase = "denoise" if (st.spb or 0) > 0 else "readout"
                    if duo_debug.INSTR:
                        self._timer.split("samp")
                    return [False] * B
                st.bon_scores = []
                st.bon_i = 0
                st.phase = "bon_score"
                if duo_debug.INSTR:
                    self._timer.split("samp")
                return [False] * B
            if st.bon_x0 is not None:
                # forced: drop the trajectory machinery for this block
                st.bon = st.bon_x0 = None
                st.bon_t = 0
            if _bon_cands is not None and _forced_r is None:
                st.bon = _bon_cands
                st.bon_scores = []
                st.bon_i = 0
                st.phase = "bon_score"
                if duo_debug.INSTR:
                    self._timer.split("samp")
                return [False] * B
            if self.commit_fusion:  # noqa: SIM102 -- see _finish_block
                # The next block's wide first forward commits this one.
                if duo_debug.INSTR:
                    self._timer.split("samp")
                self._finish_block(forward_batch, st, x)
                return [True] * B
            st.phase = "commit"
            if duo_debug.INSTR:
                self._timer.split("samp")
            return [False] * B

        # ---- denoise step st.i ------------------------------------------------
        i = st.i
        a_t = duo_math.loglinear_alpha(self._ts[i], self.eps)
        a_s = duo_math.loglinear_alpha(self._ts[i + 1], self.eps)
        a_t = a_t.reshape(1, 1, 1).expand(B, 1, 1)
        a_s = a_s.reshape(1, 1, 1).expand(B, 1, 1)
        temp = duo_math.block_temp(
            self.temp_anneal,
            st.temperature,
            self.temp_start,
            self.temp_end,
            self.temp_schedule,
            i,
            st.spb,
        )
        temp = temp * self._bon_temp_mul(st)
        # Canvas only: sampling the carried half would diverge the RNG stream
        # from the unfused path. C is 0 on a narrow window.
        C = W - Lb
        if C:
            lg, xc, updc = logits[:, C - _lo :], x[:, C:], st.upd[:, C:]
        else:
            lg, xc, updc = logits, x, st.upd
        _greedy_now = st.greedy if st.greedy is not None else self.greedy
        if _greedy_now and self.greedy_mode == "tail":
            # Greedy-tail: denoise steps sample, only the readout argmaxes.
            _greedy_now = False
        if self._fused_readout_ok(
            _greedy_now, V, lg, W, Lb, temp, top_p=st.top_p, top_k=st.top_k
        ):
            # One launch for the step, then probs @ E into a half-precision
            # staging block copied into the fp32 self-cond buffer. Not bitwise.
            from sglang.srt.dllm.kernels import denoise_step as _ro

            E = self._model_runner.model.get_input_embeddings().weight
            hidden = E.shape[1]
            canvas = forward_batch.input_ids.view(B, W)
            # Under fusion the Lb-row current half is a contiguous view at B == 1,
            # so the argmax lands in the real canvas.
            cur = canvas[:, C:] if C else canvas
            upd_cur = st.upd[:, C:] if C else st.upd
            probs, sc16 = self._readout_buffers(B * Lb, V, hidden, E.dtype, E.device)
            if _greedy_now:
                _ro.greedy_readout(
                    lg.reshape(B * Lb, -1),
                    cur.view(-1),
                    upd_cur.reshape(-1),
                    V,
                    temp,
                    probs,
                )
            else:
                # the same uniform draw sample_categorical makes, in the same order
                u = torch.rand(
                    (B * Lb, V),
                    device=lg.device,
                    dtype=torch.float32,
                    generator=st.generator,
                )
                _alphas = torch.stack([a_t.reshape(-1)[0], a_s.reshape(-1)[0]]).to(
                    torch.float32
                )
                _ro.ancestral_step(
                    lg.reshape(B * Lb, -1),
                    cur.view(-1),
                    upd_cur.reshape(-1),
                    u,
                    _alphas,
                    V,
                    temp,
                    probs,
                )
            if _TRACE_PATH:
                # what the eager branch records, computed ONLY for the trace
                _lx = torch.nn.functional.log_softmax(
                    lg[..., :V].float() / temp, dim=-1
                )
                _b2 = torch.topk(_lx, 2, dim=-1)
                _TRACE.append(
                    dict(
                        phase="belief",
                        i=int(st.i),
                        block=self._blk_idx,
                        temp=float(temp),
                        b2_ids=_b2.indices.tolist(),
                        b2_vals=_b2.values.tolist(),
                        canvas=canvas.tolist(),
                        forced=False,
                    )
                )
            if self.self_cond:
                duo_debug.sync("sc")
                torch.mm(probs, E[:V], out=sc16)
                sc_buf = self._buf.sc_buffer(hidden)
                # Carried rows stay zero, as in the eager branch.
                (sc_buf[:, C:] if C else sc_buf).view(B * Lb, hidden).copy_(sc16)
                st.sc = sc_buf
            # Block total: st.i restarts at 0 per best-of-N trajectory.
            st.steps_run = (st.bon_steps or 0) + i + 1
            self._timer.steps = st.steps_run
            st.i = i + 1
            if st.i >= st.spb:
                st.phase = "readout"
            if duo_debug.INSTR:
                self._timer.split("samp")
            return [False] * B
        x_next, log_x0 = duo_math.duo_reverse_from_logits(
            lg,
            xc,
            a_t,
            a_s,
            V,
            temp,
            self.use_float64,
            greedy=_greedy_now,
            kappa=self.kappa,
            top_p=st.top_p,
            top_k=st.top_k,
            top_p_site=self.top_p_site,
            generator=st.generator,
        )
        if _TRACE_PATH:
            # Belief top-2 (temperature-scaled log-softmax), as the reference traces.
            _post = torch.where(updc, x_next, xc)
            _b2 = torch.topk(log_x0.float(), 2, dim=-1)
            _TRACE.append(
                dict(
                    phase="belief",
                    i=int(st.i),
                    block=self._blk_idx,
                    temp=float(temp),
                    b2_ids=_b2.indices.tolist(),
                    b2_vals=_b2.values.tolist(),
                    canvas=_post.tolist(),
                    # The engine's own canvas, captured before teacher forcing.
                    forced=_TF_CANVAS is not None,
                )
            )
        _forced = None
        if _TF_CANVAS is not None:
            _blk = self._blk_idx
            if _blk < 0:
                raise DllmContractError(
                    "teacher forcing has no block index: the generation-block "
                    "counter was never advanced, so the forced canvas cannot be "
                    "aligned. Forcing against a guessed block would align step i "
                    "of one block with step i of another and report the mismatch "
                    "as a port defect."
                )
            key = (_blk, int(st.i))
            if key not in _TF_CANVAS:
                raise DllmContractError(
                    f"teacher forcing has no canvas for block {_blk} step "
                    f"{int(st.i)}. The engine is running a step the reference "
                    "did not, so the two trajectories have different lengths. "
                    "Silently skipping the force for this step would reintroduce "
                    "exactly the divergence teacher forcing exists to remove, "
                    "and the resulting 'parity' would be false."
                )
            _forced = torch.as_tensor(_TF_CANVAS[key], device=x.device, dtype=x.dtype)

        if C:
            nxt = x.clone()
            # the forced canvas is already post-update, so it replaces the where()
            nxt[:, C:] = (
                _forced if _forced is not None else torch.where(updc, x_next, xc)
            )
            forward_batch.input_ids.copy_(nxt.view(-1))
        else:
            _nx = _forced if _forced is not None else torch.where(st.upd, x_next, x)
            forward_batch.input_ids.copy_(_nx.view(-1))

        if self.self_cond:
            # sc = selfcond_embedding(log_x0) * upd, in the fixed-address sc buffer.
            emb = self._selfcond_embed(log_x0, V)
            sc_buf = self._buf.sc_buffer(emb.shape[-1])
            if C:
                # the carried half gets a zero belief
                sc_buf.zero_()
                sc_buf[:, C:].copy_(emb).mul_(self._buf.upd_f[:, C:].unsqueeze(-1))
            else:
                duo_debug.sync("sc")
                sc_buf.copy_(emb).mul_(self._buf.upd_f.unsqueeze(-1))
            st.sc = sc_buf
        # Block total across best-of-N trajectories; see the sibling write.
        st.steps_run = (st.bon_steps or 0) + i + 1
        # Mirrored for the per-block record; the only observable of adaptive_stop.
        self._timer.steps = st.steps_run
        st.i = i + 1

        settled = False
        if self.adaptive_stop and st.i < st.spb:
            # Change detection vs the canvas entering the step (still in st.x_in),
            # on denoised positions only; never fires at step 0.
            pred = log_x0.argmax(-1)
            _xin = st.x_in[:, C:] if C else st.x_in
            settled = (i > 0) and bool(((pred == _xin) | ~updc).all())
            if not settled and self.stop_entropy > 0.0:
                ent = -(log_x0.exp() * log_x0).sum(-1)
                upd_f = updc.to(ent.dtype)
                blk_ent = (ent * upd_f).sum(-1) / upd_f.sum(-1).clamp_min(1)
                settled = bool((blk_ent <= self.stop_entropy).all())
        if settled or st.i >= st.spb:
            st.phase = "readout"
        if duo_debug.INSTR:
            self._timer.split("samp")
        return [False] * B

    # ---- helpers ---------------------------------------------------------------

    def _fused_readout_ok(
        self,
        greedy: bool,
        V: int,
        lg: torch.Tensor,
        W: int,
        Lb: int,
        temp: float,
        *,
        top_p: Optional[float],
        top_k: Optional[int],
    ) -> bool:
        """Whether the fused readout kernel can replace the eager step."""
        from sglang.srt.dllm.kernels import denoise_step as _ro

        if not _ro.enabled():
            return False
        if not greedy:
            # the ancestral kernel mirrors the default posterior only
            if (
                self.kappa != 1.0
                or (top_p is not None and top_p < 1.0)
                or (top_k or 0) > 0
            ):
                return False
            if not self.use_float64:
                return False
        if _TF_CANVAS is not None or self.adaptive_stop:
            return False
        # A fused window's current half must be a contiguous view for the
        # in-place canvas write: true only at B == 1.
        if W != Lb and not (self.commit_fusion and lg.shape[0] == 1):
            return False
        if not lg.is_floating_point():
            return False
        if not (float(temp) > 0.0 and math.isfinite(float(temp))):
            return False
        E = self._model_runner.model.get_input_embeddings().weight
        # with self-conditioning off the probs are never read (GEMM skipped)
        return E.shape[0] >= V and E.dtype in (torch.bfloat16, torch.float16)

    def _readout_buffers(
        self, rows: int, V: int, hidden: int, dtype: torch.dtype, device
    ):
        """Grow-only probs [rows, V] and sc16 [rows, hidden], one pair per algorithm."""
        pb = getattr(self, "_ro_probs", None)
        if pb is None or pb.dtype != dtype or pb.shape[0] < rows or pb.shape[1] != V:
            self._retire_readout_buffer(pb)
            pb = torch.empty(
                max(rows, pb.shape[0] if pb is not None else 0),
                V,
                dtype=dtype,
                device=device,
            )
            self._ro_probs = pb
        s16 = getattr(self, "_ro_sc16", None)
        if (
            s16 is None
            or s16.dtype != dtype
            or s16.shape[0] < rows
            or s16.shape[1] != hidden
        ):
            self._retire_readout_buffer(s16)
            s16 = torch.empty(
                max(rows, s16.shape[0] if s16 is not None else 0),
                hidden,
                dtype=dtype,
                device=device,
            )
            self._ro_sc16 = s16
        return pb[:rows], s16[:rows]

    def _retire_readout_buffer(self, buf) -> None:
        # A recorded block graph may still replay into the old storage; keep it alive.
        if buf is not None and self._block_graph_enabled():
            self.__dict__.setdefault("_ro_retired", []).append(buf)

    def _selfcond_embed(self, log_x0: torch.Tensor, V: int) -> torch.Tensor:
        """probs @ E on the tied input embedding. Under TP each rank contracts its
        vocab shard and the partials all-reduce; TP=1 must not take that path."""
        model = self._model_runner.model
        emb = model.get_input_embeddings()
        E = emb.weight
        if E.shape[0] >= V:
            return (log_x0.exp().to(E.dtype) @ E[:V]).detach()
        probs = log_x0.exp().to(E.dtype)
        # Checked against V (same on every rank), so no rank raises while
        # another enters the all-reduce.
        if probs.shape[-1] != V:
            raise DllmContractError(
                f"belief last dim {probs.shape[-1]} != vocab {V} (shape "
                f"{tuple(probs.shape)}); self-conditioning needs the FULL-vocab "
                "belief on every rank, and a rank-dependent refusal here would "
                "deadlock the others."
            )
        si = getattr(emb, "shard_indices", None)
        if si is None:
            raise DllmContractError(
                f"embedding rows {E.shape[0]} < vocab {V} but the layer exposes "
                "no shard_indices, so the global->local row mapping is unknown; "
                "refusing to guess which vocab rows this rank holds."
            )
        # weight_loader maps local row i to global org_start + i (tail zero-filled);
        # clamp to V: probs has no columns past V.
        lo = si.org_vocab_start_index
        hi = min(si.org_vocab_end_index, V)
        global _logged_shard
        if not _logged_shard:
            # Logged once: this geometry is hard to deduce from a shape error.
            _logged_shard = True
            logger.warning(
                "dllm-selfcond-shard: V=%d probs=%s weight=%s org=[%d,%d) "
                "padded_org=[%d,%d) added=[%d,%d) slice=[%d,%d) width=%d",
                V,
                tuple(probs.shape),
                tuple(E.shape),
                si.org_vocab_start_index,
                si.org_vocab_end_index,
                si.padded_org_vocab_start_index,
                si.padded_org_vocab_end_index,
                si.added_vocab_start_index,
                si.added_vocab_end_index,
                lo,
                hi,
                hi - lo,
            )
        if hi - lo > E.shape[0]:
            raise DllmContractError(
                f"shard_indices reports org rows [{lo}, {hi}) = {hi - lo} for "
                f"this rank, but the weight holds only {E.shape[0]} rows, so the "
                "global->local mapping assumed here is wrong. "
                f"V={V} tp={getattr(emb, 'tp_size', '?')} "
                f"padded_org=[{si.padded_org_vocab_start_index}, "
                f"{si.padded_org_vocab_end_index}) "
                f"org=[{si.org_vocab_start_index}, {si.org_vocab_end_index}) "
                f"added=[{si.added_vocab_start_index}, {si.added_vocab_end_index}) "
                f"weight={tuple(E.shape)}"
            )

        if hi <= lo:
            # A pure-padding shard must still enter the all-reduce, or it hangs.
            partial = probs.new_zeros(probs.shape[:-1] + (E.shape[1],))
        else:
            # log_x0 is [B, T, V]: slice the last axis.
            partial = probs[..., lo:hi] @ E[: hi - lo]
        # Deferred: CPU-only test harnesses stub `sglang.srt` as a non-package.
        from sglang.srt.distributed.communication_op import (
            tensor_model_parallel_all_reduce,
        )

        return tensor_model_parallel_all_reduce(partial).detach()

    def _conv_state_static(self) -> bool:
        """True when every conv layer runs the fused kernel, so denoise forwards
        cannot modify conv state and snapshot/restore is skipped. Cached."""
        v = getattr(self, "_conv_static_cached", None)
        if v is None:
            from sglang.srt.dllm.kernels import gated_sconv as _gs

            model = getattr(self._model_runner, "model", None)
            v = model is not None and _gs.covers_all(model)
            self._conv_static_cached = v
            logger.info("dLLM conv state static (every conv layer fused): %s", v)
        return v

    def _conv_restore_each_forward(self) -> bool:
        """Whether the block loop snapshots and restores conv state around each
        forward: unless the conv state is static, or the debug switch forces it.
        Only the loop's restore follows this; the graphed prompt prefill and the
        fusion refusal keep reading _conv_state_static()."""
        v = getattr(self, "_conv_restore_cached", None)
        if v is None:
            v = (
                not self._conv_state_static()
                or envs.SGLANG_DLLM_DEBUG_CONV_RESTORE.get()
            )
            self._conv_restore_cached = v
            logger.info("dLLM conv state restored around every forward: %s", v)
        return v

    def _set_adaln_base(self, forward_batch, st, buf, a_t, n: int) -> None:
        """Select this forward's adaLN base rows from the per-(spb, phase, step)
        table into a fixed-address buffer, replacing the model's timestep MLP."""
        model = getattr(self._model_runner, "model", None)
        rows_fn = getattr(model, "adaln_rows", None)
        if rows_fn is None:
            return
        key = (st.spb, st.phase, st.i if st.phase == "denoise" else -1)
        tab = self._adaln_tables.get(key)
        if tab is None:
            if a_t is None:
                levels = torch.zeros(2, dtype=torch.float32, device=buf.sigma.device)
            else:
                s = (-a_t.clamp_min(1e-30).log()).reshape(-1).to(torch.float32)
                levels = torch.cat([torch.zeros_like(s), s])
            tab = rows_fn(levels)
            self._adaln_tables[key] = tab
            if len(self._adaln_tables) == 1:
                logger.warning(
                    "dLLM adaLN table: first per-step rows built %s (dtype=%s); "
                    "later blocks select from cache",
                    tuple(tab.shape),
                    tab.dtype,
                )
        if buf.adaln_base is None or buf.adaln_base.shape[0] != n:
            buf.adaln_base = torch.empty(
                n, tab.shape[1], dtype=tab.dtype, device=tab.device
            )
        torch.index_select(tab, 0, buf.upd_idx[:n], out=buf.adaln_base)
        forward_batch.dllm_adaln_base = buf.adaln_base

    def _adopt_conv_snapshot(self, forward_batch: ForwardBatch, st: _BatchState):
        """Pull the pre-block conv snapshot taken inside the first forward; earlier
        reads would precede the hybrid COW/clear ops and see stale slots."""
        if st.conv_snapshot is None:
            snap = forward_batch.dllm_conv_snapshot
            st.conv_snapshot = snap if snap is not None else []

    def _restore_conv(self, st: _BatchState):
        # entries reference the pool tensors directly (same batch => same slots)
        for conv_state, idx, saved in st.conv_snapshot or []:
            if saved.dim() == 4:
                # pool-level entry [layers, B, dim, k-1]: every layer in one write
                conv_state[:, idx] = saved
            else:
                conv_state[idx] = saved

    # ---- AR self-verification: ENFORCING ------------------------------------

    def _ar_slots(self, forward_batch: ForwardBatch) -> List[int]:
        """Per-row carry key: req_pool_indices, stable for the request's life."""
        rp = getattr(forward_batch, "req_pool_indices", None)
        if rp is None:
            return []
        return [int(v) for v in rp.tolist()]

    def _ar_head_verdict(self, carry_rows, xb, V: int):
        """Accept/reject for block position 0 from the carried verifier row; a row
        with no carry is force-accepted."""
        B = xb.shape[0]
        dev = xb.device
        have = [r is not None for r in carry_rows]
        if not carry_rows or not any(have):
            return torch.ones(B, dtype=torch.bool, device=dev), None, True
        # Too-wide rows are sliced; narrower rows are refused (both producers
        # store real_vocab_size).
        _bad = [
            (b, tuple(r.shape))
            for b, r in enumerate(carry_rows)
            if r is not None and (r.dim() != 1 or int(r.shape[0]) < V)
        ]
        if _bad:
            raise DllmContractError(
                f"ar_verify: carry rows {_bad} are not [>= {V}] logit rows. "
                "A narrower row would have its argmax taken over part of the "
                "vocabulary, which can name a stop token the full row would "
                "not have."
            )
        proto = next(r for r in carry_rows if r is not None)[:V]
        rows = torch.stack(
            [(r[:V] if r is not None else torch.zeros_like(proto)) for r in carry_rows]
        ).to(dev)
        tgt = xb[:, :1]
        if self.ar_verify_tau > 0.0:
            lp = torch.log_softmax(rows.float(), dim=-1)
            ok = lp.gather(-1, tgt).squeeze(-1).exp() >= self.ar_verify_tau
        else:
            ok = rows.argmax(-1) == tgt.squeeze(-1)
        missing = torch.tensor([not h for h in have], device=dev)
        return ok | missing, rows, not all(have)

    def _ar_seed_carry(self, forward_batch: ForwardBatch, logits_output) -> None:
        """Seed the verifier carry from the last prompt position's prefill logit."""
        if logits_output is None:
            raise DllmContractError(
                "ar_verify: the prompt prefill produced no logits to seed the "
                "verifier carry from. The first generated token would then be "
                "the one position no rule ever judges."
            )
        full = getattr(logits_output, "full_logits", None)
        if full is None:
            raise DllmContractError(
                "ar_verify: the prefill's logits_output carries no full_logits "
                "(the model returned last-token logits only), so the seed "
                "cannot be taken."
            )
        slots = self._ar_slots(forward_batch)
        if not slots:
            return
        bs = forward_batch.batch_size
        rows = full.view(-1, full.shape[-1])
        n = rows.shape[0]
        lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if lens is None:
            _l = getattr(forward_batch, "extend_seq_lens", None)
            lens = None if _l is None else [int(v) for v in _l.tolist()]
        if lens is None:
            # uniform rows (the exactly-window-sized clean prompt)
            if bs <= 0 or n % bs:
                raise DllmContractError(
                    f"ar_verify: cannot locate each request's last prompt "
                    f"position ({n} logit rows over batch {bs}, no extend "
                    "lengths)."
                )
            lens = [n // bs] * bs
        at = 0
        for b, ln in enumerate(lens):
            at += int(ln)
            if b < len(slots) and int(ln) > 0:
                # Real-vocab width, as _ar_enforce stores; torch.stack needs one width.
                self._ar_carry[slots[b]] = (
                    rows[at - 1, : self.real_vocab_size].detach().clone()
                )

    def _eos_ids(self):
        """The stop tokens, from the model config. Resolved once."""
        v = getattr(self, "_eos_cache", None)
        if v is None:
            mc = getattr(self._model_runner, "model_config", None)
            ids = getattr(mc, "hf_eos_token_id", None)
            if not ids:
                raise DllmContractError(
                    "ar_stop needs the model's end-of-sequence id and the "
                    "model config exposes none. Without it the head's stop "
                    "decision cannot be recognised."
                )
            v = sorted(int(i) for i in ids)
            self._eos_cache = v
        return v

    def _eos_tensor(self, device) -> torch.Tensor:
        """The stop ids as a device tensor, built once per device."""
        # Keyed on the ids too, so a reassigned list cannot hit a stale tensor.
        ids = tuple(self._eos_ids())
        key, t = getattr(self, "_eos_dev_cache", (None, None))
        if t is None or key != (ids, device):
            t = torch.tensor(ids, device=device)
            self._eos_dev_cache = ((ids, device), t)
        return t

    def _ar_redo_maybe(self, forward_batch, st, logits, x, B, W, Lb, V) -> bool:
        """Score the committed block; if the head rejects it, re-noise the worst
        positions and return True (another denoise leg). Position 0 is never rejected.
        """
        Cr = W - Lb
        lg = logits[:, Cr:, :V]
        cand = x[:, Cr:] if Cr else x
        with torch.no_grad():
            lp = torch.log_softmax(lg[:, :-1].float(), dim=-1)
            tok = lp.gather(-1, cand[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, Lb-1]
            # The one host read: the verdict changes the forward count.
            meanp = tok.mean(dim=1).exp()  # [B]
            row_fire = meanp < self.ar_redo_tau
            fire = bool(row_fire.any().item())
        if duo_debug.T_OUT:
            # Every block, fired or not, so tau can be calibrated.
            duo_debug.instr_emit(
                dict(
                    kind="ar_score",
                    block=self._blk_idx,
                    meanp=[float(v) for v in meanp.tolist()],
                    fired=bool(fire),
                    nfe=int(st.spb0 or st.spb or 0),
                )
            )
        if not fire:
            return False
        k = max(1, int(round(Lb * self.ar_redo_frac)))
        # position 0 is unscorable here, so it is never a redo candidate
        worst = torch.topk(-tok, min(k, tok.shape[1]), dim=1).indices + 1
        # A redo runs for every row; approved rows get an all-false mask.
        upd = torch.zeros(B, W, dtype=torch.bool, device=x.device)
        upd[:, Cr:].scatter_(1, worst, True)
        upd &= row_fire[:, None]
        # Re-noise only those positions; the rest stay at sigma 0 via upd.
        nz = torch.randint(
            0,
            self.real_vocab_size,
            (B, Lb),
            device=x.device,
            dtype=x.dtype,
            generator=st.generator,
        )
        canvas = forward_batch.input_ids.view(B, W)
        canvas[:, Cr:] = torch.where(upd[:, Cr:], nz, canvas[:, Cr:])
        # the block scratch follows the new mask
        duo_debug.sync("redo_upd")
        self._buf.upd.copy_(upd)
        st.upd = self._buf.upd
        self._buf.upd_f.copy_(st.upd.to(self._buf.upd_f.dtype))
        self._buf.upd_idx.copy_(st.upd.reshape(-1).to(torch.long))
        if st.spb0 is None:
            st.spb0 = st.spb
        # Bank the steps run so far before st.i restarts.
        st.bon_steps = st.steps_run
        # Full schedule by default: re-noised positions start from the uniform
        # prior and need the full sigma ladder.
        steps = self.ar_redo_steps or (st.spb0 or st.spb or 1)
        _tsk = (int(steps), str(x.device))
        _ts = self._ts_cache.get(_tsk)
        if _ts is None:
            _ts = duo_math.timestep_grid(
                steps, self.eps, self.schedule, self.rho, x.device
            )
            self._ts_cache[_tsk] = _ts
        self._ts = _ts
        st.ctx.ts = _ts
        # Re-enter from the pre-block conv state; the commit just advanced it.
        # Also under the fused conv, which writes state on commit only: without
        # the restore the redo legs use this block's own tail as the left context
        # of its first positions.
        self._adopt_conv_snapshot(forward_batch, st)
        if not st.conv_snapshot:
            raise DllmContractError(
                "ar_redo: no pre-block conv snapshot to re-enter from. Without "
                "it the redo legs convolve on top of the commit's output."
            )
        self._restore_conv(st)
        st.spb, st.i, st.sc, st.phase = steps, 0, None, "denoise"
        st.redo_left -= 1
        if duo_debug.T_OUT:
            duo_debug.instr_emit(
                dict(
                    kind="ar_redo",
                    block=self._blk_idx,
                    renoised=int(k),
                    steps=int(steps),
                    left=int(st.redo_left),
                    nfe=int(st.spb0 or 0),
                )
            )
        return True

    def _ar_adaptive_spb(self, forward_batch, default: int) -> int:
        """Steps for the coming block from the AR head's confidence in its first
        token; ar_nfe_hi when the carry is missing."""
        slots = getattr(forward_batch, "req_pool_indices", None)
        if slots is None:
            return default
        _sl = [int(v) for v in slots.tolist()]
        rows = [self._ar_carry.get(v) for v in _sl]
        if not rows or any(r is None for r in rows):
            # One fresh request forces hi for the whole batch; recorded.
            if duo_debug.T_OUT:
                duo_debug.instr_emit(
                    dict(
                        kind="ar_nfe",
                        block=self._blk_idx,
                        pmax=None,
                        nfe=int(self.ar_nfe_hi),
                        rows=len(rows),
                        fallback="no_carry",
                    )
                )
            return self._denoise_steps(self.ar_nfe_hi)
        with torch.no_grad():
            lg = torch.stack(rows, dim=0).float()
            pr = torch.softmax(lg, dim=-1)
            top2 = pr.topk(2, dim=-1).values
            # All signals every time, so one calibration run can set tau for each.
            sig = {
                "pmax": top2[:, 0],
                "margin": top2[:, 0] - top2[:, 1],
                # 1 - H/log V, so higher is more confident like the others
                "entropy": 1.0
                - (
                    -(pr * pr.clamp_min(1e-30).log()).sum(-1)
                    / math.log(max(2, pr.shape[-1]))
                ),
            }
            # MIN across rows: the least confident row decides for the batch.
            vals = {k: float(v.min().item()) for k, v in sig.items()}
        ps = [self._ar_prevscore.get(v) for v in _sl]
        vals["prevscore"] = (
            float(min(ps)) if ps and all(p is not None for p in ps) else None
        )
        pick = vals.get(self.ar_nfe_signal)
        if pick is None:
            # prevscore missing: same answer as a missing carry.
            if duo_debug.T_OUT:
                duo_debug.instr_emit(
                    dict(
                        kind="ar_nfe",
                        block=self._blk_idx,
                        nfe=int(self.ar_nfe_hi),
                        rows=len(rows),
                        fallback="no_prevscore",
                        **vals,
                    )
                )
            return self._denoise_steps(self.ar_nfe_hi)
        nfe = self.ar_nfe_lo if pick >= self.ar_nfe_tau else self.ar_nfe_hi
        if duo_debug.T_OUT:
            duo_debug.instr_emit(
                dict(
                    kind="ar_nfe",
                    block=self._blk_idx,
                    signal=self.ar_nfe_signal,
                    nfe=int(nfe),
                    rows=len(rows),
                    **vals,
                )
            )
        return self._denoise_steps(nfe)

    def _bon_temp_mul(self, st) -> float:
        """Trajectory t's temperature multiplier; 1.0 when the fan is off."""
        if not self.ar_bon_temp_spread or st.bon_x0 is None or self.ar_bon < 2:
            return 1.0
        frac = (2.0 * st.bon_t / (self.ar_bon - 1)) - 1.0  # -1 .. +1
        return 1.0 + self.ar_bon_temp_spread * frac

    def _ar_may_correct(self) -> bool:
        """Whether an AR feature may issue a forward that re-enters from the
        pre-block conv snapshot, needed even when the conv state is static."""
        return bool(
            self.ar_verify
            or self.ar_stop
            or self.ar_bon
            or self.ar_redo
            or self.ar_nfe_adapt
        )

    @staticmethod
    def _ar_ignore_eos(forward_batch, B: int) -> List[bool]:
        """Per-row SamplingParams.ignore_eos, all-false when unset."""
        v = forward_batch.dllm_ignore_eos
        if v is None:
            return [False] * B
        if len(v) != B:
            raise DllmContractError(
                f"dllm_ignore_eos has {len(v)} entries for {B} requests"
            )
        return [bool(x) for x in v]

    def _ar_stop_only(
        self,
        forward_batch: ForwardBatch,
        st: _BatchState,
        logits: torch.Tensor,
        x: torch.Tensor,
        B: int,
        W: int,
        Lb: int,
        V: int,
    ) -> None:
        """Let the AR head decide only where the request ends: commit the draft
        up to the head's first stop token, plus that token.

        The correction forward is required: the stop token's KV was never
        written, and the prefix tree keys a finished request on its fill ids.
        """
        Cr = W - Lb
        xb = x[:, Cr:]
        lgb = logits[:, Cr:, :V]
        slots = self._ar_slots(forward_batch)
        carry_rows = [self._ar_carry.get(s) for s in slots] if slots else []
        eos = self._eos_tensor(x.device)
        starts = self._start_list or [0] * B
        with torch.no_grad():
            tail = lgb[:, :-1].argmax(-1)  # predicts positions 1..Lb-1
            _, carry_stack, _ = self._ar_head_verdict(carry_rows, xb, V)
            if carry_stack is not None:
                head = carry_stack[:, :V].argmax(-1, keepdim=True)
                pred = torch.cat((head, tail), dim=1)  # [B, Lb]
            else:
                # no carry anywhere: column 0 is a placeholder, cleared below
                pred = torch.cat((xb[:, :1], tail), dim=1)
            is_stop = (pred.unsqueeze(-1) == eos).any(-1)  # [B, Lb]
            # Carry-less rows are zero-filled; argmax over zeros is id 0, maybe a stop.
            for b, r in enumerate(carry_rows if carry_rows else [None] * B):
                if r is None:
                    is_stop[b, 0] = False
            for b, s in enumerate(starts):
                if s:
                    is_stop[b, : min(int(s), Lb)] = False
            # A request that ignores EOS is not ended by one.
            for b, ig in enumerate(self._ar_ignore_eos(forward_batch, B)):
                if ig:
                    is_stop[b, :] = False
            hit = is_stop.any(dim=1)
            first = torch.argmax(is_stop.to(torch.int32), dim=1)
            hit_cpu = [bool(v) for v in hit.tolist()]
            first_cpu = [int(v) for v in first.tolist()]

        if not any(hit_cpu):
            # No stop: the block commits whole.
            self._ar_advance_carry(slots, lgb, [Lb] * B)
            return
        commit, stopped = [], []
        for b in range(B):
            if not hit_cpu[b]:
                commit.append(Lb)
                stopped.append(False)
                continue
            j = first_cpu[b]
            xb[b, j] = int(pred[b, j])
            commit.append(j + 1)
            stopped.append(True)
        at = self._buf.ar_state_at(B, x.device)
        at.copy_(torch.tensor(commit, dtype=torch.int32, device=x.device) + Cr)
        out = self._ar_forward(forward_batch, st, at)
        self._corrected_out = out
        self._commit_lens = commit
        # The carry comes from the corrected forward.
        self._ar_advance_carry(
            slots, out.logits_output.full_logits.view(B, W, -1)[:, Cr:, :V], commit
        )
        t = self._ar_totals
        t["stops"] = t.get("stops", 0) + sum(1 for v in stopped if v)
        duo_debug.instr_emit(
            dict(
                kind="ar_stop",
                block=self._blk_idx,
                committed=commit,
                stopped=stopped,
                verifiable=Lb,
                nfe=int(st.spb or 0),
            )
        )
        logger.debug("dllm-ar-stop: committed=%s stopped=%s", commit, stopped)

    def _ar_advance_carry(self, slots, lgb, commit) -> None:
        """Carry the logit row at each row's last committed position; cloned, as
        logits live in a graph output buffer. Cleared at the slot's next prefill."""
        for b, sl in enumerate(slots or []):
            self._ar_carry[sl] = lgb[b, commit[b] - 1].detach().clone()

    def _ar_forward(self, forward_batch: ForwardBatch, st: _BatchState, state_at):
        """One clean causal pass over the corrected window, from the pre-block conv
        snapshot. Replanned: an eager forward that skipped planning would attend
        through capture-time indices."""
        if duo_debug.INSTR:
            # Close the verdict into samp before the restore.
            self._timer.split("samp")
        self._adopt_conv_snapshot(forward_batch, st)
        if not st.conv_snapshot:
            raise DllmContractError(
                "ar_verify: no pre-block conv snapshot to re-enter from. "
                "Without it the corrected window convolves on top of the "
                "previous pass's output and every token it rewrites is wrong."
            )
        self._restore_conv(st)
        prev_ready = getattr(forward_batch, "attn_metadata_ready", False)
        forward_batch.attn_metadata_ready = False
        forward_batch.dllm_graph_replayed = False
        forward_batch.dllm_conv_state_at = state_at
        try:
            if duo_debug.INSTR:
                # restoration -> prep, so the buckets partition the interval
                self._timer.split("prep")
            if dllm_sync_enabled("forward"):
                dllm_device_synchronize()
            out = self._model_runner.forward(forward_batch, pp_proxy_tensors=None)
            if duo_debug.INSTR:
                # a real model invocation: count it
                self._timer.split("fwd")
            return out
        finally:
            forward_batch.dllm_conv_state_at = None
            forward_batch.attn_metadata_ready = prev_ready

    def _ar_verdict(self, lgb, xb, carry_rows, forced, starts, B, Lb, V):
        """Per-row accepted prefix (host ints), agree [B, Lb], and the AR argmax
        [B, Lb]. Positions below `starts`, in `forced`, or without a carry pass."""
        with torch.no_grad():
            if self.ar_verify_tau > 0.0:
                lp = torch.log_softmax(lgb[:, :-1].float(), dim=-1)
                p_tok = lp.gather(-1, xb[:, 1:].unsqueeze(-1)).squeeze(-1).exp()
                tail = p_tok >= self.ar_verify_tau
            else:
                tail = lgb[:, :-1].argmax(-1) == xb[:, 1:]
            head, carry_stack, missing = self._ar_head_verdict(carry_rows, xb, V)
            if missing and self.ar_repair:
                # Repair claims AR-identity; a missing carry cannot be waved through.
                raise DllmContractError(
                    "ar_repair: no verifier carry for block position 0. The "
                    "prompt prefill seeds it (_ar_seed_carry); without it that "
                    "token is never judged and the block's fixed point is not "
                    "the AR continuation of the prompt."
                )
            agree = torch.cat((head.unsqueeze(1), tail), dim=1)
            if any(s > 0 for s in starts):
                keep = torch.zeros_like(agree)
                for _b, _s in enumerate(starts):
                    keep[_b, : min(int(_s), Lb)] = True
                agree = agree | keep
            if forced is not None:
                agree = agree | forced
            r = agree.long().cumprod(dim=1).sum(dim=1)
            # Index j reads the logit at j-1; position 0 reads the carry.
            tail_ids = lgb[:, :-1].argmax(-1)
            if carry_stack is not None:
                head_ids = carry_stack[:, :V].argmax(-1)
            else:
                head_ids = xb[:, 0]
            pred_ids = torch.cat((head_ids.unsqueeze(1), tail_ids), dim=1)
        return [int(v) for v in r.tolist()], agree, pred_ids

    def _ar_enforce(
        self,
        forward_batch: ForwardBatch,
        st: _BatchState,
        logits: torch.Tensor,
        x: torch.Tensor,
        B: int,
        W: int,
        Lb: int,
        V: int,
    ) -> None:
        """Apply the AR verdict to what the block commits.

        ar_repair=0 commits the accepted prefix plus the head's token at the first
        miss; ar_repair=k substitutes and re-verifies up to k passes (the prefix
        strictly grows). A partial accept costs one clean causal correction pass.
        """
        Cr = W - Lb
        xb = x[:, Cr:]
        lgb = logits[:, Cr:, :V]
        slots = self._ar_slots(forward_batch)
        carry_rows = [self._ar_carry.get(s) for s in slots] if slots else []
        starts = self._start_list or [0] * B
        forced = torch.zeros(B, Lb, dtype=torch.bool, device=x.device)

        passes = 0
        r0 = None
        have_carry = [c is not None for c in carry_rows] if slots else [False] * B
        while True:
            r_cpu, agree, pred_ids = self._ar_verdict(
                lgb, xb, carry_rows, forced, starts, B, Lb, V
            )
            if r0 is None:
                r0 = list(r_cpu)  # the accept-prefix result, for the stats
            if all(rb >= Lb for rb in r_cpu):
                # Nothing left to fix; any pass that ran wrote the whole window.
                commit = [Lb] * B
                substituted = [False] * B
                break
            if passes >= self.ar_repair:
                # Budget spent (or accept-prefix mode): substitute the first miss.
                commit, substituted = [], []
                for b, rb in enumerate(r_cpu):
                    if rb >= Lb:
                        commit.append(Lb)
                        substituted.append(False)
                        continue
                    xb[b, rb] = pred_ids[b, rb]
                    commit.append(rb + 1)
                    substituted.append(True)
                # Checked on host: the kernel cannot without a device read per layer.
                _bad = [c for c in commit if not 1 <= c <= Lb]
                if _bad:
                    raise DllmContractError(
                        f"ar_verify: committed length {_bad} outside "
                        f"[1, {Lb}]. Zero would leave the request without "
                        "progress and more than the block would cite tokens "
                        "no forward produced."
                    )
                at = self._buf.ar_state_at(B, x.device)
                at.copy_(torch.tensor(commit, dtype=torch.int32, device=x.device) + Cr)
                out = self._ar_forward(forward_batch, st, at)
                self._corrected_out = out
                lgb = out.logits_output.full_logits.view(B, W, -1)[:, Cr:, :V]
                break
            # Repair pass: force only the first miss; the rest are re-judged.
            if self.ar_repair_width == "all":
                xb.copy_(torch.where(agree, xb, pred_ids))
            for b, rb in enumerate(r_cpu):
                if rb < Lb:
                    xb[b, rb] = pred_ids[b, rb]
                    forced[b, rb] = True
            passes += 1
            out = self._ar_forward(forward_batch, st, None)
            self._corrected_out = out
            lgb = out.logits_output.full_logits.view(B, W, -1)[:, Cr:, :V]

        # From the forward that wrote the last committed token's KV.
        self._ar_advance_carry(slots, lgb, commit)

        self._commit_lens = commit
        t = self._ar_totals
        t["blocks"] += B
        # The carried head and a carry-less position 0 were never judged; max,
        # not sum, since starts >= 1 already excludes position 0.
        _unjudged = [
            max(
                min(int(starts[b]), Lb),
                0 if (b < len(have_carry) and have_carry[b]) else 1,
            )
            for b in range(B)
        ]
        t["drafted"] += sum(max(0, Lb - u) for u in _unjudged)
        # The first verdict, before any repair.
        t["accepted"] += sum(max(0, r0[b] - _unjudged[b]) for b in range(B))
        t["bonus"] += sum(1 for v in substituted if v)
        # request-weighted, as `blocks` is
        t["passes"] += passes * B
        duo_debug.instr_emit(
            dict(
                kind="ar_enforce",
                block=self._blk_idx,
                tau=self.ar_verify_tau,
                repair=self.ar_repair,
                accepted=r0,
                committed=commit,
                starts=[int(v) for v in starts],
                unjudged=[int(v) for v in _unjudged],
                passes=passes,
                # The budget-exit branch runs one more forward; a converged block does not.
                final_fwd=bool(any(substituted)),
                # Per request: every newly emitted token is the AR head's argmax
                # (argmax rule, and every new position judged).
                ar_exact=[
                    bool(
                        self.ar_verify_tau == 0.0
                        and (
                            # with a carried head, position 0 is not newly emitted
                            int(starts[b]) > 0
                            or (b < len(have_carry) and have_carry[b])
                        )
                    )
                    for b in range(B)
                ],
                corrected=bool(passes or any(substituted)),
                verifiable=Lb,
                nfe=int(st.spb or 0),
            )
        )
        logger.debug(
            "dllm-ar-enforce: accepted=%s committed=%s passes=%d of %d "
            "(tau=%.2f, repair=%d)",
            r0,
            commit,
            passes,
            Lb,
            self.ar_verify_tau,
            self.ar_repair,
        )

    # ---- end of block (commit path and fused terminal path) -----------------
    def _finish_block(
        self, forward_batch: ForwardBatch, st: _BatchState, x: torch.Tensor
    ) -> None:
        """End-of-block finalization, shared by the commit and fused terminal paths."""
        if duo_debug.STATE_DUMP and self._model_runner is not None:
            self._blocks_done = getattr(self, "_blocks_done", 0)
            duo_debug.dump_state(self._model_runner, forward_batch, self._blocks_done)
            self._blocks_done += 1
        if duo_debug.INSTR:
            self._timer.end()
        st.phase = "done"
        if self.fdfo and st.ctx is not None and st.ctx.buf is not None:
            # back to the pool; a done row keeps its ctx but is never prepared again
            _b = st.ctx.buf
            self._buf_pool.setdefault((_b.b, _b.lb, str(_b.sigma.device)), []).append(
                _b
            )
        if _TRACE_PATH:
            _TRACE.append(dict(phase="final", i=st.i, tokens=x.tolist()))
            with open(_TRACE_PATH, "w") as f:
                json.dump(_TRACE, f)

    def prefill_view(self, forward_batch: ForwardBatch):
        """Short prompts, each padded to block width, so their prefill replays the
        clean causal block graph; anything else returns the batch unchanged.

        Pad rows sit after each prompt (unattended), write K/V to reserved slot 0,
        and dllm_conv_state_at[b] stops the short conv at the real prompt end.
        """
        if not dllm_graph_flag(
            envs.SGLANG_DLLM_ENABLE_GRAPH_PROMPT_PREFILL, self.algorithm_name
        ):
            return forward_batch
        W = self.block_size
        B = forward_batch.batch_size
        runner = getattr(self._model_runner, "decode_cuda_graph_runner", None)
        lens = forward_batch.extend_seq_lens_cpu
        prefix = forward_batch.extend_prefix_lens_cpu
        if (
            B < 1
            or lens is None
            or prefix is None
            or len(lens) != B
            # An exactly-block_size prompt takes the exact-width path and never reaches here.
            or not all(0 < int(L) < W for L in lens)
            or any(int(p) != 0 for p in prefix)
            or sum(int(L) for L in lens) != forward_batch.input_ids.numel()
            # Bidirectional prefix attention would let the prompt see its pad.
            or self.prefix_bidirectional
            # The view leaves the causal-rows slot unfilled (0 = bidirectional).
            or _TENSOR_CAUSALITY
            # Indexed into the packed batch, which the view re-lays out.
            or forward_batch.input_embeds is not None
            or forward_batch.replace_embeds is not None
            or forward_batch.replace_positions is not None
            # DP attention sizes collectives from packed counts the view does not rebuild.
            or forward_batch.global_num_tokens_cpu is not None
            # These read the prefill's logits per position.
            or self.ar_verify
            or self.ar_stop
            or self.ar_nfe_adapt
            or runner is None
            or not runner._dllm_sat_capturable
            or not self._conv_state_static()
        ):
            return forward_batch
        dev = forward_batch.input_ids.device
        lens = [int(L) for L in lens]
        pp = self._prefill_bufs(B, W, dev)
        pp["state_at"].copy_(torch.tensor(lens, dtype=torch.int32), non_blocking=True)
        # Row b's real tokens land at [b*W, b*W + L_b); the rest of its window is pad.
        dst = torch.tensor(
            [b * W + k for b, L in enumerate(lens) for k in range(L)], dtype=torch.long
        ).to(dev, non_blocking=True)
        v = copy.copy(forward_batch)
        v.input_ids = forward_batch.input_ids.new_full((B * W,), self.mask_id)
        v.input_ids[dst] = forward_batch.input_ids
        # Fresh prompts start at position 0, so every padded row is 0..W-1.
        v.positions = torch.arange(
            W, device=dev, dtype=forward_batch.positions.dtype
        ).repeat(B)
        v.out_cache_loc = forward_batch.out_cache_loc.new_zeros(B * W)
        v.out_cache_loc[dst] = forward_batch.out_cache_loc
        v.dllm_prefill_rows = (
            dst  # where each real prompt token sits (test dumps read it)
        )
        v.extend_num_tokens = B * W
        v.extend_seq_lens = torch.full_like(forward_batch.extend_seq_lens, W)
        v.extend_seq_lens_cpu = [W] * B
        v.extend_start_loc = torch.arange(
            0, B * W, W, device=dev, dtype=forward_batch.extend_start_loc.dtype
        )
        v.seq_lens = torch.full_like(forward_batch.seq_lens, W)
        if forward_batch.seq_lens_cpu is not None:
            v.seq_lens_cpu = torch.full_like(forward_batch.seq_lens_cpu, W)
        v.seq_lens_sum = B * W
        if forward_batch.orig_seq_lens is not None:
            v.orig_seq_lens = torch.full_like(forward_batch.orig_seq_lens, W)
        if forward_batch.num_token_non_padded is not None:
            v.num_token_non_padded = torch.full_like(
                forward_batch.num_token_non_padded, B * W
            )
            v.num_token_non_padded_cpu = B * W
        if forward_batch.extend_logprob_start_lens_cpu is not None:
            v.extend_logprob_start_lens_cpu = [0] * B
        # The gate is all ones, never None: fill_from skips a None field
        # without resetting its graph slot.
        v.dllm_sigma = pp["sigma"]
        v.dllm_adaln_base = pp["adaln"]
        v.dllm_selfcond = None
        v.dllm_selfcond_pos_mask = pp["pos"]
        v.dllm_causal_rows = None
        v.dllm_clean_upto = 0
        v.dllm_save_kv = True
        v.dllm_conv_state_at = pp["state_at"]
        v.dllm_prompt_prefill = False
        v.dllm_conv_capture = False
        v.dllm_conv_snapshot = None
        v.dllm_phase = None
        v.attn_metadata_ready = False
        v.forward_metadata_ready = False
        v.forward_metadata_planned_bs = None
        v.forward_metadata_planned_num_tokens = None
        v.dllm_graph_replayed = False
        return v

    def _prefill_bufs(self, B: int, W: int, dev) -> dict:
        """Fixed-address clean-prefill conditioning for B padded rows, one set per B."""
        bufs = self._pp_bufs
        if bufs is None:
            bufs = self._pp_bufs = {}
        pp = bufs.get((B, W))
        if pp is None:
            rows_fn = getattr(self._model_runner.model, "adaln_rows", None)
            n = B * W
            pp = dict(
                sigma=torch.zeros(n, dtype=torch.float32, device=dev),
                adaln=(
                    None
                    if rows_fn is None
                    else rows_fn(torch.zeros(1, dtype=torch.float32, device=dev))
                    .expand(n, -1)
                    .contiguous()
                ),
                pos=torch.ones(n, dtype=torch.float32, device=dev),
                state_at=torch.zeros(B, dtype=torch.int32, device=dev),
            )
            bufs[(B, W)] = pp
        return pp

    def forward_view_width(self, forward_batch: ForwardBatch):
        if not self.commit_fusion:
            return None
        W = self.forward_width(forward_batch) or self.window_size
        return self.block_size if W > self.block_size else None

    def forward_view(self, forward_batch: ForwardBatch, states):
        """Forwards 1..N of a fused block run on the current half only; the view
        aliases the wide batch's buffers, so every canvas write lands in both."""
        if not self.commit_fusion:
            return forward_batch
        Lb = self.block_size
        W = self.forward_width(forward_batch) or self.window_size
        C = W - Lb
        if C <= 0:
            return forward_batch
        v = self._narrow_view
        if v is None:
            v = self._build_narrow_view(forward_batch, C, Lb)
            self._narrow_view = v
        # The current half is rows [C:] at B == 1.
        v.dllm_sigma = forward_batch.dllm_sigma[C:]
        v.dllm_adaln_base = (
            None
            if forward_batch.dllm_adaln_base is None
            else forward_batch.dllm_adaln_base[C:]
        )
        v.dllm_selfcond = (
            None
            if forward_batch.dllm_selfcond is None
            else forward_batch.dllm_selfcond[C:]
        )
        # Never None: fill_from skips None without resetting the slot, so the wide
        # forward's gate would replay and silently disable self-conditioning.
        v.dllm_selfcond_pos_mask = forward_batch.dllm_selfcond_pos_mask[C:]
        v.dllm_phase = forward_batch.dllm_phase
        v.dllm_causal_override = forward_batch.dllm_causal_override
        v.dllm_causal_rows = forward_batch.dllm_causal_rows
        return v

    @staticmethod
    def _build_narrow_view(forward_batch: ForwardBatch, C: int, Lb: int):
        """Shallow copy of the wide batch over its last Lb positions; the prefix
        grows by C, seq_lens is unchanged."""
        if forward_batch.batch_size != 1:
            raise DllmContractError(
                "commit fusion's narrow view needs batch size 1 (the current "
                f"half must be a contiguous view); got {forward_batch.batch_size}"
            )
        v = copy.copy(forward_batch)
        v.input_ids = forward_batch.input_ids[C:]
        v.positions = forward_batch.positions[C:]
        v.out_cache_loc = forward_batch.out_cache_loc[C:]
        v.extend_num_tokens = Lb
        # A block graph's static batch has no extend fields.
        if forward_batch.extend_seq_lens is not None:
            v.extend_seq_lens = torch.full_like(forward_batch.extend_seq_lens, Lb)
            v.extend_seq_lens_cpu = [Lb]
        if forward_batch.extend_prefix_lens is not None:
            v.extend_prefix_lens = forward_batch.extend_prefix_lens + C
        if forward_batch.extend_prefix_lens_cpu is not None:
            v.extend_prefix_lens_cpu = [
                int(forward_batch.extend_prefix_lens_cpu[0]) + C
            ]
        if forward_batch.extend_start_loc is not None:
            v.extend_start_loc = torch.zeros_like(forward_batch.extend_start_loc)
        if forward_batch.extend_logprob_start_lens_cpu is not None:
            v.extend_logprob_start_lens_cpu = [0]
        if forward_batch.num_token_non_padded is not None:
            v.num_token_non_padded = torch.full_like(
                forward_batch.num_token_non_padded, Lb
            )
            v.num_token_non_padded_cpu = Lb
        # Persists nothing, carries nothing: an unfused denoise forward.
        v.dllm_save_kv = False
        v.dllm_conv_state_at = None
        v.dllm_clean_upto = 0
        v.dllm_conv_capture = False
        v.dllm_conv_snapshot = None
        v.dllm_prompt_prefill = False
        # A fresh plan: the wide batch's describes a different token count.
        v.attn_metadata_ready = False
        v.forward_metadata_ready = False
        v.forward_metadata_planned_bs = None
        v.forward_metadata_planned_num_tokens = None
        v.dllm_graph_replayed = False
        return v

    @staticmethod
    def _window_start(forward_batch: ForwardBatch):
        """Absolute start of row 0's window, or None; prefers the host prefix length."""
        pref = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        if pref:
            return int(pref[0])
        pos = getattr(forward_batch, "positions", None)
        if pos is None:
            return None
        try:
            return int(pos.view(forward_batch.batch_size, -1)[0, 0].item())
        except Exception:
            return None

    def after_clean_prefill(
        self, forward_batch: ForwardBatch, logits_output=None
    ) -> None:
        """Seed the AR carry; under fusion record the prompt end, the only start
        where a narrow first generation block is legitimate."""
        if self.ar_verify or self.ar_stop:
            self._ar_seed_carry(forward_batch, logits_output)
        if not self.commit_fusion:
            return
        bs = forward_batch.batch_size
        n = forward_batch.input_ids.numel()
        plen = n // bs if bs and n % bs == 0 else None
        ws = self._window_start(forward_batch)
        if plen is None or ws is None:
            raise DllmContractError(
                "commit fusion: cannot locate the prompt end "
                f"(batch={bs} tokens={n} window_start={ws}), so the one window "
                "start at which a narrow block is legitimate is unknown."
            )
        self._fusion_first_start = ws + plen

    def on_clean_prefill(self, forward_batch: ForwardBatch) -> None:
        """A clean prompt extend ran: drop the slot's AR carry."""
        if self.ar_verify or self.ar_stop or self.ar_nfe_adapt:
            # Per slot; a recycled slot must not inherit the previous tenant's carry.
            for _s in self._ar_slots(forward_batch):
                self._ar_carry.pop(_s, None)
                self._ar_prevscore.pop(_s, None)


Algorithm = DuoBlock
