from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union

import torch

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.params import dllm_graph_flag
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)

_is_npu = is_npu()

DllmRunOutput = Tuple[
    Union[LogitsProcessorOutput, torch.Tensor],
    List,
    Optional[List[int]],
    Optional[List[Any]],
    bool,
]


# Exact-token parse of SGLANG_DLLM_SYNC_SITES, shared with duo_block's _sync;
# cached at import to avoid re-parsing per forward.
_SYNC_ENV = envs.SGLANG_DLLM_SYNC_SITES.get()
# An explicitly empty value means "no barriers anywhere", distinct from unset,
# so the default barrier can be turned off.
_SYNC_SET = _SYNC_ENV is not None
_SYNC_SITES = frozenset(x.strip() for x in (_SYNC_ENV or "").split(",") if x.strip())

# Arbitrary; startup warmup uses about ten. Past it a new key runs eager rather
# than capturing a graph under live traffic.
_BLOCK_GRAPH_MAX_KEYS = 64


# Skip the per-step attention re-plan (not the buffer staging) inside a denoise
# block. Resolved lazily so a default set after import still takes effect.
def _dllm_preplan_attn() -> bool:
    return envs.SGLANG_DLLM_PREPLAN_ATTN.get()


def _default_forward_barrier() -> bool:
    """Whether the per-forward barrier is on by default: everywhere except CUDA.

    On ROCm it prevents a dLLM graph-replay fault; on CUDA the fault does not
    reproduce and the barrier is measurably slower.
    """
    try:
        import torch

        from sglang.srt.utils import is_hip

        # barrier OFF only where it was actually validated off: real CUDA.
        return not (torch.version.cuda is not None and not is_hip())
    except Exception:
        # unknown platform -> keep the safe (barrier-on) behaviour
        return True


_DEFAULT_FORWARD_BARRIER = _default_forward_barrier()


def dllm_sync_enabled(site: str) -> bool:
    """Whether to place a device barrier at `site`.

    With SGLANG_DLLM_SYNC_SITES unset the forward barrier follows the platform
    default; setting it takes full manual control.
    """
    if not _SYNC_SET:
        return site == "forward" and _DEFAULT_FORWARD_BARRIER
    return site in _SYNC_SITES or "all" in _SYNC_SITES


def dllm_device_synchronize() -> None:
    """Barrier on the active device (so NPU paths work), not unconditionally on CUDA."""
    # `current_accelerator()` is not an availability check: a CPU-only host with a
    # CUDA/ROCm wheel still reports cuda, so gate on `is_available()`.
    acc = getattr(torch, "accelerator", None)
    usable = False
    if acc is not None:
        try:
            usable = bool(acc.is_available())
        except (RuntimeError, AttributeError):
            # Only the probe is guarded; a synchronize() fault is a real device
            # error and must propagate.
            usable = False
    if usable:
        acc.synchronize()
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _dllm_sync_forward() -> None:
    if dllm_sync_enabled("forward"):
        dllm_device_synchronize()


class _ReplayedOut:
    """Stand-in for model_runner.forward's output for a block-graph replay."""

    __slots__ = ("logits_output", "can_run_graph")

    def __init__(self, logits_output):
        self.logits_output = logits_output
        self.can_run_graph = True


def _loop_exhausted(algo) -> Exception:
    from sglang.srt.dllm.params import DllmContractError

    return DllmContractError(
        f"{type(algo).__name__}: a block did not finish within max_steps="
        f"{algo.max_steps(algo.block_size)} forwards. Returning it would hand back "
        "a block whose KV was never committed; raise max_steps or fix the "
        "algorithm's step accounting."
    )


def _dump_prefill(dump_dir: str, model_runner, forward_batch, fwd, out) -> None:
    """Test-only (syncs, host copies): one file per request of the prefill batch.
    `fwd` is the batch the model ran (possibly a padded view); rows are read via it."""
    import os

    n_prompt = int(forward_batch.input_ids.numel())
    rows = getattr(fwd, "dllm_prefill_rows", None) if fwd is not forward_batch else None
    rows = (
        torch.arange(n_prompt, device=forward_batch.input_ids.device)
        if rows is None
        else rows
    )
    logits = out.logits_output.full_logits[rows].float().cpu()
    ids = forward_batch.input_ids.cpu()
    lens = forward_batch.extend_seq_lens_cpu or [n_prompt]
    backend = model_runner.attn_backend
    conv_ids = list(getattr(model_runner.model.config, "linear_layer_ids", []) or [])
    conv = None
    if conv_ids and hasattr(backend, "conv_state_pool"):
        meta = backend.conv_state_metadata(conv_ids[0], fwd)
        conv = backend.conv_state_pool()[:, meta.cache_indices].float().cpu()
    os.makedirs(dump_dir, exist_ok=True)
    start = 0
    for b, L in enumerate(int(x) for x in lens):
        rec = dict(
            ids=ids[start : start + L],
            logits=logits[start : start + L],
            replayed=bool(fwd.dllm_graph_replayed),
            batch_size=len(lens),
        )
        if conv is not None:
            rec["conv"] = conv[:, b : b + 1]
        torch.save(rec, os.path.join(dump_dir, f"{len(os.listdir(dump_dir)):05d}.pt"))
        start += L


class DllmAlgorithm:
    """dLLM algorithm: subclasses implement ``step``; the base owns the
    synchronous and FDFO (``--dllm-fdfo``) execution loops in ``run``.
    """

    # True while _block_loop prepares a block graph's forward 0.
    _graph_forward0 = False

    def __init__(self, config: DllmConfig):
        # Block-graph keys seen warm (one eager pass) and captured; the runner
        # holds the graphs.
        self.algorithm_name = config.algorithm
        self._block_graph_captured: set = set()
        self._block_graph_warm: set = set()
        # Declared here so the block-graph key can read them on any algorithm;
        # subclasses that have these features set them after super().__init__.
        self.adaptive_stop = False
        self.self_cond = False
        self.commit_fusion = False
        self.block_size = config.block_size
        # Commit fusion windows are [block k-1 clean causal | block k noised];
        # the generation region is always the last block_size positions.
        self.window_size = config.block_size
        self.mask_id = config.mask_id
        # Widths a generation forward may have; under commit fusion a
        # request's first block is block_size and later ones 2*block_size.
        self._widths = (config.block_size,)
        self.fdfo = config.first_done_first_out_mode
        # Set by _block_tail when the emitted tokens are still being copied to the
        # host; the scheduler waits on it (GenerationBatchResult.copy_done).
        self.copy_done: Optional[torch.cuda.Event] = None
        self._host_bufs = {}
        # See DllmConfig.prefix_attention: one flag for every clean-context
        # encode (prompt prefill here, commit forward in the algorithm).
        self.prefix_bidirectional = getattr(config, "prefix_bidirectional", False)

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        return [None] * forward_batch.batch_size

    def max_steps(self, block_size: int) -> int:
        return block_size + 1

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        """One denoise step, advancing ``forward_batch.input_ids``/``states`` in
        place. Returns, per block, whether it was already complete *on entry* --
        i.e. this forward persisted its final KV cache and it can be emitted.
        """
        raise NotImplementedError

    def prepare_forward(
        self,
        forward_batch: ForwardBatch,
        states: List[Any],
        first: bool,
    ) -> None:
        """Hook run before every model forward (``first=True`` before the first step).

        Default no-op (masked diffusion). Uniform-state algorithms override it
        to initialize the canvas and set per-forward conditioning
        (``dllm_sigma`` / ``dllm_selfcond`` / ``dllm_causal_override``).
        """
        return None

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]] = None,
    ) -> DllmRunOutput:
        self.copy_done = None
        if self.fdfo:
            return self._run_fdfo(model_runner, forward_batch, algo_states)
        return self._run_sync(model_runner, forward_batch)

    def forward_width(self, forward_batch: ForwardBatch) -> int:
        """Width of this forward's window per request.

        Returns 0 when the extend is not equal windows of an allowed width,
        which identifies a variable-length clean-prompt prefill.
        """
        bs = forward_batch.batch_size
        n = forward_batch.input_ids.numel()
        if bs <= 0 or n % bs:
            return 0
        w = n // bs
        return w if w in self._widths else 0

    def _block_start_list(self, forward_batch: ForwardBatch) -> List[int]:
        batch_size = forward_batch.batch_size
        w = self.forward_width(forward_batch) or self.window_size
        input_ids = forward_batch.input_ids.view(batch_size, w)[:, -self.block_size :]
        return (input_ids != self.mask_id).sum(dim=1).tolist()

    def block_start_list(self, forward_batch: ForwardBatch) -> List[int]:
        """Per-row count of already-committed tokens in the window.

        Called after ``prepare_forward(first=True)``; an algorithm that rewrites
        the canvas there must return the counts measured before rewriting.
        """
        return self._block_start_list(forward_batch)

    def on_clean_prefill(self, forward_batch: ForwardBatch) -> None:
        """Hook: a clean prompt extend (or prompt chunk) is about to run.

        The only point to invalidate per-request carry-over state, since the
        prefill branch returns before prepare_forward.
        """

    def after_clean_prefill(
        self, forward_batch: ForwardBatch, logits_output=None
    ) -> None:
        """Hook: the prompt prefill forward has just run. Default no-op.

        ``logits_output``'s last position per request predicts the first
        generated token; it is the only verifier that token will have.
        """

    def _run_sync(
        self, model_runner: ModelRunner, forward_batch: ForwardBatch
    ) -> DllmRunOutput:
        batch_size = forward_batch.batch_size

        # A non-window extend is a clean prompt prefill: one forward, nothing
        # emitted, its KV/conv state persists as the committed prefix.
        if self.forward_width(forward_batch) == 0:
            if bool((forward_batch.input_ids == self.mask_id).any()):
                from sglang.srt.dllm.params import DllmContractError

                raise DllmContractError(
                    "variable-length dLLM extend contains canvas placeholders "
                    f"-- a generation window must be one of {self._widths} "
                    "tokens per request (block_size, or 2*block_size for a "
                    "fused block carrying its predecessor)"
                )
            # Clean context: causal (None = the layer default, byte-identical
            # to an AR prefill) or bidirectional inside the prefill segment.
            forward_batch.dllm_causal_override = (
                False if self.prefix_bidirectional else None
            )
            forward_batch.dllm_graph_replayed = False
            self.on_clean_prefill(forward_batch)
            _dllm_sync_forward()
            # The model may run a padded graph-shaped view; everything after
            # the forward reads the original batch.
            fwd = self.prefill_view(forward_batch)
            out = model_runner.forward(fwd, pp_proxy_tensors=None)
            if fwd is not forward_batch:
                forward_batch.dllm_graph_replayed = fwd.dllm_graph_replayed
            _dump_dir = envs.SGLANG_DLLM_DEBUG_PREFILL_DUMP_DIR.get()
            if _dump_dir:
                _dump_prefill(_dump_dir, model_runner, forward_batch, fwd, out)
            self.after_clean_prefill(forward_batch, out.logits_output)
            return out.logits_output, [], None, None, out.can_run_graph

        states = self.init_step_state(forward_batch)
        self.prepare_forward(forward_batch, states, first=True)
        # After the hook, so a uniform-state algorithm answers without a sync.
        start_list = self.block_start_list(forward_batch)

        # Block-graph eligibility is decided before forward 0, since a captured
        # block includes forward 0.
        key = self._block_graph_key(forward_batch, states, start_list)
        action = self._block_graph_action(key) if key is not None else "eager"
        if action == "capture":
            # The whole block as one graph, forward 0 included, staged by the
            # per-forward runner. Returns None (-> eager) on an ineligible batch.
            runner = getattr(model_runner, "decode_cuda_graph_runner", None)
            self._graph_live = (forward_batch, states)
            lp = (
                runner.run_dllm_block(
                    forward_batch,
                    key,
                    self._block_loop,
                    narrow_width=self.forward_view_width(forward_batch),
                )
                if runner is not None
                else None
            )
            if lp is not None:
                self._finalize_replayed_block(forward_batch, states)
                self._block_graph_captured.add(key)
                return self._block_tail(
                    forward_batch, _ReplayedOut(lp), start_list, batch_size
                )
        _dllm_sync_forward()
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        # No mask to denoise: return empty so process_batch_result_dllm skips the
        # stream branch (matches the pre-refactor behavior).
        if all(start == self.block_size for start in start_list):
            # A clean prompt of exactly one window width lands here, not in the
            # prefill branch above, and still needs after_clean_prefill.
            if forward_batch.dllm_prompt_prefill:
                self.after_clean_prefill(forward_batch, out.logits_output)
            return out.logits_output, [], None, None, out.can_run_graph

        # NPU: attention metadata is stable across a block's denoise steps (the
        # first forward above already planned it), so mark it ready once and let
        # every later forward skip re-planning.
        if _is_npu:
            forward_batch.mark_forward_metadata_ready(replan_equivalent=True)

        if action == "warm":
            self._block_graph_warm.add(key)
        out = self._eager_loop(model_runner, forward_batch, out, states)
        # step() may issue its own forward (AR verification), superseding the
        # loop's last output.
        out = self.override_block_out(out)
        return self._block_tail(forward_batch, out, start_list, batch_size)

    def _block_tail(self, forward_batch, out, start_list, batch_size):
        # Emit only the generation half; under commit fusion the window also
        # carries block k-1's already-committed tokens.
        next_token_ids = forward_batch.input_ids.view(
            batch_size, self.forward_width(forward_batch) or self.window_size
        )[:, -self.block_size :]
        # AR verify may commit only a prefix of the canvas; the rejected tail
        # must not be emitted. None means the whole canvas.
        ends = self.commit_lengths()
        # One async device->host copy into a reused pinned buffer; the scheduler
        # waits on copy_done before reading rows.
        rows = self._copy_to_host(next_token_ids)
        if ends is None:
            next_token_ids_list = [rows[i][start_list[i] :] for i in range(batch_size)]
        else:
            if len(ends) != batch_size:
                from sglang.srt.dllm.params import DllmContractError

                raise DllmContractError(
                    f"committed lengths {ends} do not cover the batch "
                    f"({batch_size} rows). A short list would silently emit the "
                    "wrong slice for the rows it does not name."
                )
            next_token_ids_list = [
                rows[i][start_list[i] : ends[i]] for i in range(batch_size)
            ]
        # The committed length rides accept_length_per_req_cpu so the scheduler
        # can place the tokens and free the discarded positions' KV.
        return (
            out.logits_output,
            next_token_ids_list,
            ends,
            None,
            out.can_run_graph,
        )

    def _copy_to_host(self, ids: torch.Tensor):
        if not ids.is_cuda:
            return ids.tolist()
        # The buffer is reused by the next block; safe only because the dLLM path
        # forces the non-overlap scheduler (_dllm_overlap_disable), which reads
        # every row before the next forward.
        key = (tuple(ids.shape), ids.dtype)
        buf = self._host_bufs.get(key)
        if buf is None:
            buf = self._host_bufs[key] = torch.empty(
                ids.shape, dtype=ids.dtype, pin_memory=True
            )
        buf.copy_(ids, non_blocking=True)
        self.copy_done = torch.cuda.Event()
        self.copy_done.record()
        return buf

    def override_block_out(self, out):
        """Hook: the output the block should publish, if the algorithm ran a
        forward the base loop did not. Default: the loop's own last output."""
        return out

    def commit_lengths(self):
        """Per-row count of canvas positions this block commits, or None for all;
        read by the emission slice and the scheduler's KV bookkeeping."""
        return None

    def _eager_loop(self, model_runner, forward_batch, out, states):
        """Forwards 1..N dispatched one at a time; also the block-graph warmup."""
        for _loop_i in range(self.max_steps(self.block_size)):
            done = self.step(forward_batch, out.logits_output.full_logits, states)
            if all(done):
                break
            self.prepare_forward(forward_batch, states, first=False)
            _dllm_sync_forward()
            # The model may run a narrower view (commit fusion); the plan-reuse
            # marker below belongs to that view.
            fwd = self.forward_view(forward_batch, states)
            out = model_runner.forward(fwd, pp_proxy_tensors=None)
            if fwd is not forward_batch:
                # step() and the trace read the full batch's replay flag.
                forward_batch.dllm_graph_replayed = fwd.dllm_graph_replayed
            # The attention plan stays valid for the rest of the block. Use
            # attn_metadata_ready, not forward_metadata_ready: the latter also
            # skips fill_from and would freeze dllm_sigma/dllm_selfcond.
            if (
                _dllm_preplan_attn()
                and _loop_i == 0
                # Only after a real replay: an eager forward (e.g. phase-filtered
                # replay) never ran load_batch, so no live plan exists to reuse.
                and fwd.dllm_graph_replayed
            ):
                fwd.attn_metadata_ready = True
        else:
            raise _loop_exhausted(self)
        return out

    def prefill_view(self, forward_batch: ForwardBatch):
        """The batch a clean prompt prefill runs on. Default: the batch itself."""
        return forward_batch

    def forward_view_width(self, forward_batch: ForwardBatch) -> Optional[int]:
        """Per-request width of the batch forward_view returns for forwards 1..N,
        when it differs from forward 0's. Default: None (same width)."""
        return None

    def forward_view(self, forward_batch: ForwardBatch, states):
        """The batch forwards 1..N of a block run on. Default: the batch itself.

        Commit fusion returns a narrower view; step() still receives the full batch.
        """
        return forward_batch

    def _block_graph_enabled(self) -> bool:
        return dllm_graph_flag(envs.SGLANG_DLLM_ENABLE_BLOCK_GRAPH, self.algorithm_name)

    def _block_graph_key(self, forward_batch, states, start_list):
        """Everything a captured block bakes in, or None if ineligible."""
        if not self._block_graph_enabled():
            return None
        if self.adaptive_stop:
            return None
        if _SYNC_SET:
            # A synchronize inside a capture raises; refuse rather than drop
            # the requested syncs.
            return None
        # The algorithm refuses its own sync/host-copy gates.
        if self._capture_unsafe_reasons():
            return None
        # The default CUDA generator is registered with the graph, but a per-row
        # torch.Generator would be baked at capture.
        if any(getattr(s, "generator", None) is not None for s in states):
            return None
        # Greedy and sampled record different kernels; a mixed block is ineligible.
        modes = {self._block_sampler_mode(s, forward_batch) for s in states}
        if len(modes) != 1 or None in modes:
            return None
        # Under commit fusion the width tells a fused block (2*block_size, its
        # later forwards narrow) from a request's narrow first block.
        return (
            forward_batch.batch_size,
            self.forward_width(forward_batch) or self.window_size,
            self.max_steps(self.block_size),
            tuple(int(s) for s in start_list),
            self.self_cond,
            modes.pop(),
            self._block_baked_tag(states),
        )

    def _block_baked_tag(self, states) -> Optional[tuple]:
        """Per-row values a capture bakes in beyond the batch shape (None if fixed)."""
        return None

    def _block_sampler_mode(self, state, forward_batch) -> Optional[str]:
        """Hashable tag of the sampler kernels this row records, or None.

        Default: greedy only; override once a sampled path is graph-safe."""
        greedy = forward_batch.dllm_greedy
        return "greedy" if greedy is not None and all(greedy) else None

    def _capture_unsafe_reasons(self) -> list:
        """Names of gates that put a device sync or host read inside a capture."""
        return []

    def _block_loop(self, static_fb, run_forward):
        """One whole block on the static batch, for warmups and recording.

        Resets step state each call; must not re-run ``prepare_forward(first=True)``,
        since the placeholders are already replaced."""
        live_fb, states = self._graph_live
        st = states[0]
        st.i, st.phase, st.sc = 0, self._block_start_phase(st), None
        self._broadcast_state(states)
        # Re-derive forward 0's conditioning inside the recording (warmups mutate
        # it), with first-forward flags but without redoing the block setup.
        self._graph_forward0 = True
        try:
            self.prepare_forward(static_fb, states, first=False)
        finally:
            self._graph_forward0 = False
        out = run_forward(static_fb)
        for _ in range(self.max_steps(self.block_size)):
            done = self.step(static_fb, out.full_logits, states)
            if all(done):
                break
            self.prepare_forward(static_fb, states, first=False)
            out = run_forward(self.forward_view(static_fb, states))
        else:
            raise _loop_exhausted(self)
        return out

    def _block_start_phase(self, st) -> str:
        """Phase a block's first step() runs in."""
        return "denoise"

    def _finalize_replayed_block(self, forward_batch, states) -> None:
        """Advance the host-side state a replay cannot."""
        st = states[0]
        st.i = st.spb if getattr(st, "spb", None) is not None else st.i
        st.phase = "done"
        self._broadcast_state(states)

    def _prepare_is_uniform(self) -> bool:
        """True if prepare_forward conditions the batch as one block."""
        return False

    def _broadcast_state(self, states: List[Any]) -> None:
        """Hook: copy ``states[0]`` onto the other rows (lockstep algorithms)."""

    def _block_graph_action(self, key) -> str:
        """eager | warm | capture; a key needs one warm eager block before
        "capture", which records on first sighting and replays after."""
        if key is None:
            return "eager"
        if key in self._block_graph_captured:
            return "capture"  # captured => replay through the runner
        if key in self._block_graph_warm:
            return "capture"
        if len(self._block_graph_warm | self._block_graph_captured) >= (
            _BLOCK_GRAPH_MAX_KEYS
        ):
            return "eager"
        return "warm"

    def _run_fdfo(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]],
    ) -> DllmRunOutput:
        batch_size = forward_batch.batch_size

        # Clean prompt prefill, as in _run_sync: never reaches prepare_forward or
        # step, and every row reports done with its trailing prompt tokens.
        if self.forward_width(forward_batch) == 0:
            if bool((forward_batch.input_ids == self.mask_id).any()):
                from sglang.srt.dllm.params import DllmContractError

                raise DllmContractError(
                    "variable-length dLLM extend contains canvas placeholders "
                    f"-- a generation window must be one of {self._widths} "
                    "tokens per request (block_size, or 2*block_size for a "
                    "fused block carrying its predecessor)"
                )
            forward_batch.dllm_causal_override = (
                False if self.prefix_bidirectional else None
            )
            forward_batch.dllm_graph_replayed = False
            self.on_clean_prefill(forward_batch)
            _dllm_sync_forward()
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            self.after_clean_prefill(forward_batch, out.logits_output)
            lens = forward_batch.extend_seq_lens_cpu
            if lens is None:
                lens = forward_batch.extend_seq_lens.tolist()
            ids = forward_batch.input_ids.tolist()
            toks, at = [], 0
            for n in lens:
                toks.append(ids[at + max(0, n - self.block_size) : at + n])
                at += n
            return (
                out.logits_output,
                toks,
                [self.block_size] * batch_size,
                [None] * batch_size,
                out.can_run_graph,
            )

        if algo_states is None:
            algo_states = [None] * batch_size
        fresh: Optional[List[Any]] = None
        states: List[Any] = []
        for i, carried in enumerate(algo_states):
            if carried is None:
                if fresh is None:
                    fresh = self.init_step_state(forward_batch)
                states.append(fresh[i])
            else:
                states.append(carried)

        # A batch mixing fresh and carried rows is refused for uniform-prepare
        # algorithms; the scheduler's _dllm_batch_uniform avoids assembling one.
        n_fresh = sum(1 for c in algo_states if c is None)
        if 0 < n_fresh < batch_size and self._prepare_is_uniform():
            from sglang.srt.dllm.params import DllmContractError

            raise DllmContractError(
                f"FDFO batch mixes {n_fresh} fresh and {batch_size - n_fresh} "
                "carried rows; this algorithm prepares the batch as one block "
                "and cannot condition the two differently in one forward."
            )
        self.prepare_forward(forward_batch, states, first=(n_fresh == batch_size))

        _dllm_sync_forward()
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        if forward_batch.dllm_prompt_prefill:
            # A clean prompt of exactly one window width: same as _run_sync's
            # early return, every row done with its prompt tokens.
            self.after_clean_prefill(forward_batch, out.logits_output)
            W = self.forward_width(forward_batch) or self.window_size
            toks = forward_batch.input_ids.view(batch_size, W)[
                :, -self.block_size :
            ].tolist()
            return (
                out.logits_output,
                toks,
                [self.block_size] * batch_size,
                [None] * batch_size,
                out.can_run_graph,
            )
        done = self.step(forward_batch, out.logits_output.full_logits, states)
        # Lockstep algorithms run up to K steps per scheduler round (a round
        # costs more host time than a forward); masked algorithms keep K = 1.
        k_steps = 1
        if self._prepare_is_uniform():
            from sglang.srt.environ import envs as _envs

            k_steps = max(1, int(_envs.SGLANG_DLLM_FDFO_STEPS_PER_CALL.get()))
        for _k in range(1, k_steps):
            if any(done):
                break
            self.prepare_forward(forward_batch, states, first=False)
            _dllm_sync_forward()
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            # same rule as _eager_loop: after the first in-round forward that
            # actually replayed, the plan is valid for the rest of the round
            if _dllm_preplan_attn() and _k == 1 and forward_batch.dllm_graph_replayed:
                forward_batch.attn_metadata_ready = True
            done = self.step(forward_batch, out.logits_output.full_logits, states)

        accept_length_per_req_cpu = [self.block_size if d else 0 for d in done]
        next_token_ids_list = forward_batch.input_ids.view(
            batch_size, self.forward_width(forward_batch) or self.window_size
        )[:, -self.block_size :].tolist()
        states_out = [None if done[i] else states[i] for i in range(batch_size)]

        return (
            out.logits_output,
            next_token_ids_list,
            accept_length_per_req_cpu,
            states_out,
            out.can_run_graph,
        )
