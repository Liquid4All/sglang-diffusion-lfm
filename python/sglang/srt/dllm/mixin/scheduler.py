from __future__ import annotations

import json
import logging
import time
from array import array
from typing import TYPE_CHECKING, List, Optional, Set, Union

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.runtime_context import get_exec, get_schedule

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler


class SchedulerDllmMixin:
    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if get_exec().dllm.dllm_algorithm is not None
            else None
        )
        self.dllm_manager = DllmManager(dllm_config=self.dllm_config)
        _path = envs.SGLANG_DLLM_ROUND_TRACE.get()
        self._dllm_round_trace = open(_path, "a") if _path else None
        self._dllm_round_is_prefill = False

    def validate_dllm_serving_config(self: Scheduler) -> None:
        """Refuse at startup a dLLM configuration that cannot serve a block."""
        if self.dllm_config is None:
            return
        from sglang.srt.dllm.params import DllmContractError, dllm_page_size_refusal

        acfg = self.dllm_config.algorithm_config or {}
        if acfg.get("commit_fusion", False) and not getattr(
            self.tree_cache, "supports_dllm_prefix_holdback", False
        ):
            raise DllmContractError(
                "commit_fusion needs a prefix cache that holds the dLLM "
                f"prefix back one block; {type(self.tree_cache).__name__} "
                "does not (supports_dllm_prefix_holdback is False). Run with "
                "--disable-radix-cache, which selects ChunkCache."
            )
        # ChunkCache advertises the holdback unconditionally, so page size is
        # checked separately; a missing allocator raises rather than passing.
        if any(acfg.get(k, False) for k in ("commit_fusion", "ar_verify", "ar_stop")):
            refusal = dllm_page_size_refusal(
                acfg,
                int(self.page_size),
                int(self.token_to_kv_pool_allocator.page_size),
            )
            if refusal:
                raise DllmContractError(refusal)

    def get_new_batch_dllm(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        """Generate a new batch for DLLM (Diffusion LLM) scheduling."""
        _tg = time.time() if self._dllm_round_trace is not None else None
        if self.enable_priority_preemption:
            running_batch.batch_is_full = False

        # Early exit if batch is full or no requests available
        if self._should_skip_prefill(running_batch=running_batch):
            return None

        running_bs = len(running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)

        # Create prefill adder with resource constraints
        adder = self._create_dllm_prefill_adder(running_bs, running_batch=running_batch)

        # Initialize DLLM manager and transfer requests
        self.dllm_manager.init_next_round()
        self._fetch_waiting_reqs()

        # Process batches
        forward_mode = self._process_dllm_batches(adder, running_batch=running_batch)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        # Record metrics and update state
        set_time_batch(can_run_list, "set_forward_entry_time")
        self._update_state_for_batch(can_run_list, adder)

        # Create and prepare batch
        new_batch = self._create_dllm_batch(
            can_run_list, forward_mode, adder=adder, running_batch=running_batch
        )
        if self._dllm_round_trace is not None:
            new_batch.dllm_round_trace = dict(
                tg=_tg,
                t0=time.time(),
                prefill=self._dllm_round_is_prefill,
                rids=[r.rid for r in can_run_list],
            )
        return new_batch

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        rec = getattr(batch, "dllm_round_trace", None)
        if rec is not None:
            rec["tr"] = time.time()  # forward done, result processing starts
        if result.copy_done is not None:
            result.copy_done.synchronize()
        if rec is not None:
            rec["tc"] = time.time()  # GPU results on the host

        fdfo_mode = self.dllm_config.first_done_first_out_mode
        assert (
            not fdfo_mode or result.accept_length_per_req_cpu is not None
        ), "FDFO dLLM result is missing accept lengths."

        # FDFO also commits unresolved blocks so their KV can be reused.
        if fdfo_mode or result.next_token_ids:
            block_size = self.dllm_config.block_size
            algo_states = result.dllm_algo_state

            # Backstop to validate_dllm_serving_config, checked before any
            # mutation so it cannot leave a request half-committed.
            if result.accept_length_per_req_cpu is not None and not fdfo_mode:
                _ps = int(self.token_to_kv_pool_allocator.page_size)
                if _ps > 1:
                    raise ValueError(
                        f"ar_verify reached the commit path with page_size="
                        f"{_ps}; validate_dllm_serving_config should have "
                        "refused this at startup."
                    )

            self.token_to_kv_pool_allocator.free_group_begin()
            try:
                self._dllm_commit_results(
                    batch, result, fdfo_mode, block_size, algo_states
                )
                # Success path only: a partial commit must not be streamed, while
                # the allocator free group below closes on every path.
                if rec is not None:
                    rec["ts"] = time.time()
                self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
            finally:
                self.token_to_kv_pool_allocator.free_group_end()

        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=result.can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )
        if rec is not None:
            rec["t1"] = time.time()
            rec["done"] = [r.rid for r in batch.reqs if r.finished()]
            self._dllm_round_trace.write(json.dumps(rec) + "\n")
            self._dllm_round_trace.flush()

    def _dllm_commit_results(
        self: Scheduler, batch, result, fdfo_mode, block_size, algo_states
    ):
        """Apply one dLLM batch result to its requests."""
        for idx in range(batch.batch_size()):
            req = batch.reqs[idx]

            if not fdfo_mode:
                next_token_ids = result.next_token_ids[idx]
                if not isinstance(next_token_ids, list):
                    next_token_ids = next_token_ids.tolist()
                new_tokens = len(next_token_ids)
                if new_tokens == 0:
                    continue

                # Without AR verification the emitted ids are the window's
                # trailing new_tokens; under verification the block commits a
                # prefix, so they are left-aligned and the tail is discarded.
                committed = result.accept_length_per_req_cpu
                if committed is None:
                    hi = req.extend_range.end
                else:
                    hi = req.extend_range.end - block_size + int(committed[idx])
                lo = hi - new_tokens
                req.full_untruncated_fill_ids[lo:hi] = array("q", next_token_ids)
                self.metrics_reporter.num_generated_tokens += new_tokens

                req.output_ids.extend(next_token_ids)
                req.update_finish_state(new_accepted_len=new_tokens)

                if hi < req.extend_range.end:
                    self._dllm_truncate_block(req, hi)

                self._dllm_finish_if_needed(req, block_size)
                continue

            next_token_ids = result.next_token_ids[idx]

            if result.accept_length_per_req_cpu[idx] == 0:
                # Unresolved: keep partial state and KV for the next FDFO round.
                req.dllm_incomplete_ids = array("q", next_token_ids)
                req.dllm_algo_state = (
                    algo_states[idx] if algo_states is not None else None
                )
                continue

            req.dllm_incomplete_ids = array("q")
            req.dllm_algo_state = None

            # A pure prompt-prefill row (anchored grid) generated nothing; the
            # block-sized mirror below would index before the fill ids' start
            # for a prompt shorter than a block.
            if req.extend_range.end <= len(req.origin_input_ids):
                continue
            assert len(next_token_ids) == block_size

            # Mirror the resolved block into the committed fill ids so the
            # prefix cache keys on the real tokens, not the mask block, next
            # round. Index relative to extend_range.end (the truncated/
            # committed length), which can be shorter than
            # full_untruncated_fill_ids when the staging adder truncates the
            # block to the KV budget.
            req.full_untruncated_fill_ids[
                req.extend_range.end - block_size : req.extend_range.end
            ] = array("q", next_token_ids)

            len_input = len(req.origin_input_ids)
            len_fill = req.extend_range.end
            if len_fill <= len_input:
                continue

            if len_fill - len(next_token_ids) < len_input:
                next_token_ids = next_token_ids[len_input - len_fill :]

            self.metrics_reporter.num_generated_tokens += len(next_token_ids)
            req.output_ids.extend(next_token_ids)
            req.update_finish_state(new_accepted_len=len(next_token_ids))

            self._dllm_finish_if_needed(req, block_size)

    def _dllm_finish_if_needed(self: Scheduler, req: Req, block_size: int) -> None:
        """Release a finished request; first, end one whose next block would
        run past the model's context length with a length finish."""
        if (
            not req.finished()
            and req.seqlen + block_size > self.model_config.context_len
        ):
            req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))
        if req.finished():
            release_kv_cache(req, self.tree_cache)
            req.time_stats.set_completion_time()

    def _dllm_truncate_block(self: Scheduler, req: Req, new_end: int) -> None:
        """Give back the tail of a block past ``new_end`` that the AR head rejected.

        The extend range and KV lengths must shrink with the freed slots, or the
        next window reads rejected tokens as context. Mirrors
        StreamingSession._trim_overshoot.
        """
        if req.kv is None:
            raise ValueError(
                "ar_verify truncated a block for a request with no KV record; "
                "alloc_for_extend sets req.kv on every extend, so reaching "
                "here means the block ran without one."
            )
        # page_size is validated once per batch by the caller, before any
        # request state moves; token-aligned freeing below assumes it.
        end = req.kv.kv_allocated_len
        if new_end < end:
            tail = self.req_to_token_pool.req_to_token[req.req_pool_idx, new_end:end]
            self.token_to_kv_pool_allocator.free(tail)
        req.kv.kv_allocated_len = min(req.kv.kv_allocated_len, new_end)
        req.kv_committed_len = min(req.kv_committed_len, new_end)
        req.kv.swa_evicted_seqlen = min(req.kv.swa_evicted_seqlen, new_end)
        if getattr(req, "cache_protected_len", 0):
            req.cache_protected_len = min(req.cache_protected_len, new_end)
        del req.full_untruncated_fill_ids[new_end:]
        req.set_extend_range(req.extend_range.start, new_end)

    def _fetch_waiting_reqs(self: Scheduler):
        # Calculate how many requests can be added to DLLM manager
        # An unset dllm cap defers to the scheduler's resolved (memory-aware)
        # max_running_requests; this bounds the queue, not admission.
        _cap = self.dllm_config.max_running_requests
        if _cap is None:
            _cap = self.max_running_requests
        max_dllm_capacity = _cap - len(self.dllm_manager.waiting_queue)
        num_requests_to_add = min(max_dllm_capacity, len(self.waiting_queue))

        if num_requests_to_add > 0:
            requests_to_add = self.waiting_queue[:num_requests_to_add]
            self.dllm_manager.add_waiting_reqs(requests_to_add)
            self.waiting_queue = self.waiting_queue[num_requests_to_add:]

    def _should_skip_prefill(self: Scheduler, running_batch: ScheduleBatch) -> bool:
        """Check if DLLM prefill should be skipped."""
        if (
            running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True

        running_bs = len(running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.enable_priority_preemption
        ):
            running_batch.batch_is_full = True
            return True

        return False

    def _create_dllm_prefill_adder(
        self: Scheduler, running_bs: int, running_batch: ScheduleBatch
    ) -> PrefillAdder:
        """Create a prefill adder configured for DLLM scheduling."""
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=get_schedule().prefill_max_requests,
            dllm_config=self.dllm_config,
            # Without it, _init_dllm_meta budgets for a single request.
            max_running_requests=self.max_running_requests,
        )

    def _process_dllm_batches(
        self: Scheduler, adder: PrefillAdder, running_batch: ScheduleBatch
    ) -> ForwardMode:
        """Process prefill or decode batches for DLLM."""
        forward_mode = ForwardMode.DLLM_EXTEND

        # Try prefill batch first
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs and not self._dllm_prefill_due(prefill_reqs):
            prefill_reqs = []  # decode now; the prompts coalesce for a later round
        self._dllm_round_is_prefill = bool(prefill_reqs)
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
                running_batch=running_batch,
            )
        else:
            # Fall back to decode batch
            decode_reqs = self.dllm_manager.get_decode_requests()
            self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
                running_batch=running_batch,
            )

        return forward_mode

    def _dllm_prefill_due(self: Scheduler, prefill_reqs: List[Req]) -> bool:
        """Whether this round should prefill the waiting prompts (see
        SGLANG_DLLM_PREFILL_BATCH / _MAX_WAIT_MS)."""
        k = envs.SGLANG_DLLM_PREFILL_BATCH.get()
        if k <= 1 or len(prefill_reqs) >= k:
            return True
        if not self.dllm_manager.get_decode_requests():
            return True  # nothing is decoding, so waiting only adds latency
        if any(r.dllm_phase == DllmReqPhase.STAGING_PREFILL for r in prefill_reqs):
            return True  # a chunked prompt mid-prefill must not stall
        wait_s = envs.SGLANG_DLLM_PREFILL_MAX_WAIT_MS.get() / 1e3
        now = time.perf_counter()
        # An unstamped request (not queued via add_waiting_reqs) is never held back.
        return any(
            r.dllm_enqueue_time is None or now - r.dllm_enqueue_time >= wait_s
            for r in prefill_reqs
        )

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
        running_batch: ScheduleBatch,
    ) -> None:
        """Process a batch, separating staging and incoming requests."""
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            staging_result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if staging_result != AddReqResult.CONTINUE:
                return

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(
                adder, incoming_reqs, running_batch=running_batch
            )

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder
    ) -> None:
        """Update state for the batch."""

        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
            self.dllm_manager.increment_inflight_middle_chunks()

    def _create_dllm_batch(
        self: Scheduler,
        can_run_list: List[Req],
        forward_mode: ForwardMode,
        adder: PrefillAdder,
        running_batch: ScheduleBatch,
    ) -> ScheduleBatch:
        """Create and prepare a new DLLM batch."""
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=self.dllm_config,
        )
        new_batch.prepare_for_extend()
        new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None

        # Record prefill stats for logging after forward
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            PrefillStats,
        )

        new_batch.prefill_stats = PrefillStats.from_adder(
            adder, running_batch.reqs, self.enable_priority_scheduling
        )

        return new_batch

    def process_dllm_incoming_reqs(
        self: Scheduler,
        adder: PrefillAdder,
        reqs: List[Req],
        running_batch: ScheduleBatch,
    ) -> AddReqResult:
        """Process incoming DLLM requests with resource allocation and preemption."""
        res = AddReqResult.CONTINUE
        for req in reqs:
            # Check if batch is full
            running_bs = len(running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                running_batch.batch_is_full = True

            # Try preemption if batch is full
            if running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            # Prepare and add request
            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    running_batch.batch_is_full = True
                break

        return res

    def process_dllm_staging_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process staging DLLM requests with resource allocation."""
        for req in reqs:
            res = adder.add_dllm_staging_req(req)
            if res == AddReqResult.NO_TOKEN:
                return res

        return AddReqResult.CONTINUE


class DllmManager:
    """
    Manager for Diffusion LLM request scheduling.

    Maintains two queues:
    - waiting_queue: The requests waiting to be scheduled with max running requests limit
    - staging_queue: Requests allocated resources by PrefillAdder
    """

    def __init__(self, dllm_config: Optional[DllmConfig] = None):
        self.dllm_config = dllm_config
        # May be None (unset -> the scheduler's runtime cap applies).
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else None
        )
        self.waiting_queue: List[Req] = []
        self.staging_queue: List[Req] = []

    def get_prefill_requests(self) -> List[Req]:
        """Get all prefill requests from waiting queue."""
        return [req for req in self.waiting_queue if req.is_dllm_prefill()]

    def get_decode_requests(self) -> List[Req]:
        """Get all decode requests from waiting queue."""
        return [req for req in self.waiting_queue if not req.is_dllm_prefill()]

    def add_waiting_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to waiting queue with redundancy check."""
        assert self.dllm_config is not None, "Diffusion LLM config is not set."

        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        now = time.perf_counter()
        for r in reqs_to_add:
            if r.dllm_enqueue_time is None:
                r.dllm_enqueue_time = now

        # Check for duplicate request IDs
        if self._has_duplicate_reqs(reqs_to_add):
            raise RuntimeError("Redundant requests detected in dLLM requests.")

        self.waiting_queue.extend(reqs_to_add)

    def add_staging_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to staging queue (allocated by PrefillAdder)."""
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.staging_queue.extend(reqs_to_add)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        """Check if any request ID already exists in waiting queue."""
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def any_staging_reqs(self) -> bool:
        """Check if there are requests in staging queue."""
        return self.dllm_config is not None and len(self.staging_queue) > 0

    def is_empty(self) -> bool:
        """Check if both queues are empty or DLLM is not configured."""
        if self.dllm_config is None:
            return True
        return len(self.waiting_queue) == 0

    def increment_inflight_middle_chunks(self) -> None:
        """Increment chunked count for all staging requests."""
        for req in self.staging_queue:
            req.inflight_middle_chunks += 1

    def filter_finished_reqs(self) -> None:
        """Remove finished requests from both queues."""
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]

    def pop_aborted_reqs(self, abort_all: bool, rid: str) -> List[Req]:
        aborted_reqs: List[Req] = []
        seen: Set[int] = set()

        for queue_name in ("waiting_queue", "staging_queue"):
            queue = getattr(self, queue_name)
            kept_queue = []
            for req in queue:
                if abort_all or req.rid.startswith(rid):
                    req_id = id(req)
                    if req_id not in seen:
                        aborted_reqs.append(req)
                        seen.add(req_id)
                else:
                    kept_queue.append(req)
            setattr(self, queue_name, kept_queue)

        return aborted_reqs

    def init_next_round(self) -> None:
        """Initialize staging requests for next round and clear staging queue."""
        for req in self.staging_queue:
            req.init_next_round_input()
        self.staging_queue = []
