from __future__ import annotations

import enum
import logging
from array import array
from typing import TYPE_CHECKING, Optional

from sglang.srt.dllm.config import DllmConfig

logger = logging.getLogger(__name__)

# Warn once per process, not per request.
_WARNED_SAMPLING_SEED = False

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        # First entry into the dLLM waiting queue (prefill coalescing reads it).
        self.dllm_enqueue_time: Optional[float] = None
        self.dllm_incomplete_ids = array("q")
        self.dllm_algo_state = None
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config

        if self.dllm_config is not None:
            # Per-request sampling_seed is not honoured (draws use the global RNG
            # seeded by --random-seed); warned rather than refused.
            global _WARNED_SAMPLING_SEED
            if (
                not _WARNED_SAMPLING_SEED
                and getattr(
                    getattr(self, "sampling_params", None), "sampling_seed", None
                )
                is not None
            ):
                _WARNED_SAMPLING_SEED = True
                logger.warning(
                    "dLLM: per-request sampling_seed is IGNORED on the "
                    "diffusion path (the algorithm's generator is never seeded "
                    "from sampling_params). Sampling falls back to the global "
                    "RNG seeded by --random-seed, so results are reproducible "
                    "per SERVER run and single request stream, not per "
                    "request. Do not rely on sampling_seed for eval "
                    "determinism."
                )
            if getattr(self.dllm_config, "anchored", False):
                # The whole prompt is prefilled first, so every request
                # begins in PREFILL.
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL
            elif len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def dllm_prefix_holdback(self: Req) -> int:
        """How many trailing committed tokens to keep out of the prefix: one block
        under commit fusion, which writes block k-1's clean KV in block k's window."""
        cfg = self.dllm_config
        if cfg is None:
            return 0
        acfg = getattr(cfg, "algorithm_config", None) or {}
        if not acfg.get("commit_fusion", False):
            return 0
        return int(cfg.block_size)

    def determine_dllm_phase(self: Req):
        if self.dllm_incomplete_ids:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE
            return

        prefix_length = len(self.prefix_indices)

        if getattr(self.dllm_config, "anchored", False):
            # Anchored grid: PREFILL until the whole prompt is committed
            # (extends may be any length), then pure placeholder blocks.
            if prefix_length < len(self.origin_input_ids):
                self.dllm_phase = DllmReqPhase.STAGING_PREFILL
            else:
                self.dllm_phase = DllmReqPhase.STAGING_DECODE
            return

        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.full_untruncated_fill_ids) < min_required_length:
            # still incoming stage
            return

        input_block = self.full_untruncated_fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if self.dllm_incomplete_ids:
            prefix_len = len(self.prefix_indices)
            assert len(self.dllm_incomplete_ids) == self.dllm_config.block_size
            self.full_untruncated_fill_ids = (
                self.full_untruncated_fill_ids[:prefix_len] + self.dllm_incomplete_ids
            )
            # extend_range is (re)computed by the staging adder
            # (add_dllm_staging_req) before this req is scheduled, mirroring the
            # non-incomplete path which also defers it to the adder.
            return

        if getattr(self.dllm_config, "anchored", False):
            # Answer-anchored grid: no placeholder block until the entire
            # prompt is committed (prefill runs as ordinary causal extends);
            # generation blocks then start exactly at len(prompt + outputs).
            committed = len(self.prefix_indices)
            if committed < len(self.origin_input_ids):
                self.full_untruncated_fill_ids = array("q", self.origin_input_ids)
                self.dllm_initialized = True
                return
            self.dllm_block_offset = len(self.origin_input_ids) + len(self.output_ids)
            self.full_untruncated_fill_ids = (
                self.origin_input_ids
                + self.output_ids
                + array("q", [self.dllm_config.mask_id] * self.dllm_config.block_size)
            )
            self.dllm_initialized = True
            return

        self.dllm_block_offset = (
            0
            if not self.dllm_initialized
            else self.dllm_block_offset + self.dllm_config.block_size
        )
        self.full_untruncated_fill_ids = (
            self.origin_input_ids
            + self.output_ids
            + array("q", [self.dllm_config.mask_id] * self.dllm_config.block_size)
        )
        self.dllm_initialized = True

    def _update_block_offset_for_dllm(self):
        prefix_len = len(self.prefix_indices)
        if getattr(self.dllm_config, "anchored", False):
            # anchored windows start at arbitrary (prompt-determined) offsets;
            # positions come from extend_range, offsets are bookkeeping only
            if prefix_len > self.dllm_block_offset:
                self.dllm_block_offset = prefix_len
            return
        assert (
            prefix_len % self.dllm_config.block_size == 0
        ), f"Unexpected prefix len: {prefix_len}"
        if prefix_len > self.dllm_block_offset:
            self.dllm_block_offset = prefix_len
