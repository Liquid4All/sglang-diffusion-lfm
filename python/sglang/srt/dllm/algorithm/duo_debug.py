"""Measurement and debugging hooks for DuoBlock.

Per-block cost timing and profiling, the record sink the benchmarks read, per-block
KV/conv state dumps, and the device-sync barriers. All are off by default and switched
by SGLANG_DLLM_* variables (environ.py); none changes what the algorithm computes.
"""

import json
import logging
import os
import time
from typing import List

import torch

from sglang.srt.dllm.algorithm.base import dllm_device_synchronize, dllm_sync_enabled
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


# SGLANG_DLLM_SYNC_SITES=<comma list> syncs the device at named sites (forward,
# sigma, x_in, sc, upd); `forward` is on when unset, and setting it (even to "")
# takes manual control. The other sites are bisect-only and do not cover every
# write of their tensor: a barrier that removes a fault shows it suffices, not
# which access raced. Parser lives in base.py.
def sync(site: str) -> None:
    if dllm_sync_enabled(site):
        dllm_device_synchronize()


# SGLANG_DLLM_STATE_DUMP=<path>: write KV and short-conv state hashes at each block
# boundary, to compare cross-block state that token output can hide.
STATE_DUMP = envs.SGLANG_DLLM_STATE_DUMP.get()

# Per-block cost instrumentation. SGLANG_DLLM_TIMING=1 syncs around each phase
# (prep / forward / sampler; pessimistic total); SGLANG_DLLM_PROFILE=1 uses
# torch.profiler without syncs, the only one that shows launch-boundness.
TIMING = envs.SGLANG_DLLM_TIMING.get()
PROFILE = envs.SGLANG_DLLM_PROFILE.get()
INSTR = TIMING or PROFILE
# The first blocks pay triton JIT and autotuning; skip them.
T_SKIP = envs.SGLANG_DLLM_TIMING_SKIP.get()
TIMES: List[dict] = []
# The algorithm runs in the scheduler subprocess, so records go to a file; env
# vars are inherited by the child, module state is not.
T_OUT = envs.SGLANG_DLLM_INSTR_OUT.get()
# Per-record run id: /tmp is node-local, so concurrent jobs may share a sink.
T_RUN = envs.SGLANG_DLLM_RUN_ID.get()


def instr_emit(rec: dict) -> None:
    if not T_OUT:
        return
    try:
        rec = dict(rec, run=T_RUN, pid=os.getpid())
        with open(T_OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        # instrumentation must never take down a decode
        pass


def raw_bytes(t: torch.Tensor) -> bytes:
    """Exact stored bytes; bf16/fp8 are viewed as same-width ints (no .numpy())."""
    t = t.contiguous().cpu()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16).numpy().tobytes()
    if t.dtype is torch.float8_e4m3fn or t.dtype is torch.float8_e5m2:
        return t.view(torch.int8).numpy().tobytes()
    return t.numpy().tobytes()


KV_HEAD_TOKENS = 8


def dump_state(model_runner, forward_batch, block_idx: int) -> None:
    """Append this block's per-layer KV + conv state hashes to SGLANG_DLLM_STATE_DUMP."""
    import hashlib
    import json as _json

    try:
        slots = [int(v) for v in forward_batch.req_pool_indices.tolist()]
        seq_lens = [int(v) for v in forward_batch.seq_lens.tolist()]
        rec = {
            "block": block_idx,
            "slots": slots,
            "seq_lens": seq_lens,
            "kv": {},
            "conv": {},
        }

        # --- KV, per layer, for exactly the tokens this request owns ---------
        r2t = model_runner.req_to_token_pool.req_to_token
        pool = model_runner.token_to_kv_pool
        # Full-attention layers only: on HybridLinearKVPool a short-conv layer id
        # raises ValueError. Derive the ids from layer_types.
        _lt = list(getattr(model_runner.model_config.hf_config, "layer_types", []))
        full_ids = [i for i, t in enumerate(_lt) if t == "full_attention"]
        for li in full_ids:
            try:
                kb = pool.get_key_buffer(li)
                vb = pool.get_value_buffer(li)
            except (AttributeError, AssertionError, IndexError, ValueError):
                continue
            h = hashlib.sha256()
            for b, sl in enumerate(slots):
                idx = r2t[sl, : seq_lens[b]].to(torch.int64)
                h.update(raw_bytes(kb[idx]))
                h.update(raw_bytes(vb[idx]))
            rec["kv"][str(li)] = h.hexdigest()[:32]
            # Raw K/V of the first tokens per request, so a shared prompt
            # prefix's KV can be compared position by position.
            for b, sl in enumerate(slots):
                n = min(KV_HEAD_TOKENS, seq_lens[b])
                idx = r2t[sl, :n].to(torch.int64)
                rec.setdefault("kv_head", {}).setdefault(str(li), []).append(
                    {
                        "k": [raw_bytes(kb[i]).hex() for i in idx.tolist()],
                        "v": [raw_bytes(vb[i]).hex() for i in idx.tolist()],
                    }
                )

        # --- short-conv state, per layer, for this request's slot -----------
        conv = model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0]
        mamba_idx = model_runner.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )
        for li in range(conv.shape[0]):
            h = hashlib.sha256()
            for mi in mamba_idx.tolist():
                # same raw-byte rule: do not upcast before hashing
                h.update(raw_bytes(conv[li, int(mi)]))
            rec["conv"][str(li)] = h.hexdigest()[:32]

        with open(STATE_DUMP, "a") as fh:
            fh.write(_json.dumps(rec) + "\n")
    except Exception as e:  # noqa: BLE001
        # A dump failure must be visible; a silent probe reads as "no difference".
        with open(STATE_DUMP, "a") as fh:
            fh.write(
                _json.dumps(
                    {"block": block_idx, "dump_error": f"{type(e).__name__}: {e}"}
                )
                + "\n"
            )


class BlockTimer:
    """Cost of each block in phases (prep / forward / sampler), or a torch.profiler
    trace of one block. Blocks before SGLANG_DLLM_TIMING_SKIP are not measured."""

    def __init__(self) -> None:
        # Sampler steps the current block ran; set by the algorithm.
        self.steps = 0
        self._idx = -1
        self._live = False
        self._acc = {}
        self._fwds = 0
        self._prof = None
        self._profiled_once = False
        self._wall0 = self._mark = 0.0

    def begin(self) -> None:
        # Reset first, or a readout-only block inherits the previous value.
        self.steps = 0
        self._idx = self._idx + 1
        self._live = self._idx >= T_SKIP
        if not self._live:
            return
        self._acc = dict(prep=0.0, fwd=0.0, samp=0.0)
        self._fwds = 0
        # Profile exactly one block: teardown costs seconds and contaminates later blocks.
        self._prof = None
        if PROFILE and not self._profiled_once:
            self._profiled_once = True
            self._prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            self._prof.__enter__()
        torch.cuda.synchronize()
        self._wall0 = self._mark = time.perf_counter()

    def split(self, bucket: str) -> None:
        """Close one phase of the current forward (syncs under TIMING only)."""
        if not self._live:
            return
        if TIMING:
            torch.cuda.synchronize()
        now = time.perf_counter()
        self._acc[bucket] += now - self._mark
        self._mark = now
        if bucket == "fwd":
            self._fwds += 1

    def end(self) -> None:
        if not self._live:
            return
        # Close the last interval first, so prep + fwd + samp add up to wall.
        self.split("samp")
        torch.cuda.synchronize()
        wall = (time.perf_counter() - self._wall0) * 1e3
        a = {k: v * 1e3 for k, v in self._acc.items()}
        profiled = self._prof is not None
        rec = dict(
            block=self._idx,
            forwards=self._fwds,
            # readout excluded; unfused, forwards == steps_run + 2 (readout + commit)
            steps_run=self.steps,
            wall_ms=wall,
            mode="profile" if profiled else "serialized",
            **a,
        )
        if profiled:
            self._prof.__exit__(None, None, None)
            evs = self._prof.key_averages()
            # Kernel entries only: the launching CPU op also carries its child's
            # self_device_time_total, so summing everything double-counts.
            try:
                from torch.autograd import DeviceType

                # Annotation ranges (e.g. "step[DLLM_EXTEND bs=1]") are CUDA-typed too.
                def _is_kernel(e):
                    if e.device_type != DeviceType.CUDA:
                        return False
                    k = str(getattr(e, "key", ""))
                    if k.startswith("step[") or "ProfilerStep" in k:
                        return False
                    et = str(getattr(e, "kind", "") or getattr(e, "event_type", ""))
                    return "annotation" not in et.lower()

                kern = [e for e in evs if _is_kernel(e)]
            except Exception:
                kern = []

            def _dev(e):
                return max(getattr(e, "self_device_time_total", 0.0) or 0.0, 0.0)

            dev_us = sum(_dev(e) for e in kern)
            launches = sum(int(e.count) for e in kern)
            rec["kernel_ms"] = dev_us / 1e3
            rec["launches"] = launches
            rec["gap_ms"] = wall - rec["kernel_ms"]
            # Unfiltered figure, so a double-counting regression stays visible.
            rec["kernel_all_ms"] = sum(_dev(e) for e in evs) / 1e3
            rec["kernel_entries"] = len(kern)

            def _row(e):
                return dict(name=str(e.key)[:70], ms=_dev(e) / 1e3, count=int(e.count))

            rec["top_kernels"] = [
                _row(e) for e in sorted(kern, key=_dev, reverse=True)[:25]
            ]
            # By count too: dispatch cost tracks launch count, not kernel time.
            rec["top_kernels_by_count"] = [
                _row(e)
                for e in sorted(kern, key=lambda e: int(e.count), reverse=True)[:25]
            ]
            self._prof = None
        TIMES.append(rec)
        instr_emit(dict(kind="cost", **rec))
        if profiled:
            logger.warning(
                "dllm-block-profile: block=%d forwards=%d wall=%.2fms "
                "kernel=%.2fms (%.0f%%) gap=%.2fms (%.0f%%) launches=%d "
                "per_fwd_launches=%.0f",
                rec["block"],
                rec["forwards"],
                wall,
                rec["kernel_ms"],
                100.0 * rec["kernel_ms"] / max(wall, 1e-9),
                rec["gap_ms"],
                100.0 * rec["gap_ms"] / max(wall, 1e-9),
                rec["launches"],
                rec["launches"] / max(rec["forwards"], 1),
            )
        else:
            logger.warning(
                "dllm-block-timing: block=%d forwards=%d wall=%.2fms "
                "prep=%.2fms (%.0f%%) fwd=%.2fms (%.0f%%) samp=%.2fms (%.0f%%) "
                "[serialized: syncs remove overlap, total is pessimistic]",
                rec["block"],
                rec["forwards"],
                wall,
                a["prep"],
                100.0 * a["prep"] / max(wall, 1e-9),
                a["fwd"],
                100.0 * a["fwd"] / max(wall, 1e-9),
                a["samp"],
                100.0 * a["samp"] / max(wall, 1e-9),
            )
