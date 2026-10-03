#!/bin/bash
# Single-request latency: one request at a time, commit fusion inside the whole-block graph.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# The SGLang ROCm images enable AITER, which slows DuoBlock by ~30%; no effect on NVIDIA.
export SGLANG_USE_AITER=${SGLANG_USE_AITER:-0}
MODEL=${MODEL:-LiquidAI/lfm2.5-350m-diffusion-exp}
DECODE=${DECODE:-decode_nfe8_fused.yaml}
[ -f "$DECODE" ] || DECODE=$HERE/$DECODE  # a bare name is looked up next to this script
exec python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
  --dllm-algorithm DuoBlock --dllm-algorithm-config "$DECODE" \
  --dllm-prefix-attention causal --attention-backend triton --no-dllm-fdfo \
  --dllm-cuda-graph --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 \
  --max-running-requests 1 --disable-radix-cache --dtype bfloat16 --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port "${PORT:-30000}"
