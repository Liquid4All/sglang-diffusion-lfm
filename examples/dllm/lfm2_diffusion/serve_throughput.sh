#!/bin/bash
# Production serving: batching, block graphs captured at startup, prefill coalescing.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-LiquidAI/lfm2.5-350m-diffusion-exp}
DECODE=${DECODE:-decode_nfe8.yaml}
[ -f "$DECODE" ] || DECODE=$HERE/$DECODE  # a bare name is looked up next to this script
export SGLANG_DLLM_PREFILL_BATCH=4 SGLANG_DLLM_PREFILL_MAX_WAIT_MS=10
exec python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
  --dllm-algorithm DuoBlock --dllm-algorithm-config "$DECODE" \
  --dllm-prefix-attention causal --attention-backend triton --no-dllm-fdfo \
  --dllm-cuda-graph --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 \
  --max-running-requests 32 --disable-radix-cache --dtype bfloat16 --mem-fraction-static 0.85 \
  --tokenizer-worker-num 4 --dp-size "${DP:-1}" --host 0.0.0.0 --port "${PORT:-30000}"
