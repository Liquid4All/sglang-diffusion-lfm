# LFM2 block diffusion: serving with SGLang (DuoBlock)

Model: [`LiquidAI/lfm2.5-350m-diffusion-exp`](https://huggingface.co/LiquidAI/lfm2.5-350m-diffusion-exp)
(any `Lfm2ForBlockDiffusion` checkpoint works). Decode configs here: `decode_nfe8.yaml` (fast),
`decode_nfe32.yaml` (best quality), `decode_nfe4.yaml` (fastest), `decode_nfe8_fused.yaml` (NFE 8 + commit fusion, single-request only).

| script | use when | B200, NFE 8, p50 / p99 |
|---|---|---|
| `serve_throughput.sh` | **production traffic** (requests overlap) | 36 / 72 ms at 20 req/s, 47 / 86 ms at 80 req/s; ~280 req/s per GPU |
| `serve_latency.sh` | strictly one request at a time | 33 / 47 ms; queues from 5 req/s, caps at ~39 req/s |

`DP=N bash serve_throughput.sh` serves N GPUs behind one port. `DECODE=decode_nfe32.yaml` for quality.

## Per-request options

The decode config sets the server defaults; a request overrides them where it sets a value:

| request field | effect |
|---|---|
| `dllm_steps_per_block` | NFE for this request (1 to `max_steps_per_block`) |
| `temperature: 0` | greedy decoding |
| `temperature: t` | static temperature `t`, **only if the config has no `temp_start`/`temp_end`**; with bounds the anneal is kept and `t` is ignored (logged once) |
| `top_p`, `top_k` | nucleus / top-k truncation for this request |

A value equal to SGLang's default (`temperature` 1.0, `top_p` 1.0, `top_k` -1) counts as unset. Requests with different
settings are scheduled in different rounds, and a setting other than the config's captures its own block graph on first
use (startup warmup covers the config's settings only). `seed` is ignored (logged once); `return_logprob` is refused.

Load-test with an async HTTP client; a single OpenAI-SDK Python process saturates by itself.
Latencies: HTTP chat, max 160 new tokens, a 350M checkpoint of this architecture.
On MI325X (ROCm), `PYTORCH_TUNABLEOP_ENABLED=1` measured 5-7% faster per request (GEMM tuning on first use).
