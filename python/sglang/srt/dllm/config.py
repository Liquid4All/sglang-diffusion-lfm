from typing import Any, Optional

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.dllm.params import (  # noqa: F401  (re-exported)
    steps_per_block_bounds,
    steps_per_block_bounds_from_server_args,
)
from sglang.srt.server_args import ServerArgs

DLLM_PARAMS = {
    "LLaDA2MoeModelLM": {"block_size": 32, "mask_id": 156895},
    "SDARForCausalLM": {"block_size": 4, "mask_id": 151669},
    "SDARMoeForCausalLM": {"block_size": 4, "mask_id": 151669},
    # mask_id 64400 is the reserved out-of-vocab embedding slot (real
    # vocab 64400, table 65536), used only as the canvas placeholder.
    "Lfm2ForBlockDiffusion": {
        "block_size": 32,
        "mask_id": 64400,
        "init_mode": "uniform_random",
        "real_vocab_size": 64400,
        "anchored": True,
    },
}


def dllm_reserved_prompt_id(hf_config) -> Optional[int]:
    """The canvas placeholder id when it is out of vocabulary and so never a real
    prompt token (None for architectures whose mask id is an ordinary token)."""
    params = DLLM_PARAMS.get(hf_config.architectures[0], {})
    return params["mask_id"] if "real_vocab_size" in params else None


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        mask_id: int,
        max_running_requests: int,
        first_done_first_out_mode: bool = False,
        init_mode: str = "mask",
        real_vocab_size: int | None = None,
        anchored: bool = False,
        prefix_attention: str = "causal",
        cuda_graph: bool = False,
        radix_cache: bool = True,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config or {}
        self.block_size = block_size
        # mask_id doubles as the canvas placeholder; under uniform_random the model
        # never sees it, so it only needs to be an id the tokenizer never emits.
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        self.first_done_first_out_mode = first_done_first_out_mode
        assert init_mode in ("mask", "uniform_random"), init_mode
        assert init_mode == "mask" or real_vocab_size is not None, (
            "init_mode='uniform_random' needs real_vocab_size to bound the "
            "uniform draw (the model's embedding table may be larger than the "
            "real vocabulary)."
        )
        self.init_mode = init_mode
        self.real_vocab_size = real_vocab_size
        # Answer-anchored grid: generation blocks start exactly at len(prompt).
        # False = absolute block-aligned windows.
        self.anchored = anchored
        # Whether --dllm-cuda-graph is on.
        self.cuda_graph = bool(cuda_graph)
        # Commit fusion needs the prefix held back one block, which only
        # ChunkCache.cache_unfinished_req does; under a radix cache the fused
        # window never forms.
        self.radix_cache = bool(radix_cache)
        # One flag for both clean-KV sites (prompt prefill and block commit):
        # mixing causal and bidirectional produces a KV no training recipe saw.
        assert prefix_attention in ("causal", "bidirectional"), prefix_attention
        self.prefix_attention = prefix_attention
        self.prefix_bidirectional = prefix_attention == "bidirectional"

    def decode_widths(self) -> tuple:
        """Widths a dLLM generation forward can have; size buffers with max(...),
        the per-forward width comes from the batch."""
        blk = int(self.block_size)
        if self.algorithm_config.get("commit_fusion", False):
            return (blk, 2 * blk)
        return (blk,)

    @staticmethod
    def logits_rows_per_req(server_args: ServerArgs, default: int) -> int:
        """Logits rows one batch slot needs: the widest dLLM forward, else default."""
        cfg = DllmConfig.from_server_args(server_args)
        return default if cfg is None else max(default, max(cfg.decode_widths()))

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.dllm_algorithm is None:
            return None

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )

        arch = model_config.hf_config.architectures[0]
        if arch in DLLM_PARAMS:
            params = DLLM_PARAMS[arch]
            block_size = params["block_size"]
            mask_id = params["mask_id"]
            init_mode = params.get("init_mode", "mask")
            real_vocab_size = params.get("real_vocab_size")
            anchored = params.get("anchored", False)
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")

        # None defers to the runtime cap the scheduler resolves from memory;
        # DllmConfig is built before the model runner has sized anything.
        max_running_requests = server_args.max_running_requests

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(server_args.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f) or {}
            if not isinstance(algorithm_config, dict):
                raise ValueError(
                    f"{server_args.dllm_algorithm_config} must hold a YAML mapping "
                    f"of decode settings, got {type(algorithm_config).__name__}"
                )

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            # Unset -> on, except opt-in for DuoBlock, whose short forward makes
            # FDFO's per-round scheduler cost dominate.
            first_done_first_out_mode=(
                server_args.dllm_fdfo
                if server_args.dllm_fdfo is not None
                else server_args.dllm_algorithm != "DuoBlock"
            ),
            init_mode=init_mode,
            real_vocab_size=real_vocab_size,
            anchored=anchored,
            prefix_attention=server_args.dllm_prefix_attention,
            cuda_graph=bool(getattr(server_args, "dllm_cuda_graph", False)),
            radix_cache=not bool(getattr(server_args, "disable_radix_cache", False)),
        )
