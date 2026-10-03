# coding=utf-8
"""LFM2 block-diffusion (uniform-state DUO) configuration.

A stock LFM2 hybrid backbone plus:
  * adaLN-single time conditioning on a per-token sigma = -log(alpha_t).
  * self-conditioning on the previous step's x0-belief soft embedding.
"""

from transformers import CONFIG_MAPPING

from sglang.srt.configs.lfm2 import Lfm2Config


class Lfm2DiffusionConfig(Lfm2Config):
    model_type = "lfm2_diffusion"

    def __init__(
        self,
        sigma_cond_dim: int = None,
        sigma_freq_embedding_size: int = 256,
        selfcond_ffn_bias: bool = True,
        self_conditioning: bool = True,
        adaln_conditioning: bool = True,
        diffusion_block_size: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Training default is 256 (not the model dim); exported configs set it.
        self.sigma_cond_dim = sigma_cond_dim if sigma_cond_dim is not None else 256
        self.sigma_freq_embedding_size = sigma_freq_embedding_size
        self.selfcond_ffn_bias = selfcond_ffn_bias
        self.self_conditioning = self_conditioning
        self.adaln_conditioning = adaln_conditioning
        self.diffusion_block_size = diffusion_block_size


CONFIG_MAPPING._extra_content["lfm2_diffusion"] = Lfm2DiffusionConfig
