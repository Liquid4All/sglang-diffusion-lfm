"""LFM2 block-diffusion (uniform-state DUO) model for SGLang.

The backbone is LFM2's causal stack with the diffusion layer subclasses below,
so under causal prefix attention prefill is the trained token-causal clean
stream; DuoBlock switches denoise forwards (and, with bidirectional prefix
attention, the prefill) to bidirectional via forward_batch.dllm_causal_override.

Adds, matching the training reference (the parity oracle):
  * sigma/adaLN-single conditioning:
        temb        = TimestepEmbedder(sigma)                       # [T, cond]
        adaln_base  = adaln_proj(temb)                              # [T, 3*dim]
        per layer l: base = adaln_base + adaln_block_emb[l]
                     scale, shift, gate = base.chunk(3)
                     h_mod = h * (1 + scale) + shift
                     h     = h + (1 + gate) * (layer(h_mod) - h_mod)
    sigma comes from forward_batch.dllm_sigma; None -> zeros (clean text).
  * input-site self-conditioning, h = h + selfcond_ffn(selfcond_norm(sc)),
    applied only when forward_batch.dllm_selfcond is set; committed positions
    get no sc contribution (not even the FFN bias), as in training.

Weights: standard HF LFM2 names plus `model.sigma_embed.*`, `model.adaln_proj.*`,
`model.adaln_block_emb`, `model.selfcond_norm.*`, `model.selfcond_ffn.*`.
"""

import logging
import math
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.configs.lfm2_diffusion import Lfm2DiffusionConfig
from sglang.srt.dllm.kernels.adaln import adaln_post, adaln_pre
from sglang.srt.dllm.kernels.adaln_norm import active_layers as _adaln_active_layers
from sglang.srt.dllm.kernels.adaln_norm import (
    adaln_post_pre_norm as _adaln_post_pre_norm,
)
from sglang.srt.dllm.kernels.adaln_norm import adaln_pre_norm as _adaln_pre_norm
from sglang.srt.dllm.kernels.adaln_norm import consumer_norms as _adaln_consumer_norms
from sglang.srt.dllm.kernels.adaln_norm import eligible as _adaln_norm_eligible
from sglang.srt.dllm.kernels.adaln_norm import enabled as _adaln_norm_enabled
from sglang.srt.dllm.kernels.rmsnorm import triton_is_active as _triton_is_active
from sglang.srt.dllm.kernels.silu_mul import active_modules as _silu_active_modules
from sglang.srt.dllm.kernels.silu_mul import merged_modules as _silu_merged_modules
from sglang.srt.dllm.kernels.silu_mul import total_modules as _silu_total_modules
from sglang.srt.layers.attention.mamba.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.lfm2 import (
    Lfm2Attention,
    Lfm2ForCausalLM,
    Lfm2MLP,
    Lfm2ShortConv,
)
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


def _fused_add_norm_live(norm: nn.Module) -> bool:
    """Whether ``norm(x, residual)`` runs the dLLM Triton kernel, which returns
    the sum as a new tensor; the stock fused_add_rmsnorm writes it into
    ``residual`` in place, which the adaLN post step still reads."""
    return _triton_is_active(norm)


class Lfm2DiffusionAttention(Lfm2Attention):
    """Lfm2Attention whose KV write is skipped on forwards that must not commit
    (forward_batch.dllm_save_kv)."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        T = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)

        q_size = self.num_local_q_heads * self.head_dim
        kv_size = self.num_local_kv_heads * self.head_dim
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        q = q.reshape(T, self.num_local_q_heads, self.head_dim)
        k = k.reshape(T, self.num_local_kv_heads, self.head_dim)

        q = self.q_layernorm(q.reshape(-1, self.head_dim)).reshape(
            T, self.num_local_q_heads, self.head_dim
        )
        k = self.k_layernorm(k.reshape(-1, self.head_dim)).reshape(
            T, self.num_local_kv_heads, self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k)

        attn_out = self.attn(
            q.reshape(T, -1),
            k.reshape(T, -1),
            v,
            forward_batch,
            save_kv_cache=forward_batch.dllm_save_kv,
        )
        out, _ = self.out_proj(attn_out)
        return out


class Lfm2DiffusionShortConv(Lfm2ShortConv):
    """Lfm2ShortConv whose carried conv state can stand at a committed length
    shorter than the forward (forward_batch.dllm_conv_state_at)."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        meta = get_attn_backend().conv_state_metadata(self.layer_idx, forward_batch)
        conv_state = meta.layer_cache.conv[0]

        proj, _ = self.in_proj(hidden_states)
        B_gate, C_gate, x = proj.chunk(3, dim=-1)
        Bx = B_gate * x

        if forward_batch.forward_mode.is_decode():
            conv_out = causal_conv1d_update(
                Bx,
                conv_state,
                self.conv_weight,
                self.conv_bias,
                activation=None,
                conv_state_indices=meta.cache_indices,
            )
        else:
            # The taps are rewritten from Bx after the conv. Snapshot the
            # pre-forward taps first: causal_conv1d_fn overwrites them in place
            # and an n=1 row needs the previous window's newer tap.
            state_at = forward_batch.dllm_conv_state_at
            prev_taps = None
            if state_at is not None:
                prev_taps = conv_state[meta.cache_indices].clone()

            Bx_t = Bx.transpose(0, 1).contiguous()
            conv_kwargs = {}
            if meta.seq_lens_cpu is not None:
                # Host-side lengths skip the kernel's .cpu() sync, which raises
                # inside graph capture.
                conv_kwargs["seq_lens_cpu"] = meta.seq_lens_cpu
            conv_out = causal_conv1d_fn(
                Bx_t,
                self.conv_weight,
                self.conv_bias,
                query_start_loc=meta.query_start_loc,
                cache_indices=meta.cache_indices,
                has_initial_state=meta.has_initial_state,
                conv_states=conv_state,
                activation=None,
                **conv_kwargs,
            ).transpose(0, 1)

            if state_at is not None:
                self._rewrite_conv_state_at(meta, conv_state, Bx, state_at, prev_taps)

        output, _ = self.out_proj(C_gate * conv_out)
        return output

    @staticmethod
    def _rewrite_conv_state_at(meta, conv_state, Bx, state_at, prev):
        """Move each row's carried conv state back to its committed length.

        conv_state[slot] is [conv_dim, 2] = (u(t-2), u(t-1)) with u = Bx, so a
        row committing n tokens resumes from (Bx[n-2], Bx[n-1]); for n = 1 the
        older tap is the previous window's newer tap if the row has history, else 0.
        """
        qsl = getattr(meta, "query_start_loc", None)
        if qsl is None:
            raise ValueError(
                "dllm_conv_state_at needs query_start_loc to locate each "
                "request's tokens; the backend supplied none."
            )
        qsl = [int(v) for v in qsl.tolist()]
        bs = len(qsl) - 1
        n = [int(v) for v in state_at.tolist()]
        if len(n) != bs:
            raise ValueError(
                f"dllm_conv_state_at has {len(n)} entries for {bs} requests"
            )
        init = meta.has_initial_state
        init = [False] * bs if init is None else [bool(v) for v in init.tolist()]
        # Skip padded rows (sentinel slot -1 would index a real slot) and empty
        # sequences, as in the fused path.
        _ci = [int(v) for v in meta.cache_indices.tolist()]
        live = [b for b in range(bs) if _ci[b] >= 0 and (qsl[b + 1] - qsl[b]) > 0]
        if not live:
            return
        taps = []
        for b in live:
            nb = n[b]
            lo, hi = qsl[b], qsl[b + 1]
            w = hi - lo
            if not 1 <= nb <= w:
                raise ValueError(
                    f"dllm_conv_state_at[{b}]={nb} outside [1, {w}] for this "
                    "window; a committed length of zero would leave the "
                    "request without progress and one past the window would "
                    "cite tokens this forward never saw."
                )
            t1 = Bx[lo + nb - 1]
            if nb >= 2:
                t0 = Bx[lo + nb - 2]
            elif init[b] and prev is not None:
                t0 = prev[b, :, 1]
            else:
                t0 = torch.zeros_like(t1)
            taps.append(torch.stack((t0, t1), dim=-1))
        _slots = meta.cache_indices[
            torch.tensor(live, device=meta.cache_indices.device)
        ]
        conv_state[_slots] = torch.stack(taps).to(conv_state.dtype)


class Lfm2DiffusionDecoderLayer(nn.Module):
    """Lfm2DecoderLayer built from the diffusion mixers, with the fused residual
    add+norm at ffn_norm and an FFN add the next layer's adaLN-post can absorb.
    Module names match Lfm2DecoderLayer, so checkpoints load unchanged."""

    def __init__(
        self,
        config: Lfm2DiffusionConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_type = config.layer_types[layer_id]
        self.is_attention_layer = self.layer_type == "full_attention"

        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        if self.is_attention_layer:
            self.self_attn = Lfm2DiffusionAttention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                attn_type=AttentionType.DECODER,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            self.conv = Lfm2DiffusionShortConv(
                config=config,
                layer_idx=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("conv", prefix),
            )

        self.feed_forward = Lfm2MLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("feed_forward", prefix),
        )
        # Set per call by Lfm2DiffusionModel.forward.
        self._dllm_defer_ffn_add = False

    def forward(
        self,
        layer_id: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not forward_batch.forward_mode.is_idle():
            residual = hidden_states
            normed = self.operator_norm(hidden_states)

            if self.is_attention_layer:
                hidden_states = self.self_attn(positions, normed, forward_batch)
            else:
                hidden_states = self.conv(normed, forward_batch)

            if _fused_add_norm_live(self.ffn_norm):
                # The sum comes back as a new tensor, leaving `residual` intact
                # for the adaLN post step; see _fused_add_norm_live.
                ffn_in, hidden_states = self.ffn_norm(hidden_states, residual)
            else:
                hidden_states = hidden_states + residual
                ffn_in = self.ffn_norm(hidden_states)
            # Not folded into w2 as an addmm epilogue: cuBLAS adds a copy kernel
            # for the beta=1 operand, which saves nothing and changes numerics.
            ffn_out = self.feed_forward(ffn_in)
            if self._dllm_defer_ffn_add:
                # Folded into the next layer's fused adaLN-post kernel
                # (adaln_post_pre_norm, y2=); the caller pops the stash.
                self._dllm_ffn_out = ffn_out
            else:
                hidden_states = hidden_states + ffn_out

        return hidden_states, residual


class TimestepEmbedder(nn.Module):
    """DiT-style sinusoidal timestep embedding + MLP (reference math, verbatim)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class Lfm2DiffusionModel(nn.Module):
    """Lfm2Model's causal stack + adaLN-single sigma conditioning + input-site
    self-cond."""

    def __init__(
        self,
        config: Lfm2DiffusionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.num_attention_layers = sum(
            1 for lt in config.layer_types if lt == "full_attention"
        )

        def get_layer(idx: int, prefix: str, **kwargs):
            return Lfm2DiffusionDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=add_prefix("layers", prefix)
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        dim = config.hidden_size
        cond = config.sigma_cond_dim
        if config.adaln_conditioning:
            self.sigma_embed = TimestepEmbedder(cond, config.sigma_freq_embedding_size)
            self.adaln_proj = nn.Sequential(
                nn.SiLU(), nn.Linear(cond, 3 * dim, bias=True)
            )
            self.adaln_block_emb = nn.Parameter(
                torch.zeros(config.num_hidden_layers, 3 * dim)
            )
        if config.self_conditioning:
            self.selfcond_norm = RMSNorm(dim, eps=config.norm_eps)
            self.selfcond_ffn = nn.Sequential(
                nn.Linear(dim, dim, bias=bool(config.selfcond_ffn_bias)),
                nn.SiLU(),
                nn.Linear(dim, dim, bias=False),
            )

    def adaln_rows(self, sigma: torch.Tensor) -> torch.Tensor:
        """adaLN base rows for a 1-D sigma tensor: [K] -> [K, 3*hidden]."""
        temb = self.sigma_embed(sigma.reshape(-1).to(torch.float32))
        return self.adaln_proj(temb)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-block conv-state snapshot (eager-path writer; the graph runner
        # snapshots before replay instead), restored by DuoBlock before each later
        # denoise forward. Must run after the hybrid COW/clear fixups and slot
        # refill. dllm_conv_snapshot is a pre-created list shared across shallow
        # ForwardBatch copies; append exactly once.
        snap_list = forward_batch.dllm_conv_snapshot
        if (
            forward_batch.dllm_conv_capture
            and isinstance(snap_list, list)
            and len(snap_list) == 0
        ):
            from sglang.srt.model_executor.forward_context import get_attn_backend

            backend = get_attn_backend()
            if hasattr(backend, "conv_state_metadata"):
                lids = list(self.config.linear_layer_ids)
                if hasattr(backend, "conv_state_pool") and lids:
                    # One entry for all conv layers: the per-layer caches are
                    # views of one [layers, slots, dim, k-1] pool, so a single
                    # index snapshots (and _restore_conv restores) every layer.
                    meta = backend.conv_state_metadata(lids[0], forward_batch)
                    pool = backend.conv_state_pool()
                    idx = meta.cache_indices
                    snap_list.append((pool, idx, pool[:, idx].clone()))
                else:
                    for lid in lids:
                        meta = backend.conv_state_metadata(lid, forward_batch)
                        cs = meta.layer_cache.conv[0]
                        idx = meta.cache_indices
                        snap_list.append((cs, idx, cs[idx].clone()))

        h = input_embeds if input_embeds is not None else self.embed_tokens(input_ids)

        # Self-conditioning only where the algorithm supplied a belief; None on
        # prefill/commit keeps the committed context sc-free.
        sc = forward_batch.dllm_selfcond
        if sc is not None:
            assert self.config.self_conditioning, (
                "dllm_selfcond passed but the checkpoint was exported without "
                "self_conditioning -- the signal would be silently dropped"
            )
            # Applied to every token (clean positions get +c = FFN(norm(0))); under
            # commit fusion dllm_selfcond_pos_mask keeps the committed prefix sc-free.
            _sc_out = self.selfcond_ffn(self.selfcond_norm(sc.to(h.dtype)))
            _pos = forward_batch.dllm_selfcond_pos_mask
            if _pos is not None:
                _g = _pos.to(_sc_out.dtype)
                if _g.dim() == 1:
                    _g = _g.unsqueeze(-1)
                if _g.shape[0] != _sc_out.shape[0]:
                    raise ValueError(
                        f"dllm_selfcond_pos_mask has {_g.shape[0]} rows for "
                        f"{_sc_out.shape[0]} tokens; a mismatched gate would "
                        "broadcast and silence the wrong positions"
                    )
                _sc_out = _sc_out * _g
            h = h + _sc_out

        # Pre-selected by the algorithm from a per-step table when present
        # (sigma takes two values per forward); the sigma path is the fallback.
        adaln_base = forward_batch.dllm_adaln_base
        if adaln_base is None and getattr(self.config, "adaln_conditioning", False):
            sigma = forward_batch.dllm_sigma
            if sigma is None:
                # Reference eval-time default: clean text = zero noise
                sigma = torch.zeros(h.shape[0], device=h.device, dtype=torch.float32)
            temb = self.sigma_embed(sigma.reshape(-1))  # [T, cond]
            adaln_base = self.adaln_proj(temb)  # [T, 3*dim]

        residual = None
        # Clear every norm stash at entry (keyed on the hook, per layer): a forward
        # that raised mid-layer skips the post-call clear. Not try/finally, since
        # try/with blocks in these dynamo-traced frames break compilation.
        for _l in self.layers:
            if getattr(_l, "_dllm_norm_hooked", False):
                _l._dllm_normed = None

        _fuse_ok = adaln_base is not None and _adaln_norm_enabled()
        _prev = None  # deferred (h, gate, y, h_mod) awaiting the next layer
        for i in range(len(self.layers)):
            if adaln_base is not None:
                # Fused: 2 launches/layer instead of ~11 (the forward is
                # dispatch-bound). adaln_pre/post are bit-exact with the eager
                # chain and fall back to it on any shape/dtype they cannot match.
                _lyr = self.layers[i]
                # Checked per layer, not hoisted: h's dtype can change per layer
                # and must match the conditioning dtype.
                _fuse_adaln_norm = _fuse_ok and _adaln_norm_eligible(
                    h, adaln_base, self.adaln_block_emb[i]
                )
                if (
                    _fuse_adaln_norm
                    and i > 0
                    and getattr(_lyr, "_dllm_norm_hooked", False)
                    and _prev is not None
                ):
                    # adaln_post(i-1) + adaln_pre(i) + operator_norm(i) in one
                    # row-wise kernel.
                    h, h_mod, gate, _lyr._dllm_normed = _adaln_post_pre_norm(
                        _prev[0],
                        _prev[1],
                        _prev[2],
                        _prev[3],
                        adaln_base,
                        self.adaln_block_emb[i],
                        _lyr.operator_norm.weight,
                        _lyr.operator_norm.variance_epsilon,
                        y2=_prev[4],
                    )
                    _prev = None
                elif _fuse_adaln_norm and getattr(_lyr, "_dllm_norm_hooked", False):
                    # adaln_pre + operator_norm in one kernel emitting h_mod,
                    # gate and the normed input.
                    h_mod, gate, _lyr._dllm_normed = _adaln_pre_norm(
                        h,
                        adaln_base,
                        self.adaln_block_emb[i],
                        _lyr.operator_norm.weight,
                        _lyr.operator_norm.variance_epsilon,
                    )
                else:
                    h_mod, gate = adaln_pre(h, adaln_base, self.adaln_block_emb[i])
                # Residual-add fold: when the next layer runs the fused
                # post+pre+norm kernel, the FFN output is left un-added and the
                # kernel sums it (fp32 add, one bf16 rounding == torch's add).
                _defer_add = (
                    _fuse_adaln_norm
                    and i + 1 < len(self.layers)
                    and getattr(self.layers[i + 1], "_dllm_norm_hooked", False)
                )
                _lyr._dllm_defer_ffn_add = _defer_add
                y, residual = _lyr(
                    layer_id=i,
                    positions=positions,
                    hidden_states=h_mod,
                    residual=residual,
                    forward_batch=forward_batch,
                )
                _lyr._dllm_defer_ffn_add = False
                _ffn = _lyr.__dict__.pop("_dllm_ffn_out", None)
                # The stash lives for exactly this call; an idle DP-attention
                # forward skips operator_norm and would consume a leftover.
                _lyr._dllm_normed = None
                if (
                    _fuse_adaln_norm
                    and i + 1 < len(self.layers)
                    and getattr(self.layers[i + 1], "_dllm_norm_hooked", False)
                    # y's dtype is only known now; the fused kernel casts the
                    # post arithmetic to h's dtype where adaln_post promotes.
                    and _adaln_norm_eligible(h, gate, y, h_mod)
                    and (_ffn is None or _ffn.dtype == y.dtype)
                ):
                    # Deferred into the next iteration's fused kernel; the last
                    # layer falls through to the standalone post.
                    _prev = (h, gate, y, h_mod, _ffn)
                else:
                    if _ffn is not None:
                        y = y + _ffn
                    h = adaln_post(h, gate, y, h_mod)
            else:
                h, residual = self.layers[i](
                    layer_id=i,
                    positions=positions,
                    hidden_states=h,
                    residual=residual,
                    forward_batch=forward_batch,
                )

        # A deferred post past the loop would silently drop a layer.
        assert _prev is None, "adaLN post deferred past the final layer"
        return self.embedding_norm(h)


class Lfm2ForBlockDiffusion(Lfm2ForCausalLM):
    """LFM2 block-DUO diffusion LM. Served via --dllm-algorithm DuoBlock."""

    def __init__(
        self,
        config: Lfm2DiffusionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Lfm2ForCausalLM.__init__ builds a plain Lfm2Model + LogitsProcessor;
        # swap both for the diffusion variants, keeping lm_head/load_weights.
        super().__init__(config, quant_config, prefix)
        self.model = Lfm2DiffusionModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        # dLLM needs logits over every canvas position, not just the last
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)

    def adaln_rows(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.model.adaln_rows(sigma)

    def adaln_dim(self) -> int:
        return int(self.model.adaln_proj[-1].out_features)

    def adaln_dtype(self) -> torch.dtype:
        return self.model.adaln_proj[-1].weight.dtype

    def post_load_weights(self, *args, **kwargs):
        """Also install fused kernels for loaders that bypass load_weights().

        sharded_state / dummy / remote loaders call only post_load_weights();
        _install_fused_kernels is idempotent, so running it from both is safe.
        """
        r = None
        sup = getattr(super(), "post_load_weights", None)
        if callable(sup):
            r = sup(*args, **kwargs)
        self._install_fused_kernels()
        return r

    def _install_fused_kernels(self) -> None:
        """Install the opt-in fused kernels on this model's module tree.

        Each kernel falls back to the stock implementation for any shape or
        dtype it does not support; some are not bit-exact where they do run.
        """
        # Flag defaults live in sglang.srt.environ (SGLANG_DLLM_FUSED_DEFAULTS)
        # so every consumer module resolves the same answer.
        from sglang.srt.environ import envs

        # Each switch is independent; any one of them enables the installer.
        _flags = (
            envs.SGLANG_DLLM_FUSED_NORM,
            envs.SGLANG_DLLM_FUSE_MLP_GEMM,
            envs.SGLANG_DLLM_FUSE_ADALN_NORM,
            envs.SGLANG_DLLM_FUSE_SILU_MUL,
            envs.SGLANG_DLLM_FUSE_SCONV,
            envs.SGLANG_DLLM_FUSE_QKV,
            envs.SGLANG_DLLM_FUSE_DENOISE_STEP,
        )
        if not any(f.get() for f in _flags):
            return
        from sglang.srt.dllm.kernels.rmsnorm import install as _install_norm
        from sglang.srt.dllm.kernels.silu_mul import install as _install_mlp

        want_norm = envs.SGLANG_DLLM_FUSED_NORM.get()
        # adaln_norm only hooks layers whose operator_norm is the Triton kernel,
        # so its consumers request that norm even over a platform fused one.
        _adaln_consumers = (
            _adaln_consumer_norms(self.model) if _adaln_norm_enabled() else []
        )
        # ffn_norm consumers: the residual add fuses into ffn_norm only on the
        # Triton path, whose out-of-place sum keeps h_mod for the adaLN post.
        _addnorm_consumers = (
            [l.ffn_norm for l in self.model.layers if hasattr(l, "ffn_norm")]
            if want_norm
            else []
        )
        # q/k head-norm consumers: the fused QKV prologue computes those norms
        # with the Triton arithmetic, so it requires the Triton norm there.
        from sglang.srt.dllm.kernels import qkv_prologue as _qkv

        # Only modules the installer can fuse, so a refused module keeps its
        # norm arithmetic.
        _qkv_consumers = (
            [
                n
                for l in self.model.layers
                if getattr(l, "self_attn", None) is not None
                and _qkv.static_refusal(l.self_attn) is None
                for n in (l.self_attn.q_layernorm, l.self_attn.k_layernorm)
            ]
            if (_qkv.enabled() and want_norm)
            else []
        )
        _consumers = list(_adaln_consumers) + _addnorm_consumers + _qkv_consumers
        _why = []
        if _adaln_consumers:
            _why.append("the adaln+norm fusion (SGLANG_DLLM_FUSE_ADALN_NORM)")
        if _addnorm_consumers:
            _why.append("the residual add+norm fusion at ffn_norm")
        if _qkv_consumers:
            _why.append(
                "the fused QKV prologue at q/k_layernorm (SGLANG_DLLM_FUSE_QKV)"
            )
        n = _install_norm(
            self.model,
            want_norm,
            required_by=" and ".join(_why),
            consumer_modules=_consumers or None,
        )
        logger.warning("dLLM: fused rmsnorm on %d modules this pass", n)
        # The w1/w3 GEMM merge caches a derived weight copy, so it is opt-in
        # separately -- see silu_mul.install().
        merge = envs.SGLANG_DLLM_FUSE_MLP_GEMM.get()
        # An explicit SGLANG_DLLM_FUSE_SILU_MUL wins in both directions (for
        # per-kernel attribution); unset, silu_mul is implied by FUSED_NORM or
        # FUSE_MLP_GEMM, so `merge` alone can enable the installer.
        want_silu = envs.SGLANG_DLLM_FUSE_SILU_MUL.get() or (
            not envs.SGLANG_DLLM_FUSE_SILU_MUL.is_set() and (want_norm or merge)
        )
        m = _install_mlp(self.model, want_silu, merge_gemm=merge)
        logger.warning("dLLM: silu*mul on %d modules this pass", m)
        from sglang.srt.dllm.kernels import gated_sconv as _gs

        sc = _gs.install(self.model, _gs.enabled())
        logger.warning("dLLM: gated short conv fused on %d modules this pass", sc)
        qk = _qkv.install(self.model, _qkv.enabled())
        logger.warning("dLLM: qkv prologue fused on %d modules this pass", qk)
        # Hook operator_norm so the loop can hand it a precomputed norm; after
        # load_weights because the hook reads operator_norm.weight.
        an = 0
        if _adaln_norm_enabled():
            from sglang.srt.dllm.kernels import adaln_norm as _an

            an = _an.install(self.model)
            logger.warning("dLLM: adaln+norm fused on %d layers this pass", an)
        # The marker records installed-and-live model state, not this call's
        # delta or flag echoes; _fusion_installed() reads its last line.
        norm_active = sum(1 for _mod in self.model.modules() if _triton_is_active(_mod))
        adaln_active = _adaln_active_layers(self.model)
        silu_active = _silu_active_modules(self.model)
        merged_active = _silu_merged_modules(self.model)
        silu_total = _silu_total_modules(self.model)
        if _adaln_norm_enabled() and adaln_active == 0:
            logger.warning(
                "dLLM: SGLANG_DLLM_FUSE_ADALN_NORM is on but NO layer carries "
                "the fusion -- operator_norm is not the Triton kernel on any "
                "of them.",
            )
        msg = (
            f"dLLM fused kernels: rmsnorm={norm_active} modules, "
            f"mlp={silu_active} modules (mlp_total={silu_total}, "
            f"gemm_merge={merged_active} merged), "
            f"adaln_norm={adaln_active} layers, "
            f"sconv={_gs.active_modules(self.model)} modules, "
            f"qkv={_qkv.active_modules(self.model)} modules "
            f"(qkv_total={_qkv.total_modules(self.model)}), "
            # a pure switch consumed in DuoBlock.step (like preplan): the flag
            # IS the effect, so on/off is a genuine outcome here
            f"denoise_step={'on' if envs.SGLANG_DLLM_FUSE_DENOISE_STEP.get() else 'off'}, "
            f"preplan={'on' if envs.SGLANG_DLLM_PREPLAN_ATTN.get() else 'off'}, "
            # `forced` records whether SGLANG_DLLM_FORCE_FUSED decided the norm
            # install.
            f"forced="
            f"{'on' if getattr(self.model, '_dllm_norm_forced', False) else 'off'}"
        )
        logger.warning(msg)
        # Worker logs do not reach the job log, so the harness reads this
        # marker file instead.
        mk = envs.SGLANG_DLLM_FUSED_MARKER.get()
        if mk:
            try:
                with open(mk, "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> Set[str]:
        # Every conditioning tensor must load (a zero-init adaLN serves silently
        # broken); spy the stream since the parent skips unmatched tensors silently.
        _seen_ckpt_bias = False

        def _spy(ws):
            nonlocal _seen_ckpt_bias
            for name, w in ws:
                if name == "model.selfcond_ffn.0.bias":
                    _seen_ckpt_bias = True
                yield name, w

        loaded = super().load_weights(_spy(weights), is_mtp=is_mtp)
        # Installed after loading: the MLP fusion concatenates w1/w3 into a
        # derived weight, which must copy the loaded tensors.
        self._install_fused_kernels()
        required_prefixes = []
        if getattr(self.config, "adaln_conditioning", False):
            required_prefixes += [
                "model.sigma_embed.",
                "model.adaln_proj.",
                "model.adaln_block_emb",
            ]
        if getattr(self.config, "self_conditioning", False):
            required_prefixes += ["model.selfcond_norm.", "model.selfcond_ffn."]
        for pref in required_prefixes:
            assert any(n.startswith(pref) for n in loaded), (
                f"checkpoint is missing conditioning weights '{pref}*' -- "
                "refusing to serve a diffusion model with zero-init "
                "conditioning (silent partial load)"
            )
        if getattr(self.config, "self_conditioning", False):
            # Bias presence must match the config: a config bias with no
            # checkpoint tensor would keep nn.Linear's random nonzero init.
            has_bias_param = "model.selfcond_ffn.0.bias" in dict(
                self.named_parameters()
            )
            got_bias = "model.selfcond_ffn.0.bias" in loaded
            assert has_bias_param == bool(
                self.config.selfcond_ffn_bias
            ), "model construction disagrees with config.selfcond_ffn_bias"
            assert (not has_bias_param) or got_bias, (
                "config.selfcond_ffn_bias=true but the checkpoint carries no "
                "selfcond_ffn.0.bias -- the bias would keep its RANDOM default "
                "init. Export/config mismatch; refusing to serve."
            )
            # Reverse direction: a bias-free config would silently drop a
            # trained bias (the +c term).
            assert has_bias_param or not _seen_ckpt_bias, (
                "config.selfcond_ffn_bias=false but the checkpoint CARRIES a "
                "trained selfcond_ffn.0.bias -- it would be silently dropped. "
                "This checkpoint was trained bias=true; fix the config."
            )
        return loaded


EntryClass = [Lfm2ForBlockDiffusion]
