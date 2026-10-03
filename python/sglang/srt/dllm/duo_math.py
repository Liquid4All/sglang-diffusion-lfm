"""Pure math for uniform-state (DUO) block-diffusion decoding.

Torch-only port of the reference sampler internals (the parity oracle); greedy
byte-parity of the serving stack depends on this matching the oracle exactly.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def loglinear_alpha(t, eps: float = 1e-3):
    """alpha(t) = 1 - (1-eps) t  (samplers._loglinear_alpha)."""
    return 1.0 - (1.0 - eps) * t


def timestep_grid(
    steps: int, eps: float, schedule: str, rho: float, device
) -> torch.Tensor:
    """[steps+1] grid, 1 (noisy) -> 0 (clean) warped, then eps-squeezed
    (samplers._timestep_grid)."""
    u = torch.linspace(1.0, 0.0, steps + 1, device=device)
    if schedule == "linear":
        warped = u
    elif schedule == "rho":
        warped = u**rho
    elif schedule == "cosine":
        warped = 1.0 - torch.cos(u * math.pi / 2)
    elif schedule == "geometric":
        return torch.exp(torch.linspace(0.0, math.log(eps), steps + 1, device=device))
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return eps + (1.0 - eps) * warped


def anneal_scalar(
    start: float, end: float, frac: float, schedule: str = "linear"
) -> float:
    """samplers.anneal_scalar for "rev_log" and "linear"; other names raise."""
    if schedule == "rev_log":
        return start + end - start * (end / start) ** (1.0 - frac)
    if schedule == "linear":
        return start + frac * (end - start)
    raise ValueError(
        f"temp_schedule={schedule!r} is not implemented. The reference treats "
        "any unrecognised schedule as LINEAR, so computing anything else here "
        "would disagree with it silently. Use 'linear' or 'rev_log'."
    )


def block_temp(
    temp_anneal: bool,
    temperature: float,
    temp_start: float,
    temp_end: float,
    temp_schedule: str,
    i: int,
    steps: int,
) -> float:
    """samplers._block_temp: within-block temperature at denoise step i."""
    if not temp_anneal:
        return temperature
    frac = i / max(steps - 1, 1)
    return anneal_scalar(temp_start, temp_end, frac, temp_schedule)


def duo_posterior_from_x0(
    x0_probs: torch.Tensor,  # (B, L, V)
    xt: torch.Tensor,  # (B, L) int
    alpha_s: torch.Tensor,  # (B, 1, 1), >= alpha_t
    alpha_t: torch.Tensor,  # (B, 1, 1)
    vocab_size: int,
) -> torch.Tensor:
    """q(z_s | z_t, x0_hat) for the uniform-state kernel
    (reference duo_posterior_from_x0, verbatim)."""
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(xt, vocab_size).to(x0_probs.dtype)
    numerator = (
        alpha_t * vocab_size * x0_probs * xt_one_hot
        + (alpha_ts - alpha_t) * xt_one_hot
        + d_alpha * x0_probs
        + (1 - alpha_ts) * (1 - alpha_s) / vocab_size
    )
    denom = alpha_t * vocab_size * x0_probs.gather(-1, xt[..., None]) + (1 - alpha_t)
    return numerator / denom


def top_p_top_k_filter(logits: torch.Tensor, top_p: float, top_k: int) -> torch.Tensor:
    """samplers.top_p_top_k_filter: -inf out everything beyond top_k /
    the top_p nucleus (keeping the argmax always).

    top_k is clamped to the vocabulary width, as the reference does.
    """
    logits = logits.clone()
    if top_k and top_k > 0:
        kth = logits.topk(min(int(top_k), logits.shape[-1]), dim=-1).values[..., -1:]
        logits = torch.where(
            logits < kth, torch.full_like(logits, float("-inf")), logits
        )
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(-1)
        cum = probs.cumsum(-1)
        remove = cum - probs > top_p  # keep first token past the boundary
        remove_scattered = remove.scatter(-1, sorted_idx, remove)
        logits = torch.where(
            remove_scattered, torch.full_like(logits, float("-inf")), logits
        )
    return logits


def sample_categorical(
    probs: torch.Tensor, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Gumbel-max categorical draw; not RNG-stream-identical to the oracle, so
    sampling-mode parity is distributional only."""
    u = torch.rand(
        probs.shape, device=probs.device, dtype=torch.float32, generator=generator
    ).clamp_min(1e-20)
    g = -(-u.log()).log()
    return (probs.clamp_min(1e-30).log().float() + g).argmax(-1)


def duo_reverse_from_logits(
    logits: torch.Tensor,  # (B, Lb, V) raw model logits on the block
    x: torch.Tensor,  # (B, Lb) current block canvas
    alpha_t: torch.Tensor,  # (B, 1, 1)
    alpha_s: torch.Tensor,  # (B, 1, 1)
    vocab_size: int,
    temperature: float,
    use_float64: bool = True,
    greedy: bool = False,
    kappa: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    top_p_site: str = "belief",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The post-forward half of samplers._duo_reverse_step on the active block.

    Returns (x_next, log_x0). log_x0 is the temperature-scaled, untruncated
    log-softmax belief the oracle feeds to self-conditioning and adaptive stop;
    both consumers must use this tensor rather than re-deriving it.
    """
    if greedy:
        _lg = logits[..., :vocab_size].float()
        _lx = F.log_softmax(_lg / temperature, dim=-1)
        return _lx.argmax(-1), _lx
    logits = logits[..., :vocab_size].float()
    scaled = logits / temperature
    log_x0 = F.log_softmax(scaled, dim=-1)

    trunc = (top_p is not None and top_p < 1.0) or (top_k is not None and top_k > 0)
    if trunc and top_p_site not in ("belief", "posterior"):
        raise ValueError(f"top_p_site={top_p_site!r} not recognised")

    x0_probs = log_x0.exp()
    if trunc and top_p_site == "belief":
        x0_probs = F.log_softmax(
            top_p_top_k_filter(
                logits / temperature,
                1.0 if top_p is None else top_p,
                0 if top_k is None else top_k,
            ),
            dim=-1,
        ).exp()
    if use_float64:
        x0_probs = x0_probs.double()
        alpha_t = alpha_t.double()
        alpha_s = alpha_s.double()
    q_xs = duo_posterior_from_x0(x0_probs, x, alpha_s, alpha_t, vocab_size)
    if trunc and top_p_site == "posterior":
        q_xs = top_p_top_k_filter(
            q_xs.clamp_min(1e-30).log().float(),
            1.0 if top_p is None else top_p,
            0 if top_k is None else top_k,
        ).softmax(-1)
        if use_float64:
            q_xs = q_xs.double()
    if kappa < 1.0:
        q_fwd = alpha_s * x0_probs + (1.0 - alpha_s) / vocab_size
        q_xs = kappa * q_xs + (1.0 - kappa) * q_fwd
    x_next = sample_categorical(q_xs.clamp_min(1e-30), generator=generator)
    return x_next, log_x0


def readout_alpha(
    ts_last: torch.Tensor, eps: float, terminal_sigma_floor: float
) -> torch.Tensor:
    """alpha_t for the terminal readout, clamped so sigma = -log(alpha) never
    falls below the floor (samplers._block_denoise_duo, noise_removal branch).

    Takes the float32 grid tensor, not a python float: byte-parity requires the
    oracle's float32 rounding (float64 differs at ~1e-8, enough to flip a tie)."""
    a_t = loglinear_alpha(ts_last, eps)
    if terminal_sigma_floor > 0.0:
        a_t = a_t.clamp_max(math.exp(-terminal_sigma_floor))
    return a_t
