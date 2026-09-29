"""Two-stage ESS-tempered SMC for matched-compute PRM guidance.

The formal rebuttal protocol is fixed to K=8, T=128, checkpoints after steps
64 and 128, ESS target 4, resampling threshold 6, and beta grid
``{0, .05, ..., 1}``.  The implementation keeps the mathematical utilities
parameterized only so they can be tested cheaply; formal callers should leave
``strict_fixed_protocol=True``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import torch

from .protocol import derive_row_seeds, derive_seed
from .resume_denoise import (
    DenoiseConfig,
    initialize_denoise_state,
    run_denoise,
    state_from_post_step_snapshot,
)


@dataclass(frozen=True)
class SMCConfig:
    particles: int = 8
    steps: int = 128
    checkpoint_step: int = 64
    ess_target: float = 4.0
    resample_threshold: float = 6.0
    beta_grid: tuple[float, ...] = field(
        default_factory=lambda: tuple(index / 20 for index in range(21))
    )
    probability_clip: float = 1e-6
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)

    def validate(self, *, strict_fixed_protocol: bool = True) -> None:
        if self.particles <= 1:
            raise ValueError("SMC requires at least two particles")
        if self.steps <= 1:
            raise ValueError("SMC requires at least two denoising steps")
        if not (0 < self.checkpoint_step < self.steps):
            raise ValueError("checkpoint_step must be strictly inside the run")
        if not (1 <= self.ess_target <= self.particles):
            raise ValueError("ess_target outside [1, K]")
        if not (self.ess_target <= self.resample_threshold <= self.particles):
            raise ValueError("resample_threshold outside [ess_target, K]")
        if not self.beta_grid or self.beta_grid[0] != 0 or self.beta_grid[-1] != 1:
            raise ValueError("beta_grid must span exactly from 0 to 1")
        if tuple(sorted(set(self.beta_grid))) != self.beta_grid:
            raise ValueError("beta_grid must be sorted and unique")
        if not (0 < self.probability_clip < 0.5):
            raise ValueError("probability_clip must lie in (0, .5)")
        self.denoise.validate()
        if self.denoise.steps != self.steps:
            raise ValueError("SMC and denoise step counts differ")
        if strict_fixed_protocol:
            expected_grid = tuple(index / 20 for index in range(21))
            actual = (
                self.particles,
                self.steps,
                self.checkpoint_step,
                self.ess_target,
                self.resample_threshold,
                self.beta_grid,
            )
            expected = (8, 128, 64, 4.0, 6.0, expected_grid)
            if actual != expected:
                raise ValueError(
                    f"formal SMC protocol is frozen at {expected}, got {actual}"
                )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["beta_grid"] = list(self.beta_grid)
        return result


def normalize_weights(unnormalized: torch.Tensor) -> torch.Tensor:
    """Normalize non-negative weights in float64 with hard validity checks."""
    values = unnormalized.to(dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("weights must be a non-empty vector")
    if not torch.isfinite(values).all() or torch.any(values < 0):
        raise FloatingPointError("weights must be finite and non-negative")
    total = values.sum()
    if not torch.isfinite(total) or total <= 0:
        raise FloatingPointError("weight sum must be positive and finite")
    return values / total


def effective_sample_size(weights: torch.Tensor) -> float:
    normalized = normalize_weights(weights)
    return float((1.0 / normalized.square().sum()).item())


def choose_tempering_beta(
    probabilities: torch.Tensor,
    beta_grid: Sequence[float],
    ess_target: float,
) -> tuple[float, torch.Tensor, float]:
    """Choose the largest frozen-grid beta whose normalized ESS is >= target."""
    h = probabilities.to(dtype=torch.float64)
    if h.ndim != 1 or h.numel() == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if not torch.isfinite(h).all() or torch.any(h <= 0) or torch.any(h >= 1):
        raise ValueError("probabilities must be finite and strictly in (0, 1)")
    selected: tuple[float, torch.Tensor, float] | None = None
    for beta in beta_grid:
        weights = normalize_weights(h.pow(float(beta)))
        ess = effective_sample_size(weights)
        if ess + 1e-12 >= ess_target:
            selected = float(beta), weights, ess
    if selected is None:
        # beta=0 is uniform, so a valid target <= K must always be feasible.
        raise RuntimeError("no beta satisfies the ESS target")
    return selected


def systematic_resample(
    weights: torch.Tensor,
    *,
    seed: int,
) -> torch.LongTensor:
    """Systematic resampling with a deterministic CPU RNG stream."""
    normalized = normalize_weights(weights).cpu()
    particles = normalized.numel()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    offset = float(torch.rand((), generator=generator).item()) / particles
    positions = offset + torch.arange(particles, dtype=torch.float64) / particles
    cumulative = torch.cumsum(normalized, dim=0)
    cumulative[-1] = 1.0
    ancestors = torch.searchsorted(cumulative, positions, right=False)
    return ancestors.to(dtype=torch.long, device=weights.device)


def _score_particles(
    prm,
    states: torch.LongTensor,
    *,
    prompt_length: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    particles = states.shape[0]
    mask_ratios = (
        states[:, prompt_length:] == mask_token_id
    ).float().mean(dim=1)
    prompt_lengths = torch.full(
        (particles,),
        prompt_length,
        device=states.device,
        dtype=torch.long,
    )
    logits = prm(states, prompt_lengths, mask_ratios)
    if logits.shape != (particles,):
        raise ValueError(
            f"PRM must return one logit per particle, got {tuple(logits.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise FloatingPointError("PRM produced non-finite logits")
    return logits, mask_ratios


@torch.no_grad()
def prm_guided_smc_sample(
    model,
    prm,
    input_ids: torch.LongTensor,
    *,
    base_seed: int,
    global_problem_id: int | str,
    config: SMCConfig | None = None,
    strict_fixed_protocol: bool = True,
) -> tuple[torch.LongTensor, dict[str, Any]]:
    """Run the fixed two-stage SMC protocol and return all final particles."""
    smc = config or SMCConfig()
    smc.validate(strict_fixed_protocol=strict_fixed_protocol)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("SMC expects exactly one unpadded prompt")
    prompt_length = input_ids.shape[-1]
    particles = smc.particles

    stage1_seeds = derive_row_seeds(
        base_seed,
        global_problem_id,
        list(range(particles)),
        "smc_stage1",
    )
    prompt_batch = input_ids.expand(particles, -1).clone()
    initial = initialize_denoise_state(
        prompt_batch, smc.denoise, stage1_seeds
    )
    stage1_state, _ = run_denoise(
        model, initial, smc.denoise, stop_step=smc.checkpoint_step
    )
    logits1, mask_ratios1 = _score_particles(
        prm,
        stage1_state.x,
        prompt_length=prompt_length,
        mask_token_id=smc.denoise.mask_token_id,
    )
    h1 = torch.sigmoid(logits1.float()).clamp(
        min=smc.probability_clip,
        max=1 - smc.probability_clip,
    ).to(dtype=torch.float64)
    beta1, weights1, ess1 = choose_tempering_beta(
        h1, smc.beta_grid, smc.ess_target
    )
    potential1 = h1.pow(beta1)

    resampled = ess1 < smc.resample_threshold
    resampling_seed = derive_seed(
        base_seed, global_problem_id, 0, "smc_resample_step64"
    )
    if resampled:
        ancestors = systematic_resample(weights1, seed=resampling_seed)
        post_resample_weights = torch.full(
            (particles,),
            1.0 / particles,
            device=input_ids.device,
            dtype=torch.float64,
        )
    else:
        ancestors = torch.arange(particles, device=input_ids.device)
        post_resample_weights = weights1.to(device=input_ids.device)

    stage2_seeds = derive_row_seeds(
        base_seed,
        global_problem_id,
        list(range(particles)),
        "smc_stage2",
    )
    selected_attention: torch.Tensor | str
    if torch.is_tensor(stage1_state.attention_mask):
        selected_attention = stage1_state.attention_mask[ancestors].clone()
    else:
        selected_attention = stage1_state.attention_mask
    selected_positions = (
        stage1_state.position_ids[ancestors].clone()
        if stage1_state.position_ids is not None
        else None
    )
    stage2_initial = state_from_post_step_snapshot(
        stage1_state.x[ancestors].clone(),
        snapshot_step=smc.checkpoint_step - 1,
        config=smc.denoise,
        row_seeds=stage2_seeds,
        attention_mask=selected_attention,
        position_ids=selected_positions,
        logical_denoising_calls=stage1_state.logical_denoising_calls,
        physical_forward_calls=stage1_state.physical_forward_calls,
    )
    final_state, _ = run_denoise(model, stage2_initial, smc.denoise)
    logits2, mask_ratios2 = _score_particles(
        prm,
        final_state.x,
        prompt_length=prompt_length,
        mask_token_id=smc.denoise.mask_token_id,
    )
    h2 = torch.sigmoid(logits2.float()).clamp(
        min=smc.probability_clip,
        max=1 - smc.probability_clip,
    ).to(dtype=torch.float64)

    # u2_i = V1_i * H2_i / H1_{A_i}, evaluated in log space.
    selected_potential1 = potential1.to(input_ids.device)[ancestors]
    log_unnormalized2 = (
        torch.log(post_resample_weights)
        + torch.log(h2)
        - torch.log(selected_potential1)
    )
    weights2 = torch.softmax(log_unnormalized2, dim=0)
    if not torch.isfinite(weights2).all():
        raise FloatingPointError("terminal SMC weights are non-finite")

    expected_denoising_calls = particles * smc.steps
    expected_prm_calls = 2 * particles
    if final_state.logical_denoising_calls != expected_denoising_calls:
        raise AssertionError(
            "denoising call mismatch: "
            f"{final_state.logical_denoising_calls} != {expected_denoising_calls}"
        )
    unique_ancestors = int(torch.unique(ancestors).numel())
    info: dict[str, Any] = {
        "protocol": smc.to_dict(),
        "base_seed": int(base_seed),
        "global_problem_id": global_problem_id,
        "stage1": {
            "row_seeds": stage1_seeds,
            "raw_prm_logits": logits1.detach().float().cpu().tolist(),
            "probabilities": h1.cpu().tolist(),
            "mask_ratios": mask_ratios1.detach().float().cpu().tolist(),
            "beta": beta1,
            "normalized_weights": weights1.cpu().tolist(),
            "ess": ess1,
        },
        "resampling": {
            "triggered": resampled,
            "seed": resampling_seed,
            "ancestors": ancestors.detach().cpu().tolist(),
            "post_resample_weights": post_resample_weights.cpu().tolist(),
            "unique_ancestor_count": unique_ancestors,
        },
        "terminal": {
            "row_seeds": stage2_seeds,
            "raw_prm_logits": logits2.detach().float().cpu().tolist(),
            "probabilities": h2.cpu().tolist(),
            "mask_ratios": mask_ratios2.detach().float().cpu().tolist(),
            "normalized_weights": weights2.detach().cpu().tolist(),
            "top_particle_index": int(torch.argmax(logits2).item()),
            "final_lineage_count": unique_ancestors,
        },
        "call_counts": {
            "logical_denoising": final_state.logical_denoising_calls,
            "logical_prm_scoring": expected_prm_calls,
            "logical_total": (
                final_state.logical_denoising_calls + expected_prm_calls
            ),
            "physical_denoising_forwards": final_state.physical_forward_calls,
            "physical_prm_forwards": 2,
        },
    }
    return final_state.x, info
