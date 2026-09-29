"""Shard-invariant deterministic top-1/top-M PRM guidance.

The historical evaluator seeded a process once with ``base_seed + shard_id``.
Consequently, a problem's samples depended on the shard layout, traversal
order, and whether a run resumed after a partial file.  This implementation
uses the audited semantic seed protocol and one explicit generator per
candidate row.

The matched-compute protocol has two 64-step stages.  Stage one evolves K
independent candidates and keeps the top M by PRM score.  Each survivor then
spawns K/M fresh continuations for stage two.  The terminal answer is the
highest-PRM candidate.  ``M=1`` is deterministic top-1 guidance; ``M=2`` is the
diversity-retaining baseline used in the rebuttal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from .protocol import derive_row_seeds
from .resume_denoise import (
    DenoiseConfig,
    initialize_denoise_state,
    run_denoise,
    state_from_post_step_snapshot,
)


@dataclass(frozen=True)
class StableTopMConfig:
    """Configuration for the two-stage matched-compute baseline."""

    particles: int = 8
    top_m: int = 2
    steps: int = 128
    checkpoint_step: int = 64
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)

    def validate(self, *, strict_fixed_protocol: bool = True) -> None:
        if self.particles <= 0:
            raise ValueError("particles must be positive")
        if self.top_m <= 0 or self.top_m > self.particles:
            raise ValueError("top_m must lie in [1, particles]")
        if self.particles % self.top_m:
            raise ValueError("particles must be divisible by top_m")
        if not (0 < self.checkpoint_step < self.steps):
            raise ValueError("checkpoint_step must be strictly inside the run")
        self.denoise.validate()
        if self.denoise.steps != self.steps:
            raise ValueError("baseline and denoise step counts differ")
        if strict_fixed_protocol:
            expected_denoise = DenoiseConfig()
            actual = (
                self.particles,
                self.top_m,
                self.steps,
                self.checkpoint_step,
                self.denoise,
            )
            expected = (
                8,
                self.top_m,
                128,
                64,
                expected_denoise,
            )
            if self.top_m not in {1, 2} or actual != expected:
                raise ValueError(
                    "formal top-M protocol is frozen at K=8, M in {1,2}, "
                    "T=128, checkpoint=64, and the default Dream sampler; "
                    f"got {actual}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deterministic_top_indices(
    scores: torch.Tensor, count: int
) -> torch.LongTensor:
    """Select descending scores with lowest-row-id tie breaking."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty vector")
    if not (1 <= count <= scores.numel()):
        raise ValueError("count must lie in [1, len(scores)]")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("PRM produced non-finite scores")
    order = sorted(
        range(scores.numel()),
        key=lambda index: (-float(scores[index].item()), index),
    )
    return torch.tensor(order[:count], dtype=torch.long, device=scores.device)


def _score_candidates(
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
    scores = prm(states, prompt_lengths, mask_ratios)
    if scores.shape != (particles,):
        raise ValueError(
            f"PRM must return one score per candidate, got {tuple(scores.shape)}"
        )
    if not torch.isfinite(scores).all():
        raise FloatingPointError("PRM produced non-finite scores")
    return scores, mask_ratios


def _select_rows(value: torch.Tensor | str, indices: torch.LongTensor):
    if torch.is_tensor(value):
        return value[indices].clone()
    return value


@torch.no_grad()
def prm_guided_topm_stable_sample(
    model,
    prm,
    input_ids: torch.LongTensor,
    *,
    base_seed: int,
    global_problem_id: int | str,
    config: StableTopMConfig | None = None,
    strict_fixed_protocol: bool = True,
) -> tuple[torch.LongTensor, dict[str, Any]]:
    """Return all K terminal candidates and an auditable selection trace."""
    baseline = config or StableTopMConfig()
    baseline.validate(strict_fixed_protocol=strict_fixed_protocol)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("stable top-M expects exactly one unpadded prompt")

    particles = baseline.particles
    prompt_length = input_ids.shape[-1]
    particle_ids = list(range(particles))

    # Deliberately use the same semantic stage labels for M=1 and M=2.
    # Their first-stage candidate pools therefore use common random numbers.
    stage1_seeds = derive_row_seeds(
        base_seed,
        global_problem_id,
        particle_ids,
        "guided_stage1",
    )
    stage1_initial = initialize_denoise_state(
        input_ids.expand(particles, -1).clone(),
        baseline.denoise,
        stage1_seeds,
    )
    stage1_state, _ = run_denoise(
        model,
        stage1_initial,
        baseline.denoise,
        stop_step=baseline.checkpoint_step,
    )
    stage1_scores, stage1_mask_ratios = _score_candidates(
        prm,
        stage1_state.x,
        prompt_length=prompt_length,
        mask_token_id=baseline.denoise.mask_token_id,
    )
    retained = deterministic_top_indices(stage1_scores, baseline.top_m)

    children_per_parent = particles // baseline.top_m
    parent_indices = retained.repeat_interleave(children_per_parent)
    if parent_indices.numel() != particles:
        raise AssertionError("parent expansion did not produce K rows")

    stage2_seeds = derive_row_seeds(
        base_seed,
        global_problem_id,
        particle_ids,
        "guided_stage2",
    )
    stage2_initial = state_from_post_step_snapshot(
        stage1_state.x[parent_indices].clone(),
        snapshot_step=baseline.checkpoint_step - 1,
        config=baseline.denoise,
        row_seeds=stage2_seeds,
        attention_mask=_select_rows(stage1_state.attention_mask, parent_indices),
        position_ids=(
            stage1_state.position_ids[parent_indices].clone()
            if stage1_state.position_ids is not None
            else None
        ),
        logical_denoising_calls=stage1_state.logical_denoising_calls,
        physical_forward_calls=stage1_state.physical_forward_calls,
    )
    terminal_state, _ = run_denoise(
        model,
        stage2_initial,
        baseline.denoise,
    )
    terminal_scores, terminal_mask_ratios = _score_candidates(
        prm,
        terminal_state.x,
        prompt_length=prompt_length,
        mask_token_id=baseline.denoise.mask_token_id,
    )
    terminal_retained = deterministic_top_indices(
        terminal_scores, baseline.top_m
    )
    selected_particle = int(
        deterministic_top_indices(terminal_scores, 1)[0].item()
    )

    expected_counts = {
        "logical_denoising": particles * baseline.steps,
        "logical_prm_scoring": 2 * particles,
        "logical_total": particles * baseline.steps + 2 * particles,
        "physical_denoising_forwards": baseline.steps,
        "physical_prm_forwards": 2,
    }
    actual_counts = {
        "logical_denoising": terminal_state.logical_denoising_calls,
        "logical_prm_scoring": 2 * particles,
        "logical_total": (
            terminal_state.logical_denoising_calls + 2 * particles
        ),
        "physical_denoising_forwards": terminal_state.physical_forward_calls,
        "physical_prm_forwards": 2,
    }
    if actual_counts != expected_counts:
        raise AssertionError(
            f"call-count mismatch: {actual_counts} != {expected_counts}"
        )

    info: dict[str, Any] = {
        "protocol": baseline.to_dict(),
        "base_seed": int(base_seed),
        "global_problem_id": global_problem_id,
        "stage1": {
            "row_seeds": stage1_seeds,
            "raw_prm_logits": stage1_scores.detach().float().cpu().tolist(),
            "mask_ratios": (
                stage1_mask_ratios.detach().float().cpu().tolist()
            ),
            "retained_particle_ids": retained.detach().cpu().tolist(),
            "tie_break": "descending_logit_then_lowest_particle_id",
        },
        "terminal": {
            "row_seeds": stage2_seeds,
            "parent_stage1_particle_ids": (
                parent_indices.detach().cpu().tolist()
            ),
            "raw_prm_logits": terminal_scores.detach().float().cpu().tolist(),
            "mask_ratios": (
                terminal_mask_ratios.detach().float().cpu().tolist()
            ),
            "retained_particle_ids": (
                terminal_retained.detach().cpu().tolist()
            ),
            "selected_particle_id": selected_particle,
            "tie_break": "descending_logit_then_lowest_particle_id",
        },
        "call_counts": actual_counts,
    }
    return terminal_state.x, info
