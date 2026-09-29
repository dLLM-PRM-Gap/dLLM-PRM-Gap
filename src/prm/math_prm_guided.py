"""Stable-row-RNG deterministic PRM guidance for the MATH rebuttal task."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from .math_candidate_pool import fixed_denoise_config
from .protocol import derive_row_seeds
from .resume_denoise import (
    DenoiseConfig,
    initialize_denoise_state,
    run_denoise,
    state_from_post_step_snapshot,
)


GUIDED_PROTOCOL_ID = "math500_prm_guided_k8_branch64_row_rng_v1"
STAGE1_SEED_LABEL = "math_prm_guided_segment_0"
STAGE2_SEED_LABEL = "math_prm_guided_segment_1"


@dataclass(frozen=True)
class MathPRMGuidedConfig:
    particles: int = 8
    branch_every: int = 64
    denoise: DenoiseConfig = field(default_factory=fixed_denoise_config)

    def validate(self, *, strict_formal: bool = True) -> None:
        self.denoise.validate()
        if self.particles <= 1:
            raise ValueError("PRM guidance requires at least two particles")
        if not (0 < self.branch_every < self.denoise.steps):
            raise ValueError("branch_every must be strictly inside the run")
        if self.denoise.steps % self.branch_every != 0:
            raise ValueError("denoising steps must divide into equal branches")
        if strict_formal:
            expected = MathPRMGuidedConfig()
            if (
                self.particles != expected.particles
                or self.branch_every != expected.branch_every
                or asdict(self.denoise) != asdict(expected.denoise)
            ):
                raise ValueError(
                    "formal MATH guidance is fixed to K=8, branch_every=64, "
                    "and the canonical MATH sampler"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": GUIDED_PROTOCOL_ID,
            "particles": self.particles,
            "branch_every": self.branch_every,
            "segments": self.denoise.steps // self.branch_every,
            "denoise": asdict(self.denoise),
        }


def _score_states(
    prm,
    states: torch.LongTensor,
    *,
    prompt_length: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    particles = int(states.shape[0])
    mask_ratios = (states[:, prompt_length:] == mask_token_id).float().mean(dim=1)
    prompt_lengths = torch.full(
        (particles,),
        prompt_length,
        dtype=torch.long,
        device=states.device,
    )
    logits = prm(states, prompt_lengths, mask_ratios)
    if tuple(logits.shape) != (particles,):
        raise ValueError(
            f"PRM must return one logit per particle, got {tuple(logits.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise FloatingPointError("PRM produced non-finite logits")
    return logits, mask_ratios


@torch.no_grad()
def math_prm_guided_sample(
    model,
    prm,
    input_ids: torch.LongTensor,
    *,
    base_seed: int,
    global_problem_id: int | str,
    config: MathPRMGuidedConfig | None = None,
    strict_formal: bool = True,
) -> tuple[torch.LongTensor, dict[str, Any]]:
    """Run two top-1 selection stages and return all eight terminal branches."""
    guided = config or MathPRMGuidedConfig()
    guided.validate(strict_formal=strict_formal)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("guided sampling expects one unpadded prompt")
    prompt_length = int(input_ids.shape[-1])
    particle_ids = list(range(guided.particles))

    stage1_seeds = derive_row_seeds(
        base_seed,
        global_problem_id,
        particle_ids,
        STAGE1_SEED_LABEL,
    )
    stage1_initial = initialize_denoise_state(
        input_ids.expand(guided.particles, -1).clone(),
        guided.denoise,
        stage1_seeds,
    )
    stage1_state, _ = run_denoise(
        model,
        stage1_initial,
        guided.denoise,
        stop_step=guided.branch_every,
    )
    stage1_logits, stage1_ratios = _score_states(
        prm,
        stage1_state.x,
        prompt_length=prompt_length,
        mask_token_id=guided.denoise.mask_token_id,
    )
    # torch.argmax returns the first maximum, implementing the frozen
    # lowest-particle-id tie break.
    stage1_selected = int(torch.argmax(stage1_logits).item())

    stage2_seeds = derive_row_seeds(
        base_seed,
        global_problem_id,
        particle_ids,
        STAGE2_SEED_LABEL,
    )
    selected_tokens = (
        stage1_state.x[stage1_selected : stage1_selected + 1]
        .expand(guided.particles, -1)
        .clone()
    )
    selected_attention: torch.Tensor | str
    if torch.is_tensor(stage1_state.attention_mask):
        source_attention = stage1_state.attention_mask
        attention_index = stage1_selected if source_attention.shape[0] > 1 else 0
        selected_attention = (
            source_attention[attention_index : attention_index + 1]
            .expand(guided.particles, *source_attention.shape[1:])
            .clone()
        )
    else:
        selected_attention = stage1_state.attention_mask
    selected_positions = None
    if stage1_state.position_ids is not None:
        source_positions = stage1_state.position_ids
        position_index = stage1_selected if source_positions.shape[0] > 1 else 0
        selected_positions = (
            source_positions[position_index : position_index + 1]
            .expand(guided.particles, -1)
            .clone()
        )
    stage2_initial = state_from_post_step_snapshot(
        selected_tokens,
        snapshot_step=guided.branch_every - 1,
        config=guided.denoise,
        row_seeds=stage2_seeds,
        attention_mask=selected_attention,
        position_ids=selected_positions,
        logical_denoising_calls=stage1_state.logical_denoising_calls,
        physical_forward_calls=stage1_state.physical_forward_calls,
    )
    terminal_state, _ = run_denoise(
        model,
        stage2_initial,
        guided.denoise,
    )
    terminal_logits, terminal_ratios = _score_states(
        prm,
        terminal_state.x,
        prompt_length=prompt_length,
        mask_token_id=guided.denoise.mask_token_id,
    )
    if torch.any(terminal_ratios != 0):
        raise AssertionError("guided terminal branches retain MASK tokens")
    terminal_selected = int(torch.argmax(terminal_logits).item())

    expected_denoising = guided.particles * guided.denoise.steps
    expected_scorer = 2 * guided.particles
    if terminal_state.logical_denoising_calls != expected_denoising:
        raise AssertionError("guided logical denoising call count mismatch")
    if terminal_state.physical_forward_calls != guided.denoise.steps:
        raise AssertionError("guided physical denoising call count mismatch")
    info = {
        "protocol": guided.to_dict(),
        "base_seed": int(base_seed),
        "global_problem_id": str(global_problem_id),
        "stage1": {
            "row_seeds": stage1_seeds,
            "raw_prm_logits": stage1_logits.detach().float().cpu().tolist(),
            "mask_ratios": stage1_ratios.detach().float().cpu().tolist(),
            "selected_particle_id": stage1_selected,
            "tie_break": "lowest_particle_id",
        },
        "terminal": {
            "row_seeds": stage2_seeds,
            "raw_prm_logits": (terminal_logits.detach().float().cpu().tolist()),
            "mask_ratios": terminal_ratios.detach().float().cpu().tolist(),
            "selected_particle_id": terminal_selected,
            "tie_break": "lowest_particle_id",
        },
        "call_counts": {
            "logical_denoising": expected_denoising,
            "logical_prm_scoring": expected_scorer,
            "logical_total": expected_denoising + expected_scorer,
            "physical_denoising_forwards": (terminal_state.physical_forward_calls),
            "physical_prm_forwards": 2,
        },
    }
    return terminal_state.x, info
