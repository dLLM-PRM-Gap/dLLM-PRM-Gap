from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.prm.prm_guided_topm_stable import (
    StableTopMConfig,
    deterministic_top_indices,
    prm_guided_topm_stable_sample,
)
from src.prm.resume_denoise import DenoiseConfig


class TinyDream(nn.Module):
    def __init__(self, vocab_size: int, mask_token_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def forward(self, x, attention_mask, position_ids):
        batch, length = x.shape
        vocab = torch.arange(self.vocab_size, device=x.device).view(1, 1, -1)
        positions = torch.arange(length, device=x.device).view(1, -1, 1)
        context = (x.masked_fill(x == self.mask_token_id, 0).sum(-1) % 13).view(
            batch, 1, 1
        )
        center = (positions + context) % self.vocab_size
        logits = -0.1 * (vocab - center).abs().float()
        return SimpleNamespace(logits=logits.expand(batch, -1, -1).contiguous())


class TinyPRM(nn.Module):
    def forward(self, input_ids, prompt_lens, mask_ratios):
        values = torch.stack(
            [
                row[int(prompt_len.item()) :].float().sum()
                for row, prompt_len in zip(input_ids, prompt_lens)
            ]
        )
        return (values.remainder(31) - 15) / 4 - mask_ratios


def small_config(top_m: int) -> StableTopMConfig:
    mask = 37
    denoise = DenoiseConfig(
        steps=8,
        gen_length=40,
        mask_token_id=mask,
        temperature=0.8,
        alg_temp=0.6,
    )
    return StableTopMConfig(
        particles=4,
        top_m=top_m,
        steps=8,
        checkpoint_step=4,
        denoise=denoise,
    )


def run_one(problem_id: int, top_m: int):
    config = small_config(top_m)
    model = TinyDream(
        vocab_size=config.denoise.mask_token_id,
        mask_token_id=config.denoise.mask_token_id,
    )
    prompt = torch.tensor([[2, 3, 4]], dtype=torch.long)
    return prm_guided_topm_stable_sample(
        model,
        TinyPRM(),
        prompt,
        base_seed=42,
        global_problem_id=problem_id,
        config=config,
        strict_fixed_protocol=False,
    )


def test_deterministic_top_indices_breaks_ties_by_lowest_id():
    scores = torch.tensor([0.5, 0.7, 0.7, -1.0])
    assert deterministic_top_indices(scores, 3).tolist() == [1, 2, 0]
    with pytest.raises(FloatingPointError):
        deterministic_top_indices(torch.tensor([0.0, float("nan")]), 1)


@pytest.mark.parametrize("top_m", [1, 2])
def test_topm_is_reproducible_and_has_exact_matched_compute(top_m):
    first_states, first_info = run_one(17, top_m)
    second_states, second_info = run_one(17, top_m)
    assert torch.equal(first_states, second_states)
    assert first_info == second_info
    assert first_info["call_counts"] == {
        "logical_denoising": 32,
        "logical_prm_scoring": 8,
        "logical_total": 40,
        "physical_denoising_forwards": 8,
        "physical_prm_forwards": 2,
    }
    assert first_states.shape == (4, 43)
    assert len(first_info["stage1"]["retained_particle_ids"]) == top_m
    assert len(first_info["terminal"]["parent_stage1_particle_ids"]) == 4


def test_problem_stream_is_invariant_to_traversal_order():
    first_states, first_info = run_one(23, 2)
    run_one(99, 2)
    repeated_states, repeated_info = run_one(23, 2)
    assert torch.equal(first_states, repeated_states)
    assert first_info == repeated_info


def test_top1_and_topm2_share_the_same_first_stage_pool():
    _, top1 = run_one(8, 1)
    _, topm2 = run_one(8, 2)
    assert top1["stage1"]["row_seeds"] == topm2["stage1"]["row_seeds"]
    assert top1["stage1"]["raw_prm_logits"] == topm2["stage1"]["raw_prm_logits"]
    assert top1["stage1"]["mask_ratios"] == topm2["stage1"]["mask_ratios"]
    assert (
        top1["stage1"]["retained_particle_ids"][0]
        == topm2["stage1"]["retained_particle_ids"][0]
    )


def test_formal_config_is_locked():
    StableTopMConfig(top_m=1).validate(strict_fixed_protocol=True)
    StableTopMConfig(top_m=2).validate(strict_fixed_protocol=True)
    with pytest.raises(ValueError, match="formal top-M protocol"):
        StableTopMConfig(top_m=4).validate(strict_fixed_protocol=True)
    with pytest.raises(ValueError, match="formal top-M protocol"):
        StableTopMConfig(
            top_m=2,
            denoise=DenoiseConfig(temperature=0.7),
        ).validate(strict_fixed_protocol=True)
