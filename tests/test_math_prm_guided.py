from types import SimpleNamespace

import torch
from torch import nn

from src.prm.math_prm_guided import (
    MathPRMGuidedConfig,
    math_prm_guided_sample,
)
from src.prm.protocol import derive_row_seeds
from src.prm.resume_denoise import DenoiseConfig


class TinyDream(nn.Module):
    def __init__(self, vocab_size, mask_token_id):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def forward(self, x, attention_mask, position_ids):
        del attention_mask, position_ids
        batch, length = x.shape
        vocab = torch.arange(self.vocab_size).view(1, 1, -1)
        positions = torch.arange(length).view(1, -1, 1)
        context = (x.masked_fill(x == self.mask_token_id, 0).sum(-1) % 5).view(
            batch, 1, 1
        )
        center = (positions + context) % self.vocab_size
        logits = -0.2 * (vocab - center).abs().float()
        return SimpleNamespace(logits=logits.expand(batch, -1, -1).contiguous())


class TinyPRM(nn.Module):
    def forward(self, input_ids, prompt_lens, mask_ratios):
        del prompt_lens
        return input_ids.float().sum(dim=-1) / 100 + mask_ratios


def test_guided_row_rng_is_repeatable_and_call_accounting_is_exact():
    mask = 31
    config = MathPRMGuidedConfig(
        particles=3,
        branch_every=2,
        denoise=DenoiseConfig(
            steps=4,
            gen_length=12,
            mask_token_id=mask,
            temperature=0.7,
            top_p=1.0,
            alg="maskgit_plus",
            alg_temp=0.5,
        ),
    )
    kwargs = {
        "model": TinyDream(vocab_size=mask, mask_token_id=mask),
        "prm": TinyPRM(),
        "input_ids": torch.tensor([[2, 3, 4]], dtype=torch.long),
        "base_seed": 42,
        "global_problem_id": "test/algebra/1.json",
        "config": config,
        "strict_formal": False,
    }
    first, first_info = math_prm_guided_sample(**kwargs)
    torch.manual_seed(999)
    _ = torch.rand(100)
    second, second_info = math_prm_guided_sample(**kwargs)
    assert torch.equal(first, second)
    assert first_info == second_info
    assert first_info["call_counts"] == {
        "logical_denoising": 12,
        "logical_prm_scoring": 6,
        "logical_total": 18,
        "physical_denoising_forwards": 4,
        "physical_prm_forwards": 2,
    }
    assert first_info["stage1"]["row_seeds"] == derive_row_seeds(
        42,
        "test/algebra/1.json",
        [0, 1, 2],
        "math_prm_guided_segment_0",
    )
    assert first_info["terminal"]["mask_ratios"] == [0.0, 0.0, 0.0]


def test_formal_guided_config_rejects_sampler_or_branch_changes():
    MathPRMGuidedConfig().validate(strict_formal=True)
    changed = MathPRMGuidedConfig(branch_every=32)
    try:
        changed.validate(strict_formal=True)
    except ValueError as error:
        assert "fixed" in str(error)
    else:
        raise AssertionError("changed formal guidance protocol was accepted")
