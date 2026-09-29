from types import SimpleNamespace

import torch
from torch import nn

from src.prm.resume_denoise import (
    DenoiseConfig,
    DenoiseResumeState,
    initialize_denoise_state,
    replace_row_rng_seeds,
    run_denoise,
)


class TinyDream(nn.Module):
    """State-dependent logits with enough entropy to exercise both RNG draws."""

    def __init__(self, vocab_size: int, mask_token_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def forward(self, x, attention_mask, position_ids):
        batch, length = x.shape
        vocab = torch.arange(self.vocab_size, device=x.device).view(1, 1, -1)
        positions = torch.arange(length, device=x.device).view(1, -1, 1)
        context = (x.masked_fill(x == self.mask_token_id, 0).sum(-1) % 7).view(
            batch, 1, 1
        )
        center = (positions + context) % self.vocab_size
        logits = -0.17 * (vocab - center).abs().float()
        return SimpleNamespace(logits=logits.expand(batch, -1, -1).contiguous())


def test_exact_resume_at_low_middle_and_high_nonterminal_steps():
    mask = 31
    config = DenoiseConfig(
        steps=8,
        gen_length=48,
        mask_token_id=mask,
        temperature=0.7,
        alg="maskgit_plus",
        alg_temp=0.5,
    )
    model = TinyDream(vocab_size=mask, mask_token_id=mask)
    prompt = torch.tensor([[2, 3, 4]], dtype=torch.long)
    initial = initialize_denoise_state(prompt, config, [12345])
    uninterrupted, captures = run_denoise(
        model,
        initial,
        config,
        capture_after_steps=(1, 3, 6),
    )
    for step in (1, 3, 6):
        restored = DenoiseResumeState.from_payload(
            captures[step].to_payload(), device="cpu"
        )
        resumed, _ = run_denoise(model, restored, config)
        assert torch.equal(resumed.x, uninterrupted.x)
        assert resumed.logical_denoising_calls == uninterrupted.logical_denoising_calls


def test_fresh_seed_changes_continuation_and_batch_order_does_not():
    mask = 31
    config = DenoiseConfig(
        steps=8,
        gen_length=64,
        mask_token_id=mask,
        temperature=0.9,
        alg_temp=0.7,
    )
    model = TinyDream(vocab_size=mask, mask_token_id=mask)
    prompts = torch.tensor([[2, 3], [2, 3]], dtype=torch.long)
    initial = initialize_denoise_state(prompts, config, [101, 202])
    partial, _ = run_denoise(model, initial, config, stop_step=4)
    continued, _ = run_denoise(model, partial, config)

    swapped = partial.clone()
    swapped.x = swapped.x.flip(0)
    swapped.row_rng_states = tuple(reversed(swapped.row_rng_states))
    swapped.row_seeds = tuple(reversed(swapped.row_seeds))
    swapped_result, _ = run_denoise(model, swapped, config)
    assert torch.equal(continued.x, swapped_result.x.flip(0))

    fresh = replace_row_rng_seeds(partial, [303, 404])
    fresh_result, _ = run_denoise(model, fresh, config)
    assert not torch.equal(continued.x, fresh_result.x)
