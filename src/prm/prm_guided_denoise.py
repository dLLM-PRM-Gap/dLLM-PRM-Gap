"""
Phase 4: PRM-Guided Denoising via Periodic Beam Search.

Core algorithm: During Dream-7B's 128-step denoising process, periodically
branch K paths, evolve each independently, score with PRM, keep the best.

This is uniquely enabled by dLLMs — AR models cannot branch mid-generation
because each token depends on all previous tokens. In dLLMs, partial states
are self-contained and can be independently continued.

Usage:
    from prm.prm_guided_denoise import prm_guided_sample

    output, info = prm_guided_sample(
        model=dream_model,
        prm=diffusion_prm,
        input_ids=prompt_ids,
        steps=128,
        branch_every=16,
        branch_factor=4,
    )
"""

import torch
import torch.nn.functional as F


MASK_TOKEN_ID = 151666


@torch.no_grad()
def prm_guided_sample(
    model,
    prm,
    input_ids: torch.LongTensor,
    steps: int = 128,
    branch_every: int = 16,
    branch_factor: int = 4,
    temperature: float = 0.5,
    top_p: float = 1.0,
    alg: str = 'maskgit_plus',
    alg_temp: float = 0.5,
    mask_token_id: int = MASK_TOKEN_ID,
    gen_length: int = 512,
    eps: float = 1e-5,
    verbose: bool = False,
):
    """PRM-Guided denoising via Periodic Branching.

    Algorithm:
        1. Initialize: pad prompt with MASK tokens to prompt_len + gen_length
        2. For each segment of `branch_every` steps:
           a. Create K copies of current best state
           b. Evolve each copy for `branch_every` steps (standard denoising)
           c. Score all K copies with PRM (per-candidate mask_ratio)
           d. Keep the highest-scored candidate
        3. Return the final best sequence

    Args:
        model: Dream-7B model (for denoising)
        prm: DiffusionPRM model (for scoring partial states)
        input_ids: [1, prompt_len] prompt token IDs
        steps: total denoising steps (default 128)
        branch_every: steps between branch points (default 16)
        branch_factor: K = number of branches at each point (default 4)
        temperature, top_p, alg, alg_temp: denoising hyperparameters
        mask_token_id: MASK token ID for Dream-7B
        gen_length: generation length (default 512)
        eps: minimum timestep

    Returns:
        output: [1, prompt_len + gen_length] tensor (best sequence)
        info: dict with branching history and scores
    """
    from dream.generation_utils import sample_tokens

    device = input_ids.device
    prompt_len = input_ids.shape[1]
    max_length = prompt_len + gen_length

    # Initialize with MASK tokens
    x = F.pad(input_ids, (0, max_length - prompt_len), value=mask_token_id)

    # Timestep schedule
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    # Branching schedule
    num_segments = steps // branch_every
    branch_info = []

    for seg_idx in range(num_segments):
        seg_start = seg_idx * branch_every
        seg_end = min((seg_idx + 1) * branch_every, steps)

        # --- Branch: create K copies ---
        candidates = x.expand(branch_factor, -1).clone()

        # --- Evolve each candidate for branch_every steps ---
        for step_in_seg in range(seg_end - seg_start):
            abs_step = seg_start + step_in_seg
            t = timesteps[abs_step]
            s = timesteps[abs_step + 1]
            is_last = (abs_step == steps - 1)

            mask_index = (candidates == mask_token_id)
            if not mask_index.any():
                break

            # Batch forward pass for all K candidates
            logits = model(candidates, "full", None).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # Apply denoising per candidate
            _batch_denoise_step(
                candidates, logits, mask_index, t, s,
                temperature, top_p, alg, alg_temp,
                mask_token_id, is_last, device, sample_tokens,
            )

        # --- Score candidates with PRM (per-candidate mask_ratio) ---
        mask_ratios_k = (
            (candidates[:, prompt_len:] == mask_token_id)
            .float().mean(dim=1)
        )
        prompt_lens_k = torch.full(
            (branch_factor,), prompt_len, dtype=torch.long, device=device
        )

        scores = prm(candidates, prompt_lens_k, mask_ratios_k)

        # --- Keep best candidate ---
        best_idx = scores.argmax().item()
        x = candidates[best_idx].unsqueeze(0)

        seg_info = {
            'segment': seg_idx,
            'step_range': (seg_start, seg_end),
            'mask_ratio_mean': mask_ratios_k.mean().item(),
            'scores': scores.cpu().tolist(),
            'best_score': scores[best_idx].item(),
            'selected_idx': best_idx,
        }
        branch_info.append(seg_info)

        if verbose:
            print(
                f"  Segment {seg_idx}: mask_ratio={mask_ratios_k.mean():.3f} "
                f"scores={[f'{s:.3f}' for s in scores.tolist()]} "
                f"selected={best_idx}"
            )

    # Handle remaining steps after last full segment
    remaining_start = num_segments * branch_every
    if remaining_start < steps:
        for abs_step in range(remaining_start, steps):
            t = timesteps[abs_step]
            s = timesteps[abs_step + 1]
            is_last = (abs_step == steps - 1)

            mask_index = (x == mask_token_id)
            if not mask_index.any():
                break

            logits = model(x, "full", None).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            _batch_denoise_step(
                x, logits, mask_index, t, s,
                temperature, top_p, alg, alg_temp,
                mask_token_id, is_last, device, sample_tokens,
            )

    info = {
        'num_segments': num_segments,
        'branch_factor': branch_factor,
        'branch_info': branch_info,
        'total_forward_passes': (
            num_segments * branch_factor * branch_every  # denoising
            + num_segments * branch_factor               # PRM scoring
        ),
    }

    return x, info


def _batch_denoise_step(
    candidates, logits, mask_index, t, s,
    temperature, top_p, alg, alg_temp,
    mask_token_id, is_last, device, sample_tokens,
):
    """Apply one denoising step to a batch of candidates in-place.

    Each candidate may have different mask positions, so we process
    them individually (batched model forward is above, this is just
    the token selection per candidate).
    """
    K = candidates.shape[0]

    for k in range(K):
        k_mask = mask_index[k]
        if not k_mask.any():
            continue

        k_logits = logits[k][k_mask]

        if alg == 'origin':
            p_transfer = 1 - s / t if not is_last else 1
            x0 = torch.full_like(
                candidates[k][k_mask], mask_token_id, dtype=torch.long
            )
            transfer_idx = torch.rand(*x0.shape, device=device) < p_transfer
            if transfer_idx.any():
                _, sampled = sample_tokens(
                    k_logits[transfer_idx],
                    temperature=temperature, top_p=top_p
                )
                x0[transfer_idx] = sampled
            candidates[k][k_mask] = x0
        else:
            if alg == 'maskgit_plus':
                confidence, x0 = sample_tokens(
                    k_logits, temperature=temperature, top_p=top_p
                )
            elif alg == 'margin':
                confidence, x0 = sample_tokens(
                    k_logits, temperature=temperature, top_p=top_p,
                    margin_confidence=True
                )
            elif alg == 'entropy':
                confidence, x0 = sample_tokens(
                    k_logits, temperature=temperature, top_p=top_p,
                    neg_entropy=True
                )
            else:
                raise ValueError(f"Unknown alg: {alg}")

            num_mask = k_mask.sum()
            n_transfer = int(num_mask * (1 - s / t)) if not is_last else num_mask
            if n_transfer > 0:
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(confidence, n_transfer)
                else:
                    conf_scaled = confidence / alg_temp
                    conf_scaled = F.softmax(conf_scaled, dim=-1)
                    transfer_index = torch.multinomial(
                        conf_scaled, num_samples=n_transfer
                    )
                x0_ = torch.full_like(x0, mask_token_id, dtype=torch.long)
                x0_[transfer_index] = x0[transfer_index].clone()
                candidates[k][k_mask] = x0_
