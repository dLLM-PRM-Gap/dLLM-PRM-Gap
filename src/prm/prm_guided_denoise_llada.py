"""
LLaDA-version PRM-Guided Denoising via Periodic Beam Search (Q4).

Mirrors prm_guided_denoise.py but for LLaDA's low-confidence remasking sampler.
Key differences:
 - LLaDA denoise step: argmax (or top-p) → confidence → keep top-K unmasks.
 - Batched forward over K candidates to save wall-clock.
 - PRM scoring uses LLaDADiffusionPRM (separate module with LLaDA backbone).
"""

import torch
import torch.nn.functional as F


LLADA_MASK_TOKEN_ID = 126336
LLADA_EOS_TOKEN_ID = 126081


@torch.no_grad()
def prm_guided_sample_llada(
    model,
    prm,
    input_ids: torch.LongTensor,
    steps: int = 128,
    branch_every: int = 16,
    branch_factor: int = 4,
    temperature: float = 1.0,
    top_p: float = 1.0,
    gen_length: int = 512,
    mask_token_id: int = LLADA_MASK_TOKEN_ID,
    verbose: bool = False,
    **_ignored_dream_kwargs,  # silently ignore alg, alg_temp, eps, etc.
):
    """PRM-guided LLaDA denoising.

    Args:
        model: LLaDA-8B model (for logits).
        prm:   LLaDADiffusionPRM (for scoring partial states).
        input_ids: [1, prompt_len] prompt tokens.
        steps: total denoising steps.
        branch_every: steps between branch/score points.
        branch_factor: K = number of branches per segment.
        gen_length: number of generation tokens to produce after prompt.

    Returns:
        (best_ids [1, prompt_len+gen_length], info dict).
    """
    device = input_ids.device
    B, prompt_len = input_ids.shape
    assert B == 1, "prm_guided_sample_llada expects batch size 1 input"
    total_len = prompt_len + gen_length

    # Initialize: prompt + all MASK
    x = torch.full(
        (1, total_len), mask_token_id, dtype=input_ids.dtype, device=device
    )
    x[:, :prompt_len] = input_ids

    tokens_per_step = gen_length / steps
    num_segments = steps // branch_every
    branch_info = []
    total_fwd = 0

    def denoise_batch_one_step(cands, step_counter):
        """Advance K candidates one LLaDA denoising step in-place.

        cands: [K, total_len]  (caller owns the tensor).
        step_counter: global step index (1-based).
        """
        nonlocal total_fwd
        total_fwd += 1

        # Forward pass — attention_mask all 1s.
        attn = torch.ones_like(cands, dtype=torch.long)
        outputs = model(input_ids=cands, attention_mask=attn)
        logits = outputs.logits  # [K, L, V]

        if temperature != 1.0:
            logits = logits / temperature

        gen_logits = logits[:, prompt_len:, :]  # [K, G, V]
        gen_ids = cands[:, prompt_len:]         # [K, G]
        K, G, V = gen_logits.shape

        # Sample predictions
        if top_p < 1.0:
            predicted = _sample_top_p(gen_logits, top_p)
        elif temperature > 0:
            probs = F.softmax(gen_logits, dim=-1)
            predicted = torch.multinomial(
                probs.reshape(-1, V), num_samples=1
            ).reshape(K, G)
        else:
            predicted = gen_logits.argmax(dim=-1)

        probs = F.softmax(gen_logits, dim=-1)
        confidence = probs.max(dim=-1).values  # [K, G]

        # Target number unmasked after this step
        target_unmask = min(int(step_counter * tokens_per_step), gen_length)

        is_mask = (gen_ids == mask_token_id)  # [K, G]
        already = (~is_mask).sum(dim=1)        # [K]
        to_unmask = target_unmask - already    # [K]

        new_gen = gen_ids.clone()
        for k in range(K):
            n = int(to_unmask[k].item())
            if n <= 0:
                continue
            mask_pos = is_mask[k].nonzero(as_tuple=True)[0]
            if mask_pos.numel() == 0:
                continue
            n = min(n, mask_pos.numel())
            conf_at_mask = confidence[k, mask_pos]
            _, top_idx = torch.topk(conf_at_mask, n)
            positions = mask_pos[top_idx]
            new_gen[k, positions] = predicted[k, positions]

        cands[:, prompt_len:] = new_gen

    global_step = 0

    for seg_idx in range(num_segments):
        # Branch: K copies of current x
        candidates = x.expand(branch_factor, -1).clone()

        # Evolve K candidates for branch_every steps
        for _ in range(branch_every):
            global_step += 1
            if global_step > steps:
                break
            denoise_batch_one_step(candidates, global_step)

        # Score with PRM
        mask_ratios_k = (
            (candidates[:, prompt_len:] == mask_token_id).float().mean(dim=1)
        )
        prompt_lens_k = torch.full(
            (branch_factor,), prompt_len, dtype=torch.long, device=device
        )
        scores = prm(candidates, prompt_lens_k, mask_ratios_k)

        best_idx = scores.argmax().item()
        x = candidates[best_idx].unsqueeze(0)

        branch_info.append({
            'segment': seg_idx,
            'mask_ratio_mean': mask_ratios_k.mean().item(),
            'scores': scores.cpu().tolist(),
            'best_score': scores[best_idx].item(),
            'selected_idx': best_idx,
        })

        if verbose:
            print(
                f"  seg {seg_idx}: mr={mask_ratios_k.mean():.3f} "
                f"scores={[f'{s:.3f}' for s in scores.tolist()]} "
                f"pick={best_idx}"
            )

    # Handle any remaining steps with no branching (K=1)
    while global_step < steps:
        global_step += 1
        denoise_batch_one_step(x, global_step)

    info = {
        'num_segments': num_segments,
        'branch_factor': branch_factor,
        'branch_info': branch_info,
        'total_forward_passes': total_fwd,
    }
    return x, info


def _sample_top_p(logits, top_p):
    """Top-p nucleus sampling per position — shape-preserving."""
    B, L, V = logits.shape
    probs = F.softmax(logits, dim=-1)
    sp, si = probs.sort(dim=-1, descending=True)
    cs = sp.cumsum(dim=-1)
    mask = cs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sp.masked_fill_(mask, 0)
    sp = sp / sp.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    sample_idx = torch.multinomial(
        sp.reshape(-1, V), num_samples=1
    ).reshape(B, L)
    return si.gather(dim=-1, index=sample_idx.unsqueeze(-1)).squeeze(-1)
