"""
LLaDA-8B generation utility: low-confidence remasking sampling.

LLaDA's sampling scheme (per official repo ML-GSAI/LLaDA):
1. Start with all mask tokens in generation region
2. Each step: forward pass → greedy/top-p pick token at each mask position
3. Compute confidence (max prob after softmax or pre-softmax logit)
4. Keep top-K confident unmasks; re-mask the rest
5. Repeat until all unmasked

This matches Dream's denoising loop structure, so the same hook_fn callback
can be used for PRM-guided branching.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def llada_sample(
    model,
    tokenizer,
    prompt_ids,  # [B, P] int tensor
    gen_length=128,
    steps=128,
    temperature=1.0,
    top_p=1.0,
    remask_strategy="low_confidence",  # or "random"
    hook_fn=None,  # (step_idx, mask_ratio, current_ids, ...) → None or override_ids
    mask_id=126336,
    eos_id=126081,
):
    """LLaDA low-confidence remasking generation.

    Args:
        prompt_ids: (B, P) prompt tokens (no mask tokens here)
        gen_length: number of tokens to generate after prompt
        steps: denoising steps (usually = gen_length for 1 token/step, but can differ)
        hook_fn: callback after each denoising step (for PRM branching)
            Signature: hook_fn(step_idx, mask_ratio, input_ids, logits) → input_ids or None
            If returns new input_ids, use them for next step (allows branching).

    Returns:
        Tuple (final_ids, trajectory)
            final_ids: (B, P+gen_length)
            trajectory: list of (step, mask_ratio, snapshot_ids)
    """
    device = prompt_ids.device
    B, P = prompt_ids.shape
    total_len = P + gen_length

    # Initialize with mask tokens in generation region
    input_ids = torch.full((B, total_len), mask_id, dtype=prompt_ids.dtype, device=device)
    input_ids[:, :P] = prompt_ids

    # attention_mask: all 1s (no padding)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    trajectory = []
    # Initial state snapshot
    if hook_fn is not None:
        init_mask_ratio = gen_length / float(total_len)
        hook_fn(0, init_mask_ratio, input_ids.clone(), None)
    trajectory.append((0, gen_length / float(total_len), input_ids.clone()))

    # Number of tokens to unmask per step (uniformly)
    # Standard LLaDA: tokens_per_step = gen_length / steps
    tokens_per_step_float = gen_length / steps

    for step in range(1, steps + 1):
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [B, L, V]

        # Apply temperature + sampling
        if temperature != 1.0:
            logits = logits / temperature

        # Gen region only
        gen_logits = logits[:, P:, :]  # [B, gen_length, V]
        gen_ids_current = input_ids[:, P:]  # [B, gen_length]

        # Predict tokens at masked positions
        if top_p < 1.0:
            predicted = _sample_top_p(gen_logits, top_p)
        elif temperature > 0:
            probs = F.softmax(gen_logits, dim=-1)
            predicted = torch.multinomial(
                probs.reshape(-1, probs.size(-1)), num_samples=1
            ).reshape(B, -1)
        else:
            predicted = gen_logits.argmax(dim=-1)

        # Compute confidence (max softmax prob) for each position
        probs = F.softmax(gen_logits, dim=-1)
        confidence = probs.max(dim=-1).values  # [B, gen_length]

        # How many tokens should be unmasked AFTER this step?
        # At step k (1-indexed), target unmask count = k * tokens_per_step
        target_unmask = min(int(step * tokens_per_step_float), gen_length)

        # Current mask mask: which positions are still masked?
        is_mask = (gen_ids_current == mask_id)  # [B, gen_length]

        # Among currently masked positions, select top (target_unmask - already_unmasked) by confidence
        already_unmasked = (~is_mask).sum(dim=1)  # [B]
        tokens_to_unmask = target_unmask - already_unmasked

        # New gen_ids: put predicted where will be unmasked, keep mask elsewhere
        new_gen_ids = gen_ids_current.clone()

        for b in range(B):
            n_to_unmask = int(tokens_to_unmask[b].item())
            if n_to_unmask <= 0:
                continue
            # Get indices that are masked
            mask_positions = is_mask[b].nonzero(as_tuple=True)[0]
            if mask_positions.numel() == 0:
                continue
            # Confidence at those positions
            mask_confs = confidence[b, mask_positions]
            # Top-n_to_unmask by confidence
            n_to_unmask = min(n_to_unmask, mask_positions.numel())
            _, top_idx = torch.topk(mask_confs, n_to_unmask)
            unmask_positions = mask_positions[top_idx]
            new_gen_ids[b, unmask_positions] = predicted[b, unmask_positions]

        # Update input_ids
        input_ids[:, P:] = new_gen_ids

        # Mask ratio after this step
        mask_ratio = (input_ids[:, P:] == mask_id).float().mean().item()

        # Hook callback (for PRM branching)
        if hook_fn is not None:
            override = hook_fn(step, mask_ratio, input_ids.clone(), gen_logits)
            if override is not None:
                input_ids = override

        # Snapshot trajectory every ~6 steps (matches Dream's trajectory spec)
        if step % 6 == 0 or step == steps:
            trajectory.append((step, mask_ratio, input_ids.clone()))

    return input_ids, trajectory


def _sample_top_p(logits, top_p):
    """Top-p (nucleus) sampling at each position.

    Args:
        logits: [B, L, V]
    Returns:
        [B, L] sampled token ids
    """
    B, L, V = logits.shape
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    mask = cumsum > top_p
    # Shift mask right by 1 to include the first element that crosses top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_probs.masked_fill_(mask, 0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    # Sample from filtered sorted dist
    sample_idx = torch.multinomial(
        sorted_probs.reshape(-1, V), num_samples=1
    ).reshape(B, L)
    return sorted_idx.gather(dim=-1, index=sample_idx.unsqueeze(-1)).squeeze(-1)
