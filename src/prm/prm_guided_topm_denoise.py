"""Top-M retention variant of PRM-Guided.

Differences vs prm_guided_denoise.prm_guided_sample:
- At each segment, instead of retaining top-1 after PRM scoring, retain top-M.
- K total branches are distributed across M retained seeds (K/M each).
- At final segment return top-1 of last surviving M.

Uses existing _batch_denoise_step for per-candidate denoising.
"""
import torch
import torch.nn.functional as F

from prm.prm_guided_denoise import _batch_denoise_step, MASK_TOKEN_ID


@torch.no_grad()
def prm_guided_topm_sample(
    model,
    prm,
    input_ids: torch.LongTensor,
    steps: int = 128,
    branch_every: int = 64,
    branch_factor: int = 8,    # K total children per segment
    top_m: int = 2,             # M retained after each prune (M=1 = original)
    temperature: float = 0.5,
    top_p: float = 1.0,
    alg: str = 'maskgit_plus',
    alg_temp: float = 0.5,
    mask_token_id: int = MASK_TOKEN_ID,
    gen_length: int = 512,
    eps: float = 1e-5,
    verbose: bool = False,
):
    """Top-M retention PRM-Guided denoising.

    At each segment:
      - Each of M retained seeds spawns (K/M) children -> K total candidates
      - Denoise each for branch_every steps in parallel
      - PRM scores all K
      - Retain top-M by score (not top-1)
    Return top-1 of final M candidates.

    For K=8, M=2: 2 seeds * 4 children = 8 per segment, keep top 2.
    For K=8, M=1: reduces to original PRM-Guided (Algorithm 1).
    For K=8, M=8: all 8 kept (no pruning); equivalent to 8 independent chains
                  with a PRM final selector.

    Returns: (x_best [1, max_len], info dict).
    """
    from dream.generation_utils import sample_tokens
    assert branch_factor % top_m == 0, f"K={branch_factor} must be divisible by M={top_m}"
    children_per_seed = branch_factor // top_m

    device = input_ids.device
    prompt_len = input_ids.shape[1]
    max_length = prompt_len + gen_length

    # Initialize M seeds as identical copies of prompt-padded-with-MASK
    x_init = F.pad(input_ids, (0, max_length - prompt_len), value=mask_token_id)
    seeds = x_init.expand(top_m, -1).clone()          # [M, max_len]
    seed_prev_scores = torch.zeros(top_m, device=device)  # track for final return

    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    num_segments = steps // branch_every
    branch_info = []

    for seg_idx in range(num_segments):
        seg_start = seg_idx * branch_every
        seg_end = min((seg_idx + 1) * branch_every, steps)

        # Replicate each of M seeds into K/M children => K total
        # [M, max_len] -> [M, K/M, max_len] -> [K, max_len]
        candidates = (
            seeds.unsqueeze(1)
            .expand(top_m, children_per_seed, -1)
            .reshape(branch_factor, -1)
            .clone()
        )

        # Evolve all K candidates for branch_every steps
        for step_in_seg in range(seg_end - seg_start):
            abs_step = seg_start + step_in_seg
            t = timesteps[abs_step]
            s = timesteps[abs_step + 1]
            is_last = (abs_step == steps - 1)

            mask_index = (candidates == mask_token_id)
            if not mask_index.any():
                break

            logits = model(candidates, "full", None).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            _batch_denoise_step(
                candidates, logits, mask_index, t, s,
                temperature, top_p, alg, alg_temp,
                mask_token_id, is_last, device, sample_tokens,
            )

        # PRM score all K
        mask_ratios_k = (candidates[:, prompt_len:] == mask_token_id).float().mean(dim=1)
        prompt_lens_k = torch.full((branch_factor,), prompt_len, dtype=torch.long, device=device)
        scores = prm(candidates, prompt_lens_k, mask_ratios_k)

        # Retain top-M
        topm_idx = torch.topk(scores, top_m).indices
        seeds = candidates[topm_idx].clone()   # [M, max_len]
        seed_prev_scores = scores[topm_idx].clone()

        branch_info.append({
            'segment': seg_idx,
            'step_range': (seg_start, seg_end),
            'mask_ratio_mean': mask_ratios_k.mean().item(),
            'scores': scores.cpu().tolist(),
            'topm_idx': topm_idx.cpu().tolist(),
            'topm_scores': seed_prev_scores.cpu().tolist(),
        })
        if verbose:
            print(f"  Seg {seg_idx} M={top_m} K={branch_factor}: scores={scores.cpu().tolist()}, kept={topm_idx.cpu().tolist()}")

    # Return top-1 of final M
    best_of_m = seed_prev_scores.argmax().item()
    x = seeds[best_of_m].unsqueeze(0)

    info = {
        'num_segments': num_segments,
        'branch_factor': branch_factor,
        'top_m': top_m,
        'branch_info': branch_info,
        'total_forward_passes': num_segments * branch_factor * branch_every + num_segments * branch_factor,
    }
    return x, info
