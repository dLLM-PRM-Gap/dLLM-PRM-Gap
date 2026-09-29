"""
Reward models for dLLM PRM Gap: ORM (outcome) and DiffusionPRM (step-level).

Both use Dream-7B as frozen backbone with LoRA adapters.
Key difference: ORM evaluates complete solutions; DiffusionPRM evaluates
partially-masked intermediate states using bidirectional attention.
"""

import math
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

MASK_TOKEN_ID = 151666
EOS_TOKEN_ID = 151643


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for continuous values (mask ratio / step index).

    Uses a scale factor to expand the [0,1] mask_ratio range into a wider
    range that better utilizes the sinusoidal frequency bands.
    """

    def __init__(self, dim, scale=128.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        """x: [B] float values → [B, dim] embeddings."""
        x = x.float() * self.scale
        half = self.dim // 2
        denom = max(half - 1, 1)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=x.device, dtype=torch.float32)
            / denom
        )
        args = x.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        # Handle odd dim
        if emb.shape[-1] < self.dim:
            emb = nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


def create_lora_backbone(model, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """Wrap Dream-7B with LoRA on q_proj and v_proj."""
    for param in model.parameters():
        param.requires_grad = False

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )
    return get_peft_model(model, config)


def prepare_dream_attention(attention_mask):
    """Convert [B, N] bool mask to Dream's 4D attention mask + position IDs.

    Dream-7B uses bidirectional attention, so padding must be properly masked
    to prevent contamination of real token representations.
    """
    if attention_mask is None:
        return "full", None

    tok_idx = attention_mask.long().cumsum(-1) - 1
    tok_idx.masked_fill_(~attention_mask, 1)

    # [B, N] → [B, 1, N, N] for bidirectional attention
    full_mask = torch.logical_and(
        attention_mask.unsqueeze(1).unsqueeze(-2),
        attention_mask.unsqueeze(1).unsqueeze(-1),
    )
    return full_mask, tok_idx


def pool_solution_region(
    hidden_states, input_ids, prompt_lens, mask_aware=True, strategy="mean"
):
    """Pool hidden states over the solution region (before first EOS).

    Args:
        hidden_states: [B, seq_len, H]
        input_ids: [B, seq_len]
        prompt_lens: [B] int
        mask_aware: exclude MASK tokens from pooling (for PRM)
        strategy: "mean" (default, mean-pool over solution region) or
                  "last_token" (take hidden state at last non-MASK non-EOS
                  position of solution). Motivated by Ouyang 2022 / Stiennon 2020
                  standard for causal reward models — the final token of a
                  causal sequence has observed the full context, whereas
                  mean-pooling dilutes by averaging "blind" early tokens with
                  later ones.

    Returns:
        pooled: [B, H]
    """
    B, seq_len, H = hidden_states.shape
    device = hidden_states.device

    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
    in_solution = positions >= prompt_lens.unsqueeze(1)

    # Stop at first EOS: exclude EOS itself and everything after
    is_eos = (input_ids == EOS_TOKEN_ID) & in_solution
    past_first_eos = is_eos.cumsum(dim=1) > 0
    pool_mask = in_solution & ~past_first_eos

    if mask_aware:
        pool_mask = pool_mask & (input_ids != MASK_TOKEN_ID)

    if strategy == "mean":
        pool_mask_f = pool_mask.float().unsqueeze(-1)  # [B, seq_len, 1]
        pooled = (hidden_states * pool_mask_f).sum(dim=1)
        counts = pool_mask_f.sum(dim=1).clamp(min=1)
        return pooled / counts

    if strategy == "last_token":
        # For each batch row, find the LAST position where pool_mask is True.
        # If no valid position (e.g., all masked), fall back to position
        # seq_len-1 within solution; that's a corner case and shouldn't
        # occur in practice.
        mask_int = pool_mask.int()
        # last_idx[b] = max position i where mask_int[b, i] = 1; else 0
        # Trick: argmax of mask_int with later positions weighted higher via
        # multiplication by arange gives the last True position.
        weighted = mask_int * torch.arange(seq_len, device=device).unsqueeze(0)
        last_idx = weighted.argmax(dim=1)  # [B]
        # Gather: out[b] = hidden_states[b, last_idx[b], :]
        batch_idx = torch.arange(B, device=device)
        pooled = hidden_states[batch_idx, last_idx]  # [B, H]
        return pooled.float()

    raise ValueError(f"Unknown pool strategy: {strategy!r} (expected 'mean' or 'last_token')")


class OutcomeRewardModel(nn.Module):
    """ORM: evaluates complete solutions (mask_ratio=0).

    Dream-7B + LoRA → solution mean pooling → Linear → scalar reward.
    """

    def __init__(
        self, backbone, hidden_size=3584,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
    ):
        super().__init__()
        self.backbone = create_lora_backbone(
            backbone, lora_r, lora_alpha, lora_dropout
        )
        self.reward_head = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, input_ids, prompt_lens, attention_mask=None):
        dream_mask, tok_idx = prepare_dream_attention(attention_mask)
        outputs = self.backbone(
            input_ids,
            attention_mask=dream_mask,
            position_ids=tok_idx,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        pooled = pool_solution_region(
            hidden, input_ids, prompt_lens, mask_aware=False
        )
        return self.reward_head(pooled).squeeze(-1)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiffusionPRM(nn.Module):
    """Diffusion-native PRM: evaluates partially-masked dLLM states.

    Uses bidirectional attention on partial states (impossible for AR models).

    Dream-7B + LoRA → mask-aware pooling + step embedding → MLP → scalar.

    Args:
        causal: if True, use causal (left-to-right) attention instead of
                bidirectional. Used for RA7 ablation (C3 claim).
    """

    def __init__(
        self, backbone, hidden_size=3584,
        step_embed_dim=256, reward_hidden=1024,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        causal=False, no_step_embed=False, no_mask_aware=False,
        pool_strategy="mean",
    ):
        super().__init__()
        self.backbone = create_lora_backbone(
            backbone, lora_r, lora_alpha, lora_dropout
        )
        self.hidden_size = hidden_size
        self.step_embed_dim = step_embed_dim
        self.causal = causal
        self.no_step_embed = no_step_embed
        self.no_mask_aware = no_mask_aware
        self.pool_strategy = pool_strategy  # "mean" | "last_token"

        if not no_step_embed:
            self.step_embed = SinusoidalEmbedding(step_embed_dim, scale=128.0)
            self.step_proj = nn.Sequential(
                nn.Linear(step_embed_dim, step_embed_dim),
                nn.GELU(),
            )
            input_dim = hidden_size + step_embed_dim
        else:
            input_dim = hidden_size

        self.reward_head = nn.Sequential(
            nn.Linear(input_dim, reward_hidden),
            nn.GELU(),
            nn.Linear(reward_hidden, 1),
        )

    def forward(self, input_ids, prompt_lens, mask_ratios, attention_mask=None):
        if self.causal:
            # Construct 4D causal (lower-triangular) boolean mask.
            # Dream SDPA only handles tensor masks (strings like "full" skip masking),
            # so we must provide an explicit tensor for causal attention.
            B, N = input_ids.shape
            device = input_ids.device
            causal = torch.tril(
                torch.ones(N, N, device=device, dtype=torch.bool)
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            if attention_mask is not None:
                # Combine causal with padding mask
                pad = torch.logical_and(
                    attention_mask.unsqueeze(1).unsqueeze(-2),
                    attention_mask.unsqueeze(1).unsqueeze(-1),
                )  # [B, 1, N, N]
                dream_mask = causal & pad
                tok_idx = attention_mask.long().cumsum(-1) - 1
                tok_idx.masked_fill_(~attention_mask, 1)
            else:
                dream_mask = causal.expand(B, 1, N, N)
                tok_idx = None
        else:
            dream_mask, tok_idx = prepare_dream_attention(attention_mask)
        outputs = self.backbone(
            input_ids,
            attention_mask=dream_mask,
            position_ids=tok_idx,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]

        pooled = pool_solution_region(
            hidden, input_ids, prompt_lens,
            mask_aware=not self.no_mask_aware,
            strategy=self.pool_strategy,
        )

        if not self.no_step_embed:
            step_emb = self.step_proj(self.step_embed(mask_ratios))
            combined = torch.cat([pooled, step_emb], dim=-1)
        else:
            combined = pooled
        return self.reward_head(combined).squeeze(-1)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
