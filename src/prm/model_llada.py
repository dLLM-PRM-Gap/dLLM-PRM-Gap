"""
LLaDA-8B PRM adapter. Mirrors model.py but for LLaDA-8B backbone.

Key differences vs Dream-7B:
- hidden_size: 4096 (vs 3584)
- MASK_TOKEN_ID: 126336 (vs 151666)
- EOS_TOKEN_ID: 126081 (vs 151643)
- Attention: LLaDA's forward takes attention_mask (1D bool/0-1) directly;
  LLaDA's modeling internally converts to 4D bias. So we pass attention_mask
  as a normal (batch_size, seq_len) tensor without Dream's prepare_dream_attention.

For causal ablation: LLaDA's forward also accepts attention_bias (4D).
We construct a causal attention_bias tensor and pass it.
"""

import math
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

# LLaDA-8B specific token IDs (from config.json)
LLADA_MASK_TOKEN_ID = 126336
LLADA_EOS_TOKEN_ID = 126081
LLADA_HIDDEN_SIZE = 4096


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim, scale=128.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x):
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
        if emb.shape[-1] < self.dim:
            emb = nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


def create_lora_backbone(model, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """Wrap LLaDA-8B with LoRA on q_proj and v_proj."""
    for param in model.parameters():
        param.requires_grad = False

    # LLaDA uses llama block. Check attention layer names:
    # modeling_llada.py uses q_proj/k_proj/v_proj in LlamaAttention-style naming.
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )
    return get_peft_model(model, config)


def pool_solution_region_llada(
    hidden_states, input_ids, prompt_lens, mask_aware=True
):
    """Pool hidden states over solution region. Uses LLaDA's mask/EOS token IDs."""
    B, seq_len, H = hidden_states.shape
    device = hidden_states.device

    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
    in_solution = positions >= prompt_lens.unsqueeze(1)

    is_eos = (input_ids == LLADA_EOS_TOKEN_ID) & in_solution
    past_first_eos = is_eos.cumsum(dim=1) > 0
    pool_mask = in_solution & ~past_first_eos

    if mask_aware:
        pool_mask = pool_mask & (input_ids != LLADA_MASK_TOKEN_ID)

    pool_mask_f = pool_mask.float().unsqueeze(-1)
    pooled = (hidden_states * pool_mask_f).sum(dim=1)
    counts = pool_mask_f.sum(dim=1).clamp(min=1)
    return pooled / counts


def llada_causal_bias(seq_len, device, dtype=torch.float32):
    """Build 4D causal attention bias (upper-triangular = -inf)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    bias = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
    bias.masked_fill_(~mask, torch.finfo(dtype).min)
    return bias


class LLaDAOutcomeRewardModel(nn.Module):
    """ORM for LLaDA-8B backbone.

    Evaluates complete solutions. LLaDA forward returns LLaDAOutput which has
    hidden_states (when output_hidden_states=True).
    """

    def __init__(
        self, backbone, hidden_size=LLADA_HIDDEN_SIZE,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
    ):
        super().__init__()
        self.backbone = create_lora_backbone(
            backbone, lora_r, lora_alpha, lora_dropout
        )
        self.reward_head = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, input_ids, prompt_lens, attention_mask=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # LLaDAOutput.hidden_states is tuple of (n_layers+1) tensors.
        hidden = outputs.hidden_states[-1]
        pooled = pool_solution_region_llada(
            hidden, input_ids, prompt_lens, mask_aware=False
        )
        return self.reward_head(pooled).squeeze(-1)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LLaDADiffusionPRM(nn.Module):
    """dLLM PRM Gap scorer for the LLaDA-8B backbone with bidirectional attention.

    Causal ablation: pass attention_bias (4D lower-triangular) to force causal.
    """

    def __init__(
        self, backbone, hidden_size=LLADA_HIDDEN_SIZE,
        step_embed_dim=256, reward_hidden=1024,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        causal=False, no_step_embed=False, no_mask_aware=False,
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
        B, N = input_ids.shape
        device = input_ids.device

        if self.causal:
            # Pass 4D causal bias to LLaDA forward
            attention_bias = llada_causal_bias(N, device, dtype=torch.float32)
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                output_hidden_states=True,
            )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden = outputs.hidden_states[-1]
        pooled = pool_solution_region_llada(
            hidden, input_ids, prompt_lens,
            mask_aware=not self.no_mask_aware,
        )

        if not self.no_step_embed:
            step_emb = self.step_proj(self.step_embed(mask_ratios))
            combined = torch.cat([pooled, step_emb], dim=-1)
        else:
            combined = pooled
        return self.reward_head(combined).squeeze(-1)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
