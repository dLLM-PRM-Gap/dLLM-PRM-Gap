"""Deterministic, per-row Dream denoising with exact snapshot resume.

Snapshots recorded by :mod:`generate_trajectories` are post-step states.  A
snapshot tagged ``snapshot_step=i`` therefore resumes at ``i + 1``.  The
historical recorder did not persist RNG state, so exact parity is available
for states captured through this module; independent continuation labeling
can instead attach fresh semantic row seeds to an existing token snapshot.

Every row owns an explicit ``torch.Generator``.  Random draws consequently do
not depend on batch order, shard count, or the presence of other rows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


MASK_TOKEN_ID = 151666


@dataclass(frozen=True)
class DenoiseConfig:
    steps: int = 128
    temperature: float = 0.5
    top_p: float = 1.0
    top_k: int | None = None
    alg: str = "maskgit_plus"
    alg_temp: float | None = 0.5
    mask_token_id: int = MASK_TOKEN_ID
    gen_length: int = 512
    eps: float = 1e-5

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.gen_length <= 0:
            raise ValueError("gen_length must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when set")
        if self.alg not in {"origin", "maskgit_plus", "margin", "entropy"}:
            raise ValueError(f"unsupported denoising algorithm: {self.alg}")


@dataclass
class DenoiseResumeState:
    """Complete mutable state needed to continue a denoising batch."""

    x: torch.LongTensor
    next_step: int
    attention_mask: torch.Tensor | str
    position_ids: torch.Tensor | None
    row_rng_states: tuple[torch.ByteTensor, ...]
    row_seeds: tuple[int, ...]
    logical_denoising_calls: int = 0
    physical_forward_calls: int = 0

    def validate(self, config: DenoiseConfig) -> None:
        if self.x.ndim != 2:
            raise ValueError(f"x must have shape [B, L], got {tuple(self.x.shape)}")
        batch_size = self.x.shape[0]
        if len(self.row_rng_states) != batch_size:
            raise ValueError("one RNG state is required per row")
        if len(self.row_seeds) != batch_size:
            raise ValueError("one seed is required per row")
        if not (0 <= self.next_step <= config.steps):
            raise ValueError(
                f"next_step={self.next_step} outside [0, {config.steps}]"
            )
        if torch.is_tensor(self.attention_mask):
            if self.attention_mask.shape[0] not in {1, batch_size}:
                raise ValueError("attention-mask batch dimension does not match x")
            if self.attention_mask.shape[-1] != self.x.shape[-1]:
                raise ValueError("attention-mask sequence dimension does not match x")
        if self.position_ids is not None:
            if self.position_ids.shape[-1] != self.x.shape[-1]:
                raise ValueError("position_ids sequence dimension does not match x")

    def clone(self) -> "DenoiseResumeState":
        return DenoiseResumeState(
            x=self.x.clone(),
            next_step=self.next_step,
            attention_mask=(
                self.attention_mask.clone()
                if torch.is_tensor(self.attention_mask)
                else self.attention_mask
            ),
            position_ids=(
                self.position_ids.clone() if self.position_ids is not None else None
            ),
            row_rng_states=tuple(state.clone().cpu() for state in self.row_rng_states),
            row_seeds=tuple(self.row_seeds),
            logical_denoising_calls=self.logical_denoising_calls,
            physical_forward_calls=self.physical_forward_calls,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a CPU payload suitable for ``torch.save``."""
        return {
            "schema_version": 1,
            "x": self.x.detach().cpu(),
            "next_step": self.next_step,
            "attention_mask": (
                self.attention_mask.detach().cpu()
                if torch.is_tensor(self.attention_mask)
                else self.attention_mask
            ),
            "position_ids": (
                self.position_ids.detach().cpu()
                if self.position_ids is not None
                else None
            ),
            "row_rng_states": tuple(
                state.detach().cpu() for state in self.row_rng_states
            ),
            "row_seeds": tuple(self.row_seeds),
            "logical_denoising_calls": self.logical_denoising_calls,
            "physical_forward_calls": self.physical_forward_calls,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], device: str | torch.device
    ) -> "DenoiseResumeState":
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported denoise-state schema")
        target = torch.device(device)
        attention_mask = payload["attention_mask"]
        position_ids = payload.get("position_ids")
        return cls(
            x=payload["x"].to(target),
            next_step=int(payload["next_step"]),
            attention_mask=(
                attention_mask.to(target)
                if torch.is_tensor(attention_mask)
                else attention_mask
            ),
            position_ids=(
                position_ids.to(target) if position_ids is not None else None
            ),
            row_rng_states=tuple(
                state.detach().cpu().clone()
                for state in payload["row_rng_states"]
            ),
            row_seeds=tuple(int(seed) for seed in payload["row_seeds"]),
            logical_denoising_calls=int(payload.get("logical_denoising_calls", 0)),
            physical_forward_calls=int(payload.get("physical_forward_calls", 0)),
        )


def _new_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _generators_from_state(state: DenoiseResumeState) -> list[torch.Generator]:
    generators = []
    for rng_state in state.row_rng_states:
        generator = torch.Generator(device=state.x.device)
        generator.set_state(rng_state.cpu())
        generators.append(generator)
    return generators


def _prepare_attention(
    input_ids: torch.LongTensor,
    max_length: int,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | str, torch.Tensor | None]:
    if attention_mask is None or not torch.any(attention_mask == 0):
        return "full", None
    padded = F.pad(
        attention_mask.to(device=input_ids.device),
        (0, max_length - attention_mask.shape[-1]),
        value=1,
    ).bool()
    position_ids = padded.long().cumsum(-1) - 1
    position_ids.masked_fill_(~padded, 1)
    full_mask = torch.logical_and(
        padded.unsqueeze(1).unsqueeze(-2),
        padded.unsqueeze(1).unsqueeze(-1),
    )
    return full_mask, position_ids


def initialize_denoise_state(
    input_ids: torch.LongTensor,
    config: DenoiseConfig,
    row_seeds: Sequence[int],
    attention_mask: torch.Tensor | None = None,
) -> DenoiseResumeState:
    """Pad prompts with MASK and create one independent RNG stream per row."""
    config.validate()
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, prompt_length]")
    if input_ids.shape[0] != len(row_seeds):
        raise ValueError("row_seeds length must equal input batch size")
    max_length = input_ids.shape[-1] + config.gen_length
    x = F.pad(
        input_ids,
        (0, config.gen_length),
        value=config.mask_token_id,
    )
    dream_mask, position_ids = _prepare_attention(
        input_ids, max_length, attention_mask
    )
    generators = [_new_generator(input_ids.device, seed) for seed in row_seeds]
    state = DenoiseResumeState(
        x=x,
        next_step=0,
        attention_mask=dream_mask,
        position_ids=position_ids,
        row_rng_states=tuple(generator.get_state().cpu() for generator in generators),
        row_seeds=tuple(int(seed) for seed in row_seeds),
    )
    state.validate(config)
    return state


def state_from_post_step_snapshot(
    full_state: torch.LongTensor,
    snapshot_step: int,
    config: DenoiseConfig,
    *,
    row_seeds: Sequence[int] | None = None,
    row_rng_states: Sequence[torch.ByteTensor] | None = None,
    attention_mask: torch.Tensor | str = "full",
    position_ids: torch.Tensor | None = None,
    logical_denoising_calls: int = 0,
    physical_forward_calls: int = 0,
) -> DenoiseResumeState:
    """Build continuation state from a post-step token snapshot.

    Exactly one of ``row_seeds`` (fresh independent continuations) or
    ``row_rng_states`` (exact reproduction) must be supplied.
    """
    config.validate()
    if full_state.ndim != 2:
        raise ValueError("full_state must have shape [B, full_length]")
    if (row_seeds is None) == (row_rng_states is None):
        raise ValueError("provide exactly one of row_seeds or row_rng_states")
    next_step = int(snapshot_step) + 1
    if not (0 <= next_step <= config.steps):
        raise ValueError(
            f"post-step snapshot {snapshot_step} cannot resume a {config.steps}-step run"
        )
    batch_size = full_state.shape[0]
    if row_seeds is not None:
        if len(row_seeds) != batch_size:
            raise ValueError("row_seeds length must equal snapshot batch size")
        generators = [
            _new_generator(full_state.device, int(seed)) for seed in row_seeds
        ]
        states = tuple(generator.get_state().cpu() for generator in generators)
        seeds = tuple(int(seed) for seed in row_seeds)
    else:
        assert row_rng_states is not None
        if len(row_rng_states) != batch_size:
            raise ValueError("row_rng_states length must equal snapshot batch size")
        states = tuple(state.detach().cpu().clone() for state in row_rng_states)
        # The original initial seeds cannot be recovered from generator states.
        seeds = tuple(-1 for _ in range(batch_size))
    state = DenoiseResumeState(
        x=full_state.clone(),
        next_step=next_step,
        attention_mask=(
            attention_mask.clone()
            if torch.is_tensor(attention_mask)
            else attention_mask
        ),
        position_ids=position_ids.clone() if position_ids is not None else None,
        row_rng_states=states,
        row_seeds=seeds,
        logical_denoising_calls=int(logical_denoising_calls),
        physical_forward_calls=int(physical_forward_calls),
    )
    state.validate(config)
    return state


def replace_row_rng_seeds(
    state: DenoiseResumeState, row_seeds: Sequence[int]
) -> DenoiseResumeState:
    """Clone a state and attach fresh independent per-row streams."""
    if len(row_seeds) != state.x.shape[0]:
        raise ValueError("row_seeds length must equal state batch size")
    generators = [
        _new_generator(state.x.device, int(seed)) for seed in row_seeds
    ]
    cloned = state.clone()
    cloned.row_seeds = tuple(int(seed) for seed in row_seeds)
    cloned.row_rng_states = tuple(
        generator.get_state().cpu() for generator in generators
    )
    return cloned


def _top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
        F.softmax(sorted_logits, dim=-1), dim=-1
    )
    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def _top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    top_k = min(top_k, logits.shape[-1])
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)


def _sample_tokens_explicit(
    logits: torch.Tensor,
    *,
    generator: torch.Generator,
    temperature: float,
    top_p: float,
    top_k: int | None,
    margin_confidence: bool = False,
    negative_entropy: bool = False,
) -> tuple[torch.Tensor, torch.LongTensor]:
    if logits.numel() == 0:
        return (
            torch.empty(logits.shape[:-1], device=logits.device),
            torch.empty(logits.shape[:-1], device=logits.device, dtype=torch.long),
        )
    filtered = logits
    if temperature > 0:
        filtered = filtered / temperature
    if top_p < 1:
        filtered = _top_p_logits(filtered, top_p)
    if top_k is not None:
        filtered = _top_k_logits(filtered, top_k)
    probabilities = F.softmax(filtered, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("non-finite token probabilities")
    if temperature > 0:
        sampled = torch.multinomial(
            probabilities, num_samples=1, replacement=True, generator=generator
        ).squeeze(-1)
        confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, sampled = probabilities.max(dim=-1)
    if margin_confidence:
        top_two = torch.topk(probabilities, k=2, dim=-1).values
        confidence = top_two[..., 0] - top_two[..., 1]
    if negative_entropy:
        confidence = torch.sum(
            probabilities * torch.log(probabilities + 1e-10), dim=-1
        )
    return confidence, sampled


def _denoise_row(
    x: torch.LongTensor,
    logits: torch.Tensor,
    mask_index: torch.BoolTensor,
    *,
    t: torch.Tensor,
    s: torch.Tensor,
    is_last: bool,
    generator: torch.Generator,
    config: DenoiseConfig,
) -> None:
    if not mask_index.any():
        return
    masked_logits = logits[mask_index]
    if config.alg == "origin":
        probability = 1.0 if is_last else float((1 - s / t).item())
        transfer = torch.rand(
            (int(mask_index.sum().item()),),
            device=x.device,
            generator=generator,
        ) < probability
        values = torch.full(
            (transfer.shape[0],),
            config.mask_token_id,
            device=x.device,
            dtype=torch.long,
        )
        if transfer.any():
            _, sampled = _sample_tokens_explicit(
                masked_logits[transfer],
                generator=generator,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )
            values[transfer] = sampled
        x[mask_index] = values
        return

    margin = config.alg == "margin"
    negative_entropy = config.alg == "entropy"
    confidence, sampled = _sample_tokens_explicit(
        masked_logits,
        generator=generator,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        margin_confidence=margin,
        negative_entropy=negative_entropy,
    )
    num_masked = int(mask_index.sum().item())
    num_transfer = (
        num_masked
        if is_last
        else int(num_masked * float((1 - s / t).item()))
    )
    if num_transfer <= 0:
        return
    if config.alg_temp is None or config.alg_temp == 0:
        transfer_indices = torch.topk(confidence, num_transfer).indices
    else:
        transfer_probabilities = F.softmax(
            confidence / config.alg_temp, dim=-1
        )
        transfer_indices = torch.multinomial(
            transfer_probabilities,
            num_samples=num_transfer,
            replacement=False,
            generator=generator,
        )
    values = torch.full_like(sampled, config.mask_token_id, dtype=torch.long)
    values[transfer_indices] = sampled[transfer_indices]
    x[mask_index] = values


@torch.no_grad()
def run_denoise(
    model,
    state: DenoiseResumeState,
    config: DenoiseConfig,
    *,
    stop_step: int | None = None,
    capture_after_steps: Iterable[int] = (),
) -> tuple[DenoiseResumeState, dict[int, DenoiseResumeState]]:
    """Continue a batch through ``stop_step`` (exclusive).

    ``capture_after_steps`` uses zero-based, post-step indices, matching
    ``TrajectoryRecorder``.  Captures include token, attention, RNG, and call
    state and can therefore reproduce the uninterrupted continuation exactly.
    Logical denoising calls count one model-scale candidate forward per row,
    while physical calls count batched invocations.
    """
    config.validate()
    state.validate(config)
    end = config.steps if stop_step is None else int(stop_step)
    if not (state.next_step <= end <= config.steps):
        raise ValueError(
            f"stop_step={end} must lie in [{state.next_step}, {config.steps}]"
        )
    capture_set = {int(step) for step in capture_after_steps}
    if any(step < state.next_step or step >= end for step in capture_set):
        raise ValueError("capture step is outside the executed interval")

    working = state.clone()
    generators = _generators_from_state(working)
    timesteps = torch.linspace(
        1, config.eps, config.steps + 1, device=working.x.device
    )
    captures: dict[int, DenoiseResumeState] = {}
    batch_size = working.x.shape[0]

    for absolute_step in range(working.next_step, end):
        mask_index = working.x == config.mask_token_id
        outputs = model(
            working.x, working.attention_mask, working.position_ids
        )
        logits = outputs.logits
        shifted_logits = torch.cat(
            [logits[:, :1], logits[:, :-1]], dim=1
        )
        t = timesteps[absolute_step]
        s = timesteps[absolute_step + 1]
        for row in range(batch_size):
            _denoise_row(
                working.x[row],
                shifted_logits[row],
                mask_index[row],
                t=t,
                s=s,
                is_last=absolute_step == config.steps - 1,
                generator=generators[row],
                config=config,
            )
        working.next_step = absolute_step + 1
        working.logical_denoising_calls += batch_size
        working.physical_forward_calls += 1
        working.row_rng_states = tuple(
            generator.get_state().cpu() for generator in generators
        )
        if absolute_step in capture_set:
            captures[absolute_step] = working.clone()

    return working, captures
