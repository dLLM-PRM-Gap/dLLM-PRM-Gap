"""
Phase 3: Train Diffusion-native PRM on trajectory data.

DiffusionPRM = Dream-7B frozen + LoRA + mask-aware pooling + step embedding + MLP.
Trained with BCE loss on partially-masked intermediate states.

Key difference from ORM: sees ALL intermediate denoising states, not just final output.
This gives ~24x more training signal per trajectory.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_diffusion_prm.py \
        --trajectory_dir data/prm_trajectories/gsm8k \
        --output_dir checkpoints/prm_gsm8k \
        --epochs 2 --lr 1e-5
"""

import os
import sys
import json
import argparse
import time
import random
import logging
import hashlib
from functools import partial
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external", "CoDD", "eval"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external", "CoDD"))

from prm.model import DiffusionPRM  # noqa: E402
from prm.checkpointing import load_state_dict_audited  # noqa: E402
from prm.dataset import (  # noqa: E402
    DiffusionPRMDataset,
    DiffusionPRMStateDataset,
    collate_fn,
)


def atomic_save(obj, path):
    """Save checkpoint atomically: write to temp file, then replace.
    Prevents race conditions on networked filesystems where concurrent readers may see partial files."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def load_state_records(path):
    """Load a frozen state split from a PT file or JSONL."""
    if path.endswith((".pt", ".pth")):
        records = torch.load(path, map_location="cpu", weights_only=False)
    elif path.endswith(".jsonl"):
        records = []
        with open(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    else:
        raise ValueError(f"Unsupported state file: {path}")
    if not isinstance(records, list):
        raise TypeError(f"Expected a list of records in {path}")
    return records


def trainable_state_dict(model):
    """Return only parameters updated by the optimizer."""
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


RESUME_CRITICAL_CONFIG = (
    "trajectory_dir",
    "train_file",
    "val_file",
    "label_field",
    "model_path",
    "epochs",
    "batch_size",
    "grad_accum",
    "lr",
    "weight_decay",
    "warmup_fraction",
    "max_grad_norm",
    "max_steps",
    "max_val_samples",
    "save_every_steps",
    "eval_every_steps",
    "log_every_steps",
    "seed",
    "model_init_seed",
    "expected_init_sha256",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "step_embed_dim",
    "reward_hidden",
    "causal",
    "no_step_embed",
    "no_mask_aware",
    "pool_strategy",
    "formal_provenance",
)


def validate_resume_config(saved, current):
    mismatches = {
        key: {"saved": saved.get(key), "current": current.get(key)}
        for key in RESUME_CRITICAL_CONFIG
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "Resume configuration differs from the checkpoint: "
            + json.dumps(mismatches, sort_keys=True)
        )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_trajectories(trajectory_dir):
    merged = os.path.join(trajectory_dir, "trajectories_all.pt")
    if os.path.exists(merged):
        data = torch.load(merged, weights_only=False)
        logger.info(f"Loaded {len(data)} trajectories from {merged}")
        return data
    import glob

    shard_files = sorted(glob.glob(os.path.join(trajectory_dir, "shard_*.pt")))
    all_data = []
    for f in shard_files:
        shard = torch.load(f, weights_only=False)
        all_data.extend(shard)
    logger.info(f"Total: {len(all_data)} trajectories from {len(shard_files)} shards")
    return all_data


def split_by_problem(trajectories, val_fraction=0.1, seed=42):
    rng = random.Random(seed)
    problem_ids = sorted(set(t["problem_id"] for t in trajectories))
    rng.shuffle(problem_ids)
    n_val = max(50, int(len(problem_ids) * val_fraction))
    val_pids = set(problem_ids[:n_val])
    train = [t for t in trajectories if t["problem_id"] not in val_pids]
    val = [t for t in trajectories if t["problem_id"] in val_pids]
    return train, val


def evaluate(model, val_loader, device):
    """Evaluate PRM on validation data.

    Soft BCE is the primary metric. Accuracy thresholds both predictions and
    targets at 0.5 and is retained only as a secondary diagnostic.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    bucket_correct = {}  # mask_ratio bucket → (correct, total)
    bucket_size = 0.1

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)
            mask_ratios = batch["mask_ratios"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids, prompt_lens, mask_ratios, attention_mask=attention_mask
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = (logits > 0).float()
            hard_labels = (labels >= 0.5).float()
            correct += (preds == hard_labels).sum().item()
            total += len(labels)

            # Per-bucket tracking
            for mr, pred, label in zip(
                mask_ratios.cpu(), preds.cpu(), hard_labels.cpu()
            ):
                bucket = int(mr.item() / bucket_size)
                bucket = min(bucket, 9)
                key = f"{bucket * bucket_size:.1f}-{(bucket + 1) * bucket_size:.1f}"
                if key not in bucket_correct:
                    bucket_correct[key] = [0, 0]
                bucket_correct[key][1] += 1
                bucket_correct[key][0] += int(pred == label)

    model.train()
    bucket_acc = {k: v[0] / v[1] for k, v in sorted(bucket_correct.items()) if v[1] > 0}
    return total_loss / max(total, 1), correct / max(total, 1), bucket_acc


def main(
    argv: Sequence[str] | None = None,
    *,
    config_overlay: Mapping[str, object] | None = None,
):
    parser = argparse.ArgumentParser(
        description="Train DiffusionPRM on trajectory data"
    )
    parser.add_argument("--trajectory_dir", type=str, default=None)
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Frozen fitting-state PT/JSONL file for continuation-value training",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=None,
        help="Frozen validation-state PT/JSONL file",
    )
    parser.add_argument(
        "--label_field",
        type=str,
        default="binary_label",
        choices=["binary_label", "fractional_label"],
        help="Target field when --train_file/--val_file are supplied",
    )
    parser.add_argument(
        "--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B"
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints/prm_gsm8k")

    # Training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_fraction", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_init_seed",
        type=int,
        default=None,
        help=(
            "Optional seed used only for model initialization. Training data "
            "order and stochastic updates continue to use --seed."
        ),
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # PRM-specific
    parser.add_argument("--step_embed_dim", type=int, default=256)
    parser.add_argument("--reward_hidden", type=int, default=1024)
    parser.add_argument(
        "--min_mask_ratio",
        type=float,
        default=0.0,
        help="Minimum mask ratio to include in training",
    )
    parser.add_argument(
        "--max_mask_ratio",
        type=float,
        default=1.0,
        help="Maximum mask ratio to include in training",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Use causal attention (RA7 ablation for C3 claim)",
    )
    parser.add_argument(
        "--no_step_embed",
        action="store_true",
        help="Ablation RA1: remove step embedding",
    )
    parser.add_argument(
        "--no_mask_aware",
        action="store_true",
        help="Ablation RA2: include MASK tokens in pooling",
    )
    parser.add_argument(
        "--pool_strategy",
        type=str,
        default="mean",
        choices=["mean", "last_token"],
        help="Pooling strategy (Ouyang 2022 / Stiennon 2020 "
        "standard: last_token for causal RMs)",
    )

    # Checkpointing
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--eval_every_steps", type=int, default=500)
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Max training steps (0=no limit, use epochs)",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=0,
        help="Max validation samples (0=all). Subsample for speed.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--expected_init_sha256",
        type=str,
        default=None,
        help="Fail unless the initial trainable-parameter hash matches this value",
    )

    # WandB
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dllm-prm-gap")
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Set via --wandb_entity or WANDB_ENTITY env",
    )
    parser.add_argument("--run_name", type=str, default="prm_gsm8k")
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="Stable WandB ID; reused automatically when --resume is set",
    )

    args = parser.parse_args(argv)
    if bool(args.train_file) != bool(args.val_file):
        parser.error("--train_file and --val_file must be supplied together")
    if not args.train_file and not args.trajectory_dir:
        parser.error("provide --trajectory_dir or frozen --train_file/--val_file")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.resume and not args.wandb_run_id:
        resume_path = os.path.join(args.output_dir, "latest.pt")
        if os.path.exists(resume_path):
            resume_metadata = torch.load(
                resume_path, map_location="cpu", weights_only=False
            )
            args.wandb_run_id = resume_metadata.get("wandb_run_id")
    set_seed(args.seed)
    device = torch.device("cuda")

    config = dict(vars(args))
    if config_overlay is not None:
        # Formal wrappers keep their immutable, content-addressed contract in
        # every checkpoint.  Namespacing prevents it from overriding trainer
        # arguments while making resume identity fail closed.
        config["formal_provenance"] = dict(config_overlay)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=config,
            id=args.wandb_run_id,
            resume="must" if args.resume and args.wandb_run_id else "allow",
        )
        logger.info(f"WandB run: {run.id}")
        args.wandb_run_id = run.id
        config["wandb_run_id"] = run.id

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.train_file:
        train_records = load_state_records(args.train_file)
        val_records = load_state_records(args.val_file)
        train_pids = {str(record["problem_id"]) for record in train_records}
        val_pids = {str(record["problem_id"]) for record in val_records}
        overlap = train_pids & val_pids
        if overlap:
            raise ValueError(
                f"Frozen fitting/validation files overlap on {len(overlap)} problem IDs"
            )
        train_dataset = DiffusionPRMStateDataset(
            train_records, label_field=args.label_field
        )
        val_dataset = DiffusionPRMStateDataset(
            val_records, label_field=args.label_field
        )
        logger.info(
            "Loaded frozen state split: %d fitting states / %d validation "
            "states (%s target)",
            len(train_dataset),
            len(val_dataset),
            args.label_field,
        )
    else:
        trajectories = load_trajectories(args.trajectory_dir)
        train_trajs, val_trajs = split_by_problem(trajectories, seed=args.seed)
        train_dataset = DiffusionPRMDataset(
            train_trajs,
            min_mask_ratio=args.min_mask_ratio,
            max_mask_ratio=args.max_mask_ratio,
        )
        val_dataset = DiffusionPRMDataset(
            val_trajs,
            min_mask_ratio=args.min_mask_ratio,
            max_mask_ratio=args.max_mask_ratio,
        )

    # Subsample validation set for speed (142K samples → ~6K takes ~7min vs ~170min)
    if args.max_val_samples > 0 and len(val_dataset) > args.max_val_samples:
        rng = random.Random(args.seed)
        indices = list(range(len(val_dataset)))
        rng.shuffle(indices)
        val_dataset.samples = [
            val_dataset.samples[i] for i in indices[: args.max_val_samples]
        ]
        logger.info(f"Subsampled val from {len(indices)} to {len(val_dataset)} samples")

    n_pos = sum(1 for s in train_dataset.samples if s["label"] >= 0.5)
    mean_target = sum(float(s["label"]) for s in train_dataset.samples) / max(
        len(train_dataset), 1
    )
    logger.info(
        f"Train: {len(train_dataset)} samples "
        f"({n_pos} targets >= 0.5, mean target={mean_target:.4f})"
    )
    logger.info(f"Val: {len(val_dataset)} samples")

    collate = partial(collate_fn)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )

    def make_train_loader(epoch):
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)
        return DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate,
            num_workers=4,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    from dream.modeling_dream import DreamModel

    init_seed = args.seed if args.model_init_seed is None else args.model_init_seed
    set_seed(init_seed)
    logger.info(f"Loading backbone from {args.model_path}")
    backbone = DreamModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    # PEFT wrapping on CPU first: get_peft_model() iterates all params in
    # Python; doing this on GPU tensors causes thousands of CPU-GPU syncs.
    # Wrap on CPU (~10s), then move everything to GPU at once (~30s).
    logger.info("Wrapping backbone with LoRA (on CPU)...")
    model = DiffusionPRM(
        backbone,
        hidden_size=backbone.config.hidden_size,
        step_embed_dim=args.step_embed_dim,
        reward_hidden=args.reward_hidden,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        causal=args.causal,
        no_step_embed=args.no_step_embed,
        no_mask_aware=args.no_mask_aware,
        pool_strategy=args.pool_strategy,
    )
    logger.info("Moving model to GPU...")
    model = model.to(device)
    logger.info(f"Trainable params: {model.trainable_params():,}")
    initial_trainable_state = trainable_state_dict(model)
    init_sha256 = state_dict_sha256(initial_trainable_state)
    logger.info(f"Initial trainable-state SHA256: {init_sha256}")
    if args.expected_init_sha256 and init_sha256 != args.expected_init_sha256:
        raise RuntimeError(
            f"Paired initialization mismatch: expected "
            f"{args.expected_init_sha256}, got {init_sha256}"
        )
    config["init_sha256"] = init_sha256
    with open(os.path.join(args.output_dir, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    # Decouple model initialization from the stochastic training stream.  This
    # permits paired conditions to share exact initial weights while preserving
    # independent, explicitly seeded training replicates.
    set_seed(args.seed)
    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    batches_per_epoch = len(make_train_loader(0))
    updates_per_epoch = batches_per_epoch // args.grad_accum
    if updates_per_epoch <= 0:
        raise ValueError("Training split is smaller than one optimizer update")
    if args.max_steps > 0:
        total_steps = args.max_steps
        logger.info(f"Using exact max_steps={total_steps}")
    else:
        total_steps = updates_per_epoch * args.epochs
    logger.info(
        f"Total steps: {total_steps}, "
        f"effective batch: {args.batch_size * args.grad_accum}"
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_step = 0
    start_epoch = 0
    start_batch_idx = 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    if args.resume:
        ckpt_path = os.path.join(args.output_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, weights_only=False)
            validate_resume_config(ckpt.get("config", {}), config)
            saved_state = ckpt.get("trainable_state_dict", ckpt.get("model_state_dict"))
            if saved_state is None:
                raise KeyError(f"No model state found in {ckpt_path}")
            load_state_dict_audited(
                model,
                saved_state,
                adapter_only="trainable_state_dict" in ckpt,
            )
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_step = ckpt["step"]
            start_epoch = ckpt.get("next_epoch", ckpt.get("epoch", 0))
            start_batch_idx = ckpt.get("next_batch_idx", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            best_val_acc = ckpt.get("best_val_acc", 0.0)
            if ckpt.get("init_sha256") != init_sha256:
                raise RuntimeError("Resume checkpoint initialization hash mismatch")
            if "rng_state" not in ckpt:
                raise RuntimeError("Resume checkpoint lacks RNG state")
            restore_rng_state(ckpt["rng_state"])
            logger.info(
                "Resumed from step %d, epoch %d, batch %d",
                start_step,
                start_epoch,
                start_batch_idx,
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    model.train()
    global_step = start_step
    accum_loss = 0.0
    optimizer.zero_grad()
    start_time = time.time()

    epoch = start_epoch
    while global_step < total_steps:
        train_loader = make_train_loader(epoch)
        resume_batch_idx = start_batch_idx if epoch == start_epoch else 0
        if resume_batch_idx % args.grad_accum != 0:
            raise RuntimeError(
                "Resume batch must be on an optimizer-step boundary, got "
                f"{resume_batch_idx}"
            )
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < resume_batch_idx:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)
            mask_ratios = batch["mask_ratios"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids, prompt_lens, mask_ratios, attention_mask=attention_mask
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                next_epoch = epoch
                next_batch_idx = batch_idx + 1
                if next_batch_idx >= len(train_loader):
                    next_epoch = epoch + 1
                    next_batch_idx = 0

                if global_step % args.log_every_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    logger.info(
                        f"step={global_step}/{total_steps} "
                        f"loss={accum_loss:.4f} lr={lr:.2e} "
                        f"elapsed={elapsed / 60:.1f}min"
                    )
                    if args.wandb:
                        wandb.log(
                            {
                                "train/loss": accum_loss,
                                "train/lr": lr,
                            },
                            step=global_step,
                        )
                    accum_loss = 0.0

                if global_step % args.eval_every_steps == 0:
                    val_loss, val_acc, bucket_acc = evaluate(model, val_loader, device)
                    logger.info(f"  val_loss={val_loss:.4f} val_acc={val_acc:.1%}")
                    logger.info(f"  per-bucket acc: {bucket_acc}")
                    if args.wandb:
                        log_dict = {
                            "val/loss": val_loss,
                            "val/accuracy": val_acc,
                        }
                        for k, v in bucket_acc.items():
                            log_dict[f"val/acc_mr_{k}"] = v
                        wandb.log(log_dict, step=global_step)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_acc = val_acc
                        atomic_save(
                            {
                                "step": global_step,
                                "trainable_state_dict": trainable_state_dict(model),
                                "best_val_loss": best_val_loss,
                                "best_val_acc": best_val_acc,
                                "init_sha256": init_sha256,
                                "config": config,
                            },
                            os.path.join(args.output_dir, "best.pt"),
                        )
                        logger.info(
                            f"  New best soft BCE: {val_loss:.6f} "
                            f"(hard acc {val_acc:.1%})"
                        )

                if global_step % args.save_every_steps == 0:
                    ckpt = {
                        "step": global_step,
                        "epoch": epoch,
                        "next_epoch": next_epoch,
                        "next_batch_idx": next_batch_idx,
                        "trainable_state_dict": trainable_state_dict(model),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_val_acc": best_val_acc,
                        "init_sha256": init_sha256,
                        "rng_state": capture_rng_state(),
                        "wandb_run_id": args.wandb_run_id,
                        "config": config,
                    }
                    atomic_save(ckpt, os.path.join(args.output_dir, "latest.pt"))
                    logger.info(f"  Checkpoint saved at step {global_step}")

                if global_step >= total_steps:
                    break
        epoch += 1
        start_batch_idx = 0

    # ------------------------------------------------------------------
    # Final eval & save
    # ------------------------------------------------------------------
    val_loss, val_acc, bucket_acc = evaluate(model, val_loader, device)
    logger.info(f"Final: val_loss={val_loss:.4f} val_acc={val_acc:.1%}")
    logger.info(f"Final per-bucket: {bucket_acc}")
    # Persist the terminal validation readout before final checkpoints so it
    # cannot exist only in buffered logs after an interrupted network write.
    with open(os.path.join(args.output_dir, "val_final.json"), "w") as handle:
        json.dump(
            {
                "soft_bce": val_loss,
                "hard_accuracy": val_acc,
                "bucket_hard_accuracy": bucket_acc,
                "checkpoint_selection_metric": "soft_bce",
                "step": global_step,
            },
            handle,
            indent=2,
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        atomic_save(
            {
                "step": global_step,
                "trainable_state_dict": trainable_state_dict(model),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "init_sha256": init_sha256,
                "config": config,
            },
            os.path.join(args.output_dir, "best.pt"),
        )

    final_state = trainable_state_dict(model)
    final_state_sha256 = state_dict_sha256(final_state)
    atomic_save(
        {
            "step": global_step,
            "trainable_state_dict": final_state,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "final_val_acc": val_acc,
            "final_val_loss": val_loss,
            "init_sha256": init_sha256,
            "state_sha256": final_state_sha256,
            "config": config,
        },
        os.path.join(args.output_dir, "final.pt"),
    )

    summary = {
        "total_steps": global_step,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "final_val_acc": val_acc,
        "final_val_loss": val_loss,
        "bucket_accuracy": bucket_acc,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "trainable_params": model.trainable_params(),
        "init_sha256": init_sha256,
        "final_state_sha256": final_state_sha256,
        "label_field": args.label_field if args.train_file else "is_correct",
        "wandb_run_id": args.wandb_run_id,
        "wall_time_seconds": time.time() - start_time,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
