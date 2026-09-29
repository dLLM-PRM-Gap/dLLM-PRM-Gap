#!/bin/bash
# Example training script: bidir PRM with mean-pool readout on Dream-7B.
# Adjust paths, WandB entity, and resource settings for your environment.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/external/CoDD/eval:$PROJECT_ROOT/external/CoDD/lm-evaluation-harness:$PROJECT_ROOT/external/CoDD:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python src/prm/train_diffusion_prm.py \
    --trajectory_dir "${TRAJECTORY_DIR:-./data/prm_trajectories/gsm8k}" \
    --output_dir "${OUTPUT_DIR:-./checkpoints/prm_bidir_meanpool}" \
    --pool_strategy mean \
    --epochs 2 --batch_size 4 --grad_accum 8 --lr 1e-5 \
    --max_steps "${MAX_STEPS:-15000}" --seed "${SEED:-42}" \
    --eval_every_steps 2500 --max_val_samples 2000 \
    --save_every_steps 2500 --log_every_steps 100 \
    ${WANDB_ENABLE:+--wandb --wandb_project dllm-prm-gap --wandb_entity "$WANDB_ENTITY" --run_name "${RUN_NAME:-bidir_meanpool}"}
