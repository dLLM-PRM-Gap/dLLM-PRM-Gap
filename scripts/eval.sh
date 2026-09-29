#!/bin/bash
# Example evaluation: PRM-guided beam search on GSM8K test set.

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/external/CoDD/eval:$PROJECT_ROOT/external/CoDD/lm-evaluation-harness:$PROJECT_ROOT/external/CoDD:$PYTHONPATH"

PRM_CHECKPOINT="${PRM_CHECKPOINT:-./checkpoints/prm_bidir_meanpool/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_results/prm_guided_K8_be64}"
BRANCH_FACTOR="${BRANCH_FACTOR:-8}"
BRANCH_EVERY="${BRANCH_EVERY:-64}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python src/prm/eval_prm_guided_sharded.py \
    --task gsm8k --prm_checkpoint "$PRM_CHECKPOINT" \
    --branch_factor "$BRANCH_FACTOR" --branch_every "$BRANCH_EVERY" \
    --shard_id "$SHARD_ID" --num_shards "$NUM_SHARDS" \
    --output_dir "$OUTPUT_DIR" \
    --resume
