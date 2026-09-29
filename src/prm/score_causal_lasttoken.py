"""Score final-state (mask_ratio=0) trajectories with the causal last-token PRM.

Scores candidates with a causal-attention PRM whose readout is the last non-MASK
token hidden state (instead of mean pooling over the solution region). This is the
readout-ablation variant analyzed in the paper's causal-PRM section.

Usage (8-GPU parallel on local):
    for i in {0..7}; do
        CUDA_VISIBLE_DEVICES=$i python src/prm/score_causal_lasttoken.py \
            --from_trajectories data/prm_trajectories/gsm8k_test_N32_combined.pt \
            --prm_checkpoint checkpoints/prm_causal_lasttoken_v1/final.pt \
            --shard_id $i --num_shards 8 \
            --output_json eval_results/causal_lasttoken_scores_shard_$i.json &
    done; wait
"""
import os, sys, json, argparse, logging
import torch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD', 'eval'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD'))

from prm.model import DiffusionPRM

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from_trajectories", required=True)
    p.add_argument("--prm_checkpoint", required=True)
    p.add_argument("--pool_strategy", default="last_token",
                   choices=["mean", "last_token"])
    p.add_argument("--causal", action="store_true", default=True)
    p.add_argument("--model_path", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--output_json", required=True)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda")
    from dream.modeling_dream import DreamModel
    logger.info(f"[shard {args.shard_id}] Loading Dream backbone")
    backbone = DreamModel.from_pretrained(
        args.model_path, trust_remote_code=True,
        attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = DiffusionPRM(
        backbone, hidden_size=backbone.config.hidden_size,
        causal=args.causal,
        pool_strategy=args.pool_strategy,
    )
    state = torch.load(args.prm_checkpoint, map_location='cpu', weights_only=False)
    sd = state.get('model_state_dict', state.get('model', state)) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"[shard {args.shard_id}] Loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    model.to(device).eval()

    all_traj = torch.load(args.from_trajectories, weights_only=False)
    my_traj = [t for i, t in enumerate(all_traj) if i % args.num_shards == args.shard_id]
    logger.info(f"[shard {args.shard_id}] {len(my_traj)}/{len(all_traj)} traj to score")
    del all_traj

    PAD = 151643
    scores, meta = [], []
    for i in range(0, len(my_traj), args.batch_size):
        batch = my_traj[i:i + args.batch_size]
        items = []
        for t in batch:
            gs = t['gen_snapshots']
            if isinstance(gs, list):
                gen_ids = gs[-1] if len(gs) > 0 else torch.zeros(1, dtype=torch.long)
            else:
                gen_ids = gs[-1] if gs.shape[0] > 0 else torch.zeros(1, dtype=torch.long)
            items.append({'prompt_ids': t['prompt_ids'].long(), 'gen_ids': gen_ids.long()})
        max_len = max(len(b['prompt_ids']) + len(b['gen_ids']) for b in items)
        ids = torch.full((len(items), max_len), PAD, dtype=torch.long)
        pl = torch.tensor([len(b['prompt_ids']) for b in items], dtype=torch.long)
        am = torch.zeros((len(items), max_len), dtype=torch.bool)
        for j, b in enumerate(items):
            full = torch.cat([b['prompt_ids'], b['gen_ids']])
            ids[j, :len(full)] = full
            am[j, :len(full)] = True
        with torch.no_grad():
            mr_t = torch.zeros(len(items), dtype=torch.float32, device=device)
            out = model(ids.to(device), pl.to(device), mr_t, attention_mask=am.to(device))
            scores.extend(out.cpu().tolist())
        for t in batch:
            meta.append({
                'problem_id': t['problem_id'],
                'trajectory_id': t.get('trajectory_id', 0),
                'seed': t.get('seed', 0),
                'is_correct': bool(t.get('is_correct', False)),
                'answer': t.get('answer_extracted'),
            })
        if i % (args.batch_size * 50) == 0:
            print(f"[shard {args.shard_id}] scored {i+len(batch)}/{len(my_traj)}", flush=True)

    out = {
        'shard_id': args.shard_id, 'num_shards': args.num_shards,
        'pool_strategy': args.pool_strategy, 'causal': args.causal,
        'checkpoint': args.prm_checkpoint,
        'items': [{'meta': m, 'score': s} for m, s in zip(meta, scores)],
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f)
    logger.info(f"[shard {args.shard_id}] saved {args.output_json}")


if __name__ == "__main__":
    main()
