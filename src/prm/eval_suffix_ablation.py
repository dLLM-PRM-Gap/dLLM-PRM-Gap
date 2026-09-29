"""Suffix ablation: for each snapshot, zero out right-half unmasked tokens.
If bidir PRM score changes significantly, bidir genuinely uses suffix info.
"""
import os, sys, json, argparse, logging
import torch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD', 'eval'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD'))

from prm.model import DiffusionPRM, MASK_TOKEN_ID
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--from_trajectories', required=True)
    p.add_argument('--prm_checkpoint', required=True)
    p.add_argument('--model_path', default='Dream-org/Dream-v0-Instruct-7B')
    p.add_argument('--output_json', required=True)
    p.add_argument('--shard_id', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    args = p.parse_args()

    device = torch.device('cuda')
    from dream.modeling_dream import DreamModel
    backbone = DreamModel.from_pretrained(args.model_path, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16, local_files_only=True)
    prm = DiffusionPRM(backbone, hidden_size=backbone.config.hidden_size)
    state = torch.load(args.prm_checkpoint, map_location='cpu', weights_only=False)
    sd = state.get('model_state_dict', state.get('model', state))
    prm.load_state_dict(sd, strict=False)
    prm.to(device).eval()
    logger.info(f"[shard {args.shard_id}] PRM loaded")

    trajs = torch.load(args.from_trajectories, weights_only=False)
    my_trajs = [t for i, t in enumerate(trajs) if i % args.num_shards == args.shard_id]
    logger.info(f"[shard {args.shard_id}] {len(my_trajs)} traj")

    results = []
    for idx, t in enumerate(my_trajs):
        prompt_ids = t['prompt_ids'].long()
        prompt_len = len(prompt_ids)
        is_correct = bool(t.get('is_correct', False))
        for step, gen_snap, mr in zip(t.get('snapshot_steps', []), t.get('gen_snapshots', []), t.get('mask_ratios', [])):
            if gen_snap is None: continue
            gen_snap = gen_snap.long()
            gen_len = len(gen_snap)
            mid = gen_len // 2
            # Original
            ids_orig = torch.cat([prompt_ids, gen_snap]).unsqueeze(0).to(device)
            # Ablated: right half of GEN region → MASK
            gen_ablate = gen_snap.clone()
            gen_ablate[mid:] = MASK_TOKEN_ID
            ids_ablate = torch.cat([prompt_ids, gen_ablate]).unsqueeze(0).to(device)
            pl = torch.tensor([prompt_len], dtype=torch.long, device=device)
            mr_t = torch.tensor([float(mr)], dtype=torch.float32, device=device)
            am_orig = torch.ones_like(ids_orig, dtype=torch.bool)
            am_ablate = torch.ones_like(ids_ablate, dtype=torch.bool)
            with torch.no_grad():
                s_orig = prm(ids_orig, pl, mr_t, attention_mask=am_orig).item()
                s_ablate = prm(ids_ablate, pl, mr_t, attention_mask=am_ablate).item()
            results.append({'pid': t['problem_id'], 'tid': t.get('trajectory_id', 0), 'step': int(step), 'mr': float(mr), 'score_orig': s_orig, 'score_ablate': s_ablate, 'delta': s_orig - s_ablate, 'final_correct': is_correct})
        if (idx+1) % 50 == 0:
            print(f"[shard {args.shard_id}] {idx+1}/{len(my_trajs)}", flush=True)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump({'shard_id': args.shard_id, 'results': results}, f)
    logger.info(f"[shard {args.shard_id}] saved {len(results)} snapshot pairs")

if __name__ == '__main__':
    main()
