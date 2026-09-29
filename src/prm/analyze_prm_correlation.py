"""Gemini's mechanism experiment: PRM score vs final trajectory correctness,
grouped by mask_ratio bucket. Proves PRM signal is noisy at high mask_ratio.

For each trajectory in test set, PRM-score each snapshot, then correlate
that score with the FINAL correctness of that trajectory.
"""
import os, sys, json, argparse, logging
import torch, torch.nn.functional as F
from collections import defaultdict

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD', 'eval'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD'))

from prm.model import DiffusionPRM

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
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

    trajs = torch.load(args.from_trajectories, weights_only=False)
    my_trajs = [t for i, t in enumerate(trajs) if i % args.num_shards == args.shard_id]
    logger.info(f"[shard {args.shard_id}] {len(my_trajs)} trajectories")

    PAD = 151643
    results = []  # per snapshot: {problem_id, traj_id, mask_ratio, snapshot_step, prm_score, is_final_correct}

    for idx, t in enumerate(my_trajs):
        is_correct = bool(t.get('is_correct', False))
        prompt_ids = t['prompt_ids'].long()
        prompt_len = len(prompt_ids)
        snapshots = t.get('gen_snapshots', [])
        mask_ratios = t.get('mask_ratios', [])
        snap_steps = t.get('snapshot_steps', [])
        for step, gen_snap, mr in zip(snap_steps, snapshots, mask_ratios):
            if gen_snap is None: continue
            gen_snap = gen_snap.long()
            ids = torch.cat([prompt_ids, gen_snap]).unsqueeze(0).to(device)
            pl = torch.tensor([prompt_len], dtype=torch.long, device=device)
            mr_t = torch.tensor([float(mr)], dtype=torch.float32, device=device)
            am = torch.ones_like(ids, dtype=torch.bool)
            with torch.no_grad():
                score = prm(ids, pl, mr_t, attention_mask=am)
            results.append({'pid': t['problem_id'], 'tid': t.get('trajectory_id', 0), 'step': int(step), 'mr': float(mr), 'score': score.item(), 'final_correct': is_correct})
        if (idx+1) % 50 == 0:
            print(f"[shard {args.shard_id}] {idx+1}/{len(my_trajs)} trajectories processed", flush=True)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump({'shard_id': args.shard_id, 'results': results}, f)
    logger.info(f"[shard {args.shard_id}] saved {len(results)} snapshot scores")

if __name__ == '__main__':
    main()
