"""Two-pass causal: score with causal PRM on forward + reversed input.
L→R: original causal score.
R→L: reverse input tokens before scoring (simulates right-to-left causal).
Combined: average or max of both.
Compare against bidir PRM — if two-pass ≈ bidir, the key is TWO-SIDED context.
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
    p.add_argument('--causal_prm_checkpoint', required=True)
    p.add_argument('--bidir_prm_checkpoint', required=True)
    p.add_argument('--model_path', default='Dream-org/Dream-v0-Instruct-7B')
    p.add_argument('--output_json', required=True)
    p.add_argument('--shard_id', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    args = p.parse_args()

    device = torch.device('cuda')
    from dream.modeling_dream import DreamModel

    # Load CAUSAL PRM (use causal=True)
    backbone_c = DreamModel.from_pretrained(args.model_path, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16, local_files_only=True)
    causal_prm = DiffusionPRM(backbone_c, hidden_size=backbone_c.config.hidden_size, causal=True)
    state = torch.load(args.causal_prm_checkpoint, map_location='cpu', weights_only=False)
    sd = state.get('model_state_dict', state.get('model', state))
    causal_prm.load_state_dict(sd, strict=False)
    causal_prm.to(device).eval()
    logger.info(f"[shard {args.shard_id}] causal PRM loaded")

    # Load BIDIR PRM (reference)
    backbone_b = DreamModel.from_pretrained(args.model_path, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16, local_files_only=True)
    bidir_prm = DiffusionPRM(backbone_b, hidden_size=backbone_b.config.hidden_size)
    state2 = torch.load(args.bidir_prm_checkpoint, map_location='cpu', weights_only=False)
    sd2 = state2.get('model_state_dict', state2.get('model', state2))
    bidir_prm.load_state_dict(sd2, strict=False)
    bidir_prm.to(device).eval()
    logger.info(f"[shard {args.shard_id}] bidir PRM loaded")

    trajs = torch.load(args.from_trajectories, weights_only=False)
    my_trajs = [t for i, t in enumerate(trajs) if i % args.num_shards == args.shard_id]

    results = []
    for idx, t in enumerate(my_trajs):
        prompt_ids = t['prompt_ids'].long()
        prompt_len = len(prompt_ids)
        is_correct = bool(t.get('is_correct', False))
        for step, gen_snap, mr in zip(t.get('snapshot_steps', []), t.get('gen_snapshots', []), t.get('mask_ratios', [])):
            if gen_snap is None: continue
            gen_snap = gen_snap.long()
            gen_len = len(gen_snap)
            # Original order (L→R causal)
            ids_fwd = torch.cat([prompt_ids, gen_snap]).unsqueeze(0).to(device)
            # Reversed gen region only (keep prompt unchanged); R→L simulates reverse-causal
            gen_rev = gen_snap.flip(0)
            ids_bwd = torch.cat([prompt_ids, gen_rev]).unsqueeze(0).to(device)
            pl = torch.tensor([prompt_len], dtype=torch.long, device=device)
            mr_t = torch.tensor([float(mr)], dtype=torch.float32, device=device)
            am = torch.ones_like(ids_fwd, dtype=torch.bool)
            with torch.no_grad():
                s_fwd = causal_prm(ids_fwd, pl, mr_t, attention_mask=am).item()
                s_bwd = causal_prm(ids_bwd, pl, mr_t, attention_mask=am).item()
                s_bidir = bidir_prm(ids_fwd, pl, mr_t, attention_mask=am).item()
            results.append({'pid': t['problem_id'], 'tid': t.get('trajectory_id', 0), 'step': int(step), 'mr': float(mr),
                           'causal_fwd': s_fwd, 'causal_bwd': s_bwd, 'causal_twopass_avg': (s_fwd+s_bwd)/2, 'bidir': s_bidir,
                           'final_correct': is_correct})
        if (idx+1) % 50 == 0:
            print(f"[shard {args.shard_id}] {idx+1}/{len(my_trajs)}", flush=True)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump({'shard_id': args.shard_id, 'results': results}, f)

if __name__ == '__main__':
    main()
