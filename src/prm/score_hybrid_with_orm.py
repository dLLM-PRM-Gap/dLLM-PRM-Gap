"""Score PRM-Hybrid K_candidates with Dream bidir ORM to produce REAL ORM-Rerank baseline.
Input:  eval_results/prm_guided_hybrid_K8_be64_sXX/prm_guided_gsm8k_K8_shard*.json (has K_candidates with gen_tokens)
Output: eval_results/hybrid_orm_rerank/hybrid_orm_sXX.json (per-problem ORM choice + correctness)
"""
import os, sys, json, argparse, glob, logging
import torch
import numpy as np

PROJECT_ROOT = os.environ.get(
    'DIFFUSION_PRM_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD', 'eval'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD'))
from src.prm.model import OutcomeRewardModel, MASK_TOKEN_ID

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def load_orm(ckpt, model_path, device):
    from dream.modeling_dream import DreamModel
    bb = DreamModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16, local_files_only=True)
    m = OutcomeRewardModel(bb, hidden_size=bb.config.hidden_size)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd) if isinstance(sd, dict) else sd
    miss, unexp = m.load_state_dict(sd, strict=False)
    logger.info(f"Loaded ORM: missing={len(miss)}, unexpected={len(unexp)}")
    m.to(device); m.eval()
    return m

def strip_pad(ids, pad_id=151643):
    arr = np.array(ids, dtype=np.int64)
    keep = arr != pad_id
    if not keep.any(): return arr[:1]
    last_real = np.where(keep)[0][-1]
    return arr[:last_real + 1]

def score_batch(model, items, device, pad_id=151643, batch_size=4):
    scores = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        max_len = max(len(b['prompt_ids']) + len(b['gen_ids']) for b in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        pl = torch.tensor([len(b['prompt_ids']) for b in batch], dtype=torch.long)
        am = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for j, b in enumerate(batch):
            full = np.concatenate([b['prompt_ids'], b['gen_ids']])
            ids[j, :len(full)] = torch.from_numpy(full)
            am[j, :len(full)] = True
        with torch.no_grad():
            out = model(ids.to(device), pl.to(device), attention_mask=am.to(device))
            scores.extend(out.cpu().tolist())
    return scores

def process_hybrid_dir(hybrid_dir, model, device, pad_id=151643, batch_size=4):
    """Load hybrid shards, score each K candidate, save ORM-rerank results."""
    results = {}
    shards = sorted(glob.glob(os.path.join(hybrid_dir, 'prm_guided_gsm8k_K*_shard*.json')))
    shards = [s for s in shards if 'partial' not in s and 'merged' not in s]
    total = 0
    for si, shard_path in enumerate(shards):
        d = json.load(open(shard_path))
        pe = d.get('per_example', [])
        logger.info(f"[{si+1}/{len(shards)}] {os.path.basename(shard_path)}: {len(pe)} problems")
        for ex in pe:
            gi = ex['global_idx']
            prompt_ids = np.array(ex['prompt_ids'], dtype=np.int64)
            K_candidates = ex.get('K_candidates', [])
            # Build items
            items = []
            for kc in K_candidates:
                gen_ids = strip_pad(kc['gen_tokens'], pad_id=pad_id)
                items.append({'prompt_ids': prompt_ids, 'gen_ids': gen_ids})
            # Score all K candidates
            orm_scores = score_batch(model, items, device, pad_id=pad_id, batch_size=batch_size)
            # Select by ORM score
            best_idx = int(np.argmax(orm_scores))
            best = K_candidates[best_idx]
            results[gi] = {
                'orm_best_idx': best_idx,
                'orm_best_correct': bool(best['is_correct']),
                'orm_scores': [float(s) for s in orm_scores],
                'per_candidate_correct': [bool(c['is_correct']) for c in K_candidates],
                'prm_best_correct': bool(ex.get('correct_prm_best', False)),
                'oracle_correct': bool(ex.get('oracle_correct', False)),
            }
            total += 1
            if total % 50 == 0:
                n_correct = sum(1 for r in results.values() if r['orm_best_correct'])
                logger.info(f"  progress: {total} done, ORM-Rerank on Hybrid acc so far = {n_correct/total*100:.2f}%")
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hybrid_dir', required=True)
    p.add_argument('--orm_checkpoint', default=f'{PROJECT_ROOT}/checkpoints/orm_gsm8k_v2/best.pt',
                   help='Path to bidir ORM .pt checkpoint.')
    p.add_argument('--model_path', default='Dream-org/Dream-v0-Instruct-7B',
                   help='Dream backbone: HuggingFace repo ID or local path.')
    p.add_argument('--output_json', required=True)
    p.add_argument('--batch_size', type=int, default=4)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_orm(args.orm_checkpoint, args.model_path, device)
    results = process_hybrid_dir(args.hybrid_dir, model, device, batch_size=args.batch_size)

    # Aggregate
    n = len(results)
    prm_acc = sum(1 for r in results.values() if r['prm_best_correct']) / n
    orm_acc = sum(1 for r in results.values() if r['orm_best_correct']) / n
    oracle_acc = sum(1 for r in results.values() if r['oracle_correct']) / n
    out = {
        'hybrid_dir': args.hybrid_dir,
        'n_problems': n,
        'prm_best_acc': prm_acc * 100,
        'orm_rerank_on_hybrid_acc': orm_acc * 100,
        'oracle_hybrid_acc': oracle_acc * 100,
        'per_problem': results,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    json.dump(out, open(args.output_json, 'w'), indent=2)
    logger.info(f"==== Done ====")
    logger.info(f"PRM-picks: {prm_acc*100:.2f}%  ORM-rerank-on-Hybrid: {orm_acc*100:.2f}%  Oracle-on-Hybrid: {oracle_acc*100:.2f}%")
    logger.info(f"Saved: {args.output_json}")

if __name__ == '__main__':
    main()
