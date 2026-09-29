"""
Phase 4 Eval: PRM-Guided Denoising on GSM8K (Sharded).

Standalone evaluation script supporting multi-GPU sharding.
Each shard processes a subset of the test set; results are merged after.

Usage:
    # Single GPU (all 1319 problems)
    CUDA_VISIBLE_DEVICES=0 python eval_prm_guided_sharded.py \
        --task gsm8k --prm_checkpoint checkpoints/prm_gsm8k/best.pt

    # Multi-GPU sharding (e.g., 8 GPUs)
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i python eval_prm_guided_sharded.py \
            --task gsm8k --prm_checkpoint checkpoints/prm_gsm8k/best.pt \
            --shard_id $i --num_shards 8 &
    done
    wait
    # Merge
    python eval_prm_guided_sharded.py --merge_only \
        --output_dir eval_results/prm_guided --task gsm8k --num_shards 8
"""

import os
import sys
import json
import argparse
import logging
import re
import time
import random

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD', 'eval'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)


# ---------- GSM8K 8-shot prompt (matches lm-eval format) ----------

GSM8K_FEWSHOT = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. #### 29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. #### 8"
    },
]


def build_gsm8k_prompt(question):
    """Build 8-shot CoT prompt for GSM8K, matching lm-eval format."""
    prompt_parts = []
    for ex in GSM8K_FEWSHOT:
        prompt_parts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
    prompt_parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(prompt_parts)


def extract_gsm8k_answer(text):
    """Extract numeric answer from GSM8K format (#### N)."""
    matches = re.findall(r'####\s*(-?\d[\d,]*\.?\d*)', text)
    if matches:
        return matches[-1].replace(',', '')
    # Fallback: last number in text
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def normalize_answer(ans):
    """Normalize answer for comparison."""
    if ans is None:
        return None
    ans = str(ans).strip().replace(',', '')
    try:
        return str(int(float(ans)))
    except (ValueError, OverflowError):
        return ans


# ---------- MATH500 support ----------

MATH500_SYSTEM_PROMPT = (
    "Solve the following math problem step by step. Show your work clearly. "
    "Conclude by presenting the final answer in the format: \\boxed{ANSWER}"
)


def load_math500_test():
    """Load MATH-500 test set from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    examples = []
    for item in ds:
        examples.append({
            'question': item['problem'],
            'gold_answer': item['answer'],
            'full_answer': item.get('solution', ''),
        })
    return examples


def build_math500_prompt(question):
    """Build 0-shot prompt for MATH500 with boxed answer instruction."""
    return f"{MATH500_SYSTEM_PROMPT}\n\nProblem: {question}\n\nSolution:"


def extract_math500_answer(text):
    """Extract answer from \\boxed{...} format."""
    # Find all \boxed{...} patterns (handle nested braces)
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    # Fallback: look for "the answer is" pattern
    m = re.search(r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def normalize_math_answer(ans):
    """Normalize MATH answer for comparison (handles fractions, expressions)."""
    if ans is None:
        return None
    ans = str(ans).strip()
    # Remove surrounding $ signs
    ans = ans.strip('$')
    # Remove \text{} wrappers
    ans = re.sub(r'\\text\{([^}]*)\}', r'\1', ans)
    # Normalize whitespace
    ans = ' '.join(ans.split())
    # Try numeric comparison
    try:
        return str(float(ans))
    except ValueError:
        pass
    return ans


def math_answers_equal(pred, gold):
    """Compare MATH answers with tolerance for different representations."""
    if pred is None or gold is None:
        return False
    pred_n = normalize_math_answer(pred)
    gold_n = normalize_math_answer(gold)
    if pred_n == gold_n:
        return True
    # Try numeric comparison with tolerance
    try:
        return abs(float(pred_n) - float(gold_n)) < 1e-6
    except (ValueError, TypeError):
        pass
    # String comparison after stripping LaTeX
    pred_clean = re.sub(r'[\\{}\s]', '', pred_n)
    gold_clean = re.sub(r'[\\{}\s]', '', gold_n)
    return pred_clean == gold_clean


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_gsm8k_test():
    """Load GSM8K test set from HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    examples = []
    for item in ds:
        gold_answer = item['answer'].split('####')[-1].strip()
        examples.append({
            'question': item['question'],
            'gold_answer': gold_answer,
            'full_answer': item['answer'],
        })
    return examples


def merge_shards(output_dir, task, num_shards, branch_factor, top_m):
    """Merge shard results into final output."""
    all_results = []
    for sid in range(num_shards):
        shard_path = os.path.join(
            output_dir, f"prm_guided_topm{top_m}_{task}_K{branch_factor}_shard{sid}.json"
        )
        if not os.path.exists(shard_path):
            logger.warning(f"Missing shard {sid}: {shard_path}")
            continue
        with open(shard_path) as f:
            shard_data = json.load(f)
        all_results.extend(shard_data['per_example'])
        logger.info(f"Shard {sid}: {shard_data['num_correct']}/{shard_data['num_total']} "
                     f"({shard_data['accuracy']:.1%})")

    total = len(all_results)
    correct = sum(1 for r in all_results if r['correct'])
    accuracy = correct / max(total, 1)

    merged = {
        'task': task,
        'method': 'prm_guided',
        'num_total': total,
        'num_correct': correct,
        'accuracy': accuracy,
        'num_shards': num_shards,
        'per_example': all_results,
    }

    out_path = os.path.join(output_dir, f"prm_guided_topm{top_m}_{task}_K{branch_factor}_merged.json")
    with open(out_path, 'w') as f:
        json.dump(merged, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"MERGED RESULTS: {correct}/{total} = {accuracy:.1%}")
    logger.info(f"Saved to {out_path}")
    logger.info("=" * 60)
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="PRM-Guided Denoising Evaluation (Sharded)"
    )

    # Core
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--model_path", type=str,
                        default="Dream-org/Dream-v0-Instruct-7B",
                        help="Dream backbone: HuggingFace repo ID or local path.")
    parser.add_argument("--prm_checkpoint", type=str, default="checkpoints/prm_gsm8k/best.pt")
    parser.add_argument("--output_dir", type=str, default="eval_results/prm_guided")

    # Sharding
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--merge_only", action="store_true",
                        help="Only merge existing shards, no evaluation")

    # Generation
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--alg", type=str, default="maskgit_plus")
    parser.add_argument("--alg_temp", type=float, default=0.5)

    # PRM-Guided specific
    parser.add_argument("--branch_every", type=int, default=16)
    parser.add_argument("--branch_factor", type=int, default=8)
    parser.add_argument("--top_m", type=int, default=2, help="Top-M retention (M=1 reduces to original PRM-Guided)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial results if available")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Merge mode
    # ------------------------------------------------------------------
    if args.merge_only:
        merge_shards(args.output_dir, args.task, args.num_shards, args.branch_factor, args.top_m)
        return

    set_seed(args.seed + args.shard_id)
    device = torch.device("cuda")

    # ------------------------------------------------------------------
    # Load test data and compute shard bounds
    # ------------------------------------------------------------------
    if args.task == "gsm8k":
        examples = load_gsm8k_test()
    elif args.task == "math500":
        examples = load_math500_test()
    else:
        raise ValueError(f"Unknown task: {args.task}")
    total = len(examples)
    shard_size = (total + args.num_shards - 1) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = min(start_idx + shard_size, total)
    shard_examples = examples[start_idx:end_idx]

    logger.info(f"Shard {args.shard_id}/{args.num_shards}: "
                f"examples {start_idx}-{end_idx} ({len(shard_examples)} problems)")

    # ------------------------------------------------------------------
    # Load generation model
    # ------------------------------------------------------------------
    from dream.modeling_dream import DreamModel
    from transformers import AutoTokenizer

    logger.info(f"Loading Dream-7B from {args.model_path}")
    gen_model = DreamModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    gen_model.eval()

    # ------------------------------------------------------------------
    # Load PRM
    # ------------------------------------------------------------------
    from prm.model import DiffusionPRM

    logger.info(f"Loading PRM from {args.prm_checkpoint}")
    ckpt = torch.load(args.prm_checkpoint, weights_only=False, map_location='cpu')
    ckpt_config = ckpt.get('config', {})

    prm_backbone = DreamModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )

    prm = DiffusionPRM(
        prm_backbone,
        hidden_size=prm_backbone.config.hidden_size,
        step_embed_dim=ckpt_config.get('step_embed_dim', 256),
        reward_hidden=ckpt_config.get('reward_hidden', 1024),
        lora_r=ckpt_config.get('lora_r', 16),
        lora_alpha=ckpt_config.get('lora_alpha', 32),
        lora_dropout=ckpt_config.get('lora_dropout', 0.05),
        causal=ckpt_config.get('causal', False),
        no_step_embed=ckpt_config.get('no_step_embed', False),
        no_mask_aware=ckpt_config.get('no_mask_aware', False),
    )
    state_dict = ckpt.get('model_state_dict', ckpt)
    del ckpt
    prm.load_state_dict(state_dict, strict=False)
    del state_dict
    if not prm.no_step_embed:
        prm.step_embed.to(device)
        prm.step_proj.to(device)
    prm.reward_head.to(device)
    prm.eval()
    logger.info("PRM loaded successfully")

    # ------------------------------------------------------------------
    # Evaluate (with optional resume from partial)
    # ------------------------------------------------------------------
    from prm.prm_guided_topm_denoise import prm_guided_topm_sample

    results = []
    correct_count = 0
    total_time = 0.0
    resume_from = 0

    if args.resume:
        partial_path = os.path.join(
            args.output_dir,
            f"prm_guided_topm{args.top_m}_{args.task}_K{args.branch_factor}_shard{args.shard_id}_partial.json"
        )
        if os.path.exists(partial_path):
            with open(partial_path) as f:
                partial_data = json.load(f)
            resume_from = partial_data['completed']
            results = partial_data['per_example']
            correct_count = partial_data['num_correct']
            total_time = partial_data.get('avg_time_per_problem', 0) * resume_from
            logger.info(f"Resuming from partial: {resume_from}/{len(shard_examples)} done, "
                        f"acc so far={correct_count/max(resume_from,1):.1%}")
        else:
            logger.info("No partial file found, starting from scratch")

    for i, ex in enumerate(shard_examples):
        if i < resume_from:
            continue
        global_idx = start_idx + i
        if args.task == "gsm8k":
            prompt_text = build_gsm8k_prompt(ex['question'])
        else:
            prompt_text = build_math500_prompt(ex['question'])

        # Tokenize (raw text — matches lm-eval's HFLM behavior, no chat template)
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]

        # Generate with PRM guidance
        t0 = time.time()
        output, info = prm_guided_topm_sample(
            model=gen_model,
            prm=prm,
            input_ids=input_ids,
            steps=args.num_steps,
            branch_every=args.branch_every,
            branch_factor=args.branch_factor,
            top_m=args.top_m,
            temperature=args.temperature,
            top_p=args.top_p,
            alg=args.alg,
            alg_temp=args.alg_temp,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        # Decode generated text
        gen_tokens = output[0, prompt_len:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Extract and compare answer
        if args.task == "gsm8k":
            pred_answer = normalize_answer(extract_gsm8k_answer(gen_text))
            gold_answer = normalize_answer(ex['gold_answer'])
            is_correct = (pred_answer == gold_answer)
        else:  # math500
            pred_answer = extract_math500_answer(gen_text)
            gold_answer = ex['gold_answer']
            is_correct = math_answers_equal(pred_answer, gold_answer)
        correct_count += int(is_correct)

        results.append({
            'global_idx': global_idx,
            'question': ex['question'][:100],
            'gold_answer': gold_answer,
            'pred_answer': pred_answer,
            'correct': is_correct,
            'time_s': round(elapsed, 1),
            'num_segments': info['num_segments'],
        })

        # Progress logging every 10 problems
        if (i + 1) % 10 == 0 or i == 0:
            acc_so_far = correct_count / (i + 1)
            avg_time = total_time / (i + 1)
            eta = avg_time * (len(shard_examples) - i - 1)
            logger.info(
                f"[Shard {args.shard_id}] {i+1}/{len(shard_examples)} "
                f"acc={acc_so_far:.1%} avg={avg_time:.1f}s/prob "
                f"ETA={eta/60:.0f}min "
                f"{'✓' if is_correct else '✗'} pred={pred_answer} gold={gold_answer}"
            )

        # Save partial results every 10 problems (crash protection)
        if (i + 1) % 10 == 0:
            partial_path = os.path.join(
                args.output_dir,
                f"prm_guided_topm{args.top_m}_{args.task}_K{args.branch_factor}_shard{args.shard_id}_partial.json"
            )
            partial_output = {
                'task': args.task, 'method': 'prm_guided',
                'shard_id': args.shard_id, 'num_shards': args.num_shards,
                'start_idx': start_idx, 'end_idx': end_idx,
                'completed': i + 1, 'num_total': len(shard_examples),
                'num_correct': correct_count,
                'accuracy': correct_count / (i + 1),
                'avg_time_per_problem': total_time / (i + 1),
                'per_example': results,
            }
            with open(partial_path, 'w') as f:
                json.dump(partial_output, f, indent=2)
            logger.info(f"Partial results saved ({i+1}/{len(shard_examples)})")

    # ------------------------------------------------------------------
    # Save shard results
    # ------------------------------------------------------------------
    accuracy = correct_count / max(len(shard_examples), 1)

    shard_output = {
        'task': args.task,
        'method': 'prm_guided',
        'shard_id': args.shard_id,
        'num_shards': args.num_shards,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'num_total': len(shard_examples),
        'num_correct': correct_count,
        'accuracy': accuracy,
        'avg_time_per_problem': total_time / max(len(shard_examples), 1),
        'total_time': total_time,
        'config': {
            'num_steps': args.num_steps,
            'temperature': args.temperature,
            'alg_temp': args.alg_temp,
            'branch_every': args.branch_every,
            'branch_factor': args.branch_factor,
            'top_m': args.top_m,
        },
        'per_example': results,
    }

    out_path = os.path.join(
        args.output_dir,
        f"prm_guided_topm{args.top_m}_{args.task}_K{args.branch_factor}_shard{args.shard_id}.json"
    )
    with open(out_path, 'w') as f:
        json.dump(shard_output, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Shard {args.shard_id} complete: {correct_count}/{len(shard_examples)} = {accuracy:.1%}")
    logger.info(f"Avg time: {total_time/max(len(shard_examples),1):.1f}s/problem")
    logger.info(f"Total time: {total_time/60:.1f}min")
    logger.info(f"Results saved to {out_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
