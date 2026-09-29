"""
PyTorch Datasets and collation for PRM training from trajectory data.

ORMDataset: uses only the final (fully denoised) output per trajectory.
DiffusionPRMDataset: uses all intermediate partially-masked snapshots.
"""

import torch
from torch.utils.data import Dataset

MASK_TOKEN_ID = 151666
PAD_TOKEN_ID = 151643  # same as EOS in Qwen2


class ORMDataset(Dataset):
    """Dataset for ORM training — final solutions only.

    Each sample: (input_ids, prompt_len, label).
    """

    def __init__(self, trajectories):
        self.samples = []
        for traj in trajectories:
            final_gen = traj['gen_snapshots'][-1]  # last snapshot: mask_ratio=0
            prompt = traj['prompt_ids']
            self.samples.append({
                'input_ids': torch.cat([prompt.long(), final_gen.long()]),
                'prompt_len': len(prompt),
                'label': 1.0 if traj['is_correct'] else 0.0,
                'problem_id': traj['problem_id'],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DiffusionPRMDataset(Dataset):
    """Dataset for DiffusionPRM — all intermediate snapshots.

    Each sample: (input_ids, prompt_len, mask_ratio, label).
    Expands trajectories × snapshots into flat samples.
    """

    def __init__(
        self,
        trajectories,
        min_mask_ratio=0.0,
        max_mask_ratio=1.0,
        label_field=None,
    ):
        self.samples = []
        for traj in trajectories:
            prompt = traj['prompt_ids']
            prompt_len = len(prompt)
            if label_field is None:
                label = 1.0 if traj['is_correct'] else 0.0
            else:
                if label_field not in traj:
                    raise KeyError(
                        f"Trajectory {traj.get('trajectory_id')} is missing "
                        f"label field {label_field!r}"
                    )
                label = float(traj[label_field])
                if not 0.0 <= label <= 1.0:
                    raise ValueError(
                        f"{label_field} must be in [0, 1], got {label}"
                    )

            for step, gen_snap, mask_ratio in zip(
                traj['snapshot_steps'],
                traj['gen_snapshots'],
                traj['mask_ratios'],
            ):
                if not (min_mask_ratio <= mask_ratio <= max_mask_ratio):
                    continue
                self.samples.append({
                    'input_ids': torch.cat([prompt.long(), gen_snap.long()]),
                    'prompt_len': prompt_len,
                    'mask_ratio': float(mask_ratio),
                    'label': label,
                    'step': step,
                    'problem_id': traj['problem_id'],
                    'trajectory_id': traj['trajectory_id'],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DiffusionPRMStateDataset(Dataset):
    """Dataset backed by an explicitly frozen list of state records.

    Rebuttal continuation-value experiments do not split trajectories inside
    the trainer.  They pass separate fitting and validation files whose
    records contain a state and both candidate targets::

        {
            "problem_id": ...,
            "state_id": ...,
            "prompt_ids": Tensor,
            "gen_ids": Tensor,
            "mask_ratio": float,
            "step": int,
            "binary_label": float,
            "fractional_label": float,
        }

    ``input_ids`` plus ``prompt_len`` is also accepted so aggregation tools can
    avoid storing prompt and generation tensors separately.
    """

    def __init__(self, records, label_field):
        if label_field not in {"binary_label", "fractional_label"}:
            raise ValueError(
                "label_field must be 'binary_label' or 'fractional_label'"
            )
        self.samples = []
        for record in records:
            if "input_ids" in record:
                input_ids = torch.as_tensor(record["input_ids"]).long()
                prompt_len = int(record["prompt_len"])
            else:
                prompt = torch.as_tensor(record["prompt_ids"]).long()
                generation = torch.as_tensor(record["gen_ids"]).long()
                input_ids = torch.cat([prompt, generation])
                prompt_len = len(prompt)

            label = float(record[label_field])
            if not 0.0 <= label <= 1.0:
                raise ValueError(
                    f"{label_field} must be in [0, 1], got {label}"
                )
            self.samples.append({
                "input_ids": input_ids,
                "prompt_len": prompt_len,
                "mask_ratio": float(record["mask_ratio"]),
                "label": label,
                "binary_label": float(record["binary_label"]),
                "fractional_label": float(record["fractional_label"]),
                "step": int(record.get("step", -1)),
                "problem_id": record["problem_id"],
                "state_id": record["state_id"],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_token_id=PAD_TOKEN_ID):
    """Dynamic-padding collate function.

    Pads all input_ids in the batch to the longest sequence.
    Returns attention_mask to prevent padding contamination in
    Dream's bidirectional attention.
    Works for both ORM and DiffusionPRM datasets.
    """
    max_len = max(len(s['input_ids']) for s in batch)

    input_ids_list = []
    attention_masks = []
    for s in batch:
        ids = s['input_ids']
        pad_len = max_len - len(ids)
        pad = torch.full((pad_len,), pad_token_id, dtype=ids.dtype)
        input_ids_list.append(torch.cat([ids, pad]))
        attention_masks.append(torch.cat([
            torch.ones(len(ids), dtype=torch.bool),
            torch.zeros(pad_len, dtype=torch.bool),
        ]))

    result = {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_masks),
        'prompt_lens': torch.tensor(
            [s['prompt_len'] for s in batch], dtype=torch.long
        ),
        'labels': torch.tensor(
            [s['label'] for s in batch], dtype=torch.float32
        ),
    }

    if 'mask_ratio' in batch[0]:
        result['mask_ratios'] = torch.tensor(
            [s['mask_ratio'] for s in batch], dtype=torch.float32
        )

    return result
