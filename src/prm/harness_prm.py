"""
lm-eval Harness for PRM-Guided Denoising.

Integrates prm_guided_sample() with the BaselineDreamHarness interface
so PRM-guided generation can be evaluated with the same pipeline as
the baseline methods.

Usage:
    from prm.harness_prm import PRMGuidedDreamHarness

    harness = PRMGuidedDreamHarness(
        pretrained=model,
        tokenizer=tokenizer,
        prm=diffusion_prm,
        branch_every=16,
        branch_factor=4,
    )
"""

import torch
from evaluation.harness_baselines import BaselineDreamHarness


class PRMGuidedDreamHarness(BaselineDreamHarness):
    """PRM-Guided Dream-7B: Periodic Beam Search with process reward model."""

    def __init__(self, pretrained, tokenizer, prm, **args):
        super().__init__(
            pretrained, tokenizer, method_name="prm_guided", **args
        )
        self.prm = prm
        self.branch_every = args.get("branch_every", 16)
        self.branch_factor = args.get("branch_factor", 4)
        self.keep_top = args.get("keep_top", 1)

    def _do_generate(self, context, max_length):
        from prm.prm_guided_denoise import prm_guided_sample

        output, info = prm_guided_sample(
            model=self.model,
            prm=self.prm,
            input_ids=context,
            steps=self.num_steps,
            branch_every=self.branch_every,
            branch_factor=self.branch_factor,
            temperature=self.gen_temperature,
            top_p=self.gen_top_p,
            alg=self.gen_alg,
            alg_temp=self.gen_alg_temp,
            mask_token_id=self.mask_token_id,
            gen_length=self.gen_length,
        )

        # Log PRM scoring info
        self.log_profile({
            'num_segments': info['num_segments'],
            'total_forward_passes': info['total_forward_passes'],
        })

        if output.dim() == 1:
            return output.unsqueeze(0)
        return output
