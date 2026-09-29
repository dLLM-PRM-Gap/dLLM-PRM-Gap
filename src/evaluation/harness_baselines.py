"""
Evaluation harness classes for baseline methods on Dream-7B.

Extends CoDD's DreamEvalHarness to support baseline generation methods,
all using the same lm-evaluation-harness interface for fair comparison.
"""

import time
import math
import torch
import numpy as np
from lm_eval.models.huggingface import HFLM
from lm_eval.api.instance import Instance


class BaselineDreamHarness(HFLM):
    """Base harness for Dream-7B baseline evaluation.

    Subclasses override `_do_generate()` with their specific generation method.
    Profile logging (throughput, latency) is handled uniformly here.
    """

    def __init__(self, pretrained, tokenizer, method_name="baseline", **args):
        super().__init__(pretrained=pretrained, tokenizer=tokenizer, **args)
        self.model_alias = method_name
        self.mask_token_id = args.get("mask_token_id", 151666)
        self.num_steps = args.get("num_steps", 512)
        self.gen_length = args.get("gen_length", 512)
        self.gen_temperature = args.get("gen_temperature", 0.2)
        self.gen_top_p = args.get("gen_top_p", 0.95)
        self.gen_alg = args.get("gen_alg", "entropy")
        self.gen_alg_temp = args.get("gen_alg_temp", 0.0)
        self.max_gen_toks_value = args.get("max_gen_toks", 256)
        self.profile = {}

    @property
    def max_gen_toks(self) -> int:
        return self.max_gen_toks_value

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("Baseline methods are generative-only")

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError("Baseline methods are generative-only")

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        start = time.time()
        outputs = self._do_generate(context, max_length)
        end = time.time()

        # Count generated tokens (stop at EOS)
        num_tokens_generated = -context.shape[-1]
        for value in outputs[0]:
            if value == self.tokenizer.eos_token_id:
                break
            num_tokens_generated += 1

        self.log_profile({
            "num_tokens_generated": num_tokens_generated,
            "total_time": end - start,
        })

        return outputs

    def _do_generate(self, context, max_length):
        """Override in subclasses to implement specific generation method."""
        raise NotImplementedError

    def log_profile(self, profile):
        for k, v in profile.items():
            if k not in self.profile:
                self.profile[k] = []
            self.profile[k].append(v)

    def get_profile(self):
        num_tokens_generated = np.array(self.profile["num_tokens_generated"])
        total_times = np.array(self.profile["total_time"])
        throughputs = num_tokens_generated / total_times

        return {
            "throughput_mean": throughputs.mean(),
            "throughput_stderr": throughputs.std(ddof=1) / math.sqrt(len(throughputs)) if len(throughputs) > 1 else 0,
            "total_time_mean": total_times.mean(),
            "total_time_stderr": total_times.std(ddof=1) / math.sqrt(len(total_times)) if len(total_times) > 1 else 0,
            "num_tokens_generated_mean": num_tokens_generated.mean(),
            "num_tokens_generated_stderr": num_tokens_generated.std(ddof=1) / math.sqrt(len(num_tokens_generated)) if len(num_tokens_generated) > 1 else 0,
        }


class VanillaDreamHarness(BaselineDreamHarness):
    """Vanilla Dream-7B: standard factorized decoding (no dependency correction)."""

    def __init__(self, pretrained, tokenizer, **args):
        super().__init__(pretrained, tokenizer, method_name="vanilla", **args)

    def _do_generate(self, context, max_length):
        outputs = self.model.diffusion_generate(
            context,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            steps=self.num_steps,
            temperature=self.gen_temperature,
            top_p=self.gen_top_p,
            alg=self.gen_alg,
            alg_temp=self.gen_alg_temp,
        )
        # Ensure consistent output shape [B, seq_len]
        if hasattr(outputs, 'sequences'):
            return outputs.sequences
        if outputs.dim() == 1:
            return outputs.unsqueeze(0)
        return outputs


