"""Frozen data, prompt, and grading protocol for the rebuttal MATH task.

The formal MATH extension has two non-negotiable boundaries:

* scorers are fit only on the pinned Hendrycks MATH *train* split; and
* all labels and MATH500 readouts use one pinned ``math-verify`` backend.

Imports of ``datasets`` and ``math_verify`` are intentionally lazy.  This lets
the overlap, extraction, manifest, and failure-policy code be unit-tested on a
CPU-only host before the dedicated grading environment is provisioned.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import signal
import threading
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any, Callable, Mapping, Protocol, Sequence

from .protocol import sha256_text


MATH_TRAIN_DATASET = "EleutherAI/hendrycks_math"
MATH_TRAIN_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH500_DATASET = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH_TRAIN_QUESTION_LIST_SHA256 = (
    "442bfc5188e86aea9d0816f7fa84299f0fea4e8fb6910ab06a7a7effb608e859"
)
MATH500_QUESTION_LIST_SHA256 = (
    "6afbf298074d6a57f9f5e1184e58f99522b622e97154a614ae4a5d2657c7be4a"
)

# The pinned Hendrycks source contains two known empty ``\boxed{}`` answers.
# Both multiple-choice solutions explicitly establish that the answer is zero.
# Keep the correction keyed by immutable source identity and expose it in every
# dataset manifest rather than silently guessing from arbitrary solution text.
MATH_TRAIN_GOLD_OVERRIDES = {
    "number_theory:train:661": "0",
    "number_theory:train:663": "0",
}

MATH_TRAIN_CONFIG_COUNTS = {
    "algebra": 1744,
    "counting_and_probability": 771,
    "geometry": 870,
    "intermediate_algebra": 1295,
    "number_theory": 869,
    "prealgebra": 1205,
    "precalculus": 746,
}
MATH_TRAIN_EXPECTED_PROBLEMS = sum(MATH_TRAIN_CONFIG_COUNTS.values())
MATH_TRAIN_UNIQUE_PROBLEM_HASHES = 7499
MATH500_EXPECTED_PROBLEMS = 500

MATH_VERIFY_VERSION = "0.9.0"
ANTLR_RUNTIME_VERSION = "4.11.0"
MATH_GRADER_TIMEOUT_SECONDS = 30.0
MATH_GRADER_PROTOCOL_ID = "math_verify_0.9.0_raw_output_v1"
MATH_PROMPT_PROTOCOL_ID = "dream_math_chat_boxed_v1"
MATH_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Justify your final answer by first "
    "explaining your step-by-step derivation or reasoning. Conclude by "
    "presenting the final answer in the format: \\boxed{ANSWER}."
)


class MathProtocolError(RuntimeError):
    """Raised when a formal MATH protocol invariant is violated."""


class MathVerifierDependencyError(MathProtocolError):
    """Raised when the grading environment is absent or not pinned."""


def normalize_problem_text(text: str) -> str:
    """Normalize only presentation-level whitespace for overlap auditing."""
    normalized = unicodedata.normalize("NFKC", str(text))
    return " ".join(normalized.split())


def problem_sha256(text: str) -> str:
    return sha256_text(normalize_problem_text(text))


def _extract_balanced_group(text: str, opening: int) -> str | None:
    if opening >= len(text) or text[opening] != "{":
        return None
    depth = 0
    for index in range(opening, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index].strip()
            if depth < 0:
                return None
    return None


def extract_last_boxed_answer(text: str | None) -> str | None:
    """Return the contents of the last ``\\boxed``/``\\fbox`` expression.

    Nested braces are handled explicitly.  The historical MATH source also
    contains a small number of ``\\boxed ANSWER`` forms, for which the answer
    is taken up to the next math delimiter or newline.
    """
    if not text:
        return None
    source = str(text)
    occurrences = [
        (source.rfind("\\boxed"), "\\boxed"),
        (source.rfind("\\fbox"), "\\fbox"),
    ]
    start, macro = max(occurrences, key=lambda item: item[0])
    if start < 0:
        return None
    cursor = start + len(macro)
    while cursor < len(source) and source[cursor].isspace():
        cursor += 1
    if cursor < len(source) and source[cursor] == "{":
        answer = _extract_balanced_group(source, cursor)
        return answer if answer else None

    tail = re.split(r"[$\r\n]", source[cursor:], maxsplit=1)[0].strip()
    return tail or None


def math_prompt_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MATH_SYSTEM_INSTRUCTION},
        {"role": "user", "content": str(question)},
    ]


def math_prompt_template() -> str:
    """Stable pre-tokenizer representation recorded in run manifests."""
    return json.dumps(
        math_prompt_messages("{QUESTION}"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def format_math_prompt(question: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        math_prompt_messages(question),
        tokenize=False,
        add_generation_prompt=True,
    )


def _question_list_sha256(examples: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(
        json.dumps(
            [
                {
                    "global_problem_id": str(item["global_problem_id"]),
                    "problem_sha256": str(item["problem_sha256"]),
                }
                for item in examples
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _default_dataset_loader(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def load_math_train_examples(
    *,
    revision: str = MATH_TRAIN_REVISION,
    formal_run: bool = False,
    dataset_loader: Callable[..., Any] | None = None,
    cache_dir: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and concatenate the seven pinned Hendrycks MATH train configs."""
    if formal_run and revision != MATH_TRAIN_REVISION:
        raise MathProtocolError(
            f"formal MATH train revision must be {MATH_TRAIN_REVISION}"
        )
    loader = dataset_loader or _default_dataset_loader
    examples: list[dict[str, Any]] = []
    fingerprints: dict[str, str | None] = {}
    observed_counts: dict[str, int] = {}
    for config, expected_count in MATH_TRAIN_CONFIG_COUNTS.items():
        kwargs: dict[str, Any] = {
            "split": "train",
            "revision": revision,
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        dataset = loader(MATH_TRAIN_DATASET, config, **kwargs)
        observed_counts[config] = len(dataset)
        fingerprints[config] = getattr(dataset, "_fingerprint", None)
        if formal_run and len(dataset) != expected_count:
            raise MathProtocolError(
                f"{config} contains {len(dataset)} train rows; "
                f"expected {expected_count}"
            )
        for row_index, item in enumerate(dataset):
            question = str(item["problem"])
            solution = str(item["solution"])
            global_problem_id = f"{config}:train:{row_index}"
            gold_answer = MATH_TRAIN_GOLD_OVERRIDES.get(global_problem_id)
            if gold_answer is None:
                gold_answer = extract_last_boxed_answer(solution)
            if gold_answer is None:
                raise MathProtocolError(
                    f"missing boxed gold for {config}:train:{row_index}"
                )
            examples.append(
                {
                    "global_problem_id": global_problem_id,
                    "problem_ordinal": len(examples),
                    "source_config": config,
                    "source_row_index": row_index,
                    "question": question,
                    "problem_sha256": problem_sha256(question),
                    "gold_answer": gold_answer,
                    "full_gold_solution": solution,
                    "level": item.get("level"),
                    "subject": item.get("type", config),
                }
            )
    if formal_run and len(examples) != MATH_TRAIN_EXPECTED_PROBLEMS:
        raise MathProtocolError(
            f"MATH train contains {len(examples)} rows; "
            f"expected {MATH_TRAIN_EXPECTED_PROBLEMS}"
        )
    question_list_sha256 = _question_list_sha256(examples)
    if formal_run and question_list_sha256 != MATH_TRAIN_QUESTION_LIST_SHA256:
        raise MathProtocolError(
            "formal MATH train question-list SHA-256 mismatch: "
            f"{question_list_sha256} != {MATH_TRAIN_QUESTION_LIST_SHA256}"
        )
    metadata_payload = {
        "dataset_name": MATH_TRAIN_DATASET,
        "revision": revision,
        "split": "train",
        "configs": list(MATH_TRAIN_CONFIG_COUNTS),
        "config_counts": observed_counts,
        "fingerprints": fingerprints,
        "num_problems": len(examples),
        "question_list_sha256": question_list_sha256,
        "expected_question_list_sha256": MATH_TRAIN_QUESTION_LIST_SHA256,
        "gold_answer_overrides": dict(MATH_TRAIN_GOLD_OVERRIDES),
    }
    return examples, metadata_payload


def load_math500_examples(
    *,
    revision: str = MATH500_REVISION,
    formal_run: bool = False,
    dataset_loader: Callable[..., Any] | None = None,
    cache_dir: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the pinned MATH500 evaluation set using its stable ``unique_id``."""
    if formal_run and revision != MATH500_REVISION:
        raise MathProtocolError(f"formal MATH500 revision must be {MATH500_REVISION}")
    loader = dataset_loader or _default_dataset_loader
    kwargs: dict[str, Any] = {
        "split": "test",
        "revision": revision,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    dataset = loader(MATH500_DATASET, **kwargs)
    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row_index, item in enumerate(dataset):
        unique_id = item.get("unique_id")
        if formal_run and not unique_id:
            raise MathProtocolError(f"formal MATH500 row {row_index} lacks unique_id")
        source_id = str(unique_id or f"test:{row_index}")
        if source_id in seen_ids:
            raise MathProtocolError(f"duplicate MATH500 unique_id: {source_id}")
        seen_ids.add(source_id)
        question = str(item["problem"])
        examples.append(
            {
                "global_problem_id": source_id,
                "problem_ordinal": row_index,
                "source_row_index": row_index,
                "question": question,
                "problem_sha256": problem_sha256(question),
                "gold_answer": str(item["answer"]),
                "full_gold_solution": str(item.get("solution", "")),
                "level": item.get("level"),
                "subject": item.get("subject"),
            }
        )
    if formal_run and len(examples) != MATH500_EXPECTED_PROBLEMS:
        raise MathProtocolError(
            f"MATH500 contains {len(examples)} rows; "
            f"expected {MATH500_EXPECTED_PROBLEMS}"
        )
    question_list_sha256 = _question_list_sha256(examples)
    if formal_run and question_list_sha256 != MATH500_QUESTION_LIST_SHA256:
        raise MathProtocolError(
            "formal MATH500 question-list SHA-256 mismatch: "
            f"{question_list_sha256} != {MATH500_QUESTION_LIST_SHA256}"
        )
    metadata_payload = {
        "dataset_name": MATH500_DATASET,
        "revision": revision,
        "split": "test",
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "num_problems": len(examples),
        "question_list_sha256": question_list_sha256,
        "expected_question_list_sha256": MATH500_QUESTION_LIST_SHA256,
    }
    return examples, metadata_payload


def audit_problem_overlap(
    train_examples: Sequence[Mapping[str, Any]],
    evaluation_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return an exact normalized-prompt overlap report."""
    train_by_hash: dict[str, list[str]] = {}
    for item in train_examples:
        train_by_hash.setdefault(str(item["problem_sha256"]), []).append(
            str(item["global_problem_id"])
        )
    evaluation_by_hash: dict[str, list[str]] = {}
    for item in evaluation_examples:
        evaluation_by_hash.setdefault(str(item["problem_sha256"]), []).append(
            str(item["global_problem_id"])
        )
    shared = sorted(set(train_by_hash) & set(evaluation_by_hash))
    overlaps = [
        {
            "problem_sha256": digest,
            "train_problem_ids": sorted(train_by_hash[digest]),
            "evaluation_problem_ids": sorted(evaluation_by_hash[digest]),
        }
        for digest in shared
    ]
    return {
        "normalization": "unicode_nfkc_then_whitespace_collapse_v1",
        "train_problem_count": len(train_examples),
        "evaluation_problem_count": len(evaluation_examples),
        "train_unique_problem_hashes": len(train_by_hash),
        "evaluation_unique_problem_hashes": len(evaluation_by_hash),
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "passed": not overlaps,
    }


def assert_no_problem_overlap(report: Mapping[str, Any]) -> None:
    if not bool(report.get("passed")):
        overlaps = report.get("overlaps", [])
        raise MathProtocolError(
            f"MATH train/MATH500 overlap detected: {len(overlaps)} normalized prompts"
        )


def installed_math_dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("math-verify", "antlr4-python3-runtime"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def validate_math_dependency_versions(
    versions: Mapping[str, str | None],
    *,
    formal_run: bool,
) -> None:
    if versions.get("math-verify") is None:
        raise MathVerifierDependencyError("math-verify is not installed")
    if formal_run:
        expected = {
            "math-verify": MATH_VERIFY_VERSION,
            "antlr4-python3-runtime": ANTLR_RUNTIME_VERSION,
        }
        mismatches = {
            key: {"expected": value, "actual": versions.get(key)}
            for key, value in expected.items()
            if versions.get(key) != value
        }
        if mismatches:
            raise MathVerifierDependencyError(
                "formal math grader dependency mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )


class MathVerifierBackend(Protocol):
    def grade(self, model_output: str, boxed_ground_truth: str) -> float: ...

    def extract(self, model_output: str) -> str | None: ...

    def equivalent(self, left_boxed: str, right_boxed: str) -> bool: ...


class _MathVerifyBackend:
    def __init__(self) -> None:
        from math_verify import parse, verify
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        self._parse = parse
        self._verify = verify
        self._latex_extraction = (LatexExtractionConfig(),)
        self._prediction_extraction = (
            ExprExtractionConfig(),
            LatexExtractionConfig(),
        )
        self._metric = math_metric(
            gold_extraction_target=self._latex_extraction,
            pred_extraction_target=self._prediction_extraction,
        )

    def grade(self, model_output: str, boxed_ground_truth: str) -> float:
        score, _ = self._metric([boxed_ground_truth], [model_output])
        return float(score)

    def extract(self, model_output: str) -> str | None:
        parsed = self._parse(
            model_output,
            extraction_config=list(self._prediction_extraction),
            fallback_mode="no_fallback",
            extraction_mode="any_match",
            raise_on_error=True,
        )
        if not parsed:
            return None
        return str(parsed[0])

    def equivalent(self, left_boxed: str, right_boxed: str) -> bool:
        parse_kwargs = {
            "extraction_config": list(self._latex_extraction),
            "fallback_mode": "no_fallback",
            "extraction_mode": "any_match",
            "raise_on_error": True,
        }
        left = self._parse(left_boxed, **parse_kwargs)
        right = self._parse(right_boxed, **parse_kwargs)
        if not left or not right:
            return False
        return bool(self._verify(left, right, raise_on_error=True))


@dataclass(frozen=True)
class MathGradeResult:
    score: float
    correct: bool
    status: str
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MathEquivalenceResult:
    equivalent: bool
    status: str
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MathExtractionResult:
    answer: str | None
    status: str
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_error_message(error: BaseException) -> str:
    return str(error).replace("\n", " ")[:300]


@contextmanager
def _wall_time_limit(seconds: float):
    """Interrupt one grader backend call with a real POSIX wall-clock timer."""
    timeout = float(seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("grader timeout must be finite and positive")
    if (
        not hasattr(signal, "setitimer")
        or not hasattr(signal, "ITIMER_REAL")
        or threading.current_thread() is not threading.main_thread()
    ):
        raise MathProtocolError(
            "the pinned grader timeout requires POSIX setitimer on the main thread"
        )

    def handle_timeout(_signum, _frame):
        raise TimeoutError(f"math grader exceeded {timeout:g} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay > 0 or previous_interval > 0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(previous_delay - elapsed, 1e-6),
                previous_interval,
            )


def _callable_sha256(callable_obj: Callable[..., Any]) -> str:
    return sha256_text(inspect.getsource(callable_obj))


class PinnedMathVerifier:
    """Pinned raw-output grader with explicit wrapper-level failure records."""

    def __init__(
        self,
        *,
        formal_run: bool,
        backend: MathVerifierBackend | None = None,
        dependency_versions: Mapping[str, str | None] | None = None,
        timeout_seconds: float = MATH_GRADER_TIMEOUT_SECONDS,
    ) -> None:
        versions = dict(
            dependency_versions
            if dependency_versions is not None
            else installed_math_dependency_versions()
        )
        validate_math_dependency_versions(versions, formal_run=formal_run)
        self.formal_run = bool(formal_run)
        self.dependency_versions = versions
        self.backend = backend if backend is not None else _MathVerifyBackend()
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

    @staticmethod
    def boxed(answer: str) -> str:
        return "\\boxed{" + str(answer) + "}"

    def grade_output(
        self,
        model_output: str,
        ground_truth: str,
    ) -> MathGradeResult:
        try:
            with _wall_time_limit(self.timeout_seconds):
                score = float(
                    self.backend.grade(
                        str(model_output),
                        self.boxed(str(ground_truth)),
                    )
                )
            if not math.isfinite(score):
                raise FloatingPointError(f"non-finite grader score: {score}")
            return MathGradeResult(
                score=score,
                correct=score > 0.5,
                status="ok",
            )
        except Exception as error:
            status = (
                "timeout"
                if isinstance(error, TimeoutError)
                or "timeout" in type(error).__name__.lower()
                else "error"
            )
            return MathGradeResult(
                score=0.0,
                correct=False,
                status=status,
                error_type=type(error).__name__,
                error_message=_safe_error_message(error),
            )

    def compare_answers(
        self,
        left_answer: str,
        right_answer: str,
    ) -> MathEquivalenceResult:
        try:
            with _wall_time_limit(self.timeout_seconds):
                equivalent = bool(
                    self.backend.equivalent(
                        self.boxed(str(left_answer)),
                        self.boxed(str(right_answer)),
                    )
                )
            return MathEquivalenceResult(
                equivalent=equivalent,
                status="ok",
            )
        except Exception as error:
            status = (
                "timeout"
                if isinstance(error, TimeoutError)
                or "timeout" in type(error).__name__.lower()
                else "error"
            )
            return MathEquivalenceResult(
                equivalent=False,
                status=status,
                error_type=type(error).__name__,
                error_message=_safe_error_message(error),
            )

    def extract_answer(self, model_output: str) -> MathExtractionResult:
        try:
            with _wall_time_limit(self.timeout_seconds):
                answer = self.backend.extract(str(model_output))
            return MathExtractionResult(
                answer=str(answer) if answer is not None else None,
                status="ok" if answer is not None else "invalid",
            )
        except Exception as error:
            status = (
                "timeout"
                if isinstance(error, TimeoutError)
                or "timeout" in type(error).__name__.lower()
                else "error"
            )
            return MathExtractionResult(
                answer=None,
                status=status,
                error_type=type(error).__name__,
                error_message=_safe_error_message(error),
            )

    def manifest(self) -> dict[str, Any]:
        backend_type = type(self.backend)
        return {
            "protocol_id": MATH_GRADER_PROTOCOL_ID,
            "formal_run": self.formal_run,
            "dependency_versions": dict(self.dependency_versions),
            "expected_dependency_versions": {
                "math-verify": MATH_VERIFY_VERSION,
                "antlr4-python3-runtime": ANTLR_RUNTIME_VERSION,
            },
            "input_contract": "raw_model_output_against_boxed_gold",
            "metric_parse_fallback_mode": ("math_verify_0.9.0_default_first_match"),
            "majority_extraction_fallback_mode": "no_fallback",
            "backend_timeout_seconds": self.timeout_seconds,
            "backend_timeout_implementation": "posix_setitimer_itimer_real_v1",
            "exception_policy": (
                "posix_wall_timeout_or_error_scores_zero_no_manual_string_rescue"
            ),
            "grade_callable_sha256": _callable_sha256(PinnedMathVerifier.grade_output),
            "equivalence_callable_sha256": _callable_sha256(
                PinnedMathVerifier.compare_answers
            ),
            "extraction_callable_sha256": _callable_sha256(
                PinnedMathVerifier.extract_answer
            ),
            "backend_implementation": (
                f"{backend_type.__module__}.{backend_type.__qualname__}"
            ),
            "backend_grade_callable_sha256": _callable_sha256(backend_type.grade),
            "backend_equivalence_callable_sha256": _callable_sha256(
                backend_type.equivalent
            ),
            "backend_extraction_callable_sha256": _callable_sha256(
                backend_type.extract
            ),
        }
