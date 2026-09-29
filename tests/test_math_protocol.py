import time

import pytest

from src.prm.math_protocol import (
    ANTLR_RUNTIME_VERSION,
    MATH500_REVISION,
    MATH500_QUESTION_LIST_SHA256,
    MATH_TRAIN_GOLD_OVERRIDES,
    MATH_TRAIN_QUESTION_LIST_SHA256,
    MATH_TRAIN_REVISION,
    MATH_VERIFY_VERSION,
    MathProtocolError,
    MathVerifierDependencyError,
    PinnedMathVerifier,
    assert_no_problem_overlap,
    audit_problem_overlap,
    extract_last_boxed_answer,
    load_math500_examples,
    load_math_train_examples,
    math_prompt_template,
    normalize_problem_text,
    problem_sha256,
    validate_math_dependency_versions,
)


class FakeBackend:
    def grade(self, model_output, boxed_ground_truth):
        assert boxed_ground_truth.startswith("\\boxed{")
        if model_output == "raise":
            raise ValueError("synthetic parse error")
        if model_output == "timeout":
            raise TimeoutError("synthetic timeout")
        return 1.0 if model_output == "correct raw output" else 0.0

    def equivalent(self, left_boxed, right_boxed):
        if "bad" in left_boxed or "bad" in right_boxed:
            raise ValueError("synthetic equivalence error")

        def normalize(value):
            return value.replace("\\frac{1}{2}", "0.5")

        return normalize(left_boxed) == normalize(right_boxed)

    def extract(self, model_output):
        if model_output == "extract-error":
            raise ValueError("synthetic extraction error")
        if model_output.startswith("\\boxed{") and model_output.endswith("}"):
            return model_output[len("\\boxed{") : -1]
        return None


PINNED_VERSIONS = {
    "math-verify": MATH_VERIFY_VERSION,
    "antlr4-python3-runtime": ANTLR_RUNTIME_VERSION,
}


class FakeDataset(list):
    _fingerprint = "fake-fingerprint"


def test_nested_last_boxed_gold_and_space_form():
    solution = "first \\boxed{wrong}; then \\boxed{\\left(3, \\frac{\\pi}{2}\\right)}."
    assert extract_last_boxed_answer(solution) == ("\\left(3, \\frac{\\pi}{2}\\right)")
    assert extract_last_boxed_answer("answer $\\boxed 17$ done") == "17"
    assert extract_last_boxed_answer("answer \\fbox{\\frac{2}{3}}") == ("\\frac{2}{3}")
    assert extract_last_boxed_answer("no final answer") is None
    assert extract_last_boxed_answer("\\boxed{unclosed") is None


def test_problem_overlap_uses_nfkc_and_whitespace_normalization():
    train = [
        {
            "global_problem_id": "train:0",
            "problem_sha256": problem_sha256("Find  x + 1"),
        }
    ]
    evaluation = [
        {
            "global_problem_id": "test:0",
            "problem_sha256": problem_sha256("Find x + 1"),
        }
    ]
    assert normalize_problem_text("Ａ  B\nC") == "A B C"
    report = audit_problem_overlap(train, evaluation)
    assert report["overlap_count"] == 1
    assert not report["passed"]
    with pytest.raises(MathProtocolError, match="overlap"):
        assert_no_problem_overlap(report)

    clean = audit_problem_overlap(
        train,
        [
            {
                "global_problem_id": "test:1",
                "problem_sha256": problem_sha256("Different"),
            }
        ],
    )
    assert clean["passed"]
    assert_no_problem_overlap(clean)


def test_dataset_loaders_forward_pinned_revisions_and_stable_ids():
    calls = []

    def loader(path, *args, **kwargs):
        calls.append((path, args, kwargs))
        if path.endswith("hendrycks_math"):
            config = args[0]
            return FakeDataset(
                [
                    {
                        "problem": f"problem {config}",
                        "solution": "work \\boxed{7}",
                        "level": "Level 1",
                        "type": "Algebra",
                    }
                ]
            )
        return FakeDataset(
            [
                {
                    "problem": "evaluation problem",
                    "solution": "work",
                    "answer": "9",
                    "unique_id": "test/algebra/1.json",
                    "level": 1,
                    "subject": "Algebra",
                }
            ]
        )

    train, train_meta = load_math_train_examples(
        dataset_loader=loader,
        revision=MATH_TRAIN_REVISION,
    )
    evaluation, eval_meta = load_math500_examples(
        dataset_loader=loader,
        revision=MATH500_REVISION,
    )
    assert len(train) == 7
    assert train[0]["global_problem_id"] == "algebra:train:0"
    assert train[0]["gold_answer"] == "7"
    assert evaluation[0]["global_problem_id"] == "test/algebra/1.json"
    assert train_meta["revision"] == MATH_TRAIN_REVISION
    assert eval_meta["revision"] == MATH500_REVISION
    assert (
        train_meta["expected_question_list_sha256"] == MATH_TRAIN_QUESTION_LIST_SHA256
    )
    assert eval_meta["expected_question_list_sha256"] == MATH500_QUESTION_LIST_SHA256
    assert train_meta["gold_answer_overrides"] == MATH_TRAIN_GOLD_OVERRIDES
    assert all(call[2]["revision"] for call in calls)

    with pytest.raises(MathProtocolError, match="revision"):
        load_math500_examples(
            dataset_loader=loader,
            revision="moving-main",
            formal_run=True,
        )


def test_train_loader_applies_identity_keyed_empty_box_overrides():
    def loader(path, config, **kwargs):
        del path, kwargs
        count = 664 if config == "number_theory" else 1
        rows = [
            {
                "problem": f"{config} problem {index}",
                "solution": "work \\boxed{7}",
                "level": "Level 1",
                "type": config,
            }
            for index in range(count)
        ]
        if config == "number_theory":
            rows[661]["solution"] = "multiple choice conclusion \\boxed{}"
            rows[663]["solution"] = "multiple choice conclusion \\boxed{}"
        return FakeDataset(rows)

    examples, metadata = load_math_train_examples(dataset_loader=loader)
    by_id = {item["global_problem_id"]: item for item in examples}
    assert by_id["number_theory:train:661"]["gold_answer"] == "0"
    assert by_id["number_theory:train:663"]["gold_answer"] == "0"
    assert metadata["gold_answer_overrides"] == MATH_TRAIN_GOLD_OVERRIDES


def test_formal_math500_requires_source_unique_id():
    def loader(*args, **kwargs):
        del args, kwargs
        return FakeDataset(
            [
                {
                    "problem": "evaluation problem",
                    "solution": "work",
                    "answer": "9",
                    "level": 1,
                    "subject": "Algebra",
                }
            ]
        )

    with pytest.raises(MathProtocolError, match="lacks unique_id"):
        load_math500_examples(dataset_loader=loader, formal_run=True)


def test_pinned_verifier_records_failures_without_string_fallback():
    verifier = PinnedMathVerifier(
        formal_run=True,
        backend=FakeBackend(),
        dependency_versions=PINNED_VERSIONS,
    )
    correct = verifier.grade_output("correct raw output", "17")
    assert correct.correct and correct.status == "ok"

    # Even if the literal text resembles the gold, the strict backend result
    # is authoritative; there is no exact-string rescue path.
    wrong = verifier.grade_output("17", "17")
    assert not wrong.correct and wrong.status == "ok"

    error = verifier.grade_output("raise", "17")
    assert not error.correct
    assert error.status == "error"
    assert error.error_type == "ValueError"

    timeout = verifier.grade_output("timeout", "17")
    assert not timeout.correct
    assert timeout.status == "timeout"

    equivalent = verifier.compare_answers("\\frac{1}{2}", "0.5")
    assert equivalent.equivalent
    failed_equivalence = verifier.compare_answers("bad", "bad")
    assert not failed_equivalence.equivalent
    assert failed_equivalence.status == "error"

    extracted = verifier.extract_answer("\\boxed{17}")
    assert extracted.answer == "17"
    assert extracted.status == "ok"
    invalid = verifier.extract_answer("no answer")
    assert invalid.answer is None
    assert invalid.status == "invalid"
    extraction_error = verifier.extract_answer("extract-error")
    assert extraction_error.status == "error"

    manifest = verifier.manifest()
    assert manifest["dependency_versions"] == PINNED_VERSIONS
    assert manifest["exception_policy"].endswith("no_manual_string_rescue")
    assert manifest["backend_timeout_implementation"] == (
        "posix_setitimer_itimer_real_v1"
    )
    assert manifest["majority_extraction_fallback_mode"] == "no_fallback"
    assert len(manifest["grade_callable_sha256"]) == 64
    assert manifest["backend_implementation"].endswith(".FakeBackend")
    assert len(manifest["backend_extraction_callable_sha256"]) == 64


def test_pinned_verifier_enforces_real_wall_timeout():
    class SlowBackend(FakeBackend):
        def grade(self, model_output, boxed_ground_truth):
            del model_output, boxed_ground_truth
            time.sleep(0.25)
            return 1.0

    verifier = PinnedMathVerifier(
        formal_run=True,
        backend=SlowBackend(),
        dependency_versions=PINNED_VERSIONS,
        timeout_seconds=0.02,
    )
    started = time.monotonic()
    result = verifier.grade_output("anything", "17")
    assert time.monotonic() - started < 0.2
    assert not result.correct
    assert result.status == "timeout"
    assert result.error_type == "TimeoutError"


def test_formal_dependency_version_gate_is_exact():
    with pytest.raises(MathVerifierDependencyError, match="not installed"):
        validate_math_dependency_versions(
            {
                "math-verify": None,
                "antlr4-python3-runtime": ANTLR_RUNTIME_VERSION,
            },
            formal_run=False,
        )
    with pytest.raises(MathVerifierDependencyError, match="mismatch"):
        validate_math_dependency_versions(
            {
                "math-verify": MATH_VERIFY_VERSION,
                "antlr4-python3-runtime": "4.9.3",
            },
            formal_run=True,
        )
    validate_math_dependency_versions(PINNED_VERSIONS, formal_run=True)


def test_prompt_template_is_stable_and_contains_boxed_contract():
    template = math_prompt_template()
    assert "{QUESTION}" in template
    assert "\\\\boxed{ANSWER}" in template
