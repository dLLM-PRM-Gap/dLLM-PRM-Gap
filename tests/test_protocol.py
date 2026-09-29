import hashlib
import json

from src.prm.protocol import STABLE_SEED_SPEC, derive_row_seeds, derive_seed


def test_derive_seed_is_stable_and_semantic():
    expected = derive_seed(42, 17, 3, "smc_stage1")
    canonical = json.dumps(
        (42, 17, 3, "smc_stage1"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    protocol_value = int.from_bytes(hashlib.sha256(canonical).digest()[:8], "big")
    protocol_value &= (1 << 63) - 1
    assert STABLE_SEED_SPEC.startswith("sha256-first8")
    assert expected == protocol_value
    assert expected == derive_seed(42, 17, 3, "smc_stage1")
    assert 0 <= expected < 2**63
    assert expected != derive_seed(42, 17, 4, "smc_stage1")
    assert expected != derive_seed(42, 17, 3, "smc_stage2")
    assert expected != derive_seed(43, 17, 3, "smc_stage1")


def test_row_seeds_do_not_depend_on_traversal_order():
    forward = derive_row_seeds(42, 9, [0, 1, 2], "rollout")
    reverse = derive_row_seeds(42, 9, [2, 1, 0], "rollout")
    assert forward == list(reversed(reverse))
