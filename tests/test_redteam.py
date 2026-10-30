"""Red-team suite: hand-written cheating kernels the harness must reject.

Each tests/redteam/<problem>/*.cu file declares in its header comment how it must fail:
    // expect-stage: verify|bench      stage(s) the evaluation may stop at
    // expect-check: wrong_result      first verify failure's check (alternatives with |)
    // expect-hidden: true             that failure must be on a hidden shape
    // expect-diagnosis: stale         substring of one of the failure's diagnosis notes
    // expect-error: regex             searched in the evaluation error message
    // expect-rule: cublas|io          static rule(s), at least one must fire
    // sanitize: true                  evaluate with compute-sanitizer enabled
Static-stage cheats are checked without a GPU; the rest need nvcc + CUDA.
"""

import re
from pathlib import Path

import pytest

from warpsmith.evaluate import evaluate
from warpsmith.problem import load_problem
from warpsmith.static_check import check_source

REDTEAM = Path(__file__).resolve().parent / "redteam"
EXPECT_RE = re.compile(r"^//\s*(expect-[\w-]+|sanitize):\s*(.+?)\s*$", re.MULTILINE)


def expectations(path: Path) -> dict[str, str]:
    return {k.removeprefix("expect-"): v for k, v in EXPECT_RE.findall(path.read_text())}


ALL = sorted(REDTEAM.glob("*/*.cu"))
STATIC = [p for p in ALL if expectations(p).get("stage") == "static"]
DYNAMIC = [p for p in ALL if expectations(p).get("stage") != "static"]


def case_id(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def test_suite_is_well_formed():
    assert len(ALL) >= 15
    for path in ALL:
        exp = expectations(path)
        assert "stage" in exp, f"{case_id(path)} has no expect-stage"
        load_problem(path.parent.name)


@pytest.mark.parametrize("path", DYNAMIC, ids=case_id)
def test_dynamic_cheats_pass_static_checks(path):
    """Otherwise they'd be testing the static layer, not the dynamic defense they target."""
    problem = load_problem(path.parent.name)
    assert check_source(path.read_text(), problem.allowed_apis) == []


@pytest.mark.parametrize("path", STATIC, ids=case_id)
def test_static_cheats_rejected(path):
    exp = expectations(path)
    problem = load_problem(path.parent.name)
    fired = {v.rule for v in check_source(path.read_text(), problem.allowed_apis)}
    assert fired & set(exp["rule"].split("|")), f"rules fired: {sorted(fired)}"


@pytest.mark.parametrize("path", DYNAMIC, ids=case_id)
def test_dynamic_cheats_rejected(path, matmul_gpu):
    exp = expectations(path)
    problem = load_problem(path.parent.name)
    r = evaluate(problem, path.read_text(), sanitize=exp.get("sanitize") == "true")

    assert not r.passed, f"cheat was accepted: {r.bench}"
    assert r.stage in exp["stage"].split("|"), f"stopped at {r.stage}: {r.error}"

    failure = r.verify["failures"][0] if r.stage == "verify" and r.verify and r.verify["failures"] else None
    if "check" in exp:
        assert failure is not None, r.error
        assert failure["check"] in exp["check"].split("|"), failure
    if exp.get("hidden") == "true":
        assert failure["hidden"], f"caught on a visible shape, expected only hidden shapes to catch it: {failure}"
    if "diagnosis" in exp:
        assert any(exp["diagnosis"] in note for note in failure.get("diagnosis", [])), failure
    if "error" in exp:
        assert re.search(exp["error"], r.error), r.error
