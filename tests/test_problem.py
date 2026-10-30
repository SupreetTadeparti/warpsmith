import pytest
import yaml

from warpsmith.problem import load_problem


def test_load_matmul():
    p = load_problem("matmul")
    assert [a.name for a in p.inputs] == ["A", "B"]
    assert [a.name for a in p.outputs] == ["C"]
    assert p.args[0].resolve_shape({"M": 3, "N": 4, "K": 5}) == (3, 5)
    assert p.hidden_shapes, "matmul should have hidden shapes"
    assert "launch" in p.signature


def test_hidden_shapes_include_awkward_sizes():
    p = load_problem("matmul")
    assert any(any(v % 32 for v in dims.values()) for dims in p.hidden_shapes)


def test_shape_missing_dim_is_rejected(tmp_path):
    spec = yaml.safe_load(load_problem("matmul").root.joinpath("spec.yaml").read_text())
    spec["shapes"]["visible"].append({"M": 4, "N": 4})
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="missing dims"):
        load_problem(tmp_path)
