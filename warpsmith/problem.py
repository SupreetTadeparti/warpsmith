"""Problem definitions: a spec.yaml (contract, shapes, tolerances) plus reference.py.

reference.py must define:
    make_inputs(dims, gen, dist) -> dict[name, cuda tensor]
    reference(inputs, dims)      -> dict[output name, float64 tensor]
    baseline(inputs, outputs, dims)  # fast library implementation, writes into outputs
    flops(dims) -> int
    bytes_moved(dims) -> int     # compulsory DRAM traffic, used for the roofline check
and may define:
    tolerance_scale(inputs, dims) -> dict[output name, float64 tensor]
        Per-element magnitude the error is measured against (defaults to |reference|).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import yaml

PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"

ARG_KINDS = ("input", "output", "dim")


@dataclass(frozen=True)
class Arg:
    name: str
    kind: str
    dtype: str | None = None
    shape: tuple[str | int, ...] = ()

    def resolve_shape(self, dims: dict[str, int]) -> tuple[int, ...]:
        return tuple(d if isinstance(d, int) else dims[d] for d in self.shape)


@dataclass
class Problem:
    name: str
    root: Path
    description: str
    signature: str
    args: list[Arg]
    rtol: float
    atol: float
    visible_shapes: list[dict[str, int]]
    hidden_shapes: list[dict[str, int]]
    bench_shape: dict[str, int]
    distributions: list[str]
    allowed_apis: list[str] = field(default_factory=list)

    @property
    def tensor_args(self) -> list[Arg]:
        return [a for a in self.args if a.kind != "dim"]

    @property
    def inputs(self) -> list[Arg]:
        return [a for a in self.args if a.kind == "input"]

    @property
    def outputs(self) -> list[Arg]:
        return [a for a in self.args if a.kind == "output"]

    @property
    def naive_source(self) -> str:
        return (self.root / "naive.cu").read_text()

    def reference_module(self) -> ModuleType:
        path = self.root / "reference.py"
        spec = importlib.util.spec_from_file_location(f"warpsmith_problem_{self.name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def load_problem(name_or_path: str | Path) -> Problem:
    root = Path(name_or_path)
    if not (root / "spec.yaml").exists():
        root = PROBLEMS_DIR / str(name_or_path)
    spec_path = root / "spec.yaml"
    if not spec_path.exists():
        raise FileNotFoundError(f"no problem named {name_or_path!r} (looked for {spec_path})")
    spec = yaml.safe_load(spec_path.read_text())

    args = [
        Arg(a["name"], a["kind"], a.get("dtype"), tuple(a.get("shape", ())))
        for a in spec["args"]
    ]
    problem = Problem(
        name=spec["name"],
        root=root.resolve(),
        description=spec["description"].strip(),
        signature=spec["signature"].strip(),
        args=args,
        rtol=float(spec["tolerance"]["rtol"]),
        atol=float(spec["tolerance"]["atol"]),
        visible_shapes=spec["shapes"]["visible"],
        hidden_shapes=spec["shapes"].get("hidden", []),
        bench_shape=spec["shapes"]["benchmark"],
        distributions=spec.get("input_distributions", ["uniform"]),
        allowed_apis=spec.get("static", {}).get("allow", []),
    )
    _validate(problem)
    return problem


def _validate(p: Problem) -> None:
    dim_names = {a.name for a in p.args if a.kind == "dim"}
    for a in p.args:
        if a.kind not in ARG_KINDS:
            raise ValueError(f"{p.name}: arg {a.name} has unknown kind {a.kind!r}")
        if a.kind != "dim":
            if a.dtype is None or not a.shape:
                raise ValueError(f"{p.name}: tensor arg {a.name} needs dtype and shape")
            unknown = {d for d in a.shape if isinstance(d, str)} - dim_names
            if unknown:
                raise ValueError(f"{p.name}: arg {a.name} uses undeclared dims {sorted(unknown)}")
    for dims in p.visible_shapes + p.hidden_shapes + [p.bench_shape]:
        missing = dim_names - set(dims)
        if missing:
            raise ValueError(f"{p.name}: shape {dims} is missing dims {sorted(missing)}")
    if not p.outputs:
        raise ValueError(f"{p.name}: needs at least one output arg")
