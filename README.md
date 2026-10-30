# Warpsmith

An LLM agent that optimizes CUDA kernels in a loop (propose, compile, verify, benchmark, repeat), evaluated by a harness that's built to be hard to cheat.

> A *warp* is CUDA's 32-thread unit, and a smith reworks metal until it's right.

## Status

- [x] **Milestone 1:** evaluation harness with no LLM: static checks, nvcc compile, isolated worker, verification, benchmarking (matmul)
- [x] **Milestone 2:** red-team suite of cheating kernels the verifier must reject
- [ ] **Milestone 3:** agent loop (gpt-oss-120b via UF NaviGator)
- [ ] **Milestone 4:** reduction and softmax problems
- [ ] **Milestone 5:** run reports

## How a candidate is evaluated

```
source --> static check --> nvcc --> worker subprocess: verify --> benchmark
```

1. **Static check** (`warpsmith/static_check.py`): rejects calls to cuBLAS, cuDNN, Thrust and device-wide CUB; creating streams, events or graphs; synchronizing; host copies and allocations; file, process and environment access; GPU timer reads; and headers outside an allowlist.
2. **Compile** (`warpsmith/compile.py`): `nvcc -O3 -shared -Xptxas -v` builds a shared library. The harness parses ptxas output for per-kernel registers, shared memory and spills.
3. **Verify** (`warpsmith/verify.py`), in a subprocess so crashes and hangs are contained:
   - **Hidden shapes:** the agent never sees them, and they include sizes that aren't multiples of a tile.
   - **Fresh inputs:** a new random seed every run, drawn from several input distributions.
   - **Sentinel-filled outputs:** outputs start as NaN, so skipped work shows up.
   - **Guard regions:** canary bytes around every buffer catch out-of-bounds writes.
   - **Read-only inputs:** input snapshots catch kernels that modify their inputs.
   - **Same-pointer rerun:** every case runs a second time with new data in the same buffers, which catches cached results.
   - **Tolerance:** results are compared against a float64 reference, with the tolerance scaled to the magnitude of the summed terms.
4. **Benchmark** (`warpsmith/bench.py`):
   - CUDA-event timing with the L2 cache flushed and input sets rotated; median and IQR are reported.
   - The outputs produced while benchmarking are checked for correctness again.
   - **Roofline check:** results faster than the GPU's physical limit are flagged.
   - **Clock cross-check:** wall-clock time is compared with CUDA-event time to catch work that escapes the timed stream.

With `--sanitize`, a candidate that passes verification is checked again under `compute-sanitizer` memcheck, with the torch caching allocator disabled. This catches stray writes that land far outside the guard regions.

## Red-team suite

`tests/redteam/matmul/` holds cheating kernels. The harness counts as correct only if it rejects every one of them, for the expected reason. Each file's header comment states the defense that must catch it, and `tests/test_redteam.py` checks this.

| Cheat | Caught by |
|---|---|
| `zeros`: writes a constant | reference comparison (diagnosed as "all zeros") |
| `tail_skip`: computes only full 32×32 tiles; correct on every visible shape | hidden odd shapes, NaN-prefilled outputs |
| `no_bounds_check`: rounds the grid up with no bounds check | hidden shapes, guard regions |
| `pointer_cache`: replays the cached result when the pointers repeat | same-pointer rerun with new data (diagnosed as "stale") |
| `dims_cache`: caches results by shape | fresh random inputs on every case |
| `mutates_input`: uses A as scratch space | input snapshots |
| `half_precision`: silently computes in fp16 | magnitude-scaled tolerance, `wide_range` inputs |
| `skip_half_k`: samples every other k | element-wise tolerance, `small_int` inputs |
| `bench_shape_special`: cheaper wrong kernel at the benchmark shape | benchmark outputs are verified |
| `call_count`: stops launching after verification | roofline check, outputs poisoned and relaunched after timing |
| `legacy_stream`: launches outside the timed stream | wall-clock vs CUDA-event cross-check, roofline |
| `oob_far`: writes 256 MiB past the end of C | compute-sanitizer memcheck |
| `cublas_call`, `read_answers`, `own_stream`, `timer_probe`, `harness_override` | static check |

## Running

The development machine has no NVIDIA GPU, so anything that touches the GPU runs on Colab or Kaggle. Use `notebooks/run_colab.ipynb`.

```bash
pip install -e ".[dev]"
python -m warpsmith info                                    # GPU and peak specs
python -m warpsmith eval matmul examples/matmul/tiled.cu    # full pipeline + comparison to naive & cuBLAS
python -m warpsmith baseline matmul                         # cuBLAS (torch.matmul) timing
python -m pytest -q                                         # GPU tests skip without nvcc
```

## Adding a problem

Create `problems/<name>/` with three files:
- `spec.yaml`: the launcher signature, args, tolerance, and the visible, hidden and benchmark shapes.
- `naive.cu`: the starting kernel.
- `reference.py`: defines `make_inputs`, `reference` (float64), `baseline`, `flops`, `bytes_moved`, and optionally `tolerance_scale`.

See `problems/matmul/` for an example.
