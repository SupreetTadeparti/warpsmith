# warpsmith

An LLM agent that tries to make CUDA kernels faster. It writes a kernel, the harness compiles it, checks that it's correct, benchmarks it, and sends the results back so it can try again.

Most of the work went into the checking part. LLMs will happily "optimize" a kernel by skipping work or caching results, so the harness tries hard to catch that.

## How it works

Each attempt goes through:

1. a source check (no cuBLAS, no extra streams, no file reads, etc.)
2. `nvcc`
3. correctness tests in a separate process, including hidden shapes the agent never sees
4. benchmarking, with a few sanity checks on the timing

The agent gets the current best kernel, a short history of past attempts, and feedback on the last one. A new best has to pass `compute-sanitizer` too.

## Catching cheats

`tests/redteam/` has a set of kernels that cheat in different ways, like writing zeros, caching results by pointer, using fp16 or launching on a different stream. The tests make sure every one of them gets rejected.

## First results

matmul, 2048x2048, Tesla T4, gpt-oss-120b, 10 iterations:

| kernel | time |
|---|---|
| naive | 143.7 ms |
| agent's best | 25.0 ms |
| cuBLAS | 4.0 ms |

The agent got stuck because it only sees timings, not why a kernel is slow. Profiler feedback is next.

## Running it

I don't have an NVIDIA GPU, so I run everything on Colab with `notebooks/run_colab.ipynb`.

```bash
pip install -e ".[dev]"
python -m warpsmith eval matmul examples/matmul/tiled.cu
python -m warpsmith agent matmul --iterations 10
```

The agent needs `NAVIGATOR_API_KEY` set (UF NaviGator, OpenAI-compatible API).

## TODO

- profiler (ncu) feedback
- reduction and softmax problems
- run reports
