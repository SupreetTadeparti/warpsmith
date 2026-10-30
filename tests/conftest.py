import pytest

from warpsmith.compile import find_nvcc
from warpsmith.problem import load_problem


def gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return find_nvcc() is not None and torch.cuda.is_available()


@pytest.fixture(scope="session")
def matmul_gpu():
    """The matmul problem, skipping the test unless nvcc and a CUDA GPU are present."""
    if not gpu_available():
        pytest.skip("needs nvcc and a CUDA GPU")
    return load_problem("matmul")
