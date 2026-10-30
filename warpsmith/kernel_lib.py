"""Load a compiled candidate and call its extern "C" launch() with torch tensors."""

from __future__ import annotations

import ctypes
from pathlib import Path

from .problem import Problem


class ContractError(RuntimeError):
    """The shared library does not expose the expected launch() entry point."""


class KernelLib:
    def __init__(self, so_path: str | Path, problem: Problem):
        self.problem = problem
        self.lib = ctypes.CDLL(str(so_path))
        try:
            self.fn = self.lib.launch
        except AttributeError as e:
            raise ContractError('shared library has no extern "C" symbol `launch`') from e
        self.fn.restype = None
        self.fn.argtypes = [
            ctypes.c_int if a.kind == "dim" else ctypes.c_void_p for a in problem.args
        ] + [ctypes.c_void_p]
        self.lib.ws_last_error.restype = ctypes.c_int
        self.lib.ws_error_string.restype = ctypes.c_char_p
        self.lib.ws_error_string.argtypes = [ctypes.c_int]

    def __call__(self, tensors: dict, dims: dict[str, int], stream_handle: int) -> None:
        args = [dims[a.name] if a.kind == "dim" else tensors[a.name].data_ptr() for a in self.problem.args]
        self.fn(*args, stream_handle)

    def last_error(self) -> str | None:
        code = self.lib.ws_last_error()
        return None if code == 0 else self.lib.ws_error_string(code).decode()
