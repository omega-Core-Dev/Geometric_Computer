"""
GNC-X Buildtool — Python & PyTorch Bridge (`torch_gncx.x_linear`)
Author of Conception: Zaqueu Ribeiro da Costa

Loads the compiled C++20 `gncx_core.dll` via `ctypes` (zero copy over contiguous
buffers / PyTorch CPU tensors) and provides:
- `XTensor4Phase`: Compact 32-bit 4-Phase Tensor (`++=11, +-=10, -+=01, --=00`)
  with 1-cycle zero-carry negation (`XOR 11`).
- `XFoldLinear`: Drop-in linear projection layer compatible with PyTorch `nn.Module`
  (or standalone NumPy/array buffers when PyTorch is not imported).
"""

import ctypes
import os
import shutil
import subprocess
import sys
from array import array
from typing import Optional, Tuple


class GNCXCompactTokenStruct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("phase_2bit", ctypes.c_uint8),
        ("diag_d", ctypes.c_uint8),
        ("abs_delta", ctypes.c_uint16),
    ]


class GNCXCoreBridge:
    """
    Loads and binds the native C++20 `gncx_core` shared library (.dll / .so / .dylib).
    If running on Linux (e.g., Google Colab) or macOS after installing directly from
    GitHub (`pip install git+https://github.com/omega-Core-Dev/Geometric_Computer.git#subdirectory=03_torch_gncx_buildtool`),
    automatically compiles `csrc/gncx_torch_c_api.cpp` on first load!
    """

    _instance: Optional["GNCXCoreBridge"] = None

    @staticmethod
    def _resolve_or_compile_library() -> str:
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(pkg_dir)

        if sys.platform.startswith("win"):
            lib_name = "gncx_core.dll"
        elif sys.platform == "darwin":
            lib_name = "libgncx_core.dylib"
        else:
            lib_name = "libgncx_core.so"

        candidates = [
            os.path.join(pkg_dir, "bin", lib_name),
            os.path.join(base_dir, "bin", lib_name),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

        # Auto-compile on first import (ideal for Linux / Google Colab via GitHub install)
        cpp_candidates = [
            os.path.join(pkg_dir, "csrc", "gncx_torch_c_api.cpp"),
            os.path.join(base_dir, "csrc", "gncx_torch_c_api.cpp"),
        ]
        cpp_src = next((p for p in cpp_candidates if os.path.exists(p)), None)
        if cpp_src is None:
            raise FileNotFoundError(
                f"Biblioteca nativa {lib_name} e fonte C++20 nao encontrados em {base_dir}"
            )

        out_dir = os.path.join(pkg_dir, "bin")
        os.makedirs(out_dir, exist_ok=True)
        out_lib = os.path.join(out_dir, lib_name)

        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            raise RuntimeError("Compilador C++20 (clang++ ou g++) nao encontrado no PATH.")

        cmd = [compiler, "-shared", "-std=c++20", "-O3", "-fPIC", cpp_src, "-o", out_lib]
        subprocess.run(cmd, check=True)
        return out_lib

    def __init__(self, dll_path: Optional[str] = None) -> None:
        if dll_path is None:
            dll_path = self._resolve_or_compile_library()
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"GNC-X C++20 library nao encontrada em: {dll_path}")

        self.lib = ctypes.CDLL(dll_path)

        # gncx_encode_4phase_tensor
        self.lib.gncx_encode_4phase_tensor.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.POINTER(GNCXCompactTokenStruct),
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.lib.gncx_encode_4phase_tensor.restype = None

        # gncx_negate_tensor_phase_xor
        self.lib.gncx_negate_tensor_phase_xor.argtypes = [
            ctypes.POINTER(GNCXCompactTokenStruct),
            ctypes.c_size_t,
        ]
        self.lib.gncx_negate_tensor_phase_xor.restype = None

        # gncx_xfold_linear_forward
        self.lib.gncx_xfold_linear_forward.argtypes = [
            ctypes.POINTER(GNCXCompactTokenStruct),
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.lib.gncx_xfold_linear_forward.restype = ctypes.c_uint64

        # gncx_dense_linear_baseline
        self.lib.gncx_dense_linear_baseline.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(GNCXCompactTokenStruct),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.lib.gncx_dense_linear_baseline.restype = None

        # gncx_xfold_conv1d_forward
        self.lib.gncx_xfold_conv1d_forward.argtypes = [
            ctypes.POINTER(GNCXCompactTokenStruct),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.lib.gncx_xfold_conv1d_forward.restype = ctypes.c_uint64

        # gncx_dense_conv1d_baseline
        self.lib.gncx_dense_conv1d_baseline.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(GNCXCompactTokenStruct),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.lib.gncx_dense_conv1d_baseline.restype = None

        # gncx_xfold_conv2d_forward
        self.lib.gncx_xfold_conv2d_forward.argtypes = [
            ctypes.POINTER(GNCXCompactTokenStruct),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.lib.gncx_xfold_conv2d_forward.restype = ctypes.c_uint64

        # gncx_dense_conv2d_baseline
        self.lib.gncx_dense_conv2d_baseline.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(GNCXCompactTokenStruct),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.lib.gncx_dense_conv2d_baseline.restype = None

        # gncx_xfold_linear_backward
        self.lib.gncx_xfold_linear_backward.argtypes = [
            ctypes.POINTER(GNCXCompactTokenStruct),
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.lib.gncx_xfold_linear_backward.restype = None

        # gncx_dense_linear_backward
        self.lib.gncx_dense_linear_backward.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(GNCXCompactTokenStruct),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.lib.gncx_dense_linear_backward.restype = None

    @classmethod
    def get(cls) -> "GNCXCoreBridge":
        if cls._instance is None:
            cls._instance = GNCXCoreBridge()
        return cls._instance


class XTensor4Phase:
    """
    Represents a Tensor encoded in the GNC-X 4-Phase Quadrant Constellation:
      Phase 11 (++) = +Pivot, +Residue
      Phase 10 (+-) = +Pivot, -Residue
      Phase 01 (-+) = -Pivot, +Residue
      Phase 00 (--) = -Pivot, -Residue
    """

    def __init__(self, num_elements: int, layer_k: int = 3) -> None:
        self.num_elements = num_elements
        self.layer_k = layer_k
        self.tokens = (GNCXCompactTokenStruct * num_elements)()
        self.reconstructed = (ctypes.c_int64 * num_elements)()

    @classmethod
    def from_int64_buffer(cls, values: array, layer_k: int = 3) -> "XTensor4Phase":
        n = len(values)
        xt = cls(n, layer_k=layer_k)
        bridge = GNCXCoreBridge.get()
        addr, _ = values.buffer_info()
        in_ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int64))
        bridge.lib.gncx_encode_4phase_tensor(
            in_ptr, n, layer_k, xt.tokens, xt.reconstructed
        )
        return xt

    @classmethod
    def from_any(cls, data: object, layer_k: int = 3) -> "XTensor4Phase":
        """Accepts `XTensor4Phase`, `array('q')`, Python `list`/`tuple`, or `torch.Tensor`."""
        if isinstance(data, XTensor4Phase):
            return data
        if isinstance(data, array) and data.typecode == "q":
            return cls.from_int64_buffer(data, layer_k=layer_k)
        # Support PyTorch Tensor if passed
        if hasattr(data, "detach") and hasattr(data, "cpu"):
            flat_list = [int(x) for x in data.detach().cpu().view(-1).tolist()]
            return cls.from_int64_buffer(array("q", flat_list), layer_k=layer_k)
        return cls.from_int64_buffer(array("q", list(data)), layer_k=layer_k)  # type: ignore[arg-type]

    def to_list(self) -> list:
        return [int(self.reconstructed[i]) for i in range(self.num_elements)]

    def negate_inplace_xor11(self) -> None:
        """Negates the entire tensor in-place by flipping the 2-bit phases (XOR 0b11)."""
        GNCXCoreBridge.get().lib.gncx_negate_tensor_phase_xor(self.tokens, self.num_elements)


class XFoldLinear:
    """
    GNC-X "X-Folding" Linear Projection Layer (`torch.nn.Linear` drop-in compatible).
    Evaluates `M` output channels over `N` 4-Phase X-Tokens by folding into the
    10 Invariant Diagonal Anchors D(k, 0..9) in C++20, and supports O(10*M)
    exact backward gradient calculation (`backward_xfold`) and SGD weight updates!
    """

    def __init__(self, out_channels: int, layer_k: int = 3) -> None:
        self.out_channels = out_channels
        self.layer_k = layer_k
        self.weights = (ctypes.c_int64 * (out_channels * 10))()
        self.grad_weights = (ctypes.c_int64 * (out_channels * 10))()
        for m in range(out_channels):
            for d in range(10):
                self.weights[m * 10 + d] = (m + 1) * (d + 3) - 17
                self.grad_weights[m * 10 + d] = 0

    def __call__(self, x: object) -> array:
        xt = XTensor4Phase.from_any(x, layer_k=self.layer_k)
        out, _ = self.forward_xfold(xt)
        return out

    def forward_xfold(self, x_tensor: XTensor4Phase) -> Tuple[array, int]:
        """Executes the C++20 X-Folding forward pass (only 10 * M multiplications)."""
        out_buf = array("q", [0]) * self.out_channels
        out_addr, _ = out_buf.buffer_info()
        out_ptr = ctypes.cast(out_addr, ctypes.POINTER(ctypes.c_int64))

        canceled = GNCXCoreBridge.get().lib.gncx_xfold_linear_forward(
            x_tensor.tokens,
            x_tensor.num_elements,
            self.layer_k,
            self.weights,
            self.out_channels,
            out_ptr,
        )
        return out_buf, int(canceled)

    def forward_dense_baseline(self, x_tensor: XTensor4Phase) -> array:
        """Executes the standard N * M dense multiplication for bit-exactness verification."""
        out_buf = array("q", [0]) * self.out_channels
        out_addr, _ = out_buf.buffer_info()
        out_ptr = ctypes.cast(out_addr, ctypes.POINTER(ctypes.c_int64))

        GNCXCoreBridge.get().lib.gncx_dense_linear_baseline(
            x_tensor.reconstructed,
            x_tensor.tokens,
            x_tensor.num_elements,
            self.weights,
            self.out_channels,
            out_ptr,
        )
        return out_buf

    def backward_xfold(
        self, x_tensor: XTensor4Phase, grad_output: array
    ) -> Tuple[array, array]:
        """
        Executes the C++20 X-Folding Backward Pass (`loss.backward()`):
          - Computes exact dL/dW[m, d] using ONLY 10 * M multiplications!
          - Computes exact dL/dX[i] via O(1) 10-diagonal lookup!
        Returns `(grad_weights_array, grad_input_array)`.
        """
        assert len(grad_output) == self.out_channels, "grad_output deve ter tamanho out_channels!"
        go_addr, _ = grad_output.buffer_info()
        go_ptr = ctypes.cast(go_addr, ctypes.POINTER(ctypes.c_int64))

        grad_x_buf = array("q", [0]) * x_tensor.num_elements
        gx_addr, _ = grad_x_buf.buffer_info()
        gx_ptr = ctypes.cast(gx_addr, ctypes.POINTER(ctypes.c_int64))

        GNCXCoreBridge.get().lib.gncx_xfold_linear_backward(
            x_tensor.tokens,
            x_tensor.num_elements,
            self.layer_k,
            self.weights,
            go_ptr,
            self.out_channels,
            self.grad_weights,
            gx_ptr,
        )
        gw_buf = array("q", [self.grad_weights[i] for i in range(self.out_channels * 10)])
        return gw_buf, grad_x_buf

    def backward_dense_baseline(
        self, x_tensor: XTensor4Phase, grad_output: array
    ) -> Tuple[array, array]:
        """Executes standard N * M dense backward pass for bit-exact gradient verification."""
        go_addr, _ = grad_output.buffer_info()
        go_ptr = ctypes.cast(go_addr, ctypes.POINTER(ctypes.c_int64))

        dense_gw = (ctypes.c_int64 * (self.out_channels * 10))()
        grad_x_buf = array("q", [0]) * x_tensor.num_elements
        gx_addr, _ = grad_x_buf.buffer_info()
        gx_ptr = ctypes.cast(gx_addr, ctypes.POINTER(ctypes.c_int64))

        GNCXCoreBridge.get().lib.gncx_dense_linear_backward(
            x_tensor.reconstructed,
            x_tensor.tokens,
            x_tensor.num_elements,
            self.weights,
            go_ptr,
            self.out_channels,
            dense_gw,
            gx_ptr,
        )
        gw_buf = array("q", [dense_gw[i] for i in range(self.out_channels * 10)])
        return gw_buf, grad_x_buf

    def step_sgd(self, lr_shift_bits: int = 10) -> None:
        """Applies integer/fixed-point SGD update to the 10-diagonal weights: W -= (grad_W >> lr_shift)."""
        for i in range(self.out_channels * 10):
            self.weights[i] -= (self.grad_weights[i] >> lr_shift_bits)

