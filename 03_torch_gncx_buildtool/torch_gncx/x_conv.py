"""
GNC-X Buildtool — 1D & 2D 4-Phase Geometric Convolutions (`torch_gncx.x_conv`)
Author of Conception: Zaqueu Ribeiro da Costa

Provides `XFoldConv1d` and `XFoldConv2d` backed by the C++20 `gncx_core.dll`
sliding X-Sieve kernel:
  - Folds receptive windows into the 10 Invariant Diagonal Anchors D(k, 0..9).
  - Cancels opposite-phase pairs (+D vs -D) at the central '0' origin of the
    receptive window (ideal for visual edges and zero-crossing waveforms).
  - Updates sliding windows in O(1) independent of kernel size!
"""

import ctypes
from array import array
from typing import Tuple

from .x_linear import GNCXCoreBridge, XTensor4Phase


class XFoldConv1d:
    """
    1D Multi-Channel Geometric Convolution over 4-Phase X-Tokens (`torch.nn.Conv1d` compatible).
    """

    def __init__(self, out_channels: int, kernel_size: int, layer_k: int = 3) -> None:
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.layer_k = layer_k
        self.weights = (ctypes.c_int64 * (out_channels * 10))()
        for m in range(out_channels):
            for d in range(10):
                self.weights[m * 10 + d] = (m + 2) * (d + 1) - 13

    def out_length(self, length: int) -> int:
        return max(0, length - self.kernel_size + 1)

    def forward_xfold(
        self, x_tensor: XTensor4Phase, out_buf: " array | None " = None
    ) -> Tuple[array, int]:
        """Executes C++20 O(1) Sliding X-Window 1D Convolution."""
        out_len = self.out_length(x_tensor.num_elements)
        if out_buf is None:
            out_buf = array("q", [0]) * (self.out_channels * out_len)
        out_addr, _ = out_buf.buffer_info()
        out_ptr = ctypes.cast(out_addr, ctypes.POINTER(ctypes.c_int64))

        canceled = GNCXCoreBridge.get().lib.gncx_xfold_conv1d_forward(
            x_tensor.tokens,
            x_tensor.num_elements,
            self.kernel_size,
            self.layer_k,
            self.weights,
            self.out_channels,
            out_ptr,
        )
        return out_buf, int(canceled)

    def forward_dense_baseline(
        self, x_tensor: XTensor4Phase, out_buf: " array | None " = None
    ) -> array:
        """Executes standard O(L_out * K * C_out) 1D Convolution for bit-exact validation."""
        out_len = self.out_length(x_tensor.num_elements)
        if out_buf is None:
            out_buf = array("q", [0]) * (self.out_channels * out_len)
        out_addr, _ = out_buf.buffer_info()
        out_ptr = ctypes.cast(out_addr, ctypes.POINTER(ctypes.c_int64))

        GNCXCoreBridge.get().lib.gncx_dense_conv1d_baseline(
            x_tensor.reconstructed,
            x_tensor.tokens,
            x_tensor.num_elements,
            self.kernel_size,
            self.weights,
            self.out_channels,
            out_ptr,
        )
        return out_buf


class XFoldConv2d:
    """
    2D Spatial Multi-Channel Geometric Convolution over 4-Phase X-Tokens (`torch.nn.Conv2d` compatible).
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: Tuple[int, int] = (5, 5),
        layer_k: int = 3,
    ) -> None:
        self.out_channels = out_channels
        self.kernel_h, self.kernel_w = kernel_size
        self.layer_k = layer_k
        self.weights = (ctypes.c_int64 * (out_channels * 10))()
        for m in range(out_channels):
            for d in range(10):
                self.weights[m * 10 + d] = (m + 3) * (d + 2) - 19

    def out_shape(self, height: int, width: int) -> Tuple[int, int, int]:
        out_h = max(0, height - self.kernel_h + 1)
        out_w = max(0, width - self.kernel_w + 1)
        return (self.out_channels, out_h, out_w)

    def forward_xfold(
        self,
        x_tensor: XTensor4Phase,
        height: int,
        width: int,
        out_buf: " array | None " = None,
    ) -> Tuple[array, int]:
        """Executes C++20 Two-Stage Spatial X-Folding 2D Convolution."""
        assert height * width == x_tensor.num_elements, "Dimensoes H*W diferentes do XTensor4Phase!"
        c_out, out_h, out_w = self.out_shape(height, width)
        if out_buf is None:
            out_buf = array("q", [0]) * (c_out * out_h * out_w)
        out_addr, _ = out_buf.buffer_info()
        out_ptr = ctypes.cast(out_addr, ctypes.POINTER(ctypes.c_int64))

        canceled = GNCXCoreBridge.get().lib.gncx_xfold_conv2d_forward(
            x_tensor.tokens,
            height,
            width,
            self.kernel_h,
            self.kernel_w,
            self.layer_k,
            self.weights,
            self.out_channels,
            out_ptr,
        )
        return out_buf, int(canceled)

    def forward_dense_baseline(
        self,
        x_tensor: XTensor4Phase,
        height: int,
        width: int,
        out_buf: " array | None " = None,
    ) -> array:
        """Executes standard O(H_out * W_out * K_h * K_w * C_out) 2D Convolution for verification."""
        assert height * width == x_tensor.num_elements, "Dimensoes H*W diferentes do XTensor4Phase!"
        c_out, out_h, out_w = self.out_shape(height, width)
        if out_buf is None:
            out_buf = array("q", [0]) * (c_out * out_h * out_w)
        out_addr, _ = out_buf.buffer_info()
        out_ptr = ctypes.cast(out_addr, ctypes.POINTER(ctypes.c_int64))

        GNCXCoreBridge.get().lib.gncx_dense_conv2d_baseline(
            x_tensor.reconstructed,
            x_tensor.tokens,
            height,
            width,
            self.kernel_h,
            self.kernel_w,
            self.weights,
            self.out_channels,
            out_ptr,
        )
        return out_buf
