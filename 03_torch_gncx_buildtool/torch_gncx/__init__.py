"""
torch_gncx — GNC-X 4-Phase "X-Folding" Tensor Library for Python & PyTorch
Author of Conception: Zaqueu Ribeiro da Costa

Bridges the native C++20 `gncx_core.dll` into Python and PyTorch (`torch.nn.Module`),
enabling entry-level CPUs to execute multi-channel linear projections and 1D/2D
spatial convolutions with 4-Phase Quadrant cancelation (`11=++, 10=+-, 01=-+, 00=--`)
and 10-Pivot Invariant Diagonal Sieve folding.
"""

from .x_linear import GNCXCoreBridge, XTensor4Phase, XFoldLinear
from .x_conv import XFoldConv1d, XFoldConv2d

__all__ = [
    "GNCXCoreBridge",
    "XTensor4Phase",
    "XFoldLinear",
    "XFoldConv1d",
    "XFoldConv2d",
]
