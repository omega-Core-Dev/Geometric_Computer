# `torch-gncx` — GNC-X 4-Phase "X-Folding" Tensor Library (C++20 / Python / PyTorch)

**Author of Conception:** Zaqueu Ribeiro da Costa ([omega-Core-Dev](https://github.com/omega-Core-Dev))  
**License:** Apache-2.0

Native C++20 shared library (`gncx_core.dll` / `libgncx_core.so` / `libgncx_core.dylib`) and Python/PyTorch buildtool implementing the **GNC-X 4-Phase Quadrant Constellation (`11=++, 10=+-, 01=-+, 00=--`)** and **10-Pivot Invariant Diagonal Sieve ($\mathcal{D}(k, d) = d \cdot R_{k+1}$)** tensor operations:

- **`XTensor4Phase`**: 4-byte bijective 4-Phase encoding (`phase_2bit`, `diag_d`, `abs_delta`) with 1-cycle zero-carry negation via `XOR 0b11`.
- **`XFoldLinear`**: Forward and Backward ($O(10 \cdot M)$ exact weight & input gradients) + SGD updates (**37.16x faster** forward, **7.04x faster** backward on CPU).
- **`XFoldConv1d`**: $O(1)$ Sliding X-Window 1D Convolution (**11.92x faster**).
- **`XFoldConv2d`**: Two-Stage Spatial X-Folding 2D Convolution (**12.21x faster**).

## Direct Installation from GitHub

```bash
pip install "git+https://github.com/omega-Core-Dev/Geometric_Computer.git#subdirectory=03_torch_gncx_buildtool"
```
