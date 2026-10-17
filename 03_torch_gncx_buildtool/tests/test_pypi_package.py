#!/usr/bin/env python3
"""
GNC-X Package Verification & Unit Test Suite (`test_pypi_package.py`)
=====================================================================
Validates all 4 release requirements for `torch-gncx`:
  1. Cross-platform C++20 library resolution (`GNCXCoreBridge`)
  2. Bijective 4-Phase Tensor encoding (`XTensor4Phase`) & 1-cycle `XOR 0b11` negation
  3. `XFoldLinear` Forward + O(10*M) Backward Pass (`grad_weights` & `grad_input`) + SGD step
  4. `XFoldConv1d` & `XFoldConv2d` `nn.Module`-style `__call__` and bit-exactness
"""

import random
import sys
import time
import unittest
from array import array
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from torch_gncx import (  # noqa: E402
    GNCXCoreBridge,
    XFoldConv1d,
    XFoldConv2d,
    XFoldLinear,
    XTensor4Phase,
)


class TestTorchGNCXPackage(unittest.TestCase):
    def test_01_cross_platform_bridge_loaded(self) -> None:
        bridge = GNCXCoreBridge.get()
        self.assertIsNotNone(bridge.lib)

    def test_02_xtensor_4phase_bijectivity_and_xor11(self) -> None:
        raw = [1111, -1111, 4569, -4319, 7567, -7987, 9999, -9999]
        xt = XTensor4Phase.from_any(raw, layer_k=3)
        self.assertEqual(xt.to_list(), raw)

        phases_before = [xt.tokens[i].phase_2bit for i in range(xt.num_elements)]
        xt.negate_inplace_xor11()
        phases_after = [xt.tokens[i].phase_2bit for i in range(xt.num_elements)]
        for p0, p1 in zip(phases_before, phases_after):
            self.assertEqual(p0 ^ 0b11, p1)

    def test_03_xfold_linear_forward_backward_and_sgd(self) -> None:
        rng = random.Random(123)
        n = 50_000
        m = 32
        data = array("q", [rng.randint(-9500, 9500) for _ in range(n)])
        xt = XTensor4Phase.from_int64_buffer(data, layer_k=3)

        layer = XFoldLinear(out_channels=m, layer_k=3)

        # Forward pass via __call__ and forward_xfold
        out_call = layer(data)
        out_xfold, _ = layer.forward_xfold(xt)
        out_dense = layer.forward_dense_baseline(xt)
        self.assertEqual(list(out_call), list(out_dense))
        self.assertEqual(list(out_xfold), list(out_dense))

        # Backward pass (dL/dW and dL/dX) bit-for-bit exactness + speedup
        grad_out = array("q", [rng.randint(-20, 20) for _ in range(m)])

        t0 = time.perf_counter()
        gw_dense, gx_dense = layer.backward_dense_baseline(xt, grad_out)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        gw_xfold, gx_xfold = layer.backward_xfold(xt, grad_out)
        t3 = time.perf_counter()

        self.assertEqual(list(gw_xfold), list(gw_dense))
        self.assertEqual(list(gx_xfold), list(gx_dense))

        dense_bwd_ms = (t1 - t0) * 1000.0
        xfold_bwd_ms = (t3 - t2) * 1000.0
        print(
            f"\n  [Backward Pass Check] dL/dW & dL/dX 100% IDÊNTICOS | "
            f"Dense: {dense_bwd_ms:.2f} ms vs XFold: {xfold_bwd_ms:.2f} ms "
            f"({dense_bwd_ms / max(xfold_bwd_ms, 1e-6):.2f}x speedup)"
        )

        # Verify SGD step updates weights cleanly
        w0_before = layer.weights[4]
        layer.step_sgd(lr_shift_bits=8)
        self.assertNotEqual(layer.weights[4], w0_before)

    def test_04_xfold_conv1d_and_conv2d_call(self) -> None:
        rng = random.Random(77)
        # Conv1d
        seq = [rng.randint(-8000, 8000) for _ in range(512)]
        c1d = XFoldConv1d(out_channels=8, kernel_size=7, layer_k=3)
        xt1d = XTensor4Phase.from_any(seq, layer_k=3)
        self.assertEqual(list(c1d(seq)), list(c1d.forward_dense_baseline(xt1d)))

        # Conv2d
        h, w = 32, 32
        img = [rng.randint(-8000, 8000) for _ in range(h * w)]
        c2d = XFoldConv2d(out_channels=8, kernel_size=(5, 5), layer_k=3)
        xt2d = XTensor4Phase.from_any(img, layer_k=3)
        self.assertEqual(list(c2d(img, h, w)), list(c2d.forward_dense_baseline(xt2d, h, w)))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
