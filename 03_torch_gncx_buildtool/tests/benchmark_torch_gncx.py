#!/usr/bin/env python3
"""
GNC-X PyTorch / Python C++20 Buildtool — Suite de Validação & Benchmark
=======================================================================
Valida a integração entre Python/PyTorch e o kernel C++20 nativo (gncx_core.dll):
  1. Codificação 4-Phase (11=++, 10=+-, 01=-+, 00=--) em CompactXToken (32 bits)
  2. Negação algébrica de 1 ciclo via XOR 0b11 (Zero-Carry)
  3. Camada Linear XFoldLinear vs. Multiplicação Densa Tradicional (32M MACs)
  4. Camada Convolucional 1D XFoldConv1d vs. Conv1d Densa (99.17M MACs)
  5. Camada Convolucional 2D Espacial XFoldConv2d vs. Conv2d Densa (98.00M MACs)
"""

import random
import sys
import time
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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("\033[38;5;39m╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║  GNC-X BUILDTOOL: torch_gncx (C++20 Native DLL + Python/PyTorch Engine)     ║")
    print("║  4-Phase Quadrant Logic (11=++, 10=+-, 01=-+, 00=--) | Linear & Conv1d/2d   ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝\033[0m\n")

    bridge = GNCXCoreBridge.get()
    print(f"\033[1m[1] Biblioteca C++20 Carregada:\033[0m {bridge.lib._name}")

    # 2. Testa Tokenização 4-Phase e Negação XOR 0b11
    print("\n\033[1m[2] Testando Tokenização 4-Phase (CompactXToken 4 bytes) & XOR 11 Negation:\033[0m")
    sample_list = [4444 + 125, -4444 + 125, 7777 - 210, -7777 - 210, 8888, -8888, 3333 + 42, -3333 - 42]
    sample_buf = array("q", sample_list)
    xt = XTensor4Phase.from_int64_buffer(sample_buf, layer_k=3)
    recon_list = [xt.reconstructed[i] for i in range(xt.num_elements)]

    print(f"    Entrada Original (int64) : {sample_list}")
    print(f"    Reconstruído do Crivo L3 : {recon_list}")
    assert sample_list == recon_list, "Falha na bijetividade do Crivo L3!"

    phase_map = {0b11: "11 (++)", 0b10: "10 (+-)", 0b01: "01 (-+)", 0b00: "00 (--)"}
    for i in range(4):
        tok = xt.tokens[i]
        print(
            f"    x[{i}] = {sample_list[i]:6d} -> Phase={phase_map[tok.phase_2bit]} | "
            f"d={tok.diag_d} (Anchor={tok.diag_d * 1111}) | |delta|={tok.abs_delta}"
        )

    orig_phases = [xt.tokens[i].phase_2bit for i in range(4)]
    xt.negate_inplace_xor11()
    neg_phases = [xt.tokens[i].phase_2bit for i in range(4)]
    print(f"    Fases antes do XOR 11    : {[phase_map[p] for p in orig_phases]}")
    print(f"    Fases após XOR 0b11      : {[phase_map[p] for p in neg_phases]}")
    for p_old, p_new in zip(orig_phases, neg_phases):
        assert (p_old ^ 0b11) == p_new, "Falha no XOR 0b11!"
    xt.negate_inplace_xor11()
    print("    \033[38;5;46m[OK] Bijetividade 100% exata e Negação de 1 ciclo (XOR 11) validada!\033[0m")

    # 3. Benchmark XFoldLinear (500.000 elementos x 64 Canais = 32M MACs)
    num_elements = 500_000
    out_channels = 64
    print(
        f"\n\033[1m[3] Benchmark XFoldLinear ({num_elements:,} elementos x {out_channels} canais = 32.00M MACs):\033[0m"
    )
    rng = random.Random(42)
    large_buf = array("q", [0] * num_elements)
    for i in range(0, num_elements, 2):
        d = rng.randint(1, 9)
        delta = rng.randint(-450, 450)
        val = d * 1111 + delta
        large_buf[i] = val
        large_buf[i + 1] = -val if (i % 4 == 0) else (d * 1111 - delta)

    xt_large = XTensor4Phase.from_int64_buffer(large_buf, layer_k=3)
    linear_layer = XFoldLinear(out_channels=out_channels, layer_k=3)

    _ = linear_layer.forward_dense_baseline(xt_large)
    _, _ = linear_layer.forward_xfold(xt_large)

    runs = 5
    t0 = time.perf_counter()
    for _ in range(runs):
        dense_out = linear_layer.forward_dense_baseline(xt_large)
    t1 = time.perf_counter()
    dense_ms = ((t1 - t0) * 1000.0) / runs

    t2 = time.perf_counter()
    for _ in range(runs):
        xfold_out, canceled_pairs = linear_layer.forward_xfold(xt_large)
    t3 = time.perf_counter()
    xfold_ms = ((t3 - t2) * 1000.0) / runs

    exact_match = list(dense_out) == list(xfold_out)
    speedup = dense_ms / max(xfold_ms, 1e-6)

    print(f"    • Baseline Denso Tradicional (32M MULs) : {dense_ms:.3f} ms")
    print(f"    • GNC-X 4-Phase XFoldLinear (640 MULs)  : \033[38;5;46m{xfold_ms:.3f} ms\033[0m")
    print(f"    • Speedup Real via Python/C++20 Bridge  : \033[38;5;226m{speedup:.2f}x mais rápido\033[0m")
    print(f"    • Pares Opostos Cancelados no X         : {canceled_pairs:,}")
    print(
        f"    • Exatidão Bit-a-Bit (Checksums Iguais) : "
        f"\033[38;5;46m{'100% IDÊNTICO' if exact_match else 'ERRO'}\033[0m"
    )
    assert exact_match, "Divergência entre XFoldLinear e Baseline Denso!"

    # 4. Benchmark XFoldConv1d (100.000 amostras, Kernel K=31, 32 Canais = 99.17M MACs)
    seq_len = 100_000
    k1d = 31
    c1d_out = 32
    macs_1d = (seq_len - k1d + 1) * k1d * c1d_out
    print(
        f"\n\033[1m[4] Benchmark XFoldConv1d (L={seq_len:,}, Kernel={k1d}, C_out={c1d_out} = {macs_1d/1e6:.2f}M MACs):\033[0m"
    )
    xt_1d = XTensor4Phase.from_int64_buffer(large_buf[:seq_len], layer_k=3)
    conv1d_layer = XFoldConv1d(out_channels=c1d_out, kernel_size=k1d, layer_k=3)

    c1d_dense = array("q", [0]) * (c1d_out * conv1d_layer.out_length(seq_len))
    c1d_xfold = array("q", [0]) * (c1d_out * conv1d_layer.out_length(seq_len))
    _ = conv1d_layer.forward_dense_baseline(xt_1d, out_buf=c1d_dense)
    _, _ = conv1d_layer.forward_xfold(xt_1d, out_buf=c1d_xfold)

    t0 = time.perf_counter()
    for _ in range(5):
        conv1d_layer.forward_dense_baseline(xt_1d, out_buf=c1d_dense)
    t1 = time.perf_counter()
    c1d_dense_ms = ((t1 - t0) * 1000.0) / 5

    t2 = time.perf_counter()
    for _ in range(5):
        _, c1d_canceled = conv1d_layer.forward_xfold(xt_1d, out_buf=c1d_xfold)
    t3 = time.perf_counter()
    c1d_xfold_ms = ((t3 - t2) * 1000.0) / 5

    c1d_exact = c1d_dense == c1d_xfold
    c1d_speedup = c1d_dense_ms / max(c1d_xfold_ms, 1e-6)

    print(f"    • Conv1d Densa Tradicional ({macs_1d/1e6:.1f}M MULs) : {c1d_dense_ms:.3f} ms")
    print(f"    • GNC-X 4-Phase XFoldConv1d (O(1) Slide): \033[38;5;46m{c1d_xfold_ms:.3f} ms\033[0m")
    print(f"    • Speedup Real em Conv1d                : \033[38;5;226m{c1d_speedup:.2f}x mais rápido\033[0m")
    print(f"    • Pares Opostos Cancelados nas Janelas  : {c1d_canceled:,}")
    print(
        f"    • Exatidão Bit-a-Bit (Checksums Iguais) : "
        f"\033[38;5;46m{'100% IDÊNTICO' if c1d_exact else 'ERRO'}\033[0m"
    )
    assert c1d_exact, "Divergência entre XFoldConv1d e Baseline Denso!"

    # 5. Benchmark XFoldConv2d (Mapa 256x256, Kernel 7x7, 32 Canais = 98.00M MACs)
    h2d, w2d = 256, 256
    kh, kw = 7, 7
    c2d_out = 32
    macs_2d = (h2d - kh + 1) * (w2d - kw + 1) * (kh * kw) * c2d_out
    print(
        f"\n\033[1m[5] Benchmark XFoldConv2d (Grid={h2d}x{w2d}, Kernel={kh}x{kw}, C_out={c2d_out} = {macs_2d/1e6:.2f}M MACs):\033[0m"
    )
    xt_2d = XTensor4Phase.from_int64_buffer(large_buf[: h2d * w2d], layer_k=3)
    conv2d_layer = XFoldConv2d(out_channels=c2d_out, kernel_size=(kh, kw), layer_k=3)

    _, out_h, out_w = conv2d_layer.out_shape(h2d, w2d)
    c2d_dense = array("q", [0]) * (c2d_out * out_h * out_w)
    c2d_xfold = array("q", [0]) * (c2d_out * out_h * out_w)
    _ = conv2d_layer.forward_dense_baseline(xt_2d, h2d, w2d, out_buf=c2d_dense)
    _, _ = conv2d_layer.forward_xfold(xt_2d, h2d, w2d, out_buf=c2d_xfold)

    t0 = time.perf_counter()
    for _ in range(5):
        conv2d_layer.forward_dense_baseline(xt_2d, h2d, w2d, out_buf=c2d_dense)
    t1 = time.perf_counter()
    c2d_dense_ms = ((t1 - t0) * 1000.0) / 5

    t2 = time.perf_counter()
    for _ in range(5):
        _, c2d_canceled = conv2d_layer.forward_xfold(xt_2d, h2d, w2d, out_buf=c2d_xfold)
    t3 = time.perf_counter()
    c2d_xfold_ms = ((t3 - t2) * 1000.0) / 5

    c2d_exact = c2d_dense == c2d_xfold
    c2d_speedup = c2d_dense_ms / max(c2d_xfold_ms, 1e-6)

    print(f"    • Conv2d Densa Tradicional ({macs_2d/1e6:.1f}M MULs) : {c2d_dense_ms:.3f} ms")
    print(f"    • GNC-X 4-Phase XFoldConv2d (2D X-Fold) : \033[38;5;46m{c2d_xfold_ms:.3f} ms\033[0m")
    print(f"    • Speedup Real em Conv2d Espacial       : \033[38;5;226m{c2d_speedup:.2f}x mais rápido\033[0m")
    print(f"    • Pares Opostos Cancelados no Campo 2D  : {c2d_canceled:,}")
    print(
        f"    • Exatidão Bit-a-Bit (Checksums Iguais) : "
        f"\033[38;5;46m{'100% IDÊNTICO' if c2d_exact else 'ERRO'}\033[0m"
    )
    assert c2d_exact, "Divergência entre XFoldConv2d e Baseline Denso!"


if __name__ == "__main__":
    main()
