#!/usr/bin/env python3
"""
GNC-X (`torch_gncx`) — Guia Prático de Uso (Quickstart Examples)
================================================================
Autor da Concepção: Zaqueu Ribeiro da Costa (omega-Core-Dev)
Licença: Apache-2.0

Demonstra de forma simples e direta os 4 casos de uso da biblioteca:
  1. Tokenização 4-Phase (`11=++, 10=+-, 01=-+, 00=--`) e Negação Zero-Carry (`XOR 0b11`)
  2. Camada Linear (`XFoldLinear`) + Treinamento (`backward_xfold` + `step_sgd`)
  3. Convolução 1D (`XFoldConv1d`) para Séries Temporais / Áudio
  4. Convolução 2D Espacial (`XFoldConv2d`) para Imagens / Visão Computacional
"""

import sys
from array import array
from pathlib import Path

# Permite rodar direto da pasta do repositório mesmo antes do pip install
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from torch_gncx import XFoldConv1d, XFoldConv2d, XFoldLinear, XTensor4Phase  # noqa: E402


def exemplo_1_tensor_4fases_e_xor11() -> None:
    print("============================================================================")
    print(" EXEMPLO 1: Tensor de 4 Fases (XTensor4Phase) e Negação 1-Ciclo (XOR 0b11)")
    print("============================================================================")

    dados = [4569, -4319, 7567, -7987]
    # Codifica na Layer k=3 (Repunit R_4 = 1111, Diagonais D(3,d) = 0, 1111, ..., 9999)
    xt = XTensor4Phase.from_any(dados, layer_k=3)

    mapa_fases = {0b11: "11 (++)", 0b10: "10 (+-)", 0b01: "01 (-+)", 0b00: "00 (--)"}
    for i, valor in enumerate(dados):
        tok = xt.tokens[i]
        ancora = tok.diag_d * 1111
        print(
            f"  Entrada[{i}] = {valor:6d} -> Fase={mapa_fases[tok.phase_2bit]} | "
            f"Diagonal d={tok.diag_d} (Âncora={ancora}) | |Delta|={tok.abs_delta}"
        )

    # Inverte o sinal do tensor inteiro aplicando apenas XOR 0b11 nos 2 bits de fase!
    xt.negate_inplace_xor11()
    fases_invertidas = [mapa_fases[xt.tokens[i].phase_2bit] for i in range(xt.num_elements)]
    print(f"  Fases após XOR 0b11 (-X): {fases_invertidas}\n")


def exemplo_2_linear_forward_backward_sgd() -> None:
    print("============================================================================")
    print(" EXEMPLO 2: Camada Linear (XFoldLinear) + Backward Pass O(10*M) + SGD")
    print("============================================================================")

    # Entrada com pares simétricos que se aniquilam no centro '0' do X
    entrada = [1111 + 50, -1111 - 50, 4444 + 120, 4444 - 120, 8888 + 30, -8888 + 30]
    xt = XTensor4Phase.from_any(entrada, layer_k=3)

    camada_linear = XFoldLinear(out_channels=4, layer_k=3)

    # 1. Forward Pass (ou simplesmente `saida = camada_linear(entrada)`)
    saida, pares_cancelados = camada_linear.forward_xfold(xt)
    print(f"  Saída Forward (4 canais)          : {list(saida)}")
    print(f"  Pares Opostos Cancelados no X     : {pares_cancelados}")

    # 2. Backward Pass (Calcula dL/dW e dL/dX com apenas 10*M multiplicações)
    grad_saida = array("q", [1, -1, 2, -2])
    grad_pesos, grad_entrada = camada_linear.backward_xfold(xt, grad_saida)
    print(f"  Gradiente da Entrada dL/dX[0..5]  : {list(grad_entrada)}")
    print(f"  Gradiente dos Pesos dL/dW[0..3]   : {list(grad_pesos[:4])}")

    # 3. Atualização de Pesos (SGD)
    camada_linear.step_sgd(lr_shift_bits=8)
    nova_saida, _ = camada_linear.forward_xfold(xt)
    print(f"  Saída após 1 passo de treino (SGD): {list(nova_saida)}\n")


def exemplo_3_convolucao_1d() -> None:
    print("============================================================================")
    print(" EXEMPLO 3: Convolução 1D Deslizante em O(1) (XFoldConv1d)")
    print("============================================================================")

    # Série temporal de 16 amostras e kernel de janela K=5 com 2 canais de saída
    sinal_1d = [2222 + i * 10 if i % 2 == 0 else -2222 - i * 10 for i in range(16)]
    conv1d = XFoldConv1d(out_channels=2, kernel_size=5, layer_k=3)

    xt_1d = XTensor4Phase.from_any(sinal_1d, layer_k=3)
    saida_1d, cancelados_1d = conv1d.forward_xfold(xt_1d)
    out_len = conv1d.out_length(len(sinal_1d))

    print(f"  Tamanho Entrada={len(sinal_1d)}, Kernel=5 -> Saída por Canal={out_len}")
    print(f"  Canal 0 (primeiras 6 posições)    : {list(saida_1d[:6])}")
    print(f"  Pares Opostos Cancelados na Janela: {cancelados_1d}\n")


def exemplo_4_convolucao_2d_espacial() -> None:
    print("============================================================================")
    print(" EXEMPLO 4: Convolução 2D Espacial (XFoldConv2d)")
    print("============================================================================")

    # Mapa 2D 8x8 (64 pixels) com Kernel 3x3 e 4 filtros de saída -> Saída 4x6x6
    altura, largura = 8, 8
    imagem_2d = [(r * 1111 + c * 25) * (1 if (r + c) % 2 == 0 else -1) for r in range(1, altura + 1) for c in range(largura)]

    conv2d = XFoldConv2d(out_channels=4, kernel_size=(3, 3), layer_k=3)
    xt_2d = XTensor4Phase.from_any(imagem_2d, layer_k=3)

    saida_2d, cancelados_2d = conv2d.forward_xfold(xt_2d, height=altura, width=largura)
    c_out, out_h, out_w = conv2d.out_shape(altura, largura)

    print(f"  Imagem Entrada={altura}x{largura}, Kernel=3x3 -> Shape Saída=({c_out}, {out_h}, {out_w})")
    print(f"  Filtro 0 (primeira linha 6 pixels): {list(saida_2d[:out_w])}")
    print(f"  Pares Opostos Cancelados no Campo : {cancelados_2d}\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    exemplo_1_tensor_4fases_e_xor11()
    exemplo_2_linear_forward_backward_sgd()
    exemplo_3_convolucao_1d()
    exemplo_4_convolucao_2d_espacial()
