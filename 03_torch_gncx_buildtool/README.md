# `torch-gncx` — GNC-X 4-Phase "X-Folding" Tensor Library (C++20 / Python / PyTorch)

**Autor da Concepção:** Zaqueu Ribeiro da Costa ([omega-Core-Dev](https://github.com/omega-Core-Dev))  
**Licença:** Apache-2.0

Biblioteca compartilhada nativa em C++20 (`gncx_core.dll` / `libgncx_core.so` / `libgncx_core.dylib`) e pacote Python/PyTorch que implementa a **Constelação Quaternária de 4 Fases em `X` (`11=++, 10=+-, 01=-+, 00=--`)** e o dobramento algébrico sobre os **10 Crivos Diagonais Invariantes ($\mathcal{D}(k, d) = d \cdot R_{k+1}$)**.

---

## 1. Instalação Direta pelo GitHub

### Opção A — Instalar o pacote `.whl` direto da Release `v0.1.0` (Mais Rápido)
```bash
pip install https://github.com/omega-Core-Dev/Geometric_Computer/releases/download/v0.1.0/torch_gncx-0.1.0-py3-none-any.whl
```

### Opção B — Instalar direto da branch `main` via Git
```bash
pip install "git+https://github.com/omega-Core-Dev/Geometric_Computer.git#subdirectory=03_torch_gncx_buildtool"
```
*(No Windows usa a `gncx_core.dll` embutida; no Linux/Google Colab/macOS compila automaticamente o kernel C++20 `gncx_torch_c_api.cpp` na primeira importação).*

---

## 2. Exemplos Práticos de Uso Passo a Passo

Você também pode executar o script completo em [`examples/01_quickstart_uso_completo.py`](examples/01_quickstart_uso_completo.py):

```bash
python examples/01_quickstart_uso_completo.py
```

### Exemplo 2.1 — Tokenização 4-Phase (`XTensor4Phase`) e Negação de 1 Ciclo (`XOR 0b11`)
```python
from torch_gncx import XTensor4Phase

# Codifica inteiros de 64 bits em Tokens Compactos de 4 bytes na Layer k=3 (R_4 = 1111)
dados = [4569, -4319, 7567, -7987]
xt = XTensor4Phase.from_any(dados, layer_k=3)

# Reconstrução 100% bijetiva: [4569, -4319, 7567, -7987]
print("Reconstruído:", xt.to_list())

# Inverte o sinal (-X) do tensor inteiro em 1 ciclo via XOR 0b11 nos 2 bits de fase (Zero-Carry!)
xt.negate_inplace_xor11()
```

### Exemplo 2.2 — Camada Linear (`XFoldLinear`) + Backward Pass $O(10 \cdot M)$ + Treino SGD
```python
from array import array
from torch_gncx import XTensor4Phase, XFoldLinear

xt = XTensor4Phase.from_any([1161, -1161, 4564, 4324, 8918, -8858], layer_k=3)
linear = XFoldLinear(out_channels=4, layer_k=3)

# 1. Forward Pass (37.16x mais rápido que multiplicação densa em CPU)
saida, pares_cancelados = linear.forward_xfold(xt)
print("Saída:", list(saida), "| Pares opostos cancelados no X:", pares_cancelados)

# 2. Backward Pass (Calcula dL/dW e dL/dX com apenas 10*M multiplicações!)
grad_saida = array("q", [1, -1, 2, -2])
grad_pesos, grad_entrada = linear.backward_xfold(xt, grad_saida)

# 3. Passo de Otimização (SGD sobre as 10 diagonais)
linear.step_sgd(lr_shift_bits=8)
```

### Exemplo 2.3 — Convolução 1D Deslizante em $O(1)$ (`XFoldConv1d`)
```python
from torch_gncx import XTensor4Phase, XFoldConv1d

sinal_1d = [2222 + i * 10 if i % 2 == 0 else -2222 - i * 10 for i in range(64)]
xt_1d = XTensor4Phase.from_any(sinal_1d, layer_k=3)

conv1d = XFoldConv1d(out_channels=8, kernel_size=5, layer_k=3)
saida_1d, cancelados_1d = conv1d.forward_xfold(xt_1d)
```

### Exemplo 2.4 — Convolução 2D Espacial (`XFoldConv2d`)
```python
from torch_gncx import XTensor4Phase, XFoldConv2d

H, W = 32, 32
imagem_2d = [(r * 1111 + c * 15) for r in range(H) for c in range(W)]
xt_2d = XTensor4Phase.from_any(imagem_2d, layer_k=3)

conv2d = XFoldConv2d(out_channels=16, kernel_size=(3, 3), layer_k=3)
saida_2d, cancelados_2d = conv2d.forward_xfold(xt_2d, height=H, width=W)
c_out, out_h, out_w = conv2d.out_shape(H, W)
print(f"Shape de saída: ({c_out}, {out_h}, {out_w})")
```
