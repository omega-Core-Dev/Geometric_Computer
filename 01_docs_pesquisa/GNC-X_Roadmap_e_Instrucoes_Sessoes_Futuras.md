# GNC-X — Instruções de Continuidade e Roadmap para Sessões Futuras

**Autor da Concepção Original:** Zaqueu Ribeiro da Costa  
**Arquitetura:** *GNC-X (Geometric Navigational Core Architecture)*  
**Status Atual:** Monorepo Organizado | VM C++20 Validada | Buildtool `torch_gncx` (`Linear`, `Conv1d`, `Conv2d`) Validado com 100% de Exatidão Bit-a-Bit.

---

## 1. Contexto Rápido de Ambiente & Toolchain (Para Inicialização Imediata)

1. **Compilador C++20 Nativo (LLVM-MinGW UCRT 64-bit):**
   * Caminho do `clang++.exe`:
     ```text
     C:\Users\orieb\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20260908-ucrt-x86_64\bin\clang++.exe
     ```
   * Sempre compilar com `-std=c++20 -O3 -march=native -static` para evitar dependências externas de runtime DLL.
2. **Estrutura do Monorepo (`c:\Users\orieb\Documents\Geometric_Computer`):**
   * `01_docs_pesquisa\` — Whitepaper PDF original, infográfico JPEG, Especificação Formal, Manifesto/Tratado de Pesquisa Completo e este Guia de Sessões Futuras.
   * `02_gncx_vm_simulator\` — Máquina Virtual Full-Stack C++20 (`Sistema -> Processador -> Memória -> Disco GLA -> Banco de Dados Tensorial` + Compilador Fase 3 + Display Visual). Executar via `.\02_gncx_vm_simulator\build_and_run.bat`.
   * `03_torch_gncx_buildtool\` — Biblioteca dinâmica C++20 (`bin\gncx_core.dll`) + pacote Python/PyTorch (`torch_gncx`) com `XTensor4Phase`, `XFoldLinear`, `XFoldConv1d` e `XFoldConv2d`. Executar via `.\03_torch_gncx_buildtool\build_dll_and_test.bat`.

---

## 2. Invariantes Matemáticos Já Consolidados (Não Alterar)

* **Crivos Diagonais Invariantes (Base 10 Completa $d \in \{0..9\}$ em todas as Layers $L_k$):**
  $$\mathcal{D}(k, d) = d \cdot R_{k+1}, \quad R_{k+1} = \frac{10^{k+1}-1}{9}$$
  *(Nota fundamental de projeto: Todas as 10 diagonais decimais $d \in \{0..9\}$ e todas as layers $L_0..L_k$ operam acopladas simultaneamente; crivos primos são reservados exclusivamente para a última camada de refino).*
* **Lógica Quaternária de 4 Fases em `X` (`CompactXToken` — 4 bytes):**
  $$X = s_1 \cdot \mathcal{D}(k, d) + s_0 \cdot |\Delta|, \quad s_i = 2b_i - 1 \in \{-1, +1\}$$
  * `11` (`++`): $+\mathcal{D}(k, d) + |\Delta|$ (Quadrante I)
  * `10` (`+-`): $+\mathcal{D}(k, d) - |\Delta|$ (Quadrante IV)
  * `01` (`-+`): $-\mathcal{D}(k, d) + |\Delta|$ (Quadrante II)
  * `00` (`--`): $-\mathcal{D}(k, d) - |\Delta|$ (Quadrante III)
* **Negação Algébrica Zero-Carry em 1 Ciclo:**
  $$-X \iff \text{phase\_2bit} \oplus \text{0b11}$$
* **Benchmarks Empíricos Já Aferidos (`benchmark_torch_gncx.py`):**
  * `XFoldLinear` (`32M MACs`): **37,16x mais rápido** (`4,883 ms` vs `181,452 ms` — `100% IDÊNTICO`).
  * `XFoldConv1d` (`99,17M MACs`): **11,92x mais rápido** (`23,831 ms` vs `283,953 ms` — `100% IDÊNTICO`).
  * `XFoldConv2d` (`98,00M MACs`): **12,21x mais rápido** (`29,172 ms` vs `356,232 ms` — `100% IDÊNTICO`).

---

## 3. Instruções Detalhadas para os Próximos Passos (Sessões Futuras)

### Passo Futuro 1: Bloco Residual Geométrico (`XFoldResBlock2d`) & Ativação Branchless por Fase
* **Objetivo:** Construir em `03_torch_gncx_buildtool` o primeiro bloco residual completo de visão computacional sem branches (`if/else`) e sem multiplicações densas.
* **Como Implementar:**
  1. **Ativação `XPhaseReLU` / `XPhaseLeaky`:** No formato 4-Phase (`b1 b0`), o bit mais significativo `b1` já informa o sinal macroscópico do pivô diagonal! Uma ativação ReLU ou Leaky-ReLU opera em **1 ciclo bitwise** diretamente sobre `phase_2bit` e `abs_delta` usando máscara aritmética `mask = -(int64_t)(b1)`.
  2. **Skip-Connection Subtrativa/Aditiva via `XOR 0b11`:** Permitir que o ramo residual some ou subtraia a identidade $X$ invertendo apenas os 2 bits de fase (`XOR 0b11`) antes da fusão com a saída de `XFoldConv2d`.
  3. **Arquivos Alvo:**
     * Adicionar `gncx_xfold_resblock2d_forward` em `03_torch_gncx_buildtool/csrc/gncx_torch_c_api.cpp`.
     * Criar `XFoldResBlock2d` em `03_torch_gncx_buildtool/torch_gncx/x_conv.py`.

---

### Passo Futuro 2: Mecanismo de Atenção Geométrica em `X` (`XFoldAttention`)
* **Objetivo:** Levar o dobramento dos 10 Crivos Diagonais ($\mathcal{D}(k, 0..9)$) para o gargalo quadrático $O(N^2)$ dos Transformers ($Q \cdot K^T$).
* **Como Implementar:**
  1. **Decomposição 4-Phase de Queries ($Q$) e Keys ($K$):** Codificar os vetores de *Keys* e *Values* como `XTensor4Phase`.
  2. **Agrupamento por Constelação Diagonal:** Em vez de calcular $N \times N$ produtos escalares densos, agrupar as *Keys* pelas 10 âncoras diagonais $\mathcal{D}(k, d)$ e aniquilar pares de fases opostas (`11` vs `00` e `10` vs `01`) no centro `0` do X antes de projetar contra $Q$.
  3. **Arquivos Alvo:**
     * Criar `02_gncx_vm_simulator/include/gncx_x_attention.hpp`.
     * Criar `03_torch_gncx_buildtool/torch_gncx/x_attention.py` e expor em `__init__.py`.

---

### Passo Futuro 3: O "Trunfo de Refino" — Camada Final de Primologia Geométrica & Segurança Criptográfica
* **Objetivo:** Implementar a diretriz arquitetural definida pelo autor: **reservar a Primologia & Segurança como a última camada de refino sobre as layers decimais completas ($L_0..L_k$)**.
* **Como Implementar:**
  1. **Filtro de Coordenadas Primas sobre o Crivo Decimal Completo:** Enquanto as layers decimais $L_0..L_k$ garantem cobertura 100% bijetiva e transporte em $O(1)$ para qualquer inteiro $X$, a Camada de Refino Primológico inspeciona o par $(\mathcal{D}(k, d), \Delta)$ usando as propriedades modulares dos Repunits ($\gcd(d, 10) = 1$ nos setores $d \in \{1, 3, 7, 9\}$ combinados com resíduos coprimos).
  2. **Assinatura de Integridade Geométrica (Zero-Overhead Hardware MAC):** Gerar um *tag* criptográfico/verificador de integridade diretamente a partir da distribuição das 4 fases (`11, 10, 01, 00`) e das órbitas primas no Disco GLA e no `XTensor4Phase`, detectando qualquer adulteração de memória ou peso neural em $O(1)$.
  3. **Arquivos Alvo:**
     * Criar `02_gncx_vm_simulator/include/gncx_prime_refinement.hpp`.
     * Integrar como estágio final opcional no `GNCXGenomeVM` (`02_gncx_vm_simulator/include/gncx_genome_vm.hpp`) e na DLL `gncx_core.dll`.
