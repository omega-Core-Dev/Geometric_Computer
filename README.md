# GNC-X (*Geometric Navigational Core Architecture*) — Monorepo & Silicon Incubator

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](https://isocpp.org/)
[![PyTorch Bridge](https://img.shields.io/badge/Buildtool-torch__gncx-EE4C2C.svg)](./03_torch_gncx_buildtool)
[![CUDA T4](https://img.shields.io/badge/CUDA-Tesla_T4_sm__75-76B900.svg)](./04_cuda_t4_colab_benchmark)

Repositório oficial de pesquisa, simulação em C++20 e buildtools de aceleração tensorial/convolucional da arquitetura **GNC-X**, fundamentada no whitepaper *"GNC-X: O Genoma Computacional Espacial (v1.0)"* de **Zaqueu Ribeiro da Costa** ([omega-Core-Dev](https://github.com/omega-Core-Dev)).

---

## Estrutura Organizada do Diretório

Cada projeto, documentação e ferramenta de compilação reside em sua respectiva pasta isolada:

```text
Geometric_Computer/
├── README.md                                   # Índice geral do ecossistema GNC-X
├── LICENSE                                     # Licença Apache 2.0
├── NOTICE                                      # Termo de atribuição autoral (Apache 2.0)
│
├── 01_docs_pesquisa/                           # Artigos, Whitepaper original, Infográfico e Manifesto
│   ├── GNC-X - O Genoma Computacional Espacial (Whitepaper).pdf
│   ├── Gemini_Generated_Image (1).jpeg
│   ├── GNC-X_Especificacao_Formal_e_Simulador.md
│   ├── GNC-X_Manifesto_e_Tratado_de_Pesquisa_Completo.md
│   └── GNC-X_Roadmap_e_Instrucoes_Sessoes_Futuras.md
│
├── 02_gncx_vm_simulator/                       # Máquina Virtual Full-Stack & Incubadora de Silício (C++20)
│   ├── CMakeLists.txt
│   ├── build_and_run.bat                       # Compila e executa a VM completa + Monitor HTML
│   ├── include/
│   │   ├── gncx_sieve.hpp                      # Crivos Diagonais Invariantes D(k, d) = d * R_{k+1}
│   │   ├── gncx_register_matrix.hpp            # Matriz 64-bit 8x8 SWAR + Pipeline Branchless 7 estágios
│   │   ├── gncx_memory.hpp                     # Hierarquia de Memória L0..L3 (100% Bijetiva O(1))
│   │   ├── gncx_gla_disk.hpp                   # Motor de Disco GLA (mmap/Win32 + Paginação Vertical)
│   │   ├── gncx_spatial_db.hpp                 # Banco de Dados Tensorial Espacial + Filtro de Kalman
│   │   ├── gncx_compiler.hpp                   # Compilador Fase 3 + Lógica Quaternária de 4 Fases
│   │   ├── gncx_x_math.hpp                     # Algoritmo X-Folding Linear (37.16x speedup)
│   │   ├── gncx_x_conv.hpp                     # Convoluções Geométricas 4-Phase (XFoldConv1d / XFoldConv2d)
│   │   ├── gncx_visual_display.hpp             # Painel ANSI Colorido + Gerador de Monitor HTML
│   │   └── gncx_genome_vm.hpp                  # Orquestrador Full-Stack da Máquina Virtual GNC-X
│   ├── src/
│   │   └── main.cpp                            # Ponto de entrada da VM e suíte de benchmarks C++20
│   └── bin/
│       ├── gncx_simulator.exe                  # Executável nativo C++20 compilado (-O3 -march=native)
│       ├── gnc_disk.img                        # Imagem persistente do Disco GLA mapeada por geometria
│       └── gncx_visual_monitor.html            # Painel visual gerado pelo runtime C++20
│
└── 03_torch_gncx_buildtool/                    # Biblioteca Dinâmica C++20 + Buildtool Python/PyTorch
    ├── build_dll_and_test.bat                  # Compila gncx_core.dll e roda a suíte Python
    ├── csrc/
    │   └── gncx_torch_c_api.cpp                # Kernel C++20 exportado via ABI C (gncx_core.dll)
    ├── bin/
    │   └── gncx_core.dll                       # Biblioteca compartilhada nativa C++20 (-O3 -march=native)
    ├── torch_gncx/
    │   ├── __init__.py                         # Pacote Python/PyTorch torch_gncx
    │   ├── x_linear.py                         # XTensor4Phase (XOR 11) + Camada Linear XFoldLinear
    │   └── x_conv.py                           # Camadas Convolucionais XFoldConv1d e XFoldConv2d
    └── tests/
        └── benchmark_torch_gncx.py             # Suíte de testes e benchmark Python <-> C++20 DLL

└── 04_cuda_t4_colab_benchmark/                 # Portabilidade GPU (CUDA sm_75) & Colab Tesla T4
    ├── gncx_t4_benchmark.cu                    # Kernels CUDA C++ (K0 Denso, K1 GlobalAtom, K2 ShmAtom, K3 WarpOpt)
    ├── gncx_colab_t4_cuda_benchmark.py         # Profiler de Gargalos PCIe/GDDR6 + Gerador de Plots
    ├── GNC_X_Colab_T4_CUDA_Benchmark.ipynb     # Notebook pronto para Google Colab (GPU T4)
    ├── gncx_t4_cuda_telemetry_report.html      # Dashboard visual de plots de logs (PCIe vs Kernels)
    └── gncx_t4_telemetry_logs.json             # Logs estruturados de escalonamento (1M a 64M elementos)
```

---

## Como Executar Cada Ferramenta

### 1. Simulador Full-Stack da Máquina Virtual (`02_gncx_vm_simulator`)
```powershell
.\02_gncx_vm_simulator\build_and_run.bat
```

### 2. Buildtool C++20 DLL + Biblioteca Python/PyTorch (`03_torch_gncx_buildtool`)
```powershell
.\03_torch_gncx_buildtool\build_dll_and_test.bat
```

### 3. Benchmark de Gargalos de Memória GPU (CUDA / Colab Tesla T4) (`04_cuda_t4_colab_benchmark`)
```powershell
python .\04_cuda_t4_colab_benchmark\gncx_colab_t4_cuda_benchmark.py
```

---

## Instruções para Sessões Futuras (Roadmap)

Consulte o guia detalhado em [`01_docs_pesquisa/GNC-X_Roadmap_e_Instrucoes_Sessoes_Futuras.md`](01_docs_pesquisa/GNC-X_Roadmap_e_Instrucoes_Sessoes_Futuras.md) para continuar a implementação a partir do estado atual:
1. **Passo Futuro 1 (`XFoldResBlock2d`):** Fusão de bloco residual 2D com ativação branchless de 1 ciclo por máscara de fase (`b1`) e skip-connection via `XOR 0b11`.
2. **Passo Futuro 2 (`XFoldAttention`):** Mecanismo de atenção geométrica para Transformers dobrando Queries/Keys nos 10 Crivos Diagonais $\mathcal{D}(k, 0..9)$.
3. **Passo Futuro 3 (O "Trunfo de Refino" — Primologia & Segurança):** Camada final de filtragem primológica e assinatura criptográfica de integridade sobre as layers decimais completas ($L_0..L_k$).

---

## Licença & Atribuição

Distribuído sob a **Apache License, Version 2.0**. Consulte os arquivos [`LICENSE`](LICENSE) e [`NOTICE`](NOTICE) para detalhes completos.

**Copyright © 2026 Zaqueu Ribeiro da Costa ([omega-Core-Dev](https://github.com/omega-Core-Dev))**


