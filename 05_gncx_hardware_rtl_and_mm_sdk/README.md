# GNC-X (Geometric Navigational Core Architecture)
**O Genoma Computacional Espacial — Do Buddy Allocator Linear ao Mapa Geométrico Unificado e Silício PIM**

**Autor da Pesquisa Original:** Zaqueu Ribeiro da Costa (`omega-Core-Dev`)  
**Repositório Oficial:** `https://github.com/omega-Core-Dev/Geometric_Computer`

---

## 1. Estrutura Organizacional do Módulo (`05_gncx_hardware_rtl_and_mm_sdk`)

```text
05_gncx_hardware_rtl_and_mm_sdk/
├── README.md                                           # Guia técnico e resultados homologados do módulo 05
├── include/                                            # Cabeçalhos C++20 do Gerenciador de Memória Geométrica
│   ├── gncx_geometric_mm.hpp                           # Implementação base 8x8 (64 frames, Crivo D(k,d) e Φ)
│   ├── gncx_mm_64x64.hpp                               # Expansão 64x64 (4.096 frames = 16 MiB), R^-1 mod 2^m O(1),
│   │                                                   # Alocador Diagonal SWAR 64-bit e Simulador Buddy/Radix PTW
│   └── gncx_ddr3_arena.hpp                             # SDK e Gerenciador de Arena Física 16 MiB (2MB HugePage),
│                                                       # Coloração de 8 Bancos DDR3 + 64 Sets L1d e GeoTensor2D
├── lib/                                                # Biblioteca Dinâmica de Desenvolvimento SDK (C/C++/FFI)
│   ├── gncx_sdk_lib.cpp                                # Exportação C ABI (`gncx_sdk_alloc`, `gncx_sdk_free`)
│   └── libgncx_mm.so                                   # Binário compartilhado para linkagem direta (-lgncx_mm)
├── benchmark/                                          # Suíte Científica C++20 (Passos 1, 2 e Silício DDR3)
│   ├── bench_buddy_vs_gncx.cpp                         # Validação Brute-Force (8x8 e 64x64) + 1M alloc/free
│   ├── bench_ddr3_silicon.cpp                          # Benchmark Físico (__rdtscp) na DDR3 + Cache L1d/L2
│   ├── run_bench.sh                                    # Script do benchmark Buddy vs GNC-X
│   └── run_ddr3_bench.sh                               # Script de compilação da SDK + benchmark físico DDR3
└── verilog/                                            # Hardware RTL Sintetizável (IEEE 1364) e Simulador Ciclo-Exato
    ├── gncx_phi_unit.v                                 # Unidades combinacionais Φ (direta) e Φ^-1 (inversa) em 0 ciclos
    ├── gncx_matrix_ram_8x8.v                           # Matriz SRAM 8x8 Tri-Modal + geo.rot + Domain Keys + PIM X-Folding
    ├── gncx_gmmu_rocc.v                                # Coprocessador RISC-V RoCC (geo.addr, geo.ld, geo.rot, geo.st)
    ├── tb_gncx_pim.v                                   # Testbench de verificação formal em Verilog padrão
    ├── sim_gncx_rtl.cpp                                # Simulador RTL Ciclo-Exato C++20 (1M Ops Monte Carlo + VCD)
    ├── sim_gncx_rtl                                    # Binário do simulador RTL
    ├── gncx_pim_waveform.vcd                           # Forma de onda IEEE 1364 gerada (compatível GTKWave / Surfer)
    └── run_sim.sh                                      # Script de execução da simulação RTL
```

> **Integração com o Motor GNC-X LLM:**  
> O motor de inferência local multi-modelo (Gemma 2B, Gemma 4 E2B e Qwen3 1.7B em C++20 + LuaJIT) reside separadamente em `/home/oriebir/Downloads/arquivos/model2` e consome o gerenciador geométrico (`csrc/gncx_geometric_mm.hpp`) para alocação e telemetria real do KV-Cache (`/geometria`).

---

## 2. Resumo dos Resultados Homologados

### A. Verificação de Hardware RTL (`verilog/run_sim.sh`)
- **Bijeção $\Phi \leftrightarrow \Phi^{-1}$ em Hardware:** `128 / 128` órbitas testadas com `0 colisões` (explorando a propriedade $R^{-1} \equiv R \pmod 8$ para $R \in \{1, 3, 5, 7\}$).
- **Barramento Tri-Modal (256 bits/ciclo):** Leitura em 1 ciclo de Linha Horizontal (`2'b00`), Coluna Vertical GLA (`2'b01`) e Crivo Diagonal $D(k, d)$ (`2'b10`).
- **Rotação `geo.rot` em 0 Ciclos:** Em 1.000.000 de operações aleatórias, as rotações de visão matricial economizaram **32.059.904 ciclos de barramento** (`0` bits movidos contra `memcpy` linear).
- **Domain Protection Keys (4 bits):** `124.859` colisões de interseção algébrica entre processos no Single Address Space interceptadas e bloqueadas em 1 ciclo (`domain_fault = 1`).
- **PIM X-Folding Nativo:** Cancelamento simétrico de 4 fases executado diretamente nas portas de saída da malha de memória.

### B. Benchmark C++20: Buddy Allocator vs. GNC-X Diagonal (`benchmark/run_bench.sh`)
- **Validação Brute-Force $64 \times 64$ (16 MiB):** `524.288 / 524.288` estados bijetivos validados em `4,03 ms` com `0` colisões e inversos modulares exatos $R^{-1} \pmod{64}$ via Newton-Raphson $\mathcal{O}(1)$.
- **Stress Test de 1.000.000 de `alloc/free` Aleatórios:**
  - **Buddy Allocator Linear:** `423,71 ms` (`2,36 Mops/s`) | Pico de Fragmentação Externa: `27,65%`
  - **GNC-X Diagonal SWAR 64-bit:** `59,29 ms` (`16,87 Mops/s`) | Fragmentação Geométrica: `0,11% – 2,35%` (**Speedup de 7,15x**)

### C. Benchmark de Silício Real (`__rdtscp`) na DDR3 e Cache L1d (`benchmark/run_ddr3_bench.sh`)
Executado em hardware real **Intel Celeron 1005M (1.90 GHz, L1d 32 KB 8-Way, 6 GB DDR3 Dual-Channel)** usando `GNCXDDR3Arena` (`include/gncx_ddr3_arena.hpp`) e `libgncx_mm.so` (`lib/gncx_sdk_lib.cpp`) com `MADV_HUGEPAGE` (2 MB) para destravar os bits físicos `20..0`:
1. **Teste 1 — 16 Cabeças de Atenção LLM na Cache L1d 8-Way (`bits 11..6`):**
   - `posix_memalign` Linear (64 KB stride -> Set 0): `17,74 ciclos/op` (`29,94 ms`) — *Cache-Set Thrashing* severo.
   - `GNCXDDR3Arena` Colorida via $\Phi(k, d, R)$: **`2,49 ciclos/op` (`4,21 ms`) — Speedup de 7,12x**.
2. **Teste 2 — Varredura de 16 MiB na DDR3 (`bits 16..14` — Conflito de Banco vs. Crivo Diagonal 8-Bank):**
   - Stride Linear de Mesmo Banco DDR3 (Row-Buffer Miss `tRC`): `98,77 ciclos/acesso` (`13,34 ms`).
   - Crivo Diagonal GNC-X (8-Bank Interleaving $R=3$): **`2,56 ciclos/acesso` (`0,35 ms`) — Speedup de 38,53x**.
3. **Teste 3 — 500 Transposições de Tensores 2D (`512x512 float` = 1 MiB):**
   - Transposição Física Linear (`memcpy`): `3.315.474 ciclos/rot` (`874,49 ms`).
   - `GeoTensor2D::geo_rot` (Inversão Topológica de Coordenadas): **`14 ciclos/rot` (`0,0039 ms`) — 1.000 MiB de banda DDR3 economizados**.

### D. Impacto Real no Motor GNC-X LLM (`/home/oriebir/Downloads/arquivos/model2`)
Ao acoplar o alocador colorido de **8 Bancos DDR3 + 64 Conjuntos L1d** (`gncx_ram_alloc_colored`), o **ThreadPool Lock-Free Spin-Then-Park** e a **Fusão Vetorial SSSE3 (`_mm_hadd_epi32`)** nos blocos `Q4_K`/`Q6_K` com cache de época de quantização no modelo **Qwen3 1.7B Instruct (28 Camadas)**:
- **Tempo de Boot + Pinning RAM Colorido:** Caiu de `2.659 ms` para **`618,7 ms` (4,3x mais rápido)**.
- **Velocidade de Prefill:** Atingiu **`4,27 tok/s`**.
- **Velocidade de Decode Autorregressivo:** Saltou para **`3,28 tok/s` (sem CoT)** e **`3,32 tok/s` (com CoT de 229 tokens)** em um processador Dual-Core de 2013 sem AVX!

---

## 3. Como Executar os Testes

```bash
# 1. Executar a Simulação RTL de Hardware e gerar Waveform (.vcd)
./05_gncx_hardware_rtl_and_mm_sdk/verilog/run_sim.sh

# 2. Executar a Validação 64x64 (524k estados) e Benchmark 1M Ops (Buddy vs GNC-X)
./05_gncx_hardware_rtl_and_mm_sdk/benchmark/run_bench.sh

# 3. Compilar a SDK (libgncx_mm.so) e Executar o Benchmark Físico (__rdtscp) na DDR3 e L1d
./05_gncx_hardware_rtl_and_mm_sdk/benchmark/run_ddr3_bench.sh
```
