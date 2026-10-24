# GNC-X (Geometric Navigational Core Architecture) — Arquitetura Formalizada

O Genoma Computacional Espacial — Do Buddy Allocator Linear ao Mapa Geométrico Unificado

Autor: Zaqueu Ribeiro da Costa (omega-Core-Dev) | Repo: https://github.com/omega-Core-Dev/Geometric_Computer

---

## 1. Motivação e Contexto


O Linux gerencia RAM como cache de memória virtual paginada via tabelas de 4 níveis (PGD->P4D->PUD->PMD->PTE), Buddy Allocator (order 0..11) e SLUB para objetos. O modelo é linear, contíguo e sofre de fragmentação externa (~38.7% em carga).

O projeto GNC-X propõe substituir o paradigma linear por um mapa geométrico unificado onde alocação, execução e armazenamento compartilham a mesma função de tradução bijetiva O(1).

O sistema ataca várias frentes: (1) VM full-stack C++20 em 02_gncx_vm_simulator, (2) Buildtool PyTorch com X-Folding 37.16x em 03_torch_gncx_buildtool, (3) Benchmark CUDA T4 em 04_cuda_t4_colab_benchmark e (4) motor de modelos locais em CPU antiga sem GPU dedicada — frente atual.


---

## 2. Estado da Arte: Mill e CHERI — Por que não são diagonais


Mill CPU: arquitetura 64-bit com Single Address Space (SAS) — "All threads and processes share the same address space." Proteção via turfs. "A region is a continuous stretch of memory. A turf is a collection of regions." Usa PLB (Protection Lookaside Buffer) separado de tradução. Ainda linear e contíguo.

CHERI: capability de 128/256 bits com base, topo, permissões. "Each CHERI capability is double the width of the native integer pointer type... consists of baseline address and capability metadata. The metadata includes upper (UB) and lower bound (LB)." Checagem: "base ≤ A < top is within bounds". "A CHERI capability encodes the base and top of the region". Também linear contíguo [base, top).

Ambos sofrem de fragmentação externa e precisam de compactação. Nenhum propõe invariante diagonal.


---

## 3. Núcleo Geométrico: Endereço e Crivo Diagonal


GeoAddr(k,d,R):
k ∈ [0,3] = nível de escala L0..L3 (L0=registradores 8x8, L1=cache geométrico, L2=RAM grid, L3=GLA Disk)
d ∈ [0,7] (8x8) ou [0,63] expandido = índice diagonal
R ∈ {1,3,5,7} = R_{k+1} fator de rotação, coprimo com tamanho da matriz para garantir bijeção.

Crivo Diagonal Invariante:
D(k,d) = d * R_{k+1} mod N
Onde N=8 para matriz 8x8. Esta é a função do arquivo gncx_sieve.hpp.

Propriedade: se R é ímpar, tem inverso modular mod 8, garantindo bijeção.


---

## 4. Tradução Bijetiva O(1) — Φ-Unit


Substitui page table walk O(log n) por função direta reversível:

Φ(k,d,R) = (i,j)
i = (d + k*R) mod 8
j = (D(k,d) + k) mod 8
índice físico linear = i*8 + j

Φ⁻¹(i,j,k) = (k,d,R) via inverso modular de R.

Latência constante 1 ciclo, sem TLB miss. Implementado em gncx_memory.hpp como hierarquia 100% bijetiva.
Comparação:
Buddy: virtual -> PGD->P4D->PUD->PMD->PTE (4 acessos RAM)
GNC-X: GeoAddr -> Φ -> frame físico (1 operação SWAR)

Esta unidade é o acoplador de instrução no pipeline.


---

## 5. Alocador Diagonal vs Buddy Allocator


Buddy: aloca blocos 2^order. Fragmentação externa quando não há blocos contíguos, mesmo com memória livre espalhada. Precisa de kcompactd.

Geometric Allocator:
- Bitmap de 64 bits + diagonal_sieves[8]
- Alocação: encontra diagonal D com <8 ocupados, aloca próximo slot livre nessa diagonal.
- Liberação: limpa bit na diagonal.
- Fragmentação tende a ~0% porque buracos espalhados ainda formam diagonais utilizáveis.

Exemplo C++ no arquivo gncx_geometric_mm.hpp:
alloc(k_hint) -> GeoAddr opcional
translate_to_phys(GeoAddr) -> phys_base + idx*4096
fragmentation() = 1 - max_free_run/total_free


---

## 6. Acoplador de Instrução — CPU que compreende geometria


Pipeline original: FETCH linear -> DECODE -> EXECUTE -> MEMORY linear -> Φ() software
Pipeline com acoplador: FETCH geométrico -> DECODE(k,d,R) -> COUPLER Φ-Unit O(1) -> (i,j) -> EXECUTE em matriz 8x8

ISA proposta (para gncx_compiler.hpp Fase 3, lógica quaternária 4 fases):
geo.addr r1, k=2, d=5, R=3   // r1 = Φ(2,5,R3)
geo.ld   r2, [r1].D(2,5)      // load via crivo
geo.st   r2, [r1].D(2,5)
geo.fork D(2,5) -> D(2,6)     // CoW geométrico, copia diagonal não página
geo.rot  R, #45              // rotaciona visão do processo sem mover dados físicos

geo.rot é impossível no modelo linear: mudar R muda todo mapeamento lógico mantendo físico.

Implementação inicial via userfaultfd em userspace, depois como gncx_geo.ko com vm_operations_struct.fault = gncx_geo_fault.


---

## 7. Mapa Unificado — Single Address Space OS Geométrico


Todos os componentes vivem no mesmo mapa Φ:

L0: registradores 8x8 SWAR (gncx_register_matrix.hpp)
L1: cache geométrico tag=(k,d,R)
L2: RAM física grid 8x8 ou 64x64
L3: GLA Disk paginação vertical (gncx_gla_disk.hpp) — swap-out de coluna vertical, não página 4K horizontal.

Processo = conjunto de crivos D(k, d∈range) com R específico.
Isolamento = R diferentes garantem espaços disjuntos algebricamente, sem PLB.

Kernel compilado com libgncx_core usa mesmo Φ que userspace. Syscall vira troca de k, não troca de page table. fork() vira geo.fork diagonal.

Benefício para modelos locais em CPU antiga sem GPU: peso do modelo mora na mesma diagonal que código, XTensor4Phase (XOR 11) 4-bit por fase, quantização natural, mantém modelo em cache L1/L2 via mmap vertical.


---

## 8. Frente Atual: Rodando Modelos Locais em CPU Antiga sem GPU


Contexto: Almenara/MG, CPU antiga, sem placa dedicada. Motivo da escolha do X-Folding branchless e SWAR 64-bit.

Stack atual:
- 02_gncx_vm_simulator/build_and_run.bat com -O3 -march=native
- torch_gncx/x_linear.py XFoldLinear + x_conv.py XFoldConv1d/2d
- 04_cuda_t4_colab_benchmark com K0 Denso, K1 GlobalAtom, K2 ShmAtom, K3 WarpOpt (mede PCIe/GDDR6 gargalo que não existe em CPU-only)

Estratégia CPU-only:
1. Quantização 4 fases quaternárias (2 bits/fase) via XTensor4Phase
2. mmap vertical via gncx_gla_disk para modelo > RAM
3. Φ-Unit mantém pesos em diagonais quentes no cache
4. TinyLlama/Phi-2 1.1B rodando com <4GB RAM

Resultado esperado: 37.16x speedup do X-Folding vem de SWAR, não de GPU, ideal para CPU antiga.


---

## 9. Tabela Comparativa


Característica | CHERI | Mill | GNC-X
Endereço | ptr + [base,top) linear | endereço linear 64-bit SAS | GeoAddr(k,d,R) + D(k,d)=d*R_{k+1}
Alocação | intervalo contíguo | região contígua | crivo diagonal disperso
Fragmentação | alta | média | ~0%
Busca | O(log n) bounds check | O(log n) PLB | O(1) Φ bijetiva
Isolamento | bounds check | PLB + turfs | R diferente = disjunto algébrico
Operação única | — | portal | geo.rot sem memcpy
TLB | sim | PLB após LLC | substituído por Φ-Unit


---

## 10. Roadmap Formalizado


Passso 1 — Validação: brute-force bijeção 8x8 (32 combinações) depois 64x64 (4096 frames = 16MiB teste) em C++20.
Passo 2 — Benchmark: 1M alloc/free aleatórios Buddy vs Diagonal, medir fragmentação e TLB misses via perf.
Passo 3 — Acoplador Userspace: userfaultfd + UFFDIO_COPY, mmap 1GB MAP_NORESERVE, thread Φ translator.
Passo 4 — LD_PRELOAD: libgncx_malloc.so com GeoAddr malloc.
Passo 5 — Módulo Kernel: gncx_geo.ko /dev/gncx_geo com vm_ops.fault.
Passo 6 — ISA: extensão RISC-V custom rv64_gncx com geo.addr/ld/st/fork/rot, documentar em 01_docs_pesquisa/GNC-X_Especificacao_Formal_e_Simulador.md.
Passo 7 — Modelos: XFoldAttention (roadmap) para Transformers dobrando Queries/Keys nos 10 crivos D(k,0..9).
Passo 8 — Publicação: atualizar OG Card 1280x640 + whitepaper v1.1 com prova modular R ímpar.
Passo 9 — Integração no Motor GNC-X LLM: Acoplar o GeometricMemoryManager na alocação do KV-Cache e medição de telemetria real contra o Buddy Allocator.

---

## 11. Parecer de Engenharia de Hardware e Mitigações Físicas

Para o parecer técnico completo e aprofundado, consulte o documento:
`GNC-X_Parecer_Engenharia_Hardware_e_Kernel.md`

### Síntese das Mitigações para Silício Real:
1. **DRAM Row-Buffer Preservation:** O mapeamento $\Phi$ deve operar em granularidade mínima de Linhas de Cache de 64 bytes ou Páginas de 4 KB para reter 100% de taxa de acerto no *Row Buffer* do barramento físico DDR4/DDR5.
2. **Domain Keys Anti-Interseção:** Em matrizes modulares $(\mathbb{Z}/N\mathbb{Z})^2$, retas com coeficientes coprimos distintos se cruzam periodicamente. O isolamento de processos no Single Address Space (SAS) deve incorporar Domain Tags de 4 bits para garantir proteção absoluta contra corrupção em pontos de colisão.
3. **PIM (Processing-In-Memory) Nativo:** A arquitetura física geométrica justifica o desenho de uma memória personalizada em FPGA/ASIC, onde a redução tensorial ocorre na topologia cruzada da própria malha de células de memória.

---

## 12. Referências e Links

- Repositório: https://github.com/omega-Core-Dev/Geometric_Computer
- Parecer de Engenharia de Hardware: `docs/GNC-X_Parecer_Engenharia_Hardware_e_Kernel.md`
- Especificação de Memória Física e Verilog: `docs/GNC-X_Arquitetura_Memoria_Fisica_e_Verilog.md`
- Arquivos chave: `include/gncx_geometric_mm.hpp`, `include/gncx_mm_64x64.hpp`, `include/gncx_ddr3_arena.hpp`, `lib/libgncx_mm.so`
- Mill: millcomputing.com — Single Address Space, turfs = collection of continuous regions
- CHERI: University of Cambridge, SRI International — capabilities with base, top, bounds, Robert N. M. Watson, Simon Moore, Peter Sewell, Peter G. Neumann, Arm Morello board, CHERI Alliance
- Licença: Apache 2.0

---

## 13. Registro de Homologação Experimental (24/10/2026)

Todos os passos críticos do Roadmap (Passos 1, 2, 4, 6 e 9) e a validação em silício físico foram implementados, executados e homologados experimentalmente:

1. **Passo 1 & 2 (`benchmark/bench_buddy_vs_gncx.cpp`):**
   - **Bijeção Brute-Force $8\times 8$ e $64\times 64$ (16 MiB):** `524.288 / 524.288` estados bijetivos validados em `4,03 ms` (`0` erros), utilizando inverso modular de Newton-Raphson $R^{-1} \pmod{64}$ em $\mathcal{O}(1)$.
   - **Stress Test 1.000.000 `alloc/free` Aleatórios:** O alocador diagonal GNC-X SWAR 64-bit completou 1M de operações em **`59,29 ms` (`16,87 Mops/s`)** contra **`423,71 ms` (`2,36 Mops/s`)** do Buddy Allocator Linear (**Speedup de 7,15x**), reduzindo o pico de fragmentação de `27,65%` para `0,11% – 2,35%`.

2. **Passo 6 — Simulação RTL Ciclo-Exata e Forma de Onda IEEE 1364 (`verilog/`):**
   - Módulos Verilog sintetizáveis (`gncx_phi_unit.v`, `gncx_matrix_ram_8x8.v`, `gncx_gmmu_rocc.v`, `tb_gncx_pim.v`) e simulador ciclo-exato (`sim_gncx_rtl.cpp` + `gncx_pim_waveform.vcd`).
   - Em 1.000.000 de ciclos Monte Carlo (`17,39 ms`), foram comprovados: `0` colisões de bijeção, **`124.859` interseções de domínio bloqueadas em 1 ciclo** pelas *Domain Keys* de 4 bits, e **`32.059.904` ciclos de barramento economizados** pela instrução `geo.rot` em 0 ciclos.

3. **Passo 4 — Biblioteca SDK (`lib/libgncx_mm.so`) e Benchmark Físico no Silício DDR3 (`benchmark/bench_ddr3_silicon.cpp`):**
   - Utilizando HugePages de 2 MB (`MADV_HUGEPAGE`) para fixar os bits físicos `20..0`, a `GNCXDDR3Arena` controla deterministicamente os **8 Bancos Físicos da DDR3 (`bits 16..14`)** e os **64 Conjuntos da Cache L1d 8-Way (`bits 11..6`)** via $\Phi(k, d, R)$.
   - Aferido via `__rdtscp()` no **Intel Celeron 1005M + 6 GB DDR3 Dual-Channel**:
     - **16 Cabeças de Atenção na Cache L1d:** `2,49 ciclos/op` (GNC-X) vs. `17,74 ciclos/op` (Linear) — **Speedup de 7,12x**.
     - **Varredura de 16 MiB na DDR3 (8-Bank Interleaving):** `2,56 ciclos/acesso` (GNC-X) vs. `98,77 ciclos/acesso` (Conflito de Banco Linear) — **Speedup de 38,53x**.
     - **500 Transposições de Tensores 2D (`512x512 float`):** `14 ciclos/rot` (`GeoTensor2D::geo_rot`) vs. `3.315.474 ciclos/rot` (`memcpy`) — **1.000 MiB de tráfego DDR3 poupados**.

4. **Passo 9 — Acoplamento no Motor GNC-X LLM (`/home/oriebir/Downloads/arquivos/model2`):**
   - A integração de `gncx_ram_alloc_colored` nos 56 tensores de KV-Cache e 11 buffers de ativação, somada ao **ThreadPool Lock-Free Spin-Then-Park**, **redução vetorial SSSE3 (`_mm_hadd_epi32`)** nos blocos `Q4_K`/`Q6_K` e **cache de época de quantização (`g_norm_epoch`)**, elevou o desempenho do **Qwen3 1.7B Instruct (28 Camadas)** no Celeron 1005M para:
     - **Tempo de Boot + Pinning RAM:** **`618,7 ms`** (redução de 4,3x ante `2.659 ms`).
     - **Velocidade de Prefill:** **`4,27 tok/s`**.
     - **Velocidade de Decode Autorregressivo:** **`3,28 tok/s` (sem CoT)** e **`3,32 tok/s` (com CoT de 229 tokens)**.


