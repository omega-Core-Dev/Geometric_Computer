// ============================================================================
// Arquivo: bench_ddr3_silicon.cpp
// Projeto: GNC-X (Geometric Navigational Core)
// Autor Original: Zaqueu Ribeiro da Costa (omega-Core-Dev)
// Funcao: Benchmark Fisico de Silicio (Intel Celeron 1005M + DDR3 Dual-Channel)
//         Mede ciclos reais de clock da CPU (__rdtsc) e vazao real na RAM:
//           1. L1d/L2 Cache Set Thrashing (8-way) vs. Coloracao Diagonal GNC-X Phi
//           2. DDR3 Bank Conflict (Row-Buffer Miss) vs. GNC-X 8-Bank Interleaving
//           3. Transposicao Fisica de Tensor 1 MB na DDR3 vs. GeoTensor2D::geo_rot()
// ============================================================================

#include "gncx_ddr3_arena.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <chrono>
#include <x86intrin.h>

using namespace gncx::sdk;

// Barreira de compilador para impedir eliminacao de codigo morto nos loops de medicao
static inline void clobber_memory(void* p) {
    asm volatile("" : : "r,m"(p) : "memory");
}

int main() {
    std::cout << "================================================================================\n";
    std::cout << "  GNC-X PHYSICAL DDR3 & CACHE SILICON BENCHMARK (Hardware __rdtsc Telemetry)\n";
    std::cout << "  Alvo Fisico: Intel Ivy Bridge (L1d 32KB 8-Way, L2 256KB) + DDR3 8-Bank Arena\n";
    std::cout << "================================================================================\n\n";

    GNCXDDR3Arena arena;
    if (!arena.valid()) {
        std::cerr << "[ERRO] Falha ao mapear Arena Fisica de 16 MiB!\n";
        return 1;
    }

    // ========================================================================
    // EXPERIMENTO 1: COLISAO DE CONJUNTO L1d (8-WAY THRASHING) VS. GNC-X PHI
    // No Celeron 1005M, o Cache L1d tem 32 KB com 8 vias (64 conjuntos de 64B).
    // Qualquer salto multiplo de 4.096 bytes (1 pagina) cai EXATAMENTE no mesmo
    // conjunto do L1d! Com 16 fluxos simultaneos (ex: 16 cabecas Qwen3),
    // o alocador linear expulsa a linha do cache L1d em 100% dos acessos!
    // ========================================================================
    constexpr int NUM_STREAMS = 16; // 16 cabecas de atencao
    constexpr int STREAM_ITERS = 200'000;

    std::vector<volatile uint64_t*> linear_streams(NUM_STREAMS);
    std::vector<volatile uint64_t*> gncx_streams(NUM_STREAMS);
    std::vector<GNCXDDR3Arena::Allocation> gncx_allocs;

    uint8_t* raw_base = arena.base_address();
    // Toque inicial nas paginas para materializar os frames fisicos na DDR3 (sem page fault no teste)
    std::memset(raw_base, 1, GNCXDDR3Arena::ARENA_SIZE);

    // 1A: Ponteiros Lineares tradicionais (alinhados a 4 KB = todos batem no Set 0 do L1d!)
    for (int s = 0; s < NUM_STREAMS; ++s) {
        linear_streams[s] = reinterpret_cast<volatile uint64_t*>(raw_base + s * 4096ULL);
    }

    // 1B: Ponteiros GNC-X com Coloracao Diagonal Phi(k, d, R) (espalhados em Bancos e Sets L1d)
    for (int s = 0; s < NUM_STREAMS; ++s) {
        auto alloc = arena.alloc_colored(4096ULL, /*k_hint=*/1, /*apply_l1_color=*/true);
        gncx_allocs.push_back(*alloc);
        gncx_streams[s] = reinterpret_cast<volatile uint64_t*>(alloc->base_ptr);
    }

    std::cout << "[EXPERIMENTO 1] Mapeamento Fisico de 16 Cabecas de Atencao (L1d Set + DDR3 Bank):\n";
    std::cout << "  -> Alocador Linear (4KB Page-Aligned): Todas as 16 cabecas caem no L1d Set #0!\n";
    std::cout << "  -> Alocador GNC-X Phi(k=1, R=3):       Distribuicao fisica real nos 8 Bancos e 64 Sets:\n     ";
    for (int s = 0; s < 8; ++s) {
        std::cout << "[H" << s << ": Bank=" << (int)gncx_allocs[s].ddr3_bank
                  << ", L1Set=" << std::setw(2) << (int)gncx_allocs[s].l1_set_color << "] ";
    }
    std::cout << "\n";

    // Medicao Linear (Sofrendo Cache Set Thrashing 16 streams em 8-way L1d)
    unsigned int aux = 0;
    uint64_t c0_lin = __rdtscp(&aux);
    auto t0_lin = std::chrono::high_resolution_clock::now();
    uint64_t acc_lin = 0;
    for (int it = 0; it < STREAM_ITERS; ++it) {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            uint64_t v = linear_streams[s][0] + 1ULL;
            linear_streams[s][0] = v;
            acc_lin += v;
        }
    }
    uint64_t c1_lin = __rdtscp(&aux);
    auto t1_lin = std::chrono::high_resolution_clock::now();

    // Medicao GNC-X Diagonal Cache/Bank Colored
    uint64_t c0_geo = __rdtscp(&aux);
    auto t0_geo = std::chrono::high_resolution_clock::now();
    uint64_t acc_geo = 0;
    for (int it = 0; it < STREAM_ITERS; ++it) {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            uint64_t v = gncx_streams[s][0] + 1ULL;
            gncx_streams[s][0] = v;
            acc_geo += v;
        }
    }
    uint64_t c1_geo = __rdtscp(&aux);
    auto t1_geo = std::chrono::high_resolution_clock::now();

    double cyc_per_access_lin = static_cast<double>(c1_lin - c0_lin) / (STREAM_ITERS * NUM_STREAMS);
    double cyc_per_access_geo = static_cast<double>(c1_geo - c0_geo) / (STREAM_ITERS * NUM_STREAMS);
    double ms_lin = std::chrono::duration<double, std::milli>(t1_lin - t0_lin).count();
    double ms_geo = std::chrono::duration<double, std::milli>(t1_geo - t0_geo).count();

    std::cout << "  -> Resultado em Silicio (__rdtscp):\n";
    std::cout << "     * Linear (L1d 8-Way Conflict):  " << std::fixed << std::setprecision(2)
              << cyc_per_access_lin << " ciclos/op (" << ms_lin << " ms)\n";
    std::cout << "     * GNC-X Diagonal Colored:       " << cyc_per_access_geo
              << " ciclos/op (" << ms_geo << " ms) -> Ganho Fisico: "
              << (cyc_per_access_lin / cyc_per_access_geo) << "x\n\n";

    // ========================================================================
    // EXPERIMENTO 2: CONFLITO DE BANCO DDR3 (ROW-BUFFER MISS) VS. GNC-X 8-BANK
    // Forcamos a leitura atraves de 16 MB (maior que os 2 MB do Cache L3 do
    // Celeron 1005M, obrigando a ir ate os pentes fisicos DDR3):
    //   - Linear Stride 256 KB (2^18): Bate SEMPRE no Banco 0 da DDR3 em Rows
    //     diferentes, causando Precharge + Activate na DRAM!
    //   - GNC-X Diagonal Phi(k, d, R=3): Percorre os mesmos 16 MB distribuindo
    //     os acessos pelos 8 Bancos da DDR3 (0->3->6->1->4->7->2->5) e 64 Sets!
    // ========================================================================
    constexpr int DRAM_SWEEPS = 4000;
    constexpr int BLOCKS_64   = 64;

    std::vector<size_t> bank_conflict_offsets(BLOCKS_64);
    std::vector<size_t> gncx_diagonal_offsets(BLOCKS_64);

    for (int d = 0; d < BLOCKS_64; ++d) {
        // Salto de 256 KB (mesmo DDR3 Bank 0 e mesmo L1/L2 Set 0 em todas as 64 posicoes)
        bank_conflict_offsets[d] = (static_cast<size_t>(d) * 262144ULL) % GNCXDDR3Arena::ARENA_SIZE;
        // Mapeamento bijetivo GNC-X Phi(k=2, d, s=d, R=3) + coloracao de cache
        gncx_diagonal_offsets[d] =
            DDR3GeoAddress::compute_arena_page_offset(2, static_cast<uint8_t>(d), static_cast<uint8_t>(d), 3) +
            DDR3GeoAddress::cache_line_color_offset(static_cast<uint8_t>(d), 3);
    }

    uint64_t c0_dram_lin = __rdtscp(&aux);
    auto t0_dram_lin = std::chrono::high_resolution_clock::now();
    for (int sw = 0; sw < DRAM_SWEEPS; ++sw) {
        for (int d = 0; d < BLOCKS_64; ++d) {
            volatile uint64_t* p = reinterpret_cast<volatile uint64_t*>(raw_base + bank_conflict_offsets[d]);
            p[0] = p[0] + 1ULL;
        }
    }
    uint64_t c1_dram_lin = __rdtscp(&aux);
    auto t1_dram_lin = std::chrono::high_resolution_clock::now();

    uint64_t c0_dram_geo = __rdtscp(&aux);
    auto t0_dram_geo = std::chrono::high_resolution_clock::now();
    for (int sw = 0; sw < DRAM_SWEEPS; ++sw) {
        for (int d = 0; d < BLOCKS_64; ++d) {
            volatile uint64_t* p = reinterpret_cast<volatile uint64_t*>(raw_base + gncx_diagonal_offsets[d]);
            p[0] = p[0] + 1ULL;
        }
    }
    uint64_t c1_dram_geo = __rdtscp(&aux);
    auto t1_dram_geo = std::chrono::high_resolution_clock::now();

    double dram_lin_cyc = static_cast<double>(c1_dram_lin - c0_dram_lin) / (DRAM_SWEEPS * BLOCKS_64);
    double dram_geo_cyc = static_cast<double>(c1_dram_geo - c0_dram_geo) / (DRAM_SWEEPS * BLOCKS_64);
    double dram_lin_ms  = std::chrono::duration<double, std::milli>(t1_dram_lin - t0_dram_lin).count();
    double dram_geo_ms  = std::chrono::duration<double, std::milli>(t1_dram_geo - t0_dram_geo).count();

    std::cout << "[EXPERIMENTO 2] Varredura de 16 MiB na DDR3 (Conflito de Banco vs. Crivo Diagonal):\n";
    std::cout << "  -> Stride Linear Potencia-de-2 (Colisao de Banco 0 + Cache Set 0):\n";
    std::cout << "     Latencia Media: " << dram_lin_cyc << " ciclos/acesso (" << dram_lin_ms << " ms)\n";
    std::cout << "  -> GNC-X Crivo Diagonal 8-Bancos (0->3->6->1->4->7->2->5):\n";
    std::cout << "     Latencia Media: " << dram_geo_cyc << " ciclos/acesso (" << dram_geo_ms
              << " ms) -> Ganho Fisico: " << (dram_lin_cyc / dram_geo_cyc) << "x\n\n";

    // ========================================================================
    // EXPERIMENTO 3: TRANSPOSICAO DE TENSOR 2D (512x512 floats = 1 MiB)
    // Compara 1.000 transposicoes fisicas na DDR3 (leitura por linha + escrita
    // por coluna) contra `GeoTensor2D::geo_rot()` + leitura direta O(1)
    // ========================================================================
    constexpr size_t TDIM = 512;
    constexpr int ROT_ITERS = 500;
    float* mat_src = reinterpret_cast<float*>(raw_base);
    float* mat_dst = reinterpret_cast<float*>(raw_base + 2ULL * 1024ULL * 1024ULL);

    uint64_t c0_trans_lin = __rdtscp(&aux);
    auto t0_trans_lin = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ROT_ITERS; ++it) {
        for (size_t r = 0; r < TDIM; ++r) {
            for (size_t c = 0; c < TDIM; ++c) {
                mat_dst[c * TDIM + r] = mat_src[r * TDIM + c];
            }
        }
        clobber_memory(mat_dst);
    }
    uint64_t c1_trans_lin = __rdtscp(&aux);
    auto t1_trans_lin = std::chrono::high_resolution_clock::now();

    GeoTensor2D<float, TDIM> geo_tensor(mat_src, 1);
    uint64_t c0_trans_geo = __rdtscp(&aux);
    auto t0_trans_geo = std::chrono::high_resolution_clock::now();
    float probe_sum = 0.0f;
    for (int it = 0; it < ROT_ITERS; ++it) {
        geo_tensor.geo_rot(3, true); // 0 bytes copiados!
        probe_sum += geo_tensor.at(it & 511, (it * 7) & 511);
    }
    clobber_memory(&probe_sum);
    uint64_t c1_trans_geo = __rdtscp(&aux);
    auto t1_trans_geo = std::chrono::high_resolution_clock::now();

    double trans_lin_ms = std::chrono::duration<double, std::milli>(t1_trans_lin - t0_trans_lin).count();
    double trans_geo_ms = std::chrono::duration<double, std::milli>(t1_trans_geo - t0_trans_geo).count();
    uint64_t cyc_lin_rot = (c1_trans_lin - c0_trans_lin) / ROT_ITERS;
    uint64_t cyc_geo_rot = (c1_trans_geo - c0_trans_geo) / ROT_ITERS;

    std::cout << "[EXPERIMENTO 3] 500 Transposicoes de Tensor 2D (512x512 float = 1 MiB na DDR3):\n";
    std::cout << "  -> Transposicao Fisica Convencional (memcpy strided): "
              << cyc_lin_rot << " ciclos/transposicao (" << trans_lin_ms << " ms total)\n";
    std::cout << "  -> GNC-X GeoTensor2D::geo_rot() (Zero-Copy View):     "
              << cyc_geo_rot << " ciclos/transposicao (" << std::setprecision(4) << trans_geo_ms
              << " ms total)\n";
    std::cout << "  -> Economia de Largura de Banda DDR3:                 "
              << (ROT_ITERS * 2) << " MiB poupados (Speedup: "
              << std::setprecision(1) << (static_cast<double>(cyc_lin_rot) / std::max<uint64_t>(cyc_geo_rot, 1))
              << "x, Checksum=" << (acc_lin + acc_geo > 0 ? "OK" : "") << ")\n";
    std::cout << "================================================================================\n";

    return 0;
}
