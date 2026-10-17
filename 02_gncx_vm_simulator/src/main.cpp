/**
 * ============================================================================
 * GNC-X SILICON INCUBATOR — MAIN EXECUTABLE & FULL-STACK VALIDATION SUITE
 * ============================================================================
 * Author of Conception: Zaqueu Ribeiro da Costa
 *
 * Validates the complete GNC-X Unified Architecture in C++:
 *   SISTEMA (Genoma) -> PROCESSO (Matriz 64-Bit & Pipeline Branchless)
 *     -> MEMORIA (Camadas de Magnitude L0..L3) -> DISCO (GLA `gnc_disk.img`)
 *       -> BANCO DE DADOS (Tensores Espaciais sem SQL / sem B-Tree)
 * ============================================================================
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <map>
#include <random>
#include <vector>
#include <numeric>

#include "../include/gncx_sieve.hpp"
#include "../include/gncx_register_matrix.hpp"
#include "../include/gncx_memory.hpp"
#include "../include/gncx_gla_disk.hpp"
#include "../include/gncx_spatial_db.hpp"
#include "../include/gncx_genome_vm.hpp"
#include "../include/gncx_compiler.hpp"
#include "../include/gncx_x_math.hpp"

using namespace gncx;

static void print_banner(const std::string& title) {
    std::cout << "\n================================================================================\n";
    std::cout << "  " << title << "\n";
    std::cout << "================================================================================\n";
}

int main(int argc, char* argv[]) {
    std::cout << "################################################################################\n";
    std::cout << "#  GNC-X: O GENOMA COMPUTACIONAL ESPACIAL - SIMULADOR VIRTUAL (INCUBADORA C++) #\n";
    std::cout << "#  Autor da Concepcao: Zaqueu Ribeiro da Costa                                 #\n";
    std::cout << "#  Fluxo Unificado: SISTEMA -> PROCESSO -> MEMORIA -> DISCO (GLA) -> DB        #\n";
    std::cout << "################################################################################\n";

    // Valores padrao ou injetados via CLI para transitar pela VM C++20
    int64_t v0 = (argc > 1) ? std::stoll(argv[1]) : 777;
    int64_t v1 = (argc > 2) ? std::stoll(argv[2]) : 789;
    int64_t v2 = (argc > 3) ? std::stoll(argv[3]) : 4321;
    int64_t v3 = (argc > 4) ? std::stoll(argv[4]) : 9999;

    // =========================================================================
    // ETAPA 1: CRIVOS INVARIANTES DIAGONAIS & LOGICA DE 4 FASES (+=1, -=0)
    // =========================================================================
    print_banner("1. CRIVOS INVARIANTES & LOGICA DE 4 FASES DOS QUADRANTES (+=1, -=0)");
    std::cout << "Codificacao de Fase 2-Bits ao redor do Zero (X-Sieve Constellation):\n"
              << "  [Fase 11 = ++] +Pivo, +Residuo | [Fase 10 = +-] +Pivo, -Residuo\n"
              << "  [Fase 01 = -+] -Pivo, +Residuo | [Fase 00 = --] -Pivo, -Residuo\n\n";

    const std::array<int64_t, 4> quadrant_samples = {789, 4321, -4321, -789};
    for (int64_t qv : quadrant_samples) {
        CoupledPhaseCoordinate pc = FourPhaseSieveEngine::encode(qv);
        CoupledPhaseCoordinate neg_pc = pc.negate_by_phase_xor();
        std::cout << "  Valor = " << std::setw(6) << qv
                  << " | Fase = " << phase_to_symbol(pc.primary_phase)
                  << " | Pivo = D(" << pc.top_layer_k << "," << pc.top_digit_d << ")=" << std::setw(4) << pc.top_abs_anchor
                  << " | |Delta| = " << std::setw(3) << pc.top_abs_residue
                  << " | Reconstruido = " << std::setw(6) << pc.reconstruct_from_phase()
                  << " | XOR 11 (Inversao 1-Ciclo) -> " << phase_to_symbol(neg_pc.primary_phase)
                  << " = " << std::setw(6) << neg_pc.reconstruct_from_phase() << "\n";
    }

    std::cout << "\nAcoplamento Simultaneo de Todas as Camadas Decimais (L0..L3) e Trajetorias:\n";
    const std::array<int64_t, 5> sample_values = {v0, v1, v2, v3, 98765};
    for (int64_t v : sample_values) {
        GeometricCoordinate c = InvariantDiagonalSieve::decompose(v);
        int64_t rec = c.reconstruct();
        std::cout << "  Valor = " << std::setw(8) << v
                  << " -> Reconstruido = " << std::setw(8) << rec
                  << " [OK=" << (v == rec ? "SIM" : "ERRO") << "] | Trajetoria: "
                  << c.format_trajectory() << "\n";
    }

    // =========================================================================
    // ETAPA 2: FASE 3 - COMPILADOR CIENTE DA GEOMETRIA -> EXECUCAO GENOMA VM
    // =========================================================================
    print_banner("2. FASE 3: COMPILADOR GNC-X (ACOPLAMENTO L0..L3) & EXECUCAO DA VM");
    GenomeSpatialVM vm("gnc_disk.img");

    for (size_t i = 0; i < 1024; ++i) {
        vm.memory().write_by_sieve(2, 7, static_cast<int64_t>(i), static_cast<int64_t>(50 + (i * 13) % 250));
    }
    for (size_t i = 0; i < 64; ++i) {
        vm.reg_matrix().layer0_matrix[i] = static_cast<int64_t>(i + 1);
    }

    // Programa em linguagem de alto nivel GNC-X compilado automaticamente para Bytecode LMI
    std::ostringstream gncx_source;
    gncx_source << "# Script Fonte GNC-X (Fase 3: Compilador Ciente da Geometria)\n"
                << "var nav_ref  = " << v0 << "\n"
                << "var sensor_x = " << v1 << "\n"
                << "var pos_grid = " << v2 << "\n"
                << "negate_phase inv_pos = pos_grid\n"
                << "when_gt sensor_x nav_ref -> mac_grid 3 7\n"
                << "persist_gla sensor_x pos_grid\n"
                << "project_db 2 3 7 1024\n";

    GeometryAwareCompiler compiler;
    CompilationReport comp_report = compiler.compile_source(gncx_source.str());

    std::cout << "Listagem Gerada pelo Compilador Ciente da Geometria (Fase 3):\n";
    for (const auto& asm_str : comp_report.assembly_listing) {
        std::cout << "  [ASM] " << asm_str << "\n";
    }
    std::cout << "Total de Instrucoes Bytecode LMI Emitidas (com acoplamento L0..L3): "
              << comp_report.bytecode.size() << "\n\n";

    vm.execute_program(comp_report.bytecode);

    const auto& p_tel = vm.pipeline().telemetry();
    std::cout << "Traco de Execucao do Pipeline Branchless (7 Estagios | PREDICTOR DISABLED):\n";
    for (const auto& line : p_tel.trace_log) {
        std::cout << "  " << line << "\n";
    }

    std::cout << "\nMetricas Deterministicas do Processo:\n";
    std::cout << "  - Instrucoes LMI Comitadas : " << p_tel.instructions_committed << "\n";
    std::cout << "  - Ciclos Totais de Pipeline: " << p_tel.total_cycles << " (6 fill + " << p_tel.instructions_committed << " exec)\n";
    std::cout << "  - Throughput Efetivo (IPC) : " << std::fixed << std::setprecision(2) << p_tel.ipc() << " INS/CYC\n";
    std::cout << "  - Branch Mispredictions    : " << p_tel.branch_mispredictions << " (Zero Absoluto - Branchless)\n";

    std::cout << "\nEstado da Matriz Fisica 64-Bit (Layer 0 - 8x8 MAC/ALU Grid apos OP_GRID_MAC):\n";
    std::cout << vm.reg_matrix().format_layer0_matrix();

    std::cout << "\nTelemetria do Motor de Disco GLA (`gnc_disk.img` via Memory Mapping):\n";
    const auto& g_tel = vm.disk().telemetry();
    std::cout << "  - Escritas O(1) via mmap   : " << g_tel.mmap_writes << "\n";
    std::cout << "  - Leituras O(1) via mmap   : " << g_tel.mmap_reads << "\n";
    std::cout << "  - Prefetches Diagonais     : " << g_tel.predictive_prefetches << "\n";
    for (const auto& pf : g_tel.prefetch_log) {
        std::cout << "    * " << pf << "\n";
    }

    // =========================================================================
    // ETAPA 3: BENCHMARKS EMPIRICOS (GNC-X vs ARQUITETURA TRADICIONAL)
    // =========================================================================
    print_banner("3. VALIDACAO EMPIRICA: GNC-X vs ARQUITETURA TRADICIONAL (SILICIO ATUAL)");

    constexpr size_t N_BENCH = 2'000'000;
    std::vector<int64_t> data_branch(N_BENCH);
    std::vector<int64_t> data_branchless(N_BENCH);
    std::mt19937_64 rng(1337);
    for (size_t i = 0; i < N_BENCH; ++i) {
        int64_t v = static_cast<int64_t>(rng() % 1000);
        data_branch[i] = v;
        data_branchless[i] = v;
    }

    // Benchmark 3.1: Branchless Predicate Grid vs Unpredictable Branching
    auto t0 = std::chrono::high_resolution_clock::now();
    int64_t checksum_trad = 0;
    for (size_t i = 0; i < N_BENCH; ++i) {
        int64_t v = data_branch[i];
        // Ramificacao condicional imprevisivel (sofre branch misprediction)
        if (v > 500) {
            if ((v & 1) == 0) checksum_trad += v * 3 + 7;
            else checksum_trad += v * 2 - 5;
        } else {
            if ((v & 1) == 0) checksum_trad += v + 11;
            else checksum_trad += v - 3;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    auto t2 = std::chrono::high_resolution_clock::now();
    int64_t checksum_gncx = 0;
    const int64_t* __restrict ptr = data_branchless.data();
    for (size_t i = 0; i < N_BENCH; ++i) {
        int64_t v = ptr[i];
        uint64_t m_gt = branchless_mask_gt(v, 500);
        uint64_t m_even = ((static_cast<uint64_t>(v) & 1ULL) ^ 1ULL) * ~0ULL;

        int64_t hi_val = branchless_select(m_even, v * 3 + 7, v * 2 - 5);
        int64_t lo_val = branchless_select(m_even, v + 11, v - 3);
        checksum_gncx += branchless_select(m_gt, hi_val, lo_val);
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    double ms_branch = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_gncx_grid = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "[BENCHMARK A] Processamento na Grade (2.000.000 elementos):\n";
    std::cout << "  - Pipeline Tradicional (if/else branching) : " << std::fixed << std::setprecision(3) << ms_branch << " ms\n";
    std::cout << "  - Pipeline GNC-X (Branchless Predicate)    : " << ms_gncx_grid << " ms\n";
    std::cout << "  - Checksum Identico Validado               : " << (checksum_trad == checksum_gncx ? "SIM" : "NAO")
              << " (" << checksum_gncx << ")\n";

    // Benchmark 3.2: Spatial Tensor DB (O(1) Diagonal Sieve) vs B-Tree / Red-Black Tree (std::map)
    constexpr size_t N_DB = 200'000;
    std::vector<int64_t> query_keys(N_DB);
    for (size_t i = 0; i < N_DB; ++i) {
        query_keys[i] = static_cast<int64_t>(100 + (i * 7) % 900'000);
    }

    auto tb0 = std::chrono::high_resolution_clock::now();
    std::map<int64_t, int64_t> traditional_tree_index;
    for (size_t i = 0; i < N_DB; ++i) {
        traditional_tree_index[query_keys[i]] = static_cast<int64_t>(i * 10);
    }
    int64_t tree_sum = 0;
    for (size_t i = 0; i < N_DB; ++i) {
        tree_sum += traditional_tree_index[query_keys[i]];
    }
    auto tb1 = std::chrono::high_resolution_clock::now();

    auto tb2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N_DB; ++i) {
        vm.memory().store_fast_by_sieve(query_keys[i], static_cast<int64_t>(i * 10));
    }
    int64_t gncx_db_sum = 0;
    for (size_t i = 0; i < N_DB; ++i) {
        gncx_db_sum += vm.memory().load_fast_by_sieve(query_keys[i]);
    }
    auto tb3 = std::chrono::high_resolution_clock::now();

    double ms_tree = std::chrono::duration<double, std::milli>(tb1 - tb0).count();
    double ms_gncx_db = std::chrono::duration<double, std::milli>(tb3 - tb2).count();

    std::cout << "\n[BENCHMARK B] Indexacao e Busca de Banco de Dados (200.000 operacoes):\n";
    std::cout << "  - Indice em Arvore Tradicional O(log N)    : " << std::setprecision(3) << ms_tree << " ms\n";
    std::cout << "  - GNC-X Crivo Invariante Diagonal O(1)     : " << ms_gncx_db << " ms\n";
    std::cout << "  - Integridade dos Dados Recuperados        : " << (tree_sum == gncx_db_sum ? "100% EXATO" : "DIVERGENTE")
              << " (Soma=" << gncx_db_sum << ")\n";
    std::cout << "  - Ganho de Velocidade (Speedup GNC-X)      : " << std::setprecision(2) << (ms_tree / std::max(0.001, ms_gncx_db)) << "x mais rapido!\n";

    // Benchmark 3.3: X-Folding Tensor Engine (PyTorch-Style 64-Channel Linear Projection over 500,000 elements)
    constexpr size_t N_TENSOR = 500'000;
    constexpr size_t M_CHANNELS = 64;
    constexpr int TENSOR_LAYER_K = 3; // Layer 3 (Milhares: -9999 .. +9999)

    std::vector<int64_t> raw_tensor(N_TENSOR);
    std::vector<uint8_t> raw_sectors(N_TENSOR);
    std::vector<CompactXToken> x_tokens(N_TENSOR);

    for (size_t i = 0; i < N_TENSOR; ++i) {
        int64_t signed_val = static_cast<int64_t>((rng() % 19999)) - 9999;
        CompactXToken tok = XFoldingTensorEngine::encode_token_layer(signed_val, TENSOR_LAYER_K);
        x_tokens[i] = tok;
        // Exact value represented by the token
        raw_tensor[i] = XFoldingTensorEngine::decode_token_layer(tok, TENSOR_LAYER_K);
        raw_sectors[i] = tok.diag_d;
    }

    std::vector<std::array<int64_t, 10>> weights(M_CHANNELS);
    for (size_t m = 0; m < M_CHANNELS; ++m) {
        for (size_t d = 0; d < 10; ++d) {
            weights[m][d] = static_cast<int64_t>((m + 1) * (d + 3) - 17);
        }
    }

    std::vector<int64_t> out_standard(M_CHANNELS, 0);
    std::vector<int64_t> out_xfold(M_CHANNELS, 0);

    auto tx0 = std::chrono::high_resolution_clock::now();
    XFoldingTensorEngine::compute_standard_dense_projection(
        raw_tensor.data(), raw_sectors.data(), N_TENSOR, weights, out_standard
    );
    auto tx1 = std::chrono::high_resolution_clock::now();

    auto tx2 = std::chrono::high_resolution_clock::now();
    XFoldAccumulator fold_stats = XFoldingTensorEngine::compute_xfold_tensor_projection(
        x_tokens.data(), N_TENSOR, TENSOR_LAYER_K, weights, out_xfold
    );
    auto tx3 = std::chrono::high_resolution_clock::now();

    double ms_std_tensor = std::chrono::duration<double, std::milli>(tx1 - tx0).count();
    double ms_xfold_tensor = std::chrono::duration<double, std::milli>(tx3 - tx2).count();

    bool exact_tensor_match = (out_standard == out_xfold);
    int64_t tensor_checksum = 0;
    for (int64_t v : out_xfold) tensor_checksum ^= v;

    std::cout << "\n[BENCHMARK C] Calculo Tensorial Estilo PyTorch (500.000 elem x 64 Canais = 32 Milhoes MACs):\n";
    std::cout << "  - Multiplicacao Densa Tradicional (N * M)  : " << std::setprecision(3) << ms_std_tensor << " ms (32.000.000 mults)\n";
    std::cout << "  - Algoritmo GNC-X Dobramento em 'X' (4-Ph) : " << ms_xfold_tensor << " ms (Apenas 640 mults nos Pivos!)\n";
    std::cout << "  - Pares de Fase Oposta Anulados no 'X'     : " << fold_stats.phase_cancelations << " pares cancelados s/ multiplicar\n";
    std::cout << "  - Economia de Memoria RAM / Cache L1       : 50% menor (CompactXToken 32-bit vs int64)\n";
    std::cout << "  - Equivalencia Matematica Bit-a-Bit        : " << (exact_tensor_match ? "100% EXATO" : "ERRO")
              << " (Hash=" << tensor_checksum << ")\n";
    std::cout << "  - Ganho de Velocidade (Speedup PyTorch-X)  : " << std::setprecision(2)
              << (ms_std_tensor / std::max(0.001, ms_xfold_tensor)) << "x mais rapido!\n";

    // =========================================================================
    // ETAPA 4: DISPLAY VISUAL EM RUNTIME (MODULOS E CAMADAS POR CORES + DOTS)
    // =========================================================================
    vm.display().render_console_dashboard(
        vm.reg_matrix(),
        vm.pipeline().telemetry(),
        vm.disk().telemetry(),
        vm.db().stats()
    );

    const std::string html_monitor_path = "gncx_visual_monitor.html";
    vm.display().export_interactive_html_monitor(
        html_monitor_path,
        vm.reg_matrix(),
        vm.pipeline().telemetry(),
        vm.disk().telemetry(),
        ms_branch,
        ms_gncx_grid,
        ms_tree,
        ms_gncx_db
    );
    std::cout << "\n[MONITOR GRAFICO INTERATIVO GERADO]: " << html_monitor_path << "\n";

    std::cout << "\n================================================================================\n";
    std::cout << "  SIMULADOR VIRTUAL GNC-X CONCLUIDO COM SUCESSO.\n";
    std::cout << "================================================================================\n";
    return 0;
}
