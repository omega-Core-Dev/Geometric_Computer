#pragma once
/**
 * ============================================================================
 * GNC-X SILICON INCUBATOR — REAL-TIME ANSI TRUECOLOR VISUAL DISPLAY & HUD
 * ============================================================================
 * Author of Conception: Zaqueu Ribeiro da Costa
 *
 * Separates the 5 Architectural Modules by distinct signature colors and
 * renders real-time data indicator dots (● / ◉ / ·) across all layers:
 *
 *   [1. SISTEMA (Genoma / LMI OS)]       -> AZUL REAL / INDIGO   (#4d94ff) ●
 *   [2. PROCESSADOR (Matriz 64b/Pipe)]   -> CIANO NEON           (#00f5ff) ●
 *   [3. MEMORIA (Camadas L0..L3/Crivos)] -> VERDE ESMERALDA      (#00ff7b) ●
 *   [4. DISCO (Motor GLA gnc_disk.img)]  -> AMBAR / OURO         (#ffb800) ●
 *   [5. BANCO DE DADOS (Tensor DB)]      -> MAGENTA / VIOLETA    (#e84aff) ●
 * ============================================================================
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <array>
#include <vector>
#include <fstream>
#include <algorithm>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "gncx_sieve.hpp"
#include "gncx_register_matrix.hpp"
#include "gncx_memory.hpp"
#include "gncx_gla_disk.hpp"
#include "gncx_spatial_db.hpp"

namespace gncx {

namespace ansi {
    inline constexpr const char* RESET        = "\033[0m";
    inline constexpr const char* DIM_GRID     = "\033[38;2;45;75;95m";
    inline constexpr const char* WHITE_BOLD   = "\033[1;38;2;240;248;255m";

    // 5 Signature Colors for the 5 Unified Modules:
    // 1. SISTEMA (Genoma / Bytecode LMI / Orquestrador) -> Royal Electric Blue
    inline constexpr const char* MOD_SISTEMA  = "\033[1;38;2;77;148;255m";
    // 2. PROCESSADOR (Matriz 64-Bit 8x8 & Pipeline Branchless) -> Bright Cyan
    inline constexpr const char* MOD_PROC     = "\033[1;38;2;0;245;255m";
    // 3. MEMORIA (Camadas de Magnitude L0..L3 & Crivos Diagonais) -> Emerald Green
    inline constexpr const char* MOD_MEMORIA  = "\033[1;38;2;0;255;123m";
    // 4. DISCO (Motor GLA gnc_disk.img via mmap) -> Gold / Amber
    inline constexpr const char* MOD_DISCO    = "\033[1;38;2;255;184;0m";
    // 5. BANCO DE DADOS (Tensor Espacial PROJECT/KALMAN) -> Neon Magenta
    inline constexpr const char* MOD_DB       = "\033[1;38;2;232;74;255m";
}

enum class DotState : uint8_t {
    EMPTY = 0,          // · Celula em repouso
    SISTEMA_DOT = 1,    // ● Azul (Instrucao LMI / Coordenada do Genoma)
    PROC_DOT = 2,       // ◉ Ciano (Operacao na Matriz 64-Bit / ALU / Mascara)
    MEMORIA_SIEVE = 3,  // ● Verde (Crivo Invariante Diagonal D(k, d) / Hit Memoria)
    DISCO_GLA_DOT = 4,  // ◉ Ouro/Ambar (Persistencia GLA gnc_disk.img / Prefetch)
    DB_TENSOR_DOT = 5   // ◉ Magenta (Transformacao Algebrica no DB Tensorial)
};

struct VisualFrameSnapshot {
    uint64_t cycle{0};
    int active_module{1}; // 1=Sistema, 2=Processador, 3=Memoria, 4=Disco, 5=DB
    std::string instruction_mnemonic;
    std::string instruction_detail;
    int active_layer{-1};
    int active_diag_d{-1};

    // 1. SISTEMA: 12 slots de Stream de Bytecode LMI (dots indicando instrucoes)
    std::array<DotState, 12> sistema_lmi_dots{};
    // 2. PROCESSADOR: 8x8 Matriz Fisica 64-Bit (64 dots) + 7 estagios do Pipeline
    std::array<DotState, 64> proc_matrix_dots{};
    std::array<DotState, 7>  proc_pipeline_dots{};
    // 3. MEMORIA: 4 Camadas de Magnitude (L0..L3) x Grade 10x10 (400 dots)
    std::array<std::array<std::array<DotState, 10>, 10>, 4> mem_layer_grids{};
    // 4. DISCO (GLA): 4 Sub-camadas x 10 Setores Diagonais no gnc_disk.img (40 dots)
    std::array<std::array<DotState, 10>, 4> disco_gla_dots{};
    // 5. BANCO DE DADOS: 2 Tensores (Fonte L2 -> Destino L3) x 10 Setores (20 dots)
    std::array<std::array<DotState, 10>, 2> db_tensor_dots{};
};

class RuntimeVisualDisplay {
public:
    RuntimeVisualDisplay() {
        enable_ansi_console();
        reset_to_invariant_state();
    }

    static void enable_ansi_console() {
#ifdef _WIN32
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut != INVALID_HANDLE_VALUE) {
            DWORD dwMode = 0;
            if (GetConsoleMode(hOut, &dwMode)) {
                dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                SetConsoleMode(hOut, dwMode);
            }
        }
        SetConsoleOutputCP(CP_UTF8);
#endif
    }

    void reset_to_invariant_state() {
        current_.sistema_lmi_dots.fill(DotState::EMPTY);
        for (size_t i = 0; i < 64; ++i) {
            size_t r = i / 8, c = i % 8;
            current_.proc_matrix_dots[i] = (r == c) ? DotState::MEMORIA_SIEVE : DotState::EMPTY;
        }
        current_.proc_pipeline_dots.fill(DotState::PROC_DOT);

        for (int k = 0; k < 4; ++k) {
            for (int r = 0; r < 10; ++r) {
                for (int c = 0; c < 10; ++c) {
                    // A diagonal principal de cada camada e SEMPRE o Crivo Invariante de Referencia
                    current_.mem_layer_grids[k][r][c] = (r == c) ? DotState::MEMORIA_SIEVE : DotState::EMPTY;
                }
            }
        }

        for (int l = 0; l < 4; ++l) {
            for (int d = 0; d < 10; ++d) {
                current_.disco_gla_dots[l][d] = DotState::EMPTY;
            }
        }
        for (int t = 0; t < 2; ++t) {
            for (int d = 0; d < 10; ++d) {
                current_.db_tensor_dots[t][d] = DotState::EMPTY;
            }
        }
    }

    void record_lmi_step(
        size_t step_idx,
        uint64_t cycle,
        int module_id,
        const std::string& mnemonic,
        const std::string& detail,
        const GeometricCoordinate& coord,
        int target_layer,
        int target_diag_d
    ) {
        current_.cycle = cycle;
        current_.active_module = module_id;
        current_.instruction_mnemonic = mnemonic;
        current_.instruction_detail = detail;
        int lk = std::clamp(target_layer, 0, 3);
        int d = std::clamp(target_diag_d, 0, 9);
        current_.active_layer = lk;
        current_.active_diag_d = d;

        // 1. Marcar dot no Modulo SISTEMA (Genoma LMI Stream)
        if (step_idx < current_.sistema_lmi_dots.size()) {
            current_.sistema_lmi_dots[step_idx] = DotState::SISTEMA_DOT;
        }

        // 2. Atualizar Modulo PROCESSADOR (Matriz 64-Bit & Pipeline)
        for (size_t s = 0; s < 7; ++s) {
            current_.proc_pipeline_dots[s] = (module_id == 2) ? DotState::PROC_DOT : DotState::SISTEMA_DOT;
        }
        if (module_id == 1 || module_id == 2) {
            size_t cell = static_cast<size_t>((lk * 8 + d) & 63);
            current_.proc_matrix_dots[cell] = (module_id == 1) ? DotState::SISTEMA_DOT : DotState::PROC_DOT;
            if (mnemonic == "OP_GRID_MAC") {
                for (size_t i = 0; i < 64; ++i) {
                    if ((i % 3) == 0) current_.proc_matrix_dots[i] = DotState::PROC_DOT;
                }
            }
        }

        // 3. Atualizar Modulo MEMORIA (Camadas L0..L3 & Trajetoria do Crivo)
        if (coord.num_terms > 0) {
            for (uint32_t i = 0; i < coord.num_terms; ++i) {
                const auto& t = coord.terms[i];
                int tk = std::clamp(t.layer_k, 0, 3);
                int td = std::clamp(t.digit_d, 0, 9);
                int col = (td + static_cast<int>(i * 2 + 1)) % 10;
                current_.mem_layer_grids[tk][td][col] = (i == 0) ? DotState::PROC_DOT : DotState::SISTEMA_DOT;
            }
        }
        if (module_id == 3 || mnemonic == "OP_GRID_VECTOR") {
            for (int c = 0; c < 10; ++c) {
                if (c != d) current_.mem_layer_grids[lk][d][c] = DotState::PROC_DOT;
            }
        }

        // 4. Atualizar Modulo DISCO (GLA gnc_disk.img + Prefetch Vertical)
        if (module_id == 4) {
            current_.disco_gla_dots[lk][d] = DotState::DISCO_GLA_DOT;
            int next_l = (lk + 1) & 3;
            current_.disco_gla_dots[next_l][d] = DotState::SISTEMA_DOT; // Prefetch preditivo
            current_.mem_layer_grids[lk][d][(d + 3) % 10] = DotState::DISCO_GLA_DOT;
        }

        // 5. Atualizar Modulo BANCO DE DADOS (Tensor Espacial PROJECT/KALMAN)
        if (module_id == 5) {
            current_.db_tensor_dots[0][d] = DotState::MEMORIA_SIEVE;  // Fonte L2
            current_.db_tensor_dots[1][d] = DotState::DB_TENSOR_DOT;  // Destino Projetado L3
            for (int c = 0; c < 10; ++c) {
                if (c != d) current_.mem_layer_grids[3][d][c] = DotState::DB_TENSOR_DOT;
            }
        }

        frames_.push_back(current_);
    }

    static const char* render_dot(DotState s) noexcept {
        switch (s) {
            case DotState::SISTEMA_DOT:   return "\033[1;38;2;77;148;255m●\033[0m"; // Azul (Sistema/Genoma)
            case DotState::PROC_DOT:      return "\033[1;38;2;0;245;255m◉\033[0m";  // Ciano (Processador/Matriz)
            case DotState::MEMORIA_SIEVE: return "\033[1;38;2;0;255;123m●\033[0m";  // Verde (Memoria/Crivo Diagonal)
            case DotState::DISCO_GLA_DOT: return "\033[1;38;2;255;184;0m◉\033[0m";  // Ouro (Disco GLA mmap)
            case DotState::DB_TENSOR_DOT: return "\033[1;38;2;232;74;255m◉\033[0m"; // Magenta (DB Tensorial)
            case DotState::EMPTY:
            default:                      return "\033[38;2;45;75;95m·\033[0m";     // Ponto inativo da grade
        }
    }

    // Renderiza em runtime todos os 5 Modulos separados por cores no console
    void render_console_dashboard(
        const RegisterMatrix64& reg_mat,
        const PipelineTelemetry& pipe_tel,
        const GLATelemetry& gla_tel,
        const SpatialDBStats& db_stats
    ) const {
        std::cout << "\n";
        std::cout << ansi::WHITE_BOLD << "╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║         GNC-X RUNTIME VISUAL DISPLAY — ARQUITETURA SEPARADA POR CORES DE MODULOS             ║\n";
        std::cout << "║   "
                  << ansi::MOD_SISTEMA << "[● SISTEMA]" << "   "
                  << ansi::MOD_PROC    << "[◉ PROCESSADOR]" << "   "
                  << ansi::MOD_MEMORIA << "[● MEMORIA/CRIVOS]" << "   "
                  << ansi::MOD_DISCO   << "[◉ DISCO GLA]" << "   "
                  << ansi::MOD_DB      << "[◉ DB TENSOR]" << ansi::WHITE_BOLD << "    ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════════╝" << ansi::RESET << "\n";

        // =====================================================================
        // MODULO 1: SISTEMA (AZUL REAL) — O GENOMA & ORQUESTRADOR ESPACIAL
        // =====================================================================
        std::cout << ansi::MOD_SISTEMA
                  << "┌─ [MODULO 1: SISTEMA — GENOMA & BYTECODE LMI (ORQUESTRADOR ESPACIAL)] ────────────────────────┐\n"
                  << "│ " << ansi::WHITE_BOLD << "Stream de Instrucoes LMI Comitadas (Dots de Atividade do Genoma):" << ansi::MOD_SISTEMA << "                            │\n"
                  << "│   ";
        const std::array<const char*, 12> lmi_names = {
            "LDI_0", "LDI_1", "LDI_2", "LDI_3", "STI_L2", "LDI_L2",
            "MASK",  "MAC64", "VEC1K", "GLA_1", "GLA_2",  "DB_PRJ"
        };
        for (size_t i = 0; i < 12; ++i) {
            std::cout << lmi_names[i] << ":" << render_dot(current_.sistema_lmi_dots[i]) << " ";
        }
        std::cout << ansi::MOD_SISTEMA << " │\n"
                  << "└──────────────────────────────────────────────────────────────────────────────────────────────┘\n"
                  << ansi::RESET;

        // =====================================================================
        // MODULO 2: PROCESSADOR (CIANO NEON) — MATRIZ 64-BIT (8x8) & PIPELINE
        // =====================================================================
        std::cout << ansi::MOD_PROC
                  << "┌─ [MODULO 2: PROCESSADOR — MATRIZ FISICA 64-BIT (8x8) & PIPELINE BRANCHLESS] ─────────────────┐\n";
        for (int r = 0; r < 8; ++r) {
            std::cout << ansi::MOD_PROC << "│  " << ansi::DIM_GRID << "REG_ROW[" << r << "] " << ansi::RESET;
            for (int c = 0; c < 8; ++c) {
                std::cout << render_dot(current_.proc_matrix_dots[r * 8 + c]) << " ";
            }
            std::cout << ansi::MOD_PROC << "│ ";
            if (r == 0) {
                std::cout << ansi::WHITE_BOLD << "PIPELINE DETERMINISTICO (7 ESTAGIOS - SEM BRANCH):" << ansi::MOD_PROC << "            │\n";
            } else if (r == 1) {
                std::cout << "  "
                          << render_dot(current_.proc_pipeline_dots[0]) << "FETCH ──► "
                          << render_dot(current_.proc_pipeline_dots[1]) << "DECODE ──► "
                          << render_dot(current_.proc_pipeline_dots[2]) << "PREDICT[MASK] ──► "
                          << render_dot(current_.proc_pipeline_dots[3]) << "ISSUE         │\n";
            } else if (r == 2) {
                std::cout << "  ──► "
                          << render_dot(current_.proc_pipeline_dots[4]) << "EXEC(ALU/GRID) ──► "
                          << render_dot(current_.proc_pipeline_dots[5]) << "WRITE_BACK ──► "
                          << render_dot(current_.proc_pipeline_dots[6]) << "COMMIT(1/CYC) │\n";
            } else if (r == 3) {
                std::cout << "  Latencia: 1 CYC/STAGE | Throughput: "
                          << std::fixed << std::setprecision(2) << pipe_tel.ipc()
                          << " INS/CYC | Mispredict: " << pipe_tel.branch_mispredictions << "  │\n";
            } else if (r == 4) {
                std::cout << ansi::WHITE_BOLD << "Valores Ativos Layer 0 (64 MACs - Amostra Diagonal):" << ansi::MOD_PROC << "          │\n";
            } else if (r == 5) {
                std::cout << "  [0,0]=" << std::setw(5) << reg_mat.layer0_matrix[0]
                          << " [1,1]=" << std::setw(5) << reg_mat.layer0_matrix[9]
                          << " [2,2]=" << std::setw(5) << reg_mat.layer0_matrix[18]
                          << " [3,3]=" << std::setw(5) << reg_mat.layer0_matrix[27] << "               │\n";
            } else if (r == 6) {
                std::cout << "  Mascara Branchless M0 = 0x" << std::hex << reg_mat.masks[0] << std::dec
                          << " (Ativa sem if/else)        │\n";
            } else {
                std::cout << "  Registradores: R0=" << reg_mat.regs[0]
                          << " R1=" << reg_mat.regs[1]
                          << " R2=" << reg_mat.regs[2]
                          << " R3=" << reg_mat.regs[3] << "                  │\n";
            }
        }
        std::cout << ansi::MOD_PROC
                  << "└──────────────────────────────────────────────────────────────────────────────────────────────┘\n"
                  << ansi::RESET;

        // =====================================================================
        // MODULO 3: MEMORIA (VERDE ESMERALDA) — CAMADAS L0..L3 & CRIVOS DIAGONAIS
        // =====================================================================
        std::cout << ansi::MOD_MEMORIA
                  << "┌─ [MODULO 3: MEMORIA GEOMETRICA — CAMADAS DE MAGNITUDE & CRIVOS INVARIANTES DIAGONAIS] ───────┐\n"
                  << "│   " << ansi::MOD_MEMORIA << "LAYER 0: UNIDADES (0-9)    LAYER 1: DEZENAS (00-99)   LAYER 2: CENTENAS(000-999) LAYER 3: MILHARES(0000-9999)│\n";

        for (int r = 0; r < 10; ++r) {
            std::cout << ansi::MOD_MEMORIA << "│   " << ansi::DIM_GRID << "d" << r << " " << ansi::RESET;
            for (int k = 0; k < 4; ++k) {
                for (int c = 0; c < 10; ++c) {
                    std::cout << render_dot(current_.mem_layer_grids[k][r][c]) << " ";
                }
                if (k < 3) std::cout << ansi::DIM_GRID << "│ d" << r << " " << ansi::RESET;
            }
            std::cout << ansi::MOD_MEMORIA << "   │\n";
        }
        std::cout << ansi::MOD_MEMORIA
                  << "│   " << ansi::WHITE_BOLD << "Diagonal Verde (●) = Crivo Invariante D(k,d) = d * R_{k+1} | Dots Coloridos = Dados por Camada" << ansi::MOD_MEMORIA << " │\n"
                  << "└──────────────────────────────────────────────────────────────────────────────────────────────┘\n"
                  << ansi::RESET;

        // =====================================================================
        // MODULO 4: DISCO GLA (OURO/AMBAR) & MODULO 5: BANCO DE DADOS (MAGENTA)
        // =====================================================================
        std::cout << ansi::MOD_DISCO
                  << "┌─ [MODULO 4: DISCO GLA (gnc_disk.img mmap)] ──┐ "
                  << ansi::MOD_DB
                  << "┌─ [MODULO 5: BANCO DE DADOS TENSORIAL] ──────┐\n";

        for (int l = 0; l < 4; ++l) {
            std::cout << ansi::MOD_DISCO << "│  Sub-Layer L" << l << ": ";
            for (int d = 0; d < 10; ++d) {
                std::cout << render_dot(current_.disco_gla_dots[l][d]) << " ";
            }
            std::cout << ansi::MOD_DISCO << "        │ ";

            std::cout << ansi::MOD_DB << "│  ";
            if (l == 0) {
                std::cout << "Tensor Fonte (L2)  : ";
                for (int d = 0; d < 10; ++d) std::cout << render_dot(current_.db_tensor_dots[0][d]) << " ";
                std::cout << " │\n";
            } else if (l == 1) {
                std::cout << "Tensor Destino (L3): ";
                for (int d = 0; d < 10; ++d) std::cout << render_dot(current_.db_tensor_dots[1][d]) << " ";
                std::cout << " │\n";
            } else if (l == 2) {
                std::cout << "Operador: PROJECT(L2, KALMAN) -> L3         │\n";
            } else {
                std::cout << "Elementos Projetados s/ SQL: " << std::setw(6) << db_stats.elements_projected << "         │\n";
            }
        }

        std::cout << ansi::MOD_DISCO
                  << "│  Escritas mmap O(1): " << gla_tel.mmap_writes
                  << " | Prefetches: " << gla_tel.predictive_prefetches << "       │ "
                  << ansi::MOD_DB
                  << "│  Buscas O(1) Crivo Diagonal (s/ B-Tree): OK │\n"
                  << ansi::MOD_DISCO
                  << "└──────────────────────────────────────────────┘ "
                  << ansi::MOD_DB
                  << "└─────────────────────────────────────────────┘\n"
                  << ansi::RESET;
    }

    // Exporta tambem o Monitor Visual HTML5 interativo com os 5 Modulos separados por cores
    void export_interactive_html_monitor(
        const std::string& html_path,
        const RegisterMatrix64& reg_mat,
        const PipelineTelemetry& pipe_tel,
        const GLATelemetry& gla_tel,
        double bench_branch_ms,
        double bench_gncx_grid_ms,
        double bench_tree_ms,
        double bench_gncx_db_ms
    ) const {
        std::ofstream out(html_path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) return;

        out << R"HTML(<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>GNC-X Runtime Visual Display — 5 Módulos por Cores</title>
<style>
  :root {
    --mod-sistema: #4d94ff;
    --mod-proc:    #00f5ff;
    --mod-memoria: #00ff7b;
    --mod-disco:   #ffb800;
    --mod-db:      #e84aff;
    --bg-dark:     #030c14;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: radial-gradient(circle at 50% 15%, #082235 0%, #02080e 100%);
    color: #e4f6ff;
    font-family: 'Consolas', 'Courier New', monospace;
    padding: 16px;
  }
  .top-bar {
    border: 1px solid rgba(255,255,255,0.2);
    background: rgba(6, 20, 32, 0.92);
    padding: 12px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 14px;
  }
  .module-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }
  .badge {
    padding: 5px 10px;
    font-size: 11px;
    font-weight: bold;
    border-radius: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .b-sis  { border: 1px solid var(--mod-sistema); color: var(--mod-sistema); background: rgba(77,148,255,0.12); }
  .b-proc { border: 1px solid var(--mod-proc);    color: var(--mod-proc);    background: rgba(0,245,255,0.12); }
  .b-mem  { border: 1px solid var(--mod-memoria); color: var(--mod-memoria); background: rgba(0,255,123,0.12); }
  .b-disc { border: 1px solid var(--mod-disco);   color: var(--mod-disco);   background: rgba(255,184,0,0.12); }
  .b-db   { border: 1px solid var(--mod-db);      color: var(--mod-db);      background: rgba(232,74,255,0.12); }

  .dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
    transition: all 0.2s ease;
  }
  .dot-0 { background: rgba(90, 130, 160, 0.22); }
  .dot-1 { background: var(--mod-sistema); box-shadow: 0 0 8px var(--mod-sistema); transform: scale(1.2); }
  .dot-2 { background: var(--mod-proc);    box-shadow: 0 0 9px var(--mod-proc);    transform: scale(1.25); }
  .dot-3 { background: var(--mod-memoria); box-shadow: 0 0 8px var(--mod-memoria); }
  .dot-4 { background: var(--mod-disco);   box-shadow: 0 0 10px var(--mod-disco);  transform: scale(1.25); }
  .dot-5 { background: var(--mod-db);      box-shadow: 0 0 10px var(--mod-db);     transform: scale(1.25); }

  .mod-panel {
    background: rgba(5, 17, 28, 0.88);
    padding: 12px 14px;
    margin-bottom: 14px;
    border-radius: 4px;
  }
  .mod-sistema-box { border: 2px solid var(--mod-sistema); box-shadow: 0 0 16px rgba(77,148,255,0.15); }
  .mod-proc-box    { border: 2px solid var(--mod-proc);    box-shadow: 0 0 16px rgba(0,245,255,0.15); }
  .mod-memoria-box { border: 2px solid var(--mod-memoria); box-shadow: 0 0 16px rgba(0,255,123,0.15); }
  .mod-disco-box   { border: 2px solid var(--mod-disco);   box-shadow: 0 0 16px rgba(255,184,0,0.15); }
  .mod-db-box      { border: 2px solid var(--mod-db);      box-shadow: 0 0 16px rgba(232,74,255,0.15); }

  .mod-header {
    font-size: 13px;
    font-weight: bold;
    padding-bottom: 6px;
    margin-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.12);
    display: flex;
    justify-content: space-between;
  }
  .lmi-strip {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 6px;
  }
  .lmi-slot {
    border: 1px solid rgba(77,148,255,0.4);
    padding: 6px 4px;
    text-align: center;
    font-size: 10px;
    background: rgba(77,148,255,0.06);
  }
  .mid-layout {
    display: grid;
    grid-template-columns: 330px 1fr 340px;
    gap: 14px;
  }
  @media (max-width: 1200px) { .mid-layout { grid-template-columns: 1fr; } }
  .matrix-8x8 {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    gap: 4px;
  }
  .cell-8x8 {
    aspect-ratio: 1;
    border: 1px solid rgba(0,245,255,0.25);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    background: rgba(0,40,60,0.35);
  }
  .layers-2x2 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }
  .layer-card {
    border: 1px solid rgba(0,255,123,0.35);
    padding: 8px;
    background: rgba(2, 14, 22, 0.85);
  }
  .grid-10x10 {
    display: grid;
    grid-template-columns: repeat(10, 1fr);
    gap: 3px;
    margin-top: 6px;
  }
  .cell-10x10 {
    aspect-ratio: 1;
    border: 1px solid rgba(0,255,123,0.12);
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0,35,50,0.3);
  }
  .sector-row {
    display: grid;
    grid-template-columns: repeat(10, 1fr);
    gap: 4px;
    margin: 5px 0;
  }
  .sector-cell {
    border: 1px solid rgba(255,255,255,0.2);
    padding: 4px 2px;
    text-align: center;
    font-size: 9px;
  }
  .btn {
    background: rgba(0,245,255,0.15);
    border: 1px solid var(--mod-proc);
    color: var(--mod-proc);
    padding: 6px 12px;
    cursor: pointer;
    font-family: inherit;
    font-size: 11px;
    font-weight: bold;
  }
</style>
</head>
<body>

<div class="top-bar">
  <div>
    <h2 style="font-size:17px;color:#fff;">GNC-X RUNTIME VISUAL DISPLAY — MÓDULOS E CAMADAS POR CORES</h2>
    <div id="statusText" style="font-size:12px;color:var(--mod-proc);margin-top:3px;">Inicializando runtime...</div>
  </div>
  <div class="module-badges">
    <div class="badge b-sis"><span class="dot dot-1"></span>1. SISTEMA (Genoma)</div>
    <div class="badge b-proc"><span class="dot dot-2"></span>2. PROCESSADOR (Matriz 64b)</div>
    <div class="badge b-mem"><span class="dot dot-3"></span>3. MEMÓRIA (Camadas/Crivos)</div>
    <div class="badge b-disc"><span class="dot dot-4"></span>4. DISCO (GLA mmap)</div>
    <div class="badge b-db"><span class="dot dot-5"></span>5. BANCO DE DADOS (Tensor)</div>
  </div>
  <div style="display:flex;gap:6px;">
    <button class="btn" onclick="step(-1)">&#9664; ANTERIOR</button>
    <button class="btn" id="playBtn" onclick="togglePlay()">&#9208; PAUSAR</button>
    <button class="btn" onclick="step(1)">PRÓXIMO &#9654;</button>
  </div>
</div>

<!-- MODULO 1: SISTEMA (AZUL) -->
<div class="mod-panel mod-sistema-box">
  <div class="mod-header" style="color:var(--mod-sistema)">
    <span>MÓDULO 1: SISTEMA (O GENOMA &bull; ORQUESTRADOR DE CAMADAS &bull; BYTECODE LMI)</span>
    <span id="cycleCounter">CICLO: 0018</span>
  </div>
  <div class="lmi-strip" id="sistemaStrip"></div>
</div>

<div class="mid-layout">
  <!-- MODULO 2: PROCESSADOR (CIANO) -->
  <div class="mod-panel mod-proc-box">
    <div class="mod-header" style="color:var(--mod-proc)">
      <span>MÓDULO 2: PROCESSADOR (64-BIT MATRIX)</span>
      <span>1.00 INS/CYC</span>
    </div>
    <div style="font-size:11px;color:#9ad8eb;margin-bottom:6px;">Layer 0: 64 MACs/ALUs (8&times;8) &bull; PREDICTOR DISABLED</div>
    <div class="matrix-8x8" id="procMatrix"></div>
  </div>

  <!-- MODULO 3: MEMORIA (VERDE ESMERALDA) -->
  <div class="mod-panel mod-memoria-box">
    <div class="mod-header" style="color:var(--mod-memoria)">
      <span>MÓDULO 3: MEMÓRIA GEOMÉTRICA (CAMADAS DE MAGNITUDE L0..L3 &amp; CRIVOS INVARIANTES DIAGONAIS)</span>
      <span>D(k,d) = d &times; R_{k+1}</span>
    </div>
    <div class="layers-2x2" id="memLayers"></div>
  </div>

  <!-- COLUNA DIREITA: MODULO 4 (DISCO GLA) + MODULO 5 (DB TENSORIAL) -->
  <div style="display:flex;flex-direction:column;gap:14px;">
    <div class="mod-panel mod-disco-box" style="margin-bottom:0;">
      <div class="mod-header" style="color:var(--mod-disco)">
        <span>MÓDULO 4: DISCO GLA (gnc_disk.img)</span>
        <span>mmap O(1)</span>
      </div>
      <div style="font-size:11px;color:#f5d58c;margin-bottom:4px;">Sub-Camadas Persistidas &amp; Prefetch Diagonal:</div>
      <div id="discoGrid"></div>
    </div>

    <div class="mod-panel mod-db-box" style="margin-bottom:0;flex:1;">
      <div class="mod-header" style="color:var(--mod-db)">
        <span>MÓDULO 5: BANCO DE DADOS TENSORIAL</span>
        <span>Sem SQL / Sem B-Tree</span>
      </div>
      <div style="font-size:11px;color:#eeb5ff;margin-bottom:4px;">PROJECT(LAYER2, FILTER_KALMAN) &rarr; LAYER3:</div>
      <div id="dbGrid"></div>
      <div style="margin-top:10px;border-top:1px solid rgba(232,74,255,0.25);padding-top:8px;font-size:11px;">
        <div>Speedup Grade Branchless: <strong style="color:var(--mod-proc)">)HTML"
            << std::fixed << std::setprecision(2) << (bench_branch_ms / std::max(0.001, bench_gncx_grid_ms))
            << R"HTML(x mais rápido</strong></div>
        <div style="margin-top:4px;">Speedup DB Crivo Diagonal O(1): <strong style="color:var(--mod-db)">)HTML"
            << std::fixed << std::setprecision(2) << (bench_tree_ms / std::max(0.001, bench_gncx_db_ms))
            << R"HTML(x mais rápido</strong></div>
      </div>
    </div>
  </div>
</div>

<script>
const LMI_NAMES = ["LDI R0","LDI R1","LDI R2","LDI R3","STI L2","LDI L2","MASK M0","MAC 64","VEC 1K","GLA P1","GLA P2","DB PRJ"];
const LAYER0_VALS = [)HTML";
        for (size_t i = 0; i < 64; ++i) {
            out << reg_mat.layer0_matrix[i] << (i + 1 < 64 ? "," : "");
        }
        out << "];\nconst FRAMES = [\n";
        for (size_t idx = 0; idx < frames_.size(); ++idx) {
            const auto& f = frames_[idx];
            out << "  { cycle:" << f.cycle << ", mod:" << f.active_module
                << ", mnem:\"" << f.instruction_mnemonic << "\", det:\"" << f.instruction_detail << "\", ";
            out << "sis:[";
            for (size_t i = 0; i < 12; ++i) out << static_cast<int>(f.sistema_lmi_dots[i]) << (i + 1 < 12 ? "," : "");
            out << "], proc:[";
            for (size_t i = 0; i < 64; ++i) out << static_cast<int>(f.proc_matrix_dots[i]) << (i + 1 < 64 ? "," : "");
            out << "], mem:[";
            for (int k = 0; k < 4; ++k) {
                out << "[";
                for (int r = 0; r < 10; ++r) {
                    out << "[";
                    for (int c = 0; c < 10; ++c) out << static_cast<int>(f.mem_layer_grids[k][r][c]) << (c + 1 < 10 ? "," : "");
                    out << "]" << (r + 1 < 10 ? "," : "");
                }
                out << "]" << (k + 1 < 4 ? "," : "");
            }
            out << "], disc:[";
            for (int l = 0; l < 4; ++l) {
                out << "[";
                for (int d = 0; d < 10; ++d) out << static_cast<int>(f.disco_gla_dots[l][d]) << (d + 1 < 10 ? "," : "");
                out << "]" << (l + 1 < 4 ? "," : "");
            }
            out << "], db:[";
            for (int t = 0; t < 2; ++t) {
                out << "[";
                for (int d = 0; d < 10; ++d) out << static_cast<int>(f.db_tensor_dots[t][d]) << (d + 1 < 10 ? "," : "");
                out << "]" << (t + 1 < 2 ? "," : "");
            }
            out << "] }" << (idx + 1 < frames_.size() ? ",\n" : "\n");
        }
        out << R"HTML(];
const L_TITLES = [
  "LAYER 0: UNIDADES (0-9) | R1=1",
  "LAYER 1: DEZENAS (00-99) | R2=11",
  "LAYER 2: CENTENAS (000-999) | R3=111",
  "LAYER 3: MILHARES (0000-9999) | R4=1111"
];
let cur = FRAMES.length - 1;
let timer = null;

function draw(idx) {
  const f = FRAMES[idx];
  document.getElementById('cycleCounter').textContent = `CICLO: ${String(f.cycle).padStart(4,'0')} (${idx+1}/${FRAMES.length})`;
  document.getElementById('statusText').innerHTML = `<strong>[${f.mnem}]</strong> ${f.det}`;

  // 1. Sistema
  const sis = document.getElementById('sistemaStrip');
  sis.innerHTML = '';
  for (let i = 0; i < 12; i++) {
    sis.innerHTML += `<div class="lmi-slot">${LMI_NAMES[i]}<br><span class="dot dot-${f.sis[i]}" style="margin-top:4px;"></span></div>`;
  }

  // 2. Processador
  const pm = document.getElementById('procMatrix');
  pm.innerHTML = '';
  for (let i = 0; i < 64; i++) {
    pm.innerHTML += `<div class="cell-8x8"><span class="dot dot-${f.proc[i]}"></span><span style="margin-top:2px;">${LAYER0_VALS[i]}</span></div>`;
  }

  // 3. Memoria
  const ml = document.getElementById('memLayers');
  ml.innerHTML = '';
  for (let k = 0; k < 4; k++) {
    let html = `<div class="layer-card"><div style="font-size:10px;color:var(--mod-memoria);font-weight:bold;">${L_TITLES[k]}</div><div class="grid-10x10">`;
    for (let r = 0; r < 10; r++) {
      for (let c = 0; c < 10; c++) {
        html += `<div class="cell-10x10"><span class="dot dot-${f.mem[k][r][c]}"></span></div>`;
      }
    }
    html += `</div></div>`;
    ml.innerHTML += html;
  }

  // 4. Disco GLA
  const dg = document.getElementById('discoGrid');
  dg.innerHTML = '';
  for (let l = 0; l < 4; l++) {
    let row = `<div class="sector-row">`;
    for (let d = 0; d < 10; d++) {
      row += `<div class="sector-cell">L${l}:D${d}<br><span class="dot dot-${f.disc[l][d]}" style="margin-top:3px;"></span></div>`;
    }
    row += `</div>`;
    dg.innerHTML += row;
  }

  // 5. DB Tensor
  const dbg = document.getElementById('dbGrid');
  dbg.innerHTML = '';
  const labels = ["Fonte L2", "Proj L3"];
  for (let t = 0; t < 2; t++) {
    let row = `<div style="font-size:10px;margin-top:4px;">${labels[t]}:</div><div class="sector-row">`;
    for (let d = 0; d < 10; d++) {
      row += `<div class="sector-cell">T${d}<br><span class="dot dot-${f.db[t][d]}" style="margin-top:3px;"></span></div>`;
    }
    row += `</div>`;
    dbg.innerHTML += row;
  }
}

function step(d) {
  cur = (cur + d + FRAMES.length) % FRAMES.length;
  draw(cur);
}
function togglePlay() {
  const b = document.getElementById('playBtn');
  if (timer) { clearInterval(timer); timer = null; b.innerHTML = '&#9654; PLAY'; }
  else { b.innerHTML = '&#9208; PAUSAR'; timer = setInterval(() => step(1), 1000); }
}
draw(cur);
togglePlay();
</script>
</body>
</html>
)HTML";
    }

private:
    VisualFrameSnapshot current_{};
    std::vector<VisualFrameSnapshot> frames_{};
};

} // namespace gncx
