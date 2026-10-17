#pragma once
/**
 * ============================================================================
 * GNC-X SILICON INCUBATOR — THE "GENOME" SPATIAL OS & LMI BYTECODE VM
 * ============================================================================
 * Author of Conception: Zaqueu Ribeiro da Costa
 *
 * Unifies and instruments for real-time visual display:
 *   1. SISTEMA (Genoma / LMI Bytecode)
 *   2. PROCESSADOR (64-Bit Register Matrix & Branchless Pipeline)
 *   3. MEMORIA (Magnitude Layers L0..L3 & Invariant Diagonal Sieves)
 *   4. DISCO (GLA Volume `gnc_disk.img` via mmap)
 *   5. BANCO DE DADOS (Spatial Tensor Algebraic Engine)
 * ============================================================================
 */

#include <cstdint>
#include <vector>
#include <string>
#include <sstream>
#include "gncx_sieve.hpp"
#include "gncx_register_matrix.hpp"
#include "gncx_memory.hpp"
#include "gncx_gla_disk.hpp"
#include "gncx_spatial_db.hpp"
#include "gncx_visual_display.hpp"

namespace gncx {

enum class LMIOpcode : uint8_t {
    LDI_IMM,         // Load Immediate into Register R[dst] and decompose via Diagonal Sieve
    LDI_LAYER,       // Load from Geometric Memory Layer[k, d, offset] into R[dst]
    STI_LAYER,       // Store R[src] into Geometric Memory Layer[k, d, offset]
    PREDICATE_MASK,  // Compute 64-bit branchless mask M[dst] = (R[a] > R[b])
    OP_GRID_MAC,     // Execute simultaneous MAC on Layer 0 (64-unit 8x8 Register Matrix) under M[mask]
    OP_GRID_VECTOR,  // Execute branchless vector transform on Layer 1/2 (10k / 1M PEs)
    GLA_PERSIST,     // Persist R[key], R[payload] to `gnc_disk.img` via Invariant Diagonal Sieve
    DB_PROJECT       // Execute Spatial Tensor DB projection PROJECT(L_src, FILTER_KALMAN) -> L_dst
};

struct LMIInstruction {
    LMIOpcode opcode;
    int32_t arg0{0};
    int32_t arg1{0};
    int32_t arg2{0};
    int64_t imm{0};
};

class GenomeSpatialVM {
public:
    explicit GenomeSpatialVM(const std::string& disk_image_path)
        : reg_matrix_(),
          pipeline_(reg_matrix_),
          memory_(),
          disk_(disk_image_path),
          db_(memory_, disk_),
          display_() {}

    void execute_program(const std::vector<LMIInstruction>& program) {
        for (size_t idx = 0; idx < program.size(); ++idx) {
            execute_single(idx, program[idx]);
        }
    }

    void execute_single(size_t step_idx, const LMIInstruction& ins) {
        GeometricCoordinate empty_coord{};
        switch (ins.opcode) {
            case LMIOpcode::LDI_IMM: {
                int r_dst = ins.arg0 & 15;
                reg_matrix_.regs[r_dst] = ins.imm;
                GeometricCoordinate c = InvariantDiagonalSieve::decompose(ins.imm);
                reg_matrix_.layer0_matrix[r_dst] = c.primary_anchor;
                std::ostringstream oss;
                oss << "R" << r_dst << " <- " << ins.imm << " | Crivo: " << c.format_trajectory();
                pipeline_.step_instruction("LDI_IMM", oss.str());
                display_.record_lmi_step(
                    step_idx, pipeline_.telemetry().total_cycles,
                    1, "LDI_IMM", oss.str(), c, c.primary_layer_k, c.primary_diagonal_d
                );
                break;
            }
            case LMIOpcode::LDI_LAYER: {
                int r_dst = ins.arg0 & 15;
                int layer_k = ins.arg1 & 3;
                int diag_d = ins.arg2 % 10;
                int64_t val = memory_.read_by_sieve(layer_k, diag_d, ins.imm);
                reg_matrix_.regs[r_dst] = val;
                std::ostringstream oss;
                oss << "R" << r_dst << " <- LAYER" << layer_k << "[D=" << diag_d << ", off=" << ins.imm << "] (" << val << ")";
                pipeline_.step_instruction("LDI_LAYER", oss.str());
                display_.record_lmi_step(
                    step_idx, pipeline_.telemetry().total_cycles,
                    3, "LDI_LAYER", oss.str(), empty_coord, layer_k, diag_d
                );
                break;
            }
            case LMIOpcode::STI_LAYER: {
                int r_src = ins.arg0 & 15;
                int layer_k = ins.arg1 & 3;
                int diag_d = ins.arg2 % 10;
                int64_t val = reg_matrix_.regs[r_src];
                memory_.write_by_sieve(layer_k, diag_d, ins.imm, val);
                std::ostringstream oss;
                oss << "LAYER" << layer_k << "[D=" << diag_d << ", off=" << ins.imm << "] <- R" << r_src << " (" << val << ")";
                pipeline_.step_instruction("STI_LAYER", oss.str());
                display_.record_lmi_step(
                    step_idx, pipeline_.telemetry().total_cycles,
                    3, "STI_LAYER", oss.str(), empty_coord, layer_k, diag_d
                );
                break;
            }
            case LMIOpcode::PREDICATE_MASK: {
                int m_dst = ins.arg0 & 15;
                int r_a = ins.arg1 & 15;
                int r_b = ins.arg2 & 15;
                uint64_t mask = branchless_mask_gt(reg_matrix_.regs[r_a], reg_matrix_.regs[r_b]);
                reg_matrix_.masks[m_dst] = mask;
                pipeline_.telemetry_mut().predicate_masks_evaluated += 1;
                std::ostringstream oss;
                oss << "M" << m_dst << " <- MASK_GT(R" << r_a << ", R" << r_b << ") = 0x"
                    << std::hex << mask << std::dec << " (Branchless)";
                pipeline_.step_instruction("PREDICATE_MASK", oss.str());
                display_.record_lmi_step(
                    step_idx, pipeline_.telemetry().total_cycles,
                    2, "PREDICATE_MASK", oss.str(), empty_coord, 0, r_a
                );
                break;
            }
            case LMIOpcode::OP_GRID_MAC: {
                int m_idx = ins.arg0 & 15;
                int64_t factor_a = ins.arg1;
                int64_t offset_b = ins.imm;
                uint64_t mask = reg_matrix_.masks[m_idx];
                reg_matrix_.execute_layer0_mac_grid(factor_a, offset_b, mask);
                std::ostringstream oss;
                oss << "Layer0_64_MACs(x*" << factor_a << " + " << offset_b << ") under M" << m_idx;
                pipeline_.step_instruction("OP_GRID_MAC", oss.str());
                display_.record_lmi_step(
                    step_idx, pipeline_.telemetry().total_cycles,
                    2, "OP_GRID_MAC", oss.str(), empty_coord, 0, 0
                );
                break;
            }
            case LMIOpcode::OP_GRID_VECTOR: {
                int layer_k = ins.arg0 & 3;
                int diag_d = ins.arg1 % 10;
                size_t count = static_cast<size_t>(ins.arg2);
                int64_t threshold = ins.imm;
                memory_.execute_branchless_grid_transform(layer_k, diag_d, count, threshold, 2, 11);
                std::ostringstream oss;
                oss << "Layer" << layer_k << "[D=" << diag_d << ", N=" << count << "] Branchless SIMD Transform";
                pipeline_.step_instruction("OP_GRID_VECTOR", oss.str());
                display_.record_lmi_step(
                    step_idx, pipeline_.telemetry().total_cycles,
                    3, "OP_GRID_VECTOR", oss.str(), empty_coord, layer_k, diag_d
                );
                break;
            }
            case LMIOpcode::GLA_PERSIST: {
                int r_key = ins.arg0 & 15;
                int r_val = ins.arg1 & 15;
                int64_t key = reg_matrix_.regs[r_key];
                int64_t val = reg_matrix_.regs[r_val];
                size_t byte_offset = 0;
                GeometricCoordinate c = disk_.persist_by_sieve(key, val, byte_offset);
                std::ostringstream oss;
                oss << "GLA[gnc_disk.img @ 0x" << std::hex << byte_offset << std::dec
                    << "] <- Key=" << key << " (Pivot D(" << c.primary_layer_k << "," << c.primary_diagonal_d << "))";
                pipeline_.step_instruction("GLA_PERSIST", oss.str());
                display_.record_lmi_step(
                    step_idx, pipeline_.telemetry().total_cycles,
                    4, "GLA_PERSIST", oss.str(), c, c.primary_layer_k, c.primary_diagonal_d
                );
                break;
            }
            case LMIOpcode::DB_PROJECT: {
                int src_layer = ins.arg0 & 3;
                int dst_layer = ins.arg1 & 3;
                int diag_d = ins.arg2 % 10;
                size_t count = static_cast<size_t>(ins.imm);
                size_t projected = db_.project_and_filter_kalman(src_layer, dst_layer, diag_d, count, 100, 75);
                std::ostringstream oss;
                oss << "PROJECT(LAYER" << src_layer << "[D=" << diag_d << "], FILTER_KALMAN) -> LAYER"
                    << dst_layer << " (" << projected << " elementos)";
                pipeline_.step_instruction("DB_PROJECT", oss.str());
                display_.record_lmi_step(
                    step_idx, pipeline_.telemetry().total_cycles,
                    5, "DB_PROJECT", oss.str(), empty_coord, dst_layer, diag_d
                );
                break;
            }
        }
    }

    RegisterMatrix64& reg_matrix() noexcept { return reg_matrix_; }
    BranchlessPipeline& pipeline() noexcept { return pipeline_; }
    GeometricMemoryHierarchy& memory() noexcept { return memory_; }
    GLADiskEngine& disk() noexcept { return disk_; }
    SpatialTensorDB& db() noexcept { return db_; }
    RuntimeVisualDisplay& display() noexcept { return display_; }

private:
    RegisterMatrix64 reg_matrix_;
    BranchlessPipeline pipeline_;
    GeometricMemoryHierarchy memory_;
    GLADiskEngine disk_;
    SpatialTensorDB db_;
    RuntimeVisualDisplay display_;
};

} // namespace gncx
