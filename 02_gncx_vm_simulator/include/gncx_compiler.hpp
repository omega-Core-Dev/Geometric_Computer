#pragma once
/**
 * ============================================================================
 * GNC-X SILICON INCUBATOR — FASE 3: COMPILADOR / TRADUTOR CIENTE DA GEOMETRIA
 * ============================================================================
 * Author of Conception: Zaqueu Ribeiro da Costa
 *
 * Implements Phase 3 of the GNC-X Whitepaper:
 * 1. 4-Phase Quadrant Logic centered at 0 (`+ = 1`, `- = 0`):
 *      Phase 11 (3) = [+, +] (+Pivot, +Residue) -> Quadrant I
 *      Phase 10 (2) = [+, -] (+Pivot, -Residue) -> Quadrant IV
 *      Phase 01 (1) = [-, +] (-Pivot, +Residue) -> Quadrant II
 *      Phase 00 (0) = [-, -] (-Pivot, -Residue) -> Quadrant III
 *    Negation (-X) is a 1-cycle zero-carry 2-bit inversion: `phase ^ 0b11`!
 *
 * 2. Full Decimal Layer Coupling (All Decimal Layers L0..L3 active in parallel)
 *
 * 3. Geometry-Aware Translator (DSL / C-subset -> GNC-X LMI Bytecode):
 *    Parses high-level algorithmic statements, automatically maps variables
 *    to 4-Phase Quadrant Coordinates across all decimal layers, and emits
 *    branchless LMI bytecode.
 * ============================================================================
 */

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <algorithm>
#include <cctype>

#include "gncx_sieve.hpp"
#include "gncx_genome_vm.hpp"

namespace gncx {

/**
 * 2-Bit 4-Phase Quadrant Encoding (`+ = 1`, `- = 0`)
 * Bit 1 (MSB): Pivot Sign   (1 = +, 0 = -)
 * Bit 0 (LSB): Residue Sign (1 = +, 0 = -)
 */
enum class PhaseQuadrant : uint8_t {
    PHASE_NN = 0b00, // [-, -] : -D(k, d) - |Delta|
    PHASE_NP = 0b01, // [-, +] : -D(k, d) + |Delta|
    PHASE_PN = 0b10, // [+, -] : +D(k, d) - |Delta|
    PHASE_PP = 0b11  // [+, +] : +D(k, d) + |Delta|
};

constexpr const char* phase_to_symbol(PhaseQuadrant p) noexcept {
    switch (p) {
        case PhaseQuadrant::PHASE_PP: return "++ (11)";
        case PhaseQuadrant::PHASE_PN: return "+- (10)";
        case PhaseQuadrant::PHASE_NP: return "-+ (01)";
        case PhaseQuadrant::PHASE_NN: return "-- (00)";
    }
    return "++ (11)";
}

// Converts a single phase bit (0 or 1) to algebraic sign (-1 or +1) branchlessly: 2*b - 1
constexpr int64_t phase_bit_to_sign(uint8_t bit) noexcept {
    return (static_cast<int64_t>(bit & 1U) << 1) - 1;
}

// 1-Cycle Zero-Carry Negation via 2-Bit XOR 0b11!
constexpr PhaseQuadrant invert_phase_quadrant(PhaseQuadrant p) noexcept {
    return static_cast<PhaseQuadrant>(static_cast<uint8_t>(p) ^ 0b11U);
}

/**
 * Represents a Single Layer Slice inside the 4-Phase Coupled Decimal Stack.
 */
struct LayerPhaseSlice {
    int32_t layer_k{0};
    PhaseQuadrant phase{PhaseQuadrant::PHASE_PP};
    int32_t digit_d{0};
    int64_t abs_diagonal{0};
    int64_t abs_residue{0};

    constexpr int64_t evaluate_level_value() const noexcept {
        uint8_t bits = static_cast<uint8_t>(phase);
        int64_t s_piv = phase_bit_to_sign((bits >> 1) & 1U);
        return s_piv * abs_diagonal;
    }
};

/**
 * Full 4-Phase Multi-Layer Coupled Coordinate (Leveraging ALL Decimal Layers L0..L3+).
 */
struct CoupledPhaseCoordinate {
    int64_t original_value{0};
    PhaseQuadrant primary_phase{PhaseQuadrant::PHASE_PP};
    int32_t top_layer_k{0};
    int32_t top_digit_d{0};
    int64_t top_abs_anchor{0};
    int64_t top_abs_residue{0};
    // Active state across all 6 lower decimal layers L0..L5 simultaneously
    std::array<LayerPhaseSlice, 6> decimal_layers{};

    // Reconstructs the primary value using the 2-bit 4-Phase equation:
    // X = (2*b1 - 1)*D(k, d) + (2*b0 - 1)*|Delta|
    constexpr int64_t reconstruct_from_phase() const noexcept {
        if (original_value == 0) return 0;
        uint8_t bits = static_cast<uint8_t>(primary_phase);
        int64_t s_piv = phase_bit_to_sign((bits >> 1) & 1U);
        int64_t s_res = phase_bit_to_sign(bits & 1U);
        return s_piv * top_abs_anchor + s_res * top_abs_residue;
    }

    // Reconstructs by summing all coupled decimal layers L_k..L_0
    constexpr int64_t reconstruct_from_all_layers() const noexcept {
        int64_t total = 0;
        for (const auto& sl : decimal_layers) {
            if (sl.digit_d > 0) {
                total += sl.evaluate_level_value();
            }
        }
        return total;
    }

    // Negates the entire coordinate in 1 cycle by flipping the 2-bit phases (XOR 0b11)
    constexpr CoupledPhaseCoordinate negate_by_phase_xor() const noexcept {
        CoupledPhaseCoordinate neg = *this;
        neg.original_value = -original_value;
        neg.primary_phase = invert_phase_quadrant(primary_phase);
        for (auto& sl : neg.decimal_layers) {
            if (sl.digit_d > 0) {
                sl.phase = invert_phase_quadrant(sl.phase);
            }
        }
        return neg;
    }
};

class FourPhaseSieveEngine {
public:
    static CoupledPhaseCoordinate encode(int64_t value) noexcept {
        CoupledPhaseCoordinate out{};
        out.original_value = value;
        for (int k = 0; k < 6; ++k) {
            out.decimal_layers[k].layer_k = k;
        }

        if (value == 0) {
            return out;
        }

        GeometricCoordinate base = InvariantDiagonalSieve::decompose(value);
        out.top_layer_k = base.primary_layer_k;
        out.top_digit_d = base.primary_diagonal_d;
        out.top_abs_anchor = DIAGONAL_SIEVES[base.primary_layer_k][base.primary_diagonal_d];

        int64_t signed_anchor = (value >= 0) ? out.top_abs_anchor : -out.top_abs_anchor;
        int64_t signed_res = value - signed_anchor;
        out.top_abs_residue = (signed_res >= 0) ? signed_res : -signed_res;

        uint8_t b1 = (value >= 0) ? 1U : 0U;
        uint8_t b0 = (signed_res >= 0) ? 1U : 0U;
        out.primary_phase = static_cast<PhaseQuadrant>((b1 << 1) | b0);

        // Populate ALL Decimal Layers L0..L5 simultaneously from the decomposition terms
        for (uint32_t i = 0; i < base.num_terms; ++i) {
            const auto& t = base.terms[i];
            if (t.layer_k >= 0 && t.layer_k < 6) {
                auto& slice = out.decimal_layers[t.layer_k];
                // Accumulate if multiple terms land in L0
                int64_t prev_val = (slice.digit_d > 0) ? slice.evaluate_level_value() : 0;
                int64_t new_val = prev_val + static_cast<int64_t>(t.sign) * t.diagonal_value;
                uint8_t sb1 = (new_val >= 0) ? 1U : 0U;
                uint8_t sb0 = b0;
                slice.phase = static_cast<PhaseQuadrant>((sb1 << 1) | sb0);
                slice.abs_diagonal = (new_val >= 0) ? new_val : -new_val;
                slice.digit_d = static_cast<int32_t>(slice.abs_diagonal / REPUNITS[t.layer_k]);
            }
        }

        return out;
    }
};

/**
 * Symbol Table Entry for the Geometry-Aware GNC-X Compiler (Phase 3)
 */
struct CompiledSymbol {
    std::string name;
    int32_t assigned_reg;
    CoupledPhaseCoordinate phase_coord;
};

struct CompilationReport {
    std::vector<CompiledSymbol> symbols;
    std::vector<LMIInstruction> bytecode;
    std::vector<std::string> assembly_listing;
};

/**
 * Phase 3: Geometry-Aware Compiler / Translator (GNC-X DSL -> LMI Bytecode)
 *
 * Translates a high-level GNC-X source script into LMI Bytecode while automatically:
 * 1. Assigning 4-Phase Quadrant Codes (`++=11, +-=10, -+=01, --=00`)
 * 2. Coupling variables across all Decimal Magnitude Layers (L0..L3)
 * 3. Generating 100% branchless predicate masks and GLA/DB spatial instructions
 */
class GeometryAwareCompiler {
public:
    CompilationReport compile_source(const std::string& source_code) {
        CompilationReport report{};
        std::istringstream stream(source_code);
        std::string line;
        int32_t next_reg = 0;

        while (std::getline(stream, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty() || trimmed[0] == '#' || trimmed.substr(0, 2) == "//") {
                continue;
            }

            std::istringstream ls(trimmed);
            std::string keyword;
            ls >> keyword;

            if (keyword == "var" || keyword == "let") {
                // Syntax: var <name> = <integer_literal>
                std::string var_name, eq_sign;
                int64_t val = 0;
                ls >> var_name >> eq_sign >> val;

                int32_t reg = next_reg++ & 15;
                CoupledPhaseCoordinate pcoord = FourPhaseSieveEngine::encode(val);
                symbol_table_[var_name] = CompiledSymbol{var_name, reg, pcoord};
                report.symbols.push_back(symbol_table_[var_name]);

                // 1. Emit LDI_IMM for the primary register
                report.bytecode.push_back(LMIInstruction{LMIOpcode::LDI_IMM, reg, 0, 0, val});

                // 2. Emit STI_LAYER across ALL active Decimal Layers (L0..L3) simultaneously!
                for (int k = 0; k <= 3; ++k) {
                    const auto& sl = pcoord.decimal_layers[k];
                    if (sl.digit_d > 0) {
                        int64_t phase_tagged_offset = static_cast<int64_t>(sl.phase) * 16 + sl.digit_d;
                        report.bytecode.push_back(LMIInstruction{
                            LMIOpcode::STI_LAYER, reg, k, sl.digit_d, phase_tagged_offset
                        });
                    }
                }

                std::ostringstream asm_line;
                asm_line << "ALLOC_4PHASE " << std::setw(8) << var_name << " -> R" << reg
                         << " | Fase=" << phase_to_symbol(pcoord.primary_phase)
                         << " | Pivô=D(" << pcoord.top_layer_k << "," << pcoord.top_digit_d << ")=" << pcoord.top_abs_anchor
                         << " | |Delta|=" << pcoord.top_abs_residue;
                report.assembly_listing.push_back(asm_line.str());
            } else if (keyword == "negate_phase") {
                // Syntax: negate_phase <dst_var> = <src_var>
                std::string dst_name, eq_sign, src_name;
                ls >> dst_name >> eq_sign >> src_name;
                if (symbol_table_.count(src_name)) {
                    const auto& src_sym = symbol_table_[src_name];
                    CoupledPhaseCoordinate neg_coord = src_sym.phase_coord.negate_by_phase_xor();
                    int32_t reg = next_reg++ & 15;
                    int64_t neg_val = neg_coord.reconstruct_from_phase();
                    symbol_table_[dst_name] = CompiledSymbol{dst_name, reg, neg_coord};
                    report.symbols.push_back(symbol_table_[dst_name]);

                    report.bytecode.push_back(LMIInstruction{LMIOpcode::LDI_IMM, reg, 0, 0, neg_val});
                    std::ostringstream asm_line;
                    asm_line << "XOR_PHASE_11 " << std::setw(8) << dst_name << " -> R" << reg
                             << " | " << phase_to_symbol(src_sym.phase_coord.primary_phase)
                             << " ^ 11 -> " << phase_to_symbol(neg_coord.primary_phase)
                             << " (Valor=" << neg_val << ", Zero Carry!)";
                    report.assembly_listing.push_back(asm_line.str());
                }
            } else if (keyword == "when_gt") {
                // Syntax: when_gt <varA> <varB> -> mac_grid <factor> <bias>
                std::string var_a, var_b, arrow, cmd;
                int32_t factor = 3;
                int64_t bias = 7;
                ls >> var_a >> var_b >> arrow >> cmd >> factor >> bias;
                int32_t r_a = symbol_table_.count(var_a) ? symbol_table_[var_a].assigned_reg : 0;
                int32_t r_b = symbol_table_.count(var_b) ? symbol_table_[var_b].assigned_reg : 1;

                report.bytecode.push_back(LMIInstruction{LMIOpcode::PREDICATE_MASK, 0, r_a, r_b, 0});
                report.bytecode.push_back(LMIInstruction{LMIOpcode::OP_GRID_MAC, 0, factor, 0, bias});

                std::ostringstream asm_line;
                asm_line << "BRANCHLESS_IF (" << var_a << " > " << var_b << ") -> M0 + OP_GRID_MAC(x*"
                         << factor << " + " << bias << ")";
                report.assembly_listing.push_back(asm_line.str());
            } else if (keyword == "persist_gla") {
                // Syntax: persist_gla <varKey> <varPayload>
                std::string var_k, var_p;
                ls >> var_k >> var_p;
                int32_t r_k = symbol_table_.count(var_k) ? symbol_table_[var_k].assigned_reg : 0;
                int32_t r_p = symbol_table_.count(var_p) ? symbol_table_[var_p].assigned_reg : 1;
                report.bytecode.push_back(LMIInstruction{LMIOpcode::GLA_PERSIST, r_k, r_p, 0, 0});
                report.assembly_listing.push_back("GLA_MMAP_PERSIST " + var_k + ", " + var_p);
            } else if (keyword == "project_db") {
                // Syntax: project_db L<src> -> L<dst> sector <d> count <N>
                int32_t src_l = 2, dst_l = 3, diag_d = 7;
                int64_t count = 1024;
                ls >> src_l >> dst_l >> diag_d >> count;
                report.bytecode.push_back(LMIInstruction{LMIOpcode::DB_PROJECT, src_l, dst_l, diag_d, count});
                report.assembly_listing.push_back(
                    "DB_TENSOR_PROJECT L" + std::to_string(src_l) + "[D=" + std::to_string(diag_d) +
                    "] -> L" + std::to_string(dst_l) + " (N=" + std::to_string(count) + ")"
                );
            }
        }

        return report;
    }

private:
    static std::string trim(const std::string& s) {
        size_t start = 0;
        while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
        size_t end = s.size();
        while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
        return s.substr(start, end - start);
    }

    std::unordered_map<std::string, CompiledSymbol> symbol_table_{};
};

} // namespace gncx
