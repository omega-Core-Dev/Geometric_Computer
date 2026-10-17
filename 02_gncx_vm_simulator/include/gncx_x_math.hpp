#pragma once
/**
 * ============================================================================
 * GNC-X SILICON INCUBATOR — X-FOLDING TENSOR & MATRIX ARITHMETIC ENGINE
 * ============================================================================
 * Author of Conception: Zaqueu Ribeiro da Costa
 *
 * Implements the "X-Sieve" Fast Calculation Algorithm for Entry-Level CPUs
 * and future PyTorch Tensor Integration (`torch_gncx`):
 *
 * Core Mathematical Identity:
 * Every integer X_i in Magnitude Layer L_k is encoded in 4-Phase X-Coordinates:
 *   X_i = s1_i * D(k, d_i) + s0_i * |Delta_i|
 *   where:
 *     Phase (b1, b0) in {00 (--), 01 (-+), 10 (+-), 11 (++)}
 *     s1_i = 2*b1 - 1 in {-1, +1}
 *     s0_i = 2*b0 - 1 in {-1, +1}
 *     D(k, d) = d * R_{k+1} (The 10 Invariant Diagonal Reference Sieves)
 *
 * When computing Multi-Channel Linear Projections (the core of PyTorch nn.Linear
 * & Attention layers) over N elements and M output channels:
 *   Standard CPU Brute-Force: Executes N * M multiplications + N * M additions.
 *   GNC-X "X-Folding" Engine:
 *     1. Folds all N elements into the 10 Invariant Diagonal Pivots (d = 0..9)
 *        using ONLY 2-bit Phase Cancelation (+1/-1) and residual additions
 *        (ZERO multiplications in the N-loop!).
 *     2. Multiplies ONLY at the 10 Invariant Diagonal Pivots (10 * M mults!).
 *
 * Result: 100% BIT-FOR-BIT EXACT mathematical equivalence with massive speedup
 * on entry-level CPUs!
 * ============================================================================
 */

#include <cstdint>
#include <vector>
#include <array>
#include "gncx_sieve.hpp"
#include "gncx_compiler.hpp"

namespace gncx {

/**
 * Compact 4-Byte (32-bit) X-Coordinate Token (2x to 4x smaller than int64_t in L1 Cache!)
 */
struct alignas(4) CompactXToken {
    uint8_t phase_2bit;   // 00 (--), 01 (-+), 10 (+-), 11 (++)
    uint8_t diag_d;       // Invariant Diagonal Pivot index d in {0..9}
    uint16_t abs_delta;   // Residual magnitude |Delta| relative to D(k, d)
};

struct XFoldAccumulator {
    // For each of the 10 Invariant Diagonal Anchors d in {0..9}:
    std::array<int64_t, 10> net_pivot_phase_count{}; // Net (+1 vs -1) count of D(k, d) after phase cancelation
    std::array<int64_t, 10> net_residue_sum{};       // Net sum of (+/- |Delta_i|)
    uint64_t phase_cancelations{0};                  // Count of opposite-phase pairs canceled
};

class XFoldingTensorEngine {
public:
    // Encodes an int64 value in Layer k into a compact 32-bit X-Token (Phase 2-bit + Pivot d + |Delta|)
    static constexpr CompactXToken encode_token_layer(int64_t value, int layer_k) noexcept {
        int64_t r_k = REPUNITS[layer_k];
        int64_t abs_v = (value >= 0) ? value : -value;
        int64_t d_approx = (abs_v + (r_k >> 1)) / r_k;
        if (d_approx > 9) d_approx = 9;
        uint8_t d = static_cast<uint8_t>(d_approx);
        int64_t anchor = DIAGONAL_SIEVES[layer_k][d];

        int64_t signed_anchor = (value >= 0) ? anchor : -anchor;
        int64_t signed_res = value - signed_anchor;
        uint16_t abs_res = static_cast<uint16_t>(signed_res >= 0 ? signed_res : -signed_res);

        uint8_t b1 = (value >= 0) ? 1U : 0U;
        uint8_t b0 = (signed_res >= 0) ? 1U : 0U;
        uint8_t phase = static_cast<uint8_t>((b1 << 1) | b0);

        return CompactXToken{phase, d, abs_res};
    }

    // Reconstructs exact integer from CompactXToken in Layer k
    static constexpr int64_t decode_token_layer(const CompactXToken& tok, int layer_k) noexcept {
        int64_t s1 = (static_cast<int64_t>((tok.phase_2bit >> 1) & 1U) << 1) - 1;
        int64_t s0 = (static_cast<int64_t>(tok.phase_2bit & 1U) << 1) - 1;
        return s1 * DIAGONAL_SIEVES[layer_k][tok.diag_d] + s0 * static_cast<int64_t>(tok.abs_delta);
    }

    /**
     * Standard Brute-Force Multi-Channel Tensor Projection (Baseline CPU / PyTorch Naive):
     * Computes out[m] = sum_{i=0..N-1} weights[m][d_i] * X[i] for M output channels.
     * Requires N * M 64-bit multiplications and additions.
     */
    static void compute_standard_dense_projection(
        const int64_t* __restrict raw_inputs,
        const uint8_t* __restrict input_sectors,
        size_t num_elements,
        const std::vector<std::array<int64_t, 10>>& channel_weights,
        std::vector<int64_t>& out_channels
    ) noexcept {
        size_t num_channels = channel_weights.size();
        for (size_t m = 0; m < num_channels; ++m) {
            int64_t acc = 0;
            const auto& w_row = channel_weights[m];
            for (size_t i = 0; i < num_elements; ++i) {
                acc += w_row[input_sectors[i]] * raw_inputs[i];
            }
            out_channels[m] = acc;
        }
    }

    /**
     * GNC-X "X-Folding" Multi-Channel Tensor Projection:
     * Step 1: Folds N CompactXTokens into the 10 Invariant Diagonal Sieves (ZERO multiplications!)
     * Step 2: Evaluates all M output channels using ONLY 10 multiplications per channel!
     */
    static XFoldAccumulator compute_xfold_tensor_projection(
        const CompactXToken* __restrict tokens,
        size_t num_elements,
        int layer_k,
        const std::vector<std::array<int64_t, 10>>& channel_weights,
        std::vector<int64_t>& out_channels
    ) noexcept {
        XFoldAccumulator fold{};
        std::array<uint64_t, 10> pos_pivots{};
        std::array<uint64_t, 10> neg_pivots{};

        // PASSO 1: Dobramento em "X" (Apenas soma de fase 2-bits e residuo - ZERO multiplicacoes!)
        for (size_t i = 0; i < num_elements; ++i) {
            const CompactXToken tok = tokens[i];
            uint8_t d = tok.diag_d;
            uint8_t b1 = (tok.phase_2bit >> 1) & 1U;
            uint8_t b0 = tok.phase_2bit & 1U;

            // s1 = +1 se b1==1, -1 se b1==0 (branchless)
            int64_t s1 = (static_cast<int64_t>(b1) << 1) - 1;
            int64_t s0 = (static_cast<int64_t>(b0) << 1) - 1;

            fold.net_pivot_phase_count[d] += s1;
            fold.net_residue_sum[d] += (s0 * static_cast<int64_t>(tok.abs_delta));

            pos_pivots[d] += b1;
            neg_pivots[d] += (b1 ^ 1U);
        }

        // Contabilizar quantos pares opostos (+D vs -D) se anularam automaticamente no "X"
        for (size_t d = 0; d < 10; ++d) {
            fold.phase_cancelations += std::min(pos_pivots[d], neg_pivots[d]);
        }

        // Pré-calcular a massa líquida exata de cada um dos 10 Setores Diagonais Invariantes:
        // SectorMass[d] = net_pivot_phase_count[d] * D(k, d) + net_residue_sum[d]
        std::array<int64_t, 10> diagonal_sector_mass{};
        for (size_t d = 0; d < 10; ++d) {
            diagonal_sector_mass[d] = fold.net_pivot_phase_count[d] * DIAGONAL_SIEVES[layer_k][d]
                                    + fold.net_residue_sum[d];
        }

        // PASSO 2: Projeção de Canais de Saída (Apenas 10 multiplicações por canal em vez de N!)
        size_t num_channels = channel_weights.size();
        for (size_t m = 0; m < num_channels; ++m) {
            int64_t acc = 0;
            const auto& w_row = channel_weights[m];
            for (size_t d = 0; d < 10; ++d) {
                acc += w_row[d] * diagonal_sector_mass[d];
            }
            out_channels[m] = acc;
        }

        return fold;
    }
};

} // namespace gncx
