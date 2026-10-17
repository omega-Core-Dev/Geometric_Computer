#pragma once
/**
 * ============================================================================
 * GNC-X SILICON INCUBATOR — GEOMETRIC MEMORY HIERARCHY (LAYERS L0..L3)
 * ============================================================================
 * Author of Conception: Zaqueu Ribeiro da Costa
 *
 * Incubates the GNC-X Magnitude Memory Layers on contiguous host memory buffers,
 * partitioned by the 10 Invariant Diagonal Reference Sieves (d in 0..9):
 * - Layer 0 (64 cells): 1 cycle latency (Register Matrix / L1 immediate)
 * - Layer 1 (10,000 vector cells): 1 cycle latency (L1 cache tier)
 * - Layer 2 (1,000,000 PE cells): 10 cycles latency (L2/L3 cache tier)
 * - Layer 3 (1,000,000 DRAM cells): 100 cycles latency (DRAM tier)
 * ============================================================================
 */

#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include "gncx_sieve.hpp"
#include "gncx_register_matrix.hpp"

namespace gncx {

struct MemoryLayerSpec {
    int layer_id;
    const char* name;
    size_t total_cells;
    size_t sector_stride;
    uint64_t cycle_latency;
};

class GeometricMemoryHierarchy {
public:
    static constexpr std::array<MemoryLayerSpec, 4> SPECS{{
        {0, "Layer 0 (Unidades / 64 MACs)",       64,         6,       1},
        {1, "Layer 1 (Dezenas / 10k Vetorial)",   10'000,     1'000,   1},
        {2, "Layer 2 (Centenas / 1M PEs)",        1'000'000,  100'000, 10},
        {3, "Layer 3 (Milhares-Milhoes / DRAM)",  2'000'000,  200'000, 100}
    }};

    GeometricMemoryHierarchy() {
        for (size_t k = 0; k < 4; ++k) {
            buffers_[k].resize(SPECS[k].total_cells, 0);
            access_counts_[k] = 0;
        }
    }

    // Algebraic O(1) slot address anchored directly on the Invariant Diagonal Pivot D(k, d) = d * R_{k+1}
    // Bijective across all layers: slot = D(k, d) + signed_residue
    static constexpr size_t compute_spatial_index(int magnitude_k, int diagonal_d, int64_t signed_offset) noexcept {
        int mem_tier = (magnitude_k <= 0) ? 0 : ((magnitude_k <= 1) ? 1 : ((magnitude_k == 2) ? 2 : 3));
        const auto& spec = SPECS[mem_tier];
        int k_clamped = (magnitude_k < 0) ? 0 : ((magnitude_k > MAX_LAYER_K) ? MAX_LAYER_K : magnitude_k);
        int d_clamped = (diagonal_d < 0) ? 0 : ((diagonal_d > 9) ? 9 : diagonal_d);
        int64_t diagonal_pivot = DIAGONAL_SIEVES[k_clamped][d_clamped];
        int64_t raw_idx = diagonal_pivot + signed_offset;
        if (raw_idx < 0) raw_idx = -raw_idx;
        return static_cast<size_t>(raw_idx) % spec.total_cells;
    }

    int64_t read_by_sieve(int magnitude_k, int diagonal_d, int64_t signed_offset = 0) noexcept {
        int mem_tier = (magnitude_k <= 0) ? 0 : ((magnitude_k <= 1) ? 1 : ((magnitude_k == 2) ? 2 : 3));
        size_t idx = compute_spatial_index(magnitude_k, diagonal_d, signed_offset);
        access_counts_[mem_tier] += 1;
        accumulated_cycles_ += SPECS[mem_tier].cycle_latency;
        return buffers_[mem_tier][idx];
    }

    size_t write_by_sieve(int magnitude_k, int diagonal_d, int64_t signed_offset, int64_t value) noexcept {
        int mem_tier = (magnitude_k <= 0) ? 0 : ((magnitude_k <= 1) ? 1 : ((magnitude_k == 2) ? 2 : 3));
        size_t idx = compute_spatial_index(magnitude_k, diagonal_d, signed_offset);
        buffers_[mem_tier][idx] = value;
        access_counts_[mem_tier] += 1;
        accumulated_cycles_ += SPECS[mem_tier].cycle_latency;
        return idx;
    }

    // Ultra-fast O(1) Store via Invariant Diagonal Sieve (bijective: D(k, d) + signed_residue)
    void store_fast_by_sieve(int64_t key_value, int64_t payload) noexcept {
        int32_t k = 0;
        int32_t d = 0;
        int64_t signed_res = 0;
        InvariantDiagonalSieve::locate_primary_sieve(key_value, k, d, signed_res);
        write_by_sieve(k, d, signed_res, payload);
    }

    // Ultra-fast O(1) Load via Invariant Diagonal Sieve (bijective: D(k, d) + signed_residue)
    int64_t load_fast_by_sieve(int64_t key_value) noexcept {
        int32_t k = 0;
        int32_t d = 0;
        int64_t signed_res = 0;
        InvariantDiagonalSieve::locate_primary_sieve(key_value, k, d, signed_res);
        return read_by_sieve(k, d, signed_res);
    }

    // Full Trajectory Coordinate Store (for inspection & multi-layer auditing)
    GeometricCoordinate store_by_geometric_value(int64_t key_value, int64_t payload, int& out_layer, size_t& out_slot) noexcept {
        GeometricCoordinate coord = InvariantDiagonalSieve::decompose(key_value);
        out_layer = (coord.primary_layer_k <= 1) ? 1 : ((coord.primary_layer_k == 2) ? 2 : 3);
        out_slot = write_by_sieve(coord.primary_layer_k, coord.primary_diagonal_d, coord.signed_residue, payload);
        return coord;
    }

    // Full Trajectory Coordinate Load
    int64_t load_by_geometric_value(int64_t key_value, GeometricCoordinate& out_coord) noexcept {
        out_coord = InvariantDiagonalSieve::decompose(key_value);
        return read_by_sieve(out_coord.primary_layer_k, out_coord.primary_diagonal_d, out_coord.signed_residue);
    }

    // Branchless Grid Execution on any Layer (simulating 10,000 Vector Units or 1,000,000 PEs)
    void execute_branchless_grid_transform(
        int layer_k,
        int diagonal_d,
        size_t count,
        int64_t threshold,
        int64_t scale_if_gt,
        int64_t bias_if_le
    ) noexcept {
        int lk = layer_k & 3;
        size_t base = compute_spatial_index(lk, diagonal_d, 0);
        size_t limit = std::min(count, SPECS[lk].total_cells - base);
        int64_t* __restrict ptr = buffers_[lk].data() + base;

        // Pure branchless vectorized loop (auto-vectorizes to AVX2/AVX-512 on C++ compiler)
        for (size_t i = 0; i < limit; ++i) {
            int64_t val = ptr[i];
            uint64_t mask = branchless_mask_gt(val, threshold);
            int64_t path_true = val * scale_if_gt;
            int64_t path_false = val + bias_if_le;
            ptr[i] = branchless_select(mask, path_true, path_false);
        }

        access_counts_[lk] += 1;
        accumulated_cycles_ += SPECS[lk].cycle_latency;
    }

    std::vector<int64_t>& raw_layer(int layer_k) noexcept { return buffers_[layer_k & 3]; }
    const std::vector<int64_t>& raw_layer(int layer_k) const noexcept { return buffers_[layer_k & 3]; }
    uint64_t access_count(int layer_k) const noexcept { return access_counts_[layer_k & 3]; }
    uint64_t accumulated_cycles() const noexcept { return accumulated_cycles_; }

private:
    std::array<std::vector<int64_t>, 4> buffers_{};
    std::array<uint64_t, 4> access_counts_{};
    uint64_t accumulated_cycles_{0};
};

} // namespace gncx
