#pragma once
/**
 * ============================================================================
 * GNC-X 4-PHASE GEOMETRIC CONVOLUTION ENGINE (`XFoldConv1d` & `XFoldConv2d`)
 * ============================================================================
 * Author of Conception: Zaqueu Ribeiro da Costa
 *
 * Implements 1D and 2D Multi-Channel Convolutions over the 4-Phase X-Sieve:
 *   - Phase 11 (++) : +D(k,d) + |Delta|
 *   - Phase 10 (+-) : +D(k,d) - |Delta|
 *   - Phase 01 (-+) : -D(k,d) + |Delta|
 *   - Phase 00 (--) : -D(k,d) - |Delta|
 *
 * Key Algorithmic Breakthroughs for Entry-Level CPUs:
 *   1. Receptive-Field X-Folding: Inside any K_h x K_w spatial window, tokens
 *      are folded into the 10 Invariant Diagonal Anchors D(k, 0..9) using ONLY
 *      2-bit phase additions/subtractions.
 *   2. Sliding X-Window Recurrence O(1): As the kernel slides horizontally from
 *      x to (x + 1), the 10 diagonal sector masses are updated by subtracting
 *      the exiting column strip and adding the entering column strip—eliminating
 *      redundant inner-kernel loops!
 *   3. Local Phase Annihilation: Edges and symmetric zero-crossings inside the
 *      receptive field cancel opposite pivots (+D and -D) at the central '0'.
 * ============================================================================
 */

#include "gncx_x_math.hpp"
#include <vector>
#include <array>
#include <algorithm>

namespace gncx::conv {

struct Conv2DBenchmarkResult {
    double dense_time_ms;
    double xfold_time_ms;
    double speedup;
    uint64_t total_macs_equivalent;
    uint64_t opposite_phase_pairs_canceled;
    bool bit_exact_match;
};

} // namespace gncx::conv
