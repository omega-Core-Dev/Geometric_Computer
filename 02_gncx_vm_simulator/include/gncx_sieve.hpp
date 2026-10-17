#pragma once
/**
 * ============================================================================
 * GNC-X SILICON INCUBATOR — CORE ALGEBRAIC & INVARIANT DIAGONAL SIEVE ENGINE
 * ============================================================================
 * Author of Conception: Zaqueu Ribeiro da Costa
 * Architecture: Geometric Navigational Core (GNC-X)
 *
 * Implements in native C++ (zero overhead, 64-bit hardware aligned):
 * 1. Decimal Magnitude Layers (L_k, k = 0..17)
 * 2. Repunit Generators: R_{k+1} = (10^{k+1} - 1) / 9
 * 3. Invariant Diagonal Reference Sieves: D(k, d) = d * R_{k+1}, d in {0..9}
 * 4. Vertical Lifting & Projection Operators: D(k+1, d) = 10 * D(k, d) + d
 * 5. O(1) Geometric Coordinate Decomposition & Structural Primology Sieve
 * ============================================================================
 */

#include <cstdint>
#include <array>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

namespace gncx {

constexpr int MAX_LAYER_K = 17; // Fits safely in signed/unsigned 64-bit integer
constexpr int DIAGONAL_ANCHORS = 10; // d in {0..9}

// Compile-time precomputation of Powers of 10
constexpr std::array<int64_t, MAX_LAYER_K + 2> compute_powers_of_10() {
    std::array<int64_t, MAX_LAYER_K + 2> p{};
    int64_t val = 1;
    for (int i = 0; i <= MAX_LAYER_K + 1; ++i) {
        p[i] = val;
        if (i <= MAX_LAYER_K) val *= 10;
    }
    return p;
}

// Compile-time precomputation of Repunits R_{k+1} = 1, 11, 111, 1111, ...
constexpr std::array<int64_t, MAX_LAYER_K + 1> compute_repunits() {
    std::array<int64_t, MAX_LAYER_K + 1> r{};
    int64_t val = 1;
    for (int k = 0; k <= MAX_LAYER_K; ++k) {
        r[k] = val;
        if (k < MAX_LAYER_K) val = val * 10 + 1;
    }
    return r;
}

// Compile-time 2D Matrix of Invariant Diagonal Sieves D(k, d) = d * R_{k+1}
constexpr std::array<std::array<int64_t, DIAGONAL_ANCHORS>, MAX_LAYER_K + 1> compute_diagonal_sieves() {
    std::array<std::array<int64_t, DIAGONAL_ANCHORS>, MAX_LAYER_K + 1> table{};
    auto rep = compute_repunits();
    for (int k = 0; k <= MAX_LAYER_K; ++k) {
        for (int d = 0; d < DIAGONAL_ANCHORS; ++d) {
            table[k][d] = static_cast<int64_t>(d) * rep[k];
        }
    }
    return table;
}

inline constexpr auto POWERS_OF_10 = compute_powers_of_10();
inline constexpr auto REPUNITS = compute_repunits();
inline constexpr auto DIAGONAL_SIEVES = compute_diagonal_sieves();

/**
 * Single coordinate term in the Layer-by-Layer Diagonal Sieve Decomposition.
 */
struct SieveTerm {
    int32_t layer_k;         // Magnitude layer k (0=Units, 1=Tens, 2=Hundreds, 3=Thousands...)
    int32_t sign;            // +1 or -1 relative displacement direction
    int32_t digit_d;         // Invariant diagonal pivot index d in {0..9}
    int64_t diagonal_value;  // D(k, d) = d * R_{k+1}
};

/**
 * Canonical GNC-X Algebraic Coordinate for any integer/address X.
 * Replaces linear memory pointers, filesystem inodes, and B-Tree keys.
 */
struct alignas(64) GeometricCoordinate {
    int64_t original_value{0};
    int32_t primary_layer_k{0};
    int32_t primary_diagonal_d{0};
    int64_t primary_anchor{0};
    int64_t signed_residue{0};
    uint32_t num_terms{0};
    std::array<SieveTerm, 20> terms{};

    // Pure algebraic reconstruction: X = sum(sign_i * D(k_i, d_i))
    constexpr int64_t reconstruct() const noexcept {
        int64_t acc = 0;
        for (uint32_t i = 0; i < num_terms; ++i) {
            acc += static_cast<int64_t>(terms[i].sign) * terms[i].diagonal_value;
        }
        return acc;
    }

    std::string format_trajectory() const {
        if (num_terms == 0) return "L0[D(0,0)=0]";
        std::ostringstream oss;
        for (uint32_t i = 0; i < num_terms; ++i) {
            const auto& t = terms[i];
            if (i == 0) {
                if (t.sign < 0) oss << "-";
                oss << "L" << t.layer_k << "[D(" << t.layer_k << "," << t.digit_d << ")=" << t.diagonal_value << "]";
            } else {
                oss << (t.sign >= 0 ? " + " : " - ")
                    << "L" << t.layer_k << "[D(" << t.layer_k << "," << t.digit_d << ")=" << t.diagonal_value << "]";
            }
        }
        return oss.str();
    }
};

/**
 * Core Invariant Diagonal Sieve Engine (Compile-time & Hardware Branchless).
 */
class InvariantDiagonalSieve {
public:
    // Vertical Lifting Operator L_k -> L_{k+1}: D(k+1, d) = 10 * D(k, d) + d
    static constexpr int64_t lift_anchor(int k, int d) noexcept {
        return 10 * DIAGONAL_SIEVES[k][d] + d;
    }

    // Vertical Projection Operator L_k -> L_{k-1}: D(k-1, d) = D(k, d) / 10
    static constexpr int64_t project_anchor(int k, int d) noexcept {
        return (k <= 0) ? 0 : (DIAGONAL_SIEVES[k][d] / 10);
    }

    // Determines magnitude layer k in O(1)
    static constexpr int determine_layer(int64_t value) noexcept {
        int64_t v = value >= 0 ? value : -value;
        if (v < 10) return 0;
        int k = 0;
        while (k < MAX_LAYER_K && v >= POWERS_OF_10[k + 1]) {
            ++k;
        }
        return k;
    }

    // Finds the nearest invariant diagonal anchor D(k, d) in layer k in O(1)
    static constexpr void nearest_diagonal_in_layer(
        int64_t abs_value,
        int k,
        int32_t& out_d,
        int64_t& out_anchor,
        int64_t& out_residue
    ) noexcept {
        int64_t r_k = REPUNITS[k];
        int64_t d_approx = (abs_value + (r_k >> 1)) / r_k;
        if (d_approx < 0) d_approx = 0;
        if (d_approx > 9) d_approx = 9;
        out_d = static_cast<int32_t>(d_approx);
        out_anchor = DIAGONAL_SIEVES[k][out_d];
        out_residue = abs_value - out_anchor;
    }

    // Ultra-fast 1-cycle O(1) Primary Sieve Locator: maps any value to (k, d, signed_residue)
    static constexpr void locate_primary_sieve(
        int64_t value,
        int32_t& out_k,
        int32_t& out_d,
        int64_t& out_signed_residue
    ) noexcept {
        int64_t abs_val = (value >= 0) ? value : -value;
        int k = determine_layer(abs_val);
        int64_t r_k = REPUNITS[k];
        int64_t d_approx = (abs_val + (r_k >> 1)) / r_k;
        if (d_approx > 9) d_approx = 9;
        int32_t d = static_cast<int32_t>(d_approx);
        int64_t anchor = DIAGONAL_SIEVES[k][d];
        out_k = k;
        out_d = d;
        out_signed_residue = (value >= 0) ? (abs_val - anchor) : -(abs_val - anchor);
    }

    // Decomposes any 64-bit integer into its exact Layer-by-Layer Diagonal Sieve Coordinate
    static constexpr GeometricCoordinate decompose(int64_t value) noexcept {
        GeometricCoordinate coord{};
        coord.original_value = value;

        if (value == 0) {
            coord.terms[0] = SieveTerm{0, 1, 0, 0};
            coord.num_terms = 1;
            return coord;
        }

        int32_t base_sign = (value >= 0) ? 1 : -1;
        int64_t rem = (value >= 0) ? value : -value;
        int primary_k = determine_layer(rem);

        int32_t prim_d = 0;
        int64_t prim_anchor = 0;
        int64_t prim_res = 0;
        nearest_diagonal_in_layer(rem, primary_k, prim_d, prim_anchor, prim_res);

        coord.primary_layer_k = primary_k;
        coord.primary_diagonal_d = prim_d;
        coord.primary_anchor = base_sign * prim_anchor;
        coord.signed_residue = base_sign * prim_res;

        int32_t curr_sign = base_sign;
        int k = primary_k;

        while (rem > 0 && k >= 0 && coord.num_terms < 18) {
            int32_t d = 0;
            int64_t anchor = 0;
            int64_t residue = 0;
            nearest_diagonal_in_layer(rem, k, d, anchor, residue);
            if (d > 0) {
                coord.terms[coord.num_terms++] = SieveTerm{k, curr_sign, d, anchor};
                if (residue >= 0) {
                    rem = residue;
                } else {
                    rem = -residue;
                    curr_sign = -curr_sign;
                }
            }
            --k;
        }

        while (rem > 0 && coord.num_terms < 20) {
            int32_t d0 = (rem > 9) ? 9 : static_cast<int32_t>(rem);
            coord.terms[coord.num_terms++] = SieveTerm{0, curr_sign, d0, d0};
            rem -= d0;
        }

        return coord;
    }

    // Factorizes R_{k+1} to expose the structural prime barriers of Layer L_k
    static std::vector<int64_t> repunit_prime_factors(int k) {
        std::vector<int64_t> factors;
        int64_t temp = REPUNITS[k];
        for (int64_t d = 2; d * d <= temp; d += (d == 2 ? 1 : 2)) {
            while (temp % d == 0) {
                factors.push_back(d);
                temp /= d;
            }
        }
        if (temp > 1) factors.push_back(temp);
        return factors;
    }
};

} // namespace gncx
