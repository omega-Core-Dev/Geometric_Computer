/**
 * ============================================================================
 * GNC-X BUILDTOOL FOR PYTORCH (`gncx_core.dll` / `libgncx_core.so`)
 * ============================================================================
 * Author of Conception: Zaqueu Ribeiro da Costa
 *
 * Native C++20 Shared Library exposing the GNC-X "X-Folding" 4-Phase Tensor
 * Arithmetic Engine via an ABI-stable C interface (`extern "C"`) for zero-copy
 * integration with Python, NumPy, and PyTorch (`torch.nn.Module`).
 * ============================================================================
 */

#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#define GNCX_API extern "C" __declspec(dllexport)
#else
#define GNCX_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr int MAX_K = 17;

constexpr std::array<int64_t, MAX_K + 1> make_repunits() {
    std::array<int64_t, MAX_K + 1> r{};
    int64_t v = 1;
    for (int k = 0; k <= MAX_K; ++k) {
        r[k] = v;
        if (k < MAX_K) v = v * 10 + 1;
    }
    return r;
}

constexpr std::array<std::array<int64_t, 10>, MAX_K + 1> make_sieves() {
    std::array<std::array<int64_t, 10>, MAX_K + 1> s{};
    auto r = make_repunits();
    for (int k = 0; k <= MAX_K; ++k) {
        for (int d = 0; d < 10; ++d) {
            s[k][d] = static_cast<int64_t>(d) * r[k];
        }
    }
    return s;
}

constexpr auto REPUNITS = make_repunits();
constexpr auto DIAGONAL_SIEVES = make_sieves();

} // namespace

#pragma pack(push, 1)
struct GNCXCompactToken {
    uint8_t phase_2bit; // 00 (--), 01 (-+), 10 (+-), 11 (++)
    uint8_t diag_d;     // Invariant Diagonal Pivot d in {0..9}
    uint16_t abs_delta; // Residual |Delta|
};
#pragma pack(pop)

/**
 * Encodes an array of int64 values into 4-byte GNCXCompactTokens in Layer `layer_k`
 * and optionally writes the reconstructed exact values to `out_reconstructed`.
 */
GNCX_API void gncx_encode_4phase_tensor(
    const int64_t* __restrict input_vals,
    size_t num_elements,
    int32_t layer_k,
    GNCXCompactToken* __restrict out_tokens,
    int64_t* __restrict out_reconstructed
) {
    int lk = std::clamp(layer_k, 0, MAX_K);
    int64_t r_k = REPUNITS[lk];

    for (size_t i = 0; i < num_elements; ++i) {
        int64_t val = input_vals[i];
        int64_t abs_v = (val >= 0) ? val : -val;
        int64_t d_approx = (abs_v + (r_k >> 1)) / r_k;
        if (d_approx > 9) d_approx = 9;
        uint8_t d = static_cast<uint8_t>(d_approx);
        int64_t anchor = DIAGONAL_SIEVES[lk][d];

        int64_t signed_anchor = (val >= 0) ? anchor : -anchor;
        int64_t signed_res = val - signed_anchor;
        uint16_t abs_res = static_cast<uint16_t>(signed_res >= 0 ? signed_res : -signed_res);

        uint8_t b1 = (val >= 0) ? 1U : 0U;
        uint8_t b0 = (signed_res >= 0) ? 1U : 0U;
        uint8_t phase = static_cast<uint8_t>((b1 << 1) | b0);

        out_tokens[i] = GNCXCompactToken{phase, d, abs_res};

        if (out_reconstructed != nullptr) {
            int64_t s1 = (static_cast<int64_t>(b1) << 1) - 1;
            int64_t s0 = (static_cast<int64_t>(b0) << 1) - 1;
            out_reconstructed[i] = s1 * anchor + s0 * static_cast<int64_t>(abs_res);
        }
    }
}

/**
 * 1-Cycle Zero-Carry Negation (`-X`) of an entire X-Token Tensor via 2-bit XOR `0b11`!
 */
GNCX_API void gncx_negate_tensor_phase_xor(
    GNCXCompactToken* __restrict tokens,
    size_t num_elements
) {
    for (size_t i = 0; i < num_elements; ++i) {
        tokens[i].phase_2bit ^= 0x03U;
    }
}

/**
 * GNC-X "X-Folding" Multi-Channel Linear Layer Forward Pass:
 * Computes out_channels[m] = sum_{i=0..N-1} weights[m * 10 + d_i] * X_i
 *
 * Returns the number of opposite-phase pairs (+D vs -D) automatically canceled
 * at the central '0' origin of the X-Sieve!
 */
GNCX_API uint64_t gncx_xfold_linear_forward(
    const GNCXCompactToken* __restrict tokens,
    size_t num_elements,
    int32_t layer_k,
    const int64_t* __restrict weights_m_by_10,
    size_t num_channels,
    int64_t* __restrict out_channels
) {
    int lk = std::clamp(layer_k, 0, MAX_K);
    std::array<int64_t, 10> net_pivot_phase{};
    std::array<int64_t, 10> net_residue_sum{};
    std::array<uint64_t, 10> pos_pivots{};
    std::array<uint64_t, 10> neg_pivots{};

    // Step 1: Pure 4-Phase Folding across N elements (ZERO multiplications!)
    for (size_t i = 0; i < num_elements; ++i) {
        const GNCXCompactToken tok = tokens[i];
        uint8_t d = tok.diag_d;
        uint8_t b1 = (tok.phase_2bit >> 1) & 1U;
        uint8_t b0 = tok.phase_2bit & 1U;

        int64_t s1 = (static_cast<int64_t>(b1) << 1) - 1;
        int64_t mask0 = -static_cast<int64_t>(b0 ^ 1U);
        int64_t signed_delta = (static_cast<int64_t>(tok.abs_delta) ^ mask0) - mask0;

        net_pivot_phase[d] += s1;
        net_residue_sum[d] += signed_delta;

        pos_pivots[d] += b1;
        neg_pivots[d] += (b1 ^ 1U);
    }

    uint64_t canceled_pairs = 0;
    std::array<int64_t, 10> sector_mass{};
    for (size_t d = 0; d < 10; ++d) {
        canceled_pairs += std::min(pos_pivots[d], neg_pivots[d]);
        sector_mass[d] = net_pivot_phase[d] * DIAGONAL_SIEVES[lk][d] + net_residue_sum[d];
    }

    // Step 2: Project all M output channels using ONLY 10 multiplications per channel!
    for (size_t m = 0; m < num_channels; ++m) {
        int64_t acc = 0;
        const int64_t* __restrict w_row = weights_m_by_10 + (m * 10);
        for (size_t d = 0; d < 10; ++d) {
            acc += w_row[d] * sector_mass[d];
        }
        out_channels[m] = acc;
    }

    return canceled_pairs;
}

/**
 * Baseline Dense Linear Forward Pass (Brute-Force N * M multiplications) for verification.
 */
GNCX_API void gncx_dense_linear_baseline(
    const int64_t* __restrict raw_inputs,
    const GNCXCompactToken* __restrict tokens,
    size_t num_elements,
    const int64_t* __restrict weights_m_by_10,
    size_t num_channels,
    int64_t* __restrict out_channels
) {
    for (size_t m = 0; m < num_channels; ++m) {
        int64_t acc = 0;
        const int64_t* __restrict w_row = weights_m_by_10 + (m * 10);
        for (size_t i = 0; i < num_elements; ++i) {
            acc += w_row[tokens[i].diag_d] * raw_inputs[i];
        }
        out_channels[m] = acc;
    }
}

/**
 * ============================================================================
 * GNC-X "X-FOLDING" 1D CONVOLUTION (`XFoldConv1d`)
 * ============================================================================
 * Slides a 1D receptive window of size `kernel_size` across `length` tokens
 * (`out_length = length - kernel_size + 1`).
 * Uses O(1) Sliding X-Window recurrence: as the window advances by 1 step,
 * subtracts the exiting token and adds the entering token in the 10 Diagonal
 * Sieve buckets, then projects all `out_channels` filters!
 */
GNCX_API uint64_t gncx_xfold_conv1d_forward(
    const GNCXCompactToken* __restrict tokens,
    size_t length,
    size_t kernel_size,
    int32_t layer_k,
    const int64_t* __restrict weights_cout_by_10,
    size_t out_channels,
    int64_t* __restrict out_tensor // shape: [out_channels, out_length]
) {
    if (length < kernel_size || kernel_size == 0) return 0;
    const size_t out_len = length - kernel_size + 1;
    const int lk = std::clamp(layer_k, 0, MAX_K);

    std::array<int64_t, 10> net_pivot_phase{};
    std::array<int64_t, 10> net_residue_sum{};
    std::array<uint64_t, 10> pos_pivots{};
    std::array<uint64_t, 10> neg_pivots{};
    uint64_t total_canceled_pairs = 0;

    auto add_token = [&](const GNCXCompactToken& tok) {
        uint8_t d = tok.diag_d;
        uint8_t b1 = (tok.phase_2bit >> 1) & 1U;
        uint8_t b0 = tok.phase_2bit & 1U;
        int64_t s1 = (static_cast<int64_t>(b1) << 1) - 1;
        int64_t mask0 = -static_cast<int64_t>(b0 ^ 1U);
        int64_t signed_delta = (static_cast<int64_t>(tok.abs_delta) ^ mask0) - mask0;

        net_pivot_phase[d] += s1;
        net_residue_sum[d] += signed_delta;
        pos_pivots[d] += b1;
        neg_pivots[d] += (b1 ^ 1U);
    };

    auto remove_token = [&](const GNCXCompactToken& tok) {
        uint8_t d = tok.diag_d;
        uint8_t b1 = (tok.phase_2bit >> 1) & 1U;
        uint8_t b0 = tok.phase_2bit & 1U;
        int64_t s1 = (static_cast<int64_t>(b1) << 1) - 1;
        int64_t mask0 = -static_cast<int64_t>(b0 ^ 1U);
        int64_t signed_delta = (static_cast<int64_t>(tok.abs_delta) ^ mask0) - mask0;

        net_pivot_phase[d] -= s1;
        net_residue_sum[d] -= signed_delta;
        pos_pivots[d] -= b1;
        neg_pivots[d] -= (b1 ^ 1U);
    };

    auto decode_val = [&](const GNCXCompactToken& tok) -> int64_t {
        uint8_t d = tok.diag_d;
        uint8_t b1 = (tok.phase_2bit >> 1) & 1U;
        uint8_t b0 = tok.phase_2bit & 1U;
        int64_t s1 = (static_cast<int64_t>(b1) << 1) - 1;
        int64_t mask0 = -static_cast<int64_t>(b0 ^ 1U);
        int64_t signed_delta = (static_cast<int64_t>(tok.abs_delta) ^ mask0) - mask0;
        return s1 * DIAGONAL_SIEVES[lk][d] + signed_delta;
    };

    // Pre-decode exact 4-Phase values in a single sequential pass
    std::vector<int64_t> decoded_vals(length);
    for (size_t i = 0; i < length; ++i) {
        decoded_vals[i] = decode_val(tokens[i]);
    }

    // Prime the initial window [0 .. kernel_size - 1]
    for (size_t k = 0; k < kernel_size; ++k) {
        add_token(tokens[k]);
    }

    std::array<int64_t, 10> initial_sector_mass{};
    for (size_t d = 0; d < 10; ++d) {
        total_canceled_pairs += std::min(pos_pivots[d], neg_pivots[d]);
        initial_sector_mass[d] = net_pivot_phase[d] * DIAGONAL_SIEVES[lk][d] + net_residue_sum[d];
    }

    // Count opposite-phase cancellations across the 1D sliding window in one pass
    for (size_t x = 0; x + 1 < out_len; ++x) {
        remove_token(tokens[x]);
        add_token(tokens[x + kernel_size]);
        for (size_t d = 0; d < 10; ++d) {
            total_canceled_pairs += std::min(pos_pivots[d], neg_pivots[d]);
        }
    }

    // Sequential L1-cache-friendly O(1) sliding projection per output channel!
    for (size_t m = 0; m < out_channels; ++m) {
        const int64_t* __restrict w_row = weights_cout_by_10 + (m * 10);
        int64_t* __restrict out_row = out_tensor + (m * out_len);

        // Initial window at x = 0 folded into the 10 Invariant Diagonal Sieves
        int64_t acc = 0;
        for (size_t d = 0; d < 10; ++d) {
            acc += w_row[d] * initial_sector_mass[d];
        }
        out_row[0] = acc;

        for (size_t x = 0; x + 1 < out_len; ++x) {
            const size_t enter_idx = x + kernel_size;
            acc += (w_row[tokens[enter_idx].diag_d] * decoded_vals[enter_idx]
                  - w_row[tokens[x].diag_d] * decoded_vals[x]);
            out_row[x + 1] = acc;
        }
    }

    return total_canceled_pairs;
}

GNCX_API void gncx_dense_conv1d_baseline(
    const int64_t* __restrict raw_inputs,
    const GNCXCompactToken* __restrict tokens,
    size_t length,
    size_t kernel_size,
    const int64_t* __restrict weights_cout_by_10,
    size_t out_channels,
    int64_t* __restrict out_tensor
) {
    if (length < kernel_size || kernel_size == 0) return;
    const size_t out_len = length - kernel_size + 1;

    for (size_t m = 0; m < out_channels; ++m) {
        const int64_t* __restrict w_row = weights_cout_by_10 + (m * 10);
        int64_t* __restrict out_row = out_tensor + (m * out_len);
        for (size_t x = 0; x < out_len; ++x) {
            int64_t acc = 0;
            for (size_t k = 0; k < kernel_size; ++k) {
                size_t idx = x + k;
                acc += w_row[tokens[idx].diag_d] * raw_inputs[idx];
            }
            out_row[x] = acc;
        }
    }
}

/**
 * ============================================================================
 * GNC-X "X-FOLDING" 2D SPATIAL CONVOLUTION (`XFoldConv2d`)
 * ============================================================================
 * Performs a 2D spatial convolution over an [H x W] 4-Phase X-Token grid with
 * a [K_h x K_w] receptive field across `out_channels` filters.
 *
 * Uses Two-Stage Spatial X-Folding:
 *   Stage A: Vertical Column-Strip Folding of height K_h into the 10 Diagonal
 *            Sieve anchors.
 *   Stage B: Horizontal Sliding X-Window of width K_w across W_out, updating
 *            the 2D window's 10 Diagonal masses in O(1) per horizontal step!
 */
GNCX_API uint64_t gncx_xfold_conv2d_forward(
    const GNCXCompactToken* __restrict tokens,
    size_t height,
    size_t width,
    size_t kernel_h,
    size_t kernel_w,
    int32_t layer_k,
    const int64_t* __restrict weights_cout_by_10,
    size_t out_channels,
    int64_t* __restrict out_tensor // shape: [out_channels, out_h, out_w]
) {
    if (height < kernel_h || width < kernel_w || kernel_h == 0 || kernel_w == 0) return 0;
    const size_t out_h = height - kernel_h + 1;
    const size_t out_w = width - kernel_w + 1;
    const size_t out_spatial = out_h * out_w;
    const int lk = std::clamp(layer_k, 0, MAX_K);

    // Pre-decode exact signed contribution per token into a flat column-strip cache
    // col_mass[x][d], col_pos[x][d], col_neg[x][d] for the current row band [y .. y + K_h - 1]
    std::vector<std::array<int64_t, 10>> col_mass(width);
    std::vector<std::array<uint32_t, 10>> col_pos(width);
    std::vector<std::array<uint32_t, 10>> col_neg(width);

    auto accumulate_row_into_cols = [&](size_t r, int64_t dir) {
        const GNCXCompactToken* __restrict row_ptr = tokens + (r * width);
        for (size_t x = 0; x < width; ++x) {
            const GNCXCompactToken tok = row_ptr[x];
            uint8_t d = tok.diag_d;
            uint8_t b1 = (tok.phase_2bit >> 1) & 1U;
            uint8_t b0 = tok.phase_2bit & 1U;
            int64_t s1 = (static_cast<int64_t>(b1) << 1) - 1;
            int64_t mask0 = -static_cast<int64_t>(b0 ^ 1U);
            int64_t signed_delta = (static_cast<int64_t>(tok.abs_delta) ^ mask0) - mask0;
            int64_t val = s1 * DIAGONAL_SIEVES[lk][d] + signed_delta;

            col_mass[x][d] += dir * val;
            if (dir > 0) {
                col_pos[x][d] += b1;
                col_neg[x][d] += (b1 ^ 1U);
            } else {
                col_pos[x][d] -= b1;
                col_neg[x][d] -= (b1 ^ 1U);
            }
        }
    };

    // Prime initial vertical strip [0 .. kernel_h - 1]
    for (size_t r = 0; r < kernel_h; ++r) {
        accumulate_row_into_cols(r, +1);
    }

    uint64_t total_canceled_pairs = 0;
    std::vector<int64_t> col_proj(out_channels * width, 0);

    for (size_t y = 0; y < out_h; ++y) {
        // 1. Project each column's 10 folded Diagonal Sieve masses into the M channels
        for (size_t m = 0; m < out_channels; ++m) {
            const int64_t* __restrict w_row = weights_cout_by_10 + (m * 10);
            int64_t* __restrict cp_row = col_proj.data() + (m * width);
            for (size_t x = 0; x < width; ++x) {
                int64_t s = 0;
                for (size_t d = 0; d < 10; ++d) {
                    s += w_row[d] * col_mass[x][d];
                }
                cp_row[x] = s;
            }
        }

        // 2. Count opposite-phase cancellations across the 2D sliding window
        std::array<uint32_t, 10> win_pos{};
        std::array<uint32_t, 10> win_neg{};
        for (size_t c = 0; c < kernel_w; ++c) {
            for (size_t d = 0; d < 10; ++d) {
                win_pos[d] += col_pos[c][d];
                win_neg[d] += col_neg[c][d];
            }
        }
        for (size_t x = 0; x < out_w; ++x) {
            for (size_t d = 0; d < 10; ++d) {
                total_canceled_pairs += std::min(win_pos[d], win_neg[d]);
            }
            if (x + 1 < out_w) {
                size_t enter_c = x + kernel_w;
                for (size_t d = 0; d < 10; ++d) {
                    win_pos[d] += (col_pos[enter_c][d] - col_pos[x][d]);
                    win_neg[d] += (col_neg[enter_c][d] - col_neg[x][d]);
                }
            }
        }

        // 3. Horizontal O(1) Zero-Multiplication Sliding Window per output channel!
        for (size_t m = 0; m < out_channels; ++m) {
            const int64_t* __restrict cp_row = col_proj.data() + (m * width);
            int64_t* __restrict out_row = out_tensor + (m * out_spatial + y * out_w);

            int64_t acc = 0;
            for (size_t c = 0; c < kernel_w; ++c) {
                acc += cp_row[c];
            }
            out_row[0] = acc;

            for (size_t x = 0; x + 1 < out_w; ++x) {
                acc += (cp_row[x + kernel_w] - cp_row[x]);
                out_row[x + 1] = acc;
            }
        }

        // Vertical O(1) slide: subtract row y, add row (y + kernel_h)
        if (y + 1 < out_h) {
            accumulate_row_into_cols(y, -1);
            accumulate_row_into_cols(y + kernel_h, +1);
        }
    }

    return total_canceled_pairs;
}

GNCX_API void gncx_dense_conv2d_baseline(
    const int64_t* __restrict raw_inputs,
    const GNCXCompactToken* __restrict tokens,
    size_t height,
    size_t width,
    size_t kernel_h,
    size_t kernel_w,
    const int64_t* __restrict weights_cout_by_10,
    size_t out_channels,
    int64_t* __restrict out_tensor
) {
    if (height < kernel_h || width < kernel_w || kernel_h == 0 || kernel_w == 0) return;
    const size_t out_h = height - kernel_h + 1;
    const size_t out_w = width - kernel_w + 1;
    const size_t out_spatial = out_h * out_w;

    for (size_t m = 0; m < out_channels; ++m) {
        const int64_t* __restrict w_row = weights_cout_by_10 + (m * 10);
        int64_t* __restrict out_ch = out_tensor + (m * out_spatial);
        for (size_t y = 0; y < out_h; ++y) {
            for (size_t x = 0; x < out_w; ++x) {
                int64_t acc = 0;
                for (size_t ky = 0; ky < kernel_h; ++ky) {
                    const size_t row_offset = (y + ky) * width + x;
                    for (size_t kx = 0; kx < kernel_w; ++kx) {
                        const size_t idx = row_offset + kx;
                        acc += w_row[tokens[idx].diag_d] * raw_inputs[idx];
                    }
                }
                out_ch[y * out_w + x] = acc;
            }
        }
    }
}

/**
 * ============================================================================
 * GNC-X "X-FOLDING" BACKWARD PASS (`gncx_xfold_linear_backward`)
 * ============================================================================
 * Computes exact gradients for training (`loss.backward()`):
 *   1. Weight Gradients: dL/dW[m, d] = grad_output[m] * sector_mass[d]
 *      (Only 10 * M multiplications regardless of N!)
 *   2. Input Gradients:  dL/dX[i]    = sum_m grad_output[m] * W[m, d_i]
 *      (Computed via 10 diagonal gradient buckets + O(1) lookup per token!)
 */
GNCX_API void gncx_xfold_linear_backward(
    const GNCXCompactToken* __restrict tokens,
    size_t num_elements,
    int32_t layer_k,
    const int64_t* __restrict weights_m_by_10,
    const int64_t* __restrict grad_output_m,
    size_t num_channels,
    int64_t* __restrict out_grad_weights_m_by_10,
    int64_t* __restrict out_grad_input_n
) {
    int lk = std::clamp(layer_k, 0, MAX_K);
    std::array<int64_t, 10> net_pivot_phase{};
    std::array<int64_t, 10> net_residue_sum{};

    for (size_t i = 0; i < num_elements; ++i) {
        const GNCXCompactToken tok = tokens[i];
        uint8_t d = tok.diag_d;
        uint8_t b1 = (tok.phase_2bit >> 1) & 1U;
        uint8_t b0 = tok.phase_2bit & 1U;

        int64_t s1 = (static_cast<int64_t>(b1) << 1) - 1;
        int64_t mask0 = -static_cast<int64_t>(b0 ^ 1U);
        int64_t signed_delta = (static_cast<int64_t>(tok.abs_delta) ^ mask0) - mask0;

        net_pivot_phase[d] += s1;
        net_residue_sum[d] += signed_delta;
    }

    std::array<int64_t, 10> sector_mass{};
    for (size_t d = 0; d < 10; ++d) {
        sector_mass[d] = net_pivot_phase[d] * DIAGONAL_SIEVES[lk][d] + net_residue_sum[d];
    }

    std::array<int64_t, 10> diag_grad{};
    for (size_t m = 0; m < num_channels; ++m) {
        int64_t g_m = grad_output_m[m];
        const int64_t* __restrict w_row = weights_m_by_10 + (m * 10);
        int64_t* __restrict gw_row = out_grad_weights_m_by_10 + (m * 10);
        for (size_t d = 0; d < 10; ++d) {
            gw_row[d] = g_m * sector_mass[d];
            diag_grad[d] += g_m * w_row[d];
        }
    }

    if (out_grad_input_n != nullptr) {
        for (size_t i = 0; i < num_elements; ++i) {
            out_grad_input_n[i] = diag_grad[tokens[i].diag_d];
        }
    }
}

GNCX_API void gncx_dense_linear_backward(
    const int64_t* __restrict raw_inputs,
    const GNCXCompactToken* __restrict tokens,
    size_t num_elements,
    const int64_t* __restrict weights_m_by_10,
    const int64_t* __restrict grad_output_m,
    size_t num_channels,
    int64_t* __restrict out_grad_weights_m_by_10,
    int64_t* __restrict out_grad_input_n
) {
    for (size_t m = 0; m < num_channels * 10; ++m) {
        out_grad_weights_m_by_10[m] = 0;
    }
    for (size_t m = 0; m < num_channels; ++m) {
        int64_t g_m = grad_output_m[m];
        int64_t* __restrict gw_row = out_grad_weights_m_by_10 + (m * 10);
        for (size_t i = 0; i < num_elements; ++i) {
            gw_row[tokens[i].diag_d] += g_m * raw_inputs[i];
        }
    }
    if (out_grad_input_n != nullptr) {
        for (size_t i = 0; i < num_elements; ++i) {
            uint8_t d = tokens[i].diag_d;
            int64_t acc = 0;
            for (size_t m = 0; m < num_channels; ++m) {
                acc += grad_output_m[m] * weights_m_by_10[m * 10 + d];
            }
            out_grad_input_n[i] = acc;
        }
    }
}


