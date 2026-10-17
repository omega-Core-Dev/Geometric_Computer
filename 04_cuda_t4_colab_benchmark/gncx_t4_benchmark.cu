
// ============================================================================
// GNC-X CUDA C++ KERNELS FOR NVIDIA TESLA T4 (Turing sm_75)
// Compile on Colab T4: nvcc -O3 -arch=sm_75 gncx_t4_benchmark.cu -o gncx_t4_benchmark
// ============================================================================
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>

__constant__ int64_t d_DIAG_ANCHORS_L3[10] = {
    0LL, 1111LL, 2222LL, 3333LL, 4444LL, 5555LL, 6666LL, 7777LL, 8888LL, 9999LL
};

// Packed 32-bit token for 100% coalesced 128-byte Warp transactions on GDDR6:
// bits [1:0]   = phase_2bit (00=--, 01=-+, 10=+-, 11=++)
// bits [15:8]  = diag_d     (0..9)
// bits [31:16] = abs_delta  (uint16_t)
using PackedXToken32 = uint32_t;

// ----------------------------------------------------------------------------
// KERNEL 0: Dense Brute-Force Baseline on GPU (N * M multiplications)
// ----------------------------------------------------------------------------
__global__ void kernel_dense_baseline(
    const PackedXToken32* __restrict__ tokens,
    size_t n,
    const int64_t* __restrict__ weights_m_by_10,
    size_t out_channels,
    unsigned long long* __restrict__ out_channels_acc
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t raw = tokens[idx];
    uint32_t phase = raw & 0x3U;
    uint32_t d = (raw >> 8) & 0xFFU;
    int64_t abs_delta = static_cast<int64_t>(raw >> 16);

    int64_t b1 = static_cast<int64_t>((phase >> 1) & 1U);
    int64_t b0 = static_cast<int64_t>(phase & 1U);
    int64_t s1 = (b1 << 1) - 1;
    int64_t s0 = (b0 << 1) - 1;
    int64_t val = s1 * d_DIAG_ANCHORS_L3[d] + s0 * abs_delta;

    for (size_t m = 0; m < out_channels; ++m) {
        int64_t prod = weights_m_by_10[m * 10 + d] * val;
        atomicAdd(&out_channels_acc[m], static_cast<unsigned long long>(prod));
    }
}

// ----------------------------------------------------------------------------
// KERNEL 1 (BOTTLENECK DEMO): Naive Global Atomic X-Fold
// All threads hammer the same 10 global memory addresses `global_sector_mass[0..9]`!
// Demonstrates massive L2 atomic serialization across all 40 SMs of the T4.
// ----------------------------------------------------------------------------
__global__ void kernel_xfold_naive_global_atomic(
    const PackedXToken32* __restrict__ tokens,
    size_t n,
    unsigned long long* __restrict__ global_sector_mass
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t raw = tokens[idx];
    uint32_t phase = raw & 0x3U;
    uint32_t d = (raw >> 8) & 0xFFU;
    int64_t abs_delta = static_cast<int64_t>(raw >> 16);

    int64_t b1 = static_cast<int64_t>((phase >> 1) & 1U);
    int64_t b0 = static_cast<int64_t>(phase & 1U);
    int64_t s1 = (b1 << 1) - 1;
    int64_t s0 = (b0 << 1) - 1;
    int64_t val = s1 * d_DIAG_ANCHORS_L3[d] + s0 * abs_delta;

    // Bottleneck: 2,560+ concurrent threads colliding on 10 global addresses!
    atomicAdd(&global_sector_mass[d], static_cast<unsigned long long>(val));
}

// ----------------------------------------------------------------------------
// KERNEL 2: Shared Memory X-Fold (Unpadded 10-bin shared array)
// Reduces global traffic, but still suffers intra-block shared atomic contention.
// ----------------------------------------------------------------------------
__global__ void kernel_xfold_shared_atomic(
    const PackedXToken32* __restrict__ tokens,
    size_t n,
    unsigned long long* __restrict__ global_sector_mass
) {
    __shared__ unsigned long long s_mass[10];
    if (threadIdx.x < 10) s_mass[threadIdx.x] = 0ULL;
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t raw = tokens[idx];
        uint32_t phase = raw & 0x3U;
        uint32_t d = (raw >> 8) & 0xFFU;
        int64_t abs_delta = static_cast<int64_t>(raw >> 16);

        int64_t b1 = static_cast<int64_t>((phase >> 1) & 1U);
        int64_t b0 = static_cast<int64_t>(phase & 1U);
        int64_t s1 = (b1 << 1) - 1;
        int64_t s0 = (b0 << 1) - 1;
        int64_t val = s1 * d_DIAG_ANCHORS_L3[d] + s0 * abs_delta;

        atomicAdd(&s_mass[d], static_cast<unsigned long long>(val));
    }
    __syncthreads();

    if (threadIdx.x < 10) {
        atomicAdd(&global_sector_mass[threadIdx.x], s_mass[threadIdx.x]);
    }
}

// ----------------------------------------------------------------------------
// KERNEL 3 (GNC-X OPTIMAL T4 KERNEL):
// Coalesced uint32 Grid-Stride + Register-Privatized 10-Pivot Accumulators
// + Warp-Shuffle (__shfl_down_sync) + Bank-Conflict-Free Padded Shared Memory!
// ZERO atomic collisions inside the inner loop!
// ----------------------------------------------------------------------------
__global__ void kernel_xfold_warp_privatized_optimal(
    const PackedXToken32* __restrict__ tokens,
    size_t n,
    unsigned long long* __restrict__ global_sector_mass
) {
    // Each thread keeps its own 10 Invariant Diagonal accumulators in registers!
    int64_t reg_pivot_phase[10] = {0};
    int64_t reg_residue_sum[10] = {0};

    // Grid-stride loop: 100% coalesced 128-byte Warp loads, ZERO atomic ops in loop!
    size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        uint32_t raw = tokens[idx];
        uint32_t phase = raw & 0x3U;
        uint32_t d = (raw >> 8) & 0xFFU;
        int64_t abs_delta = static_cast<int64_t>(raw >> 16);

        int64_t b1 = static_cast<int64_t>((phase >> 1) & 1U);
        int64_t b0 = static_cast<int64_t>(phase & 1U);
        int64_t s1 = (b1 << 1) - 1;
        int64_t mask0 = -(b0 ^ 1LL);
        int64_t signed_delta = (abs_delta ^ mask0) - mask0;

        #pragma unroll
        for (int k = 0; k < 10; ++k) {
            int64_t match = -static_cast<int64_t>(d == static_cast<uint32_t>(k));
            reg_pivot_phase[k] += (s1 & match);
            reg_residue_sum[k] += (signed_delta & match);
        }
    }

    // Fold thread's 10 diagonals in registers + Warp Shuffle reduction (__shfl_down_sync)
    // Stride 16 in shared memory avoids 32-bank conflicts across warps!
    __shared__ int64_t warp_bins[8][16];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    #pragma unroll
    for (int d = 0; d < 10; ++d) {
        int64_t val = reg_pivot_phase[d] * d_DIAG_ANCHORS_L3[d] + reg_residue_sum[d];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            warp_bins[warp_id][d] = val;
        }
    }
    __syncthreads();

    // First 10 threads reduce across the 8 warps of the block and emit 1 atomicAdd per block
    if (threadIdx.x < 10) {
        int64_t block_sum = 0;
        int num_warps = blockDim.x >> 5;
        for (int w = 0; w < num_warps; ++w) {
            block_sum += warp_bins[w][threadIdx.x];
        }
        atomicAdd(&global_sector_mass[threadIdx.x], static_cast<unsigned long long>(block_sum));
    }
}
