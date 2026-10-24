#pragma once
// ============================================================================
// Arquivo: gncx_mm_64x64.hpp
// Projeto: GNC-X (Geometric Navigational Core)
// Autor Original: Zaqueu Ribeiro da Costa (omega-Core-Dev)
// Funcao:
//   1. Tradutor Bijetivo O(1) Phi / Phi^-1 em 8x8 (64 frames) e 64x64 (4096 frames = 16 MiB)
//   2. Alocador Diagonal Geometrico 64x64 com Rotacao Multi-R e Domain Keys (4 bits)
//   3. Alocador Buddy Binario (Ordens 0..12) fiel ao modelo linear do Linux Kernel
// ============================================================================

#include <cstdint>
#include <array>
#include <bitset>
#include <vector>
#include <optional>
#include <algorithm>
#include <cassert>

namespace gncx::bench {

// ----------------------------------------------------------------------------
// 1. Endereco Geometrico Expandido 64x64 (4096 Frames = 16 MiB @ 4KB/frame)
// ----------------------------------------------------------------------------
struct GeoAddr64 {
    uint8_t k;          // Nivel de escala L0..L3 (0..3)
    uint8_t d;          // Posicao no crivo diagonal (0..63)
    uint8_t s;          // Indice da orbita diagonal / crivo paralelo (0..63)
    uint8_t R;          // Fator de rotacao impar coprimo com 64 (R in {1,3,5,...,63})
    uint8_t domain_id;  // Domain Protection Key (0..15)
    bool operator==(const GeoAddr64& o) const = default;
};

// ----------------------------------------------------------------------------
// 2. Unidade Phi(k, d, s, R) O(1) para Matriz N x N (N = 8 ou N = 64)
//    Propriedade: Para qualquer N = 2^m e R impar:
//      D(d, R) = (d * R) mod N
//      i = (d + k * R) mod N
//      j = (D(d, R) + k + s) mod N
//    Para s = 0, reduz-se exatamente a formula classica do GNC-X:
//      i = (d + k * R) mod N,  j = (d * R + k) mod N
//    Para s in [0, N-1] e d in [0, N-1], forma uma bijecao 100% sobre N x N!
// ----------------------------------------------------------------------------
template <uint32_t N>
struct PhiUnitNxN {
    static_assert((N & (N - 1)) == 0 && N >= 8 && N <= 64, "N deve ser potencia de 2 entre 8 e 64");
    static constexpr uint32_t MASK = N - 1;

    // Inverso multiplicativo modular R^-1 mod 2^m em O(1) via Newton-Raphson:
    // Para qualquer R impar, x_{k+1} = x_k * (2 - R * x_k) mod 2^m.
    // Duas iteracoes bastam para m <= 8 (ate N = 256).
    static constexpr uint8_t mod_inverse_pow2(uint8_t R) {
        uint8_t inv = R; // Aproximacao inicial exata mod 8 (pois R^2 == 1 mod 8)
        inv = static_cast<uint8_t>(inv * (2U - R * inv));
        inv = static_cast<uint8_t>(inv * (2U - R * inv));
        return static_cast<uint8_t>(inv & MASK);
    }

    // Traducao Direta O(1): GeoAddr -> (i, j)
    static constexpr std::pair<uint8_t, uint8_t> forward(uint8_t k, uint8_t d, uint8_t R, uint8_t s = 0) {
        const uint32_t k_m = k & 0x3U;
        const uint32_t d_m = d & MASK;
        const uint32_t s_m = s & MASK;
        const uint32_t R_m = (R | 1U) & MASK;

        const uint32_t D = (d_m * R_m) & MASK;
        const uint8_t  i = static_cast<uint8_t>((d_m + k_m * R_m) & MASK);
        const uint8_t  j = static_cast<uint8_t>((D + k_m + s_m) & MASK);
        return {i, j};
    }

    // Indice Fisico Linear O(1): 0 .. (N*N - 1)
    static constexpr uint32_t to_linear(uint8_t k, uint8_t d, uint8_t R, uint8_t s = 0) {
        auto [i, j] = forward(k, d, R, s);
        return (static_cast<uint32_t>(i) * N) + static_cast<uint32_t>(j);
    }

    // Traducao Inversa Exata O(1): (i, j, k, R) -> (d, s)
    // E para s = 0 (orbita principal), recupera d tanto por i quanto por j via R^-1!
    static constexpr GeoAddr64 inverse(uint8_t i, uint8_t j, uint8_t k, uint8_t R, uint8_t domain_id = 1) {
        const uint32_t k_m = k & 0x3U;
        const uint32_t R_m = (R | 1U) & MASK;
        const uint32_t d   = (static_cast<uint32_t>(i) + N - ((k_m * R_m) & MASK)) & MASK;
        const uint32_t D   = (d * R_m) & MASK;
        const uint32_t s   = (static_cast<uint32_t>(j) + (2U * N) - D - k_m) & MASK;
        return GeoAddr64{
            static_cast<uint8_t>(k_m),
            static_cast<uint8_t>(d),
            static_cast<uint8_t>(s),
            static_cast<uint8_t>(R_m),
            domain_id
        };
    }

    // Recuperacao de d exclusivamente pela coluna j quando s=0 (usando inverso modular R^-1)
    static constexpr uint8_t recover_d_from_j_s0(uint8_t j, uint8_t k, uint8_t R) {
        const uint32_t k_m   = k & 0x3U;
        const uint32_t R_m   = (R | 1U) & MASK;
        const uint32_t inv_R = mod_inverse_pow2(static_cast<uint8_t>(R_m));
        const uint32_t D_rec = (static_cast<uint32_t>(j) + N - (k_m & MASK)) & MASK;
        return static_cast<uint8_t>((D_rec * inv_R) & MASK);
    }
};

using Phi8x8   = PhiUnitNxN<8>;
using Phi64x64 = PhiUnitNxN<64>;

// ----------------------------------------------------------------------------
// 3. Simulador de Page Table Walk de 4 Niveis (Linux Radix Tree PGD->P4D->PUD->PMD->PTE)
//    Usado para comparar o custo de traducao virtual->fisico com TLB vs. Phi O(1)
// ----------------------------------------------------------------------------
class SimulatedRadixPageTable {
    static constexpr uint32_t NUM_PAGES = 4096;
    static constexpr uint32_t TLB_SIZE  = 64; // L1 dTLB tipica de 64 entradas

    // 4 niveis de tabela simulados em memoria
    alignas(64) uint32_t pgd_[16]{};
    alignas(64) uint32_t pud_[64]{};
    alignas(64) uint32_t pmd_[256]{};
    alignas(64) uint32_t pte_[NUM_PAGES]{};

    struct TlbEntry {
        uint32_t vpn{0xFFFFFFFFU};
        uint32_t pfn{0};
    };
    std::array<TlbEntry, TLB_SIZE> tlb_{};
    uint64_t tlb_hits_{0};
    uint64_t tlb_misses_{0};

public:
    SimulatedRadixPageTable() {
        for (uint32_t i = 0; i < 16; ++i)  pgd_[i] = i * 4;
        for (uint32_t i = 0; i < 64; ++i)  pud_[i] = i * 4;
        for (uint32_t i = 0; i < 256; ++i) pmd_[i] = i * 16;
        for (uint32_t i = 0; i < NUM_PAGES; ++i) pte_[i] = (i ^ 0x5A5U) & 0xFFFU;
    }

    uint64_t translate(uint32_t vpn) {
        vpn &= 0xFFFU;
        uint32_t slot = vpn & (TLB_SIZE - 1);
        if (tlb_[slot].vpn == vpn) {
            tlb_hits_++;
            return 0x100000000ULL + static_cast<uint64_t>(tlb_[slot].pfn) * 4096ULL;
        }
        tlb_misses_++;
        // Page table walk de 4 niveis
        uint32_t l1 = pgd_[(vpn >> 8) & 0xFU];
        uint32_t l2 = pud_[l1 + ((vpn >> 6) & 0x3U)];
        uint32_t l3 = pmd_[l2 + ((vpn >> 4) & 0x3U)];
        uint32_t pfn = pte_[l3 + (vpn & 0xFU)];
        tlb_[slot] = {vpn, pfn};
        return 0x100000000ULL + static_cast<uint64_t>(pfn) * 4096ULL;
    }

    uint64_t hits() const { return tlb_hits_; }
    uint64_t misses() const { return tlb_misses_; }
};

// ----------------------------------------------------------------------------
// 4. Alocador Buddy Binario Linear (4096 Frames = 16 MiB, Ordens 0..6)
//    Comportamento identico ao __alloc_pages() do Linux:
//    - Uma requisicao de 2^order paginas exige um bloco FISICAMENTE CONTIGUO
//      e alinhado em multiplo de 2^order.
// ----------------------------------------------------------------------------
class BuddyAllocator4096 {
public:
    static constexpr uint32_t TOTAL_FRAMES = 4096; // 16 MiB
    static constexpr uint32_t MAX_ORDER    = 6;    // Ate blocos de 64 paginas (256 KB)

    struct BlockHandle {
        uint32_t base_frame;
        uint8_t  order;
    };

private:
    std::bitset<TOTAL_FRAMES> phys_bitmap_;
    uint32_t used_frames_{0};
    uint64_t alloc_success_{0};
    uint64_t alloc_failures_frag_{0}; // Falhou com RAM livre suficiente por falta de contiguidade!

public:
    BuddyAllocator4096() { phys_bitmap_.reset(); }

    std::optional<BlockHandle> alloc(uint8_t order) {
        if (order > MAX_ORDER) return std::nullopt;
        const uint32_t block_size = 1U << order;
        if (TOTAL_FRAMES - used_frames_ < block_size) {
            return std::nullopt; // Realmente sem memoria total
        }

        // Procura bloco alinhado de tamanho 2^order totalmente livre
        for (uint32_t base = 0; base < TOTAL_FRAMES; base += block_size) {
            bool all_free = true;
            for (uint32_t offset = 0; offset < block_size; ++offset) {
                if (phys_bitmap_.test(base + offset)) {
                    all_free = false;
                    break;
                }
            }
            if (all_free) {
                for (uint32_t offset = 0; offset < block_size; ++offset) {
                    phys_bitmap_.set(base + offset);
                }
                used_frames_ += block_size;
                alloc_success_++;
                return BlockHandle{base, order};
            }
        }

        // Existem paginas livres suficientes no total, mas nenhuma contigua de ordem `order`!
        alloc_failures_frag_++;
        return std::nullopt;
    }

    void free(BlockHandle h) {
        const uint32_t block_size = 1U << h.order;
        for (uint32_t offset = 0; offset < block_size; ++offset) {
            uint32_t idx = h.base_frame + offset;
            if (idx < TOTAL_FRAMES && phys_bitmap_.test(idx)) {
                phys_bitmap_.reset(idx);
                used_frames_--;
            }
        }
    }

    // Fragmentacao externa classica para blocos contiguos: 1 - (maior_bloco_contiguo / total_livre)
    float external_fragmentation() const {
        uint32_t total_free = TOTAL_FRAMES - used_frames_;
        if (total_free == 0) return 0.0f;
        uint32_t max_run = 0, cur = 0;
        for (uint32_t i = 0; i < TOTAL_FRAMES; ++i) {
            if (!phys_bitmap_.test(i)) {
                cur++;
            } else {
                max_run = std::max(max_run, cur);
                cur = 0;
            }
        }
        max_run = std::max(max_run, cur);
        return 1.0f - static_cast<float>(max_run) / static_cast<float>(total_free);
    }

    uint32_t used_frames() const { return used_frames_; }
    uint64_t alloc_success() const { return alloc_success_; }
    uint64_t alloc_failures_frag() const { return alloc_failures_frag_; }
};

// ----------------------------------------------------------------------------
// 5. Alocador Diagonal Geometrico GNC-X 64x64 (4096 Frames = 16 MiB)
//    - Organiza os 4096 frames em 64 crivos diagonais D(k, d, s) de 64 frames cada
//    - Usa multiplas mascaras de 64 bits (SWAR 1 ciclo via __builtin_ctzll / popcount)
//    - Uma requisicao de 2^order paginas aloca N slots livres em um crivo diagonal
//      (ou em crivos adjacentes sob a mesma chave de dominio e rotacao R),
//      nao exigindo contiguidade fisica linear!
// ----------------------------------------------------------------------------
class GeometricAllocator4096 {
public:
    static constexpr uint32_t N            = 64;
    static constexpr uint32_t TOTAL_FRAMES = N * N; // 4096 frames = 16 MiB
    static constexpr std::array<uint8_t, 4> ROT = {1, 3, 5, 7};

    struct GeoBlockHandle {
        uint8_t  k;
        uint8_t  R;
        uint8_t  sieve_s;
        uint8_t  domain_id;
        uint64_t slot_mask; // Quais slots d in [0..63] foram alocados neste crivo
    };

private:
    // Cada crivo s in [0..63] sob a visao base possui um bitmap SWAR de 64 bits (1 = ocupado)
    std::array<uint64_t, N> sieve_bitmaps_{};
    std::bitset<TOTAL_FRAMES> phys_bitmap_;
    std::array<uint8_t, TOTAL_FRAMES> domain_tags_{};

    uint32_t used_frames_{0};
    uint64_t alloc_success_{0};
    uint64_t alloc_failures_frag_{0};
    uint32_t cursor_s_{0};

public:
    GeometricAllocator4096() {
        sieve_bitmaps_.fill(0ULL);
        phys_bitmap_.reset();
        domain_tags_.fill(0);
    }

    // Aloca 2^order paginas (1 a 64 paginas) em um crivo diagonal usando instrucoes SWAR 64-bit
    std::optional<GeoBlockHandle> alloc(uint8_t order, uint8_t k_hint = 2, uint8_t domain_id = 1) {
        const uint32_t needed = 1U << order;
        if (needed > N || (TOTAL_FRAMES - used_frames_ < needed)) {
            return std::nullopt;
        }

        const uint8_t k  = k_hint & 0x3U;
        const uint8_t Rk = ROT[k];

        // Varre os 64 crivos diagonais procurando um com pelo menos `needed` slots livres
        for (uint32_t step = 0; step < N; ++step) {
            const uint8_t s = static_cast<uint8_t>((cursor_s_ + step) & 63U);
            const uint64_t free_mask = ~sieve_bitmaps_[s];
            const uint32_t free_count = static_cast<uint32_t>(__builtin_popcountll(free_mask));

            if (free_count >= needed) {
                uint64_t chosen_slots = 0ULL;
                uint64_t rem = free_mask;
                for (uint32_t i = 0; i < needed; ++i) {
                    uint32_t d = static_cast<uint32_t>(__builtin_ctzll(rem));
                    chosen_slots |= (1ULL << d);
                    rem &= (rem - 1ULL); // limpa LSB

                    uint32_t phys_idx = Phi64x64::to_linear(k, static_cast<uint8_t>(d), Rk, s);
                    phys_bitmap_.set(phys_idx);
                    domain_tags_[phys_idx] = domain_id;
                }
                sieve_bitmaps_[s] |= chosen_slots;
                used_frames_      += needed;
                alloc_success_++;
                cursor_s_ = (s + 1U) & 63U;
                return GeoBlockHandle{k, Rk, s, domain_id, chosen_slots};
            }
        }

        alloc_failures_frag_++;
        return std::nullopt;
    }

    void free(const GeoBlockHandle& h) {
        uint64_t slots = h.slot_mask;
        uint32_t freed = static_cast<uint32_t>(__builtin_popcountll(slots));
        sieve_bitmaps_[h.sieve_s] &= ~slots;

        while (slots != 0ULL) {
            uint32_t d = static_cast<uint32_t>(__builtin_ctzll(slots));
            slots &= (slots - 1ULL);
            uint32_t phys_idx = Phi64x64::to_linear(h.k, static_cast<uint8_t>(d), h.R, h.sieve_s);
            phys_bitmap_.reset(phys_idx);
            domain_tags_[phys_idx] = 0;
        }
        used_frames_ -= freed;
    }

    // Fragmentacao Efetiva Geometrica:
    // No GNC-X, dois slots quaisquer dentro do mesmo crivo diagonal s sao enderecados
    // em O(1) via SWAR sem exigir adjacencia linear. Portanto, a maior corrida util
    // em um crivo e o popcount de bits livres do crivo, e entre crivos e a soma dos crivos!
    float geometric_fragmentation_order3() const {
        uint32_t total_free = TOTAL_FRAMES - used_frames_;
        if (total_free < 8) return 0.0f;
        uint32_t usable_in_diagonals = 0;
        for (uint32_t s = 0; s < N; ++s) {
            uint32_t fc = static_cast<uint32_t>(__builtin_popcountll(~sieve_bitmaps_[s]));
            if (fc >= 8) usable_in_diagonals += fc;
        }
        return 1.0f - (static_cast<float>(usable_in_diagonals) / static_cast<float>(total_free));
    }

    uint32_t used_frames() const { return used_frames_; }
    uint64_t alloc_success() const { return alloc_success_; }
    uint64_t alloc_failures_frag() const { return alloc_failures_frag_; }
};

} // namespace gncx::bench
