#pragma once
// ============================================================================
// Arquivo: gncx_ddr3_arena.hpp
// Projeto: GNC-X (Geometric Navigational Core) — SDK & Low-Level Memory Manager
// Autor Original: Zaqueu Ribeiro da Costa (omega-Core-Dev)
// Funcao:
//   Gerenciador de Memoria Fisica/Virtual Consciente da Topologia DDR3/DDR4
//   e da Hierarquia de Cache L1d/L2/L3 (Intel Ivy Bridge / x86_64):
//     1. Arena de 2 MB / 16 MB alinhada via mmap + MADV_HUGEPAGE (destrava
//        o controle direto dos bits fisicos 20..0 pelo software).
//     2. Distribuicao Bijetiva O(1) via Phi(k, d, R, s) nos 8 Bancos da DDR3
//        (Bits 16..14) e nos 64 Conjuntos do Cache L1d (Bits 11..6).
//     3. GeoTensor2D<T>: Estrutura de Tensor com rotacao/transposicao
//        instantanea `geo_rot()` em 0 ciclos de copia (sem memcpy).
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <array>
#include <optional>
#include <utility>
#include <algorithm>
#include <sys/mman.h>
#include <unistd.h>

namespace gncx::sdk {

// ----------------------------------------------------------------------------
// 1. Mapeamento de Bits Fisicos DDR3 Dual-Channel + Cache L1d/L2 (2 MB THP)
//    Dentro de uma Huge Page de 2 MB (2^21 bytes):
//      - Bits  5..0  (64 B):   Burst 8n da DDR3 / Linha de Cache
//      - Bits 11..6  (4 KB):   Conjunto do Cache L1d (64 sets, 8-way)
//      - Bits 13..12 (16 KB):  Cor do Cache L2 (512 sets)
//      - Bits 16..14 (128 KB): Banco Fisico da DDR3 (Bank 0..7)
//      - Bits 20..17 (2 MB):   Linha Fisica da DDR3 (Row 0..15 na Huge Page)
// ----------------------------------------------------------------------------
struct DDR3GeoAddress {
    static constexpr std::array<uint8_t, 4> ROT = {1, 3, 5, 7};

    // Calcula o deslocamento fisico otimizado dentro de uma Arena de 16 MB (4096 paginas de 4KB)
    // combinando:
    //   - Selecao de Banco DDR3 (bits 16..14) via crivo diagonal j = (d * R + k + s) mod 8
    //   - Coloracao de Cache L1d/L2 (bits 11..6) via deslocamento coprimo anti-aliasing
    static constexpr size_t compute_arena_page_offset(uint8_t k, uint8_t d, uint8_t s, uint8_t R_custom = 0) {
        const uint32_t k_m = k & 0x3U;
        const uint32_t R   = (R_custom != 0) ? ((R_custom | 1U) & 0x3FU) : ROT[k_m];
        const uint32_t d_m = d & 0x3FU; // 0..63
        const uint32_t s_m = s & 0x3FU; // 0..63

        // Coordenadas bijetivas (i, j) na grade 64x64 (4096 frames de 4 KB = 16 MiB)
        const uint32_t D = (d_m * R) & 0x3FU;
        const uint32_t i = (d_m + k_m * R) & 0x3FU;
        const uint32_t j = (D + k_m + s_m) & 0x3FU;

        // O indice baixo de j (bits 2..0) vai diretamente para os bits 16..14 (DDR3 Bank 0..7)!
        // Como R e impar, quando d avanca 0,1,2..7, o banco DDR3 avanca por todos os 8 bancos sem repetir!
        const uint32_t ddr3_bank = j & 0x7U;          // 3 bits -> bits 16..14
        const uint32_t sub_col   = (j >> 3U) & 0x3U;  // 2 bits -> bits 13..12 (L2 Color)
        const uint32_t high_col  = (j >> 5U) & 0x1U;  // 1 bit  -> bit  17
        const uint32_t row_id    = i & 0x3FU;         // 6 bits -> bits 23..18 (DDR3 Row)

        const size_t page_aligned_offset =
            (static_cast<size_t>(row_id)    << 18U) |
            (static_cast<size_t>(high_col)  << 17U) |
            (static_cast<size_t>(ddr3_bank) << 14U) |
            (static_cast<size_t>(sub_col)   << 12U);

        return page_aligned_offset; // Multiplo exato de 4096 bytes (0 .. 16 MiB - 4 KiB)
    }

    // Coloracao Intra-Pagina de Linha de Cache (64 bytes):
    // Evita que multiplos tensores ou cabecas de atencao alinhem no Set 0 do Cache L1d!
    static constexpr size_t cache_line_color_offset(uint8_t d, uint8_t R) {
        const uint32_t color_set = (static_cast<uint32_t>(d) * (R | 1U)) & 0x3FU; // 0..63 sets L1d
        return static_cast<size_t>(color_set) * 64ULL; // 0 .. 4032 bytes
    }
};

// ----------------------------------------------------------------------------
// 2. Arena Fisica/Virtual GNC-X (16 MiB alinhada a Huge Pages de 2 MB)
// ----------------------------------------------------------------------------
class GNCXDDR3Arena {
public:
    static constexpr size_t HUGEPAGE_ALIGNMENT = 2ULL * 1024ULL * 1024ULL;  // 2 MiB
    static constexpr size_t ARENA_SIZE         = 16ULL * 1024ULL * 1024ULL; // 16 MiB (4096 frames)
    static constexpr size_t FRAME_SIZE         = 4096ULL;
    static constexpr uint32_t N                = 64;

    struct Allocation {
        void*    base_ptr;     // Ponteiro direto para uso em C/C++
        uint8_t  k;
        uint8_t  d;
        uint8_t  s;
        uint8_t  R;
        uint8_t  ddr3_bank;    // Qual dos 8 bancos DDR3 recebeu o bloco inicial
        uint8_t  l1_set_color; // Qual dos 64 sets do L1d recebeu o inicio
        uint64_t slot_mask;
    };

private:
    uint8_t* raw_mmap_ptr_{nullptr};
    size_t   raw_mmap_len_{0};
    uint8_t* aligned_base_{nullptr};

    // 64 crivos diagonais (cada uint64_t representa 64 frames de 4 KB = 256 KB)
    std::array<uint64_t, N> sieve_bitmaps_{};
    uint32_t used_frames_{0};
    uint32_t cursor_s_{0};

public:
    GNCXDDR3Arena() {
        sieve_bitmaps_.fill(0ULL);
        // Aloca com folga de 2 MB para garantir alinhamento estrito em fronteira de Huge Page (2 MB)
        raw_mmap_len_ = ARENA_SIZE + HUGEPAGE_ALIGNMENT;
        void* p = ::mmap(nullptr, raw_mmap_len_, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED) {
            raw_mmap_ptr_ = static_cast<uint8_t*>(p);
            uintptr_t addr = reinterpret_cast<uintptr_t>(raw_mmap_ptr_);
            uintptr_t aligned = (addr + HUGEPAGE_ALIGNMENT - 1ULL) & ~(HUGEPAGE_ALIGNMENT - 1ULL);
            aligned_base_ = reinterpret_cast<uint8_t*>(aligned);

            // Solicita ao Kernel Linux a fusao em Paginas Gigantes Fisicas de 2 MB (destrava bits 20..0)
            #ifdef MADV_HUGEPAGE
            ::madvise(aligned_base_, ARENA_SIZE, MADV_HUGEPAGE);
            #endif
        }
    }

    ~GNCXDDR3Arena() {
        if (raw_mmap_ptr_ && raw_mmap_ptr_ != MAP_FAILED) {
            ::munmap(raw_mmap_ptr_, raw_mmap_len_);
        }
    }

    GNCXDDR3Arena(const GNCXDDR3Arena&) = delete;
    GNCXDDR3Arena& operator=(const GNCXDDR3Arena&) = delete;

    bool valid() const { return aligned_base_ != nullptr; }
    uint8_t* base_address() const { return aligned_base_; }

    // Aloca um bloco geometrico colorido por Banco DDR3 e Conjunto L1d em O(1) SWAR
    std::optional<Allocation> alloc_colored(size_t bytes, uint8_t k_hint = 1, bool apply_l1_color = false) {
        if (!aligned_base_) return std::nullopt;
        const size_t eff_bytes = bytes + (apply_l1_color ? 4096ULL : 0ULL);
        const uint32_t frames_needed = static_cast<uint32_t>((eff_bytes + FRAME_SIZE - 1ULL) / FRAME_SIZE);
        if (frames_needed == 0 || frames_needed > N || (N * N - used_frames_ < frames_needed)) {
            return std::nullopt;
        }

        const uint8_t k = k_hint & 0x3U;
        const uint8_t R = DDR3GeoAddress::ROT[k];

        for (uint32_t step = 0; step < N; ++step) {
            const uint8_t s = static_cast<uint8_t>((cursor_s_ + step) & 63U);
            const uint64_t free_mask = ~sieve_bitmaps_[s];
            if (static_cast<uint32_t>(__builtin_popcountll(free_mask)) >= frames_needed) {
                uint32_t first_d = static_cast<uint32_t>(__builtin_ctzll(free_mask));
                uint64_t mask = 0ULL;
                uint64_t rem  = free_mask;
                for (uint32_t i = 0; i < frames_needed; ++i) {
                    uint32_t d_bit = static_cast<uint32_t>(__builtin_ctzll(rem));
                    mask |= (1ULL << d_bit);
                    rem &= (rem - 1ULL);
                }
                sieve_bitmaps_[s] |= mask;
                used_frames_      += frames_needed;
                cursor_s_          = (s + 1U) & 63U;

                size_t page_off = DDR3GeoAddress::compute_arena_page_offset(k, static_cast<uint8_t>(first_d), s, R);
                size_t l1_off   = apply_l1_color ? DDR3GeoAddress::cache_line_color_offset(static_cast<uint8_t>(first_d + s), R) : 0ULL;
                uint8_t* ptr    = aligned_base_ + page_off + l1_off;

                uintptr_t raw_addr = reinterpret_cast<uintptr_t>(ptr);
                uint8_t bank       = static_cast<uint8_t>((raw_addr >> 14U) & 0x7U);
                uint8_t l1_set     = static_cast<uint8_t>((raw_addr >> 6U)  & 0x3FU);

                return Allocation{
                    ptr, k, static_cast<uint8_t>(first_d), s, R, bank, l1_set, mask
                };
            }
        }
        return std::nullopt;
    }

    void free(const Allocation& a) {
        uint32_t count = static_cast<uint32_t>(__builtin_popcountll(a.slot_mask));
        sieve_bitmaps_[a.s] &= ~a.slot_mask;
        used_frames_ -= count;
    }

    uint32_t used_frames() const { return used_frames_; }
};

// ----------------------------------------------------------------------------
// 3. GeoTensor2D<T>: Tensor 2D com Mapeamento Diagonal e `geo_rot` Zero-Copy
//    Permite ler linhas, colunas, diagonais e transpor matrizes em 0 ciclos
//    de movimentacao de memoria!
// ----------------------------------------------------------------------------
template <typename T, size_t DIM>
class GeoTensor2D {
    static_assert((DIM & (DIM - 1)) == 0, "DIM deve ser potencia de 2");
    T*      data_;
    uint8_t rot_R_{1};      // Fator coprimo ativo
    bool    transposed_{false};

public:
    explicit GeoTensor2D(T* backing_buffer, uint8_t initial_R = 1)
        : data_(backing_buffer), rot_R_(initial_R | 1U), transposed_(false) {}

    // EQUIVALENTE EM SOFTWARE DA INSTRUCAO `geo.rot` EM HARDWARE:
    // Transpoe e rotaciona a visao do tensor em 0 ciclos de copia (sem memcpy)!
    inline void geo_rot(uint8_t new_R = 0, bool toggle_transpose = true) noexcept {
        if (new_R != 0) rot_R_ = new_R | 1U;
        if (toggle_transpose) transposed_ = !transposed_;
    }

    // Acesso direto O(1) com descorrelacao opcional de banco/cache
    inline T& at(size_t row, size_t col) noexcept {
        size_t r = transposed_ ? col : row;
        size_t c = transposed_ ? row : col;
        // Aplica permutacao bijetiva na coluna se rot_R_ != 1
        size_t phys_c = (c * rot_R_) & (DIM - 1U);
        return data_[r * DIM + phys_c];
    }

    inline const T& at(size_t row, size_t col) const noexcept {
        size_t r = transposed_ ? col : row;
        size_t c = transposed_ ? row : col;
        size_t phys_c = (c * rot_R_) & (DIM - 1U);
        return data_[r * DIM + phys_c];
    }

    uint8_t current_R() const noexcept { return rot_R_; }
    bool is_transposed() const noexcept { return transposed_; }
};

} // namespace gncx::sdk
