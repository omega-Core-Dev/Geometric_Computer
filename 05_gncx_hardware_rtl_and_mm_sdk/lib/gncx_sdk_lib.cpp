// ============================================================================
// Arquivo: gncx_sdk_lib.cpp
// Projeto: GNC-X (Geometric Navigational Core) — Biblioteca Dinamica SDK (.so)
// Autor Original: Zaqueu Ribeiro da Costa (omega-Core-Dev)
// Funcao: Exporta a C ABI da Arena Geometrica DDR3 (`libgncx_mm.so`) para que
//         qualquer software em C, C++, Rust, Python (ctypes) ou LuaJIT (FFI)
//         ja saia com alocacao colorida por Banco DDR3 + Conjunto de Cache L1d!
// ============================================================================

#include "gncx_ddr3_arena.hpp"
#include <mutex>

namespace {
    std::mutex g_arena_mtx;
    gncx::sdk::GNCXDDR3Arena* g_default_arena = nullptr;

    gncx::sdk::GNCXDDR3Arena* get_arena() {
        if (!g_default_arena) {
            g_default_arena = new gncx::sdk::GNCXDDR3Arena();
        }
        return g_default_arena;
    }
}

extern "C" {

struct GNCXBlockDesc {
    void*    ptr;
    uint8_t  k;
    uint8_t  d;
    uint8_t  s;
    uint8_t  R;
    uint8_t  ddr3_bank;
    uint8_t  l1_set_color;
    uint64_t slot_mask;
};

// Aloca memoria geometricamente distribuida nos 8 Bancos da DDR3 e 64 Sets L1d
GNCXBlockDesc gncx_sdk_alloc(size_t bytes, uint8_t k_hint, int apply_l1_color) {
    std::lock_guard<std::mutex> lock(g_arena_mtx);
    auto* arena = get_arena();
    auto res = arena->alloc_colored(bytes, k_hint, apply_l1_color != 0);
    if (!res.has_value()) {
        return GNCXBlockDesc{nullptr, 0, 0, 0, 0, 0, 0, 0ULL};
    }
    return GNCXBlockDesc{
        res->base_ptr,
        res->k,
        res->d,
        res->s,
        res->R,
        res->ddr3_bank,
        res->l1_set_color,
        res->slot_mask
    };
}

// Libera bloco em O(1) SWAR
void gncx_sdk_free(GNCXBlockDesc desc) {
    if (!desc.ptr) return;
    std::lock_guard<std::mutex> lock(g_arena_mtx);
    auto* arena = get_arena();
    gncx::sdk::GNCXDDR3Arena::Allocation a{
        desc.ptr, desc.k, desc.d, desc.s, desc.R,
        desc.ddr3_bank, desc.l1_set_color, desc.slot_mask
    };
    arena->free(a);
}

// Retorna quantos frames de 4 KB estao em uso na Arena de 16 MiB
uint32_t gncx_sdk_used_frames() {
    std::lock_guard<std::mutex> lock(g_arena_mtx);
    return get_arena()->used_frames();
}

} // extern "C"
