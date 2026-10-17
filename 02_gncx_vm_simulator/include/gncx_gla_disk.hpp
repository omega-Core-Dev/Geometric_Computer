#pragma once
/**
 * ============================================================================
 * GNC-X SILICON INCUBATOR — GLA DISK ENGINE (GEOMETRIC LAYER ADDRESSING)
 * ============================================================================
 * Author of Conception: Zaqueu Ribeiro da Costa
 *
 * Replaces traditional filesystems (NTFS/EXT4) with a single unified spatial
 * volume (`gnc_disk.img`) mapped directly into the process address space via
 * OS memory mapping (Win32 CreateFileMapping/MapViewOfFile or POSIX mmap).
 *
 * - O(1) Algebraic Coordinate Indexing:
 *   offset = (layer * stride_layer) + (diagonal_d * stride_diagonal) + (pos * sizeof(Cell))
 * - Predictive Vertical Diagonal Pagination:
 *   Accessing D(k, d) predictively prefetches D(k+1, d) = 10 * D(k, d) + d
 * ============================================================================
 */

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <cstring>
#include "gncx_sieve.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace gncx {

struct alignas(16) GLADiskCell {
    int64_t key_value;
    int64_t payload_value;
};

struct GLATelemetry {
    uint64_t mmap_reads{0};
    uint64_t mmap_writes{0};
    uint64_t predictive_prefetches{0};
    uint64_t prefetch_hits{0};
    std::vector<std::string> prefetch_log;
};

class GLADiskEngine {
public:
    static constexpr size_t SUB_LAYERS = 4;
    static constexpr size_t DIAGONAL_SECTORS = 10;
    static constexpr size_t POSITIONS_PER_SECTOR = 8192;
    static constexpr size_t CELL_BYTES = sizeof(GLADiskCell); // 16 bytes
    static constexpr size_t STRIDE_DIAGONAL = POSITIONS_PER_SECTOR * CELL_BYTES; // 131,072 bytes
    static constexpr size_t STRIDE_LAYER = DIAGONAL_SECTORS * STRIDE_DIAGONAL;   // 1,310,720 bytes
    static constexpr size_t TOTAL_DISK_BYTES = SUB_LAYERS * STRIDE_LAYER;        // 5,242,880 bytes (~5 MB)

    explicit GLADiskEngine(const std::string& disk_path) : disk_path_(disk_path) {
        for (auto& row : prefetched_table_) {
            row.fill(false);
        }
        open_or_create_mmap();
    }

    ~GLADiskEngine() {
        close_mmap();
    }

    // Algebraic O(1) byte offset calculation (zero filesystem tree traversal)
    static constexpr size_t compute_algebraic_offset(int layer_k, int diagonal_d, size_t position) noexcept {
        size_t l_idx = static_cast<size_t>(layer_k >= 0 ? layer_k : -layer_k) % SUB_LAYERS;
        size_t d_idx = static_cast<size_t>(diagonal_d >= 0 ? diagonal_d : -diagonal_d) % DIAGONAL_SECTORS;
        size_t p_idx = position % POSITIONS_PER_SECTOR;
        return (l_idx * STRIDE_LAYER) + (d_idx * STRIDE_DIAGONAL) + (p_idx * CELL_BYTES);
    }

    // Writes (key, payload) directly to its algebraic coordinate on `gnc_disk.img`
    size_t write_coordinate(int layer_k, int diagonal_d, size_t position, int64_t key, int64_t payload) noexcept {
        size_t offset = compute_algebraic_offset(layer_k, diagonal_d, position);
        if (mapped_view_) {
            auto* cell = reinterpret_cast<GLADiskCell*>(mapped_view_ + offset);
            cell->key_value = key;
            cell->payload_value = payload;
            telemetry_.mmap_writes += 1;
            trigger_predictive_pagination(layer_k, diagonal_d);
        }
        return offset;
    }

    // Reads (key, payload) in O(1) directly from `gnc_disk.img`
    GLADiskCell read_coordinate(int layer_k, int diagonal_d, size_t position) noexcept {
        size_t offset = compute_algebraic_offset(layer_k, diagonal_d, position);
        GLADiskCell result{0, 0};
        if (mapped_view_) {
            const auto* cell = reinterpret_cast<const GLADiskCell*>(mapped_view_ + offset);
            result = *cell;
            telemetry_.mmap_reads += 1;
            trigger_predictive_pagination(layer_k, diagonal_d);
        }
        return result;
    }

    // Decomposes `key_value` via the Invariant Diagonal Sieve and persists it in O(1)
    GeometricCoordinate persist_by_sieve(int64_t key_value, int64_t payload, size_t& out_byte_offset) noexcept {
        GeometricCoordinate coord = InvariantDiagonalSieve::decompose(key_value);
        size_t pos = static_cast<size_t>(coord.signed_residue >= 0 ? coord.signed_residue : -coord.signed_residue);
        out_byte_offset = write_coordinate(coord.primary_layer_k, coord.primary_diagonal_d, pos, key_value, payload);
        return coord;
    }

    // Fetches a persisted record in O(1) using only its Invariant Diagonal Sieve coordinate
    GLADiskCell fetch_by_sieve(int64_t key_value, GeometricCoordinate& out_coord, size_t& out_byte_offset) noexcept {
        out_coord = InvariantDiagonalSieve::decompose(key_value);
        size_t pos = static_cast<size_t>(out_coord.signed_residue >= 0 ? out_coord.signed_residue : -out_coord.signed_residue);
        out_byte_offset = compute_algebraic_offset(out_coord.primary_layer_k, out_coord.primary_diagonal_d, pos);
        return read_coordinate(out_coord.primary_layer_k, out_coord.primary_diagonal_d, pos);
    }

    const GLATelemetry& telemetry() const noexcept { return telemetry_; }
    uint8_t* raw_mapped_view() noexcept { return mapped_view_; }

private:
    void trigger_predictive_pagination(int layer_k, int diagonal_d) {
        size_t l_idx = static_cast<size_t>(layer_k >= 0 ? layer_k : -layer_k) % SUB_LAYERS;
        size_t d_idx = static_cast<size_t>(diagonal_d >= 0 ? diagonal_d : -diagonal_d) % DIAGONAL_SECTORS;

        if (prefetched_table_[l_idx][d_idx]) {
            telemetry_.prefetch_hits += 1;
        }

        size_t next_layer = (l_idx + 1) % SUB_LAYERS;
        if (!prefetched_table_[next_layer][d_idx] && mapped_view_) {
            // Touch target page in mmap to warm OS page cache predictively
            size_t prefetch_offset = compute_algebraic_offset(static_cast<int>(next_layer), static_cast<int>(d_idx), 0);
            volatile uint8_t warm_byte = mapped_view_[prefetch_offset];
            (void)warm_byte;

            prefetched_table_[next_layer][d_idx] = true;
            telemetry_.predictive_prefetches += 1;

            if (telemetry_.prefetch_log.size() < 12) {
                int64_t lifted = InvariantDiagonalSieve::lift_anchor(static_cast<int>(l_idx), static_cast<int>(d_idx));
                std::ostringstream oss;
                oss << "L" << l_idx << ":D(" << l_idx << "," << d_idx << ") -> Prefetch Vertical L"
                    << next_layer << ":D(" << next_layer << "," << d_idx << ")=" << lifted;
                telemetry_.prefetch_log.push_back(oss.str());
            }
        }
    }

    void open_or_create_mmap() {
#ifdef _WIN32
        file_handle_ = CreateFileA(
            disk_path_.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr,
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        if (file_handle_ == INVALID_HANDLE_VALUE) return;

        LARGE_INTEGER file_size;
        file_size.QuadPart = static_cast<LONGLONG>(TOTAL_DISK_BYTES);
        SetFilePointerEx(file_handle_, file_size, nullptr, FILE_BEGIN);
        SetEndOfFile(file_handle_);

        mapping_handle_ = CreateFileMappingA(
            file_handle_,
            nullptr,
            PAGE_READWRITE,
            0,
            static_cast<DWORD>(TOTAL_DISK_BYTES),
            nullptr
        );
        if (!mapping_handle_) return;

        mapped_view_ = static_cast<uint8_t*>(MapViewOfFile(
            mapping_handle_,
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            TOTAL_DISK_BYTES
        ));
#else
        fd_ = open(disk_path_.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd_ < 0) return;
        if (ftruncate(fd_, static_cast<off_t>(TOTAL_DISK_BYTES)) != 0) return;
        void* addr = mmap(nullptr, TOTAL_DISK_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (addr != MAP_FAILED) {
            mapped_view_ = static_cast<uint8_t*>(addr);
        }
#endif
    }

    void close_mmap() {
#ifdef _WIN32
        if (mapped_view_) {
            FlushViewOfFile(mapped_view_, 0);
            UnmapViewOfFile(mapped_view_);
            mapped_view_ = nullptr;
        }
        if (mapping_handle_) {
            CloseHandle(mapping_handle_);
            mapping_handle_ = nullptr;
        }
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (mapped_view_) {
            msync(mapped_view_, TOTAL_DISK_BYTES, MS_SYNC);
            munmap(mapped_view_, TOTAL_DISK_BYTES);
            mapped_view_ = nullptr;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
#endif
    }

    std::string disk_path_;
    uint8_t* mapped_view_{nullptr};
    GLATelemetry telemetry_{};
    std::array<std::array<bool, DIAGONAL_SECTORS>, SUB_LAYERS> prefetched_table_{};

#ifdef _WIN32
    HANDLE file_handle_{INVALID_HANDLE_VALUE};
    HANDLE mapping_handle_{nullptr};
#else
    int fd_{-1};
#endif
};

} // namespace gncx
