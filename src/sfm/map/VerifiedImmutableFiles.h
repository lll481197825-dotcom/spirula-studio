#pragma once
#include "core/HostMemory.h"
#include "core/Sha256.h"
#include "sfm/core/Cancel.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <cwctype>
#include <tuple>
#include <string>
#include <limits>
#include <stdexcept>

namespace sfm::regional {
class VerifiedImmutableFiles {
public:
    using Path = std::filesystem::path;
    struct Statistics { uint64_t bytesHashChecked = 0, leaseHits = 0, leaseMisses = 0; };
    explicit VerifiedImmutableFiles(size_t budget = 1u << 20) : budget_(budget) {}
    VerifiedImmutableFiles(const VerifiedImmutableFiles&) = delete;
    VerifiedImmutableFiles& operator=(const VerifiedImmutableFiles&) = delete;
    const Statistics& statistics() const { return stats_; }
    uint64_t residentBytes() const { return resident_; }
    size_t leasedFiles() const {
#ifdef _WIN32
        return files_.size();
#else
        return 0;
#endif
    }
    void clear() {
#ifdef _WIN32
        paths_.clear(); files_.clear(); directories_.clear();
#endif
        resident_ = 0;
    }
    void verify(const Path& input, const std::string& expected, uint64_t expectedBytes = UINT64_MAX) {
        if (expected.size() != 64) throw std::runtime_error("immutable verification requires a complete SHA256");
        const auto path = std::filesystem::absolute(input).lexically_normal();
#ifdef _WIN32
        holdDirectories(path.parent_path());
        auto handle = std::make_unique<Handle>(open(path, false)); const auto state = identity(handle->value, false);
        if (expectedBytes != UINT64_MAX && state.bytes != expectedBytes) throw std::runtime_error("immutable lease file size differs");
        const auto name = pathKey(path); const auto knownPath = paths_.find(name);
        if (knownPath != paths_.end() && !(knownPath->second.state == state)) throw std::runtime_error("immutable lease path identity changed");
        auto known = files_.find(state);
        if (known != files_.end()) {
            if (known->second.sha != expected) throw std::runtime_error("immutable lease expected SHA changed");
            if (knownPath == paths_.end()) addPath(name, state, std::move(handle));
            ++stats_.leaseHits; return;
        }
        guard(charge(path) + 4096 + 65536); ++stats_.leaseMisses;
        spirula::Sha256 sha; std::array<uint8_t, 65536> buffer; uint64_t total = 0;
        for (;;) {
            cancel::check(); DWORD received = 0;
            if (!ReadFile(handle->value, buffer.data(), DWORD(buffer.size()), &received, nullptr)) throw std::runtime_error("immutable lease SHA read failed");
            if (!received) break;
            sha.update(buffer.data(), received); total += received; stats_.bytesHashChecked += received;
        }
        if (total != state.bytes || !(identity(handle->value, false) == state) || sha.hex() != expected)
            throw std::runtime_error("immutable lease content SHA or physical identity differs");
        const auto cost = charge(path) + 512; guard(cost + 4608 + uint64_t(name.capacity()) * sizeof(wchar_t));
        addPath(name, state, std::move(handle));
        files_.emplace(state, File{expected}); resident_ += cost;
#else
        guard(65536);
        if (!std::filesystem::is_regular_file(path) || std::filesystem::is_symlink(path)) throw std::runtime_error("immutable verification requires a regular file");
        const auto size = std::filesystem::file_size(path); const auto time = std::filesystem::last_write_time(path);
        if (expectedBytes != UINT64_MAX && size != expectedBytes) throw std::runtime_error("immutable verification file size differs");
        ++stats_.leaseMisses; std::ifstream file(path, std::ios::binary); if (!file) throw std::runtime_error("cannot read immutable dependency");
        spirula::Sha256 sha; std::array<uint8_t, 65536> buffer;
        while (file) { cancel::check(); file.read(reinterpret_cast<char*>(buffer.data()), buffer.size()); const auto n = size_t(file.gcount()); sha.update(buffer.data(), n); stats_.bytesHashChecked += n; }
        if (!file.eof() || size != std::filesystem::file_size(path) || time != std::filesystem::last_write_time(path) || sha.hex() != expected)
            throw std::runtime_error("immutable dependency SHA changed");
#endif
    }
private:
    void guard(uint64_t extra) const { if (resident_ > budget_ || extra > budget_ - resident_) throw std::length_error("immutable verification leases exceed allowance"); }
    static uint64_t charge(const Path& path) { return 512 + uint64_t(path.native().capacity()) * sizeof(Path::value_type); }
    size_t budget_; uint64_t resident_ = 0; Statistics stats_;
#ifdef _WIN32
    struct Handle {
        HANDLE value = INVALID_HANDLE_VALUE;
        explicit Handle(HANDLE h) : value(h) {}
        ~Handle() { if (value != INVALID_HANDLE_VALUE) CloseHandle(value); }
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
    };
    struct Identity {
        DWORD volume = 0, high = 0, low = 0; uint64_t bytes = 0;
        bool operator<(const Identity& b) const { return std::tie(volume, high, low, bytes) < std::tie(b.volume, b.high, b.low, b.bytes); }
        bool operator==(const Identity& b) const { return volume == b.volume && high == b.high && low == b.low && bytes == b.bytes; }
    };
    struct File { std::string sha; };
    struct LeasedPath { Identity state; std::unique_ptr<Handle> handle; };
    static HANDLE open(const Path& path, bool directory) {
        const DWORD access = GENERIC_READ;
        const DWORD sharing = directory ? FILE_SHARE_READ | FILE_SHARE_WRITE : FILE_SHARE_READ;
        const DWORD flags = FILE_FLAG_OPEN_REPARSE_POINT | (directory ? FILE_FLAG_BACKUP_SEMANTICS : FILE_FLAG_SEQUENTIAL_SCAN);
        const auto h = CreateFileW(path.c_str(), access, sharing, nullptr, OPEN_EXISTING, flags, nullptr);
        if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("immutable verification cannot acquire deny-write/delete lease: " + path.u8string());
        return h;
    }
    static Identity identity(HANDLE handle, bool directory) {
        BY_HANDLE_FILE_INFORMATION info{};
        if (!GetFileInformationByHandle(handle, &info) || (!info.nFileIndexHigh && !info.nFileIndexLow) || (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) ||
            bool(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != directory)
            throw std::runtime_error("immutable verification physical file or directory identity invalid");
        return {info.dwVolumeSerialNumber, info.nFileIndexHigh, info.nFileIndexLow, (uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow};
    }
    static std::wstring pathKey(const Path& path) { auto key = path.native(); for (auto& c : key) c = wchar_t(std::towlower(c)); return key; }
    void addPath(const std::wstring& name, Identity state, std::unique_ptr<Handle> handle) {
        const uint64_t cost = 4608 + uint64_t(name.capacity()) * sizeof(wchar_t); guard(cost);
        paths_.emplace(name, LeasedPath{state, std::move(handle)}); resident_ += cost;
    }
    void holdDirectories(const Path& path) {
        if (path.empty()) return;
        const auto name = pathKey(path);
        const auto known = directories_.find(name);
        if (known != directories_.end()) { (void)identity(known->second->value, true); return; }
        const auto parent = path.parent_path(); if (parent != path) holdDirectories(parent);
        guard(charge(path) + 4096); auto handle = std::make_unique<Handle>(open(path, true)); (void)identity(handle->value, true);
        directories_.emplace(name, std::move(handle)); resident_ += charge(path) + 4096;
    }
    std::map<Identity, File> files_;
    std::map<std::wstring, LeasedPath> paths_;
    std::map<std::wstring, std::unique_ptr<Handle>> directories_;
#endif
};
}
