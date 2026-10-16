#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif

namespace sfm::storage {
namespace fs = std::filesystem;

inline uint64_t addBytes(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) throw std::length_error("disk budget overflow");
    return a + b;
}

inline uint64_t scaleBytes(uint64_t bytes, uint64_t factor) {
    if (factor && bytes > std::numeric_limits<uint64_t>::max() / factor) throw std::length_error("disk budget overflow");
    return bytes * factor;
}

inline void checkAvailable(const fs::path& path, uint64_t required, uint64_t available,
                           uint64_t reserve = uint64_t(1) << 30) {
    const uint64_t total = addBytes(required, reserve);
    if (available < total) throw std::runtime_error("insufficient disk space at " + path.u8string() +
        ": need " + std::to_string(total) + " bytes including reserve, available " + std::to_string(available) +
        "; accepted checkpoints retained; select a larger regional work directory");
}

inline void requireSpace(const fs::path& directory, uint64_t required, uint64_t reserve = uint64_t(1) << 30) {
    fs::path existing = fs::absolute(directory);
    while (!fs::exists(existing)) {
        const auto parent = existing.parent_path();
        if (parent == existing) throw std::runtime_error("cannot locate regional work volume");
        existing = parent;
    }
    if (!fs::is_directory(existing)) throw std::runtime_error("regional work path is not a directory: " + existing.u8string());
    checkAvailable(directory, required, fs::space(existing).available, reserve);
}

inline fs::path regionalRoot(const fs::path& default_root, const std::string& work_directory) {
    return work_directory.empty() ? default_root : fs::absolute(fs::u8path(work_directory)) / default_root.filename();
}

inline void replaceFile(const fs::path& tmp, const fs::path& path) {
#ifdef _WIN32
    if (!MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::system_error(int(GetLastError()), std::system_category(), "checkpoint replacement failed");
#else
    fs::rename(tmp, path);
#endif
}

// Model checkpoints are replaced atomically; links never share an in-place writer.
inline void retainFile(const fs::path& source, const fs::path& target) {
    if (fs::exists(target) && fs::equivalent(source, target)) return;
    fs::create_directories(target.parent_path());
    fs::path tmp = target; tmp += ".part";
    fs::remove(tmp);
    std::error_code error;
    fs::create_hard_link(source, tmp, error);
    if (error) {
        requireSpace(target.parent_path(), fs::file_size(source));
        fs::copy_file(source, tmp);
    }
    if (fs::file_size(source) != fs::file_size(tmp)) throw std::runtime_error("retained checkpoint size mismatch");
    replaceFile(tmp, target);
}
} // namespace sfm::storage
