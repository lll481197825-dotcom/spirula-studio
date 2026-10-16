#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace sfm::resume {
namespace fs = std::filesystem;

// Preserve legacy names for ordinary directories; aliases identify their target.
inline fs::path featureIdentityRoot(const fs::path& directory) {
    const auto absolute = fs::absolute(directory).lexically_normal();
#ifdef _WIN32
    HANDLE handle = CreateFileW(absolute.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
        throw std::system_error(int(GetLastError()), std::system_category(), "cannot resolve feature directory");
    const DWORD size = GetFinalPathNameByHandleW(handle, nullptr, 0, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    std::wstring target(size, L'\0');
    const DWORD count = size ? GetFinalPathNameByHandleW(handle, target.data(), size,
        FILE_NAME_NORMALIZED | VOLUME_NAME_DOS) : 0;
    const DWORD error = GetLastError();
    CloseHandle(handle);
    if (!count || count >= size)
        throw std::system_error(int(error), std::system_category(), "cannot resolve feature directory");
    target.resize(count);
    if (target.compare(0, 8, L"\\\\?\\UNC\\") == 0) target = L"\\\\" + target.substr(8);
    else if (target.compare(0, 4, L"\\\\?\\") == 0) target.erase(0, 4);
    const auto physical = fs::path(target).lexically_normal();
    const auto a = absolute.native(), b = physical.native();
    if (CompareStringOrdinal(a.c_str(), int(a.size()), b.c_str(), int(b.size()), TRUE) == CSTR_EQUAL)
        return directory;
#else
    const auto physical = fs::canonical(directory);
    if (physical == absolute) return directory;
#endif
    return physical;
}

inline std::string featureDirectoryIdentity(const fs::path& directory) {
    const auto identity = featureIdentityRoot(directory);
    std::vector<std::string> rows;
    for (const auto& entry : fs::recursive_directory_iterator(directory))
        if (entry.is_regular_file()) {
            const auto name = identity / entry.path().lexically_relative(directory);
            rows.push_back(name.generic_string() + ":" + std::to_string(entry.file_size()) + ":" +
                std::to_string(int64_t(entry.last_write_time().time_since_epoch().count())));
        }
    std::sort(rows.begin(), rows.end());
    uint64_t hash = 1469598103934665603ull;
    for (const auto& row : rows)
        for (unsigned char c : row) { hash ^= c; hash *= 1099511628211ull; }
    char result[64];
    std::snprintf(result, sizeof result, "%016llx.%zu", (unsigned long long)hash, rows.size());
    return result;
}

inline void requireReusable(const std::string& mode, bool reusable, const char* stage) {
    if (mode == "require" && !reusable)
        throw std::runtime_error(std::string("required ") + stage +
            " reuse is not valid; cached data retained; no automatic recomputation");
}
} // namespace sfm::resume
