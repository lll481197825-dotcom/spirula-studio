#pragma once

#include <algorithm>
#include <cstddef>

#include "core/HostMemory.h"

namespace sfm {

struct MappingMemoryLimits {
    size_t working_bytes = 0;
    size_t graph_bytes = 0;
    size_t bundle_bytes = 0;
};

inline MappingMemoryLimits mappingMemoryLimits(size_t total, size_t available,
                                               int requested_mb = 0) {
    constexpr size_t mib = size_t{1024} * 1024;
    size_t budget = requested_mb > 0 ? size_t(requested_mb) * mib
                                    : total ? total - total / 5 : 2048 * mib;
    if (total) budget = std::min(budget, total - total / 5);
    if (available) {
        const size_t reserve = std::min<size_t>(4096 * mib, available / 4);
        budget = std::min(budget, available - reserve);
    }
    return {budget, std::min(512 * mib, budget / 8), budget / 2};
}

inline size_t availableBundleBudget(size_t configured) {
    const size_t available = spirula::availableRamBytes();
    return available ? std::min(configured, available - available / 4) : configured;
}

}  // namespace sfm
