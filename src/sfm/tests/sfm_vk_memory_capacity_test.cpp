#include "sfm/vk/VkContext.h"
#include "sfm/tests/TestMain.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static void checkCapacity(const VkDeviceMemoryCapacity& capacity) {
    require(capacity.budget_bytes <= capacity.heap_bytes, "budget exceeds physical heap");
    require(capacity.available_bytes <= capacity.budget_bytes, "available exceeds budget");
    require(capacity.reserve_bytes <= capacity.available_bytes, "reserve exceeds available");
    require(capacity.fallback != capacity.budget_known, "unknown capacity presented as known");
    require(capacity.available_bytes == (capacity.budget_bytes > capacity.usage_bytes ?
            capacity.budget_bytes - capacity.usage_bytes : 0), "available includes allocated usage");
}

static int run(int argc, char** argv) {
    constexpr uint64_t GiB = 1ull << 30;
    spirula::vkmemory::Budget budget{10 * GiB, 3 * GiB, 7 * GiB};
    auto known = VkDeviceMemoryCapacity::fromHeap(12 * GiB, GiB, budget, true);
    checkCapacity(known);
    require(known.budget_known && known.available_bytes == 7 * GiB,
            "driver budget or process usage ignored");
    auto fallback = VkDeviceMemoryCapacity::fromHeap(12 * GiB, GiB, {}, false);
    checkCapacity(fallback);
    require(fallback.fallback && fallback.available_bytes == 5 * GiB,
            "fallback treats total VRAM as available");
    auto exhausted = VkDeviceMemoryCapacity::fromHeap(12 * GiB, 7 * GiB, {}, false);
    checkCapacity(exhausted);
    require(!exhausted.available_bytes && !exhausted.reserve_bytes,
            "exhausted fallback underflowed");
    auto zeroBudget = VkDeviceMemoryCapacity::fromHeap(12 * GiB, GiB, {}, true);
    checkCapacity(zeroBudget);
    require(zeroBudget.fallback, "invalid zero driver budget is trusted");
    VkContext context;
    require(!context.deviceMemoryCapacity().available_bytes,
            "uninitialized context reports capacity");
    if (argc > 1 && std::strcmp(argv[1], "--device-query") == 0) {
        context.init({});
        auto capacity = context.deviceMemoryCapacity();
        checkCapacity(capacity);
        require(capacity.heap_bytes && capacity.heap_index < VK_MAX_MEMORY_HEAPS,
                "selected device has no local heap");
        std::printf("device=%s selector=%s heap=%u total=%llu budget=%llu usage=%llu available=%llu reserve=%llu known=%d fallback=%d\n",
                    context.deviceName().c_str(), context.selector().c_str(), capacity.heap_index,
                    (unsigned long long)capacity.heap_bytes, (unsigned long long)capacity.budget_bytes,
                    (unsigned long long)capacity.usage_bytes, (unsigned long long)capacity.available_bytes,
                    (unsigned long long)capacity.reserve_bytes, capacity.budget_known, capacity.fallback);
    }
    std::puts("capacity invariants passed");
    return 0;
}

int main(int argc, char** argv) { return sfmTestMain(argc, argv, run); }
