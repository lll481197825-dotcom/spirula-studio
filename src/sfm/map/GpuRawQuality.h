#pragma once
#include "sfm/core/Model.h"
#include <memory>

namespace sfm::regional {
struct GpuQualityRow {
    Vec3 value;
    Vec2 xy;
    uint32_t image = 0, kind = 0;
};
static_assert(sizeof(GpuQualityRow) == 48, "quality shader row layout differs");

struct GpuQualityPartial {
    double cost = 0, finite_cost = 0;
    uint64_t observations = 0, invalid_projections = 0, retained_observations = 0;
    uint64_t directional_observations = 0, fallback_rows = 0;
};
enum class GpuRawQualityStatus {
    Ready, UnsupportedCamera, UnsupportedGate, InvalidInput, BudgetExceeded,
    DeviceUnavailable, RequiresCpuFallback
};
struct GpuRawQualityOptions {
    size_t max_rows = 65536;
    // Host allowance includes staging/driver estimate; device allowance is payload buffers.
    uint64_t host_budget_bytes = 256ull << 20, device_budget_bytes = 256ull << 20;
    int device = -1;
    std::string selector;
};
struct GpuRawQualityResult {
    GpuRawQualityStatus status = GpuRawQualityStatus::InvalidInput;
    std::vector<GpuQualityPartial> partials;
    std::string reason;
    bool ready() const { return status == GpuRawQualityStatus::Ready; }
};

// Each instance owns an exclusive quality-only Vulkan context; no BA module is reused.
// Non-ready partials are diagnostics only. Recompute that complete batch on the CPU.
class GpuRawQuality {
public:
    explicit GpuRawQuality(GpuRawQualityOptions options = {});
    ~GpuRawQuality();
    GpuRawQuality(const GpuRawQuality&) = delete;
    GpuRawQuality& operator=(const GpuRawQuality&) = delete;
    static bool supports(CamModel model);
    GpuRawQualityStatus begin(const Reconstruction& poses, double gate);
    GpuRawQualityResult evaluate(const std::vector<GpuQualityRow>& rows);
    const std::string& reason() const;
    size_t capacity() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
