#include "sfm/map/GpuRawQuality.h"
#include "sfm/vk/EmbeddedSpirv.h"
#include "sfm/vk/VkContext.h"
#include <limits>

namespace sfm::regional {
namespace {
struct View {
    double rotation[9]{}, translation[3]{}, intr[12]{}, pixelScale = 0;
    uint32_t model = 0, registered = 0;
};
struct DevicePartial {
    double cost = 0, finiteCost = 0;
    uint32_t observations = 0, invalid = 0, retained = 0, directions = 0, fallback = 0, pad = 0;
};
static_assert(sizeof(View) == 208 && sizeof(DevicePartial) == 40, "quality shader layout differs");
constexpr uint64_t contextReserve = (128ull + 64ull) << 20;
uint64_t checkedAdd(uint64_t a, uint64_t b) {
    if (a > UINT64_MAX - b) throw std::length_error("GPU quality byte count overflow"); return a + b;
}
uint64_t checkedMultiply(uint64_t a, uint64_t b) {
    if (a && b > UINT64_MAX / a) throw std::length_error("GPU quality byte count overflow"); return a * b;
}
bool finite(Vec3 value) { return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z); }
bool nativeProjectionCompatible(const Camera& c) {
    if (c.sx1 != 0 || c.sy1 != 0) return false;
    if (c.model == CamModel::FullOpenCV) return true;
    if (c.k3 != 0 || c.k4 != 0 || c.k5 != 0 || c.k6 != 0) return false;
    if (c.model == CamModel::OpenCV) return true;
    if (c.p1 != 0 || c.p2 != 0) return false;
    if (c.model == CamModel::Radial) return c.fx == c.fy;
    if (c.k1 != 0 || c.k2 != 0) return false;
    return c.model == CamModel::Pinhole || (c.model == CamModel::SimplePinhole && c.fx == c.fy);
}
}
struct GpuRawQuality::Impl {
    explicit Impl(GpuRawQualityOptions value) : options(std::move(value)) {}
    GpuRawQualityOptions options;
    GpuRawQualityStatus status = GpuRawQualityStatus::InvalidInput;
    std::string reason = "quality begin has not run";
    std::unique_ptr<VkContext> context;
    GpuBuffer rowBuffer, viewBuffer, partialBuffer;
    std::vector<View> views;
    std::vector<DevicePartial> partials;
    float gate = 0;
    GpuRawQualityStatus fail(GpuRawQualityStatus state, const std::string& message) {
        status = state; reason = message; return status;
    }
    void releaseBuffers() {
        if (!context) return;
        context->destroyBuffer(rowBuffer); context->destroyBuffer(viewBuffer); context->destroyBuffer(partialBuffer);
    }
    ~Impl() { releaseBuffers(); }
};
GpuRawQuality::GpuRawQuality(GpuRawQualityOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
GpuRawQuality::~GpuRawQuality() = default;
bool GpuRawQuality::supports(CamModel model) {
    return model == CamModel::SimplePinhole || model == CamModel::Pinhole || model == CamModel::Radial ||
           model == CamModel::OpenCV || model == CamModel::FullOpenCV;
}
const std::string& GpuRawQuality::reason() const { return impl_->reason; }
size_t GpuRawQuality::capacity() const { return impl_->options.max_rows; }
GpuRawQualityStatus GpuRawQuality::begin(const Reconstruction& poses, double gate) {
    auto& p = *impl_; p.status = GpuRawQualityStatus::InvalidInput; p.reason.clear();
    if (!(gate > 0) || !std::isfinite(gate)) return p.fail(GpuRawQualityStatus::InvalidInput,"quality gate is not positive finite");
    if (double(float(gate)) != gate) return p.fail(GpuRawQualityStatus::UnsupportedGate,"native Huber parameter cannot represent this double gate exactly");
    if (!p.options.max_rows || p.options.max_rows > (1u << 20) || !p.options.host_budget_bytes || !p.options.device_budget_bytes)
        return p.fail(GpuRawQualityStatus::BudgetExceeded,"quality batch capacity/budget invalid");
    uint64_t viewSlots = 0;
    for (const auto& item : poses.images) if (item.second.registered) viewSlots = std::max(viewSlots,uint64_t(item.first) + 1);
    if (!viewSlots || viewSlots > UINT32_MAX) return p.fail(GpuRawQualityStatus::InvalidInput,"quality requires bounded registered image IDs");
    try {
        const auto rowBytes = checkedMultiply(p.options.max_rows,sizeof(GpuQualityRow));
        const auto viewBytes = checkedMultiply(viewSlots,sizeof(View));
        const auto partialBytes = checkedMultiply((p.options.max_rows + 2047) / 2048,sizeof(DevicePartial));
        const auto buffers = checkedAdd(checkedAdd(rowBytes,viewBytes),partialBytes);
        const auto hostBytes = checkedAdd(contextReserve,checkedAdd(buffers,partialBytes));
        if (hostBytes > p.options.host_budget_bytes || buffers > p.options.device_budget_bytes || partialBytes > VkContext::stagingCapacity())
            return p.fail(GpuRawQualityStatus::BudgetExceeded,"quality owned-buffer/staging forecast exceeds allowance");
        p.views.assign(size_t(viewSlots),{});
        for (const auto& item : poses.images) if (item.second.registered) {
            const auto camera = poses.cameras.find(item.second.camera_id);
            if (camera == poses.cameras.end() || item.first != item.second.id)
                return p.fail(GpuRawQualityStatus::InvalidInput,"quality image/camera identity invalid");
            const auto& c = camera->second;
            if (!supports(c.model) || !nativeProjectionCompatible(c))
                return p.fail(GpuRawQualityStatus::UnsupportedCamera,"quality camera requires exact CPU projection");
            if (!(c.pixel_scale > 0) || !std::isfinite(c.pixel_scale) || !(c.fx > 0) || !(c.fy > 0) ||
                c.width <= 0 || c.height <= 0 || !finite(item.second.pose.t))
                return p.fail(GpuRawQualityStatus::InvalidInput,"quality camera/pose metadata invalid");
            auto& v = p.views[item.first]; v.model = uint32_t(camBaModel(c.model)); v.registered = 1; v.pixelScale = c.pixel_scale;
            packIntrinsics(c,v.intr); std::copy(item.second.pose.R.begin(),item.second.pose.R.end(),v.rotation);
            v.translation[0] = item.second.pose.t.x; v.translation[1] = item.second.pose.t.y; v.translation[2] = item.second.pose.t.z;
            for (double value : v.rotation) if (!std::isfinite(value)) return p.fail(GpuRawQualityStatus::InvalidInput,"quality rotation is nonfinite");
            for (double value : v.intr) if (!std::isfinite(value)) return p.fail(GpuRawQualityStatus::InvalidInput,"quality intrinsics are nonfinite");
        }
        size_t words = 0; const auto* code = findSpirv("quality_double_huber",&words);
        if (!code || !words) return p.fail(GpuRawQualityStatus::DeviceUnavailable,"F64 quality shader was not compiled");
        if (!p.context) {
            auto context = std::make_unique<VkContext>(); VkContextOptions options;
            options.needFloat64 = true; options.deviceIndex = p.options.device; options.selector = p.options.selector;
            context->init(options); p.context = std::move(context);
        }
        if (p.rowBuffer.size != rowBytes || p.viewBuffer.size != viewBytes || p.partialBuffer.size != partialBytes) {
            p.releaseBuffers(); const auto available = p.context->deviceMemoryCapacity();
            if (available.available_bytes < available.reserve_bytes || buffers > available.available_bytes - available.reserve_bytes)
                return p.fail(GpuRawQualityStatus::BudgetExceeded,"quality buffers exceed current selected GPU availability");
            if (std::max({rowBytes,viewBytes,partialBytes}) > p.context->maxBufferBytes())
                return p.fail(GpuRawQualityStatus::BudgetExceeded,"quality storage buffer exceeds selected GPU limit");
            p.rowBuffer = p.context->createBuffer(rowBytes); p.viewBuffer = p.context->createBuffer(viewBytes); p.partialBuffer = p.context->createBuffer(partialBytes);
        }
        p.context->createDescriptors({p.rowBuffer.buf,p.viewBuffer.buf,p.partialBuffer.buf});
        p.context->loadPipelines(code,words * sizeof(uint32_t),{"quality_raw"});
        p.context->upload(p.viewBuffer,p.views.data(),viewBytes);
        p.partials.resize((p.options.max_rows + 2047) / 2048); p.gate = float(gate);
        p.status = GpuRawQualityStatus::Ready; return p.status;
    } catch (const std::length_error& e) {
        p.releaseBuffers(); return p.fail(GpuRawQualityStatus::BudgetExceeded,e.what());
    } catch (const std::bad_alloc& e) {
        p.releaseBuffers(); return p.fail(GpuRawQualityStatus::BudgetExceeded,e.what());
    } catch (const std::exception& e) {
        p.releaseBuffers(); p.context.reset(); return p.fail(GpuRawQualityStatus::DeviceUnavailable,e.what());
    }
}
GpuRawQualityResult GpuRawQuality::evaluate(const std::vector<GpuQualityRow>& rows) {
    auto& p = *impl_; GpuRawQualityResult result; result.status = p.status; result.reason = p.reason;
    if (p.status != GpuRawQualityStatus::Ready) return result;
    if (rows.size() > p.options.max_rows) { result.status = GpuRawQualityStatus::BudgetExceeded; result.reason = "quality input exceeds bounded batch capacity"; return result; }
    for (const auto& row : rows) {
        if (row.kind > 2 || !finite(row.value) || !std::isfinite(row.xy.x) || !std::isfinite(row.xy.y) ||
            row.image >= p.views.size() || !p.views[row.image].registered ||
            (row.kind == 1 && std::abs(row.value.dot(row.value) - 1) > 1e-10) ||
            (row.kind == 2 && (row.value.x != 0 || row.value.y != 0 || row.value.z != 0))) {
            result.status = GpuRawQualityStatus::InvalidInput; result.reason = "quality row measurement/type/pose invalid"; return result;
        }
    }
    if (rows.empty()) return result;
    try {
        const size_t count = (rows.size() + 2047) / 2048;
        p.context->upload(p.rowBuffer,rows.data(),rows.size() * sizeof(GpuQualityRow));
        auto command = p.context->begin(); Push push; push.u0 = uint32_t(rows.size()); push.u1 = uint32_t(p.views.size()); push.f0 = p.gate;
        p.context->dispatch(command,"quality_raw",uint32_t(count),push); p.context->barrier(command);
        p.context->recordDownload(command,p.partialBuffer,count * sizeof(DevicePartial)); p.context->submit(command);
        std::memcpy(p.partials.data(),p.context->stagingDownloadPtr(),count * sizeof(DevicePartial));
        result.partials.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto& q = p.partials[i];
            if (q.observations != std::min<size_t>(2048,rows.size() - i * 2048) || q.retained > q.observations ||
                q.directions > q.observations - q.retained || q.invalid > q.observations - q.retained || q.fallback > q.observations ||
                !std::isfinite(q.cost) || q.cost < 0 || !std::isfinite(q.finiteCost) || q.finiteCost < 0 || q.finiteCost > q.cost) {
                result.partials.clear(); result.status = GpuRawQualityStatus::RequiresCpuFallback; result.reason = "quality GPU partial coverage/finite sum differs"; return result;
            }
            result.partials.push_back({q.cost,q.finiteCost,q.observations,q.invalid,q.retained,q.directions,q.fallback});
            if (q.fallback) { result.status = GpuRawQualityStatus::RequiresCpuFallback; result.reason = "quality invalid/near-domain residual requires complete CPU batch"; }
        }
    } catch (const std::exception& e) {
        result.partials.clear(); result.status = GpuRawQualityStatus::DeviceUnavailable; result.reason = e.what();
        p.fail(result.status,result.reason); p.releaseBuffers(); p.context.reset();
    }
    return result;
}
}
