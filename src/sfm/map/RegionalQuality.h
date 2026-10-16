#pragma once

#include "sfm/map/RegionalStore.h"
#include "sfm/ba/CpuParallel.h"
#include <exception>
#include <limits>
#include <unordered_map>

namespace sfm::regional {
struct Quality {
    double cost = 0; uint64_t observations = 0;
    double finite_cost = std::numeric_limits<double>::quiet_NaN();
    uint64_t invalid_projections = 0;
    uint64_t retained_observations = 0, directional_observations = 0;
};
struct PhotoQuality { uint64_t observations = 0, inliers = 0; double error = 0; uint64_t invalid_projections = 0, retained_observations = 0; };
inline void accumulateQuality(Quality& total, const Quality& local) {
    total.cost += local.cost; total.observations += local.observations;
    total.finite_cost += local.finite_cost; total.invalid_projections += local.invalid_projections;
    total.retained_observations += local.retained_observations;
    total.directional_observations += local.directional_observations;
}
struct ValidationObservation { uint64_t point = 0; uint32_t image = 0, reserved = 0; Vec2 xy; };

inline void writeProjectionQuality(const fs::path& path, const Quality& q) {
    if (!q.observations || q.retained_observations > q.observations || q.directional_observations > q.observations-q.retained_observations ||
        q.invalid_projections > q.observations-q.retained_observations || !std::isfinite(q.cost) || q.cost < 0 ||
        !std::isfinite(q.finite_cost) || q.finite_cost < 0) throw std::runtime_error("invalid regional projection quality");
    model_store_detail::Writer out(path.string() + ".part");
    const uint64_t schema=q.retained_observations || q.directional_observations ? 2 : 1;
    out.scalar(schema); out.scalar(q.cost); out.scalar(q.observations);
    out.scalar(q.finite_cost); out.scalar(q.invalid_projections);
    if(schema==2) {out.scalar(q.retained_observations);out.scalar(q.directional_observations);} out.finish();
    replaceFile(path.string() + ".part", path);
}
inline Quality readProjectionQuality(const fs::path& path) {
    model_store_detail::Reader in(path);
    const auto schema=in.scalar<uint64_t>();
    if (schema != 1 && schema != 2) throw std::runtime_error("invalid regional projection ledger");
    Quality q; q.cost = in.scalar<double>(); q.observations = in.scalar<uint64_t>();
    q.finite_cost = in.scalar<double>(); q.invalid_projections = in.scalar<uint64_t>();
    if(schema==2) {q.retained_observations=in.scalar<uint64_t>();q.directional_observations=in.scalar<uint64_t>();} in.finish();
    if (!q.observations || q.retained_observations > q.observations || q.directional_observations > q.observations-q.retained_observations ||
        q.invalid_projections > q.observations-q.retained_observations || !std::isfinite(q.cost) || q.cost < 0 ||
        !std::isfinite(q.finite_cost) || q.finite_cost < 0) throw std::runtime_error("invalid regional projection ledger values");
    return q;
}

inline void writeValidation(const fs::path& path, const Reconstruction& rec, uint64_t model) {
    uint64_t count = 0;
    for (const auto& p : rec.points3D) count += p.second.track.size();
    model_store_detail::Writer out(path.string() + ".part");
    out.scalar<uint64_t>(0x314c415653535353ull); out.scalar(model); out.scalar(count);
    for (const auto& p : rec.points3D) {
        cancel::check();
        for (const auto& e : p.second.track) {
            const auto& im = rec.images.at(e.image_id);
            if (!im.registered || e.point2D_idx >= im.points2D.size()) throw std::runtime_error("invalid regional validation track");
            out.scalar(ValidationObservation{p.first, e.image_id, 0, im.points2D[e.point2D_idx]});
        }
    }
    out.finish(); replaceFile(path.string() + ".part", path);
}

inline Quality validationCost(const fs::path& path, const Reconstruction& local,
                              const Reconstruction& committed, uint64_t model, double gate, int threads = 0,
                              std::vector<PhotoQuality>* photos = nullptr) {
    if (!(gate > 0) || !std::isfinite(gate)) throw std::invalid_argument("invalid regional validation gate");
    model_store_detail::Reader in(path);
    if (in.scalar<uint64_t>() != 0x314c415653535353ull || in.scalar<uint64_t>() != model)
        throw std::runtime_error("regional validation identity changed");
    uint64_t count = in.scalar<uint64_t>();
    if (count > (UINT64_MAX - 24) / sizeof(ValidationObservation) ||
        fs::file_size(path) != 24 + count * sizeof(ValidationObservation))
        throw std::runtime_error("regional validation file size mismatch");
    struct View { const Image* image; const Camera* camera; };
    std::unordered_map<uint32_t, View> views;
    for (const auto& item : committed.images) {
        const auto camera = committed.cameras.find(item.second.camera_id);
        if (camera != committed.cameras.end()) views.emplace(item.first, View{&item.second, &camera->second});
    }
    const auto& prepared = views;
    constexpr size_t batch = 262144, grain = 4096;
    std::vector<ValidationObservation> rows(size_t(std::min<uint64_t>(count, batch)));
    auto& pool = bacpu::Pool::get();
    const int workers = threads > 0 ? std::min(threads, pool.size()) : pool.size();
    Quality result; result.finite_cost = 0;
    for (uint64_t offset = 0; offset < count;) {
        cancel::check(); const size_t size = size_t(std::min<uint64_t>(count - offset, batch));
        in.read(rows.data(), size);
        const int tasks = int((size + grain - 1) / grain);
        std::vector<double> costs(tasks);
        std::vector<double> finite_costs(tasks);
        std::vector<uint64_t> invalid_projections(tasks);
        std::vector<std::map<uint32_t, PhotoQuality>> photo_costs(photos ? tasks : 0);
        std::mutex failure_mutex; std::exception_ptr failure; std::atomic<bool> failed{false};
        pool.run(tasks, workers, [&](int task, int) {
            if (failed.load(std::memory_order_relaxed)) return;
            try {
                cancel::check(); uint64_t point = 0; bool has_point = false; Vec3 xyz{};
                const size_t end = std::min(size, (size_t(task) + 1) * grain);
                for (size_t o = size_t(task) * grain; o < end; ++o) {
                    const auto& row = rows[o]; const auto& view = prepared.at(row.image);
                    if (!view.image->registered || row.reserved || !(view.camera->pixel_scale > 0))
                        throw std::runtime_error("invalid committed validation camera");
                    if (!has_point || point != row.point) { xyz = local.points3D.at(row.point).xyz; point = row.point; has_point = true; }
                    if (!std::isfinite(xyz.norm())) throw std::runtime_error("nonfinite committed validation point");
                    bool valid = true;
                    double e = reprojErrorAt(*view.camera, view.image->pose, row.xy, xyz, &valid) / view.camera->pixel_scale;
                    if (!std::isfinite(e)) throw std::runtime_error("nonfinite committed validation residual");
                    const double cost = e <= gate ? e * e : gate * (2 * e - gate);
                    costs[task] += cost;
                    if (valid) finite_costs[task] += cost; else ++invalid_projections[task];
                    if (photos) {
                        auto& photo = photo_costs[task][row.image];
                        ++photo.observations; photo.inliers += e <= gate; photo.error += e;
                        photo.invalid_projections += !valid;
                    }
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failure) failure = std::current_exception();
                failed.store(true, std::memory_order_relaxed);
            }
        });
        if (failure) std::rethrow_exception(failure);
        for (double cost : costs) result.cost += cost;
        for (double cost : finite_costs) result.finite_cost += cost;
        for (uint64_t count : invalid_projections) result.invalid_projections += count;
        if (photos) for (const auto& task : photo_costs) for (const auto& kv : task) {
            auto& photo = photos->at(kv.first); photo.observations += kv.second.observations;
            photo.inliers += kv.second.inliers; photo.error += kv.second.error;
            photo.invalid_projections += kv.second.invalid_projections;
        }
        result.observations += size; offset += size;
    }
    in.finish();
    if (!std::isfinite(result.cost) || !std::isfinite(result.finite_cost)) throw std::runtime_error("regional validation cost overflow");
    return result;
}

inline bool photoQualityAccepted(const PhotoQuality& photo, double gate) {
    return !photo.invalid_projections && photo.observations >= 12 && photo.inliers * 2 >= photo.observations &&
        std::isfinite(photo.error) && photo.error <= gate * photo.observations;
}

inline uint32_t qualityStop(Quality candidate, Quality accepted, double baseline, double tolerance, bool last) {
    if (!candidate.observations || candidate.observations != accepted.observations ||
        candidate.retained_observations != accepted.retained_observations || candidate.directional_observations != accepted.directional_observations ||
        !std::isfinite(candidate.cost) || candidate.cost < 0 || !std::isfinite(baseline) || baseline < 0)
        throw std::runtime_error("regional validation observations or cost changed");
    if (candidate.cost > accepted.cost * (1 + 1e-8) + 1e-8) return 2;
    if (std::isfinite(candidate.finite_cost) && std::isfinite(accepted.finite_cost)) {
        if (candidate.invalid_projections > accepted.invalid_projections) return 2;
        if (candidate.finite_cost > accepted.finite_cost * (1 + 1e-8) + 1e-8) return 2;
    }
    if (!last && (candidate.invalid_projections || accepted.invalid_projections)) return 0;
    return last || std::abs(accepted.cost - candidate.cost) <= baseline * tolerance ? 1u : 0u;
}
} // namespace sfm::regional
