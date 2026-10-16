#pragma once

#include "sfm/Pipeline.h"
#include "sfm/map/Orient.h"
#include "sfm/map/AttitudeGauge.h"

namespace sfm::regional {
using sfm::slog::Tag;

struct MeasuredAlignment {
    bool ok = false;
    Sim3 transform;
    int positions = 0, inliers = 0;
    double rms = 0, yaw_deg = 0, up_spread_deg = 0, heading_spread_deg = 0;
};

inline AttitudeRef regionalRecordedAttitude(const Reconstruction& rec, const std::string& images,
                                            bool pixels_turned) {
    if (images.empty()) return {};
    std::vector<Reconstruction> names(1);
    for (const auto& kv : rec.images) if (kv.second.registered) {
        Image image; image.id = kv.first; image.name = kv.second.name; image.registered = true;
        names[0].images.emplace(kv.first, std::move(image));
    }
    resolveImageNames(names, images);
    return attitudeRefFromImages(names[0], images, pixels_turned);
}

inline MeasuredAlignment measuredRegionalAlignment(const Reconstruction& rec, PriorSource& gps,
                                                   const AttitudeRef& attitude) {
    MeasuredAlignment out;
    const auto orientation = fitAttitudeGauge(rec, attitude, true);
    if (!orientation.ok || !orientation.north || size_t(orientation.up.votes) * 5 < rec.numRegistered() * 4)
        return out;
    MetricRef reference;
    for (const auto& kv : rec.images) {
        Vec3 position;
        if (kv.second.registered && gps.position(kv.first, position)) {
            reference.centres.push_back(transformPoint(orientation.T, cameraCenter(kv.second.pose)));
            reference.targets.push_back(position); reference.image_ids.push_back(kv.first);
        }
    }
    auto fit = fitMetricGauge(reference, 5., MetricAxes::Horizontal, .01);
    out.positions = fit.n; out.inliers = fit.inliers;
    out.yaw_deg = std::atan2(fit.T.R[3], fit.T.R[0]) * 180. / M_PI;
    out.up_spread_deg = orientation.up.spread_deg; out.heading_spread_deg = orientation.heading.spread_deg;
    if (!fit.ok || size_t(fit.n) * 5 < rec.numRegistered() * 4 || fit.inliers * 5 < fit.n * 4 ||
        std::abs(out.yaw_deg) > 10. || fit.scale_unc > 1. || fit.rot_unc_deg > 1.) return out;
    int spatial_inliers = 0; double squared = 0;
    for (size_t i = 0; i < reference.centres.size(); ++i) {
        double error = (transformPoint(fit.T, reference.centres[i]) - reference.targets[i]).norm();
        if (error <= fit.max_error) { ++spatial_inliers; squared += error * error; }
    }
    if (spatial_inliers * 5 < fit.n * 4) return out;
    out.inliers = spatial_inliers; out.rms = std::sqrt(squared / spatial_inliers);
    out.transform = composeSim3(fit.T, orientation.T); out.ok = true; return out;
}

inline bool measuredRegionalGauge(const Reconstruction& rec, PriorSource& gps, const AttitudeRef& attitude) {
    auto fit = measuredRegionalAlignment(rec, gps, attitude);
    if (!fit.ok) return false;
    const auto& r = fit.transform.R;
    double angle = std::acos(std::clamp((r[0] + r[4] + r[8] - 1.) * .5, -1., 1.)) * 180. / M_PI;
    if (angle > 10. || std::abs(fit.transform.scale - 1.) > .01) return false;
    size_t seen = 0, consistent = 0;
    for (const auto& kv : rec.images) {
        Vec3 position;
        if (kv.second.registered && gps.position(kv.first, position)) {
            ++seen; consistent += (cameraCenter(kv.second.pose) - position).norm() <= 5.;
        }
    }
    return seen >= 3 && consistent * 5 >= seen * 4;
}

inline Reconstruction measuredRegionalViews(const Reconstruction& rec, const std::vector<uint8_t>& positioned) {
    Reconstruction measured;
    for (const auto& kv : rec.images) if (kv.second.registered && positioned.at(kv.first)) {
        Image image; image.id = kv.first; image.camera_id = kv.second.camera_id;
        image.pose = kv.second.pose; image.registered = true; image.name = kv.second.name;
        measured.images.emplace(kv.first, std::move(image));
    }
    return measured;
}

inline std::set<uint32_t> measuredGaugeAnchors(const Reconstruction& rec) {
    const Image* first = nullptr; const Image* farthest = nullptr; double baseline = 0;
    for (const auto& kv : rec.images) if (kv.second.registered) {
        if (!first) first = &kv.second;
        double distance = (cameraCenter(kv.second.pose) - cameraCenter(first->pose)).norm();
        if (distance > baseline) { baseline = distance; farthest = &kv.second; }
    }
    if (!first || !farthest || baseline < 1e-6) return {};
    return {first->id, farthest->id};
}

struct SharedAlignmentView {
    uint32_t id = 0;
    double alignment_px = 0, local_px = 0, centre_error = 0, rotation_deg = 0;
    bool inlier = false;
};
struct SharedAlignmentReport {
    size_t shared = 0, inliers = 0;
    double median_px = 0, p95_px = 0;
    bool fitted = false, accepted = false;
    Sim3 transform;
    std::vector<SharedAlignmentView> views;
};

inline bool alignRegional(Reconstruction& rec, const Reconstruction& summary, PriorSource& gps,
                   double max_error, bool require_overlap = false, SharedAlignmentReport* report = nullptr) {
    if (report) *report = {};
    struct View { uint32_t id; Pose a, b; Camera cam; std::vector<Vec3> points; std::vector<Vec2> xy; };
    std::vector<View> views;
    for (const auto& kv : rec.images) {
        auto found = summary.images.find(kv.first);
        if (!kv.second.registered || found == summary.images.end()) continue;
        const Image& im = kv.second;
        View v; v.id = kv.first; v.a = im.pose; v.b = found->second.pose; v.cam = rec.cameras.at(im.camera_id);
        size_t step = std::max<size_t>(1, im.numPoint3D() / 32), seen = 0;
        for (size_t f = 0; f < im.point3D_ids.size(); ++f) {
            auto point = rec.points3D.find(im.point3D_ids[f]); if (point == rec.points3D.end()) continue;
            if (seen++ % step) continue; v.points.push_back(point->second.xyz); v.xy.push_back(im.points2D[f]);
        }
        if (!v.points.empty()) views.push_back(std::move(v));
    }
    if (report) report->shared = views.size();
    if (views.size() >= 3) {
        auto fit = [&](const std::vector<int>& indices) {
            std::vector<Pose> a, b; for (int i : indices) { a.push_back(views[i].a); b.push_back(views[i].b); }
            Sim3 t; std::vector<Sim3> out; if (estimateSim3FromPoses(a, b, t)) out.push_back(t); return out;
        };
        auto residual = [&](const Sim3& t, int i) {
            const View& v = views[i]; double sum = 0;
            for (size_t k = 0; k < v.points.size(); ++k)
                sum += std::min(max_error * 10, reprojErrorAt(v.cam, v.b, v.xy[k], transformPoint(t, v.points[k])) / v.cam.pixel_scale);
            double error = sum / v.points.size(); return error * error;
        };
        RansacOptions opt; opt.max_error = max_error; opt.max_num_trials = 500; opt.seed = 0;
        auto fit_result = loransac<Sim3>(int(views.size()), 2, fit, fit, residual, opt);
        if (report) {
            report->inliers = fit_result.num_inliers; report->fitted = fit_result.success;
            report->transform = fit_result.model;
            if (report->fitted) {
                std::vector<double> errors;
                for (size_t i = 0; i < views.size(); ++i) {
                    const auto& v = views[i]; SharedAlignmentView row; row.id = v.id;
                    row.alignment_px = std::sqrt(residual(fit_result.model, int(i)));
                    for (size_t k = 0; k < v.points.size(); ++k)
                        row.local_px += std::min(max_error * 10, reprojErrorAt(v.cam, v.a, v.xy[k], v.points[k]) / v.cam.pixel_scale);
                    row.local_px /= v.points.size();
                    const Pose predicted = transformPose(fit_result.model, v.a);
                    row.centre_error = (cameraCenter(predicted) - cameraCenter(v.b)).norm();
                    const auto rotation = mul(predicted.R, transpose(v.b.R));
                    row.rotation_deg = std::acos(std::clamp((rotation[0] + rotation[4] + rotation[8] - 1.) * .5, -1., 1.)) * 180. / M_PI;
                    row.inlier = fit_result.inlier_mask[i] != 0;
                    report->views.push_back(row); errors.push_back(row.alignment_px);
                }
                std::sort(errors.begin(), errors.end());
                report->median_px = errors[errors.size() / 2];
                report->p95_px = errors[std::min(errors.size() - 1, errors.size() * 95 / 100)];
            }
        }
        if (fit_result.success && fit_result.num_inliers >= 3 && fit_result.num_inliers * 2 >= views.size()) {
            if (report) report->accepted = true;
            applySim3(rec, fit_result.model); return true;
        }
        if (require_overlap) {
            slog::diag(Tag::Map, "[regional] shared alignment rejected: %zu views, %d inliers, threshold %.2f px\n",
                views.size(), fit_result.num_inliers, max_error);
            return false;
        }
    }
    if (require_overlap) {
        slog::diag(Tag::Map, "[regional] shared alignment rejected: %zu supported views; need at least 3\n", views.size());
        return false;
    }
    MetricRef ref;
    for (const auto& kv : rec.images) {
        Vec3 target;
        if (kv.second.registered && gps.position(kv.first, target)) {
            ref.centres.push_back(cameraCenter(kv.second.pose)); ref.targets.push_back(target); ref.image_ids.push_back(kv.first);
        }
    }
    auto fit = fitMetricGauge(ref, 5., MetricAxes::Full, 0.01);
    if (!fit.ok) {
        slog::diag(Tag::Map, "[regional] alignment unresolved: %zu registered, %zu shared views, GPS gauge failed (%d)\n",
                   size_t(rec.numRegistered()), views.size(), int(fit.reason));
        return false;
    }
    applySim3(rec, fit.T); return true;
}

inline bool stageRegional(Reconstruction& rec, const Reconstruction& summary, PriorSource& gps,
                          double max_error, SharedAlignmentReport* report = nullptr) {
    SharedAlignmentReport evidence;
    bool compatible = alignRegional(rec, summary, gps, max_error, !summary.images.empty(), &evidence);
    if (report) *report = std::move(evidence);
    return compatible;
}

} // namespace sfm::regional
