#pragma once
#include "sfm/map/CanonicalScene.h"
#include "sfm/map/RegionalStore.h"

namespace sfm::regional {
inline double reprojLandmarkError(const Camera& camera, const Pose& pose, const Vec2& xy,
                                 const CanonicalLandmark& landmark, bool* valid = nullptr) {
    validateLandmark(landmark);
    if (landmark.kind == LandmarkKind::RetainedObservation) throw std::logic_error("retained observation has no projection residual");
    if (landmark.kind == LandmarkKind::FiniteXYZ) return reprojErrorAt(camera, pose, xy, landmark.value, valid);
    Pose rotationOnly = pose; rotationOnly.t = {};
    return reprojErrorAt(camera, rotationOnly, xy, landmark.value, valid);
}
// Current canonical support can be a residual of a formerly multi-view source track.
inline bool isRetainedObservationTrack(const std::vector<Observation>& track) {
    if (track.empty()) throw std::runtime_error("canonical root has no immutable raw support");
    const auto image = uint32_t(track.front().point >> 32);
    for (const auto& row : track) {
        if (!std::isfinite(row.xy.x) || !std::isfinite(row.xy.y)) throw std::runtime_error("retained raw coordinate is nonfinite");
        if (uint32_t(row.point >> 32) != image) return false;
    }
    return true;
}
namespace landmark_geometry_detail {
struct Objective { uint64_t invalid = 0; double finite = 0; };
inline Objective objective(CanonicalLandmark value, const std::vector<Observation>& track, const Reconstruction& poses, double gate) {
    Objective result;
    for (const auto& row : track) {
        const auto& image = poses.images.at(uint32_t(row.point >> 32)); const auto& camera = poses.cameras.at(image.camera_id);
        if (!image.registered || !(camera.pixel_scale > 0) || !std::isfinite(camera.pixel_scale)) throw std::runtime_error("landmark support measurement unavailable");
        bool valid = true; const double error = reprojLandmarkError(camera, image.pose, row.xy, value, &valid) / camera.pixel_scale;
        if (!valid || !std::isfinite(error)) { ++result.invalid; continue; }
        result.finite += error <= gate ? error * error : gate * (2 * error - gate);
    }
    return result;
}
inline std::pair<Vec3, Vec3> tangent(Vec3 d) {
    const Vec3 first = (std::abs(d.x) < .9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0}).cross(d).normalized();
    return {first, d.cross(first).normalized()};
}
}
inline CanonicalLandmark refineLandmark(CanonicalLandmark initial, const std::vector<Observation>& track,
                                       const Reconstruction& poses, double gate) {
    validateLandmark(initial); if (!(gate > 0) || !std::isfinite(gate)) throw std::invalid_argument("landmark refinement gate invalid");
    if (initial.kind == LandmarkKind::RetainedObservation) { if (!isRetainedObservationTrack(track)) throw std::logic_error("retained support gained a second photo; typed depth admission required"); return initial; }
    if (initial.kind == LandmarkKind::FiniteXYZ) return {refineUnifiedPoint(initial.value, track, poses, gate), initial.kind};
    using namespace landmark_geometry_detail;
    auto before = objective(initial, track, poses, gate);
    auto improves = [&](CanonicalLandmark candidate, const Objective& next) {
        if (next.invalid > before.invalid || !std::isfinite(next.finite)) return false;
        if (next.invalid == before.invalid) return next.finite < before.finite;
        double priorCommon = 0, nextCommon = 0;
        for (const auto& row : track) {
            const auto& image = poses.images.at(uint32_t(row.point >> 32)); const auto& camera = poses.cameras.at(image.camera_id); bool a = true, b = true;
            const double x = reprojLandmarkError(camera, image.pose, row.xy, initial, &a) / camera.pixel_scale;
            const double y = reprojLandmarkError(camera, image.pose, row.xy, candidate, &b) / camera.pixel_scale;
            if (!a || !b) continue;
            priorCommon += x <= gate ? x * x : gate * (2 * x - gate); nextCommon += y <= gate ? y * y : gate * (2 * y - gate);
        }
        return nextCommon <= priorCommon * (1 + 1e-10) + 1e-10;
    };
    if (before.invalid) {
        Vec3 mean{}; for (const auto& row : track) { const auto& image = poses.images.at(uint32_t(row.point >> 32)); mean = mean + mul(transpose(image.pose.R), poses.cameras.at(image.camera_id).bearing(row.xy)).normalized(); }
        if (mean.norm() > 1e-12) { CanonicalLandmark candidate{mean.normalized(), LandmarkKind::UnitDirection}; const auto after = objective(candidate, track, poses, gate); if (improves(candidate, after)) { initial = candidate; before = after; } }
    }
    for (int iteration = 0; iteration < 10; ++iteration) {
        const auto basis = tangent(initial.value); const Vec3 axes[2]{basis.first, basis.second};
        double h00 = 0, h01 = 0, h11 = 0, g0 = 0, g1 = 0;
        for (const auto& row : track) {
            const auto& image = poses.images.at(uint32_t(row.point >> 32)); const auto& camera = poses.cameras.at(image.camera_id);
            bool valid = true; reprojLandmarkError(camera, image.pose, row.xy, initial, &valid); if (!valid) continue;
            const auto pixel = camera.project(mul(image.pose.R, initial.value));
            const double rx = (camera.isSpherical() ? std::remainder(pixel.x - row.xy.x, double(camera.width)) : pixel.x - row.xy.x) / camera.pixel_scale;
            const double ry = (pixel.y - row.xy.y) / camera.pixel_scale; if (!std::isfinite(rx) || !std::isfinite(ry)) continue;
            const double residual = std::hypot(rx, ry), weight = residual > gate ? gate / residual : 1.;
            double jx[2], jy[2]; const double epsilon = 1e-6;
            for (int k = 0; k < 2; ++k) {
                const auto moved = camera.project(mul(image.pose.R, (initial.value + axes[k] * epsilon).normalized()));
                jx[k] = (camera.isSpherical() ? std::remainder(moved.x - pixel.x, double(camera.width)) : moved.x - pixel.x) / (epsilon * camera.pixel_scale);
                jy[k] = (moved.y - pixel.y) / (epsilon * camera.pixel_scale);
            }
            h00 += weight * (jx[0] * jx[0] + jy[0] * jy[0]); h01 += weight * (jx[0] * jx[1] + jy[0] * jy[1]);
            h11 += weight * (jx[1] * jx[1] + jy[1] * jy[1]); g0 += weight * (jx[0] * rx + jy[0] * ry); g1 += weight * (jx[1] * rx + jy[1] * ry);
        }
        h00 += 1e-8 * std::max(1., h00); h11 += 1e-8 * std::max(1., h11); const double determinant = h00 * h11 - h01 * h01;
        if (!(determinant > 0) || !std::isfinite(determinant)) break;
        const double x = (h11 * g0 - h01 * g1) / determinant, y = (h00 * g1 - h01 * g0) / determinant;
        if (!std::isfinite(x) || !std::isfinite(y)) break;
        bool improved = false;
        for (double alpha : {1., .5, .25, .125}) {
            CanonicalLandmark candidate{(initial.value - (axes[0] * x + axes[1] * y) * alpha).normalized(), LandmarkKind::UnitDirection};
            const auto after = objective(candidate, track, poses, gate);
            if (improves(candidate, after)) { initial = candidate; before = after; improved = true; break; }
        }
        if (!improved || std::hypot(x, y) < 1e-10) break;
    }
    validateLandmark(initial); return initial;
}
struct UnitDirectionFit {
    bool accepted = false;
    Vec3 value{};
    double maxWorldRayAngleDegrees = 0, maxError = 0;
    uint64_t observations = 0, inliers = 0, distinctPhotos = 0;
};
// Direction certification asserts angular observability, not physical infinite depth.
inline UnitDirectionFit fitUnitDirection(const std::vector<Observation>& track, const Reconstruction& poses,
                                        double gate, double minTriAngleDegrees) {
    if (!(gate > 0) || !std::isfinite(gate) || !(minTriAngleDegrees > 0) || !std::isfinite(minTriAngleDegrees) || minTriAngleDegrees >= 180)
        throw std::invalid_argument("direction certification requires original positive pixel/parallax gates");
    UnitDirectionFit result; std::vector<Vec3> rays; rays.reserve(track.size()); std::set<uint32_t> photos; Vec3 mean{};
    for (const auto& row : track) {
        const auto id = uint32_t(row.point >> 32); const auto& image = poses.images.at(id); const auto& camera = poses.cameras.at(image.camera_id);
        if (!image.registered || !(camera.pixel_scale > 0) || !std::isfinite(camera.pixel_scale)) throw std::runtime_error("direction support lacks registered native camera");
        const auto ray = mul(transpose(image.pose.R), camera.bearing(row.xy)).normalized();
        if (!step::finite(ray) || std::abs(ray.dot(ray) - 1) > 1e-10) throw std::runtime_error("direction support world ray invalid");
        for (const auto& previous : rays) result.maxWorldRayAngleDegrees = std::max(result.maxWorldRayAngleDegrees, std::acos(std::clamp(ray.dot(previous), -1., 1.)) * 180. / M_PI);
        rays.push_back(ray); photos.insert(id); mean = mean + ray; ++result.observations;
    }
    result.distinctPhotos = photos.size();
    if (photos.size() < 2 || !(result.maxWorldRayAngleDegrees < minTriAngleDegrees) || !(mean.norm() > 1e-12)) return result;
    auto candidate = refineLandmark({mean.normalized(), LandmarkKind::UnitDirection}, track, poses, gate); result.value = candidate.value;
    for (const auto& row : track) {
        const auto& image = poses.images.at(uint32_t(row.point >> 32)); const auto& camera = poses.cameras.at(image.camera_id);
        bool valid = true; const double error = reprojLandmarkError(camera, image.pose, row.xy, candidate, &valid) / camera.pixel_scale;
        if (valid && std::isfinite(error) && error <= gate) ++result.inliers;
        result.maxError = std::max(result.maxError, error);
    }
    result.accepted = result.observations > 0 && result.inliers == result.observations && std::isfinite(result.maxError); return result;
}
}
