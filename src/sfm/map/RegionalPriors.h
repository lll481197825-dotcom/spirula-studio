#pragma once

#include "sfm/map/RegionalAlignment.h"
#include "sfm/map/RegionalLandmarks.h"
#include "sfm/map/SensorPriors.h"

namespace sfm::regional {

inline PosePriors regionalGpsPriors(const Reconstruction& rec, PriorSource& gps) {
    PosePriors out; out.gps.ok = true; out.gps.gate = 5.;
    out.gps.sigma_h = kTrustedGpsSigma;
    for (const auto& kv : rec.images) if (kv.second.registered) {
        Vec3 position;
        if (!gps.position(kv.first, position) || !std::isfinite(position.norm()))
            throw std::runtime_error("regional GPS prior missing for registered photo");
        PriorCentre c; c.n = 1; c.img[0] = kv.first; c.b = position;
        c.sigma = {kTrustedGpsSigma, kTrustedGpsSigma, kTrustedGpsSigma};
        c.cauchy = kTrustedGpsCauchy; out.centres.push_back(c);
    }
    return out;
}

struct RegionalGpsQuality {
    size_t photos = 0, inliers = 0;
    double cost = 0;
    bool accepted() const { return photos >= 3 && inliers * 5 >= photos * 4; }
};
inline RegionalGpsQuality regionalGpsQuality(const Reconstruction& rec, PriorSource& gps) {
    RegionalGpsQuality out;
    for (const auto& kv : rec.images) if (kv.second.registered) {
        Vec3 p;
        if (!gps.position(kv.first, p)) throw std::runtime_error("regional GPS audit missing photo");
        const double e = (cameraCenter(kv.second.pose) - p).norm();
        if (!std::isfinite(e)) throw std::runtime_error("regional GPS audit nonfinite pose");
        ++out.photos; out.inliers += e <= 5.;
        out.cost += e <= 5. ? e * e : 5. * (2. * e - 5.);
    }
    return out;
}
inline bool regionalGpsGauge(const Reconstruction& rec, PriorSource& gps, const AttitudeRef& attitude) {
    if (!regionalGpsQuality(rec, gps).accepted()) return false;
    size_t recorded = 0;
    for (uint32_t id : attitude.image_ids) {
        const auto at = rec.images.find(id);
        recorded += at != rec.images.end() && at->second.registered;
    }
    if (recorded >= 3 && recorded * 5 >= rec.numRegistered() * 4)
        return measuredRegionalGauge(rec, gps, attitude);
    MetricRef ref;
    for (const auto& kv : rec.images) if (kv.second.registered) {
        Vec3 p; if (!gps.position(kv.first, p)) return false;
        ref.centres.push_back(cameraCenter(kv.second.pose)); ref.targets.push_back(p); ref.image_ids.push_back(kv.first);
    }
    const auto fit = fitMetricGauge(ref, 5., MetricAxes::Full, .01);
    if (!fit.ok || std::abs(fit.T.scale - 1.) > .01) return false;
    const double angle = std::acos(std::clamp((fit.T.R[0] + fit.T.R[4] + fit.T.R[8] - 1.) * .5, -1., 1.)) * 180. / M_PI;
    if (angle > 10.) return false;
    if (recorded >= 3) {
        const auto measured = fitAttitudeGauge(rec, attitude, true);
        if (!measured.ok || !measured.north) return false;
        const auto& R = measured.T.R;
        if (std::acos(std::clamp((R[0] + R[4] + R[8] - 1.) * .5, -1., 1.)) * 180. / M_PI > 10.) return false;
    }
    return true;
}
inline bool regionalGpsRoundAccepted(const RegionalGpsQuality& next, const RegionalGpsQuality& previous) {
    if (next.photos != previous.photos) return false;
    if (previous.accepted()) return next.accepted();
    return next.inliers >= previous.inliers && next.cost <= previous.cost * (1 + 1e-8) + 1e-8;
}

inline void addRegionalGpsTargets(std::set<uint32_t>& targets, const Reconstruction& rec, PriorSource& gps) {
    for (const auto& kv : rec.images) if (kv.second.registered) {
        Vec3 p;
        if (!gps.position(kv.first, p)) throw std::runtime_error("regional GPS target missing photo");
        if ((cameraCenter(kv.second.pose) - p).norm() > 5.) targets.insert(kv.first);
    }
}

struct RegionalSourceFrame {
    std::string name;
    Reconstruction views;
    bool positioned = false, certified = false;
};
using RegionalPoseAnchors = std::map<uint32_t, Pose>;

inline Reconstruction regionalSourceViews(const RegionalSourceFrame& source, const Reconstruction& poses) {
    Reconstruction views;
    for (const auto& kv : source.views.images) if (kv.second.registered) {
        const auto& image = poses.images.at(kv.first);
        if (!image.registered || image.camera_id != kv.second.camera_id)
            throw std::runtime_error("regional source camera identity changed");
        Image copy; copy.id = kv.first; copy.camera_id = image.camera_id;
        copy.registered = true; copy.pose = image.pose; views.images.emplace(kv.first, std::move(copy));
    }
    return views;
}

inline bool regionalSourceAccepted(const RegionalSourceFrame& source, const Reconstruction& poses,
                                  PriorSource& gps, const AttitudeRef& attitude) {
    const auto views = regionalSourceViews(source, poses);
    return (!source.positioned || regionalGpsQuality(views, gps).accepted()) &&
        (!source.certified || regionalGpsGauge(views, gps, attitude));
}

inline std::set<uint32_t> protectRegionalSources(Reconstruction& poses, const std::vector<RegionalSourceFrame>& sources,
    PriorSource& gps, const AttitudeRef& attitude, RegionalPoseAnchors& anchors) {
    Reconstruction candidate = poses;
    auto held = anchors; std::set<uint32_t> changed;
    for (const auto& kv : held) {
        candidate.images.at(kv.first).pose = kv.second; changed.insert(kv.first);
    }
    for (size_t attempt = 0; attempt <= sources.size(); ++attempt) {
        bool accepted = true;
        for (const auto& source : sources) {
            if (regionalSourceAccepted(source, candidate, gps, attitude)) continue;
            accepted = false;
            for (const auto& kv : source.views.images) if (kv.second.registered) {
                candidate.images.at(kv.first).pose = kv.second.pose;
                held[kv.first] = kv.second.pose; changed.insert(kv.first);
            }
        }
        if (accepted) {
            for (const auto& kv : candidate.images) poses.images.at(kv.first).pose = kv.second.pose;
            anchors = std::move(held); return changed;
        }
    }
    std::string failed;
    for (const auto& source : sources) if (!regionalSourceAccepted(source, candidate, gps, attitude))
        failed += (failed.empty() ? "" : ", ") + source.name;
    throw std::runtime_error("regional certified visual source poses conflict: " + failed);
}

template<class ReadModel> inline void restoreSharedMembers(SharedIndex& shared, ReadModel read) {
    for (size_t m = 0; m < shared.models; ++m) {
        cancel::check(); auto rec = read(m);
        for (const auto& item : shared.by_model.at(m)) {
            const auto& point = rec.points3D.at(item.first);
            auto& p = shared.landmarks.at(item.second);
            for (const auto& e : point.track) {
                const uint64_t key = (uint64_t(e.image_id) << 32) | e.point2D_idx;
                const auto obs = std::lower_bound(p.observations.begin(), p.observations.end(), key,
                    [](const SharedObservation& a, uint64_t b) { return a.key < b; });
                const auto& xy = rec.images.at(e.image_id).points2D.at(e.point2D_idx);
                if (obs == p.observations.end() || obs->key != key || obs->xy.x != xy.x || obs->xy.y != xy.y)
                    throw std::runtime_error("regional migration changed original shared observations");
            }
            for (auto& member : p.members) if (member.model == m) member.xyz = point.xyz;
        }
    }
    for (auto& p : shared.landmarks) if (p.valid) {
        p.xyz = {};
        for (const auto& member : p.members) p.xyz = p.xyz + member.xyz;
        p.xyz = p.xyz * (1. / p.members.size());
    }
}

} // namespace sfm::regional
