#pragma once

#include "sfm/map/ExportVerifiedScene.h"
#include "sfm/map/RegionalPlan.h"
#include "sfm/map/RegionalSourceHistory.h"

namespace sfm::regional {
inline bool canonicalMetricVerified(const std::map<uint32_t, ExportInputPhoto>& inputs,
                                    const std::map<uint32_t, ExportConstraintPhoto>& photos) {
    if (inputs.empty() || inputs.size() != photos.size()) return false;
    for (const auto& input : inputs) {
        const auto at = photos.find(input.first);
        if (at == photos.end() || at->second.image != input.first || !at->second.gps_verified || !at->second.scale_verified ||
            !std::isfinite(at->second.gps_error_metres) || !std::isfinite(at->second.scale_error_fraction)) return false;
    }
    return true;
}
inline ExportFinalAcceptance canonicalCaptureAcceptance(const Reconstruction& poses,
    const std::map<uint32_t, ExportInputPhoto>& inputs, const std::vector<Region>& plan,
    PriorSource& gps, const AttitudeRef& attitude, const std::vector<RegionalSourceFrame>& sources,
    const std::vector<RegionalSourceFrame>& history, bool historicalRequired, const fs::path& evidence) {
    ExportFinalAcceptance out; out.historical_evaluated = true; out.historical_required = historicalRequired;
    const size_t positioned = std::count_if(history.begin(), history.end(), [](const auto& source) { return source.positioned; });
    const size_t certified = std::count_if(history.begin(), history.end(), [](const auto& source) { return source.certified; });
    out.historical_verified = (!historicalRequired || (history.size() == 20 && positioned == 20 && certified == 18)) &&
        history::verifyFinal(history, poses, gps, attitude, evidence / "historical-source-final-proof.csv");
    std::vector<uint32_t> owners(inputs.empty() ? 0 : size_t(inputs.rbegin()->first) + 1, UINT32_MAX);
    std::vector<double> scale(plan.size(), std::numeric_limits<double>::quiet_NaN());
    std::ofstream scales(evidence / "regional-scale-final-proof.csv");
    scales << std::setprecision(17) << "owner_region,registered_photos,fit_valid,scale_error_fraction,fit_rms_m,scope\n";
    for (size_t region = 0; region < plan.size(); ++region) {
        Reconstruction subset;
        for (uint32_t id : plan[region].core) {
            if (id >= owners.size() || !inputs.count(id) || owners[id] != UINT32_MAX) throw std::runtime_error("canonical acceptance input ownership differs");
            owners[id] = uint32_t(region); const auto at = poses.images.find(id);
            if (at != poses.images.end() && at->second.registered) subset.images.emplace(*at);
        }
        const auto fit = measuredRegionalAlignment(subset, gps, attitude);
        if (fit.ok) scale[region] = std::abs(fit.transform.scale - 1.0);
        scales << region << ',' << subset.numRegistered() << ',' << fit.ok << ',' << scale[region] << ',' << fit.rms
               << ",owner_core_GPS_Sim3_audit_no_parameter_change\n";
    }
    scales.flush(); if (!scales) throw std::runtime_error("cannot write canonical scale proof");
    std::map<uint32_t, Mat3> recorded;
    for (size_t i = 0; i < attitude.image_ids.size(); ++i) recorded.emplace(attitude.image_ids[i], attitude.world_from_cam.at(i));
    std::map<uint32_t, std::vector<std::string>> sourceFailures;
    for (const auto& frame : sources) {
        bool accepted = false;
        try { accepted = regionalSourceAccepted(frame, poses, gps, attitude); } catch (const std::out_of_range&) {}
        if (!accepted) for (const auto& image : frame.views.images) if (image.second.registered)
            sourceFailures[image.first].push_back("original_source_constraint_failed:" + frame.name);
    }
    for (const auto& input : inputs) {
        ExportConstraintPhoto proof; proof.image = input.first; proof.path = input.second.path;
        const auto image = poses.images.find(input.first);
        if (image == poses.images.end() || !image->second.registered) proof.reasons.push_back("not_visually_registered");
        else {
            Vec3 position;
            if (gps.position(input.first, position)) {
                proof.gps_error_metres = (cameraCenter(image->second.pose) - position).norm();
                proof.gps_verified = std::isfinite(proof.gps_error_metres) && proof.gps_error_metres <= 5;
            }
            if (input.first < owners.size() && owners[input.first] != UINT32_MAX) {
                proof.scale_error_fraction = scale.at(owners[input.first]);
                proof.scale_verified = std::isfinite(proof.scale_error_fraction) && proof.scale_error_fraction <= .01;
            }
            const auto reference = recorded.find(input.first);
            if (reference != recorded.end()) {
                const auto& w = reference->second; const auto& r = image->second.pose.R;
                const Vec3 expected{w[6], w[7], w[8]}, predicted = mul(r, Vec3{0,0,1});
                const double up = std::atan2(expected.cross(predicted).norm(), expected.dot(predicted)) * 180 / M_PI;
                const auto residual = mul(w, r);
                const double heading = std::abs(std::atan2(residual[3] - residual[1], residual[0] + residual[4]) * 180 / M_PI);
                proof.attitude_error_degrees = std::max(up, heading);
                proof.attitude_verified = std::isfinite(proof.attitude_error_degrees) && proof.attitude_error_degrees <= 10;
            }
        }
        if (!proof.gps_verified) proof.reasons.push_back("absolute_GPS_above_5m_or_unknown");
        if (!proof.scale_verified) proof.reasons.push_back("owner_core_scale_above_1_percent_or_unobservable");
        if (!proof.attitude_verified) proof.reasons.push_back("recorded_attitude_above_10_degrees_or_unknown");
        const auto failed = sourceFailures.find(input.first);
        if (failed != sourceFailures.end()) proof.reasons.insert(proof.reasons.end(), failed->second.begin(), failed->second.end());
        out.photos.push_back(std::move(proof));
    }
    return out;
}
}
