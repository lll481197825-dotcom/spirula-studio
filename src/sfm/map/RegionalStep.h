#pragma once

#include "sfm/core/Model.h"

namespace sfm::regional::step {
inline void validateAlpha(double alpha) {
    if (!std::isfinite(alpha) || !(alpha > 0) || alpha > 1)
        throw std::invalid_argument("regional parameter step requires 0 < alpha <= 1");
}
inline bool finite(Vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
inline void validatePose(const Pose& pose) {
    if (!finite(pose.t)) throw std::runtime_error("nonfinite regional step pose");
    for (double value : pose.R) if (!std::isfinite(value)) throw std::runtime_error("nonfinite regional step rotation");
    const Mat3 identity = mul(pose.R, transpose(pose.R));
    for (size_t i = 0; i < identity.size(); ++i)
        if (std::abs(identity[i] - (i % 4 == 0 ? 1. : 0.)) > 1e-6)
            throw std::runtime_error("regional step rotation is not orthonormal");
    if (det3(pose.R) < 0) throw std::runtime_error("regional step rotation is a reflection");
}
inline Pose interpolatePose(const Pose& accepted, const Pose& candidate, double alpha) {
    validateAlpha(alpha); validatePose(accepted); validatePose(candidate);
    if (alpha == 1) return candidate;
    if (accepted.R == candidate.R && accepted.t.x == candidate.t.x && accepted.t.y == candidate.t.y && accepted.t.z == candidate.t.z)
        return accepted;
    Pose result;
    result.R = mul(so3Exp(so3Log(mul(candidate.R, transpose(accepted.R))) * alpha), accepted.R);
    const Vec3 centre = cameraCenter(accepted) * (1 - alpha) + cameraCenter(candidate) * alpha;
    result.t = mul(result.R, centre) * -1.; validatePose(result); return result;
}
inline std::array<double, 14> values(const Camera& camera) {
    return {camera.fx, camera.fy, camera.cx, camera.cy, camera.k1, camera.k2, camera.p1, camera.p2,
            camera.k3, camera.k4, camera.k5, camera.k6, camera.sx1, camera.sy1};
}
inline std::array<double*, 14> fields(Camera& camera) {
    return {&camera.fx, &camera.fy, &camera.cx, &camera.cy, &camera.k1, &camera.k2, &camera.p1, &camera.p2,
            &camera.k3, &camera.k4, &camera.k5, &camera.k6, &camera.sx1, &camera.sy1};
}
inline void validateCamera(const Camera& camera) {
    for (double value : values(camera)) if (!std::isfinite(value)) throw std::runtime_error("nonfinite regional step intrinsics");
    if (!(camera.fx > 0) || !(camera.fy > 0) || !(camera.pixel_scale > 0) || !std::isfinite(camera.pixel_scale) ||
        camera.width <= 0 || camera.height <= 0) throw std::runtime_error("invalid regional step camera");
}
inline Camera interpolateCamera(const Camera& accepted, const Camera& candidate, double alpha) {
    validateAlpha(alpha); validateCamera(accepted); validateCamera(candidate);
    if (accepted.id != candidate.id || accepted.model != candidate.model || accepted.width != candidate.width ||
        accepted.height != candidate.height || accepted.pixel_scale != candidate.pixel_scale)
        throw std::runtime_error("regional step changed camera identity or measurement scale");
    std::array<double, 12> previous{}, next{}, blended{};
    packIntrinsics(accepted, previous.data()); packIntrinsics(candidate, next.data());
    const int count = camNumParams(accepted.model);
    for (int i = 0; i < count; ++i) blended[size_t(i)] = previous[size_t(i)] * (1 - alpha) + next[size_t(i)] * alpha;
    Camera decodedBefore = accepted, decodedNext = candidate;
    unpackIntrinsics(decodedBefore, previous.data()); unpackIntrinsics(decodedNext, next.data());
    const auto decodedA = values(decodedBefore), decodedB = values(decodedNext), a = values(accepted), b = values(candidate);
    auto represented = fields(decodedNext);
    for (size_t i = 0; i < represented.size(); ++i) if (decodedA[i] == decodedB[i]) *represented[i] = a[i];
    if (values(decodedNext) != b) throw std::runtime_error("regional step changed parameters outside the camera BA layout");
    if (alpha == 1) return candidate;
    Camera result = accepted; unpackIntrinsics(result, blended.data());
    auto output = fields(result);
    for (size_t i = 0; i < output.size(); ++i) if (decodedA[i] == decodedB[i]) *output[i] = a[i];
    validateCamera(result); return result;
}
}
