#include "sfm/map/CanonicalAcceptance.h"
#include <iostream>

using namespace sfm;
using namespace sfm::regional;
class ExactGps : public PriorSource {
public:
    std::vector<Vec3> positions;
    bool has(uint32_t i) const override { return i < positions.size(); }
    bool relativeRotation(uint32_t, uint32_t, Mat3&, double&) const override { return false; }
    std::vector<uint32_t> neighbours(uint32_t) const override { return {}; }
    PosePriors factors(const std::vector<PosedImage>&) override { return {}; }
    bool position(uint32_t i, Vec3& p) const override { if (!has(i)) return false; p = positions[i]; return true; }
};
static void check(bool condition, const char* why) { if (!condition) throw std::runtime_error(why); }
int main(int argc, char** argv) try {
    const std::filesystem::path root = argc > 1 ? std::filesystem::u8path(argv[1]) : std::filesystem::path("canonical-acceptance-fixture");
    if (std::filesystem::exists(root)) throw std::runtime_error("fixture output must be a new directory"); std::filesystem::create_directories(root);
    Reconstruction poses; ExactGps gps; AttitudeRef attitude; Region region;
    std::map<uint32_t, ExportInputPhoto> inputs;
    for (uint32_t id = 0; id < 24; ++id) {
        const Vec3 centre{double(id % 6) * 20, double(id / 6) * 25, double(id % 3) + 30}; gps.positions.push_back(centre);
        Image image; image.id = id; image.camera_id = 0; image.name = "photo-" + std::to_string(id); image.registered = true;
        image.pose = {mat3Identity(), centre * -1}; poses.images.emplace(id, image); inputs.emplace(id, ExportInputPhoto{"original/" + image.name, image.name});
        attitude.image_ids.push_back(id); attitude.world_from_cam.push_back(mat3Identity()); region.core.push_back(id); region.images.push_back(id);
    }
    attitude.registered = 24;
    auto audit = [&](const Reconstruction& candidate, const AttitudeRef& reference = AttitudeRef{}, bool required = false) {
        return canonicalCaptureAcceptance(candidate, inputs, {region}, gps, reference, {}, {}, required, root);
    };
    const auto accepted = audit(poses, attitude);
    check(accepted.photos.size() == 24 && accepted.historical_evaluated && !accepted.historical_required && accepted.historical_verified, "original inventory or explicit historical N/A lost");
    for (const auto& photo : accepted.photos) check(photo.gps_verified && photo.scale_verified && photo.attitude_verified && photo.gps_error_metres == 0 && photo.scale_error_fraction < 1e-9 && photo.attitude_error_degrees < 1e-9, "exact ENU scene did not pass measured constraints");
    auto scaled = poses;
    for (auto& item : scaled.images) item.second.pose.t = item.second.pose.t * 1.04;
    const auto rejected_scale = audit(scaled, attitude);
    for (const auto& photo : rejected_scale.photos) check(!photo.scale_verified && photo.scale_error_fraction > .01, "GPS scale distortion was absorbed or ignored");
    auto shifted = poses; shifted.images.at(3).pose.t.x += 6;
    const auto rejected_gps = audit(shifted, attitude);
    check(!rejected_gps.photos.at(3).gps_verified && rejected_gps.photos.at(3).gps_error_metres == 6, "absolute GPS residual was refitted away");
    auto omitted = poses; omitted.images.erase(7); const auto missing = audit(omitted, attitude);
    check(missing.photos.size() == 24 && !missing.photos.at(7).gps_verified && !missing.photos.at(7).attitude_verified && !missing.photos.at(7).reasons.empty(), "missing input escaped final proof");
    const auto unknown = audit(poses);
    for (const auto& photo : unknown.photos) check(!photo.attitude_verified && !photo.scale_verified, "unknown attitude/metric fit falsely passed");
    const auto absent_history = audit(poses, attitude, true); check(!absent_history.historical_verified, "empty proof replaced required historical sources");
    auto conflicting = region; conflicting.core.push_back(3); bool blocked = false;
    try { canonicalCaptureAcceptance(poses, inputs, {conflicting}, gps, attitude, {}, {}, false, root); } catch (const std::exception&) { blocked = true; }
    check(blocked, "ambiguous input owner accepted");
    for (const auto& image : poses.images) check(image.second.pose.R == mat3Identity() && image.second.pose.t.x == -gps.positions[image.first].x, "acceptance modified input poses");
    std::cout << "canonical capture acceptance: exact input coverage, measured scale/GPS/attitude, unknowns and historical proof passed\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
