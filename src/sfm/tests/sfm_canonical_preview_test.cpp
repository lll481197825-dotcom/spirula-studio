#include "sfm/map/CanonicalPreview.h"
#include "sfm/map/CanonicalAcceptance.h"
#include "sfm/core/Progress.h"
#include <iostream>

using namespace sfm;
using namespace sfm::regional;
static void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
int main(int argc, char** argv) try {
    const auto root = std::filesystem::u8path(argc > 1 ? argv[1] : "canonical-preview-test");
    require(!std::filesystem::exists(root), "fixture requires a new directory");
    CanonicalScene scene(root, 64u << 20); CanonicalScene::Binding binding;
    binding.rawCatalog = "preview-immutable-raw"; binding.topology = "preview-topology";
    binding.sources.push_back({"input", std::string(64, 'a'), 120, 240});
    Reconstruction poses; poses.cameras.emplace(0, Camera::defaultFor(0, 1000, 800, 700, CamModel::OpenCV));
    Image image; image.id = 0; image.camera_id = 0; image.name = "input.jpg"; image.registered = true; image.pose = {mat3Identity(), {1, 2, 3}}; poses.images.emplace(0, image);
    scene.beginLayoutCandidate(binding, poses, [&](const auto& emit) { for (uint64_t id = 0; id < 120; ++id) emit({0, id, 1000 + id}, id); },
        [&](const auto& emit) { for (uint64_t id = 0; id < 120; ++id) emit(id, {double(id), .1 * id, 10}); });
    scene.applyLandmarkTypes({{0, {0, 0, 1}}}); scene.applyRetainedObservations({12}); scene.commit("typed-preview-fixture");
    const auto identity = scene.identity(); const auto head = spirula::sha256_file((root / "head.bin").u8string());
    const auto preview = canonicalPreview(scene, 10);
    require(preview.points3D.size() == 8 && preview.images.at(0).pose.R == poses.images.at(0).pose.R && preview.images.at(0).pose.t.x == 1, "bounded preview lost actual cameras or displayed nonmetric roots");
    for (const auto& point : preview.points3D) {
        const auto actual = scene.landmark(point.first);
        require(actual.kind == LandmarkKind::FiniteXYZ && point.second.xyz.x == actual.value.x && point.second.xyz.y == actual.value.y && point.second.xyz.z == actual.value.z, "preview invented or moved a point");
    }
    require(canonicalPreview(scene, 0).points3D.empty() && canonicalPreview(scene, 1000).points3D.size() == 118, "preview point capacity differs");
    progress::set_dir((root / "live-preview").u8string()); progress::model(preview, true); progress::flush(); progress::set_dir("");
    model_store_detail::Reader live(root / "live-preview/model.bin");
    require(live.scalar<uint32_t>() == 0x4d504b56 && live.scalar<uint32_t>() == 4, "SfM point preview was not serialized as a native live model");
    live.seek(20); require(live.scalar<uint64_t>() == preview.points3D.size(), "SfM preview discarded actual finite XYZ before GUI readback");
    require(scene.identity() == identity && spirula::sha256_file((root / "head.bin").u8string()) == head, "preview altered accepted geometry");
    std::map<uint32_t, ExportInputPhoto> inputs{{0, {"input.jpg", "input.jpg"}}};
    ExportConstraintPhoto proof; proof.image = 0; proof.gps_verified = proof.scale_verified = true; proof.gps_error_metres = .2; proof.scale_error_fraction = .001;
    proof.reasons.push_back("original_deferred_source_observations_outstanding");
    std::map<uint32_t, ExportConstraintPhoto> constraints{{0, proof}};
    require(canonicalMetricVerified(inputs, constraints), "partial original-coverage state incorrectly erased measured GPS scale");
    constraints.at(0).scale_verified = false; require(!canonicalMetricVerified(inputs, constraints), "unverified metric scale claimed metres");
    constraints.at(0) = proof; constraints.at(0).gps_error_metres = std::numeric_limits<double>::quiet_NaN();
    require(!canonicalMetricVerified(inputs, constraints) && !canonicalMetricVerified(inputs, {}), "missing/nonfinite metric evidence claimed metres");
    std::cout << "PASS bounded real finite preview, immutable accepted head and independent metric evidence\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
