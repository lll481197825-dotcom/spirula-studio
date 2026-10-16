#include "sfm/map/SceneVisualEvidence.h"
#include "sfm/map/ImmutableSceneAliases.h"
#include <iostream>

using namespace sfm;
using namespace sfm::regional;
static void check(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
int main(int argc, char** argv) try {
    const auto root = argc > 1 ? std::filesystem::u8path(argv[1]) : std::filesystem::path("scene-visual-evidence-fixture");
    if (std::filesystem::exists(root)) throw std::runtime_error("fixture directory must be new");
    std::filesystem::create_directories(root);
    Reconstruction model; model.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::Pinhole);
    for (uint32_t id = 0; id < 6; ++id) {
        Image image; image.id = id; image.camera_id = 0; image.registered = true; image.name = "photo-" + std::to_string(id);
        image.pose = {mat3Identity(), {-.3 * id, 0, 0}}; image.points2D.resize(13); image.point3D_ids.assign(13, kInvalidPoint3D);
        model.images.emplace(id, image);
    }
    for (uint32_t group = 0; group < 3; ++group) for (uint32_t feature = 0; feature < 12; ++feature) {
        const uint64_t id = group * 12 + feature; Point3D point; point.xyz = {.1 * feature, .05 * group, 5 + .02 * feature};
        for (uint32_t image = group * 2; image < group * 2 + 2; ++image) {
            auto& view = model.images.at(image); view.points2D[feature] = model.cameras.at(0).project(point.xyz + view.pose.t);
            view.point3D_ids[feature] = id; point.track.push_back({image, feature});
        }
        if (!id) for (int duplicate = 0; duplicate < 4; ++duplicate) point.track.push_back({0, 0});
        model.points3D.emplace(id, point);
    }
    Point3D bridge; bridge.xyz = {.4, .3, 6}; bridge.track = {{1, 12}, {2, 12}};
    for (const auto& row : bridge.track) {
        auto& image = model.images.at(row.image_id); auto xy = model.cameras.at(0).project(bridge.xyz + image.pose.t);
        if (row.image_id == 2) xy.x += 100;
        image.points2D[row.point2D_idx] = xy; image.point3D_ids[row.point2D_idx] = 36;
    }
    model.points3D.emplace(36, bridge); model.next_point3D_id = 37;
    RegionalWorkingStore catalog(root / "catalog", 64u << 20);
    for (int source = 0; source < 2; ++source) {
        const auto path = root / ("source-" + std::to_string(source) + ".bin"); model_store_detail::writeModel(model, path);
        catalog.importModel(path, model, "source-" + std::to_string(source));
    }
    const std::vector<uint32_t> owners{0, 0, 1, 1, 2, 2}; ImmutableSceneAliases aliases(root / "aliases", 64u << 20);
    aliases.prepare(catalog, owners); ImmutableSceneTopology index(root / "index", 64u << 20);
    index.prepare(catalog, owners, aliases.identity(), aliases.points(), [&](auto key) { return aliases.canonical(key); });
    auto poses = catalog.poses(); poses.points3D.clear(); for (auto& image : poses.images) { image.second.points2D.clear(); image.second.point3D_ids.clear(); }
    CanonicalScene scene(root / "scene", 64u << 20);
    scene.initialize(CanonicalScene::bindingFor(catalog, index.identity()), poses, [&](const auto& emit) { aliases.emitAliases(emit); },
        [&](const auto& emit) { index.visitSeeds([&](const auto& seed) { emit(seed.root, seed.xyz); }); });
    const auto identity = scene.identity(); const auto raw = sceneQuality(index, scene, 3, 32u << 20, 2);
    const auto evidence = sceneVisualEvidence(index, scene, 3, 32u << 20, 2);
    check(raw.observations == 156 && evidence.uniqueObservations == 74, "duplicate source/raw rows inflated distinct visual evidence");
    check(evidence.photos.at(0).observations == 12 && evidence.photos.at(0).inliers == 12, "duplicate original feature counted as additional visual support");
    check(evidence.photos.at(2).observations == 13 && evidence.photos.at(2).inliers == 12, "outlier bridge became visual inlier");
    check(evidence.landmarks.at(0) == 12 && evidence.landmarks.at(2) == 12, "duplicate features or rejected bridge inflated distinct inlier landmarks");
    check(evidence.landmarks.at(1) == 12, "single-inlier-view bridge was counted as a triangulated landmark");
    check(evidence.components == 3 && evidence.component[0] == evidence.component[1] && evidence.component[1] != evidence.component[2] && evidence.component[4] != evidence.component[2], "outlier-only raw bridge joined distinct visual components");
    bool refused = false; try { sceneVisualEvidence(index, scene, 3, 1u << 20, 2); } catch (const std::length_error&) { refused = true; }
    check(refused && scene.identity() == identity, "visual proof exceeded allowance or mutated accepted geometry");
    scene.beginCandidate(); auto moved = scene.poses().images.at(0).pose; moved.t.x += 10; scene.setPose(0, moved);
    const auto oneView = sceneVisualEvidence(index, scene, 3, 32u << 20, 2);
    check(photoQualityAccepted(oneView.photos.at(1), 3) && oneView.photos.at(1).inliers == 13 && oneView.landmarks.at(1) == 0,
        "thirteen good single-view projections falsely met the distinct triangulated-landmark contract");
    check(oneView.component.at(1) != oneView.component.at(0) && oneView.component.at(1) != oneView.component.at(2), "single-view-only evidence invented visual connectivity");
    scene.abort("single-valid-view-evidence-fixture"); check(scene.identity() == identity, "single-view proof changed accepted scene authority");
    Quality before, after; before.observations = after.observations = 10; before.invalid_projections = after.invalid_projections = 2;
    before.cost = after.cost = 2e30; before.finite_cost = 100; after.finite_cost = 50;
    check(!sceneFiniteImprovementSmall(before, after, .001), "invalid sentinel hid genuine finite improvement as convergence");
    after.finite_cost = 99.99999; check(sceneFiniteImprovementSmall(before, after, .001), "small actual finite improvement failed convergence");
    after.invalid_projections = 1; check(!sceneFiniteImprovementSmall(before, after, .001), "invalid projection reduction was mistaken for convergence");
    before.invalid_projections = after.invalid_projections = 10;
    check(!sceneFiniteImprovementSmall(before, after, .001), "unobservable all-invalid scene became converged");
    std::cout << "distinct visual evidence: raw multiplicity, unique features, inlier-only connectivity and bounded read-only proof passed\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
