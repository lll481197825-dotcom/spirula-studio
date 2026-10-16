#include "sfm/map/CanonicalAdmission.h"
#include <iostream>
#include <cstring>

using namespace sfm;
using namespace sfm::regional;
using Path = std::filesystem::path;
static void check(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
template<class Run> static void rejects(Run run, const char* reason) {
    bool failed = false; try { run(); } catch (const std::exception&) { failed = true; } check(failed, reason);
}
static Reconstruction model(bool incoming) {
    Reconstruction out; out.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::Pinhole); out.cameras[0].pixel_scale = 2;
    for (uint32_t id = 0; id < 4; ++id) {
        Image image; image.id = id; image.camera_id = 0; image.name = "photo-" + std::to_string(id); image.registered = true;
        image.pose = {mat3Identity(), {-.4 * id, 0, 0}}; image.points2D.resize(2); image.point3D_ids.assign(2, kInvalidPoint3D); out.images[id] = image;
    }
    for (uint32_t point = 0; point < 2; ++point) {
        const Vec3 truth{.2 * point, .1, 5 + .3 * point}; Point3D value; value.xyz = truth;
        const uint64_t id = (incoming ? 20 : 10) + point;
        for (auto& item : out.images) {
            auto xy = out.cameras.at(0).project(truth + item.second.pose.t);
            if (!point && item.first == 2) xy.x += 8;
            item.second.points2D[point] = xy;
        }
        std::vector<uint32_t> support = incoming && point == 1 ? std::vector<uint32_t>{2, 3, 2} :
            incoming ? std::vector<uint32_t>{0, 1, 0, 2} : std::vector<uint32_t>{0, 1, 0};
        for (uint32_t image : support) {
            auto& view = out.images.at(image); view.point3D_ids[point] = id; value.track.push_back({image, point});
        }
        if (incoming) value.xyz = value.xyz + Vec3{point ? .04 : 500, .03, .02};
        out.points3D.emplace(id, value);
    }
    out.next_point3D_id = incoming ? 22 : 12; return out;
}
int main(int argc, char** argv) try {
    const Path root = argc > 1 ? std::filesystem::u8path(argv[1]) : Path("canonical-admission-seed-fixture");
    if (std::filesystem::exists(root)) throw std::runtime_error("fixture requires a new output directory"); std::filesystem::create_directories(root);
    RegionalWorkingStore catalog(root / "catalog", 128u << 20); auto old = model(false), incoming = model(true);
    const auto oldPath = root / "old.bin", newPath = root / "new.bin";
    model_store_detail::writeModel(old, oldPath); const auto oldHash = spirula::sha256_file(oldPath.u8string()); catalog.importModel(oldPath, old, "old");
    const std::vector<uint32_t> owners(4, 0); ImmutableSceneTopology oldIndex(root / "old-index", 32u << 20);
    oldIndex.prepare(catalog, owners, "accepted-stable-roots", 2, [](auto key) { return key.ordinal; });
    const auto manifestHash = spirula::sha256_file((root / "old-index" / oldIndex.identity() / "manifest.bin").u8string());
    const auto copy = oldIndex.forkImmutable(root / "forked-index"); ImmutableSceneTopology forked(root / "forked-index", 32u << 20); forked.restore(oldIndex.identity());
    check(copy == std::filesystem::absolute(root / "forked-index") / oldIndex.identity() && forked.identity() == oldIndex.identity() && forked.observationCount() == 6 &&
        spirula::sha256_file((copy / "manifest.bin").u8string()) == manifestHash, "fork rebuilt or changed accepted immutable topology");
    for (const char* name : {"rows.bin", "point-ranges.bin", "point-owners.bin", "photo-ranges.bin", "photo-rows.bin", "seeds.bin", "aliases.bin"})
        check(spirula::sha256_file((copy / name).u8string()) == spirula::sha256_file((root / "old-index" / oldIndex.identity() / name).u8string()), "fork changed an immutable index byte");
    rejects([&] { oldIndex.forkImmutable(root / "forked-index"); }, "fork overwrote existing destination");
    rejects([&] { oldIndex.forkImmutable(root / "old-index" / "inside"); }, "fork wrote into source tree");
    CanonicalScene scene(root / "scene", 128u << 20); const auto poses = catalog.poses();
    scene.initialize(CanonicalScene::bindingFor(catalog, oldIndex.identity()), poses, [&](const auto& emit) { oldIndex.emitAliases(emit); },
        [&](const auto& emit) { oldIndex.visitSeeds([&](const auto& seed) { emit(seed.root, seed.xyz); }); });
    const auto head = scene.identity(); const auto oldXYZ = scene.pointXYZ(0);
    model_store_detail::writeModel(incoming, newPath); const auto newHash = spirula::sha256_file(newPath.u8string()); catalog.importModel(newPath, incoming, "incoming");
    ImmutableSceneTopology next(root / "next-index", 32u << 20);
    next.prepare(catalog, owners, "single-accepted-root-association", 4, [](auto key) { return key.source == 0 ? key.ordinal : key.ordinal == 0 ? 0 : 3; });
    auto seed = prepareCanonicalAdmissionSeed(oldIndex, next, scene, poses, 3, 64u << 20, 2);
    scene.beginLayoutCandidate(CanonicalScene::bindingFor(catalog, next.identity()), poses, [&](const auto& emit) { next.emitAliases(emit); }, seed.producer);
    const auto preservedXYZ = scene.pointXYZ(0);
    check(std::memcmp(&preservedXYZ, &oldXYZ, sizeof(Vec3)) == 0 && seed.evidence->oldRoots == 2 && seed.evidence->newRoots == 1 &&
        seed.evidence->oldPreservedRoots == 2 && seed.evidence->oldRefinedRoots == 0,
        "incoming bad geometry or observation blindly reseeded accepted old root");
    check(seed.evidence->oldBefore.observations == 6 && seed.evidence->oldAfter.observations == 6 && seed.evidence->newBefore.observations == 3 &&
        seed.evidence->incomingOldRootRaw == 4 && seed.evidence->newAfter.cost < seed.evidence->newBefore.cost,
        "admission seed dropped duplicate old/new raw or skipped new complete support refinement");
    const auto full = sceneQuality(next, scene, 3, 32u << 20, 2);
    check(full.observations == 13 && full.invalid_projections == 0 && std::abs(full.finite_cost - 15) < 1e-8,
        "ignored-at-seed incoming association raw vanished from complete objective or pixel units changed");
    scene.abort("admission-seed-geometry-proof"); check(scene.identity() == head, "seed proof changed accepted head");
    auto changedPose = poses; changedPose.images.at(0).pose.t.x += 1e-12;
    auto poseSeed = prepareCanonicalAdmissionSeed(oldIndex, next, scene, changedPose, 3, 64u << 20, 2);
    scene.beginLayoutCandidate(CanonicalScene::bindingFor(catalog, next.identity()), changedPose, [&](const auto& emit) { next.emitAliases(emit); }, poseSeed.producer);
    check(poseSeed.evidence->oldRefinedRoots == 2 && poseSeed.evidence->oldPreservedRoots == 0 && poseSeed.evidence->oldBefore.observations == 6 &&
        canonicalPrefixAccepted(poseSeed.evidence->oldBefore, poseSeed.evidence->oldAfter), "a real tiny pose change was hidden by an epsilon or lost old support");
    scene.abort("changed-pose-old-support-proof");
    auto changedK = poses; changedK.cameras.at(0).fx *= 1.001;
    auto kSeed = prepareCanonicalAdmissionSeed(oldIndex, next, scene, changedK, 3, 64u << 20, 2);
    scene.beginLayoutCandidate(CanonicalScene::bindingFor(catalog, next.identity()), changedK, [&](const auto& emit) { next.emitAliases(emit); }, kSeed.producer);
    check(kSeed.evidence->oldRefinedRoots == 2 && kSeed.evidence->oldPreservedRoots == 0 && kSeed.evidence->oldBefore.observations == 6 &&
        kSeed.evidence->oldAfter.cost < kSeed.evidence->oldBefore.cost && canonicalPrefixAccepted(kSeed.evidence->oldBefore, kSeed.evidence->oldAfter),
        "global K change missed old camera views or used incoming geometry as the old target");
    scene.abort("changed-camera-old-support-proof");
    for (int schema = 0; schema < 4; ++schema) {
        auto incompatible = poses;
        if (schema == 0) incompatible.cameras.at(0).pixel_scale = 3;
        else if (schema == 1) ++incompatible.cameras.at(0).width;
        else if (schema == 2) incompatible.cameras.at(0).model = CamModel::OpenCV;
        else ++incompatible.images.at(0).exif_orientation;
        rejects([&] { prepareCanonicalAdmissionSeed(oldIndex, next, scene, incompatible, 3, 64u << 20, 2); },
            "admission silently reinterpreted old measurement units or orientation");
        check(scene.identity() == head && !scene.editing(), "schema rejection altered accepted authority");
    }
    ImmutableSceneTopology merged(root / "uncertified-merge-index", 32u << 20);
    merged.prepare(catalog, owners, "uncertified-two-old-roots", 4, [](auto key) { return key.source == 1 && key.ordinal == 1 ? 3 : 0; });
    auto rejection = std::make_shared<CanonicalAdmissionSeedEvidence>();
    rejects([&] { prepareCanonicalAdmissionSeed(oldIndex, merged, scene, poses, 3, 64u << 20, 2, {}, rejection); }, "multiple accepted roots were fused without a geometry certificate");
    check(rejection->rejectedOldRoot == 1 && rejection->rejectedCandidateRoot == 0 && !rejection->rejection.empty() && scene.identity() == head,
        "uncertified merge rejected without explicit original point identity evidence");
    check(oldHash == spirula::sha256_file(oldPath.u8string()) && newHash == spirula::sha256_file(newPath.u8string()) &&
        manifestHash == spirula::sha256_file((root / "old-index" / oldIndex.identity() / "manifest.bin").u8string()), "seed/fork changed immutable original evidence");
    std::cout << "immutable fork and controlled old-only/new-fullsupport admission seed tests passed\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
