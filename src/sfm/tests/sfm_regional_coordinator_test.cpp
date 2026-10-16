#include <sstream>
#include "sfm/core/Resume.h"
#include "sfm/map/RegionalPlan.h"
#include "sfm/map/RegionalQuality.h"
#include "sfm/map/RegionalCoordinator.h"
#include "sfm/tests/TestMain.h"
#include <iostream>
#include <iomanip>

using namespace sfm;
using Store = regional::RegionalWorkingStore;
static void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
static bool close(double a, double b) { return std::abs(a - b) <= std::max(1., std::abs(a)) * 1e-9; }

static void projectionDependencyTest() {
    Reconstruction before;
    before.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::FullOpenCV);
    before.cameras[1] = Camera::defaultFor(1, 800, 600, 500, CamModel::Pinhole);
    for (uint32_t id = 0; id < 4; ++id) {
        Image image; image.id = id; image.camera_id = id == 3 ? 1 : 0; image.registered = true;
        image.pose = {mat3Identity(), {0, 0, 0}}; before.images[id] = image;
    }
    auto after = before;
    check(regional::changedProjectionImages(before, after).empty(), "unchanged parameters dirtied projection support");
    after.images.at(1).pose.t.x = .001;
    check(regional::changedProjectionImages(before, after) == std::set<uint32_t>{1}, "pose update did not isolate its image");
    after = before; after.cameras.at(0).k6 = 1e-12;
    check(regional::changedProjectionImages(before, after) == std::set<uint32_t>({0, 1, 2}),
          "global distortion update missed images outside the solved block");
    after.images.at(3).pose.R[1] = 1e-12;
    check(regional::changedProjectionImages(before, after) == std::set<uint32_t>({0, 1, 2, 3}),
          "combined camera and pose support is incomplete");
    after = before; after.cameras.at(1).pixel_scale = 2;
    check(regional::changedProjectionImages(before, after) == std::set<uint32_t>{3}, "measurement scale support missed its camera group");
}

static Reconstruction fixture(uint32_t source) {
    Reconstruction out; out.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::Pinhole);
    for (uint32_t image = 0; image < 4; ++image) {
        Image im; im.id = image; im.camera_id = 0; im.name = "photo-" + std::to_string(image); im.registered = true;
        im.pose = {mat3Identity(), {-.6 * image, .13 * (image % 2), .02 * image}};
        im.points2D.resize(100); im.point3D_ids.assign(100, kInvalidPoint3D); out.images.emplace(image, im);
    }
    for (uint32_t point = 0; point < 100; ++point) {
        Vec3 truth{.2 * (point % 10) - .7, .16 * (point / 10) - .5, 5. + .07 * (point % 13)};
        for (auto& image : out.images) image.second.points2D[point] = out.cameras[0].project(mul(image.second.pose.R, truth) + image.second.pose.t);
        if (point >= 60 && ((source == 0 && point >= 80) || (source == 1 && point < 80))) continue;
        Point3D p; p.xyz = truth + Vec3{.025, -.018, .045}; const uint64_t id = uint64_t(source + 1) * 1000 + point;
        for (auto& image : out.images) {
            if (point >= 60 && ((point < 80 && image.first > 1) || (point >= 80 && image.first < 2))) continue;
            p.track.push_back({image.first, point}); image.second.point3D_ids[point] = id;
        }
        if (!source && !point) p.track.push_back({0, 0});
        out.points3D.emplace(id, p);
    }
    out.images.at(1).pose.t.x += .04; out.images.at(3).pose.t.y -= .03;
    out.next_point3D_id = uint64_t(source + 1) * 1000 + 100; return out;
}
static std::string modelHash(const Reconstruction& model, const std::filesystem::path& file) {
    model_store_detail::writeModel(model, file); std::ifstream in(file, std::ios::binary); spirula::Sha256 sha;
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()); sha.update(bytes.data(), bytes.size()); return sha.hex();
}
static std::string fileHash(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary); check(bool(in), "cannot hash test evidence"); spirula::Sha256 sha;
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    check(!in.bad(), "test evidence hash read failed"); sha.update(bytes.data(), bytes.size()); return sha.hex();
}
static regional::Quality reference(Store& store, regional::RegionalTopology& topology, const std::filesystem::path& root);
static void topologyStepTest(const std::filesystem::path& root) {
    namespace fs = std::filesystem; const auto dir = root / "common-step"; fs::create_directory(dir);
    Store store(dir / "work", 128u << 20);
    for (uint32_t source = 0; source < 2; ++source) {
        auto model = fixture(source); const auto path = dir / ("source-" + std::to_string(source) + ".bin");
        model_store_detail::writeModel(model, path); store.importModel(path, model);
    }
    regional::RegionalTopology topology(dir / "topology", 128u << 20); topology.prepare(store, {0, 0, 1, 1});
    std::map<uint64_t, std::pair<Store::PointKey, Vec3>> groups;
    for (size_t source = 0; source < store.sourceCount(); ++source) store.visitSourceParameters(source, [&](Store::PointKey key, Vec3 xyz) {
        if (topology.shared(key)) groups.emplace(topology.canonical(key), std::make_pair(key, topology.xyz(key, xyz)));
    });
    check(groups.size() == topology.groups(), "common step fixture lacks full group coverage");
    const auto accepted = dir / "accepted.bin", candidate = dir / "candidate.bin"; topology.saveParameters(accepted);
    auto write = [&](const fs::path& path, const std::string& identity, bool nan, bool trailing) {
        model_store_detail::Writer out(path); out.scalar<uint64_t>(0x36504f5453474552ull); out.string(identity); out.scalar<uint64_t>(groups.size());
        size_t at = 0; for (const auto& group : groups) {
            auto xyz = group.second.second + Vec3{.2, -.1, .05};
            if (nan && ++at == groups.size()) xyz.z = std::numeric_limits<double>::quiet_NaN();
            out.scalar(xyz);
        }
        if (trailing) out.scalar<uint8_t>(7); out.finish(); resume::store(path.string() + ".sha256", fileHash(path));
    };
    write(candidate, topology.identity(), false, false); topology.interpolateParameters(accepted, candidate, 1);
    for (const auto& group : groups) {
        const auto expected = group.second.second + Vec3{.2, -.1, .05}; const auto actual = topology.xyz(group.second.first, {});
        check(actual.x == expected.x && actual.y == expected.y && actual.z == expected.z, "common alpha=1 changed candidate bits");
    }
    topology.interpolateParameters(accepted, candidate, .5);
    std::vector<Vec3> unchanged;
    for (const auto& group : groups) {
        const auto actual = topology.xyz(group.second.first, {}); unchanged.push_back(actual);
        check((actual - (group.second.second + Vec3{.1, -.05, .025})).norm() < 1e-12, "common half step inconsistent");
    }
    auto refused = [&](const fs::path& bad) {
        bool threw = false; try { topology.interpolateParameters(accepted, bad, .5); } catch (const std::exception&) { threw = true; }
        check(threw, "corrupted common proposal was accepted"); size_t at = 0;
        for (const auto& group : groups) {
            const auto value = topology.xyz(group.second.first, {}), original = unchanged.at(at++);
            check(value.x == original.x && value.y == original.y && value.z == original.z, "failed common proposal partially changed XYZ");
        }
    };
    const auto badSha = dir / "bad-sha.bin"; write(badSha, topology.identity(), false, false); resume::store(badSha.string() + ".sha256", std::string(64, '0')); refused(badSha);
    const auto badIdentity = dir / "bad-identity.bin"; write(badIdentity, topology.identity() + "-foreign", false, false); refused(badIdentity);
    const auto trailing = dir / "trailing.bin"; write(trailing, topology.identity(), false, true); refused(trailing);
    const auto nonfinite = dir / "nonfinite.bin"; write(nonfinite, topology.identity(), true, false); refused(nonfinite);
    topology.loadParameters(accepted);
    for (const auto& group : groups) check((topology.xyz(group.second.first, {}) - group.second.second).norm() == 0, "common accepted restore changed XYZ");
}
static void halfStepGateTest(const std::filesystem::path& root, bool promotePrivate = false) {
    namespace fs = std::filesystem; const auto dir = root / (promotePrivate ? "promoted-half-step-gate" : "half-step-gate"); fs::create_directory(dir);
    Store store(dir / "work", 128u << 20);
    for (uint32_t source = 0; source < 2; ++source) {
        auto model = fixture(source); const auto path = dir / ("source-" + std::to_string(source) + ".bin");
        model_store_detail::writeModel(model, path); store.importModel(path, model);
    }
    const std::vector<uint32_t> owners{0, 0, 1, 1}; regional::RegionalTopology topology(dir / "topology", 128u << 20); topology.prepare(store, owners);
    uint64_t expectedObservations = 561;
    if (promotePrivate) {
        const auto oldPrivate = store.localKey(0, 1060);
        check(!topology.shared(oldPrivate), "promotion fixture point was already shared");
        store.beginGeneration(); store.setPointXYZ(oldPrivate, store.pointXYZ(oldPrivate) + Vec3{.4, -.15, .1}); store.commitGeneration();
        topology.saveParameters(dir / "pre-append-common.bin");
        auto added = fixture(2);
        for (auto& image : added.images) image.second.point3D_ids.assign(image.second.points2D.size(), kInvalidPoint3D);
        Point3D bridge = added.points3D.at(3060); added.points3D.clear();
        bridge.xyz = bridge.xyz + Vec3{-.3, .2, -.1}; bridge.track.clear();
        for (auto& image : added.images) { bridge.track.push_back({image.first, 60}); image.second.point3D_ids.at(60) = 3060; }
        added.points3D.emplace(3060, std::move(bridge));
        const auto path = dir / "new-bridge.bin"; model_store_detail::writeModel(added, path); store.importModel(path, added); topology.prepare(store, owners);
        check(topology.counters().new_sources == 1 && topology.shared(oldPrivate), "incremental input did not promote the old private point");
        check((store.pointXYZ(oldPrivate) - topology.xyz(oldPrivate, {})).norm() > .01, "promotion fixture failed to expose distinct accepted source/common coordinates");
        const auto addedKey = store.localKey(2, 3060);
        check(topology.canonical(oldPrivate) == topology.canonical(addedKey) &&
              (store.pointXYZ(oldPrivate) - store.pointXYZ(addedKey)).norm() > .1, "promoted aliases did not retain different prior XYZ");
        expectedObservations += 4;
    }
    SfmConfig cfg; cfg.threads = 2; cfg.regional_iterations = 1; cfg.max_error = 3; cfg.mapper.ba_real = "cpu";
    regional::RegionalPoseAnchors held{{0, store.poses().images.at(0).pose}, {2, store.poses().images.at(2).pose}};
    std::vector<regional::Region> regions(2); regions[0].core = {0, 1}; regions[1].core = {2, 3};
    const auto initial = regional::workingQuality(store, topology, 3, 2); const auto generation = store.generation();
    regional::RegionalCoordinator coordinator(store, topology, cfg, nullptr, held, 128u << 20, dir / "parameters");
    int calls = 0; Reconstruction fullProposal;
    const auto result = coordinator.run(regions, [&](const Reconstruction& proposal, const Reconstruction& before) {
        ++calls;
        if (calls == 1) { fullProposal = proposal; return std::vector<std::string>{"full_proposal_source_gate_test_reject"}; }
        check(calls == 2, "first half trial failed to provide a genuine accepted descent");
        const auto visual = regional::workingQuality(store, topology, 3, 2);
        check(visual.observations == expectedObservations && visual.invalid_projections <= initial.invalid_projections &&
              visual.finite_cost < initial.finite_cost && visual.cost < initial.cost, "half trial bypassed original visual descent");
        const auto physical = reference(store, topology, dir);
        check(physical.observations == visual.observations && physical.invalid_projections == visual.invalid_projections &&
              close(physical.finite_cost, visual.finite_cost) && close(physical.cost, visual.cost), "gate inspected virtual common XYZ that differ from the physical ModelStore Reader");
        for (uint32_t image : {1u, 3u}) {
            const Vec3 midpoint = (cameraCenter(before.images.at(image).pose) + cameraCenter(fullProposal.images.at(image).pose)) * .5;
            check((cameraCenter(proposal.images.at(image).pose) - midpoint).norm() < 1e-10, "gate did not inspect a uniform half-step pose");
        }
        return std::vector<std::string>{};
    });
    check(calls == 2 && result.rounds == 1 && result.rejected.empty() && store.generation() == generation + 1, "half-step acceptance did not commit exactly one generation");
    const auto prefix = dir / "parameters" / ("generation-" + std::to_string(generation + 1));
    check(resume::recorded(prefix.string() + "-accepted-step.txt").find("0.500000") == 0, "accepted half-step evidence missing");
    for (const char* suffix : {"-full-step-projection.bin", "-candidate-common.bin", "-step-1-projection.bin", "-step-1-poses.bin", "-step-1-common.bin"})
        check(fs::exists(prefix.string() + suffix), "complete full/trial candidate evidence missing");
    size_t retained = 0; for (const auto& file : fs::directory_iterator(dir / "work")) if (file.path().filename().string().find("rejected-parameters-") == 0) {
        ++retained; check(fs::exists(file.path() / "poses.bin"), "full proposal poses not retained");
        const auto saved = model_store_detail::readModel(file.path() / "poses.bin");
        check(modelHash(saved, dir / "retained-pose-check.bin") == modelHash(fullProposal, dir / "original-proposal-check.bin"), "accepted half step overwrote the complete original proposal");
        for (size_t source = 0; source < store.sourceCount(); ++source) check(fs::exists(file.path() / ("points-" + std::to_string(source) + ".xyz")), "full proposal private XYZ not retained");
    }
    check(retained == 1, "successful half-step should retain exactly the original full proposal");
    Store restored(dir / "work", 128u << 20); regional::RegionalTopology restoredTopology(dir / "topology", 128u << 20); restoredTopology.prepare(restored, owners);
    restoredTopology.loadParameters(dir / "parameters" / ("common-" + std::to_string(restored.generation()) + ".bin"));
    const auto final = regional::workingQuality(restored, restoredTopology, 3, 2), originalReader = reference(restored, restoredTopology, dir);
    check(final.observations == expectedObservations && final.invalid_projections == originalReader.invalid_projections &&
          close(final.finite_cost, originalReader.finite_cost) && close(final.cost, result.quality.cost) && close(final.cost, originalReader.cost), "half-step checkpoint restore/original raw reader differs");
    for (size_t source = 0; source < restored.sourceCount(); ++source) restored.visitSourceParameters(source, [&](Store::PointKey key, Vec3 xyz) {
        const auto canonical = restoredTopology.xyz(key, xyz);
        check(xyz.x == canonical.x && xyz.y == canonical.y && xyz.z == canonical.z, "committed source XYZ differs from authoritative common parameters");
    });
}
static void privateWindowTest(const std::filesystem::path& root) {
    Reconstruction model; model.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::Pinhole);
    for (uint32_t image = 0; image < 64; ++image) {
        Image im; im.id = image; im.camera_id = 0; im.registered = true; im.name = "window-photo-" + std::to_string(image);
        im.pose = {mat3Identity(), {-.025 * image, .003 * (image % 3), 0}}; im.points2D.resize(16384); im.point3D_ids.assign(16384, kInvalidPoint3D); model.images.emplace(image, std::move(im));
    }
    for (uint32_t point = 0; point < 80; ++point) {
        Vec3 truth{.16 * (point % 10) - .7, .12 * (point / 10) - .4, 5. + .03 * (point % 13)};
        Point3D item; item.xyz = truth + Vec3{.025, -.018, .045};
        for (auto& image : model.images) { image.second.points2D[point] = model.cameras.at(0).project(truth + image.second.pose.t); image.second.point3D_ids[point] = point + 1000; item.track.push_back({image.first, point}); }
        model.points3D.emplace(point + 1000, std::move(item));
    }
    const auto source = root / "private-window-source.bin"; model_store_detail::writeModel(model, source);
    Store store(root / "private-window-work", 128u << 20); store.importModel(source, model);
    regional::RegionalTopology topology(root / "private-window-topology", 128u << 20); topology.prepare(store, std::vector<uint32_t>(64, 0));
    check(topology.groups() == 0, "private single-owner landmarks were treated as common points");
    auto cut = store.gather({0, 1}, {}, false); check(cut.boundaryPoints.size() == 80, "private window interfaces were not identified");
    SfmConfig cfg; cfg.threads = 2; cfg.regional_iterations = 1; cfg.max_error = 3; cfg.mapper.ba_real = "double";
    regional::RegionalPoseAnchors held{{0, store.poses().images.at(0).pose}, {63, store.poses().images.at(63).pose}};
    regional::Region region; for (uint32_t image = 0; image < 64; ++image) region.core.push_back(image);
    regional::RegionalCoordinator controller(store, topology, cfg, nullptr, held, 64u << 20, root / "private-window-parameters");
    auto result = controller.run({region}, [](const Reconstruction&, const Reconstruction&) { return std::vector<std::string>{}; });
    check(result.rounds == 1 && result.quality.observations == 5120, "private interface coordination failed raw acceptance");
    check((store.pointXYZ(store.localKey(0, 1000)) - model.points3D.at(1000).xyz).norm() > 1e-6, "budget-split private landmark stayed frozen forever");
    uint64_t raw = 0; store.visitObservations([&](const Store::RawObservation&) { ++raw; }); check(raw == 5120, "private interface triangulation changed original observations");
}
static void forcedPhotoTaskTest(const std::filesystem::path& root) {
    for (size_t limit : {size_t(1), size_t(2)}) {
        const auto dir = root / ("forced-photo-" + std::to_string(limit)); std::filesystem::create_directory(dir);
        Store store(dir / "work", 128u << 20);
        for (uint32_t source = 0; source < 2; ++source) {
            auto model = fixture(source); const auto path = dir / ("source-" + std::to_string(source) + ".bin"); model_store_detail::writeModel(model, path); store.importModel(path, model);
        }
        const std::vector<uint32_t> owners{0, 0, 1, 1}; regional::RegionalTopology topology(dir / "topology", 128u << 20); topology.prepare(store, owners);
        SfmConfig cfg; cfg.threads = 2; cfg.regional_iterations = 1; cfg.max_error = 3; cfg.mapper.ba_real = "double";
        regional::RegionalPoseAnchors held{{0, store.poses().images.at(0).pose}, {2, store.poses().images.at(2).pose}};
        std::vector<regional::Region> regions(2); regions[0].core = {0, 1}; regions[1].core = {2, 3};
        const auto initial = regional::workingQuality(store, topology, 3, 2); const auto key = store.localKey(0, 1060); const auto xyz = store.pointXYZ(key);
        const auto movable = store.gather({1, 3}, {}, false);
        auto fixedPointCost = [&](uint32_t id, const Reconstruction& poses) {
            const auto& image = poses.images.at(id); const auto& camera = poses.cameras.at(image.camera_id); double cost = 0;
            for (const auto& point : movable.model.points3D) for (const auto& row : point.second.track) if (row.image_id == id) {
                const auto predicted = camera.project(mul(image.pose.R, point.second.xyz) + image.pose.t);
                const auto observed = movable.model.images.at(id).points2D.at(row.point2D_idx);
                const double dx = predicted.x - observed.x, dy = predicted.y - observed.y; cost += dx * dx + dy * dy;
            }
            return cost;
        };
        const auto beforePoses = store.poses(); const double fixedBefore1 = fixedPointCost(1, beforePoses), fixedBefore3 = fixedPointCost(3, beforePoses);
        regional::RegionalCoordinator controller(store, topology, cfg, nullptr, held, 128u << 20, dir / "parameters", limit);
        auto result = controller.run(regions, [](const Reconstruction&, const Reconstruction&) { return std::vector<std::string>{}; });
        check(result.rounds == 1 && result.quality.observations == 561, "forced photo limit failed full raw acceptance");
        check(result.quality.cost < initial.cost, "forced photo limit failed to improve original objective");
        if (limit == 1) {
            check((store.pointXYZ(key) - xyz).norm() > 1e-6, "single-photo tasks never refined private boundary point");
            for (uint32_t image : {1u, 3u}) check((store.poses().images.at(image).pose.t - beforePoses.images.at(image).pose.t).norm() > 1e-6,
                "single-photo task reported success without refining its movable camera pose");
            check(fixedPointCost(1, store.poses()) < fixedBefore1 && fixedPointCost(3, store.poses()) < fixedBefore3,
                "single-photo camera refinement did not improve visual residuals against unchanged XYZ");
        }
        Store restored(dir / "work", 128u << 20); regional::RegionalTopology restoredTopology(dir / "topology", 128u << 20); restoredTopology.prepare(restored, owners);
        auto generation = regional::acceptedCommonGeneration(restored); check(bool(generation), "forced-photo checkpoint phase missing");
        restoredTopology.restoreAcceptedParameters(dir / "parameters" / ("common-" + std::to_string(*generation) + ".bin"));
        auto quality = regional::workingQuality(restored, restoredTopology, 3, 2);
        check(quality.observations == 561 && close(quality.cost, result.quality.cost), "forced-photo checkpoint restore changed raw quality");
    }
}
static regional::Quality reference(Store& store, regional::RegionalTopology& topology, const std::filesystem::path& root) {
    (void)topology;
    regional::Quality result; result.finite_cost = 0;
    for (size_t source = 0; source < store.sourceCount(); ++source) {
        auto model = store.materialize(source);
        const auto physical = root / ("materialized-reader-" + std::to_string(source) + ".bin");
        model_store_detail::writeModel(model, physical); model = model_store_detail::readModel(physical);
        const auto path = root / ("validation-" + std::to_string(source) + ".bin"); regional::writeValidation(path, model, source);
        regional::accumulateQuality(result, regional::validationCost(path, model, store.poses(), source, 3, 2));
    }
    return result;
}
class CachedGps : public PriorSource {
public:
    std::map<uint32_t, Vec3> positions;
    explicit CachedGps(const std::filesystem::path& file) {
        std::ifstream input(file); check(bool(input), "cannot read cached native ENU input"); std::string line;
        check(bool(std::getline(input, line)), "empty cached ENU input"); if (!line.empty() && line.back() == '\r') line.pop_back();
        const bool csv = line == "image_id,x,y,z";
        uint32_t sequential = 0;
        if (!csv) {
            std::istringstream row(line); Vec3 p; std::string trailing;
            check(bool(row >> p.x >> p.y >> p.z) && !(row >> trailing) && std::isfinite(p.norm()), "cached ENU input is neither CSV nor sequential XYZ");
            positions.emplace(sequential++, p);
        }
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back(); if (line.empty()) continue;
            std::replace(line.begin(), line.end(), ',', ' '); std::istringstream row(line); uint32_t image; Vec3 p; std::string trailing;
            image = sequential;
            if (csv) check(bool(row >> image), "invalid cached ENU image ID"); else ++sequential;
            check(bool(row >> p.x >> p.y >> p.z) && !(row >> trailing) && std::isfinite(p.norm()), "invalid cached ENU record");
            check(positions.emplace(image, p).second, "duplicate cached ENU image ID");
        }
        check(input.eof(), "cached ENU read failed");
    }
    bool has(uint32_t id) const override { return positions.count(id) != 0; }
    bool position(uint32_t id, Vec3& p) const override { auto it = positions.find(id); if (it == positions.end()) return false; p = it->second; return true; }
    bool relativeRotation(uint32_t, uint32_t, Mat3&, double&) const override { return false; }
    std::vector<uint32_t> neighbours(uint32_t) const override { return {}; }
    PosePriors factors(const std::vector<PosedImage>&) override { return {}; }
};
static int real(const std::filesystem::path& sourceA, const std::filesystem::path& sourceB,
    const std::filesystem::path& consensusPath, const std::filesystem::path& gpsPath, const std::filesystem::path& root) {
    namespace fs = std::filesystem; fs::create_directories(root); CachedGps gps(gpsPath);
    const size_t budget = size_t(2) << 30; Store store(root / "work", budget / 2);
    for (const auto& source : {sourceA, sourceB}) if (store.knownSource(source) == SIZE_MAX) {
        auto model = model_store_detail::readModel(source); store.importModel(source, model);
    }
    const auto consensus = model_store_detail::readModel(consensusPath, false);
    if (!fs::exists(root / "seed.done")) {
        store.beginGeneration();
        try {
            for (const auto& image : store.poses().images) {
                auto found = consensus.images.find(image.first);
                check(found != consensus.images.end() && found->second.registered && found->second.camera_id == image.second.camera_id, "real source image missing from accepted common ENU consensus");
                store.setPose(image.first, found->second.pose);
            }
            for (const auto& camera : store.poses().cameras) store.setCamera(consensus.cameras.at(camera.first)); store.commitGeneration();
            resume::store(root / "seed.done", std::to_string(store.generation()));
        } catch (...) { store.abortGeneration(); throw; }
    }
    std::vector<std::set<uint32_t>> sourceImages(2); uint32_t last = 0;
    for (size_t source = 0; source < 2; ++source) for (const auto& image : store.sourceViews(source).images) if (image.second.registered) {
        sourceImages[source].insert(image.first); last = std::max(last, image.first);
    }
    check(last < 10000000, "real input image IDs exceed bounded catalogue"); std::vector<uint32_t> owners(size_t(last) + 1, UINT32_MAX);
    std::vector<regional::Region> regions(2); size_t shared = 0;
    for (const auto& image : store.poses().images) if (image.second.registered) {
        uint32_t owner = sourceImages[0].count(image.first) ? 0 : 1;
        if (sourceImages[0].count(image.first) && sourceImages[1].count(image.first)) owner = uint32_t(shared++ % 2);
        owners.at(image.first) = owner; regions[owner].core.push_back(image.first); Vec3 p; check(gps.position(image.first, p), "actual cached ENU missing selected source photo");
    }
    check(!regions[0].core.empty() && !regions[1].core.empty(), "real sources do not provide two observable blocks");
    regional::RegionalTopology topology(root / "topology", budget); topology.prepare(store, owners);
    SfmConfig cfg; cfg.threads = 2; cfg.regional_iterations = 1; cfg.max_error = 3; cfg.mapper.ba_real = "double";
    regional::RegionalPoseAnchors held;
    const auto initial = regional::workingQuality(store, topology, 3, cfg.threads), baseline = reference(store, topology, root);
    check(initial.observations == baseline.observations && close(initial.cost, baseline.cost), "real flat and original source ledger cost differ before BA");
    const auto initialGps = regional::regionalGpsQuality(store.poses(), gps); regional::RegionalCoordinator controller(store, topology, cfg, &gps, held, budget, root / "parameters");
    auto outcome = controller.run(regions, [&](const Reconstruction& next, const Reconstruction& previous) {
        const auto candidate = regional::regionalGpsQuality(next, gps), accepted = regional::regionalGpsQuality(previous, gps);
        std::vector<std::string> reasons;
        if (candidate.inliers < accepted.inliers) reasons.push_back("actual_gps_inlier_count_decreased");
        if (candidate.cost > accepted.cost * (1 + 1e-8) + 1e-8) reasons.push_back("actual_fixed_enu_gps_cost_increased");
        return reasons;
    });
    const auto final = regional::workingQuality(store, topology, 3, cfg.threads), full = reference(store, topology, root);
    check(final.observations == initial.observations && full.observations == initial.observations && close(final.cost, full.cost), "real original observation ledger changed after native BA");
    store.verifyImmutable(); const auto gpsAfter = regional::regionalGpsQuality(store.poses(), gps);
    Store restored(root / "work", budget / 2); regional::RegionalTopology restoredTopology(root / "topology", budget); restoredTopology.prepare(restored, owners);
    const auto parameter = root / "parameters" / ("common-" + std::to_string(restored.generation()) + ".bin");
    if (fs::exists(parameter)) restoredTopology.loadParameters(parameter);
    const auto restoreQuality = regional::workingQuality(restored, restoredTopology, 3, cfg.threads);
    check(restoreQuality.observations == final.observations && close(restoreQuality.cost, final.cost), "real committed/rejected checkpoint restore changed quality");
    std::ofstream out(root / "real-probe.txt"); out << std::setprecision(17)
        << "source_models=2\nshared_points=" << topology.groups() << "\noriginal_observations=" << initial.observations
        << "\ninitial_cost=" << initial.cost << "\nfinal_cost=" << final.cost << "\ninvalid_before=" << initial.invalid_projections
        << "\ninvalid_after=" << final.invalid_projections << "\naccepted_rounds=" << outcome.rounds
        << "\ngps_before=" << initialGps.inliers << '/' << initialGps.photos << "\ngps_after=" << gpsAfter.inliers << '/' << gpsAfter.photos
        << "\ngps_cost_before=" << initialGps.cost << "\ngps_cost_after=" << gpsAfter.cost << "\nrestore_cost=" << restoreQuality.cost << '\n';
    for (const auto& reason : outcome.rejected) out << "rejected=" << reason << '\n'; out.flush(); check(bool(out), "cannot save native real probe evidence");
    std::cout << "real native coordinator probe preserved complete original observations; accepted rounds " << outcome.rounds << '\n'; return 0;
}
static int test(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--feature-diagnostic") {
        Store store(argv[2], size_t(1) << 30); auto views = store.sourceViews(0); auto other = store.sourceViews(1);
        std::ofstream out(argv[3]); out << std::setprecision(17) << "image_id,local_feature,count_a,count_b,x_a,y_a,x_b,y_b\n";
        uint64_t photos = 0, differences = 0, observedConflicts = 0, placeholders = 0;
        for (const auto& image : views.images) if (image.second.registered && other.images.count(image.first) && other.images.at(image.first).registered) {
            ++photos; std::vector<Vec2> first; const auto a = store.sourceImageFeatureCount(0, image.first), b = store.sourceImageFeatureCount(1, image.first);
            std::vector<uint8_t> usedA(size_t(a), 0), usedB(size_t(b), 0);
            store.visitImageObservations(image.first, [&](const Store::RawObservation& row) {
                if (row.point.source == 0) usedA.at(row.feature) = 1; if (row.point.source == 1) usedB.at(row.feature) = 1;
            });
            check(a < 10000000, "diagnostic photo features exceed memory limit"); first.reserve(size_t(a));
            store.visitSourceImageFeatures(0, image.first, [&](uint64_t, Vec2 xy, uint64_t) { first.push_back(xy); }); size_t printed = 0;
            store.visitSourceImageFeatures(1, image.first, [&](uint64_t feature, Vec2 xy, uint64_t) {
                if (feature >= first.size() || first[size_t(feature)].x != xy.x || first[size_t(feature)].y != xy.y) {
                    ++differences;
                    if (feature < usedA.size() && usedA[size_t(feature)] && usedB.at(size_t(feature))) ++observedConflicts; else ++placeholders;
                    if (printed++ < 8) out << image.first << ',' << feature << ',' << a << ',' << b << ','
                        << (feature < first.size() ? first[size_t(feature)].x : 0) << ',' << (feature < first.size() ? first[size_t(feature)].y : 0) << ',' << xy.x << ',' << xy.y << '\n';
                }
            });
        }
        out.flush(); check(bool(out), "feature diagnostic write failed"); std::cout << "overlap photos " << photos << ", local feature XY differences " << differences
            << ", both actually observed conflicts " << observedConflicts << ", differences in unobserved placeholder slots " << placeholders << '\n'; return 0;
    }
    if (argc == 7 && std::string(argv[1]) == "--real") return real(argv[2], argv[3], argv[4], argv[5], argv[6]);
    ModelStore scratch(""); const auto root = scratch.directory(); Store store(root / "work", 128u << 20);
    for (uint32_t source = 0; source < 2; ++source) {
        const auto model = fixture(source); const auto path = root / ("source-" + std::to_string(source) + ".bin");
        model_store_detail::writeModel(model, path); store.importModel(path, model);
    }
    regional::RegionalTopology topology(root / "topology", 128u << 20); const std::vector<uint32_t> owners{0, 0, 1, 1}; topology.prepare(store, owners);
    check(topology.groups() == 60, "full shared topology point count changed");
    auto onePhoto = store.gather({0}, {}, false);
    check(!onePhoto.boundaryPoints.empty() && onePhoto.model.images.size() == 1, "single-photo fixed landmark gather missing");
    for (const auto& point : onePhoto.model.points3D) check(!point.second.track.empty(), "single-photo observation dropped");
    BundleOptions single; single.real = RealCfg::F64; single.solver = "cg"; single.max_iters = 4;
    single.refine_intrinsics = false; single.fixed_points = &onePhoto.boundaryPoints; SolverStats singleStats; single.stats = &singleStats;
    auto layout = buildBundle(onePhoto.model, single);
    check(layout.P.num_images == 1 && layout.P.num_points == onePhoto.model.points3D.size() && layout.P.num_obs > 0,
        "single-photo fixed common observations did not enter native BA");
    runGlobalBA(onePhoto.model, single); check(std::isfinite(singleStats.final_cost) && singleStats.final_cost < singleStats.initial_cost,
        "single-photo fixed-point native Vulkan refinement failed");
    auto initial = regional::workingQuality(store, topology, 3, 2); auto expected = reference(store, topology, root);
    check(initial.observations == 561 && initial.observations == expected.observations && close(initial.cost, expected.cost) && close(initial.finite_cost, expected.finite_cost), "flat validation differs from original raw ledger");
    SfmConfig cfg; cfg.threads = 2; cfg.regional_iterations = 1; cfg.max_error = 3; cfg.mapper.ba_real = "double";
    cfg.mapper.refine_principal_point = false; cfg.mapper.refine_extra_params = false;
    regional::RegionalPoseAnchors held; held.emplace(0, store.poses().images.at(0).pose); held.emplace(2, store.poses().images.at(2).pose);
    std::vector<regional::Region> regions(2); regions[0].core = {0, 1}; regions[1].core = {2, 3};
    regional::RegionalCoordinator coordinator(store, topology, cfg, nullptr, held, 128u << 20, root / "coordination");
    const uint64_t beforeGeneration = store.generation(); const auto poseHash = modelHash(store.poses(), root / "before-poses.bin");
    std::vector<std::string> sourceHashes;
    for (size_t source = 0; source < store.sourceCount(); ++source) sourceHashes.push_back(modelHash(store.materialize(source), root / ("before-" + std::to_string(source) + ".bin")));
    topology.saveParameters(root / "common-before-rejection.bin");
    const auto commonHash = fileHash(root / "common-before-rejection.bin"), manifestHash = fileHash(root / "work" / "manifest.bin");
    int calls = 0;
    auto rejected = coordinator.run(regions, [&](const Reconstruction&, const Reconstruction&) { ++calls; return std::vector<std::string>{"gps_source_gate_test_reject"}; });
    check(calls == 13 && rejected.rounds == 0 && !rejected.converged && !rejected.rejected.empty() && store.generation() == beforeGeneration, "controlled source gate did not veto every trial and roll back round");
    check(std::find(rejected.rejected.begin(), rejected.rejected.end(), "complete_objective_backtracking_exhausted") != rejected.rejected.end(), "exhausted trials did not report rejection");
    check(fileHash(root / "work" / "manifest.bin") == manifestHash, "all-gates failure changed accepted manifest");
    topology.saveParameters(root / "common-after-rejection.bin");
    check(fileHash(root / "common-after-rejection.bin") == commonHash, "all-gates failure changed accepted common XYZ");
    size_t preserved = 0;
    for (const auto& entry : std::filesystem::directory_iterator(root / "work")) if (entry.path().filename().string().find("rejected-parameters-") == 0) {
        ++preserved; check(std::filesystem::exists(entry.path() / "poses.bin"), "rejected full/trial pose evidence missing");
        for (size_t source = 0; source < 2; ++source) check(std::filesystem::exists(entry.path() / ("points-" + std::to_string(source) + ".xyz")), "rejected full/trial XYZ evidence missing");
    }
    check(preserved == 13, "all-gates rejection failed to preserve the full proposal and twelve trials");
    check(modelHash(store.poses(), root / "after-poses.bin") == poseHash, "rejected round changed canonical poses or intrinsics");
    for (size_t source = 0; source < store.sourceCount(); ++source) check(modelHash(store.materialize(source), root / ("after-" + std::to_string(source) + ".bin")) == sourceHashes[source], "rejected round changed XYZ or original tracks");
    Store reopened(root / "work", 128u << 20); regional::RegionalTopology restored(root / "topology", 128u << 20); restored.prepare(reopened, owners);
    check(reopened.generation() == beforeGeneration, "rejected round advanced durable head");
    regional::RegionalCoordinator repeated(reopened, restored, cfg, nullptr, held, 128u << 20, root / "coordination");
    auto replay = repeated.run(regions, [](const Reconstruction&, const Reconstruction&) { return std::vector<std::string>{"gps_source_gate_test_reject"}; });
    check(replay.rounds == 0 && replay.rejected == rejected.rejected, "checkpoint restore replay changed rejection");
    regional::RegionalCoordinator accepted(reopened, restored, cfg, nullptr, held, 128u << 20, root / "accepted");
    auto result = accepted.run(regions, [](const Reconstruction&, const Reconstruction&) { return std::vector<std::string>{}; });
    check(result.rounds == 1 && result.quality.observations == 561 && result.quality.cost <= initial.cost, "native block coordination did not improve complete raw objective");
    Store finalStore(root / "work", 128u << 20); regional::RegionalTopology finalTopology(root / "topology", 128u << 20); finalTopology.prepare(finalStore, owners);
    finalTopology.loadParameters(root / "accepted" / ("common-" + std::to_string(finalStore.generation()) + ".bin"));
    auto final = regional::workingQuality(finalStore, finalTopology, 3, 2); auto full = reference(finalStore, finalTopology, root);
    check(final.observations == 561 && close(final.cost, full.cost) && close(final.cost, result.quality.cost), "accepted checkpoint restore changed original quality");
    projectionDependencyTest();
    topologyStepTest(root);
    halfStepGateTest(root);
    halfStepGateTest(root, true);
    privateWindowTest(root);
    forcedPhotoTaskTest(root);
    std::cout << "regional coordinator tests passed: 561 original observations, native BA, gate rollback, restore\n"; return 0;
}
int main(int argc, char** argv) { return sfmTestMain(argc, argv, test); }
