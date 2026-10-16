#include "sfm/map/ExportVerifiedScene.h"
#include <iostream>
#include <thread>
using namespace sfm;
using namespace sfm::regional;
using Path = std::filesystem::path;
static void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class Action> static void rejects(Action action, const char* why) { bool rejected = false; try { action(); } catch (const std::exception&) { rejected = true; } require(rejected, why); }
static std::string sha(const Path& path) { auto value = spirula::sha256_file(path.u8string()); require(!value.empty(), "fixture SHA failed"); return value; }
static bool same(const Quality& a, const Quality& b) { return a.observations == b.observations && a.retained_observations == b.retained_observations && a.directional_observations == b.directional_observations && a.invalid_projections == b.invalid_projections && a.cost == b.cost && a.finite_cost == b.finite_cost; }
static bool same(const PhotoQuality& a, const PhotoQuality& b) { return a.observations == b.observations && a.retained_observations == b.retained_observations && a.inliers == b.inliers && a.invalid_projections == b.invalid_projections && a.error == b.error; }
static Vec3 direction() { return Vec3{.2, .05, 1}.normalized(); }
static Reconstruction fixture() {
    Reconstruction rec; rec.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::OpenCV); rec.cameras[0].pixel_scale = 2;
    for (uint32_t id = 0; id < 2; ++id) { Image image; image.id = id; image.camera_id = 0; image.name = "photo-" + std::to_string(id) + ".jpg"; image.registered = true;
        image.pose = {mat3Identity(), {-.5 * id, 0, 0}}; image.points2D.resize(14); image.point3D_ids.assign(14, kInvalidPoint3D); rec.images.emplace(id, std::move(image)); }
    for (uint32_t id = 0; id < 14; ++id) {
        Point3D point; point.xyz = id < 12 ? Vec3{.02 * id, .04 * (id % 4), 5 + .03 * id} : id == 12 ? direction() : Vec3{999, -888, -3};
        point.rgb[0] = 30; point.rgb[1] = 60; point.rgb[2] = 90;
        for (auto& item : rec.images) { const auto xy = id == 13 ? Vec2{10, 20} : rec.cameras.at(0).project(id == 12 ? direction() : point.xyz + item.second.pose.t); item.second.points2D[id] = {double(float(xy.x)), double(float(xy.y))}; }
        if (id < 13) for (uint32_t repeat = 0; repeat < 120; ++repeat) { point.track.push_back({0, id}); point.track.push_back({1, id}); }
        else for (uint32_t repeat = 0; repeat < 300; ++repeat) point.track.push_back({0, id});
        for (const auto& row : point.track) rec.images.at(row.image_id).point3D_ids[row.point2D_idx] = id;
        rec.points3D.emplace(id, std::move(point));
    }
    rec.next_point3D_id = 14; return rec;
}
int main(int argc, char** argv) try {
    const Path root = argc > 1 ? std::filesystem::u8path(argv[1]) : std::filesystem::temp_directory_path() / ("formal-reader-parallel-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(!std::filesystem::exists(root), "fixture requires new owned output"); std::filesystem::create_directories(root / "features"); std::filesystem::create_directory(root / "inputs");
    const auto original = fixture(); const auto source = root / "source.bin"; model_store_detail::writeModel(original, source); const auto sourceHash = sha(source);
    std::vector<std::string> featureHashes;
    for (const auto& item : original.images) {
        FeatureSet features; features.width = 800; features.height = 600; features.extract_width = 400; features.extract_height = 300; features.keypoints.resize(14); features.descriptors.resize(14 * features.dim);
        for (size_t i = 0; i < 14; ++i) { features.keypoints[i].x = float(item.second.points2D[i].x); features.keypoints[i].y = float(item.second.points2D[i].y); }
        const auto path = root / "features" / (item.second.name + ".bin"); writeFeatures(path.u8string(), features); featureHashes.push_back(sha(path));
        std::ofstream image(root / "inputs" / item.second.name, std::ios::binary); image << "synthetic native fixture; no real-photo acceptance claim";
    }
    RegionalWorkingStore catalog(root / "catalog", 64u << 20); catalog.importModel(source, original, "input"); CanonicalScene scene(root / "scene", 64u << 20);
    auto poses = catalog.poses(); poses.points3D.clear(); for (auto& item : poses.images) { item.second.points2D.clear(); item.second.point3D_ids.clear(); }
    scene.beginLayoutCandidate(CanonicalScene::bindingFor(catalog, "formal-reader-parallel-fixture"), poses,
        [&](const auto& emit) { for (uint64_t id = 0; id < 14; ++id) emit(catalog.key(0, id), id); },
        [&](const auto& emit) { for (const auto& point : original.points3D) emit(point.first, point.second.xyz); });
    scene.applyLandmarkTypes({{12, direction()}}); scene.applyRetainedObservations({13}); scene.commit("typed-reader-fixture");
    const auto headHash = sha(root / "scene/head.bin"), pointsHash = sha(scene.acceptedSnapshot().directory / "points.bin");
    VerifiedExportOptions options; options.cache_bytes = 8u << 20; options.reader_memory_bytes = 8u << 20; options.accepted_raw.finite_cost = 0;
    CanonicalParameterView parameters(scene, 1u << 20);
    catalog.visitObservations([&](const auto& row) { const auto point = parameters.landmark(parameters.canonical(row.point)); const auto& image = scene.poses().images.at(row.image); PhotoQuality photo;
        verified_export_detail::projection(options.accepted_raw, photo, scene.poses().cameras.at(image.camera_id), image, row.xy, point.value, 3, point.kind); });
    for (const auto& image : original.images) options.inputs[image.first] = {(root / "inputs" / image.second.name).u8string(), image.second.name};
    auto featurePath = [&](uint32_t image) { return root / "features" / (original.images.at(image).name + ".bin"); };
    ExportFinalGate gate = [&](const Reconstruction& actual, const Quality&, const auto&, const ExportStats&) {
        ExportFinalAcceptance proof; proof.historical_evaluated = true;
        const double originalDistance = (cameraCenter(original.images.at(0).pose) - cameraCenter(original.images.at(1).pose)).norm();
        const double actualDistance = (cameraCenter(actual.images.at(0).pose) - cameraCenter(actual.images.at(1).pose)).norm();
        for (const auto& item : actual.images) { ExportConstraintPhoto row; row.image = item.first; row.path = options.inputs.at(item.first).path;
            row.gps_error_metres = (cameraCenter(item.second.pose) - cameraCenter(original.images.at(item.first).pose)).norm(); row.gps_verified = row.gps_error_metres <= 5;
            row.scale_error_fraction = std::abs(actualDistance / originalDistance - 1); row.scale_verified = row.scale_error_fraction <= .01;
            double trace = 0; for (size_t k = 0; k < 9; ++k) trace += item.second.pose.R[k] * original.images.at(item.first).pose.R[k];
            row.attitude_error_degrees = std::acos(std::clamp((trace - 1) * .5, -1., 1.)) * 180 / M_PI; row.attitude_verified = row.attitude_error_degrees <= 10; proof.photos.push_back(row); }
        return proof;
    };
    options.threads = 1; const auto serial = exportVerifiedScene(catalog, scene, featurePath, root / "serial-stage", root / "serial", options, gate);
    const auto producer = std::this_thread::get_id(); uint64_t loaded = 0, rawDone = 0;
    options.threads = 0; options.reportphase = [&](const char* phase, uint64_t done, uint64_t total) { require(std::this_thread::get_id() == producer && done <= total, "formal progress not producer-owned/bounded");
        if (std::string(phase) == "native-reader-load") { require(done >= loaded, "native load progress unordered"); loaded = done; }
        if (std::string(phase) == "native-reader-raw") { require(done >= rawDone, "native raw progress unordered"); rawDone = done; } };
    const auto parallel = exportVerifiedScene(catalog, scene, featurePath, root / "parallel-stage", root / "parallel", options, gate);
    for (const auto& row : parallel.failures) std::cerr << row.reason << '\n';
    require(serial.reader_verified && parallel.reader_verified && serial.verified && parallel.verified && same(serial.raw, parallel.raw), "threads1/auto formal Reader complete objective differs");
    require(parallel.raw.observations == 3420 && parallel.finite_raw == 2880 && parallel.directional_raw == 240 && parallel.retained_raw == 300 && loaded == 3 && rawDone == 3420, "formal raw multiset/type/progress coverage changed");
    require(parallel.reader_parameter_cache_bytes && parallel.reader_raw_batches && serial.raw_photos.size() == parallel.raw_photos.size(), "formal Reader flat cache was not used or row batches absent");
    for (const auto& item : serial.raw_photos) require(same(item.second, parallel.raw_photos.at(item.first)), "formal per-photo reduction not deterministic");
    auto actual = verified_export_detail::strictReadBack(root / "parallel", options.reader_memory_bytes); verified_export_detail::sameParameters(scene.poses(), actual, options);
    const auto dirs = verified_export_detail::strictReadDirections(root / "parallel", actual, 1u << 20);
    const auto retained = verified_export_detail::strictReadRetained(root / "parallel", actual, dirs, 1u << 20);
    auto helperOptions = options; helperOptions.reportphase = {}; VerifiedExportReport fallback;
    verified_export_detail::verifyOriginalRawLedger(catalog, scene, root / "parallel", actual, dirs, retained, helperOptions, 200u << 10, fallback);
    require(!fallback.reader_parameter_cache_bytes && fallback.raw.observations == parallel.raw.observations && fallback.finite_raw == 2880 && fallback.directional_raw == 240 && fallback.retained_raw == 300 &&
        std::abs(fallback.raw.cost - parallel.raw.cost) < std::max(1e-8, parallel.raw.cost * 1e-8), "bounded fallback lost original raw or failed a small admitted fixture");
    std::filesystem::create_directory(root / "bad-alias"); std::filesystem::copy_file(root / "parallel/raw-alias-map.bin", root / "bad-alias/raw-alias-map.bin");
    { std::fstream file(root / "bad-alias/raw-alias-map.bin", std::ios::binary | std::ios::in | std::ios::out); file.seekp(24); const uint64_t wrong = 2; file.write(reinterpret_cast<const char*>(&wrong), sizeof(wrong)); }
    rejects([&] { VerifiedExportReport report; verified_export_detail::verifyOriginalRawLedger(catalog, scene, root / "bad-alias", actual, dirs, retained, helperOptions, 1u << 20, report); }, "wrong native alias silently accepted");
    auto wrong = actual; wrong.images.at(0).points2D.at(0).x += 1;
    rejects([&] { VerifiedExportReport report; verified_export_detail::verifyOriginalRawLedger(catalog, scene, root / "parallel", wrong, dirs, retained, helperOptions, 1u << 20, report); }, "wrong original feature XY silently accepted");
    auto badDirections = dirs; badDirections.at(13).track.pop_back();
    rejects([&] { VerifiedExportReport report; verified_export_detail::verifyOriginalRawLedger(catalog, scene, root / "parallel", actual, badDirections, retained, helperOptions, 1u << 20, report); }, "missing directional raw track silently accepted");
    auto badRetained = retained; badRetained.at(14).clear();
    rejects([&] { VerifiedExportReport report; verified_export_detail::verifyOriginalRawLedger(catalog, scene, root / "parallel", actual, dirs, badRetained, helperOptions, 1u << 20, report); }, "missing retained raw track silently accepted");
    require(sha(source) == sourceHash && sha(root / "scene/head.bin") == headHash && sha(scene.acceptedSnapshot().directory / "points.bin") == pointsHash, "formal Reader changed immutable source/accepted authority");
    size_t at = 0; for (const auto& image : original.images) require(sha(featurePath(image.first)) == featureHashes.at(at++), "formal Reader changed source features");
    std::cout << "PASS formal Reader threads1/auto exact three-ledger/photo reductions, finite/directional/retained raw, bounded fallback and bad alias/XY/tracks rejection; no real-photo acceptance claim\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
