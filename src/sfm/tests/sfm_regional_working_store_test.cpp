#include "sfm/map/RegionalWorkingStore.h"
#include <iostream>
#include <chrono>

using namespace sfm;
using Store = regional::RegionalWorkingStore;

static void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class Call> static void rejects(Call call, const char* message) {
    bool rejected = false; try { call(); } catch (const std::exception&) { rejected = true; } check(rejected, message);
}
static Reconstruction fixture(uint32_t imageOffset = 0) {
    Reconstruction m; Camera camera; camera.id = 4; camera.width = 1000; camera.height = 800;
    camera.fx = camera.fy = 700; camera.cx = 500; camera.cy = 400; m.cameras.emplace(4, camera);
    for (uint32_t id = 1; id <= 4; ++id) {
        Image im; im.id = id + imageOffset; im.camera_id = 4; im.name = "photo-" + std::to_string(im.id);
        im.pose = {mat3Identity(), {double(id), 0, 0}}; im.registered = true;
        im.points2D = {{12.25, 32.5}, {124.5, 99.5}, {200, 700}};
        im.point3D_ids.assign(3, kInvalidPoint3D); m.images.emplace(im.id, std::move(im));
    }
    for (uint64_t id : {7ull, 99ull, 700ull}) {
        Point3D p; p.xyz = {double(id), 2, 10}; p.error = double(id) * 0.01;
        p.rgb[0] = 33; p.rgb[1] = 104; p.rgb[2] = 201;
        p.track = {{1 + imageOffset, 0}, {2 + imageOffset, 0}, {2 + imageOffset, 0}, {3 + imageOffset, 1}};
        if (id == 700) p.track = {{4 + imageOffset, 2}};
        m.points3D.emplace(id, p);
    }
    m.images.at(1 + imageOffset).point3D_ids[0] = 99;
    m.images.at(2 + imageOffset).point3D_ids[0] = 99;
    m.images.at(3 + imageOffset).point3D_ids[1] = 99;
    m.images.at(4 + imageOffset).point3D_ids[2] = 700;
    m.next_point3D_id = 701; m.rig_detached.insert(1 + imageOffset); return m;
}
static std::string fileHash(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary); check(bool(in), "cannot hash test source"); spirula::Sha256 sha;
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()); sha.update(bytes.data(), bytes.size()); return sha.hex();
}
static std::string digest(const Reconstruction& model, const std::filesystem::path& path) {
    model_store_detail::writeModel(model, path); return fileHash(path);
}
static void readOnlyStoreTest(const std::filesystem::path& root, const std::filesystem::path& work) {
    namespace fs = std::filesystem;
    const auto copy = root / "readonly-raw-source"; fs::copy(work, copy, fs::copy_options::recursive);
    size_t count; uint64_t generation;
    { Store read(copy, 32u << 20, Store::OpenMode::ReadOnly); count = read.sourceCount(); generation = read.generation(); }
    const auto nextSource = copy / ("source-" + std::to_string(count)), nextParameters = copy / ("parameters-" + std::to_string(generation + 1));
    for (const auto& path : {nextSource, nextParameters}) { fs::create_directory(path); std::ofstream out(path / "partial.bin"); out << "retained immutable import evidence"; }
    auto tree = [](const fs::path& path, bool locked = false) {
        std::map<std::string, std::string> result;
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            const auto key = entry.path().lexically_relative(path).generic_u8string();
            result.emplace(key, entry.is_directory() ? "directory" : locked && entry.path().filename() == "transaction.lock" ?
                "exclusive-lock-size=" + std::to_string(entry.file_size()) : fileHash(entry.path()));
        }
        return result;
    };
    const auto before = tree(copy);
    {
        Store read(copy, 32u << 20, Store::OpenMode::ReadOnly); read.verifyImmutable();
        const auto key = read.key(0, 0); const auto forkPath = read.forkImmutablePrefix(root / "readonly-prefix-fork", 1);
        Store fork(forkPath, 32u << 20); check(fork.sourceCount() == 1 && fork.sourceRawIdentity(0) == read.sourceRawIdentity(0), "readonly raw prefix fork changed identity");
        uint64_t rows = 0; read.visitObservations([&](const auto&) { ++rows; }); check(rows == 18, "readonly full raw multiplicity lost");
        rejects([&] { read.importModel(read.sourcePath(0), fixture()); }, "readonly immutable source import started");
        rejects([&] { read.beginGeneration(); }, "readonly raw parameter generation started");
        rejects([&] { read.setPointXYZ(key, {1, 2, 3}); }, "readonly raw XYZ writable");
        rejects([&] { read.applyPointUpdates({{key, {1, 2, 3}}}); }, "readonly raw batch writable");
        rejects([&] { read.setPose(1, read.poses().images.at(1).pose); }, "readonly raw pose writable");
        rejects([&] { read.setCamera(read.poses().cameras.begin()->second); }, "readonly raw calibration writable");
        rejects([&] { read.rewriteSourceXYZ(0, [](auto) {}); }, "readonly raw XYZ producer writable");
        rejects([&] { read.updateSourceXYZ(0, [](auto, Vec3 xyz) { return xyz; }); }, "readonly raw XYZ updater writable");
        rejects([&] { read.commitGeneration(); }, "readonly raw manifest writable");
        rejects([&] { read.abortGeneration(); }, "readonly raw candidate retention writable");
        rejects([&] { read.interpolateGeneration(copy, 0.5); }, "readonly raw interpolation writable");
        rejects([&] { Store writer(copy, 32u << 20); }, "readonly raw lease allowed a writer");
    }
    check(tree(copy) == before, "readonly raw open/fork/refusal changed source paths or SHA");
    {
        Store writer(copy, 32u << 20); writer.beginGeneration(); const auto locked = tree(copy, true);
        rejects([&] { Store read(copy, 32u << 20, Store::OpenMode::ReadOnly); }, "readonly raw ignored active writer");
        check(tree(copy, true) == locked, "readonly raw refusal changed active writer evidence"); writer.abortGeneration();
    }
    const auto noLock = root / "readonly-raw-no-lock"; fs::copy(copy, noLock, fs::copy_options::recursive);
    fs::remove(noLock / "transaction.lock"); const auto noLockBefore = tree(noLock);
    { Store read(noLock, 32u << 20, Store::OpenMode::ReadOnly); }
    check(tree(noLock) == noLockBefore && !fs::exists(noLock / "transaction.lock"), "readonly raw created a missing lock");
    const auto absent = root / "readonly-raw-absent";
    rejects([&] { Store read(absent, 32u << 20, Store::OpenMode::ReadOnly); }, "readonly raw accepted missing source");
    check(!fs::exists(absent), "readonly raw created a missing root");
    const auto cold = root / "readonly-cold-photo-cache";
    { Store writer(cold, 32u << 20); writer.importModel(root / "original.bin", fixture()); }
    const auto coldBefore = tree(cold);
    { Store read(cold, 32u << 20, Store::OpenMode::ReadOnly); rejects([&] { read.preparePhotoObservationCache(0); }, "readonly raw built a missing derived cache"); }
    check(tree(cold) == coldBefore, "readonly photo-cache refusal wrote source evidence");
}
static void compactSlots(const std::filesystem::path& root) {
    auto first = fixture(); first.images.at(1).points2D[1] = {}; first.images.at(2).points2D[1] = {};
    auto second = fixture(); second.points3D.clear();
    for (auto& image : second.images) { std::fill(image.second.points2D.begin(), image.second.points2D.end(), Vec2{}); std::fill(image.second.point3D_ids.begin(), image.second.point3D_ids.end(), kInvalidPoint3D); }
    Point3D p; p.xyz = {1, 2, 3}; p.track = {{1, 1}, {2, 1}}; second.points3D.emplace(999, p);
    for (uint32_t image : {1u, 2u}) { second.images.at(image).points2D[1] = {124.5, 99.5}; second.images.at(image).point3D_ids[1] = 999; }
    const auto a = root / "compact-a.bin", b = root / "compact-b.bin"; model_store_detail::writeModel(first, a); model_store_detail::writeModel(second, b);
    Store store(root / "compact", 32u << 20); store.importModel(a, first); store.importModel(b, second);
    const auto gathered = store.gather({1, 2}, {}, false); check(gathered.model.images.at(1).points2D[1].x == 124.5, "zero placeholder blocked a genuine observed feature");
    check(digest(store.materialize(0), root / "compact-roundtrip-a.bin") == fileHash(a) && digest(store.materialize(1), root / "compact-roundtrip-b.bin") == fileHash(b), "compact placeholder source evidence was rewritten");
    auto missing = second; missing.images.at(3).registered = false; const auto unregistered = root / "unregistered.bin";
    model_store_detail::writeModel(missing, unregistered); Store staged(root / "registration", 32u << 20); staged.importModel(unregistered, missing);
    check(!staged.poses().images.at(3).registered, "unregistered metadata changed before recovery");
    auto recovered = second; recovered.points3D.at(999).track.push_back({3, 1}); recovered.images.at(3).points2D[1] = {124.5, 99.5}; recovered.images.at(3).point3D_ids[1] = 999;
    const auto recoveredFile = root / "registration-recovered.bin"; model_store_detail::writeModel(recovered, recoveredFile);
    staged.importModel(recoveredFile, recovered); check(staged.poses().images.at(3).registered, "real recovered registration was discarded");
    Store replay(root / "registration", 32u << 20); check(replay.poses().images.at(3).registered, "recovered registration was not durable");
    auto conflicting = second; conflicting.images.at(1).points2D[1].x += 1;
    const auto c = root / "compact-conflict.bin"; model_store_detail::writeModel(conflicting, c); store.importModel(c, conflicting);
    rejects([&] { store.gather({1, 2}, {}, false); }, "two genuinely observed conflicting XY coordinates were accepted");
}
static void lruReaders(const std::filesystem::path& root) {
    Store store(root / "lru-readers", 32u << 20); const auto model = fixture();
    for (size_t source = 0; source < 10; ++source) {
        const auto path = root / ("lru-source-" + std::to_string(source) + ".bin"); model_store_detail::writeModel(model, path); store.importModel(path, model);
    }
    for (size_t source = 2; source < 10; ++source) store.key(source, 0);
    store.resetCounters(); const Store::PointKey key{0, 0, 7};
    for (size_t query = 0; query < 1000; ++query) check((store.pointXYZ(key) - model.points3D.at(7).xyz).norm() == 0, "LRU changed native parameter values");
    const auto io = store.counters(); check(io.reader_opens == 2 && io.reader_evictions == 2 && io.reader_hits == 1998,
        "active metadata/XYZ Readers were evicted by path instead of actual last use");
}
static void retainedRejection(const std::filesystem::path& root) {
    const auto work = root / "retained-rejection", source = root / "retained-source.bin"; const auto model = fixture();
    model_store_detail::writeModel(model, source); Store store(work, 32u << 20); store.importModel(source, model);
    const auto manifestHash = fileHash(work / "manifest.bin"), acceptedHash = fileHash(work / "parameters-0/points-0.xyz");
    store.resetCounters(); uint64_t parameterCount = 0;
    store.visitSourceParameters(0, [&](Store::PointKey point, Vec3 value) {
        check(point.ordinal == parameterCount++ && value.x == model.points3D.at(point.localPoint).xyz.x, "sequential parameter source key/XYZ changed");
    });
    check(parameterCount == 3 && store.counters().raw_bytes == 0 && store.counters().parameter_bytes == 3 * sizeof(Vec3),
        "sequential parameter traversal read original raw tracks or missed XYZ");
    store.beginGeneration(); const auto key = store.key(0, 0); store.setPointXYZ(key, {123, 456, 789});
    auto pose = store.poses().images.at(1).pose; pose.t.y = 17; store.setPose(1, pose);
    auto camera = store.poses().cameras.at(4); camera.fx = 777; store.setCamera(camera); store.abortGeneration();
    check(!std::filesystem::exists(work / "parameters-1"), "rejected next-generation name blocks continuation");
    std::filesystem::path rejected;
    for (const auto& entry : std::filesystem::directory_iterator(work)) if (entry.path().filename().u8string().find("rejected-parameters-1-") == 0) rejected = entry.path();
    check(!rejected.empty(), "rejected XYZ candidate was deleted");
    const auto candidate = model_store_detail::readModel(rejected / "poses.bin");
    check(candidate.images.at(1).pose.t.y == 17 && candidate.cameras.at(4).fx == 777, "rejected pose/intrinsic metadata was not retained");
    model_store_detail::Reader xyz(rejected / "points-0.xyz", 64u << 10); const auto value = xyz.scalar<Vec3>();
    check(value.x == 123 && value.y == 456 && value.z == 789, "rejected native XYZ parameters were not retained");
    check(fileHash(work / "manifest.bin") == manifestHash && fileHash(work / "parameters-0/points-0.xyz") == acceptedHash,
        "retaining rejection changed accepted head or XYZ");
    Store reopened(work, 32u << 20); check(reopened.pointXYZ(key).x == model.points3D.at(7).xyz.x && reopened.poses().cameras.at(4).fx == 700,
        "accepted restore used rejected candidate parameters"); reopened.beginGeneration(); reopened.abortGeneration();
}
static void atomicRewrite(const std::filesystem::path& root) {
    const auto work = root / "atomic-rewrite", source = root / "atomic-source.bin"; const auto model = fixture();
    model_store_detail::writeModel(model, source); Store store(work, 32u << 20); store.importModel(source, model);
    const auto rawPath = work / "source-0/observations.bin", acceptedPath = work / "parameters-0/points-0.xyz";
    const auto rawHash = fileHash(rawPath), acceptedHash = fileHash(acceptedPath), sourceHash = fileHash(source);
    store.beginGeneration(); const auto next = work / "parameters-1/points-0.xyz"; const auto nextHash = fileHash(next);
    const auto key = store.key(0, 0); store.pointXYZ(key); // Prime an XYZ Reader before replacement.
    rejects([&] { store.rewriteSourceXYZ(0, [&](auto emit) { emit(Vec3{1, 2, 3}); throw std::runtime_error("producer failed"); }); }, "producer failure accepted");
    check(fileHash(next) == nextHash, "producer failure replaced active XYZ");
    rejects([&] { store.rewriteSourceXYZ(0, [&](auto emit) { emit(Vec3{1, 2, 3}); }); }, "incomplete point stream accepted");
    rejects([&] { store.rewriteSourceXYZ(0, [&](auto emit) { for (size_t i = 0; i < 4; ++i) emit(Vec3{1, 2, 3}); }); }, "excess point stream accepted");
    rejects([&] { store.rewriteSourceXYZ(0, [&](auto emit) { emit(Vec3{NAN, 2, 3}); }); }, "nonfinite point stream accepted");
    check(fileHash(next) == nextHash, "invalid point stream replaced active XYZ");
    size_t parts = 0; for (const auto& file : std::filesystem::directory_iterator(next.parent_path()))
        if (file.path().filename().u8string().find("points-0.xyz.private.part") == 0) ++parts;
    check(parts == 4, "failed sequential parameter candidates were not retained");
    store.rewriteSourceXYZ(0, [&](auto emit) {
        store.visitSourcePoints(0, [&](Store::PointKey point, Vec3 value, const auto& track) {
            check(track.size() == model.points3D.at(point.localPoint).track.size(), "rewrite lost original support/duplicate track");
            emit(value + Vec3{0, 4, 0});
        });
    });
    check(store.pointXYZ(key).y == 6, "atomic rewrite retained a stale cached XYZ Reader");
    store.commitGeneration("private-rewrite-test"); Store reopened(work, 32u << 20);
    check(reopened.pointXYZ(key).y == 6 && reopened.generationPhase() == "private-rewrite-test", "atomic XYZ rewrite was not durable");
    check(fileHash(rawPath) == rawHash && fileHash(source) == sourceHash && fileHash(acceptedPath) == acceptedHash,
        "atomic rewrite changed immutable raw/source/previous accepted generation");
}
static void photoMajorCache(const std::filesystem::path& root) {
    const auto work = root / "photo-major", source = root / "photo-major-source.bin"; const auto model = fixture();
    model_store_detail::writeModel(model, source); Store store(work, 32u << 20); store.importModel(source, model);
    const auto rawHash = fileHash(work / "source-0/observations.bin"), indexHash = fileHash(work / "source-0/photo-index.bin"), manifestHash = fileHash(work / "manifest.bin");
    const auto data = store.preparePhotoObservationCache(0); check(store.counters().photo_cache_builds == 1, "cold photo-major cache was not built once");
    store.resetCounters(); std::vector<Store::RawObservation> actual;
    for (uint32_t image : {1u, 2u, 3u, 4u}) store.visitImageObservations(image, [&](const auto& row) { actual.push_back(row); });
    check(actual.size() == 9 && store.counters().photo_cache_builds == 0 && store.counters().seeks == 4, "photo-major cache lost duplicates or used per-row random seeks");
    for (size_t i = 1; i < actual.size(); ++i) check(std::tie(actual[i-1].image, actual[i-1].ordinal) < std::tie(actual[i].image, actual[i].ordinal), "photo-major rows reordered original observations within photo");
    Store warm(work, 32u << 20); check(warm.preparePhotoObservationCache(0) == data && warm.counters().photo_cache_builds == 0, "warm photo-major cache rebuilt frozen source");
    check(fileHash(work / "source-0/observations.bin") == rawHash && fileHash(work / "source-0/photo-index.bin") == indexHash && fileHash(work / "manifest.bin") == manifestHash,
        "derived photo-major cache changed original immutable rows/index or generation");
    { std::fstream file(data, std::ios::binary | std::ios::in | std::ios::out); file.seekp(0); char byte = 71; file.write(&byte, 1); }
    rejects([&] { store.preparePhotoObservationCache(0); }, "in-process tampered photo cache was accepted");
    Store corrupt(work, 32u << 20); rejects([&] { corrupt.preparePhotoObservationCache(0); }, "restored photo cache checksum mismatch was accepted");
    const auto missingWork = root / "photo-missing", missingSource = root / "photo-missing-source.bin";
    model_store_detail::writeModel(model, missingSource); Store missing(missingWork, 32u << 20); missing.importModel(missingSource, model);
    std::filesystem::resize_file(missingWork / "source-0/photo-index.bin", 0);
    rejects([&] { missing.preparePhotoObservationCache(0); }, "missing original photo-index rows were silently rebuilt/accepted");
    const auto partialWork = root / "photo-partial", partialSource = root / "photo-partial-source.bin";
    model_store_detail::writeModel(model, partialSource); Store partial(partialWork, 32u << 20); partial.importModel(partialSource, model);
    const auto partialData = partial.preparePhotoObservationCache(0); std::filesystem::rename(partialData.parent_path() / "ready.bin", partialData.parent_path() / "preserved-ready.bin");
    Store interrupted(partialWork, 32u << 20); rejects([&] { interrupted.preparePhotoObservationCache(0); }, "incomplete photo-major candidate was silently reused/replaced");
    check(std::filesystem::exists(partialData) && std::filesystem::exists(partialData.parent_path() / "preserved-ready.bin"), "incomplete derived cache evidence was deleted");
    auto externalModel = fixture(); externalModel.points3D.clear(); Point3D point; point.xyz = {1, 2, 10};
    for (size_t i = 0; i < 150000; ++i) point.track.push_back({uint32_t(1 + i % 2), 0});
    externalModel.points3D.emplace(7, std::move(point)); const auto externalSource = root / "photo-external-source.bin";
    model_store_detail::writeModel(externalModel, externalSource); Store external(root / "photo-external", 32u << 20); external.importModel(externalSource, externalModel);
    const auto externalData = external.preparePhotoObservationCache(0);
    check(external.counters().photo_cache_written_bytes > std::filesystem::file_size(externalData), "external bounded merge path was not exercised");
    uint64_t rows = 0; for (uint32_t image : {1u, 2u}) external.visitImageObservations(image, [&](const auto& row) {
        check(row.point.localPoint == 7 && row.ordinal % 2 == image - 1 && row.xy.x == 12.25, "external merge changed original raw key/XY/order"); ++rows;
    });
    check(rows == 150000, "external merge dropped duplicate observations");
}
static void retainedStep(const std::filesystem::path& root) {
    namespace fs = std::filesystem;
    const auto work = root / "retained-step", sourceA = root / "step-a.bin", sourceB = root / "step-b.bin";
    auto first = fixture(), second = fixture(10);
    model_store_detail::writeModel(first, sourceA); model_store_detail::writeModel(second, sourceB);
    Store store(work, 32u << 20); store.importModel(sourceA, first); store.importModel(sourceB, second);
    const auto generation = store.generation(); const auto key = store.localKey(0, 99);
    const auto manifestHash = fileHash(work / "manifest.bin"); const auto acceptedXYZ = work / ("parameters-" + std::to_string(generation)) / "points-0.xyz";
    const auto acceptedHash = fileHash(acceptedXYZ), sourceHashA = fileHash(sourceA), sourceHashB = fileHash(sourceB);
    std::map<fs::path, std::string> immutable;
    for (size_t source = 0; source < 2; ++source) for (const auto& file : fs::directory_iterator(work / ("source-" + std::to_string(source))))
        if (file.is_regular_file()) immutable.emplace(file.path(), fileHash(file.path()));
    const Pose before = store.poses().images.at(1).pose;
    Pose next{so3Exp({0, M_PI, 0}), {}}; const Vec3 nextCentre{-5, 2, 4}; next.t = mul(next.R, nextCentre) * -1.;
    store.beginGeneration();
    for (size_t source = 0; source < 2; ++source)
        store.updateSourceXYZ(source, [](Store::PointKey, Vec3 value) { return value + Vec3{20, 6, -2}; });
    store.setPose(1, next); auto camera = store.poses().cameras.at(4);
    camera.fx = camera.fy = 900; camera.cx = 540; camera.k1 = 0.2; store.setCamera(camera);
    const auto candidate = store.abortGeneration();
    check(candidate.parent_path() == work && fs::exists(candidate / "poses.bin"), "rejected step path not returned or candidate missing");
    check(fileHash(work / "manifest.bin") == manifestHash && fileHash(acceptedXYZ) == acceptedHash, "reject changed accepted step head");
    rejects([&] { store.interpolateGeneration(candidate, 0.5); }, "step outside transaction accepted");
    auto copyCandidate = [&](const std::string& suffix, bool outside = false) {
        const auto copy = (outside ? root : work) / ("rejected-parameters-" + std::to_string(generation + 1) + "-" + suffix);
        fs::create_directories(copy);
        for (const auto& file : fs::directory_iterator(candidate)) fs::copy_file(file.path(), copy / file.path().filename());
        return copy;
    };
    store.beginGeneration();
    const auto pending = work / ("parameters-" + std::to_string(generation + 1));
    const auto pendingHash = fileHash(pending / "points-0.xyz"), pendingPose = digest(store.poses(), root / "step-pending-poses.bin");
    auto invalid = [&](const fs::path& path, double alpha, const char* message) {
        rejects([&] { store.interpolateGeneration(path, alpha); }, message);
        check(fileHash(pending / "points-0.xyz") == pendingHash && digest(store.poses(), root / "step-after-rejection.bin") == pendingPose,
              "invalid retained step partially changed pending parameters");
    };
    for (double alpha : std::array<double, 5>{0., -0.5, 1.01, NAN, INFINITY}) invalid(candidate, alpha, "invalid step alpha accepted");
    invalid(copyCandidate("outside", true), 0.5, "outside candidate path accepted");
    for (int type = 0; type < 6; ++type) {
        const auto copy = copyCandidate("metadata-" + std::to_string(type));
        auto rec = model_store_detail::readModel(copy / "poses.bin");
        if (type == 0) rec.images.at(1).id = 999;
        if (type == 1) rec.images.at(1).name += "-wrong";
        if (type == 2) rec.images.at(1).camera_id = 123;
        if (type == 3) rec.cameras.at(4).pixel_scale *= 2;
        if (type == 4) rec.cameras.at(4).width += 1;
        if (type == 5) rec.cameras.at(4).model = CamModel::OpenCV;
        model_store_detail::writeModel(rec, copy / "poses.bin"); invalid(copy, 0.5, "changed candidate metadata accepted");
    }
    const auto truncated = copyCandidate("truncated");
    fs::resize_file(truncated / "points-1.xyz", fs::file_size(truncated / "points-1.xyz") - 1);
    invalid(truncated, 0.5, "truncated final source XYZ accepted");
    const auto nonfinite = copyCandidate("nonfinite");
    {
        std::fstream file(nonfinite / "points-1.xyz", std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(std::streamoff(fs::file_size(nonfinite / "points-1.xyz") - sizeof(Vec3)));
        const Vec3 bad{NAN, 2, 3}; file.write(reinterpret_cast<const char*>(&bad), sizeof(bad));
    }
    invalid(nonfinite, 0.5, "nonfinite final source XYZ accepted");
    const auto badPose = copyCandidate("bad-pose");
    auto pose = model_store_detail::readModel(badPose / "poses.bin"); pose.images.at(1).pose.R[0] = NAN;
    model_store_detail::writeModel(pose, badPose / "poses.bin"); invalid(badPose, 0.5, "nonfinite candidate pose accepted");
    store.interpolateGeneration(candidate, 0.5);
    const auto middle = store.pointXYZ(key); check(middle.x == 109 && middle.y == 5 && middle.z == 9, "XYZ did not use common alpha");
    const auto centre = cameraCenter(store.poses().images.at(1).pose);
    check((centre - (cameraCenter(before) + nextCentre) * 0.5).norm() < 1e-12, "camera centre interpolation used translation instead");
    check(std::abs(so3Log(store.poses().images.at(1).pose.R).norm() - M_PI * 0.5) < 1e-12, "pi rotation did not interpolate on SO3");
    check(store.poses().cameras.at(4).fx == 800 && store.poses().cameras.at(4).cx == 520 && store.poses().cameras.at(4).k1 == 0.1,
          "intrinsics did not use the same retained step alpha");
    store.interpolateGeneration(candidate, 0.25); check(store.pointXYZ(key).x == 104, "second trial blended from previous trial instead of accepted generation");
    store.interpolateGeneration(candidate, 1);
    check(fileHash(pending / "points-0.xyz") == fileHash(candidate / "points-0.xyz") &&
          digest(store.poses(), root / "step-endpoint.bin") == fileHash(candidate / "poses.bin"), "alpha1 is not the exact retained endpoint");
    store.interpolateGeneration(candidate, 0.5); store.commitGeneration("retained-step-test");
    Store restored(work, 32u << 20);
    check(restored.generation() == generation + 1 && restored.generationPhase() == "retained-step-test" && restored.pointXYZ(key).x == 109 &&
          restored.poses().cameras.at(4).fx == 800 && (cameraCenter(restored.poses().images.at(1).pose) - centre).norm() < 1e-12,
          "committed retained step did not restore exactly");
    uint64_t observations = 0; restored.visitObservations([&](const auto&) { ++observations; }); check(observations == 18, "retained step lost duplicate original observations");
    check(fileHash(sourceA) == sourceHashA && fileHash(sourceB) == sourceHashB && fileHash(acceptedXYZ) == acceptedHash, "retained step modified original source or previous accepted XYZ");
    for (const auto& file : immutable) check(fileHash(file.first) == file.second, "retained step changed immutable source/flat bytes");
    auto pinhole = Camera::defaultFor(7, 800, 600, 500, CamModel::Pinhole), nextPinhole = pinhole;
    pinhole.k6 = nextPinhole.k6 = 123; nextPinhole.fx = 900;
    check(regional::step::interpolateCamera(pinhole, nextPinhole, 0.5).k6 == 123, "inactive camera metadata was zeroed by unpackIntrinsics");
    nextPinhole.k6 = 124; rejects([&] { regional::step::interpolateCamera(pinhole, nextPinhole, 0.5); }, "inactive non-BA camera parameter changed");
}
static void retainedRigStep(const std::filesystem::path& root) {
    const auto work = root / "rig-step", source = root / "rig-step-source.bin";
    auto model = fixture(); auto secondCamera = model.cameras.at(4); secondCamera.id = 5; model.cameras.emplace(5, secondCamera);
    model.images.at(1).pose = {mat3Identity(), {}};
    model.images.at(2).camera_id = 5; model.images.at(2).pose = {mat3Identity(), {-2, 0, 0}};
    RigCalib rig; rig.resize(2); rig.ref = 0; rig.cam_from_rig[0] = {mat3Identity(), {}};
    rig.cam_from_rig[1] = {mat3Identity(), {-2, 0, 0}}; rig.fixed = {1, 1}; rig.established = {1, 1}; model.rigs.push_back(rig);
    model_store_detail::writeModel(model, source); Store store(work, 32u << 20); store.importModel(source, model);
    check(store.poses().rigs.empty() && store.sourceViews(0).rigs.size() == 1, "source-rig fixture did not exercise an empty global rig catalogue");
    const Mat3 rotated = so3Exp({0, 0, M_PI}); const Pose nextFirst{rotated, {}}, nextSecond{rotated, {-2, 0, 0}};
    store.beginGeneration(); store.setPose(1, nextFirst); store.setPose(2, nextSecond); const auto sourceCandidate = store.abortGeneration();
    const auto originalManifest = fileHash(work / "manifest.bin"); store.beginGeneration();
    bool sourceRejected = false;
    try { store.interpolateGeneration(sourceCandidate, 0.5); }
    catch (const std::runtime_error& error) { sourceRejected = std::string(error.what()) == "regional backtracking requires rig-frame interpolation"; }
    check(sourceRejected && store.poses().images.at(1).pose.R == mat3Identity() && fileHash(work / "manifest.bin") == originalManifest,
          "source rig escaped half-step protection through empty global rig metadata");
    store.interpolateGeneration(sourceCandidate, 1);
    check(digest(store.poses(), root / "source-rig-step-endpoint.bin") == fileHash(sourceCandidate / "poses.bin"), "source-rig alpha1 endpoint was not exact");
    store.abortGeneration();
    store.beginGeneration(); const_cast<Reconstruction&>(store.poses()).rigs = model.rigs; store.commitGeneration("rig-step-accepted-fixture");
    const auto first = store.poses().images.at(1).pose, second = store.poses().images.at(2).pose;
    check(std::abs((cameraCenter(first) - cameraCenter(second)).norm() - 2) < 1e-12 &&
          std::abs((cameraCenter(nextFirst) - cameraCenter(nextSecond)).norm() - 2) < 1e-12, "rig endpoints did not retain fixed baseline");
    const auto middleFirst = regional::step::interpolatePose(first, nextFirst, 0.5), middleSecond = regional::step::interpolatePose(second, nextSecond, 0.5);
    check((cameraCenter(middleFirst) - cameraCenter(middleSecond)).norm() < 1e-12, "rig counterexample did not collapse independent centre interpolation");
    store.beginGeneration(); store.setPose(1, nextFirst); store.setPose(2, nextSecond); const auto candidate = store.abortGeneration();
    const auto manifest = fileHash(work / "manifest.bin"); store.beginGeneration();
    const auto pending = work / ("parameters-" + std::to_string(store.generation() + 1)); const auto xyz = fileHash(pending / "points-0.xyz");
    bool rejected = false;
    try { store.interpolateGeneration(candidate, 0.5); }
    catch (const std::runtime_error& error) { rejected = std::string(error.what()) == "regional backtracking requires rig-frame interpolation"; }
    check(rejected && fileHash(pending / "points-0.xyz") == xyz && fileHash(work / "manifest.bin") == manifest && store.poses().images.at(1).pose.R == first.R,
          "rig half-step was accepted, changed state or hid the required rig-frame diagnostic");
    store.interpolateGeneration(candidate, 1);
    check(digest(store.poses(), root / "rig-step-endpoint.bin") == fileHash(candidate / "poses.bin"), "rig alpha1 endpoint was not exact");
    store.abortGeneration(); Store restored(work, 32u << 20);
    check(restored.poses().rigs.size() == 1 && restored.poses().images.at(1).pose.R == first.R, "rig rejection changed durable accepted parameters");
}
static void resolvedMaterialization(const std::filesystem::path& root) {
    const auto work = root / "resolved-materialization", source = root / "resolved-source.bin";
    auto original = fixture(); model_store_detail::writeModel(original, source);
    Store store(work, 32u << 20); store.importModel(source, original);
    const auto sourceHash = fileHash(source), manifestHash = fileHash(work / "manifest.bin"), parametersHash = fileHash(work / "parameters-0/points-0.xyz");
    std::map<std::filesystem::path, std::string> frozen;
    for (const auto& file : std::filesystem::directory_iterator(work / "source-0"))
        if (file.is_regular_file()) frozen.emplace(file.path(), fileHash(file.path()));
    check(digest(store.materialize(0), root / "resolved-default.bin") == sourceHash, "default materialization changed original private-coordinate contract");
    size_t calls = 0;
    auto resolved = store.materialize(0, [&](Store::PointKey key, Vec3 value) {
        ++calls; check(key.source == 0 && value.x == original.points3D.at(key.localPoint).xyz.x, "resolver lost source/local point identity or stored XYZ");
        return key.localPoint == 99 ? Vec3{9, 8, 7} : value;
    });
    auto expected = original; expected.points3D.at(99).xyz = {9, 8, 7};
    const auto resolvedPath = root / "resolved-reader.bin";
    check(calls == original.points3D.size() && digest(resolved, resolvedPath) == digest(expected, root / "resolved-reference.bin"),
          "resolved materialization changed tracks, duplicate raw observations, features, assignments, RGB/error or view metadata");
    const auto read = model_store_detail::readModel(resolvedPath);
    check(read.points3D.at(99).xyz.z == 7 && read.points3D.at(99).track.size() == 4 && read.points3D.at(99).track[1].image_id == read.points3D.at(99).track[2].image_id,
          "native Reader lost resolved XYZ or duplicate original track entries");
    rejects([&] { store.materialize(0, [](Store::PointKey, Vec3) { return Vec3{NAN, 2, 3}; }); }, "nonfinite resolver result materialized");
    rejects([&] { store.materialize(0, [](Store::PointKey, Vec3) { return Vec3{1, INFINITY, 3}; }); }, "infinite resolver result materialized");
    check(fileHash(source) == sourceHash && fileHash(work / "manifest.bin") == manifestHash && fileHash(work / "parameters-0/points-0.xyz") == parametersHash && store.pointXYZ(store.localKey(0, 99)).x == 99,
          "read-only coordinate resolution changed source, accepted XYZ or head");
    for (const auto& file : frozen) check(fileHash(file.first) == file.second, "coordinate resolution modified immutable working data");
}
static void run() {
    ModelStore temporary(""); const auto root = temporary.directory();
    const auto original = root / "original.bin", second = root / "second.bin", work = root / "working";
    const auto m = fixture(); model_store_detail::writeModel(m, original);
    compactSlots(root);
    lruReaders(root);
    retainedRejection(root);
    atomicRewrite(root);
    photoMajorCache(root);
    retainedStep(root);
    retainedRigStep(root);
    resolvedMaterialization(root);
    auto originalHash = fileHash(original);
    Store store(work, 32u << 20); check(store.importModel(original, m, "original-evidence") == 0, "wrong source slot");
    check(store.sourceLabel(0) == "original-evidence" && store.findSourceLabel("original-evidence") == 0, "stable source label lost");
    check(store.importModel(original, m) == 0 && store.sourceCount() == 1, "same source reimported");
    check(store.poses().images.at(1).points2D.empty(), "global pose table retained feature arrays");
    check(store.sourceViews(0).rig_detached.count(1) == 1, "immutable rig metadata lost");
    auto k = store.localKey(0, 99); check(k.ordinal == 1 && k.localPoint == 99, "local point mapping changed");
    uint64_t rawCount = 0, repeated = 0;
    store.visitSourcePoints(0, [&](Store::PointKey p, Vec3 xyz, const std::vector<Store::RawObservation>& track) {
        check(xyz.x == m.points3D.at(p.localPoint).xyz.x, "point parameter changed on import");
        const auto& expected = m.points3D.at(p.localPoint).track; check(track.size() == expected.size(), "raw track duplicates removed");
        for (size_t i = 0; i < track.size(); ++i) {
            const auto& row = track[i]; check(row.image == expected[i].image_id && row.feature == expected[i].point2D_idx, "raw track order changed");
            check(row.xy.x == m.images.at(row.image).points2D[row.feature].x, "raw feature XY changed"); ++rawCount;
        }
    });
    check(rawCount == 9, "raw observation coverage incomplete");
    store.visitImageObservations(2, [&](const Store::RawObservation& row) { check(row.image == 2, "photo index gave another image"); ++repeated; });
    check(repeated == 4, "photo index collapsed repeated raw features");
    uint64_t selectedCount = 0, selectedRaw = 0;
    store.visitSelectedPoints({store.localKey(0, 700), k, k}, [&](Store::PointKey key, Vec3 xyz, const std::vector<Store::RawObservation>& track) {
        ++selectedCount; selectedRaw += track.size(); check(xyz.x == m.points3D.at(key.localPoint).xyz.x, "selected point parameter changed");
    });
    check(selectedCount == 2 && selectedRaw == 5, "selected point streaming dropped raw duplicates or repeated selection");
    store.resetCounters(); auto gather = store.gather({1}); auto stats = store.counters();
    check(gather.model.points3D.size() == 2 && gather.model.images.size() == 3, "active gather expanded unrelated points or lost boundary");
    check(gather.fixedImages == std::set<uint32_t>({2, 3}), "boundary cameras not fixed");
    check(gather.model.points3D.begin()->second.track.size() == 4, "gather changed duplicate raw track");
    check(stats.source_full_reads == 0 && stats.raw_bytes < std::filesystem::file_size(original), "warm gather reread full source model");
    auto truncated = store.gather({1}, {}, false); check(truncated.model.images.size() == 1 && truncated.model.points3D.begin()->second.track.size() == 1, "target-only gather boundary filter failed");
    check(truncated.boundaryPoints.size() == 2, "truncated boundary points were not fixed");
    check(digest(store.materialize(0), root / "roundtrip.bin") == originalHash, "materialized source is not byte equivalent");
    rejects([&] { store.setPointXYZ(k, {1, 2, 3}); }, "parameter mutation outside transaction accepted");
    store.beginGeneration(); store.setPointXYZ(k, {9, 8, 7});
    Pose changed = store.poses().images.at(1).pose; changed.t.x = 200; store.setPose(1, changed);
    Camera c = store.poses().cameras.at(4); c.fx = 800; store.setCamera(c);
    store.commitGeneration("initial-seed-v1"); check(store.generation() == 1, "committed generation did not advance");
    {
        Store restored(work, 32u << 20); check(restored.pointXYZ(k).z == 7 && restored.poses().images.at(1).pose.t.x == 200 && restored.poses().cameras.at(4).fx == 800, "parameter restore lost updates");
        check(restored.sourceLabel(0) == "original-evidence", "atomic source label not restored");
        check(restored.generationPhase() == "initial-seed-v1" && restored.sourcePhotoObservationCount(0, 2) == 4, "atomic seed phase or photo counts not restored");
        check(restored.point(k).track.size() == 4, "updated XYZ changed original track");
        restored.beginGeneration(); restored.updateSourceXYZ(0, [](Store::PointKey, Vec3 xyz) { xyz.y += 100; return xyz; });
        check(restored.pointXYZ(k).y == 108, "bulk sequential XYZ update failed"); restored.abortGeneration();
        check(restored.pointXYZ(k).y == 8 && restored.generation() == 1, "aborted transaction mutated accepted generation");
        restored.beginGeneration(); restored.applyPointUpdates({{k, {13, 14, 15}}, {restored.localKey(0, 7), {1, 1, 1}}});
        check(restored.pointXYZ(k).z == 15, "batched parameter update failed"); restored.abortGeneration();
    }
    auto m2 = fixture(10); model_store_detail::writeModel(m2, second); check(store.importModel(second, m2) == 1, "append source slot changed");
    check(store.localKey(0, 99) == k && store.pointXYZ(k).z == 7, "source append invalidated old point parameters");
    {
        Store restored(work, 32u << 20); check(restored.sourceCount() == 2 && restored.pointXYZ(k).z == 7, "appended source restore failed");
        uint64_t count = 0; restored.visitObservations([&](const Store::RawObservation&) { ++count; }); check(count == 18, "multi-source raw observations lost");
    }
    const auto acceptedXYZ = work / "parameters-1" / "points-0.xyz"; const auto acceptedHash = fileHash(acceptedXYZ);
    store.beginGeneration(); store.setPointXYZ(k, {3, 4, 5}); store.commitGeneration();
    check(fileHash(acceptedXYZ) == acceptedHash && store.pointXYZ(k).z == 5, "copy-on-write parameter generation modified old hardlink");
    const auto acceptedGeneration = store.generation();
    {
        Store interrupted(work, 32u << 20); interrupted.beginGeneration(); interrupted.setPointXYZ(k, {123, 456, 789});
        rejects([&] { Store overlapping(work, 32u << 20); }, "live transaction was mistaken for an interrupted generation");
    }
    {
        Store recovered(work, 32u << 20); check(recovered.generation() == acceptedGeneration && recovered.pointXYZ(k).z == 5, "interrupted generation was accepted");
        recovered.beginGeneration(); recovered.setPointXYZ(k, {3, 4, 6}); recovered.commitGeneration();
        check(recovered.generation() == acceptedGeneration + 1, "interrupted checkpoint could not replay");
    }
    rejects([&] { store.beginGeneration(); }, "stale writer replaced a newer accepted generation");
    check(fileHash(original) == originalHash, "immutable original model modified");
    readOnlyStoreTest(root, work);
    {
        std::fstream file(work / "source-0" / "observations.bin", std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(5); char bad = 53; file.write(&bad, 1); file.flush();
    }
    rejects([&] { Store broken(work, 32u << 20); }, "corrupted immutable data restored");
}
static void realProbe(const std::filesystem::path& source, const std::filesystem::path& context, const std::filesystem::path& output) {
    namespace fs = std::filesystem; fs::create_directories(output);
    auto start = std::chrono::steady_clock::now(); const auto full = model_store_detail::readModel(source);
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
    const double fullRead = elapsed(); auto views = model_store_detail::readModel(context, false); std::set<uint32_t> targets;
    for (const auto& image : views.images) if (image.second.registered && full.images.count(image.first)) targets.insert(image.first);
    check(!targets.empty(), "real working store probe has no overlapping context photos");
    Store store(output / "store", size_t(1) << 30); start = std::chrono::steady_clock::now(); store.importModel(source, full); const double import = elapsed();
    Reconstruction expected;
    for (const auto& pair : full.points3D) {
        bool selected = false; for (const auto& e : pair.second.track) if (targets.count(e.image_id)) selected = true;
        if (!selected) continue;
        Point3D point = pair.second; point.track.clear(); const uint64_t id = expected.next_point3D_id++;
        for (const auto& e : pair.second.track) if (targets.count(e.image_id)) {
            if (!expected.images.count(e.image_id)) {
                Image image = full.images.at(e.image_id); std::fill(image.point3D_ids.begin(), image.point3D_ids.end(), kInvalidPoint3D);
                expected.cameras.emplace(image.camera_id, full.cameras.at(image.camera_id)); expected.images.emplace(e.image_id, std::move(image));
            }
            expected.images.at(e.image_id).point3D_ids.at(e.point2D_idx) = id; point.track.push_back(e);
        }
        expected.points3D.emplace(id, std::move(point));
    }
    const auto expectedHash = digest(expected, output / "reference.bin"); expected = {};
    store.resetCounters(); start = std::chrono::steady_clock::now(); auto first = store.gather(targets, {}, false); const double gather = elapsed();
    check(digest(first.model, output / "gather.bin") == expectedHash, "real indexed gather differs from full-model reference");
    const auto count = first.model.points3D.size(); first = {}; start = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; ++i) { auto warm = store.gather(targets, {}, false); check(warm.model.points3D.size() == count, "warm gather point coverage changed"); }
    const double warm = elapsed() / 3; const auto io = store.counters();
    std::ofstream out(output / "probe.txt");
    out << "full_model_read_seconds=" << fullRead << "\nimport_seconds=" << import << "\ngather_seconds=" << gather << "\nwarm_gather_seconds=" << warm
        << "\ntarget_photos=" << targets.size() << "\nactive_points=" << count << "\nsource_full_deserializations_during_gather=" << io.source_full_reads
        << "\nlogical_raw_bytes_four_gathers=" << io.raw_bytes << "\nlogical_index_bytes_four_gathers=" << io.index_bytes << "\nlogical_parameter_bytes_four_gathers=" << io.parameter_bytes
        << "\nreference_model_sha256=" << expectedHash << '\n';
    out.flush(); check(bool(out), "cannot save real probe evidence"); std::cout << "real working store probe passed\n";
}
int main(int argc, char** argv) {
    try { if (argc == 5 && std::string(argv[1]) == "--real-probe") realProbe(argv[2], argv[3], argv[4]); else run(); std::cout << "regional working store tests passed\n"; return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
