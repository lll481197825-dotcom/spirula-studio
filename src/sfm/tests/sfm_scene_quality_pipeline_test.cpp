#include "sfm/map/SceneQuality.h"
#include "sfm/map/ImmutableSceneAliases.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
using namespace sfm;
using namespace sfm::regional;
using Path = std::filesystem::path;
static void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
template<class Action> static void rejects(Action action, const char* why) { bool caught = false; try { action(); } catch (const std::exception&) { caught = true; } require(caught, why); }
static std::string sha(const Path& path) { auto value = spirula::sha256_file(path.u8string()); require(!value.empty(), "fixture SHA failed"); return value; }
static bool same(const Quality& a, const Quality& b) {
    return a.observations == b.observations && a.invalid_projections == b.invalid_projections && a.retained_observations == b.retained_observations &&
        a.directional_observations == b.directional_observations && a.cost == b.cost && a.finite_cost == b.finite_cost;
}
static bool same(const PhotoQuality& a, const PhotoQuality& b) {
    return a.observations == b.observations && a.retained_observations == b.retained_observations && a.inliers == b.inliers && a.invalid_projections == b.invalid_projections && a.error == b.error;
}
static Quality serialReference(const ImmutableSceneTopology& index, const CanonicalScene& scene, double gate, size_t allowance,
                               std::vector<PhotoQuality>* photos = nullptr, SceneRawBatchStatistics* statistics = nullptr) {
    if (photos) photos->assign(size_t(index.imageSlots()), {});
    Quality result; result.finite_cost = 0;
    visitSceneRawBatches(index, scene, allowance, [&](const std::vector<SceneRawValue>& batch) {
        const size_t tasks = (batch.size() + 2047) / 2048;
        std::vector<Quality> sums(tasks); for (auto& q : sums) q.finite_cost = 0;
        std::vector<std::map<uint32_t, PhotoQuality>> local(photos ? tasks : 0);
        for (size_t task = 0; task < tasks; ++task) for (size_t i = task * 2048; i < std::min(batch.size(), (task + 1) * 2048); ++i) {
            const auto& value = batch[i]; const auto& image = scene.poses().images.at(value.row.image); const auto& camera = scene.poses().cameras.at(image.camera_id);
            require(image.registered && camera.pixel_scale > 0, "serial raw view invalid"); auto& q = sums[task]; ++q.observations;
            if (value.kind == LandmarkKind::RetainedObservation) { ++q.retained_observations; if (photos) ++local[task][value.row.image].retained_observations; continue; }
            q.directional_observations += value.kind == LandmarkKind::UnitDirection;
            bool valid = true; const double error = reprojLandmarkError(camera, image.pose, value.row.xy, {value.xyz, value.kind}, &valid) / camera.pixel_scale;
            require(std::isfinite(error), "serial residual nonfinite"); const double cost = error <= gate ? error * error : gate * (2 * error - gate);
            q.cost += cost; if (valid) q.finite_cost += cost; else ++q.invalid_projections;
            if (photos) { auto& p = local[task][value.row.image]; ++p.observations; p.inliers += valid && error <= gate; p.invalid_projections += !valid; p.error += error; }
        }
        for (const auto& q : sums) accumulateQuality(result, q);
        if (photos) for (const auto& task : local) for (const auto& item : task) { auto& p = photos->at(item.first); p.observations += item.second.observations; p.retained_observations += item.second.retained_observations; p.inliers += item.second.inliers; p.invalid_projections += item.second.invalid_projections; p.error += item.second.error; }
    }, false, statistics);
    return result;
}
static Quality legacyParallel(const ImmutableSceneTopology& index, const CanonicalScene& scene, double gate, size_t allowance, int threads) {
    struct View { const Image* image = nullptr; const Camera* camera = nullptr; };
    std::vector<View> views(size_t(index.imageSlots()));
    for (const auto& item : scene.poses().images) if (item.first < views.size()) views[item.first] = {&item.second, &scene.poses().cameras.at(item.second.camera_id)};
    auto& pool = bacpu::Pool::get(); const int workers = threads > 0 ? std::min(threads, pool.size()) : pool.size();
    Quality result; result.finite_cost = 0;
    visitSceneRawBatches(index, scene, allowance, [&](const auto& batch) {
        const int tasks = int((batch.size() + 2047) / 2048); std::vector<Quality> sums(tasks); for (auto& q : sums) q.finite_cost = 0;
        pool.runChecked(tasks, workers, [&](int task, int) {
            auto& q = sums[task]; const auto end = std::min(batch.size(), (size_t(task) + 1) * 2048);
            for (size_t i = size_t(task) * 2048; i < end; ++i) {
                const auto& value = batch[i]; const auto& view = views.at(value.row.image);
                require(view.image && view.camera && view.image->registered && view.camera->pixel_scale > 0, "legacy raw view invalid"); ++q.observations;
                if (value.kind == LandmarkKind::RetainedObservation) { ++q.retained_observations; continue; }
                q.directional_observations += value.kind == LandmarkKind::UnitDirection;
                bool valid = true; const double error = reprojLandmarkError(*view.camera, view.image->pose, value.row.xy, {value.xyz, value.kind}, &valid) / view.camera->pixel_scale;
                require(std::isfinite(error), "legacy residual nonfinite"); const double cost = error <= gate ? error * error : gate * (2 * error - gate);
                q.cost += cost; if (valid) q.finite_cost += cost; else ++q.invalid_projections;
            }
        });
        for (const auto& q : sums) accumulateQuality(result, q);
    }, false);
    return result;
}
static Vec3 value(uint64_t point) { if (point == 0) return {.1, .1, 8}; if (point == 1) return {.2, .1, 8}; if (point == 2) return {1, 0, 1}; return {999, -888, -3}; }
static Reconstruction fixture() {
    Reconstruction r; r.cameras[4] = Camera::defaultFor(4, 1000, 800, 700, CamModel::OpenCV); r.cameras[4].pixel_scale = 2;
    for (uint32_t id = 0; id < 3; ++id) {
        Image image; image.id = id; image.camera_id = 4; image.name = "image-" + std::to_string(id); image.registered = true;
        image.pose = {mat3Identity(), {-double(id) * .5, 0, 0}}; image.points2D.resize(4); image.point3D_ids.assign(4, kInvalidPoint3D);
        for (uint32_t point = 0; point < 4; ++point) image.points2D[point] = r.cameras.at(4).project(point == 2 ? value(point) : value(point) + image.pose.t);
        r.images.emplace(id, std::move(image));
    }
    for (uint32_t point = 0; point < 4; ++point) {
        Point3D p; p.xyz = value(point); const uint32_t repetitions = point < 2 ? 35000 : 500;
        if (point < 3) for (uint32_t repeat = 0; repeat < repetitions; ++repeat) { p.track.push_back({0, point}); p.track.push_back({1, point}); }
        else for (uint32_t repeat = 0; repeat < 1000; ++repeat) p.track.push_back({2, point});
        for (const auto& row : p.track) r.images.at(row.image_id).point3D_ids[row.point2D_idx] = point;
        r.points3D.emplace(point, std::move(p));
    }
    r.next_point3D_id = 4; return r;
}
static Reconstruction poses(const RegionalWorkingStore& catalog) {
    auto r = catalog.poses(); r.points3D.clear(); for (auto& image : r.images) { image.second.points2D.clear(); image.second.point3D_ids.clear(); } return r;
}
static void initialize(CanonicalScene& scene, const RegionalWorkingStore& catalog, const ImmutableSceneTopology& index,
                       const Reconstruction& p, bool wrongAlias = false, bool retainedDirection = false) {
    scene.beginLayoutCandidate(CanonicalScene::bindingFor(catalog, index.identity()), p,
        [&](const auto& emit) { index.emitAliases([&](auto key, uint64_t root) { emit(key, wrongAlias && key.source == 1 ? 4 : root); }); },
        [&](const auto& emit) { for (uint64_t i = 0; i < 4 + uint64_t(wrongAlias); ++i) emit(i, value(i == 4 ? 1 : i)); });
    if (!retainedDirection) scene.applyLandmarkTypes({{2, {std::sqrt(.5), 0, std::sqrt(.5)}}});
    scene.applyRetainedObservations(retainedDirection ? std::vector<uint64_t>{2, 3} : std::vector<uint64_t>{3});
}
static uint64_t processReads() {
#ifdef _WIN32
    IO_COUNTERS io{}; if (!GetProcessIoCounters(GetCurrentProcess(), &io)) throw std::runtime_error("process IO counters unavailable"); return io.ReadTransferCount;
#else
    return 0;
#endif
}
static double processCpuSeconds() {
#ifdef _WIN32
    FILETIME created{}, ended{}, kernel{}, user{};
    require(GetProcessTimes(GetCurrentProcess(), &created, &ended, &kernel, &user), "process CPU counters unavailable");
    return (double((uint64_t(kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime) + double((uint64_t(user.dwHighDateTime) << 32) | user.dwLowDateTime)) / 1e7;
#else
    return 0;
#endif
}
static int real(int argc, char** argv) {
    require(argc >= 7, "--real <scene-root> <index-root> <new-output> <allowance-MiB> <threads> [--gpu] [--reverse]");
    const Path sceneRoot = std::filesystem::absolute(std::filesystem::u8path(argv[2])); const Path indexRoot = std::filesystem::absolute(std::filesystem::u8path(argv[3]));
    const Path output = std::filesystem::absolute(std::filesystem::u8path(argv[4])); require(!std::filesystem::exists(output), "real benchmark output must be new");
    auto contains = [](const Path& parent, const Path& child) { const auto rel = child.lexically_relative(parent); return rel.empty() || (!rel.is_absolute() && !rel.empty() && *rel.begin() != ".."); };
    require(!contains(sceneRoot, output) && !contains(indexRoot, output) && !contains(output, sceneRoot) && !contains(output, indexRoot), "real benchmark output overlaps native evidence");
    const auto mib = std::stoull(argv[5]); require(mib >= 16 && mib <= SIZE_MAX / (1u << 20), "invalid benchmark allowance"); const size_t allowance = size_t(mib) << 20;
    const int threads = std::stoi(argv[6]); require(threads >= 0, "invalid benchmark threads");
    std::filesystem::create_directories(output);
    std::string originalHead;
    { CanonicalScene source(sceneRoot, allowance, CanonicalScene::OpenMode::ReadOnly); originalHead = sha(sceneRoot / "head.bin"); source.forkAccepted(output / "pinned-scene"); }
    CanonicalScene scene(output / "pinned-scene", allowance, CanonicalScene::OpenMode::ReadOnly);
    const auto pinnedHead = sha(output / "pinned-scene/head.bin"); require(pinnedHead == originalHead, "native accepted fork changed checkpoint identity");
    ImmutableSceneTopology index(indexRoot, allowance); index.restore(scene.binding().topology);
    std::ofstream log(output / "quality-benchmark.txt"); require(bool(log), "cannot write owned benchmark report"); log << std::setprecision(17);
    const auto hardware = std::max(1u, std::thread::hardware_concurrency());
    const int poolThreads = bacpu::Pool::get().size(), effectiveThreads = threads > 0 ? std::min(threads, poolThreads) : poolThreads;
    log << "requested_threads=" << threads << "\npool_threads=" << poolThreads << "\neffective_threads=" << effectiveThreads
        << "\nstream_residual_threads=" << (effectiveThreads > 1 ? effectiveThreads - 1 : 1) << "\nhardware_threads=" << hardware
        << "\nthread_override=" << (spirula::env("SFM_BA_THREADS") ? spirula::env("SFM_BA_THREADS") : "unset")
        << "\ncache_protocol=fresh_native_scene_per_variant_then_same_scene_warm_repeat;strict_open_outside_timer;serial_raw_prewarm_first\n";
    auto measure = [&](const char* name, auto action) {
        CanonicalScene measured(output / "pinned-scene", allowance, CanonicalScene::OpenMode::ReadOnly);
        require(measured.identity() == scene.identity(), "benchmark variant opened a different pinned snapshot");
        Quality first;
        for (int repeat = 0; repeat < 2; ++repeat) {
            const std::string label = std::string(name) + (repeat ? ".warm" : "");
            measured.resetCounters(); const auto reads = processReads(); const auto cpuStarted = processCpuSeconds(); const auto started = std::chrono::steady_clock::now(); auto q = action(measured);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); const auto io = measured.counters();
            log << label << ".seconds=" << seconds << '\n' << label << ".process_logical_read_bytes=" << processReads() - reads << '\n' << label << ".raw=" << q.observations << '\n'
                << label << ".cpu_percent_whole_machine=" << 100 * (processCpuSeconds() - cpuStarted) / seconds / hardware << '\n'
                << label << ".invalid=" << q.invalid_projections << '\n' << label << ".finite_cost=" << q.finite_cost << '\n' << label << ".cost=" << q.cost << '\n'
                << label << ".retained=" << q.retained_observations << '\n' << label << ".directional=" << q.directional_observations << '\n' << label << ".alias_bytes=" << io.aliasBytes << '\n'
                << label << ".point_read_bytes=" << io.pointReadBytes << '\n' << label << ".file_opens=" << io.fileOpens << '\n'; log.flush();
            if (!repeat) first = q; else require(same(first, q), "fresh/warm benchmark changed deterministic typed raw objective");
        }
        return first;
    };
    const auto old = measure("serial_reference", [&](const auto& measured) { return serialReference(index, measured, 3, allowance); });
    bool gpuRequested = false, reverse = false; for (int i = 7; i < argc; ++i) { const std::string flag = argv[i]; if (flag == "--gpu") gpuRequested = true; else if (flag == "--reverse") reverse = true; else throw std::invalid_argument("unknown benchmark option"); }
    Quality legacy, cpu;
    auto runLegacy = [&] { legacy = measure("legacy_parallel_cpu", [&](const auto& measured) { return legacyParallel(index, measured, 3, allowance, threads); }); };
    auto runStream = [&] { cpu = measure("stream_swap_pipeline_cpu", [&](const auto& measured) { return sceneQuality(index, measured, 3, allowance, threads); }); };
    if (reverse) { runStream(); runLegacy(); } else { runLegacy(); runStream(); }
    require(same(old, legacy), "legacy production parallel path changed the serial objective oracle");
    require(same(old, cpu), "real CPU pipeline changed complete deterministic raw objective");
    if (gpuRequested) {
        GpuRawQuality gpu; SceneQualityEvaluation evaluation;
        const auto device = measure("gpu_quality", [&](const auto& measured) { evaluation = evaluateSceneQuality(index, measured, 3, allowance, threads, nullptr, {}, nullptr, &gpu); return evaluation.quality; });
        require(evaluation.complete && old.observations == device.observations && old.invalid_projections == device.invalid_projections && old.retained_observations == device.retained_observations && old.directional_observations == device.directional_observations &&
            std::abs(old.cost-device.cost) <= std::max(1e-8, old.cost*1e-8) && std::abs(old.finite_cost-device.finite_cost) <= std::max(1e-8, old.finite_cost*1e-8), "real GPU quality differs from complete CPU oracle");
        log << "gpu_rows=" << evaluation.gpuRows << "\ncpu_fallback_rows=" << evaluation.cpuRows << '\n';
    }
    require(sha(output / "pinned-scene/head.bin") == pinnedHead, "real readonly benchmark changed pinned accepted head");
    log << "source_root=" << sceneRoot.u8string() << "\nsource_head_at_pin=" << originalHead << "\nsource_head_at_finish=" << sha(sceneRoot / "head.bin") << "\npinned_head=" << pinnedHead << "\npinned_head_unchanged=true\n";
    log << "passed=true\nsource_readonly=true\nfull_capture_verified=false\ncomparison_order=serial_reference_then_" << (reverse ? "stream_then_legacy" : "legacy_then_stream") << "_each_fresh_and_warm\n"; log.flush();
    std::cout << "PASS real readonly complete raw quality; pinned accepted head unchanged; source head progress is informational; no capture acceptance claim\n"; return 0;
}
int main(int argc, char** argv) try {
    if (argc > 1 && (std::string(argv[1]) == "--real" || std::string(argv[1]) == "--existing")) return real(argc, argv);
    const Path root = argc > 1 ? std::filesystem::u8path(argv[1]) : std::filesystem::temp_directory_path() / ("scene-quality-pipeline-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(!std::filesystem::exists(root), "fixture output must be new"); std::filesystem::create_directories(root);
    constexpr size_t allowance = 64u << 20; const auto input = fixture(); const auto source = root / "source.bin"; model_store_detail::writeModel(input, source); const auto immutable = sha(source);
    RegionalWorkingStore catalog(root / "catalog", allowance); catalog.importModel(source, input, "source");
    ImmutableSceneAliases aliases(root / "aliases", allowance); aliases.prepare(catalog, {0, 1, 2}); ImmutableSceneTopology index(root / "indices", allowance);
    index.prepare(catalog, {0, 1, 2}, aliases.identity(), aliases.points(), [&](auto key) { return aliases.canonical(key); });
    CanonicalScene scene(root / "scene", allowance); initialize(scene, catalog, index, poses(catalog)); scene.commit("quality-pipeline-fixture");
    const auto head = sha(root / "scene/head.bin"), accepted = scene.identity();
    CanonicalParameterView roomy(scene, 1u << 20); require(roomy.flat(), "small canonical view did not use a bounded cache");
    const auto bitmap = storage::scaleBytes(scene.pointSlots() / 64 + !!(scene.pointSlots() % 64), sizeof(uint64_t));
    const auto exactViewBudget = storage::addBytes(roomy.bytes(), storage::addBytes(bitmap, 3u << 16));
    scene.resetCounters(); CanonicalParameterView tooSmall(scene, size_t(exactViewBudget - 1));
    require(!tooSmall.flat() && !tooSmall.bytes() && !scene.counters().aliasBytes && !scene.counters().pointReadBytes, "below-boundary parameter cache allocated/read before scratch was admitted");
    CanonicalParameterView exactView(scene, size_t(exactViewBudget)); require(exactView.flat() && exactView.bytes() == roomy.bytes(), "three-reader scratch exact boundary did not fit");
    std::vector<PhotoQuality> referencePhotos, parallelPhotos; SceneRawBatchStatistics slow, fast;
    const auto reference = serialReference(index, scene, 3, allowance, &referencePhotos, &slow);
    const auto mainThread = std::this_thread::get_id(); uint64_t lastProgress = 0;
    const auto parallel = sceneQuality(index, scene, 3, allowance, 0, &parallelPhotos, [&](const char*, uint64_t done, uint64_t) { require(std::this_thread::get_id() == mainThread && done >= lastProgress, "progress escaped the producer or became unordered"); lastProgress = done; });
    require(same(reference, parallel) && reference.observations == 142000 && reference.retained_observations == 1000 && reference.directional_observations == 1000 && !reference.invalid_projections, "pipeline changed full typed original raw multiset/objective");
    require(referencePhotos.size() == parallelPhotos.size(), "photo table shape differs"); for (size_t i = 0; i < referencePhotos.size(); ++i) require(same(referencePhotos[i], parallelPhotos[i]), "parallel photo reduction is not bitwise deterministic");
    std::vector<PhotoQuality> sixteenPhotos; const auto sixteen = sceneQuality(index, scene, 3, 16u << 20, 0, &sixteenPhotos);
    require(same(reference, sixteen) && sixteenPhotos.size() == referencePhotos.size(), "16 MiB dynamic photo batches changed complete typed raw or reduction order");
    for (size_t i = 0; i < sixteenPhotos.size(); ++i) require(same(sixteenPhotos[i], referencePhotos[i]), "16 MiB photo table changed typed counts or exact reductions");
    std::vector<PhotoQuality> tightPhotos; const auto tight = sceneQuality(index, scene, 3, 2u << 20, 0, &tightPhotos);
    require(tight.observations == reference.observations && tight.retained_observations == reference.retained_observations && tight.directional_observations == reference.directional_observations &&
        tight.invalid_projections == reference.invalid_projections && std::abs(tight.cost-reference.cost) < 1e-12 && std::abs(tight.finite_cost-reference.finite_cost) < 1e-12, "2 MiB raw-reader budget omitted rows or changed typed objective");
    for (size_t i = 0; i < tightPhotos.size(); ++i) require(tightPhotos[i].observations == referencePhotos[i].observations && tightPhotos[i].retained_observations == referencePhotos[i].retained_observations && tightPhotos[i].inliers == referencePhotos[i].inliers && tightPhotos[i].invalid_projections == referencePhotos[i].invalid_projections && std::abs(tightPhotos[i].error-referencePhotos[i].error) < 1e-12, "2 MiB photo counts/residuals changed");
    rejects([&] { sceneQuality(index, scene, 3, (2u << 20)-1, 0, &tightPhotos); }, "below-minimum raw-reader workspace was silently admitted");
    uint64_t visited = 0; visitSceneRawBatches(index, scene, allowance, [&](const auto& rows) { visited += rows.size(); }, true, &fast);
    require(!slow.flatParameters && fast.flatParameters && fast.aliasRows == scene.pointSlots() && fast.landmarkRoots == scene.canonicalPoints() && fast.rawRows == visited && visited == reference.observations, "flat cache omitted source aliases/canonical/raw coverage");
    require(same(reference, sceneQuality(index, scene, 3, allowance, 1)), "single-thread objective changed");
    scene.resetCounters();
    const auto acceptedEvaluation = evaluateSceneQuality(index, scene, 3, allowance, 0, nullptr, {}, &reference);
    require(acceptedEvaluation.complete && acceptedEvaluation.rejected.empty() && same(reference, acceptedEvaluation.quality) && acceptedEvaluation.cpuRows == reference.observations, "acceptable candidate did not complete full validation");
    require(scene.counters().aliasBytes == 0 && scene.counters().pointReadBytes == scene.canonicalPoints() * sizeof(Vec3), "stream quality rebuilt flat aliases or reread every source slot instead of one parameter per canonical root");
    require(bacpu::Pool::get().size() == 1 ? acceptedEvaluation.rawBatchSwaps == 0 : acceptedEvaluation.rawBatchSwaps >= 3, "multiple full raw batches did not reuse the swapped worker buffer");
    const auto inlineEvaluation = evaluateSceneQuality(index, scene, 3, allowance, 1);
    require(inlineEvaluation.complete && !inlineEvaluation.rawBatchSwaps && same(reference, inlineEvaluation.quality), "single-worker path entered asynchronous buffer ownership or changed complete quality");
    auto malformed = reference; --malformed.observations; rejects([&] { evaluateSceneQuality(index, scene, 3, allowance, 0, nullptr, {}, &malformed); }, "bounded evaluator accepted a different raw cohort");
    scene.beginCandidate(); scene.applyPointUpdates({{1, {.2, .1, -8}}});
    const auto invalid = evaluateSceneQuality(index, scene, 3, allowance, 0, nullptr, {}, &reference);
    require(!invalid.complete && invalid.rejected == "invalid_projections_increased" && invalid.quality.observations < reference.observations && invalid.quality.invalid_projections, "late invalid point was not explicitly rejected with partial coverage");
    auto badBaseline = sceneQuality(index, scene, 3, allowance, 0); badBaseline.cost *= .5; scene.applyPointUpdates({{1, {100, .1, -8}}});
    const auto invalidCost = evaluateSceneQuality(index, scene, 3, allowance, 0, nullptr, {}, &badBaseline);
    require(!invalidCost.complete && invalidCost.rejected == "visual_cost_increased", "independent total-cost cutoff was confused with finite cost"); scene.abort("invalid-cost-diagnostic");
    scene.beginCandidate(); scene.applyPointUpdates({{1, {.21, .1, 8}}}); const auto finite = evaluateSceneQuality(index, scene, 3, allowance, 0, nullptr, {}, &reference);
    require(!finite.complete && finite.rejected == "finite_visual_cost_increased" && finite.quality.observations < reference.observations, "late finite objective increase was not bounded"); scene.abort("finite-cost-diagnostic");
    std::atomic<bool> cancelled{false}; cancel::set_token(&cancelled); bool caughtCancel = false;
    try { sceneQuality(index, scene, 3, allowance, 0, nullptr, [&](const char*, uint64_t, uint64_t) { cancelled.store(true); }); } catch (const Cancelled&) { caughtCancel = true; }
    cancel::set_token(nullptr); require(caughtCancel && same(reference, sceneQuality(index, scene, 3, allowance, 0)), "cancellation left an unjoined or unusable quality worker");
    rejects([&] { sceneQuality(index, scene, 3, allowance, 0, nullptr, [](const char*, uint64_t, uint64_t) { throw std::runtime_error("fixture progress failure"); }); }, "progress exception was swallowed");
    auto unregistered = poses(catalog); unregistered.images.at(1).registered = false; CanonicalScene invalidView(root / "invalid-view", allowance);
    initialize(invalidView, catalog, index, unregistered); rejects([&] { sceneQuality(index, invalidView, 3, allowance, 0); }, "consumer exception was swallowed or deadlocked"); invalidView.abort("invalid-view-diagnostic");
    auto duplicate = input; for (auto it = duplicate.points3D.begin(); it != duplicate.points3D.end();) if (it->first != 1) it = duplicate.points3D.erase(it); else ++it;
    for (auto& item : duplicate.images) for (size_t i = 0; i < item.second.point3D_ids.size(); ++i) if (i != 1) item.second.point3D_ids[i] = kInvalidPoint3D;
    duplicate.points3D.at(1).track = {{0, 1}, {1, 1}}; model_store_detail::writeModel(duplicate, root / "duplicate-source.bin"); catalog.importModel(root / "duplicate-source.bin", duplicate, "duplicate-source");
    ImmutableSceneAliases joinedAliases(root / "aliases", allowance); joinedAliases.prepare(catalog, {0, 1, 2}); ImmutableSceneTopology joined(root / "indices", allowance);
    joined.prepare(catalog, {0, 1, 2}, joinedAliases.identity(), joinedAliases.points(), [&](auto key) { return joinedAliases.canonical(key); });
    CanonicalScene badAlias(root / "bad-alias", allowance); initialize(badAlias, catalog, joined, poses(catalog), true);
    rejects([&] { sceneQuality(joined, badAlias, 3, allowance, 0); }, "flat cache accepted a mismatched original local/root alias"); badAlias.abort("bad-alias-diagnostic");
    CanonicalScene badRetained(root / "bad-retained", allowance); initialize(badRetained, catalog, joined, poses(catalog), false, true);
    rejects([&] { sceneQuality(joined, badRetained, 3, allowance, 0); }, "typed retained rows with two photos passed producer validation"); badRetained.abort("bad-retained-diagnostic");
    require(sha(source) == immutable && sha(root / "scene/head.bin") == head && scene.identity() == accepted && same(reference, sceneQuality(index, scene, 3, allowance, 0)), "quality diagnostics changed immutable source/accepted parameters");
    std::cout << "PASS typed full raw pipeline: exact serial/parallel/photo reduction, flat alias binding, bounded rejection, cancellation/producer/consumer/progress failures; no geometry success claim\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
