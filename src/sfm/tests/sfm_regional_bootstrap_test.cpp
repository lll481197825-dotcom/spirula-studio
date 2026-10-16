#include "sfm/Pipeline.h"
#include "sfm/map/RegionalBootstrap.h"
#include "sfm/map/RegionalAlignment.h"
#include "sfm/map/RegionalTransform.h"
#include "sfm/map/RegionalRepairCache.h"
#include "sfm/tests/TestMain.h"
#include <cstdio>
#include <iomanip>

using namespace sfm;
namespace fs = std::filesystem;
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct CachedPositions : PriorSource {
    std::vector<Vec3> values;
    bool has(uint32_t) const override { return false; }
    bool relativeRotation(uint32_t, uint32_t, Mat3&, double&) const override { return false; }
    std::vector<uint32_t> neighbours(uint32_t) const override { return {}; }
    PosePriors factors(const std::vector<PosedImage>&) override { return {}; }
    bool position(uint32_t id, Vec3& value) const override {
        if (id >= values.size()) return false; value = values[id]; return true;
    }
};

static Reconstruction fixture() {
    Reconstruction rec; rec.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::Pinhole);
    for (uint32_t i = 0; i < 4; ++i) {
        Image im; im.id = i; im.camera_id = 0; im.registered = true; im.pose.t = {-.5 * i, 0, 0};
        im.points2D.resize(30); im.point3D_ids.assign(30, kInvalidPoint3D); rec.images[i] = im;
    }
    for (uint64_t k = 0; k < 30; ++k) {
        Point3D p; p.xyz = {.05 * double(k), .1 * (k % 5), 5 + .01 * k};
        for (auto& kv : rec.images) {
            auto& im = kv.second; im.points2D[k] = rec.cameras[0].project(p.xyz + im.pose.t);
            im.point3D_ids[k] = k; p.track.push_back({im.id, uint32_t(k)});
        }
        rec.points3D[k] = p;
    }
    rec.next_point3D_id = 30; return rec;
}

static int cacheProbe(int argc, char** argv) {
    require(argc == 5, "cache probe requires donor model, context model and report directory");
    const fs::path donor_path = argv[2]; const auto context_views = model_store_detail::readModel(argv[3], false);
    const fs::path output = argv[4]; fs::create_directories(output);
    regional::Region context; for (const auto& kv : context_views.images) context.images.push_back(kv.first);
    const auto begin = now(); auto donor = model_store_detail::readModel(donor_path);
    auto expected = regional::anchoredRepairSeed(donor, context, donor);
    const auto full_seconds = now() - begin;
    regional::RepairModelCache cache(size_t(1) << 30); cache.sync({donor_path});
    auto poses = model_store_detail::readModel(donor_path, false); donor = {};
    const auto cold = now(); auto subset = cache.subset(0, context.images);
    auto actual = regional::anchoredRepairSeed(subset, context, poses);
    const auto cold_seconds = now() - cold;
    regional::RepairDigest a, b; model_store_detail::serializeModel(expected, a); model_store_detail::serializeModel(actual, b);
    require(a.result() == b.result(), "indexed repair seed changed real geometry, feature identities or observation order");
    const auto warm = now();
    for (size_t k = 0; k < 5; ++k) {
        auto next = cache.subset(0, context.images); auto seed = regional::anchoredRepairSeed(next, context, poses);
        regional::RepairDigest check; model_store_detail::serializeModel(seed, check); require(check.result() == a.result(), "warm cache changed seed");
    }
    const auto warm_seconds = (now() - warm) / 5;
    std::ofstream report(output / "cache-probe.txt");
    report << std::setprecision(17) << "full_read_and_seed_seconds=" << full_seconds << "\ncold_index_and_seed_seconds=" << cold_seconds
        << "\nwarm_subset_and_seed_seconds=" << warm_seconds << "\nindex_bytes=" << cache.bytes() << "\nheader_reads=" << cache.header_reads
        << "\nindex_builds=" << cache.index_builds << "\nindex_hits=" << cache.index_hits << "\nseed_images=" << actual.images.size()
        << "\nseed_points=" << actual.points3D.size() << "\nseed_serialized_identity=" << a.result() << "\nfull_capture_verified=false\n";
    report.flush(); require(bool(report), "cannot write cache replay evidence");
    std::printf("PASS real donor cache: full %.3fs cold %.3fs warm %.3fs, %zu index bytes; exact seed identity %s\n",
        full_seconds, cold_seconds, warm_seconds, cache.bytes(), a.result().c_str()); return 0;
}

static void cacheFixture() {
    auto rec = fixture(); rec.points3D[50] = rec.points3D.at(0); rec.next_point3D_id = 51;
    const auto path = fs::temp_directory_path() / ("regional-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    model_store_detail::writeModel(rec, path);
    regional::RepairModelCache cache(1u << 20); cache.sync({path});
    regional::Region context; context.core = {3}; context.images = {0,1,2};
    const auto ranked = cache.rank(context, {0,1,2}); require(ranked == std::vector<std::pair<size_t,size_t>>{{3,0}}, "catalog changed donor votes");
    const auto full = model_store_detail::readModel(path); const auto expected = regional::anchoredRepairSeed(full, context, full);
    const auto subset = cache.subset(0, context.images); const auto actual = regional::anchoredRepairSeed(subset, context, full);
    regional::RepairDigest a, b; model_store_detail::serializeModel(expected, a); model_store_detail::serializeModel(actual, b);
    require(a.result() == b.result(), "subset lost duplicate raw observations or changed a feature assignment");
    cache.sync({path}); cache.subset(0, context.images);
    require(cache.header_reads == 1 && cache.index_builds == 1 && cache.index_hits == 1, "unchanged model was rescanned");
    auto next = full; next.points3D.at(1).xyz.x += .001; model_store_detail::writeModel(next, path);
    fs::last_write_time(path, fs::last_write_time(path) + std::chrono::seconds(1)); cache.sync({path});
    auto changed = cache.subset(0, context.images);
    require(changed.points3D.at(1).xyz.x == next.points3D.at(1).xyz.x && cache.header_reads == 2 && cache.index_builds == 2,
        "replaced donor reused stale offsets or geometry");
    const auto stamp = regional::repairGeometryIdentity(context, full, &expected);
    auto moved = full; moved.images.at(1).pose.t.x += .001;
    require(stamp != regional::repairGeometryIdentity(context, moved, &expected), "failure memo ignores anchor pose changes");
    auto modified = expected; modified.points3D.at(1).xyz.x += .001;
    require(stamp != regional::repairGeometryIdentity(context, full, &modified), "failure memo ignores secondary seed geometry");
    auto required = context; required.core = {2};
    require(stamp != regional::repairGeometryIdentity(required, full, &expected), "failure memo conflates required photo queues");
    MatchesIndex pairs; pairs.images.resize(9); regional::RepairPairCatalog graph;
    for (uint32_t i = 0; i < 8; ++i) { MatchesIndex::Entry e{}; e.image1 = i; e.image2 = i + 1; e.count = 20 + i; pairs.pairs.push_back(e); }
    for (size_t limit : {2u,4u,8u}) {
        const auto a = regional::repairContext(pairs, {0,1}, {3,5,7}, limit);
        const auto b = regional::repairContext(pairs, {0,1}, {3,5,7}, limit, &graph);
        require(a.images == b.images && a.core == b.core, "indexed repair graph changed support selection");
    }
    MatchesIndex::Entry e{}; e.image1 = 0; e.image2 = 7; e.count = 100; pairs.pairs.push_back(e);
    require(regional::repairContext(pairs, {0,1}, {3,5,7}, 4).images == regional::repairContext(pairs, {0,1}, {3,5,7}, 4, &graph).images,
        "new verified repair pair not added to graph dependencies");
    fs::remove(path);
}

static int probe(int argc, char** argv) {
    require(argc == 7 || argc == 9 || argc == 10, "repair probe requires matches, features, weak model, donor, scratch, optional second donor, cached GPS and source photos");
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    MatchesIndex index; require(indexMatches(argv[2], index) && index.complete, "real matches incomplete");
    auto original = model_store_detail::readModel(argv[4]); auto donor = model_store_detail::readModel(argv[5]);
    const auto required = regional::registeredIds(original);
    CameraSetup cameras; loadCameraSetup(index.metadata, cameras);
    SfmConfig cfg; cfg.quality = "medium"; cfg.quiet = true; cfg.threads = 8; cfg.mapper.threads = 8;
    cfg.mapper.ba_solver = "cg"; cfg.mapper.ba_real = "double";
    std::vector<PresetChange> changes; applyPresets(cfg, {}, changes);
    const fs::path scratch = argv[6]; fs::create_directories(scratch);
    Reconstruction poses = donor; std::optional<Reconstruction> extra; CachedPositions gps;
    if (argc >= 9) {
        extra = model_store_detail::readModel(argv[7]); regional::SharedAlignmentReport alignment;
        if (std::string(argv[1]) != "--coordinated-repair-probe")
            require(regional::alignRegional(*extra, donor, gps, cfg.max_error, true, &alignment), "second donor has no verified common frame");
        for (const auto& kv : extra->images) poses.images.emplace(kv.first, kv.second);
        model_store_detail::Reader input(argv[8]); const auto n = input.count(28); gps.values.resize(n);
        for (size_t k = 0; k < n; ++k) { const auto id = input.scalar<uint32_t>(); const auto p = input.scalar<Vec3>(); gps.values.at(id) = p; } input.finish();
        require(gps.values.size() == index.images.size(), "cached GPS does not cover actual capture");
        regional::loadRepairMatches(index, scratch / "supplement");
    }
    std::unique_ptr<PriorSource> actual_priors;
    if (argc == 10) {
        cfg.sensor_pairs = true; cfg.metric_gps = "full";
        actual_priors = makeExifGpsPriors(cfg, argv[9], index.metadata, cameras, true);
        require(bool(actual_priors), "actual source GPS unavailable");
    }
    const auto anchors = regional::registeredIds(poses);
    std::string previous_context;
    for (size_t limit : {64u, 128u, 256u, 512u, 1024u}) {
        if (argc >= 9 && limit > 64) regional::supplementRepairMatches(cfg, index, required, anchors, gps.values, cameras,
            argv[3], scratch / "supplement", std::min<size_t>(256, limit / 2), size_t(6) << 30);
        auto context = regional::repairContext(index, required, anchors, limit);
        const auto position_gate = std::string(argv[1]) == "--coordinated-repair-probe" ? &gps : nullptr;
        auto seed = regional::anchoredRepairSeed(donor, context, poses, cfg.max_error, position_gate);
        if (extra) regional::mergeAnchoredRepairSeed(seed, regional::anchoredRepairSeed(*extra, context, poses, cfg.max_error, position_gate));
        std::string context_key; for (uint32_t id : context.images) context_key += std::to_string(id) + ",";
        context_key += std::to_string(index.pairs.size()) + ":" + std::to_string(seed.points3D.size());
        if (context_key == previous_context) { std::puts("ANCHOR_REPAIR identical support skipped"); continue; } previous_context = context_key;
        std::printf("ANCHOR_REPAIR required=%zu support=%zu verified_seed_views=%u points=%zu limit=%zu\n",
            required.size(), context.images.size() - required.size(), seed.numRegistered(), seed.points3D.size(), limit);
        require(seed.numRegistered() >= 3 && !seed.points3D.empty(), "no validated donor seed");
        auto models = regional::reconstructRepair(cfg, index, context, argv[2], argv[3], "", cameras, actual_priors.get(), &seed, size_t(6) << 30, true);
        for (size_t m = 0; m < models.size(); ++m) {
            auto& rec = models[m]; regional::atomicModel(scratch / ("candidate-" + std::to_string(limit) + "-" + std::to_string(m) + ".bin"), rec);
            std::printf("ANCHOR_REPAIR candidate=%zu registered=%u all_required=%d\n", m, rec.numRegistered(), int(regional::coversImages({rec}, required)));
            if (!regional::coversImages({rec}, required)) continue;
            const auto count = regional::preserveRepairObservations(original, rec, cfg.max_error);
            regional::atomicModel(scratch / "rehabilitated.bin", rec);
            std::printf("PASS actual anchored repair, %zu original photos and %llu observations retained; GPS/global/export acceptance pending\n",
                required.size(), (unsigned long long)count); return 0;
        }
    }
    throw std::runtime_error("real anchored repair unresolved; candidate evidence retained");
}

static int test(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--repair-cache-probe") return cacheProbe(argc, argv);
    cacheFixture();
    if (argc > 1 && (std::string(argv[1]) == "--anchor-repair-probe" || std::string(argv[1]) == "--coordinated-repair-probe")) return probe(argc, argv);
    regional::SharedIndex index; index.models = 3;
    for (uint64_t k = 0; k < 80; ++k) {
        Vec3 p{double(k % 7), double(k % 11), double(k % 13)};
        regional::SharedLandmark l; l.key = k; l.xyz = p; l.members = {{0,k,p},{1,k,p + Vec3{3,0,0}}}; index.landmarks.push_back(l);
    }
    auto strict = index; bool rejected = false;
    try { regional::alignSharedBlocks(strict); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "strict graph silently dropped a model");
    const auto fit = regional::alignSharedBlocks(index, 50, 256u << 20, {}, {{2,{0,0,0},{1e6,0,0}}}, true);
    require(fit.deferred == std::vector<size_t>{2}, "wrong pending model set");
    require(std::abs(fit.transforms[1].t.x + 3) < 1e-5, "unobservable GPS target contaminated backbone");
    require(fit.transforms[2].scale == 1 && fit.transforms[2].t.norm() < 1e-8, "deferred model got a fabricated transform");
    auto retained = regional::retainSharedModels(index, {2}, {}); require(retained.models == 2 && retained.landmarks.size() == 80, "filtered graph lost the reliable component");

    auto original = fixture(); regional::Region context; context.images = {0,1,2}; context.core = {3};
    CachedPositions metric; metric.values = {{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
    regional::FixedMetricPriorSource fixed_metric(metric);
    const auto factors = fixed_metric.factors({{0,0,Pose{}},{1,0,Pose{}}});
    require(factors.gps.ok && factors.gps.A == mat3Identity() && factors.gps.t.norm() == 0 && factors.centres.size() == 2 &&
        (factors.centres[1].b - metric.values[1]).norm() == 0, "anchored repair refitted its established metric frame");
    std::vector<Vec3> points, bearings;
    for (size_t k = 0; k < 90; ++k) {
        const auto p = original.points3D.at(k % 30).xyz; points.push_back(p + (k < 30 ? Vec3{} : Vec3{20,0,0})); bearings.push_back(p.normalized());
    }
    const auto unlimited = ransacPnP(points, bearings, 500, 3, 4);
    const auto bounded = ransacPnP(points, bearings, 500, 3, 4, 3000, [](const Pose& p) { return cameraCenter(p).norm() <= 5; });
    require(unlimited.success && unlimited.num_inliers > bounded.num_inliers && bounded.success && bounded.num_inliers >= 28 &&
        cameraCenter(bounded.pose).norm() < 1e-3, "wrong-place majority hid an eligible visual PnP solution");
    auto seed = regional::anchoredRepairSeed(original, context, original);
    require(seed.numRegistered() == 3 && !seed.images.count(3), "anchor seed exceeded context");
    auto corrupt = original; corrupt.images[1].pose.t = {100,100,100};
    auto checked = regional::anchoredRepairSeed(original, context, corrupt);
    require(!checked.images.count(1), "invalid anchor entered repair seed");
    auto recovered = original;
    for (auto& kv : recovered.images) kv.second.point3D_ids[0] = kInvalidPoint3D;
    recovered.points3D.erase(0);
    const auto observations = regional::preserveRepairObservations(original, recovered, 3);
    require(observations == 120 && countObservations(recovered) == 120, "repair erased original observations");
    require(regional::verifyOriginalObservations(original, recovered) == 120, "original observation lineage not verifiable");
    auto changed = original; changed.images[0].points2D[0].x += 1;
    rejected = false; try { regional::preserveRepairObservations(original, changed, 3); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "repair silently changed feature identity");

    std::vector<regional::Observation> rays;
    for (const auto& kv : original.images) rays.push_back({0,uint64_t(kv.first) << 32,kv.second.points2D[0]});
    const auto refined = regional::refineUnifiedPoint({0,0,-5}, rays, original, 3);
    require(refined.z > 0 && (refined - original.points3D[0].xyz).norm() < 1e-4, "invalid projection penalty prevented valid retriangulation");
    std::puts("PASS observable backbone, explicit deferred coverage, verified repair anchors, observation conservation and valid numerical objective");
    return 0;
}
int main(int argc, char** argv) { return sfmTestMain(argc, argv, test); }
