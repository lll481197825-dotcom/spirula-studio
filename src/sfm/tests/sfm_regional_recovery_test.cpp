#include "sfm/map/RegionalAlignment.h"
#include "sfm/map/RegionalRecovery.h"
#include "sfm/map/RegionalStore.h"
#include "sfm/tests/TestMain.h"
#include "sfm/feature/LearnedMatcher.h"
#include "sfm/feature/Verification.h"
#include "sfm/map/AttitudeGauge.h"
#include "sfm/map/RepairFeatures.h"
#include "sfm/map/RegionalAdjustment.h"
#include "sfm/map/RegionalTransform.h"
#include "sfm/map/RegionalQuality.h"
#include <chrono>
#include <cstdio>
#include <regex>

using namespace sfm;
namespace fs = std::filesystem;
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct NoPrior : PriorSource {
    bool has(uint32_t) const override { return false; }
    bool relativeRotation(uint32_t, uint32_t, Mat3&, double&) const override { return false; }
    std::vector<uint32_t> neighbours(uint32_t) const override { return {}; }
    PosePriors factors(const std::vector<PosedImage>&) override { return {}; }
};
struct Positions : NoPrior {
    std::map<uint32_t, Vec3> values;
    bool position(uint32_t id, Vec3& value) const override {
        auto found = values.find(id); if (found == values.end()) return false;
        value = found->second; return true;
    }
};

static Reconstruction fixture() {
    Reconstruction rec; rec.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::Pinhole);
    for (uint32_t i = 0; i < 4; ++i) {
        Image im; im.id = 10 + i; im.camera_id = 0; im.registered = true;
        im.pose.t = {-.5 * i, 0, 0}; im.points2D.resize(12); im.point3D_ids.assign(12, kInvalidPoint3D);
        rec.images[im.id] = im;
    }
    for (uint32_t k = 0; k < 8; ++k) {
        Point3D p; p.xyz = {.2 * k, .1 * (k % 3), 5 + .05 * k};
        for (auto& kv : rec.images) {
            auto& im = kv.second; Vec3 q = p.xyz + im.pose.t;
            im.points2D[k + 2] = {500 * q.x / q.z + 400, 500 * q.y / q.z + 300};
            im.point3D_ids[k + 2] = k + 1; p.track.push_back({im.id, k + 2});
        }
        rec.points3D[k + 1] = p;
    }
    rec.next_point3D_id = 9; return rec;
}

static int probe(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    require(argc == 7 || argc == 8, "--probe requires matches, features, seed, anchor model, scratch output, optional source images");
    std::puts("PROBE indexing saved matches; original input files are read-only");
    MatchesIndex index; require(indexMatches(argv[2], index) && index.complete, "incomplete probe matches");
    auto seed = model_store_detail::readModel(argv[4]);
    auto summary = model_store_detail::readModel(argv[5]);
    auto required = regional::registeredIds(seed), anchors = regional::registeredIds(summary);
    fs::path scratch = argv[6]; fs::create_directories(scratch);
    CameraSetup cameras; loadCameraSetup(index.metadata, cameras);
    SfmConfig cfg; cfg.quality = "medium"; cfg.quiet = true; cfg.threads = 4; cfg.mapper.threads = 4;
    cfg.mapper.ba_solver = "cg";
    std::vector<PresetChange> changes; applyPresets(cfg, {}, changes);
    auto supplement = scratch / "matches"; regional::loadRepairMatches(index, supplement);
    std::vector<Vec3> positions(index.images.size());
    if (argc == 8) {
        std::puts("PROBE reading full capture GPS for external bridge candidates");
        cfg.sensor_pairs = true;
        auto priors = makeExifGpsPriors(cfg, argv[7], index.metadata, cameras, true);
        require(bool(priors), "probe source images have no usable GPS");
        for (uint32_t i = 0; i < positions.size(); ++i)
            require(priors->position(i, positions[i]), "probe GPS coverage is incomplete");
        std::printf("PROBE GPS read for all %zu input photos\n", positions.size());
    }
    NoPrior none;
    for (size_t limit : {64u, 128u, 256u, 512u, 1024u}) {
        if (argc == 8) regional::supplementRepairMatches(cfg, index, required, anchors, positions, cameras,
            argv[3], supplement, std::min<size_t>(256, limit / 2), size_t(6) << 30);
        auto context = regional::repairContext(index, required, anchors, limit);
        size_t common = 0; for (uint32_t i : context.images) common += anchors.count(i);
        std::printf("PROBE required=%zu support=%zu anchored=%zu limit=%zu\n", required.size(), context.images.size() - required.size(), common, limit);
        if (common < 3) continue;
        auto models = regional::reconstructRepair(cfg, index, context, argv[2], argv[3], "", cameras, nullptr, &seed, size_t(6) << 30);
        std::vector<Reconstruction> aligned;
        for (size_t m = 0; m < models.size(); ++m) {
            regional::atomicModel(scratch / ("local-" + std::to_string(limit) + "-" + std::to_string(m) + ".bin"), models[m]);
            bool ok = regional::alignRegional(models[m], summary, none, cfg.max_error, true);
            std::printf("PROBE model=%zu registered=%zu shared-alignment=%d\n", m, size_t(models[m].numRegistered()), int(ok));
            if (ok) aligned.push_back(std::move(models[m]));
        }
        if (regional::coversImages(aligned, required)) {
            for (size_t m = 0; m < aligned.size(); ++m) regional::atomicModel(scratch / ("aligned-" + std::to_string(m) + ".bin"), aligned[m]);
            std::ofstream report(scratch / "result.json"); report << "{\"required\":" << required.size()
                << ",\"support_limit\":" << limit << ",\"all_required_recovered\":true}\n";
            std::puts("PASS: all original component photos recovered through actual matched anchors"); return 0;
        }
    }
    throw std::runtime_error("real component recovery did not cover every original photo; probe geometry retained");
}

static int test(int argc, char** argv) {
    if (argc > 1 && (std::string(argv[1]) == "--gps-coordination-probe" || std::string(argv[1]) == "--gps-position-probe")) {
        const bool positions_only = std::string(argv[1]) == "--gps-position-probe";
        require(argc == 6, "--gps-coordination-probe requires legacy coordination, cached GPS, images and scratch");
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        const fs::path legacy = argv[2], scratch = argv[5]; fs::create_directories(scratch);
        Positions gps; model_store_detail::Reader in(argv[3]); const size_t count = in.count(28);
        for (size_t k = 0; k < count; ++k) {
            const auto id = in.scalar<uint32_t>(); const auto p = in.scalar<Vec3>(); gps.values.emplace(id, p);
        }
        in.finish();
        std::vector<std::pair<std::pair<unsigned, unsigned>, fs::path>> ordered;
        const std::regex pattern("pass-0-region-([0-9]+)-([0-9]+)\\.bin");
        for (const auto& entry : fs::directory_iterator(legacy.parent_path())) {
            std::smatch match; const auto name = entry.path().filename().string();
            if (std::regex_match(name, match, pattern)) ordered.push_back({
                {unsigned(std::stoul(match[1])), unsigned(std::stoul(match[2]))}, entry.path()});
        }
        std::sort(ordered.begin(), ordered.end()); require(!ordered.empty(), "missing original models");
        auto consensus = model_store_detail::readModel(legacy / "round-0-consensus.bin", false);
        const auto attitude = positions_only ? AttitudeRef{} : regional::regionalRecordedAttitude(consensus, argv[4], false);
        std::set<size_t> roots;
        std::set<size_t> certified_roots;
        std::vector<regional::SharedGpsAnchor> gps_anchors;
        for (size_t m = 0; m < ordered.size(); ++m) {
            const auto views = model_store_detail::readModel(ordered[m].second, false);
            if (regional::regionalGpsQuality(views, gps).accepted()) roots.insert(m);
            if (!positions_only && regional::regionalGpsGauge(views, gps, attitude)) certified_roots.insert(m);
            for (const auto& kv : views.images) if (kv.second.registered) {
                Vec3 p; require(gps.position(kv.first, p), "probe GPS missing registered photo");
                gps_anchors.push_back({uint32_t(m), cameraCenter(kv.second.pose), p});
            }
        }
        auto shared = regional::readSharedIndex(legacy / "round-0-landmarks.bin", ordered.size(), 256u << 20);
        std::printf("GPS_PROBE restoring %zu source models, %zu GPS-consistent frames, %zu certified gauges, position-only=%d\n",
            ordered.size(), roots.size(), certified_roots.size(), int(positions_only));
        regional::restoreSharedMembers(shared, [&](size_t m) {
            std::printf("GPS_PROBE source %zu/%zu\n", m + 1, ordered.size());
            return model_store_detail::readModel(ordered[m].second);
        });
        const auto fit = regional::guardedSharedAlignment(shared, 50, 256u << 20, roots, gps_anchors);
        regional::writeSharedIndex(scratch / "shared.bin", shared);
        SfmConfig cfg; cfg.threads = 8; cfg.mapper.ba_real = "double"; cfg.quiet = true; VkContext context;
        for (size_t m = 0; m < ordered.size(); ++m) {
            auto views = model_store_detail::readModel(ordered[m].second, false);
            const auto source_gps = regional::regionalGpsQuality(views, gps);
            applySim3(views, fit.transforms[m]);
            const auto aligned_gps = regional::regionalGpsQuality(views, gps);
            const bool certified = regional::measuredRegionalGauge(views, gps, attitude);
            std::printf("GPS_PROBE %s source=%zu/%zu aligned=%zu/%zu source_cost=%.9g aligned_cost=%.9g fixed=%d certified=%d\n",
                ordered[m].second.filename().string().c_str(), source_gps.inliers, source_gps.photos,
                aligned_gps.inliers, aligned_gps.photos, source_gps.cost, aligned_gps.cost, int(roots.count(m)), positions_only ? -1 : int(certified));
            if (roots.count(m)) require(std::abs(source_gps.cost - aligned_gps.cost) < 1e-6,
                "shared coordination moved a certified GPS model");
            if (certified_roots.count(m)) require(certified, "shared coordination lost certified attitude gauge");
            if (ordered[m].first.second || (ordered[m].first.first != 1 && ordered[m].first.first != 6)) continue;
            auto rec = model_store_detail::readModel(ordered[m].second); applySim3(rec, fit.transforms[m]);
            const auto fixed = shared.inject(m, rec);
            const bool certified_before = !positions_only && regional::regionalGpsGauge(views, gps, attitude);
            regional::boundedRegionalBA(rec, views, cfg, fixed, size_t(9) << 30, context, certified_before, &gps);
            const auto refined = regional::regionalGpsQuality(rec, gps);
            std::printf("GPS_PROBE window %s GPS=%zu/%zu cost=%.9g certified=%d\n",
                ordered[m].second.filename().string().c_str(), refined.inliers, refined.photos, refined.cost,
                positions_only ? -1 : int(regional::measuredRegionalGauge(rec, gps, attitude)));
            regional::atomicModel(scratch / ordered[m].second.filename(), rec);
            if (certified_before) require(regional::regionalGpsGauge(rec, gps, attitude), "GPS-constrained window lost certified frame");
            require(refined.accepted(), "GPS-constrained window lost absolute position consistency");
        }
        std::puts("PASS real source identity migration, absolute GPS frames and constrained windows; full visual export not evaluated"); return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--measured-gauge-audit") {
        require(argc == 5, "--measured-gauge-audit requires coordination, images and cached positions");
        std::setvbuf(stdout, nullptr, _IONBF, 0); fs::path root = argv[2]; Positions positions;
        model_store_detail::Reader in(argv[4]); size_t count = in.count(28);
        for (size_t k = 0; k < count; ++k) {
            uint32_t id = in.scalar<uint32_t>(); positions.values.emplace(id, in.scalar<Vec3>());
        }
        in.finish(); std::vector<uint8_t> marked(positions.values.rbegin()->first + 1);
        std::ifstream recorded(root.parent_path() / "measured-positioned.txt"); uint32_t id;
        while (recorded >> id) marked.at(id) = 1;
        model_store_detail::Reader head(root / "head.bin"); size_t pass = head.scalar<uint64_t>(); head.finish();
        auto consensus = model_store_detail::readModel(root / ("round-" + std::to_string(pass) + "-consensus.bin"), false);
        auto attitude = regional::regionalRecordedAttitude(regional::measuredRegionalViews(consensus, marked), argv[3], false);
        const std::string prefix = "pass-" + std::to_string(pass) + "-region-";
        size_t failed = 0, models = 0;
        for (const auto& entry : fs::directory_iterator(root)) {
            auto name = entry.path().filename().string();
            if (name.rfind(prefix, 0) != 0 || entry.path().extension() != ".bin") continue;
            auto rec = model_store_detail::readModel(entry.path(), false); auto measured = regional::measuredRegionalViews(rec, marked);
            if (measured.images.empty()) continue;
            ++models; bool ok = regional::measuredRegionalGauge(measured, positions, attitude); failed += !ok;
            auto original = model_store_detail::readModel(root.parent_path() / name, false);
            auto original_measured = regional::measuredRegionalViews(original, marked);
            auto original_ok = regional::measuredRegionalGauge(original_measured, positions, attitude);
            auto diagnostic = [&](const Reconstruction& views) {
                size_t good = 0; double total = 0;
                for (const auto& kv : views.images) { Vec3 p; if (!positions.position(kv.first, p)) continue;
                    double error = (cameraCenter(kv.second.pose) - p).norm(); good += error <= 5.; total += error; }
                return std::make_pair(good, total / std::max<size_t>(1, views.numRegistered()));
            };
            auto a = diagnostic(original_measured), b = diagnostic(measured);
            std::printf("MEASURED_AUDIT %s measured=%zu registered=%zu source_certified=%d joint_certified=%d source_GPS=%zu joint_GPS=%zu source_mean=%.9g joint_mean=%.9g\n", name.c_str(),
                size_t(measured.numRegistered()), size_t(rec.numRegistered()), int(original_ok), int(ok), a.first, b.first, a.second, b.second);
        }
        std::printf("MEASURED_AUDIT models=%zu failed=%zu\n", models, failed); return failed ? 1 : 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--coord-gauge-probe") {
        require(argc == 9, "--coord-gauge-probe requires coordination, matches, images, scratch, model, index and count");
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        fs::path root = argv[2], scratch = argv[5]; fs::create_directories(scratch);
        SfmConfig cfg; cfg.quality = "medium"; cfg.sensor_pairs = true; cfg.threads = 8;
        std::vector<PresetChange> changes; applyPresets(cfg, {}, changes);
        Positions positions;
        if (fs::exists(scratch / "gps.bin")) {
            model_store_detail::Reader in(scratch / "gps.bin"); size_t count = in.count(28);
            for (size_t k = 0; k < count; ++k) {
                uint32_t id = in.scalar<uint32_t>(); auto p = in.scalar<Vec3>(); positions.values.emplace(id, p);
            }
            in.finish();
        } else {
            MatchesIndex index; require(indexMatches(argv[3], index) && index.complete, "invalid gauge probe matches");
            CameraSetup cameras; loadCameraSetup(index.metadata, cameras);
            auto gps = makeExifGpsPriors(cfg, argv[4], index.metadata, cameras, true);
            require(bool(gps), "gauge probe requires GPS");
            for (uint32_t id = 0; id < index.images.size(); ++id) { Vec3 p; if (gps->position(id, p)) positions.values.emplace(id, p); }
            model_store_detail::Writer out(scratch / "gps.bin"); out.scalar<uint64_t>(positions.values.size());
            for (const auto& kv : positions.values) { out.scalar(kv.first); out.scalar(kv.second); } out.finish();
        }
        auto* gps = &positions;
        std::set<uint32_t> measured_ids;
        std::ifstream recorded(root.parent_path() / "measured-positioned.txt"); uint32_t measured_id;
        while (recorded >> measured_id) measured_ids.insert(measured_id);
        auto rec = model_store_detail::readModel(argv[6]);
        auto attitude = regional::regionalRecordedAttitude(rec, argv[4], cfg.exif_orientation == "apply");
        auto report = [&](const char* stage) {
            auto orientation = fitAttitudeGauge(rec, attitude, true);
            auto fit = regional::measuredRegionalAlignment(rec, *gps, attitude);
            size_t seen = 0, good = 0; double sum = 0, maximum = 0;
            for (const auto& kv : rec.images) {
                Vec3 p; if (!kv.second.registered || !gps->position(kv.first, p)) continue;
                double e = (cameraCenter(kv.second.pose) - p).norm(); ++seen; good += e <= 5.; sum += e; maximum = std::max(maximum, e);
            }
            std::printf("GAUGE %s images=%zu certified=%d fit=%d up=%d north=%d votes=%d up_spread=%.9g heading_spread=%.9g scale=%.9g yaw=%.9g GPS=%zu/%zu mean=%.9g max=%.9g rms=%.9g\n",
                stage, size_t(rec.numRegistered()), int(regional::measuredRegionalGauge(rec, *gps, attitude)), int(fit.ok), int(orientation.ok), int(orientation.north),
                orientation.up.votes, orientation.up.spread_deg, orientation.heading.spread_deg, fit.transform.scale, fit.yaw_deg, good, seen, sum / std::max<size_t>(1, seen), maximum, fit.rms);
            std::vector<uint8_t> marked(positions.values.rbegin()->first + 1);
            for (uint32_t id : measured_ids) marked.at(id) = 1;
            auto measured = regional::measuredRegionalViews(rec, marked);
            size_t measured_good = 0;
            for (const auto& kv : measured.images) { Vec3 p; if (gps->position(kv.first, p)) measured_good += (cameraCenter(kv.second.pose) - p).norm() <= 5.; }
            std::printf("MEASURED %s photos=%zu GPS_good=%zu certified=%d\n", stage, size_t(measured.numRegistered()), measured_good,
                int(regional::measuredRegionalGauge(measured, *gps, attitude)));
        };
        report("accepted-input");
        auto coordinated = std::move(rec);
        auto original = root.parent_path() / fs::path(argv[6]).filename();
        if (fs::exists(original)) { rec = model_store_detail::readModel(original, false); report("before-coordination"); }
        rec = std::move(coordinated);
        auto consensus = model_store_detail::readModel(root / "round-0-consensus.bin");
        auto shared = regional::readSharedIndex(root / "round-0-landmarks.bin", std::stoull(argv[8]), 128u << 20);
        auto fixed = shared.inject(std::stoull(argv[7]), rec); VkContext context;
        regional::boundedRegionalBA(rec, consensus, cfg, fixed, size_t(9) << 30, context, true);
        report("candidate"); regional::atomicModel(scratch / "candidate.bin", rec);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--resume-root-probe") {
        require(argc == 4, "--resume-root-probe requires preferred root and current signature file");
        const auto signature = resume::recorded(argv[3]);
        require(!signature.empty(), "--resume-root-probe signature is missing, empty or unreadable");
        auto selected = regional::compatibleRegionalRoot(argv[2], signature);
        std::printf("RESUME_ROOT %s\n", selected.string().c_str()); return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--polish-probe") {
        require(argc >= 7, "--polish-probe requires consensus, shared points, scratch and two models");
        auto consensus = model_store_detail::readModel(argv[2]); fs::path scratch = argv[4]; fs::create_directories(scratch);
        auto shared = regional::readSharedIndex(argv[3], argc - 5, 128u << 20);
        std::vector<fs::path> models;
        for (int arg = 5; arg < argc; ++arg) {
            auto rec = model_store_detail::readModel(argv[arg]); auto m = models.size();
            models.push_back(scratch / ("model-" + std::to_string(m) + ".bin")); regional::atomicModel(models.back(), rec);
            regional::writeValidation(scratch / ("validation-" + std::to_string(m) + ".bin"), rec, m);
        }
        double audited_cost = 0;
        auto audit = [&] {
            audited_cost = 0;
            std::vector<regional::PhotoQuality> photos(consensus.images.rbegin()->first + 1);
            for (size_t m = 0; m < models.size(); ++m) {
                auto rec = model_store_detail::readModel(models[m]); shared.inject(m, rec);
                auto q = regional::validationCost(scratch / ("validation-" + std::to_string(m) + ".bin"), rec, consensus, m, 3, 8, &photos);
                audited_cost += q.cost;
            }
            return photos;
        };
        SfmConfig cfg; cfg.threads = 8; cfg.mapper.ba_real = "double"; cfg.quiet = true; VkContext context;
        audit(); const double baseline_cost = audited_cost;
        for (size_t pass = 0; pass < 3; ++pass) {
            auto photos = audit(); std::set<uint32_t> targets;
            for (const auto& kv : consensus.images) if (!regional::photoQualityAccepted(photos.at(kv.first), 3)) targets.insert(kv.first);
            std::printf("POLISH pass=%zu targets=%zu full_cost=%.9g\n", pass, targets.size(), audited_cost); if (targets.empty()) break;
            regional::refineRegionalPoses(models.size(), [&](size_t m) {
                auto rec = model_store_detail::readModel(models[m]); shared.inject(m, rec); return rec;
            }, consensus, targets, scratch / ("poses-" + std::to_string(pass)), cfg, size_t(6) << 30, context);
            regional::triangulateShared(shared, consensus, 3);
            for (size_t m = 0; m < models.size(); ++m) {
                auto rec = model_store_detail::readModel(models[m]); regional::synchronizeRegionalCameras(rec, consensus);
                auto fixed = shared.inject(m, rec); regional::triangulatePrivate(rec, fixed, 3, cfg.threads, &targets);
                regional::atomicModel(models[m], rec);
            }
        }
        auto photos = audit(); size_t failed = 0; std::ofstream out(scratch / "photo-quality.csv");
        out << "image_id,observations,inliers,mean_error_px,passed\n";
        for (const auto& kv : consensus.images) {
            auto q = photos.at(kv.first); bool good = regional::photoQualityAccepted(q, 3); failed += !good;
            out << kv.first << ',' << q.observations << ',' << q.inliers << ',' << q.error / std::max<uint64_t>(1, q.observations) << ',' << good << '\n';
        }
        std::printf("POLISH final %zu/%zu passed\n", size_t(consensus.numRegistered()) - failed, size_t(consensus.numRegistered()));
        std::printf("POLISH original-observation cost %.9g -> %.9g\n", baseline_cost, audited_cost);
        regional::atomicModel(scratch / "consensus.bin", consensus); regional::writeSharedIndex(scratch / "shared.bin", shared);
        require(audited_cost <= baseline_cost * (1 + 1e-8), "full-observation pose refinement increased global original-observation cost");
        require(!failed, "full-observation pose refinement left unresolved photos"); return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--local-quality-probe") {
        require(argc == 4, "--local-quality-probe requires model and report directory");
        auto rec = model_store_detail::readModel(argv[2]); fs::path scratch = argv[3]; fs::create_directories(scratch);
        regional::writeValidation(scratch / "observations.bin", rec, 0);
        std::vector<regional::PhotoQuality> photos(rec.images.rbegin()->first + 1);
        auto q = regional::validationCost(scratch / "observations.bin", rec, rec, 0, 3, 8, &photos);
        std::ofstream report(scratch / "photo-quality.csv"); report << "image_id,observations,inliers,mean_error_px\n";
        for (const auto& kv : rec.images) if (kv.second.registered) {
            const auto& p = photos.at(kv.first);
            report << kv.first << ',' << p.observations << ',' << p.inliers << ',' << (p.observations ? p.error / p.observations : 0) << '\n';
        }
        std::printf("LOCAL images=%zu observations=%llu cost=%.9g\n", size_t(rec.numRegistered()), (unsigned long long)q.observations, q.cost); return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--joint-probe") {
        require(argc >= 6, "--joint-probe requires reference, scratch, and at least two candidate models");
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        auto reference = model_store_detail::readModel(argv[2]); fs::path scratch = argv[3]; fs::create_directories(scratch);
        std::vector<fs::path> paths; NoPrior none; std::map<uint32_t, size_t> owners;
        int first = 4;
        if (std::string(argv[first]) == "--plan") {
            require(argc >= 8, "--joint-probe --plan requires a plan and two models");
            model_store_detail::Reader plan(argv[first + 1]); require(plan.scalar<uint32_t>() == 1, "invalid joint probe plan");
            size_t count = plan.count(16);
            for (size_t r = 0; r < count; ++r) {
                for (uint32_t id : plan.vector<uint32_t>()) require(owners.emplace(id, r).second, "duplicate core owner");
                plan.vector<uint32_t>();
            }
            plan.finish(); first += 2;
        }
        std::vector<size_t> regions;
        for (int arg = first; arg < argc; ++arg) {
            auto rec = model_store_detail::readModel(argv[arg]); regional::SharedAlignmentReport evidence;
            require(regional::stageRegional(rec, reference, none, 3, &evidence), "real model has no observable visual bridge");
            std::map<size_t, size_t> counts;
            for (const auto& kv : rec.images) if (kv.second.registered && owners.count(kv.first)) ++counts[owners.at(kv.first)];
            size_t region = SIZE_MAX, most = 0;
            for (const auto& kv : counts) if (kv.second > most) { region = kv.first; most = kv.second; }
            regions.push_back(region);
            paths.push_back(scratch / ("input-" + std::to_string(paths.size()) + ".bin")); regional::atomicModel(paths.back(), rec);
        }
        auto shared = regional::buildSharedIndex(paths.size(), [&](size_t m) { return model_store_detail::readModel(paths[m]); },
            scratch / "index", 32u << 20, 128u << 20, 100000);
        auto fit = regional::guardedSharedAlignment(shared, 50, 128u << 20);
        require(fit.visual_components == 1, "real model graph is not visually connected");
        Reconstruction summary; regional::CameraEstimates cameras; std::map<uint32_t, uint32_t> scores;
        auto publish = [&](const Reconstruction& rec, Reconstruction& target, size_t model) {
            for (const auto& kv : rec.images) if (kv.second.registered) {
                uint32_t score = kv.second.numPoint3D();
                if (owners.count(kv.first) && owners.at(kv.first) == regions.at(model)) score += 1u << 30;
                if (target.images.count(kv.first) && scores[kv.first] >= score) continue;
                Image im; im.id = kv.first; im.camera_id = kv.second.camera_id; im.pose = kv.second.pose; im.registered = true;
                target.images[kv.first] = std::move(im); scores[kv.first] = score;
            }
        };
        for (size_t m = 0; m < paths.size(); ++m) {
            auto rec = model_store_detail::readModel(paths[m]); applySim3(rec, fit.transforms[m]);
            regional::atomicModel(paths[m], rec); regional::writeValidation(scratch / ("validation-" + std::to_string(m) + ".bin"), rec, m);
            publish(rec, summary, m); regional::addCameraEstimates(cameras, rec);
        }
        regional::applyCameraEstimates(summary, cameras); regional::triangulateShared(shared, summary, 3);
        auto quality = [&](const Reconstruction& poses, const regional::SharedIndex& points,
                           std::vector<regional::PhotoQuality>* photos = nullptr) {
            regional::Quality total;
            for (size_t m = 0; m < paths.size(); ++m) {
                auto rec = model_store_detail::readModel(paths[m]); points.inject(m, rec);
                auto q = regional::validationCost(scratch / ("validation-" + std::to_string(m) + ".bin"), rec, poses, m, 3, 8, photos);
                total.cost += q.cost; total.observations += q.observations;
            }
            return total;
        };
        auto accepted = quality(summary, shared); SfmConfig cfg; cfg.threads = 8; cfg.mapper.ba_real = "double";
        cfg.mapper.ba_solver = "cg"; cfg.quiet = true; VkContext context;
        std::printf("JOINT initial images=%zu landmarks=%zu observations=%llu cost=%.9g\n", size_t(summary.numRegistered()),
            shared.landmarks.size(), (unsigned long long)accepted.observations, accepted.cost);
        for (size_t pass = 1; pass <= size_t(cfg.regional_iterations); ++pass) {
            Reconstruction next; scores.clear(); cameras.clear();
            auto previous_paths = paths; std::vector<fs::path> candidate_paths;
            for (size_t m = 0; m < paths.size(); ++m) {
                auto rec = model_store_detail::readModel(paths[m]); auto fixed = shared.inject(m, rec);
                regional::boundedRegionalBA(rec, summary, cfg, fixed, size_t(6) << 30, context);
                auto candidate = scratch / ("pass-" + std::to_string(pass) + "-" + std::to_string(m) + ".bin");
                regional::atomicModel(candidate, rec); candidate_paths.push_back(candidate);
                publish(rec, next, m); regional::addCameraEstimates(cameras, rec);
            }
            paths = candidate_paths;
            regional::applyCameraEstimates(next, cameras); auto points = shared; regional::triangulateShared(points, next, 3);
            for (size_t m = 0; m < paths.size(); ++m) {
                auto rec = model_store_detail::readModel(paths[m]);
                regional::synchronizeRegionalCameras(rec, next);
                auto fixed = points.inject(m, rec);
                regional::triangulatePrivate(rec, fixed, 3, cfg.threads);
                regional::atomicModel(paths[m], rec);
            }
            std::vector<regional::PhotoQuality> photos(next.images.rbegin()->first + 1);
            auto q = quality(next, points, &photos);
            for (size_t polish = 0; polish < 3 && q.cost <= accepted.cost * (1 + 1e-8) + 1e-8; ++polish) {
                auto targets = regional::regionalQualityTargets(next, photos, 3);
                if (targets.empty()) break;
                auto candidate = regional::refineRegionalQuality(paths.size(),
                    [&](size_t m) { return model_store_detail::readModel(paths[m]); },
                    [&](size_t m, const Reconstruction& rec) { regional::atomicModel(paths[m], rec); },
                    quality, next, points, targets, photos, scratch / ("poses-" + std::to_string(pass) + "-" + std::to_string(polish)),
                    cfg, size_t(6) << 30, context);
                bool improved = candidate.cost < q.cost;
                q = candidate; if (!improved) break;
            }
            std::printf("JOINT pass=%zu images=%zu observations=%llu cost=%.9g previous=%.9g\n", pass, size_t(next.numRegistered()),
                (unsigned long long)q.observations, q.cost, accepted.cost);
            require(q.observations == accepted.observations, "real joint coordination changed original observations");
            auto stop = regional::qualityStop(q, accepted, accepted.cost, cfg.regional_cost_tolerance,
                pass == size_t(cfg.regional_iterations));
            if (stop == 2) {
                paths = previous_paths; std::puts("JOINT rejected candidate round; previous accepted geometry retained"); break;
            }
            shared = std::move(points); summary = std::move(next); accepted = q;
            if (stop) break;
        }
        std::vector<regional::PhotoQuality> tail_photos(summary.images.rbegin()->first + 1);
        auto tail_quality = quality(summary, shared, &tail_photos);
        for (size_t pass = 1; pass <= size_t(cfg.regional_iterations); ++pass) {
            auto targets = regional::regionalQualityTargets(summary, tail_photos, 3);
            if (targets.empty()) break;
            auto previous_paths = paths; Reconstruction next = summary; auto points = shared;
            std::vector<fs::path> candidates;
            for (size_t m = 0; m < paths.size(); ++m)
                candidates.push_back(scratch / ("quality-" + std::to_string(pass) + "-" + std::to_string(m) + ".bin"));
            paths = candidates;
            auto q = regional::refineRegionalQuality(paths.size(),
                [&](size_t m) { return model_store_detail::readModel(previous_paths[m]); },
                [&](size_t m, const Reconstruction& rec) { regional::atomicModel(paths[m], rec); },
                quality, next, points, targets, tail_photos, scratch / ("quality-poses-" + std::to_string(pass)),
                cfg, size_t(6) << 30, context);
            auto stop = regional::qualityStop(q, tail_quality, accepted.cost, 0, false);
            std::printf("JOINT quality pass=%zu targets=%zu observations=%llu cost=%.9g previous=%.9g stop=%u\n",
                pass, targets.size(), (unsigned long long)q.observations, q.cost, tail_quality.cost, stop);
            if (stop == 2) { paths = previous_paths; break; }
            summary = std::move(next); shared = std::move(points); tail_quality = q;
            if (stop && regional::regionalQualityTargets(summary, tail_photos, 3).size() >= targets.size()) break;
        }
        regional::atomicModel(scratch / "consensus.bin", summary);
        regional::writeSharedIndex(scratch / "shared.bin", shared);
        std::vector<regional::PhotoQuality> photos(summary.images.rbegin()->first + 1);
        for (size_t m = 0; m < paths.size(); ++m) {
            auto rec = model_store_detail::readModel(paths[m]); shared.inject(m, rec);
            regional::validationCost(scratch / ("validation-" + std::to_string(m) + ".bin"), rec, summary, m, 3, 8, &photos);
        }
        size_t failed = 0; std::ofstream report(scratch / "photo-quality.csv");
        report << "image_id,observations,inliers,mean_error_px,passed\n";
        for (const auto& kv : summary.images) {
            const auto& q = photos.at(kv.first); bool good = regional::photoQualityAccepted(q, 3); failed += !good;
            report << kv.first << ',' << q.observations << ',' << q.inliers << ','
                << (q.observations ? q.error / q.observations : 0) << ',' << good << '\n';
        }
        std::printf("JOINT final per-photo quality: %zu/%zu passed\n", size_t(summary.numRegistered()) - failed, size_t(summary.numRegistered()));
        require(!failed, "real joint coordination left individual camera geometry unresolved; report retained");
        std::puts("PASS real visual bridge, connected shared-landmark graph, full original observations and improving coordination"); return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--alignment-region-probe") {
        require(argc == 5, "--alignment-region-probe requires reference model, region plan and candidate");
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        auto reference = model_store_detail::readModel(argv[2]);
        auto candidate = model_store_detail::readModel(argv[4]);
        model_store_detail::Reader plan(argv[3]); require(plan.scalar<uint32_t>() == 1, "invalid region probe plan");
        size_t count = plan.count(16); NoPrior none;
        for (size_t r = 0; r < count; ++r) {
            auto core = plan.vector<uint32_t>(); plan.vector<uint32_t>(); Reconstruction subset;
            for (uint32_t id : core) {
                auto at = reference.images.find(id), local = candidate.images.find(id);
                if (at != reference.images.end() && local != candidate.images.end() && local->second.registered)
                    subset.images.emplace(id, at->second);
            }
            if (subset.images.size() < 3) continue;
            auto copy = candidate; regional::SharedAlignmentReport report;
            bool ok = regional::alignRegional(copy, subset, none, 3, true, &report);
            std::printf("REGION_ALIGNMENT region=%zu shared=%zu inliers=%zu median_clipped_px=%.6f p95_clipped_px=%.6f accepted=%d\n",
                r + 1, report.shared, report.inliers, report.median_px, report.p95_px, int(ok));
        }
        plan.finish(); return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--alignment-probe") {
        require(argc >= 5, "--alignment-probe requires checkpoint root or reference model, output reference, candidate models");
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        Reconstruction reference;
        const fs::path input = argv[2];
        if (fs::is_regular_file(input)) reference = model_store_detail::readModel(input);
        else {
            model_store_detail::Reader plan(input / "plan.bin");
            require(plan.scalar<uint32_t>() == 1, "invalid alignment probe plan");
            std::vector<std::set<uint32_t>> owned(plan.count(16));
            for (auto& ids : owned) {
                auto core = plan.vector<uint32_t>(); ids.insert(core.begin(), core.end());
                plan.vector<uint32_t>();
            }
            plan.finish();
            std::vector<std::pair<size_t, fs::path>> stages;
            const std::string prefix = "aligned-v1-local-region-";
            for (const auto& entry : fs::directory_iterator(input)) {
                auto name = entry.path().filename().string();
                if (name.rfind(prefix, 0) != 0 || entry.path().extension() != ".done") continue;
                size_t region = std::stoull(name.substr(prefix.size()));
                require(region < owned.size(), "alignment probe region out of range");
                stages.emplace_back(region, entry.path());
            }
            std::sort(stages.begin(), stages.end());
            std::map<uint32_t, uint32_t> scores;
            for (const auto& stage : stages) {
                auto base = stage.second; base.replace_extension();
                model_store_detail::Reader manifest(stage.second);
                size_t count = manifest.count(16);
                for (size_t m = 0; m < count; ++m) {
                    auto path = fs::path(base.string() + "-" + std::to_string(m) + ".bin");
                    uint64_t bytes = manifest.scalar<uint64_t>(); manifest.scalar<uint64_t>();
                    require(fs::file_size(path) == bytes, "alignment probe checkpoint changed");
                    auto rec = model_store_detail::readModel(path);
                    for (const auto& kv : rec.images) if (kv.second.registered) {
                        uint32_t score = kv.second.numPoint3D() + (owned[stage.first].count(kv.first) ? (1u << 30) : 0);
                        if (reference.images.count(kv.first) && scores[kv.first] >= score) continue;
                        Image im; im.id = kv.first; im.camera_id = kv.second.camera_id; im.name = kv.second.name;
                        im.pose = kv.second.pose; im.registered = true; im.exif_orientation = kv.second.exif_orientation;
                        reference.images[kv.first] = std::move(im); scores[kv.first] = score;
                    }
                    std::printf("REFERENCE stage=%s registered=%zu total=%zu\n", path.filename().string().c_str(),
                        size_t(rec.numRegistered()), size_t(reference.numRegistered()));
                }
                manifest.finish();
            }
        }
        regional::atomicModel(argv[3], reference);
        NoPrior none;
        for (int arg = 4; arg < argc; ++arg) {
            auto rec = model_store_detail::readModel(argv[arg]); regional::SharedAlignmentReport report;
            bool ok = regional::alignRegional(rec, reference, none, 3, true, &report);
            std::printf("ALIGNMENT candidate=%s registered=%zu points=%zu shared=%zu fitted=%d inliers=%zu median_clipped_px=%.6f p95_clipped_px=%.6f accepted=%d\n",
                argv[arg], size_t(rec.numRegistered()), rec.points3D.size(), report.shared, int(report.fitted),
                report.inliers, report.median_px, report.p95_px, int(ok));
            const auto rows_path = fs::path(argv[3]).parent_path() / (fs::path(argv[arg]).stem().string() + "-alignment.csv");
            std::ofstream rows(rows_path);
            rows << "image_id,inlier,alignment_clipped_px,local_clipped_px,centre_error,rotation_deg\n";
            for (const auto& row : report.views)
                rows << row.id << ',' << row.inlier << ',' << row.alignment_px << ',' << row.local_px << ','
                     << row.centre_error << ',' << row.rotation_deg << '\n';
            rows.flush(); require(bool(rows), "cannot write alignment probe observations");
        }
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--repair-speed-probe") {
        require(argc == 5, "--repair-speed-probe requires baseline matches, features and repair directory");
        MatchesIndex index; require(indexMatches(argv[2], index) && index.complete, "invalid speed probe baseline");
        size_t original = index.pairs.size(); regional::loadRepairMatches(index, argv[4]);
        std::vector<std::pair<uint32_t, uint32_t>> pairs;
        for (size_t i = original; i < index.pairs.size() && pairs.size() < 512; ++i)
            pairs.emplace_back(index.pairs[i].image1, index.pairs[i].image2);
        require(pairs.size() == 512, "not enough real positive pairs for the speed probe");
        CameraSetup cameras; loadCameraSetup(index.metadata, cameras);
        SfmConfig cfg; cfg.quality = "medium"; cfg.quiet = true;
        std::vector<PresetChange> changes; applyPresets(cfg, {}, changes);
        auto run = [&](bool fast) {
            auto start = std::chrono::steady_clock::now(); std::vector<TwoViewMatches> result;
            if (fast) {
                regional::RepairFeatureCache cache(index, argv[3], cameras, size_t(768) << 20);
                MatchOptions options = cfg.match; options.batch_pairs = 64; options.descriptor_budget_bytes = size_t(512) << 20;
                for (const auto& image : index.images) options.max_feature_count = std::max(options.max_feature_count, image.num_features);
                auto matcher = createFeatureMatcher(cfg.matcher, options, cfg.lightglue, cfg.loma_match);
                for (size_t begin = 0; begin < pairs.size();) {
                    size_t end = cache.nextEnd(pairs, begin); uint64_t putative = 0;
                    auto part = regional::verifyRepairPairs(cfg, cache, *matcher,
                        {pairs.begin() + begin, pairs.begin() + end}, putative);
                    for (auto& pair : part) result.push_back(std::move(pair)); begin = end;
                }
                std::printf("FAST_CACHE loads=%zu peak_mib=%.2f\n", cache.loads(), cache.peakBytes() / double(1 << 20));
            } else for (size_t begin = 0; begin < pairs.size(); begin += 32) {
                size_t end = std::min(begin + 32, pairs.size()); std::set<uint32_t> used;
                for (size_t p = begin; p < end; ++p) { used.insert(pairs[p].first); used.insert(pairs[p].second); }
                std::vector<uint32_t> ids(used.begin(), used.end()); std::map<uint32_t, uint32_t> local;
                std::vector<FeatureSet> features; std::vector<Camera> percam;
                for (uint32_t i : ids) {
                    local[i] = uint32_t(features.size()); percam.push_back(cameras.cameras.at(cameras.ids.at(i)));
                    features.push_back(readFeatures((fs::path(argv[3]) / (index.images.at(i).name + ".bin")).string(), true, true));
                }
                std::vector<std::pair<uint32_t, uint32_t>> local_pairs;
                for (size_t p = begin; p < end; ++p) local_pairs.emplace_back(local.at(pairs[p].first), local.at(pairs[p].second));
                MatchOptions options = cfg.match; options.batch_pairs = 16; options.descriptor_budget_bytes = size_t(128) << 20;
                auto matcher = createFeatureMatcher(cfg.matcher, options, cfg.lightglue, cfg.loma_match);
                auto bearings = precomputeBearings(features, percam, true, cfg.threads);
                VerificationOptions verify; verify.two_view = cfg.twoview; verify.num_threads = cfg.threads;
                verify.report_events = false; verify.match_batch_pairs = 16; verify.bearings = &bearings; verify.cameras = &percam;
                auto part = verifyPairs(features, local_pairs, [&](size_t b, size_t e, auto& out) {
                    matcher->matchBatch(features, local_pairs, b, e, out);
                }, verify);
                for (auto& pair : part) {
                    pair.image1 = ids.at(pair.image1); pair.image2 = ids.at(pair.image2); result.push_back(std::move(pair));
                }
            }
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::printf("SPEED %s pairs=%zu kept=%zu seconds=%.5f pairs_per_s=%.1f\n", fast ? "persistent" : "legacy32",
                pairs.size(), result.size(), seconds, pairs.size() / seconds);
            return std::make_pair(seconds, std::move(result));
        };
        auto legacy = run(false), fast = run(true);
        require(legacy.second.size() == fast.second.size(), "speed optimization changed geometric acceptance");
        for (size_t i = 0; i < legacy.second.size(); ++i) {
            const auto& a = legacy.second[i]; const auto& b = fast.second[i];
            require(a.image1 == b.image1 && a.image2 == b.image2 && a.config == b.config && a.matches.size() == b.matches.size(),
                    "speed optimization changed pair identity or geometry");
            for (size_t j = 0; j < a.matches.size(); ++j)
                require(a.matches[j].idx1 == b.matches[j].idx1 && a.matches[j].idx2 == b.matches[j].idx2,
                        "speed optimization changed verified feature correspondences");
        }
        std::printf("PASS identical verified results; measured speedup %.3fx\n", legacy.first / fast.first);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--match-source-probe") {
        require(argc == 4, "--match-source-probe requires baseline matches and cached repair directory");
        MatchesIndex index; require(indexMatches(argv[2], index) && index.complete, "invalid baseline matches");
        size_t baseline_pairs = index.pairs.size();
        regional::loadRepairMatches(index, argv[3]);
        {
            std::ifstream baseline(argv[2], std::ios::binary);
            std::map<uint32_t, std::ifstream> unbounded;
            for (size_t i = baseline_pairs; i < index.pairs.size(); ++i) {
                const auto& pair = index.pairs[i];
                auto found = unbounded.find(pair.source);
                if (found == unbounded.end()) found = unbounded.emplace(pair.source,
                    std::ifstream(index.sources.at(pair.source - 1), std::ios::binary)).first;
                if (!found->second) {
                    std::printf("OLD_READER_OPEN_FAILURE sources=%zu source=%u pair=%zu\n", unbounded.size(), pair.source, i);
                    break;
                }
            }
        }
        regional::RegionalMatchReader reader(index, argv[2]); std::vector<FeatureMatch> matches;
        uint64_t total = 0, checksum = 0; size_t peak = 0;
        for (size_t i = baseline_pairs; i < index.pairs.size(); ++i) {
            const auto& pair = index.pairs[i]; reader.read(pair, matches);
            require(matches.size() == pair.count, "real cached pair lost matches");
            for (const auto& match : matches) {
                require(match.idx1 < index.images.at(pair.image1).num_features &&
                        match.idx2 < index.images.at(pair.image2).num_features, "cached feature ID out of range");
                checksum += uint64_t(match.idx1) * 4294967291ull + match.idx2;
            }
            total += matches.size(); peak = std::max(peak, reader.openFiles());
        }
        require(peak <= regional::RegionalMatchReader::capacity, "real source handle budget exceeded");
        std::printf("PASS real cached sources=%zu pairs=%zu matches=%llu peak_open=%zu checksum=%llu\n",
            index.sources.size(), index.pairs.size() - baseline_pairs, (unsigned long long)total,
            peak, (unsigned long long)checksum);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--sensor-probe") {
        require(argc == 4 || argc == 5, "--sensor-probe requires saved geometry, source images, optional scratch output");
        std::vector<Reconstruction> models{model_store_detail::readModel(argv[2])};
        resolveImageNames(models, argv[3]); auto& model = models[0];
        auto attitude = fitAttitudeGauge(model, attitudeRefFromImages(model, argv[3], true), true);
        std::printf("ATTITUDE ok=%d north=%d up_votes=%d up_spread=%.5f heading_spread=%.5f\n",
            int(attitude.ok), int(attitude.north), attitude.up.votes, attitude.up.spread_deg, attitude.heading.spread_deg);
        require(attitude.ok && attitude.north, "recorded attitude did not constrain component orientation");
        applySim3(model, attitude.T); MetricRef reference; metricRefFromGps(model, argv[3], reference);
        auto fit = fitMetricGauge(reference, 5., MetricAxes::Horizontal, .01);
        std::printf("GPS ok=%d reason=%d inliers=%d/%d rms=%.5f yaw_correction=%.5f scale_unc=%.5f\n",
            int(fit.ok), int(fit.reason), fit.inliers, fit.n, fit.rms,
            std::atan2(fit.T.R[3], fit.T.R[0]) * 180. / M_PI, fit.scale_unc);
        require(fit.ok, "GPS did not constrain levelled component scale and position");
        Positions measured_positions;
        for (size_t i = 0; i < reference.image_ids.size(); ++i)
            measured_positions.values[reference.image_ids[i]] = reference.targets[i];
        auto accepted = regional::measuredRegionalAlignment(model, measured_positions,
            attitudeRefFromImages(model, argv[3], true));
        std::printf("PRODUCTION_MEASURED ok=%d inliers=%d/%d spatial_rms=%.5f\n",
            int(accepted.ok), accepted.inliers, accepted.positions, accepted.rms);
        require(accepted.ok, "production measured recovery refused component");
        applySim3(model, accepted.transform);
        require(regional::measuredRegionalGauge(model, measured_positions, attitudeRefFromImages(model, argv[3], true)),
            "production measured recovery could not certify its final gauge");
        if (argc == 5) {
            fs::path scratch = argv[4]; fs::create_directories(scratch);
            regional::atomicModel(scratch / "measured-component.bin", model);
            std::ofstream report(scratch / "result.json");
            report << "{\"registered\":" << model.numRegistered() << ",\"gps_inliers\":" << accepted.inliers
                << ",\"gps_positions\":" << accepted.positions << ",\"gps_spatial_rms_m\":" << accepted.rms
                << ",\"measured_gauge_certified\":true,\"visual_connection_resolved\":false,\"full_capture_verified\":false}\n";
        }
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--pair-probe") {
        require(argc == 4, "--pair-probe requires two saved feature files");
        std::vector<FeatureSet> features;
        for (int i = 2; i < argc; ++i) {
            features.push_back(readFeatures(argv[i], true, true));
            const auto& f = features.back();
            std::printf("FEATURE count=%u bytes=%zu dim=%u dtype=%u size=%dx%d scale=%.4f\n",
                f.count(), f.descBytes(), f.dim, unsigned(f.dtype), f.width, f.height, f.pixelScale());
        }
        SfmConfig cfg; cfg.quality = "medium"; cfg.threads = 4;
        std::vector<PresetChange> changes; applyPresets(cfg, {}, changes);
        auto matcher = createFeatureMatcher(cfg.matcher, cfg.match, cfg.lightglue, cfg.loma_match);
        std::vector<std::pair<uint32_t, uint32_t>> pairs{{0, 1}};
        VerificationOptions opt; opt.two_view = cfg.twoview; opt.num_threads = 4; opt.report_events = false;
        uint64_t putative = 0;
        auto verified = verifyPairs(features, pairs, [&](size_t b, size_t e, auto& out) {
            matcher->matchBatch(features, pairs, b, e, out);
        }, opt, &putative);
        std::printf("PAIR descriptor_matches=%llu verified_pairs=%zu inliers=%zu\n",
            (unsigned long long)putative, verified.size(), verified.empty() ? size_t(0) : size_t(verified[0].matches.size()));
        return verified.empty() ? 1 : 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--probe") return probe(argc, argv);
    auto original = fixture(); regional::RegionInput input; input.global = {10, 11, 12, 13}; input.original.resize(4);
    auto flight = original; Positions positions; AttitudeRef attitude;
    Sim3 target{2., so3Exp({0, 0, .05}), {40, 80, 120}};
    for (auto& kv : flight.images) {
        kv.second.pose.t.x = -100. * (kv.first - 10);
        positions.values[kv.first] = transformPoint(target, cameraCenter(kv.second.pose));
        attitude.image_ids.push_back(kv.first); attitude.world_from_cam.push_back(mat3Identity());
    }
    auto measured = regional::measuredRegionalAlignment(flight, positions, attitude);
    require(measured.ok && measured.inliers == 4, "recorded attitude and GPS could not constrain a straight flight");
    applySim3(flight, measured.transform);
    require(regional::measuredRegionalGauge(flight, positions, attitude), "accepted measured frame could not be certified");
    {
        auto compact = flight; Positions gps;
        const Vec3 origin = cameraCenter(compact.images.begin()->second.pose);
        for (auto& kv : compact.images) {
            const Vec3 centre = origin + (cameraCenter(kv.second.pose) - origin) * .3;
            kv.second.pose.t = mul(kv.second.pose.R, centre) * -1.; gps.values[kv.first] = centre;
        }
        require(regional::regionalGpsGauge(compact, gps, attitude), "compact visual source is not certified");
        regional::RegionalSourceFrame frame{"compact", compact, true, true};
        auto owner = compact;
        for (auto& kv : owner.images) {
            const Vec3 centre = origin + (cameraCenter(kv.second.pose) - origin) * 1.025;
            kv.second.pose.t = mul(kv.second.pose.R, centre) * -1.;
        }
        require(regional::regionalGpsQuality(owner, gps).accepted() && !regional::regionalGpsGauge(owner, gps, attitude),
            "ownership scale conflict was not reproduced inside the position gate");
        regional::RegionalPoseAnchors anchors;
        const auto affected = regional::protectRegionalSources(owner, {frame}, gps, attitude, anchors);
        require(affected.size() == 4 && anchors.size() == 4 && regional::regionalGpsGauge(owner, gps, attitude),
            "ownership reconciliation did not restore the source certificate");
        for (const auto& kv : compact.images)
            require(owner.images.at(kv.first).pose.R == kv.second.pose.R &&
                (owner.images.at(kv.first).pose.t - kv.second.pose.t).norm() < 1e-12,
                "source protection generated a pose instead of retaining the original visual pose");
        owner.images.begin()->second.pose.t.x += .2;
        regional::protectRegionalSources(owner, {frame}, gps, attitude, anchors);
        require((owner.images.begin()->second.pose.t - compact.images.begin()->second.pose.t).norm() < 1e-12,
            "a subsequent ownership merge overwrote a certified visual anchor");
    }
    const auto absolute_priors = regional::regionalGpsPriors(flight, positions);
    auto drifted = flight; for (auto& kv : drifted.images) kv.second.pose.t.z += 20.;
    const auto drifted_priors = regional::regionalGpsPriors(drifted, positions);
    require(absolute_priors.centres.size() == flight.numRegistered() && drifted_priors.centres.size() == flight.numRegistered(),
        "regional GPS factors omitted positioned photos");
    for (size_t i = 0; i < absolute_priors.centres.size(); ++i) {
        const auto& a = absolute_priors.centres[i]; const auto& b = drifted_priors.centres[i];
        require(a.A[0] == mat3Identity() && b.A[0] == a.A[0] && (b.b - a.b).norm() == 0.,
            "regional GPS factors refitted away an absolute-frame drift");
    }
    require(!regional::regionalGpsRoundAccepted(regional::regionalGpsQuality(drifted, positions),
        regional::regionalGpsQuality(flight, positions)), "GPS regression was accepted as a new round");
    require(regional::regionalGpsRoundAccepted({4266, 4244, 10423.1783}, {4266, 4263, 9669.99727}),
        "valid joint refinement was rejected despite passing the unchanged GPS gate");
    std::set<uint32_t> gps_targets; regional::addRegionalGpsTargets(gps_targets, drifted, positions);
    require(gps_targets == regional::registeredIds(drifted), "GPS-conflicting photos were omitted from pose repair");
    {
        Positions gps;
        for (const auto& kv : original.images) gps.values[kv.first] = cameraCenter(kv.second.pose);
        auto shifted_points = original; std::set<uint64_t> fixed;
        for (auto& p : shifted_points.points3D) { p.second.xyz.x += .2; fixed.insert(p.first); }
        SfmConfig cfg; cfg.mapper.ba_real = "cpu"; cfg.threads = 2; VkContext context;
        auto without = shifted_points, with = shifted_points;
        regional::boundedRegionalBA(without, original, cfg, fixed, size_t(1) << 30, context);
        regional::boundedRegionalBA(with, original, cfg, fixed, size_t(1) << 30, context, false, &gps);
        const auto a = regional::regionalGpsQuality(without, gps), b = regional::regionalGpsQuality(with, gps);
        require(b.cost < a.cost - 1e-8, "bounded regional BA did not apply absolute GPS factors");
        regional::RegionalPoseAnchors anchors;
        anchors.emplace(10, original.images.at(10).pose);
        auto held_model = shifted_points;
        regional::boundedRegionalBA(held_model, original, cfg, fixed, size_t(1) << 30, context, false, &gps, &anchors);
        require(held_model.images.at(10).pose.R == anchors.at(10).R &&
            (held_model.images.at(10).pose.t - anchors.at(10).t).norm() < 1e-12,
            "window BA moved a protected original visual pose");
        const auto scratch = fs::temp_directory_path() / ("regional-gps-pose-test-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        auto loose_poses = original, constrained_poses = original;
        const auto targets = regional::registeredIds(original);
        regional::refineRegionalPoses(1, [&](size_t) { return shifted_points; }, loose_poses, targets,
            scratch / "without", cfg, size_t(1) << 30, context);
        regional::refineRegionalPoses(1, [&](size_t) { return shifted_points; }, constrained_poses, targets,
            scratch / "with", cfg, size_t(1) << 30, context, &gps);
        require(regional::regionalGpsQuality(constrained_poses, gps).cost <
            regional::regionalGpsQuality(loose_poses, gps).cost - 1e-8,
            "full-observation pose refinement did not apply absolute GPS factors");
        auto held_poses = original;
        regional::refineRegionalPoses(1, [&](size_t) { return shifted_points; }, held_poses, targets,
            scratch / "held", cfg, size_t(1) << 30, context, &gps, &anchors);
        require(held_poses.images.at(10).pose.R == anchors.at(10).R &&
            (held_poses.images.at(10).pose.t - anchors.at(10).t).norm() < 1e-12,
            "full-observation refinement moved a protected original visual pose");
        fs::remove_all(scratch);
    }
    auto mixed = flight; auto mixed_attitude = attitude; auto mixed_positions = positions;
    std::vector<uint8_t> marked(24); for (uint32_t id = 10; id < 14; ++id) marked[id] = 1;
    for (uint32_t id = 20; id < 24; ++id) {
        auto im = flight.images.at(10); im.id = id; mixed_positions.values[id] = cameraCenter(im.pose);
        im.pose.t.x += 1000 + 100 * (id - 20); mixed.images.emplace(id, im);
        mixed_attitude.image_ids.push_back(id); mixed_attitude.world_from_cam.push_back(mat3Identity());
    }
    require(!regional::measuredRegionalGauge(mixed, mixed_positions, mixed_attitude), "fixture did not expose unrelated visual-camera gauge conflicts");
    auto measured_views = regional::measuredRegionalViews(mixed, marked);
    require(measured_views.numRegistered() == 4 && regional::measuredRegionalGauge(measured_views, mixed_positions, mixed_attitude),
        "unrelated visual cameras invalidated the measured photo constraints");
    for (auto& kv : measured_views.images) kv.second.pose.t.z += 100;
    require(!regional::measuredRegionalGauge(measured_views, mixed_positions, mixed_attitude), "actual measured-photo contradictions were hidden");
    require(regional::measuredGaugeAnchors(flight) == std::set<uint32_t>({10, 13}),
        "measured BA did not hold a spanning camera baseline");
    auto contradicted = attitude;
    for (auto& r : contradicted.world_from_cam) r = so3Exp({0, 0, .8});
    require(!regional::measuredRegionalAlignment(flight, positions, contradicted).ok,
        "contradictory measured heading was hidden by a GPS fit");
    auto bad_height = positions; bad_height.values[10].z += 100.; bad_height.values[12].z -= 100.;
    require(!regional::measuredRegionalAlignment(flight, bad_height, attitude).ok,
        "horizontal residual concealed contradictory altitude");
    for (auto& ids : input.original) ids = {2, 3, 4, 5, 6, 7, 8, 9};
    input.features.resize(4);
    for (size_t i = 0; i < input.global.size(); ++i) for (uint32_t f : input.original[i]) {
        auto xy = original.images.at(input.global[i]).points2D.at(f);
        input.features[i].keypoints.push_back({float(xy.x), float(xy.y), 1, 0, 0});
    }
    auto compact = regional::compactSeed(original, input);
    require(compact.images.size() == 4 && compact.images.at(0).point3D_ids.at(0) == 1, "compact seed image or feature identity changed");
    MatchesIndex index; index.images.resize(14); for (auto& im : index.images) im.num_features = 12;
    regional::restoreFeatureIds(compact, input, index);
    require(regional::registeredIds(compact) == regional::registeredIds(original), "seed round trip lost a photo");
    require(compact.points3D.at(1).track[0].image_id == 10 && compact.points3D.at(1).track[0].point2D_idx == 2,
        "seed track round trip changed original feature IDs");
    bool omitted = false; input.global.pop_back(); input.original.pop_back();
    try { regional::compactSeed(original, input); } catch (const std::runtime_error&) { omitted = true; }
    require(omitted, "repair silently omitted a source photo");
    NoPrior none; Reconstruction empty;
    auto shifted = original; applySim3(shifted, {1.3, so3Exp({.2, -.1, .04}), {4, 2, 1}});
    regional::SharedAlignmentReport report;
    require(!regional::alignRegional(shifted, empty, none, 3, true, &report) && !report.fitted && report.shared == 0,
        "empty first-block anchors were accepted or misreported");
    require(regional::alignRegional(shifted, original, none, 3, true, &report), "collinear flight cameras could not align through shared observations");
    require(report.accepted && report.fitted && report.shared == 4 && report.inliers == 4 && report.p95_px < 1e-7,
        "shared-camera alignment diagnostic does not reflect its acceptance");
    require(report.views.size() == 4 && report.views[0].local_px < 1e-7 && report.views[0].centre_error < 1e-7 &&
        report.views[0].rotation_deg < 1e-5, "alignment diagnostic did not distinguish local and cross-block geometry");
    for (const auto& kv : original.points3D) require((shifted.points3D.at(kv.first).xyz - kv.second.xyz).norm() < 1e-7,
        "shared-camera recovery moved geometry incorrectly");
    auto inconsistent = original;
    for (auto& kv : inconsistent.images) {
        double displacement = kv.first - 10;
        kv.second.pose.R = so3Exp({.3 * displacement, -.2 * displacement, .1 * displacement});
        kv.second.pose.t = kv.second.pose.t + Vec3{displacement * displacement, -displacement, .2 * displacement};
    }
    auto retained = original;
    require(!regional::alignRegional(retained, inconsistent, none, 3, true, &report) && !report.accepted && report.shared == 4,
        "inconsistent cross-block poses were accepted or their evidence was lost");
    for (const auto& kv : original.points3D)
        require((retained.points3D.at(kv.first).xyz - kv.second.xyz).norm() == 0,
            "failed diagnostic alignment changed the retained local geometry");
    auto expanded = original;
    for (uint32_t id = 14; id < 20; ++id) {
        auto im = original.images.at(10); im.id = id; im.pose.t.x = -.5 * (id - 10);
        for (uint32_t k = 0; k < 8; ++k) {
            auto point = expanded.points3D.at(k + 1).xyz + im.pose.t;
            im.points2D[k + 2] = expanded.cameras.at(0).project(point);
            expanded.points3D.at(k + 1).track.push_back({id, k + 2});
        }
        expanded.images[id] = std::move(im);
    }
    auto warped = expanded;
    for (auto& kv : warped.images) if (kv.first >= 13) {
        double distance = kv.first - 12;
        kv.second.pose.R = so3Exp({.3 * distance, -.2 * distance, .1 * distance});
        kv.second.pose.t.x += distance * distance;
    }
    auto staged = expanded;
    require(!regional::alignRegional(staged, warped, none, 3, true), "fixture did not reproduce the pre-coordination rejection");
    require(!regional::stageRegional(staged, warped, none, 3, &report), "rejected weak bridge entered coordination as a trusted model");
    require(!report.accepted && report.inliers >= 3 && regional::coversImages({staged}, regional::registeredIds(expanded)),
        "provisional bridge was reported as validated or lost a photo");
    regional::synchronizeRegionalCameras(staged, expanded);
    for (const auto& kv : expanded.images) {
        require((cameraCenter(staged.images.at(kv.first).pose) - cameraCenter(kv.second.pose)).norm() < 1e-9,
            "overlapping model kept an independent camera pose after coordination");
        require(staged.images.at(kv.first).point3D_ids == kv.second.point3D_ids,
            "consensus synchronization changed original feature identities");
    }
    auto missing = original; missing.images.erase(13);
    require(!regional::coversImages({missing}, regional::registeredIds(original)), "incomplete repair passed photo coverage");
    require(regional::coversImages({original}, regional::registeredIds(original)), "complete repair rejected");
    MatchesIndex links; links.images.resize(10);
    links.pairs = {{0, 3, 500, 0, 0}, {1, 4, 100, 0, 0}, {2, 5, 50, 0, 0}, {0, 6, 90, 0, 0}, {6, 7, 80, 0, 0}};
    auto context = regional::repairContext(links, {0, 1, 2}, {3, 4, 5, 7}, 8);
    require(context.core == std::vector<uint32_t>({0, 1, 2}) && context.images == std::vector<uint32_t>({0, 1, 2, 3, 4, 5, 6, 7}),
        "repair did not include every required photo and direct/two-hop matched anchors");
    auto isolated = regional::repairContext(links, {8, 9}, {3, 4, 5}, 8);
    require(isolated.images == std::vector<uint32_t>({8, 9}), "repair invented an unverified connection");
    auto scratch = fs::temp_directory_path() / ("sfm-repair-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(scratch / "features"); fs::create_directories(scratch / "matches");
    {
        auto canonical = scratch / "resume-roots" / "canonical", legacy = canonical.parent_path() / "legacy";
        fs::create_directories(canonical); fs::create_directories(legacy);
        resume::store(canonical / "signature.txt", "features=stable\n");
        resume::store(legacy / "signature.txt", "features=stable\nblock-cache-mb=3000\n");
        std::ofstream(legacy / "local-region-0.done").put('\n');
        require(regional::compatibleRegionalRoot(canonical, "features=stable\n") == legacy,
            "cache capacity migration discarded existing local geometry");
        require(regional::compatibleRegionalRoot(canonical, "features=changed\n") != legacy,
            "cache migration reused geometry over changed feature identities");
        std::ofstream(canonical / "pass-0-region-0.done").put('\n');
        require(regional::compatibleRegionalRoot(canonical, "features=stable\n") == canonical,
            "resume ignored the compatible root with more completed regions");
    }
    {
        auto poses = expanded; poses.images.at(10).pose.t.x += .2; poses.images.at(11).pose.t.y -= .1;
        SfmConfig cfg; cfg.mapper.ba_real = "cpu"; cfg.quiet = true; VkContext device;
        regional::refineRegionalPoses(2, [&](size_t) { return expanded; }, poses, {10, 11},
            scratch / "pose-refinement", cfg, size_t(64) << 20, device);
        for (uint32_t id : {10u, 11u})
            require((cameraCenter(poses.images.at(id).pose) - cameraCenter(expanded.images.at(id).pose)).norm() < 1e-4,
                "full original observations across models did not recover the consensus camera");
        require((cameraCenter(poses.images.at(12).pose) - cameraCenter(expanded.images.at(12).pose)).norm() == 0,
            "targeted consensus solve changed an unrelated camera");
        poses.images.at(10).pose.t.x += .1;
        regional::refineRegionalPoses(2, [&](size_t) { return expanded; }, poses, {10},
            scratch / "single-pose-refinement", cfg, size_t(64) << 20, device);
        require((cameraCenter(poses.images.at(10).pose) - cameraCenter(expanded.images.at(10).pose)).norm() < 1e-4,
            "last unresolved photo was silently skipped by full-observation refinement");
    }
    MatchesDatabase base; base.images = {{"a", 12}, {"b", 12}, {"c", 12}};
    base.cameras = {original.cameras.at(0)}; base.camera_ids = {0, 0, 0};
    base.pairs = {{0, 1, 3, {{1, 2}, {2, 3}}}}; writeMatches((scratch / "base.bin").string(), base);
    MatchesIndex augmented; require(indexMatches((scratch / "base.bin").string(), augmented), "baseline test matches unreadable");
    MatchesDatabase extra = base; extra.images = {base.images[1], base.images[2]}; extra.camera_ids = {0, 0};
    extra.pairs = {{0, 1, 3, {{4, 5}, {5, 6}}}};
    auto path = scratch / "matches" / "pairs.bin"; writeMatches(path.string(), extra);
    { std::ofstream ids(path.string() + ".ids", std::ios::binary); const uint32_t table[] = {2, 1, 2};
      ids.write(reinterpret_cast<const char*>(table), sizeof(table)); }
    regional::loadRepairMatches(augmented, scratch / "matches");
    regional::loadRepairMatches(augmented, scratch / "matches");
    require(augmented.pairs.size() == 2 && augmented.pairs[1].source == 1 && augmented.pairs[1].image1 == 1 &&
            augmented.pairs[1].image2 == 2, "supplemental image IDs or idempotent resume changed");
    auto negative = extra; negative.images = {base.images[0], base.images[2]}; negative.pairs = {{0, 1, 0, {}}};
    auto negative_path = scratch / "matches" / "negative.bin"; writeMatches(negative_path.string(), negative);
    { std::ofstream ids(negative_path.string() + ".ids", std::ios::binary); const uint32_t table[] = {2, 0, 2};
      ids.write(reinterpret_cast<const char*>(table), sizeof(table)); }
    regional::loadRepairMatches(augmented, scratch / "matches");
    require(augmented.pairs.size() == 2 && augmented.attempted_pairs.count(uint64_t(2)) &&
        augmented.attempted_pairs.count((uint64_t(1) << 32) | 2), "negative repair evidence was lost on resume");
    FeatureSet features; features.width = 800; features.height = 600; features.keypoints.resize(12);
    features.descriptors.resize(12 * features.dim);
    for (const auto& im : base.images) writeFeatures((scratch / "features" / (im.name + ".bin")).string(), features);
    {
        MatchesIndex cache_index; cache_index.images.resize(20); CameraSetup setup;
        setup.cameras[0] = base.cameras[0]; setup.ids.assign(20, 0);
        FeatureSet full = features; full.keypoints.resize(64); full.descriptors.resize(64 * full.dim);
        for (uint32_t i = 0; i < 20; ++i) {
            full.keypoints[0].x = float(i); full.descriptors[0] = uint8_t(i);
            cache_index.images[i] = {"cached-" + std::to_string(i), 64};
            writeFeatures((scratch / "features" / (cache_index.images[i].name + ".bin")).string(), full);
        }
        regional::RepairFeatureCache cache(cache_index, scratch / "features", setup, 160000);
        for (size_t pass = 0; pass < 2; ++pass) for (uint32_t i = 0; i < 20; ++i) {
            uint32_t j = (i + 1) % 20; cache.pin({i, j});
            require(cache.features[i].keypoints[0].x == float(i) && cache.features[j].descData()[0] == uint8_t(j),
                    "repair feature eviction changed stable photo identity");
            require(cache.view[i] == &cache.features[i] && cache.bytes() <= cache.budget(), "repair cache violated stable pointers or RAM budget");
        }
        size_t loads = cache.loads(); cache.pin({19, 0});
        require(cache.loads() == loads && loads > 20, "repair cache did not reuse resident photos or exercise eviction");
        std::vector<std::pair<uint32_t, uint32_t>> planned(3000, {19, 0});
        require(cache.nextEnd(planned, 0) == 2048, "repair still imposes tiny commits despite available RAM");
    }
    regional::Region all; all.core = all.images = {0, 1, 2};
    auto loaded = regional::loadRegion(augmented, all, scratch / "base.bin", scratch / "features", size_t(1) << 30);
    require(loaded.db.pairs.size() == 2 && loaded.original[1] == std::vector<uint32_t>({2, 3, 4, 5}) &&
            loaded.original[2] == std::vector<uint32_t>({5, 6}), "supplemental pair offsets or feature identity changed");
    {
    fs::create_directories(scratch / "many");
    for (size_t i = 0; i < 700; ++i) {
        auto file = scratch / "many" / ("pairs-" + std::to_string(i) + ".bin");
        fs::copy_file(path, file); fs::copy_file(path.string() + ".ids", file.string() + ".ids");
    }
    MatchesIndex many; require(indexMatches((scratch / "base.bin").string(), many), "many-source baseline unreadable");
    regional::loadRepairMatches(many, scratch / "many");
    regional::RegionalMatchReader reader(many, scratch / "base.bin"); std::vector<FeatureMatch> rows;
    for (size_t pass = 0; pass < 2; ++pass) for (const auto& pair : many.pairs) {
        reader.read(pair, rows); require(rows.size() == 2, "eviction dropped a match");
        require(rows[0].idx1 == (pair.source ? 4u : 1u) && rows[0].idx2 == (pair.source ? 5u : 2u),
                "eviction changed match payload");
        require(reader.openFiles() <= regional::RegionalMatchReader::capacity, "source handles grew without bound");
    }
    auto many_loaded = regional::loadRegion(many, all, scratch / "base.bin", scratch / "features", size_t(1) << 30);
    require(many_loaded.db.pairs.size() == 701 && many_loaded.original == loaded.original,
            "many-source region lost pairs or changed feature identity");
    auto missing_pair = many.pairs.back(); missing_pair.source = uint32_t(many.sources.size() + 1);
    many.sources.push_back((scratch / "missing.bin").string());
    bool missing_rejected = false;
    try { reader.read(missing_pair, rows); }
    catch (const std::runtime_error& error) {
        missing_rejected = std::string(error.what()).find("missing.bin") != std::string::npos;
    }
    require(missing_rejected, "missing source was skipped or omitted its diagnostic path");
    auto truncated = many.pairs.back(); truncated.offset = UINT64_C(1) << 40;
    bool truncated_rejected = false;
    try { reader.read(truncated, rows); }
    catch (const std::runtime_error& error) {
        truncated_rejected = std::string(error.what()).find("offset=1099511627776") != std::string::npos;
    }
    require(truncated_rejected && rows.empty(), "short source read was skipped or omitted its diagnostic offset");
    reader.read(many.pairs.back(), rows); require(rows.size() == 2, "failed read poisoned the cached stream");
    }
    { std::ofstream ids(path.string() + ".ids", std::ios::binary); const uint32_t table[] = {2, 0, 2};
      ids.write(reinterpret_cast<const char*>(table), sizeof(table)); }
    bool rejected = false;
    try { MatchesIndex bad; indexMatches((scratch / "base.bin").string(), bad); regional::loadRepairMatches(bad, scratch / "matches"); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "repair silently accepted changed photo identity"); fs::remove_all(scratch);
    std::puts("PASS: retained photo coverage, compact seed round trip, deferred first block, shared-camera collinear recovery and verified bridge planning");
    return 0;
}
int main(int argc, char** argv) { return sfmTestMain(argc, argv, test); }
