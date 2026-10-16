#include "sfm/Pipeline.h"
#include "sfm/core/Resume.h"
#include "sfm/core/Progress.h"
#include "sfm/map/RegionalStore.h"
#include "sfm/map/RegionalPlan.h"
#include "sfm/map/RegionalRecovery.h"
#include "sfm/map/RegionalAlignment.h"
#include "sfm/map/RegionalAdjustment.h"
#include "sfm/map/RegionalTransform.h"
#include "sfm/map/RegionalWindows.h"
#include "sfm/map/RegionalQuality.h"
#include "sfm/map/RegionalPrefetch.h"
#include "sfm/map/RegionalBootstrap.h"
#include "sfm/map/RegionalRepairCache.h"
#include "sfm/map/SceneCoordinator.h"
#include "sfm/map/NativeSceneTasks.h"
#include "sfm/map/ImmutableSceneAliases.h"
#include "sfm/map/CanonicalRecovery.h"
#include "sfm/map/HeldRepairSeed.h"
#include "sfm/map/CanonicalInitialization.h"
#include "sfm/map/CanonicalOriginalCoverage.h"
#include "sfm/map/CanonicalAdmission.h"
#include "sfm/map/CanonicalImageIdentity.h"
#include "sfm/map/CanonicalAcceptance.h"
#include "sfm/map/CanonicalCapacity.h"
#include "sfm/map/CanonicalResume.h"
#include "sfm/map/CanonicalLayout.h"
#include "sfm/map/CanonicalExportAttempt.h"
#include "sfm/map/CanonicalPreview.h"
#include "sfm/map/SceneVisualEvidence.h"
#include "sfm/map/RegionalSourceHistory.h"
#include "sfm/map/Orient.h"
#include "i18n/TimeFormat.h"
#include <array>
#include <iomanip>
#include <sstream>
#include <thread>

namespace sfm {
namespace {
namespace fs = std::filesystem;
using regional::Region;
using regional::RegionInput;
using sfm::slog::Tag;
namespace M = spirula::i18n::msg::sfm;

struct SavedRegion { fs::path path; size_t memory = 0; fs::path origin; };
struct MapIdsScope {
    explicit MapIdsScope(const std::vector<uint32_t>& ids) { events::set_map_image_ids(ids); }
    ~MapIdsScope() { events::set_map_image_ids({}); }
};

using regional::signatureHash;

void savePlan(const fs::path& root, const std::vector<Region>& plan) {
    model_store_detail::Writer out(root / "plan.part");
    out.scalar<uint32_t>(1); out.scalar<uint64_t>(plan.size());
    for (const Region& r : plan) { out.vector(r.core); out.vector(r.images); }
    out.finish(); regional::replaceFile(root / "plan.part", root / "plan.bin");
}


fs::path stageFile(const fs::path& root, size_t pass, size_t region) {
    return root / ("pass-" + std::to_string(pass) + "-region-" + std::to_string(region));
}

std::vector<SavedRegion> savedStage(const fs::path& file) {
    std::vector<SavedRegion> result;
    if (!fs::exists(file.string() + ".done")) return result;
    model_store_detail::Reader in(file.string() + ".done");
    size_t count = in.count(16);
    for (size_t m = 0; m < count; ++m) {
        SavedRegion item; item.path = file.string() + "-" + std::to_string(m) + ".bin";
        uint64_t bytes = in.scalar<uint64_t>(); item.memory = in.scalar<uint64_t>();
        if (!fs::exists(item.path) || fs::file_size(item.path) != bytes) throw std::runtime_error("regional checkpoint file changed");
        result.push_back(item);
    }
    in.finish();
    std::istringstream origins(resume::recorded(file.string() + ".origins")); std::string origin;
    for (auto& item : result) {
        if (std::getline(origins, origin)) {
            if (origin.empty() || fs::path(origin).filename() != fs::path(origin)) throw std::runtime_error("invalid regional source lineage");
            item.origin = origin;
        } else item.origin = item.path.filename();
    }
    if (std::getline(origins, origin)) throw std::runtime_error("regional source lineage count changed");
    return result;
}

std::vector<SavedRegion> commitStage(const fs::path& file, const std::vector<Reconstruction>& models) {
    std::vector<SavedRegion> result;
    for (size_t m = 0; m < models.size(); ++m) {
        SavedRegion item{file.string() + "-" + std::to_string(m) + ".bin", modelResidentBytes(models[m])};
        regional::atomicModel(item.path, models[m]); result.push_back(item);
    }
    model_store_detail::Writer out(file.string() + ".part"); out.scalar<uint64_t>(result.size());
    for (const auto& r : result) { out.scalar<uint64_t>(fs::file_size(r.path)); out.scalar<uint64_t>(r.memory); }
    out.finish(); regional::replaceFile(file.string() + ".part", file.string() + ".done");
    return result;
}

Reconstruction readSaved(const SavedRegion& item, size_t budget) {
    if (item.memory > budget / 2) throw std::length_error("regional checkpoint exceeds current memory budget");
    return model_store_detail::readModel(item.path);
}

std::optional<Reconstruction> repairSeed(const fs::path& root, const fs::path& stage,
                                       const std::set<uint32_t>& required, size_t budget) {
    std::vector<SavedRegion> candidates;
    for (const auto& entry : fs::directory_iterator(root)) {
        const auto name = entry.path().filename().string();
        if (entry.path().extension() != ".done" ||
            (name.rfind("repair-v2-" + stage.filename().string() + "-", 0) != 0 &&
             name.rfind("repair-v3-" + stage.filename().string() + "-", 0) != 0)) continue;
        auto base = entry.path(); base.replace_extension();
        for (auto& item : savedStage(base)) if (item.memory <= budget / 2) candidates.push_back(std::move(item));
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.memory > b.memory; });
    for (const auto& item : candidates) {
        auto rec = readSaved(item, budget);
        const auto ids = regional::registeredIds(rec);
        if (std::includes(ids.begin(), ids.end(), required.begin(), required.end())) return rec;
    }
    return std::nullopt;
}

void addSummary(Reconstruction& summary, const Reconstruction& rec, const Region& region,
                std::vector<uint32_t>& source, size_t preview_points, const PriorSource* gps = nullptr) {
    for (const auto& kv : rec.images) {
        if (!kv.second.registered) continue;
        bool owned = std::binary_search(region.core.begin(), region.core.end(), kv.first);
        auto found = summary.images.find(kv.first);
        uint32_t weight = std::min(kv.second.numPoint3D(), (1u << 30) - 1); Vec3 position;
        const bool positioned = gps && gps->position(kv.first, position) && (cameraCenter(kv.second.pose) - position).norm() <= 5;
        uint32_t score = weight | (owned ? 1u << 30 : 0) | (positioned ? 1u << 31 : 0);
        if (found != summary.images.end() && source.at(kv.first) >= score) continue;
        Image im; im.id = kv.first; im.camera_id = kv.second.camera_id; im.name = kv.second.name;
        im.pose = kv.second.pose; im.registered = true; im.exif_orientation = kv.second.exif_orientation;
        summary.images[kv.first] = std::move(im); source.at(kv.first) = score;
        events::map_placed(kv.first);
    }
    size_t step = std::max<size_t>(1, rec.points3D.size() / std::max<size_t>(1, preview_points)), seen = 0;
    for (const auto& kv : rec.points3D) {
        if (summary.points3D.size() >= progress::kMaxPoints) break;
        if (seen++ % step) continue;
        Point3D p; p.xyz = kv.second.xyz; std::copy(kv.second.rgb, kv.second.rgb + 3, p.rgb);
        summary.points3D.emplace(summary.next_point3D_id++, std::move(p));
    }
}

using regional::alignRegional;
using regional::stageRegional;


using regional::RegionalCosts;
using regional::CameraEstimates;
using regional::addCameraEstimates;
using regional::applyCameraEstimates;
using regional::boundedRegionalBA;

void commitSavedStage(const fs::path& stage, const std::vector<SavedRegion>& saved, const RegionalCosts& costs) {
    std::string origins;
    for (const auto& item : saved) origins += (item.origin.empty() ? item.path.filename() : item.origin).string() + "\n";
    resume::store(stage.string() + ".origins", origins);
    model_store_detail::Writer c(stage.string() + ".costs.part");
    c.scalar(costs); c.finish(); regional::replaceFile(stage.string() + ".costs.part", stage.string() + ".costs");
    model_store_detail::Writer out(stage.string() + ".part"); out.scalar<uint64_t>(saved.size());
    for (const auto& model : saved) { out.scalar<uint64_t>(fs::file_size(model.path)); out.scalar<uint64_t>(model.memory); }
    out.finish(); regional::replaceFile(stage.string() + ".part", stage.string() + ".done");
}
RegionalCosts readStageCosts(const fs::path& stage) {
    model_store_detail::Reader in(stage.string() + ".costs"); auto costs = in.scalar<RegionalCosts>(); in.finish();
    if (!std::isfinite(costs.initial) || !std::isfinite(costs.final) || costs.initial < 0 || costs.final < 0)
        throw std::runtime_error("invalid regional coordination cost checkpoint");
    return costs;
}
} // namespace

std::optional<AutoResult> run_regional(SfmConfig cfg, const std::string& matches,
                                      const std::string& features, const std::string& images,
                                      const fs::path& sparse, MatchesIndex* indexed) {
    if (cfg.pairs != "spatial-blocks") return std::nullopt;
    MatchesIndex index;
    if (indexed) index = std::move(*indexed);
    else if (!indexMatches(matches, index)) throw std::runtime_error("regional mapping cannot index matches.bin");
    if (!index.complete) throw std::runtime_error("regional mapping needs a complete matches.bin");
    if (!index.metadata.hasCameras()) throw std::runtime_error("regional mapping requires cached camera setup; use auto to create it");
    CameraSetup cameras; loadCameraSetup(index.metadata, cameras);
    SensorCaptures sensors = loadSensorCaptures(cfg, !cfg.quiet);
    std::unique_ptr<PriorSource> gps = makeSensorPriors(cfg, sensors, index.metadata, cameras.ids);
    if (!gps) { cfg.sensor_pairs = true; gps = makeExifGpsPriors(cfg, images, index.metadata, cameras, !cfg.quiet); }
    std::vector<Vec3> positions(index.images.size());
    for (uint32_t i = 0; i < index.images.size(); ++i) {
        cancel::check();
        if (!gps || !gps->position(i, positions[i]) || !std::isfinite(positions[i].x) ||
            !std::isfinite(positions[i].y) || !std::isfinite(positions[i].z)) {
            slog::warn(Tag::Map, M::match_blocks_need_gps, {(long long)i, (long long)index.images.size()});
            throw std::runtime_error("regional mapping GPS input incomplete; retained caches will not be replaced by whole-capture mapping");
        }
        uint32_t count = 0;
        if (!peekFeatures((fs::path(features) / (index.images[i].name + ".bin")).string(), count) || count != index.images[i].num_features)
            throw std::runtime_error("regional mapping: cached matches do not index the current feature files");
    }
    const std::string featureDigest = regionalInputDigest(features);
    std::string signature = "regional-v2\n" + regional::regionalBaseSignature(stageSignature(cfg, CMD_MAP)) +
        regional::regionalBaseSignature(stageSignature(cfg, CMD_AUTO)) +
        "matches=" + fs::absolute(matches).string() + "\nbytes=" + std::to_string(fs::file_size(matches)) +
        "\ntime=" + std::to_string(fs::last_write_time(matches).time_since_epoch().count()) +
        "\nfeatures=" + featureDigest + "\n";
    fs::path root = fs::path(features).parent_path() / ".regional" / std::to_string(signatureHash(signature));
    const auto previous_root = regional::compatibleRegionalRoot(root, signature);
    root = previous_root;
    root = storage::regionalRoot(root, cfg.regional_work_dir);
    root = regional::compatibleRegionalRoot(root, signature);
    if (fs::exists(previous_root / "plan.bin") && !fs::exists(root / "plan.bin"))
        throw std::runtime_error("regional work directory changed: migrate existing checkpoints from " +
            previous_root.u8string() + " to " + root.u8string() + "; local models will not be recomputed");
    storage::requireSpace(root, 0);
    fs::create_directories(root);
    if (fs::exists(root / "signature.txt") && !resume::equivalentSignatures(resume::recorded(root / "signature.txt"), signature))
        throw std::runtime_error("regional checkpoint signature collision");
    if (!fs::exists(root / "signature.txt")) resume::store(root / "signature.txt", signature);
    configureMappingMemory(cfg, root.string());
    const size_t budget = cfg.bup.model_memory_bytes;
    const fs::path repair_matches = root / ("supplement-v1-" + std::to_string(signatureHash(stageSignature(cfg, CMD_MATCH))));
    std::vector<fs::path> supplemental;
    for (const auto& e : fs::directory_iterator(root)) if (e.is_directory() &&
        e.path().filename().string().rfind("supplement-v1-", 0) == 0) supplemental.push_back(e.path());
    std::sort(supplemental.begin(), supplemental.end());
    for (const auto& directory : supplemental) regional::loadRepairMatches(index, directory);
    std::vector<Reconstruction> original_names(1);
    for (uint32_t id = 0; id < index.images.size(); ++id) {
        Image image; image.id = id; image.name = index.images[id].name; image.registered = true;
        original_names[0].images.emplace(id, std::move(image));
    }
    resolveImageNames(original_names, images);
    std::map<uint32_t, regional::ExportInputPhoto> original_inputs;
    std::set<std::string> original_paths;
    const auto original_root = fs::canonical(fs::u8path(images));
    for (uint32_t id = 0; id < index.images.size(); ++id) {
        const auto name = fs::u8path(original_names[0].images.at(id).name);
        if (name.is_absolute() || name.empty() || std::any_of(name.begin(), name.end(), [](const auto& component) { return component == ".."; }))
            throw std::runtime_error("original photo identity is not a capture-relative name");
        const auto path = original_root / name;
        if (!fs::is_regular_file(path) || !original_paths.insert(fs::canonical(path).u8string()).second)
            throw std::runtime_error("original photo path missing or duplicate in complete input inventory: " + path.u8string());
        original_inputs.emplace(id, regional::ExportInputPhoto{path.u8string(), index.images[id].name, name.generic_u8string()});
    }
    original_names.clear(); original_names.shrink_to_fit();
    std::vector<Region> plan;
    if (fs::exists(root / "plan.bin")) plan = regional::readRegionalPlan(root, index.images.size());
    else { plan = regional::planRegions(index, positions, size_t(cfg.block_size)); savePlan(root, plan); }
    slog::out(Tag::Map, M::regional_plan, {(long long)plan.size(), cfg.block_size, root.string()});
    events::stage_begin(Stage::Map, index.images.size()); events::map_begin(index.images.size());
    Reconstruction summary; summary.cameras = cameras.cameras;
    std::optional<AttitudeRef> capture_attitude;
    auto captureAttitude = [&]() -> const AttitudeRef& {
        if (!capture_attitude) {
            Reconstruction names;
            for (uint32_t i = 0; i < index.images.size(); ++i) {
                Image im; im.id = i; im.name = index.images[i].name; im.registered = true;
                names.images.emplace(i, std::move(im));
            }
            slog::diag(Tag::Map, "[regional] reading common recorded-attitude reference for %zu input photos\n", index.images.size());
            capture_attitude = regional::regionalRecordedAttitude(names, images, cfg.exif_orientation == "apply");
            slog::diag(Tag::Map, "[regional] recorded attitude available for %zu/%zu photos\n", capture_attitude->image_ids.size(), index.images.size());
        }
        return *capture_attitude;
    };
    std::vector<uint32_t> source(index.images.size());
    std::vector<std::vector<SavedRegion>> completed(plan.size());
    regional::RepairModelCache repair_models(budget / 32);
    regional::RepairPairCatalog repair_pairs;
    std::set<std::string> rejected_repair_inputs;
    size_t rejected_repair_bytes = 0;
    struct Deferred { size_t region; SavedRegion original; };
    std::vector<Deferred> deferred;
    std::vector<regional::RegionalSourceFrame> original_source_frames;
    bool source_frames_recorded = false;
    std::optional<regional::SharedIndex> bootstrap_sample;
    struct Component { SavedRegion local; fs::path stage; std::vector<SavedRegion> aligned; bool empty = false; };
    std::vector<std::vector<Component>> components(plan.size());
    std::vector<uint8_t> region_seen(plan.size());
    std::vector<uint8_t> locally_registered(index.images.size());
    std::vector<uint8_t> measured_positioned(index.images.size());
    std::vector<uint8_t> final_verified(index.images.size());
    std::vector<uint8_t> quality_checked(index.images.size()), visual_passed(index.images.size()), gps_passed(index.images.size()), attitude_passed(index.images.size()), scale_passed(index.images.size());
    {
        std::istringstream recorded(resume::recorded(root / "measured-positioned.txt")); size_t id;
        while (recorded >> id) {
            if (id >= measured_positioned.size()) throw std::runtime_error("invalid measured photo checkpoint");
            measured_positioned[id] = 1;
        }
    }
    std::vector<uint32_t> verified_pairs(index.images.size());
    for (const auto& p : index.pairs) if (p.count) { verified_pairs.at(p.image1)++; verified_pairs.at(p.image2)++; }
    auto publish = [&](const std::vector<SavedRegion>& saved, size_t r) {
        for (const auto& item : saved) {
            auto rec = readSaved(item, budget); addSummary(summary, rec, plan[r], source, progress::kMaxPoints / plan.size(), gps.get());
        }
    };
    auto recordMeasured = [&](const std::set<uint32_t>& registered) {
        for (uint32_t id : registered) measured_positioned.at(id) = 1;
        std::string ids;
        for (size_t id = 0; id < measured_positioned.size(); ++id)
            if (measured_positioned[id]) ids += std::to_string(id) + "\n";
        resume::store(root / "measured-positioned.txt", ids);
    };
    auto audit = [&]() {
        std::vector<uint32_t> owner(index.images.size());
        for (uint32_t r = 0; r < plan.size(); ++r) for (uint32_t id : plan[r].core) owner.at(id) = r;
        std::ofstream out(root / "photo-status.csv.part");
        out << "image_id,name,owner_region,verified_pairs,status\n";
        for (uint32_t id = 0; id < index.images.size(); ++id) {
            const char* state = summary.images.count(id) ? (final_verified[id] ? "aligned" : quality_checked[id] ?
                !visual_passed[id] ? "quality_failed" : !gps_passed[id] ? "gps_failed" : !scale_passed[id] ? "scale_failed" : !attitude_passed[id] ? "attitude_failed" : "validated_pose_pending_export" :
                measured_positioned[id] ? "positioned_gps_attitude" : "awaiting_coordination") : locally_registered[id] ? "unaligned_component" :
                !region_seen.at(owner[id]) ? "pending_region" : !index.images[id].num_features ? "no_features" :
                verified_pairs[id] ? "unregistered" : "no_verified_matches";
            std::string name = index.images[id].name; size_t at = 0;
            while ((at = name.find('"', at)) != std::string::npos) { name.insert(at, 1, '"'); at += 2; }
            out << id << ",\"" << name << "\"," << owner[id] << ',' << verified_pairs[id] << ',' << state << '\n';
        }
        out.flush(); if (!out) throw std::runtime_error("cannot write regional photo status"); out.close();
        regional::replaceFile(root / "photo-status.csv.part", root / "photo-status.csv");
    };
    auto finalizeRegion = [&](size_t r) {
        if (components[r].empty()) return;
        fs::path stage = stageFile(root, 0, r); std::vector<SavedRegion> saved;
        for (const auto& task : components[r]) for (const auto& item : task.aligned) {
            SavedRegion target{stage.string() + "-" + std::to_string(saved.size()) + ".bin", item.memory};
            target.origin = target.path.filename();
            storage::retainFile(item.path, target.path); saved.push_back(target);
        }
        if (!saved.empty()) { commitSavedStage(stage, saved, {}); completed[r] = std::move(saved); }
    };
    std::unique_ptr<regional::RegionalWorkingStore> active_working;
    std::unique_ptr<regional::CanonicalScene> active_scene;
    std::unique_ptr<regional::ImmutableSceneTopology> active_index;
    std::set<uint32_t> recovery_targets, recovered_pose_targets;
    std::map<std::string, size_t> working_sources;
    std::vector<size_t> working_slots;
    auto source_label = [](const SavedRegion& item) { return (item.origin.empty() ? item.path.filename() : item.origin).u8string(); };
    auto require_fixed_frame = [&](const SavedRegion& item) {
        if (!active_scene || !active_scene->initialized()) throw std::logic_error("fixed canonical frame requires an accepted scene");
        const auto hash = spirula::sha256_file(item.path.u8string());
        if (hash.empty() || resume::recorded(item.path.string() + ".fixed-frame.txt") !=
            "held-canonical-frame-v1\n" + active_scene->identity() + "\n" + hash + "\n")
            throw std::runtime_error("cached recovery source lacks unchanged accepted-frame provenance; original evidence retained");
    };
    auto repair = [&](size_t r, const std::set<uint32_t>& required, const Reconstruction* seed,
                      const fs::path& accepted_stage, bool keep_all = true) {
        const bool canonical_repair = active_scene && active_scene->initialized();
        const auto& authority = canonical_repair ? active_scene->acceptedPoses() : summary;
        auto anchors = regional::registeredIds(authority);
        if (canonical_repair) for (uint32_t id : recovery_targets) anchors.erase(id);
        if (!canonical_repair) for (uint32_t id : required) anchors.erase(id);
        auto previous_seed = canonical_repair ? std::optional<Reconstruction>{} : repairSeed(root, accepted_stage, required, budget);
        const Reconstruction* current_seed = previous_seed ? &*previous_seed : seed;
        if (current_seed && !canonical_repair) {
            auto candidate = *current_seed;
            if (stageRegional(candidate, summary, *gps, cfg.max_error)) {
                auto saved = commitStage(accepted_stage, {candidate}); publish(saved, r); audit(); return saved;
            }
        }
        std::vector<SavedRegion> donors; std::vector<fs::path> donor_paths;
        for (const auto& models : completed) for (const auto& item : models) { donors.push_back(item); donor_paths.push_back(item.path); }
        if (!active_working) repair_models.sync(donor_paths);
        const size_t cap = std::max<size_t>(64, std::min<size_t>(1024, size_t(cfg.block_size)));
        for (size_t limit = 64; limit <= cap; limit = limit == cap ? cap + 1 : std::min(cap, limit * 2)) {
            cancel::check();
            const size_t previous = index.pairs.size();
            try {
                if (limit > 64) regional::supplementRepairMatches(cfg, index, required, anchors, positions, cameras,
                    features, repair_matches, std::min<size_t>(256, limit / 2), budget);
            } catch (const regional::RegionInputOverBudget&) {
                slog::diag(Tag::Map, "[regional] supplemental match batch reached memory budget; verified results retained\n");
            }
            for (size_t p = previous; p < index.pairs.size(); ++p) {
                verified_pairs.at(index.pairs[p].image1)++; verified_pairs.at(index.pairs[p].image2)++;
            }
            audit(); Region context = regional::repairContext(index, required, anchors, limit, &repair_pairs);
            std::optional<Reconstruction> anchored_seed;
            const auto ranked_donors = active_working ? std::vector<std::pair<size_t, size_t>>{} : repair_models.rank(context, anchors);
            if (active_working) {
                std::set<uint32_t> targets;
                for (auto id : context.images) if (anchors.count(id)) targets.insert(id);
                if (!targets.empty()) {
                    if (!active_scene || !active_index) throw std::runtime_error("canonical recovery state missing");
                    auto gathered = regional::gatherCanonicalSeed(*active_index, *active_scene, targets,
                        [&](uint32_t image) { return features / fs::u8path(index.images.at(image).name + ".bin"); }, budget / 8);
                    anchored_seed = regional::anchoredRepairSeed(gathered, context, authority, cfg.max_error, gps.get(), true);
                    if (anchored_seed->points3D.empty()) anchored_seed.reset();
                }
            } else if (!ranked_donors.empty()) {
                anchored_seed.emplace(); std::set<uint32_t> outstanding_anchors;
                for (uint32_t id : context.images) if (anchors.count(id)) outstanding_anchors.insert(id);
                for (const auto& ranked : ranked_donors) {
                    const auto m = ranked.second;
                    bool useful = false; for (uint32_t id : outstanding_anchors) useful |= repair_models.contains(m, id);
                    if (!useful) continue;
                    Reconstruction views;
                    try { views = repair_models.subset(m, context.images); }
                    catch (const std::length_error&) { views = readSaved(donors.at(m), budget); }
                    auto selected = regional::anchoredRepairSeed(views, context, summary, cfg.max_error, gps.get());
                    regional::mergeAnchoredRepairSeed(*anchored_seed, selected);
                    for (const auto& kv : selected.images) outstanding_anchors.erase(kv.first);
                    if (modelResidentBytes(*anchored_seed) > budget / 8)
                        throw regional::RegionInputOverBudget();
                    if (outstanding_anchors.empty()) break;
                }
                if (anchored_seed->points3D.empty()) anchored_seed.reset();
            }
            const Reconstruction* run_seed = anchored_seed ? &*anchored_seed : canonical_repair ? nullptr : current_seed;
            if (current_seed && !canonical_repair) {
                const auto retained = regional::registeredIds(*current_seed);
                context.images.insert(context.images.end(), retained.begin(), retained.end());
                std::sort(context.images.begin(), context.images.end());
                context.images.erase(std::unique(context.images.begin(), context.images.end()), context.images.end());
            }
            size_t common = 0; std::string key = "repair-input-v9-measurement-closed-canonical-track-bindings\n" + signature + "\n" + accepted_stage.filename().string() +
                "\nkeep-all=" + std::to_string(keep_all) + "\n";
            for (uint32_t id : context.images) { common += anchors.count(id); key += std::to_string(id) + ","; }
            key += repair_matches.filename().string();
            if (active_scene) key += "\ncanonical-snapshot=" + active_scene->identity();
            key += regional::repairGeometryIdentity(context, authority, run_seed);
            for (const auto& ranked : ranked_donors) key += repair_models.stamp(ranked.second) + "\n";
            for (auto p : repair_pairs.incident(context.images)) { const auto& pair = index.pairs.at(p); if (pair.count &&
                std::binary_search(context.images.begin(), context.images.end(), pair.image1) &&
                std::binary_search(context.images.begin(), context.images.end(), pair.image2)) {
                key += std::to_string(pair.image1) + ":" + std::to_string(pair.image2) + ":" +
                    std::to_string(pair.count) + ":" + std::to_string(pair.offset) + ":" +
                    (pair.source ? index.sources.at(pair.source - 1) : matches) + ";";
            } }
            if (common < 3 && (run_seed || keep_all)) continue;
            if (canonical_repair && (!anchored_seed || common < 3)) continue;
            if (rejected_repair_inputs.count(key)) continue;
            fs::path attempt = root / ("repair-v9-" + accepted_stage.filename().string() + "-" +
                std::to_string(signatureHash(key)) + (anchored_seed ? "-anchored" : run_seed ? "-grow" : "-seed"));
            if (resume::recorded(attempt.string() + ".rejected") == key) continue;
            auto saved = savedStage(attempt); std::vector<Reconstruction> models;
            try {
                if (!fs::exists(attempt.string() + ".done")) {
                    slog::diag(Tag::Map, "[regional] repair region %zu: %zu required images, %zu support, %zu registered anchors\n",
                        r + 1, required.size(), context.images.size() - required.size(), common);
                    models = regional::reconstructRepair(cfg, index, context, matches, features, images, cameras, gps.get(), run_seed, budget, bool(anchored_seed));
                    saved = commitStage(attempt, models);
                } else for (const auto& item : saved) models.push_back(readSaved(item, budget));
            } catch (const regional::RegionInputOverBudget&) { break; }
              catch (const BAOverBudget&) { break; }
              catch (const BAIndexCapacity&) { break; }
              catch (const MergeOverBudget&) { break; }
            std::vector<Reconstruction> aligned; std::set<uint32_t> measured_ids;
            for (size_t model_id = 0; model_id < models.size(); ++model_id) if (models[model_id].numRegistered() >= 3 && !models[model_id].points3D.empty()) {
                auto& model = models[model_id];
                bool accepted = false;
                if (canonical_repair) {
                    regional::canonicalizeImageIdentity(model, original_inputs);
                    regional::RestoredCanonicalFeatureBindings bindings;
                    try {
                        bindings = regional::restoreAcceptedFeatureBindings(*active_index, *active_scene, model, budget / 8, cfg.max_error);
                    } catch (const regional::CanonicalRepairBindingRejected& e) {
                        const auto evidence = fs::path(attempt.string() + ".binding-rejected-" + std::to_string(model_id) + ".txt");
                        if (!fs::exists(evidence)) resume::store(evidence, "accepted=" + active_scene->identity() + "\nreason=" + e.what() + "\n");
                        slog::diag(Tag::Map, "[canonical-repair] visual binding candidate rejected; original attempt retained: %s\n", e.what());
                        continue;
                    }
                    HeldRepairSeedContract(*anchored_seed).requireUnchanged(model, "regional.accepted-feature-binding");
                    slog::diag(Tag::Map, "[canonical-repair] original feature bindings restored: %llu features, %llu raw rows moved; %llu raw rows retained\n",
                        (unsigned long long)bindings.reassociated_features, (unsigned long long)bindings.moved_raw_rows,
                        (unsigned long long)bindings.raw_rows);
                    const auto measured = regional::regionalGpsQuality(model, *gps);
                    accepted = measured.photos >= 3 && measured.inliers == measured.photos;
                } else accepted = stageRegional(model, summary, *gps, cfg.max_error);
                if (!accepted && !keep_all && !canonical_repair) {
                    auto measured = regional::measuredRegionalAlignment(model, *gps,
                        captureAttitude());
                    if (measured.ok) {
                        applySim3(model, measured.transform); accepted = true;
                        for (uint32_t id : regional::registeredIds(model)) if (required.count(id)) measured_ids.insert(id);
                        slog::diag(Tag::Map, "[regional] previously unregistered photos positioned by measured GPS/attitude: %zu, 3D RMS %.4f m; visual connection unresolved\n",
                            measured_ids.size(), measured.rms);
                    }
                }
                if (accepted) aligned.push_back(std::move(model));
            }
            size_t recovered = 0;
            std::set<uint32_t> covered;
            for (const auto& model : aligned) for (uint32_t id : regional::registeredIds(model)) covered.insert(id);
            for (uint32_t id : required) recovered += covered.count(id);
            if ((keep_all && recovered == required.size()) || (!keep_all && recovered)) {
                if (!measured_ids.empty()) recordMeasured(measured_ids);
                auto accepted = commitStage(accepted_stage, aligned);
                if (canonical_repair) for (const auto& item : accepted) {
                    const auto hash = spirula::sha256_file(item.path.u8string());
                    if (hash.empty()) throw std::runtime_error("held recovery immutable model hash missing");
                    resume::store(item.path.string() + ".fixed-frame.txt", "held-canonical-frame-v1\n" + active_scene->identity() + "\n" + hash + "\n");
                }
                publish(accepted, r); audit();
                slog::diag(Tag::Map, "[regional] repair staged: %zu/%zu required images; final visual quality pending, remaining photos retained\n", recovered, required.size());
                return accepted;
            }
            if (key.size() <= budget / 64) {
                if (rejected_repair_bytes > budget / 64 - key.size()) { rejected_repair_inputs.clear(); rejected_repair_bytes = 0; }
                if (rejected_repair_inputs.insert(key).second) rejected_repair_bytes += key.size();
            }
            resume::store(attempt.string() + ".rejected", key);
        }
        return std::vector<SavedRegion>{};
    };
    auto recover = [&](bool after_coordination) {
        for (;;) {
            bool advanced = false;
            for (size_t r = 0; !after_coordination && r < plan.size(); ++r) for (auto& task : components[r]) if (!task.empty && task.aligned.empty()) {
                cancel::check(); auto rec = readSaved(task.local, budget); auto required = regional::registeredIds(rec);
                if (required.empty()) continue;
                if (rec.numRegistered() >= 3 && !rec.points3D.empty() && stageRegional(rec, summary, *gps, cfg.max_error)) {
                    task.aligned = commitStage(task.stage, {rec}); publish(task.aligned, r);
                } else {
                    task.aligned = repair(r, required, &rec, task.stage);
                    if (task.aligned.empty()) {
                        auto measured = regional::measuredRegionalAlignment(rec, *gps,
                            captureAttitude());
                        if (measured.ok) {
                            applySim3(rec, measured.transform);
                            resume::store(task.stage.string() + ".sensor", "gps-attitude-v1\n");
                            recordMeasured(required);
                            task.aligned = commitStage(task.stage, {rec});
                            publish(task.aligned, r);
                            slog::diag(Tag::Map,
                                "[regional] measured positioning, visual connection unresolved: %zu photos, GPS %d/%d inliers, 3D RMS %.4f m, up spread %.3f deg, heading spread %.3f deg, GPS yaw correction %.3f deg\n",
                                required.size(), measured.inliers, measured.positions, measured.rms,
                                measured.up_spread_deg, measured.heading_spread_deg, measured.yaw_deg);
                        }
                    }
                }
                if (!task.aligned.empty()) { advanced = true; finalizeRegion(r); progress::model(summary, true); }
            }
            for (size_t r = 0; after_coordination && r < plan.size(); ++r) {
                std::set<uint32_t> missing;
                for (uint32_t id : plan[r].core) if ((!summary.images.count(id) || recovery_targets.count(id)) && !recovered_pose_targets.count(id)) missing.insert(id);
                std::vector<uint32_t> ids(missing.begin(), missing.end());
                for (size_t begin = 0; begin < ids.size(); begin += 128) {
                    std::set<uint32_t> group(ids.begin() + begin, ids.begin() + std::min(ids.size(), begin + 128));
                    std::string key; for (uint32_t id : group) key += std::to_string(id) + ",";
                    if (active_scene) key += active_scene->identity();
                    fs::path stage = root / ("missing-canonical-v3-region-" + std::to_string(r) + "-" + std::to_string(signatureHash(key)));
                    auto saved = savedStage(stage);
                    if (!fs::exists(stage.string() + ".done")) saved = repair(r, group, nullptr, stage, false);
                    bool new_input = false;
                    for (const auto& candidate : saved) new_input |= !working_sources.count(source_label(candidate));
                    if (!saved.empty() && new_input) {
                        for (const auto& item : saved) require_fixed_frame(item);
                        publish(saved, r);
                        advanced = true;
                        for (const auto& item : saved) {
                            const auto restored = model_store_detail::readModel(item.path, false);
                            for (uint32_t id : group) {
                                const auto at = restored.images.find(id);
                                if (at != restored.images.end() && at->second.registered) recovered_pose_targets.insert(id);
                            }
                        }
                        {
                            fs::path target_stage = root / ("recovered-v3-region-" + std::to_string(r));
                            auto additions = savedStage(target_stage);
                            for (const auto& item : saved) {
                                SavedRegion target{target_stage.string() + "-" + std::to_string(additions.size()) + ".bin", item.memory, fs::u8path(source_label(item))};
                                storage::retainFile(item.path, target.path);
                                if (active_scene) storage::retainFile(item.path.string() + ".fixed-frame.txt", target.path.string() + ".fixed-frame.txt");
                                additions.push_back(target); completed[r].push_back(target);
                            }
                            commitSavedStage(target_stage, additions, {});
                        }
                    }
                }
            }
            audit();
            if (!advanced || after_coordination) break;
        }
        size_t pending = 0; for (const auto& tasks : components) for (const auto& task : tasks) pending += !task.empty && task.aligned.empty();
        regional::atomicModel(root / "recovery-consensus.bin", summary);
        slog::diag(Tag::Map, "[regional] recovery queue: %zu pending components, %zu unregistered photos; retained for post-coordination recovery\n",
            pending, index.images.size() - summary.numRegistered());
        if (!after_coordination) for (size_t r = 0; r < plan.size(); ++r) finalizeRegion(r);
    };
    audit();
    double started = now();
    for (size_t r = 0; r < plan.size(); ++r) {
        cancel::check(); fs::path stage = stageFile(root, 0, r); completed[r] = savedStage(stage);
        if (completed[r].empty() && !fs::exists(stage.string() + ".done")) {
            slog::out(Tag::Map, M::regional_region, {(long long)(r + 1), (long long)plan.size(), 0,
                (long long)plan[r].core.size(), (long long)(plan[r].images.size() - plan[r].core.size())});
            try {
                std::string input_key;
                for (uint32_t i : plan[r].images) input_key += std::to_string(i) + ",";
                fs::path local_stage = root / ("local-region-" + std::to_string(r) + "-" + std::to_string(signatureHash(input_key)));
                std::vector<SavedRegion> cached_local = savedStage(local_stage);
                std::vector<Reconstruction> models;
                if (!fs::exists(local_stage.string() + ".done")) {
                    models = regional::reconstructRepair(cfg, index, plan[r], matches, features, images, cameras, gps.get(), nullptr, budget);
                    cached_local = commitStage(local_stage, models);
                } else {
                    slog::diag(Tag::Map, "[regional] resumed local geometry for region %zu/%zu\n", r + 1, plan.size());
                }
                models.clear(); models.shrink_to_fit();
                auto& tasks = components[r]; tasks.clear();
                for (size_t m = 0; m < cached_local.size(); ++m) {
                    Component task{cached_local[m], root / ("aligned-v1-" + local_stage.filename().string() + "-" + std::to_string(m))};
                    auto rec = readSaved(task.local, budget);
                    task.empty = rec.numRegistered() == 0;
                    for (uint32_t id : regional::registeredIds(rec)) locally_registered.at(id) = 1;
                    task.aligned = savedStage(task.stage);
                    if (fs::exists(task.stage.string() + ".sensor"))
                        for (uint32_t id : regional::registeredIds(rec)) measured_positioned.at(id) = 1;
                    if (!fs::exists(task.stage.string() + ".done") && rec.numRegistered() >= 3 && !rec.points3D.empty() &&
                        stageRegional(rec, summary, *gps, cfg.max_error)) task.aligned = commitStage(task.stage, {rec});
                    publish(task.aligned, r);
                    if (task.aligned.empty()) slog::diag(Tag::Map,
                        "[regional] region %zu component %zu deferred: %zu images retained for connection repair\n", r + 1, m, size_t(rec.numRegistered()));
                    tasks.push_back(std::move(task));
                }
                finalizeRegion(r);
            } catch (const std::exception& e) {
                if (!dynamic_cast<const MergeOverBudget*>(&e) && !dynamic_cast<const BAOverBudget*>(&e) &&
                    !dynamic_cast<const BAIndexCapacity*>(&e) && !dynamic_cast<const regional::RegionInputOverBudget*>(&e)) { audit(); throw; }
                if (plan[r].core.size() < 32) throw;
                size_t middle = plan[r].core.size() / 2;
                Region a, b; a.core.assign(plan[r].core.begin(), plan[r].core.begin() + middle);
                b.core.assign(plan[r].core.begin() + middle, plan[r].core.end());
                auto halo = [&](Region& child) {
                    child.images = child.core; std::set<uint32_t> own(child.core.begin(), child.core.end());
                    std::map<uint32_t, uint64_t> votes;
                    for (const auto& p : index.pairs) {
                        if (own.count(p.image1) && !own.count(p.image2)) votes[p.image2] += p.count;
                        if (own.count(p.image2) && !own.count(p.image1)) votes[p.image1] += p.count;
                    }
                    std::vector<std::pair<uint64_t, uint32_t>> rank; for (const auto& kv : votes) rank.emplace_back(kv.second, kv.first);
                    std::sort(rank.rbegin(), rank.rend());
                    for (size_t k = 0; k < std::min(rank.size(), std::max<size_t>(8, child.core.size() / 4)); ++k) child.images.push_back(rank[k].second);
                    std::sort(child.images.begin(), child.images.end());
                };
                halo(a); halo(b); plan[r] = std::move(a); plan.push_back(std::move(b)); completed.resize(plan.size()); components.resize(plan.size()); region_seen.resize(plan.size()); savePlan(root, plan);
                slog::diag(Tag::Map, "[regional] region exceeded working budget; split core and retry, completed regions retained\n");
                --r; continue;
            }
        } else slog::diag(Tag::Map, "[regional] resumed completed region %zu/%zu\n", r + 1, plan.size());
        for (auto& item : completed[r]) {
            const auto replacement = savedStage(root / ("rehabilitated-v3-" + item.path.filename().string()));
            if (!replacement.empty()) item = replacement.at(0);
        }
        const auto additions = savedStage(root / ("recovered-v3-region-" + std::to_string(r)));
        completed[r].insert(completed[r].end(), additions.begin(), additions.end());
        for (const auto& item : completed[r]) { auto rec = readSaved(item, budget); addSummary(summary, rec, plan[r], source, progress::kMaxPoints / plan.size(), gps.get()); }
        region_seen[r] = 1;
        events::progress(Stage::Map, r + 1, plan.size());
        progress::model(summary, true); audit();
    }
    recover(false);
    for (size_t r = 0; r < components.size(); ++r) for (const auto& task : components[r])
        if (!task.empty && task.aligned.empty()) {
            auto original = task.local; original.origin = original.path.filename(); deferred.push_back({r, original});
        }
    std::vector<SavedRegion> all;
    fs::path coordination;

    const size_t cache = regional::regionalWorkingCache(budget);
    const size_t shared_memory = std::min<size_t>(256u << 20, budget / 16);
    regional::RegionalPoseAnchors source_anchors;
    bool initial_anchors_restored = false;
    std::vector<regional::RegionalSourceFrame> historical_frames;
    const bool historical_required = index.images.size() == 122415;
    std::vector<uint32_t> owners(index.images.size(), UINT32_MAX);
    for (uint32_t r = 0; r < plan.size(); ++r) for (auto image : plan[r].core) owners.at(image) = r;
    uint64_t scene_round = 0;
    std::string displayed_scene_phase;
    auto sceneProgress = [&](const char* phase, uint64_t done, uint64_t total) {
        const std::string name = phase;
        const Stage stage = name.find("seed") != std::string::npos ? Stage::Seed :
            name.find("sort") != std::string::npos || name.find("index") != std::string::npos ? Stage::Merge :
            name.find("source") != std::string::npos ? Stage::Load : Stage::Refine;
        if (name != displayed_scene_phase) { events::stage_begin(stage, total); displayed_scene_phase = name; }
        events::progress(stage, done, total);
        std::ostringstream value;
        value << "{\"schema\":\"canonical-scene-v1\",\"phase\":\"" << phase << "\",\"done\":" << done << ",\"total\":" << total
              << ",\"generation\":" << (active_scene && active_scene->initialized() ? active_scene->generation() : 0)
              << ",\"round\":" << scene_round << ",\"host_budget_bytes\":" << budget
              << ",\"host_available_bytes\":" << spirula::availableRamBytes() << ",\"full_capture_verified\":false}\n";
        resume::store(coordination / "working-progress.json", value.str());
        static double last_log = 0;
        if (done == total || now() - last_log >= 5) {
            slog::diag(Tag::Map, "[canonical-scene] %s: %llu/%llu, generation %llu, round %llu\n", phase,
                (unsigned long long)done, (unsigned long long)total, (unsigned long long)(active_scene && active_scene->initialized() ? active_scene->generation() : 0),
                (unsigned long long)scene_round); last_log = now();
        }
    };
    auto coordinate = [&]() -> bool {
        all.clear(); for (const auto& v : completed) all.insert(all.end(), v.begin(), v.end());
        if (all.empty()) throw std::runtime_error("regional mapping produced no visual models");
        const auto coordinate_identity = std::to_string(signatureHash(regional::regionalBaseSignature(signature)));
        coordination = root / ("canonical-native-v2-" + coordinate_identity);
        if (!fs::exists(coordination) && regional::forkCanonicalResume(root / ("canonical-native-v1-" + coordinate_identity), coordination, budget / 4))
            slog::diag(Tag::Map, "[canonical-scene] restored accepted immutable prefix; rejected recovery sources retained separately\n");
        fs::create_directories(coordination);
        resume::store(coordination / "coverage.txt", "full_capture_verified=false\nCanonical scene acceptance pending\n");
        const auto& recorded_attitude = captureAttitude();
        if (!source_frames_recorded) {
            const auto evidence_root = root / "source-evidence-v1";
            auto evidence = resume::recorded(evidence_root / "inputs.txt");
            if (evidence.empty()) {
                for (size_t r = 0; r < plan.size(); ++r) for (const auto& item : savedStage(stageFile(root, 0, r))) {
                    const auto path = evidence_root / item.path.filename(); fs::create_directories(path.parent_path());
                    if (!fs::exists(path)) storage::retainFile(item.path, path);
                    evidence += path.filename().string() + ":" + std::to_string(fs::file_size(path)) + "\n";
                }
                for (const auto& task : deferred) {
                    const auto path = evidence_root / task.original.path.filename();
                    if (!fs::exists(path)) storage::retainFile(task.original.path, path);
                    evidence += path.filename().string() + ":" + std::to_string(fs::file_size(path)) + "\n";
                }
                resume::store(evidence_root / "inputs.txt", evidence);
            }
            std::istringstream manifest(evidence); std::string row;
            while (std::getline(manifest, row)) {
                const auto split = row.find(':');
                if (split == std::string::npos || fs::path(row.substr(0, split)).filename() != fs::path(row.substr(0, split))) throw std::runtime_error("invalid immutable regional source manifest");
                const auto path = evidence_root / row.substr(0, split);
                if (fs::file_size(path) != std::stoull(row.substr(split + 1))) throw std::runtime_error("immutable original source changed");
                const auto views = model_store_detail::readModel(path, false);
                const bool positioned = regional::regionalGpsQuality(views, *gps).accepted();
                const bool certified = regional::regionalGpsGauge(views, *gps, recorded_attitude);
                if (positioned || certified) original_source_frames.push_back({path.filename().string(), views, positioned, certified});
            }
            historical_frames = regional::history::loadRequired(root, featureDigest, index.images.size(), *gps, recorded_attitude, historical_required);
            source_frames_recorded = true;
        }
        if (!active_working) {
            active_working = std::make_unique<regional::RegionalWorkingStore>(coordination / "immutable-raw", budget / 4);
            for (size_t source = 0; source < active_working->sourceCount(); ++source)
                if (!working_sources.emplace(active_working->sourceLabel(source), source).second) throw std::runtime_error("immutable source label conflict");
        }
        if (!active_scene) active_scene = std::make_unique<regional::CanonicalScene>(coordination / "scene", budget / 4);
        std::vector<std::pair<std::string, fs::path>> original_inputs_for_stage;
        if (!active_scene->initialized()) {
            std::map<std::string, fs::path> original_stage_sources;
            for (const auto& item : all) original_stage_sources.emplace(source_label(item), item.path);
            for (const auto& task : deferred) original_stage_sources.emplace(source_label(task.original), task.original.path);
            for (const auto& row : original_stage_sources) original_inputs_for_stage.push_back(row);
        }
        const auto initialization_directory = coordination / "initialization";
        if (active_scene->initialized() && !initial_anchors_restored) {
            const auto initial = regional::loadCanonicalInitialStage(initialization_directory, regional::regionalBaseSignature(signature), {}, budget / 4, true);
            if (!initial) throw std::runtime_error("accepted canonical scene lacks original held-source stage");
            for (uint32_t image : initial->fixedImages) {
                const auto& held = initial->poses.images.at(image); const auto& current = active_scene->poses().images.at(image);
                if (!held.registered || current.name != held.name || current.camera_id != held.camera_id)
                    throw std::runtime_error("canonical original held-source image identity differs");
                source_anchors.emplace(image, held.pose);
            }
            initial_anchors_restored = true;
            std::set<std::string> present;
            for (const auto& item : active_scene->binding().sources) present.insert(item.label);
            for (const auto& region : completed) for (const auto& item : region) present.insert(source_label(item));
            for (const auto& task : deferred) present.insert(source_label(task.original));
            for (const auto& record : initial->deferred) if (present.insert(record.label).second) {
                if (record.region >= completed.size()) throw std::runtime_error("restored original deferred source has invalid region");
                deferred.push_back({size_t(record.region), {record.original, size_t(record.memory), fs::u8path(record.label)}});
            }
        }
        const auto prepared_initialization = active_scene->initialized() ? std::optional<regional::CanonicalInitialStage>{} :
            regional::loadCanonicalInitialStage(initialization_directory, regional::regionalBaseSignature(signature), original_inputs_for_stage, budget / 4);
        if (!active_scene->initialized() && active_working->sourceCount() && !prepared_initialization)
            throw std::runtime_error("immutable catalogue lacks proven initial P/K/source stage; incomplete data retained");
        Reconstruction starting_poses = active_scene->initialized() ? active_scene->poses() : summary;
        bool appended = false;
        std::vector<SavedRegion> imported;
        if (!active_scene->initialized()) {
            if (prepared_initialization) {
                starting_poses = prepared_initialization->poses;
                regional::canonicalizeImageIdentity(starting_poses, original_inputs);
                source_anchors.clear();
                for (uint32_t image : prepared_initialization->fixedImages) source_anchors.emplace(image, starting_poses.images.at(image).pose);
                std::set<std::string> selected, already_deferred;
                for (const auto& source : prepared_initialization->sources) selected.insert(source.label);
                for (const auto& task : deferred) already_deferred.insert(source_label(task.original));
                for (const auto& record : prepared_initialization->deferred) {
                    if (record.region >= completed.size()) throw std::runtime_error("initial deferred region identity differs");
                    if (already_deferred.insert(record.label).second) deferred.push_back({size_t(record.region), {record.original, size_t(record.memory), fs::u8path(record.label)}});
                }
                for (auto& region : completed) region.erase(std::remove_if(region.begin(), region.end(), [&](const SavedRegion& item) {
                    if (selected.count(source_label(item))) return false;
                    if (!already_deferred.count(source_label(item))) throw std::runtime_error("initial stage source selection coverage differs");
                    return true;
                }), region.end());
                all.clear(); for (const auto& region : completed) all.insert(all.end(), region.begin(), region.end());
                for (const auto& source : prepared_initialization->sources) imported.push_back({source.aligned, size_t(source.memory), fs::u8path(source.label)});
                if (active_working->sourceCount() > imported.size()) throw std::runtime_error("initial immutable source catalogue has unexpected suffix");
                for (size_t slot = 0; slot < active_working->sourceCount(); ++slot)
                    if (active_working->sourceLabel(slot) != prepared_initialization->sources.at(slot).label ||
                        active_working->knownSource(prepared_initialization->sources.at(slot).aligned) != slot)
                        throw std::runtime_error("initial immutable source prefix differs from proven stage");
                sceneProgress("initial-stage-restored", active_working->sourceCount(), imported.size());
            } else {
            regional::CanonicalInitialStage initial_stage;
            regional::SharedIndex sample;
            if (bootstrap_sample && bootstrap_sample->models == all.size()) { sample = std::move(*bootstrap_sample); bootstrap_sample.reset(); }
            else sample = regional::buildSharedIndex(all.size(), [&](size_t m) { return readSaved(all.at(m), budget); },
                coordination / "bootstrap", cache, shared_memory, size_t(cfg.regional_landmarks),
                [&](size_t m) { return model_store_detail::readModel(all.at(m).path, false); });
            std::set<size_t> roots; std::vector<regional::SharedGpsAnchor> gps_anchors;
            for (size_t m = 0; m < all.size(); ++m) {
                const auto views = model_store_detail::readModel(all[m].path, false);
                if (regional::regionalGpsQuality(views, *gps).accepted()) roots.insert(m);
                for (const auto& im : views.images) if (im.second.registered) gps_anchors.push_back({uint32_t(m), cameraCenter(im.second.pose), positions.at(im.first)});
            }
            const auto fit = regional::guardedSharedAlignment(sample, 50, shared_memory, roots, gps_anchors, true);
            if (!fit.deferred.empty()) {
                const std::set<size_t> weak(fit.deferred.begin(), fit.deferred.end()); size_t m = 0;
                summary.images.clear(); summary.points3D.clear(); std::fill(source.begin(), source.end(), 0);
                for (size_t r = 0; r < completed.size(); ++r) {
                    std::vector<SavedRegion> retained;
                    for (const auto& item : completed[r]) {
                        if (weak.count(m++)) deferred.push_back({r, item});
                        else { retained.push_back(item); addSummary(summary, model_store_detail::readModel(item.path, false), plan[r], source, 0, gps.get()); }
                    }
                    completed[r] = std::move(retained);
                }
                bootstrap_sample = regional::retainSharedModels(std::move(sample), weak, regional::registeredIds(summary));
                audit(); return false;
            }
            starting_poses.images.clear(); starting_poses.points3D.clear(); std::fill(source.begin(), source.end(), 0);
            std::map<uint32_t, uint64_t> calibration_support; size_t model = 0;
            for (size_t r = 0; r < completed.size(); ++r) {
                for (const auto& item : completed[r]) {
                    auto rec = readSaved(item, budget); regional::validateCanonicalImageIdentity(rec, original_inputs);
                    const auto transform = fit.transforms.at(model++); applySim3(rec, transform);
                    const auto path = regional::canonical_initial_detail::aligned(coordination / "aligned-inputs", rec);
                    imported.push_back({path, modelResidentBytes(rec), fs::u8path(source_label(item))});
                    initial_stage.sources.push_back({source_label(item), item.path, path, modelResidentBytes(rec), transform});
                    addSummary(starting_poses, rec, plan[r], source, 0, gps.get());
                    std::map<uint32_t, uint64_t> support;
                    for (const auto& im : rec.images) if (im.second.registered) support[im.second.camera_id] += im.second.numPoint3D();
                    for (const auto& k : support) if (k.second > calibration_support[k.first]) {
                        starting_poses.cameras[k.first] = rec.cameras.at(k.first); calibration_support[k.first] = k.second;
                    }
                }
            }
            regional::protectRegionalSources(starting_poses, original_source_frames, *gps, recorded_attitude, source_anchors);
            regional::canonicalizeImageIdentity(starting_poses, original_inputs);
            starting_poses.points3D.clear();
            for (auto& image : starting_poses.images) { image.second.points2D.clear(); image.second.point3D_ids.clear(); }
            initial_stage.poses = starting_poses;
            for (const auto& image : source_anchors) initial_stage.fixedImages.push_back(image.first);
            for (const auto& task : deferred) initial_stage.deferred.push_back({source_label(task.original), task.original.path, task.original.memory, task.region});
            regional::saveCanonicalInitialStage(initialization_directory, regional::regionalBaseSignature(signature), initial_stage);
            }
        } else {
            const auto accepted_sources = active_scene->sourceCount();
            if (!(regional::CanonicalScene::bindingFor(*active_working, active_scene->binding().topology, true, accepted_sources) == active_scene->binding()))
                throw std::runtime_error("accepted canonical immutable source prefix changed");
            std::set<std::string> accepted_labels;
            for (const auto& source : active_scene->binding().sources) accepted_labels.insert(source.label);
            std::vector<std::pair<std::string, fs::path>> admission_originals;
            for (const auto& item : all) if (!accepted_labels.count(source_label(item))) admission_originals.emplace_back(source_label(item), item.path);
            const auto admission_directory = coordination / ("admission-" + active_scene->identity());
            const auto admission_signature = active_scene->identity() + "." + regional::regionalBaseSignature(signature);
            auto admission = regional::loadCanonicalInitialStage(admission_directory, admission_signature, admission_originals, budget / 4);
            if (!admission && active_working->sourceCount() != accepted_sources)
                throw std::runtime_error("pending immutable raw suffix lacks proven canonical admission proposal; retained");
            if (admission) {
                starting_poses = admission->poses;
                regional::canonicalizeImageIdentity(starting_poses, original_inputs);
                recovered_pose_targets = std::set<uint32_t>(admission->targets.begin(), admission->targets.end());
                for (const auto& source : admission->sources) imported.push_back({source.aligned, size_t(source.memory), fs::u8path(source.label)});
                sceneProgress("admission-stage-restored", active_working->sourceCount() - accepted_sources, imported.size());
            } else if (!admission_originals.empty()) {
                regional::CanonicalInitialStage proposal;
                for (const auto& item : all) if (!accepted_labels.count(source_label(item))) {
                    auto rec = readSaved(item, budget); regional::validateCanonicalImageIdentity(rec, original_inputs);
                    require_fixed_frame(item);
                    regional::canonicalizeSourceImageIdentity(rec, original_inputs, active_working->poses());
                    const auto path = regional::canonical_initial_detail::aligned(coordination / "aligned-inputs", rec);
                    imported.push_back({path, modelResidentBytes(rec), fs::u8path(source_label(item))});
                    proposal.sources.push_back({source_label(item), item.path, path, modelResidentBytes(rec), Sim3{}});
                    for (const auto& im : rec.images) if (im.second.registered && (!starting_poses.images.count(im.first) || !starting_poses.images.at(im.first).registered || recovered_pose_targets.count(im.first))) {
                        auto copy = im.second; copy.points2D.clear(); copy.point3D_ids.clear(); starting_poses.images[im.first] = std::move(copy);
                    }
                }
                regional::canonicalizeImageIdentity(starting_poses, original_inputs);
                proposal.poses = starting_poses; proposal.targets.assign(recovered_pose_targets.begin(), recovered_pose_targets.end());
                regional::saveCanonicalInitialStage(admission_directory, admission_signature, proposal);
            }
            if (active_working->sourceCount() > accepted_sources + imported.size()) throw std::runtime_error("admission raw suffix exceeds proven source proposal");
            for (size_t source = accepted_sources; source < active_working->sourceCount(); ++source) {
                const auto& prepared = imported.at(source - accepted_sources);
                if (active_working->sourceLabel(source) != source_label(prepared) || active_working->knownSource(prepared.path) != source)
                    throw std::runtime_error("admission immutable raw suffix identity differs");
            }
            appended = !imported.empty();
        }
        for (const auto& item : imported) {
            const auto label = source_label(item); const auto existing = working_sources.find(label);
            if (existing != working_sources.end()) {
                if (active_working->knownSource(item.path) != existing->second) throw std::runtime_error("prepared immutable source identity conflicts");
                continue;
            }
            const auto slot = active_working->importModel(item.path, readSaved(item, budget), label);
            working_sources.emplace(label, slot); appended = true;
            sceneProgress("immutable-source-import", active_working->sourceCount(), all.size());
        }
        working_slots.clear(); for (const auto& item : all) working_slots.push_back(working_sources.at(source_label(item)));
        if (active_working->sourceCount() != working_sources.size()) throw std::runtime_error("canonical immutable source ownership incomplete");
        regional::VerifiedExportOptions forecast_options; forecast_options.inputs = original_inputs; forecast_options.owners = &owners;
        forecast_options.cache_bytes = cache; forecast_options.reader_memory_bytes = budget / 2; forecast_options.max_error = cfg.max_error;
        forecast_options.historical_required = historical_required;
        const auto export_forecast = regional::forecastVerifiedExport(*active_working, starting_poses,
            [&](uint32_t image) { return features / fs::u8path(index.images.at(image).name + ".bin"); }, forecast_options,
            coordination / "export-capacity-forecast", sparse / "canonical-isolated-capacity-forecast");
        regional::CanonicalDiskInputs capacity; capacity.imageSlots = owners.size(); capacity.rounds = cfg.regional_iterations;
        capacity.compactRejectedSteps = true; capacity.backtrackSteps = 12; capacity.exportPeakBytes = export_forecast.required_peak_bytes;
        capacity.posesBytes = model_store_detail::serializedModelBytes(starting_poses);
        uint64_t binding_bytes = 24 + 64 + 128;
        for (size_t slot = 0; slot < active_working->sourceCount(); ++slot) {
            capacity.rawRows = storage::addBytes(capacity.rawRows, active_working->sourceObservationCount(slot));
            capacity.pointSlots = storage::addBytes(capacity.pointSlots, active_working->sourcePointCount(slot));
            binding_bytes = storage::addBytes(binding_bytes, storage::addBytes(32 + 64, active_working->sourceLabel(slot).size()));
        }
        capacity.snapshotMetadataBytes = storage::addBytes(binding_bytes, 292 + 256);
        capacity.stepReferenceBytes = storage::addBytes(binding_bytes, 476 + 256 + 128);
        const auto probe = coordination / ("capacity-hardlink-probe-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        const auto link = fs::path(probe.string() + ".link");
        { std::ofstream out(probe, std::ios::binary); out << "canonical-capacity"; out.flush(); if (!out) throw std::runtime_error("cannot write owned hardlink capacity probe"); }
        std::error_code link_error; fs::create_hard_link(probe, link, link_error);
        capacity.aliasHardlinks = !link_error && fs::equivalent(probe, link);
        if (capacity.aliasHardlinks) {
            const uint64_t existing_links = active_scene->initialized() ? fs::hard_link_count(active_scene->acceptedSnapshot().directory / "aliases.bin") : 1;
            capacity.aliasHardlinks = storage::addBytes(existing_links, storage::scaleBytes(capacity.rounds, capacity.backtrackSteps + 2)) < 1000;
        }
        if (fs::exists(link)) fs::remove(link); fs::remove(probe);
        const bool same_layout = active_scene->initialized() && active_working->sourceCount() == active_scene->sourceCount();
        capacity.aliasesPrepared = same_layout; capacity.indexPrepared = same_layout; capacity.layoutPrepared = same_layout;
        const auto disk_forecast = regional::canonicalCohortDiskForecast(capacity);
        std::ostringstream disk_proof; disk_proof << "raw=" << capacity.rawRows << "\npoint_slots=" << capacity.pointSlots
            << "\nfeatures=" << export_forecast.features << "\nrounds_this_cohort=" << capacity.rounds
            << "\nactual_hardlink_support=" << capacity.aliasHardlinks << "\nexport_peak=" << capacity.exportPeakBytes
            << "\nadditional_peak=" << disk_forecast.peakAdditional << "\nreserve=" << (uint64_t(2) << 30)
            << "\nalias_cold=" << disk_forecast.aliasCold << "\nindex_cold=" << disk_forecast.indexCold
            << "\nparameter_phase=" << disk_forecast.parameterPhase << "\nfull_capture_verified=false\n";
        resume::store(coordination / ("cohort-" + std::to_string(active_working->sourceCount()) + "-capacity.txt"), disk_proof.str());
        sceneProgress("cohort-disk-admission", disk_forecast.peakAdditional, disk_forecast.peakAdditional);
        regional::requireCanonicalCohortSpace(coordination, disk_forecast);
        auto layout = regional::prepareCanonicalLayout(coordination, *active_working, *active_scene, owners, budget / 4, sceneProgress);
        auto next_index = std::move(layout.index);
        const auto& binding = layout.binding;
        if (active_scene->initialized() && (layout.phase == regional::CanonicalLayoutPhase::AppendSources) != appended)
            throw std::runtime_error("canonical source admission and layout transition disagree");
        if (layout.phase == regional::CanonicalLayoutPhase::Cold) {
            auto directions=std::make_shared<std::vector<std::pair<uint64_t,Vec3>>>();
            auto retained=std::make_shared<std::vector<uint64_t>>();
            try {
                active_scene->beginLayoutCandidate(binding, starting_poses, [&](const auto& emit) { next_index->emitAliases(emit); },
                    regional::completeSceneSeed(*next_index, starting_poses, cfg.max_error, budget / 2, cfg.threads, sceneProgress, directions, cfg.mapper.min_tri_angle_deg,retained));
                active_scene->applyLandmarkTypes(*directions);
                active_scene->applyRetainedObservations(*retained);
                const auto validate = [&](const regional::CanonicalScene& candidate) {
                    const auto q = regional::sceneQuality(*next_index, candidate, cfg.max_error, budget / 2, cfg.threads, nullptr, sceneProgress);
                    if (q.observations != candidate.observations() || !std::isfinite(q.cost) || !std::isfinite(q.finite_cost))
                        throw std::runtime_error("initial canonical original-observation ledger is incomplete or nonfinite");
                    for (const auto& frame : original_source_frames) if (!regional::regionalSourceAccepted(frame, candidate.poses(), *gps, recorded_attitude))
                        throw std::runtime_error("initial canonical source constraint failed");
                    for (const auto& frame : historical_frames) if (!regional::regionalSourceAccepted(frame, candidate.poses(), *gps, recorded_attitude))
                        throw std::runtime_error("initial canonical historical source constraint failed");
                    regional::writeProjectionQuality(coordination / "initial-projection.bin", q);
                };
                validate(*active_scene); active_scene->commit("canonical-initial-fullsupport-baseline-v2");
            } catch (...) {if(active_scene->editing()) active_scene->abort("initial-typed-validation-failed");throw;}
        } else if (layout.phase == regional::CanonicalLayoutPhase::AppendSources) {
            if (!active_index || active_index->identity() != active_scene->acceptedSnapshot().binding.topology) {
                active_index = std::make_unique<regional::ImmutableSceneTopology>(coordination / "indices", budget / 4);
                active_index->restore(active_scene->acceptedSnapshot().binding.topology);
            }
            const auto before_prefix = regional::scenePrefixQuality(*active_index, *active_scene, cfg.max_error, budget / 2, cfg.threads, true, sceneProgress);
            const auto before_poses = active_scene->acceptedPoses();
            const auto before_visual = regional::sceneVisualEvidence(*active_index, *active_scene, cfg.max_error, budget / 2, cfg.threads, sceneProgress);
            const auto admission_evidence = coordination / ("admission-proof-" + active_scene->identity());
            fs::create_directories(admission_evidence / "before"); fs::create_directories(admission_evidence / "candidate");
            const auto before_constraints = regional::canonicalCaptureAcceptance(before_poses, original_inputs, plan, *gps, recorded_attitude,
                original_source_frames, historical_frames, historical_required, admission_evidence / "before");
            auto seed_evidence = std::make_shared<regional::CanonicalAdmissionSeedEvidence>();
            try {
                const auto admission_seed = regional::prepareCanonicalAdmissionSeed(*active_index, *next_index, *active_scene,
                    starting_poses, cfg.max_error, budget / 2, cfg.threads, sceneProgress, seed_evidence, cfg.mapper.min_tri_angle_deg);
                active_scene->beginLayoutCandidate(binding, starting_poses, [&](const auto& emit) { next_index->emitAliases(emit); },
                    admission_seed.producer);
                active_scene->applyLandmarkTypes(seed_evidence->newDirections);
                active_scene->applyRetainedObservations(seed_evidence->newRetainedObservations);
                auto admitted = [&] {
                    const auto q = regional::sceneQuality(*next_index, *active_scene, cfg.max_error, budget / 2, cfg.threads, nullptr, sceneProgress);
                    if (!std::isfinite(q.cost) || !std::isfinite(q.finite_cost) || q.observations != active_scene->observations()) throw std::runtime_error("new canonical cohort raw validation incomplete");
                    const auto prefix = regional::scenePrefixQuality(*active_index, *active_scene, cfg.max_error, budget / 2, cfg.threads, false, sceneProgress);
                    std::vector<std::string> failures;
                    if (!regional::canonicalPrefixAccepted(before_prefix, prefix)) failures.push_back("same_oldraw_visual_regression");
                    if (q.invalid_projections) failures.push_back("new_cohort_invalid_projection");
                    Reconstruction old_candidate = before_poses;
                    for (auto& image : old_candidate.images) image.second.pose = active_scene->poses().images.at(image.first).pose;
                    old_candidate.cameras = active_scene->poses().cameras;
                    if (!regional::regionalGpsRoundAccepted(regional::regionalGpsQuality(old_candidate, *gps), regional::regionalGpsQuality(before_poses, *gps)) ||
                        (regional::regionalGpsGauge(before_poses, *gps, recorded_attitude) && !regional::regionalGpsGauge(old_candidate, *gps, recorded_attitude))) failures.push_back("old_photo_absolute_GPS_or_gauge_regression");
                    for (const auto& frame : original_source_frames) if (!regional::regionalSourceAccepted(frame, active_scene->poses(), *gps, recorded_attitude)) failures.push_back("original_source_constraint_failed:" + frame.name);
                    for (const auto& frame : historical_frames) if (!regional::regionalSourceAccepted(frame, active_scene->poses(), *gps, recorded_attitude)) failures.push_back("historical_source_constraint_failed:" + frame.name);
                    const auto constraints = regional::canonicalCaptureAcceptance(active_scene->poses(), original_inputs, plan, *gps, recorded_attitude,
                        original_source_frames, historical_frames, historical_required, admission_evidence / "candidate");
                    if (before_constraints.photos.size() != constraints.photos.size()) throw std::runtime_error("canonical admission input constraint coverage differs");
                    for (size_t photo = 0; photo < constraints.photos.size(); ++photo) {
                        const auto& previous = before_constraints.photos[photo]; const auto& current = constraints.photos[photo];
                        if (previous.image != current.image) throw std::runtime_error("canonical admission input constraint identity differs");
                        const auto before = before_poses.images.find(current.image), after = active_scene->poses().images.find(current.image);
                        const bool promoted = after != active_scene->poses().images.end() && after->second.registered &&
                            (before == before_poses.images.end() || !before->second.registered || recovered_pose_targets.count(current.image));
                        if (((promoted || previous.gps_verified) && !current.gps_verified) ||
                            ((promoted || previous.scale_verified) && !current.scale_verified) ||
                            ((promoted || previous.attitude_verified) && !current.attitude_verified)) failures.push_back("promoted_or_previously_verified_photo_constraint_failed:" + std::to_string(current.image));
                    }
                    if (before_constraints.historical_verified && !constraints.historical_verified) failures.push_back("historical_source_verification_regression");
                    const auto visual = regional::sceneVisualEvidence(*next_index, *active_scene, cfg.max_error, budget / 2, cfg.threads, sceneProgress);
                    for (uint32_t image : recovered_pose_targets) if (!active_scene->poses().images.count(image) ||
                        !regional::photoQualityAccepted(visual.photos.at(image), cfg.max_error) || visual.landmarks.at(image) < 12 ||
                        (visual.components > 1 && visual.component.at(image) != visual.largest)) failures.push_back("recovered_photo_visual_support_failed:" + std::to_string(image));
                    for (const auto& image : before_poses.images) if (image.second.registered &&
                        regional::photoQualityAccepted(before_visual.photos.at(image.first), cfg.max_error) && before_visual.landmarks.at(image.first) >= 12 &&
                        (!regional::photoQualityAccepted(visual.photos.at(image.first), cfg.max_error) || visual.landmarks.at(image.first) < 12 ||
                            (before_visual.components == 1 && visual.components != 1))) failures.push_back("previously_verified_photo_visual_regression:" + std::to_string(image.first));
                    std::ostringstream ledger; ledger << std::setprecision(17) << "accepted_identity=" << active_scene->acceptedSnapshot().identity
                        << "\nnew_raw=" << q.observations << "\nnew_invalid=" << q.invalid_projections << "\nnew_finite=" << q.finite_cost
                        << "\nold_raw=" << before_prefix.observations << "\nold_before_invalid=" << before_prefix.invalid_projections
                        << "\nold_after_invalid=" << prefix.invalid_projections << "\nold_before_finite=" << before_prefix.finite_cost
                        << "\nold_after_finite=" << prefix.finite_cost << "\ncomponents=" << visual.components << "\naccepted=" << failures.empty() << "\n";
                    for (const auto& failure : failures) ledger << "failed=" << failure << "\n";
                    resume::store(admission_evidence / ("assessment-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt"), ledger.str());
                    if (!failures.empty()) {
                        slog::diag(Tag::Map, "[canonical-admission] rejected: %s; old raw %llu finite %.17g -> %.17g; new invalid %llu\n",
                            failures.front().c_str(), (unsigned long long)prefix.observations, before_prefix.finite_cost, prefix.finite_cost,
                            (unsigned long long)q.invalid_projections); return false;
                    }
                    regional::writeProjectionQuality(coordination / "admission-projection.bin", q);
                    return true;
                };
                if (!admitted()) throw std::runtime_error("held recovery admission failed complete raw/visual/GPS/scale/attitude/source validation; unchanged accepted scene and detailed candidate proof retained");
                active_scene->commit("canonical-new-cohort-fullraw-prefix-v1");
            } catch (...) {
                std::ostringstream proof; proof << "old_preserved_roots=" << seed_evidence->oldPreservedRoots << "\nold_refined_roots=" << seed_evidence->oldRefinedRoots
                    << "\nseed_rejection=" << seed_evidence->rejection << "\nfull_capture_verified=false\n";
                resume::store(admission_evidence / "seed-rejection.txt", proof.str());
                if (active_scene->editing()) active_scene->abort("cohort-append-rejected"); throw;
            }
        } else active_scene->requireBinding(binding);
        active_index = std::move(next_index); recovered_pose_targets.clear();
        const auto priors = regional::regionalGpsPriors(active_scene->poses(), *gps);
        regional::SceneCoordinator::Options options; options.gate = cfg.max_error; options.allowance = budget / 2;
        auto& bundle = options.bundle; bundle.real = realCfgFromName(cfg.mapper.ba_real); bundle.max_iters = 20;
        bundle.loss = cfg.mapper.ba_loss; bundle.loss_param = float(cfg.mapper.ba_loss_param); bundle.solver = "cg";
        bundle.device_selector = cfg.mapper.device_selector; bundle.device = cfg.mapper.device; bundle.threads = cfg.threads;
        bundle.refine_intrinsics = true; bundle.refine_principal_point = cfg.mapper.refine_principal_point;
        bundle.refine_extra_params = cfg.mapper.refine_extra_params; bundle.priors = &priors;
        bundle.host_budget_bytes = budget / 2; bundle.over_budget_throws = true;
        auto gates = [&](const Reconstruction& before, const Reconstruction& candidate) {
            std::vector<std::string> reasons;
            if (!regional::regionalGpsRoundAccepted(regional::regionalGpsQuality(candidate, *gps), regional::regionalGpsQuality(before, *gps))) reasons.push_back("absolute_GPS_cost_or_inlier_regression");
            if (regional::regionalGpsGauge(before, *gps, recorded_attitude) && !regional::regionalGpsGauge(candidate, *gps, recorded_attitude)) reasons.push_back("established_metric_gauge_lost");
            for (const auto& frame : original_source_frames) if (!regional::regionalSourceAccepted(frame, candidate, *gps, recorded_attitude)) reasons.push_back("original_source_constraint_failed:" + frame.name);
            for (const auto& frame : historical_frames) if (!regional::regionalSourceAccepted(frame, candidate, *gps, recorded_attitude)) reasons.push_back("historical_source_constraint_failed:" + frame.name);
            return reasons;
        };
        regional::SceneCoordinator controller(*active_scene, *active_index, options, gates, sceneProgress);
        uint64_t external = storage::addBytes(active_working->residentBytes(), modelResidentBytes(summary));
        for (const auto& frame : original_source_frames) external = storage::addBytes(external, modelResidentBytes(frame.views));
        for (const auto& frame : historical_frames) external = storage::addBytes(external, modelResidentBytes(frame.views));
        external = storage::addBytes(external, storage::scaleBytes(index.pairs.capacity(), sizeof(index.pairs[0])));
        regional::NativeSceneTasks tasks(*active_index, *active_scene, plan, source_anchors, bundle, budget, sceneProgress, external);
        std::vector<regional::CanonicalBundle> execution_shell;
        auto cohort_rounds = regional::canonicalCompletedRounds(coordination, *active_scene);
        scene_round = std::max(scene_round, cohort_rounds);
        for (uint64_t round = cohort_rounds; !regional::canonicalRoundsFinished(coordination, *active_scene) && round < uint64_t(cfg.regional_iterations); ++round) {
            controller.setRoundOrdinal(round + 1);
            ++scene_round; const double began = now(); const auto task_before = tasks.statistics(); const auto outcome = controller.round(execution_shell, tasks.execute());
            const auto& task_after = tasks.statistics(); const auto& timing = outcome.statistics;
            std::ostringstream ledger; ledger << std::setprecision(17) << "generation=" << active_scene->generation()
                << "\nbefore_invalid=" << outcome.before.invalid_projections << "\nbefore_finite=" << outcome.before.finite_cost
                << "\nraw=" << outcome.after.observations << "\ninvalid=" << outcome.after.invalid_projections
                << "\nfinite=" << outcome.after.finite_cost << "\ncost=" << outcome.after.cost << "\naccepted=" << outcome.accepted
                << "\nalpha=" << outcome.alpha << "\nseconds=" << now() - began
                << "\nbaseline_validation_seconds=" << timing.baselineSeconds << "\nblock_seconds=" << timing.blockSeconds
                << "\nintrinsics_seconds=" << timing.intrinsicSeconds << "\ntriangulation_seconds=" << timing.triangulationSeconds
                << "\ncandidate_validation_seconds=" << timing.candidateQualitySeconds << "\nsource_gates_seconds=" << timing.gatesSeconds
                << "\ncpu_confirmation_seconds=" << timing.cpuConfirmationSeconds
                << "\nsnapshot_seconds=" << timing.snapshotSeconds << "\nvalidation_rows_evaluated=" << timing.qualityRows
                << "\ncomplete_evaluations=" << timing.completeEvaluations << "\nbounded_rejections=" << timing.boundedRejections
                << "\nbaseline_cache_hits=" << timing.baselineCacheHits
                << "\ngpu_quality_rows=" << timing.gpuQualityRows << "\ncpu_quality_rows=" << timing.cpuQualityRows
                << "\nsource_gate_rejections=" << timing.gateRejections
                << "\nprepare_seconds=" << task_after.prepareSeconds - task_before.prepareSeconds
                << "\nreload_seconds=" << task_after.reloadSeconds - task_before.reloadSeconds
                << "\ncontext_seconds=" << task_after.contextSeconds - task_before.contextSeconds
                << "\nsolve_wave_seconds=" << task_after.waveSeconds - task_before.waveSeconds
                << "\nsolver_init_worker_seconds=" << task_after.solverInitSeconds - task_before.solverInitSeconds
                << "\nsolver_compute_worker_seconds=" << task_after.solverComputeSeconds - task_before.solverComputeSeconds
                << "\napply_seconds=" << task_after.applySeconds - task_before.applySeconds << "\n";
            for (const auto& reason : outcome.rejected) ledger << "rejected=" << reason << "\n";
            regional::storeCanonicalRoundAssessment(coordination, *active_scene, round + 1, ledger.str());
            if (outcome.accepted && progress::enabled()) progress::model(regional::canonicalPreview(*active_scene, progress::kMaxPoints), true);
            const bool small_improvement = outcome.accepted && regional::sceneFiniteImprovementSmall(outcome.before, outcome.after, cfg.regional_cost_tolerance);
            const bool round_limit = round + 1 == uint64_t(cfg.regional_iterations);
            const bool finished = !outcome.accepted || round_limit || small_improvement;
            const auto accepted_rounds = regional::canonicalCompletedRounds(coordination, *active_scene);
            regional::storeCanonicalRounds(coordination, *active_scene, accepted_rounds, finished,
                !outcome.accepted ? "rejected" : round_limit ? "round_limit" : small_improvement ? "small_improvement" : "running", round + 1);
            if (!outcome.accepted) { sceneProgress("candidate-rejected-preserved", scene_round, scene_round); break; }
            if (finished) break;
        }
        if (!regional::canonicalRoundsFinished(coordination, *active_scene) &&
            regional::canonicalCompletedRounds(coordination, *active_scene) >= uint64_t(cfg.regional_iterations))
            regional::storeCanonicalRounds(coordination, *active_scene,
                regional::canonicalCompletedRounds(coordination, *active_scene), true, "round_limit");
        summary = active_scene->poses(); if (progress::enabled()) progress::model(regional::canonicalPreview(*active_scene, progress::kMaxPoints), true); audit();
        std::vector<regional::PhotoQuality> photos;
        const auto quality = regional::sceneQuality(*active_index, *active_scene, cfg.max_error, budget / 2, cfg.threads, &photos, sceneProgress);
        regional::writeProjectionQuality(coordination / "accepted-projection.bin", quality);
        const auto acceptance = regional::canonicalCaptureAcceptance(summary, original_inputs, plan, *gps, recorded_attitude,
            original_source_frames, historical_frames, historical_required, coordination);
        recovery_targets.clear();
        for (const auto& p : acceptance.photos) if (!p.gps_verified || !p.scale_verified || !p.attitude_verified) recovery_targets.insert(p.image);
        const auto visual = regional::sceneVisualEvidence(*active_index, *active_scene, cfg.max_error, budget / 2, cfg.threads, sceneProgress);
        for (const auto& image : summary.images) if (!regional::photoQualityAccepted(visual.photos.at(image.first), cfg.max_error) || visual.landmarks.at(image.first) < 12 ||
            (visual.components > 1 && visual.component.at(image.first) != visual.largest)) recovery_targets.insert(image.first);
        sceneProgress("accepted-scene-audit", quality.observations, quality.observations);
        return true;
    };
    while (!coordinate()) {}
    std::optional<std::vector<regional::CanonicalOriginalRequirement>> original_requirements;
    auto required_originals = [&]() -> const std::vector<regional::CanonicalOriginalRequirement>& {
        if (!original_requirements) {
            const auto initial = regional::loadCanonicalInitialStage(coordination / "initialization",
                regional::regionalBaseSignature(signature), {}, budget / 4, true);
            if (!initial) throw std::runtime_error("original completeness lacks verified initial source stage");
            original_requirements = regional::canonicalOriginalRequirements(*initial, root / "source-evidence-v1",
                resume::recorded(root / "source-evidence-v1" / "inputs.txt"));
        }
        return *original_requirements;
    };
    auto normalize_original_names = [&](Reconstruction& model) { regional::canonicalizeImageIdentity(model, original_inputs); };
    auto repair_deferred = [&]() {
        std::vector<Deferred> remaining;
        for (const auto& task : deferred) {
            const auto label = source_label(task.original);
            const auto accepted = std::find_if(active_scene->binding().sources.begin(), active_scene->binding().sources.end(),
                [&](const auto& source) { return source.label == label; });
            if (accepted != active_scene->binding().sources.end()) {
                const auto& originals = required_originals();
                const auto original = std::find_if(originals.begin(), originals.end(), [&](const auto& source) { return source.label == label; });
                if (original == originals.end()) throw std::runtime_error("deferred source is absent from original stage identity");
                const auto proof = regional::verifyCanonicalOriginalCoverage(*active_working, *active_scene, {*original}, normalize_original_names, budget / 2);
                if (proof.complete) {
                    slog::diag(Tag::Map, "[canonical-repair] stale deferred task already admitted: %s, %llu original raw rows verified\n",
                        label.c_str(), (unsigned long long)proof.rows.at(0).matched);
                    continue;
                }
            }
            auto original = readSaved(task.original, budget); const auto required = regional::registeredIds(original);
            const auto stage = root / ("deferred-repair-v3-" + task.original.path.filename().string() + "-" + std::to_string(signatureHash(active_scene->identity())));
            auto saved = savedStage(stage);
            if (saved.empty()) saved = repair(task.region, required, nullptr, stage);
            if (saved.empty()) { remaining.push_back(task); continue; }
            bool retained = false;
            for (const auto& candidate : saved) {
                require_fixed_frame(candidate);
                auto recovered = readSaved(candidate, budget);
                if (!regional::coversImages({recovered}, required)) continue;
                regional::canonicalizeImageIdentity(original, original_inputs);
                regional::canonicalizeImageIdentity(recovered, original_inputs);
                const auto before_coverage = recovered;
                const uint64_t observations = regional::preserveRepairObservations(original, recovered, cfg.max_error);
                try {
                    regional::restoreAcceptedFeatureBindings(*active_index, *active_scene, recovered, budget / 8, cfg.max_error);
                } catch (const regional::CanonicalRepairBindingRejected& e) {
                    const auto token = std::chrono::steady_clock::now().time_since_epoch().count();
                    const auto rejected_stage = root / ("weak-binding-rejected-" + std::to_string(token));
                    commitStage(rejected_stage, {recovered});
                    resume::store(rejected_stage.string() + ".reason", "accepted=" + active_scene->identity() + "\nreason=" + e.what() + "\n");
                    slog::diag(Tag::Map, "[canonical-repair] weak source binding rejected; full candidate retained: %s\n", e.what());
                    continue;
                }
                HeldRepairSeedContract(before_coverage).requireUnchanged(recovered, "regional.weak-original-coverage");
                const auto rehabilitation = root / ("rehabilitated-v3-" + task.original.path.filename().string());
                auto accepted = commitStage(rehabilitation, {recovered});
                const auto hash = spirula::sha256_file(accepted.at(0).path.u8string());
                if (hash.empty()) throw std::runtime_error("rehabilitated immutable model hash missing");
                resume::store(accepted.at(0).path.string() + ".fixed-frame.txt", "held-canonical-frame-v1\n" + active_scene->identity() + "\n" + hash + "\n");
                accepted.at(0).origin = task.original.origin;
                commitSavedStage(rehabilitation, accepted, {});
                completed[task.region].insert(completed[task.region].end(), accepted.begin(), accepted.end());
                for (uint32_t id : required) {
                    const auto at = recovered.images.find(id);
                    if (at != recovered.images.end() && at->second.registered && recovery_targets.count(id)) recovered_pose_targets.insert(id);
                }
                publish(accepted, task.region); retained = true;
                resume::store(rehabilitation.string() + ".coverage", "Original photos=" + std::to_string(required.size()) +
                    "\nOriginal observations=" + std::to_string(observations) + "\nfull_capture_verified=false\n");
                slog::diag(Tag::Map, "[regional] deferred model rehabilitated: %zu photos, %llu original observations preserved; full quality pending\n",
                    required.size(), (unsigned long long)observations); break;
            }
            if (!retained) remaining.push_back(task);
        }
        deferred = std::move(remaining); audit();
    };
    for (;;) {
        const size_t before = summary.numRegistered();
        size_t before_sources = 0; for (const auto& region : completed) before_sources += region.size();
        const size_t outstanding = deferred.size(); repair_deferred();
        recover(true);
        size_t after_sources = 0; for (const auto& region : completed) after_sources += region.size();
        if (summary.numRegistered() == before && deferred.size() == outstanding && before_sources == after_sources) break;
        while (!coordinate()) {}
    }

    if (!active_working || !active_scene || !active_index) throw std::runtime_error("final canonical state missing");
    active_working->verifyImmutable();
    {
        const auto& io=active_working->counters(); std::ostringstream proof;
        proof << "scope=single_process_immutable_file_leases\nsha_bytes_actually_read=" << io.bytesHashChecked
            << "\nlease_reuses=" << io.leaseHits << "\nlease_cold_validations=" << io.leaseMisses << "\nleased_physical_files=" << io.leasedFiles
            << "\nterminal_raw_validation=complete\nformal_Reader_validation=required\n";
        resume::store(coordination / "immutable-verification-proof.txt",proof.str());
    }
    const auto final_quality = regional::sceneQuality(*active_index, *active_scene, cfg.max_error, budget / 2, cfg.threads, nullptr, sceneProgress);
    regional::atomicModel(root / "consensus.bin", active_scene->poses());
    const auto original_coverage = regional::verifyCanonicalOriginalCoverage(*active_working, *active_scene,
        required_originals(), normalize_original_names, budget / 2);
    const bool complete_originals = original_coverage.complete;
    resume::store(coordination / "original-observation-coverage.csv", original_coverage.csv());
    events::stage_begin(Stage::Finish);
    const auto export_attempt = regional::reserveCanonicalExportAttempt(coordination, sparse, *active_scene);
    const auto& pending = export_attempt.pending;
    regional::VerifiedExportOptions options; options.cache_bytes = cache; options.reader_memory_bytes = budget / 2;
    options.threads = cfg.threads;
    options.reportphase = sceneProgress;
    options.phase_bytes = [&](const char* phase, uint64_t bytes, uint64_t peak) {
        const std::string label = std::string("formal-export-") + phase; sceneProgress(label.c_str(), bytes, peak);
    };
    options.max_error = cfg.max_error; options.accepted_raw = final_quality; options.owners = &owners;
    options.inputs = original_inputs; options.historical_required = historical_required;
    options.input_path = [&](uint32_t id) { return original_inputs.at(id).path; };
    const auto report = regional::exportVerifiedScene(*active_working, *active_scene,
        [&](uint32_t image) { return fs::u8path(features) / fs::u8path(index.images.at(image).name + ".bin"); },
        export_attempt.scratch, pending, options,
        [&](const Reconstruction& readback, const regional::Quality&, const std::map<uint32_t, regional::PhotoQuality>&, const regional::ExportStats&) {
            auto proof = regional::canonicalCaptureAcceptance(readback, original_inputs, plan, *gps, captureAttitude(),
                original_source_frames, historical_frames, historical_required, coordination);
            if (!complete_originals) for (auto& photo : proof.photos) photo.reasons.push_back("original_deferred_source_observations_outstanding");
            return proof;
        });
    const auto& stats = report.stats;
    if (!report.verified || !complete_originals) {
        const bool metric_proven = regional::canonicalMetricVerified(original_inputs, report.constraints);
        if (metric_proven) {
            ModelGauge gauge; gauge.metric = true;
            gauge.oriented = std::all_of(report.constraints.begin(), report.constraints.end(), [](const auto& item) { return item.second.attitude_verified; });
            gauge.scale = "gps"; gauge.up = gauge.oriented ? "gps+attitude" : "gps"; writeGauge(pending, gauge);
        }
        std::ostringstream reasons; reasons << "full_capture_verified=false\nreader_verified=" << report.reader_verified
            << "\noriginal_sources_complete=" << complete_originals << "\ninputs=" << index.images.size() << "\n";
        for (const auto& failure : report.failures) reasons << failure.image << ':' << failure.reason << '\n';
        resume::store(coordination / "coverage.txt", reasons.str());
        for (const auto& p : report.constraints) {
            const auto id = p.first; quality_checked.at(id) = 1; gps_passed.at(id) = p.second.gps_verified;
            attitude_passed.at(id) = p.second.attitude_verified; const auto q = report.unique_quality.find(id);
            scale_passed.at(id) = p.second.scale_verified;
            const auto landmarks = report.unique_landmarks.find(id);
            visual_passed.at(id) = q != report.unique_quality.end() && regional::photoQualityAccepted(q->second, cfg.max_error) &&
                landmarks != report.unique_landmarks.end() && landmarks->second >= 12 && report.stats.photo_components == 1;
        }
        audit(); events::stage_end(Stage::Finish); events::stage_end(Stage::Map);
        AutoResult result; result.images = index.images.size(); result.registered = summary.numRegistered();
        result.points = stats.points; result.models = stats.images ? 1 : 0; result.partial = true; result.exit_code = 3;
        result.mean_reproj = stats.error; result.median_reproj = stats.median; result.sparse_dir = pending;
        result.metric = metric_proven;
        Event event; event.kind = Event::Kind::Result; event.stage = Stage::Finish; event.images = result.images;
        event.registered = result.registered; event.points = result.points; event.models = result.models;
        event.mean_reproj = result.mean_reproj; event.partial = true; event.metric = metric_proven; events::emit(event);
        slog::diag(Tag::Map, "[canonical-scene] full capture rejected: %zu input photos, %zu failed checks; accepted scene and isolated export preserved at %s\n",
            index.images.size(), report.failures.size(), pending.u8string().c_str());
        return result;
    }
    std::fill(final_verified.begin(), final_verified.end(), 1);
    std::fill(measured_positioned.begin(), measured_positioned.end(), 0);
    resume::store(root / "measured-positioned.txt", ""); audit();
    fs::copy_file(root / "photo-status.csv", pending / "photo-status.csv", fs::copy_options::overwrite_existing);
    const size_t measured_count = std::count(measured_positioned.begin(), measured_positioned.end(), uint8_t(1));
    resume::store(pending / "regional-measured-photos.txt", std::to_string(measured_count) + "\n");
    ModelGauge gauge; gauge.metric = gauge.oriented = true; gauge.scale = "gps";
    gauge.up = measured_count ? "gps+attitude" : "gps"; writeGauge(pending, gauge);
    resume::store(pending / "regional-complete.txt", signature + "canonical-scene-v1\n" + active_scene->identity());
    fs::create_directories(sparse);
    if (fs::exists(sparse / "0")) {
        fs::path previous = sparse / ("regional-previous-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::rename(sparse / "0", previous);
    }
    fs::rename(pending, sparse / "0");
    resume::store(coordination / "coverage.txt", "full_capture_verified=true\nAll input photos, original observations, per-photo constraints, connectivity and formal export passed\n");
    events::stage_end(Stage::Finish); events::stage_end(Stage::Map);
    AutoResult result; result.images = index.images.size(); result.registered = stats.images;
    result.points = stats.points; result.models = 1; result.mean_reproj = stats.error;
    result.median_reproj = stats.median;
    result.metric = true; result.partial = false;
    result.sparse_dir = sparse; result.exit_code = stats.images < 2 || stats.points == 0 ? 2 : result.partial ? 3 : 0;
    Event event; event.kind = Event::Kind::Result; event.stage = Stage::Finish; event.images = result.images;
    event.registered = result.registered; event.points = result.points; event.models = 1;
    event.mean_reproj = result.mean_reproj; event.partial = result.partial; event.metric = result.metric; events::emit(event);
    slog::out(Tag::Run, M::sum_map, {spirula::i18n::format_duration(now() - started), (long long)result.registered,
        (long long)result.images, (long long)result.points, (long long)summary.cameras.size()});
    slog::out(Tag::Run, M::sum_model_error, {slog::num(stats.error, 3), slog::num(stats.median, 3), (long long)stats.observations});
    slog::diag(Tag::Map, "[regional] canonical readonly export: %llu images, %llu points, %llu unique observations; %llu conflicting tracks rejected\n",
               (unsigned long long)stats.images, (unsigned long long)stats.points, (unsigned long long)stats.observations,
               (unsigned long long)stats.conflicts);
    slog::diag(Tag::Map, "[regional] cross-region tracks: %llu, connected region groups: %zu\n",
               (unsigned long long)stats.cross_region_points, stats.components);
    if (measured_count) slog::diag(Tag::Map,
        "[regional] %zu photos positioned by measured GPS/attitude; visual connectivity unresolved, result remains partial; see photo-status.csv\n",
        measured_count);
    slog::out(Tag::Run, M::sum_written, {(sparse / "0").string()});
    progress::gauge(true, true); if (progress::enabled()) progress::model(regional::canonicalPreview(*active_scene, progress::kMaxPoints), true);
    return result;
}
} // namespace sfm
