#include "sfm/Pipeline.h"
#include "sfm/map/RegionalRecovery.h"
#include "sfm/feature/LearnedMatcher.h"
#include "sfm/feature/Verification.h"
#include "sfm/map/RegionalStore.h"
#include "sfm/map/RepairFeatures.h"
#include "sfm/map/RegionalPriors.h"

namespace sfm::regional {

PosePriors FixedMetricPriorSource::factors(const std::vector<PosedImage>& images) {
    auto factors = source_.factors(images); factors.dropAbsoluteCentres();
    factors.gps = {}; factors.gps.ok = true; factors.gps.gate = 5; factors.gps.sigma_h = kTrustedGpsSigma;
    factors.up_w = {0, 0, 1}; if (factors.level.ok) factors.level.up_w = factors.up_w;
    for (const auto& image : images) {
        Vec3 position;
        if (!source_.position(image.image, position) || !std::isfinite(position.norm()))
            throw std::runtime_error("anchored repair GPS missing from established metric frame");
        PriorCentre centre; centre.n = 1; centre.img[0] = image.image; centre.b = position;
        centre.sigma = {kTrustedGpsSigma, kTrustedGpsSigma, kTrustedGpsSigma};
        factors.centres.push_back(centre);
    }
    return factors;
}

namespace {
using Pair = std::pair<uint32_t, uint32_t>;
std::vector<uint32_t> repairIds(const std::filesystem::path& path, const MatchesIndex& local,
                               const MatchesIndex& global) {
    std::ifstream file(path.string() + ".ids", std::ios::binary);
    uint32_t count = 0; file.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!file || count != local.images.size() || count > global.images.size())
        throw std::runtime_error("invalid repair match image table: " + path.string());
    std::vector<uint32_t> ids(count); file.read(reinterpret_cast<char*>(ids.data()), count * sizeof(uint32_t));
    std::set<uint32_t> unique;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (!file || ids[i] >= global.images.size() || !unique.insert(ids[i]).second ||
            local.images[i].name != global.images[ids[i]].name ||
            local.images[i].num_features != global.images[ids[i]].num_features)
            throw std::runtime_error("repair match image identity changed: " + path.string());
    }
    return ids;
}

void appendRepairMatches(MatchesIndex& index, const std::filesystem::path& path) {
    if (std::find(index.sources.begin(), index.sources.end(), path.string()) != index.sources.end()) return;
    MatchesIndex local;
    if (!indexMatches(path.string(), local) || !local.complete || !local.metadata.hasCameras())
        throw std::runtime_error("incomplete repair matches: " + path.string());
    auto ids = repairIds(path, local, index);
    index.sources.push_back(path.string());
    for (auto pair : local.pairs) {
        pair.image1 = ids.at(pair.image1); pair.image2 = ids.at(pair.image2);
        auto key = std::minmax(pair.image1, pair.image2);
        index.attempted_pairs.insert((uint64_t(key.first) << 32) | key.second);
        if (!pair.count) continue;
        if (!pair.config) throw std::runtime_error("unverified repair matches: " + path.string());
        pair.source = uint32_t(index.sources.size()); index.pairs.push_back(pair);
    }
}
} // namespace

void loadRepairMatches(MatchesIndex& index, const std::filesystem::path& directory) {
    if (!std::filesystem::exists(directory)) return;
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        if (entry.is_regular_file() && entry.path().extension() == ".bin") files.push_back(entry.path());
    std::sort(files.begin(), files.end());
    for (const auto& path : files) appendRepairMatches(index, path);
}

std::vector<TwoViewMatches> verifyRepairPairs(const SfmConfig& cfg, RepairFeatureCache& cache,
    IFeatureMatcher& matcher, const std::vector<Pair>& pairs, uint64_t& putative) {
    std::set<uint32_t> pinned;
    for (const auto& p : pairs) { pinned.insert(p.first); pinned.insert(p.second); }
    cache.pin(pinned);
    VerificationOptions verify; verify.two_view = cfg.twoview; verify.num_threads = cfg.threads;
    verify.report_events = false; verify.match_batch_pairs = 64;
    verify.bearings = &cache.bearings; verify.cameras = &cache.cameras;
    auto* brute = dynamic_cast<BruteForceMatcher*>(&matcher);
    return verifyPairs(cache.features, pairs,
        [&](size_t b, size_t e, std::vector<std::vector<FeatureMatch>>& out) {
            if (brute) brute->matchBatch(cache.view, pairs, b, e, out);
            else matcher.matchBatch(cache.features, pairs, b, e, out);
        }, verify, &putative);
}

size_t supplementRepairMatches(const SfmConfig& cfg, MatchesIndex& index, const std::set<uint32_t>& required,
    const std::set<uint32_t>& anchors, const std::vector<Vec3>& positions, const CameraSetup& cameras,
    const std::filesystem::path& features, const std::filesystem::path& directory, size_t neighbours, size_t budget) {
    if (positions.size() != index.images.size() || !neighbours || required.empty()) return 0;
    std::vector<Vec3> external_positions, anchor_positions;
    std::vector<uint32_t> external_ids, anchor_ids;
    for (uint32_t i = 0; i < positions.size(); ++i) if (!required.count(i)) {
        external_positions.push_back(positions[i]); external_ids.push_back(i);
        if (anchors.count(i)) { anchor_positions.push_back(positions[i]); anchor_ids.push_back(i); }
    }
    SpatialBlockIndex external(external_positions), anchored(anchor_positions);
    std::set<Pair> candidates;
    auto add = [&](uint32_t a, uint32_t b) {
        if (a != b && !required.count(b) && index.images.at(a).num_features && index.images.at(b).num_features)
            candidates.emplace(std::min(a, b), std::max(a, b));
    };
    for (uint32_t i : required) {
        cancel::check();
        for (uint32_t j : external.nearest(positions.at(i), neighbours)) add(i, external_ids.at(j));
        for (uint32_t j : anchored.nearest(positions.at(i), std::min<size_t>(32, neighbours))) add(i, anchor_ids.at(j));
        for (uint32_t gap = 1; gap <= 32; gap *= 2) {
            if (i >= gap) add(i, i - gap);
            if (size_t(i) + gap < index.images.size()) add(i, i + gap);
        }
    }
    for (const auto& p : index.pairs) if (p.count) candidates.erase({std::min(p.image1, p.image2), std::max(p.image1, p.image2)});
    for (auto at = candidates.begin(); at != candidates.end();)
        if (index.attempted_pairs.count((uint64_t(at->first) << 32) | at->second)) at = candidates.erase(at);
        else ++at;
    if (candidates.empty()) return 0;
    std::filesystem::create_directories(directory);
    slog::diag(slog::Tag::Map, "[regional] supplemental matching: %zu required photos, %zu new candidates, %zu GPS neighbours\n",
               required.size(), candidates.size(), neighbours);
    std::vector<Pair> pairs(candidates.begin(), candidates.end()); size_t previous = index.pairs.size();
    RepairFeatureCache cache(index, features, cameras, std::min<size_t>(2ull << 30, budget / 8));
    MatchOptions options = cfg.match; options.batch_pairs = 64;
    options.descriptor_budget_bytes = std::min<size_t>(512ull << 20, std::min(options.descriptor_budget_bytes, budget / 8));
    for (const auto& image : index.images) options.max_feature_count = std::max(options.max_feature_count, image.num_features);
    auto matcher = createFeatureMatcher(cfg.matcher, options, cfg.lightglue, cfg.loma_match);
    const auto started = std::chrono::steady_clock::now(); double verify_seconds = 0, persist_seconds = 0;
    for (size_t begin = 0; begin < pairs.size();) {
        cancel::check(); const size_t end = cache.nextEnd(pairs, begin);
        std::string key; std::set<uint32_t> used;
        for (size_t p = begin; p < end; ++p) {
            key += std::to_string(pairs[p].first) + "," + std::to_string(pairs[p].second) + ";";
            used.insert(pairs[p].first); used.insert(pairs[p].second);
        }
        auto path = directory / ("pairs-" + std::to_string(signatureHash(key)) + ".bin");
        std::vector<uint32_t> ids(used.begin(), used.end()); std::map<uint32_t, uint32_t> local;
        MatchesDatabase db; db.cameras = index.metadata.cameras; db.focal_prior = index.metadata.focal_prior;
        db.focal_measured = index.metadata.focal_measured;
        for (uint32_t i : ids) {
            local[i] = uint32_t(db.images.size()); db.images.push_back(index.images.at(i));
            db.camera_ids.push_back(cameras.ids.at(i));
        }
        std::vector<Pair> local_pairs;
        for (size_t p = begin; p < end; ++p) local_pairs.emplace_back(local.at(pairs[p].first), local.at(pairs[p].second));
        uint64_t putative = 0; const auto before_verify = std::chrono::steady_clock::now();
        db.pairs = verifyRepairPairs(cfg, cache, *matcher, std::vector<Pair>(pairs.begin() + begin, pairs.begin() + end), putative);
        verify_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - before_verify).count();
        const auto before_persist = std::chrono::steady_clock::now();
        for (auto& pair : db.pairs) { pair.image1 = local.at(pair.image1); pair.image2 = local.at(pair.image2); }
        std::set<Pair> verified;
        for (const auto& p : db.pairs) verified.emplace(p.image1, p.image2);
        for (const auto& p : local_pairs) if (!verified.count(p)) db.pairs.push_back({p.first, p.second, 0, {}});
        writeMatches(path.string() + ".part", db);
        std::ofstream table(path.string() + ".ids.part", std::ios::binary);
        uint32_t count = uint32_t(ids.size()); table.write(reinterpret_cast<const char*>(&count), sizeof(count));
        table.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(uint32_t)); table.flush();
        if (!table) throw std::runtime_error("cannot write repair match image table"); table.close();
        replaceFile(path.string() + ".ids.part", path.string() + ".ids");
        MatchesIndex check;
        if (!indexMatches(path.string() + ".part", check) || !check.complete || !check.metadata.hasCameras())
            throw std::runtime_error("cannot persist supplemental matches");
        replaceFile(path.string() + ".part", path); appendRepairMatches(index, path);
        persist_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - before_persist).count();
        slog::diag(slog::Tag::Map, "[regional] supplemental matching %zu/%zu candidates, %llu descriptor matches, %zu verified connections retained\n",
                   end, pairs.size(), (unsigned long long)putative, index.pairs.size() - previous);
        begin = end;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    slog::diag(slog::Tag::Map, "[regional] repair throughput %.1f pairs/s; verify/read %.2fs, persist %.2fs; feature loads %zu, cache peak %.1f MiB\n",
        pairs.size() / std::max(seconds, .001), verify_seconds, persist_seconds, cache.loads(), cache.peakBytes() / double(1ull << 20));
    return index.pairs.size() - previous;
}

std::vector<Reconstruction> reconstructRepair(const SfmConfig& cfg, const MatchesIndex& index,
    const Region& region, const std::filesystem::path& matches, const std::filesystem::path& features,
    const std::string& images, const CameraSetup& cameras, PriorSource* gps,
    const Reconstruction* seed, size_t budget, bool hold_seed) {
    if (hold_seed && (!seed || seed->points3D.empty()))
        throw std::runtime_error("held repair requires a nonempty immutable visual seed");
    RegionInput input = loadRegion(index, region, matches, features, budget,hold_seed ? seed : nullptr);
    struct Scope {
        Scope(const std::vector<uint32_t>& ids) { events::set_map_image_ids(ids); }
        ~Scope() { events::set_map_image_ids({}); }
    } scope(input.global);
    SfmConfig local = cfg; local.mapper.initial_cameras = cameras.cameras;
    if (seed) for (const auto& camera : seed->cameras) local.mapper.initial_cameras[camera.first] = camera.second;
    local.mapper.report_progress = false;
    local.mapper.known_focal_cameras = cameras.focal_known; local.mapper.given_focal_cameras = cameras.focal_given;
    local.mapper.measured_focal_cameras = cameras.focal_measured;
    std::unique_ptr<RemappedPriorSource> priors;
    if (gps) priors = std::make_unique<RemappedPriorSource>(*gps, input.global);
    std::unique_ptr<FixedMetricPriorSource> metric_priors;
    if (hold_seed && priors) metric_priors = std::make_unique<FixedMetricPriorSource>(*priors);
    RigTable rigs = buildRigs(input.db, local, !cfg.quiet); SequenceTable seqs = buildSequences(input.db, local, !cfg.quiet);
    Mapper mapper(input.db, input.features, local.mapper, input.db.camera_ids, &rigs, &seqs,
                  cfg.sensor_map ? (metric_priors ? static_cast<PriorSource*>(metric_priors.get()) : priors.get()) : nullptr);
    std::vector<Reconstruction> models;
    std::optional<Reconstruction> held_compact_seed;
    if (seed && !seed->points3D.empty()) {
        Reconstruction compact = compactSeed(*seed, input, hold_seed);
        if (hold_seed && compact.points3D.empty()) {
            slog::diag(slog::Tag::Map, "[regional] anchored context has no retained visual seed tracks; expand support without unanchored fallback\n");
            return {};
        }
        for (const auto& camera : cameras.cameras) compact.cameras.emplace(camera.first, camera.second);
        if (!compact.points3D.empty()) {
            auto grown = mapper.continueFrom(compact, nullptr, {}, uint32_t(input.global.size()), hold_seed);
            if (hold_seed) HeldRepairSeedContract(compact).requireUnchanged(grown,"regional.compact-exit");
            if (hold_seed) held_compact_seed.emplace(std::move(compact));
            models.push_back(std::move(grown));
        }
    }
    if (models.empty()) {
        AssembleStats stats; models = runMapper(mapper, input.db, input.features, local, stats);
        double finish = 0; models = finishModels(mapper, std::move(models), local, !cfg.quiet, finish);
    }
    // A held seed carries the canonical feature-name frame through the full
    // compact/global identity check. Physical export names are an explicit
    // mapping at the caller; resolving them here would mutate held metadata.
    if (!images.empty() && !hold_seed) resolveImageNames(models, images);
    for (auto& model : models) {
        restoreFeatureIds(model, input, index,hold_seed ? seed : nullptr);
        if (held_compact_seed) HeldRepairSeedContract(*held_compact_seed).requireRemappedUnchanged(
            model,input.global,input.original,[&](uint32_t image) { return index.images.at(image).num_features; },"regional.global-exit");
    }
    if (!images.empty()) recolorPoints(models, cfg);
    return models;
}

} // namespace sfm::regional
