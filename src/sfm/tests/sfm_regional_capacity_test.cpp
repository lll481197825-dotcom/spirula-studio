#include "sfm/map/RegionalTopology.h"
#include "sfm/map/RegionalPlan.h"
#include "sfm/map/RegionalCoordinator.h"
#include "sfm/tests/TestMain.h"
#include "core/HostMemory.h"
#include "sfm/map/MemoryPolicy.h"
#include <iomanip>
#include <iostream>

using namespace sfm;
using Store = regional::RegionalWorkingStore;
using Topology = regional::RegionalTopology;
namespace fs = std::filesystem;

static void check(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
class AuditHash {
public:
    void bytes(const void* data, size_t count) {
        const auto* source = static_cast<const uint8_t*>(data);
        while (count) {
            const auto copied = std::min(count, chunk_.size() - used_); std::memcpy(chunk_.data() + used_, source, copied);
            source += copied; used_ += copied; count -= copied; if (used_ == chunk_.size()) flush();
        }
    }
    template<class T> void scalar(const T& value) {
        if (used_ + sizeof(value) > chunk_.size()) flush();
        std::memcpy(chunk_.data() + used_, &value, sizeof(value)); used_ += sizeof(value);
    }
    std::string finish() { flush(); return hash_.hex(); }
private:
    void flush() { if (used_) hash_.update(chunk_.data(), used_); used_ = 0; }
    spirula::Sha256 hash_; std::array<uint8_t, 64u << 10> chunk_{}; size_t used_ = 0;
};
struct ModelHashSink {
    AuditHash hash;
    template<class T> void scalar(const T& value) { hash.scalar(value); }
    template<class T> void vector(const std::vector<T>& values) { scalar(uint64_t(values.size())); hash.bytes(values.data(), values.size() * sizeof(T)); }
    void string(const std::string& value) { scalar(uint64_t(value.size())); hash.bytes(value.data(), value.size()); }
};
template<class T> static void append(AuditHash& hash, const T& value) { hash.scalar(value); }
struct Memory {
    uint64_t resident = 0, privateBytes = 0, peakResident = 0, peakCommit = 0;
};
static Memory memory() {
    Memory value; value.resident = spirula::processRamBytes();
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters))) {
        value.privateBytes = counters.PrivateUsage; value.peakResident = counters.PeakWorkingSetSize; value.peakCommit = counters.PeakPagefileUsage;
    }
#endif
    return value;
}
static void reportMemory(std::ostream& out, const char* phase, size_t budget) {
    const auto value = memory(); out << "memory_" << phase << "=" << value.resident << ',' << value.privateBytes << ',' << value.peakResident << ',' << value.peakCommit << '\n'; out.flush();
    std::cout << "[capacity-audit] " << phase << " resident/private/peak-resident/peak-commit bytes " << value.resident << '/' << value.privateBytes << '/'
        << value.peakResident << '/' << value.peakCommit << '\n' << std::flush;
    check(value.peakCommit <= storage::addBytes(budget, 128u << 20), "complete capacity process peak exceeds budget plus 128 MiB instrumentation margin");
}
static std::string fileHash(const fs::path& path) {
    std::ifstream input(path, std::ios::binary); check(bool(input), "cannot open capacity audit artifact");
    spirula::Sha256 hash; std::vector<uint8_t> chunk(1u << 20);
    while (input) { input.read(reinterpret_cast<char*>(chunk.data()), chunk.size()); hash.update(chunk.data(), size_t(input.gcount())); }
    check(input.eof(), "capacity artifact read failed"); return hash.hex();
}
struct Snapshot {
    uint64_t points = 0, observations = 0, common = 0, groups = 0;
    std::string lineage, raw, tracks;
    bool operator==(const Snapshot& b) const {
        return points == b.points && observations == b.observations && common == b.common && groups == b.groups &&
            lineage == b.lineage && raw == b.raw && tracks == b.tracks;
    }
};
static Snapshot snapshot(Store& store, const Topology& topology) {
    Snapshot result; result.groups = topology.groups(); AuditHash lineage, raw, tracks;
    for (size_t source = 0; source < store.sourceCount(); ++source) {
        uint64_t ordinal = 0, observation = 0;
        store.visitSourcePoints(source, [&](Store::PointKey key, Vec3 xyz, const std::vector<Store::RawObservation>& rows) {
            check(key.source == source && key.ordinal == ordinal++, "source point ordinal changed during capacity audit");
            const auto root = topology.canonical(key); const uint8_t shared = topology.shared(key); const auto value = topology.xyz(key, xyz);
            append(lineage, key.source); append(lineage, key.ordinal); append(lineage, key.localPoint); append(lineage, root); append(lineage, shared);
            append(lineage, value.x); append(lineage, value.y); append(lineage, value.z); ++result.points;
            for (const auto& row : rows) {
                check(row.point == key && row.ordinal == observation++, "original observation lineage or order changed");
                append(raw, row.point.source); append(raw, row.point.ordinal); append(raw, row.point.localPoint);
                append(raw, row.ordinal); append(raw, row.image); append(raw, row.feature); append(raw, row.xy.x); append(raw, row.xy.y); ++result.observations;
            }
        });
        check(ordinal == store.sourcePointCount(source) && observation == store.sourceObservationCount(source), "full source cardinality differs from durable manifest");
        std::cout << "[capacity-audit] full original source " << source + 1 << '/' << store.sourceCount()
            << ", points " << result.points << ", observations " << result.observations << '\n' << std::flush;
    }
    topology.visitCommonObservations([&](regional::Observation row) {
        append(tracks, row.key); append(tracks, row.point); append(tracks, row.xy.x); append(tracks, row.xy.y); ++result.common;
    });
    result.lineage = lineage.finish(); result.raw = raw.finish(); result.tracks = tracks.finish(); return result;
}
static uint64_t countPoints(Store& store) {
    uint64_t count = 0; for (size_t source = 0; source < store.sourceCount(); ++source) count = storage::addBytes(count, store.sourcePointCount(source)); return count;
}
static uint64_t countRaw(Store& store) {
    uint64_t count = 0; for (size_t source = 0; source < store.sourceCount(); ++source) count = storage::addBytes(count, store.sourceObservationCount(source)); return count;
}
static Snapshot originalReference(Store& store, size_t budget) {
    Snapshot result; AuditHash raw;
    for (size_t source = 0; source < store.sourceCount(); ++source) {
        const auto model = model_store_detail::readModel(store.sourcePath(source));
        check(modelResidentBytes(model) <= budget / 2, "one original native source exceeds reference budget");
        uint64_t ordinal = 0, observation = 0;
        for (const auto& point : model.points3D) {
            for (const auto& row : point.second.track) {
                const auto xy = model.images.at(row.image_id).points2D.at(row.point2D_idx);
                append(raw, uint64_t(source)); append(raw, ordinal); append(raw, point.first); append(raw, observation++);
                append(raw, row.image_id); append(raw, row.point2D_idx); append(raw, xy.x); append(raw, xy.y); ++result.observations;
            }
            ++ordinal; ++result.points;
        }
        check(ordinal == store.sourcePointCount(source) && observation == store.sourceObservationCount(source), "original native Reader source cardinality differs from imported flat store");
        std::cout << "[capacity-audit] original native Reader " << source + 1 << '/' << store.sourceCount()
            << ", points " << result.points << ", observations " << result.observations << '\n' << std::flush;
    }
    result.raw = raw.finish(); return result;
}
static std::string parameterHash(const fs::path& working, uint64_t generation) {
    const auto directory = working / ("parameters-" + std::to_string(generation)); std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(directory)) if (entry.is_regular_file()) paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end()); AuditHash hash;
    for (const auto& path : paths) {
        const auto name = path.filename().u8string(), digest = fileHash(path); append(hash, uint64_t(name.size()));
        for (auto byte : name) append(hash, byte); for (auto byte : digest) append(hash, byte);
    }
    return hash.finish();
}
class CapturedPositions : public PriorSource {
public:
    CapturedPositions(const fs::path& path, size_t images) {
        std::ifstream input(path); check(bool(input), "cannot read captured sequential ENU positions");
        Vec3 value; positions_.reserve(images);
        while (input >> value.x >> value.y >> value.z) {
            check(std::isfinite(value.norm()) && positions_.size() < images, "captured ENU position is nonfinite or exceeds image catalogue"); positions_.push_back(value);
        }
        check(input.eof() && positions_.size() == images, "captured ENU position count differs from complete image catalogue");
    }
    bool has(uint32_t id) const override { return id < positions_.size(); }
    bool position(uint32_t id, Vec3& value) const override { if (!has(id)) return false; value = positions_[id]; return true; }
    bool relativeRotation(uint32_t, uint32_t, Mat3&, double&) const override { return false; }
    std::vector<uint32_t> neighbours(uint32_t) const override { return {}; }
    PosePriors factors(const std::vector<PosedImage>&) override { return {}; }
private:
    std::vector<Vec3> positions_;
};
static void retainDirectory(const fs::path& source, const fs::path& destination) {
    fs::create_directory(destination);
    for (const auto& entry : fs::directory_iterator(source)) {
        check(entry.is_regular_file() && !entry.is_symlink(), "immutable clone source contains an unexpected directory or symlink");
        const auto target = destination / entry.path().filename(); fs::create_hard_link(entry.path(), target);
        check(fs::equivalent(entry.path(), target), "immutable source clone unexpectedly copied complete data");
    }
}
static uint64_t win32ReadBytes() {
#if defined(_WIN32)
    IO_COUNTERS counters{}; check(bool(GetProcessIoCounters(GetCurrentProcess(), &counters)), "cannot read native process IO counters"); return counters.ReadTransferCount;
#else
    return 0;
#endif
}
static int gatherProbe(const fs::path& working, const fs::path& planRoot, const fs::path& output, size_t budget,
    uint64_t generation, size_t images, size_t blocks, size_t firstBlock = 0) {
    check(!fs::exists(output), "gather IO probe output already exists"); fs::create_directories(output);
    std::cout << "[gather-io] native full working-store restore begins\n" << std::flush; Store store(working, budget / 2);
    check(store.sourceCount() == 134 && store.generation() == generation, "gather IO probe input cohort or generation differs");
    auto plan = regional::readRegionalPlan(planRoot, images); check(blocks > 0 && firstBlock < plan.size() && blocks <= plan.size() - firstBlock, "invalid gather block selection");
    std::stable_sort(plan.begin(), plan.end(), [](const regional::Region& a, const regional::Region& b) { return a.core.size() > b.core.size(); });
    std::ofstream report(output / "gather-io.txt"); report << std::setprecision(17) << "source_models=" << store.sourceCount() << "\ngeneration=" << generation << "\nfull_capture_verified=false\n";
    for (size_t block = firstBlock; block < firstBlock + blocks; ++block) {
        std::set<uint32_t> targets; for (auto image : plan[block].core) if (store.poses().images.count(image) && store.poses().images.at(image).registered) targets.insert(image);
        store.resetCounters(); const auto cacheRead = win32ReadBytes(); const auto cacheStarted = std::chrono::steady_clock::now();
        std::cout << "[photo-major] block " << block + 1 << " independent cache preparation/verification begins\n" << std::flush;
        for (size_t source = 0; source < store.sourceCount(); ++source) {
            bool needed = false; for (auto image : targets) if (store.sourcePhotoObservationCount(source, image)) { needed = true; break; }
            if (needed) store.preparePhotoObservationCache(source);
        }
        const auto cacheSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - cacheStarted).count(); const auto cacheIo = store.counters();
        report << "cache_block=" << block + 1 << "\ncache_prepare_seconds=" << cacheSeconds << "\ncache_win32_read_bytes=" << win32ReadBytes() - cacheRead
            << "\ncache_build_sources=" << cacheIo.photo_cache_builds << "\ncache_original_input_bytes=" << cacheIo.photo_cache_input_bytes
            << "\ncache_written_bytes=" << cacheIo.photo_cache_written_bytes << "\ncache_build_seconds=" << cacheIo.photo_cache_build_seconds << '\n'; report.flush();
        reportMemory(report, "photo_cache", budget);
        for (size_t source = store.sourceCount() - 8; source < store.sourceCount(); ++source) if (store.sourcePointCount(source)) store.key(source, 0);
        store.resetCounters(); const auto beforeRead = win32ReadBytes(); const auto started = std::chrono::steady_clock::now();
        std::cout << "[gather-io] block " << block + 1 << ", " << targets.size() << " target photos; eight source-point Readers primed\n" << std::flush;
        auto gathered = store.gather(targets, {}, false); const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const auto readBytes = win32ReadBytes() - beforeRead; const auto counters = store.counters(); ModelHashSink sink; model_store_detail::serializeModel(gathered.model, sink);
        AuditHash identity; for (const auto& point : gathered.pointKeys) { append(identity, point.first); append(identity, point.second.source); append(identity, point.second.ordinal); append(identity, point.second.localPoint); }
        for (auto point : gathered.boundaryPoints) append(identity, point); uint64_t raw = 0, features = 0;
        for (const auto& point : gathered.model.points3D) raw += point.second.track.size(); for (const auto& image : gathered.model.images) features += image.second.points2D.size();
        report << "block=" << block + 1 << "\nphotos=" << gathered.model.images.size() << "\npoints=" << gathered.model.points3D.size() << "\nraw=" << raw << "\nfeatures=" << features
            << "\nelapsed_seconds=" << elapsed << "\nwin32_read_bytes=" << readBytes << "\nlogical_raw_bytes=" << counters.raw_bytes << "\nlogical_index_bytes=" << counters.index_bytes
            << "\nlogical_parameter_bytes=" << counters.parameter_bytes << "\nmodel_native_sha256=" << sink.hash.finish() << "\nlineage_boundary_sha256=" << identity.finish() << '\n'; report.flush();
        report << "reader_hits=" << counters.reader_hits << "\nreader_opens=" << counters.reader_opens << "\nreader_evictions=" << counters.reader_evictions << '\n'; report.flush();
        reportMemory(report, "gather", budget); std::cout << "[gather-io] completed block " << block + 1 << ", " << elapsed << "s, Win32 read " << readBytes << " bytes\n" << std::flush;
    }
    check(bool(report), "gather IO evidence write failed"); return 0;
}
static int largestPhotoCacheProbe(const fs::path& working, const fs::path& output, size_t budget, uint64_t generation) {
    check(!fs::exists(output), "largest-source photo cache evidence output already exists"); fs::create_directories(output);
    std::cout << "[photo-major-largest] strict full 134-source native restore begins\n" << std::flush;
    Store store(working, budget / 2); check(store.sourceCount() == 134 && store.generation() == generation, "largest photo cache input identity differs");
    size_t selected = 0; for (size_t source = 1; source < store.sourceCount(); ++source)
        if (store.sourceObservationCount(source) > store.sourceObservationCount(selected)) selected = source;
    const auto sourceRoot = working / ("source-" + std::to_string(selected)); std::map<std::string, std::string> hashes;
    for (const char* file : {"views.bin", "features.bin", "points.bin", "observations.bin", "photo-index.bin", "initial.xyz"}) hashes.emplace(file, fileHash(sourceRoot / file));
    const auto manifestHash = fileHash(working / "manifest.bin"), parametersHash = parameterHash(working, generation);
    std::ofstream report(output / "largest-source-cache.txt"); report << std::setprecision(17) << "full_capture_verified=false\nsources=134\nselected_source=" << selected
        << "\npoints=" << store.sourcePointCount(selected) << "\nraw=" << store.sourceObservationCount(selected) << "\nrecord_bytes=" << sizeof(Store::RawObservation)
        << "\nbudget_bytes=" << budget << '\n'; report.flush();
    std::cout << "[photo-major-largest] selected source " << selected << ", " << store.sourceObservationCount(selected) << " complete original observations; cold cache begins\n" << std::flush;
    store.resetCounters(); const auto reads = win32ReadBytes(); const auto started = std::chrono::steady_clock::now();
    const auto cache = store.preparePhotoObservationCache(selected); const auto io = store.counters();
    report << "cold_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << "\nwin32_read_bytes=" << win32ReadBytes() - reads
        << "\ncache_build_sources=" << io.photo_cache_builds << "\ncache_original_input_bytes=" << io.photo_cache_input_bytes << "\ncache_written_bytes=" << io.photo_cache_written_bytes
        << "\ncache_path=" << cache.u8string() << "\ncache_sha256=" << fileHash(cache) << '\n'; report.flush(); reportMemory(report, "largest_photo_cache", budget);
    for (const auto& file : hashes) check(fileHash(sourceRoot / file.first) == file.second, "derived cache changed original source files");
    check(fileHash(working / "manifest.bin") == manifestHash && parameterHash(working, generation) == parametersHash, "derived cache changed production parameters/generation");
    report << "all_six_immutable_hashes_unchanged=true\nmanifest_and_parameters_unchanged=true\npassed=true\n"; report.flush();
    check(bool(report), "largest photo cache proof write failed"); std::cout << "[photo-major-largest] complete largest-source cache capacity proof passed\n"; return 0;
}
static std::pair<std::string, uint64_t> originalRawPrefix(Store& store, size_t sources) {
    AuditHash hash; uint64_t observations = 0;
    store.visitObservations([&](const Store::RawObservation& row) {
        if (row.point.source >= sources) return;
        append(hash, row.point.source); append(hash, row.point.ordinal); append(hash, row.point.localPoint);
        append(hash, row.ordinal); append(hash, row.image); append(hash, row.feature); append(hash, row.xy.x); append(hash, row.xy.y); ++observations;
    });
    return {hash.finish(), observations};
}
static int appendProbe(const fs::path& baseProbe, const fs::path& planRoot, const fs::path& output, size_t budget,
    uint64_t generation, size_t images, size_t selectedPhotos) {
    check(!fs::exists(output), "real append candidate directory already exists"); const auto original = fs::canonical(baseProbe / "working");
    const auto manifestHash = fileHash(original / "manifest.bin"), parametersHash = parameterHash(original, generation);
    fs::create_directories(output); const auto clone = output / "working"; fs::create_directory(clone);
    fs::copy_file(original / "manifest.bin", clone / "manifest.bin");
    for (size_t source = 0; source < 134; ++source) retainDirectory(original / ("source-" + std::to_string(source)), clone / ("source-" + std::to_string(source)));
    retainDirectory(original / ("parameters-" + std::to_string(generation)), clone / ("parameters-" + std::to_string(generation)));
    std::cout << "[capacity-append] full-resident clone native restore begins\n" << std::flush;
    Store store(clone, budget / 2); check(store.sourceCount() == 134 && countPoints(store) == 89671309 && store.generation() == generation,
        "real append baseline does not contain complete accepted source cohort");
    auto plan = regional::readRegionalPlan(planRoot, images); std::vector<uint32_t> owners(images, UINT32_MAX);
    for (uint32_t region = 0; region < plan.size(); ++region) for (auto image : plan[region].core) owners.at(image) = region;
    fs::create_directory(output / "topology"); fs::copy_file(baseProbe / "topology/head.bin", output / "topology/head.bin");
    std::ofstream report(output / "append-probe.txt"); report << "full_capture_verified=false\noriginal_sources=134\noriginal_points=89671309\nbudget_bytes=" << budget << '\n'; report.flush();
    Snapshot appended; uint64_t appendedPoints = 0, appendedRaw = 0; const auto rawBefore = originalRawPrefix(store, 134);
    {
        Topology topology(output / "topology", budget); topology.prepare(store, owners);
        topology.restoreAcceptedParameters(baseProbe / "parameters" / ("common-" + std::to_string(generation) + ".bin"));
        reportMemory(report, "append_baseline", budget);
        std::set<uint32_t> targets;
        for (auto image : plan.front().core) if (store.poses().images.count(image) && store.poses().images.at(image).registered) {
            bool supported = false; for (size_t source = 0; source < store.sourceCount(); ++source) supported |= store.sourcePhotoObservationCount(source, image) != 0;
            if (supported) targets.insert(image); if (targets.size() == selectedPhotos) break;
        }
        check(!targets.empty(), "real append context has no visually supported captured photo");
        auto context = store.gather(targets, {}, false); std::set<uint64_t> fixed; topology.inject(context, fixed);
        check(!context.model.points3D.empty(), "real append context has no actual original 3D tracks");
        appendedPoints = context.model.points3D.size(); for (const auto& point : context.model.points3D) appendedRaw += point.second.track.size();
        const auto source = output / "real-observation-context.bin"; model_store_detail::writeModel(context.model, source);
        const auto slot = store.importModel(source, context.model, "actual-captured-observation-context");
        check(slot == 134 && store.sourceCount() == 135 && countPoints(store) == 89671309 + appendedPoints && countRaw(store) == rawBefore.second + appendedRaw,
            "real append source range or raw delta lost complete original coverage");
        context = {}; store.resetCounters(); const auto started = std::chrono::steady_clock::now(); topology.prepare(store, owners);
        check(topology.counters().new_sources == 1, "real append unexpectedly rebuilt original source cohort");
        const auto restored = topology.restoreAcceptedParameters(baseProbe / "parameters" / ("common-" + std::to_string(generation) + ".bin"));
        check(restored == Topology::ParameterRestore::InitializedAppend, "real source append lost accepted parameter ancestor contract");
        reportMemory(report, "real_append", budget);
        const auto& counts = topology.counters(); report << "new_context_photos=" << targets.size() << "\nnew_context_points=" << appendedPoints << "\nnew_context_raw=" << appendedRaw
            << "\nappend_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
            << "\naffected_old_photo_rows=" << counts.old_photo_rows << "\naffected_old_private_points=" << counts.old_private_points
            << "\nunchanged_groups_reused=" << counts.reused_groups << "\ndelta_rows=" << counts.delta_rows << '\n'; report.flush();
        appended = snapshot(store, topology); topology.saveParameters(output / "append-common.bin");
    }
    const auto rawAfter = originalRawPrefix(store, 134); check(rawBefore == rawAfter, "real incremental append modified or dropped an original observation");
    store.resetCounters();
    {
        Topology warm(output / "topology", budget); warm.prepare(store, owners);
        check(store.counters().raw_bytes == 0 && warm.counters().new_sources == 0, "warm appended topology rescanned original raw sources");
        warm.restoreAcceptedParameters(output / "append-common.bin"); check(appended == snapshot(store, warm), "warm actual appended aliases, parameters or full tracks differ");
    }
    reportMemory(report, "append_warm", budget);
    check(fileHash(original / "manifest.bin") == manifestHash && parameterHash(original, generation) == parametersHash,
        "isolated append changed prior accepted baseline manifest or parameters");
    report << "original_raw_sha256=" << rawBefore.first << "\noriginal_raw_count=" << rawBefore.second
        << "\noriginal_raw_unchanged=true\nwarm_equals_append=true\nprior_accepted_manifest_and_parameters_unchanged=true\npassed=true\n";
    report.flush(); check(bool(report), "real append evidence write failed"); std::cout << "[capacity-append] complete full-resident real source append and warm audit passed\n"; return 0;
}
static int blockProbe(const fs::path& working, const fs::path& planRoot, const fs::path& consensusPath,
    const fs::path& capacityRoot, const fs::path& gpsPath, const fs::path& output, size_t budget, uint64_t generation,
    size_t images, size_t blocks, int threads, const fs::path& photoCacheRoot = {}) {
    check(!fs::exists(output), "block capacity candidate output already exists");
    const auto original = fs::canonical(working), target = fs::absolute(output).lexically_normal();
    check(target.u8string().find(original.u8string()) != 0, "block candidate cannot be inside original working store");
    const auto manifestHash = fileHash(original / "manifest.bin"), parametersHash = parameterHash(original, generation);
    fs::create_directories(output); const auto clone = output / "working"; fs::create_directory(clone);
    fs::copy_file(original / "manifest.bin", clone / "manifest.bin");
    for (size_t source = 0; source < 134; ++source) retainDirectory(original / ("source-" + std::to_string(source)), clone / ("source-" + std::to_string(source)));
    retainDirectory(original / ("parameters-" + std::to_string(generation)), clone / ("parameters-" + std::to_string(generation)));
    if (!photoCacheRoot.empty()) {
        const auto input = photoCacheRoot / "photo-observations-v1"; check(fs::is_directory(input), "requested frozen photo cache root is missing");
        fs::create_directory(clone / "photo-observations-v1");
        for (const auto& entry : fs::directory_iterator(input)) {
            check(entry.is_directory() && !entry.is_symlink() && fs::exists(entry.path() / "ready.bin"), "incomplete derived photo cache evidence cannot be reused");
            retainDirectory(entry.path(), clone / "photo-observations-v1" / entry.path().filename());
        }
    }
    std::cout << "[capacity-block] immutable source hardlinks retained; strict native clone restore begins\n" << std::flush;
    Store store(clone, budget / 2); check(store.sourceCount() == 134 && countPoints(store) == 89671309 &&
        store.generation() == generation && store.generationPhase().empty(), "block clone does not match complete unseeded 134-source capture");
    const auto consensus = model_store_detail::readModel(consensusPath, false); CapturedPositions gps(gpsPath, images);
    auto plan = regional::readRegionalPlan(planRoot, images); std::vector<uint32_t> owners(images, UINT32_MAX);
    for (uint32_t region = 0; region < plan.size(); ++region) for (auto image : plan[region].core) owners.at(image) = region;
    check(blocks > 0 && blocks <= plan.size() && threads > 0, "invalid block probe selection");
    std::stable_sort(plan.begin(), plan.end(), [](const regional::Region& a, const regional::Region& b) { return a.core.size() > b.core.size(); }); plan.resize(blocks);
    fs::create_directory(output / "topology"); fs::copy_file(capacityRoot / "topology/head.bin", output / "topology/head.bin");
    Topology topology(output / "topology", budget); topology.prepare(store, owners); topology.restoreAcceptedParameters(capacityRoot / "common-native.bin");
    fs::create_directory(output / "parameters"); const auto beforeSeed = output / "parameters/before-seed-common.bin";
    topology.saveParameters(beforeSeed); store.beginGeneration();
    try {
        for (const auto& image : consensus.images) if (store.poses().images.count(image.first)) store.setPose(image.first, image.second.pose);
        for (const auto& camera : consensus.cameras) if (store.poses().cameras.count(camera.first)) store.setCamera(camera.second);
        topology.triangulate(store.poses(), 3, threads); topology.synchronize(store);
        const uint64_t seedBase = storage::addBytes(topology.memoryBytes(), storage::addBytes(store.residentBytes(),
            storage::addBytes(modelResidentBytes(consensus), std::min<size_t>(128u << 20, budget / 16))));
        check(seedBase < budget, "full private seed has no remaining host allowance");
        std::cout << "[capacity-block] initial full private source calibration begins; base " << seedBase << " remaining " << budget - seedBase << " bytes\n" << std::flush;
        const auto privateStarted = std::chrono::steady_clock::now();
        const auto privateSeed = regional::refinePrivateGeometry(store, topology, store.poses(), 3, threads, budget - seedBase,
            output / "parameters/private-support", nullptr, [&](const regional::PrivateGeometryProgress& progress) {
                std::ostringstream state; state << "{\"phase\":\"initial_private_triangulation\",\"private_source\":" << progress.source
                    << ",\"private_sources\":" << progress.sourceTotal << ",\"private_points_done\":" << progress.pointsDone
                    << ",\"private_points_total\":" << progress.pointsTotal << ",\"private_groups_done\":" << progress.groupsDone
                    << ",\"private_groups_total\":" << progress.groupsTotal << ",\"full_capture_verified\":false}\n";
                resume::store(output / "parameters/working-progress.json", state.str());
            });
        std::ofstream seedReport(output / "private-seed.txt"); seedReport << std::setprecision(17)
            << "full_capture_verified=false\ninitial_private_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - privateStarted).count()
            << "\nobservations=" << privateSeed.before.observations << "\ninvalid_before=" << privateSeed.before.invalid << "\ninvalid_after=" << privateSeed.after.invalid
            << "\nfinite_before=" << privateSeed.before.finite << "\nfinite_after=" << privateSeed.after.finite
            << "\ntotal_before=" << privateSeed.before.total << "\ntotal_after=" << privateSeed.after.total
            << "\nprivate_ledger=" << privateSeed.ledger.u8string() << "\nprivate_original_ledger_pass=" << regional::privateGeometryAccepted(privateSeed) << '\n';
        seedReport.flush(); check(bool(seedReport), "private seed evidence write failed"); reportMemory(seedReport, "initial_private_seed", budget);
        check(regional::privateGeometryAccepted(privateSeed), "initial full private raw ledger regressed; candidate retained");
        const auto seedGeneration = store.generation() + 1;
        topology.saveParameters(output / "parameters" / ("common-" + std::to_string(seedGeneration) + ".bin"));
        store.commitGeneration("initial-seed-private-v2:" + std::to_string(seedGeneration));
    } catch (...) {
        topology.saveParameters(output / "parameters/initial-seed-candidate-common.bin"); store.abortGeneration(); topology.loadParameters(beforeSeed); throw;
    }
    std::ofstream report(output / "block-probe.txt"); report << std::setprecision(17) << "full_capture_verified=false\nsources=" << store.sourceCount()
        << "\noriginal_points=" << countPoints(store) << "\noriginal_observations=" << countRaw(store) << "\nselected_blocks=" << blocks << "\nthreads=" << threads
        << "\nbudget_bytes=" << budget << '\n'; report.flush(); reportMemory(report, "block_seed", budget);
    const auto initial = regional::workingQuality(store, topology, 3, threads); const auto initialGps = regional::regionalGpsQuality(store.poses(), gps);
    const auto seedGeneration = store.generation(); const auto seedHash = parameterHash(clone, seedGeneration);
    report << "initial_visual_cost=" << initial.cost << "\ninitial_finite_cost=" << initial.finite_cost << "\ninvalid_before=" << initial.invalid_projections
        << "\ngps_before=" << initialGps.inliers << '/' << initialGps.photos << "\ngps_cost_before=" << initialGps.cost
        << "\noriginal_manifest_sha256=" << manifestHash << "\noriginal_parameters_sha256=" << parametersHash << "\nseed_parameters_sha256=" << seedHash << '\n'; report.flush();
    regional::RegionalPoseAnchors held; const auto heldPath = consensusPath.parent_path() / "round-0-source-held.bin";
    if (fs::exists(heldPath)) for (const auto& image : model_store_detail::readModel(heldPath, false).images) if (image.second.registered) held.emplace(image.first, image.second.pose);
    SfmConfig cfg; cfg.threads = threads; cfg.regional_iterations = 1; cfg.max_error = 3; cfg.mapper.ba_real = "double";
    regional::RegionalCoordinator controller(store, topology, cfg, &gps, held, budget, output / "parameters");
    const auto started = std::chrono::steady_clock::now();
    const auto result = controller.run(plan, [&](const Reconstruction& candidate, const Reconstruction& before) {
        return regional::regionalGpsRoundAccepted(regional::regionalGpsQuality(candidate, gps), regional::regionalGpsQuality(before, gps)) ?
            std::vector<std::string>{} : std::vector<std::string>{"actual_absolute_GPS_cost_or_inlier_regression"};
    });
    const auto final = regional::workingQuality(store, topology, 3, threads); const auto finalGps = regional::regionalGpsQuality(store.poses(), gps);
    check(initial.observations == final.observations && final.observations == countRaw(store), "block native BA changed complete original observation coverage");
    if (!result.rounds) {
        check(store.generation() == seedGeneration && parameterHash(clone, seedGeneration) == seedHash, "block rejected round changed accepted pose/camera/XYZ parameter files");
        check(std::abs(initial.cost - final.cost) <= std::max(1., std::abs(initial.cost)) * 1e-9, "block rejected round did not restore full common/raw visual objective");
    }
    report << "initial_visual_cost=" << initial.cost << "\nfinal_visual_cost=" << final.cost << "\ninvalid_before=" << initial.invalid_projections
        << "\ninvalid_after=" << final.invalid_projections << "\ngps_before=" << initialGps.inliers << '/' << initialGps.photos
        << "\ngps_after=" << finalGps.inliers << '/' << finalGps.photos << "\ngps_cost_before=" << initialGps.cost << "\ngps_cost_after=" << finalGps.cost
        << "\naccepted_rounds=" << result.rounds << "\nround_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << '\n';
    for (const auto& reason : result.rejected) report << "rejected=" << reason << '\n';
    const auto& io = store.counters(); report << "logical_raw_bytes=" << io.raw_bytes << "\nlogical_index_bytes=" << io.index_bytes << "\nlogical_parameter_bytes=" << io.parameter_bytes << '\n';
    check(fileHash(original / "manifest.bin") == manifestHash && parameterHash(original, generation) == parametersHash,
        "isolated block probe changed production manifest or accepted parameters");
    report << "original_manifest_and_parameters_unchanged=true\ncompleted=true\n"; report.flush(); check(bool(report), "block probe evidence write failed");
    reportMemory(report, "block_round", budget);
    std::cout << "[capacity-block] complete full-resident native block probe finished; accepted rounds " << result.rounds << '\n'; return result.rounds ? 0 : 2;
}
static int recordOutcome(const fs::path& probe, const fs::path& gpsPath, size_t images) {
    model_store_detail::Reader header(probe / "working/manifest.bin", 64u << 10);
    check(header.scalar<uint64_t>() == 0x31534b524f575253ull && header.scalar<uint32_t>() == 3, "unexpected native working manifest prefix");
    const auto generation = header.scalar<uint64_t>(); const auto phase = header.string(); const auto sources = header.scalar<uint64_t>();
    const auto candidateGeneration = phase.find("joint-round-v1:") == 0 || phase.find("joint-round-private-v2:") == 0 ? generation : generation + 1;
    const auto prefix = probe / "parameters" / ("generation-" + std::to_string(candidateGeneration));
    const auto quality = regional::readProjectionQuality(prefix.string() + "-projection.bin"); CapturedPositions gps(gpsPath, images);
    const auto before = model_store_detail::readModel(probe / "working" / ("parameters-" + std::to_string(candidateGeneration - 1)) / "poses.bin", false);
    const auto candidate = model_store_detail::readModel(prefix.string() + "-poses.bin", false);
    const auto beforeGps = regional::regionalGpsQuality(before, gps), candidateGps = regional::regionalGpsQuality(candidate, gps);
    std::ofstream out(probe / "native-outcome-recovery.txt"); out << std::setprecision(17) << "full_capture_verified=false\naccepted_generation=" << generation
        << "\naccepted_phase=" << phase << "\nsources=" << sources << "\ncandidate_generation=" << candidateGeneration
        << "\ncandidate_observations=" << quality.observations << "\ncandidate_total_visual_cost=" << quality.cost
        << "\ncandidate_finite_visual_cost=" << quality.finite_cost << "\ncandidate_invalid_projections=" << quality.invalid_projections
        << "\ngps_before=" << beforeGps.inliers << '/' << beforeGps.photos << "\ngps_candidate=" << candidateGps.inliers << '/' << candidateGps.photos
        << "\ngps_cost_before=" << beforeGps.cost << "\ngps_cost_candidate=" << candidateGps.cost
        << "\nabsolute_GPS_round_gate=" << regional::regionalGpsRoundAccepted(candidateGps, beforeGps)
        << "\nnext_parameters_dir_exists=" << fs::exists(probe / "working" / ("parameters-" + std::to_string(generation + 1))) << '\n';
    if (fs::exists(prefix.string() + "-rejected.txt")) out << "rejected=" << resume::recorded(prefix.string() + "-rejected.txt");
    out.flush(); check(bool(out), "native outcome recovery write failed"); return 0;
}
static int capture(const fs::path& working, const fs::path& planRoot, const fs::path& consensusPath,
    const fs::path& output, size_t budget, size_t expectedSources, uint64_t expectedPoints, size_t images) {
    check(!fs::exists(output), "capacity candidate output already exists; choose a new isolated directory");
    const auto original = fs::canonical(working), target = fs::absolute(output).lexically_normal();
    check(target.u8string().find(original.u8string()) != 0, "capacity candidate cannot be inside original working store");
    const auto originalManifest = fileHash(original / "manifest.bin");
    std::cout << "[capacity-audit] strict native working-store restore begin; original source hashes verified once\n" << std::flush;
    Store store(original, budget / 2);
    check(store.sourceCount() == expectedSources && countPoints(store) == expectedPoints, "capture capacity input does not match expected complete source/point counts");
    const auto generation = store.generation(); const auto phase = store.generationPhase();
    const auto originalParameters = parameterHash(original, generation);
    const auto consensus = model_store_detail::readModel(consensusPath, false);
    check(!consensus.images.empty(), "capacity seed consensus is empty");
    const auto plan = regional::readRegionalPlan(planRoot, images);
    std::vector<uint32_t> owners(images, UINT32_MAX);
    for (uint32_t region = 0; region < plan.size(); ++region) for (auto image : plan[region].core) owners.at(image) = region;
    for (const auto& image : store.poses().images) if (image.second.registered) {
        const auto at = consensus.images.find(image.first);
        check(at != consensus.images.end() && at->second.name == image.second.name && at->second.camera_id == image.second.camera_id,
            "capacity canonical seed lost global source image identity");
    }
    fs::create_directories(output); std::ofstream report(output / "capacity.txt");
    report << std::setprecision(17) << "sources=" << store.sourceCount() << "\noriginal_points=" << countPoints(store)
        << "\noriginal_observations=" << countRaw(store) << "\nsource_generation=" << generation << "\nsource_phase=" << phase
        << "\ninput_images=" << images << "\nplan_blocks=" << plan.size() << "\nbudget_bytes=" << budget
        << "\nsource_manifest_sha256=" << originalManifest << "\nfull_capture_verified=false\n"; report.flush();
    reportMemory(report, "native_restore", budget);
    const auto reference = originalReference(store, budget); reportMemory(report, "original_native_reference", budget);
    report << "original_reader_raw_sha256=" << reference.raw << '\n'; report.flush();
    const auto start = std::chrono::steady_clock::now(); Snapshot cold;
    {
        Topology topology(output / "topology", budget); topology.prepare(store, owners);
        reportMemory(report, "cold_topology", budget);
        std::cout << "[capacity-audit] complete topology accepted; isolated canonical consensus seed triangulation begin\n" << std::flush;
        const auto seedStarted = std::chrono::steady_clock::now(); topology.triangulate(consensus, 3, 2);
        reportMemory(report, "seed_triangulate", budget);
        report << "candidate_seed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - seedStarted).count() << '\n'; report.flush();
        cold = snapshot(store, topology); topology.saveParameters(output / "common-native.bin");
        check(cold.raw == reference.raw && cold.points == reference.points && cold.observations == reference.observations,
            "complete cold flat raw/duplicates/feature XY differs from original native Reader");
        reportMemory(report, "cold_full_audit", budget);
        report << "cold_groups=" << cold.groups << "\ncommon_observations=" << cold.common << "\nraw_sha256=" << cold.raw
            << "\nlineage_sha256=" << cold.lineage << "\ncommon_track_sha256=" << cold.tracks
            << "\ncold_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
            << "\nresident_after_cold=" << spirula::processRamBytes() << '\n'; report.flush();
    }
    store.resetCounters();
    {
        Topology restored(output / "topology", budget); restored.prepare(store, owners);
        reportMemory(report, "warm_topology", budget);
        check(store.counters().raw_bytes == 0 && restored.counters().new_sources == 0, "warm complete topology reread or imported original raw sources");
        restored.restoreAcceptedParameters(output / "common-native.bin");
        const auto warm = snapshot(store, restored); check(cold == warm, "warm full topology/checkpoint differs from complete cold lineage, XYZ or original tracks");
        reportMemory(report, "warm_full_audit", budget);
    }
    check(store.generation() == generation && store.generationPhase() == phase && fileHash(original / "manifest.bin") == originalManifest &&
        parameterHash(original, generation) == originalParameters,
        "capacity audit modified original durable working generation");
    report << "warm_equals_cold=true\noriginal_working_manifest_unchanged=true\npassed=true\n"; report.flush();
    check(bool(report), "capacity report write failed"); std::cout << "complete capture capacity test passed\n"; return 0;
}
static size_t probeBudget(const char* argument) {
    const double mib = std::stod(argument); check(std::isfinite(mib) && (mib == 0 || mib >= 32) && mib < 1048576, "invalid native probe memory budget MiB");
    const auto physical = spirula::physicalRamBytes(), available = spirula::availableRamBytes();
    const auto budget = mib == 0 ? mappingMemoryLimits(physical, available).working_bytes : size_t(mib * (1u << 20));
    check(budget >= (32u << 20), "native automatic probe budget below minimum");
    std::cout << "[capacity-budget] physical " << physical << " available " << available << " budget " << budget << " requested MiB " << mib << '\n' << std::flush;
    return budget;
}
static int run(int argc, char** argv) {
    if (argc == 6 && std::string(argv[1]) == "--photo-cache-source-probe") {
        return largestPhotoCacheProbe(argv[2], argv[3], probeBudget(argv[4]), std::stoull(argv[5]));
    }
    if (argc == 5 && std::string(argv[1]) == "--record-block-outcome") return recordOutcome(argv[2], argv[3], size_t(std::stoull(argv[4])));
    if ((argc == 8 || argc == 9 || argc == 10) && std::string(argv[1]) == "--gather-probe") {
        return gatherProbe(argv[2], argv[3], argv[4], probeBudget(argv[5]), std::stoull(argv[6]), size_t(std::stoull(argv[7])), argc >= 9 ? size_t(std::stoull(argv[8])) : 2,
            argc == 10 ? size_t(std::stoull(argv[9])) : 0);
    }
    if ((argc == 8 || argc == 9) && std::string(argv[1]) == "--capture-append-probe") {
        const double mib = std::stod(argv[5]); check(std::isfinite(mib) && mib >= 32 && mib < 1048576, "invalid real append budget MiB");
        return appendProbe(argv[2], argv[3], argv[4], size_t(mib * (1u << 20)), std::stoull(argv[6]), size_t(std::stoull(argv[7])),
            argc == 9 ? size_t(std::stoull(argv[8])) : 4);
    }
    if ((argc == 11 || argc == 13 || argc == 14) && std::string(argv[1]) == "--capture-block-probe") {
        return blockProbe(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], probeBudget(argv[8]), std::stoull(argv[9]),
            size_t(std::stoull(argv[10])), argc >= 13 ? size_t(std::stoull(argv[11])) : 2, argc >= 13 ? std::stoi(argv[12]) : 2,
            argc == 14 ? fs::path(argv[13]) : fs::path{});
    }
    if (argc == 5 && std::string(argv[1]) == "--inspect-workstore") {
        const size_t budget = size_t(std::stod(argv[4]) * (1u << 20)); const fs::path working = argv[2];
        const auto before = fileHash(working / "manifest.bin");
        std::cout << "[capacity-audit] strict warm restore begins, including native source/flat/parameter SHA verification\n" << std::flush;
        Store store(working, budget / 2); std::ofstream out(argv[3]); out << "sources=" << store.sourceCount()
            << "\npoints=" << countPoints(store) << "\nobservations=" << countRaw(store) << "\ngeneration=" << store.generation()
            << "\nphase=" << store.generationPhase() << "\nmanifest_sha256=" << before << '\n';
        for (size_t source = 0; source < store.sourceCount(); ++source) {
            check(store.findSourceLabel(store.sourceLabel(source)) == source, "warm restored source labels duplicate or changed");
            out << "source=" << source << ',' << store.sourceLabel(source) << ',' << store.sourceIdentity(source) << ','
                << store.sourcePointCount(source) << ',' << store.sourceObservationCount(source) << '\n';
        }
        check(fileHash(working / "manifest.bin") == before, "inspect changed original manifest");
        out << "original_manifest_unchanged=true\nfull_capture_verified=false\n"; out.flush(); check(bool(out), "inspect evidence write failed");
        std::cout << "[capacity-audit] warm restored " << store.sourceCount() << " sources / " << countPoints(store)
            << " points / " << countRaw(store) << " original observations without import\n"; return 0;
    }
    if (argc == 10 && std::string(argv[1]) == "--capture-workstore") {
        const double mib = std::stod(argv[6]); check(std::isfinite(mib) && mib >= 32 && mib < 1048576, "invalid capacity budget MiB");
        return capture(argv[2], argv[3], argv[4], argv[5], size_t(mib * (1u << 20)), size_t(std::stoull(argv[7])), std::stoull(argv[8]), size_t(std::stoull(argv[9])));
    }
    if (argc != 1) throw std::runtime_error("usage: --capture-workstore WORK PLAN_ROOT CONSENSUS ISOLATED_OUTPUT BUDGET_MIB EXPECTED_SOURCES EXPECTED_POINTS INPUT_IMAGES; or --inspect-workstore WORK REPORT BUDGET_MIB");
    ModelStore scratch(""); const auto root = scratch.directory(); Store store(root / "working", 32u << 20);
    Reconstruction model; model.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::Pinhole);
    for (uint32_t image = 0; image < 2; ++image) {
        Image value; value.id = image; value.name = "capacity-photo-" + std::to_string(image); value.registered = true; value.camera_id = 0;
        value.pose = {mat3Identity(), {-.5 * image, 0, 0}}; value.points2D.resize(6); value.point3D_ids.assign(6, kInvalidPoint3D);
        model.images.emplace(image, std::move(value));
    }
    for (uint32_t point = 0; point < 6; ++point) {
        Point3D value; value.xyz = {.12 * point, .1 * (point % 3), 5. + .03 * point}; value.track = {{0, point}, {1, point}};
        for (auto& image : model.images) { image.second.points2D[point] = model.cameras[0].project(value.xyz + image.second.pose.t); image.second.point3D_ids[point] = 30 + point; }
        model.points3D.emplace(30 + point, value);
    }
    const auto consensus = root / "consensus.bin"; model_store_detail::writeModel(model, consensus);
    const auto sourceA = root / "source-a.bin"; model.points3D.at(30).track.push_back({0, 0}); model_store_detail::writeModel(model, sourceA); store.importModel(sourceA, model);
    const auto sourceB = root / "source-b.bin"; model.points3D.at(30).track.pop_back(); model_store_detail::writeModel(model, sourceB); store.importModel(sourceB, model);
    model_store_detail::Writer plan(root / "plan.bin"); plan.scalar<uint32_t>(1); plan.scalar<uint64_t>(2);
    for (uint32_t image = 0; image < 2; ++image) { plan.vector(std::vector<uint32_t>{image}); plan.vector(std::vector<uint32_t>{0, 1}); } plan.finish();
    return capture(root / "working", root, consensus, root / "candidate", 64u << 20, 2, 12, 2);
}
int main(int argc, char** argv) { return sfmTestMain(argc, argv, run); }
