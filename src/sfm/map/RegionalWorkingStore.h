#pragma once

#include <cmath>
#include <cerrno>
#include <queue>
#include <memory>
#include <tuple>
#include "core/Sha256.h"
#include "sfm/map/ModelStore.h"
#include "sfm/map/RegionalStep.h"
#include "sfm/map/VerifiedImmutableFiles.h"
#include "sfm/core/Cancel.h"
#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace sfm::regional {

class RegionalWorkingStore {
    using Path = std::filesystem::path;
    using Reader = model_store_detail::Reader;
    using Writer = model_store_detail::Writer;
    static constexpr uint64_t magic = 0x31534b524f575253ull;
public:
    struct PointKey {
        uint64_t source = 0, ordinal = 0, localPoint = 0;
        bool operator<(const PointKey& b) const { return std::tie(source, ordinal, localPoint) < std::tie(b.source, b.ordinal, b.localPoint); }
        bool operator==(const PointKey& b) const { return source == b.source && ordinal == b.ordinal && localPoint == b.localPoint; }
    };
    struct RawObservation {
        PointKey point;
        uint64_t ordinal = 0;
        uint32_t image = 0, feature = 0;
        Vec2 xy;
    };
    using XYZResolver = std::function<Vec3(PointKey, Vec3)>;
    struct IoCounters {
        uint64_t source_full_reads = 0, raw_bytes = 0, index_bytes = 0, parameter_bytes = 0, seeks = 0;
        uint64_t reader_hits = 0, reader_opens = 0, reader_evictions = 0;
        uint64_t photo_cache_builds = 0, photo_cache_input_bytes = 0, photo_cache_written_bytes = 0;
        uint64_t bytesHashChecked = 0, leaseHits = 0, leaseMisses = 0, leasedFiles = 0;
        double photo_cache_build_seconds = 0;
    };
    struct Gather {
        Reconstruction model;
        std::map<uint64_t, PointKey> pointKeys;
        std::set<uint32_t> fixedImages;
        std::set<uint64_t> boundaryPoints;
    };
private:
    struct PointRecord {
        uint64_t local = 0, first = 0, count = 0;
        Vec3 original;
        double error = 0;
        uint8_t rgb[3]{};
    };
    struct ImageRange {
        uint64_t features = 0, count = 0, photoFirst = 0, photoCount = 0;
    };
    struct PhotoRef {
        uint32_t image = 0;
        uint64_t observation = 0;
        bool operator<(const PhotoRef& b) const { return std::tie(image, observation) < std::tie(b.image, b.observation); }
    };
    struct Source {
        std::string path, label, digest;
        uint64_t bytes = 0, points = 0, observations = 0;
        int64_t mtime = 0;
        std::map<uint32_t, ImageRange> images;
        std::vector<std::string> hashes;
    };
    struct FeatureRecord { Vec2 xy; uint64_t assignment = kInvalidPoint3D; };
    class FileGuard {
    public:
        explicit FileGuard(const Path& path, bool readOnly = false) {
#ifdef _WIN32
            handle_ = CreateFileW(path.c_str(), readOnly ? GENERIC_READ : GENERIC_READ | GENERIC_WRITE,
                readOnly ? FILE_SHARE_READ : 0, nullptr, readOnly ? OPEN_EXISTING : OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (readOnly && handle_ == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_NOT_FOUND) return;
            if (handle_ == INVALID_HANDLE_VALUE) throw std::runtime_error("working store transaction is locked");
#else
            handle_ = readOnly ? open(path.c_str(), O_RDONLY) : open(path.c_str(), O_CREAT | O_RDWR, 0600);
            if (readOnly && handle_ < 0 && errno == ENOENT) return;
            if (handle_ < 0 || flock(handle_, (readOnly ? LOCK_SH : LOCK_EX) | LOCK_NB) != 0) {
                if (handle_ >= 0) close(handle_); throw std::runtime_error("working store transaction is locked");
            }
#endif
        }
        ~FileGuard() {
#ifdef _WIN32
            if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
            if (handle_ >= 0) { flock(handle_, LOCK_UN); close(handle_); }
#endif
        }
        FileGuard(const FileGuard&) = delete;
        FileGuard& operator=(const FileGuard&) = delete;
    private:
#ifdef _WIN32
        HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
        int handle_ = -1;
#endif
    };
public:
    enum class OpenMode { ReadWrite, ReadOnly };
    RegionalWorkingStore(const Path& root, size_t budget, OpenMode mode = OpenMode::ReadWrite)
        : root_(std::filesystem::absolute(root)), budget_(budget), read_only_(mode == OpenMode::ReadOnly), verified_(std::max<size_t>(256u << 10, budget / 32)) {
        if (budget < (16u << 20)) throw std::length_error("working store requires a 16 MiB minimum budget");
        if (read_only_) {
            if (!std::filesystem::is_directory(root_) || !std::filesystem::is_regular_file(root_ / "manifest.bin"))
                throw std::runtime_error("read-only working store requires an existing accepted manifest");
        } else std::filesystem::create_directories(root_);
        if (std::filesystem::exists(root_ / "manifest.bin")) restore();
    }
    ~RegionalWorkingStore() { readers_.clear(); transaction_.reset(); }
    const Reconstruction& poses() const { return poses_; }
    const IoCounters& counters() const { return counters_; }
    uint64_t residentBytes() const {
        uint64_t bytes = storage::addBytes(storage::addBytes(metadataResident(), modelResidentBytes(savedPoses_)), sizeof(reader_tick_));
        bytes = storage::addBytes(bytes, storage::scaleBytes(photo_dependencies_.size() + photo_caches_.size(), 1024));
        for (const auto& reader : readers_) bytes = storage::addBytes(bytes, (64u << 10) + sizeof(Reader) + sizeof(CachedReader) + 256 +
            storage::scaleBytes(reader.first.native().capacity(), sizeof(Path::value_type)));
        return bytes;
    }
    void resetCounters() { counters_ = {}; }
    size_t sourceCount() const { return sources_.size(); }
    uint64_t sourcePointCount(size_t source) const { return sources_.at(source).points; }
    uint64_t sourceObservationCount(size_t source) const { return sources_.at(source).observations; }
    uint64_t sourcePhotoObservationCount(size_t source, uint32_t image) const {
        const auto& images = sources_.at(source).images; auto found = images.find(image); return found == images.end() ? 0 : found->second.photoCount;
    }
    uint64_t sourceImageFeatureCount(size_t source, uint32_t image) const {
        const auto& images = sources_.at(source).images; auto found = images.find(image); return found == images.end() ? 0 : found->second.count;
    }
    uint64_t sourcePhotoFeatureCount(size_t source, uint32_t image) const { return sourceImageFeatureCount(source, image); }
    template<class Visit> void visitSourceImageFeatures(size_t source, uint32_t image, Visit visit) const {
        const auto& range = sources_.at(source).images.at(image); Reader in(sourceDir(source) / "features.bin", 64u << 10);
        in.seek(storage::scaleBytes(range.features, sizeof(FeatureRecord))); ++counters_.seeks;
        for (uint64_t i = 0; i < range.count; ++i) { auto record = in.scalar<FeatureRecord>(); counters_.raw_bytes += sizeof(record); visit(i, record.xy, record.assignment); }
    }
    const std::string& generationPhase() const { return phase_; }
    std::string sourceIdentity(size_t source) const { const auto& s = sources_.at(source); return s.digest + "." + std::to_string(s.bytes); }
    std::string sourceRawIdentity(size_t source) const {
        const auto& row = sources_.at(source); spirula::Sha256 sha;
        for (const auto& hash : row.hashes) { const uint64_t bytes = hash.size(); sha.update(reinterpret_cast<const uint8_t*>(&bytes), sizeof(bytes)); sha.update(reinterpret_cast<const uint8_t*>(hash.data()), hash.size()); }
        return sha.hex();
    }
    const std::string& sourceLabel(size_t source) const { return sources_.at(source).label; }
    size_t findSourceLabel(const std::string& label) const {
        for (size_t i = 0; i < sources_.size(); ++i) if (sources_[i].label == label) return i; return SIZE_MAX;
    }
    size_t knownSource(const Path& original) const {
        const auto path = std::filesystem::canonical(original).u8string();
        for (size_t i = 0; i < sources_.size(); ++i) if (sources_[i].path == path) {
            if (sources_[i].bytes != std::filesystem::file_size(original) || sources_[i].mtime != stamp(original))
                throw std::runtime_error("immutable working source identity changed");
            return i;
        }
        return SIZE_MAX;
    }
    size_t findSource(const Path& original) const { return knownSource(original); }
    Path forkImmutablePrefix(const Path& destination, size_t count) const {
        if (editing_ || !count || count > sources_.size()) throw std::invalid_argument("immutable fork requires a nonempty accepted source prefix");
        requireCurrentHead();
        const Path source = std::filesystem::canonical(root_), target = std::filesystem::weakly_canonical(std::filesystem::absolute(destination));
        const auto relative = target.lexically_relative(source);
        if (target == source || (!relative.empty() && *relative.begin() != Path("..")) || std::filesystem::exists(target))
            throw std::invalid_argument("immutable fork requires a new destination outside its source root");
        for (Path parent = target.parent_path(); !parent.empty();) {
            if (std::filesystem::exists(parent) && std::filesystem::equivalent(parent, source))
                throw std::invalid_argument("immutable fork destination is physically inside its source root");
            const Path next = parent.parent_path(); if (next == parent) break; parent = next;
        }
        for (size_t i = 0; i < count; ++i) {
            size_t h = 0; for (const auto* file : immutableFiles()) verifyDependency(sourceDir(i) / file, sources_[i].hashes.at(h++));
        }
        std::filesystem::create_directories(target.parent_path());
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        const Path stage = target.parent_path() / (target.filename().u8string() + ".fork-part-" + std::to_string(tick));
        RegionalWorkingStore fork(stage, budget_); fork.sources_.assign(sources_.begin(), sources_.begin() + count);
        fork.poses_ = poses_; std::set<uint32_t> images, cameras;
        for (size_t i = 0; i < count; ++i) for (const auto& entry : sources_[i].images) images.insert(entry.first);
        for (auto it = fork.poses_.images.begin(); it != fork.poses_.images.end();) {
            if (!images.count(it->first)) it = fork.poses_.images.erase(it); else { cameras.insert(it->second.camera_id); ++it; }
        }
        for (auto it = fork.poses_.cameras.begin(); it != fork.poses_.cameras.end();) {
            if (!cameras.count(it->first)) it = fork.poses_.cameras.erase(it); else ++it;
        }
        fork.phase_ = "immutable-prefix-fork-v1"; fork.metadata_bytes_ = fork.measureMetadata(); fork.checkBudget(fork.metadataResident());
        const Path parameters = fork.parameterDir(0); std::filesystem::create_directories(parameters);
        for (size_t i = 0; i < count; ++i) {
            std::filesystem::create_directory(fork.sourceDir(i));
            for (const auto* file : immutableFiles()) std::filesystem::create_hard_link(sourceDir(i) / file, fork.sourceDir(i) / file);
            std::filesystem::create_hard_link(sourceDir(i) / "initial.xyz", parameters / fork.xyzName(i));
        }
        fork.savePoses(parameters); fork.saveManifest(); fork.verifyImmutable();
        fork.verified_.clear(); requireCurrentHead(); std::filesystem::rename(stage, target); return target;
    }
    uint64_t generation() const { return generation_; }
    Path sourcePath(size_t source) const { return std::filesystem::u8path(sources_.at(source).path); }
    Reconstruction sourceViews(size_t source) const { (void)sources_.at(source); return model_store_detail::readModel(sourceDir(source) / "views.bin"); }

    size_t importModel(const Path& original, const Reconstruction& model, const std::string& label = {}) {
        requireWritable();
        if (editing_) throw std::logic_error("cannot append source during a parameter transaction");
        FileGuard guard(root_ / "transaction.lock");
        requireCurrentHead();
        const size_t known = knownSource(original); if (known != SIZE_MAX) {
            if (!label.empty() && sources_[known].label != label) throw std::runtime_error("working source label identity changed"); return known;
        }
        Source source; source.path = std::filesystem::canonical(original).u8string();
        source.label = label.empty() ? source.path : label;
        if (findSourceLabel(source.label) != SIZE_MAX) throw std::runtime_error("duplicate immutable working source label");
        source.bytes = std::filesystem::file_size(original); source.mtime = stamp(original);
        source.digest = hash(original);
        DigestOutput digest;
        model_store_detail::serializeModel(model, digest);
        if (digest.finish() != source.digest) throw std::runtime_error("imported model differs from frozen source");
        const size_t slot = sources_.size(); const Path dir = sourceDir(slot);
        if (std::filesystem::exists(dir)) throw std::runtime_error("uncommitted working source directory exists");
        storage::requireSpace(root_, storage::scaleBytes(source.bytes, 3));
        std::filesystem::create_directory(dir);
        Reconstruction views; views.cameras = model.cameras; views.rigs = model.rigs;
        views.rig_detached = model.rig_detached; views.next_point3D_id = model.next_point3D_id;
        for (const auto& pair : model.images) {
            Image image; image.id = pair.second.id; image.camera_id = pair.second.camera_id; image.name = pair.second.name;
            image.pose = pair.second.pose; image.registered = pair.second.registered; image.exif_orientation = pair.second.exif_orientation;
            views.images.emplace(pair.first, std::move(image));
        }
        model_store_detail::writeModel(views, dir / "views.bin");
        Writer featureOut(dir / "features.bin"), pointOut(dir / "points.bin"), rawOut(dir / "observations.bin"), xyzOut(dir / "initial.xyz");
        uint64_t featureOffset = 0;
        for (const auto& pair : model.images) {
            const Image& im = pair.second;
            if (pair.first != im.id || im.points2D.size() != im.point3D_ids.size()) throw std::runtime_error("invalid working source image");
            source.images.emplace(pair.first, ImageRange{featureOffset, im.points2D.size(), 0, 0});
            for (size_t i = 0; i < im.points2D.size(); ++i) { FeatureRecord f{}; f.xy = im.points2D[i]; f.assignment = im.point3D_ids[i]; featureOut.scalar(f); ++featureOffset; }
        }
        std::vector<PhotoRef> chunk;
        const size_t chunkLimit = std::max<size_t>(1, budget_ / 16 / sizeof(PhotoRef));
        chunk.reserve(chunkLimit); size_t runs = 0;
        auto flush = [&] {
            if (chunk.empty()) return;
            std::sort(chunk.begin(), chunk.end()); Writer out(dir / ("run-" + std::to_string(runs++) + ".bin"));
            for (auto& ref : chunk) out.scalar(ref); out.finish(); chunk.clear();
        };
        for (const auto& pair : model.points3D) {
            PointRecord p{}; p.local = pair.first; p.first = source.observations; p.count = pair.second.track.size();
            p.original = pair.second.xyz; p.error = pair.second.error; std::copy(std::begin(pair.second.rgb), std::end(pair.second.rgb), p.rgb);
            pointOut.scalar(p); xyzOut.scalar(p.original);
            for (const auto& track : pair.second.track) {
                const auto& im = model.images.at(track.image_id);
                if (track.point2D_idx >= im.points2D.size()) throw std::runtime_error("invalid raw working observation");
                RawObservation row{}; row.point = {slot, source.points, p.local}; row.ordinal = source.observations++;
                row.image = track.image_id; row.feature = track.point2D_idx; row.xy = im.points2D[track.point2D_idx];
                rawOut.scalar(row); chunk.push_back({row.image, row.ordinal}); if (chunk.size() == chunkLimit) flush();
            }
            ++source.points;
        }
        flush(); featureOut.finish(); pointOut.finish(); rawOut.finish(); xyzOut.finish();
        makePhotoIndex(dir, runs, source);
        for (const char* file : immutableFiles()) source.hashes.push_back(hash(dir / file));
        auto proposed = poses_;
        for (const auto& pair : views.cameras) proposed.cameras.emplace(pair);
        for (const auto& pair : views.images) {
            auto at = proposed.images.find(pair.first);
            if (at != proposed.images.end() && (at->second.name != pair.second.name || at->second.camera_id != pair.second.camera_id))
                throw std::runtime_error("global image identity conflicts between source models");
            if (at != proposed.images.end() && !at->second.registered && pair.second.registered) at->second = pair.second;
            else proposed.images.emplace(pair);
        }
        uint64_t memory = modelResidentBytes(proposed) + metadataBytes(source);
        for (const auto& existing : sources_) memory = storage::addBytes(memory, metadataBytes(existing));
        checkBudget(storage::addBytes(memory, modelResidentBytes(poses_) + modelResidentBytes(views)));
        if (source.bytes != std::filesystem::file_size(original) || source.mtime != stamp(original)) throw std::runtime_error("source changed during import");
        const uint64_t previous = generation_;
        const bool accepted = std::filesystem::exists(root_ / "manifest.bin");
        const uint64_t next = accepted ? generation_ + 1 : generation_;
        const Path current = parameterDir(next);
        if (std::filesystem::exists(current)) throw std::runtime_error("uncommitted import parameter generation exists");
        std::filesystem::create_directory(current);
        if (accepted) for (size_t i = 0; i < sources_.size(); ++i)
            storage::retainFile(parameterDir(generation_) / xyzName(i), current / xyzName(i));
        std::filesystem::copy_file(dir / "initial.xyz", current / xyzName(slot));
        const auto oldMetadata = metadata_bytes_;
        sources_.push_back(std::move(source)); auto oldPoses = std::move(poses_); poses_ = std::move(proposed); generation_ = next; metadata_bytes_ = measureMetadata();
        try { savePoses(current); saveManifest(); }
        catch (...) { generation_ = previous; poses_ = std::move(oldPoses); sources_.pop_back(); metadata_bytes_ = oldMetadata; throw; }
        rememberPhotoDependencies(slot);
        return slot;
    }

    PointKey key(size_t source, uint64_t ordinal) const {
        const auto p = pointRecord(source, ordinal); return {source, ordinal, p.local};
    }
    PointKey localKey(size_t source, uint64_t local) const {
        uint64_t lo = 0, hi = sources_.at(source).points;
        while (lo < hi) { uint64_t mid = lo + (hi - lo) / 2; auto p = pointRecord(source, mid); if (p.local < local) lo = mid + 1; else hi = mid; }
        if (lo == sources_.at(source).points) throw std::out_of_range("working point identity");
        auto result = key(source, lo); if (result.localPoint != local) throw std::out_of_range("working point identity"); return result;
    }
    Vec3 pointXYZ(const PointKey& p) const {
        validateKey(p); return readAt<Vec3>(activeParameters() / xyzName(p.source), p.ordinal, counters_.parameter_bytes);
    }
    struct PointAppearance { uint8_t rgb[3]{}; double error = 0; };
    PointAppearance pointAppearance(const PointKey& key) const {
        const auto record = validateKey(key); PointAppearance out; out.error = record.error;
        std::copy(std::begin(record.rgb), std::end(record.rgb), out.rgb); return out;
    }
    Point3D point(const PointKey& p) const {
        const auto meta = validateKey(p); Point3D result; result.xyz = pointXYZ(p); result.error = meta.error;
        std::copy(std::begin(meta.rgb), std::end(meta.rgb), result.rgb);
        checkBudget(metadataResident() + storage::scaleBytes(meta.count, sizeof(TrackElement)));
        result.track.reserve(size_t(meta.count));
        visitPoint(p, [&](const RawObservation& row) { result.track.push_back({row.image, row.feature}); }); return result;
    }
    template<class Visit> void visitPoint(const PointKey& p, Visit visit) const {
        auto meta = validateKey(p); auto held = cachedReader(sourceDir(p.source) / "observations.bin"); Reader& reader = *held;
        reader.seek(storage::scaleBytes(meta.first, sizeof(RawObservation))); ++counters_.seeks;
        for (uint64_t i = 0; i < meta.count; ++i) {
            auto row = reader.scalar<RawObservation>(); counters_.raw_bytes += sizeof(row);
            if (!(row.point == p) || row.ordinal != meta.first + i) throw std::runtime_error("working observation point identity corrupted"); visit(row);
        }
    }
    template<class Visit> void visitObservations(Visit visit) const {
        for (size_t source = 0; source < sources_.size(); ++source) {
            Reader reader(sourceDir(source) / "observations.bin", 64u << 10);
            for (uint64_t i = 0; i < sources_[source].observations; ++i) {
                auto row = reader.scalar<RawObservation>(); counters_.raw_bytes += sizeof(row);
                if (row.point.source != source || row.ordinal != i) throw std::runtime_error("working observation ordering corrupted"); visit(row);
            }
            reader.finish();
        }
    }
    template<class Visit> void visitSourcePoints(size_t source, Visit visit) const {
        visitSourcePointsDetailed(source, [&](PointKey key, const PointRecord&, Vec3 value, const std::vector<RawObservation>& track) {
            visit(key, value, track);
        });
    }
    template<class Visit> void visitSourceParameters(size_t source, Visit visit) const {
        const auto& src = sources_.at(source); Reader points(sourceDir(source) / "points.bin", 64u << 10);
        Reader xyz(activeParameters() / xyzName(source), 64u << 10); uint64_t first = 0;
        for (uint64_t ordinal = 0; ordinal < src.points; ++ordinal) {
            const auto meta = points.scalar<PointRecord>(); const auto value = xyz.scalar<Vec3>();
            counters_.index_bytes += sizeof(meta); counters_.parameter_bytes += sizeof(value);
            if (meta.first != first || meta.count > src.observations - first) throw std::runtime_error("working parameter point ranges corrupted");
            visit(PointKey{source, ordinal, meta.local}, value); first += meta.count;
        }
        if (first != src.observations) throw std::runtime_error("working parameter original coverage incomplete");
        points.finish(); xyz.finish();
    }
    template<class Visit> void visitSelectedPoints(const std::vector<PointKey>& selection, Visit visit) const {
        checkBudget(metadataResident() + storage::scaleBytes(selection.size(), sizeof(size_t)));
        std::vector<size_t> order(selection.size()); for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return selection[a] < selection[b]; });
        size_t first = 0; std::vector<RawObservation> track;
        while (first < order.size()) {
            const size_t source = size_t(selection[order[first]].source); const auto& src = sources_.at(source);
            Reader points(sourceDir(source) / "points.bin", 64u << 10), raw(sourceDir(source) / "observations.bin", 64u << 10);
            Reader xyz(activeParameters() / xyzName(source), 64u << 10); PointKey previous{}; bool havePrevious = false;
            do {
                const auto key = selection[order[first++]]; if (havePrevious && key == previous) continue; previous = key; havePrevious = true;
                if (key.ordinal >= src.points) throw std::out_of_range("selected working point ordinal");
                points.seek(storage::scaleBytes(key.ordinal, sizeof(PointRecord))); xyz.seek(storage::scaleBytes(key.ordinal, sizeof(Vec3))); counters_.seeks += 2;
                const auto meta = points.scalar<PointRecord>(); const auto value = xyz.scalar<Vec3>(); counters_.index_bytes += sizeof(meta); counters_.parameter_bytes += sizeof(value);
                if (meta.local != key.localPoint || meta.first > src.observations || meta.count > src.observations - meta.first)
                    throw std::runtime_error("selected working point lineage/range changed");
                checkBudget(metadataResident() + order.size() * sizeof(size_t) + storage::scaleBytes(meta.count + track.capacity(), sizeof(RawObservation)));
                track.clear(); if (track.capacity() < meta.count) track.reserve(size_t(meta.count));
                raw.seek(storage::scaleBytes(meta.first, sizeof(RawObservation))); ++counters_.seeks;
                for (uint64_t i = 0; i < meta.count; ++i) {
                    auto row = raw.scalar<RawObservation>(); counters_.raw_bytes += sizeof(row);
                    if (!(row.point == key) || row.ordinal != meta.first + i) throw std::runtime_error("selected raw observation order changed"); track.push_back(row);
                }
                visit(key, value, track);
            } while (first < order.size() && selection[order[first]].source == source);
        }
    }
    template<class Visit> void visitSourcePointsDetailed(size_t source, Visit visit, bool originalOnly = false) const {
        const auto& src = sources_.at(source);
        Reader points(sourceDir(source) / "points.bin", 64u << 10), raw(sourceDir(source) / "observations.bin", 64u << 10);
        std::unique_ptr<Reader> xyz; if (!originalOnly) xyz = std::make_unique<Reader>(activeParameters() / xyzName(source), 64u << 10);
        uint64_t first = 0;
        std::vector<RawObservation> track;
        for (uint64_t ordinal = 0; ordinal < src.points; ++ordinal) {
            auto meta = points.scalar<PointRecord>(); auto value = originalOnly ? meta.original : xyz->scalar<Vec3>();
            counters_.index_bytes += sizeof(meta); if (!originalOnly) counters_.parameter_bytes += sizeof(value);
            if (meta.first != first || meta.count > src.observations - first) throw std::runtime_error("working point ranges corrupted");
            checkBudget(metadataResident() + storage::scaleBytes(meta.count, sizeof(RawObservation)));
            track.clear(); if (track.capacity() < meta.count) {
                checkBudget(metadataResident() + storage::scaleBytes(meta.count + track.capacity(), sizeof(RawObservation)));
                track.reserve(size_t(meta.count));
            }
            PointKey key{source, ordinal, meta.local};
            for (uint64_t j = 0; j < meta.count; ++j) {
                auto row = raw.scalar<RawObservation>(); counters_.raw_bytes += sizeof(row);
                if (!(row.point == key) || row.ordinal != first + j) throw std::runtime_error("working point track order corrupted"); track.push_back(row);
            }
            visit(key, meta, value, track); first += meta.count;
        }
        if (first != src.observations) throw std::runtime_error("working observation coverage incomplete");
        points.finish(); raw.finish(); if (xyz) xyz->finish();
    }
    template<class Visit> void visitImmutableSourcePointsDetailed(size_t source, Visit visit) const {
        visitSourcePointsDetailed(source, std::move(visit), true);
    }
    // Producer emits one XYZ per original point ordinal. Reads of this source still
    // observe its active parameters until the entire new stream has been verified.
    template<class Producer> void rewriteSourceXYZ(size_t source, Producer produce) {
        requireEditing(); const auto& src = sources_.at(source);
        const Path destination = pending_ / xyzName(source);
        Path temporary = pending_ / (xyzName(source) + ".private.part");
        for (uint64_t suffix = 1; std::filesystem::exists(temporary); ++suffix)
            temporary = pending_ / (xyzName(source) + ".private.part-" + std::to_string(suffix));
        storage::requireSpace(root_, storage::scaleBytes(src.points, sizeof(Vec3)));
        checkBudget(storage::addBytes(residentBytes(), 1u << 20));
        {
            Writer out(temporary); uint64_t emitted = 0;
            auto emit = [&](Vec3 value) {
                if (emitted == src.points) throw std::runtime_error("too many rewritten working points");
                if (!finite(value)) throw std::runtime_error("nonfinite rewritten working point");
                out.scalar(value); ++emitted; counters_.parameter_bytes += sizeof(value);
            };
            produce(emit);
            if (emitted != src.points) throw std::runtime_error("incomplete rewritten working point coverage");
            out.finish();
        }
        if (std::filesystem::file_size(temporary) != storage::scaleBytes(src.points, sizeof(Vec3)))
            throw std::runtime_error("rewritten working parameter size mismatch");
        readers_.erase(destination); storage::replaceFile(temporary, destination);
    }
    template<class Update> void updateSourceXYZ(size_t source, Update update) {
        requireEditing(); const auto& src = sources_.at(source);
        Reader points(sourceDir(source) / "points.bin", 64u << 10), xyz(pending_ / xyzName(source), 64u << 10);
        const Path temporary = pending_ / (xyzName(source) + ".part"); Writer out(temporary);
        for (uint64_t ordinal = 0; ordinal < src.points; ++ordinal) {
            auto meta = points.scalar<PointRecord>(); auto value = xyz.scalar<Vec3>();
            counters_.index_bytes += sizeof(meta); counters_.parameter_bytes += sizeof(value);
            auto updated = update(PointKey{source, ordinal, meta.local}, value);
            if (!finite(updated)) throw std::runtime_error("nonfinite working point update"); out.scalar(updated); counters_.parameter_bytes += sizeof(updated);
        }
        points.finish(); xyz.finish(); out.finish(); readers_.erase(pending_ / xyzName(source)); storage::replaceFile(temporary, pending_ / xyzName(source));
    }
    template<class Visit> void visitImageObservations(uint32_t image, Visit visit) const {
        for (size_t source = 0; source < sources_.size(); ++source) {
            auto found = sources_[source].images.find(image); if (found == sources_[source].images.end()) continue;
            const auto& range = found->second;
            if (!range.photoCount) continue;
            Reader raw(preparePhotoObservationCache(source), 64u << 10);
            raw.seek(storage::scaleBytes(range.photoFirst, sizeof(RawObservation))); ++counters_.seeks;
            uint64_t previous = 0;
            for (uint64_t i = 0; i < range.photoCount; ++i) {
                auto row = raw.scalar<RawObservation>(); counters_.raw_bytes += sizeof(row);
                if (row.image != image || row.point.source != source || row.ordinal >= sources_[source].observations ||
                    row.feature >= range.count || row.point.ordinal >= sources_[source].points || (i && row.ordinal <= previous))
                    throw std::runtime_error("working photo index identity corrupted"); visit(row);
                previous = row.ordinal;
            }
        }
    }
    Path preparePhotoObservationCache(size_t source) const {
        const auto& src = sources_.at(source); checkPhotoDependencies(source);
        auto cached = photo_caches_.find(source);
        if (cached != photo_caches_.end()) {
            if (!(cached->second.data == fileState(cached->second.path)) || !(cached->second.marker == fileState(cached->second.path.parent_path() / "ready.bin")))
                throw std::runtime_error("photo-major cache changed; candidate retained");
            return cached->second.path;
        }
        FileGuard lock(root_ / "photo-cache.lock", read_only_);
        spirula::Sha256 binding; const auto identity = src.hashes.at(3) + src.hashes.at(4);
        binding.update(reinterpret_cast<const uint8_t*>(identity.data()), identity.size());
        // Full dependency identities remain in the marker; bounded path length
        // also permits use in deeply nested Windows regional work directories.
        const Path dir = root_ / "photo-observations-v1" / ("source-" + std::to_string(source) + "-" + binding.hex().substr(0, 24));
        const Path data = dir / "observations.bin", marker = dir / "ready.bin";
        if (std::filesystem::exists(dir)) {
            if (!std::filesystem::exists(marker)) throw std::runtime_error("incomplete photo-major cache retained; explicit recovery required");
            Reader ready(marker, 64u << 10);
            if (ready.scalar<uint64_t>() != photoMagic || ready.scalar<uint32_t>() != 1 || ready.scalar<uint64_t>() != source ||
                ready.scalar<uint64_t>() != sizeof(RawObservation) || ready.scalar<uint64_t>() != src.observations ||
                ready.string() != src.hashes.at(3) || ready.string() != src.hashes.at(4))
                throw std::runtime_error("photo-major cache identity mismatch; candidate retained");
            const auto expected = ready.string(); ready.finish();
            if (std::filesystem::file_size(data) != storage::scaleBytes(src.observations, sizeof(RawObservation)) || hash(data) != expected)
                throw std::runtime_error("photo-major cache checksum mismatch; candidate retained");
        } else { requireWritable(); buildPhotoObservationCache(source, dir); }
        checkPhotoDependencies(source);
        photo_caches_.emplace(source, PhotoCache{data, fileState(data), fileState(marker)}); return data;
    }
    Gather gather(const std::set<uint32_t>& targets, const std::vector<PointKey>& activePoints = {}, bool includeBoundary = true) const {
        // Cold sort runs must not coexist with the large gathered point/track maps.
        for (size_t source = 0; source < sources_.size(); ++source) {
            bool needed = false;
            for (auto image : targets) if (sourcePhotoObservationCount(source, image)) { needed = true; break; }
            if (needed) preparePhotoObservationCache(source);
        }
        Gather out; std::set<PointKey> points(activePoints.begin(), activePoints.end());
        std::map<uint32_t, std::vector<uint8_t>> observed;
        std::vector<RawObservation> selected;
        if (!includeBoundary) {
            uint64_t count = 0;
            for (const auto& source : sources_) for (auto id : targets) {
                auto found = source.images.find(id); if (found != source.images.end()) count = storage::addBytes(count, found->second.photoCount);
            }
            checkBudget(metadataResident() + storage::scaleBytes(count, sizeof(RawObservation))); selected.reserve(size_t(count));
        }
        const uint64_t selectedBytes = storage::scaleBytes(selected.capacity(), sizeof(RawObservation));
        for (auto id : targets) visitImageObservations(id, [&](const RawObservation& row) {
            points.insert(row.point); if (!includeBoundary) selected.push_back(row);
            checkBudget(metadataResident() + points.size() * 80 + selectedBytes);
        });
        std::sort(selected.begin(), selected.end(), [](const RawObservation& a, const RawObservation& b) {
            return a.point < b.point || (!(b.point < a.point) && a.ordinal < b.ordinal);
        });
        size_t selectedAt = 0;
        size_t resident = metadataResident() + points.size() * 80 + size_t(selectedBytes);
        auto addImage = [&](uint32_t image, size_t source) {
            if (out.model.images.count(image)) return;
            const auto& range = sources_.at(source).images.at(image);
            resident = size_t(storage::addBytes(resident, 640 + storage::scaleBytes(range.count, sizeof(Vec2) + sizeof(uint64_t) + sizeof(uint8_t)))); checkBudget(resident);
            Image im = poses_.images.at(image); im.points2D.resize(size_t(range.count)); im.point3D_ids.assign(size_t(range.count), kInvalidPoint3D);
            Reader reader(sourceDir(source) / "features.bin", 64u << 10); reader.seek(storage::scaleBytes(range.features, sizeof(FeatureRecord))); ++counters_.seeks;
            for (size_t i = 0; i < im.points2D.size(); ++i) { auto f = reader.scalar<FeatureRecord>(); im.points2D[i] = f.xy; counters_.raw_bytes += sizeof(f); }
            out.model.cameras.emplace(im.camera_id, poses_.cameras.at(im.camera_id));
            observed.emplace(image, std::vector<uint8_t>(size_t(range.count), 0));
            if (!targets.count(image)) out.fixedImages.insert(image); out.model.images.emplace(image, std::move(im));
        };
        for (const auto& p : points) {
            const auto meta = validateKey(p);
            const size_t first = selectedAt;
            while (selectedAt < selected.size() && selected[selectedAt].point == p) ++selectedAt;
            const size_t count = selectedAt - first;
            resident = size_t(storage::addBytes(resident, 320 + storage::scaleBytes(includeBoundary ? meta.count : count, sizeof(TrackElement)))); checkBudget(resident);
            Point3D item; item.xyz = pointXYZ(p); item.error = meta.error; std::copy(std::begin(meta.rgb), std::end(meta.rgb), item.rgb);
            const uint64_t id = out.model.next_point3D_id++;
            item.track.reserve(size_t(includeBoundary ? meta.count : count));
            auto append = [&](const RawObservation& row) {
                if (!targets.count(row.image)) out.boundaryPoints.insert(id);
                if (!includeBoundary && !targets.count(row.image)) return;
                addImage(row.image, size_t(p.source)); auto& image = out.model.images.at(row.image);
                if (row.feature >= image.points2D.size() || (observed.at(row.image)[row.feature] &&
                    (image.points2D[row.feature].x != row.xy.x || image.points2D[row.feature].y != row.xy.y)))
                    throw std::runtime_error("global feature identity conflicts between sources: source " + std::to_string(p.source) +
                        ", image " + std::to_string(row.image) + ", local feature " + std::to_string(row.feature) +
                        ", XY " + std::to_string(row.xy.x) + "," + std::to_string(row.xy.y));
                observed.at(row.image)[row.feature] = 1; image.points2D[row.feature] = row.xy;
                image.point3D_ids[row.feature] = id; item.track.push_back({row.image, row.feature});
            };
            if (includeBoundary) visitPoint(p, append);
            else {
                if (count > meta.count) throw std::runtime_error("working photo index repeats an observation");
                if (count < meta.count) out.boundaryPoints.insert(id);
                for (size_t i = first; i < selectedAt; ++i) append(selected[i]);
            }
            out.model.points3D.emplace(id, std::move(item)); out.pointKeys.emplace(id, p);
        }
        return out;
    }

    void beginGeneration() {
        requireWritable();
        if (editing_) throw std::logic_error("working parameter transaction already active");
        auto guard = std::make_unique<FileGuard>(root_ / "transaction.lock");
        requireCurrentHead();
        checkBudget(metadataResident() + modelResidentBytes(poses_));
        pending_ = parameterDir(generation_ + 1);
        if (std::filesystem::exists(pending_)) throw std::runtime_error("uncommitted parameter generation exists");
        uint64_t bytes = 0; for (const auto& source : sources_) bytes = storage::addBytes(bytes, storage::scaleBytes(source.points, sizeof(Vec3)));
        storage::requireSpace(root_, bytes); std::filesystem::create_directory(pending_);
        for (size_t i = 0; i < sources_.size(); ++i) std::filesystem::copy_file(parameterDir(generation_) / xyzName(i), pending_ / xyzName(i));
        savedPoses_ = poses_; editing_ = true; transaction_ = std::move(guard);
    }
    void setPointXYZ(const PointKey& key, const Vec3& xyz) {
        requireEditing(); validateKey(key); if (!finite(xyz)) throw std::runtime_error("nonfinite working point update");
        std::fstream file(pending_ / xyzName(key.source), std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(std::streamoff(storage::scaleBytes(key.ordinal, sizeof(Vec3)))); file.write(reinterpret_cast<const char*>(&xyz), sizeof(xyz));
        file.flush(); if (!file) throw std::runtime_error("cannot write working point parameter"); counters_.parameter_bytes += sizeof(xyz); ++counters_.seeks;
        readers_.erase(pending_ / xyzName(key.source));
    }
    void applyPointUpdates(const std::vector<std::pair<PointKey, Vec3>>& updates) {
        requireEditing(); checkBudget(metadataResident() + storage::scaleBytes(updates.size(), sizeof(size_t)));
        std::vector<size_t> order(updates.size()); for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return updates[a].first < updates[b].first || (!(updates[b].first < updates[a].first) && a < b);
        });
        for (auto item : order) { validateKey(updates[item].first); if (!finite(updates[item].second)) throw std::runtime_error("nonfinite working parameter batch"); }
        size_t first = 0;
        while (first < order.size()) {
            const auto source = updates[order[first]].first.source;
            const Path path = pending_ / xyzName(source); readers_.erase(path);
            std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
            if (!file) throw std::runtime_error("cannot open working parameter batch");
            uint64_t expected = UINT64_MAX;
            do {
                const auto& update = updates[order[first++]];
                if (update.first.ordinal != expected) { file.seekp(std::streamoff(storage::scaleBytes(update.first.ordinal, sizeof(Vec3)))); ++counters_.seeks; }
                file.write(reinterpret_cast<const char*>(&update.second), sizeof(Vec3)); counters_.parameter_bytes += sizeof(Vec3); expected = update.first.ordinal + 1;
            } while (first < order.size() && updates[order[first]].first.source == source);
            file.flush(); if (!file) throw std::runtime_error("cannot flush working parameter batch");
        }
    }
    void setPose(uint32_t image, const Pose& pose) { requireEditing(); if (!finite(pose.t)) throw std::runtime_error("nonfinite working pose"); for (double x : pose.R) if (!std::isfinite(x)) throw std::runtime_error("nonfinite working rotation"); poses_.images.at(image).pose = pose; }
    void setCamera(const Camera& camera) {
        requireEditing();
        for (double x : {camera.fx, camera.fy, camera.cx, camera.cy, camera.k1, camera.k2, camera.p1, camera.p2, camera.k3, camera.k4, camera.k5, camera.k6, camera.sx1, camera.sy1, camera.pixel_scale})
            if (!std::isfinite(x)) throw std::runtime_error("nonfinite working camera parameter");
        poses_.cameras.at(camera.id) = camera;
    }
    void commitGeneration(const std::string& phase = {}) {
        requireEditing(); savePoses(pending_); ++generation_; editing_ = false;
        const auto previousPhase = phase_; if (!phase.empty()) phase_ = phase;
        try { saveManifest(); } catch (...) { --generation_; phase_ = previousPhase; editing_ = true; throw; }
        savedPoses_ = {}; pending_.clear(); readers_.clear(); transaction_.reset();
    }
    Path abortGeneration() {
        requireEditing();
        if (std::filesystem::canonical(pending_).parent_path() != std::filesystem::canonical(root_))
            throw std::runtime_error("rejected parameter generation escaped working root");
        savePoses(pending_); readers_.clear();
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count(); Path retained;
        for (size_t trial = 0; trial < 100; ++trial) {
            const Path kept = root_ / ("rejected-parameters-" + std::to_string(generation_ + 1) + "-" + std::to_string(tick) + "-" + std::to_string(trial));
            if (std::filesystem::exists(kept)) continue; std::filesystem::rename(pending_, kept); retained = kept; break;
        }
        if (retained.empty()) throw std::runtime_error("cannot retain rejected parameter generation");
        poses_ = std::move(savedPoses_); editing_ = false; pending_.clear(); transaction_.reset();
        return retained;
    }
    void interpolateGeneration(const Path& candidatePath, double alpha) {
        requireEditing(); step::validateAlpha(alpha);
        if (alpha < 1 && !savedPoses_.rigs.empty())
            throw std::runtime_error("regional backtracking requires rig-frame interpolation");
        if (alpha < 1) for (size_t source = 0; source < sources_.size(); ++source) {
            const Path viewsPath = sourceDir(source) / "views.bin";
            const uint64_t bytes = std::filesystem::file_size(viewsPath);
            checkBudget(storage::addBytes(metadataResident(), storage::scaleBytes(bytes, 10)));
            if (hash(viewsPath) != sources_[source].hashes.at(0)) throw std::runtime_error("regional step source view identity changed");
            const auto views = sourceViews(source);
            checkBudget(storage::addBytes(metadataResident(), modelResidentBytes(views)));
            if (!views.rigs.empty()) throw std::runtime_error("regional backtracking requires rig-frame interpolation");
        }
        const Path candidate = std::filesystem::canonical(candidatePath), root = std::filesystem::canonical(root_);
        const std::string prefix = "rejected-parameters-" + std::to_string(generation_ + 1) + "-";
        if (candidate.parent_path() != root || candidate.filename().u8string().rfind(prefix, 0) != 0)
            throw std::runtime_error("regional step candidate is outside the retained working generation");
        std::set<std::string> expected{"poses.bin"};
        for (size_t source = 0; source < sources_.size(); ++source) expected.insert(xyzName(source));
        for (const auto& file : std::filesystem::directory_iterator(candidate)) {
            if (!file.is_regular_file() || std::filesystem::is_symlink(file.symlink_status()) ||
                std::filesystem::canonical(file.path()).parent_path() != candidate || !expected.erase(file.path().filename().u8string()))
                throw std::runtime_error("regional step candidate contains an unexpected parameter file");
        }
        if (!expected.empty()) throw std::runtime_error("regional step candidate parameter file missing");
        const uint64_t poseBytes = std::filesystem::file_size(candidate / "poses.bin");
        checkBudget(storage::addBytes(metadataResident(), storage::addBytes(storage::scaleBytes(poseBytes, 10), 1u << 20)));
        auto trial = model_store_detail::readModel(candidate / "poses.bin");
        checkBudget(storage::addBytes(metadataResident(), storage::addBytes(storage::scaleBytes(modelResidentBytes(trial), 2), 1u << 20)));
        const auto& accepted = savedPoses_;
        if (trial.cameras.size() != accepted.cameras.size() || trial.images.size() != accepted.images.size() || !trial.points3D.empty())
            throw std::runtime_error("regional step candidate pose catalogue changed");
        auto normalized = trial;
        for (auto& item : normalized.cameras) {
            const auto found = accepted.cameras.find(item.first);
            if (found == accepted.cameras.end()) throw std::runtime_error("regional step candidate camera ID changed");
            step::interpolateCamera(found->second, item.second, alpha); item.second = found->second;
        }
        for (auto& item : normalized.images) {
            const auto found = accepted.images.find(item.first);
            if (found == accepted.images.end()) throw std::runtime_error("regional step candidate image ID changed");
            step::interpolatePose(found->second.pose, item.second.pose, alpha); item.second.pose = found->second.pose;
        }
        DigestOutput oldMetadata, newMetadata;
        model_store_detail::serializeModel(accepted, oldMetadata); model_store_detail::serializeModel(normalized, newMetadata);
        if (oldMetadata.finish() != newMetadata.finish()) throw std::runtime_error("regional step candidate immutable pose metadata changed");
        normalized = {};
        for (size_t source = 0; source < sources_.size(); ++source) {
            const uint64_t bytes = storage::scaleBytes(sources_[source].points, sizeof(Vec3));
            if (std::filesystem::file_size(candidate / xyzName(source)) != bytes ||
                std::filesystem::file_size(parameterDir(generation_) / xyzName(source)) != bytes)
                throw std::runtime_error("regional step point parameter count changed");
            Reader before(parameterDir(generation_) / xyzName(source), 64u << 10), next(candidate / xyzName(source), 64u << 10);
            for (uint64_t ordinal = 0; ordinal < sources_[source].points; ++ordinal) {
                if (ordinal % 65536 == 0) cancel::check();
                if (!finite(before.scalar<Vec3>()) || !finite(next.scalar<Vec3>())) throw std::runtime_error("nonfinite regional step point parameter");
                counters_.parameter_bytes += 2 * sizeof(Vec3);
            }
            before.finish(); next.finish();
        }
        for (size_t source = 0; source < sources_.size(); ++source) rewriteSourceXYZ(source, [&](auto emit) {
            Reader before(parameterDir(generation_) / xyzName(source), 64u << 10), next(candidate / xyzName(source), 64u << 10);
            for (uint64_t ordinal = 0; ordinal < sources_[source].points; ++ordinal) {
                if (ordinal % 65536 == 0) cancel::check();
                const Vec3 a = before.scalar<Vec3>(), b = next.scalar<Vec3>();
                emit(alpha == 1 ? b : a * (1 - alpha) + b * alpha); counters_.parameter_bytes += 2 * sizeof(Vec3);
            }
            before.finish(); next.finish();
        });
        for (auto& item : trial.images) item.second.pose = step::interpolatePose(accepted.images.at(item.first).pose, item.second.pose, alpha);
        for (auto& item : trial.cameras) item.second = step::interpolateCamera(accepted.cameras.at(item.first), item.second, alpha);
        poses_ = std::move(trial);
    }
    Reconstruction materialize(size_t source, const XYZResolver& resolver = {}, bool originalOnly = false) const {
        Reconstruction out = model_store_detail::readModel(sourceDir(source) / "views.bin");
        uint64_t resident = modelResidentBytes(out), featureAt = 0; Reader in(sourceDir(source) / "features.bin", 64u << 10);
        for (auto& pair : out.images) {
            const auto& range = sources_.at(source).images.at(pair.first);
            if (!originalOnly) {
                const auto& canonical = poses_.images.at(pair.first);
                pair.second.pose = canonical.pose; pair.second.registered = canonical.registered; out.cameras.at(pair.second.camera_id) = poses_.cameras.at(pair.second.camera_id);
            }
            if (featureAt != range.features) throw std::runtime_error("working feature ranges changed order");
            checkBudget(metadataResident() + resident + storage::scaleBytes(range.count, sizeof(Vec2) + sizeof(uint64_t)));
            pair.second.points2D.resize(size_t(range.count)); pair.second.point3D_ids.resize(size_t(range.count));
            for (uint64_t i = 0; i < range.count; ++i) { auto row = in.scalar<FeatureRecord>(); pair.second.points2D[size_t(i)] = row.xy; pair.second.point3D_ids[size_t(i)] = row.assignment; counters_.raw_bytes += sizeof(row); }
            resident = storage::addBytes(resident, model_memory_detail::vectorBytes(pair.second.points2D) + model_memory_detail::vectorBytes(pair.second.point3D_ids)); featureAt += range.count;
        }
        in.finish();
        visitSourcePointsDetailed(source, [&](PointKey key, const PointRecord& meta, Vec3 xyz, const std::vector<RawObservation>& track) {
            const uint64_t entry = sizeof(std::pair<const uint64_t, Point3D>) + 64 + storage::scaleBytes(track.size(), sizeof(TrackElement));
            checkBudget(metadataResident() + resident + entry + storage::scaleBytes(track.capacity(), sizeof(RawObservation)));
            if (resolver) xyz = resolver(key, xyz);
            if (!finite(xyz)) throw std::runtime_error("nonfinite resolved working materialization point");
            Point3D point; point.xyz = xyz; point.error = meta.error; std::copy(std::begin(meta.rgb), std::end(meta.rgb), point.rgb);
            point.track.reserve(track.size()); for (const auto& row : track) point.track.push_back({row.image, row.feature});
            out.points3D.emplace_hint(out.points3D.end(), key.localPoint, std::move(point)); resident = storage::addBytes(resident, entry);
        }, originalOnly);
        checkBudget(metadataResident() + modelResidentBytes(out));
        return out;
    }
    void verifyImmutable() const {
        for (size_t i = 0; i < sources_.size(); ++i) {
            const auto& source = sources_[i]; const Path path = std::filesystem::u8path(source.path);
            if (source.bytes != std::filesystem::file_size(path) || source.mtime != stamp(path)) throw std::runtime_error("working original source identity mismatch");
            verifyDependency(path, source.digest, source.bytes);
            size_t index = 0; for (const char* file : immutableFiles()) verifyDependency(sourceDir(i) / file, source.hashes.at(index++));
            rememberPhotoDependencies(i);
        }
        checkBudget(metadataResident());
    }
private:
    static constexpr uint64_t photoMagic = 0x315448504f575253ull;
    struct FileState {
        uint64_t bytes = 0; int64_t time = 0;
        bool operator==(const FileState& b) const { return bytes == b.bytes && time == b.time; }
    };
    struct PhotoDependencies { FileState raw, index; };
    struct PhotoCache { Path path; FileState data, marker; };
    static FileState fileState(const Path& path) { return {std::filesystem::file_size(path), stamp(path)}; }
    void rememberPhotoDependencies(size_t source) const {
        photo_dependencies_[source] = {fileState(sourceDir(source) / "observations.bin"), fileState(sourceDir(source) / "photo-index.bin")};
    }
    void checkPhotoDependencies(size_t source) const {
        const auto found = photo_dependencies_.find(source);
        if (found == photo_dependencies_.end() || !(found->second.raw == fileState(sourceDir(source) / "observations.bin")) ||
            !(found->second.index == fileState(sourceDir(source) / "photo-index.bin")))
            throw std::runtime_error("immutable photo cache input changed; candidate retained");
    }
    static bool photoRowLess(const RawObservation& a, const RawObservation& b) { return std::tie(a.image, a.ordinal) < std::tie(b.image, b.ordinal); }
    void buildPhotoObservationCache(size_t source, const Path& dir) const {
        const auto started = std::chrono::steady_clock::now(); const auto& src = sources_.at(source);
        const uint64_t bytes = storage::scaleBytes(src.observations, sizeof(RawObservation));
        storage::requireSpace(root_, storage::addBytes(storage::scaleBytes(bytes, 3), 4u << 20));
        const size_t chunkLimit = std::max<size_t>(1, size_t(std::min<uint64_t>(src.observations, budget_ / 4 / sizeof(RawObservation))));
        checkBudget(storage::addBytes(residentBytes(), storage::addBytes(storage::scaleBytes(chunkLimit, sizeof(RawObservation)), 6u << 20)));
        std::filesystem::create_directories(dir);
        std::vector<RawObservation> chunk; chunk.reserve(chunkLimit); size_t runs = 0;
        auto flush = [&] {
            if (chunk.empty()) return;
            std::sort(chunk.begin(), chunk.end(), photoRowLess); Writer out(dir / ("run-" + std::to_string(runs++) + ".bin"));
            for (const auto& row : chunk) out.scalar(row); out.finish(); counters_.photo_cache_written_bytes += chunk.size() * sizeof(RawObservation); chunk.clear();
        };
        {
            Reader points(sourceDir(source) / "points.bin", 64u << 10), raw(sourceDir(source) / "observations.bin", 64u << 10);
            uint64_t first = 0;
            for (uint64_t ordinal = 0; ordinal < src.points; ++ordinal) {
                const auto meta = points.scalar<PointRecord>(); counters_.index_bytes += sizeof(meta);
                if (meta.first != first || meta.count > src.observations - first) throw std::runtime_error("photo cache raw point range corrupted");
                for (uint64_t i = 0; i < meta.count; ++i) {
                    auto row = raw.scalar<RawObservation>(); counters_.photo_cache_input_bytes += sizeof(row);
                    const auto image = src.images.find(row.image);
                    if (!(row.point == PointKey{source, ordinal, meta.local}) || row.ordinal != first + i || image == src.images.end() || row.feature >= image->second.count)
                        throw std::runtime_error("photo cache raw lineage corrupted");
                    chunk.push_back(row); if (chunk.size() == chunkLimit) flush();
                }
                first += meta.count;
            }
            if (first != src.observations) throw std::runtime_error("photo cache raw coverage incomplete");
            points.finish(); raw.finish();
        }
        flush(); std::vector<RawObservation>().swap(chunk);
        size_t firstRun = 0, nextRun = runs;
        while (runs > 1) {
            size_t made = 0;
            for (size_t first = 0; first < runs; first += 32) {
                const size_t end = std::min(runs, first + 32);
                struct Head { RawObservation row; size_t reader; bool operator<(const Head& b) const { return photoRowLess(b.row, row); } };
                std::vector<std::unique_ptr<Reader>> inputs; std::vector<uint64_t> remaining; std::priority_queue<Head> heads;
                for (size_t run = first; run < end; ++run) {
                    const Path path = dir / ("run-" + std::to_string(firstRun + run) + ".bin"); const auto size = std::filesystem::file_size(path);
                    if (size % sizeof(RawObservation)) throw std::runtime_error("truncated photo-major sort run retained");
                    inputs.emplace_back(std::make_unique<Reader>(path, 64u << 10)); remaining.push_back(size / sizeof(RawObservation));
                    if (remaining.back()) { --remaining.back(); heads.push({inputs.back()->scalar<RawObservation>(), inputs.size() - 1}); }
                }
                Writer out(dir / ("run-" + std::to_string(nextRun + made++) + ".bin"));
                while (!heads.empty()) {
                    const auto head = heads.top(); heads.pop(); out.scalar(head.row); counters_.photo_cache_written_bytes += sizeof(RawObservation);
                    if (remaining[head.reader]) { --remaining[head.reader]; heads.push({inputs[head.reader]->scalar<RawObservation>(), head.reader}); }
                }
                out.finish(); for (auto& input : inputs) input->finish(); inputs.clear();
                for (size_t run = first; run < end; ++run) std::filesystem::remove(dir / ("run-" + std::to_string(firstRun + run) + ".bin"));
            }
            firstRun = nextRun; nextRun += made; runs = made;
        }
        const Path data = dir / "observations.bin";
        if (runs) std::filesystem::rename(dir / ("run-" + std::to_string(firstRun) + ".bin"), data);
        else { Writer empty(data); empty.finish(); }
        {
            Reader rows(data, 64u << 10), index(sourceDir(source) / "photo-index.bin", 64u << 10);
            uint64_t photoOffset = 0;
            for (const auto& image : src.images) if (image.second.photoCount) {
                if (image.second.photoFirst != photoOffset || image.second.photoCount > src.observations - photoOffset)
                    throw std::runtime_error("original photo index range incomplete; derived candidate retained");
                photoOffset += image.second.photoCount;
            }
            if (photoOffset != src.observations) throw std::runtime_error("original photo index coverage incomplete; derived candidate retained");
            RawObservation previous{};
            for (uint64_t i = 0; i < src.observations; ++i) {
                const auto row = rows.scalar<RawObservation>(); const auto ref = index.scalar<PhotoRef>(); counters_.index_bytes += sizeof(ref);
                const auto image = src.images.find(row.image);
                if (row.image != ref.image || row.ordinal != ref.observation || (i && !photoRowLess(previous, row)) || image == src.images.end() ||
                    i < image->second.photoFirst || i - image->second.photoFirst >= image->second.photoCount)
                    throw std::runtime_error("photo-major candidate differs from original photo index; candidate retained");
                previous = row;
            }
            rows.finish(); index.finish();
        }
        checkPhotoDependencies(source); const auto digest = hash(data);
        Writer marker(dir / "ready.part"); marker.scalar(photoMagic); marker.scalar<uint32_t>(1); marker.scalar<uint64_t>(source);
        marker.scalar<uint64_t>(sizeof(RawObservation)); marker.scalar(src.observations); marker.string(src.hashes.at(3)); marker.string(src.hashes.at(4)); marker.string(digest); marker.finish();
        storage::replaceFile(dir / "ready.part", dir / "ready.bin"); ++counters_.photo_cache_builds;
        counters_.photo_cache_build_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }
    struct DigestOutput {
        spirula::Sha256 sha;
        template<class T> void scalar(const T& value) { sha.update(reinterpret_cast<const uint8_t*>(&value), sizeof(T)); }
        template<class T> void vector(const std::vector<T>& value) { scalar(uint64_t(value.size())); sha.update(reinterpret_cast<const uint8_t*>(value.data()), value.size() * sizeof(T)); }
        void string(const std::string& value) { scalar(uint64_t(value.size())); sha.update(reinterpret_cast<const uint8_t*>(value.data()), value.size()); }
        std::string finish() { return sha.hex(); }
    };
    static std::vector<const char*> immutableFiles() { return {"views.bin", "features.bin", "points.bin", "observations.bin", "photo-index.bin", "initial.xyz"}; }
    static int64_t stamp(const Path& path) { return int64_t(std::filesystem::last_write_time(path).time_since_epoch().count()); }
    void verifyDependency(const Path& path, const std::string& digest, uint64_t bytes = UINT64_MAX) const {
        const auto before = verified_.statistics();
        auto account = [&] {
            const auto& after = verified_.statistics();
            counters_.bytesHashChecked += after.bytesHashChecked - before.bytesHashChecked;
            counters_.leaseHits += after.leaseHits - before.leaseHits; counters_.leaseMisses += after.leaseMisses - before.leaseMisses;
            counters_.leasedFiles = verified_.leasedFiles();
        };
        try { verified_.verify(path, digest, bytes); } catch (...) { account(); throw; }
        account();
    }
    static std::string hash(const Path& path) {
        std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("cannot hash working store dependency");
        spirula::Sha256 sha; std::vector<uint8_t> buffer(64u << 10);
        while (in) { in.read(reinterpret_cast<char*>(buffer.data()), std::streamsize(buffer.size())); sha.update(buffer.data(), size_t(in.gcount())); }
        if (!in.eof()) throw std::runtime_error("working dependency hash read failed"); return sha.hex();
    }
    static bool finite(Vec3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
    Path sourceDir(size_t source) const { return root_ / ("source-" + std::to_string(source)); }
    Path parameterDir(uint64_t generation) const { return root_ / ("parameters-" + std::to_string(generation)); }
    static std::string xyzName(size_t source) { return "points-" + std::to_string(source) + ".xyz"; }
    Path activeParameters() const { return editing_ ? pending_ : parameterDir(generation_); }
    void requireWritable() const { if (read_only_) throw std::logic_error("read-only working store rejects mutation"); }
    void requireEditing() const { requireWritable(); if (!editing_) throw std::logic_error("working parameter update requires a transaction"); }
    void requireCurrentHead() const {
        if (read_only_ && hash(root_ / "manifest.bin") != readonly_manifest_hash_)
            throw std::runtime_error("read-only working manifest changed");
        if (!std::filesystem::exists(root_ / "manifest.bin")) { if (!sources_.empty()) throw std::runtime_error("accepted working manifest missing"); return; }
        Reader in(root_ / "manifest.bin", 64u << 10);
        if (in.scalar<uint64_t>() != magic) throw std::runtime_error("accepted working generation identity changed");
        const auto version = in.scalar<uint32_t>();
        if ((version != 2 && version != 3) || in.scalar<uint64_t>() != generation_)
            throw std::runtime_error("accepted working generation changed; reopen store before writing");
    }
    size_t metadataBytes(const Source& source) const { return source.images.size() * (sizeof(ImageRange) + 96) + source.path.capacity() + source.label.capacity() + 1024; }
    size_t measureMetadata() const { size_t result = modelResidentBytes(poses_); for (const auto& source : sources_) result += metadataBytes(source); return result; }
    size_t metadataResident() const { return size_t(storage::addBytes(metadata_bytes_, verified_.residentBytes())); }
    void checkBudget(uint64_t bytes) const { if (bytes > budget_ - (6u << 20)) throw std::length_error("working store active set exceeds memory budget"); }
    template<class T> T readAt(const Path& path, uint64_t ordinal, uint64_t& bytes) const {
        auto held = cachedReader(path); held->seek(storage::scaleBytes(ordinal, sizeof(T))); ++counters_.seeks;
        auto result = held->scalar<T>(); bytes += sizeof(T); return result;
    }
    std::shared_ptr<Reader> cachedReader(const Path& path) const {
        auto found = readers_.find(path);
        if (found != readers_.end()) { found->second.used = ++reader_tick_; ++counters_.reader_hits; return found->second.reader; }
        auto reader = std::make_shared<Reader>(path, 64u << 10); ++counters_.reader_opens;
        if (readers_.size() == 8) {
            const auto oldest = std::min_element(readers_.begin(), readers_.end(), [](const auto& a, const auto& b) { return a.second.used < b.second.used; });
            readers_.erase(oldest); ++counters_.reader_evictions;
        }
        readers_.emplace(path, CachedReader{reader, ++reader_tick_}); return reader;
    }
    PointRecord pointRecord(size_t source, uint64_t ordinal) const {
        if (ordinal >= sources_.at(source).points) throw std::out_of_range("working point ordinal");
        return readAt<PointRecord>(sourceDir(source) / "points.bin", ordinal, counters_.index_bytes);
    }
    PointRecord validateKey(const PointKey& key) const { auto p = pointRecord(size_t(key.source), key.ordinal); if (p.local != key.localPoint) throw std::runtime_error("working point lineage identity mismatch"); return p; }
    void savePoses(const Path& dir) const { model_store_detail::writeModel(poses_, dir / "poses.bin"); }
    void saveManifest() const {
        requireWritable();
        Writer out(root_ / "manifest.part"); out.scalar(magic); out.scalar<uint32_t>(3); out.scalar(generation_); out.string(phase_); out.scalar<uint64_t>(sources_.size());
        for (const auto& source : sources_) {
            out.string(source.path); out.string(source.label); out.string(source.digest); out.scalar(source.bytes); out.scalar(source.mtime); out.scalar(source.points); out.scalar(source.observations);
            out.scalar<uint64_t>(source.images.size()); for (const auto& pair : source.images) { out.scalar(pair.first); out.scalar(pair.second); }
            for (const auto& digest : source.hashes) out.string(digest);
        }
        out.string(hash(parameterDir(generation_) / "poses.bin"));
        for (size_t i = 0; i < sources_.size(); ++i) out.string(hash(parameterDir(generation_) / xyzName(i)));
        out.finish(); storage::replaceFile(root_ / "manifest.part", root_ / "manifest.bin");
    }
    void restore() {
        std::unique_ptr<FileGuard> guard;
        if (read_only_) { read_lease_ = std::make_unique<FileGuard>(root_ / "transaction.lock", true); readonly_manifest_hash_ = hash(root_ / "manifest.bin"); }
        else guard = std::make_unique<FileGuard>(root_ / "transaction.lock");
        if (std::filesystem::file_size(root_ / "manifest.bin") > budget_ / 4) throw std::length_error("working manifest exceeds memory budget");
        Reader in(root_ / "manifest.bin"); if (in.scalar<uint64_t>() != magic) throw std::runtime_error("unsupported working store identity");
        const auto version = in.scalar<uint32_t>(); if (version != 2 && version != 3) throw std::runtime_error("unsupported working store version");
        generation_ = in.scalar<uint64_t>(); if (version == 3) phase_ = in.string();
        const size_t count = in.count(64); sources_.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            Source source; source.path = in.string(); source.label = in.string(); source.digest = in.string(); source.bytes = in.scalar<uint64_t>(); source.mtime = in.scalar<int64_t>(); source.points = in.scalar<uint64_t>(); source.observations = in.scalar<uint64_t>();
            if (source.label.empty() || findSourceLabel(source.label) != SIZE_MAX) throw std::runtime_error("duplicate working source label");
            size_t n = in.count(sizeof(uint32_t) + sizeof(ImageRange));
            for (size_t j = 0; j < n; ++j) { auto image = in.scalar<uint32_t>(); auto range = in.scalar<ImageRange>(); if (!source.images.emplace(image, range).second) throw std::runtime_error("duplicate working photo index"); }
            for (size_t j = 0; j < immutableFiles().size(); ++j) source.hashes.push_back(in.string()); sources_.push_back(std::move(source));
            metadata_bytes_ = size_t(storage::addBytes(metadata_bytes_, metadataBytes(sources_.back()))); checkBudget(metadataResident());
        }
        if (in.string() != hash(parameterDir(generation_) / "poses.bin")) throw std::runtime_error("working pose checkpoint checksum mismatch");
        for (size_t i = 0; i < sources_.size(); ++i) if (in.string() != hash(parameterDir(generation_) / xyzName(i))) throw std::runtime_error("working point checkpoint checksum mismatch");
        in.finish(); verifyImmutable();
        if (std::filesystem::file_size(parameterDir(generation_) / "poses.bin") > budget_ / 4) throw std::length_error("working pose table exceeds memory budget");
        poses_ = model_store_detail::readModel(parameterDir(generation_) / "poses.bin"); metadata_bytes_ = measureMetadata(); checkBudget(metadataResident());
        if (read_only_) requireCurrentHead();
        else { preserveInterrupted(parameterDir(generation_ + 1)); preserveInterrupted(sourceDir(sources_.size())); }
    }
    void preserveInterrupted(const Path& path) const {
        if (!std::filesystem::exists(path)) return;
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        for (size_t trial = 0; trial < 100; ++trial) {
            const Path kept = root_ / ("interrupted-" + path.filename().u8string() + "-" + std::to_string(tick) + "-" + std::to_string(trial));
            if (std::filesystem::exists(kept)) continue; std::filesystem::rename(path, kept); return;
        }
        throw std::runtime_error("cannot preserve interrupted working generation");
    }
    void makePhotoIndex(const Path& dir, size_t runs, Source& source) const {
        size_t first = 0, next = runs;
        while (runs > 1) {
            size_t made = 0;
            for (size_t start = 0; start < runs; start += 16) {
                size_t end = std::min(runs, start + 16);
                struct Head { PhotoRef ref; size_t stream; bool operator<(const Head& b) const { return b.ref < ref; } };
                std::vector<std::ifstream> streams(end - start); std::priority_queue<Head> queue;
                auto get = [](std::ifstream& in, PhotoRef& ref) { in.read(reinterpret_cast<char*>(&ref), sizeof(ref)); if (in.gcount() == sizeof(ref)) return true; if (in.gcount() || !in.eof()) throw std::runtime_error("truncated working photo sort run"); return false; };
                for (size_t j = start; j < end; ++j) { auto& in = streams[j - start]; in.open(dir / ("run-" + std::to_string(first + j) + ".bin"), std::ios::binary); if (!in) throw std::runtime_error("cannot open working photo sort run"); PhotoRef ref; if (get(in, ref)) queue.push({ref, j - start}); }
                Writer out(dir / ("run-" + std::to_string(next + made++) + ".bin"));
                while (!queue.empty()) { auto h = queue.top(); queue.pop(); out.scalar(h.ref); if (get(streams[h.stream], h.ref)) queue.push(h); } out.finish();
                streams.clear(); for (size_t j = start; j < end; ++j) std::filesystem::remove(dir / ("run-" + std::to_string(first + j) + ".bin"));
            }
            first = next; next += made; runs = made;
        }
        const Path path = dir / "photo-index.bin";
        if (runs) std::filesystem::rename(dir / ("run-" + std::to_string(first) + ".bin"), path);
        else { Writer empty(path); empty.finish(); }
        Reader in(path, 64u << 10);
        for (uint64_t i = 0; i < source.observations; ++i) { auto ref = in.scalar<PhotoRef>(); auto& range = source.images.at(ref.image); if (!range.photoCount) range.photoFirst = i; ++range.photoCount; } in.finish();
    }
    Path root_, pending_;
    size_t budget_;
    size_t metadata_bytes_ = sizeof(Reconstruction);
    uint64_t generation_ = 0;
    std::string phase_;
    bool editing_ = false, read_only_ = false;
    std::string readonly_manifest_hash_;
    Reconstruction poses_, savedPoses_;
    std::vector<Source> sources_;
    mutable IoCounters counters_;
    struct CachedReader { std::shared_ptr<Reader> reader; uint64_t used = 0; };
    mutable std::map<Path, CachedReader> readers_;
    mutable std::map<size_t, PhotoDependencies> photo_dependencies_;
    mutable std::map<size_t, PhotoCache> photo_caches_;
    mutable uint64_t reader_tick_ = 0;
    std::unique_ptr<FileGuard> transaction_, read_lease_;
    mutable VerifiedImmutableFiles verified_;
};

} // namespace sfm::regional
