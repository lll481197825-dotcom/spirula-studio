#pragma once

#include "sfm/map/RegionalWorkingStore.h"
#include "sfm/map/RegionalTopologyStorage.h"
#include "sfm/map/RegionalStore.h"
#include "sfm/ba/CpuParallel.h"
#include "sfm/core/Events.h"
#include "sfm/core/Log.h"
#include "sfm/core/Resume.h"
#include <sstream>

namespace sfm::regional {

class RegionalTopology {
    using Key = RegionalWorkingStore::PointKey;
    using Raw = RegionalWorkingStore::RawObservation;
    struct Group { uint64_t root = 0, first = 0, count = 0, weight = 0; Vec3 xyz; uint64_t file = 0; };
    struct File { std::string path, hash; uint64_t bytes = 0; };
    struct Parent { std::string identity, parameterHash; uint64_t groups = 0; };
    struct Moment { uint64_t point = 0, weight = 0; Vec3 xyz; };
public:
    struct Counters { uint64_t new_sources = 0, old_photo_rows = 0, old_private_points = 0, reused_groups = 0, delta_rows = 0, parameter_disk_bytes = 0; };
    RegionalTopology(const fs::path& root, size_t budget) : root_(fs::absolute(root)), budget_(budget) {
        if (budget < (8u << 20)) throw std::length_error("regional topology requires 8 MiB");
        fs::create_directories(root_);
    }
    size_t groups() const { return groups_.size(); }
    size_t points() const { return roots_.size(); }
    const std::string& identity() const { return identity_; }
    const std::string& ownerIdentity() const { return ownerHash_; }
    uint64_t canonicalPoint(uint64_t point) const { return roots_.at(size_t(point)); }
    topology_storage::Bits privateAliasRoots() const {
        checkMemory(topology_storage::Bits::allocationBytes(roots_.size())); topology_storage::Bits aliases(roots_.size());
        for (size_t point = 0; point < roots_.size(); ++point) {
            if (point % 1048576 == 0) cancel::check(); const auto root = roots_[point];
            if (root != point && !sharedRoots_.at(size_t(root))) aliases.at(size_t(root)) = true;
        }
        return aliases;
    }
    uint64_t residentBytes() const { return resident(); }
    uint64_t peakCapacityBound() const { return peakCapacityBound_; }
    uint64_t memoryBytes() const { return resident(); }
    uint64_t peakMemoryBytes() const { return peakCapacityBound_; }
    const Counters& counters() const { return counters_; }
    uint64_t canonical(const Key& key) const { return roots_.at(size_t(global(key))); }
    bool shared(const Key& key) const { return group(key) != kInvalidPoint3D; }
    Vec3 xyz(const Key& key, Vec3 privateXYZ) const {
        const auto id = group(key); return id == kInvalidPoint3D ? privateXYZ : groups_.at(id).xyz;
    }
    enum class ParameterRestore { Exact, InitializedAppend };
    void restore() {
        if (!offsets_.empty()) return;
        if (!fs::exists(root_ / "head.bin")) throw std::runtime_error("regional topology has no committed head");
        load();
    }
    void prepare(RegionalWorkingStore& store, const std::vector<uint32_t>& owners) {
        const auto acceptedParameters = lastParametersPath_;
        try { prepareImpl(store, owners); }
        catch (...) {
            const auto failure = std::current_exception(); resetState();
            if (fs::exists(root_ / "head.bin")) load();
            if (!acceptedParameters.empty()) restoreAcceptedParameters(acceptedParameters);
            std::rethrow_exception(failure);
        }
    }
private:
    void prepareImpl(RegionalWorkingStore& store, const std::vector<uint32_t>& owners) {
        counters_ = {};
        std::vector<uint64_t> offsets{0}; std::vector<std::string> sources;
        checkMemory(storage::scaleBytes(store.sourceCount() + 1, sizeof(uint64_t) + 128));
        for (size_t source = 0; source < store.sourceCount(); ++source) {
            offsets.push_back(storage::addBytes(offsets.back(), store.sourcePointCount(source)));
            sources.push_back(store.sourceIdentity(source));
        }
        for (const auto& image : store.poses().images) if (image.first >= owners.size())
            throw std::runtime_error("regional owner catalogue incomplete");
        spirula::Sha256 hash; hash.update(reinterpret_cast<const uint8_t*>(owners.data()), owners.size() * sizeof(uint32_t)); const auto ownerHash = hash.hex();
        if (offsets_.empty() && fs::exists(root_ / "head.bin")) load();
        if (sources_ == sources && ownerHash_ == ownerHash && offsets_ == offsets) { validate(); return; }
        const bool append = ownerHash_ == ownerHash && sources_.size() < sources.size() &&
            std::equal(sources_.begin(), sources_.end(), sources.begin()) &&
            offsets_.size() <= offsets.size() && std::equal(offsets_.begin(), offsets_.end(), offsets.begin());
        const auto started = std::chrono::steady_clock::now();
        auto phaseTime = started;
        auto elapsed = [&] { const auto now = std::chrono::steady_clock::now(); const double seconds = std::chrono::duration<double>(now - phaseTime).count(); phaseTime = now; return seconds; };
        const size_t oldSources = append ? sources_.size() : 0;
        slog::diag(slog::Tag::Map, "[working-topology] %s: %zu immutable old sources, %zu new sources; new-source raw scan begin\n",
            append ? "incremental append" : "cold build", oldSources, store.sourceCount() - oldSources);
        if (append && lastParametersIdentity_ == identity_ && !lastParametersHash_.empty())
            parents_.push_back({identity_, lastParametersHash_, groups_.size()});
        if (!append) parents_.clear();
        const size_t oldPoints = append ? roots_.size() : 0;
        checkMemory(storage::scaleBytes(owners.size(), 5));
        std::vector<uint32_t> uses(owners.size()); std::vector<uint8_t> newlyShared(owners.size());
        for (size_t source = 0; source < store.sourceCount(); ++source) {
            const auto views = store.sourceViews(source);
            for (const auto& image : views.images) if (store.sourcePhotoObservationCount(source, image.first)) {
                if (!image.second.registered || image.first >= owners.size()) throw std::runtime_error("topology source observation lacks visual photo identity");
                ++uses[image.first];
            }
        }
        for (size_t image = 0; image < uses.size(); ++image)
            newlyShared[image] = uses[image] > 1 && (!append || image >= usage_.size() || usage_[image] < 2);
        if (!append) { roots_.clear(); roots_.shrink_to_fit(); sharedRoots_.release(); groups_.release(); files_.clear(); ledgers_.clear(); usage_.clear(); readers_.clear(); }
        const uint64_t total = offsets.back();
        if (total > SIZE_MAX) throw std::length_error("regional topology exceeds address space");
        checkMemory(topology_storage::Indices::allocationBytes(total));
        roots_.resize(size_t(total));
        offsets_ = std::move(offsets);
        checkMemory(topology_storage::Bits::allocationBytes(total) * 2 + topology_storage::Bits::allocationBytes(oldPoints) + owners.size() * 5);
        topology_storage::Bits affected{size_t(total)}, candidate{size_t(total)}, oldShared{oldPoints};
        scratch_ = affected.bytes() + candidate.bytes() + oldShared.bytes() + owners.size() * 5;
        for (size_t i = 0; i < oldPoints; ++i) oldShared[i] = sharedRoots_.at(size_t(roots_[i]));
        for (size_t i = oldPoints; i < roots_.size(); ++i) { roots_[i] = i; affected[i] = 1; }
        const auto stage = root_ / ("build-" + std::to_string(store.sourceCount()) + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directory(stage);
        uint64_t newObservations = 0, newPoints = 0;
        for (size_t source = oldSources; source < store.sourceCount(); ++source) {
            newObservations = storage::addBytes(newObservations, store.sourceObservationCount(source));
            newPoints = storage::addBytes(newPoints, store.sourcePointCount(source));
        }
        storage::requireSpace(stage, storage::addBytes(storage::scaleBytes(newObservations, 6 * sizeof(Observation)),
            storage::addBytes(storage::scaleBytes(newPoints, sizeof(Moment)), storage::scaleBytes(total, 2 * sizeof(uint64_t) + sizeof(Group)))));
        std::set<uint32_t> photos;
        const auto newRows = stage / "new-rows.bin", moments = stage / "moments.bin", featureRows = stage / "feature-delta.bin";
        {
            std::ofstream rawOut(newRows, std::ios::binary), featureOut(featureRows, std::ios::binary), momentOut(moments, std::ios::binary);
            for (size_t source = oldSources; source < store.sourceCount(); ++source) {
                ++counters_.new_sources;
                store.visitSourcePoints(source, [&](Key key, Vec3 xyz, const std::vector<Raw>& track) {
                    cancel::check(); const auto point = global(key);
                    record(momentOut, Moment{point, track.size(), xyz});
                    uint32_t firstOwner = UINT32_MAX;
                    for (const auto& obs : track) {
                        const uint64_t feature = (uint64_t(obs.image) << 32) | obs.feature;
                        if (obs.image >= owners.size()) throw std::runtime_error("regional raw photo owner missing");
                        record(rawOut, Observation{point, feature, obs.xy});
                        const uint32_t owner = owners[obs.image];
                        if (firstOwner == UINT32_MAX) firstOwner = owner; else candidate[size_t(point)] |= firstOwner != owner;
                        if (uses[obs.image] > 1) {
                            record(featureOut, Observation{feature, point, obs.xy});
                            checkMemory(scratch_ + photos.size() * 48 + 48);
                            photos.insert(obs.image); ++counters_.delta_rows;
                        }
                    }
                });
                events::progress(Stage::Merge, source - oldSources + 1, store.sourceCount() - oldSources);
                slog::diag(slog::Tag::Map, "[working-topology] new-source scan %zu/%zu: source %zu, %llu raw observations\n",
                    source - oldSources + 1, store.sourceCount() - oldSources, source, (unsigned long long)store.sourceObservationCount(source));
            }
            slog::diag(slog::Tag::Map, "[working-topology] new-source raw scan %.3fs; old-photo index lookup begin, %zu affected photos\n", elapsed(), photos.size());
            size_t photoDone = 0;
            if (append) for (uint32_t image : photos) {
                cancel::check();
                store.visitImageObservations(image, [&](const Raw& obs) {
                    if (obs.point.source >= oldSources) return;
                    record(featureOut, Observation{(uint64_t(obs.image) << 32) | obs.feature, global(obs.point), obs.xy});
                    ++counters_.old_photo_rows; ++counters_.delta_rows;
                });
                if (++photoDone % 128 == 0 || photoDone == photos.size()) events::progress(Stage::Merge, photoDone, photos.size());
            }
            rawOut.flush(); featureOut.flush(); momentOut.flush();
            if (!rawOut || !featureOut || !momentOut) throw std::runtime_error("cannot save topology delta");
        }
        slog::diag(slog::Tag::Map, "[working-topology] old-photo index lookup %.3fs: %llu old rows; delta feature sort begin, %llu records\n",
            elapsed(), (unsigned long long)counters_.old_photo_rows, (unsigned long long)counters_.delta_rows);
        const auto featureSortBudget = batchBudget(budget_ / 16, 1, 512u << 10);
        checkMemory(scratch_ + featureSortBudget + (512u << 10));
        auto ordered = sortObservations(featureRows, stage / "feature-sort", size_t(featureSortBudget));
        slog::diag(slog::Tag::Map, "[working-topology] delta feature sort %.3fs; union begin\n", elapsed());
        {
            std::ifstream in(ordered, std::ios::binary); Observation row; bool have = nextRecord(in, row);
            while (have) {
                cancel::check(); const auto feature = row.key; std::vector<uint64_t> points; bool added = false;
                do {
                    if (points.empty() || points.back() != row.point) {
                        if (points.size() == points.capacity()) {
                            const size_t cap = std::max<size_t>(8, points.size() * 2);
                            checkMemory(scratch_ + (points.capacity() + cap) * sizeof(uint64_t)); points.reserve(cap);
                        }
                        points.push_back(row.point);
                    }
                    added |= row.point >= oldPoints || newlyShared.at(size_t(row.key >> 32)); have = nextRecord(in, row);
                } while (have && row.key == feature);
                if (!added || points.size() < 2) continue;
                const auto first = find(points.front()); affected.at(size_t(first)) = 1;
                for (size_t i = 1; i < points.size(); ++i) {
                    const auto root = find(points[i]); affected.at(size_t(root)) = 1; join(first, root);
                }
            }
        }
        slog::diag(slog::Tag::Map, "[working-topology] delta union %.3fs; in-memory root compression %zu points\n", elapsed(), roots_.size());
        for (size_t i = 0; i < roots_.size(); ++i) {
            roots_[i] = find(i);
            if ((i + 1) % 1048576 == 0 || i + 1 == roots_.size()) events::progress(Stage::Merge, i + 1, roots_.size());
        }
        for (size_t i = 0; i < affected.size(); ++i) {
            if (affected[i]) affected.at(size_t(roots_[i])) = 1;
            candidate.at(size_t(roots_[i])) |= candidate[i] || roots_[i] != i || (i < oldPoints && oldShared[i]);
        }
        slog::diag(slog::Tag::Map, "[working-topology] root compression %.3fs; affected complete-track gather begin\n", elapsed());
        const auto grouped = stage / "group-delta.bin";
        checkMemory(scratch_ + topology_storage::Bits::allocationBytes(groups_.size()));
        topology_storage::Bits reused(groups_.size());
        scratch_ += reused.bytes();
        {
            std::ofstream out(grouped, std::ios::binary), momentOut(moments, std::ios::binary | std::ios::app);
            std::ifstream in(newRows, std::ios::binary); Observation row;
            while (nextRecord(in, row)) { row.key = roots_.at(size_t(row.key)); if (candidate.at(size_t(row.key))) record(out, row); }
            for (size_t g = 0; g < groups_.size(); ++g) {
                const auto oldRoot = groups_.at(g).root;
                const auto nextRoot = roots_.at(size_t(oldRoot));
                if (!affected.at(size_t(nextRoot))) { reused[g] = 1; ++counters_.reused_groups; continue; }
                readTrack(groups_[g], [&](Observation obs) { obs.key = nextRoot; record(out, obs); });
                record(momentOut, Moment{nextRoot, groups_[g].weight, groups_[g].xyz});
            }
            for (size_t point = 0; point < oldPoints; ++point) {
                if (!affected.at(size_t(roots_[point])) || !candidate.at(size_t(roots_[point])) || oldShared[point]) continue;
                const size_t source = size_t(std::upper_bound(offsets_.begin(), offsets_.end(), point) - offsets_.begin() - 1);
                const auto key = store.key(source, point - offsets_[source]); uint64_t count = 0;
                store.visitPoint(key, [&](const Raw& obs) {
                    record(out, Observation{roots_[point], (uint64_t(obs.image) << 32) | obs.feature, obs.xy}); ++count;
                });
                record(momentOut, Moment{roots_[point], count, store.pointXYZ(key)}); ++counters_.old_private_points;
            }
            out.flush(); momentOut.flush(); if (!out || !momentOut) throw std::runtime_error("cannot save affected topology groups");
        }
        slog::diag(slog::Tag::Map, "[working-topology] complete-track gather %.3fs: %llu old private points, %llu unchanged groups reused; affected group sort begin\n",
            elapsed(), (unsigned long long)counters_.old_private_points, (unsigned long long)counters_.reused_groups);
        affected.release(); candidate.release(); oldShared.release(); photos.clear();
        scratch_ = reused.bytes() + owners.size() * 5;
        const auto groupSortBudget = batchBudget(budget_ / 16, 1, 512u << 10);
        checkMemory(scratch_ + groupSortBudget + (512u << 10));
        auto sorted = sortObservations(grouped, stage / "point-sort", size_t(groupSortBudget));
        scratch_ = reused.bytes() + owners.size() * 5;
        slog::diag(slog::Tag::Map, "[working-topology] affected group sort %.3fs; canonical full-track validation/write begin\n", elapsed());
        const auto reusedMetadata = stage / "groups-reused.bin", affectedMetadata = stage / "groups-affected.bin";
        checkMemory(scratch_ + (64u << 10));
        {
            std::ofstream out(reusedMetadata, std::ios::binary);
            for (size_t g = 0; g < groups_.size(); ++g) if (reused[g]) record(out, groups_[g]);
            out.flush(); if (!out) throw std::runtime_error("cannot save reused topology metadata");
        }
        const size_t fileSlot = files_.size(); uint64_t first = 0;
        checkMemory(scratch_ + topology_storage::Bits::allocationBytes(total) + (128u << 10));
        topology_storage::Bits quarantined{size_t(total)};
        scratch_ += quarantined.bytes() + (128u << 10);
        {
            std::ifstream in(sorted, std::ios::binary); std::ofstream out(stage / "tracks.bin", std::ios::binary);
            std::ofstream metadata(affectedMetadata, std::ios::binary);
            std::ofstream conflicts(stage / "conflicts.csv"); conflicts << "source_point_root,unique_observations,reason\n";
            Observation row; bool have = nextRecord(in, row);
            while (have) {
                cancel::check(); const uint64_t root = row.key;
                std::vector<Observation> track; uint64_t previous = kInvalidPoint3D;
                uint32_t previousImage = UINT32_MAX, firstBlock = UINT32_MAX; bool conflict = false, multipleBlocks = false, xyConflict = false, invalidXY = false;
                Vec2 previousXY{};
                do {
                    invalidXY |= !std::isfinite(row.xy.x) || !std::isfinite(row.xy.y);
                    if (row.point != previous) {
                        const auto image = uint32_t(row.point >> 32);
                        if (image == previousImage) conflict = true;
                        previousImage = image; previous = row.point; previousXY = row.xy;
                        const uint32_t owner = owners.at(image);
                        if (firstBlock == UINT32_MAX) firstBlock = owner; else multipleBlocks |= firstBlock != owner;
                        if (track.size() == track.capacity()) {
                            const size_t cap = std::max<size_t>(8, track.size() * 2);
                            checkMemory(scratch_ + (track.capacity() + cap) * sizeof(Observation));
                            track.reserve(cap);
                        }
                        track.push_back(row);
                    } else if (row.xy.x != previousXY.x || row.xy.y != previousXY.y) xyConflict = true;
                    have = nextRecord(in, row);
                } while (have && row.key == root);
                if (conflict || xyConflict || invalidXY) {
                    conflicts << root << ',' << track.size() << ',' << (invalidXY ? "nonfinite_feature_xy" : xyConflict ? "same_feature_different_xy" : "multiple_features_in_one_photo") << '\n';
                    quarantined.at(size_t(root)) = 1; continue;
                }
                if (!multipleBlocks || track.size() < 2) continue;
                record(metadata, Group{root, first, track.size(), 0, {}, fileSlot});
                for (const auto& obs : track) record(out, obs);
                first += track.size();
            }
            out.flush(); conflicts.flush(); metadata.flush();
            if (!out || !conflicts || !metadata) throw std::runtime_error("cannot save topology tracks, metadata or conflict ledger");
        }
        groups_.release(); sharedRoots_.release(); reused.release();
        scratch_ = quarantined.bytes() + owners.size() * 5 + (128u << 10);
        {
            std::ifstream oldMetadata(reusedMetadata, std::ios::binary), newMetadata(affectedMetadata, std::ios::binary);
            Group oldGroup, newGroup; bool oldHave = nextRecord(oldMetadata, oldGroup), newHave = nextRecord(newMetadata, newGroup);
            uint64_t previous = kInvalidPoint3D;
            while (oldHave || newHave) {
                if (groups_.size() % 65536 == 0) cancel::check();
                const bool takeOld = oldHave && (!newHave || oldGroup.root < newGroup.root); const auto group = takeOld ? oldGroup : newGroup;
                if (previous != kInvalidPoint3D && group.root <= previous) throw std::runtime_error("incremental topology metadata root order or overlap changed");
                if (groups_.size() == groups_.capacity()) {
                    checkMemory(scratch_ + topology_storage::Chunks<Group>::block * (sizeof(Group) + sizeof(void*)));
                    groups_.reserve(groups_.size() + 1);
                }
                groups_.push_back(group); previous = group.root;
                if (takeOld) oldHave = nextRecord(oldMetadata, oldGroup); else newHave = nextRecord(newMetadata, newGroup);
            }
        }
        rebuildSharedRoots();
        {
            std::ifstream in(moments, std::ios::binary); Moment row;
            while (nextRecord(in, row)) {
                const auto root = roots_.at(size_t(row.point)); const auto id = groupForRoot(root);
                if (id == kInvalidPoint3D || !row.weight) continue;
                auto& g = groups_.at(size_t(id));
                if (g.file != fileSlot) continue;
                const uint64_t before = g.weight; g.weight = storage::addBytes(g.weight, row.weight);
                const double fraction = double(row.weight) / g.weight;
                g.xyz = before ? g.xyz * (1 - fraction) + row.xyz * fraction : row.xyz;
            }
        }
        for (size_t i = 0; i < roots_.size(); ++i) if (quarantined.at(size_t(roots_[i]))) roots_[i] = i;
        quarantined.release(); reused.release();
        const auto trackPath = stage / "tracks.bin";
        files_.push_back({trackPath.u8string(), digest(trackPath), fs::file_size(trackPath)});
        const auto ledgerPath = stage / "conflicts.csv";
        ledgers_.push_back({ledgerPath.u8string(), digest(ledgerPath), fs::file_size(ledgerPath)});
        usage_ = std::move(uses); sources_ = std::move(sources); ownerHash_ = ownerHash; scratch_ = 0;
        validate();
        slog::diag(slog::Tag::Map, "[working-topology] canonical tracks %.3fs: %zu complete shared groups; head commit begin\n", elapsed(), groups_.size());
        save(stage);
        slog::diag(slog::Tag::Map, "[working-topology] head commit %.3fs; topology total %.3fs, original raw observations unchanged\n",
            elapsed(), std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        slog::diag(slog::Tag::Map, "[working-topology] capacity: %zu points, %zu shared groups, roots %llu bytes, group chunks %llu bytes, shared-root rank %llu bytes, resident %llu bytes, phase peak bound %llu / %llu bytes\n",
            roots_.size(), groups_.size(), (unsigned long long)roots_.bytes(), (unsigned long long)groups_.bytes(), (unsigned long long)sharedRoots_.bytes(),
            (unsigned long long)resident(), (unsigned long long)peakCapacityBound_, (unsigned long long)(budget_ / 3));
        const auto stagePath = fs::weakly_canonical(stage);
        for (const auto& temporary : {newRows, moments, featureRows, grouped, ordered, sorted, reusedMetadata, affectedMetadata}) {
            const auto actual = fs::weakly_canonical(temporary); const auto parent = actual.parent_path();
            if (parent != stagePath && parent.parent_path() != stagePath) throw std::runtime_error("topology scratch path escapes owned build directory");
            std::error_code error; fs::remove(actual, error);
            if (error) slog::diag(slog::Tag::Map, "[working-topology] committed scratch retained: %s\n", actual.u8string().c_str());
        }
    }
public:
    using TriangulationProgress = std::function<void(uint64_t, uint64_t)>;
    void triangulate(const Reconstruction& poses, double gate, int threads, const TriangulationProgress& onProgress = {}) {
        struct StageScope {
            bool active = true;
            ~StageScope() { if (active) { try { events::stage_end(Stage::Merge); } catch (...) {} } }
            void finish() { active = false; events::stage_end(Stage::Merge); }
        };
        events::stage_begin(Stage::Merge, int64_t(groups_.size())); StageScope phase;
        struct Job { size_t id; std::vector<Observation> track; };
        auto& pool = bacpu::Pool::get(); const int workers = threads > 0 ? std::min(threads, pool.size()) : pool.size();
        const auto trackBatchBudget = batchBudget(budget_ / 16, 4, trackReaderReserve() + (128u << 10));
        const auto started = std::chrono::steady_clock::now(); auto lastLog = started;
        std::atomic<uint64_t> completed{0}; std::atomic<int64_t> nextReport{2000}; std::mutex progressMutex;
        uint64_t lastPublished = kInvalidPoint3D;
        auto publish = [&](bool final) {
            const auto now = std::chrono::steady_clock::now();
            const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - started).count();
            if (!final) {
                auto deadline = nextReport.load(std::memory_order_relaxed);
                if (milliseconds < deadline || !nextReport.compare_exchange_strong(deadline, milliseconds + 2000, std::memory_order_relaxed)) return;
            }
            std::lock_guard<std::mutex> lock(progressMutex); const auto done = completed.load(std::memory_order_acquire);
            if (done == lastPublished) return; lastPublished = done;
            events::progress(Stage::Merge, int64_t(done), int64_t(groups_.size()));
            if (onProgress) onProgress(done, groups_.size());
            if (now - lastLog >= std::chrono::seconds(2)) {
                slog::diag(slog::Tag::Map, "[working-topology] shared triangulation: %llu/%zu groups completed, %.3fs elapsed\n",
                    (unsigned long long)done, groups_.size(), milliseconds / 1000.); lastLog = now;
            }
        };
        publish(true);
        size_t next = 0;
        while (next < groups_.size()) {
            std::vector<Job> jobs; uint64_t bytes = 0;
            while (next < groups_.size()) {
                const auto& g = groups_[next]; const auto needed = storage::scaleBytes(g.count, sizeof(Observation));
                if (!jobs.empty() && bytes + needed > trackBatchBudget) break;
                checkMemory((bytes + needed) * 2 + jobs.capacity() * sizeof(Job) + trackReaderReserve() + (128u << 10));
                if (needed > trackBatchBudget) throw std::length_error("regional common track exceeds remaining batch budget");
                Job job{next, {}}; job.track.reserve(size_t(g.count));
                readTrack(g, [&](const Observation& row) { job.track.push_back(row); });
                if (jobs.size() == jobs.capacity()) {
                    const size_t cap = std::max<size_t>(8, jobs.size() * 2);
                    checkMemory((bytes + needed) * 2 + (jobs.capacity() + cap) * sizeof(Job) + trackReaderReserve() + (128u << 10)); jobs.reserve(cap);
                }
                bytes += needed; ++next; jobs.push_back(std::move(job));
            }
            pool.runChecked(int(jobs.size()), workers, [&](int index, int) {
                cancel::check(); auto& job = jobs.at(size_t(index)); auto& g = groups_.at(job.id);
                g.xyz = refineUnifiedPoint(g.xyz, job.track, poses, gate);
                completed.fetch_add(1, std::memory_order_release); publish(false);
            });
        }
        if (completed.load(std::memory_order_acquire) != groups_.size()) throw std::runtime_error("regional shared triangulation completion count changed");
        publish(true);
        phase.finish();
    }
    void inject(RegionalWorkingStore::Gather& gathered, std::set<uint64_t>& fixed) const {
        for (const auto& item : gathered.pointKeys) {
            const auto id = group(item.second); if (id == kInvalidPoint3D) continue;
            gathered.model.points3D.at(item.first).xyz = groups_.at(id).xyz; fixed.insert(item.first);
        }
    }
    template<class Visit> uint64_t visitCanonicalMembers(const std::set<uint64_t>& canonicalRoots,
        RegionalWorkingStore& store, Visit visit) const {
        if (canonicalRoots.empty()) return 0;
        if (store.sourceCount() + 1 != offsets_.size()) throw std::runtime_error("canonical member source catalogue changed");
        const size_t words = roots_.size() / 64 + (roots_.size() % 64 != 0);
        checkMemory(storage::scaleBytes(words, sizeof(uint64_t)));
        std::vector<uint64_t> selected(words);
        for (const auto root : canonicalRoots) {
            if (root >= roots_.size() || roots_.at(size_t(root)) != root) throw std::out_of_range("requested point is not a canonical root");
            selected[size_t(root / 64)] |= uint64_t(1) << (root % 64);
        }
        uint64_t aliases = 0;
        for (size_t source = 0; source < store.sourceCount(); ++source) {
            if (store.sourcePointCount(source) != offsets_[source + 1] - offsets_[source] || store.sourceIdentity(source) != sources_.at(source))
                throw std::runtime_error("canonical member source identity or range changed");
            for (uint64_t ordinal = 0; ordinal < offsets_[source + 1] - offsets_[source]; ++ordinal) {
                if (ordinal % 1048576 == 0) cancel::check();
                const auto root = roots_.at(size_t(offsets_[source] + ordinal));
                if ((selected[size_t(root / 64)] >> (root % 64)) & 1) {
                    visit(store.key(source, ordinal)); ++aliases;
                }
            }
        }
        return aliases;
    }
    template<class Visit> uint64_t visitCanonicalMembers(const topology_storage::Bits& selected,
        RegionalWorkingStore& store, Visit visit) const {
        if (selected.size() != roots_.size()) throw std::runtime_error("canonical member bitmap belongs to another point catalogue");
        if (store.sourceCount() + 1 != offsets_.size()) throw std::runtime_error("canonical member source catalogue changed");
        checkMemory(selected.bytes());
        for (size_t point = 0; point < roots_.size(); ++point) {
            if (point % 1048576 == 0) cancel::check();
            if (selected[point] && roots_[point] != point) throw std::out_of_range("requested bitmap point is not a canonical root");
        }
        uint64_t aliases = 0;
        for (size_t source = 0; source < store.sourceCount(); ++source) {
            if (store.sourcePointCount(source) != offsets_[source + 1] - offsets_[source] || store.sourceIdentity(source) != sources_.at(source))
                throw std::runtime_error("canonical member source identity or range changed");
            for (uint64_t ordinal = 0; ordinal < offsets_[source + 1] - offsets_[source]; ++ordinal) {
                if (ordinal % 1048576 == 0) cancel::check();
                const auto root = roots_.at(size_t(offsets_[source] + ordinal));
                if (selected[size_t(root)]) { visit(store.key(source, ordinal)); ++aliases; }
            }
        }
        return aliases;
    }
    template<class Visit> void visitCommonObservations(Visit visit) const {
        for (size_t g = 0; g < groups_.size(); ++g) readTrack(groups_[g], visit);
    }
    void synchronize(RegionalWorkingStore& store) const {
        for (size_t source = 0; source < store.sourceCount(); ++source)
            store.updateSourceXYZ(source, [&](Key key, Vec3 value) { return xyz(key, value); });
    }
    void saveParameters(const fs::path& path) const {
        model_store_detail::Writer out(path.string() + ".part"); out.scalar<uint64_t>(magic); out.string(identity_);
        out.scalar<uint64_t>(groups_.size()); for (size_t g = 0; g < groups_.size(); ++g) out.scalar(groups_[g].xyz);
        out.finish(); replaceFile(path.string() + ".part", path);
        lastParametersHash_ = digest(path); lastParametersIdentity_ = identity_; lastParametersPath_ = path;
        resume::store(fs::u8path(path.u8string() + ".sha256"), lastParametersHash_);
    }
    void loadParameters(const fs::path& path) {
        if (restoreAcceptedParameters(path) != ParameterRestore::Exact)
            throw std::runtime_error("regional parameter file belongs to initialized ancestor, not current topology");
    }
    void interpolateParameters(const fs::path& accepted, const fs::path& candidate, double alpha) {
        if (!(alpha > 0 && alpha <= 1) || !std::isfinite(alpha))
            throw std::invalid_argument("invalid common parameter step");
        restore();
        for (const auto& path : {accepted, candidate}) {
            const auto expected = resume::recorded(fs::u8path(path.u8string() + ".sha256"));
            if (expected.empty() || digest(path) != expected)
                throw std::runtime_error("common parameter step checksum changed");
        }
        checkMemory(scratch_ + storage::scaleBytes(groups_.size(), sizeof(Vec3)) + (128u << 10));
        model_store_detail::Reader before(accepted, 64u << 10), after(candidate, 64u << 10);
        for (auto* input : {&before, &after}) {
            if (input->scalar<uint64_t>() != magic || boundedString(*input) != identity_ ||
                input->count(sizeof(Vec3)) != groups_.size())
                throw std::runtime_error("common parameter step topology changed");
        }
        std::vector<Vec3> values; values.reserve(groups_.size());
        for (size_t group = 0; group < groups_.size(); ++group) {
            if (group % 65536 == 0) cancel::check();
            const auto a = before.scalar<Vec3>(), b = after.scalar<Vec3>();
            const auto value = alpha == 1 ? b : a * (1 - alpha) + b * alpha;
            if (!std::isfinite(a.norm()) || !std::isfinite(b.norm()) || !std::isfinite(value.norm()))
                throw std::runtime_error("nonfinite common parameter step");
            values.push_back(value);
        }
        before.finish(); after.finish();
        for (size_t group = 0; group < groups_.size(); ++group) groups_[group].xyz = values[group];
        lastParametersHash_.clear(); lastParametersIdentity_.clear(); lastParametersPath_.clear();
    }
    ParameterRestore restoreAcceptedParameters(const fs::path& path) {
        restore();
        const auto expected = resume::recorded(fs::u8path(path.u8string() + ".sha256"));
        const auto actual = digest(path);
        if (expected.empty() || expected != actual) throw std::runtime_error("regional common parameter checksum changed");
        model_store_detail::Reader in(path, 64u << 10);
        const auto format = in.scalar<uint64_t>();
        if (format != magic && format != legacyMagic) throw std::runtime_error("invalid regional common parameter format");
        const auto identity = boundedString(in); const auto count = in.count(sizeof(Vec3));
        if (identity == identity_ && count == groups_.size()) {
            restoreValues(in, count);
            lastParametersHash_ = actual; lastParametersIdentity_ = identity_; lastParametersPath_ = path;
            return ParameterRestore::Exact;
        }
        const auto parent = std::find_if(parents_.begin(), parents_.end(), [&](const Parent& candidate) {
            return candidate.identity == identity && candidate.groups == count && candidate.parameterHash == actual;
        });
        if (parent == parents_.end()) throw std::runtime_error("regional topology parameters changed identity or initialized provenance");
        for (size_t g = 0; g < count; ++g) if (!std::isfinite(in.scalar<Vec3>().norm())) throw std::runtime_error("nonfinite ancestor common parameter");
        in.finish(); return ParameterRestore::InitializedAppend;
    }

private:
    static constexpr uint64_t legacyMagic = 0x35504f5453474552ull;
    static constexpr uint64_t magic = 0x36504f5453474552ull;
    uint64_t global(const Key& key) const {
        if (offsets_.empty() || key.source >= offsets_.size() - 1 || key.ordinal >= offsets_[key.source + 1] - offsets_[key.source])
            throw std::out_of_range("regional topology point source");
        return offsets_[key.source] + key.ordinal;
    }
    uint64_t groupForRoot(uint64_t root) const {
        return sharedRoots_.at(size_t(root)) ? sharedRoots_.rank(size_t(root)) : kInvalidPoint3D;
    }
    uint64_t group(const Key& key) const { return groupForRoot(roots_.at(size_t(global(key)))); }
    void rebuildSharedRoots() {
        checkMemory(scratch_ + topology_storage::RankedBits::allocationBytes(roots_.size()));
        sharedRoots_.release(); sharedRoots_.resize(roots_.size());
        for (size_t g = 0; g < groups_.size(); ++g) sharedRoots_.at(size_t(groups_[g].root)) = true;
        sharedRoots_.buildRanks();
    }
    uint64_t find(uint64_t i) {
        uint64_t root = i; while (roots_.at(size_t(root)) != root) root = roots_[size_t(root)];
        while (i != root) { const uint64_t next = roots_[size_t(i)]; roots_[size_t(i)] = root; i = next; }
        return root;
    }
    void join(uint64_t a, uint64_t b) { a = find(a); b = find(b); if (a != b) roots_[size_t(std::max(a, b))] = std::min(a, b); }
    uint64_t resident() const {
        uint64_t bytes = usage_.capacity() * sizeof(uint32_t) + readers_.size() * ((64u << 10) + 256) + offsets_.capacity() * sizeof(uint64_t) + roots_.bytes() + sharedRoots_.bytes() + groups_.bytes();
        for (const auto& parent : parents_) bytes = storage::addBytes(bytes, sizeof(Parent) + parent.identity.capacity() + parent.parameterHash.capacity());
        for (const auto& source : sources_) bytes = storage::addBytes(bytes, source.capacity() + 32);
        for (const auto& file : files_) bytes = storage::addBytes(bytes, file.path.capacity() + file.hash.capacity() + sizeof(File));
        for (const auto& file : ledgers_) bytes = storage::addBytes(bytes, file.path.capacity() + file.hash.capacity() + sizeof(File));
        return bytes;
    }
    void checkMemory(uint64_t extra = 0) const {
        const auto requested = storage::addBytes(resident(), extra);
        peakCapacityBound_ = std::max(peakCapacityBound_, requested);
        if (requested > budget_ / 3)
            throw std::length_error("complete regional topology exceeds bounded working memory: " + std::to_string(requested) +
                " > " + std::to_string(budget_ / 3) + " bytes (roots=" + std::to_string(roots_.bytes()) +
                ", groups=" + std::to_string(groups_.bytes()) + ", shared=" + std::to_string(sharedRoots_.bytes()) + ", phase-extra=" + std::to_string(extra) + ")");
    }
    uint64_t batchBudget(uint64_t maximum, uint64_t multiplicity, uint64_t fixed) const {
        const uint64_t used = storage::addBytes(resident(), storage::addBytes(scratch_, fixed));
        if (used >= budget_ / 3 || !multiplicity) throw std::length_error("regional topology has no remaining phase memory");
        const uint64_t available = (budget_ / 3 - used) / multiplicity;
        const uint64_t result = std::min(maximum, available);
        if (result < sizeof(Observation)) throw std::length_error("regional topology has no remaining observation batch memory");
        return result;
    }
    void restoreValues(model_store_detail::Reader& in, size_t count) {
        const auto copyBytes = storage::scaleBytes(count + topology_storage::Chunks<Vec3>::block, sizeof(Vec3));
        if (storage::addBytes(resident(), storage::addBytes(scratch_, copyBytes)) <= budget_ / 3) {
            checkMemory(scratch_ + copyBytes); topology_storage::Chunks<Vec3> values; values.reserve(count);
            for (size_t g = 0; g < count; ++g) {
                const auto value = in.scalar<Vec3>(); if (!std::isfinite(value.norm())) throw std::runtime_error("nonfinite regional common parameter"); values.push_back(value);
            }
            in.finish(); for (size_t g = 0; g < count; ++g) groups_[g].xyz = values[g]; return;
        }
        checkMemory(scratch_ + (9u << 18));
        storage::requireSpace(root_, storage::scaleBytes(count, 2 * sizeof(Vec3)) + (1u << 20));
        const auto label = "parameters-restore-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        const auto candidate = root_ / (label + "-candidate.bin"), rollback = root_ / (label + "-rollback.bin");
        {
            model_store_detail::Writer next(candidate), previous(rollback);
            for (size_t g = 0; g < count; ++g) {
                const auto value = in.scalar<Vec3>(); if (!std::isfinite(value.norm())) throw std::runtime_error("nonfinite regional common parameter");
                next.scalar(value); previous.scalar(groups_[g].xyz);
            }
            in.finish(); next.finish(); previous.finish();
        }
        auto apply = [&](const fs::path& path) {
            model_store_detail::Reader values(path, 64u << 10);
            for (size_t g = 0; g < count; ++g) groups_[g].xyz = values.scalar<Vec3>(); values.finish();
        };
        auto cleanup = [&] {
            std::error_code error; fs::remove(candidate, error); error.clear(); fs::remove(rollback, error);
        };
        try { apply(candidate); }
        catch (...) {
            const auto failure = std::current_exception();
            try { apply(rollback); } catch (...) { std::throw_with_nested(std::runtime_error("regional parameter restore and rollback both failed; owned scratch retained")); }
            cleanup(); std::rethrow_exception(failure);
        }
        cleanup();
        counters_.parameter_disk_bytes = storage::addBytes(counters_.parameter_disk_bytes, storage::scaleBytes(count, 2 * sizeof(Vec3)));
        slog::diag(slog::Tag::Map, "[working-topology] parameter restore used bounded disk staging: %zu groups, %llu bytes per XYZ stream\n", count, (unsigned long long)storage::scaleBytes(count, sizeof(Vec3)));
    }
    static std::string digest(const fs::path& path) {
        std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("cannot hash regional topology dependency");
        spirula::Sha256 hash; std::array<char, 65536> buffer;
        while (in) { in.read(buffer.data(), buffer.size()); hash.update(reinterpret_cast<const uint8_t*>(buffer.data()), size_t(in.gcount())); }
        if (!in.eof()) throw std::runtime_error("regional topology hash read failed"); return hash.hex();
    }
    template<class T> std::vector<T> boundedVector(model_store_detail::Reader& in) const {
        const size_t count = in.count(sizeof(T)); checkMemory(storage::scaleBytes(count, sizeof(T)));
        std::vector<T> values(count); in.read(values.data(), count); return values;
    }
    std::string boundedString(model_store_detail::Reader& in) const {
        const size_t count = in.count(1); checkMemory(count); std::string value(count, '\0'); in.read(value.data(), count); return value;
    }
    template<class Visit> void readTrack(const Group& group, Visit visit) const {
        const auto& file = files_.at(size_t(group.file));
        if (group.first > file.bytes / sizeof(Observation) || group.count > file.bytes / sizeof(Observation) - group.first)
            throw std::runtime_error("regional common track range invalid");
        auto found = readers_.find(group.file);
        if (found == readers_.end()) {
            const size_t capacity = std::max<size_t>(1, std::min<size_t>(32, budget_ / 96 / (64u << 10)));
            if (readers_.size() == capacity) readers_.erase(readers_.begin());
            checkMemory(scratch_ + (64u << 10) + 256);
            found = readers_.emplace(group.file, std::make_shared<model_store_detail::Reader>(fs::u8path(file.path), 64u << 10)).first;
        }
        auto& reader = *found->second;
        reader.seek(storage::scaleBytes(group.first, sizeof(Observation)));
        for (uint64_t i = 0; i < group.count; ++i) {
            const auto row = reader.scalar<Observation>();
            if (row.key != group.root) throw std::runtime_error("regional common track identity corrupted"); visit(row);
        }
    }
    uint64_t trackReaderReserve() const {
        const size_t capacity = std::max<size_t>(1, std::min<size_t>(32, budget_ / 96 / (64u << 10)));
        return (capacity > readers_.size() ? capacity - readers_.size() : 0) * uint64_t((64u << 10) + 256);
    }
    void validate() const {
        checkMemory();
        if (offsets_.size() != sources_.size() + 1 || offsets_.empty() || offsets_.front() || offsets_.back() != roots_.size() || sharedRoots_.size() != roots_.size() || !std::is_sorted(offsets_.begin(), offsets_.end()))
            throw std::runtime_error("invalid regional topology source ranges");
        for (size_t i = 0; i < roots_.size(); ++i) if (roots_[i] >= roots_.size() || roots_[size_t(roots_[i])] != roots_[i])
            throw std::runtime_error("invalid regional topology root");
        for (size_t g = 0; g < groups_.size(); ++g) {
            const auto& group = groups_[g]; const auto& file = files_.at(size_t(group.file));
            if (group.root >= roots_.size() || roots_[size_t(group.root)] != group.root || groupForRoot(group.root) != g || (g && groups_[g - 1].root >= group.root) ||
                group.count < 2 || !group.weight || !std::isfinite(group.xyz.norm()) || group.first > file.bytes / sizeof(Observation) ||
                group.count > file.bytes / sizeof(Observation) - group.first)
                throw std::runtime_error("invalid regional topology common track");
        }
        for (const auto& file : files_) if (file.bytes % sizeof(Observation) || fs::file_size(fs::u8path(file.path)) != file.bytes)
            throw std::runtime_error("regional topology track file size changed");
    }
    void save(const fs::path& stage) {
        checkMemory(scratch_ + (2u << 20) + (64u << 10));
        std::ostringstream label; label << ownerHash_ << '\n'; for (const auto& source : sources_) label << source << '\n';
        spirula::Sha256 hash; const auto text = label.str(); hash.update(reinterpret_cast<const uint8_t*>(text.data()), text.size()); identity_ = hash.hex();
        model_store_detail::Writer out(stage / "topology.bin"); out.scalar<uint64_t>(magic); out.string(identity_); out.string(ownerHash_);
        out.scalar<uint64_t>(sources_.size()); for (const auto& source : sources_) out.string(source);
        out.vector(usage_); out.vector(offsets_); out.scalar<uint64_t>(roots_.size()); out.scalar<uint8_t>(roots_.wide() ? 8 : 4);
        for (size_t i = 0; i < roots_.size(); ++i) { if (roots_.wide()) out.scalar<uint64_t>(roots_[i]); else out.scalar<uint32_t>(uint32_t(roots_[i])); }
        out.scalar<uint64_t>(groups_.size()); for (size_t g = 0; g < groups_.size(); ++g) out.scalar(groups_[g]);
        out.scalar<uint64_t>(files_.size()); for (const auto& file : files_) { out.string(file.path); out.string(file.hash); out.scalar(file.bytes); }
        out.scalar<uint64_t>(ledgers_.size()); for (const auto& file : ledgers_) { out.string(file.path); out.string(file.hash); out.scalar(file.bytes); }
        out.scalar<uint64_t>(parents_.size()); for (const auto& parent : parents_) { out.string(parent.identity); out.string(parent.parameterHash); out.scalar(parent.groups); }
        out.finish();
        model_store_detail::Writer head(root_ / "head.part"); head.scalar<uint64_t>(magic); head.string((stage / "topology.bin").u8string()); head.string(digest(stage / "topology.bin"));
        head.finish(); replaceFile(root_ / "head.part", root_ / "head.bin");
        lastParametersHash_.clear(); lastParametersIdentity_.clear(); lastParametersPath_.clear();
    }
    void load() {
        model_store_detail::Reader head(root_ / "head.bin", 64u << 10);
        const auto format = head.scalar<uint64_t>();
        if (format != magic && format != legacyMagic) throw std::runtime_error("unsupported regional topology head");
        const auto path = fs::u8path(boundedString(head)); const auto expected = boundedString(head); head.finish();
        if (digest(path) != expected) throw std::runtime_error("regional topology checkpoint checksum changed");
        model_store_detail::Reader in(path, 64u << 10); if (in.scalar<uint64_t>() != format) throw std::runtime_error("invalid regional topology checkpoint");
        identity_ = boundedString(in); ownerHash_ = boundedString(in);
        const auto count = in.count(8); checkMemory(storage::scaleBytes(count, 96)); sources_.reserve(count);
        for (size_t i = 0; i < count; ++i) sources_.push_back(boundedString(in));
        usage_ = boundedVector<uint32_t>(in); offsets_ = boundedVector<uint64_t>(in);
        const auto pointCount = in.count(format == legacyMagic ? 8 : 4);
        const auto width = format == legacyMagic ? uint8_t(8) : in.scalar<uint8_t>();
        if ((width != 4 && width != 8) || offsets_.empty() || offsets_.back() != pointCount || (width == 4 && pointCount > UINT32_MAX)) throw std::runtime_error("invalid topology point width or range");
        checkMemory(topology_storage::Indices::allocationBytes(pointCount)); roots_.resize(pointCount);
        for (size_t i = 0; i < pointCount; ++i) roots_[i] = width == 4 ? uint64_t(in.scalar<uint32_t>()) : in.scalar<uint64_t>();
        uint64_t legacyMaximum = 0; bool legacyGroups = false;
        if (format == legacyMagic) {
            if (in.count(8) != pointCount) throw std::runtime_error("invalid legacy topology group map");
            for (size_t i = 0; i < pointCount; ++i) { const auto id = in.scalar<uint64_t>(); if (id != kInvalidPoint3D) { legacyGroups = true; legacyMaximum = std::max(legacyMaximum, id); } }
        }
        const auto groupCount = in.count(sizeof(Group));
        checkMemory(storage::scaleBytes(groupCount + topology_storage::Chunks<Group>::block, sizeof(Group))); groups_.reserve(groupCount);
        for (size_t g = 0; g < groupCount; ++g) groups_.push_back(in.scalar<Group>());
        if (legacyGroups && legacyMaximum >= groupCount) throw std::runtime_error("invalid legacy topology group reference");
        rebuildSharedRoots();
        const auto files = in.count(24); checkMemory(storage::scaleBytes(files, sizeof(File) + 256)); files_.reserve(files);
        for (size_t i = 0; i < files; ++i) { File file; file.path = boundedString(in); file.hash = boundedString(in); file.bytes = in.scalar<uint64_t>(); files_.push_back(std::move(file)); }
        const auto ledgers = in.count(24); checkMemory(storage::scaleBytes(ledgers, sizeof(File) + 256)); ledgers_.reserve(ledgers);
        for (size_t i = 0; i < ledgers; ++i) { File file; file.path = boundedString(in); file.hash = boundedString(in); file.bytes = in.scalar<uint64_t>(); ledgers_.push_back(std::move(file)); }
        const auto parents = in.count(24); checkMemory(storage::scaleBytes(parents, sizeof(Parent) + 128)); parents_.reserve(parents);
        for (size_t i = 0; i < parents; ++i) { Parent parent; parent.identity = boundedString(in); parent.parameterHash = boundedString(in); parent.groups = in.scalar<uint64_t>(); parents_.push_back(std::move(parent)); }
        in.finish(); validate();
        for (const auto& file : ledgers_) if (fs::file_size(fs::u8path(file.path)) != file.bytes || digest(fs::u8path(file.path)) != file.hash)
            throw std::runtime_error("regional immutable conflict ledger checksum changed");
        for (const auto& file : files_) if (digest(fs::u8path(file.path)) != file.hash) throw std::runtime_error("regional immutable common-track checksum changed");
    }
    void resetState() {
        usage_ = {}; usage_.shrink_to_fit(); offsets_ = {}; roots_.clear(); roots_.shrink_to_fit(); sharedRoots_.release(); groups_.release(); files_ = {}; ledgers_ = {}; sources_ = {}; parents_ = {};
        offsets_.shrink_to_fit();
        files_.shrink_to_fit(); ledgers_.shrink_to_fit(); sources_.shrink_to_fit(); parents_.shrink_to_fit(); readers_.clear(); scratch_ = 0; ownerHash_.clear(); identity_.clear(); lastParametersHash_.clear(); lastParametersIdentity_.clear(); lastParametersPath_.clear();
    }
    fs::path root_; size_t budget_; uint64_t scratch_ = 0;
    std::vector<uint64_t> offsets_; topology_storage::Indices roots_; topology_storage::RankedBits sharedRoots_; std::vector<uint32_t> usage_;
    topology_storage::Chunks<Group> groups_; std::vector<File> files_, ledgers_; std::vector<std::string> sources_;
    std::string ownerHash_, identity_; Counters counters_; std::vector<Parent> parents_;
    mutable std::string lastParametersHash_, lastParametersIdentity_; mutable fs::path lastParametersPath_;
    mutable std::map<uint64_t, std::shared_ptr<model_store_detail::Reader>> readers_;
    mutable uint64_t peakCapacityBound_ = 0;
};

} // namespace sfm::regional
