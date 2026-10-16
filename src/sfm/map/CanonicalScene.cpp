#include "sfm/map/CanonicalScene.h"
#include "core/HostMemory.h"
#include <array>
#include <cerrno>
#include <exception>
#include <sstream>

namespace sfm::regional {
namespace {
using Reader = model_store_detail::Reader;
using Writer = model_store_detail::Writer;
using Path = std::filesystem::path;
constexpr uint64_t sceneMagic = 0x314e4353454e4143ull;
constexpr uint64_t headMagic = 0x314448534e414343ull;
constexpr uint64_t stepMagic = 0x3146455250545343ull;
struct Alias { uint64_t root = 0, local = 0; };
std::string digest(const Path& path) {
    const auto result = spirula::sha256_file(path.string());
    if (result.empty()) throw std::runtime_error("cannot hash canonical scene file");
    return result;
}
std::string finiteKindsHash(uint64_t count) {
    spirula::Sha256 hash; const std::array<uint8_t, 65536> zero{};
    while (count) { const auto n = size_t(std::min<uint64_t>(count, zero.size())); hash.update(zero.data(), n); count -= n; }
    return hash.hex();
}
std::string token() { return std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()); }
void requireFinite(Vec3 value) { if (!step::finite(value)) throw std::runtime_error("nonfinite canonical scene XYZ"); }
void writeBinding(Writer& out, const CanonicalScene::Binding& binding) {
    out.string(binding.rawCatalog); out.string(binding.topology); out.scalar<uint64_t>(binding.sources.size());
    for (const auto& source : binding.sources) { out.string(source.label); out.string(source.digest); out.scalar(source.points); out.scalar(source.observations); }
}
CanonicalScene::Binding readBinding(Reader& in, size_t budget) {
    CanonicalScene::Binding binding; binding.rawCatalog = in.string(); binding.topology = in.string();
    const auto count = in.count(32); if (count > budget / 256) throw std::length_error("canonical source catalogue exceeds budget");
    binding.sources.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        CanonicalScene::SourceIdentity source; source.label = in.string(); source.digest = in.string();
        source.points = in.scalar<uint64_t>(); source.observations = in.scalar<uint64_t>(); binding.sources.push_back(std::move(source));
    }
    return binding;
}
void validatePoses(const Reconstruction& poses) {
    if (!poses.points3D.empty()) throw std::invalid_argument("canonical pose table contains geometry");
    for (const auto& camera : poses.cameras) { if (camera.first != camera.second.id) throw std::runtime_error("canonical camera ID mismatch"); step::validateCamera(camera.second); }
    for (const auto& image : poses.images) {
        if (image.first != image.second.id || !poses.cameras.count(image.second.camera_id)) throw std::runtime_error("canonical photo/camera identity mismatch");
        if (!image.second.points2D.empty() || !image.second.point3D_ids.empty()) throw std::invalid_argument("canonical pose table contains raw features");
        step::validatePose(image.second.pose);
    }
}
struct DigestWriter {
    spirula::Sha256 hash;
    template<class T> void scalar(const T& value) { hash.update(reinterpret_cast<const uint8_t*>(&value), sizeof(value)); }
    template<class T> void vector(const std::vector<T>& values) { scalar<uint64_t>(values.size()); hash.update(reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T)); }
    void string(const std::string& value) { scalar<uint64_t>(value.size()); hash.update(reinterpret_cast<const uint8_t*>(value.data()), value.size()); }
};
void samePoseIdentity(const Reconstruction& before, const Reconstruction& after) {
    if (before.images.size() != after.images.size() || before.cameras.size() != after.cameras.size()) throw std::runtime_error("canonical proposal changed pose catalogue");
    for (const auto& image : before.images) {
        const auto found = after.images.find(image.first);
        if (found == after.images.end() || image.second.name != found->second.name || image.second.camera_id != found->second.camera_id ||
            image.second.registered != found->second.registered) throw std::runtime_error("canonical proposal changed photo identity");
    }
    for (const auto& camera : before.cameras) {
        const auto found = after.cameras.find(camera.first); if (found == after.cameras.end()) throw std::runtime_error("canonical proposal changed camera identity");
        (void)step::interpolateCamera(camera.second, found->second, 1);
    }
    auto a = before, b = after;
    for (auto& image : b.images) image.second.pose = a.images.at(image.first).pose;
    for (auto& camera : b.cameras) camera.second = a.cameras.at(camera.first);
    DigestWriter ha, hb; model_store_detail::serializeModel(a, ha); model_store_detail::serializeModel(b, hb);
    if (ha.hash.hex() != hb.hash.hex()) throw std::runtime_error("canonical proposal changed model metadata");
}
class Lock {
public:
    explicit Lock(const Path& path, bool readOnly = false) {
#ifdef _WIN32
        handle = CreateFileW(path.c_str(), readOnly ? GENERIC_READ : GENERIC_READ | GENERIC_WRITE,
            readOnly ? FILE_SHARE_READ : 0, nullptr, readOnly ? OPEN_EXISTING : OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (readOnly && handle == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_NOT_FOUND) return;
        if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("canonical scene transaction locked");
#else
        handle = readOnly ? open(path.c_str(), O_RDONLY) : open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (readOnly && handle < 0 && errno == ENOENT) return;
        if (handle < 0 || flock(handle, (readOnly ? LOCK_SH : LOCK_EX) | LOCK_NB)) { if (handle >= 0) close(handle); throw std::runtime_error("canonical scene transaction locked"); }
#endif
    }
    ~Lock() {
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
        if (handle >= 0) { flock(handle, LOCK_UN); close(handle); }
#endif
    }
private:
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int handle = -1;
#endif
};
}
void validateLandmark(const CanonicalLandmark& landmark) {
    requireFinite(landmark.value);
    if (landmark.kind != LandmarkKind::FiniteXYZ && landmark.kind != LandmarkKind::UnitDirection && landmark.kind != LandmarkKind::RetainedObservation)
        throw std::runtime_error("unknown canonical landmark kind");
    if (landmark.kind == LandmarkKind::RetainedObservation && (landmark.value.x != 0 || landmark.value.y != 0 || landmark.value.z != 0))
        throw std::runtime_error("retained observation has no geometric payload");
    if (landmark.kind == LandmarkKind::UnitDirection && std::abs(landmark.value.dot(landmark.value) - 1.) > 1e-10)
        throw std::runtime_error("canonical direction is not a unit vector");
}
CanonicalLandmark interpolateLandmark(const CanonicalLandmark& before, const CanonicalLandmark& after, double alpha) {
    if (!std::isfinite(alpha) || alpha < 0 || alpha > 1) throw std::invalid_argument("landmark alpha must be in [0,1]"); validateLandmark(before); validateLandmark(after);
    if (before.kind != after.kind) throw std::runtime_error("canonical interpolation changed landmark kind");
    if (alpha == 0) return before; if (alpha == 1) return after;
    if (before.value.x == after.value.x && before.value.y == after.value.y && before.value.z == after.value.z) return before;
    if (before.kind == LandmarkKind::FiniteXYZ) return {before.value * (1 - alpha) + after.value * alpha, before.kind};
    const double dot = std::clamp(before.value.dot(after.value), -1., 1.);
    if (dot <= -1. + 1e-12) throw std::runtime_error("antipodal direction interpolation is ambiguous");
    Vec3 value;
    if (dot > 1. - 1e-10) value = (before.value * (1 - alpha) + after.value * alpha).normalized();
    else { const double angle = std::acos(dot), denominator = std::sin(angle);
        value = (before.value * (std::sin((1 - alpha) * angle) / denominator) + after.value * (std::sin(alpha * angle) / denominator)).normalized(); }
    CanonicalLandmark result{value, before.kind}; validateLandmark(result); return result;
}
bool CanonicalScene::SourceIdentity::operator==(const SourceIdentity& b) const {
    return label == b.label && digest == b.digest && points == b.points && observations == b.observations;
}
bool CanonicalScene::Binding::operator==(const Binding& b) const { return rawCatalog == b.rawCatalog && topology == b.topology && sources == b.sources; }
struct CanonicalScene::Impl {
    Path root, active;
    size_t budget = 0;
    Snapshot accepted, current;
    Reconstruction poseTable, acceptedPoses;
    std::vector<uint64_t> offsets;
    std::unique_ptr<Lock> lock, readLease;
    mutable std::unique_ptr<Reader> aliases, xyz, kinds;
    struct AliasPage { uint64_t tick = 0; std::vector<Alias> values; };
    mutable std::map<uint64_t, AliasPage> aliasPages;
    mutable uint64_t aliasTick = 0, aliasBytes = 0;
    uint64_t poseBytes = 0, savedPoseBytes = 0;
    mutable std::unique_ptr<Reader> acceptedAliases, acceptedXYZ, acceptedKinds;
    mutable std::map<uint64_t, AliasPage> acceptedPages;
    mutable uint64_t acceptedAliasBytes = 0;
    mutable Counters io;
    bool edit = false, haveAccepted = false, readOnly = false, layoutEdit = false;
    bool interpolationReady = false;
    Snapshot interpolationProposal;
    double interpolationAlpha = 0;
    std::string interpolationPoseHash;
    uint64_t memory() const {
        uint64_t bytes = sizeof(Impl) + poseBytes + savedPoseBytes + aliasBytes + acceptedAliasBytes + offsets.capacity() * sizeof(uint64_t);
        bytes = storage::addBytes(bytes, (aliases ? 65536 : 0) + (xyz ? 65536 : 0) + (acceptedAliases ? 65536 : 0) + (acceptedXYZ ? 65536 : 0) + (kinds ? 65536 : 0) + (acceptedKinds ? 65536 : 0));
        for (const auto& s : current.binding.sources) bytes = storage::addBytes(bytes, sizeof(s) + s.label.capacity() + s.digest.capacity());
        for (const auto& s : accepted.binding.sources) bytes = storage::addBytes(bytes, sizeof(s) + s.label.capacity() + s.digest.capacity());
        for (const auto& s : interpolationProposal.binding.sources) bytes = storage::addBytes(bytes, sizeof(s) + s.label.capacity() + s.digest.capacity());
        return bytes + current.identity.capacity() + accepted.identity.capacity() + current.phase.capacity() + accepted.phase.capacity() + 4096;
    }
    void guard(uint64_t extra = 0) const {
        if (storage::addBytes(memory(), extra) > budget) throw std::length_error("canonical scene memory budget exceeded");
        const auto available = spirula::availableRamBytes();
        if (available && extra > available - std::min<size_t>(available / 4, size_t(4) << 30)) throw std::length_error("canonical scene physical RAM headroom exceeded");
    }
    void clearReaders() const { aliases.reset(); xyz.reset(); kinds.reset(); }
    void setOffsets() {
        offsets.clear(); offsets.push_back(0); uint64_t obs = 0; std::set<std::string> labels;
        if (current.binding.rawCatalog.empty() || current.binding.topology.empty()) throw std::runtime_error("canonical scene missing immutable identities");
        for (const auto& s : current.binding.sources) {
            if (s.label.empty() || s.digest.empty() || !labels.insert(s.label).second) throw std::runtime_error("canonical scene duplicate/empty source identity");
            offsets.push_back(storage::addBytes(offsets.back(), s.points)); obs = storage::addBytes(obs, s.observations);
        }
        if (offsets.back() != current.pointSlots || obs != current.observations) throw std::runtime_error("canonical scene catalogue counts disagree");
        guard();
    }
    Reader& aliasReader() const { if (!aliases) { aliases = std::make_unique<Reader>(active / "aliases.bin", 65536); ++io.fileOpens; } return *aliases; }
    Reader& xyzReader() const { if (!xyz) { xyz = std::make_unique<Reader>(active / "points.bin", 65536); ++io.fileOpens; } return *xyz; }
    LandmarkKind kind(uint64_t id, bool saved = false) const {
        const auto& snapshot = saved ? accepted : current;
        if (id >= snapshot.pointSlots) throw std::out_of_range("canonical landmark kind index");
        if (snapshot.schema == 1) return LandmarkKind::FiniteXYZ;
        auto& reader = saved ? acceptedKinds : kinds;
        if (!reader) { guard(65536); reader = std::make_unique<Reader>((saved ? accepted.directory : active) / "kinds.bin", 65536); ++io.fileOpens; }
        reader->seek(id); const auto value = reader->scalar<uint8_t>();
        if (value > uint8_t(snapshot.schema >= 3 ? LandmarkKind::RetainedObservation : LandmarkKind::UnitDirection)) throw std::runtime_error("unknown canonical landmark kind");
        return LandmarkKind(value);
    }
    void requireAcceptedKinds() const {
        if (!haveAccepted) return;
        if (accepted.schema >= 2) {
            if (std::filesystem::is_symlink(accepted.directory / "kinds.bin") ||
                std::filesystem::file_size(accepted.directory / "kinds.bin") != accepted.pointSlots ||
                digest(accepted.directory / "kinds.bin") != accepted.kindHash) throw std::runtime_error("accepted canonical landmark kinds changed");
        } else if (std::filesystem::exists(accepted.directory / "kinds.bin")) throw std::runtime_error("legacy accepted snapshot has unbound landmark types");
    }
    void writeKinds() {
        editing(); requireAcceptedKinds(); const auto part = active / "kinds.part"; storage::requireSpace(active, current.pointSlots);
        Writer out(part); std::unique_ptr<Reader> before;
        if (haveAccepted && accepted.schema >= 2) before = std::make_unique<Reader>(accepted.directory / "kinds.bin", 65536);
        std::unique_ptr<Reader> oldAliases; if (haveAccepted) oldAliases = std::make_unique<Reader>(accepted.directory / "aliases.bin", 65536);
        Reader nextAliases(active / "aliases.bin", 65536);
        for (uint64_t id = 0; id < current.pointSlots; ++id) {
            if (!(id & 65535)) cancel::check(); const auto next = nextAliases.scalar<Alias>(); uint8_t value = 0;
            if (haveAccepted && id < accepted.pointSlots) {
                const auto old = oldAliases->scalar<Alias>();
                if (old.root != next.root || old.local != next.local) throw std::runtime_error("canonical layout changed an accepted alias/type identity");
                if (before) value = before->scalar<uint8_t>();
            }
            out.scalar(value);
        }
        if (haveAccepted && current.pointSlots < accepted.pointSlots) throw std::runtime_error("canonical layout lost accepted landmarks");
        nextAliases.finish(); if (oldAliases) oldAliases->finish(); if (before) before->finish(); out.finish();
        kinds.reset(); storage::replaceFile(part, active / "kinds.bin"); current.schema = 3; current.kindHash = digest(active / "kinds.bin");
    }
    Alias alias(uint64_t id) const {
        if (id >= current.pointSlots) throw std::out_of_range("canonical scene point index");
        const uint64_t page = id / 4096, start = page * 4096; auto found = aliasPages.find(page);
        if (found == aliasPages.end()) {
            const size_t limit = std::max<size_t>(1, budget / (16 * 65536));
            if (aliasPages.size() >= limit) {
                auto oldest = aliasPages.begin(); for (auto at = aliasPages.begin(); at != aliasPages.end(); ++at) if (at->second.tick < oldest->second.tick) oldest = at;
                aliasBytes -= oldest->second.values.capacity() * sizeof(Alias) + sizeof(AliasPage) + 64; aliasPages.erase(oldest);
            }
            const size_t count = size_t(std::min<uint64_t>(4096, current.pointSlots - start)); guard(count * sizeof(Alias) + 65536 + sizeof(AliasPage) + 64);
            AliasPage values; values.values.resize(count); auto& in = aliasReader(); in.seek(storage::scaleBytes(start, sizeof(Alias))); in.read(values.values.data(), count);
            io.aliasBytes += count * sizeof(Alias); aliasBytes += values.values.capacity() * sizeof(Alias) + sizeof(AliasPage) + 64;
            found = aliasPages.emplace(page, std::move(values)).first;
        }
        found->second.tick = ++aliasTick; return found->second.values[size_t(id - start)];
    }
    Alias acceptedAlias(uint64_t id) const {
        if (!haveAccepted || id >= accepted.pointSlots) throw std::out_of_range("accepted canonical point index");
        const uint64_t page = id / 4096, start = page * 4096; auto found = acceptedPages.find(page);
        if (found == acceptedPages.end()) {
            const size_t limit = std::max<size_t>(1, budget / (16 * 65536));
            if (acceptedPages.size() >= limit) {
                auto oldest = acceptedPages.begin(); for (auto at = acceptedPages.begin(); at != acceptedPages.end(); ++at) if (at->second.tick < oldest->second.tick) oldest = at;
                acceptedAliasBytes -= oldest->second.values.capacity() * sizeof(Alias) + sizeof(AliasPage) + 64; acceptedPages.erase(oldest);
            }
            const size_t count = size_t(std::min<uint64_t>(4096, accepted.pointSlots - start)); guard(count * sizeof(Alias) + 65536 + sizeof(AliasPage) + 64);
            if (!acceptedAliases) { acceptedAliases = std::make_unique<Reader>(accepted.directory / "aliases.bin", 65536); ++io.fileOpens; }
            AliasPage values; values.values.resize(count); acceptedAliases->seek(storage::scaleBytes(start, sizeof(Alias))); acceptedAliases->read(values.values.data(), count);
            io.aliasBytes += count * sizeof(Alias); acceptedAliasBytes += values.values.capacity() * sizeof(Alias) + sizeof(AliasPage) + 64;
            found = acceptedPages.emplace(page, std::move(values)).first;
        }
        found->second.tick = ++aliasTick; return found->second.values[size_t(id - start)];
    }
    void requireRoot(uint64_t id) const { if (alias(id).root != id) throw std::invalid_argument("canonical scene rejects alias XYZ access"); }
    void writable() const { if (readOnly) throw std::logic_error("canonical read-only scene rejects mutation"); }
    void editing() const { writable(); if (!edit) throw std::logic_error("canonical scene mutation requires candidate transaction"); }
    Snapshot inspect(const Path& directory, const std::string& expected = {}, const PointProducer* virtualPoints = nullptr,
                     const std::string& virtualHash = {}) const {
        if (!std::filesystem::is_directory(directory) || std::filesystem::is_symlink(directory)) throw std::runtime_error("canonical snapshot directory invalid");
        if (std::filesystem::file_size(directory / "snapshot.bin") > budget / 8) throw std::length_error("canonical snapshot manifest exceeds budget");
        const auto identity = digest(directory / "snapshot.bin"); if (!expected.empty() && identity != expected) throw std::runtime_error("canonical snapshot manifest checksum mismatch");
        Reader in(directory / "snapshot.bin", 65536); if (in.scalar<uint64_t>() != sceneMagic) throw std::runtime_error("unsupported canonical snapshot magic");
        Snapshot snapshot; snapshot.schema = in.scalar<uint32_t>(); if (snapshot.schema != 1 && snapshot.schema != 2 && snapshot.schema != 3) throw std::runtime_error("unsupported canonical snapshot schema");
        snapshot.directory = directory; snapshot.identity = identity;
        snapshot.generation = in.scalar<uint64_t>(); snapshot.pointSlots = in.scalar<uint64_t>(); snapshot.canonicalPoints = in.scalar<uint64_t>();
        snapshot.observations = in.scalar<uint64_t>(); snapshot.phase = in.string(); snapshot.binding = readBinding(in, budget);
        const std::vector<std::string> names = snapshot.schema == 1 ? std::vector<std::string>{"aliases.bin", "points.bin", "poses.bin"} : std::vector<std::string>{"aliases.bin", "points.bin", "poses.bin", "kinds.bin"};
        for (const auto& name : names) {
            const auto hash = in.string(); const auto bytes = in.scalar<uint64_t>(); const auto path = directory / name;
            if (name == "aliases.bin") snapshot.aliasHash = hash; else if (name == "points.bin") snapshot.pointHash = hash; else if (name == "poses.bin") snapshot.poseHash = hash; else snapshot.kindHash = hash;
            if (name == "points.bin" && virtualPoints) {
                if (bytes != storage::scaleBytes(snapshot.pointSlots, sizeof(Vec3)) || hash != virtualHash) throw std::runtime_error("canonical archived step point hash/count differs");
            } else if (std::filesystem::is_symlink(path) || std::filesystem::file_size(path) != bytes || digest(path) != hash) throw std::runtime_error("canonical snapshot file checksum mismatch");
        }
        in.finish();
        if (snapshot.schema == 1) {
            if (std::filesystem::exists(directory / "kinds.bin")) throw std::runtime_error("legacy finite snapshot contains unbound landmark types");
            snapshot.kindHash = finiteKindsHash(snapshot.pointSlots);
        } else if (std::filesystem::file_size(directory / "kinds.bin") != snapshot.pointSlots) throw std::runtime_error("canonical kind layout mismatch");
        if (std::filesystem::file_size(directory / "aliases.bin") != storage::scaleBytes(snapshot.pointSlots, sizeof(Alias)) ||
            (!virtualPoints && std::filesystem::file_size(directory / "points.bin") != storage::scaleBytes(snapshot.pointSlots, sizeof(Vec3)))) throw std::runtime_error("canonical snapshot point layout mismatch");
        uint64_t expectedSlots = 0, expectedObservations = 0; std::set<std::string> labels;
        for (const auto& s : snapshot.binding.sources) {
            if (s.label.empty() || s.digest.empty() || !labels.insert(s.label).second) throw std::runtime_error("canonical snapshot source identity invalid");
            expectedSlots = storage::addBytes(expectedSlots, s.points); expectedObservations = storage::addBytes(expectedObservations, s.observations);
        }
        if (snapshot.binding.rawCatalog.empty() || snapshot.binding.topology.empty() || expectedSlots != snapshot.pointSlots || expectedObservations != snapshot.observations)
            throw std::runtime_error("canonical snapshot raw catalogue mismatch");
        const uint64_t bitmapBytes = storage::scaleBytes(snapshot.pointSlots / 64 + !!(snapshot.pointSlots % 64), sizeof(uint64_t)); guard(bitmapBytes + 5 * 65536);
        std::vector<uint64_t> roots(size_t(bitmapBytes / 8)); Reader a(directory / "aliases.bin", 65536);
        uint64_t rootCount = 0; std::unique_ptr<Reader> typeReader; if (snapshot.schema >= 2) typeReader = std::make_unique<Reader>(directory / "kinds.bin", 65536);
        auto validate = [&](uint64_t i, Vec3 value) {
            if (i % 65536 == 0) cancel::check(); const auto row = a.scalar<Alias>(); requireFinite(value);
            if (row.root > i || row.root >= snapshot.pointSlots) throw std::runtime_error("canonical alias root ordering invalid");
            const auto kind = typeReader ? typeReader->scalar<uint8_t>() : uint8_t(0);
            if (kind > (snapshot.schema >= 3 ? 2 : 1) || (row.root != i && kind)) throw std::runtime_error("canonical alias/landmark kind invalid");
            if (row.root == i) { validateLandmark({value, LandmarkKind(kind)}); roots[size_t(i / 64)] |= uint64_t(1) << (i % 64); ++rootCount; }
            else if (!(roots[size_t(row.root / 64)] & (uint64_t(1) << (row.root % 64))) || value.x != 0 || value.y != 0 || value.z != 0)
                throw std::runtime_error("canonical alias contains writable XYZ or nonroot mapping");
        };
        if (virtualPoints) {
            uint64_t next = 0; (*virtualPoints)([&](uint64_t i, Vec3 value) { if (i != next++ || i >= snapshot.pointSlots) throw std::runtime_error("canonical archived step point order differs"); validate(i, value); });
            if (next != snapshot.pointSlots) throw std::runtime_error("canonical archived step point coverage incomplete");
        } else {
            Reader p(directory / "points.bin", 65536); for (uint64_t i = 0; i < snapshot.pointSlots; ++i) validate(i, p.scalar<Vec3>()); p.finish();
        }
        a.finish(); if (typeReader) typeReader->finish(); if (rootCount != snapshot.canonicalPoints) throw std::runtime_error("canonical root count mismatch");
        const auto poseBytes = std::filesystem::file_size(directory / "poses.bin"); guard(storage::scaleBytes(poseBytes, 10));
        const auto model = model_store_detail::readModel(directory / "poses.bin"); guard(modelResidentBytes(model)); validatePoses(model);
        return snapshot;
    }
    struct StepReference {
        Snapshot base, proposal;
        Binding binding;
        std::string trialIdentity, aliasHash, pointHash, poseHash, kindHash;
        uint32_t schema = 1;
        uint64_t pointBytes = 0;
        double alpha = 0;
    };
    Path ownSnapshot(const std::string& name, const char* prefix) const {
        if (Path(name).filename() != Path(name) || name.find(prefix) != 0) throw std::runtime_error("canonical step reference path invalid");
        const auto path = root / name;
        if (std::filesystem::is_symlink(path) || std::filesystem::canonical(path).parent_path() != std::filesystem::canonical(root))
            throw std::runtime_error("canonical step reference outside owned scene");
        return path;
    }
    StepReference readStep(const Path& directory) const {
        if (std::filesystem::canonical(directory).parent_path() != std::filesystem::canonical(root) ||
            directory.filename().u8string().find("rejected-") != 0 || std::filesystem::is_symlink(directory))
            throw std::runtime_error("canonical archived step outside rejected scene snapshots");
        const auto path = directory / "step-reference.bin";
        if (std::filesystem::is_symlink(path) || std::filesystem::file_size(path) > budget / 16) throw std::runtime_error("canonical step reference framing invalid");
        const auto receiptPath = directory / "step-reference-ready.bin";
        if (std::filesystem::is_symlink(receiptPath) || std::filesystem::file_size(receiptPath) > 4096) throw std::runtime_error("canonical step receipt framing invalid");
        Reader receipt(receiptPath, 65536);
        if (receipt.scalar<uint64_t>() != stepMagic || receipt.scalar<uint32_t>() != 1) throw std::runtime_error("canonical step receipt schema invalid");
        const auto recipeHash = receipt.string(), trialHash = receipt.string(); const auto receiptAlpha = receipt.scalar<double>(); receipt.finish();
        if (digest(path) != recipeHash) throw std::runtime_error("canonical step producer recipe checksum mismatch");
        Reader in(path, 65536); if (in.scalar<uint64_t>() != stepMagic) throw std::runtime_error("canonical step reference magic invalid");
        const auto schema = in.scalar<uint32_t>(); if (schema != 1 && schema != 2) throw std::runtime_error("canonical step reference schema invalid");
        const auto baseName = in.string(), baseHash = in.string(), proposalName = in.string(), proposalHash = in.string();
        StepReference result; result.schema = schema; result.alpha = in.scalar<double>(); step::validateAlpha(result.alpha);
        result.trialIdentity = in.string(); result.aliasHash = in.string(); result.pointHash = in.string(); result.poseHash = in.string();
        if (schema == 2) result.kindHash = in.string();
        result.pointBytes = in.scalar<uint64_t>(); result.binding = readBinding(in, budget); in.finish();
        if (result.trialIdentity != trialHash || std::memcmp(&result.alpha, &receiptAlpha, sizeof(result.alpha)) != 0)
            throw std::runtime_error("canonical step producer alpha/trial receipt differs");
        result.base = inspect(ownSnapshot(baseName, "accepted-"), baseHash);
        result.proposal = inspect(ownSnapshot(proposalName, "rejected-"), proposalHash);
        if (!(result.binding == result.base.binding) || !(result.binding == result.proposal.binding) ||
            result.proposal.generation != result.base.generation + 1 || result.base.aliasHash != result.proposal.aliasHash ||
            result.aliasHash != result.base.aliasHash || result.base.kindHash != result.proposal.kindHash || (schema == 2 && result.kindHash != result.base.kindHash) || result.pointBytes != storage::scaleBytes(result.base.pointSlots, sizeof(Vec3)))
            throw std::runtime_error("canonical step reference immutable layout/base/proposal differs");
        return result;
    }
    PointProducer stepPoints(const StepReference& reference) const {
        return [this, reference](const PointEmit& emit) {
            guard(4 * 65536); Reader aliases(reference.base.directory / "aliases.bin", 65536),
                before(reference.base.directory / "points.bin", 65536), after(reference.proposal.directory / "points.bin", 65536);
            std::unique_ptr<Reader> types; if (reference.base.schema >= 2) types = std::make_unique<Reader>(reference.base.directory / "kinds.bin", 65536);
            for (uint64_t id = 0; id < reference.base.pointSlots; ++id) {
                if (!(id & 65535)) cancel::check(); const auto row = aliases.scalar<Alias>(); const auto a = before.scalar<Vec3>(), b = after.scalar<Vec3>();
                const auto kind = types ? LandmarkKind(types->scalar<uint8_t>()) : LandmarkKind::FiniteXYZ;
                emit(id, row.root == id ? interpolateLandmark({a, kind}, {b, kind}, reference.alpha).value : Vec3{});
            }
            aliases.finish(); before.finish(); after.finish(); if (types) types->finish();
        };
    }
    std::string expectedStepPoseHash(const StepReference& reference) const {
        const auto beforeBytes = std::filesystem::file_size(reference.base.directory / "poses.bin"), afterBytes = std::filesystem::file_size(reference.proposal.directory / "poses.bin");
        guard(storage::scaleBytes(storage::addBytes(beforeBytes, afterBytes), 20));
        auto before = model_store_detail::readModel(reference.base.directory / "poses.bin"), after = model_store_detail::readModel(reference.proposal.directory / "poses.bin");
        samePoseIdentity(before, after);
        if (reference.alpha < 1 && (!before.rigs.empty() || !after.rigs.empty())) throw std::runtime_error("canonical archived step requires rig-frame interpolation");
        for (auto& image : after.images) image.second.pose = step::interpolatePose(before.images.at(image.first).pose, image.second.pose, reference.alpha);
        for (auto& camera : after.cameras) camera.second = step::interpolateCamera(before.cameras.at(camera.first), camera.second, reference.alpha);
        DigestWriter hash; model_store_detail::serializeModel(after, hash); return hash.hash.hex();
    }
    Snapshot inspectStep(const Path& directory) const {
        const auto reference = readStep(directory); spirula::Sha256 hash; const auto producer = stepPoints(reference);
        producer([&](uint64_t, Vec3 value) { requireFinite(value); hash.update(reinterpret_cast<const uint8_t*>(&value), sizeof(value)); });
        const auto reconstructedHash = hash.hex();
        if (reconstructedHash != reference.pointHash || expectedStepPoseHash(reference) != reference.poseHash)
            throw std::runtime_error("canonical rejected step reconstructed XYZ/P/K hash differs");
        const auto snapshot = inspect(directory, reference.trialIdentity, &producer, reconstructedHash);
        if (!(snapshot.binding == reference.binding) || snapshot.generation != reference.proposal.generation ||
            snapshot.pointSlots != reference.base.pointSlots || snapshot.kindHash != reference.base.kindHash || snapshot.aliasHash != reference.aliasHash || snapshot.poseHash != reference.poseHash)
            throw std::runtime_error("canonical rejected step snapshot binding differs");
        return snapshot;
    }
    std::string headIdentity() const {
        if (!std::filesystem::exists(root / "head.bin")) return {};
        Reader in(root / "head.bin", 65536); if (in.scalar<uint64_t>() != headMagic) throw std::runtime_error("canonical head schema mismatch");
        const auto name = in.string(), hash = in.string(); in.finish();
        if (Path(name).filename() != Path(name) || name.find("accepted-") != 0) throw std::runtime_error("canonical head path invalid");
        if (digest(root / name / "snapshot.bin") != hash) throw std::runtime_error("canonical head identity mismatch"); return hash;
    }
    void requireHead() const {
        if (headIdentity() != (haveAccepted ? accepted.identity : std::string{})) throw std::runtime_error("canonical accepted head changed");
    }
    void restore() {
        std::unique_ptr<Lock> transaction;
        if (readOnly) readLease = std::make_unique<Lock>(root / "transaction.lock", true);
        else transaction = std::make_unique<Lock>(root / "transaction.lock");
        Reader in(root / "head.bin", 65536);
        if (in.scalar<uint64_t>() != headMagic) throw std::runtime_error("canonical head schema mismatch");
        const auto name = in.string(), hash = in.string(); in.finish();
        if (Path(name).filename() != Path(name) || name.find("accepted-") != 0) throw std::runtime_error("canonical head path invalid");
        current = inspect(root / name, hash); accepted = current; active = current.directory;
        poseTable = model_store_detail::readModel(active / "poses.bin"); poseBytes = modelResidentBytes(poseTable); haveAccepted = true; setOffsets();
        requireHead();
        for (const auto& entry : std::filesystem::directory_iterator(root)) if (!readOnly && entry.is_directory() && entry.path().filename().u8string().find("candidate-") == 0)
            std::filesystem::rename(entry.path(), root / ("interrupted-" + entry.path().filename().u8string() + "-" + token()));
    }
    void start() {
        writable();
        if (edit) throw std::logic_error("canonical candidate already active");
        lock = std::make_unique<Lock>(root / "transaction.lock");
        try { requireHead(); guard(modelResidentBytes(poseTable) + (4u << 20)); acceptedPoses = poseTable; savedPoseBytes = modelResidentBytes(acceptedPoses);
            interpolationReady = false; interpolationProposal = {}; interpolationPoseHash.clear();
            layoutEdit = false; current = accepted; current.identity.clear(); current.generation = haveAccepted ? accepted.generation + 1 : 0;
            current.directory = root / ("candidate-" + std::to_string(current.generation) + "-" + token());
            std::filesystem::create_directory(current.directory); active = current.directory; edit = true; clearReaders();
        } catch (...) { lock.reset(); throw; }
    }
    void startParameters(bool copyPoints) {
        if (!haveAccepted) throw std::logic_error("canonical scene uninitialized"); start();
        const auto bytes = std::filesystem::file_size(accepted.directory / "points.bin");
        storage::requireSpace(active, storage::addBytes(bytes, current.pointSlots));
        storage::retainFile(accepted.directory / "aliases.bin", active / "aliases.bin");
        if (copyPoints) { std::filesystem::copy_file(accepted.directory / "points.bin", active / "points.bin"); io.pointCopiedBytes += bytes; }
        if (accepted.schema >= 2) { storage::retainFile(accepted.directory / "kinds.bin", active / "kinds.bin"); current.schema = 3; }
        else writeKinds();
    }
    void writePoints(const PointProducer& producer) {
        editing(); guard(2u << 20); const auto part = active / ("points-" + token() + ".part");
        storage::requireSpace(active, storage::scaleBytes(current.pointSlots, sizeof(Vec3))); Reader in(active / "aliases.bin", 65536); Writer out(part);
        uint64_t next = 0, emitted = 0;
        auto aliasesThrough = [&](uint64_t target) {
            while (next < target) { const auto row = in.scalar<Alias>(); if (row.root == next) throw std::runtime_error("canonical producer omitted root XYZ"); out.scalar(Vec3{}); ++next; }
        };
        producer([&](uint64_t id, Vec3 value) {
            validateLandmark({value, kind(id)}); if (id < next || id >= current.pointSlots) throw std::runtime_error("canonical producer point order invalid"); aliasesThrough(id);
            if (in.scalar<Alias>().root != id) throw std::runtime_error("canonical producer emitted alias XYZ"); out.scalar(value); ++next; ++emitted;
        });
        aliasesThrough(current.pointSlots); in.finish(); if (emitted != current.canonicalPoints) throw std::runtime_error("canonical producer root count mismatch");
        out.finish(); clearReaders(); storage::replaceFile(part, active / "points.bin"); io.pointWrittenBytes += storage::scaleBytes(current.pointSlots, sizeof(Vec3));
    }
    Snapshot seal(const std::string& phase) {
        editing(); clearReaders(); validatePoses(poseTable); guard(2u << 20);
        model_store_detail::writeModel(poseTable, active / "poses.bin");
        Writer out(active / "snapshot.part"); out.scalar(sceneMagic); out.scalar<uint32_t>(3); out.scalar(current.generation);
        out.scalar(current.pointSlots); out.scalar(current.canonicalPoints); out.scalar(current.observations); out.string(phase); writeBinding(out, current.binding);
        const std::array<std::string, 4> names{"aliases.bin", "points.bin", "poses.bin", "kinds.bin"};
        const std::array<std::string, 4> hashes{digest(active / names[0]), digest(active / names[1]), digest(active / names[2]), digest(active / names[3])};
        if (!current.aliasHash.empty() && hashes[0] != current.aliasHash) throw std::runtime_error("canonical candidate mutated immutable alias layout");
        requireAcceptedKinds();
        if (haveAccepted && !layoutEdit && hashes[3] != accepted.kindHash) throw std::runtime_error("ordinary candidate changed immutable landmark kinds");
        if (haveAccepted && layoutEdit) {
            Reader candidateTypes(active / "kinds.bin", 65536); std::unique_ptr<Reader> oldTypes;
            if (accepted.schema >= 2) oldTypes = std::make_unique<Reader>(accepted.directory / "kinds.bin", 65536);
            for (uint64_t id = 0; id < accepted.pointSlots; ++id) {
                const auto old = oldTypes ? oldTypes->scalar<uint8_t>() : uint8_t(0);
                if (candidateTypes.scalar<uint8_t>() != old) throw std::runtime_error("layout candidate changed an accepted landmark kind");
            }
            if (oldTypes) oldTypes->finish();
        }
        for (size_t file = 0; file < names.size(); ++file) { out.string(hashes[file]); out.scalar<uint64_t>(std::filesystem::file_size(active / names[file])); }
        out.finish(); storage::replaceFile(active / "snapshot.part", active / "snapshot.bin");
        current = inspect(active); return current;
    }
};
CanonicalScene::CanonicalScene(const Path& root, size_t budget, OpenMode mode) : impl_(std::make_unique<Impl>()) {
    impl_->root = std::filesystem::absolute(root);
    impl_->readOnly = mode == OpenMode::ReadOnly;
    if (!budget) {
        const auto available = spirula::availableRamBytes(), physical = spirula::physicalRamBytes();
        if (!available || !physical) throw std::runtime_error("canonical auto budget needs actual hardware memory");
        budget = std::min<size_t>(physical / 5 * 4, available - std::min<size_t>(size_t(4) << 30, available / 4));
    }
    if (budget < (16u << 20)) throw std::length_error("canonical scene requires 16 MiB budget"); impl_->budget = budget;
    if (impl_->readOnly) {
        if (!std::filesystem::is_directory(impl_->root) || !std::filesystem::is_regular_file(impl_->root / "head.bin"))
            throw std::runtime_error("canonical read-only scene requires an existing accepted head");
    } else std::filesystem::create_directories(impl_->root);
    if (std::filesystem::exists(impl_->root / "head.bin")) impl_->restore();
}
CanonicalScene::~CanonicalScene() = default;
CanonicalScene::Binding CanonicalScene::bindingFor(const RegionalWorkingStore& catalog, const std::string& topologyIdentity, bool verifyImmutable, size_t maxSources) {
    if (verifyImmutable) catalog.verifyImmutable(); Binding binding; binding.topology = topologyIdentity; DigestWriter out; const auto count = std::min(catalog.sourceCount(), maxSources); out.scalar<uint64_t>(count);
    for (size_t s = 0; s < count; ++s) {
        SourceIdentity row{catalog.sourceLabel(s), catalog.sourceIdentity(s) + "." + catalog.sourceRawIdentity(s), catalog.sourcePointCount(s), catalog.sourceObservationCount(s)};
        out.string(row.label); out.string(row.digest); out.scalar(row.points); out.scalar(row.observations); binding.sources.push_back(std::move(row));
    }
    binding.rawCatalog = out.hash.hex(); return binding;
}
bool CanonicalScene::initialized() const { return impl_->haveAccepted; }
bool CanonicalScene::editing() const { return impl_->edit; }
const Reconstruction& CanonicalScene::poses() const { return impl_->poseTable; }
const Reconstruction& CanonicalScene::acceptedPoses() const { if (!initialized()) throw std::logic_error("canonical accepted poses unavailable"); return editing() ? impl_->acceptedPoses : impl_->poseTable; }
const CanonicalScene::Binding& CanonicalScene::binding() const { return impl_->current.binding; }
const CanonicalScene::Snapshot& CanonicalScene::acceptedSnapshot() const { if (!initialized()) throw std::logic_error("canonical scene has no accepted snapshot"); return impl_->accepted; }
CanonicalScene::Path CanonicalScene::forkAccepted(const Path& destination) const {
    if (!initialized() || editing()) throw std::logic_error("canonical fork requires an idle accepted snapshot");
    impl_->requireHead();
    const Path source = std::filesystem::canonical(impl_->root), target = std::filesystem::weakly_canonical(std::filesystem::absolute(destination));
    const auto relative = target.lexically_relative(source);
    if (target == source || (!relative.empty() && *relative.begin() != Path("..")) || std::filesystem::exists(target))
        throw std::invalid_argument("canonical fork requires a new destination outside the source root");
    for (Path parent = target.parent_path(); !parent.empty();) {
        if (std::filesystem::exists(parent) && std::filesystem::equivalent(parent, source))
            throw std::invalid_argument("canonical fork destination is physically inside its source root");
        const Path next = parent.parent_path(); if (next == parent) break; parent = next;
    }
    const auto accepted = impl_->inspect(impl_->accepted.directory, impl_->accepted.identity);
    std::filesystem::create_directories(target.parent_path());
    const Path stage = target.parent_path() / (target.filename().u8string() + ".fork-part-" + token());
    std::filesystem::create_directory(stage); std::filesystem::create_directory(stage / accepted.directory.filename());
    for (const char* file : {"snapshot.bin", "aliases.bin", "points.bin", "poses.bin"})
        std::filesystem::create_hard_link(accepted.directory / file, stage / accepted.directory.filename() / file);
    if (accepted.schema >= 2) std::filesystem::create_hard_link(accepted.directory / "kinds.bin", stage / accepted.directory.filename() / "kinds.bin");
    Writer head(stage / "head.bin"); head.scalar(headMagic); head.string(accepted.directory.filename().u8string()); head.string(accepted.identity); head.finish();
    const auto verified = impl_->inspect(stage / accepted.directory.filename(), accepted.identity);
    if (!(verified.binding == accepted.binding) || verified.observations != accepted.observations) throw std::runtime_error("canonical fork binding or multiset changed");
    impl_->requireHead(); std::filesystem::rename(stage, target); return target;
}
CanonicalScene::Snapshot CanonicalScene::inspectRetainedSnapshot(const Path& directory) const {
    const auto path = std::filesystem::canonical(directory), root = std::filesystem::canonical(impl_->root);
    const auto name = path.filename().u8string();
    if (std::filesystem::is_symlink(directory) || path.parent_path() != root ||
        (name.find("accepted-") != 0 && name.find("rejected-") != 0))
        throw std::invalid_argument("canonical retained snapshot must be a direct owned accepted or rejected directory");
    return std::filesystem::exists(path / "points.bin") ? impl_->inspect(path) : impl_->inspectStep(path);
}
uint64_t CanonicalScene::generation() const { return impl_->current.generation; }
const std::string& CanonicalScene::identity() const { return impl_->current.identity; }
size_t CanonicalScene::sourceCount() const { return binding().sources.size(); }
uint64_t CanonicalScene::pointSlots() const { return impl_->current.pointSlots; }
uint64_t CanonicalScene::canonicalPoints() const { return impl_->current.canonicalPoints; }
uint64_t CanonicalScene::observations() const { return impl_->current.observations; }
uint64_t CanonicalScene::canonical(PointKey key) const {
    if (key.source >= sourceCount() || key.ordinal >= binding().sources[size_t(key.source)].points) throw std::out_of_range("canonical source point identity");
    const auto row = impl_->alias(storage::addBytes(impl_->offsets[size_t(key.source)], key.ordinal));
    if (row.local != key.localPoint) throw std::runtime_error("canonical source local point identity changed"); return row.root;
}
uint64_t CanonicalScene::canonicalGlobal(uint64_t id) const { return impl_->alias(id).root; }
bool CanonicalScene::isCanonical(uint64_t id) const { return impl_->alias(id).root == id; }
CanonicalLandmark CanonicalScene::landmark(uint64_t id) const {
    impl_->requireRoot(id); auto& in = impl_->xyzReader(); in.seek(storage::scaleBytes(id, sizeof(Vec3))); auto value = in.scalar<Vec3>();
    impl_->io.pointReadBytes += sizeof(Vec3); CanonicalLandmark result{value, impl_->kind(id)}; validateLandmark(result); return result;
}
Vec3 CanonicalScene::pointXYZ(uint64_t id) const { const auto value = landmark(id); if (value.kind != LandmarkKind::FiniteXYZ) throw std::logic_error("nonmetric landmark has no finite XYZ"); return value.value; }
std::string CanonicalScene::landmarkKindsHash() const { if (impl_->current.schema == 1) return impl_->current.kindHash; if (impl_->current.kindHash.empty()) impl_->current.kindHash = digest(impl_->active / "kinds.bin"); return impl_->current.kindHash; }
Vec3 CanonicalScene::xyz(PointKey key) const { return pointXYZ(canonical(key)); }
uint64_t CanonicalScene::acceptedCanonical(PointKey key) const {
    const auto& saved = acceptedSnapshot();
    if (key.source >= saved.binding.sources.size() || key.ordinal >= saved.binding.sources[size_t(key.source)].points) throw std::out_of_range("accepted source point identity");
    uint64_t global = key.ordinal; for (size_t source = 0; source < key.source; ++source) global = storage::addBytes(global, saved.binding.sources[source].points);
        const auto row = impl_->acceptedAlias(global);
    if (row.local != key.localPoint) throw std::runtime_error("accepted source local ID changed"); return row.root;
}
CanonicalLandmark CanonicalScene::acceptedLandmark(uint64_t id) const {
    const auto& saved = acceptedSnapshot(); if (id >= saved.pointSlots) throw std::out_of_range("accepted canonical point index");
        if (impl_->acceptedAlias(id).root != id) throw std::invalid_argument("accepted alias XYZ access rejected");
    if (!impl_->acceptedXYZ) { impl_->guard(65536); impl_->acceptedXYZ = std::make_unique<Reader>(saved.directory / "points.bin", 65536); ++impl_->io.fileOpens; }
    impl_->acceptedXYZ->seek(storage::scaleBytes(id, sizeof(Vec3))); auto value = impl_->acceptedXYZ->scalar<Vec3>(); impl_->io.pointReadBytes += sizeof(Vec3); CanonicalLandmark result{value, impl_->kind(id, true)}; validateLandmark(result); return result;
}
Vec3 CanonicalScene::acceptedPointXYZ(uint64_t id) const { const auto value = acceptedLandmark(id); if (value.kind != LandmarkKind::FiniteXYZ) throw std::logic_error("accepted nonmetric landmark has no finite XYZ"); return value.value; }
void CanonicalScene::visitCanonicalLandmarks(const LandmarkEmit& visit) const {
    Reader a(impl_->active / "aliases.bin", 65536), p(impl_->active / "points.bin", 65536); impl_->io.fileOpens += 2;
    std::unique_ptr<Reader> types; if (impl_->current.schema >= 2) types = std::make_unique<Reader>(impl_->active / "kinds.bin", 65536);
    for (uint64_t id = 0; id < pointSlots(); ++id) { if (!(id & 65535)) cancel::check(); const auto row = a.scalar<Alias>(); const auto xyz = p.scalar<Vec3>();
        const auto kind = types ? LandmarkKind(types->scalar<uint8_t>()) : LandmarkKind::FiniteXYZ;
        impl_->io.aliasBytes += sizeof(Alias); impl_->io.pointReadBytes += sizeof(Vec3);
        if (row.root == id) { CanonicalLandmark value{xyz, kind}; validateLandmark(value); visit(id, value); } }
    a.finish(); p.finish(); if (types) types->finish();
}
void CanonicalScene::visitSourceAliases(const AliasEmit& visit) const {
    if (!visit || impl_->offsets.size() != sourceCount() + 1 || impl_->offsets.back() != pointSlots())
        throw std::runtime_error("canonical source alias layout is incomplete");
    const auto bitmapBytes = storage::scaleBytes(pointSlots() / 64 + !!(pointSlots() % 64), sizeof(uint64_t));
    impl_->guard(storage::addBytes(bitmapBytes, 65536));
    std::vector<uint64_t> roots(size_t(bitmapBytes / sizeof(uint64_t)), 0);
    Reader in(impl_->active / "aliases.bin", 65536); ++impl_->io.fileOpens;
    uint64_t global = 0, rootCount = 0;
    for (size_t source = 0; source < sourceCount(); ++source) {
        if (impl_->offsets[source] != global) throw std::runtime_error("canonical source alias offset differs");
        uint64_t previousLocal = 0; bool haveLocal = false;
        for (uint64_t ordinal = 0; ordinal < binding().sources[source].points; ++ordinal, ++global) {
            if (!(global & 65535)) cancel::check(); const auto row = in.scalar<Alias>(); impl_->io.aliasBytes += sizeof(Alias);
            if (row.root > global || row.root >= pointSlots() || row.local == kInvalidPoint3D || (haveLocal && row.local <= previousLocal))
                throw std::runtime_error("canonical source alias root/local metadata invalid");
            previousLocal = row.local; haveLocal = true;
            if (row.root == global) { roots[size_t(global / 64)] |= uint64_t(1) << (global % 64); ++rootCount; }
            else if (!(roots[size_t(row.root / 64)] & (uint64_t(1) << (row.root % 64)))) throw std::runtime_error("canonical source alias refers to a noncanonical root");
            visit(PointKey{uint64_t(source), ordinal, row.local}, row.root);
        }
    }
    in.finish(); if (global != pointSlots() || rootCount != canonicalPoints()) throw std::runtime_error("canonical source alias coverage differs");
}
void CanonicalScene::visitCanonicalPoints(const PointEmit& visit) const {
    visitCanonicalLandmarks([&](uint64_t id, CanonicalLandmark value) { if (value.kind != LandmarkKind::FiniteXYZ) throw std::logic_error("finite XYZ visitor encountered a direction"); visit(id, value.value); });
}
void CanonicalScene::requireBinding(const Binding& expected) const {
    const auto& current = binding(); if (current == expected) return;
    std::ostringstream error; error << "canonical scene immutable raw/topology binding changed: accepted_topology=" << current.topology
        << "; requested_topology=" << expected.topology << "; accepted_raw=" << current.rawCatalog
        << "; requested_raw=" << expected.rawCatalog << "; accepted_sources=" << current.sources.size()
        << "; requested_sources=" << expected.sources.size();
    for (size_t source = 0; source < std::min(current.sources.size(), expected.sources.size()); ++source)
        if (!(current.sources[source] == expected.sources[source])) error << "; changed_source=" << source << ':' << current.sources[source].label;
    throw std::runtime_error(error.str());
}
Reconstruction CanonicalScene::materialize(const RegionalWorkingStore& catalog, size_t source) const {
    if (source >= sourceCount()) throw std::out_of_range("canonical materialization source not accepted");
    requireBinding(bindingFor(catalog, binding().topology, false, sourceCount()));
    auto out = catalog.materialize(source, [&](PointKey key, Vec3) { return xyz(key); }, true);
    for (auto& image : out.images) { const auto& value = poses().images.at(image.first); image.second.pose = value.pose; image.second.registered = value.registered; }
    for (auto& camera : out.cameras) camera.second = poses().cameras.at(camera.first);
    return out;
}
void CanonicalScene::initialize(const Binding& binding, const Reconstruction& poses, const AliasProducer& aliases, const PointProducer& seed, const std::string& phase) {
    if (initialized()) throw std::logic_error("canonical scene already initialized"); beginLayoutCandidate(binding, poses, aliases, seed); commit(phase);
}
void CanonicalScene::initializeValidated(const Binding& binding, const Reconstruction& poses, const AliasProducer& aliases,
                                         const PointProducer& seed, const std::function<void(const CanonicalScene&)>& validator,
                                         const std::string& phase) {
    if (initialized() || !validator) throw std::logic_error("canonical initial acceptance requires an uninitialized scene and full validator");
    try { beginLayoutCandidate(binding, poses, aliases, seed); validator(*this); commit(phase); }
    catch (...) { const auto failure = std::current_exception(); if (editing()) abort("initial-validation-rejected"); std::rethrow_exception(failure); }
}
void CanonicalScene::beginLayoutCandidate(const Binding& binding, const Reconstruction& poses, const AliasProducer& aliases, const PointProducer& seed) {
    validatePoses(poses);
    if (initialized()) {
        if (binding.sources.size() < impl_->accepted.binding.sources.size()) throw std::invalid_argument("canonical layout removed accepted source identities");
        for (size_t source = 0; source < impl_->accepted.binding.sources.size(); ++source)
            if (!(binding.sources[source] == impl_->accepted.binding.sources[source])) throw std::invalid_argument("canonical layout changed accepted source identities");
    }
    impl_->guard(modelResidentBytes(poses) + (4u << 20)); impl_->start(); impl_->layoutEdit = true;
    impl_->current.binding = binding; impl_->current.aliasHash.clear(); impl_->current.pointSlots = 0; impl_->current.canonicalPoints = 0; impl_->current.observations = 0;
    for (const auto& s : binding.sources) { impl_->current.pointSlots = storage::addBytes(pointSlots(), s.points); impl_->current.observations = storage::addBytes(observations(), s.observations); }
    impl_->poseTable = poses; impl_->poseBytes = modelResidentBytes(poses); impl_->aliasPages.clear(); impl_->aliasBytes = 0; impl_->setOffsets();
    storage::requireSpace(impl_->active, storage::scaleBytes(pointSlots(), sizeof(Alias) + sizeof(Vec3) + sizeof(uint8_t)));
    Writer out(impl_->active / "aliases.bin"); uint64_t next = 0;
    aliases([&](PointKey key, uint64_t root) {
        if (key.source >= sourceCount() || key.ordinal >= binding.sources[size_t(key.source)].points || impl_->offsets[size_t(key.source)] + key.ordinal != next || root > next)
            throw std::runtime_error("canonical alias producer identity/order invalid");
        out.scalar(Alias{root, key.localPoint}); if (root == next) ++impl_->current.canonicalPoints; ++next;
    });
    if (next != pointSlots()) throw std::runtime_error("canonical alias producer incomplete"); out.finish(); impl_->writeKinds(); impl_->writePoints(seed);
}
void CanonicalScene::beginCandidate() { impl_->startParameters(true); }
void CanonicalScene::beginInterpolatedCandidate(const Snapshot& proposal, double alpha) {
    if (editing()) throw std::logic_error("canonical candidate already active");
    step::validateAlpha(alpha);
    try { impl_->startParameters(false); interpolateCandidate(proposal, alpha); }
    catch (...) { const auto failure = std::current_exception(); if (editing()) abort("interpolation-initialization-rejected"); std::rethrow_exception(failure); }
}
void CanonicalScene::setPose(uint32_t image, const Pose& pose) { impl_->editing(); impl_->interpolationReady = false; step::validatePose(pose); impl_->poseTable.images.at(image).pose = pose; impl_->current.identity.clear(); }
void CanonicalScene::setCamera(const Camera& camera) {
    impl_->editing(); impl_->interpolationReady = false; const auto& before = poses().cameras.at(camera.id); (void)step::interpolateCamera(before, camera, 1); impl_->poseTable.cameras.at(camera.id) = camera; impl_->current.identity.clear();
}
void CanonicalScene::applyPointUpdates(const std::vector<std::pair<uint64_t, Vec3>>& updates) {
    impl_->editing(); impl_->interpolationReady = false; impl_->guard(storage::scaleBytes(updates.size(), sizeof(updates.front())));
    uint64_t previous = 0; bool first = true;
    for (const auto& update : updates) { if (!first && update.first <= previous) throw std::invalid_argument("canonical XYZ batch must be unique and sorted"); impl_->requireRoot(update.first); if (impl_->kind(update.first) != LandmarkKind::FiniteXYZ) throw std::logic_error("finite XYZ update attempted on a direction"); requireFinite(update.second); previous = update.first; first = false; }
    impl_->clearReaders(); impl_->current.identity.clear(); std::fstream file(impl_->active / "points.bin", std::ios::in | std::ios::out | std::ios::binary); if (!file) throw std::runtime_error("canonical XYZ writer open failed");
    for (const auto& update : updates) { file.seekp(std::streamoff(storage::scaleBytes(update.first, sizeof(Vec3)))); file.write(reinterpret_cast<const char*>(&update.second), sizeof(Vec3)); impl_->io.pointWrittenBytes += sizeof(Vec3); }
    file.flush(); if (!file) throw std::runtime_error("canonical XYZ batch write failed"); impl_->current.identity.clear();
}
void CanonicalScene::applyLandmarkUpdates(const std::vector<std::pair<uint64_t, CanonicalLandmark>>& updates) {
    impl_->editing(); impl_->guard(storage::scaleBytes(updates.size(), sizeof(updates.front())));
    uint64_t previous = 0; bool first = true;
    for (const auto& update : updates) { if (!first && update.first <= previous) throw std::invalid_argument("canonical landmark batch must be sorted and unique");
        impl_->requireRoot(update.first); validateLandmark(update.second);
        if (update.second.kind != impl_->kind(update.first)) throw std::logic_error("ordinary update changed landmark kind"); previous = update.first; first = false; }
    impl_->clearReaders(); impl_->interpolationReady = false; impl_->current.identity.clear();
    std::fstream file(impl_->active / "points.bin", std::ios::in | std::ios::out | std::ios::binary); if (!file) throw std::runtime_error("canonical landmark writer open failed");
    for (const auto& update : updates) { file.seekp(std::streamoff(storage::scaleBytes(update.first, sizeof(Vec3)))); file.write(reinterpret_cast<const char*>(&update.second.value), sizeof(Vec3)); impl_->io.pointWrittenBytes += sizeof(Vec3); }
    file.flush(); if (!file) throw std::runtime_error("canonical landmark batch write failed");
}
void CanonicalScene::applyLandmarkTypes(const std::vector<std::pair<uint64_t, Vec3>>& updates) {
    impl_->editing(); if (!impl_->layoutEdit) throw std::logic_error("landmark classification requires a layout candidate");
    impl_->guard(storage::scaleBytes(updates.size(), sizeof(updates.front()))); uint64_t previous = 0; bool first = true;
    for (const auto& update : updates) {
        if (!first && update.first <= previous) throw std::invalid_argument("landmark classification must be sorted and unique");
        impl_->requireRoot(update.first); if ((initialized() && update.first < impl_->accepted.pointSlots) || impl_->kind(update.first) != LandmarkKind::FiniteXYZ)
            throw std::logic_error("landmark classification cannot change an accepted kind");
        validateLandmark({update.second, LandmarkKind::UnitDirection}); previous = update.first; first = false;
    }
    impl_->clearReaders(); impl_->interpolationReady = false; impl_->current.identity.clear(); impl_->current.kindHash.clear();
    std::fstream values(impl_->active / "points.bin", std::ios::in | std::ios::out | std::ios::binary), types(impl_->active / "kinds.bin", std::ios::in | std::ios::out | std::ios::binary);
    if (!values || !types) throw std::runtime_error("canonical landmark classification writer open failed");
    for (const auto& update : updates) { const uint8_t kind = 1; values.seekp(std::streamoff(storage::scaleBytes(update.first, sizeof(Vec3)))); values.write(reinterpret_cast<const char*>(&update.second), sizeof(Vec3)); types.seekp(std::streamoff(update.first)); types.write(reinterpret_cast<const char*>(&kind), 1); }
    values.flush(); types.flush(); if (!values || !types) throw std::runtime_error("canonical landmark classification write failed");
}
void CanonicalScene::applyRetainedObservations(const std::vector<uint64_t>& updates) {
    impl_->editing(); if (!impl_->layoutEdit) throw std::logic_error("retained classification requires a layout candidate");
    impl_->guard(storage::scaleBytes(updates.size(), sizeof(uint64_t))); uint64_t previous = 0; bool first = true;
    for (const auto id : updates) {
        if (!first && id <= previous) throw std::invalid_argument("retained classification must be sorted and unique");
        impl_->requireRoot(id);
        if ((initialized() && id < impl_->accepted.pointSlots) || impl_->kind(id) != LandmarkKind::FiniteXYZ)
            throw std::logic_error("retained classification cannot change an accepted or classified kind");
        previous = id; first = false;
    }
    impl_->clearReaders(); impl_->interpolationReady = false; impl_->current.identity.clear(); impl_->current.kindHash.clear(); impl_->current.schema = 3;
    std::fstream values(impl_->active / "points.bin", std::ios::in | std::ios::out | std::ios::binary), types(impl_->active / "kinds.bin", std::ios::in | std::ios::out | std::ios::binary);
    if (!values || !types) throw std::runtime_error("canonical retained classification writer open failed");
    const Vec3 zero{}; const uint8_t kind = uint8_t(LandmarkKind::RetainedObservation);
    for (const auto id : updates) { values.seekp(std::streamoff(storage::scaleBytes(id, sizeof(Vec3)))); values.write(reinterpret_cast<const char*>(&zero), sizeof(zero)); types.seekp(std::streamoff(id)); types.write(reinterpret_cast<const char*>(&kind), 1); }
    values.flush(); types.flush(); if (!values || !types) throw std::runtime_error("canonical retained classification write failed");
}
void CanonicalScene::rewriteCanonicalPoints(const PointProducer& producer) { impl_->editing(); impl_->interpolationReady = false; impl_->writePoints(producer); impl_->current.identity.clear(); }
CanonicalScene::Snapshot CanonicalScene::sealCandidate(const std::string& phase) { return impl_->seal(phase); }
CanonicalScene::Snapshot CanonicalScene::commit(const std::string& phase) {
    impl_->editing(); impl_->requireHead(); impl_->seal(phase); impl_->clearReaders();
    const auto final = impl_->root / ("accepted-" + std::to_string(generation()) + "-" + token());
    std::filesystem::rename(impl_->active, final); impl_->active = final; impl_->current.directory = final;
    Writer head(impl_->root / "head.part"); head.scalar(headMagic); head.string(final.filename().u8string()); head.string(impl_->current.identity); head.finish();
    storage::replaceFile(impl_->root / "head.part", impl_->root / "head.bin"); impl_->accepted = impl_->current; impl_->acceptedPoses = {}; impl_->savedPoseBytes = 0; impl_->acceptedAliases.reset(); impl_->acceptedXYZ.reset(); impl_->acceptedKinds.reset(); impl_->acceptedPages.clear(); impl_->acceptedAliasBytes = 0;
    impl_->haveAccepted = true; impl_->edit = false; impl_->lock.reset(); return impl_->accepted;
}
CanonicalScene::Path CanonicalScene::abort(const std::string& reason) {
    impl_->editing(); impl_->clearReaders();
    Path preserved = impl_->active;
    try { if (impl_->current.identity.empty() || !std::filesystem::exists(impl_->active / "snapshot.bin")) impl_->seal(reason); } catch (...) { impl_->clearReaders(); }
    const auto target = impl_->root / ("rejected-" + std::to_string(generation()) + "-" + token());
    std::filesystem::rename(preserved, target); impl_->poseTable = std::move(impl_->acceptedPoses); impl_->poseBytes = modelResidentBytes(impl_->poseTable); impl_->savedPoseBytes = 0; impl_->aliasPages.clear(); impl_->aliasBytes = 0; impl_->current = impl_->accepted;
    impl_->active = impl_->accepted.directory; impl_->edit = false; impl_->interpolationReady = false; impl_->interpolationProposal = {}; impl_->interpolationPoseHash.clear(); impl_->lock.reset(); if (initialized()) impl_->setOffsets(); return target;
}
CanonicalScene::Path CanonicalScene::preserveInterpolatedRejection(const Snapshot& proposal, double alpha, const std::string& reason) {
    impl_->editing();
    if (!initialized() || !impl_->interpolationReady || proposal.identity != impl_->interpolationProposal.identity ||
        std::memcmp(&alpha, &impl_->interpolationAlpha, sizeof(alpha)) != 0 || proposal.directory != impl_->interpolationProposal.directory)
        return abort(reason + "-full-preserved");
    bool proven = false;
    try {
        const auto base = impl_->accepted, full = impl_->inspect(proposal.directory, proposal.identity);
        const auto trial = impl_->current.identity.empty() ? sealCandidate(reason) : impl_->inspect(impl_->active, impl_->current.identity);
        Impl::StepReference reference; reference.base = base; reference.proposal = full; reference.alpha = alpha;
        reference.trialIdentity = trial.identity; reference.aliasHash = trial.aliasHash; reference.pointHash = trial.pointHash;
        reference.poseHash = trial.poseHash; reference.kindHash = trial.kindHash; reference.schema = 2; reference.binding = trial.binding; reference.pointBytes = storage::scaleBytes(trial.pointSlots, sizeof(Vec3));
        if (!(trial.binding == base.binding) || !(trial.binding == full.binding) || base.aliasHash != full.aliasHash ||
            trial.aliasHash != base.aliasHash || base.kindHash != full.kindHash || trial.kindHash != base.kindHash || trial.generation != base.generation + 1 || full.generation != trial.generation ||
            trial.poseHash != impl_->interpolationPoseHash || impl_->expectedStepPoseHash(reference) != trial.poseHash)
            throw std::runtime_error("rejected interpolation metadata changed; retain full candidate");
        const auto producer = impl_->stepPoints(reference); Reader actual(impl_->active / "points.bin", 65536); spirula::Sha256 hash;
        producer([&](uint64_t, Vec3 expected) {
            const auto value = actual.scalar<Vec3>(); if (std::memcmp(&value, &expected, sizeof(value)) != 0) throw std::runtime_error("rejected interpolation XYZ changed; retain full candidate");
            hash.update(reinterpret_cast<const uint8_t*>(&expected), sizeof(expected));
        }); actual.finish();
        if (hash.hex() != trial.pointHash) throw std::runtime_error("rejected interpolation full point SHA differs");
        Writer out(impl_->active / "step-reference.part"); out.scalar(stepMagic); out.scalar<uint32_t>(2);
        out.string(base.directory.filename().u8string()); out.string(base.identity); out.string(full.directory.filename().u8string()); out.string(full.identity);
        out.scalar(alpha); out.string(trial.identity); out.string(trial.aliasHash); out.string(trial.pointHash); out.string(trial.poseHash);
        out.string(trial.kindHash); out.scalar(reference.pointBytes); writeBinding(out, trial.binding); out.finish();
        storage::replaceFile(impl_->active / "step-reference.part", impl_->active / "step-reference.bin");
        Writer receipt(impl_->active / "step-reference-ready.part"); receipt.scalar(stepMagic); receipt.scalar<uint32_t>(1);
        receipt.string(digest(impl_->active / "step-reference.bin")); receipt.string(trial.identity); receipt.scalar(alpha); receipt.finish();
        storage::replaceFile(impl_->active / "step-reference-ready.part", impl_->active / "step-reference-ready.bin"); proven = true;
    } catch (const std::exception& error) {
        proven = false; try { std::ofstream note(impl_->active / "step-reference-refusal.txt"); note << error.what() << '\n'; } catch (...) { }
    } catch (...) { proven = false; }
    const auto target = abort(reason + (proven ? "-exact-step-reference" : "-full-proof-fallback"));
    if (proven) {
        try {
            impl_->inspectStep(target); impl_->inspect(target);
            if (std::filesystem::canonical(target).parent_path() != std::filesystem::canonical(impl_->root) || target.filename().u8string().find("rejected-") != 0)
                throw std::runtime_error("rejected step owned removal target invalid");
            std::filesystem::remove(target / "points.bin");
        } catch (const std::exception& error) {
            try { std::ofstream note(target / "step-reference-refusal.txt"); note << error.what() << '\n'; } catch (...) { }
        } catch (...) { }
    }
    return target;
}
CanonicalScene::Snapshot CanonicalScene::verifyRejectedStep(const Path& directory) const {
    if (std::filesystem::canonical(directory).parent_path() != std::filesystem::canonical(impl_->root) || directory.filename().u8string().find("rejected-") != 0)
        throw std::runtime_error("rejected verification path outside scene");
    return std::filesystem::exists(directory / "points.bin") ? impl_->inspect(directory) : impl_->inspectStep(directory);
}
void CanonicalScene::visitRejectedStepLandmarks(const Path& directory, const LandmarkEmit& visit) const {
    const auto snapshot = verifyRejectedStep(directory); Reader aliases(directory / "aliases.bin", 65536);
    std::unique_ptr<Reader> types; if (snapshot.schema >= 2) types = std::make_unique<Reader>(directory / "kinds.bin", 65536);
    auto typed = [&](uint64_t id, Vec3 value) { const auto kind = types ? LandmarkKind(types->scalar<uint8_t>()) : LandmarkKind::FiniteXYZ; if (aliases.scalar<Alias>().root == id) visit(id, {value, kind}); };
    if (std::filesystem::exists(directory / "points.bin")) {
        Reader points(directory / "points.bin", 65536);
        for (uint64_t id = 0; id < snapshot.pointSlots; ++id) typed(id, points.scalar<Vec3>());
        points.finish();
    } else {
        const auto reference = impl_->readStep(directory); impl_->stepPoints(reference)(typed);
    }
    aliases.finish(); if (types) types->finish();
}
void CanonicalScene::visitRejectedStepPoints(const Path& directory, const PointEmit& visit) const {
    visitRejectedStepLandmarks(directory, [&](uint64_t id, CanonicalLandmark value) { if (value.kind != LandmarkKind::FiniteXYZ) throw std::logic_error("finite rejected visitor encountered a direction"); visit(id, value.value); });
}
void CanonicalScene::interpolateCandidate(const Snapshot& proposal, double alpha) {
    impl_->editing(); impl_->interpolationReady = false; step::validateAlpha(alpha); if (proposal.identity.empty()) throw std::runtime_error("canonical interpolation requires sealed proposal identity"); if (!initialized()) throw std::logic_error("canonical interpolation requires accepted baseline");
    const auto candidatePath = std::filesystem::canonical(proposal.directory), ownRoot = std::filesystem::canonical(impl_->root);
    if (candidatePath.parent_path() != ownRoot || candidatePath.filename().u8string().find("rejected-") != 0) throw std::runtime_error("canonical interpolation candidate outside rejected snapshots");
    const auto full = impl_->inspect(candidatePath, proposal.identity);
    if (!(full.binding == impl_->accepted.binding) || full.pointSlots != impl_->accepted.pointSlots || full.generation != impl_->accepted.generation + 1)
        throw std::runtime_error("canonical interpolation immutable layout mismatch");
    if (full.kindHash != impl_->accepted.kindHash) throw std::runtime_error("canonical interpolation landmark kinds changed");
    if (digest(candidatePath / "aliases.bin") != digest(impl_->accepted.directory / "aliases.bin")) throw std::runtime_error("canonical interpolation alias identity changed");
    auto proposedPoses = model_store_detail::readModel(candidatePath / "poses.bin"); samePoseIdentity(impl_->acceptedPoses, proposedPoses);
    if (alpha < 1 && (!impl_->acceptedPoses.rigs.empty() || !proposedPoses.rigs.empty())) throw std::runtime_error("canonical step requires rig-frame interpolation");
    impl_->writePoints([&](const PointEmit& emit) {
        Reader aliases(impl_->accepted.directory / "aliases.bin", 65536), before(impl_->accepted.directory / "points.bin", 65536), after(candidatePath / "points.bin", 65536);
        for (uint64_t id = 0; id < pointSlots(); ++id) { auto row = aliases.scalar<Alias>(); auto a = before.scalar<Vec3>(), b = after.scalar<Vec3>(); if (row.root == id) { const auto kind = impl_->kind(id, true); emit(id, interpolateLandmark({a, kind}, {b, kind}, alpha).value); } }
        aliases.finish(); before.finish(); after.finish();
    });
    for (auto& image : proposedPoses.images) image.second.pose = step::interpolatePose(impl_->acceptedPoses.images.at(image.first).pose, image.second.pose, alpha);
    for (auto& camera : proposedPoses.cameras) camera.second = step::interpolateCamera(impl_->acceptedPoses.cameras.at(camera.first), camera.second, alpha);
    impl_->poseTable = std::move(proposedPoses); impl_->current.identity.clear();
    DigestWriter poseHash; model_store_detail::serializeModel(impl_->poseTable, poseHash); impl_->interpolationPoseHash = poseHash.hash.hex();
    impl_->interpolationProposal = proposal; impl_->interpolationAlpha = alpha; impl_->interpolationReady = true;
}
uint64_t CanonicalScene::residentBytes() const { return impl_->memory(); }
const CanonicalScene::Counters& CanonicalScene::counters() const { return impl_->io; }
void CanonicalScene::resetCounters() { impl_->io = {}; }
}
