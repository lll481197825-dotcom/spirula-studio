#include "sfm/map/RegionalTopology.h"
#include "sfm/tests/TestMain.h"
#include <iostream>

using namespace sfm;
namespace fs = std::filesystem;
using Store = regional::RegionalWorkingStore;
using Topology = regional::RegionalTopology;
static void check(bool ok, const char* reason) { if (!ok) throw std::runtime_error(reason); }
template<class Call> static void rejects(Call call, const char* reason) {
    bool failed = false; try { call(); } catch (const std::exception&) { failed = true; } check(failed, reason);
}
static Reconstruction base() {
    Reconstruction rec; rec.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::Pinhole);
    for (uint32_t image = 0; image < 10; ++image) {
        Image im; im.id = image; im.camera_id = 0; im.name = "photo-" + std::to_string(image); im.registered = true;
        im.pose = {mat3Identity(), {-.2 * image, 0, 0}};
        for (uint32_t feature = 0; feature < 8; ++feature) im.points2D.push_back({200. + feature, 100. + image});
        im.point3D_ids.assign(8, kInvalidPoint3D); rec.images.emplace(image, im);
    }
    return rec;
}
static void point(Reconstruction& rec, uint64_t id, std::initializer_list<TrackElement> track, double x) {
    Point3D p; p.xyz = {x, 2, 8}; p.track = track;
    for (const auto& row : track) rec.images.at(row.image_id).point3D_ids.at(row.point2D_idx) = id;
    rec.points3D.emplace(id, p); rec.next_point3D_id = std::max(rec.next_point3D_id, id + 1);
}
static std::vector<regional::Observation> tracks(const Topology& topology) {
    std::vector<regional::Observation> result;
    topology.visitCommonObservations([&](regional::Observation row) { result.push_back(row); }); return result;
}
static void equivalent(Store& store, const Topology& a, const Topology& b) {
    check(a.groups() == b.groups(), "incremental group count differs from cold build");
    for (size_t source = 0; source < store.sourceCount(); ++source)
        for (uint64_t ordinal = 0; ordinal < store.sourcePointCount(source); ++ordinal) {
            const auto key = store.key(source, ordinal); const auto ax = a.xyz(key, store.pointXYZ(key)), bx = b.xyz(key, store.pointXYZ(key));
            check(a.canonical(key) == b.canonical(key) && a.shared(key) == b.shared(key), "incremental point identity differs from cold build");
            check((ax - bx).norm() < 1e-12, "incremental parameter differs from cold build");
        }
    const auto at = tracks(a), bt = tracks(b); check(at.size() == bt.size(), "incremental common track count differs");
    for (size_t i = 0; i < at.size(); ++i)
        check(at[i].key == bt[i].key && at[i].point == bt[i].point && at[i].xy.x == bt[i].xy.x && at[i].xy.y == bt[i].xy.y,
            "incremental common feature or XY differs from cold build");
}
static std::string fileHash(const fs::path& path) {
    std::ifstream in(path, std::ios::binary); spirula::Sha256 hash; std::array<uint8_t, 65536> bytes;
    while (in) { in.read(reinterpret_cast<char*>(bytes.data()), bytes.size()); hash.update(bytes.data(), size_t(in.gcount())); }
    check(in.eof(), "test checksum read failed"); return hash.hex();
}
static void legacyCopy(const fs::path& current, const fs::path& legacy) {
    constexpr uint64_t oldMagic = 0x35504f5453474552ull;
    struct StoredGroup { uint64_t root, first, count, weight; Vec3 xyz; uint64_t file; };
    fs::create_directory(legacy); model_store_detail::Reader head(current / "head.bin"); head.scalar<uint64_t>();
    const auto path = fs::u8path(head.string()); head.string(); head.finish(); model_store_detail::Reader in(path);
    check(in.scalar<uint64_t>() == 0x36504f5453474552ull, "test expected compact checkpoint");
    const auto destination = legacy / "topology.bin"; model_store_detail::Writer out(destination); out.scalar(oldMagic);
    out.string(in.string()); out.string(in.string()); const auto sources = in.scalar<uint64_t>(); out.scalar(sources);
    for (size_t source = 0; source < sources; ++source) out.string(in.string());
    out.vector(in.vector<uint32_t>()); out.vector(in.vector<uint64_t>());
    const auto count = in.scalar<uint64_t>(); const auto width = in.scalar<uint8_t>(); std::vector<uint64_t> roots(count);
    for (auto& root : roots) root = width == 4 ? in.scalar<uint32_t>() : in.scalar<uint64_t>();
    const auto groups = in.vector<StoredGroup>(); std::vector<uint64_t> groupOf(count, kInvalidPoint3D);
    for (size_t g = 0; g < groups.size(); ++g) groupOf.at(size_t(groups[g].root)) = g;
    out.vector(roots); out.vector(groupOf); out.vector(groups);
    for (int catalogue = 0; catalogue < 3; ++catalogue) {
        const auto entries = in.scalar<uint64_t>(); out.scalar(entries);
        for (size_t entry = 0; entry < entries; ++entry) { out.string(in.string()); out.string(in.string()); out.scalar(in.scalar<uint64_t>()); }
    }
    in.finish(); out.finish(); model_store_detail::Writer legacyHead(legacy / "head.bin"); legacyHead.scalar(oldMagic);
    legacyHead.string(destination.u8string()); legacyHead.string(fileHash(destination)); legacyHead.finish();
}
static void capacityLayout() {
    using namespace regional::topology_storage;
    constexpr size_t points = 89671309; Indices indices; indices.resize(points);
    check(!indices.wide() && indices.bytes() == points * uint64_t(4), "real point capacity did not use exact compact indices");
    indices[points - 1] = points - 1; check(uint64_t(indices[points - 1]) == points - 1, "compact identity truncated");
    rejects([&] { indices[1] = uint64_t(UINT32_MAX) + 1; }, "compact root silently truncated a 64-bit identity");
    check(Indices::allocationBytes(uint64_t(UINT32_MAX) + 1) == (uint64_t(UINT32_MAX) + 1) * 8, "wide root layout unavailable");
    Indices wideIndices(true); wideIndices.resize(3); const uint64_t wideIdentity = uint64_t(UINT32_MAX) + 123456;
    wideIndices[0] = wideIdentity; wideIndices[1] = wideIndices[0];
    check(wideIndices.wide() && uint64_t(wideIndices[1]) == wideIdentity && wideIndices.bytes() == 24, "wide identities truncated in small forced-width fixture");
    rejects([&] { Indices::allocationBytes(UINT64_MAX); }, "wide capacity arithmetic overflowed");
    RankedBits shared; shared.resize(points);
    const std::vector<size_t> selected{0, 63, 64, 511, 512, 513, points - 1};
    for (auto root : selected) shared[root] = true; shared.buildRanks();
    for (size_t g = 0; g < selected.size(); ++g) check(shared.rank(selected[g]) == g && shared[selected[g]], "shared root rank changed a group identity");
    check(!shared[65] && shared.rank(514) == 6, "bitmap shared membership or rank incorrect");
    Chunks<std::array<uint64_t, 8>> groups;
    for (uint64_t g = 0; g <= 2097152; ++g) groups.push_back({g, 0, 0, 0, 0, 0, 0, 0});
    check(groups.capacity() == 2097152 + Chunks<std::array<uint64_t, 8>>::block, "group capacity doubled at production failure threshold");
    check(groups.at(1048576)[0] == 1048576 && groups.at(2097152)[0] == 2097152, "chunk boundary corrupted group identity");
    const auto bytes = indices.bytes() + shared.bytes() + groups.bytes() + Bits::allocationBytes(points) * 3;
    check(bytes < (520ull << 20), "real-capacity topology structural working set exceeds 520 MiB");
    std::cout << "PASS: 89,671,309 compact roots and 2,097,153 chunked groups, " << bytes << " structural bytes including three live bitmaps\n";
}
static int run(int, char**) {
    capacityLayout();
    ModelStore temporary(""); const auto root = temporary.directory();
    const auto owners = std::vector<uint32_t>{0, 0, 1, 1, 2, 2, 3, 3, 4, 5};
    auto a = base(); point(a, 10, {{0, 0}, {1, 0}}, 1); point(a, 11, {{0, 1}, {2, 1}}, 2);
    point(a, 12, {{4, 2}}, 3); point(a, 13, {{2, 3}, {3, 3}}, 4); point(a, 14, {{8, 6}, {9, 6}}, 5);
    auto b = base(); point(b, 20, {{0, 1}, {2, 1}, {2, 1}}, 6); point(b, 21, {{4, 4}, {5, 4}}, 7);
    auto c = base(); point(c, 30, {{1, 0}, {3, 0}}, 8); point(c, 31, {{4, 2}, {5, 2}}, 9);
    point(c, 32, {{0, 1}, {3, 1}}, 10); point(c, 33, {{6, 5}, {7, 5}}, 11);
    const auto pa = root / "a.bin", pb = root / "b.bin", pc = root / "c.bin";
    model_store_detail::writeModel(a, pa); model_store_detail::writeModel(b, pb); model_store_detail::writeModel(c, pc);
    Store store(root / "work", 128u << 20); store.importModel(pa, a); store.importModel(pb, b);
    Topology incremental(root / "topology", 128u << 20); incremental.prepare(store, owners);
    check(incremental.groups() == 2, "cold shared topology lost a complete common track");
    const auto oldRaw = store.sourceObservationCount(0) + store.sourceObservationCount(1);
    incremental.saveParameters(root / "old-parameters.bin");
    {
        legacyCopy(root / "topology", root / "legacy-topology"); Topology legacy(root / "legacy-topology", 128u << 20);
        legacy.prepare(store, owners); equivalent(store, incremental, legacy);
        const auto parameters = root / "legacy-parameters.bin"; fs::copy_file(root / "old-parameters.bin", parameters);
        std::fstream file(parameters, std::ios::binary | std::ios::in | std::ios::out); const uint64_t oldMagic = 0x35504f5453474552ull;
        file.write(reinterpret_cast<const char*>(&oldMagic), sizeof(oldMagic)); file.close(); resume::store(parameters.string() + ".sha256", fileHash(parameters));
        legacy.loadParameters(parameters); equivalent(store, incremental, legacy);
    }
    store.resetCounters(); incremental.prepare(store, owners);
    check(store.counters().raw_bytes == 0 && incremental.counters().new_sources == 0, "same topology reread raw source");
    {
        Topology reopened(root / "topology", 128u << 20); store.resetCounters(); reopened.prepare(store, owners);
        equivalent(store, incremental, reopened); check(store.counters().source_full_reads == 0, "warm topology reread original model");
    }
    store.importModel(pc, c); store.resetCounters(); incremental.prepare(store, owners);
    check(incremental.counters().new_sources == 1 && incremental.counters().old_private_points == 2 && incremental.counters().reused_groups == 1,
        "append did not reuse old groups or gather just the affected old private points");
    check(store.sourceObservationCount(0) + store.sourceObservationCount(1) == oldRaw, "append changed immutable raw observations");
    Topology fresh(root / "fresh", 128u << 20); fresh.prepare(store, owners); equivalent(store, incremental, fresh);
    {
        Topology beforeJoint(root / "topology", 128u << 20); beforeJoint.restore();
        check(beforeJoint.restoreAcceptedParameters(root / "old-parameters.bin") == Topology::ParameterRestore::InitializedAppend,
            "accepted ancestor not identified after committed append before joint acceptance");
        equivalent(store, incremental, beforeJoint);
        const auto fake = root / "fake-old-parameters.bin"; std::filesystem::copy_file(root / "old-parameters.bin", fake);
        model_store_detail::Reader input(fake); input.scalar<uint64_t>(); input.string(); input.scalar<uint64_t>(); const auto position = input.position();
        std::fstream changed(fake, std::ios::binary | std::ios::in | std::ios::out); changed.seekp(std::streamoff(position));
        const double alternate = 123; changed.write(reinterpret_cast<const char*>(&alternate), sizeof(alternate)); changed.close();
        std::ifstream file(fake, std::ios::binary); std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), {}); spirula::Sha256 hash; hash.update(bytes.data(), bytes.size());
        resume::store(fake.string() + ".sha256", hash.hex());
        rejects([&] { beforeJoint.restoreAcceptedParameters(fake); }, "ancestor identity accepted with different parameter provenance");
    }
    {
        std::vector<Store::PointKey> aliases; store.resetCounters();
        const auto canonicalRoot = incremental.canonical(store.localKey(0, 11));
        const auto count = incremental.visitCanonicalMembers({canonicalRoot}, store, [&](Store::PointKey key) { aliases.push_back(key); });
        check(count == 3 && aliases == std::vector<Store::PointKey>({store.localKey(0, 11), store.localKey(1, 20), store.localKey(2, 32)}),
            "canonical alias enumeration lost a member or changed source/ordinal order");
        check(store.counters().raw_bytes == 0, "canonical alias enumeration read raw observations");
        regional::topology_storage::Bits selected(incremental.points()); selected[size_t(canonicalRoot)] = true;
        std::vector<Store::PointKey> bitmapAliases; store.resetCounters();
        check(incremental.visitCanonicalMembers(selected, store, [&](Store::PointKey key) { bitmapAliases.push_back(key); }) == count && bitmapAliases == aliases,
            "bitmap canonical alias enumeration changed complete support or source order");
        check(store.counters().raw_bytes == 0, "bitmap canonical alias enumeration read raw observations");
        rejects([&] { regional::topology_storage::Bits wrong(selected.size() + 1); incremental.visitCanonicalMembers(wrong, store, [](Store::PointKey) {}); },
            "wrong point-catalogue bitmap accepted");
        selected[size_t(store.sourcePointCount(0) + store.localKey(1, 20).ordinal)] = true;
        rejects([&] { incremental.visitCanonicalMembers(selected, store, [](Store::PointKey) {}); }, "noncanonical bitmap member request accepted");
        rejects([&] { incremental.visitCanonicalMembers({store.sourcePointCount(0) + store.localKey(1, 20).ordinal}, store, [](Store::PointKey) {}); },
            "noncanonical member request accepted");
    }
    rejects([&] { incremental.loadParameters(root / "old-parameters.bin"); }, "old parameter checkpoint accepted across topology append");
    incremental.saveParameters(root / "parameters.bin");
    Topology restored(root / "topology", 128u << 20); restored.prepare(store, owners); restored.loadParameters(root / "parameters.bin"); equivalent(store, incremental, restored);
    size_t cancelledBegin = 0, cancelledEnd = 0;
    events::set_sink([&](const Event& event) { if (event.stage == Stage::Merge) { cancelledBegin += event.kind == Event::Kind::StageBegin; cancelledEnd += event.kind == Event::Kind::StageEnd; } });
    std::atomic<bool> stop{true}; cancel::set_token(&stop);
    rejects([&] { restored.triangulate(store.poses(), 3, 2); }, "worker cancellation did not propagate"); cancel::set_token(nullptr);
    events::set_sink({}); check(cancelledBegin == 1 && cancelledEnd == 1, "cancelled shared triangulation left its native stage open");
    auto conflict = base(); conflict.images.at(0).points2D[1].x += .5; point(conflict, 90, {{0, 1}, {2, 1}}, 12);
    const auto pd = root / "d.bin"; model_store_detail::writeModel(conflict, pd); store.importModel(pd, conflict);
    incremental.prepare(store, owners); Topology conflictFresh(root / "conflict-fresh", 128u << 20); conflictFresh.prepare(store, owners);
    equivalent(store, incremental, conflictFresh); check(!incremental.shared(store.localKey(0, 11)), "same-feature XY conflict not quarantined");
    auto ambiguous = base(); point(ambiguous, 100, {{0, 0}, {0, 1}, {3, 0}}, 20);
    const auto ambiguousSource = root / "ambiguous.bin"; model_store_detail::writeModel(ambiguous, ambiguousSource); store.importModel(ambiguousSource, ambiguous);
    incremental.prepare(store, owners); Topology ambiguousFresh(root / "ambiguous-fresh", 128u << 20); ambiguousFresh.prepare(store, owners);
    equivalent(store, incremental, ambiguousFresh); check(!incremental.shared(store.localKey(0, 10)), "multiple features in one photo not quarantined");
    {
        auto first = base(); point(first, 1, {{0, 0}, {2, 0}}, 1); point(first, 2, {{0, 0}, {2, 0}}, 3);
        auto next = base(); point(next, 3, {{0, 1}, {3, 1}}, 5);
        const auto firstPath = root / "boundary-old.bin", nextPath = root / "boundary-new.bin";
        model_store_detail::writeModel(first, firstPath); model_store_detail::writeModel(next, nextPath);
        Store boundaryStore(root / "boundary-work", 128u << 20); boundaryStore.importModel(firstPath, first);
        Topology boundaryTopo(root / "boundary-topology", 128u << 20); boundaryTopo.prepare(boundaryStore, owners);
        check(boundaryTopo.groups() == 2, "single-source competing private features unexpectedly merged");
        boundaryStore.importModel(nextPath, next); boundaryTopo.prepare(boundaryStore, owners);
        Topology boundaryFresh(root / "boundary-fresh", 128u << 20); boundaryFresh.prepare(boundaryStore, owners);
        equivalent(boundaryStore, boundaryTopo, boundaryFresh);
        check(boundaryTopo.canonical(boundaryStore.localKey(0, 1)) == boundaryTopo.canonical(boundaryStore.localKey(0, 2)),
            "old-only feature was not revisited when its photo became shared");
    }
    store.verifyImmutable();
    auto large = base();
    for (auto& image : large.images) { image.second.points2D.resize(200000, {100, 200}); image.second.point3D_ids.resize(200000, kInvalidPoint3D); }
    for (uint32_t id = 0; id < 200000; ++id) point(large, id + 100, {{0, id}, {2, id}}, double(id));
    const auto largeSource = root / "large-source.bin"; model_store_detail::writeModel(large, largeSource);
    Store largeStore(root / "large-work", 128u << 20); largeStore.importModel(largeSource, large);
    Topology largeTopology(root / "large-topology", 128u << 20); largeTopology.prepare(largeStore, owners);
    {
        const auto parameters = root / "large-parameters.bin"; largeTopology.saveParameters(parameters);
        const auto constrainedBudget = size_t((largeTopology.memoryBytes() + (7u << 19)) * 3);
        Topology staged(root / "large-topology", constrainedBudget); staged.prepare(largeStore, owners); staged.loadParameters(parameters);
        check(staged.counters().parameter_disk_bytes == 200000 * uint64_t(2 * sizeof(Vec3)), "parameter restoration did not stage within remaining memory");
        std::vector<std::pair<uint64_t, uint64_t>> completed;
        size_t phaseBegin = 0, phaseEnd = 0;
        events::set_sink([&](const Event& event) { if (event.stage == Stage::Merge) { phaseBegin += event.kind == Event::Kind::StageBegin; phaseEnd += event.kind == Event::Kind::StageEnd; } });
        staged.triangulate(largeStore.poses(), 3, 2, [&](uint64_t done, uint64_t total) { completed.emplace_back(done, total); });
        events::set_sink({}); check(phaseBegin == 1 && phaseEnd == 1, "shared triangulation did not own a complete native progress stage");
        check(!completed.empty() && completed.front().first == 0 && completed.back() == std::make_pair(uint64_t(200000), uint64_t(200000)),
            "triangulation did not publish real initial/final completed groups");
        for (size_t i = 1; i < completed.size(); ++i) check(completed[i].first > completed[i - 1].first && completed[i].first <= completed[i].second,
            "completed shared triangulation progress went backwards or exceeded its total");
        check(staged.peakMemoryBytes() <= constrainedBudget / 3, "adaptive triangulation exceeded remaining phase budget");
        staged.loadParameters(parameters); equivalent(largeStore, largeTopology, staged);
        auto added = large; added.points3D.begin()->second.xyz.x += 1;
        const auto appendPath = root / "large-append.bin"; model_store_detail::writeModel(added, appendPath); largeStore.importModel(appendPath, added);
        largeStore.resetCounters(); staged.prepare(largeStore, owners);
        check(staged.groups() == 200000 && staged.counters().new_sources == 1 && largeStore.counters().source_full_reads == 0,
            "bounded append changed source reuse or complete group coverage");
        check(staged.peakMemoryBytes() <= constrainedBudget / 3, "append kept old and new complete group tables in memory");
        Topology appendedFresh(root / "large-append-fresh", 128u << 20); appendedFresh.prepare(largeStore, owners); equivalent(largeStore, staged, appendedFresh);
        Topology appendedRestored(root / "large-topology", constrainedBudget); appendedRestored.restore();
        check(appendedRestored.restoreAcceptedParameters(parameters) == Topology::ParameterRestore::InitializedAppend,
            "bounded append lost accepted ancestor parameter identity");
        equivalent(largeStore, staged, appendedRestored); largeStore.verifyImmutable();
    }
    rejects([&] { Topology budgetGuard(root / "large-topology", 8u << 20); budgetGuard.prepare(largeStore, owners); },
        "warm topology allocated arrays beyond supplied memory budget");
    Topology lowBudget(root / "topology", 8u << 20); lowBudget.prepare(store, owners);
    {
        std::filesystem::path tracksPath;
        for (const auto& directory : std::filesystem::directory_iterator(root / "topology"))
            if (directory.is_directory() && std::filesystem::file_size(directory.path() / "tracks.bin")) { tracksPath = directory.path() / "tracks.bin"; break; }
        check(!tracksPath.empty(), "no persisted shared track artifact");
        std::fstream file(tracksPath, std::ios::binary | std::ios::in | std::ios::out); char byte; file.read(&byte, 1); byte ^= 1; file.seekp(0); file.write(&byte, 1); file.close();
        rejects([&] { Topology corrupted(root / "topology", 128u << 20); corrupted.prepare(store, owners); }, "warm topology did not verify track checksum");
    }
    std::cout << "PASS: exact complete cold/warm/append topology, private-to-shared promotion, unchanged track-file reuse, raw duplicates, XY quarantine, checkpoint identity and worker exceptions\n";
    return 0;
}
int main(int argc, char** argv) { return sfmTestMain(argc, argv, run); }
