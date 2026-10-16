#include "sfm/map/RegionalWorkingStore.h"
#include "sfm/tests/TestMain.h"
#include <iostream>

using namespace sfm;
using Store = regional::RegionalWorkingStore;
using Files = regional::VerifiedImmutableFiles;
namespace fs = std::filesystem;
static void check(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
template<class F> static void rejects(F call, const char* why) {
    bool failed = false; try { call(); } catch (const std::exception&) { failed = true; } check(failed, why);
}
static std::string sha(const fs::path& file) { const auto value = spirula::sha256_file(file.u8string()); check(!value.empty(), "test SHA unavailable"); return value; }
static Reconstruction model(uint32_t offset = 0) {
    Reconstruction out; out.cameras.emplace(0, Camera::defaultFor(0, 640, 480, 500, CamModel::Pinhole));
    Point3D point; point.xyz = {0, 0, 10}; point.track = {{offset, 0}, {offset + 1, 0}, {offset + 1, 0}};
    for (uint32_t i = 0; i < 2; ++i) {
        Image image; image.id = offset + i; image.name = "image-" + std::to_string(image.id); image.camera_id = 0;
        image.registered = true; image.pose = {mat3Identity(), {-.3 * i, 0, 0}};
        image.points2D = {out.cameras.at(0).project(point.xyz + image.pose.t)}; image.point3D_ids = {0}; out.images.emplace(image.id, image);
    }
    out.points3D.emplace(0, point); out.next_point3D_id = 1; return out;
}
static uint64_t sourceBytes(const fs::path& original, const fs::path& work, size_t source) {
    uint64_t total = fs::file_size(original);
    for (const char* file : {"views.bin", "features.bin", "points.bin", "observations.bin", "photo-index.bin", "initial.xyz"})
        total += fs::file_size(work / ("source-" + std::to_string(source)) / file);
    return total;
}
static void fileLeaseTest(const fs::path& root) {
    const auto directory = root / "files"; fs::create_directory(directory); const auto path = directory / "immutable.bin";
    { std::ofstream out(path, std::ios::binary); out << "unchanged raw measurement evidence"; }
    const auto expected = sha(path); const auto time = fs::last_write_time(path); const auto size = fs::file_size(path);
#ifdef _WIN32
    HANDLE writer = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(writer != INVALID_HANDLE_VALUE, "test preexisting writer unavailable");
    { Files files; rejects([&] { files.verify(path, expected); }, "existing writable handle bypassed immutable lease"); }
    CloseHandle(writer);
#endif
    {
        Files files; files.verify(path, expected, size);
        const auto first = files.statistics(); check(first.bytesHashChecked == size && first.leaseMisses == 1, "first immutable file was not fully hashed");
        files.verify(path, expected, size);
#ifdef _WIN32
        check(files.statistics().bytesHashChecked == first.bytesHashChecked && files.statistics().leaseHits == 1 && files.leasedFiles() == 1, "same-file lease repeated SHA reads");
        HANDLE write = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        check(write == INVALID_HANDLE_VALUE, "verified immutable file remained writable"); if (write != INVALID_HANDLE_VALUE) CloseHandle(write);
        rejects([&] { fs::rename(path, directory / "renamed.bin"); }, "verified file could be renamed");
        rejects([&] { fs::remove(path); }, "verified file could be deleted");
        rejects([&] { fs::rename(directory, root / "renamed-directory"); }, "verified ancestor directory could be renamed");
        const auto hardlink = directory / "hardlink.bin"; fs::create_hard_link(path, hardlink); files.verify(hardlink, expected, size);
        check(files.leasedFiles() == 1 && files.statistics().bytesHashChecked == size && files.statistics().leaseHits == 2, "shared hardlink did not reuse physical identity lease");
        rejects([&] { fs::remove(hardlink); }, "verified alternate hardlink pathname could be unlinked/replaced");
        rejects([&] { fs::rename(hardlink, directory / "renamed-hardlink.bin"); }, "verified alternate hardlink pathname could be renamed/replaced");
        HANDLE aliasWriter = CreateFileW(hardlink.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        check(aliasWriter == INVALID_HANDLE_VALUE, "verified alternate hardlink remained writable"); if (aliasWriter != INVALID_HANDLE_VALUE) CloseHandle(aliasWriter);
        rejects([&] { files.verify(hardlink, std::string(64, '0'), size); }, "lease reused a different expected SHA");
#else
        check(files.statistics().bytesHashChecked == size * 2 && !files.statistics().leaseHits, "nonmandatory platform blindly cached immutable verification");
#endif
        rejects([&] { files.verify(path, expected, size + 1); }, "wrong expected immutable size accepted");
        check(sha(path) == expected, "lease tests changed source bytes");
    }
    { std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out); check(bool(out), "released immutable lease did not allow writer"); out.put('X'); out.flush(); check(bool(out), "test content mutation failed"); }
    fs::last_write_time(path, time); check(fs::file_size(path) == size, "test mutation changed length");
    Files cold; rejects([&] { cold.verify(path, expected, size); }, "cold same-size/same-mtime corruption was trusted");
    check(cold.statistics().bytesHashChecked == size && !cold.statistics().leaseHits, "cold corruption did not receive complete SHA read");
    Files tiny(1); rejects([&] { tiny.verify(path, sha(path), size); }, "lease metadata exceeded explicit memory allowance");
}
static void storeLeaseTest(const fs::path& root) {
    const auto first = root / "first.bin", second = root / "second.bin", work = root / "store";
    const auto a = model(), b = model(10); model_store_detail::writeModel(a, first); model_store_detail::writeModel(b, second);
    const auto beforeA = sha(first), beforeB = sha(second);
    Store store(work, 32u << 20); store.importModel(first, a, "first"); store.resetCounters(); store.verifyImmutable();
    const uint64_t bytesA = sourceBytes(first, work, 0); check(store.counters().bytesHashChecked == bytesA, "first complete Store immutable hash bytes incorrect");
    store.resetCounters(); store.verifyImmutable();
#ifdef _WIN32
    check(!store.counters().bytesHashChecked && store.counters().leaseHits == 7 && store.counters().leasedFiles == 7, "repeat Store verify reread frozen content");
#else
    check(store.counters().bytesHashChecked == bytesA, "conservative platform skipped strong Store recheck");
#endif
    {
        Store reopened(work, 32u << 20); check(reopened.counters().bytesHashChecked == bytesA, "new Store lifetime reused an untrusted previous lease");
        reopened.resetCounters(); reopened.verifyImmutable();
#ifdef _WIN32
        check(!reopened.counters().bytesHashChecked && reopened.counters().leaseHits == 7, "restore receipt was not reused by same Store");
#endif
    }
    store.importModel(second, b, "second"); store.resetCounters(); store.verifyImmutable(); const uint64_t bytesB = sourceBytes(second, work, 1);
#ifdef _WIN32
    check(store.counters().bytesHashChecked == bytesB && store.counters().leaseHits == 7 && store.counters().leaseMisses == 7, "append invalidated frozen prefix or skipped new suffix SHA");
#else
    check(store.counters().bytesHashChecked == bytesA + bytesB, "append complete conservative verification differs");
#endif
    uint64_t raw = 0; store.visitObservations([&](const auto&) { ++raw; }); check(raw == 6, "verification lease changed original raw multiplicity");
    const auto forkPath = store.forkImmutablePrefix(root / "prefix-fork", 1); Store fork(forkPath, 32u << 20);
    check(fork.sourceCount() == 1 && fork.sourceRawIdentity(0) == store.sourceRawIdentity(0), "leased raw prefix fork changed binding");
    raw = 0; fork.visitObservations([&](const auto&) { ++raw; }); check(raw == 3, "leased fork collapsed duplicate raw");
    const auto key = store.key(0, 0); const auto frozen = sha(work / "source-0/initial.xyz");
    store.beginGeneration(); store.setPointXYZ(key, {1, 2, 30}); store.commitGeneration();
    check(sha(work / "source-0/initial.xyz") == frozen && fork.pointXYZ(fork.key(0, 0)).z == 10 && store.pointXYZ(key).z == 30, "lease blocked COW or accepted/source XYZ was modified");
    check(sha(first) == beforeA && sha(second) == beforeB, "immutable verification changed original models");
}
static int run(int argc, char** argv) {
    ModelStore temporary(""); const auto root = argc > 1 ? fs::u8path(argv[1]) : temporary.directory() / "verification-lease";
    check(!fs::exists(root), "verification fixture requires a new directory"); fs::create_directories(root);
    fileLeaseTest(root); storeLeaseTest(root);
    std::cout << "PASS per-Store immutable verification: strong cold SHA, deny-write/delete leases, same-file/hardlink reuse, append suffix, restore, prefix fork and COW/raw retention\n"; return 0;
}
int main(int argc, char** argv) { return sfmTestMain(argc, argv, run); }
