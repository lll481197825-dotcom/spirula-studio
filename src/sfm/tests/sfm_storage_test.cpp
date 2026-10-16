#include "sfm/core/DiskStorage.h"
#include "sfm/core/Resume.h"
#include "sfm/map/RegionalStore.h"
#include "sfm/SfmConfig.h"
#include "sfm/tests/TestMain.h"
#include <chrono>
#include <cstdio>

namespace fs = std::filesystem;
using namespace sfm;

static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

static int storageTest(int, char**) {
    const auto root = fs::temp_directory_path() / ("sfm-storage-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    storage::checkAvailable(root, 100, 200, 100);
    bool rejected = false;
    try { storage::checkAvailable(root, 101, 200, 100); }
    catch (const std::runtime_error& e) { rejected = std::string(e.what()).find("accepted checkpoints retained") != std::string::npos; }
    require(rejected, "disk preflight failed to protect accepted state");
    rejected = false;
    try { storage::scaleBytes(UINT64_MAX, 2); } catch (const std::length_error&) { rejected = true; }
    require(rejected, "disk estimate overflow was ignored");
    const auto first = storage::regionalRoot(root / "dataset/.regional/capture", (root / "work").u8string());
    require(first == root / "work/capture" && first / "epoch-a" != first / "epoch-b",
            "new coordination epochs escaped their selected work root");
    SfmConfig cfg;
    const auto map_signature = stageSignature(cfg, CMD_MAP), auto_signature = stageSignature(cfg, CMD_AUTO);
    cfg.regional_work_dir = (root / "other-work").u8string();
    require(stageSignature(cfg, CMD_MAP) == map_signature && stageSignature(cfg, CMD_AUTO) == auto_signature,
            "changing the disk invalidated geometry or matching identity");
    Reconstruction original;
    Camera camera; camera.id = 0; original.cameras.emplace(0, camera);
    Image image; image.id = 0; image.name = "photo.jpg"; image.registered = true;
    image.points2D = {{1, 2}, {3, 4}}; image.point3D_ids = {7, 7}; original.images.emplace(0, image);
    Point3D point; point.xyz = {1, 2, 3}; point.track = {{0, 0}, {0, 1}}; original.points3D.emplace(7, point);
    original.next_point3D_id = 8;
    const auto source = root / "source.bin", target = root / "retained.bin";
    regional::atomicModel(source, original);
    require(fs::file_size(source) == model_store_detail::serializedModelBytes(original), "serialized disk budget differs from writer");
    storage::retainFile(source, target);
    require(fs::equivalent(source, target), "same-volume model reuse copied the payload");
    auto changed = original; changed.points3D.at(7).xyz.x = 42;
    regional::atomicModel(target, changed);
    require(!fs::equivalent(source, target) && model_store_detail::readModel(source).points3D.at(7).xyz.x == 1,
            "replacing a retained model changed its source or accepted baseline");
    storage::retainFile(source, target);
    regional::atomicModel(source, changed);
    require(model_store_detail::readModel(target).points3D.at(7).xyz.x == 1,
            "later source replacement changed a retained checkpoint");
    const auto signature = root / "signature.txt";
    resume::store(signature, "accepted"); resume::store(signature, "next");
    require(resume::recorded(signature) == "next" && !fs::exists(root / "signature.txt.part"), "resume commit is incomplete");
    rejected = false;
    try { resume::store(signature / "blocked", "bad"); } catch (const std::exception&) { rejected = true; }
    require(rejected && resume::recorded(signature) == "next", "resume write failure was silent or destroyed previous state");
    const auto unicode = root / fs::u8path(u8"检查点") / "resume.txt";
    resume::store(unicode, "unicode");
    require(resume::recorded(unicode) == "unicode", "checkpoint paths lost Unicode characters");
    fs::remove_all(root);
    std::puts("PASS"); return 0;
}
int main(int argc, char** argv) { return sfmTestMain(argc, argv, storageTest); }
