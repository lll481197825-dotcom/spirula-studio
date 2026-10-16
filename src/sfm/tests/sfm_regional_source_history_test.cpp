#include "sfm/map/RegionalSourceHistory.h"
#include "sfm/tests/TestMain.h"
#include "data/JsonWrite.h"
#include <chrono>
#include <cstdio>

using namespace sfm;
namespace history = sfm::regional::history;
static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> static void rejected(F body, const char* message) {
    bool failed = false; try { body(); } catch (const std::exception&) { failed = true; }
    require(failed, message);
}
struct Positions : PriorSource {
    std::vector<Vec3> coordinates;
    bool has(uint32_t) const override { return false; }
    bool relativeRotation(uint32_t, uint32_t, Mat3&, double&) const override { return false; }
    std::vector<uint32_t> neighbours(uint32_t) const override { return {}; }
    PosePriors factors(const std::vector<PosedImage>&) override { return {}; }
    bool position(uint32_t id, Vec3& value) const override {
        if (id >= coordinates.size()) return false;
        value = coordinates[id]; return true;
    }
};
static Reconstruction fixture(Positions& gps) {
    Reconstruction model; model.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::Pinhole);
    gps.coordinates = {{0,0,0},{10,0,1},{0,10,2},{10,10,3},{20,4,5},{2,20,7}};
    for (uint32_t id = 0; id < gps.coordinates.size(); ++id) {
        Image image; image.id = id; image.camera_id = 0; image.registered = true;
        image.name = "photo-" + std::to_string(id); image.pose.t = gps.coordinates[id] * -1.; model.images[id] = image;
    }
    return model;
}
static int test(int argc, char** argv) {
    namespace fs = std::filesystem;
    if (argc >= 2 && std::string(argv[1]) == "--preflight") {
        if (argc != 6) throw std::runtime_error("--preflight manifest.bin ENU-rows.txt recorded-attitude.bin output.csv");
        Positions gps; std::ifstream positions(fs::u8path(argv[3])); Vec3 value;
        if (!positions) throw std::runtime_error("original ENU cache missing");
        while (positions >> value.x >> value.y >> value.z) {
            if (!std::isfinite(value.norm())) throw std::runtime_error("nonfinite original GPS coordinate");
            gps.coordinates.push_back(value);
        }
        require(gps.coordinates.size() == 122415 && positions.eof(), "original GPS input identity incomplete");
        AttitudeRef attitude; model_store_detail::Reader input(fs::u8path(argv[4]));
        const auto count = input.count(76);
        std::set<uint32_t> ids;
        for (size_t i = 0; i < count; ++i) {
            const auto id = input.scalar<uint32_t>(); const auto rotation = input.scalar<Mat3>();
            require(id < 122415 && ids.insert(id).second, "invalid recorded historical attitude identity");
            attitude.image_ids.push_back(id); attitude.world_from_cam.push_back(rotation);
        }
        input.finish(); attitude.registered = int(count);
        const auto frames = history::loadPinned(fs::u8path(argv[2]), "d936da1775a11c18.122415", 122415, gps, attitude);
        std::ofstream output(fs::u8path(argv[5])); require(bool(output), "cannot open historical preflight report");
        output << "source,photos,position_required,gauge_required,gps_inliers,position_passed,gauge_passed\n";
        for (const auto& frame : frames) {
            const auto quality = regional::regionalGpsQuality(frame.views, gps);
            output << history::csvCell(frame.name) << ',' << quality.photos << ',' << frame.positioned << ','
                << frame.certified << ',' << quality.inliers << ',' << quality.accepted() << ','
                << regional::regionalGpsGauge(frame.views, gps, attitude) << '\n';
        }
        output.flush(); require(bool(output), "historical preflight report write failed");
        std::printf("PREFLIGHT immutable historical sources=%zu certified=18 fixed ENU; no final candidate audited\n", frames.size());
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--pin") {
        if (argc != 4) throw std::runtime_error("--pin input-map.json output-manifest.bin");
        const auto manifest = history::pinJson(fs::u8path(argv[2]), fs::u8path(argv[3]), "d936da1775a11c18.122415", 122415);
        require(manifest.sources.size() == 20, "actual historical source count must be 20");
        size_t certificate = 0, position = 0;
        for (const auto& source : manifest.sources) { certificate += source.certified; position += source.positioned; }
        require(certificate == 18 && position == 20, "actual historical flags must be 20/18");
        std::printf("PINNED historical position=%zu certified=%zu SHA256=%s\n", position, certificate,
            history::fullHash(fs::u8path(argv[3])).c_str());
        return 0;
    }
    const auto root = fs::temp_directory_path() / ("sfm-history-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    Positions gps; const auto model = fixture(gps); AttitudeRef attitude;
    history::Manifest manifest; manifest.feature_digest = "fixture-identity.6"; manifest.input_count = 6;
    for (size_t source = 0; source < 20; ++source) {
        const auto path = root / ("source-" + std::to_string(source) + ".bin");
        model_store_detail::writeModel(model, path);
        manifest.sources.push_back({"source-" + std::to_string(source), path.u8string(), history::fullHash(path), 6, true, source < 18});
    }
    const auto pinned = root / "historical-source-certificates.bin";
    history::writeManifest(manifest, pinned);
    const auto marker = root / "historical-source-certificates.required";
    { std::ofstream output(marker); output << history::fullHash(pinned) << '\n'; }
    const auto frames = history::loadRequired(root, manifest.feature_digest, 6, gps, attitude, true);
    require(frames.size() == 20 && !frames[19].certified, "historical certificate flags silently recomputed");
    require(history::verifyFinal(frames, model, gps, attitude, root / "pass.csv"), "known complete historical source failed");
    rejected([&] { history::loadPinned(pinned, "wrong-digest", 6, gps, attitude); }, "wrong feature identity accepted");
    rejected([&] { history::loadPinned(pinned, manifest.feature_digest, 7, gps, attitude); }, "wrong original count accepted");
    auto candidate = model; candidate.images.erase(0);
    require(!history::verifyFinal(frames, candidate, gps, attitude, root / "missing.csv"), "missing original photo accepted");
    candidate = model; candidate.images[0].camera_id = 1;
    require(!history::verifyFinal(frames, candidate, gps, attitude, root / "camera.csv"), "changed original camera accepted");
    candidate = model; for (auto& item : candidate.images) item.second.pose.t.x += 100;
    require(!history::verifyFinal(frames, candidate, gps, attitude, root / "gps.csv"), "refit absorbed wrong fixed GPS position");
    require(candidate.images[0].pose.t.x == 100, "history audit changed candidate poses");
    auto bad = manifest; bad.sources[0].certified = false; history::writeManifest(bad, pinned);
    rejected([&] { history::loadRequired(root, manifest.feature_digest, 6, gps, attitude, true); }, "changed pinned flags accepted");
    rejected([&] { history::loadPinned(pinned, manifest.feature_digest, 6, gps, attitude); }, "lost historical certificate accepted");
    history::writeManifest(manifest, pinned);
    { std::ofstream output(fs::u8path(manifest.sources[0].path), std::ios::binary | std::ios::app); output.put('x'); }
    rejected([&] { history::loadPinned(pinned, manifest.feature_digest, 6, gps, attitude); }, "modified original source accepted");
    fs::remove(pinned);
    rejected([&] { history::loadRequired(root, manifest.feature_digest, 6, gps, attitude, true); }, "missing required manifest accepted");
    const auto fresh = root / "fresh"; fs::create_directories(fresh);
    require(history::loadRequired(fresh, "fresh.6", 6, gps, attitude, false).empty(), "fresh project incorrectly requires history");
    fs::remove_all(root);
    std::puts("PASS immutable source SHA, fixed GPS, original photo/camera membership, 20/18 obligations, required proof, no pose mutation");
    return 0;
}
int main(int argc, char** argv) { return sfmTestMain(argc, argv, test); }
