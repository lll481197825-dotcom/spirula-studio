#include "sfm/map/NativeSceneTasks.h"
#include "sfm/map/SceneVisualEvidence.h"
#include "sfm/tests/TestMain.h"
#include <iostream>

using namespace sfm;
using namespace sfm::regional;
using Path = std::filesystem::path;
static void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
template<class Action> static void rejects(Action action, const char* reason) {
    bool caught = false; try { action(); } catch (const std::exception&) { caught = true; } require(caught,reason);
}
static Vec3 finiteTruth(uint32_t point) { return {.2 * (point % 4) - .3, .15 * (point / 4) - .2, 5 + .07 * point}; }
static Vec3 directionTruth() { return Vec3{.1,.02,1}.normalized(); }
static Reconstruction fixture(bool retained, bool retainedOnly) {
    Reconstruction out; Camera camera = Camera::defaultFor(0,800,600,500,CamModel::Pinhole); camera.pixel_scale = 2; out.cameras.emplace(0,camera);
    for (uint32_t id = 0; id < 2; ++id) {
        Image image; image.id = id; image.camera_id = 0; image.name = "photo-" + std::to_string(id); image.registered = true;
        image.pose = {mat3Identity(),{-.6 * id,0,0}}; image.points2D.resize(24); image.point3D_ids.assign(24,kInvalidPoint3D); out.images.emplace(id,image);
    }
    if (!retainedOnly) for (uint32_t point = 0; point < 12; ++point) {
        Point3D value; const bool direction = point == 11; const auto truth = direction ? directionTruth() : finiteTruth(point);
        value.xyz = direction ? truth : truth + Vec3{.02,.01,.03};
        const uint64_t id = 1000 + point;
        for (auto& image : out.images) {
            image.second.points2D[point] = camera.project(direction ? truth : truth + image.second.pose.t);
            image.second.point3D_ids[point] = id; value.track.push_back({image.first,point});
        }
        out.points3D.emplace(id,value);
    }
    if (retained) for (uint32_t point = 0; point < 12; ++point) {
        const uint32_t image = point % 2, feature = point + 12; const uint64_t id = 2000 + point;
        Point3D value; value.xyz = {1e6 + point,-1e6,-3}; value.track = {{image,feature},{image,feature}};
        out.images.at(image).points2D[feature] = {double(31 * point + 4),double(17 * point + 9)};
        out.images.at(image).point3D_ids[feature] = id; out.points3D.emplace(id,value);
    }
    out.next_point3D_id = retained ? 2012 : 1012; return out;
}
struct Fixture {
    Path source;
    std::string sourceSHA;
    RegionalWorkingStore catalog;
    ImmutableSceneTopology index;
    CanonicalScene scene;
    Fixture(const Path& root, bool retained, bool retainedOnly)
        : source(root/"source.bin"), catalog(root/"catalog",64u<<20), index(root/"index",32u<<20), scene(root/"scene",64u<<20) {
        std::filesystem::create_directories(root); const auto original = fixture(retained,retainedOnly);
        model_store_detail::writeModel(original,source); sourceSHA = spirula::sha256_file(source.string()); catalog.importModel(source,original,"retained-bundle-source");
        index.prepare(catalog,{0,0},"retained-bundle-full-raw",catalog.sourcePointCount(0),[](auto key) { return key.ordinal; });
        scene.beginLayoutCandidate(CanonicalScene::bindingFor(catalog,index.identity()),catalog.poses(),
            [&](const auto& emit) { index.emitAliases(emit); },[&](const auto& emit) { index.visitSeeds([&](const auto& seed) { emit(seed.root,seed.xyz); }); });
        std::vector<uint64_t> retainedRoots; std::vector<std::pair<uint64_t,Vec3>> directions;
        catalog.visitSourcePoints(0,[&](auto key,const Vec3&,const auto&) {
            if (key.localPoint >= 2000) retainedRoots.push_back(scene.canonical(key));
            else if (key.localPoint == 1011) directions.push_back({scene.canonical(key),directionTruth()});
        });
        scene.applyLandmarkTypes(directions); scene.applyRetainedObservations(retainedRoots); scene.commit("retained-bundle-fixture");
    }
    void originalUnchanged() const { require(spirula::sha256_file(source.string()) == sourceSHA,"retained gather/solve modified original source raw"); }
};
static BundleOptions options() {
    BundleOptions out; out.real = RealCfg::CPU; out.threads = 1; out.max_iters = 3; out.refine_intrinsics = false;
    out.loss = "huber"; out.loss_param = 3; return out;
}
static void sameProblem(const BAProblem& a, const BAProblem& b) {
    require(a.num_points == b.num_points && a.num_obs == b.num_obs && a.num_images == b.num_images,"retained raw changed BA dimensions");
    require(a.obs_image == b.obs_image && a.obs_point == b.obs_point && a.obs_ranges == b.obs_ranges && a.obs_xy == b.obs_xy,"retained raw changed geometric measurements");
    require(a.points == b.points && a.poses == b.poses && a.intr == b.intr && a.fixed_points == b.fixed_points && a.fixed_frames == b.fixed_frames && a.point_kinds == b.point_kinds,"retained raw changed native parameters/masks/types");
}
static void mixed(const Path& root) {
    Fixture baseline(root/"baseline",false,false), retained(root/"retained",true,false); auto run = options();
    CanonicalBundle reference, actual; reference.prepare(baseline.index,baseline.scene,{0,1},0,{0},run,32u<<20);
    actual.prepare(retained.index,retained.scene,{0,1},0,{0},run,32u<<20);
    require(baseline.scene.observations() == 24 && retained.scene.observations() == 48 && retained.index.observationCount() == 48,"fixture raw duplicate count changed");
    require(reference.authority_observations == 24 && !reference.retained_observations && actual.authority_observations == 48 && actual.retained_observations == 24,"geometric/retained authority accounting differs");
    require(actual.problem.num_points == 12 && actual.problem.num_obs == 24 && actual.points == reference.points,"retained roots entered native BA point table");
    for (auto kind : actual.problem.point_kinds) require(kind <= 1,"canonical-only retained kind reached native shader/CPU BA");
    sameProblem(reference.problem,actual.problem);
    const auto quality = sceneQuality(retained.index,retained.scene,3,16u<<20,1);
    require(quality.observations == 48 && quality.retained_observations == 24,"full raw authority lost retained duplicates");
    const auto beforeEvidence = sceneVisualEvidence(retained.index,retained.scene,3,16u<<20,1);
    for (uint32_t id = 0; id < 2; ++id) require(beforeEvidence.landmarks.at(id) == 11 && beforeEvidence.landmarks.at(id) < 12,"retained/directional rows fabricated twelve metric landmarks");
    baseline.scene.beginCandidate(); retained.scene.beginCandidate(); const auto referenceResult = reference.solve(run), actualResult = actual.solve(run);
    require(referenceResult.stats.final_cost == actualResult.stats.final_cost,"retained raw influenced actual native camera solve objective"); sameProblem(reference.problem,actual.problem);
    reference.apply(baseline.scene); actual.apply(retained.scene);
    for (const auto& image : baseline.scene.poses().images) {
        const auto& a = image.second.pose; const auto& b = retained.scene.poses().images.at(image.first).pose;
        require(a.R == b.R && a.t.x == b.t.x && a.t.y == b.t.y && a.t.z == b.t.z,"retained raw changed solved camera geometry");
    }
    uint64_t retainedRows = 0; retained.index.visitRaw([&](const auto& row) { if (retained.scene.landmark(row.canonical).kind == LandmarkKind::RetainedObservation) ++retainedRows; });
    require(retainedRows == 24 && retained.scene.observations() == 48,"native solve removed retained raw from authority");
    baseline.scene.abort("retained-camera-influence-proof"); retained.scene.abort("retained-camera-influence-proof"); baseline.originalUnchanged(); retained.originalUnchanged();
}
static void retainedOnly(const Path& root) {
    Fixture fixture(root,true,true); auto run = options(); CanonicalBundle task;
    task.prepare(fixture.index,fixture.scene,{0,1},0,{},run,32u<<20);
    require(!task.problem.num_points && !task.problem.num_obs && task.authority_observations == 24 && task.retained_observations == 24,"rank-zero block uploaded an unobservable point/row");
    rejects([&] { task.solve(run); },"rank-zero bundle fabricated a successful zero-cost solve");
    const auto before = fixture.scene.poses(); NativeSceneTasks scheduler(fixture.index,fixture.scene,{{{0,1},{0,1}}},{},run,256u<<20);
    fixture.scene.beginCandidate(); std::vector<CanonicalBundle> unused; scheduler.execute()(unused,fixture.scene,run);
    const auto stats = scheduler.statistics(); require(stats.completed == 1 && stats.retainedOnlyTasks == 1 && stats.retainedOnlyRows == 24 && !stats.gpuCompleted && !stats.contextCreates && stats.solverComputeSeconds == 0,"rank-zero task allocated/ran a geometric solver");
    for (const auto& image : before.images) { const auto& a = image.second.pose; const auto& b = fixture.scene.poses().images.at(image.first).pose; require(a.R == b.R && a.t.x == b.t.x && a.t.y == b.t.y && a.t.z == b.t.z,"rank-zero task changed a camera"); }
    const auto visual = sceneVisualEvidence(fixture.index,fixture.scene,3,16u<<20,1); for (auto count : visual.landmarks) require(!count,"retained-only rows fabricated metric visual evidence");
    require(fixture.index.observationCount() == 24 && fixture.scene.observations() == 24,"rank-zero task discarded original raw");
    fixture.scene.abort("retained-only-no-solver-proof"); fixture.originalUnchanged();
}
static int test(int argc,char** argv) {
    const Path root = argc > 1 ? std::filesystem::u8path(argv[1]) : std::filesystem::temp_directory_path()/("retained-bundle-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (std::filesystem::exists(root)) throw std::runtime_error("fixture destination must be new"); std::filesystem::create_directories(root);
    mixed(root/"mixed"); retainedOnly(root/"retained-only"); std::cout<<"PASS retained raw authority, native BA exclusion, camera independence, rank-zero scheduling and metric-landmark guard; CPU only; full_capture_verified=false\n"; return 0;
}
int main(int argc,char** argv) { return sfmTestMain(argc,argv,test); }
