#include "sfm/map/GpuRawQuality.h"
#include "sfm/map/CanonicalLandmarkGeometry.h"
#include "sfm/tests/TestMain.h"
#include <iomanip>
#include <iostream>

using namespace sfm;
using namespace sfm::regional;
static void require(bool condition, const std::string& reason) { if (!condition) throw std::runtime_error(reason); }
static Reconstruction poses(CamModel model, double scale) {
    Reconstruction result; auto camera = Camera::defaultFor(4,4000,3000,2300,model); camera.pixel_scale = scale;
    if (model == CamModel::Radial || model == CamModel::OpenCV || model == CamModel::FullOpenCV) { camera.k1 = .015; camera.k2 = -.001; }
    if (model == CamModel::OpenCV || model == CamModel::FullOpenCV) { camera.p1 = .0003; camera.p2 = -.0002; camera.fy = 2290; }
    if (model == CamModel::FullOpenCV) { camera.k3 = .00001; camera.k4 = .0002; camera.k5 = -.00001; camera.k6 = .000001; }
    result.cameras.emplace(4,camera); Image image; image.id = 7; image.camera_id = 4; image.registered = true; image.name = "quality-photo";
    image.pose = {angleAxisToRotation({.07,-.04,.03}),{-.4,.1,-.2}}; result.images.emplace(7,image); return result;
}
static std::vector<GpuQualityRow> rows(const Reconstruction& model) {
    const auto& image = model.images.at(7); const auto& camera = model.cameras.at(4); std::vector<GpuQualityRow> output;
    for (uint32_t i = 0; i < 5007; ++i) {
        const uint32_t kind = i % 11 == 0 ? 2 : (i % 5 == 0 ? 1 : 0);
        Vec3 point{.13 * std::sin(i * .3),.2 * std::cos(i * .4),5 + .01 * (i % 80)};
        if (kind == 1) point = Vec3{point.x,point.y,1}.normalized();
        Vec2 observation = camera.project(mul(image.pose.R,point) + (kind == 0 ? image.pose.t : Vec3{}));
        observation.x += camera.pixel_scale * (.3 + .07 * (i % 80)); observation.y -= camera.pixel_scale * .21;
        if (kind == 2) { point = {}; observation = {double(i),-double(i)}; }
        output.push_back({point,observation,7,kind});
    }
    return output;
}
static std::vector<GpuQualityPartial> reference(const Reconstruction& model, const std::vector<GpuQualityRow>& input, double gate) {
    std::vector<GpuQualityPartial> result((input.size() + 2047) / 2048);
    for (size_t i = 0; i < input.size(); ++i) {
        const auto& row = input[i]; auto& q = result[i / 2048]; ++q.observations;
        if (row.kind == 2) { ++q.retained_observations; continue; }
        q.directional_observations += row.kind == 1;
        const auto& image = model.images.at(row.image); const auto& camera = model.cameras.at(image.camera_id);
        bool valid = true; double error = reprojLandmarkError(camera,image.pose,row.xy,{row.value,LandmarkKind(row.kind)},&valid) / camera.pixel_scale;
        const auto cost = error <= gate ? error * error : gate * (2 * error - gate);
        q.cost += cost; if (valid) q.finite_cost += cost; else ++q.invalid_projections;
    }
    return result;
}
static double compare(const GpuRawQualityResult& gpu, const std::vector<GpuQualityPartial>& cpu) {
    require(gpu.ready(),"GPU quality unavailable: " + gpu.reason); require(gpu.partials.size() == cpu.size(),"quality partial ordering/count differs");
    double maxRelative = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        const auto& a = gpu.partials[i]; const auto& b = cpu[i];
        require(a.observations == b.observations && a.invalid_projections == b.invalid_projections &&
                a.retained_observations == b.retained_observations && a.directional_observations == b.directional_observations && !a.fallback_rows,"quality typed coverage differs");
        for (auto pair : {std::pair<double,double>{a.cost,b.cost},{a.finite_cost,b.finite_cost}}) {
            const auto relative = std::abs(pair.first - pair.second) / std::max(1.,pair.second); maxRelative = std::max(maxRelative,relative);
            require(relative < 2e-10,"F64 quality cost differs from native CPU double");
        }
    }
    return maxRelative;
}
static void cpuOnlyTests() {
    GpuRawQuality quality;
    require(quality.evaluate({}).status == GpuRawQualityStatus::InvalidInput,"uninitialized quality context accepted");
    auto base = poses(CamModel::OpenCV,2);
    require(quality.begin(base,0) == GpuRawQualityStatus::InvalidInput,"invalid gate accepted");
    require(quality.begin(base,.1) == GpuRawQualityStatus::UnsupportedGate,"double gate silently rounded to float");
    auto wide = poses(CamModel::Equirect,2);
    require(quality.begin(wide,3) == GpuRawQualityStatus::UnsupportedCamera,"spherical camera silently evaluated by pinhole shader");
    auto foreign = base; foreign.cameras.at(4).k3 = .01;
    require(quality.begin(foreign,3) == GpuRawQualityStatus::UnsupportedCamera,"unused native camera coefficient silently dropped");
    GpuRawQualityOptions limitedOptions; limitedOptions.host_budget_bytes = limitedOptions.device_budget_bytes = 1024; GpuRawQuality limited(limitedOptions);
    require(limited.begin(base,3) == GpuRawQualityStatus::BudgetExceeded,"quality ignored explicit bounded allowance");
    auto sparse = base; auto image = sparse.images.at(7); sparse.images.clear(); image.id = 0xFFFFFFFE; sparse.images.emplace(image.id,image);
    require(quality.begin(sparse,3) == GpuRawQualityStatus::BudgetExceeded,"sparse global image slots allocated without budget guard");
}
static void gpuTests() {
    GpuRawQuality quality; double maxRelative = 0;
    for (auto model : {CamModel::SimplePinhole,CamModel::Pinhole,CamModel::Radial,CamModel::OpenCV,CamModel::FullOpenCV}) for (double scale : {2.,3.}) {
        auto current = poses(model,scale); auto input = rows(current);
        require(quality.begin(current,3) == GpuRawQualityStatus::Ready,"quality begin failed: " + quality.reason());
        auto actual = quality.evaluate(input); const auto referenceValue = reference(current,input,3);
        maxRelative = std::max(maxRelative,compare(actual,referenceValue));
        const auto again = quality.evaluate(input); require(again.ready() && again.partials.size() == actual.partials.size(),"repeat quality failed");
        for (size_t i = 0; i < actual.partials.size(); ++i) require(actual.partials[i].cost == again.partials[i].cost,"fixed reduction order changed between identical batches");
        std::cout << "GPU quality model=" << camBaModel(model) << " pixel_scale=" << scale << " rows=" << input.size() << " partials=" << actual.partials.size() << " max_relative=" << maxRelative << std::endl;
        if (model != CamModel::OpenCV || scale != 2) continue;
        auto changed = input; changed.back().xy.x += 60;
        const auto changedResult = quality.evaluate(changed); compare(changedResult,reference(current,changed,3));
        require(changedResult.partials.back().cost > actual.partials.back().cost,"last batch row was skipped or stale");
        auto bad = input; bad.back().value = {0,0,-20}; bad.back().kind = 0;
        auto badResult = quality.evaluate(bad); require(badResult.status == GpuRawQualityStatus::RequiresCpuFallback,"behind finite point silently certified");
        uint64_t invalid = 0; for (const auto& part : badResult.partials) invalid += part.invalid_projections; require(invalid == 1,"behind-row diagnostic invalid count differs");
        auto boundaryRows = input; const auto& image = current.images.at(7);
        boundaryRows.back().value = mul(transpose(image.pose.R),Vec3{0,0,1e-8} - image.pose.t); boundaryRows.back().kind = 0;
        require(quality.evaluate(boundaryRows).status == GpuRawQualityStatus::RequiresCpuFallback,"cheirality-boundary roundoff did not fall back");
        auto antipodal = input; antipodal.back().value = Vec3{0,0,-1}; antipodal.back().kind = 1;
        require(quality.evaluate(antipodal).status == GpuRawQualityStatus::RequiresCpuFallback,"antipodal direction silently certified");
        auto malformed = input; malformed.back().kind = 3;
        require(quality.evaluate(malformed).status == GpuRawQualityStatus::InvalidInput,"unknown primitive accepted");
        malformed = input; malformed.back().kind = 2; malformed.back().value = {1,0,0};
        require(quality.evaluate(malformed).status == GpuRawQualityStatus::InvalidInput,"retained point acquired fake XYZ");
        malformed = input; malformed.resize(quality.capacity() + 1);
        require(quality.evaluate(malformed).status == GpuRawQualityStatus::BudgetExceeded,"batch capacity ignored");
        std::vector<GpuQualityRow> directions; for (const auto& row : input) if (row.kind == 1) directions.push_back(row);
        const auto directionalBefore = quality.evaluate(directions); require(directionalBefore.ready(),"direction-only quality failed");
        current.images.at(7).pose.t = {500,-200,800};
        require(quality.begin(current,3) == GpuRawQualityStatus::Ready,"changed view upload failed");
        const auto directionalAfter = quality.evaluate(directions); compare(directionalAfter,reference(current,directions,3));
        for (size_t i = 0; i < directionalBefore.partials.size(); ++i) require(directionalBefore.partials[i].cost == directionalAfter.partials[i].cost,"translation influenced homogeneous direction");
        current = poses(model,scale); current.cameras.at(4).fx += 60; current.cameras.at(4).fy += 30;
        require(quality.begin(current,3) == GpuRawQualityStatus::Ready,"intrinsics refresh failed");
        const auto updated = quality.evaluate(input); compare(updated,reference(current,input,3));
        require(updated.partials.front().cost != actual.partials.front().cost,"begin reused stale candidate intrinsics");
    }
    std::cout << "PASS all 5 supported camera models, 10 F64 GPU/scaled cases; max_relative=" << maxRelative << std::endl;
}
static int test(int argc, char** argv) {
    std::cout << std::setprecision(17); bool cpuOnly = false;
    for (int i = 1; i < argc; ++i) cpuOnly = cpuOnly || std::string(argv[i]) == "--cpu-only";
    cpuOnlyTests(); if (!cpuOnly) gpuTests(); else std::cout << "GPU explicitly not run (--cpu-only)\n";
    std::cout << "PASS quality-only Vulkan contract; full_capture_verified=false\n"; return 0;
}
int main(int argc, char** argv) { return sfmTestMain(argc,argv,test); }
