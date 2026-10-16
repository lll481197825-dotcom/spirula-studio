#include "sfm/ba/Solver.h"
#include "sfm/tests/SyntheticBA.h"
#include "sfm/tests/TestMain.h"
#include <iostream>
#include <iomanip>

using namespace sfm;
static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> static void rejects(F fn, const char* message) {
    bool rejected = false; try { fn(); } catch (const std::exception&) { rejected = true; } require(rejected, message);
}
static double relative(const std::vector<double>& a, const std::vector<double>& b) {
    require(a.size() == b.size(), "normal-equation dimensions differ"); double error = 0, scale = 1;
    for (size_t i = 0; i < a.size(); ++i) { error = std::max(error, std::fabs(a[i] - b[i])); scale = std::max(scale, std::fabs(a[i])); }
    return error / scale;
}
static std::vector<double> intrinsic(uint32_t model, double pixelScale) {
    int count = 0; const auto* initial = synth::defaultIntr(model, count); std::vector<double> result(initial, initial + count);
    if (model < 2) { result[0] = model == 0 ? result[0] - std::log(pixelScale) : result[0] / pixelScale; return result; }
    Camera camera; for (const auto& info : kCamModelInfo) if (uint32_t(info.ba_model) == model) camera.model = info.model;
    unpackIntrinsics(camera, result.data()); camera.fx /= pixelScale; camera.fy /= pixelScale; camera.cx /= pixelScale; camera.cy /= pixelScale;
    packIntrinsics(camera, result.data()); return result;
}
template<class M> static void truthProjection(const double pose[6], const double* ext, const double* intr,
                                               const double point[3], bool direction, double xy[2]) {
    double camera[3]; bacpu::angleAxisRotate<double>(pose, point, camera);
    if (!direction) for (int axis = 0; axis < 3; ++axis) camera[axis] += pose[3 + axis];
    if (ext) {
        double rotated[3]; bacpu::angleAxisRotate<double>(ext, camera, rotated);
        for (int axis = 0; axis < 3; ++axis) camera[axis] = rotated[axis] + (direction ? 0 : ext[3 + axis]);
    }
    M::template project<double>(intr, camera, xy);
}
static BAProblem problem(uint32_t model, bool finite, bool freeIntrinsics, double pixelScale = 2) {
    BAProblem p; p.num_images = 3; p.num_points = finite ? 48 : 24; p.num_obs = p.num_images * p.num_points;
    p.identityFrames(); p.poses.resize(18); p.intr = intrinsic(model, pixelScale); p.total_intr = uint32_t(p.intr.size());
    const uint32_t free = freeIntrinsics && model != 9 ? p.total_intr : 0;
    p.groups.push_back({0, 18, free, model}); p.image_group.assign(3, 0); p.pose_dim = 18; p.n_dim = 18 + free; p.free_intr = free;
    for (uint32_t image = 0; image < 3; ++image) {
        const Vec3 aa{.01 * image, -.015 * image, .008 * image};
        const auto R = angleAxisToRotation(aa); const Vec3 center{100 + .3 * image, -80 + .2 * image, 12 + .1 * image};
        const auto t = mul(R, center) * -1.; p.poses[6 * image] = aa.x; p.poses[6 * image + 1] = aa.y; p.poses[6 * image + 2] = aa.z;
        p.poses[6 * image + 3] = t.x; p.poses[6 * image + 4] = t.y; p.poses[6 * image + 5] = t.z;
    }
    p.points.resize(3 * p.num_points); p.point_kinds.resize(p.num_points); p.fixed_points.assign(p.num_points, 1); p.obs_ranges.push_back(0);
    for (uint32_t point = 0; point < p.num_points; ++point) {
        const bool direction = !finite || point >= 24; p.point_kinds[point] = uint8_t(direction);
        const double sign = model < 2 ? -1. : 1.;
        const Vec3 value = direction ? Vec3{.03 * (point % 24) - .35, .1 * std::cos(point), sign}.normalized() : Vec3{100 + 2 * std::sin(point), -80 + 2 * std::cos(point), 12 + sign * (10 + .07 * point)};
        p.points[3 * point] = value.x; p.points[3 * point + 1] = value.y; p.points[3 * point + 2] = value.z;
        for (uint32_t image = 0; image < 3; ++image) {
            double xy[2];
            bacpu::withModel(model, [&](auto camera) { truthProjection<decltype(camera)>(&p.poses[6 * image], nullptr, p.intr.data(), &p.points[3 * point], direction, xy); });
            require(std::isfinite(xy[0]) && std::isfinite(xy[1]), "fixture direction is not in front of its observation");
            p.obs_image.push_back(image); p.obs_point.push_back(point); p.obs_xy.push_back(xy[0] + .13); p.obs_xy.push_back(xy[1] - .07);
        }
        p.obs_ranges.push_back(uint32_t(p.obs_image.size()));
    }
    finalizeTables(p); return p;
}
static SolverOptions options(RealCfg real = RealCfg::CPU) {
    SolverOptions result; result.real = real; result.solver = SolverSel::Dense; result.cg_fallback = CgFallback::Off;
    result.verbose = false; result.loss = "trivial"; result.threads = 2; result.max_iters = 12; result.rtol = 1e-12; result.init_damping = .001; return result;
}
static BAProblem legacySubset(const BAProblem& source, bool directions) {
    BAProblem result = source; result.num_points = result.num_obs = 0; result.points.clear(); result.point_kinds.clear(); result.fixed_points.clear();
    result.obs_image.clear(); result.obs_point.clear(); result.obs_xy.clear(); result.obs_ranges = {0};
    if (directions) {
        for (uint32_t frame = 0; frame < result.num_frames; ++frame) for (uint32_t axis = 3; axis < 6; ++axis) result.poses[6 * frame + axis] = 0;
        for (const auto& member : result.members) for (uint32_t axis = 3; axis < 6; ++axis) result.exts[member.ext_offset + axis] = 0;
    }
    for (uint32_t point = 0; point < source.num_points; ++point) if (source.directionalPoint(point) == directions) {
        result.points.insert(result.points.end(), source.points.begin() + 3 * point, source.points.begin() + 3 * point + 3); result.fixed_points.push_back(1);
        for (uint32_t row = source.obs_ranges[point]; row < source.obs_ranges[point + 1]; ++row) {
            result.obs_image.push_back(source.obs_image[row]); result.obs_point.push_back(result.num_points);
            result.obs_xy.push_back(source.obs_xy[2 * row]); result.obs_xy.push_back(source.obs_xy[2 * row + 1]);
        }
        ++result.num_points; result.obs_ranges.push_back(uint32_t(result.obs_image.size()));
    }
    result.num_obs = uint32_t(result.obs_image.size()); finalizeTables(result); return result;
}
static void addLegacySystem(const BAProblem& problem, bool directions, const std::vector<double>& matrix,
                            const std::vector<double>& gradient, std::vector<double>& sumMatrix, std::vector<double>& sumGradient) {
    auto translationColumn = [&](uint32_t column) {
        if (column < problem.pose_dim) return column % 6 >= 3;
        for (const auto& member : problem.members) {
            uint32_t free = 0;
            for (uint32_t axis = 0; axis < 6; ++axis) if (member.mask & (1u << axis)) {
                if (free < member.n_free && column == member.ext_col + free) return axis >= 3;
                ++free;
            }
        }
        return false;
    };
    if (sumMatrix.empty()) { sumMatrix.assign(matrix.size(),0); sumGradient.assign(gradient.size(),0); }
    for (uint32_t row = 0; row < problem.n_dim; ++row) {
        if (directions && translationColumn(row)) continue;
        sumGradient[row] += gradient[row];
        for (uint32_t column = 0; column <= row; ++column) if (!directions || !translationColumn(column))
            sumMatrix[size_t(row) * (row + 1) / 2 + column] += matrix[size_t(row) * (row + 1) / 2 + column];
    }
}
template<class M> static double derivative(uint32_t model, bool rig, bool zeroRotation, double pixelScale) {
    const auto intrinsics = intrinsic(model, pixelScale); auto intr = intrinsics; constexpr int NI = M::kNumIntr;
    const int poseDof = rig ? 12 : 6, dof = poseDof + NI;
    double pose[6]{.12, -.2, .07, 123, -456, 789}, ext[6]{.05, -.03, .08, 5, 12, -6};
    if (zeroRotation) std::fill(pose, pose + 3, 0.);
    const auto d = Vec3{.2, -.1, model < 2 ? -1. : 1.}.normalized(); double point[3]{d.x, d.y, d.z}, obs[2]{}, projected[2];
    truthProjection<M>(pose, rig ? ext : nullptr, intr.data(), point, true, projected);
    obs[0] = projected[0] + .3; obs[1] = projected[1] - .4;
    double residual[2], jc[2 * kMaxCamDof], jp[6];
    if (rig) bacpu::jacobianRig<M>(pose, ext, intr.data(), point, obs, residual, jc, jp, true);
    else bacpu::jacobian<M>(pose, intr.data(), point, obs, residual, jc, jp, true);
    double worst = 0;
    for (int column = 0; column < dof; ++column) {
        double* parameter = column < 6 ? pose + column : rig && column < 12 ? ext + column - 6 : intr.data() + column - poseDof;
        const auto initial = *parameter, epsilon = 1e-6 * std::max(1., std::fabs(initial)); double a[2], b[2];
        *parameter = initial + epsilon;
        if (rig) bacpu::residualRig<M>(pose, ext, intr.data(), point, obs, a, true); else bacpu::residual<M>(pose, intr.data(), point, obs, a, true);
        *parameter = initial - epsilon;
        if (rig) bacpu::residualRig<M>(pose, ext, intr.data(), point, obs, b, true); else bacpu::residual<M>(pose, intr.data(), point, obs, b, true);
        *parameter = initial;
        for (int row = 0; row < 2; ++row) {
            const auto numeric = (a[row] - b[row]) / (2 * epsilon), analytic = jc[row * dof + column];
            require(std::isfinite(numeric) && std::isfinite(analytic), "typed derivative is nonfinite");
            worst = std::max(worst, std::fabs(numeric - analytic) / std::max(1., std::fabs(numeric) + std::fabs(analytic)));
            if ((column >= 3 && column < 6) || (rig && column >= 9 && column < 12)) require(analytic == 0 && numeric == 0, "direction translation derivative is nonzero");
        }
    }
    for (double value : jp) require(value == 0, "fixed direction acquired movable 3D derivatives");
    require(worst < 2e-6, "typed rotational/intrinsic Jacobian differs from central differences"); return worst;
}
static void cpuTests() {
    double worst = 0;
    for (uint32_t model = 0; model < kNumModels; ++model) bacpu::withModel(model, [&](auto camera) {
        for (bool rig : {false, true}) for (bool zero : {false, true}) for (double scale : {.5, 2.})
            worst = std::max(worst, derivative<decltype(camera)>(model, rig, zero, scale));
    });
    auto p = problem(3, false, false); auto bad = p; bad.point_kinds.resize(1); rejects([&]{finalizeTables(bad);}, "point-kind size mismatch accepted");
    bad = p; bad.point_kinds[0] = 2; rejects([&]{finalizeTables(bad);}, "unknown point kind accepted");
    bad = p; bad.fixed_points[0] = 0; rejects([&]{finalizeTables(bad);}, "movable direction accepted");
    bad = p; bad.points[0] *= 2; rejects([&]{finalizeTables(bad);}, "nonunit direction accepted");
    auto finite = problem(3, true, false); finite.point_kinds.clear(); auto zeros = finite; zeros.point_kinds.assign(zeros.num_points, 0);
    bacpu::Solver legacy(finite, options()), explicitFinite(zeros, options()); legacy.init(); explicitFinite.init();
    require(legacy.computeCost() == explicitFinite.computeCost(), "empty kinds changed finite BA math");
    bacpu::Solver cpu(p, options()); cpu.init(); const auto before = cpu.computeCost(); cpu.assembleOnly(.001);
    const auto gradient = cpu.gradient(); for (uint32_t frame = 0; frame < 3; ++frame) for (uint32_t axis = 3; axis < 6; ++axis) require(gradient[6 * frame + axis] == 0, "direction camera system has translation information");
    const auto fixed = p.points; std::vector<double> translations; for (uint32_t frame = 0; frame < 3; ++frame) for (uint32_t axis = 3; axis < 6; ++axis) translations.push_back(p.poses[6 * frame + axis]);
    cpu.solve(); require(cpu.stats().final_cost < before * .1, "direction BA failed to improve actual rotation residuals"); require(p.points == fixed, "CPU BA changed fixed directions");
    size_t translation = 0; for (uint32_t frame = 0; frame < 3; ++frame) for (uint32_t axis = 3; axis < 6; ++axis) require(p.poses[6 * frame + axis] == translations[translation++], "direction-only BA changed translation");
    p.point_kinds[0] = 0; rejects([&]{cpu.computeCost();}, "CPU cached type mutation accepted");
    for (uint32_t model : {0u,3u,6u,8u,9u}) {
        auto antipodal = problem(model, false, false); for (int axis = 0; axis < 3; ++axis) antipodal.points[axis] = -antipodal.points[axis];
        rejects([&]{bacpu::Solver invalid(antipodal, options()); invalid.init();}, "antipodal direction accepted by initial BA");
    }
    auto seam = problem(9, false, false); seam.points[0] = 1e-7; seam.points[1] = 0; seam.points[2] = -std::sqrt(1 - 1e-14);
    double pose[6]{0,0,0,100,-40,70}, xy[2]{640,240}, r[2], jc[16], jp[6]; const auto intr = intrinsic(9,1);
    bacpu::jacobian<bacpu::EquirectModel>(pose,intr.data(),seam.points.data(),xy,r,jc,jp,true);
    require(std::fabs(r[0]) < 1e-3 && std::fabs(r[1]) < 1e-12, "direction panorama seam was not wrapped");
    std::cout<<"CPU typed derivatives all10 models/rigs/scales; max_relative="<<worst<<"; fixed/antipodal/cache/seam contracts PASS\n";
}
static void gpuCase(VkContext& context, uint32_t model, RealCfg real, double scale, bool rig = false) {
    auto base = problem(model, true, true, scale);
    if (rig) {
        base.image_member = {0, 1, 0}; base.members = {{0, 0, 0, kExtAll}, {6, 18, 6, kExtAll}};
        base.exts = {.01, 0, .02, 1, -.5, .2, .03, -.02, .05, 2, -.7, 1};
        base.ext_dim = 6; base.groups[0].intr_col = 24; base.n_dim = 24 + base.free_intr;
        for (uint32_t row = 0; row < base.num_obs; ++row) {
            const auto image = base.obs_image[row], point = base.obs_point[row]; double xy[2];
            const auto& member = base.members[base.image_member[image]];
            bacpu::withModel(model, [&](auto camera) { truthProjection<decltype(camera)>(&base.poses[6 * image], &base.exts[member.ext_offset], base.intr.data(), &base.points[3 * point], base.directionalPoint(point), xy); });
            require(std::isfinite(xy[0]) && std::isfinite(xy[1]), "rig fixture is not in its projection domain");
            base.obs_xy[2 * row] = xy[0] + .13; base.obs_xy[2 * row + 1] = xy[1] - .07;
        }
        finalizeTables(base);
    }
    auto cpuProblem = base, gpuProblem = base;
    const auto cpuOptions = options(), gpuOptions = options(real);
    bacpu::Solver cpu(cpuProblem, cpuOptions); cpu.init(); cpu.assembleOnly(.001);
    const auto expectedCost = cpu.computeCost();
    const double tolerance = real == RealCfg::F64 ? 2e-8 : 8e-4;
    double legacyHostCost = 0, legacyDeviceCost = 0;
    std::vector<double> legacyHostMatrix, legacyHostGradient, legacyDeviceMatrix, legacyDeviceGradient;
    for (bool directions : {false,true}) {
        auto control = legacySubset(base,directions); bacpu::Solver host(control,cpuOptions); host.init(); legacyHostCost += host.computeCost();
        host.assembleOnly(.001); addLegacySystem(control,directions,host.packedS(),host.gradient(),legacyHostMatrix,legacyHostGradient);
        BundleSolver device(control,gpuOptions,&context); device.init(); legacyDeviceCost += device.computeCost(); device.debugAssemble(.001);
        addLegacySystem(control,directions,device.debugPackedS(),device.debugG(),legacyDeviceMatrix,legacyDeviceGradient);
    }
    BundleSolver gpu(gpuProblem, gpuOptions, &context); gpu.init(); require(gpu.real() == real, "direction GPU parity silently fell back to another scalar backend");
    const auto actualCost = gpu.computeCost();
    const double legacyNumericError = std::fabs(legacyDeviceCost - legacyHostCost), roundoff = tolerance * std::max(1.,expectedCost);
    std::cout<<"GPU diagnostic model="<<model<<" real="<<realCfgName(real)<<" actual_real="<<realCfgName(gpu.real())<<" rig="<<rig<<" pixel_scale="<<scale<<" expected_cost="<<expectedCost<<" actual_cost="<<actualCost<<" legacy_host_cost="<<legacyHostCost<<" legacy_device_cost="<<legacyDeviceCost<<" legacy_numeric_error="<<legacyNumericError<<" typed_delta="<<(actualCost-legacyDeviceCost)<<" kind_offset="<<base.pointKindTableOffset()<<std::endl;
    require(std::fabs(expectedCost - legacyHostCost) < 1e-12 * std::max(1.,expectedCost), "typed CPU cost differs from independent legacy finite controls");
    require(std::fabs(actualCost - legacyDeviceCost) < roundoff, "typed GPU cost differs from independent legacy finite controls");
    require(std::fabs(expectedCost - actualCost) < legacyNumericError + roundoff, "typed CPU/GPU cost exceeds measured legacy numeric error");
    gpu.debugAssemble(.001); const auto hostMatrix = cpu.packedS(), hostGradient = cpu.gradient(), deviceMatrix = gpu.debugPackedS(), deviceGradient = gpu.debugG();
    const auto matrixError = relative(hostMatrix,deviceMatrix), gradientError = relative(hostGradient,deviceGradient);
    const auto legacyMatrixError = relative(legacyHostMatrix,legacyDeviceMatrix), legacyGradientError = relative(legacyHostGradient,legacyDeviceGradient);
    const auto typedMatrixDelta = relative(legacyDeviceMatrix,deviceMatrix), typedGradientDelta = relative(legacyDeviceGradient,deviceGradient);
    std::cout<<"GPU system diagnostic model="<<model<<" rig="<<rig<<" matrix="<<matrixError<<" gradient="<<gradientError<<" legacy_matrix="<<legacyMatrixError<<" legacy_gradient="<<legacyGradientError<<" typed_matrix_delta="<<typedMatrixDelta<<" typed_gradient_delta="<<typedGradientDelta<<std::endl;
    require(relative(legacyHostMatrix,hostMatrix) < 1e-10 && relative(legacyHostGradient,hostGradient) < 1e-10, "typed CPU system differs from independent projected legacy finite controls");
    require(typedMatrixDelta < tolerance && typedGradientDelta < tolerance, "typed GPU system differs from independent projected legacy finite controls");
    require(matrixError < legacyMatrixError + tolerance && gradientError < legacyGradientError + tolerance, "typed CPU/GPU systems exceed measured legacy numeric error");
    const auto fixed = gpuProblem.points; gpu.solve(); gpu.downloadParams();
    for (uint32_t point = 0; point < gpuProblem.num_points; ++point) if (gpuProblem.directionalPoint(point))
        for (uint32_t axis = 0; axis < 3; ++axis) require(gpuProblem.points[3*point+axis] == fixed[3*point+axis], "GPU roundtrip changed a fixed direction");
    gpuProblem.point_kinds[24] = 0; rejects([&]{gpu.uploadParams();}, "GPU cached kind mutation accepted"); gpuProblem.point_kinds[24] = 1;
    std::cout<<"GPU model="<<model<<" real="<<realCfgName(real)<<" rig="<<rig<<" pixel_scale="<<scale<<" matrix="<<matrixError<<" gradient="<<gradientError<<" PASS\n";
}
static void gpuTests() {
    VkContext context;
    for (uint32_t model = 0; model < kNumModels; ++model) gpuCase(context,model,RealCfg::F64,2);
    gpuCase(context,3,RealCfg::F64,.5);
    gpuCase(context,3,RealCfg::F64,2,true);
    for (RealCfg real : {RealCfg::F32,RealCfg::DF64}) if (realSupportedByDevice(real, context.caps())) {
        const auto blob = std::string("ba_") + realCfgName(real) + "_trivial"; size_t words = 0;
        if (!sfm::findSpirv(blob.c_str(),&words)) { std::cout<<"GPU "<<realCfgName(real)<<" explicitly not run: embedded shader variant is not configured\n"; continue; }
        VkContext scalarContext; gpuCase(scalarContext,3,real,2);
    }
    for (uint32_t model : {3u,6u,8u,9u}) {
        auto p = problem(model,false,false); BundleSolver solver(p,options(RealCfg::F64),&context); solver.init(); const auto cost = solver.computeCost();
        const auto original = p.points;
        for (int axis = 0; axis < 3; ++axis) p.points[axis] = -p.points[axis]; solver.uploadParams();
        require(!std::isfinite(solver.computeCost()), "GPU antipodal direction became a finite good projection");
        p.points = original; for (uint32_t image = 0; image < p.num_images; ++image) for (uint32_t axis = 3; axis < 6; ++axis) p.poses[6*image+axis] += 1e6;
        solver.uploadParams(); require(solver.computeCost() == cost, "GPU direction cost depends on translation");
        solver.debugAssemble(.001); const auto g = solver.debugG(); for (uint32_t image = 0; image < p.num_images; ++image) for (uint32_t axis = 3; axis < 6; ++axis) require(g[6*image+axis] == 0, "GPU direction translation gradient is not zero");
    }
}
static void diagnoseCosts() {
    VkContext context;
    for (uint32_t model : {0u,3u,6u,8u,9u}) for (bool zeroRotation : {true,false}) {
        auto base = problem(model,false,false);
        for (uint32_t image = 0; image < base.num_images; ++image) {
            for (uint32_t axis = 3; axis < 6; ++axis) base.poses[6 * image + axis] = 0;
            if (zeroRotation) for (uint32_t axis = 0; axis < 3; ++axis) base.poses[6 * image + axis] = 0;
        }
        for (uint32_t row = 0; row < base.num_obs; ++row) {
            double xy[2]; const auto image = base.obs_image[row], point = base.obs_point[row];
            bacpu::withModel(model,[&](auto camera) { truthProjection<decltype(camera)>(&base.poses[6*image],nullptr,base.intr.data(),&base.points[3*point],true,xy); });
            base.obs_xy[2 * row] = xy[0] + .13; base.obs_xy[2 * row + 1] = xy[1] - .07;
        }
        double device[2], host[2];
        for (int typed = 0; typed < 2; ++typed) {
            auto p = base; if (!typed) p.point_kinds.clear();
            bacpu::Solver cpu(p,options()); cpu.init(); host[typed] = cpu.computeCost();
            BundleSolver gpu(p,options(RealCfg::F64),&context); gpu.init(); device[typed] = gpu.computeCost();
        }
        std::cout<<"CONTROL model="<<model<<" zero_rotation="<<zeroRotation<<" host_finite="<<host[0]<<" host_direction="<<host[1]<<" device_finite="<<device[0]<<" device_direction="<<device[1]<<" native_error="<<(device[0]-host[0])<<" typed_delta="<<(device[1]-device[0])<<std::endl;
    }
}
static int test(int argc,char** argv) {
    std::cout<<std::setprecision(17); bool cpuOnly = false, diagnostic = false; for (int i=1;i<argc;++i) { cpuOnly = cpuOnly || std::string(argv[i])=="--cpu-only"; diagnostic = diagnostic || std::string(argv[i])=="--diagnose-cost"; }
    if (diagnostic) { diagnoseCosts(); return 0; }
    cpuTests(); if (!cpuOnly) gpuTests(); else std::cout<<"GPU tests explicitly not run (--cpu-only)\n";
    std::cout<<"PASS native fixed directional BA; full_capture_verified=false\n"; return 0;
}
int main(int argc,char** argv) { return sfmTestMain(argc,argv,test); }
