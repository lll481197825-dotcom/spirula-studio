#pragma once
#include "sfm/map/SceneQuality.h"
#include "sfm/map/Bundle.h"

namespace sfm::regional {
class CanonicalBundle {
public:
    BAProblem problem;
    PosePriors priors;
    std::vector<uint32_t> images, cameras;
    std::vector<double> cameraPixelScales;
    std::vector<uint64_t> points;
    std::string topologyIdentity;
    std::string landmarkKindsIdentity;
    uint64_t forecastHostBytes = 0;
    uint64_t authority_observations = 0, retained_observations = 0;
    uint64_t residentBytes() const { return storage::addBytes(bundleProblemHostBytes(problem),
        storage::scaleBytes(points.capacity(), sizeof(uint64_t)) + storage::scaleBytes(images.capacity() + cameras.capacity(), sizeof(uint32_t)) + storage::scaleBytes(cameraPixelScales.capacity(), sizeof(double)) +
        storage::scaleBytes(priors.rotations.capacity(), sizeof(PriorRotation)) + storage::scaleBytes(priors.ups.capacity(), sizeof(PriorUp)) + storage::scaleBytes(priors.centres.capacity(), sizeof(PriorCentre)) + landmarkKindsIdentity.capacity()); }

    void prepare(const ImmutableSceneTopology& index, const CanonicalScene& scene,
                 std::vector<uint32_t> selected, uint32_t owner, const std::set<uint32_t>& anchors,
                 const BundleOptions& options, size_t allowance) {
        if (index.identity() != scene.binding().topology) throw std::runtime_error("canonical bundle identity differs");
        if (options.rigs && options.use_rigs && !options.rigs->empty()) throw std::runtime_error("canonical bundle rig layout is not implemented");
        images = std::move(selected); images.insert(images.end(), anchors.begin(), anchors.end());
        std::sort(images.begin(), images.end()); images.erase(std::unique(images.begin(), images.end()), images.end());
        if (images.empty() || images.size() > UINT32_MAX) throw std::length_error("invalid canonical bundle photo count");
        std::vector<uint32_t> local(size_t(index.imageSlots()), UINT32_MAX);
        if (storage::scaleBytes(local.size(), sizeof(uint32_t)) > allowance / 8) throw std::length_error("canonical bundle photo map exceeds allowance");
        for (size_t i = 0; i < images.size(); ++i) { if (images[i] >= local.size() || !scene.poses().images.at(images[i]).registered) throw std::runtime_error("canonical bundle image not registered"); local[images[i]] = uint32_t(i); }
        points.clear();
        const uint64_t seenWords = (index.pointSlots() + 63) / 64;
        if (storage::scaleBytes(seenWords, sizeof(uint64_t)) > allowance / 8) throw std::length_error("canonical bundle point directory exceeds allowance");
        std::vector<uint64_t> seen(size_t(seenWords), 0);
        for (uint32_t image : images) index.visitPhotoRoots(image, [&](uint64_t root) {
            auto& word = seen.at(size_t(root / 64)); const uint64_t bit = uint64_t(1) << (root % 64);
            if (word & bit) return; word |= bit;
            if (points.size() >= allowance / (16 * sizeof(uint64_t))) throw std::length_error("canonical bundle candidate directory exceeds allowance; split photo task");
            points.push_back(root);
        });
        seen.clear(); seen.shrink_to_fit();
        std::sort(points.begin(), points.end()); points.erase(std::unique(points.begin(), points.end()), points.end());
        uint64_t observations = 0, maximumTrack = 0; size_t geometricPoints = 0;
        authority_observations = retained_observations = 0;
        for (uint64_t root : points) {
            const auto landmark = scene.landmark(root); validateLandmark(landmark); uint64_t n = 0;
            index.visitPoint(root, [&](const auto& row) {
                if (landmark.kind == LandmarkKind::RetainedObservation && scene.canonical(row.point) != root) throw std::runtime_error("canonical retained raw alias differs");
                if (local[row.image] != UINT32_MAX) ++n;
            });
            authority_observations = storage::addBytes(authority_observations, n);
            if (landmark.kind == LandmarkKind::RetainedObservation) { retained_observations = storage::addBytes(retained_observations, n); continue; }
            points[geometricPoints++] = root; observations = storage::addBytes(observations, n); maximumTrack = std::max(maximumTrack, n);
        }
        points.resize(geometricPoints);
        if (points.size() > options.index_limit || observations > options.index_limit) throw std::length_error("canonical bundle native index capacity exceeded; split photo task");
        bundle_detail::HostShape shape; shape.images = std::max<size_t>(2, images.size()); shape.points = points.size(); shape.observations = observations; shape.max_track = maximumTrack; shape.image_span = images.size(); shape.max_dof = 6;
        shape.pair_entries = (long double)observations * std::max<uint64_t>(1, maximumTrack);
        BundleOptions fixedK = options; fixedK.refine_intrinsics = false;
        forecastHostBytes = storage::addBytes(bundle_detail::estimateHostBytes(shape, fixedK, false), storage::scaleBytes(points.capacity(), sizeof(uint64_t)) + storage::scaleBytes(local.size(), sizeof(uint32_t)));
        forecastHostBytes = storage::addBytes(forecastHostBytes, storage::scaleBytes(points.size(), 6));
        bundle_detail::checkHostBytes(forecastHostBytes, allowance);
        problem = BAProblem{}; auto& p = problem; p.index_limit = options.index_limit; p.num_images = uint32_t(images.size()); p.num_points = uint32_t(points.size()); p.num_obs = uint32_t(observations); p.identityFrames(); p.pose_dim = 6 * p.num_frames; p.n_dim = p.pose_dim;
        p.fixed_frames.assign(images.size(), 0); p.image_group.resize(images.size());
        cameras.clear(); cameraPixelScales.clear(); std::map<uint32_t, uint32_t> groupIndex;
        for (size_t i = 0; i < images.size(); ++i) {
            const auto& im = scene.poses().images.at(images[i]); const auto& camera = scene.poses().cameras.at(im.camera_id);
            if (!(camera.pixel_scale > 0) || !std::isfinite(camera.pixel_scale)) throw std::runtime_error("invalid canonical camera pixel scale");
            p.fixed_frames[i] = anchors.count(images[i]) != 0;
            auto inserted = groupIndex.emplace(camera.id, uint32_t(groupIndex.size()));
            if (inserted.second) { const uint32_t model = uint32_t(camBaModel(camera.model)); p.groups.push_back({uint32_t(p.intr.size()), uint32_t(images.size() * 6), 0, model}); cameras.push_back(camera.id); cameraPixelScales.push_back(camera.pixel_scale); p.intr.resize(p.intr.size() + kModels[model].n_intr); }
            p.image_group[i] = inserted.first->second;
        }
        p.poses.resize(images.size() * 6); p.points.resize(points.size() * 3); p.fixed_points.assign(points.size(), 0); p.point_kinds.assign(points.size(), 0);
        p.obs_image.reserve(size_t(observations)); p.obs_point.reserve(size_t(observations)); p.obs_xy.reserve(size_t(observations * 2)); p.obs_ranges.push_back(0);
        for (size_t point = 0; point < points.size(); ++point) {
            const uint64_t root = points[point]; bool outside = false, mutableSupport = false; uint32_t lastImage = UINT32_MAX, distinct = 0; std::vector<RegionalWorkingStore::PointKey> checked;
            const auto landmark = scene.landmark(root); validateLandmark(landmark); p.point_kinds[point] = uint8_t(landmark.kind);
            index.visitPoint(root, [&](const auto& row) {
                if (std::find(checked.begin(), checked.end(), row.point) == checked.end()) { if (scene.canonical(row.point) != root) throw std::runtime_error("canonical bundle raw alias differs"); checked.push_back(row.point); }
                const uint32_t image = local[row.image]; if (image == UINT32_MAX) { outside = true; return; }
                mutableSupport = mutableSupport || !p.fixed_frames.at(image);
                if (image != lastImage) { ++distinct; lastImage = image; }
                const double scale = cameraPixelScales.at(p.image_group.at(image));
                p.obs_image.push_back(image); p.obs_point.push_back(uint32_t(point)); p.obs_xy.push_back(row.xy.x / scale); p.obs_xy.push_back(row.xy.y / scale);
            });
            p.fixed_points[point] = landmark.kind == LandmarkKind::UnitDirection || outside || !mutableSupport || distinct < 2 || index.pointOwner(root) != owner; p.obs_ranges.push_back(uint32_t(p.obs_image.size()));
        }
        if (p.obs_image.size() != observations || storage::addBytes(p.obs_image.size(), retained_observations) != authority_observations)
            throw std::runtime_error("canonical bundle geometric/retained authority coverage differs");
        priors = PosePriors{};
        if (options.priors) {
            const auto& in = *options.priors; priors.up_w = in.up_w; priors.huber = in.huber; priors.gps = in.gps; priors.level = in.level;
            auto has = [&](uint32_t image) { return image < local.size() && local[image] != UINT32_MAX; };
            for (auto row : in.rotations) if (has(row.i) && has(row.j)) { row.i = local[row.i]; row.j = local[row.j]; priors.rotations.push_back(row); }
            for (auto row : in.ups) if (has(row.i)) { row.i = local[row.i]; priors.ups.push_back(row); }
            for (auto row : in.centres) { bool complete = row.n > 0 && row.n <= 3; for (int i = 0; i < row.n; ++i) complete = complete && has(row.img[i]); if (complete) { for (int i = 0; i < row.n; ++i) row.img[i] = local[row.img[i]]; priors.centres.push_back(row); } }
        }
        p.total_intr = uint32_t(p.intr.size()); topologyIdentity = index.identity(); landmarkKindsIdentity = scene.landmarkKindsHash(); reload(scene); finalizeTables(p);
    }
    void reload(const CanonicalScene& scene) {
        if (scene.binding().topology != topologyIdentity) throw std::runtime_error("canonical bundle topology changed");
        if (scene.landmarkKindsHash() != landmarkKindsIdentity) throw std::runtime_error("canonical bundle landmark kinds changed; prepare the layout again");
        for (size_t i = 0; i < images.size(); ++i) bundle_detail::packPose(scene.poses().images.at(images[i]).pose, &problem.poses[6 * i]);
        for (size_t i = 0; i < cameras.size(); ++i) {
            Camera camera = scene.poses().cameras.at(cameras[i]); const double scale = cameraPixelScales.at(i);
            if (camera.pixel_scale != scale || uint32_t(camBaModel(camera.model)) != problem.groups[i].model) throw std::runtime_error("canonical bundle camera measurement layout changed");
            camera.fx /= scale; camera.fy /= scale; camera.cx /= scale; camera.cy /= scale;
            packIntrinsics(camera, &problem.intr[problem.groups[i].intr_offset]);
        }
        for (size_t i = 0; i < points.size(); ++i) {
            const auto landmark = scene.landmark(points[i]); validateLandmark(landmark);
            if (problem.point_kinds.at(i) != uint8_t(landmark.kind)) throw std::runtime_error("canonical bundle cached point kind differs");
            const Vec3 value = landmark.value; problem.points[3 * i] = value.x; problem.points[3 * i + 1] = value.y; problem.points[3 * i + 2] = value.z;
        }
        problem.priors = priors.empty() ? nullptr : &priors;
    }
    BundleRun solve(const BundleOptions& options, VkContext* context = nullptr) {
        if (!problem.num_obs) throw std::runtime_error("canonical bundle has no geometric observations; retained measurements cannot enter BA");
        problem.priors = priors.empty() ? nullptr : &priors; return solveBundle(problem, bundleSolverOptions(options), context);
    }
    void apply(CanonicalScene& scene) const {
        if (!problem.num_obs) throw std::runtime_error("canonical bundle has no geometric solution to apply");
        if (!scene.editing() || topologyIdentity != scene.binding().topology) throw std::runtime_error("canonical bundle apply outside matching candidate");
        if (scene.landmarkKindsHash() != landmarkKindsIdentity) throw std::runtime_error("canonical bundle apply landmark kinds differ");
        validateBAPointKinds(problem);
        for (size_t i = 0; i < points.size(); ++i) if (problem.directionalPoint(uint32_t(i))) {
            const auto landmark = scene.landmark(points[i]);
            if (landmark.kind != LandmarkKind::UnitDirection || landmark.value.x != problem.points[3 * i] ||
                landmark.value.y != problem.points[3 * i + 1] || landmark.value.z != problem.points[3 * i + 2])
                throw std::runtime_error("canonical block BA changed a fixed direction");
        }
        for (size_t i = 0; i < images.size(); ++i) if (!problem.fixed_frames[i]) scene.setPose(images[i], bundle_detail::unpackPose(&problem.poses[6 * i]));
        std::vector<std::pair<uint64_t, Vec3>> updates; updates.reserve(points.size());
        for (size_t i = 0; i < points.size(); ++i) if (!problem.fixed_points[i]) updates.push_back({points[i], {problem.points[3 * i], problem.points[3 * i + 1], problem.points[3 * i + 2]}});
        scene.applyPointUpdates(updates);
    }
};
}
