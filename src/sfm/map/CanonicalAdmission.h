#pragma once
#include "sfm/map/SceneQuality.h"
#include "sfm/map/CanonicalLandmarkGeometry.h"

namespace sfm::regional {
inline Quality scenePrefixQuality(const ImmutableSceneTopology& oldIndex, const CanonicalScene& scene,
                                  double gate, size_t allowance, int threads = 0, bool accepted = false,
                                  ImmutableSceneTopology::Progress progress = {}) {
    if (!scene.initialized() || oldIndex.identity() != scene.acceptedSnapshot().binding.topology ||
        oldIndex.observationCount() != scene.acceptedSnapshot().observations || !(gate > 0) || !std::isfinite(gate))
        throw std::runtime_error("canonical admission old raw snapshot identity differs");
    if (allowance < (2u << 20)) throw std::length_error("canonical prefix quality requires 2 MiB");
    progress = scene_index_detail::throttledProgress(std::move(progress));
    const auto& poses = accepted ? scene.acceptedPoses() : scene.poses();
    struct View { const Image* image = nullptr; const Camera* camera = nullptr; };
    std::vector<View> views(size_t(oldIndex.imageSlots()));
    for (const auto& item : poses.images) if (item.first < views.size()) views[item.first] = {&item.second, &poses.cameras.at(item.second.camera_id)};
    const uint64_t metadata = storage::scaleBytes(views.capacity(), sizeof(View));
    if (metadata >= allowance / 2) throw std::length_error("canonical prefix photo table exceeds allowance");
    struct PrefixValue { ImmutableSceneTopology::Row row; CanonicalLandmark landmark; };
    const size_t capacity = std::min<size_t>(65536, (allowance - size_t(metadata)) / (4 * sizeof(PrefixValue)));
    std::vector<PrefixValue> batch; batch.reserve(capacity);
    auto& pool = bacpu::Pool::get(); const int workers = threads > 0 ? std::min(threads, pool.size()) : pool.size();
    Quality result; result.finite_cost = 0; uint64_t oldRoot = UINT64_MAX, newRoot = 0; CanonicalLandmark landmark{};
    std::vector<RegionalWorkingStore::PointKey> checked;
    auto flush = [&] {
        if (batch.empty()) return;
        const int tasks = int((batch.size() + 2047) / 2048); std::vector<Quality> sums(tasks); for (auto& q : sums) q.finite_cost = 0;
        pool.runChecked(tasks, workers, [&](int task, int) {
            auto& q = sums[task]; const size_t end = std::min(batch.size(), (size_t(task) + 1) * 2048);
            for (size_t i = size_t(task) * 2048; i < end; ++i) {
                const auto& value = batch[i]; const auto& view = views.at(value.row.image);
                if (!view.image || !view.camera || !view.image->registered || !(view.camera->pixel_scale > 0))
                    throw std::runtime_error("canonical prefix raw observation has no global pose/camera");
                ++q.observations;
                if (value.landmark.kind == LandmarkKind::RetainedObservation) { ++q.retained_observations; continue; }
                q.directional_observations += value.landmark.kind == LandmarkKind::UnitDirection;
                bool valid = true; const double error = reprojLandmarkError(*view.camera, view.image->pose, value.row.xy, value.landmark, &valid) / view.camera->pixel_scale;
                if (!std::isfinite(error)) throw std::runtime_error("canonical prefix residual nonfinite");
                const double cost = error <= gate ? error * error : gate * (2 * error - gate);
                q.cost += cost; if (valid) q.finite_cost += cost; else ++q.invalid_projections;
            }
        });
        for (const auto& q : sums) accumulateQuality(result, q); batch.clear();
        if (progress) progress(accepted ? "accepted-prefix-fullraw" : "candidate-prefix-fullraw", result.observations, oldIndex.observationCount());
    };
    oldIndex.visitRaw([&](const ImmutableSceneTopology::Row& row) {
        if (row.canonical != oldRoot) {
            oldRoot = row.canonical; checked.clear();
            if (scene.acceptedCanonical(row.point) != oldRoot) throw std::runtime_error("canonical prefix accepted alias differs");
            newRoot = accepted ? oldRoot : scene.canonical(row.point);
            landmark = accepted ? scene.acceptedLandmark(oldRoot) : scene.landmark(newRoot);
        }
        if (std::find(checked.begin(), checked.end(), row.point) == checked.end()) {
            if (scene.acceptedCanonical(row.point) != oldRoot || (!accepted && scene.canonical(row.point) != newRoot))
                throw std::runtime_error("canonical admission split an accepted shared alias");
            if (checked.size() >= allowance / (8 * sizeof(RegionalWorkingStore::PointKey))) throw std::length_error("canonical prefix alias checks exceed allowance");
            checked.push_back(row.point);
        }
        batch.push_back({row, landmark}); if (batch.size() == capacity) flush();
    });
    flush();
    if (result.observations != oldIndex.observationCount() || !std::isfinite(result.cost) || !std::isfinite(result.finite_cost))
        throw std::runtime_error("canonical prefix original multiset coverage incomplete");
    return result;
}
inline bool canonicalPrefixAccepted(const Quality& before, const Quality& after) {
    return before.observations == after.observations && before.retained_observations == after.retained_observations && before.directional_observations == after.directional_observations && std::isfinite(after.cost) && std::isfinite(after.finite_cost) &&
        after.invalid_projections <= before.invalid_projections && after.finite_cost <= before.finite_cost * (1 + 1e-8) + 1e-8 &&
        after.cost <= before.cost * (1 + 1e-8) + 1e-8;
}

struct CanonicalAdmissionSeedEvidence {
    Quality oldBefore, oldAfter, newBefore, newAfter;
    uint64_t oldRoots = 0, oldPreservedRoots = 0, oldRefinedRoots = 0, newRoots = 0, incomingOldRootRaw = 0, rejectedOldRoot = UINT64_MAX;
    uint64_t rejectedCandidateRoot = UINT64_MAX;
    RegionalWorkingStore::PointKey rejectedPoint;
    std::string rejection;
    std::vector<std::pair<uint64_t, Vec3>> newDirections;
    std::vector<uint64_t> newRetainedObservations;
    CanonicalAdmissionSeedEvidence() { oldBefore.finite_cost = oldAfter.finite_cost = newBefore.finite_cost = newAfter.finite_cost = 0; }
};
struct CanonicalAdmissionSeed {
    CanonicalScene::PointProducer producer;
    std::shared_ptr<CanonicalAdmissionSeedEvidence> evidence;
};

// Old accepted aliases retain their IDs. New observations are evaluated by the
// admission transaction, never used to reseed old geometry implicitly.
inline CanonicalAdmissionSeed prepareCanonicalAdmissionSeed(const ImmutableSceneTopology& oldIndex,
    const ImmutableSceneTopology& nextIndex, const CanonicalScene& scene, const Reconstruction& startingPoses,
    double gate, size_t allowance, int threads = 0, ImmutableSceneTopology::Progress progress = {},
    std::shared_ptr<CanonicalAdmissionSeedEvidence> evidence = {}, double minTriAngleDegrees = 0) {
    if (!scene.initialized() || scene.editing() || oldIndex.identity() != scene.acceptedSnapshot().binding.topology ||
        oldIndex.pointSlots() != scene.acceptedSnapshot().pointSlots || oldIndex.observationCount() != scene.acceptedSnapshot().observations ||
        nextIndex.pointSlots() < oldIndex.pointSlots() || nextIndex.observationCount() < oldIndex.observationCount() ||
        !(gate > 0) || !std::isfinite(gate) || !std::isfinite(minTriAngleDegrees) || minTriAngleDegrees < 0 || minTriAngleDegrees >= 180 || allowance < (16u << 20))
        throw std::invalid_argument("canonical admission seed requires a complete accepted prefix and bounded full support");
    progress = scene_index_detail::throttledProgress(std::move(progress));
    if (!evidence) evidence = std::make_shared<CanonicalAdmissionSeedEvidence>();
    *evidence = CanonicalAdmissionSeedEvidence{};
    const auto accepted = scene.acceptedSnapshot(); uint64_t oldKeys = 0;
    nextIndex.emitAliases([&](auto key, uint64_t candidateRoot) {
        if (key.source >= accepted.binding.sources.size()) return;
        ++oldKeys; const auto originalRoot = scene.acceptedCanonical(key);
        if (candidateRoot != originalRoot) {
            evidence->rejectedOldRoot = originalRoot; evidence->rejectedCandidateRoot = candidateRoot; evidence->rejectedPoint = key;
            evidence->rejection = "uncertified_old_canonical_rewrite: source=" + std::to_string(key.source) +
                ",ordinal=" + std::to_string(key.ordinal) + ",local=" + std::to_string(key.localPoint) +
                ",accepted=" + std::to_string(originalRoot) + ",candidate=" + std::to_string(candidateRoot);
            throw std::runtime_error(evidence->rejection);
        }
    });
    if (oldKeys != accepted.pointSlots) throw std::runtime_error("canonical admission seed omitted accepted alias keys");
    if (storage::scaleBytes(oldIndex.imageSlots(), sizeof(uint8_t)) > allowance / 16) throw std::length_error("admission changed-view mask exceeds allowance");
    std::vector<uint8_t> changed(size_t(oldIndex.imageSlots()), 0);
    const auto& acceptedPoses = scene.acceptedPoses();
    auto intrinsics = [](const Camera& c) { return std::tie(c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.p1, c.p2, c.k3, c.k4, c.k5, c.k6, c.sx1, c.sy1); };
    for (const auto& item : acceptedPoses.cameras) {
        const auto found = startingPoses.cameras.find(item.first);
        if (found == startingPoses.cameras.end() || found->second.id != item.second.id || found->second.model != item.second.model ||
            found->second.width != item.second.width || found->second.height != item.second.height || found->second.pixel_scale != item.second.pixel_scale)
            throw std::runtime_error("canonical admission old camera measurement schema changed");
    }
    for (const auto& item : acceptedPoses.images) {
        const auto found = startingPoses.images.find(item.first);
        if (found == startingPoses.images.end() || found->second.name != item.second.name || found->second.camera_id != item.second.camera_id ||
            found->second.exif_orientation != item.second.exif_orientation || (item.second.registered && !found->second.registered))
            throw std::runtime_error("canonical admission seed changed old image identity/measurement/registration");
        if (item.first >= changed.size()) throw std::runtime_error("canonical admission old image outside immutable topology");
        const auto& a = item.second.pose; const auto& b = found->second.pose;
        changed[item.first] = a.R != b.R || a.t.x != b.t.x || a.t.y != b.t.y || a.t.z != b.t.z ||
            intrinsics(acceptedPoses.cameras.at(item.second.camera_id)) != intrinsics(startingPoses.cameras.at(found->second.camera_id));
    }
    for (const auto& item : startingPoses.images) step::validatePose(item.second.pose);
    for (const auto& item : startingPoses.cameras) step::validateCamera(item.second);
    CanonicalAdmissionSeed out; out.evidence = evidence;
    out.producer = [&oldIndex, &nextIndex, &scene, &startingPoses, gate, allowance, threads, progress, evidence, accepted, minTriAngleDegrees, changed = std::move(changed)](const auto& emit) {
        if (scene.acceptedSnapshot().identity != accepted.identity) throw std::runtime_error("canonical admission seed accepted head changed");
        *evidence = CanonicalAdmissionSeedEvidence{};
        struct Job { uint64_t root = 0; CanonicalLandmark before, after; bool newDirection = false, newRetained = false; bool old = false, affected = false; std::vector<Observation> track; Quality beforeQuality, afterQuality; };
        auto quality = [&](CanonicalLandmark landmark, const std::vector<Observation>& track) {
            Quality q; q.finite_cost = 0;
            for (const auto& row : track) {
                const auto& image = startingPoses.images.at(uint32_t(row.point >> 32)); const auto& camera = startingPoses.cameras.at(image.camera_id);
                if (!image.registered) throw std::runtime_error("admission seed support image is unregistered");
                ++q.observations;
                if (landmark.kind == LandmarkKind::RetainedObservation) { ++q.retained_observations; continue; }
                q.directional_observations += landmark.kind == LandmarkKind::UnitDirection;
                bool valid = true; const auto error = reprojLandmarkError(camera, image.pose, row.xy, landmark, &valid) / camera.pixel_scale;
                if (!std::isfinite(error)) throw std::runtime_error("admission seed support residual nonfinite");
                const auto cost = error <= gate ? error * error : gate * (2 * error - gate);
                q.invalid_projections += !valid; q.cost += cost; if (valid) q.finite_cost += cost;
            }
            return q;
        };
        std::vector<Job> jobs; uint64_t bytes = 0, done = 0, total = 0;
        nextIndex.visitSeeds([&](const auto&) { ++total; });
        auto& pool = bacpu::Pool::get(); const auto workers = threads > 0 ? std::min(threads, pool.size()) : pool.size();
        auto flush = [&] {
            if (jobs.empty()) return;
            pool.runChecked(int(jobs.size()), workers, [&](int i, int) {
                auto& job = jobs[i]; cancel::check();
                if (!job.old && isRetainedObservationTrack(job.track)) { job.before = job.after = {{}, LandmarkKind::RetainedObservation}; job.newRetained = true; }
                job.beforeQuality = quality(job.before, job.track);
                job.after = job.old && !job.affected ? job.before : refineLandmark(job.before, job.track, startingPoses, gate);
                job.afterQuality = job.old && !job.affected ? job.beforeQuality : quality(job.after, job.track);
                if (!job.old && job.after.kind == LandmarkKind::FiniteXYZ && job.afterQuality.invalid_projections && minTriAngleDegrees > 0) {
                    const auto direction = fitUnitDirection(job.track, startingPoses, gate, minTriAngleDegrees);
                    if (direction.accepted) { job.after = {direction.value, LandmarkKind::UnitDirection}; job.afterQuality = quality(job.after, job.track); job.newDirection = true; }
                }
            });
            for (const auto& job : jobs) {
                if (job.old) { ++evidence->oldRoots; if (job.affected) ++evidence->oldRefinedRoots; else ++evidence->oldPreservedRoots; accumulateQuality(evidence->oldBefore, job.beforeQuality); accumulateQuality(evidence->oldAfter, job.afterQuality); }
                else { ++evidence->newRoots; accumulateQuality(evidence->newBefore, job.beforeQuality); accumulateQuality(evidence->newAfter, job.afterQuality); }
                if (job.newRetained) {
                    if (storage::addBytes(storage::scaleBytes(evidence->newRetainedObservations.size() + 1, sizeof(uint64_t)), storage::scaleBytes(evidence->newDirections.size(), sizeof(evidence->newDirections.front()))) > allowance / 8)
                        throw std::length_error("typed admission evidence exceeds allowance");
                    evidence->newRetainedObservations.push_back(job.root);
                }
                if (job.newDirection) {
                    if (storage::addBytes(storage::scaleBytes(evidence->newDirections.size() + 1, sizeof(evidence->newDirections.front())), storage::scaleBytes(evidence->newRetainedObservations.size(), sizeof(uint64_t))) > allowance / 8)
                        throw std::length_error("direction admission evidence exceeds allowance");
                    evidence->newDirections.push_back({job.root, job.after.value});
                }
                emit(job.root, job.after.value); ++done;
            }
            jobs.clear(); bytes = 0; if (progress) progress("admission-seed-fullsupport", done, total);
        };
        nextIndex.visitSeeds([&](const auto& seed) {
            cancel::check(); const bool old = seed.root < accepted.pointSlots;
            const auto& supportIndex = old ? oldIndex : nextIndex; const auto range = supportIndex.pointRange(seed.root);
            const auto need = storage::addBytes(sizeof(Job), storage::scaleBytes(range.count, sizeof(Observation) + sizeof(RegionalWorkingStore::PointKey) + sizeof(PointView) + sizeof(Vec3) + 64));
            if (need > allowance / 2) throw std::length_error("admission seed fullsupport track exceeds allowance");
            if (bytes + need > allowance / 2 || jobs.size() >= 4096) flush();
            Job job; job.root = seed.root; job.old = old; job.before = old ? scene.acceptedLandmark(seed.root) : CanonicalLandmark{seed.xyz, LandmarkKind::FiniteXYZ}; job.after = job.before;
            job.track.reserve(size_t(range.count)); std::vector<RegionalWorkingStore::PointKey> checked;
            supportIndex.visitPoint(seed.root, [&](const auto& row) {
                if (std::find(checked.begin(), checked.end(), row.point) == checked.end()) {
                    if (old && scene.acceptedCanonical(row.point) != seed.root) throw std::runtime_error("admission old support alias differs");
                    if (!old && row.point.source < accepted.binding.sources.size()) throw std::runtime_error("admission new root contains old immutable support");
                    checked.push_back(row.point);
                }
                job.track.push_back({seed.root, (uint64_t(row.image) << 32) | row.feature, row.xy});
                if (old) job.affected = job.affected || changed.at(row.image);
            });
            if (old) {
                const auto nextRange = nextIndex.pointRange(seed.root);
                if (nextRange.count < range.count) throw std::runtime_error("admission new layout lost old root raw support");
                evidence->incomingOldRootRaw = storage::addBytes(evidence->incomingOldRootRaw, nextRange.count - range.count);
            }
            jobs.push_back(std::move(job)); bytes += need;
        });
        flush();
        if (evidence->oldRoots != accepted.canonicalPoints || evidence->oldBefore.observations != oldIndex.observationCount() ||
            storage::addBytes(storage::addBytes(evidence->oldBefore.observations, evidence->newBefore.observations), evidence->incomingOldRootRaw) != nextIndex.observationCount())
            throw std::runtime_error("canonical admission seed complete raw accounting differs");
        if (!canonicalPrefixAccepted(evidence->oldBefore, evidence->oldAfter)) {
            evidence->rejection = "admission_old_support_retriangulation_stage_regressed"; throw std::runtime_error(evidence->rejection);
        }
    };
    return out;
}
}
