// Reconstructing the atoms of a bottom-up run, concurrently.
//
// An atom is a few dozen images (sfm/map/Partition.h), and the incremental
// mapper over one is almost entirely host work: a two-view RANSAC per candidate
// seed, then P3P and triangulation per image, with bundle adjustments too small
// to occupy the device between them. Measured on a 5402-image capture, the atom
// phase was 721 s of a 1137 s mapping stage, on one core, one atom at a time.
//
// Atoms are independent by construction, so this runs them in parallel. What
// makes that possible without threading the mapper's state machine is to give
// each atom its *own* mapper over its own **sub-database**: the atom's images
// renumbered from zero, the verified pairs among them, and their features. Two
// consequences, both of which matter more than the parallelism:
//
//   * A Mapper over 48 images allocates 48 image records, not 5402. The shared
//     mapper's `rec_` carries a points2D / point3D_ids pair for every image in
//     the database whatever the restriction, so building a fresh mapper per
//     atom is *cheaper* than restricting one -- and resetModel(), which the
//     seed retries call, walks the whole database rather than the atom.
//   * A private object needs no locking anywhere. The only shared thing is the
//     Vulkan context each bundle adjustment runs on, and that is per worker:
//     a context is not thread-safe, and creating one costs far more than
//     reconstructing an atom, so it is created once per worker and handed to
//     every mapper that worker builds (Mapper::useBaContext).
//
// Camera ids stay global throughout. The intrinsics an atom must not fit for
// itself are bootstrapped once over the whole database and handed to every
// worker (Mapper::startingCameras), and the joint refinement that follows the
// atom phase shares intrinsics per group -- which is only meaningful if the
// group ids agree across atoms.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "sfm/core/Events.h"
#include "sfm/map/AtomScheduling.h"
#include "sfm/core/Features.h"
#include "sfm/core/Matches.h"
#include "sfm/core/Model.h"
#include "sfm/core/PriorSource.h"
#include "sfm/map/Mapper.h"
#include "sfm/vk/VkContext.h"
#include "sfm/core/Log.h"

namespace sfm {

struct AtomOptions {
    // Zero uses all hardware CPU slots subject to aggregate RAM/VRAM credits.
    int threads = 0;
    int max_threads = 0;  // optional legacy user ceiling, zero has no fixed cap
    size_t host_budget_bytes = 0, device_budget_bytes = 0, caller_owned_bytes = 0;
    // Bundle-adjustment cadence inside an atom. Far looser than the mapper's
    // global default because a forty-image solve does not fill the hardware,
    // and what it protects against is a bad registration in a model small
    // enough for the joint refinement afterwards to absorb.
    double ba_growth = 2.0;
    // Whether the atom's closing refinement is a tight one. See
    // MapperOptions::ba_final_tight: everything an atom converges is re-solved
    // jointly above it, three times over.
    //
    // Loosening the *growth* tolerance was tried alongside and is not the same
    // trade: rtol 1e-3 with patience 2 took a 379-image capture's median
    // relative rotation from 0.286 to 0.672 deg and its AUC@10 from 96.1 to
    // 92.5, to save two seconds. Iteration count is not where an atom's time is.
    bool tight_final_ba = false;
    // Seed attempts an atom may spend looking for further components inside
    // itself. An atom that fragments is usually dust, and what it leaves
    // behind is picked up by the tree's growth passes.
    int model_trials = 4;
    // Seed attempts for the atom's *primary* model, and how much of the atom
    // that model must cover to be kept. The mapper's own defaults (8 and 0.5)
    // are written for a whole capture, where failing to cover half of it is a
    // failure worth retrying from another seed. An atom is not a capture: a
    // partial one is merged and grown by the levels above, so trial
    // reconstructions spent covering it are work the schedule redoes anyway.
    // 0 keeps the mapper's own value.
    int init_trials = 8;
    double min_model_fraction = 0;
    bool verbose = true;
};

struct AtomStats {
    size_t atoms = 0;      // clusters handed in
    size_t models = 0;     // reconstructions they produced
    size_t registered = 0; // summed over models, so overlap counts twice
    size_t empty = 0;      // atoms that produced nothing
    int threads = 1;
    double secs = 0;
};

namespace detail {

// The atom's images renumbered 0..n-1, with the pairs among them.
struct SubDatabase {
    MatchesDatabase db;
    std::vector<FeatureSet> feats;
    std::vector<uint32_t> cam_ids;   // per local image, the *global* camera id
    std::vector<uint32_t> to_global; // local id -> database id
    RigTable rigs;                   // the run's rigs over the local ids
    SequenceTable seqs;              // ... and its sequences
    std::unique_ptr<RemappedPriorSource> priors;   // ... and its sensor priors
};

// `adj[i]` = indices into db.pairs of every pair image i takes part in. Built
// once for the whole database; a per-atom scan of db.pairs would be O(atoms x
// pairs), which on a capture with 300 atoms and 700k pairs is the dominant
// cost of the phase it is meant to make cheap.
inline std::vector<std::vector<uint32_t>> pairAdjacency(const MatchesDatabase& db) {
    std::vector<std::vector<uint32_t>> adj(db.images.size());
    std::vector<uint32_t> deg(db.images.size(), 0);
    for (const TwoViewMatches& p : db.pairs) {
        if (p.image1 < deg.size()) deg[p.image1]++;
        if (p.image2 < deg.size()) deg[p.image2]++;
    }
    for (size_t i = 0; i < adj.size(); i++) adj[i].reserve(deg[i]);
    for (uint32_t k = 0; k < db.pairs.size(); k++) {
        const TwoViewMatches& p = db.pairs[k];
        if (p.image1 < adj.size()) adj[p.image1].push_back(k);
        if (p.image2 < adj.size()) adj[p.image2].push_back(k);
    }
    return adj;
}

// Carve out one atom. `local` is scratch sized to the database, holding
// UINT32_MAX for images outside the atom; the caller reuses it across atoms so
// this stays O(atom), and it is left clean on return.
inline SubDatabase carveAtom(const MatchesDatabase& db, const std::vector<FeatureSet>& feats,
                             const std::vector<uint32_t>& cam_ids,
                             const std::vector<std::vector<uint32_t>>& adj,
                             const std::vector<uint32_t>& images,
                             std::vector<uint32_t>& local, const RigTable* rigs = nullptr,
                             const SequenceTable* seqs = nullptr, PriorSource* priors = nullptr) {
    SubDatabase s;
    s.to_global = images;
    std::sort(s.to_global.begin(), s.to_global.end());
    s.to_global.erase(std::unique(s.to_global.begin(), s.to_global.end()), s.to_global.end());
    for (uint32_t i = 0; i < s.to_global.size(); i++) local[s.to_global[i]] = i;

    s.db.images.resize(s.to_global.size());
    s.feats.resize(s.to_global.size());
    s.cam_ids.resize(s.to_global.size());
    for (uint32_t i = 0; i < s.to_global.size(); i++) {
        const uint32_t g = s.to_global[i];
        s.db.images[i] = db.images[g];
        // Everything the mapper reads and nothing else. Descriptors are half a
        // megabyte an image and no mapping code touches them -- the `map`
        // subcommand does not even load them -- but `auto` still holds the ones
        // matching used, and copying those per atom would cost more memory than
        // the whole reconstruction.
        const FeatureSet& f = feats[g];
        FeatureSet& sf = s.feats[i];
        sf.width = f.width;
        sf.height = f.height;
        sf.extract_width = f.extract_width;
        sf.extract_height = f.extract_height;
        sf.exif_focal = f.exif_focal;
        sf.exif_camera = f.exif_camera;
        sf.exif_orientation = f.exif_orientation;
        sf.keypoints = f.keypoints;
        sf.colors = f.colors;
        s.cam_ids[i] = cam_ids.empty() ? 1 : cam_ids[g];
    }
    // Take each pair from its lower endpoint so it is collected exactly once.
    for (uint32_t i = 0; i < s.to_global.size(); i++) {
        const uint32_t g = s.to_global[i];
        for (uint32_t k : adj[g]) {
            const TwoViewMatches& p = db.pairs[k];
            if (p.image1 != g) continue;
            if (p.image2 >= local.size() || local[p.image2] == UINT32_MAX) continue;
            s.db.pairs.push_back(p);
            s.db.pairs.back().image1 = i;
            s.db.pairs.back().image2 = local[p.image2];
        }
    }
    if (rigs) s.rigs = rigs->subset(local, s.to_global.size());
    if (seqs) s.seqs = seqs->subset(local, s.to_global.size());
    if (priors) s.priors = std::make_unique<RemappedPriorSource>(*priors, s.to_global);
    for (uint32_t g : s.to_global) local[g] = UINT32_MAX;
    return s;
}

// Renumber a reconstruction built on a sub-database back to database ids.
// Camera ids are global already and are left alone.
inline void toGlobalIds(Reconstruction& m, const std::vector<uint32_t>& to_global) {
    std::map<uint32_t, Image> images;
    for (auto& kv : m.images) {
        if (kv.first >= to_global.size()) continue;
        Image im = std::move(kv.second);
        im.id = to_global[kv.first];
        images[im.id] = std::move(im);
    }
    m.images = std::move(images);
    for (auto& kv : m.points3D)
        for (TrackElement& e : kv.second.track)
            if (e.image_id < to_global.size()) e.image_id = to_global[e.image_id];
    std::set<uint32_t> detached;
    for (uint32_t id : m.rig_detached)
        if (id < to_global.size()) detached.insert(to_global[id]);
    m.rig_detached = std::move(detached);
}

}  // namespace detail

using AtomModelSink = std::function<void(size_t, size_t, Reconstruction&&)>;

namespace detail {
inline uint64_t atomSharedInputBytes(const MatchesDatabase& db, const std::vector<FeatureSet>& feats,
    const std::vector<std::vector<uint32_t>>& adj, const MapperOptions& base,
    const std::vector<std::vector<uint32_t>>& atoms) {
    using namespace atom_scheduling; uint64_t bytes = sizeof(db) + sizeof(feats) + sizeof(adj) + sizeof(atoms);
    bytes = add(bytes, multiply(db.images.capacity(), sizeof(ImageEntry)));
    bytes = add(bytes, multiply(db.pairs.capacity(), sizeof(TwoViewMatches)));
    bytes = add(bytes, multiply(feats.capacity(), sizeof(FeatureSet)));
    bytes = add(bytes, multiply(adj.capacity() + atoms.capacity(), sizeof(std::vector<uint32_t>)));
    uint64_t features = 0, matches = 0;
    for (const auto& image : db.images) bytes = add(bytes, image.name.capacity() + 1);
    for (const auto& pair : db.pairs) { matches = add(matches, pair.matches.size()); bytes = add(bytes, multiply(pair.matches.size(), sizeof(FeatureMatch))); }
    for (const auto& f : feats) { features = add(features, f.keypoints.size()); bytes = add(bytes, multiply(f.keypoints.capacity(), sizeof(Keypoint))); bytes = add(bytes, f.colors.capacity() + f.descriptors.capacity() + f.exif_camera.capacity() + 1); }
    for (const auto& rows : adj) bytes = add(bytes, multiply(rows.capacity(), sizeof(uint32_t)));
    for (const auto& rows : atoms) bytes = add(bytes, multiply(rows.capacity(), sizeof(uint32_t)));
    // The outer mapper retains its correspondence cache and feature score tables.
    const auto graph = add(multiply(matches, 16), multiply(features + db.images.size(), 8));
    bytes = add(bytes, base.graph_cache_bytes ? std::min<uint64_t>(graph, base.graph_cache_bytes) : graph);
    return add(bytes, add(multiply(features, 16), multiply(db.images.size(), 512)));
}
inline atom_scheduling::Job forecastAtomResources(const MatchesDatabase& db, const std::vector<FeatureSet>& feats,
    const std::vector<std::vector<uint32_t>>& adj, const std::vector<uint32_t>& atom,
    const MapperOptions& mo, bool gpu, uint64_t auxiliaryBytes, std::vector<uint8_t>& membership) {
    using namespace atom_scheduling; auto ids = atom; std::sort(ids.begin(), ids.end()); ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    uint64_t features = 0, matches = 0, ownedMatches = 0, largestPair = 0, pairs = 0;
    for (const auto id : ids) { if (id >= feats.size() || id >= db.images.size()) throw std::out_of_range("atom image resource identity"); membership.at(id) = 1; features = add(features, feats[id].keypoints.size()); }
    for (const auto id : ids) for (uint32_t index : adj.at(id)) {
        const auto& pair = db.pairs.at(index); if (pair.image1 != id || !membership.at(pair.image2)) continue;
        matches = add(matches, pair.matches.size()); ownedMatches = add(ownedMatches, pair.matches.isView() ? 0 : pair.matches.size()); largestPair = std::max<uint64_t>(largestPair, pair.matches.size()); ++pairs;
    }
    for (const auto id : ids) membership[id] = 0;
    // Every live point owns at least two unique image/features in a mapper model.
    const uint64_t points = features / 2;
    const uint64_t model = add(multiply(features, 40), add(multiply(points, sizeof(Point3D) + 80), multiply(ids.size(), sizeof(Image) + sizeof(Camera) + 256)));
    const auto trials = std::max(1, mo.max_init_trials), further = std::max(1, mo.max_model_trials);
    const uint64_t components = std::min<uint64_t>(std::max(1, mo.max_num_models), add(trials, further));
    const uint64_t copies = add(std::max<uint64_t>(trials, components), 3);
    const auto models = multiply(model, copies);
    // CSR fill, seed memo/RANSAC, correspondence/PnP/triangulation temporaries.
    auto nonBundle = add(models, add(multiply(features, sizeof(Keypoint) + 3 + 256), multiply(ownedMatches, 16)));
    nonBundle = add(nonBundle, add(multiply(matches, 18), add(multiply(largestPair, 256), multiply(pairs, sizeof(TwoViewGeometry) + sizeof(TwoViewMatches) + 160))));
    nonBundle = add(nonBundle, add(multiply(db.images.size(), sizeof(uint32_t)), multiply(auxiliaryBytes, copies + 3)));
    bundle_detail::HostShape shape; shape.images = ids.size(); shape.points = points; shape.observations = features;
    shape.max_track = ids.size(); shape.image_span = ids.size(); shape.max_dof = kMaxCamDof; shape.copy_bytes = model;
    shape.pair_entries = (long double)features * (ids.size() + 1.L) / 2;
    BundleOptions options; options.solver = mo.ba_solver; options.real = RealCfg::CPU;
    uint64_t solver = bundle_detail::estimateHostBytes(shape, options, false);
    options.real = realCfgFromName(mo.ba_real);
    solver = std::max<uint64_t>(solver, bundle_detail::estimateHostBytes(shape, options, false));
    // Native BA forecasts/allocation checks remain authoritative inside the credit.
    solver = add(solver, std::max<uint64_t>(1u << 20, solver / 8));
    nonBundle = add(nonBundle, std::max<uint64_t>(1u << 20, nonBundle / 8));
    Job job; job.credit.host = add(nonBundle, solver); job.credit.device = gpu ? solver : 0;
    job.solver_host = solver; job.solver_device = job.credit.device; job.model_bytes = models; return job;
}
} // namespace detail

// The optional sink runs on atom workers with global image and camera ids.
inline std::vector<Reconstruction> reconstructAtoms(
    const MatchesDatabase& db, const std::vector<FeatureSet>& feats,
    const MapperOptions& base, const std::vector<uint32_t>& cam_ids,
    const std::map<uint32_t, Camera>& start_cams,
    const std::vector<std::vector<uint32_t>>& atoms, const AtomOptions& opt, AtomStats& st,
    const RigTable* rigs = nullptr, const SequenceTable* seqs = nullptr,
    PriorSource* priors = nullptr, const AtomModelSink& sink = {}) {
    const auto t0 = std::chrono::steady_clock::now();
    st.atoms = atoms.size();

    MapperOptions mo = base;
    mo.verbose = false;
    // An atom is not the capture: it numbers its images within itself and its
    // model is one of hundreds. The loop below reports it in database ids once
    // it is done, and a snapshot from here would show one atom as "the model".
    mo.report_progress = false;
    mo.ba_growth_ratio = std::max(1.0 + 1e-9, opt.ba_growth);
    mo.ba_final_tight = opt.tight_final_ba;
    // Every solve an atom runs is a coarse one (nothing here is the final
    // answer), so one scalar configuration covers the worker, and one context
    // with it -- which is what makes the per-worker context in the first place.
    if (!opt.tight_final_ba) mo.ba_real = mo.ba_real_coarse;
    mo.max_model_trials = opt.model_trials;
    // The focal was settled over the whole database before this ran. An atom
    // that searched again would be answering from a few dozen images the
    // question D48 exists to keep it away from, and would pay a trial
    // reconstruction per atom to do it.
    mo.focal_trials = 0;
    if (opt.init_trials > 0) mo.max_init_trials = opt.init_trials;
    if (opt.min_model_fraction > 0) mo.min_model_fraction = opt.min_model_fraction;
    mo.initial_cameras = start_cams;
    for (const auto& kv : start_cams) mo.known_focal_cameras.insert(kv.first);

    const std::vector<std::vector<uint32_t>> adj = detail::pairAdjacency(db);

    // Hardware slots are shared by outer tasks and their internal host passes.
    // Every task also holds aggregate host/device credits until its models spill.
    const unsigned hc = std::max(1u, std::thread::hardware_concurrency());
    const unsigned cpuSlots = base.threads > 0 ? std::min<unsigned>(hc, unsigned(base.threads)) : hc;
    const auto wantedReal = realCfgFromName(mo.ba_real), wantedCoarse = realCfgFromName(mo.ba_real_coarse);
    VkDeviceCaps capabilities;
    if (wantedReal != RealCfg::CPU || wantedCoarse != RealCfg::CPU) capabilities = mo.device_selector.empty() ? VkContext::probeCaps(deviceOnlyOpt(mo.device, mo.device_selector)) : cachedDeviceCaps(mo.device_selector);
    const auto chosen = pickRealForDevice(wantedReal, capabilities), chosenCoarse = pickRealForDevice(wantedCoarse, capabilities);
    const bool gpu = chosen != RealCfg::CPU || chosenCoarse != RealCfg::CPU;
    std::unique_ptr<VkContext> firstContext; VkContextOptions atomContextOptions;
    VkDeviceMemoryCapacity deviceCapacity;
    uint64_t host = opt.host_budget_bytes ? opt.host_budget_bytes : base.host_budget_bytes;
    const auto available = spirula::availableRamBytes();
    if (!host) host = mappingMemoryLimits(spirula::physicalRamBytes(), available).working_bytes;
    if (available) host = std::min<uint64_t>(host, available - std::min<uint64_t>(available / 8, 512u << 20));
    uint64_t auxiliary = 0;
    if (rigs) { auxiliary = atom_scheduling::add(auxiliary, rigs->of_image.capacity() * sizeof(RigSlot)); for (const auto& rig : rigs->rigs) { auxiliary = atom_scheduling::add(auxiliary, rig.members.capacity() * sizeof(RigMemberDef) + rig.frames.capacity() * sizeof(std::vector<uint32_t>)); for (const auto& frame : rig.frames) auxiliary = atom_scheduling::add(auxiliary, frame.capacity() * sizeof(uint32_t)); } }
    if (seqs) { auxiliary = atom_scheduling::add(auxiliary, 12ull * seqs->seq.capacity() + 4ull * seqs->length.capacity()); for (const auto& rows : seqs->members) for (const auto& member : rows) auxiliary = atom_scheduling::add(auxiliary, member.capacity() + sizeof(std::string)); }
    const auto shared = atom_scheduling::add(detail::atomSharedInputBytes(db, feats, adj, base, atoms), atom_scheduling::add(opt.caller_owned_bytes, auxiliary));
    std::vector<atom_scheduling::Job> forecasts; forecasts.reserve(atoms.size()); std::vector<uint8_t> membership(db.images.size());
    uint64_t retained = 0;
    for (const auto& atom : atoms) { forecasts.push_back(detail::forecastAtomResources(db, feats, adj, atom, mo, gpu, auxiliary, membership)); if (!sink) retained = atom_scheduling::add(retained, forecasts.back().model_bytes); }
    const auto fixedHost = atom_scheduling::add(shared, atom_scheduling::add(retained, forecasts.capacity() * sizeof(atom_scheduling::Job) + membership.capacity() + (1u << 20)));
    if (fixedHost >= host) throw std::length_error("atom shared inputs/retained outputs exceed aggregate RAM budget");
    if (gpu) {
        uint64_t largest = 0; for (const auto& job : forecasts) largest = std::max(largest, job.credit.host);
        if (atom_scheduling::add(largest, 128u << 20) > host - fixedHost)
            throw std::length_error("atom RAM credits cannot hold first GPU context and one complete task");
    }
    if (gpu) {
        firstContext = std::make_unique<VkContext>();
        atomContextOptions.needFloat64 = chosen == RealCfg::F64 || chosenCoarse == RealCfg::F64;
        atomContextOptions.needFloatAtomics = (chosen != RealCfg::CPU && chosen != RealCfg::DF64) || (chosenCoarse != RealCfg::CPU && chosenCoarse != RealCfg::DF64);
        atomContextOptions.needInt64Atomics = chosen == RealCfg::DF64 || chosenCoarse == RealCfg::DF64;
        atomContextOptions.selector = mo.device_selector; atomContextOptions.deviceIndex = mo.device;
        firstContext->init(atomContextOptions); deviceCapacity = firstContext->deviceMemoryCapacity();
        if (deviceCapacity.available_bytes <= deviceCapacity.reserve_bytes) throw std::length_error("atom GPU capacity unavailable after driver safety reserve");
    }
    atom_scheduling::Limits limits; limits.hardware = cpuSlots; limits.requested = opt.threads > 0 ? uint32_t(opt.threads) : 0;
    if (opt.max_threads > 0) limits.requested = limits.requested ? std::min(limits.requested, uint32_t(opt.max_threads)) : uint32_t(opt.max_threads);
    limits.host = host - fixedHost; limits.context_host = gpu ? 128u << 20 : 0;
    limits.device = gpu ? deviceCapacity.available_bytes - deviceCapacity.reserve_bytes : 0;
    if (gpu && opt.device_budget_bytes) limits.device = std::min<uint64_t>(limits.device, opt.device_budget_bytes);
    limits.context_device = gpu ? 128u << 20 : 0; limits.first_device_context_accounted = gpu;
    const auto scheduling = atom_scheduling::plan(forecasts, limits); const int nt = int(scheduling.workers); st.threads = nt;
    atom_scheduling::Pool resources(forecasts, scheduling.pool, cpuSlots, scheduling.workers);
    if (opt.verbose) slog::diag(slog::Tag::Map, "[bup] resource-adaptive atom workers %d/%u CPU slots (requested %d); shared/retained %.1f MiB, host pool %.1f MiB, device pool %.1f MiB, context reservations host %.1f/device %.1f MiB; driver budget %s\n", nt, cpuSlots, opt.threads, fixedHost / 1048576., scheduling.pool.host / 1048576., scheduling.pool.device / 1048576., scheduling.contexts_host / 1048576., scheduling.contexts_device / 1048576., gpu ? (deviceCapacity.budget_known ? "available" : "conservative fallback") : "native CPU");

    std::vector<std::vector<Reconstruction>> per_atom(atoms.size());
    std::vector<size_t> model_counts(atoms.size(), 0);
    std::vector<size_t> registrations(atoms.size(), 0);
    std::mutex log_mu;
    std::atomic<size_t> done{0};
    // A worker whose device fails has to report itself: an exception crossing
    // a thread boundary is std::terminate, and a terminate handler is a worse
    // way to learn about a lost device than the message the failure carries.
    std::mutex err_mu;
    std::exception_ptr first_error;

    auto worker = [&](int workerIndex) {
        try {
            // One context for every atom this worker builds. Creating a Vulkan
            // device takes longer than reconstructing an atom, and it cannot be
            // shared with another thread.
            VkContext ownedContext; VkContext& ctx = workerIndex == 0 && firstContext ? *firstContext : ownedContext;
            if (gpu && !ctx.initialized()) ctx.init(atomContextOptions);
            std::vector<uint32_t> local(db.images.size(), UINT32_MAX);
            for (;;) {
                const auto lease = resources.acquire(); const size_t i = lease.task; if (i == atoms.size()) break;
                struct Release { atom_scheduling::Pool& pool; size_t task; ~Release() { pool.release(task); } } release{resources, i};
                MapperOptions taskOptions = mo; taskOptions.threads = int(lease.threads);
                taskOptions.host_budget_bytes = size_t(forecasts[i].solver_host); taskOptions.device_budget_bytes = size_t(forecasts[i].solver_device);
                detail::SubDatabase sub =
                    detail::carveAtom(db, feats, cam_ids, adj, atoms[i], local, rigs, seqs, priors);
                Mapper m(sub.db, sub.feats, taskOptions, sub.cam_ids, rigs ? &sub.rigs : nullptr,
                         seqs ? &sub.seqs : nullptr, sub.priors.get());
                m.useBaContext(&ctx);
                uint32_t reg = 0;
                auto built = m.run(); uint64_t returnedBytes = 0;
                for (const auto& rec : built) returnedBytes = atom_scheduling::add(returnedBytes, modelResidentBytes(rec));
                if (returnedBytes > forecasts[i].model_bytes) throw std::length_error("atom returned models exceed admitted model memory forecast");
                for (Reconstruction& r : built) {
                    if (r.numRegistered() < 2) continue;
                    detail::toGlobalIds(r, sub.to_global);
                    reg += r.numRegistered();
                    for (const auto& kv : r.images)
                        if (kv.second.registered) events::map_placed(kv.first);
                    const size_t component = model_counts[i]++;
                    if (sink) {
                        sink(i, component, std::move(r));
                        r = Reconstruction{};
                    } else per_atom[i].push_back(std::move(r));
                }
                registrations[i] = reg;
                if (opt.verbose) {
                    const size_t n = ++done;
                    std::lock_guard<std::mutex> lk(log_mu);
                    slog::diag(slog::Tag::Map,
                               "[bup] atom %zu/%zu: %zu images -> %zu model(s), %u registered; CPU lease %u, returned models %.1f MiB / forecast %.1f MiB, task host/device credits %.1f/%.1f MiB",
                               n, atoms.size(), atoms[i].size(), model_counts[i], reg, lease.threads, returnedBytes / 1048576., forecasts[i].model_bytes / 1048576., forecasts[i].credit.host / 1048576., forecasts[i].credit.device / 1048576.);
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lk(err_mu);
            if (!first_error) first_error = std::current_exception();
            resources.stop();
        }
    };
    if (nt == 1) {
        worker(0);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nt);
        for (int t = 0; t < nt; t++) pool.emplace_back(worker, t);
        for (std::thread& t : pool) t.join();
    }
    if (first_error) std::rethrow_exception(first_error);
    const auto resourceStats = resources.stats();
    if (opt.verbose) slog::diag(slog::Tag::Map, "[bup] atom reservation peaks: %u active tasks, %u/%u CPU leases, host credits %.1f/%.1f MiB, device credits %.1f/%.1f MiB; %llu resource waits\n", resourceStats.peak_active, resourceStats.peak_cpu, cpuSlots, resourceStats.peak_host / 1048576., scheduling.pool.host / 1048576., resourceStats.peak_device / 1048576., scheduling.pool.device / 1048576., (unsigned long long)resourceStats.waits);

    std::vector<Reconstruction> models;
    st.models = 0;
    for (size_t i = 0; i < per_atom.size(); ++i) {
        std::vector<Reconstruction>& v = per_atom[i];
        if (!model_counts[i]) st.empty++;
        st.models += model_counts[i];
        st.registered += registrations[i];
        for (Reconstruction& r : v) {
            models.push_back(std::move(r));
        }
    }
    st.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return models;
}

}  // namespace sfm
