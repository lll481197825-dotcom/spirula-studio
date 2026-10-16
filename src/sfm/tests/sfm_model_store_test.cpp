#include "sfm/map/ModelStore.h"
#include "sfm/map/RegionalPrefetch.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <set>
#include <thread>

using namespace sfm;

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static Reconstruction fixture() {
    Reconstruction m;
    Camera c;
    c.id = 101; c.model = CamModel::FullOpenCV; c.width = 5280; c.height = 3956;
    int k = 0;
    for (double* p : {&c.fx, &c.fy, &c.cx, &c.cy, &c.k1, &c.k2, &c.p1, &c.p2,
                      &c.k3, &c.k4, &c.k5, &c.k6, &c.sx1, &c.sy1, &c.pixel_scale})
        *p = (++k) * 0.123456789012345;
    m.cameras.emplace(101, c);
    for (uint32_t id : {20u, 92u, 701u}) {
        Image im;
        im.id = id; im.camera_id = c.id; im.name = "capture/subfolder/photo-" + std::to_string(id);
        for (size_t j = 0; j < im.pose.R.size(); ++j) im.pose.R[j] = (j + id) * 0.019876543210987;
        im.pose.t = {1.23456789012345, -3.56789012345678, 99.23456789012345};
        im.registered = id != 701; im.exif_orientation = 6;
        im.points2D = {{12.25, 29.125}, {-1.5, 399.01}};
        im.point3D_ids = {id == 701 ? kInvalidPoint3D : 987654321012345ull, kInvalidPoint3D};
        im.points2D.reserve(7); im.point3D_ids.reserve(8);
        m.images.emplace(id, std::move(im));
    }
    Point3D p;
    p.xyz = {0.123456789012345, 2.34567890123456, -44.5678901234567};
    p.rgb[0] = 7; p.rgb[1] = 201; p.rgb[2] = 253; p.error = 1.6170123456789;
    p.track = {{20, 0}, {92, 0}};
    p.track.reserve(11);
    m.points3D.emplace(987654321012345ull, p);
    m.next_point3D_id = 987654321012399ull;
    m.rigs.resize(3);
    RigCalib& rig = m.rigs[1];
    rig.ref = 1; rig.resize(2); rig.user_scale = 17.2345678901234;
    rig.cam_from_rig[0] = m.images.at(20).pose;
    rig.cam_from_rig[1] = m.images.at(92).pose;
    rig.established = {1, 1}; rig.fixed = {0, 1}; rig.support = {23, 24};
    rig.spread_deg = {0.012345678901234, 0.987654321098765}; rig.declined_at = {123, 456};
    m.rig_detached = {20, 701};
    return m;
}

static std::vector<char> bytes(const std::filesystem::path& p) {
    std::ifstream file(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

static void testRoundtrip() {
    ModelStore store("");
    const Reconstruction m = fixture();
    const auto meta = store.put(m);
    check(meta.registered_images == std::vector<uint32_t>({20, 92}), "wrong registered-image metadata");
    check(meta.resident_bytes == modelResidentBytes(m), "wrong resident-byte metadata");
    const Reconstruction loaded = store.load(meta, meta.resident_bytes);
    check(loaded.images.size() == 3 && !loaded.images.at(701).registered, "unregistered image lost");
    check(loaded.images.at(20).exif_orientation == 6, "EXIF orientation lost");
    check(loaded.cameras.at(101).pixel_scale == m.cameras.at(101).pixel_scale, "pixel scale lost");
    check(loaded.rigs.size() == 3 && loaded.rig_detached == m.rig_detached, "rig state lost");
    check(loaded.next_point3D_id == m.next_point3D_id, "next point id changed");
    check(modelResidentBytes(loaded) <= meta.resident_bytes, "reload exceeds saved memory estimate");
    const auto views = model_store_detail::readModel(store.path(meta), false);
    check(views.points3D.empty() && views.images.size() == loaded.images.size(), "camera scan loaded geometry or lost images");
    for (const auto& kv : loaded.images) {
        const auto& view = views.images.at(kv.first);
        check(view.id == kv.second.id && view.camera_id == kv.second.camera_id && view.name == kv.second.name &&
            view.registered == kv.second.registered && view.exif_orientation == kv.second.exif_orientation &&
            view.pose.R == kv.second.pose.R && (view.pose.t - kv.second.pose.t).norm() == 0 &&
            view.points2D.empty() && view.point3D_ids.empty(), "camera-only scan changed metadata or retained feature arrays");
    }
    check(views.cameras.at(101).pixel_scale == loaded.cameras.at(101).pixel_scale &&
        views.cameras.at(101).fx == loaded.cameras.at(101).fx, "camera-only scan changed calibration");
    const auto again = store.put(loaded);
    check(bytes(store.path(meta)) == bytes(store.path(again)), "model fields changed during disk roundtrip");
    bool rejected = false;
    try { store.load(meta, meta.resident_bytes - 1); } catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "over-budget model loaded");
    const auto empty = store.put({});
    check(store.load(empty).images.empty(), "empty model failed");
    Reconstruction reserved = m;
    const size_t before = modelResidentBytes(reserved);
    reserved.images.at(20).points2D.reserve(10000);
    check(modelResidentBytes(reserved) > before + 9000 * sizeof(Vec2), "capacity ignored in memory estimate");
}

static void testGroups() {
    std::vector<StoredModel> models = {
        {0, 70, {1, 2, 3}}, {1, 60, {3, 4, 5}}, {2, 50, {5, 6}},
        {3, 40, {100, 101}}, {4, 40, {101, 102}},
    };
    const auto groups = storedModelGroups(models, 140);
    check(groups == std::vector<std::vector<size_t>>({{0, 1}, {2}, {3, 4}}), "overlap grouping changed");
    std::set<size_t> seen;
    for (const auto& group : groups) {
        size_t sum = 0;
        for (size_t i : group) { sum += models[i].resident_bytes; check(seen.insert(i).second, "model grouped twice"); }
        check(sum <= 140, "group exceeds resident budget");
    }
    check(seen.size() == models.size(), "grouping dropped a model");
    check(storedModelBytes(models) == 260, "wrong total memory estimate");
    bool rejected = false;
    try { storedModelGroups(models, 69); } catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "oversized atom did not fail");
    check(storedModelGroups({}, 0).empty(), "empty grouping failed");
}

static void testBufferedRecords() {
    ModelStore store("");
    const auto path = store.directory() / "records.bin";
    std::vector<uint64_t> large(400000);
    for (size_t i = 0; i < large.size(); ++i) large[i] = uint64_t(i) * 987654321;
    const std::string text((1u << 20) + 37, 'x');
    {
        model_store_detail::Writer out(path);
        out.scalar<uint32_t>(17); out.vector(std::vector<uint32_t>{1, 2, 3});
        out.string(text); out.vector(large); out.scalar<uint64_t>(29); out.finish();
    }
    {
        model_store_detail::Reader in(path);
        check(in.scalar<uint32_t>() == 17 && in.skipVector<uint32_t>() == 3, "buffered skip changed record position");
        check(in.string() == text && in.vector<uint64_t>() == large && in.scalar<uint64_t>() == 29,
              "buffer boundary changed spilled records");
        in.finish();
        const auto moved = store.directory() / "finished-records.bin";
        std::filesystem::rename(path, moved);
        std::filesystem::copy_file(moved, path);
        check(std::filesystem::remove(moved), "finished reader kept checkpoint locked during cleanup");
    }
    {
        model_store_detail::Reader in(path);
        in.scalar<uint32_t>(); in.skipVector<uint32_t>(); in.string();
        check(in.skipVector<uint64_t>() == large.size() && in.scalar<uint64_t>() == 29,
              "large buffered seek changed next record");
        in.finish();
    }
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    bool rejected = false;
    try {
        model_store_detail::Reader in(path);
        in.scalar<uint32_t>(); in.skipVector<uint32_t>(); in.string(); in.vector<uint64_t>(); in.scalar<uint64_t>();
    } catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "buffered reader accepted a truncated final record");
}

static void testReadAhead() {
    const auto main_thread = std::this_thread::get_id();
    std::atomic<int> asynchronous{0};
    regional::ModelReadAhead reader(3, 16u << 20, [](size_t) { return 1u << 20; }, [&](size_t i) {
        if (std::this_thread::get_id() != main_thread) ++asynchronous;
        auto rec = fixture(); rec.next_point3D_id = i; return rec;
    });
    for (int pass = 0; pass < 2; ++pass) for (size_t i = 0; i < 3; ++i)
        check(reader.get(i).next_point3D_id == i, "prefetch reordered a model or pass");
    check(asynchronous == 4, "prefetch did not overlap reads within memory budget");
    regional::ModelReadAhead bounded(2, 4u << 20, [](size_t) { return 1u << 20; }, [&](size_t) {
        check(std::this_thread::get_id() == main_thread, "prefetch exceeded combined model allowance");
        return fixture();
    });
    bounded.get(0); bounded.get(1);
    regional::ModelReadAhead failed(2, 16u << 20, [](size_t) { return 1u << 20; }, [&](size_t i) {
        if (i) throw std::runtime_error("prefetched model unavailable");
        return fixture();
    });
    failed.get(0); bool rejected = false;
    try { failed.get(1); } catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "prefetch hid a model read failure");

    const size_t cpu = std::max(1u, std::thread::hardware_concurrency());
    if (cpu > 1) {
        constexpr size_t allowance = 14u << 20, modelBytes = 1u << 20, buffers = 2u << 20;
        const size_t expected = std::min<size_t>(cpu - 1, (allowance - modelBytes) / (modelBytes + buffers));
        std::mutex mutex; std::condition_variable ready;
        size_t active = 0, peak = 0; bool release = false;
        regional::ModelReadAhead wide(9, allowance * 2, [modelBytes](size_t) { return modelBytes; }, [&](size_t i) {
            if (std::this_thread::get_id() != main_thread) {
                std::unique_lock<std::mutex> lock(mutex);
                peak = std::max(peak, ++active); ready.notify_all();
                ready.wait(lock, [&] { return release; }); --active;
            }
            auto rec = fixture(); rec.next_point3D_id = i; return rec;
        });
        check(wide.get(0).next_point3D_id == 0, "wide prefetch changed the first model");
        bool reached;
        {
            std::unique_lock<std::mutex> lock(mutex);
            reached = ready.wait_for(lock, std::chrono::seconds(5), [&] { return active == expected; });
            release = true; ready.notify_all();
        }
        check(reached, "prefetch retained an artificial three-reader ceiling");
        for (size_t i = 1; i < 9; ++i)
            check(wide.get(i).next_point3D_id == i, "wide prefetch reordered models");
        check(peak == expected && peak * (modelBytes + buffers) + modelBytes <= allowance,
              "wide prefetch exceeded CPU or combined model/buffer allowance");
        if (cpu >= 5) check(peak > 3, "prefetch did not use available readers beyond three");
    }
}

static void testConcurrencyAndCleanup() {
    std::filesystem::path dir;
    {
        ModelStore store("");
        dir = store.directory();
        std::vector<StoredModel> stored(12);
        std::vector<std::thread> workers;
        for (size_t i = 0; i < stored.size(); ++i)
            workers.emplace_back([&, i] { stored[i] = store.put(fixture()); });
        for (auto& worker : workers) worker.join();
        std::set<uint64_t> files;
        for (const auto& meta : stored) {
            check(files.insert(meta.file).second, "concurrent spill overwrote a model");
            check(store.load(meta).points3D.size() == 1, "concurrent spill damaged a model");
        }
        store.erase(stored[0]);
        check(!std::filesystem::exists(store.path(stored[0])), "obsolete model not removed");
    }
    check(!std::filesystem::exists(dir), "scratch directory not removed");
    bool rejected = false;
    try {
        ModelStore store("");
        dir = store.directory();
        auto meta = store.put(fixture());
        std::filesystem::resize_file(store.path(meta), 12);
        store.load(meta);
    } catch (const std::runtime_error&) { rejected = true; }
    check(rejected && !std::filesystem::exists(dir), "truncated model or failure cleanup failed");
    {
        ModelStore store("");
        const auto meta = store.put(fixture());
        std::fstream file(store.path(meta), std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(20);
        const uint64_t bad = ~uint64_t{0};
        file.write(reinterpret_cast<const char*>(&bad), sizeof bad); file.close();
        rejected = false;
        try { store.load(meta); } catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "corrupt count did not fail before allocation");
    }
}

int main() {
    try {
        testRoundtrip(); testGroups(); testBufferedRecords(); testReadAhead(); testConcurrencyAndCleanup();
        std::puts("sfm_model_store_test: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sfm_model_store_test: %s\n", e.what());
        return 1;
    }
}
