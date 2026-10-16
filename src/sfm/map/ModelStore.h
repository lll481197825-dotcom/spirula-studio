#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "sfm/core/Model.h"
#include "sfm/core/DiskStorage.h"
#include "sfm/core/ModelMemory.h"

namespace sfm {

namespace model_store_detail {

using model_memory_detail::addBytes;

class Writer {
public:
    explicit Writer(const std::filesystem::path& path) : file_(path, std::ios::binary), buffer_(1u << 20) {
        if (!file_) throw std::runtime_error("cannot create spilled model");
    }
    ~Writer() { if (used_) file_.write(buffer_.data(), std::streamsize(used_)); }
    template <class T> void scalar(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&v, sizeof(T));
    }
    template <class T> void vector(const std::vector<T>& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        scalar<uint64_t>(v.size());
        if (v.size() > SIZE_MAX / sizeof(T)) throw std::length_error("spilled model vector overflow");
        bytes(v.data(), v.size() * sizeof(T));
    }
    void string(const std::string& v) {
        scalar<uint64_t>(v.size());
        bytes(v.data(), v.size());
    }
    void finish() {
        flush();
        file_.flush();
        if (!file_) throw std::runtime_error("cannot flush spilled model");
        file_.close();
        if (!file_) throw std::runtime_error("cannot close spilled model");
    }
private:
    void flush() {
        if (used_) file_.write(buffer_.data(), std::streamsize(used_));
        if (!file_) throw std::runtime_error("cannot write spilled model");
        used_ = 0;
    }
    void bytes(const void* source, size_t n) {
        const auto* p = static_cast<const char*>(source);
        while (n) {
            if (!used_ && n >= buffer_.size()) {
                file_.write(p, std::streamsize(n));
                if (!file_) throw std::runtime_error("cannot write spilled model array");
                return;
            }
            const size_t count = std::min(n, buffer_.size() - used_);
            std::memcpy(buffer_.data() + used_, p, count);
            used_ += count; p += count; n -= count;
            if (used_ == buffer_.size()) flush();
        }
    }
    std::ofstream file_;
    std::vector<char> buffer_;
    size_t used_ = 0;
};

struct ModelReadIndex {
    Reconstruction views;
    std::map<uint32_t, uint64_t> images;
    std::map<uint32_t, std::vector<uint64_t>> points;
    size_t bytes = 0;
};

class Reader {
public:
    explicit Reader(const std::filesystem::path& path, size_t buffer_bytes = 1u << 20)
        : file_(path, std::ios::binary), remaining_(std::filesystem::file_size(path)), buffer_(std::max<size_t>(1, buffer_bytes)), size_(remaining_) {
        if (!file_) throw std::runtime_error("cannot open spilled model");
    }
    uint64_t position() const { return size_ - remaining_; }
    void seek(uint64_t offset) {
        if (offset > size_) throw std::runtime_error("spilled model offset out of range");
        if (end_ && offset >= buffer_position_ && offset <= buffer_position_ + end_) begin_ = size_t(offset - buffer_position_);
        else { file_.seekg(std::streamoff(offset), std::ios::beg); begin_ = end_ = 0; }
        remaining_ = size_ - offset;
        if (!file_) throw std::runtime_error("cannot seek spilled model");
    }
    template <class T> T scalar() {
        static_assert(std::is_trivially_copyable_v<T>);
        T v{};
        bytes(&v, sizeof(T));
        return v;
    }
    template <class T> void read(T* destination, size_t count) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (count > remaining_ / sizeof(T)) throw std::runtime_error("truncated spilled model array");
        bytes(destination, count * sizeof(T));
    }
    size_t count(size_t minimum_bytes) {
        const uint64_t n = scalar<uint64_t>();
        if (n > SIZE_MAX || (minimum_bytes && n > remaining_ / minimum_bytes))
            throw std::runtime_error("invalid spilled model count");
        return static_cast<size_t>(n);
    }
    template <class T> std::vector<T> vector() {
        static_assert(std::is_trivially_copyable_v<T>);
        const size_t n = count(sizeof(T));
        std::vector<T> v(n);
        bytes(v.data(), n * sizeof(T));
        return v;
    }
    template <class T> size_t skipVector() {
        const size_t n = count(sizeof(T));
        const uint64_t bytes = uint64_t(n) * sizeof(T);
        require(bytes);
        if (bytes <= end_ - begin_) begin_ += size_t(bytes);
        else {
            file_.seekg(std::streamoff(bytes - (end_ - begin_)), std::ios::cur);
            begin_ = end_ = 0;
        }
        remaining_ -= bytes;
        if (!file_) throw std::runtime_error("cannot skip spilled model vector");
        return n;
    }
    std::string string() {
        const size_t n = count(1);
        std::string v(n, '\0');
        bytes(v.data(), n);
        return v;
    }
    void finish() {
        if (remaining_) throw std::runtime_error("unexpected trailing spilled model data");
        file_.close();
        if (file_.fail()) throw std::runtime_error("cannot close spilled model reader");
    }
private:
    void require(uint64_t n) {
        if (n > remaining_) throw std::runtime_error("truncated spilled model");
    }
    void bytes(void* destination, size_t n) {
        require(n); auto* p = static_cast<char*>(destination);
        while (n) {
            if (begin_ == end_) {
                begin_ = end_ = 0;
                if (n >= buffer_.size()) {
                    file_.read(p, std::streamsize(n));
                    if (!file_) throw std::runtime_error("cannot read spilled model array");
                    remaining_ -= n; return;
                }
                buffer_position_ = position();
                end_ = size_t(std::min<uint64_t>(remaining_, buffer_.size()));
                file_.read(buffer_.data(), std::streamsize(end_));
                if (!file_) throw std::runtime_error("cannot read spilled model buffer");
            }
            const size_t count = std::min(n, end_ - begin_);
            std::memcpy(p, buffer_.data() + begin_, count);
            begin_ += count; p += count; n -= count; remaining_ -= count;
        }
    }
    std::ifstream file_;
    uint64_t remaining_ = 0;
    std::vector<char> buffer_;
    size_t begin_ = 0, end_ = 0;
    uint64_t size_ = 0, buffer_position_ = 0;
};

template<class Output> inline void writeCamera(Output& out, const Camera& c) {
    out.scalar(c.id); out.scalar(c.width); out.scalar(c.height);
    out.template scalar<int32_t>(static_cast<int32_t>(c.model));
    for (double v : {c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.p1, c.p2, c.k3, c.k4,
                     c.k5, c.k6, c.sx1, c.sy1, c.pixel_scale}) out.scalar(v);
}

inline Camera readCamera(Reader& in) {
    Camera c;
    c.id = in.scalar<uint32_t>(); c.width = in.scalar<int>(); c.height = in.scalar<int>();
    c.model = static_cast<CamModel>(in.scalar<int32_t>());
    for (double* v : {&c.fx, &c.fy, &c.cx, &c.cy, &c.k1, &c.k2, &c.p1, &c.p2,
                      &c.k3, &c.k4, &c.k5, &c.k6, &c.sx1, &c.sy1, &c.pixel_scale})
        *v = in.scalar<double>();
    return c;
}

template<class Output> inline void serializeModel(const Reconstruction& m, Output& out) {
    out.template scalar<uint64_t>(0x31504c4c49505353ull);
    out.template scalar<uint32_t>(1);
    out.scalar(m.next_point3D_id);
    out.template scalar<uint64_t>(m.cameras.size());
    for (const auto& kv : m.cameras) {
        out.scalar(kv.first); writeCamera(out, kv.second);
    }
    out.template scalar<uint64_t>(m.images.size());
    for (const auto& kv : m.images) {
        const Image& im = kv.second;
        out.scalar(kv.first); out.scalar(im.id); out.scalar(im.camera_id); out.string(im.name);
        out.scalar(im.pose); out.template scalar<uint8_t>(im.registered ? 1 : 0);
        out.scalar(im.exif_orientation); out.vector(im.points2D); out.vector(im.point3D_ids);
    }
    out.template scalar<uint64_t>(m.points3D.size());
    for (const auto& kv : m.points3D) {
        const Point3D& p = kv.second;
        out.scalar(kv.first); out.scalar(p.xyz);
        for (uint8_t v : p.rgb) out.scalar(v);
        out.scalar(p.error); out.vector(p.track);
    }
    out.template scalar<uint64_t>(m.rigs.size());
    for (const RigCalib& r : m.rigs) {
        out.scalar(r.ref); out.vector(r.cam_from_rig); out.vector(r.established);
        out.vector(r.fixed); out.vector(r.support); out.vector(r.spread_deg);
        out.vector(r.declined_at); out.scalar(r.user_scale);
    }
    out.template scalar<uint64_t>(m.rig_detached.size());
    for (uint32_t id : m.rig_detached) out.scalar(id);
}

class ByteCounter {
public:
    template<class T> void scalar(const T&) { bytes = storage::addBytes(bytes, sizeof(T)); }
    template<class T> void vector(const std::vector<T>& v) {
        scalar(uint64_t(v.size())); bytes = storage::addBytes(bytes, storage::scaleBytes(v.size(), sizeof(T)));
    }
    void string(const std::string& v) { scalar(uint64_t(v.size())); bytes = storage::addBytes(bytes, v.size()); }
    uint64_t bytes = 0;
};

inline uint64_t serializedModelBytes(const Reconstruction& m) {
    ByteCounter count; serializeModel(m, count); return count.bytes;
}

inline void writeModel(const Reconstruction& m, const std::filesystem::path& path) {
    storage::requireSpace(path.parent_path(), serializedModelBytes(m));
    Writer out(path); serializeModel(m, out); out.finish();
}

inline std::pair<uint32_t, Image> readStoredImage(Reader& in, bool geometry) {
    const auto id = in.scalar<uint32_t>(); Image im;
    im.id = in.scalar<uint32_t>(); im.camera_id = in.scalar<uint32_t>(); im.name = in.string();
    im.pose = in.scalar<Pose>(); im.registered = in.scalar<uint8_t>() != 0;
    im.exif_orientation = in.scalar<uint8_t>();
    size_t features = 0, assignments = 0;
    if (geometry) { im.points2D = in.vector<Vec2>(); im.point3D_ids = in.vector<uint64_t>(); features = im.points2D.size(); assignments = im.point3D_ids.size(); }
    else { features = in.skipVector<Vec2>(); assignments = in.skipVector<uint64_t>(); }
    if (features != assignments) throw std::runtime_error("invalid spilled image");
    return {id, std::move(im)};
}

inline std::pair<uint64_t, Point3D> readStoredPoint(Reader& in) {
    const auto id = in.scalar<uint64_t>(); Point3D p; p.xyz = in.scalar<Vec3>();
    for (uint8_t& v : p.rgb) v = in.scalar<uint8_t>();
    p.error = in.scalar<double>(); p.track = in.vector<TrackElement>();
    return {id, std::move(p)};
}

inline Reconstruction readModel(const std::filesystem::path& path, bool include_geometry = true,
                                ModelReadIndex* index = nullptr, size_t index_limit = SIZE_MAX) {
    if (index && include_geometry) throw std::invalid_argument("model index must be built without retaining full geometry");
    if (index) *index = {};
    Reader in(path);
    if (in.scalar<uint64_t>() != 0x31504c4c49505353ull || in.scalar<uint32_t>() != 1)
        throw std::runtime_error("unsupported spilled model format");
    Reconstruction m;
    m.next_point3D_id = in.scalar<uint64_t>();
    const size_t cameras = in.count(sizeof(uint32_t) * 5 + sizeof(double) * 15);
    for (size_t i = 0; i < cameras; ++i) {
        const auto id = in.scalar<uint32_t>();
        if (!m.cameras.emplace(id, readCamera(in)).second)
            throw std::runtime_error("duplicate spilled camera id");
    }
    const size_t images = in.count(sizeof(uint32_t) * 3 + sizeof(Pose) + 26);
    for (size_t i = 0; i < images; ++i) {
        const auto offset = in.position(); auto item = readStoredImage(in, include_geometry);
        if (index) {
            index->images.emplace(item.first, offset);
            index->bytes = storage::addBytes(index->bytes, sizeof(Image) + item.second.name.capacity() + 256);
            if (index->bytes > index_limit) throw std::length_error("model observation index exceeds memory budget");
        }
        if (!m.images.emplace(item.first, std::move(item.second)).second)
            throw std::runtime_error("invalid spilled image");
    }
    if (!include_geometry && !index) return m;
    const size_t points = in.count(sizeof(uint64_t) * 2 + sizeof(Vec3) + sizeof(double) + 3);
    for (size_t i = 0; i < points; ++i) {
        const auto offset = in.position(); auto item = readStoredPoint(in);
        if (index) for (const auto& e : item.second.track) {
            if (!m.images.count(e.image_id)) throw std::runtime_error("model observation index references missing image");
            auto& offsets = index->points[e.image_id];
            if (offsets.size() == offsets.capacity()) {
                const size_t capacity = std::max<size_t>(8, storage::scaleBytes(offsets.capacity(), 2));
                index->bytes = storage::addBytes(index->bytes, storage::scaleBytes(capacity - offsets.capacity(), sizeof(uint64_t)));
                if (index->bytes > index_limit) throw std::length_error("model observation index exceeds memory budget");
                offsets.reserve(capacity);
            }
            offsets.push_back(offset);
        }
        if (include_geometry && !m.points3D.emplace(item.first, std::move(item.second)).second)
            throw std::runtime_error("duplicate spilled point id");
    }
    const size_t rigs = in.count(sizeof(int) + sizeof(uint64_t) * 6 + sizeof(double));
    m.rigs.resize(rigs);
    for (RigCalib& r : m.rigs) {
        r.ref = in.scalar<int>(); r.cam_from_rig = in.vector<Pose>();
        r.established = in.vector<uint8_t>(); r.fixed = in.vector<uint8_t>();
        r.support = in.vector<uint32_t>(); r.spread_deg = in.vector<double>();
        r.declined_at = in.vector<uint32_t>(); r.user_scale = in.scalar<double>();
    }
    const size_t detached = in.count(sizeof(uint32_t));
    for (size_t i = 0; i < detached; ++i)
        if (!m.rig_detached.insert(in.scalar<uint32_t>()).second)
            throw std::runtime_error("duplicate spilled detached image id");
    in.finish();
    if (index) index->views = m;
    return m;
}

inline Reconstruction readModelSubset(const std::filesystem::path& path, const ModelReadIndex& index,
                                     const std::vector<uint32_t>& images) {
    Reconstruction result; result.cameras = index.views.cameras; result.rigs = index.views.rigs;
    result.next_point3D_id = index.views.next_point3D_id; result.rig_detached = index.views.rig_detached;
    Reader in(path, 64u << 10); std::map<uint64_t, size_t> points;
    const std::set<uint32_t> unique(images.begin(), images.end());
    for (uint32_t id : unique) {
        auto at = index.images.find(id); if (at == index.images.end() || !index.views.images.at(id).registered) continue;
        in.seek(at->second); auto image = readStoredImage(in, true);
        if (image.first != id) throw std::runtime_error("model observation index image identity changed");
        result.images.emplace(id, std::move(image.second));
        auto tracks = index.points.find(id); if (tracks != index.points.end()) for (auto offset : tracks->second) ++points[offset];
    }
    for (const auto& p : points) if (p.second >= 2) {
        in.seek(p.first); auto point = readStoredPoint(in);
        result.points3D.emplace(point.first, std::move(point.second));
    }
    return result;
}

}  // namespace model_store_detail

struct StoredModel {
    uint64_t file = 0;
    size_t resident_bytes = 0;
    std::vector<uint32_t> registered_images;
};

class ModelStore {
public:
    explicit ModelStore(const std::string& parent) {
        namespace fs = std::filesystem;
        const fs::path base = parent.empty() ? fs::temp_directory_path() : fs::path(parent);
        fs::create_directories(base);
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        static std::atomic<uint64_t> serial{0};
        for (int trial = 0; trial < 100; ++trial) {
            root_ = fs::absolute(base) / ("bup-models-" + std::to_string(tick) + "-" +
                                         std::to_string(serial++));
            if (fs::create_directory(root_)) return;
        }
        throw std::runtime_error("cannot create an exclusive model scratch directory");
    }
    ModelStore(const ModelStore&) = delete;
    ModelStore& operator=(const ModelStore&) = delete;
    ~ModelStore() {
        std::error_code ignored;
        std::filesystem::remove_all(root_, ignored);
    }
    StoredModel put(const Reconstruction& m) {
        StoredModel meta;
        meta.file = next_++;
        meta.resident_bytes = modelResidentBytes(m);
        for (const auto& kv : m.images)
            if (kv.second.registered) meta.registered_images.push_back(kv.first);
        model_store_detail::writeModel(m, path(meta));
        return meta;
    }
    Reconstruction load(const StoredModel& meta, size_t limit = SIZE_MAX) const {
        if (meta.resident_bytes > limit)
            throw std::runtime_error("spilled model exceeds the model memory budget");
        if (std::filesystem::file_size(path(meta)) > meta.resident_bytes)
            throw std::runtime_error("spilled model file exceeds its memory estimate");
        Reconstruction m = model_store_detail::readModel(path(meta));
        if (modelResidentBytes(m) > limit)
            throw std::runtime_error("loaded model exceeds the model memory budget");
        return m;
    }
    void erase(const StoredModel& meta) { std::filesystem::remove(path(meta)); }
    std::filesystem::path path(const StoredModel& meta) const {
        return root_ / (std::to_string(meta.file) + ".model");
    }
    const std::filesystem::path& directory() const { return root_; }
private:
    std::filesystem::path root_;
    std::atomic<uint64_t> next_{0};
};

inline size_t storedModelBytes(const std::vector<StoredModel>& models) {
    size_t bytes = 0;
    for (const StoredModel& m : models)
        bytes = model_store_detail::addBytes(bytes, m.resident_bytes);
    return bytes;
}

inline std::vector<std::vector<size_t>> storedModelGroups(
    const std::vector<StoredModel>& models, size_t limit) {
    std::map<uint32_t, std::vector<size_t>> owners;
    for (size_t i = 0; i < models.size(); ++i) {
        if (models[i].resident_bytes > limit)
            throw std::runtime_error("one atom model exceeds the model memory budget");
        for (uint32_t id : models[i].registered_images) owners[id].push_back(i);
    }
    std::vector<uint8_t> used(models.size(), 0);
    std::vector<std::vector<size_t>> groups;
    for (size_t done = 0; done < models.size();) {
        size_t seed = models.size();
        for (size_t i = 0; i < models.size(); ++i)
            if (!used[i] && (seed == models.size() ||
                models[i].registered_images.size() > models[seed].registered_images.size())) seed = i;
        std::vector<size_t> group;
        std::map<size_t, size_t> overlap;
        size_t bytes = 0, chosen = seed;
        for (;;) {
            group.push_back(chosen); used[chosen] = 1; ++done;
            bytes += models[chosen].resident_bytes;
            for (uint32_t id : models[chosen].registered_images)
                for (size_t peer : owners[id])
                    if (!used[peer]) ++overlap[peer];
            chosen = models.size();
            size_t best = 0;
            for (const auto& kv : overlap)
                if (!used[kv.first] && kv.second > best &&
                    models[kv.first].resident_bytes <= limit - bytes) {
                    chosen = kv.first; best = kv.second;
                }
            if (chosen == models.size()) break;
        }
        groups.push_back(std::move(group));
    }
    return groups;
}

}  // namespace sfm
