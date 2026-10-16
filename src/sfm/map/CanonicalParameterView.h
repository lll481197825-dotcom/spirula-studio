#pragma once
#include "sfm/map/CanonicalScene.h"

namespace sfm::regional {
// Fresh read-only parameters for one traversal; do not retain across scene mutations.
// The allowance includes cache allocation and sequential native-reader scratch.
class CanonicalParameterView {
public:
    using PointKey = CanonicalScene::PointKey;
    CanonicalParameterView(const CanonicalScene& scene, size_t byteBudget) : scene_(scene) {
        const uint64_t required = storage::addBytes(storage::scaleBytes(scene.pointSlots(), sizeof(Alias) + sizeof(Parameter)),
            storage::scaleBytes(scene.binding().sources.size() + 1, sizeof(uint64_t)));
        const uint64_t scratch = storage::addBytes(storage::scaleBytes(scene.pointSlots() / 64 + !!(scene.pointSlots() % 64), sizeof(uint64_t)), 3u << 16);
        if (required > byteBudget || scratch > byteBudget - required) return;
        offsets_.reserve(scene.binding().sources.size() + 1); offsets_.push_back(0);
        for (const auto& source : scene.binding().sources) offsets_.push_back(storage::addBytes(offsets_.back(), source.points));
        if (offsets_.back() != scene.pointSlots()) throw std::runtime_error("canonical parameter view source offsets differ");
        aliases_.resize(size_t(scene.pointSlots())); parameters_.resize(size_t(scene.pointSlots()));
        if (storage::addBytes(bytes(), scratch) > byteBudget) throw std::length_error("canonical parameter view allocation exceeds allowance");
        uint64_t next = 0;
        scene.visitSourceAliases([&](PointKey key, uint64_t root) {
            if (key.source >= offsets_.size() - 1 || key.ordinal >= offsets_[size_t(key.source) + 1] - offsets_[size_t(key.source)] ||
                offsets_[size_t(key.source)] + key.ordinal != next || root > next || root >= scene.pointSlots())
                throw std::runtime_error("canonical parameter view source/ordinal/root differs");
            aliases_[size_t(next++)] = {root, key.localPoint};
        });
        if (next != scene.pointSlots()) throw std::runtime_error("canonical parameter view alias coverage incomplete");
        uint64_t roots = 0;
        scene.visitCanonicalLandmarks([&](uint64_t root, CanonicalLandmark value) {
            if (root >= aliases_.size() || aliases_[size_t(root)].root != root || parameters_[size_t(root)].kind != absent())
                throw std::runtime_error("canonical parameter view root identity differs");
            validateLandmark(value); parameters_[size_t(root)] = {value.value, value.kind}; ++roots;
        });
        if (roots != scene.canonicalPoints()) throw std::runtime_error("canonical parameter view root coverage incomplete");
        for (size_t i = 0; i < aliases_.size(); ++i) {
            if (!(i & 65535)) cancel::check();
            if (aliases_[i].root == i && parameters_[i].kind == absent()) throw std::runtime_error("canonical parameter view root has no parameter");
        }
        flat_ = true;
    }
    CanonicalParameterView(const CanonicalParameterView&) = delete;
    CanonicalParameterView& operator=(const CanonicalParameterView&) = delete;
    bool flat() const { return flat_; }
    uint64_t bytes() const {
        return storage::addBytes(storage::addBytes(storage::scaleBytes(aliases_.capacity(), sizeof(Alias)),
            storage::scaleBytes(parameters_.capacity(), sizeof(Parameter))), storage::scaleBytes(offsets_.capacity(), sizeof(uint64_t)));
    }
    uint64_t canonical(PointKey key) const {
        if (!flat_) return scene_.canonical(key);
        if (key.source >= offsets_.size() - 1 || key.ordinal >= offsets_[size_t(key.source) + 1] - offsets_[size_t(key.source)])
            throw std::runtime_error("canonical parameter view source/point ordinal differs");
        const auto& alias = aliases_[size_t(offsets_[size_t(key.source)] + key.ordinal)];
        if (alias.local != key.localPoint) throw std::runtime_error("canonical parameter view original point ID differs");
        return alias.root;
    }
    CanonicalLandmark landmark(uint64_t root) const {
        if (!flat_) return scene_.landmark(root);
        if (root >= aliases_.size() || aliases_[size_t(root)].root != root || parameters_[size_t(root)].kind == absent())
            throw std::runtime_error("canonical parameter view requested a nonroot parameter");
        const auto& parameter = parameters_[size_t(root)]; return {parameter.value, parameter.kind};
    }
private:
    struct Alias { uint64_t root = 0, local = 0; };
    struct Parameter { Vec3 value{}; LandmarkKind kind = absent(); };
    static constexpr LandmarkKind absent() { return static_cast<LandmarkKind>(255); }
    const CanonicalScene& scene_;
    std::vector<Alias> aliases_;
    std::vector<Parameter> parameters_;
    std::vector<uint64_t> offsets_;
    bool flat_ = false;
};
}
