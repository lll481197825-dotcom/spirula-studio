#pragma once

#include "sfm/map/RegionalWorkingStore.h"
#include <memory>

namespace sfm::regional {
enum class LandmarkKind : uint8_t { FiniteXYZ = 0, UnitDirection = 1, RetainedObservation = 2 };
struct CanonicalLandmark { Vec3 value; LandmarkKind kind = LandmarkKind::FiniteXYZ; };
void validateLandmark(const CanonicalLandmark& landmark);
CanonicalLandmark interpolateLandmark(const CanonicalLandmark& before, const CanonicalLandmark& after, double alpha);
class CanonicalScene {
public:
    using Path = std::filesystem::path;
    using PointKey = RegionalWorkingStore::PointKey;
    using Landmark = CanonicalLandmark;
    struct SourceIdentity {
        std::string label, digest;
        uint64_t points = 0, observations = 0;
        bool operator==(const SourceIdentity& b) const;
    };
    struct Binding {
        std::string rawCatalog, topology;
        std::vector<SourceIdentity> sources;
        bool operator==(const Binding& b) const;
    };
    struct Snapshot {
        Path directory;
        uint64_t generation = 0, pointSlots = 0, canonicalPoints = 0, observations = 0;
        std::string identity, phase, aliasHash, pointHash, poseHash, kindHash;
        uint32_t schema = 3;
        Binding binding;
    };
    struct Counters { uint64_t aliasBytes = 0, pointReadBytes = 0, pointWrittenBytes = 0, fileOpens = 0, pointCopiedBytes = 0; };
    using AliasEmit = std::function<void(PointKey, uint64_t)>;
    using AliasProducer = std::function<void(const AliasEmit&)>;
    using PointEmit = std::function<void(uint64_t, Vec3)>;
    using PointProducer = std::function<void(const PointEmit&)>;
    using LandmarkEmit = std::function<void(uint64_t, CanonicalLandmark)>;

    enum class OpenMode { ReadWrite, ReadOnly };
    CanonicalScene(const Path& root, size_t budget, OpenMode mode = OpenMode::ReadWrite);
    ~CanonicalScene();
    CanonicalScene(const CanonicalScene&) = delete;
    CanonicalScene& operator=(const CanonicalScene&) = delete;
    static Binding bindingFor(const RegionalWorkingStore& catalog, const std::string& topologyIdentity, bool verifyImmutable = true, size_t maxSources = SIZE_MAX);
    bool initialized() const;
    bool editing() const;
    const Reconstruction& poses() const;
    const Reconstruction& acceptedPoses() const;
    const Binding& binding() const;
    const Snapshot& acceptedSnapshot() const;
    Path forkAccepted(const Path& destination) const;
    Snapshot inspectRetainedSnapshot(const Path& directory) const;
    uint64_t generation() const;
    const std::string& identity() const;
    size_t sourceCount() const;
    uint64_t pointSlots() const;
    uint64_t canonicalPoints() const;
    uint64_t observations() const;
    uint64_t canonical(PointKey key) const;
    uint64_t canonicalGlobal(uint64_t sourceGlobalOrdinal) const;
    bool isCanonical(uint64_t point) const;
    CanonicalLandmark landmark(uint64_t canonicalPoint) const;
    CanonicalLandmark acceptedLandmark(uint64_t canonicalPoint) const;
    std::string landmarkKindsHash() const;
    Vec3 pointXYZ(uint64_t canonicalPoint) const;
    Vec3 xyz(PointKey key) const;
    uint64_t acceptedCanonical(PointKey key) const;
    Vec3 acceptedPointXYZ(uint64_t canonicalPoint) const;
    void visitCanonicalPoints(const PointEmit& visit) const;
    void visitSourceAliases(const AliasEmit& visit) const;
    void visitCanonicalLandmarks(const LandmarkEmit& visit) const;
    void requireBinding(const Binding& expected) const;
    Reconstruction materialize(const RegionalWorkingStore& catalog, size_t source) const;

    void initialize(const Binding& binding, const Reconstruction& poses, const AliasProducer& aliases,
                    const PointProducer& canonicalSeed, const std::string& phase = "initial");
    void initializeValidated(const Binding& binding, const Reconstruction& poses, const AliasProducer& aliases,
                             const PointProducer& canonicalSeed, const std::function<void(const CanonicalScene&)>& validator,
                             const std::string& phase = "initial-validated");
    void beginCandidate();
    void beginInterpolatedCandidate(const Snapshot& proposal, double alpha);
    void beginLayoutCandidate(const Binding& binding, const Reconstruction& poses,
                              const AliasProducer& aliases, const PointProducer& canonicalSeed);
    void setPose(uint32_t image, const Pose& pose);
    void setCamera(const Camera& camera);
    void applyPointUpdates(const std::vector<std::pair<uint64_t, Vec3>>& updates);
    void applyLandmarkUpdates(const std::vector<std::pair<uint64_t, CanonicalLandmark>>& updates);
    void applyLandmarkTypes(const std::vector<std::pair<uint64_t, Vec3>>& newDirections);
    void applyRetainedObservations(const std::vector<uint64_t>& newRoots);
    void rewriteCanonicalPoints(const PointProducer& producer);
    Snapshot sealCandidate(const std::string& phase = "proposal");
    Snapshot commit(const std::string& phase);
    Path abort(const std::string& reason = "rejected");
    Path preserveInterpolatedRejection(const Snapshot& proposal, double alpha, const std::string& reason);
    Snapshot verifyRejectedStep(const Path& directory) const;
    void visitRejectedStepPoints(const Path& directory, const PointEmit& visit) const;
    void visitRejectedStepLandmarks(const Path& directory, const LandmarkEmit& visit) const;
    void interpolateCandidate(const Snapshot& proposal, double alpha);
    uint64_t residentBytes() const;
    const Counters& counters() const;
    void resetCounters();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
