#pragma once

#include "sfm/core/DiskStorage.h"

namespace sfm::regional {

struct CanonicalDiskInputs {
    uint64_t rawRows = 0, pointSlots = 0, imageSlots = 0;
    uint64_t posesBytes = 0, snapshotMetadataBytes = 0, stepReferenceBytes = 0;
    uint64_t exportPeakBytes = 0, futureFixedBytes = 0;
    uint64_t rounds = 8, backtrackSteps = 12;
    bool compactRejectedSteps = false, aliasHardlinks = false;
    bool aliasesPrepared = false, indexPrepared = false, layoutPrepared = false;
};

struct CanonicalCohortDiskForecast {
    uint64_t aliasCold = 0, aliasRetained = 0;
    uint64_t indexCold = 0, indexRetained = 0;
    uint64_t layoutSeed = 0, roundArchives = 0, trialTemporary = 0;
    uint64_t parameterPhase = 0, exportPhase = 0, peakAdditional = 0;
};

// Prepared artifacts already occupy the volume's reported free space. Each
// appended cohort needs a fresh forecast; rounds is not a whole-run limit.
inline CanonicalCohortDiskForecast canonicalCohortDiskForecast(const CanonicalDiskInputs& in) {
    using storage::addBytes; using storage::scaleBytes;
    if (!in.snapshotMetadataBytes || !in.exportPeakBytes ||
        (in.compactRejectedSteps && in.backtrackSteps && !in.stepReferenceBytes))
        throw std::invalid_argument("canonical disk forecast requires explicit snapshot/export/reference bounds");
    const uint64_t xyz = scaleBytes(in.pointSlots, 25), aliases = scaleBytes(in.pointSlots, 16);
    const uint64_t fullSnapshots = scaleBytes(in.rounds, 2);
    const uint64_t rejectedTrials = scaleBytes(in.rounds, in.backtrackSteps);
    const uint64_t allSnapshots = addBytes(fullSnapshots, rejectedTrials);
    CanonicalCohortDiskForecast out;
    if (!in.aliasesPrepared) {
        out.aliasCold = addBytes(scaleBytes(in.rawRows, 96), scaleBytes(in.pointSlots, 8));
        out.aliasRetained = scaleBytes(in.pointSlots, 8);
    }
    if (!in.indexPrepared) {
        // Retained raw input and point/photo output coexist with sorting runs;
        // seed sorting needs three 40-byte tables even for zero-raw points.
        const auto phasePeak = addBytes(std::max({
            addBytes(scaleBytes(in.rawRows, 192), scaleBytes(in.pointSlots, 72)),
            addBytes(scaleBytes(in.rawRows, 200), scaleBytes(in.pointSlots, 92)),
            addBytes(scaleBytes(in.rawRows, 192), scaleBytes(in.pointSlots, 172))}),
            addBytes(scaleBytes(in.imageSlots, 16), 4096));
        out.indexCold = std::max(phasePeak,
            addBytes(scaleBytes(in.rawRows, 232), scaleBytes(in.pointSlots, 100)));
        out.indexRetained = addBytes(addBytes(scaleBytes(in.rawRows, 80), scaleBytes(in.pointSlots, 92)), scaleBytes(in.imageSlots, 16));
    }
    if (!in.layoutPrepared) out.layoutSeed = addBytes(addBytes(xyz, aliases), addBytes(in.posesBytes, in.snapshotMetadataBytes));
    out.roundArchives = addBytes(scaleBytes(fullSnapshots, xyz),
        scaleBytes(allSnapshots, addBytes(in.posesBytes, in.snapshotMetadataBytes)));
    out.roundArchives = addBytes(out.roundArchives, in.compactRejectedSteps
        ? scaleBytes(rejectedTrials, in.stepReferenceBytes) : scaleBytes(rejectedTrials, xyz));
    // The controller retains one full failed archive and closes the round.
    if (in.compactRejectedSteps && rejectedTrials) out.roundArchives = addBytes(out.roundArchives, xyz);
    if (!in.aliasHardlinks) out.roundArchives = addBytes(out.roundArchives, scaleBytes(allSnapshots, aliases));
    // Interpolation retains the current dense array while its replacement is
    // streamed. Snapshot/reference parts also coexist until atomic publication.
    out.trialTemporary = addBytes(scaleBytes(xyz, 2), scaleBytes(in.snapshotMetadataBytes, 2));
    if (in.compactRejectedSteps) out.trialTemporary = addBytes(out.trialTemporary, in.stepReferenceBytes);
    if (!in.aliasHardlinks) out.trialTemporary = addBytes(out.trialTemporary, aliases);
    const uint64_t retained = addBytes(addBytes(out.aliasRetained, out.indexRetained), addBytes(out.layoutSeed, out.roundArchives));
    out.parameterPhase = addBytes(retained, out.trialTemporary);
    out.exportPhase = addBytes(retained, in.exportPeakBytes);
    out.peakAdditional = addBytes(in.futureFixedBytes, std::max({out.aliasCold,
        addBytes(out.aliasRetained, out.indexCold), out.parameterPhase, out.exportPhase}));
    return out;
}

inline void requireCanonicalCohortSpace(const std::filesystem::path& directory,
    const CanonicalCohortDiskForecast& forecast, uint64_t reserve = uint64_t(2) << 30) {
    storage::requireSpace(directory, forecast.peakAdditional, reserve);
}

} // namespace sfm::regional
