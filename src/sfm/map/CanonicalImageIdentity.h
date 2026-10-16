#pragma once
#include "sfm/map/ModelStore.h"

namespace sfm::regional {
template<class Inputs> void validateCanonicalImageIdentity(const Reconstruction& source, const Inputs& inputs) {
    if (!source.rigs.empty() || !source.rig_detached.empty()) throw std::runtime_error("canonical regional rig-frame scheduling is unverified; preserve source and refuse");
    for (const auto& item : source.images) {
        const auto input = inputs.find(item.first);
        if (input == inputs.end() || item.second.id != item.first || input->second.name.empty() ||
            (item.second.name != input->second.name && item.second.name != input->second.export_name))
            throw std::runtime_error("canonical source image ID/name differs from explicit feature/input mapping: " + std::to_string(item.first));
    }
}
template<class Inputs> void canonicalizeImageIdentity(Reconstruction& poses, const Inputs& inputs) {
    validateCanonicalImageIdentity(poses, inputs);
    for (auto& item : poses.images) item.second.name = inputs.at(item.first).name;
}
template<class Inputs> void canonicalizeSourceImageIdentity(Reconstruction& source, const Inputs& inputs,
                                                           const Reconstruction& existingSources) {
    validateCanonicalImageIdentity(source, inputs);
    for (auto& item : source.images) {
        const auto& input = inputs.at(item.first); const auto existing = existingSources.images.find(item.first);
        if (input.export_name.empty()) throw std::runtime_error("immutable source requires explicit original photo name");
        if (existing == existingSources.images.end()) item.second.name = input.export_name;
        else {
            if (existing->second.name != input.name && existing->second.name != input.export_name)
                throw std::runtime_error("immutable source photo name differs from explicit original input mapping");
            item.second.name = existing->second.name;
        }
    }
}
}
