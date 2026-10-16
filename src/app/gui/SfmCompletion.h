#pragma once

namespace gui {
struct SfmCompletionPolicy {
    bool warning = false;
    bool run_dense = false;
    bool defer_dense = false;
    bool unverified = false;
};
// Relative-scale SfM can support dense reconstruction; an incomplete result cannot.
inline SfmCompletionPolicy sfm_completion_policy(bool partial, bool metric, bool dense_enabled,
                                                bool verdict_known = true) {
    return {partial || !metric, dense_enabled && !partial, dense_enabled && partial,
            !verdict_known && !partial && metric};
}
}
