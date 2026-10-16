#include "common/camera.slang"
#include "common/loss.slang"

struct QualityRow {
    double value[3]; double xy[2]; uint image; uint kind;
};
struct QualityView {
    double rotation[9]; double translation[3]; double intr[12];
    double pixelScale; uint model; uint registered;
};
struct QualityPartial {
    double cost; double finiteCost;
    uint observations; uint invalid; uint retained; uint directions; uint fallback; uint pad;
};
[[vk::binding(0)]] StructuredBuffer<QualityRow> rows;
[[vk::binding(1)]] StructuredBuffer<QualityView> views;
[[vk::binding(2)]] RWStructuredBuffer<QualityPartial> output;
[[vk::push_constant]] cbuffer PC {
    uint rowCount; uint viewCount; uint unused0; uint unused1;
    float gate; float unused2; float unused3; float unused4;
};

RVec2 projection(QualityView view, RVec3 pc) {
    if (view.model == 2) { Real p[5]; for (uint i = 0; i < 5; ++i) p[i] = view.intr[i]; return PinholeRadialModel.fromArray(p).project(pc); }
    if (view.model == 3) { Real p[8]; for (uint i = 0; i < 8; ++i) p[i] = view.intr[i]; return OpenCVModel.fromArray(p).project(pc); }
    if (view.model == 4) { Real p[3]; for (uint i = 0; i < 3; ++i) p[i] = view.intr[i]; return SimplePinholeModel.fromArray(p).project(pc); }
    if (view.model == 5) { Real p[4]; for (uint i = 0; i < 4; ++i) p[i] = view.intr[i]; return PinholeModel.fromArray(p).project(pc); }
    Real p[12]; for (uint i = 0; i < 12; ++i) p[i] = view.intr[i]; return FullOpenCVModel.fromArray(p).project(pc);
}

QualityPartial emptyPartial() {
    QualityPartial q; q.cost = q.finiteCost = 0.0;
    q.observations = q.invalid = q.retained = q.directions = q.fallback = q.pad = 0;
    return q;
}
QualityPartial evaluateRow(uint index) {
    QualityPartial q = emptyPartial(); q.observations = 1;
    QualityRow row = rows[index];
    if (row.kind == 2) { q.retained = 1; return q; }
    q.directions = row.kind == 1 ? 1 : 0;
    if (row.image >= viewCount) { q.fallback = 1; return q; }
    QualityView view = views[row.image];
    if (!view.registered) { q.fallback = 1; return q; }
    RVec3 pc = rvec3(
        view.rotation[0] * row.value[0] + view.rotation[1] * row.value[1] + view.rotation[2] * row.value[2],
        view.rotation[3] * row.value[0] + view.rotation[4] * row.value[1] + view.rotation[5] * row.value[2],
        view.rotation[6] * row.value[0] + view.rotation[7] * row.value[1] + view.rotation[8] * row.value[2]);
    if (row.kind == 0) pc = pc + rvec3(view.translation[0], view.translation[1], view.translation[2]);
    double zScale = abs(view.rotation[6] * row.value[0]) + abs(view.rotation[7] * row.value[1]) + abs(view.rotation[8] * row.value[2]);
    if (row.kind == 0) zScale += abs(view.translation[2]);
    if (!isfinite(pc.x) || !isfinite(pc.y) || !isfinite(pc.z)) { q.invalid = q.fallback = 1; return q; }
    if (pc.z < 1e-8) { q.invalid = q.fallback = 1; return q; }
    if (abs(pc.z - 1e-8) <= 1e-10 * max(1.0, zScale)) { q.fallback = 1; return q; }
    if (view.model == 7) {
        double x = pc.x / pc.z, y = pc.y / pc.z, r2 = x * x + y * y;
        double a = r2 * view.intr[7], b = r2 * r2 * view.intr[8], c = r2 * r2 * r2 * view.intr[9];
        if (abs(1.0 + a + b + c) <= 1e-10 * (1.0 + abs(a) + abs(b) + abs(c))) { q.fallback = 1; return q; }
    }
    RVec2 xy = projection(view, pc);
    double dx = (xy.x - row.xy[0]) / view.pixelScale, dy = (xy.y - row.xy[1]) / view.pixelScale;
    double squared = dx * dx + dy * dy;
    if (!isfinite(squared) || squared < 0.0) { q.invalid = q.fallback = 1; return q; }
    double cost = HuberLoss.cost(squared, gate);
    if (!isfinite(cost) || cost < 0.0) { q.fallback = 1; return q; }
    q.cost = q.finiteCost = cost; return q;
}
QualityPartial addPartials(QualityPartial a, QualityPartial b) {
    a.cost += b.cost; a.finiteCost += b.finiteCost;
    a.observations += b.observations; a.invalid += b.invalid; a.retained += b.retained;
    a.directions += b.directions; a.fallback += b.fallback; return a;
}
groupshared QualityPartial partials[256];
[shader("compute")] [numthreads(256, 1, 1)]
void quality_raw(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    QualityPartial q = emptyPartial();
    for (uint item = 0; item < 8; ++item) {
        uint index = group.x * 2048 + lane * 8 + item;
        if (index < rowCount) q = addPartials(q, evaluateRow(index));
    }
    partials[lane] = q; GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride != 0; stride /= 2) {
        if (lane < stride) partials[lane] = addPartials(partials[lane], partials[lane + stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0) output[group.x] = partials[0];
}
