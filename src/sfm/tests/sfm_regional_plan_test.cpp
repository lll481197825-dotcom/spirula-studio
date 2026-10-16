#include "sfm/map/RegionalPlan.h"
#include "sfm/Pipeline.h"
#include "sfm/tests/TestMain.h"
#include <chrono>
#include <iomanip>
#include <sstream>

using namespace sfm;
namespace fs = std::filesystem;
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
static void coverage(const std::vector<regional::Region>& plan, size_t count) {
    std::vector<uint32_t> seen(count);
    for (const auto& block : plan) {
        require(!block.core.empty(), "empty spatial core");
        require(std::is_sorted(block.images.begin(), block.images.end()) &&
            std::adjacent_find(block.images.begin(), block.images.end()) == block.images.end(), "duplicate halo image");
        for (uint32_t id : block.core) {
            require(id < count && !seen[id]++, "invalid or duplicate core owner");
            require(std::binary_search(block.images.begin(), block.images.end(), id), "core absent from work set");
        }
    }
    require(std::count(seen.begin(), seen.end(), 1) == count, "missing core owner");
}
static void same(const std::vector<regional::Region>& a, const std::vector<regional::Region>& b) {
    require(a.size() == b.size(), "different spatial cluster count");
    for (size_t i = 0; i < a.size(); ++i)
        require(a[i].core == b[i].core && a[i].images == b[i].images, "spatial plan not deterministic");
}
static std::string csv(const std::string& value) {
    std::string out = "\"";
    for (char c : value) { if (c == '"') out += '"'; out += c; }
    return out + "\"";
}
static int probe(const MatchesIndex& index, const std::vector<Vec3>& positions, size_t target, const fs::path& path) {
    const auto begin = std::chrono::steady_clock::now();
    regional::SpatialPlanStats stats;
    const auto plan = regional::planRegions(index, positions, target, &stats);
    coverage(plan, positions.size());
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    fs::create_directories(path);
    std::ofstream out(path / "owners.csv"); require(bool(out), "cannot write spatial probe owners");
    out << "image_id,region,name,gps_e,gps_n,gps_u\n" << std::setprecision(17);
    for (size_t r = 0; r < plan.size(); ++r) for (uint32_t id : plan[r].core)
        out << id << ',' << r << ',' << csv(index.images[id].name) << ',' << positions[id].x << ',' << positions[id].y << ',' << positions[id].z << '\n';
    std::ofstream report(path / "summary.txt"); require(bool(report), "cannot write spatial probe summary");
    report << std::setprecision(17) << "images=" << positions.size() << "\nclusters=" << stats.clusters
        << "\npositioned=" << stats.positioned << "\nmissing_positions=" << stats.missing_positions
        << "\nrounds=" << stats.rounds << "\nattempts=" << stats.attempts << "\ndistances=" << stats.distances
        << "\nbudget_limited=" << stats.budget_limited << "\nsmallest=" << stats.smallest << "\nlargest=" << stats.largest
        << "\ncompactness=" << stats.compactness << "\nseconds=" << seconds << "\nfull_capture_verified=false\n";
    std::printf("PASS: %zu unique owners, %zu regions, %zu positioned, %zu missing GPS, %.3f s\n",
        positions.size(), plan.size(), stats.positioned, stats.missing_positions, seconds);
    return 0;
}
static int test(int argc, char** argv) {
    if (argc == 6 && std::string(argv[1]) == "--capture") {
        MatchesIndex index; require(indexMatches(argv[2], index) && index.complete, "cannot index capture matches");
        SfmConfig cfg; cfg.sensor_pairs = true; cfg.sensor_map = true;
        auto prior = makeExifGpsPriors(cfg, argv[3], index.metadata, CameraSetup{}, false);
        require(bool(prior), "capture has no GPS prior source");
        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::vector<Vec3> positions(index.images.size(), Vec3{nan, nan, nan});
        for (uint32_t id = 0; id < positions.size(); ++id) prior->position(id, positions[id]);
        return probe(index, positions, std::stoull(argv[4]), fs::u8path(argv[5]));
    }
    if (argc == 5 && std::string(argv[1]) == "--positions") {
        std::ifstream input(fs::u8path(argv[2])); require(bool(input), "cannot read positions");
        MatchesIndex index; std::vector<Vec3> positions; std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            std::istringstream row(line); std::string x, y, z;
            require(bool(row >> x >> y >> z), "position row requires x y z");
            positions.push_back({std::stod(x), std::stod(y), std::stod(z)});
            index.images.push_back({std::to_string(index.images.size()), 0});
        }
        return probe(index, positions, std::stoull(argv[3]), fs::u8path(argv[4]));
    }
    MatchesIndex index; std::vector<Vec3> gps;
    for (uint32_t i = 0; i < 120; ++i) {
        index.images.push_back({std::to_string(i), 0});
        gps.push_back({double(i / 40) * 1000 + double(i % 8), double((i % 40) / 8), double(i % 13) * 1000});
    }
    regional::SpatialPlanStats stats;
    auto plan = regional::planRegions(index, gps, 40, &stats);
    coverage(plan, 120); require(plan.size() == 3 && stats.attempts == 3, "wrong target cluster count or attempts");
    for (const auto& r : plan) require(r.core.size() == 40, "separated GPS islands mixed");
    same(plan, regional::planRegions(index, gps, 40));
    for (auto& p : gps) p.z = -p.z;
    same(plan, regional::planRegions(index, gps, 40));
    for (auto& p : gps) { p.x += 100000; p.y -= 4000; }
    same(plan, regional::planRegions(index, gps, 40));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    gps[1] = {nan, nan, nan}; index.pairs.push_back({1, 119, 500, 0, 0});
    plan = regional::planRegions(index, gps, 40, &stats); coverage(plan, 120);
    require(stats.missing_positions == 1, "missing GPS not recorded");
    bool anchored = false;
    for (const auto& r : plan) if (std::binary_search(r.core.begin(), r.core.end(), uint32_t(1)))
        anchored = std::binary_search(r.core.begin(), r.core.end(), uint32_t(119));
    require(anchored, "missing GPS ignored verified connection");
    index.pairs.clear(); std::fill(gps.begin(), gps.end(), Vec3{nan, nan, nan});
    plan = regional::planRegions(index, gps, 40, &stats); coverage(plan, 120);
    require(stats.positioned == 0 && stats.missing_positions == 120 && stats.smallest == 40 && stats.largest == 40,
        "all missing GPS not explicitly balanced");
    std::fill(gps.begin(), gps.end(), Vec3{7, 8, 9});
    plan = regional::planRegions(index, gps, 40, &stats); coverage(plan, 120);
    require(stats.smallest == 40 && stats.largest == 40, "coincident GPS produced empty or unbalanced cores");
    bool exhausted = false;
    try { regional::SpatialPlanOptions lowBudgetOptions; lowBudgetOptions.distance_budget = 1; regional::planRegions(index, gps, 40, nullptr, lowBudgetOptions); }
    catch (const std::length_error&) { exhausted = true; }
    require(exhausted, "distance work budget ignored");
    index.pairs.push_back({0, 130, 1, 0, 0});
    bool invalid = false;
    try { regional::planRegions(index, gps, 40); } catch (const std::invalid_argument&) { invalid = true; }
    require(invalid, "invalid pair index accepted");
    std::puts("PASS: deterministic XY kmeans, unique owners, isolated islands, separate halo, missing and coincident GPS, bounded work");
    return 0;
}
int main(int argc, char** argv) { return sfmTestMain(argc, argv, test); }
