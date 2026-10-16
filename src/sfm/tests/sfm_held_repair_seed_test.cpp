#include "sfm/map/Mapper.h"
#include "sfm/tests/TestMain.h"
#ifndef SS_HELD_SEED_OLD_PROBE
#include "sfm/map/HeldRepairSeed.h"
#endif
#include <cstdio>
#include <cstring>

using namespace sfm;
static void require(bool v, const char* message) { if (!v) throw std::runtime_error(message); }

struct Fixture {
    static constexpr uint32_t images = 5, points = 64;
    MatchesDatabase db;
    std::vector<FeatureSet> features;
    Reconstruction seed;
    MapperOptions options;
    Fixture(bool splitSeed = false) : features(images) {
        const uint32_t seedCount = points + uint32_t(splitSeed), count = seedCount + 8;
        const auto cam = Camera::defaultFor(7, 640, 480, 550, CamModel::Pinhole);
        seed.cameras.emplace(7, cam); seed.next_point3D_id = 20000;
        db.images.resize(images);
        const std::array<Vec3, images> centers{{{0,0,0},{1,0,0},{0,1,0},{1,1,0},{.4,.3,0}}};
        for (uint32_t id = 0; id < images; ++id) {
            db.images[id] = {"held" + std::to_string(id), count};
            auto& f = features[id]; f.width = 640; f.height = 480;
            f.keypoints.resize(count);
            Image im; im.id = id; im.camera_id = 7; im.name = db.images[id].name;
            im.pose = {mat3Identity(), centers[id] * -1.0}; im.registered = id < images - 1;
            im.points2D.resize(count); im.point3D_ids.assign(count, kInvalidPoint3D);
            for (uint32_t p = 0; p < count; ++p) {
                const Vec3 xyz{-.7 + .2 * (p % 8), -.7 + .2 * (p / 8), 6 + .1 * (p % 3)};
                const auto xy = cam.project(xyz - centers[id]);
                f.keypoints[p] = {float(xy.x), float(xy.y), 1, 0, 0};
                im.points2D[p] = {double(f.keypoints[p].x), double(f.keypoints[p].y)};
            }
            if (im.registered) seed.images.emplace(id, std::move(im));
        }
        for (uint32_t p = 0; p < seedCount; ++p) {
            const Vec3 truth{-.7 + .2 * (p % 8), -.7 + .2 * (p / 8), 6 + .1 * (p % 3)};
            const uint64_t base = p == 0 ? 0 : p < points ? 1000 + 17 * p : 10000;
            for (uint32_t part = 0; part < (p == points ? 2u : 1u); ++part) {
                const uint64_t point = base + part; Point3D value;
                value.xyz = truth + Vec3{.001,-.001,.001};
                for (uint32_t id = 0; id < images - 1; ++id) {
                    if (p == points && id / 2 != part) continue;
                    value.track.push_back({id,p}); seed.images.at(id).point3D_ids[p] = point;
                }
                seed.points3D.emplace(point, std::move(value));
            }
        }
        for (uint32_t a = 0; a < images; ++a) for (uint32_t b = a + 1; b < images; ++b) {
            TwoViewMatches pair; pair.image1 = a; pair.image2 = b;
            pair.config = int(TwoViewConfig::Uncalibrated);
            for (uint32_t p = 0; p < count; ++p) pair.matches.push_back({p,p});
            db.pairs.push_back(std::move(pair));
        }
        options.verbose = false; options.threads = 2; options.focal = 550;
        options.initial_cameras[7] = cam; options.known_focal_cameras.insert(7);
        options.ba_real = "cpu"; options.ba_real_coarse = "cpu";
        options.ba_solver = "cg"; options.ba_final_tight = false;
        options.refine_principal_point = false; options.refine_extra_params = false;
        options.min_num_pnp_inliers = 15; options.min_pnp_inlier_ratio = .25;
        options.max_reproj_error = 3; options.retri_scale = .75; options.merge_tracks = true;
        options.ba_refine_change = -1; // Exercise both refinement/track-completion rounds.
    }
};

static int test(int, char**) {
    for (bool split : {false, true}) {
        Fixture f(split); const auto original = f.seed;
        Mapper mapper(f.db, f.features, f.options, std::vector<uint32_t>(Fixture::images,7));
        Mapper::GrowStats stats; const auto grown = mapper.continueFrom(f.seed, &stats, {}, Fixture::images, true);
        require(stats.registered == 1 && grown.numRegistered() == Fixture::images,
                "held continuation did not really register its target");
        size_t retained = 0, fixed = 0, movedByFeature = 0; double maxByFeature = 0;
        for (const auto& point : original.points3D) {
            const auto found = grown.points3D.find(point.first);
            retained += found != grown.points3D.end();
            fixed += found != grown.points3D.end() &&
                std::memcmp(&found->second.xyz,&point.second.xyz,sizeof(Vec3)) == 0;
            const auto first = point.second.track.front();
            const auto assigned = grown.images.at(first.image_id).point3D_ids.at(first.point2D_idx);
            if (assigned != kInvalidPoint3D && grown.points3D.count(assigned)) {
                const double delta = (grown.points3D.at(assigned).xyz - point.second.xyz).norm();
                movedByFeature += delta > 1e-10; maxByFeature = std::max(maxByFeature,delta);
            }
        }
        std::printf("held actual Mapper split=%d registered=%u seed_points=%zu retained_IDs=%zu exact_XYZ=%zu\n",
                    int(split),stats.registered,original.points3D.size(),retained,fixed);
        std::printf("held original-feature lookup moved_XYZ=%zu max_delta=%.17g\n",movedByFeature,maxByFeature);
#ifdef SS_HELD_SEED_OLD_PROBE
        require(retained != original.points3D.size() || fixed != retained,
                "old held fixture did not expose point ID/XYZ contract loss");
#else
        HeldRepairSeedContract contract(original); contract.requireUnchanged(grown);
        require(retained == original.points3D.size() && fixed == retained,
                "held point identity or XYZ changed");
        if (split) require(grown.points3D.count(10000) && grown.points3D.count(10001),
                           "two distinct seed roots were fused");
        require(grown.points3D.size() > original.points3D.size(),"new private geometry was not triangulated");
        auto changed = grown; changed.points3D.at(original.points3D.begin()->first).xyz.x += .001;
        bool rejected = false; try { contract.requireUnchanged(changed); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected,"XYZ drift bypassed held contract");
        changed = grown; const auto point = original.points3D.begin()->first;
        const auto obs = original.points3D.begin()->second.track.front();
        changed.images.at(obs.image_id).point3D_ids.at(obs.point2D_idx) = 10001;
        rejected = false; try { contract.requireUnchanged(changed); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected,"original feature reassignment bypassed held contract");
        changed = grown; changed.points3D.at(point).track.erase(changed.points3D.at(point).track.begin());
        rejected = false; try { contract.requireUnchanged(changed); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected,"original seed observation removal bypassed held contract");
        changed = grown; changed.images.at(0).pose.t.x += .001;
        rejected = false; try { contract.requireUnchanged(changed); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected,"seed pose change bypassed held contract");
        changed = grown; changed.cameras.at(7).fx += .1;
        rejected = false; try { contract.requireUnchanged(changed); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected,"seed calibration change bypassed held contract");
        changed = grown; changed.images.at(0).name += ".JPG";
        rejected = false; try { contract.requireUnchanged(changed); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected,"physical export-name conversion bypassed held canonical metadata identity");
        auto malformed = f.seed; malformed.images.at(0).points2D.at(0).x += .5;
        rejected = false; try { mapper.continueFrom(malformed,nullptr,{},Fixture::images,true); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected,"adoption silently changed original measurement coordinates");
        const auto retry = mapper.continueFrom(f.seed,nullptr,{},Fixture::images,true);
        contract.requireUnchanged(retry);
        malformed = f.seed; malformed.rig_detached.insert(0);
        rejected = false; try { HeldRepairSeedContract unsupported(malformed); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected,"unsupported seed rig geometry was treated as fully protected");
        auto thin = f.options; thin.min_tri_angle_deg = 85;
        Mapper parallax(f.db,f.features,thin,std::vector<uint32_t>(Fixture::images,7));
        const auto kept = parallax.continueFrom(f.seed,nullptr,{},Fixture::images,true);
        contract.requireUnchanged(kept);
        auto duplicate = original;
        const auto duplicatePoint = duplicate.points3D.begin()->first;
        duplicate.points3D.at(duplicatePoint).track.push_back(duplicate.points3D.at(duplicatePoint).track.front());
        HeldRepairSeedContract duplicateContract(duplicate);
        changed = duplicate; changed.points3D.at(duplicatePoint).track.pop_back();
        rejected = false; try { duplicateContract.requireUnchanged(changed); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected,"duplicate seed raw multiplicity was reduced to presence");
        const std::vector<uint32_t> global{5,9,14,18,27};
        std::vector<std::vector<uint32_t>> originalFeatures(Fixture::images);
        auto remapped = grown; remapped.images.clear();
        const size_t fullFeatures = grown.images.at(0).points2D.size() * 2 + 1;
        for (const auto& kv : grown.images) {
            Image image = kv.second; image.id = global.at(kv.first);
            image.points2D.assign(fullFeatures,{}); image.point3D_ids.assign(fullFeatures,kInvalidPoint3D);
            for (size_t feature = 0; feature < kv.second.points2D.size(); ++feature) {
                const auto mapped = uint32_t(feature * 2 + 1); originalFeatures.at(kv.first).push_back(mapped);
                image.points2D.at(mapped) = kv.second.points2D.at(feature);
                image.point3D_ids.at(mapped) = kv.second.point3D_ids.at(feature);
            }
            remapped.images.emplace(image.id,std::move(image));
        }
        for (auto& point : remapped.points3D) for (auto& row : point.second.track) {
            row.point2D_idx = originalFeatures.at(row.image_id).at(row.point2D_idx);
            row.image_id = global.at(row.image_id);
        }
        contract.requireRemappedUnchanged(remapped,global,originalFeatures,[&](uint32_t) { return fullFeatures; });
        remapped.images.at(global.at(0)).points2D.at(1).x += .1;
        rejected = false; try {
            contract.requireRemappedUnchanged(remapped,global,originalFeatures,[&](uint32_t) { return fullFeatures; });
        } catch (const std::runtime_error&) { rejected = true; }
        require(rejected,"restored global feature coordinate drift bypassed held contract");
        auto single = original;
        for (auto& point : single.points3D) {
            for (size_t i = 1; i < point.second.track.size(); ++i) {
                const auto row = point.second.track[i];
                single.images.at(row.image_id).point3D_ids.at(row.point2D_idx) = kInvalidPoint3D;
            }
            point.second.track.resize(1);
        }
        Mapper singleMapper(f.db,f.features,f.options,std::vector<uint32_t>(Fixture::images,7));
        const auto singleGrown = singleMapper.continueFrom(single,nullptr,{},Fixture::images,true);
        HeldRepairSeedContract(single).requireUnchanged(singleGrown);
        require(singleGrown.numRegistered() == Fixture::images,
                "known fixed XYZ with one local observation could not seed actual PnP");
        if (!split) {
            Mapper ordinary(f.db,f.features,f.options,std::vector<uint32_t>(Fixture::images,7));
            const auto fresh = ordinary.continueFrom(f.seed,nullptr,{},Fixture::images,false);
            require(fresh.numRegistered() == Fixture::images && !fresh.points3D.count(1000),
                    "ordinary continuation adoption behavior changed");
            const auto first = original.points3D.begin()->second.track.front();
            const auto actual = fresh.images.at(first.image_id).point3D_ids.at(first.point2D_idx);
            require(actual != kInvalidPoint3D &&
                (fresh.points3D.at(actual).xyz-original.points3D.begin()->second.xyz).norm() > .0001,
                "ordinary continuation unexpectedly fixed unheld structure");
        }
#endif
        require(original.images.at(0).pose.R == f.seed.images.at(0).pose.R &&
                original.points3D.size() == f.seed.points3D.size(),"immutable seed input changed");
    }
    return 0;
}
int main(int argc,char** argv) { return sfmTestMain(argc,argv,test); }
