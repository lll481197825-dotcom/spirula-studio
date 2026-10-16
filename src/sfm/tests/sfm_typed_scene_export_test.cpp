#include "sfm/map/ExportVerifiedScene.h"
#include "sfm/map/SceneVisualEvidence.h"
#include "sfm/map/ImmutableSceneAliases.h"
#include "sfm/map/CanonicalRecovery.h"
#include "sfm/tests/TestMain.h"
#include <iostream>

using namespace sfm;
using namespace sfm::regional;
namespace fs = std::filesystem;
static void check(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
static void run(const fs::path& root, uint32_t finiteCount, bool directionalBridge = false, uint32_t retainedCount=0) {
    fs::create_directories(root / "features");
    Reconstruction rec; rec.cameras[0] = Camera::defaultFor(0, 800, 600, 500, CamModel::Pinhole);
    const Vec3 direction = Vec3{.1,.05,1}.normalized();
    for (uint32_t id = 0; id < 3; ++id) {
        Image image; image.id = id; image.camera_id = 0; image.registered = true;
        image.name = "photo-" + std::to_string(id) + ".jpg"; image.pose.t = {-.5 * id,0,0};
        image.points2D.resize(finiteCount + 2 + retainedCount); image.point3D_ids.assign(finiteCount + 2 + retainedCount,kInvalidPoint3D);
        image.points2D.back()={400,300};
        rec.images.emplace(id,std::move(image));
    }
    for (uint32_t id = 0; id <= finiteCount; ++id) {
        Point3D point; point.xyz = {.02 * id,.03 * (id % 4),5 + .02 * id};
        for (auto& item : rec.images) {
            auto& image = item.second;
            if (directionalBridge && id < finiteCount && item.first == 2) continue;
            const auto xy = rec.cameras.at(0).project(id == finiteCount ? direction : point.xyz + image.pose.t);
            image.points2D[id] = {double(float(xy.x)),double(float(xy.y))}; image.point3D_ids[id] = id;
            point.track.push_back({item.first,id});
        }
        if (id == finiteCount) point.track.push_back({0,id});
        rec.points3D.emplace(id,std::move(point));
    }
    for(uint32_t k=0;k<retainedCount;++k) {
        const auto id=finiteCount+1+k; Point3D point;point.xyz={.02*k,.1,4};auto& image=rec.images.at(0);
        const auto xy=rec.cameras.at(0).project(point.xyz);image.points2D[id]={double(float(xy.x)),double(float(xy.y))};image.point3D_ids[id]=id;
        point.track={{0,id},{0,id}};rec.points3D.emplace(id,std::move(point));
    }
    rec.next_point3D_id = finiteCount + 1 + retainedCount;
    const auto file = root / "source.bin"; model_store_detail::writeModel(rec,file);
    RegionalWorkingStore catalog(root / "catalog",64u << 20); catalog.importModel(file,rec,"original");
    std::vector<uint32_t> owners{0,0,0}; ImmutableSceneAliases aliases(root / "aliases",32u << 20);
    aliases.prepare(catalog,owners); ImmutableSceneTopology topology(root / "index",32u << 20);
    topology.prepare(catalog,owners,aliases.identity(),aliases.points(),[&](auto key) { return aliases.canonical(key); });
    CanonicalScene scene(root / "scene",64u << 20);
    scene.beginLayoutCandidate(CanonicalScene::bindingFor(catalog,topology.identity()),catalog.poses(),
        [&](const auto& emit) { topology.emitAliases(emit); },[&](const auto& emit) { topology.visitSeeds([&](const auto& seed) { emit(seed.root,seed.xyz); }); });
    const auto dirRoot = scene.canonical(catalog.key(0,finiteCount));
    scene.applyLandmarkTypes({{dirRoot,direction}});
    std::vector<uint64_t> retainedRoots;for(uint32_t k=0;k<retainedCount;++k)retainedRoots.push_back(scene.canonical(catalog.key(0,finiteCount+1+k)));
    scene.applyRetainedObservations(retainedRoots);scene.commit("typed-raw-fixture");
    bool refused = false; try { scene.pointXYZ(dirRoot); } catch (const std::exception&) { refused = true; }
    check(refused,"direction silently supplied a fake finite point");
    const auto quality = sceneQuality(topology,scene,3,32u << 20,2);
    const auto visual = sceneVisualEvidence(topology,scene,3,32u << 20,2);
    const auto finiteRaw = uint64_t(finiteCount) * (directionalBridge ? 2 : 3);
    check(quality.observations == finiteRaw + 4 + 2*retainedCount && quality.retained_observations==2*retainedCount && !quality.invalid_projections,"typed raw duplicate coverage/projection failed");
    for (uint32_t id = 0; id < 3; ++id) check(visual.landmarks.at(id) == (directionalBridge && id == 2 ? 0 : finiteCount),"direction inflated metric visual landmark count");
    check(visual.components == (directionalBridge ? 2 : 1),"direction-only bridge joined independent metric components");
    for (const auto& item : rec.images) {
        FeatureSet f; f.width=800; f.height=600; f.keypoints.resize(item.second.points2D.size()); f.descriptors.resize(f.keypoints.size()*f.dim);
        for (size_t k=0;k<f.keypoints.size();++k) {f.keypoints[k].x=float(item.second.points2D[k].x);f.keypoints[k].y=float(item.second.points2D[k].y);}
        writeFeatures((root/"features"/(item.second.name+".bin")).u8string(),f);
    }
    VerifiedExportOptions options; options.cache_bytes=8u<<20; options.reader_memory_bytes=64u<<20; options.accepted_raw=quality;
    for (const auto& image : rec.images) options.inputs[image.first]={(root/"inputs"/image.second.name).u8string(),image.second.name};
    auto gate = [&](const Reconstruction& actual,const Quality&,const std::map<uint32_t,PhotoQuality>&,const ExportStats&) {
        ExportFinalAcceptance proof; proof.historical_evaluated=true;
        for (const auto& image : actual.images) {
            ExportConstraintPhoto row; row.image=image.first;row.path=options.inputs.at(image.first).path;
            row.gps_error_metres=(cameraCenter(image.second.pose)-cameraCenter(rec.images.at(image.first).pose)).norm();
            row.gps_verified=row.gps_error_metres<=5; row.scale_error_fraction=0;row.scale_verified=true;
            row.attitude_error_degrees=0;row.attitude_verified=true; proof.photos.push_back(row);
        }
        return proof;
    };
    const auto result=exportVerifiedScene(catalog,scene,[&](uint32_t id) {return root/"features"/(rec.images.at(id).name+".bin");},root/"scratch",root/"formal",options,gate);
    for (const auto& failure : result.failures) std::cerr << failure.reason << '\n';
    check(result.reader_verified && result.finite_raw==finiteRaw && result.directional_raw==4 &&
        result.retained_raw==2*retainedCount && result.retained_points==retainedCount && result.retained_unique==retainedCount &&
        result.directional_points==1 && result.directional_unique==3 && result.lineage_rows==quality.observations,
        "actual typed native formal Reader lost kind, identity or duplicate raw rows");
    check(result.verified == (finiteCount>=12 && !directionalBridge),"insufficient finite landmarks or a direction-only bridge passed metric gates");
    const auto actual=verified_export_detail::strictReadBack(root/"formal",64u<<20);
    const auto directions=verified_export_detail::strictReadDirections(root/"formal",actual,16u<<20);
    check(actual.points3D.size()==finiteCount && directions.size()==1 && directions.at(dirRoot+1).track.size()==3,
        "COLMAP finite subset and typed direction sidecar were conflated");
    const auto retained=verified_export_detail::strictReadRetained(root/"formal",actual,directions,16u<<20);
    check(retained.size()==retainedCount,"retained raw was lost or promoted to formal metric geometry");
    if(retainedCount) {
        const auto changed=exportVerifiedScene(catalog,scene,[&](uint32_t id){return root/"features"/(rec.images.at(id).name+".bin");},
            root/"retained-changed-scratch",root/"retained-changed-formal",options,
            [&](const Reconstruction& model,const Quality& q,const std::map<uint32_t,PhotoQuality>& photos,const ExportStats& stats){
                auto result=gate(model,q,photos,stats);std::ofstream file(root/"retained-changed-formal/retained-observations.bin",std::ios::binary|std::ios::app);file.put('x');file.close();return result;
            });
        check(!changed.verified && !changed.reader_verified,"retained sixth file changed after Reader without invalidating acceptance");
        fs::copy(root/"formal",root/"retained-multiview",fs::copy_options::recursive);
        {
            std::fstream file(root/"retained-multiview/retained-observations.bin",std::ios::binary|std::ios::in|std::ios::out);
            const uint64_t rows=2;const uint32_t image=1,feature=finiteCount+1;
            file.seekp(32);file.write(reinterpret_cast<const char*>(&rows),8);file.seekp(48);file.write(reinterpret_cast<const char*>(&image),4);file.write(reinterpret_cast<const char*>(&feature),4);
        }
        bool rejected=false;try{verified_export_detail::strictReadRetained(root/"retained-multiview",actual,directions,16u<<20);}catch(const std::exception&){rejected=true;}
        check(rejected,"retained Reader accepted a second view as an unconstrained record");
    }
    if (finiteCount == 12 && !directionalBridge && !retainedCount) {
        fs::copy(root/"formal",root/"nonunit-copy",fs::copy_options::recursive);
        {
            std::fstream file(root/"nonunit-copy/directional-points.bin",std::ios::binary|std::ios::in|std::ios::out);
            const double wrong=2;file.seekp(32);file.write(reinterpret_cast<const char*>(&wrong),sizeof(wrong));
        }
        bool rejected=false;try {verified_export_detail::strictReadDirections(root/"nonunit-copy",actual,16u<<20);}catch(const std::exception&){rejected=true;}
        check(rejected,"actual native direction Reader accepted a nonunit replacement");
        const auto changed=exportVerifiedScene(catalog,scene,[&](uint32_t id){return root/"features"/(rec.images.at(id).name+".bin");},
            root/"changed-scratch",root/"changed-formal",options,
            [&](const Reconstruction& model,const Quality& q,const std::map<uint32_t,PhotoQuality>& photos,const ExportStats& stats){
                auto result=gate(model,q,photos,stats);std::ofstream file(root/"changed-formal/directional-points.bin",std::ios::binary|std::ios::app);file.put('x');file.close();return result;
            });
        check(!changed.verified && !changed.reader_verified,"directional fifth file changed after Reader without invalidating acceptance");
        auto surplusOptions=options;
        surplusOptions.phase_bytes=[&](const char* phase,uint64_t,uint64_t){
            if(std::string(phase)!="points-written")return;
            std::ifstream in(root/"surplus-formal/directional-points.bin",std::ios::binary);
            std::vector<char> bytes{std::istreambuf_iterator<char>(in),{}};in.close();bytes.insert(bytes.begin()+72,8,0);
            const uint64_t count=4;const uint32_t photo=0,feature=finiteCount+1;
            std::memcpy(bytes.data()+56,&count,8);std::memcpy(bytes.data()+72,&photo,4);std::memcpy(bytes.data()+76,&feature,4);
            std::ofstream file(root/"surplus-formal/directional-points.bin",std::ios::binary);file.write(bytes.data(),bytes.size());
        };
        const auto surplus=exportVerifiedScene(catalog,scene,[&](uint32_t id){return root/"features"/(rec.images.at(id).name+".bin");},
            root/"surplus-scratch",root/"surplus-formal",surplusOptions,gate);
        check(!surplus.verified && !surplus.reader_verified,"pre-SHA invented directional feature escaped native bidirectional coverage checks");
    }
    auto seed=gatherCanonicalSeed(topology,scene,{0,1,2},[&](uint32_t id) {return root/"features"/(rec.images.at(id).name+".bin");},64u<<20);
    check(seed.points3D.size()==finiteCount && !seed.points3D.count(dirRoot),"direction became a PnP translation anchor");
    std::cout << "typed formal native Reader finite=" << finiteCount << " direction=1 raw=" << quality.observations << " accepted=" << result.verified << '\n';
}
static int test(int argc,char** argv) {
    const auto root=argc>1?fs::u8path(argv[1]):fs::temp_directory_path()/"typed-scene-export-fixture";
    if(fs::exists(root))throw std::runtime_error("typed fixture must be new");
    run(root/"twelve",12);run(root/"eleven",11);run(root/"direction-only-bridge",12,true);
    run(root/"retained-twelve",12,false,4);run(root/"retained-cannot-inflate-eleven",11,false,16);return 0;
}
int main(int argc,char** argv) {return sfmTestMain(argc,argv,test);}
