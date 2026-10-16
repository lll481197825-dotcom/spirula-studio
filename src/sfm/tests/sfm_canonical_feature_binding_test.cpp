#include "sfm/map/CanonicalRecovery.h"
#include "sfm/map/ImmutableSceneAliases.h"
#include "sfm/map/HeldRepairSeed.h"
#include "sfm/map/SceneVisualEvidence.h"
#include "sfm/tests/TestMain.h"
#include <iostream>
using namespace sfm;
static void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
static bool xyzEqual(Vec3 a, Vec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
template<class F> static void rejects(F f, const char* why) {
    bool failed=false; try { f(); } catch(const std::exception&) { failed=true; } require(failed,why);
}
static uint64_t rows(const Reconstruction& m) { uint64_t n=0; for(const auto& p:m.points3D)n+=p.second.track.size();return n; }
static Reconstruction original() {
    Reconstruction m; Camera c;c.id=7;c.width=640;c.height=480;c.fx=c.fy=500;c.cx=320;c.cy=240;m.cameras.emplace(7,c);
    for(uint32_t id=0;id<3;++id){Image i;i.id=id;i.camera_id=7;i.name="photo-"+std::to_string(id);i.registered=true;i.pose={mat3Identity(),{double(id),0,0}};i.points2D={{10,20},{30,40}};i.point3D_ids={kInvalidPoint3D,kInvalidPoint3D};m.images.emplace(id,i);}
    Point3D a;a.xyz={1,2,10};a.track={{0,0},{2,0},{2,0}};m.points3D.emplace(0,a);
    Point3D b;b.xyz={3,2,10};b.track={{1,0}};m.points3D.emplace(1,b);
    Point3D cpoint;cpoint.xyz={5,2,10};cpoint.track={{2,1}};m.points3D.emplace(2,cpoint);
    for(const auto& p:m.points3D)for(const auto& r:p.second.track)m.images.at(r.image_id).point3D_ids[r.point2D_idx]=p.first;
    m.next_point3D_id=3;return m;
}
static std::string nativeHash(const Reconstruction& model, const std::filesystem::path& path) {
    model_store_detail::writeModel(model,path);return spirula::sha256_file(path.u8string());
}
static void wholeTrackTest(const std::filesystem::path& root) {
    namespace fs=std::filesystem;fs::create_directories(root);
    Reconstruction original;Camera camera=Camera::defaultFor(7,640,480,500,CamModel::Pinhole);camera.pixel_scale=2;original.cameras.emplace(7,camera);
    for(uint32_t id:{0u,2u}){Image image;image.id=id;image.camera_id=7;image.name="anchor-"+std::to_string(id);image.registered=true;image.pose={mat3Identity(),{double(id),0,0}};image.points2D.resize(2);image.point3D_ids={0,1};original.images.emplace(id,image);}
    for(uint64_t id=0;id<2;++id){Point3D point;point.xyz={double(id),0,10};point.track={{0,uint32_t(id)},{2,uint32_t(id)}};for(const auto& row:point.track){auto& image=original.images.at(row.image_id);image.points2D[row.point2D_idx]=camera.project(point.xyz+image.pose.t);}original.points3D.emplace(id,point);}original.next_point3D_id=2;
    const auto source=root/"source.bin";const auto sourceSHA=nativeHash(original,source);
    regional::RegionalWorkingStore catalog(root/"catalog",32u<<20);catalog.importModel(source,original,"whole-track-source");
    regional::ImmutableSceneAliases aliases(root/"aliases",32u<<20);aliases.prepare(catalog,{0,0,0});
    regional::ImmutableSceneTopology index(root/"index",32u<<20);index.prepare(catalog,{0,0,0},aliases.identity(),aliases.points(),[&](auto key){return aliases.canonical(key);});
    auto poses=catalog.poses();poses.points3D.clear();for(auto& item:poses.images){item.second.points2D.clear();item.second.point3D_ids.clear();}
    regional::CanonicalScene scene(root/"scene",32u<<20);scene.initialize(regional::CanonicalScene::bindingFor(catalog,index.identity()),poses,[&](auto emit){aliases.emitAliases(emit);},[&](auto emit){index.visitSeeds([&](auto seed){emit(seed.root,seed.xyz);});});
    const auto headSHA=spirula::sha256_file((root/"scene/head.bin").u8string());
    auto candidate=original;candidate.images.erase(2);candidate.points3D.clear();candidate.images.at(0).point3D_ids={2,kInvalidPoint3D};
    Image newPhoto=candidate.images.at(0);newPhoto.id=1;newPhoto.name="new-photo-1";newPhoto.pose.t={1,0,0};newPhoto.point3D_ids={2,kInvalidPoint3D};newPhoto.points2D[0]=camera.project(scene.pointXYZ(0)+newPhoto.pose.t);candidate.images.emplace(1,newPhoto);
    Point3D privatePoint;privatePoint.xyz=scene.pointXYZ(0);privatePoint.track={{0,0},{1,0},{1,0}};candidate.points3D.emplace(2,privatePoint);candidate.next_point3D_id=3;
    auto rejectUnchanged=[&](Reconstruction incoming,const char* token,const char* reason){
        const auto before=nativeHash(incoming,root/(std::string(token)+"-before.bin"));bool typed=false;
        try{regional::restoreAcceptedFeatureBindings(index,scene,incoming,4u<<20,3);}catch(const regional::CanonicalRepairBindingRejected& error){typed=std::string(error.what()).find(reason)!=std::string::npos;}
        require(typed,"geometric binding rejection was untyped or absent");require(nativeHash(incoming,root/(std::string(token)+"-after.bin"))==before,"geometric preflight rejection mutated original candidate");
    };
    auto bridge=candidate;bridge.images.emplace(2,original.images.at(2));bridge.images.at(2).point3D_ids={kInvalidPoint3D,2};bridge.points3D.at(2).track.push_back({2,1});
    rejectUnchanged(bridge,"multiple-roots","multiple accepted canonical roots");
    auto off=candidate;off.images.at(1).points2D[0].x+=6.01;rejectUnchanged(off,"off-authority","full visual support");
    auto badXY=candidate;badXY.images.at(0).points2D[0].x+=1;bool hard=false;
    try{regional::restoreAcceptedFeatureBindings(index,scene,badXY,4u<<20,3);}catch(const regional::CanonicalRepairBindingRejected&){throw std::runtime_error("identity fault misclassified as ordinary geometry rejection");}catch(const std::runtime_error&){hard=true;}
    require(hard,"original XY identity drift accepted");
    const auto before=candidate;const auto report=regional::restoreAcceptedFeatureBindings(index,scene,candidate,4u<<20,3);
    require(report.moved_raw_rows==3&&report.reassociated_features==2&&report.removed_empty_private_points==1&&rows(candidate)==3,"whole track or duplicate raw count lost");
    require(candidate.points3D.size()==1&&candidate.points3D.count(0)&&!candidate.points3D.count(2)&&candidate.points3D.at(0).track.size()==3&&xyzEqual(candidate.points3D.at(0).xyz,scene.pointXYZ(0)),"whole track authority root0 changed");
    for(const auto& image:before.images){const auto& after=candidate.images.at(image.first);require(after.pose.R==image.second.pose.R&&xyzEqual(after.pose.t,image.second.pose.t)&&after.name==image.second.name,"whole-track binding changed P or photo metadata");require(after.point3D_ids[0]==0,"new photo lost canonical root0 support");for(size_t f=0;f<after.points2D.size();++f)require(after.points2D[f].x==image.second.points2D[f].x&&after.points2D[f].y==image.second.points2D[f].y,"whole-track binding altered XY");}
    require(HeldRepairSeedContract::sameCamera(before.cameras.at(7),candidate.cameras.at(7)),"whole-track binding changed K");
    const auto once=nativeHash(candidate,root/"once.bin");const auto twice=regional::restoreAcceptedFeatureBindings(index,scene,candidate,4u<<20,3);
    require(!twice.reassociated_features&&!twice.moved_raw_rows&&!twice.borrowed_points&&twice.raw_rows==3&&nativeHash(candidate,root/"twice.bin")==once,"new-photo canonical whole-track binding was not idempotent");
    const auto restoredPath=root/"restored.bin";model_store_detail::writeModel(candidate,restoredPath);auto native=model_store_detail::readModel(restoredPath);require(rows(native)==3&&native.points3D.at(0).track[1].image_id==1&&native.points3D.at(0).track[2].image_id==1,"native Reader dropped new-photo duplicate rows");
    regional::RegionalWorkingStore resultCatalog(root/"result-catalog",32u<<20);resultCatalog.importModel(restoredPath,native,"restored-track");
    regional::ImmutableSceneAliases resultAliases(root/"result-aliases",32u<<20);resultAliases.prepare(resultCatalog,{0,0});
    regional::ImmutableSceneTopology resultIndex(root/"result-index",32u<<20);resultIndex.prepare(resultCatalog,{0,0},resultAliases.identity(),resultAliases.points(),[&](auto key){return resultAliases.canonical(key);});
    auto resultPoses=resultCatalog.poses();resultPoses.points3D.clear();for(auto& image:resultPoses.images){image.second.points2D.clear();image.second.point3D_ids.clear();}
    regional::CanonicalScene resultScene(root/"result-scene",32u<<20);resultScene.initialize(regional::CanonicalScene::bindingFor(resultCatalog,resultIndex.identity()),resultPoses,[&](auto emit){resultAliases.emitAliases(emit);},[&](auto emit){resultIndex.visitSeeds([&](auto seed){emit(seed.root,seed.xyz);});});
    const auto evidence=regional::sceneVisualEvidence(resultIndex,resultScene,3,16u<<20,1);
    require(resultIndex.observationCount()==3&&evidence.uniqueObservations==2&&evidence.landmarks.at(0)==1&&evidence.landmarks.at(1)==1&&evidence.components==1,"restoration split a valid two-view track into singleton visual support");
    auto scaled=candidate;const auto track=scaled.points3D.at(0);scaled.points3D.clear();scaled.points3D.emplace(2,track);scaled.next_point3D_id=3;scaled.images.at(0).point3D_ids[0]=2;scaled.images.at(1).point3D_ids[0]=2;scaled.images.at(1).points2D[0].x+=5.5;
    regional::restoreAcceptedFeatureBindings(index,scene,scaled,4u<<20,3);require(rows(scaled)==3&&scaled.images.at(1).point3D_ids[0]==0,"visual gate ignored original pixel_scale=2");
    require(sourceSHA==spirula::sha256_file(source.u8string())&&headSHA==spirula::sha256_file((root/"scene/head.bin").u8string()),"whole-track test changed immutable source or accepted head");
}
static int test(int argc,char** argv) {
    namespace fs=std::filesystem;
    require(argc==2,"requires a new isolated fixture directory");const auto root=fs::u8path(argv[1]);require(!fs::exists(root),"fixture directory must be new");fs::create_directories(root);
    auto source=original();const auto path=root/"original.bin";model_store_detail::writeModel(source,path);
    const auto sourceSHA=spirula::sha256_file(path.u8string());
    regional::RegionalWorkingStore catalog(root/"catalog",32u<<20);catalog.importModel(path,source,"original");
    regional::ImmutableSceneAliases aliases(root/"aliases",32u<<20);aliases.prepare(catalog,{0,0,0});
    regional::ImmutableSceneTopology topology(root/"indices",32u<<20);topology.prepare(catalog,{0,0,0},aliases.identity(),aliases.points(),[&](auto key){return aliases.canonical(key);});
    auto poses=catalog.poses();for(auto& im:poses.images){im.second.points2D.clear();im.second.point3D_ids.clear();}poses.points3D.clear();
    regional::CanonicalScene scene(root/"scene",32u<<20);scene.initialize(regional::CanonicalScene::bindingFor(catalog,topology.identity()),poses,[&](auto emit){aliases.emitAliases(emit);},[&](auto emit){topology.visitSeeds([&](auto p){emit(p.root,p.xyz);});});
    const auto headSHA=spirula::sha256_file((root/"scene/head.bin").u8string());
    Reconstruction held=source;held.images.erase(2);held.points3D.erase(2);for(auto& p:held.points3D)p.second.track.erase(std::remove_if(p.second.track.begin(),p.second.track.end(),[](auto r){return r.image_id==2;}),p.second.track.end());
    HeldRepairSeedContract fixed(held);
    auto candidate=source;candidate.images.at(2).pose.t.x+=.125;
    candidate.points3D.at(0).track={{0,0}};candidate.points3D.at(1).track.push_back({2,0});candidate.points3D.at(1).track.push_back({2,0});candidate.images.at(2).point3D_ids[0]=1;
    auto privatePoint=candidate.points3D.at(2);candidate.points3D.erase(2);privatePoint.xyz={50,20,10};candidate.points3D.emplace(3,privatePoint);candidate.images.at(2).point3D_ids[1]=3;candidate.next_point3D_id=4;
    const auto before=candidate;fixed.requireUnchanged(candidate);
    auto malformed=candidate;malformed.points3D.at(0).xyz.x+=.001;
    rejects([&]{regional::restoreAcceptedFeatureBindings(topology,scene,malformed,4u<<20);},"borrowed XYZ drift was overwritten");
    malformed=candidate;malformed.images.at(2).points2D[0].x+=1;
    rejects([&]{regional::restoreAcceptedFeatureBindings(topology,scene,malformed,4u<<20);},"original feature XY drift was accepted");
    malformed=candidate;rejects([&]{regional::restoreAcceptedFeatureBindings(topology,scene,malformed,1);},"scratch budget ignored");
    const auto restored=regional::restoreAcceptedFeatureBindings(topology,scene,candidate,4u<<20);
    fixed.requireUnchanged(candidate);
    require(restored.reassociated_features==2&&restored.moved_raw_rows==3&&restored.borrowed_points==1&&restored.removed_empty_private_points==1,"restored counts incorrect");
    require(rows(candidate)==rows(before)&&rows(candidate)==5&&candidate.images.at(2).point3D_ids[0]==0&&candidate.images.at(2).point3D_ids[1]==2,"duplicate raw or authoritative roots lost");
    require(candidate.points3D.at(0).track.size()==3&&candidate.points3D.at(1).track.size()==1&&!candidate.points3D.count(3)&&xyzEqual(candidate.points3D.at(2).xyz,scene.pointXYZ(2)),"wrong source row retained or private empty point survived");
    for(const auto& im:before.images){const auto& after=candidate.images.at(im.first);require(after.name==im.second.name&&after.pose.R==im.second.pose.R&&xyzEqual(after.pose.t,im.second.pose.t),"corrected P or image metadata changed");for(size_t f=0;f<after.points2D.size();++f)require(after.points2D[f].x==im.second.points2D[f].x&&after.points2D[f].y==im.second.points2D[f].y,"incoming XY changed");}
    require(HeldRepairSeedContract::sameCamera(before.cameras.at(7),candidate.cameras.at(7)),"corrected K changed");
    const auto again=regional::restoreAcceptedFeatureBindings(topology,scene,candidate,4u<<20);require(!again.reassociated_features&&again.raw_rows==5,"restoration not idempotent");
    auto absent=candidate;absent.points3D.at(0).track.erase(std::remove_if(absent.points3D.at(0).track.begin(),absent.points3D.at(0).track.end(),[](auto r){return r.image_id==2;}),absent.points3D.at(0).track.end());absent.images.at(2).point3D_ids[0]=kInvalidPoint3D;
    absent.images.at(2).points2D[0]={0,0};
    const auto unassigned=regional::restoreAcceptedFeatureBindings(topology,scene,absent,4u<<20);require(!unassigned.reassociated_features&&rows(absent)==3&&absent.images.at(2).point3D_ids[0]==kInvalidPoint3D,"missing incoming old feature was invented");
    require(sourceSHA==spirula::sha256_file(path.u8string())&&headSHA==spirula::sha256_file((root/"scene/head.bin").u8string()),"old source or accepted head changed");
    wholeTrackTest(root/"whole-track");
    std::cout<<"PASS canonical binding: old corrections/raw=5 preserved; new-photo two-view root0 track/raw=3 retained; typed bridge/off-XYZ rejection is preflight; pixel_scale=2, idempotence, P/K/source/head SHA preserved. Two-view fixture is not full-capture acceptance.\n";return 0;
}
int main(int argc,char**argv){return sfmTestMain(argc,argv,test);}
