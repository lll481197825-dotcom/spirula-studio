#include "sfm/map/RegionalRecovery.h"
#include "sfm/map/RegionalBootstrap.h"
#include "sfm/tests/TestMain.h"
#include <chrono>
#include <cstdio>

using namespace sfm;
namespace fs = std::filesystem;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> static void rejects(F operation,const char* message){bool failed=false;try{operation();}catch(const std::runtime_error&){failed=true;}require(failed,message);}

static int test(int argc,char** argv){
    const auto base=argc>1?fs::u8path(argv[1]):fs::temp_directory_path();
    const auto root=base/("held-feature-compaction-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root/"features");
    FeatureSet features;features.width=640;features.height=480;features.extract_width=320;features.extract_height=240;
    features.keypoints.resize(12);features.descriptors.resize(12*features.dim);
    for(uint32_t f=1;f<features.count();++f)features.keypoints[f]={float(12.25+f*2),float(20.5+f*3),1,0,0};
    auto camera=Camera::defaultFor(7,640,480,550,CamModel::Pinhole);camera.pixel_scale=features.pixelScale();
    MatchesDatabase db;db.images={{"outside-a",12},{"seed-a",12},{"outside-b",12},{"seed-b",12}};
    db.cameras={camera};db.camera_ids={7,7,7,7};db.pairs={{1,3,3,{{2,1},{6,4}}}};
    writeMatches((root/"matches.bin").u8string(),db);MatchesIndex index;require(indexMatches((root/"matches.bin").u8string(),index)&&index.complete,"fixture matches incomplete");
    for(const auto& image:db.images)writeFeatures((root/"features"/fs::u8path(image.name+".bin")).u8string(),features);
    Reconstruction seed;seed.cameras[7]=camera;seed.next_point3D_id=1000;
    for(uint32_t id:{1u,3u}){Image image;image.id=id;image.camera_id=7;image.name=db.images[id].name;image.registered=true;
        image.pose={mat3Identity(),{double(id),0,0}};image.point3D_ids.assign(features.count(),kInvalidPoint3D);
        for(const auto& p:features.keypoints)image.points2D.push_back({double(p.x),double(p.y)});seed.images.emplace(id,std::move(image));}
    Point3D single;single.xyz={1,2,3};single.track={{1,9},{1,9}};seed.points3D.emplace(0,single);seed.images.at(1).point3D_ids[9]=0;
    Point3D joined;joined.xyz={4,5,6};joined.track={{1,5},{3,8}};seed.points3D.emplace(17,joined);
    seed.images.at(1).point3D_ids[5]=17;seed.images.at(3).point3D_ids[8]=17;
    HeldRepairSeedContract(seed).requireUnchanged(seed,"fixture.original");const auto original=seed;
    regional::Region region;region.core=region.images={1,3};const size_t budget=64u<<20;
    auto ordinary=regional::loadRegion(index,region,root/"matches.bin",root/"features",budget);
    require(ordinary.original==std::vector<std::vector<uint32_t>>{{2,6},{1,4}},"ordinary matched-feature compaction changed");
    rejects([&]{regional::compactSeed(seed,ordinary,true);},"held compaction silently dropped assigned raw without closure");
    auto input=regional::loadRegion(index,region,root/"matches.bin",root/"features",budget,&seed);
    require(input.original==std::vector<std::vector<uint32_t>>{{2,5,6,9},{1,4,8}},"held assigned features did not join sorted original identity map");
    require(input.features[0].count()==4&&input.features[1].count()==3,"unrelated unused features entered held graph");
    require(input.db.pairs[0].matches[0].idx1==0&&input.db.pairs[0].matches[0].idx2==0&&
        input.db.pairs[0].matches[1].idx1==2&&input.db.pairs[0].matches[1].idx2==1,"held closure did not remap original matching endpoints");
    auto compact=regional::compactSeed(seed,input,true);
    require(compact.points3D.at(0).track.size()==2&&compact.points3D.at(17).track.size()==2,"single-view seed or duplicate raw multiplicity lost");
    require(compact.points3D.at(0).xyz.x==single.xyz.x&&compact.points3D.at(0).xyz.y==single.xyz.y&&compact.points3D.at(0).xyz.z==single.xyz.z,"held XYZ changed during compaction");
    auto restored=compact;regional::restoreFeatureIds(restored,input,index,&seed);HeldRepairSeedContract(seed).requireUnchanged(restored,"fixture.full-original-closure");
    for(const auto& image:seed.images)for(size_t f=0;f<features.count();++f)require(restored.images.at(image.first).points2D[f].x==image.second.points2D[f].x&&
        restored.images.at(image.first).points2D[f].y==image.second.points2D[f].y,"unrelated held XY was replaced by zero placeholder");
    require(restored.images.at(1).points2D[0].x==0&&restored.images.at(1).points2D[0].y==0,"real zero coordinate treated as missing");
    auto bad=seed;bad.images.at(1).points2D[9].x+=.125;
    rejects([&]{regional::loadRegion(index,region,root/"matches.bin",root/"features",budget,&bad);},"wrong assigned source XY passed real FeatureBin identity check");
    bad=seed;bad.images.at(1).points2D[0].x+=.125;
    rejects([&]{regional::loadRegion(index,region,root/"matches.bin",root/"features",budget,&bad);},"wrong unused source XY passed real FeatureBin identity check");
    auto badInput=input;badInput.features[0].keypoints[0].x+=.125f;
    rejects([&]{regional::compactSeed(seed,badInput,true);},"loaded original XY changed while compacting held seed");
    auto changed=compact;changed.images.at(0).points2D[0].x+=.125;
    rejects([&]{regional::restoreFeatureIds(changed,input,index,&seed);},"global closure overwrote a genuinely changed loaded XY");
    changed=compact;changed.points3D.at(0).track.pop_back();
    rejects([&]{regional::restoreFeatureIds(changed,input,index,&seed);},"global closure reduced raw duplicate multiplicity to presence");
    bool memoryRejected=false;try{regional::loadRegion(index,region,root/"matches.bin",root/"features",1,&seed);}catch(const regional::RegionInputOverBudget&){memoryRejected=true;}
    require(memoryRejected,"held feature closure bypassed original memory admission");
    regional::loadRegion(index,region,root/"matches.bin",root/"features",2000);
    memoryRejected=false;try{regional::loadRegion(index,region,root/"matches.bin",root/"features",2000,&seed);}catch(const regional::RegionInputOverBudget&){memoryRejected=true;}
    require(memoryRejected,"whole context measurement closure was absent from input memory admission");
    auto anchor=seed;anchor.images.erase(3);anchor.points3D.at(17).track.resize(1);
    auto targetInput=regional::loadRegion(index,region,root/"matches.bin",root/"features",budget,&anchor);
    require(targetInput.original[1]==std::vector<uint32_t>{1,4},"unmatched target measurements were added as artificial graph features");
    auto candidate=regional::compactSeed(anchor,targetInput,true);auto target=seed.images.at(3);target.id=1;
    target.points2D.clear();target.point3D_ids.assign(targetInput.features[1].count(),kInvalidPoint3D);
    for(const auto& p:targetInput.features[1].keypoints)target.points2D.push_back({double(p.x),double(p.y)});
    candidate.images.emplace(target.id,std::move(target));auto badTarget=candidate;badTarget.images.at(1).points2D[0].x+=.125;
    rejects([&]{regional::restoreFeatureIds(badTarget,targetInput,index,&anchor);},"closure silently overwrote changed loaded target XY");
    regional::restoreFeatureIds(candidate,targetInput,index,&anchor);
    require(candidate.images.at(3).points2D[10].x==features.keypoints[10].x&&candidate.images.at(3).points2D[10].y==features.keypoints[10].y&&
        candidate.images.at(3).point3D_ids[10]==kInvalidPoint3D,"unmatched real target measurement was not restored without invented geometry");
    Reconstruction weak;weak.cameras=seed.cameras;weak.next_point3D_id=89;
    for(uint32_t id:{1u,3u}){auto image=seed.images.at(id);image.point3D_ids.assign(features.count(),kInvalidPoint3D);weak.images.emplace(id,std::move(image));}
    Point3D weakPoint;weakPoint.xyz=single.xyz;weakPoint.track={{1,9},{3,10}};weak.points3D.emplace(88,weakPoint);
    weak.images.at(1).point3D_ids[9]=weak.images.at(3).point3D_ids[10]=88;
    require(regional::preserveRepairObservations(weak,candidate,3)==2&&regional::verifyOriginalObservations(weak,candidate)==2,
        "real original target raw outside context matching was lost after measurement closure");
    HeldRepairSeedContract(anchor).requireUnchanged(candidate,"fixture.weak-original-raw-preserved");
    for(size_t f=0;f<features.count();++f)require(seed.images.at(1).points2D[f].x==original.images.at(1).points2D[f].x&&
        seed.images.at(1).points2D[f].y==original.images.at(1).points2D[f].y,"immutable seed measurements changed");
    require(seed.points3D.at(0).track.size()==2,"immutable seed multiplicity changed");
    std::printf("PASS held feature closure: single-view fixedXYZ, duplicate raw, sorted match maps, all trueXY, unmatched weak target raw and strict rejection; ordinary scope unchanged\n");return 0;
}
int main(int argc,char** argv){return sfmTestMain(argc,argv,test);}
