#include "sfm/map/ImmutableSceneAliases.h"
#include "sfm/map/SceneCoordinator.h"
#include "sfm/map/ExportVerifiedScene.h"
#include "sfm/map/RegionalBootstrap.h"
#include "sfm/core/Matches.h"
#include <iostream>
#include <iomanip>

using namespace sfm;
using namespace sfm::regional;
using Path = std::filesystem::path;
static constexpr size_t allowance = size_t(128) << 20;
static MatchesIndex inputIndex;
static void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
static bool equalQuality(const Quality& a, const Quality& b) {
    return a.observations == b.observations && a.invalid_projections == b.invalid_projections &&
        std::abs(a.cost-b.cost) <= std::max(1e-8,std::abs(a.cost)*1e-8) &&
        std::abs(a.finite_cost-b.finite_cost) <= std::max(1e-8,std::abs(a.finite_cost)*1e-8);
}
static Path featurePath(const Path& features,const Image& image) {
    require(image.id<inputIndex.images.size(),"actual native source image ID missing from original matches metadata");
    const auto& original=inputIndex.images.at(image.id).name; auto stem=std::filesystem::u8path(image.name); stem.replace_extension();
    require(image.name==original || stem.generic_u8string()==original,"actual original source name differs from native input feature identity");
    return features / std::filesystem::u8path(original + ".bin");
}
static std::string sha(const Path& file) { const auto result=spirula::sha256_file(file.u8string()); require(!result.empty(),"actual source SHA unavailable"); return result; }
static uint64_t bytes(const Path& directory) {
    uint64_t result=0; for(const auto& entry:std::filesystem::recursive_directory_iterator(directory)) if(entry.is_regular_file()) result+=entry.file_size(); return result;
}
static void probeProgress(const char* phase,uint64_t done,uint64_t total) { std::cout<<"PHASE "<<phase<<" "<<done<<"/"<<total<<std::endl; }
static void peek(const Path& root,const std::vector<Path>& sources) {
    std::filesystem::create_directories(root); std::ofstream proof(root/"source-peek.csv");
    proof<<"source,path,bytes,SHA256,registered_photos,points,raw_rows,unique_features\n";
    std::set<uint64_t> first; uint64_t shared=0,totalRaw=0;
    for(size_t source=0;source<sources.size();++source) {
        const auto model=model_store_detail::readModel(sources[source]); std::set<uint64_t> features;
        require(model.rigs.empty() && model.rig_detached.empty(),"actual rig source codec not supported by this closed diagnostic probe");
        for(const auto& point:model.points3D) for(const auto& row:point.second.track) features.insert((uint64_t(row.image_id)<<32)|row.point2D_idx);
        if(!source) first=features; else for(uint64_t key:features) shared+=first.count(key);
        const auto raw=countObservations(model); totalRaw+=raw;
        proof<<source<<','<<verified_export_detail::csv(sources[source].u8string())<<','<<std::filesystem::file_size(sources[source])<<','<<sha(sources[source])
            <<','<<model.numRegistered()<<','<<model.points3D.size()<<','<<raw<<','<<features.size()<<'\n';
        std::cout<<"REAL_SOURCE "<<source<<" images="<<model.numRegistered()<<" points="<<model.points3D.size()<<" raw="<<raw<<" unique_features="<<features.size()<<'\n';
    }
    proof.flush(); require(bool(proof),"cannot write source peek proof");
    std::cout<<"REAL_QUEUE whole_sources="<<sources.size()<<" full_raw="<<totalRaw<<" shared_original_features="<<shared<<" observation_sampling=false\n";
}
static void checkInputs(RegionalWorkingStore& catalog,const Path& features,const Path& root,bool create) {
    const auto manifest=root/"input-source-feature-sha.csv";
    if(create) {
        std::ofstream out(manifest); out<<"kind,id,path,SHA256\n";
        for(size_t source=0;source<catalog.sourceCount();++source) out<<"source,"<<source<<','<<verified_export_detail::csv(catalog.sourcePath(source).u8string())<<','<<sha(catalog.sourcePath(source))<<'\n';
        for(const auto& image:catalog.poses().images) {
            const auto file=featurePath(features,image.second);
            const auto f=readFeatures(file.string(),false); const auto& camera=catalog.poses().cameras.at(image.second.camera_id);
            require(f.width==camera.width && f.height==camera.height && std::abs(f.pixelScale()-camera.pixel_scale)<1e-12,"actual feature camera metadata differs");
            out<<"feature,"<<image.first<<','<<verified_export_detail::csv(file.u8string())<<','<<sha(file)<<'\n';
        }
        out.flush(); require(bool(out),"cannot flush original source/feature SHA proof");
    } else {
        std::ifstream in(manifest); require(bool(in),"actual source/feature SHA proof missing"); std::string line; std::getline(in,line);
        std::map<std::string,std::string> original;
        while(std::getline(in,line)) {const auto at=line.find_last_of(','); if(at!=std::string::npos) original[line.substr(0,at)]=line.substr(at+1);}
        for(size_t source=0;source<catalog.sourceCount();++source) {
            const auto key="source,"+std::to_string(source)+","+verified_export_detail::csv(catalog.sourcePath(source).u8string());
            require(original.at(key)==sha(catalog.sourcePath(source)),"original production source changed");
        }
        for(const auto& image:catalog.poses().images) {
            const auto file=featurePath(features,image.second); const auto key="feature,"+std::to_string(image.first)+","+verified_export_detail::csv(file.u8string());
            require(original.at(key)==sha(file),"original feature file changed");
        }
    }
}
static std::vector<uint32_t> readOwners(const Path& root) {
    model_store_detail::Reader in(root/"owners.bin"); const auto result=in.vector<uint32_t>(); in.finish(); return result;
}
static void prepare(const Path& root,const Path& features,const std::vector<Path>& sources) {
    require(!std::filesystem::exists(root/"catalog"),"real probe catalogue already exists; use --export or a fresh owned directory");
    RegionalWorkingStore catalog(root/"catalog",allowance);
    for(size_t source=0;source<sources.size();++source) { const auto model=model_store_detail::readModel(sources[source]);
        require(model.rigs.empty() && model.rig_detached.empty(),"actual rig source codec not supported by this closed diagnostic probe");
        catalog.importModel(sources[source],model,"actual-source-"+std::to_string(source)); }
    checkInputs(catalog,features,root,true);
    std::vector<uint32_t> owners(catalog.poses().images.rbegin()->first+1,UINT32_MAX);
    uint64_t ordinal=0; std::ofstream plan(root/"plan.txt"); plan<<"schema=diagnostic-fixed-photo-owner-partition-v1\nscope=whole two original source models; not full 10650 capture\n";
    for(const auto& image:catalog.poses().images) {const uint32_t owner=ordinal++<catalog.poses().images.size()/2?0:1; owners[image.first]=owner; plan<<owner<<','<<image.first<<'\n';}
    plan.flush(); require(bool(plan),"cannot write fixed diagnostic owner plan");
    {model_store_detail::Writer out(root/"owners.bin"); out.vector(owners); out.finish();}
    ImmutableSceneAliases aliases(root/"aliases",allowance); aliases.prepare(catalog,owners,probeProgress);
    ImmutableSceneTopology index(root/"index",allowance);
    index.prepare(catalog,owners,aliases.identity(),aliases.points(),[&](auto key){return aliases.canonical(key);},{},"original-aligned-source-seed-v1",probeProgress);
    CanonicalScene scene(root/"scene",allowance);
    scene.initializeValidated(CanonicalScene::bindingFor(catalog,index.identity()),catalog.poses(),
        [&](const auto& emit){index.emitAliases(emit);},completeSceneSeed(index,catalog.poses(),3,allowance,2,probeProgress),
        [&](const CanonicalScene& candidate){const auto quality=sceneQuality(index,candidate,3,allowance,2,nullptr,probeProgress);
            require(quality.observations==aliases.observations() && std::isfinite(quality.cost) && std::isfinite(quality.finite_cost),"diagnostic initial fullraw coverage/finite cost failed");},
        "diagnostic-complete-support-initial-not-capture-certified");
    const auto quality=sceneQuality(index,scene,3,allowance,2);
    std::ofstream manifest(root/"queue-contract.txt"); manifest<<std::setprecision(17)
        <<"scope=actual_Gaoxin_two_whole_source_models\nfull_capture_verified=false\noriginal_observation_sampling=false\n"
        <<"catalog="<<(root/"catalog").u8string()<<"\nscene="<<(root/"scene").u8string()<<"\nindex="<<(root/"index").u8string()
        <<"\nalias_identity="<<aliases.identity()<<"\ntopology_identity="<<index.identity()<<"\nseed_identity=original-aligned-source-seed-v1\n"
        <<"scene_identity="<<scene.identity()<<"\ngeneration="<<scene.generation()<<"\npoint_slots="<<scene.pointSlots()<<"\ncanonical_points="<<scene.canonicalPoints()
        <<"\nraw="<<quality.observations<<"\ninvalid="<<quality.invalid_projections<<"\nfinite="<<quality.finite_cost<<"\ntotal="<<quality.cost
        <<"\nactual_current_disk_bytes="<<bytes(root)<<'\n'; manifest.flush(); require(bool(manifest),"cannot write bound real queue contract");
    checkInputs(catalog,features,root,false);
    std::cout<<"REAL_SCENE_READY images="<<scene.poses().numRegistered()<<" canonical="<<scene.canonicalPoints()<<" raw="<<quality.observations
        <<" invalid="<<quality.invalid_projections<<" finite="<<quality.finite_cost<<" full_capture_verified=false disk="<<bytes(root)<<'\n';
}
static void verifyExport(const Path& root,const Path& features,const std::string& label="formal-isolated") {
    require(!label.empty() && label!=".." && label.find_first_of("/\\")==std::string::npos,"real exporter output label must be one new owned path component");
    RegionalWorkingStore catalog(root/"catalog",allowance); CanonicalScene scene(root/"scene",allowance);
    require(scene.poses().rigs.empty() && scene.poses().rig_detached.empty(),"actual rig scene codec not supported by this closed diagnostic probe");
    ImmutableSceneTopology index(root/"index",allowance); index.restore(scene.binding().topology);
    scene.requireBinding(CanonicalScene::bindingFor(catalog,index.identity())); checkInputs(catalog,features,root,false);
    const auto before=scene.acceptedSnapshot(); const auto expected=sceneQuality(index,scene,3,allowance,2,nullptr,probeProgress);
    const auto materialized=root/("native-"+label);require(!std::filesystem::exists(materialized),"native materialized proof must use a new owned output label");
    Quality native; native.finite_cost=0; std::filesystem::create_directories(materialized);
    std::ofstream rows(materialized/"source-reader-quality.csv"); rows<<std::setprecision(17)<<"source,original_raw,native_raw,invalid,finite,total,multiset_preserved\n";
    for(size_t source=0;source<catalog.sourceCount();++source) {
        const auto original=model_store_detail::readModel(catalog.sourcePath(source)); const auto model=scene.materialize(catalog,source);
        const auto file=materialized/("source-"+std::to_string(source)+".bin"), journal=materialized/("raw-"+std::to_string(source)+".bin");
        model_store_detail::writeModel(model,file); writeValidation(journal,model,source); auto actual=model_store_detail::readModel(file);
        verified_export_detail::sameParameters(model,actual,VerifiedExportOptions{});
        const auto preserved=verifyOriginalObservations(original,actual); require(preserved==catalog.sourceObservationCount(source),"native materialized original raw multiset changed");
        const auto quality=validationCost(journal,actual,actual,source,3,2); accumulateQuality(native,quality);
        rows<<source<<','<<preserved<<','<<quality.observations<<','<<quality.invalid_projections<<','<<quality.finite_cost<<','<<quality.cost<<",true\n";
    }
    rows.flush(); require(bool(rows),"cannot write actual native Reader proof"); require(equalQuality(expected,native),"actual source ModelStore Reader and canonical fullraw ledger differ");
    VerifiedExportOptions options; options.cache_bytes=32u<<20; options.reader_memory_bytes=size_t(2)<<30; options.accepted_raw=expected;
    auto owners=readOwners(root); options.owners=&owners;
    for(const auto& image:scene.poses().images) options.inputs[image.first]={(Path("E:/高新区")/std::filesystem::u8path(image.second.name)).u8string(),image.second.name};
    options.input_path=[&](uint32_t id){return options.inputs.at(id).path;};
    const auto result=exportVerifiedScene(catalog,scene,[&](uint32_t image){return featurePath(features,scene.poses().images.at(image));},
        root/(label+"-stage"),root/label,options,[&](const Reconstruction& actual,const Quality&,const std::map<uint32_t,PhotoQuality>&,const ExportStats&){
            ExportFinalAcceptance proof; proof.historical_evaluated=true;
            for(const auto& input:options.inputs) {ExportConstraintPhoto row; row.image=input.first; row.path=input.second.path;
                row.reasons.push_back("architecture-only-probe: original EXIF GPS/scale/attitude full acceptance intentionally unknown"); proof.photos.push_back(row);}
            require(actual.numRegistered()==scene.poses().numRegistered(),"actual formal Reader input image count changed"); return proof;});
    for(const auto& failure:result.failures) if(failure.image==UINT32_MAX) std::cout<<"FORMAL_GATE "<<failure.reason<<'\n';
    require(result.reader_verified && !result.verified && equalQuality(expected,result.raw) && result.lineage_rows==expected.observations,"actual formal Reader raw/canonical equivalence failed or unknown capture gates passed");
    checkInputs(catalog,features,root,false);
    require(scene.generation()==before.generation && scene.acceptedSnapshot().pointHash==before.pointHash && scene.acceptedSnapshot().poseHash==before.poseHash,"read-only export changed accepted Scene");
    std::ofstream proof(root/("real-reader-export-proof-"+label+".json")); proof<<std::setprecision(17)
        <<"{\"scope\":\"actual_Gaoxin_two_whole_original_source_models\",\"whole_sources\":"<<catalog.sourceCount()
        <<",\"original_raw\":"<<expected.observations<<",\"materialized_raw\":"<<native.observations<<",\"formal_raw\":"<<result.raw.observations
        <<",\"canonical_points\":"<<scene.canonicalPoints()<<",\"formal_unique_rows\":"<<result.stats.observations
        <<",\"ledger_finite\":"<<expected.finite_cost<<",\"materialized_finite\":"<<native.finite_cost<<",\"formal_finite\":"<<result.raw.finite_cost
        <<",\"invalid\":"<<result.raw.invalid_projections<<",\"original_multiset_preserved\":true,\"actual_COLMAP_native_Reader\":true,\"unique_XYZ_authority\":true"
        <<",\"source_feature_SHA_unchanged\":true,\"accepted_scene_read_only\":true,\"GPS_scale_attitude\":\"unknown_not_capture_validation\",\"full_capture_verified\":false,\"disk_bytes\":"<<bytes(root)<<"}\n";
    proof.flush(); require(bool(proof),"cannot flush actual real native proof");
    std::cout<<"REAL_NATIVE_EXPORT raw="<<result.raw.observations<<" unique="<<result.stats.observations<<" invalid="<<result.raw.invalid_projections
        <<" canonical_finite="<<expected.finite_cost<<" materialized_finite="<<native.finite_cost<<" formal_finite="<<result.raw.finite_cost
        <<" source_feature_SHA_unchanged=true accepted_scene_unchanged=true full_capture_verified=false disk="<<bytes(root)<<'\n';
}
int main(int argc,char** argv) try {
    std::cout<<std::setprecision(17); require(argc>=4,"usage: --peek|--prepare|--export proofRoot features [wholeSourceA wholeSourceB]");
    const std::string mode=argv[1]; const auto root=std::filesystem::u8path(argv[2]),features=std::filesystem::u8path(argv[3]);
    std::vector<Path> sources; for(int i=4;i<argc;++i) sources.push_back(std::filesystem::u8path(argv[i]));
    require(indexMatches((features.parent_path()/"matches.bin").string(),inputIndex) && inputIndex.complete,"actual native matches input metadata incomplete");
    if(mode=="--peek") {require(sources.size()==2,"probe requires exact two complete original sources"); peek(root,sources);}
    else if(mode=="--prepare") {require(sources.size()==2,"probe requires exact two complete original sources"); prepare(root,features,sources);}
    else if(mode=="--export") verifyExport(root,features,argc>4?argv[4]:"formal-isolated");
    else throw std::runtime_error("unknown actual native probe mode");
    return 0;
} catch(const std::exception& error) {std::cerr<<"REAL_NATIVE_PROBE_ERROR "<<error.what()<<'\n';return 1;}
