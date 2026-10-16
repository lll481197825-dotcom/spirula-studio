#include "sfm/map/ExportVerifiedScene.h"
#include "sfm/map/CanonicalScene.h"
#include "sfm/map/RegionalBootstrap.h"
#include "sfm/tests/TestMain.h"
#include <iostream>

using namespace sfm;
namespace fs=std::filesystem;
using Store=regional::RegionalWorkingStore;
using Scene=regional::CanonicalScene;
static void check(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> static void rejects(F fn,const char* message) { bool failed=false; try { fn(); } catch(const std::exception&) { failed=true; } check(failed,message); }

static Reconstruction fixture() {
    Reconstruction rec; rec.cameras[0]=Camera::defaultFor(0,800,600,500,CamModel::Pinhole);
    for(uint32_t i=0;i<3;++i) {
        Image im; im.id=i; im.camera_id=0; im.registered=true; im.name="photo-"+std::to_string(i)+".jpg";
        im.pose.t={-.5*i,0,0}; im.points2D.resize(12); im.point3D_ids.resize(12); rec.images[i]=im;
    }
    for(uint64_t p=0;p<12;++p) {
        Point3D point; point.xyz={.02*double(p),.04*double(p%4),5+.03*double(p)};
        point.rgb[0]=30; point.rgb[1]=60; point.rgb[2]=90;
        for(auto& kv:rec.images) {
            const auto xy=rec.cameras[0].project(point.xyz+kv.second.pose.t);
            kv.second.points2D[p]={double(float(xy.x)),double(float(xy.y))}; kv.second.point3D_ids[p]=p;
            point.track.push_back({kv.first,uint32_t(p)});
            if(kv.first==0) point.track.push_back({kv.first,uint32_t(p)});
        }
        rec.points3D[p]=point;
    }
    rec.next_point3D_id=12; return rec;
}
static regional::Quality fullQuality(const Store& store,const Scene& scene) {
    regional::Quality result; result.finite_cost=0;
    store.visitObservations([&](const Store::RawObservation& row) { regional::PhotoQuality photo;
        const auto& im=scene.poses().images.at(row.image);
        regional::verified_export_detail::projection(result,photo,scene.poses().cameras.at(im.camera_id),im,row.xy,scene.xyz(row.point),3); });
    return result;
}
static regional::ExportFinalAcceptance groundTruthProof(const Reconstruction& model,const Reconstruction& truth,
    const regional::VerifiedExportOptions& options) {
    regional::ExportFinalAcceptance proof; proof.historical_evaluated=true;
    auto first=model.images.begin(),second=std::next(first);
    const double before=(cameraCenter(truth.images.at(first->first).pose)-cameraCenter(truth.images.at(second->first).pose)).norm();
    const double after=(cameraCenter(first->second.pose)-cameraCenter(second->second.pose)).norm();
    for(const auto& item:model.images) {
        regional::ExportConstraintPhoto row; row.image=item.first; row.path=options.input_path(item.first);
        row.gps_error_metres=(cameraCenter(item.second.pose)-cameraCenter(truth.images.at(item.first).pose)).norm();
        row.gps_verified=std::isfinite(row.gps_error_metres) && row.gps_error_metres<=5;
        row.scale_error_fraction=std::abs(after/before-1); row.scale_verified=std::isfinite(row.scale_error_fraction) && row.scale_error_fraction<=.01;
        double trace=0; for(size_t k=0;k<9;++k) trace+=item.second.pose.R[k]*truth.images.at(item.first).pose.R[k];
        row.attitude_error_degrees=std::acos(std::clamp((trace-1)*.5,-1.,1.))*180/M_PI;
        row.attitude_verified=std::isfinite(row.attitude_error_degrees) && row.attitude_error_degrees<=10;
        proof.photos.push_back(row);
    } return proof;
}
static int test(int argc,char** argv) {
    const fs::path root=argc>1?fs::u8path(argv[1]):fs::temp_directory_path()/("verified-scene-export-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root/"features"); const auto rec=fixture();
    const auto source=root/"source-a.bin", second=root/"source-b.bin";
    model_store_detail::writeModel(rec,source); model_store_detail::writeModel(rec,second);
    const auto sourceSHA=spirula::sha256_file(source.u8string());
    for(const auto& item:rec.images) {
        FeatureSet f; f.width=800; f.height=600; f.keypoints.resize(12); f.descriptors.resize(12*f.dim);
        for(size_t k=0;k<12;++k) { f.keypoints[k].x=float(item.second.points2D[k].x); f.keypoints[k].y=float(item.second.points2D[k].y); }
        writeFeatures((root/"features"/(item.second.name+".bin")).u8string(),f);
    }
    auto damaged=rec; damaged.points3D.at(0).track.erase(damaged.points3D.at(0).track.begin()+1);
    rejects([&]{ regional::verifyOriginalObservations(rec,damaged); },"multiplicity 2->1 did not fail verification");
    const auto kept=regional::preserveRepairObservations(rec,damaged,3);
    check(kept==48 && countObservations(damaged)==48 && regional::verifyOriginalObservations(rec,damaged)==48,"preserve did not restore missing duplicate row");
    model_store_detail::writeModel(damaged,root/"repaired-multiset.bin");
    check(countObservations(model_store_detail::readModel(root/"repaired-multiset.bin"))==48,"repaired duplicate lost native serialization");
    std::cout<<"RAW_MULTISET original=48 damaged=47 verify_rejected=true preserved_return="<<kept<<" actual_after=48 native_reader=48\n";
    Store catalog(root/"catalog",64u<<20); catalog.importModel(source,rec,"source-a"); catalog.importModel(second,rec,"source-b");
    Scene scene(root/"scene",64u<<20); auto binding=Scene::bindingFor(catalog,"fixture-topology");
    scene.initialize(binding,catalog.poses(),[&](const Scene::AliasEmit& emit) {
        for(size_t s=0;s<2;++s) for(uint64_t p=0;p<12;++p) emit(catalog.key(s,p),p);
    },[&](const Scene::PointEmit& emit) { for(const auto& item:rec.points3D) { auto xyz=item.second.xyz; if(item.first==0) xyz.x+=.002; emit(item.first,xyz); } });
    const auto initialQuality=fullQuality(catalog,scene);
    scene.beginCandidate(); scene.applyPointUpdates({{0,rec.points3D.at(0).xyz}});
    const auto proposal=scene.sealCandidate("export-native-candidate-fixture");
    const auto candidateQuality=fullQuality(catalog,scene); regional::Quality candidateReader; candidateReader.finite_cost=0;
    for(size_t sourceId=0;sourceId<2;++sourceId) {
        const auto model=scene.materialize(catalog,sourceId); const auto file=root/("candidate-"+std::to_string(sourceId)+".bin");
        const auto journal=root/("candidate-validation-"+std::to_string(sourceId)+".bin");
        model_store_detail::writeModel(model,file); regional::writeValidation(journal,model,sourceId);
        auto actualModel=model_store_detail::readModel(file);
        regional::verified_export_detail::sameParameters(model,actualModel,regional::VerifiedExportOptions{});
        regional::accumulateQuality(candidateReader,regional::validationCost(journal,actualModel,actualModel,sourceId,3,1));
    }
    check(candidateQuality.cost<initialQuality.cost && candidateQuality.invalid_projections==0 &&
        candidateReader.observations==96 && std::abs(candidateReader.cost-candidateQuality.cost)<1e-8,
        "candidate native Reader/full raw improvement gate failed");
    scene.commit("accepted-after-native-full-raw-fixture");
    check(scene.generation()==proposal.generation && !scene.editing(),"candidate did not become the bound accepted snapshot");
    std::cout<<"CANDIDATE_ACCEPTED native_raw="<<candidateReader.observations<<" cost_before="<<initialQuality.cost
        <<" cost_candidate="<<candidateQuality.cost<<" reader_cost="<<candidateReader.cost<<" invalid=0\n";
    regional::VerifiedExportOptions options; options.cache_bytes=8u<<20; options.reader_memory_bytes=64u<<20;
    options.accepted_raw=fullQuality(catalog,scene); options.input_path=[&](uint32_t id){return (root/"inputs"/rec.images.at(id).name).u8string();};
    for(const auto& item:rec.images) options.inputs[item.first]={options.input_path(item.first),item.second.name};
    auto paths=[&](uint32_t id){return root/"features"/(rec.images.at(id).name+".bin");};
    const regional::ExportFinalGate accepted=[&](const Reconstruction& model,const regional::Quality&,const std::map<uint32_t,regional::PhotoQuality>&,const regional::ExportStats&){
        return groundTruthProof(model,rec,options);
    };
    auto result=regional::exportVerifiedScene(catalog,scene,paths,root/"export-stage",root/"export",options,accepted);
    for(const auto& failure:result.failures) std::cerr<<"EXPORT_FAIL "<<failure.reason<<'\n';
    check(result.verified && result.reader_verified && result.lineage_rows==96 && result.raw.observations==96 && result.stats.observations==36 && result.stats.points==12,"complete raw/native formal export failed");
    check(result.forecast.features==36 && result.observed_owned_bytes_at_checkpoints>0 && result.observed_owned_bytes_at_checkpoints<=result.forecast.required_peak_bytes,
        "real typed feature/export file checkpoint capacity forecast was not measured or exceeded");
    auto actual=Reconstruction::readBinary((root/"export").u8string());
    std::vector<double> actualResiduals;
    for(const auto& item:actual.points3D) { const auto xyz=scene.pointXYZ(item.first-1);
        check(item.second.xyz.x==xyz.x && item.second.xyz.y==xyz.y && item.second.xyz.z==xyz.z,"export secretly refined canonical XYZ");
        for(const auto& obs:item.second.track) {const auto& image=actual.images.at(obs.image_id); actualResiduals.push_back(reprojErrorAt(actual.cameras.at(image.camera_id),image.pose,image.points2D.at(obs.point2D_idx),item.second.xyz));} }
    std::sort(actualResiduals.begin(),actualResiduals.end());
    check(result.stats.components==1 && std::abs(result.stats.median-(actualResiduals[17]+actualResiduals[18])*.5)<1e-14,"formal exact residual median/connectivity statistics differ");
    std::vector<uint32_t> owners{0,0,1}; auto regionOptions=options; regionOptions.owners=&owners;
    const auto regionResult=regional::exportVerifiedScene(catalog,scene,paths,root/"owner-stage",root/"owner-export",regionOptions,accepted);
    check(regionResult.verified && regionResult.stats.components==1 && regionResult.stats.cross_region_points==12,"formal owner-region connectivity statistics differ");
    auto stemModel=rec; for(auto& image:stemModel.images) image.second.name=fs::u8path(image.second.name).stem().u8string();
    const auto stemSource=root/"stem-source.bin"; model_store_detail::writeModel(stemModel,stemSource);
    Store stemCatalog(root/"stem-catalog",64u<<20); stemCatalog.importModel(stemSource,stemModel,"stem-source");
    Scene stemScene(root/"stem-scene",64u<<20); stemScene.initialize(Scene::bindingFor(stemCatalog,"stem-topology"),stemCatalog.poses(),
        [&](const Scene::AliasEmit& emit){for(uint64_t p=0;p<12;++p) emit(stemCatalog.key(0,p),p);},
        [&](const Scene::PointEmit& emit){for(const auto& point:stemModel.points3D) emit(point.first,point.second.xyz);});
    auto stemOptions=options; stemOptions.accepted_raw=fullQuality(stemCatalog,stemScene);
    for(const auto& image:stemModel.images) stemOptions.inputs.at(image.first)={options.input_path(image.first),image.second.name,rec.images.at(image.first).name};
    const auto stemResult=regional::exportVerifiedScene(stemCatalog,stemScene,paths,root/"stem-stage",root/"stem-export",stemOptions,
        [&](const Reconstruction& model,const regional::Quality&,const std::map<uint32_t,regional::PhotoQuality>&,const regional::ExportStats&){return groundTruthProof(model,rec,stemOptions);});
    check(stemResult.verified && stemResult.reader_verified,"explicit original stem to actual photo filename mapping failed");
    const auto stemActual=Reconstruction::readBinary((root/"stem-export").u8string());
    for(const auto& image:stemActual.images) check(image.second.name==rec.images.at(image.first).name && stemScene.poses().images.at(image.first).name==stemModel.images.at(image.first).name,"formal photo names or immutable internal feature stems changed");
    auto missingStem=stemOptions; for(auto& image:missingStem.inputs) image.second.export_name.clear();
    const auto missingStemResult=regional::exportVerifiedScene(stemCatalog,stemScene,paths,root/"missing-stem-stage",root/"missing-stem-export",missingStem,accepted);
    check(!missingStemResult.verified && !missingStemResult.reader_verified,"missing original filename mapping was guessed or silently published");
    Scene rigScene(root/"unsupported-rig-scene",64u<<20); auto rigPoses=catalog.poses(); rigPoses.rigs.emplace_back();
    rigScene.initialize(binding,rigPoses,[&](const Scene::AliasEmit& emit){for(size_t s=0;s<2;++s)for(uint64_t p=0;p<12;++p)emit(catalog.key(s,p),p);},
        [&](const Scene::PointEmit& emit){for(const auto& point:rec.points3D)emit(point.first,point.second.xyz);});
    const auto rigResult=regional::exportVerifiedScene(catalog,rigScene,paths,root/"unsupported-rig-stage",root/"unsupported-rig-export",options,accepted);
    check(!rigResult.verified && !rigResult.reader_verified,"unsupported rig metadata silently dropped by verified export");
    std::cout<<"ORIGINAL_PHOTO_NAME explicit_stem_mapping=true actual_native_Reader_name=true internal_stem_unchanged=true missing_mapping_closed=true\n";
    check(fs::file_size(root/"export"/"raw-alias-map.bin")==16+96*16,"raw mapping duplicated input XY or lost row coverage");
    auto denied=regional::exportVerifiedScene(catalog,scene,paths,root/"denied-stage",root/"denied",options,
        [&](const Reconstruction& model,const regional::Quality& quality,const std::map<uint32_t,regional::PhotoQuality>& photos,const regional::ExportStats& stats){
            auto proof=accepted(model,quality,photos,stats); proof.photos.front().gps_verified=true; proof.photos.front().gps_error_metres=6;
            proof.photos.front().reasons.push_back("GPS exceeds original 5m limit"); return proof;});
    check(!denied.verified && denied.reader_verified && fs::exists(root/"denied"/"export-per-photo.csv"),"GPS gate bypassed or partial artifact not reported");
    auto unknown=regional::exportVerifiedScene(catalog,scene,paths,root/"unknown-stage",root/"unknown",options,{});
    check(!unknown.verified && unknown.reader_verified && !unknown.failures.empty() && fs::exists(root/"unknown"/"points3D.bin"),"unknown external full gate accepted or isolated formal artifact lost");
    auto emptyProof=regional::exportVerifiedScene(catalog,scene,paths,root/"empty-proof-stage",root/"empty-proof",options,
        [](const Reconstruction&,const regional::Quality&,const std::map<uint32_t,regional::PhotoQuality>&,const regional::ExportStats&){return regional::ExportFinalAcceptance{};});
    check(!emptyProof.verified && emptyProof.reader_verified,"present but no-op full gate accepted unknown constraints");
    auto missingInputOptions=options; missingInputOptions.inputs[7]={(root/"inputs"/"missing-original.jpg").u8string(),"missing-original.jpg"};
    const auto missingInput=regional::exportVerifiedScene(catalog,scene,paths,root/"missing-input-stage",root/"missing-input",missingInputOptions,accepted);
    check(missingInput.reader_verified && !missingInput.verified && fs::exists(root/"missing-input"/"points3D.bin"),"omitted original input was hidden by registered scene count");
    auto unknownInventoryOptions=options; unknownInventoryOptions.inputs.clear();
    const auto unknownInventory=regional::exportVerifiedScene(catalog,scene,paths,root/"unknown-inventory-stage",root/"unknown-inventory",unknownInventoryOptions,accepted);
    check(unknownInventory.reader_verified && !unknownInventory.verified,"unknown original input inventory was accepted as a full capture");
    auto historyOptions=options; historyOptions.historical_required=true;
    const auto historyMissing=regional::exportVerifiedScene(catalog,scene,paths,root/"missing-history-stage",root/"missing-history",historyOptions,accepted);
    check(historyMissing.reader_verified && !historyMissing.verified,"missing required historical source certification accepted");
    bool poisoned=false;
    const auto poisonXYZ=regional::exportVerifiedScene(catalog,scene,[&](uint32_t image){
        if(!poisoned && fs::is_regular_file(root/"poison-xyz"/"points3D.bin")) { poisoned=true; std::fstream file(root/"poison-xyz"/"points3D.bin",std::ios::in|std::ios::out|std::ios::binary);
            file.seekp(16); double wrong=.123; file.write(reinterpret_cast<const char*>(&wrong),sizeof(wrong)); }
        return paths(image);
    },root/"poison-xyz-stage",root/"poison-xyz",options,accepted);
    check(!poisonXYZ.verified && !poisonXYZ.reader_verified,"actual serialized XYZ mutation bypassed native raw acceptance");
    const auto changedFile=regional::exportVerifiedScene(catalog,scene,paths,root/"changed-file-stage",root/"changed-file",options,
        [&](const Reconstruction& model,const regional::Quality& quality,const std::map<uint32_t,regional::PhotoQuality>& photos,const regional::ExportStats& stats){
            auto proof=accepted(model,quality,photos,stats); std::fstream file(root/"changed-file"/"images.bin",std::ios::in|std::ios::out|std::ios::binary);
            file.seekp(44); double wrong=.02; file.write(reinterpret_cast<const char*>(&wrong),sizeof(wrong)); return proof;
        });
    check(!changedFile.verified && !changedFile.reader_verified,"actual formal file mutation after native acceptance was published");
    auto badCostOptions=options; badCostOptions.accepted_raw.cost+=1;
    auto badCost=regional::exportVerifiedScene(catalog,scene,paths,root/"bad-cost-stage",root/"bad-cost",badCostOptions,accepted);
    check(!badCost.verified && !badCost.reader_verified,"wrong complete raw objective accepted");
    Scene conflict(root/"conflict-scene",64u<<20);
    conflict.initialize(binding,catalog.poses(),[&](const Scene::AliasEmit& emit){
        for(size_t s=0;s<2;++s) for(uint64_t p=0;p<12;++p) emit(catalog.key(s,p),s*12+p);
    },[&](const Scene::PointEmit& emit){for(uint64_t p=0;p<24;++p) emit(p,rec.points3D.at(p%12).xyz);});
    auto conflictOptions=options; conflictOptions.accepted_raw=fullQuality(catalog,conflict);
    const auto conflicted=regional::exportVerifiedScene(catalog,conflict,paths,root/"conflict-stage",root/"conflict",conflictOptions,accepted);
    check(!conflicted.verified && !conflicted.reader_verified && !conflicted.failures.empty() &&
        conflicted.failures.front().reason.find("conflicting canonical")!=std::string::npos,
        "one feature/two canonical identities was silently merged or hidden");
    rejects([&]{scene.pointXYZ(12);},"source alias had a second readable XYZ authority");
    scene.beginCandidate(); rejects([&]{scene.applyPointUpdates({{12,{0,0,5}}});},"source alias accepted an independent XYZ write"); scene.abort("alias-write-negative-fixture");
    auto few=rec; for(auto at=few.points3D.begin();at!=few.points3D.end();) {
        if(at->first>=4) {for(const auto& row:at->second.track) few.images.at(row.image_id).point3D_ids.at(row.point2D_idx)=kInvalidPoint3D; at=few.points3D.erase(at);} else {
            at->second.track.clear(); for(uint32_t image=0;image<3;++image) for(int repeat=0;repeat<3;++repeat) at->second.track.push_back({image,uint32_t(at->first)}); ++at;
        }
    }
    const auto fewFile=root/"few-unique-source.bin",fewSecond=root/"few-unique-source-b.bin";
    model_store_detail::writeModel(few,fewFile); model_store_detail::writeModel(few,fewSecond);
    Store fewCatalog(root/"few-unique-catalog",64u<<20); fewCatalog.importModel(fewFile,few,"few-a"); fewCatalog.importModel(fewSecond,few,"few-b");
    Scene fewScene(root/"few-unique-scene",64u<<20); const auto fewBinding=Scene::bindingFor(fewCatalog,"few-topology");
    fewScene.initialize(fewBinding,fewCatalog.poses(),[&](const Scene::AliasEmit& emit){for(size_t s=0;s<2;++s) for(uint64_t p=0;p<4;++p) emit(fewCatalog.key(s,p),p);},
        [&](const Scene::PointEmit& emit){for(const auto& item:few.points3D) emit(item.first,item.second.xyz);});
    auto fewOptions=options; fewOptions.accepted_raw=fullQuality(fewCatalog,fewScene);
    const auto fewResult=regional::exportVerifiedScene(fewCatalog,fewScene,paths,root/"few-unique-stage",root/"few-unique",fewOptions,accepted);
    check(fewResult.reader_verified && !fewResult.verified && fewResult.raw.observations==72 && fewResult.stats.min_photo_observations==4,
        "repeated raw rows inflated unique visual minimum into a false acceptance");
    auto proveModel=[&](const std::string& label,const Reconstruction& model,bool expectedPass,bool expectedReader) {
        const auto directory=root/label;fs::create_directories(directory/"features");const auto file=directory/"original.bin";
        model_store_detail::writeModel(model,file);Store store(directory/"catalog",64u<<20);store.importModel(file,model,label);
        Scene state(directory/"scene",64u<<20);state.initialize(Scene::bindingFor(store,label+"-topology"),store.poses(),
            [&](const Scene::AliasEmit& emit){for(uint64_t p=0;p<store.sourcePointCount(0);++p)emit(store.key(0,p),p);},
            [&](const Scene::PointEmit& emit){uint64_t p=0;for(const auto& point:model.points3D)emit(p++,point.second.xyz);});
        auto opt=options;opt.inputs.clear();opt.accepted_raw=fullQuality(store,state);
        for(const auto& image:model.images){FeatureSet f;f.width=800;f.height=600;f.keypoints.resize(image.second.points2D.size());f.descriptors.resize(f.count()*f.dim);
            for(size_t k=0;k<f.count();++k){f.keypoints[k].x=float(image.second.points2D[k].x);f.keypoints[k].y=float(image.second.points2D[k].y);}
            writeFeatures((directory/"features"/(image.second.name+".bin")).u8string(),f);
            opt.inputs[image.first]={options.input_path(image.first),image.second.name};}
        auto exported=regional::exportVerifiedScene(store,state,[&](uint32_t id){return directory/"features"/(model.images.at(id).name+".bin");},
            directory/"stage",directory/"output",opt,[&](const Reconstruction& actual,const regional::Quality&,const std::map<uint32_t,regional::PhotoQuality>&,const regional::ExportStats&){return groundTruthProof(actual,model,opt);});
        check(exported.verified==expectedPass && exported.reader_verified==expectedReader,"distinct feature/view/landmark contract fixture verdict differs");return exported;
    };
    auto multi=rec;multi.images.at(0).points2D.push_back(multi.images.at(0).points2D.at(0));multi.images.at(0).point3D_ids.push_back(0);multi.points3D.at(0).track.push_back({0,12});
    const auto multiResult=proveModel("same-image-distinct-features",multi,true,true);
    check(multiResult.raw.observations==49 && multiResult.stats.observations==37 && multiResult.unique_photos.at(0)==13 && multiResult.unique_landmarks.at(0)==12,
        "valid same-image feature aliases were dropped or inflated landmark evidence");
    auto onePoint=rec;onePoint.points3D.clear();Point3D only=rec.points3D.at(0);only.track.clear();
    for(auto& image:onePoint.images){for(size_t k=0;k<12;++k){image.second.points2D[k]=image.second.points2D[0];image.second.point3D_ids[k]=0;only.track.push_back({image.first,uint32_t(k)});}}
    onePoint.points3D[0]=only;
    const auto onePointResult=proveModel("one-XYZ-twelve-features",onePoint,false,true);
    check(onePointResult.unique_photos.at(0)==12 && onePointResult.unique_landmarks.at(0)==1 && onePointResult.raw.observations==36,"twelve features at one XYZ inflated visual landmark evidence");
    auto oneView=onePoint;oneView.images.erase(1);oneView.images.erase(2);auto& oneTrack=oneView.points3D.at(0).track;
    oneTrack.erase(std::remove_if(oneTrack.begin(),oneTrack.end(),[](const auto& row){return row.image_id!=0;}),oneTrack.end());
    const auto oneViewResult=proveModel("one-view-twelve-features",oneView,false,false);
    check(oneViewResult.failures.size()<10 && oneViewResult.lineage_rows==12,"one-view degeneracy was accepted or its repeated report grew with raw rows");
    auto bridge=rec;bridge.points3D.clear();bridge.images[3]=bridge.images.at(2);bridge.images[3].id=3;bridge.images[3].name="photo-3.jpg";bridge.images[3].pose.t.x=-1.5;
    for(auto& image:bridge.images){image.second.points2D.resize(12);image.second.point3D_ids.assign(12,kInvalidPoint3D);}
    for(uint64_t p=0;p<24;++p){Point3D point;point.xyz=rec.points3D.at(p%12).xyz+Vec3{p>=12?2.:0.,0,0};
        for(uint32_t id:{uint32_t(p<12?0:2),uint32_t(p<12?1:3)}){const auto xy=bridge.cameras.at(0).project(point.xyz+bridge.images.at(id).pose.t);
            bridge.images.at(id).points2D.at(p%12)={double(float(xy.x)),double(float(xy.y))};bridge.images.at(id).point3D_ids.at(p%12)=p;point.track.push_back({id,uint32_t(p%12)});}
        bridge.points3D[p]=point;}
    Point3D connector;connector.xyz={.1,.1,6};for(uint32_t id:{1u,2u}){auto xy=bridge.cameras.at(0).project(connector.xyz+bridge.images.at(id).pose.t);if(id==2)xy.x+=100;
        bridge.images.at(id).points2D.push_back({double(float(xy.x)),double(float(xy.y))});bridge.images.at(id).point3D_ids.push_back(24);connector.track.push_back({id,12});}
    bridge.points3D[24]=connector;
    // Its new fourth photo is an actual fixture identity; no extension guessing.
    const auto oldInputPath=options.input_path; options.input_path=[&](uint32_t id){return id==3?(root/"inputs"/"photo-3.jpg").u8string():oldInputPath(id);};
    const auto bridgeResult=proveModel("outlier-only-visual-bridge",bridge,false,true);options.input_path=oldInputPath;
    check(bridgeResult.stats.photo_components==2 && bridgeResult.raw.observations==50 && bridgeResult.unique_landmarks.at(2)==12,
        "an outlier-only raw track was accepted as genuine visual connectivity");
    std::cout<<"DISTINCT_FEATURE_PAIR native_raw=49 unique_features=37 image0_landmarks=12 accepted=true one_XYZ_twelve_features_rejected=true one_view_twelve_features_rejected=true\n";
    for(const auto type:{CamModel::OpenCV,CamModel::Equirect}) {
        const auto dir=root/(type==CamModel::OpenCV?"opencv-pixel-scale-2":"equirect-seam-pixel-scale-half"); fs::create_directories(dir/"features");
        auto model=rec; auto& camera=model.cameras.at(0); camera=Camera::defaultFor(0,800,600,500,type);
        camera.pixel_scale=type==CamModel::OpenCV?2:.5;
        if(type==CamModel::OpenCV) { camera.k1=.015; camera.k2=-.0003; camera.p1=.0002; camera.p2=-.0001; camera.fx=520; camera.fy=480; }
        for(auto& item:model.images) {
            const double angle=type==CamModel::OpenCV?.2:.0004;
            item.second.pose.R={std::cos(angle),0,std::sin(angle),0,1,0,-std::sin(angle),0,std::cos(angle)};
            item.second.pose.t={type==CamModel::OpenCV?-.5*item.first:-.001*item.first,0,0};
        }
        for(auto& item:model.points3D) {
            if(type==CamModel::Equirect) item.second.xyz={(item.first%2?.002:-.002)+.0001*item.first,.03,-5-.02*item.first};
            for(auto& image:model.images) { const auto xy=camera.project(mul(image.second.pose.R,item.second.xyz)+image.second.pose.t);
                image.second.points2D.at(item.first)={double(float(xy.x)),double(float(xy.y))}; }
        }
        for(const auto& image:model.images) {
            FeatureSet f; f.width=800; f.height=600; f.keypoints.resize(12); f.descriptors.resize(12*f.dim);
            f.extract_width=int(f.width/camera.pixel_scale); f.extract_height=int(f.height/camera.pixel_scale);
            for(size_t k=0;k<12;++k) {f.keypoints[k].x=float(image.second.points2D[k].x); f.keypoints[k].y=float(image.second.points2D[k].y);}
            writeFeatures((dir/"features"/(image.second.name+".bin")).u8string(),f);
        }
        const auto file=dir/"source.bin"; model_store_detail::writeModel(model,file);
        Store codecCatalog(dir/"catalog",64u<<20); codecCatalog.importModel(file,model,"codec");
        Scene codecScene(dir/"scene",64u<<20); const auto codecBinding=Scene::bindingFor(codecCatalog,"codec-topology");
        codecScene.initialize(codecBinding,codecCatalog.poses(),[&](const Scene::AliasEmit& emit){for(uint64_t p=0;p<12;++p) emit(codecCatalog.key(0,p),p);},
            [&](const Scene::PointEmit& emit){for(const auto& item:model.points3D) emit(item.first,item.second.xyz);});
        auto codecOptions=options; codecOptions.accepted_raw=fullQuality(codecCatalog,codecScene);
        codecOptions.input_path=[&](uint32_t id){return (dir/"inputs"/model.images.at(id).name).u8string();};
        codecOptions.inputs.clear(); for(const auto& image:model.images) codecOptions.inputs[image.first]={codecOptions.input_path(image.first),image.second.name};
        const auto codecResult=regional::exportVerifiedScene(codecCatalog,codecScene,[&](uint32_t id){return dir/"features"/(model.images.at(id).name+".bin");},
            dir/"stage",dir/"output",codecOptions,[&](const Reconstruction& actual,const regional::Quality&,const std::map<uint32_t,regional::PhotoQuality>&,const regional::ExportStats&){return groundTruthProof(actual,model,codecOptions);});
        for(const auto& failed:codecResult.failures) std::cerr<<"CODEC_FAIL "<<failed.reason<<'\n';
        check(codecResult.verified && codecResult.reader_verified && codecResult.raw.observations==48 && std::abs(codecResult.raw.cost-codecOptions.accepted_raw.cost)<1e-8,
            "nonunit pixel scale/nonidentity rotation/distorted or spherical native codec changed physical residuals");
        std::cout<<"CAMERA_CODEC type="<<camColmapId(type)<<" pixel_scale="<<camera.pixel_scale<<" nonidentity_rotation=true raw=48 native_cost="<<codecResult.raw.cost<<" equivalent=true\n";
    }
    fs::create_directories(root/"corrupt-formal");
    for(const auto name:{"cameras.bin","images.bin","points3D.bin"}) fs::copy_file(root/"export"/name,root/"corrupt-formal"/name);
    { std::fstream output(root/"corrupt-formal"/"points3D.bin",std::ios::in|std::ios::out|std::ios::binary); output.seekp(8+51); uint32_t wrong=999; output.write(reinterpret_cast<const char*>(&wrong),4); }
    rejects([&]{ regional::verified_export_detail::strictReadBack(root/"corrupt-formal",64u<<20); },"native parser auto-repair hid a corrupted formal track");
    fs::create_directories(root/"duplicate-pair-formal");for(const auto name:{"cameras.bin","images.bin","points3D.bin"})fs::copy_file(root/"export"/name,root/"duplicate-pair-formal"/name);
    {std::fstream output(root/"duplicate-pair-formal"/"points3D.bin",std::ios::in|std::ios::out|std::ios::binary);output.seekp(67);uint32_t zero=0;output.write(reinterpret_cast<const char*>(&zero),4);output.write(reinterpret_cast<const char*>(&zero),4);}
    rejects([&]{regional::verified_export_detail::strictReadBack(root/"duplicate-pair-formal",64u<<20);},"duplicate exact image/feature pair bypassed strict reciprocal Reader");
    fs::create_directories(root/"truncated-formal");
    for(const auto name:{"cameras.bin","images.bin","points3D.bin"}) fs::copy_file(root/"export"/name,root/"truncated-formal"/name);
    fs::resize_file(root/"truncated-formal"/"points3D.bin",fs::file_size(root/"truncated-formal"/"points3D.bin")-1);
    rejects([&]{regional::verified_export_detail::strictReadBack(root/"truncated-formal",64u<<20);},"truncated formal file accepted or silently repaired");
    check(spirula::sha256_file(source.u8string())==sourceSHA,"original immutable source changed");
    std::cout<<"VERIFIED_SCENE_EXPORT raw=96 unique=36 canonical=12 actual_COLMAP_native_reader=true XYZ_unchanged=true unknown_gate_closed=true GPS_failure_isolated=true source_SHA_unchanged=true\n";
    if(argc>2) {
        const auto weakPath=fs::u8path(argv[2]); const auto weakSHA=spirula::sha256_file(weakPath.u8string());
        const auto weak=model_store_detail::readModel(weakPath); const auto weakRaw=countObservations(weak);
        auto same=weak; check(regional::verifyOriginalObservations(weak,same)==weakRaw,"actual weak raw source self multiset failed");
        const auto weakMultiset=regional::raw_lineage::originalMultiset(weak,same);
        check(regional::preserveRepairObservations(weak,same,3)==weakRaw && countObservations(same)==weakRaw,"actual weak raw preservation failed");
        bool removed=false; for(auto& item:same.points3D) { std::set<uint64_t> seen;
            for(auto at=item.second.track.begin();at!=item.second.track.end();++at) if(!seen.insert(regional::raw_lineage::featureKey(*at)).second) { item.second.track.erase(at); removed=true; break; }
            if(removed) break;
        }
        if(removed) {
            rejects([&]{regional::verifyOriginalObservations(weak,same);},"actual weak source duplicate 2->1 went undetected");
            check(regional::preserveRepairObservations(weak,same,3)==weakRaw && countObservations(same)==weakRaw,"actual weak source duplicate could not be restored");
            model_store_detail::writeModel(same,root/"real-weak-restored.bin");
            check(regional::verifyOriginalObservations(weak,model_store_detail::readModel(root/"real-weak-restored.bin"))==weakRaw,"real weak restored multiset failed native Reader");
        }
        auto derivedOriginal=weak; derivedOriginal.points3D.begin()->second.track.push_back(derivedOriginal.points3D.begin()->second.track.front());
        auto derivedCurrent=weak;
        rejects([&]{regional::verifyOriginalObservations(derivedOriginal,derivedCurrent);},"real-geometry derived duplicate multiplicity 2->1 went undetected");
        check(regional::preserveRepairObservations(derivedOriginal,derivedCurrent,3)==weakRaw+1 && countObservations(derivedCurrent)==weakRaw+1,
            "real-geometry derived duplicate row was not preserved");
        model_store_detail::writeModel(derivedCurrent,root/"real-geometry-derived-duplicate.bin");
        check(regional::verifyOriginalObservations(derivedOriginal,model_store_detail::readModel(root/"real-geometry-derived-duplicate.bin"))==weakRaw+1,
            "real-geometry derived duplicate native Reader proof failed");
        check(spirula::sha256_file(weakPath.u8string())==weakSHA,"actual weak original source SHA changed");
        std::cout<<"REAL_WEAK_SOURCE raw="<<weakRaw<<" unique_feature_keys="<<weakMultiset.required.size()<<" native_multiset_verified=true duplicate_damage_rejected_and_restored="<<removed<<" original_SHA_unchanged=true\n";
        std::cout<<"DERIVED_REAL_GEOMETRY_DUPLICATE_FIXTURE augmented_raw="<<weakRaw+1<<" damaged_raw="<<weakRaw
            <<" verify_rejected=true preserved_return="<<weakRaw+1<<" native_actual_rows="<<weakRaw+1<<" original_production_source_modified=false\n";
    }
    return 0;
}
int main(int argc,char** argv){return sfmTestMain(argc,argv,test);}
