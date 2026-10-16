#include "sfm/Pipeline.h"
#include "sfm/map/CanonicalResume.h"
#include "sfm/map/CanonicalInitialization.h"
#include "sfm/map/CanonicalImageIdentity.h"
#include "sfm/map/CanonicalRecovery.h"
#include "sfm/map/RegionalLandmarks.h"
#include "sfm/map/RegionalRecovery.h"
#include "sfm/map/RegionalBootstrap.h"
#include "sfm/map/CanonicalAcceptance.h"
#include "sfm/map/SceneVisualEvidence.h"
#include "sfm/map/CanonicalOriginalCoverage.h"
#include "sfm/map/CanonicalPreview.h"
#include "sfm/tests/TestMain.h"
#include <fstream>
#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>
using namespace sfm;
namespace fs=std::filesystem;
static void check(bool value,const std::string& why){if(!value)throw std::runtime_error(why);}
static std::string sha(const fs::path& path){auto h=spirula::sha256_file(path.u8string());check(!h.empty(),"SHA unavailable: "+path.u8string());return h;}
static std::string read(const fs::path& path){std::ifstream in(path,std::ios::binary);check(bool(in),"input missing: "+path.u8string());return {std::istreambuf_iterator<char>(in),{}};}
static void write(const fs::path& path,const std::string& value){fs::create_directories(path.parent_path());std::ofstream out(path,std::ios::binary);out<<value;out.flush();check(bool(out),"proof write failed: "+path.u8string());}
template<class T> static void flag(std::vector<std::string>& args,const char* name,const T& initial,const std::string& value,const char* choices){
    if((std::string(name)=="focal"&&value=="0")||(std::string(name)=="distortion"&&value=="none"))return;
    if constexpr(std::is_same_v<T,bool>){check(value=="on"||value=="off","invalid signed bool");args.push_back(std::string(value=="on"?"--":"--no-")+name);}
    else {args.push_back(std::string("--")+name);if constexpr(std::is_same_v<T,std::string>)args.push_back(value=="none"&&initial.empty()&&(!choices||!*choices)?"":value);else args.push_back(value);}
}
struct Replay {
    fs::path original,previous,dataset,photos,proof,work,root,v1,v2;
    SfmConfig cfg;std::string signature;std::vector<std::string> args;
};
static Replay configuration(int argc,char** argv){
    check(argc>=7,"usage: --config|--prepare|--resume-boundary|--stage-recovered|--complete|--complete-natural originalRoot previousCoord dataset photos newProofRoot [recovered0 recovered1]");
    Replay r;r.original=fs::u8path(argv[2]);r.previous=fs::u8path(argv[3]);r.dataset=fs::u8path(argv[4]);r.photos=fs::u8path(argv[5]);r.proof=fs::u8path(argv[6]);r.work=r.proof/"regional-workspaces";
    const auto recorded=read(r.original/"signature.txt");std::map<std::string,std::string> values;std::vector<std::pair<std::string,std::string>> overrides;
    std::istringstream input(recorded);std::string line;bool autoStage=false;
    while(std::getline(input,line)){const auto at=line.find('=');if(at==std::string::npos)continue;const auto key=line.substr(0,at),value=line.substr(at+1);
        if(key=="quality")autoStage=true;
        if(autoStage&&key=="matches")break;
        if(key.rfind("override ",0)==0){check(!autoStage,"camera override outside signed map stage");overrides.emplace_back(key.substr(9),value);continue;}
        if(!autoStage)continue;
        const auto added=values.emplace(key,value);check(added.second||added.first->second==value,"inconsistent signed auto setting: "+key);
    }
    check(autoStage,"signed automatic pipeline stage missing");
    SfmConfig defaults;r.args={r.photos.u8string(),"--output",(r.proof/"dataset").u8string()};std::set<std::string> flags;
#define REPLAY_SETTING(member,name,cmds,tier,group,lo,hi,choices,help) \
    if((uint32_t(cmds)&CMD_AUTO)&&(tier)!=Tier::Alias&&values.count(name)&&flags.insert(name).second)flag(r.args,name,defaults.member,values.at(name),choices);
    SFM_CONFIG_FIELDS(REPLAY_SETTING)
#undef REPLAY_SETTING
    for(const auto& item:overrides){check(item.second.find_first_of(",;")==std::string::npos,"replay requires explicit native extended camera parameter handling");r.args.push_back("--camera-model");r.args.push_back(item.first+"="+item.second);}
    r.args.insert(r.args.end(),{"--threads","0","--regional-work-dir",r.work.u8string(),"--reuse-features","require","--reuse-matches","require"});
    AutoRequest request;const auto error=parse_auto_args(r.args,request);check(error.empty(),"official argument parser: "+error);r.cfg=request.cfg;
    r.signature="regional-v2\n"+regional::regionalBaseSignature(stageSignature(r.cfg,CMD_MAP))+regional::regionalBaseSignature(stageSignature(r.cfg,CMD_AUTO))+
        "matches="+fs::absolute(r.dataset/"matches.bin").string()+"\nbytes="+std::to_string(fs::file_size(r.dataset/"matches.bin"))+
        "\ntime="+std::to_string(fs::last_write_time(r.dataset/"matches.bin").time_since_epoch().count())+"\nfeatures="+regionalInputDigest(r.dataset/"features")+"\n";
    check(resume::equivalentSignatures(recorded,r.signature),"actual official config/path/feature identity differs from original signature; no fork or mapping launched");
    r.root=r.work/r.original.filename();const auto id=std::to_string(regional::signatureHash(regional::regionalBaseSignature(r.signature)));r.v1=r.root/("canonical-native-v1-"+id);r.v2=r.root/("canonical-native-v2-"+id);
    return r;
}
static void recordConfiguration(const Replay& r){write(r.proof/"actual-native-signature.txt",r.signature);std::ostringstream out;for(const auto& a:r.args)out<<a<<'\n';write(r.proof/"official-arguments.txt",out.str());}
static void prepare(const Replay& r){
    check(!fs::exists(r.proof),"new disjoint proof directory required");
    const auto original=fs::canonical(r.original),target=fs::weakly_canonical(r.proof);for(auto p=target;!p.empty();){check(p!=original,"replay destination inside original checkpoint");auto parent=p.parent_path();if(parent==p)break;p=parent;}
    fs::create_directories(r.root);storage::requireSpace(r.root,12ull<<30);recordConfiguration(r);
    for(const auto* name:{"plan.bin","signature.txt"})fs::copy_file(r.original/name,r.root/name);
    std::ostringstream baseline;baseline<<"previous_head_sha="<<sha(r.previous/"scene/head.bin")<<"\nprevious_manifest_sha="<<sha(r.previous/"immutable-raw/manifest.bin")<<'\n';
    for(const auto& entry:fs::directory_iterator(r.original)){if(!entry.is_regular_file())continue;const auto name=entry.path().filename().u8string();if(name.rfind("pass-0-region-",0)!=0)continue;
        if(entry.path().extension()==".bin"){storage::retainFile(entry.path(),r.root/entry.path().filename());check(fs::equivalent(entry.path(),r.root/entry.path().filename()),"same-volume model retention copied instead of hardlinked");baseline<<"original_model="<<entry.path().u8string()<<'|'<<sha(entry.path())<<'\n';}
        else if(entry.path().extension()==".done"||entry.path().extension()==".origins"||entry.path().extension()==".costs")fs::copy_file(entry.path(),r.root/entry.path().filename());
    }
    fs::create_directory(r.root/"source-evidence-v1");for(const auto& entry:fs::directory_iterator(r.original/"source-evidence-v1")){check(entry.is_regular_file(),"original source evidence is not flat");if(entry.path().extension()==".bin"){storage::retainFile(entry.path(),r.root/"source-evidence-v1"/entry.path().filename());check(fs::equivalent(entry.path(),r.root/"source-evidence-v1"/entry.path().filename()),"original source evidence must remain same-volume hardlinks");}else fs::copy_file(entry.path(),r.root/"source-evidence-v1"/entry.path().filename());}
    check(regional::forkCanonicalResume(r.previous,r.v1,1u<<30),"native accepted-prefix fork unavailable");
    write(r.proof/"baseline.txt",baseline.str());std::cout<<"PREPARED official Regional.cpp isolated checkpoint "<<r.root.u8string()<<"; features/matches/photos remain original; full_capture_verified=false\n";
}
struct BoundaryReached:std::exception{const char* what()const noexcept override{return "intentional resume/noappend audit boundary";}};
static void boundary(const Replay& r){
    check(fs::exists(r.proof/"baseline.txt"),"native checkpoint preparation missing");bool reached=false;
    {RunContext context;context.set_log([&](slog::Tag,slog::Level,const std::string& text){std::cout<<text;std::cout.flush();if(text.find("[canonical-scene] accepted-scene-audit:")!=std::string::npos)throw BoundaryReached();});
        try{const auto result=run_regional(r.cfg,(r.dataset/"matches.bin").u8string(),(r.dataset/"features").u8string(),r.photos.u8string(),r.proof/"sparse");check(bool(result),"formal regional branch not selected");}
        catch(const BoundaryReached&){reached=true;}
    }
    check(reached,"formal noappend boundary was not reached");regional::CanonicalScene scene(r.v2/"scene",1u<<30,regional::CanonicalScene::OpenMode::ReadOnly);
    regional::CanonicalScene original(r.previous/"scene",1u<<30,regional::CanonicalScene::OpenMode::ReadOnly);
    check(scene.identity()==original.identity()&&scene.binding()==original.binding()&&scene.generation()==8&&scene.sourceCount()==4&&scene.observations()==34207004,"formal noappend changed accepted geometry/source scope");
    check(regional::canonicalCompletedRounds(r.v2,scene)==8,"formal noappend lost accepted round ordinal");
    write(r.proof/"formal-resume-boundary.txt","formal_Regional_cpp_boundary_reached=true\naccepted_identity="+scene.identity()+"\ntopology="+scene.binding().topology+"\nraw=34207004\nsource_count=4\naccepted_rounds=8\nfull_capture_verified=false\nalgorithm_completed=false\n");
    std::cout<<"PASS FORMAL RESUME/NOAPPEND boundary; this is not complete reconstruction acceptance\n";
}
static void stageRecovered(const Replay& r,const fs::path& first,const fs::path& second){
    check(fs::exists(r.proof/"formal-resume-boundary.txt"),"formal resume/noappend proof is required before admission");
    regional::CanonicalScene scene(r.v2/"scene",1u<<30,regional::CanonicalScene::OpenMode::ReadOnly);regional::RegionalWorkingStore catalog(r.v2/"immutable-raw",1u<<30,regional::RegionalWorkingStore::OpenMode::ReadOnly);
    regional::ImmutableSceneTopology topology(r.v2/"indices",1u<<30);topology.restore(scene.binding().topology);auto existing=catalog.poses();
    MatchesIndex index;check(indexMatches((r.dataset/"matches.bin").u8string(),index)&&index.complete&&index.images.size()==10650,"original cached input scope differs");
    std::vector<Reconstruction> names(1);for(uint32_t id=0;id<index.images.size();++id){Image image;image.id=id;image.name=index.images[id].name;names[0].images.emplace(id,image);}resolveImageNames(names,r.photos.u8string());
    struct Name{std::string name,export_name;};std::map<uint32_t,Name> inputs;for(const auto& row:names[0].images){check(fs::is_regular_file(r.photos/fs::u8path(row.second.name)),"actual original photo name mapping missing");inputs.emplace(row.first,Name{index.images[row.first].name,row.second.name});}
    const std::set<uint32_t> targets{327,328,329,330,331,332,2399,2400,2401,4900,4901,4902,4903,4904,4905,4906,4907,4908,4909,4910,4911,8642};
    regional::CanonicalInitialStage proposal;proposal.poses=scene.poses();proposal.targets.assign(targets.begin(),targets.end());
    const std::array<fs::path,2> sources{first,second};std::ostringstream proof;
    for(size_t region=0;region<sources.size();++region){const auto originalSHA=sha(sources[region]);auto model=model_store_detail::readModel(sources[region]);regional::canonicalizeImageIdentity(model,inputs);
        const auto corrected=regional::restoreAcceptedFeatureBindings(topology,scene,model,256u<<20);check(!corrected.reassociated_features,"prepared recovered source still has unverified old-root reassociation");
        const auto stem=r.root/("recovered-v3-region-"+std::to_string(region));const auto original=fs::path(stem.u8string()+"-0.bin");check(!fs::exists(original),"recovered stage already exists");model_store_detail::writeModel(model,original);
        model_store_detail::Writer done(stem.u8string()+".done");done.scalar<uint64_t>(1);done.scalar<uint64_t>(fs::file_size(original));done.scalar<uint64_t>(modelResidentBytes(model));done.finish();write(stem.u8string()+".origins",original.filename().u8string()+"\n");
        write(original.u8string()+".fixed-frame.txt","held-canonical-frame-v1\n"+scene.identity()+"\n"+sha(original)+"\n");
        for(const auto& image:model.images)if(image.second.registered&&(!proposal.poses.images.count(image.first)||!proposal.poses.images.at(image.first).registered||targets.count(image.first))){auto copy=image.second;copy.points2D.clear();copy.point3D_ids.clear();proposal.poses.images[image.first]=std::move(copy);}
        auto immutable=model;regional::canonicalizeSourceImageIdentity(immutable,inputs,existing);const auto aligned=regional::canonical_initial_detail::aligned(r.v2/"aligned-inputs",immutable);
        proposal.sources.push_back({original.filename().u8string(),original,aligned,modelResidentBytes(immutable),Sim3{}});check(sha(sources[region])==originalSHA,"original recovered input changed");proof<<"input="<<sources[region].u8string()<<'|'<<originalSHA<<"\nowned_fixed_frame="<<original.u8string()<<'|'<<sha(original)<<"\n";
    }
    regional::canonicalizeImageIdentity(proposal.poses,inputs);const auto signature=scene.identity()+"."+regional::regionalBaseSignature(r.signature);
    regional::saveCanonicalInitialStage(r.v2/("admission-"+scene.identity()),signature,proposal);proof<<"explicit_recovered_targets=22\nold_root_bindings_exact=true\nfull_capture_verified=false\n";write(r.proof/"formal-admission-inputs.txt",proof.str());
    std::cout<<"PREPARED genuine fixed-frame recovered22 admission for formal Regional.cpp; no candidate accepted\n";
}
static void auditAccepted(Replay r) {
    using namespace regional;
    check(!fs::exists(r.proof), "accepted audit output must be a new isolated directory");
    const auto target = fs::weakly_canonical(fs::absolute(r.proof));
    for (const auto& source : {r.original, r.previous, r.dataset, r.photos}) {
        const auto protectedRoot = fs::canonical(source);
        for (auto parent = target; !parent.empty();) {
            check(parent != protectedRoot, "accepted audit output must be outside all original input/checkpoint trees");
            const auto next = parent.parent_path(); if (next == parent) break; parent = next;
        }
    }
    fs::create_directories(r.proof);
    RunContext context; context.set_progress_dir((r.proof / "progress").u8string());
    context.set_log([](slog::Tag, slog::Level, const std::string& value) { std::cout << value; std::cout.flush(); });
    configureMappingMemory(r.cfg, r.proof.u8string()); const auto budget = r.cfg.bup.model_memory_bytes;
    const auto head = sha(r.previous / "scene/head.bin"), manifest = sha(r.previous / "immutable-raw/manifest.bin"), matchesSHA = sha(r.dataset / "matches.bin");
    CanonicalScene scene(r.previous / "scene", budget / 4, CanonicalScene::OpenMode::ReadOnly);
    RegionalWorkingStore catalog(r.previous / "immutable-raw", budget / 4, RegionalWorkingStore::OpenMode::ReadOnly);
    ImmutableSceneTopology topology(r.previous / "indices", budget / 4); topology.restore(scene.binding().topology);
    scene.requireBinding(CanonicalScene::bindingFor(catalog, topology.identity(), true, scene.sourceCount()));
    MatchesIndex index; check(indexMatches((r.dataset / "matches.bin").u8string(), index) && index.complete && index.images.size() == 10650, "accepted audit requires the actual complete10650 Gaoxin input");
    CameraSetup cameras; loadCameraSetup(index.metadata, cameras); auto sensors = loadSensorCaptures(r.cfg, !r.cfg.quiet);
    std::unique_ptr<PriorSource> gps = makeSensorPriors(r.cfg, sensors, index.metadata, cameras.ids);
    if (!gps) { r.cfg.sensor_pairs = true; gps = makeExifGpsPriors(r.cfg, r.photos.u8string(), index.metadata, cameras, !r.cfg.quiet); }
    check(bool(gps), "accepted audit has no original GPS source");
    std::vector<Reconstruction> names(1); Reconstruction attitudeNames;
    for (uint32_t id = 0; id < index.images.size(); ++id) { Image im; im.id = id; im.name = index.images[id].name; names[0].images.emplace(id, im); im.registered = true; attitudeNames.images.emplace(id, std::move(im)); }
    resolveImageNames(names, r.photos.u8string()); std::map<uint32_t, ExportInputPhoto> inputs; std::set<std::string> paths;
    for (const auto& item : names[0].images) {
        const auto path = fs::canonical(r.photos / fs::u8path(item.second.name)); check(paths.insert(path.u8string()).second, "accepted audit duplicated an original photo");
        inputs.emplace(item.first, ExportInputPhoto{path.u8string(), index.images[item.first].name, item.second.name});
    }
    const auto attitude = regionalRecordedAttitude(attitudeNames, r.photos.u8string(), r.cfg.exif_orientation == "apply");
    const auto plan = readRegionalPlan(r.original, index.images.size());
    const auto initial = loadCanonicalInitialStage(r.previous / "initialization", regionalBaseSignature(read(r.original / "signature.txt")), {}, budget / 4, true);
    check(bool(initial), "accepted audit has no verified original initialization");
    const auto required = canonicalOriginalRequirements(*initial, r.original / "source-evidence-v1", read(r.original / "source-evidence-v1/inputs.txt"));
    const auto originals = verifyCanonicalOriginalCoverage(catalog, scene, required, [&](auto& rec) { canonicalizeImageIdentity(rec, inputs); }, budget / 4);
    write(r.proof / "original-observation-coverage.csv", originals.csv()); check(originals.complete, "accepted original multiset completeness failed");
    std::vector<RegionalSourceFrame> sources;
    for (const auto& requiredSource : required) {
        auto views = model_store_detail::readModel(requiredSource.path, false);
        const bool positioned = regionalGpsQuality(views, *gps).accepted(), certified = regionalGpsGauge(views, *gps, attitude);
        if (positioned || certified) sources.push_back({requiredSource.label, std::move(views), positioned, certified});
    }
    auto reportProgress = [&](const char* phase, uint64_t done, uint64_t total) { std::cout << "AUDIT " << phase << ' ' << done << '/' << total << '\n'; std::cout.flush(); };
    const auto expected = sceneQuality(topology, scene, r.cfg.max_error, budget / 2, 0, nullptr, reportProgress);
    progress::model(canonicalPreview(scene, progress::kMaxPoints), true);
    std::vector<uint32_t> owners(index.images.size(), UINT32_MAX);
    for (uint32_t owner = 0; owner < plan.size(); ++owner) for (uint32_t id : plan[owner].core) owners.at(id) = owner;
    VerifiedExportOptions options; options.cache_bytes = 128u << 20; options.reader_memory_bytes = budget / 2;
    options.max_error = r.cfg.max_error; options.accepted_raw = expected; options.inputs = inputs; options.owners = &owners; options.threads = 0; options.reportphase = reportProgress;
    options.phase_bytes = [&](const char* phase, uint64_t bytes, uint64_t peak) { reportProgress(phase, bytes, peak); };
    const auto formal = r.proof / "dataset/sparse/0";
    const auto report = exportVerifiedScene(catalog, scene, [&](uint32_t id) { return r.dataset / "features" / fs::u8path(index.images.at(id).name + ".bin"); }, r.proof / "export-scratch", formal, options,
        [&](const Reconstruction& actual, const Quality&, const auto&, const ExportStats&) { return canonicalCaptureAcceptance(actual, inputs, plan, *gps, attitude, sources, {}, false, r.proof); });
    check(report.verified && report.reader_verified && report.stats.images == 10650 && report.stats.photo_components == 1 && report.raw.observations == scene.observations() && !report.raw.invalid_projections && canonicalMetricVerified(inputs, report.constraints), "complete original accepted audit/formal Reader/photo constraints failed");
    ModelGauge gauge; gauge.metric = gauge.oriented = true; gauge.scale = "gps"; gauge.up = "gps+attitude"; writeGauge(formal, gauge);
    check(sha(r.previous / "scene/head.bin") == head && sha(r.previous / "immutable-raw/manifest.bin") == manifest && sha(r.dataset / "matches.bin") == matchesSHA, "accepted audit changed source checkpoint, raw catalogue or matches");
    write(r.proof / "final-audit.txt", "full_capture_verified=true\nimages=10650\nregistered=10650\noriginal_source_multisets_verified=true\nformal_native_reader_verified=true\nsource_checkpoint_unchanged=true\nfeatures_matches_and_geometry_not_recomputed=true\naccepted_identity=" + scene.identity() + "\nformal_sparse=" + formal.u8string() + "\n");
    std::cout << "PASS actual10650 accepted audit, full original multisets, per-photo metric/visual/connectivity and new formal nativeReader; no reconstruction rerun\n";
}
static void complete(const Replay& r,bool natural){
    check(fs::exists(r.proof/(natural?"baseline.txt":"formal-admission-inputs.txt")),"formal native checkpoint/admission inputs missing");recordConfiguration(r);const auto originalHead=sha(r.previous/"scene/head.bin"),originalManifest=sha(r.previous/"immutable-raw/manifest.bin");
    std::ofstream output(r.proof/"formal-entry-complete.txt");output<<std::setprecision(17)<<"entry=sfm::run_regional\nrecovery= "<<(natural?"production_natural":"native_cached_22_admission")<<"\nfull_capture_verified=false\n";output.flush();
    try{RunContext context;context.set_progress_dir((r.proof/"progress").u8string());context.set_log([](slog::Tag,slog::Level,const std::string& text){std::cout<<text;std::cout.flush();});const auto result=run_regional(r.cfg,(r.dataset/"matches.bin").u8string(),(r.dataset/"features").u8string(),r.photos.u8string(),r.proof/"sparse");check(bool(result),"formal regional branch not selected");
        const auto formal=fs::is_regular_file(result->sparse_dir/"0/cameras.bin")?result->sparse_dir/"0":result->sparse_dir;
        output<<"algorithm_returned=true\nexit_code="<<result->exit_code<<"\nregistered="<<result->registered<<"\nimages="<<result->images<<"\nmodels="<<result->models<<"\npartial="<<result->partial<<"\nsparse_root="<<result->sparse_dir.u8string()<<"\nformal_sparse="<<formal.u8string()<<'\n';output.flush();
        const auto coverage=read(r.v2/"coverage.txt");check(result->exit_code==0&&!result->partial&&result->images==10650&&result->registered==10650&&coverage.find("full_capture_verified=true")!=std::string::npos,"formal complete result did not pass all capture gates");
        for(const auto* file:{"cameras.bin","images.bin","points3D.bin","directional-points.bin"})check(fs::is_regular_file(formal/file),"formal typed artifact absent");
        check(sha(r.previous/"scene/head.bin")==originalHead&&sha(r.previous/"immutable-raw/manifest.bin")==originalManifest,"original failure checkpoint changed");output<<"full_capture_verified=true\nactual_formal_export_present=true\noriginal_checkpoint_head_manifest_SHA_unchanged=true\n";output.flush();std::cout<<"PASS COMPLETE FORMAL REGIONAL REPLAY 10650: actual native full capture/Reader/export gates returned true\n";
    }catch(const std::exception& e){output<<"algorithm_failed=true\nerror="<<e.what()<<"\nfull_capture_verified=false\n";output.flush();check(sha(r.previous/"scene/head.bin")==originalHead&&sha(r.previous/"immutable-raw/manifest.bin")==originalManifest,"original checkpoint changed on failed replay");throw;}
}
static void weakSeedDiagnostic(Replay r,const fs::path& weak,const fs::path& output){
    check(!fs::exists(output),"weak seed diagnostic requires new isolated output");fs::create_directories(output);
    std::ofstream proof(output/"held-seed-probe.txt");proof<<std::setprecision(17)<<"scope=one actual weak-source context through official reconstruction\nfull_capture_verified=false\n";proof.flush();
    const auto sourceSHA=sha(weak),headSHA=sha(r.v2/"scene/head.bin");
    RunContext log;log.set_log([](slog::Tag,slog::Level,const std::string& value){std::cout<<value;std::cout.flush();});
    configureMappingMemory(r.cfg,output.u8string());const auto budget=r.cfg.bup.model_memory_bytes;
    regional::CanonicalScene scene(r.v2/"scene",budget/4,regional::CanonicalScene::OpenMode::ReadOnly);
    regional::ImmutableSceneTopology topology(r.v2/"indices",budget/4);topology.restore(scene.binding().topology);
    check(scene.generation()==8&&scene.observations()==34207004,"diagnostic parent differs from actual failing accepted8");
    MatchesIndex index;check(indexMatches((r.dataset/"matches.bin").u8string(),index)&&index.complete,"original matches incomplete");
    std::vector<fs::path> supplements;for(const auto& e:fs::directory_iterator(r.root))if(e.is_directory()&&e.path().filename().u8string().rfind("supplement-v1-",0)==0)supplements.push_back(e.path());
    std::sort(supplements.begin(),supplements.end());for(const auto& p:supplements)regional::loadRepairMatches(index,p);
    CameraSetup cameras;loadCameraSetup(index.metadata,cameras);auto sensors=loadSensorCaptures(r.cfg,!r.cfg.quiet);
    std::unique_ptr<PriorSource> gps=makeSensorPriors(r.cfg,sensors,index.metadata,cameras.ids);if(!gps){r.cfg.sensor_pairs=true;gps=makeExifGpsPriors(r.cfg,r.photos.u8string(),index.metadata,cameras,!r.cfg.quiet);}check(bool(gps),"actual EXIF prior missing");
    std::vector<Reconstruction> names(1);Reconstruction attitudeNames;
    for(uint32_t id=0;id<index.images.size();++id){Image image;image.id=id;image.name=index.images[id].name;names[0].images.emplace(id,image);image.registered=true;attitudeNames.images.emplace(id,std::move(image));}
    resolveImageNames(names,r.photos.u8string());std::map<uint32_t,regional::ExportInputPhoto> inputs;
    for(const auto& p:names[0].images)inputs.emplace(p.first,regional::ExportInputPhoto{(r.photos/fs::u8path(p.second.name)).u8string(),index.images[p.first].name,p.second.name});
    const auto attitude=regional::regionalRecordedAttitude(attitudeNames,r.photos.u8string(),r.cfg.exif_orientation=="apply");
    const auto plan=regional::readRegionalPlan(r.root,index.images.size());const auto capture=regional::canonicalCaptureAcceptance(scene.poses(),inputs,plan,*gps,attitude,{}, {},false,output);
    std::set<uint32_t> recovery;for(const auto& p:capture.photos)if(!p.gps_verified||!p.scale_verified||!p.attitude_verified)recovery.insert(p.image);
    const auto visual=regional::sceneVisualEvidence(topology,scene,r.cfg.max_error,budget/2,r.cfg.threads);
    for(const auto& p:scene.poses().images)if(!regional::photoQualityAccepted(visual.photos.at(p.first),r.cfg.max_error)||visual.landmarks.at(p.first)<12||(visual.components>1&&visual.component.at(p.first)!=visual.largest))recovery.insert(p.first);
    auto original=model_store_detail::readModel(weak);regional::canonicalizeImageIdentity(original,inputs);const auto required=regional::registeredIds(original);
    auto anchors=regional::registeredIds(scene.acceptedPoses());for(auto id:recovery)anchors.erase(id);regional::RepairPairCatalog pairs;
    const auto context=regional::repairContext(index,required,anchors,64,&pairs);std::set<uint32_t> selected;for(auto id:context.images)if(anchors.count(id))selected.insert(id);
    proof<<"parent_identity="<<scene.identity()<<"\nweak_source_sha="<<sourceSHA<<"\nrequired="<<required.size()<<"\ncontext="<<context.images.size()<<"\nanchors="<<selected.size()<<"\ncontext_ids=";for(auto id:context.images)proof<<id<<',';proof<<'\n';proof.flush();
    check(required.size()==33&&context.images.size()==97&&selected.size()==97,"diagnostic context is not the actual failing33/64/97 scope");
    auto seed=regional::gatherCanonicalSeed(topology,scene,selected,[&](uint32_t id){return r.dataset/"features"/fs::u8path(index.images.at(id).name+".bin");},budget/8);
    seed=regional::anchoredRepairSeed(seed,context,scene.acceptedPoses(),r.cfg.max_error,gps.get(),true);check(seed.images.size()==97,"diagnostic actual anchor seed differs");
    auto local=regional::loadRegion(index,context,r.dataset/"matches.bin",r.dataset/"features",budget,&seed);auto compact=regional::compactSeed(seed,local,true);auto roundtrip=compact;regional::restoreFeatureIds(roundtrip,local,index,&seed);
    uint64_t xyChanges=0,assignedChanges=0,lostPoints=0,rawBefore=0,rawAfter=0;
    for(const auto& p:seed.images)for(size_t f=0;f<p.second.points2D.size();++f){const auto& after=roundtrip.images.at(p.first);xyChanges+=p.second.points2D[f].x!=after.points2D[f].x||p.second.points2D[f].y!=after.points2D[f].y;assignedChanges+=p.second.point3D_ids[f]!=kInvalidPoint3D&&p.second.point3D_ids[f]!=after.point3D_ids[f];}
    for(const auto& p:seed.points3D){rawBefore+=p.second.track.size();lostPoints+=!roundtrip.points3D.count(p.first);}for(const auto& p:roundtrip.points3D)rawAfter+=p.second.track.size();
    proof<<"pure_compact_roundtrip_xy_changes="<<xyChanges<<"\nassigned_feature_changes="<<assignedChanges<<"\nlost_seed_points="<<lostPoints<<"\nseed_unique_rows="<<rawBefore<<"\nroundtrip_unique_rows="<<rawAfter<<'\n';
    try{HeldRepairSeedContract(seed).requireUnchanged(roundtrip,"probe.pure-compact-roundtrip");proof<<"pure_roundtrip_held_pass=true\n";}catch(const std::exception& e){proof<<"pure_roundtrip_held_pass=false\npure_roundtrip_error="<<e.what()<<'\n';}proof.flush();
    model_store_detail::writeModel(seed,output/"actual-held-global-seed.bin");model_store_detail::writeModel(compact,output/"actual-held-compact-seed.bin");
    try{auto models=regional::reconstructRepair(r.cfg,index,context,r.dataset/"matches.bin",r.dataset/"features",r.photos.u8string(),cameras,gps.get(),&seed,budget,true);check(!models.empty(),"actual official held reconstruction returned no model");
        for(auto& model:models){regional::canonicalizeImageIdentity(model,inputs);regional::restoreAcceptedFeatureBindings(topology,scene,model,budget/8);HeldRepairSeedContract(seed).requireUnchanged(model,"probe.parent-global-anchor-exit");}
        proof<<"actual_official_held_pass=true\nfull_capture_verified=false\n";proof.flush();
    }catch(const std::exception& e){proof<<"actual_official_held_pass=false\nactual_official_error="<<e.what()<<"\nfull_capture_verified=false\n";proof.flush();check(sha(weak)==sourceSHA&&sha(r.v2/"scene/head.bin")==headSHA,"diagnostic changed readonly original state");throw;}
    check(sha(weak)==sourceSHA&&sha(r.v2/"scene/head.bin")==headSHA,"diagnostic changed readonly original state");
}
static void admissionInvalid(const fs::path& coordination,const fs::path& rejected,const fs::path& output){
    using Catalog=regional::RegionalWorkingStore;using NativeReader=model_store_detail::Reader;
    check(!fs::exists(output),"admission diagnostic requires a new output directory");
    const auto root=fs::canonical(coordination),target=fs::weakly_canonical(fs::absolute(output));
    for(auto p=target;!p.empty();){check(p!=root,"diagnostic output cannot be inside the retained cohort");const auto parent=p.parent_path();if(parent==p)break;p=parent;}
    const auto headHash=sha(root/"scene/head.bin"),catalogHash=sha(root/"immutable-raw/manifest.bin");
    regional::CanonicalScene accepted(root/"scene",256u<<20,regional::CanonicalScene::OpenMode::ReadOnly);
    const auto snapshot=accepted.inspectRetainedSnapshot(rejected);
    check(snapshot.generation==accepted.generation()+1&&snapshot.binding.sources.size()==8&&accepted.sourceCount()==4,
          "diagnostic must inspect the retained 4-to-8-source admission");
    Catalog catalog(root/"immutable-raw",256u<<20,Catalog::OpenMode::ReadOnly);
    check(regional::CanonicalScene::bindingFor(catalog,snapshot.binding.topology,false)==snapshot.binding,"rejected scene and immutable source catalogue differ");
    check(fs::is_regular_file(snapshot.directory/"points.bin"),"diagnostic requires the actual retained full candidate, not a step recipe");
    auto poses=model_store_detail::readModel(snapshot.directory/"poses.bin");
    std::vector<uint64_t> offsets{0};for(const auto& source:snapshot.binding.sources)offsets.push_back(offsets.back()+source.points);
    struct AliasRow{uint64_t root=0,local=0;};static_assert(sizeof(AliasRow)==16);
    NativeReader aliases(snapshot.directory/"aliases.bin",65536),points(snapshot.directory/"points.bin",65536);
    std::map<uint64_t,std::pair<uint64_t,Vec3>> parameters;
    auto parameter=[&](Catalog::PointKey key){const uint64_t slot=offsets.at(size_t(key.source))+key.ordinal;
        const auto found=parameters.find(slot);if(found!=parameters.end())return found->second;
        check(parameters.size()<300000,"incoming parameter diagnostic exceeded its bounded cache");
        aliases.seek(slot*sizeof(AliasRow));const auto alias=aliases.scalar<AliasRow>();
        check(alias.local==key.localPoint&&alias.root<snapshot.pointSlots,"native candidate alias/local identity differs");
        points.seek(alias.root*sizeof(Vec3));const auto xyz=points.scalar<Vec3>();
        const auto value=std::make_pair(alias.root,xyz);parameters.emplace(slot,value);return value;};
    struct BadPoint{Catalog::PointKey key;uint64_t root=0;Vec3 original,xyz;uint64_t trackRows=0;};
    std::map<Catalog::PointKey,BadPoint> badPoints;std::set<uint64_t> badRoots,diagnosticRoots;
    const std::set<uint32_t> failedPhotos{20,21,22,23,24,25,26,27,28,30,31,38,39,40,41,42,43,44,50,52,53,350};
    struct PhotoStats{uint64_t raw=0,invalid=0,inliers=0,overGate=0,sourceInvalid=0;double finite=0,sourceFinite=0;};std::map<std::pair<size_t,uint32_t>,PhotoStats> photos;
    uint64_t scanned=0,invalid=0,overGate=0;fs::create_directories(output);
    std::ofstream rows(output/"invalid-rows.csv"),high(output/"high-residual-rows.csv"),tracks(output/"diagnostic-point-tracks.csv"),proof(output/"admission-invalid-proof.txt");
    rows<<std::setprecision(17);high<<std::setprecision(17);tracks<<std::setprecision(17);proof<<std::setprecision(17);
    const std::string columns="source,label,point_ordinal,local_point,raw_ordinal,image,feature,root,borrowed_old_root,xyz_x,xyz_y,xyz_z,source_x,source_y,source_z,depth,bearing_depth,native_valid,pixel_error,source_pose_valid,source_pose_depth,source_pose_pixel_error\n";
    rows<<columns;high<<columns;
    tracks<<"source,point_ordinal,local_point,raw_ordinal,image,feature,root,xy_x,xy_y,depth,native_valid,pixel_error,source_pose_valid,source_pose_depth,source_pose_pixel_error\n";
    for(size_t source=accepted.sourceCount();source<catalog.sourceCount();++source){
        auto sourcePoses=catalog.sourceViews(source);uint64_t sourceRows=0,sourceInvalid=0;
        catalog.visitImmutableSourcePointsDetailed(source,[&](auto key,const auto&,Vec3 original,const auto& track){
            const auto value=parameter(key);for(const auto& row:track){++sourceRows;++scanned;
                const auto& image=poses.images.at(row.image);const auto& camera=poses.cameras.at(image.camera_id);bool valid=true;
                const double error=reprojErrorAt(camera,image.pose,row.xy,value.second,&valid)/camera.pixel_scale;check(std::isfinite(error),"native incoming residual nonfinite");
                const auto pc=mul(image.pose.R,value.second)+image.pose.t;const auto& originalImage=sourcePoses.images.at(row.image);const auto& originalCamera=sourcePoses.cameras.at(originalImage.camera_id);
                bool originalValid=true;const double originalError=reprojErrorAt(originalCamera,originalImage.pose,row.xy,original,&originalValid)/originalCamera.pixel_scale;
                auto& photo=photos[{source,row.image}];++photo.raw;photo.invalid+=!valid;photo.inliers+=valid&&error<=3;photo.overGate+=valid&&error>3;photo.sourceInvalid+=!originalValid;
                if(valid)photo.finite+=error<=3?error*error:3*(2*error-3);if(originalValid)photo.sourceFinite+=originalError<=3?originalError*originalError:3*(2*originalError-3);
                if(!valid){++invalid;++sourceInvalid;badRoots.insert(value.first);diagnosticRoots.insert(value.first);badPoints.emplace(key,BadPoint{key,value.first,original,value.second,track.size()});}
                const bool selectedHigh=valid&&error>3&&failedPhotos.count(row.image);if(selectedHigh){++overGate;diagnosticRoots.insert(value.first);}
                if(valid&&!selectedHigh)continue;
                const auto beforePc=mul(originalImage.pose.R,original)+originalImage.pose.t;
                auto& destination=valid?high:rows;destination<<source<<','<<catalog.sourceLabel(source)<<','<<key.ordinal<<','<<key.localPoint<<','<<row.ordinal<<','<<row.image<<','<<row.feature<<','<<value.first<<','<<(value.first<accepted.pointSlots())<<','
                    <<value.second.x<<','<<value.second.y<<','<<value.second.z<<','<<original.x<<','<<original.y<<','<<original.z<<','<<pc.z<<','<<pc.dot(camera.bearing(row.xy))<<','<<valid<<','<<error<<','<<originalValid<<','<<beforePc.z<<','
                    <<originalError<<'\n';
            }});
        check(sourceRows==catalog.sourceObservationCount(source),"incoming immutable raw coverage differs");
        proof<<"source="<<source<<"|"<<catalog.sourceLabel(source)<<"|"<<catalog.sourcePath(source).u8string()<<"|raw="<<sourceRows<<"|invalid="<<sourceInvalid<<"|source_identity="<<catalog.sourceIdentity(source)<<'\n';
    }
    check(scanned==snapshot.observations-accepted.observations(),"incoming-only raw coverage does not match admission scope");
    std::map<uint64_t,std::vector<Catalog::RawObservation>> supports;
    for(size_t source=accepted.sourceCount();source<catalog.sourceCount();++source){const auto sourcePoses=catalog.sourceViews(source);
        catalog.visitImmutableSourcePointsDetailed(source,[&](auto key,const auto&,Vec3 original,const auto& track){const auto value=parameter(key);if(!diagnosticRoots.count(value.first))return;
            for(const auto& row:track){check(supports[value.first].size()<100000,"bad-root support exceeded bounded diagnostic");supports[value.first].push_back(row);
                const auto& image=poses.images.at(row.image);const auto& camera=poses.cameras.at(image.camera_id);bool valid=true;const auto error=reprojErrorAt(camera,image.pose,row.xy,value.second,&valid)/camera.pixel_scale;
                const auto& sourceImage=sourcePoses.images.at(row.image);const auto& sourceCamera=sourcePoses.cameras.at(sourceImage.camera_id);bool originalValid=true;const auto sourceError=reprojErrorAt(sourceCamera,sourceImage.pose,row.xy,original,&originalValid)/sourceCamera.pixel_scale;
                tracks<<source<<','<<key.ordinal<<','<<key.localPoint<<','<<row.ordinal<<','<<row.image<<','<<row.feature<<','<<value.first<<','<<row.xy.x<<','<<row.xy.y<<','<<(mul(image.pose.R,value.second)+image.pose.t).z<<','<<valid<<','<<error<<','<<originalValid<<','<<(mul(sourceImage.pose.R,original)+sourceImage.pose.t).z<<','<<sourceError<<'\n';
            }});
    }
    std::ofstream geometry(output/"bad-root-geometry.csv");geometry<<std::setprecision(17)<<"root,borrowed_old_root,incoming_raw,distinct_images,distinct_centers,max_ray_angle_deg,unweighted_ray_rank_valid,ray_x,ray_y,ray_z,ray_invalid,ray_finite_cost,refined_invalid,refined_finite_cost\n";
    for(const auto& item:supports){Mat3 H{};Vec3 g{};std::set<uint32_t> images;std::vector<Vec3> centers,bearings;std::vector<regional::Observation> observations;
        for(const auto& row:item.second){const auto& image=poses.images.at(row.image);const auto& camera=poses.cameras.at(image.camera_id);const auto center=cameraCenter(image.pose),direction=mul(transpose(image.pose.R),camera.bearing(row.xy)).normalized();
            images.insert(row.image);centers.push_back(center);bearings.push_back(direction);observations.push_back({item.first,(uint64_t(row.image)<<32)|row.feature,row.xy});
            const double v[3]{direction.x,direction.y,direction.z};Mat3 A{};for(int a=0;a<3;++a)for(int b=0;b<3;++b){A[3*a+b]=(a==b?1.:0.)-v[a]*v[b];H[3*a+b]+=A[3*a+b];}g=g+mul(A,center);
        }
        size_t uniqueCenters=0;for(size_t i=0;i<centers.size();++i){bool duplicate=false;for(size_t j=0;j<i;++j)duplicate=duplicate||(centers[i]-centers[j]).norm()<1e-9;uniqueCenters+=!duplicate;}
        double angle=0;for(size_t i=0;i<bearings.size();++i)for(size_t j=0;j<i;++j)angle=std::max(angle,std::acos(std::clamp(bearings[i].dot(bearings[j]),-1.,1.))*180/3.14159265358979323846);
        bool rank=false;const auto ray=mul(inverse3(H,&rank),g);auto evaluate=[&](Vec3 xyz){std::pair<uint64_t,double> result{};for(const auto& row:item.second){const auto& image=poses.images.at(row.image);const auto& camera=poses.cameras.at(image.camera_id);bool valid=true;const auto error=reprojErrorAt(camera,image.pose,row.xy,xyz,&valid)/camera.pixel_scale;if(!valid||!std::isfinite(error))++result.first;else result.second+=error<=3?error*error:3*(2*error-3);}return result;};
        const auto q=rank?evaluate(ray):std::make_pair(UINT64_MAX,0.);const auto refined=rank?regional::refineUnifiedPoint(ray,observations,poses,3):ray;const auto qr=rank?evaluate(refined):q;
        geometry<<item.first<<','<<(item.first<accepted.pointSlots())<<','<<item.second.size()<<','<<images.size()<<','<<uniqueCenters<<','<<angle<<','<<rank<<','<<ray.x<<','<<ray.y<<','<<ray.z<<','<<q.first<<','<<q.second<<','<<qr.first<<','<<qr.second<<'\n';
    }
    std::ofstream perPhoto(output/"incoming-photo-quality.csv");perPhoto<<std::setprecision(17)<<"source,image,raw,invalid,inliers,over_3px,finite_cost,source_invalid,source_finite_cost,listed_failed_photo\n";
    for(const auto& item:photos)perPhoto<<item.first.first<<','<<item.first.second<<','<<item.second.raw<<','<<item.second.invalid<<','<<item.second.inliers<<','<<item.second.overGate<<','<<item.second.finite<<','<<item.second.sourceInvalid<<','<<item.second.sourceFinite<<','<<failedPhotos.count(item.first.second)<<'\n';
    std::map<std::pair<uint32_t,uint32_t>,std::vector<std::pair<uint64_t,Catalog::RawObservation>>> featureRows;
    for(const auto& support:supports)for(const auto& row:support.second)featureRows[{row.image,row.feature}].push_back({support.first,row});
    std::ofstream origin(output/"original-weak-point-map.csv");origin<<std::setprecision(17)<<"weak_name,weak_point,image,feature,incoming_source,incoming_point,candidate_root,weak_track_rows,weak_xyz_x,weak_xyz_y,weak_xyz_z,weak_pose_depth,weak_pose_valid,candidate_camera_weak_xyz_depth\n";
    const auto evidence=root.parent_path()/"source-evidence-v1";for(const auto& entry:fs::directory_iterator(evidence)){const auto name=entry.path().filename().u8string();if(name.rfind("local-region-",0)!=0||entry.path().extension()!=".bin")continue;
        const auto before=sha(entry.path());auto weak=model_store_detail::readModel(entry.path());
        for(const auto& point:weak.points3D)for(const auto& row:point.second.track){const auto matched=featureRows.find({row.image_id,row.point2D_idx});if(matched==featureRows.end())continue;for(const auto& correspondence:matched->second){const auto& incoming=correspondence.second;
            const auto& image=weak.images.at(row.image_id);const auto& camera=weak.cameras.at(image.camera_id);bool valid=true;reprojErrorAt(camera,image.pose,image.points2D.at(row.point2D_idx),point.second.xyz,&valid);
            const auto& current=poses.images.at(row.image_id);origin<<name<<','<<point.first<<','<<row.image_id<<','<<row.point2D_idx<<','<<incoming.point.source<<','<<incoming.point.localPoint<<','<<correspondence.first<<','<<point.second.track.size()<<','<<point.second.xyz.x<<','<<point.second.xyz.y<<','<<point.second.xyz.z<<','<<(mul(image.pose.R,point.second.xyz)+image.pose.t).z<<','<<valid<<','<<(mul(current.pose.R,point.second.xyz)+current.pose.t).z<<'\n';
        }}
        check(sha(entry.path())==before,"original weak model changed");proof<<"original_weak="<<entry.path().u8string()<<'|'<<before<<'\n';
    }
    check(sha(root/"scene/head.bin")==headHash&&sha(root/"immutable-raw/manifest.bin")==catalogHash,"readonly admission diagnostic changed head or catalogue");
    proof<<"snapshot="<<snapshot.directory.u8string()<<"\nidentity="<<snapshot.identity<<"\npoint_sha="<<snapshot.pointHash<<"\npose_sha="<<snapshot.poseHash<<"\nalias_sha="<<snapshot.aliasHash<<"\naccepted_identity="<<accepted.identity()<<"\nincoming_raw_scanned="<<scanned<<"\ninvalid_rows="<<invalid<<"\ninvalid_source_points="<<badPoints.size()<<"\ninvalid_canonical_roots="<<badRoots.size()<<"\nlisted_photos_high_residual_rows="<<overGate<<"\ndiagnostic_roots="<<diagnosticRoots.size()<<"\nold_raw_rescanned=0\noriginal_head_manifest_SHA_unchanged=true\nfull_capture_verified=false\n";
    rows.flush();high.flush();tracks.flush();geometry.flush();origin.flush();perPhoto.flush();proof.flush();check(bool(rows)&&bool(high)&&bool(tracks)&&bool(geometry)&&bool(origin)&&bool(perPhoto)&&bool(proof),"admission diagnostic receipt write failed");
    check(invalid==41,"incoming diagnostic does not reproduce the actual 41 invalid projections");std::cout<<"PASS read-only incoming admission diagnosis: "<<invalid<<" invalid rows in "<<badRoots.size()<<" canonical roots; no old raw rescan or geometry edits\n";
}
static void admissionRays(const fs::path& coordination,const fs::path& rejected,const fs::path& firstProof,const fs::path& output){
    using Catalog=regional::RegionalWorkingStore;using NativeReader=model_store_detail::Reader;
    check(!fs::exists(output),"ray diagnostic requires a new isolated output");const auto root=fs::canonical(coordination),target=fs::weakly_canonical(fs::absolute(output));
    for(auto p=target;!p.empty();){check(p!=root,"ray diagnostic output must be outside the retained cohort");const auto parent=p.parent_path();if(parent==p)break;p=parent;}
    const auto headHash=sha(root/"scene/head.bin"),catalogHash=sha(root/"immutable-raw/manifest.bin");
    regional::CanonicalScene accepted(root/"scene",256u<<20,regional::CanonicalScene::OpenMode::ReadOnly);const auto snapshot=accepted.inspectRetainedSnapshot(rejected);
    check(accepted.generation()==8&&snapshot.generation==9&&snapshot.binding.sources.size()==8,"ray diagnostic admission scope changed");
    Catalog catalog(root/"immutable-raw",256u<<20,Catalog::OpenMode::ReadOnly);
    check(regional::CanonicalScene::bindingFor(catalog,snapshot.binding.topology,false)==snapshot.binding,"ray diagnostic source binding differs");
    auto poses=model_store_detail::readModel(snapshot.directory/"poses.bin");
    std::map<Catalog::PointKey,uint64_t> selection;auto parse=[&](const fs::path& path,bool onlyNew){std::istringstream file(read(path));std::string line;std::getline(file,line);
        while(std::getline(file,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();if(line.empty())continue;std::vector<std::string> fields;std::istringstream row(line);std::string value;while(std::getline(row,value,','))fields.push_back(value);
            check(fields.size()==22,"ray input CSV schema differs");if(onlyNew&&fields[8]!="0")continue;Catalog::PointKey key{std::stoull(fields[0]),std::stoull(fields[2]),std::stoull(fields[3])};const auto point=std::stoull(fields[7]);
            auto found=selection.emplace(key,point);check(found.second||found.first->second==point,"diagnostic input changes a canonical alias");}};
    parse(firstProof/"invalid-rows.csv",false);parse(firstProof/"high-residual-rows.csv",true);check(selection.size()==15,"ray diagnostic must select exactly13 invalid and2 high-residual source points");
    std::vector<uint64_t> offsets{0};for(const auto& source:snapshot.binding.sources)offsets.push_back(offsets.back()+source.points);
    struct AliasRow{uint64_t root=0,local=0;};NativeReader aliases(snapshot.directory/"aliases.bin",65536),points(snapshot.directory/"points.bin",65536);
    struct RootTrack{Catalog::PointKey key;Vec3 original,xyz;std::vector<Catalog::RawObservation> rows;};std::map<uint64_t,RootTrack> tracks;
    std::vector<Catalog::PointKey> keys;for(const auto& selected:selection)keys.push_back(selected.first);
    catalog.visitSelectedPoints(keys,[&](auto key,Vec3 original,const auto& rows){const auto id=selection.at(key),slot=offsets.at(size_t(key.source))+key.ordinal;
        aliases.seek(slot*sizeof(AliasRow));const auto alias=aliases.scalar<AliasRow>();check(alias.root==id&&alias.local==key.localPoint,"selected canonical root does not match native aliases");
        points.seek(id*sizeof(Vec3));const auto xyz=points.scalar<Vec3>();check(rows.size()<1024,"selected ray track exceeds bounded diagnostic");
        check(tracks.emplace(id,RootTrack{key,original,xyz,rows}).second,"selected ray root has multiple source aliases; requires complete support diagnostic");});
    fs::create_directories(output);std::ofstream cameras(output/"actual-cameras.csv"),rays(output/"actual-rays.csv"),summary(output/"ray-observability.csv"),bindings(output/"accepted-feature-bindings.csv"),weakGeometry(output/"weak-ray-observability.csv");
    cameras<<std::setprecision(17);rays<<std::setprecision(17);summary<<std::setprecision(17);weakGeometry<<std::setprecision(17);
    cameras<<"scope,camera,model,model_name,colmap_model,width,height,fx,fy,cx,cy,pixel_scale,k1,k2,p1,p2,k3,k4,k5,k6\n";
    auto cameraRow=[&](const std::string& scope,const Camera& c){cameras<<scope<<','<<c.id<<','<<int(c.model)<<','<<camInfo(c.model).cli_name<<','<<camColmapId(c.model)<<','<<c.width<<','<<c.height<<','<<c.fx<<','<<c.fy<<','<<c.cx<<','<<c.cy<<','<<c.pixel_scale<<','<<c.k1<<','<<c.k2<<','<<c.p1<<','<<c.p2<<','<<c.k3<<','<<c.k4<<','<<c.k5<<','<<c.k6<<'\n';};
    for(const auto& camera:poses.cameras)cameraRow("canonical",camera.second);
    rays<<"root,source,local_point,image,feature,xy_x,xy_y,center_x,center_y,center_z,bearing_x,bearing_y,bearing_z,directional_front,directional_pixel_error,directional_angle_deg,candidate_depth\n";
    summary<<"root,rows,distinct_images,max_center_separation,max_world_ray_angle_deg,mean_direction_x,mean_direction_y,mean_direction_z,directional_invalid,directional_inliers,max_directional_pixel_error,directional_finite_cost,accepted_bound_features,distinct_accepted_roots,finite_depth_certified\n";
    bindings<<"root,image,feature,old_source,old_local_point,accepted_root,xy_equal\n";
    std::map<std::pair<uint32_t,uint32_t>,std::set<uint64_t>> oldRoots;
    std::set<std::pair<uint32_t,uint32_t>> measurements;for(const auto& track:tracks)for(const auto& row:track.second.rows)measurements.insert({row.image,row.feature});
    for(size_t source=0;source<accepted.sourceCount();++source){const auto views=catalog.sourceViews(source);std::map<uint32_t,std::set<uint32_t>> selected;
        for(const auto& pair:measurements)if(views.images.count(pair.first))selected[pair.first].insert(pair.second);
        for(const auto& photo:selected)catalog.visitSourceImageFeatures(source,photo.first,[&](uint64_t feature,Vec2 xy,uint64_t assignment){if(!photo.second.count(uint32_t(feature))||assignment==kInvalidPoint3D)return;
            const auto key=catalog.localKey(source,assignment);const auto canonical=accepted.acceptedCanonical(key);oldRoots[{photo.first,uint32_t(feature)}].insert(canonical);
            for(const auto& track:tracks)for(const auto& row:track.second.rows)if(row.image==photo.first&&row.feature==feature)bindings<<track.first<<','<<row.image<<','<<row.feature<<','<<source<<','<<assignment<<','<<canonical<<','<<(row.xy.x==xy.x&&row.xy.y==xy.y)<<'\n';});
    }
    auto angles=[](const std::vector<Vec3>& directions){double maximum=0;for(size_t i=0;i<directions.size();++i)for(size_t j=0;j<i;++j)maximum=std::max(maximum,std::acos(std::clamp(directions[i].dot(directions[j]),-1.,1.))*180/3.14159265358979323846);return maximum;};
    auto separation=[](const std::vector<Vec3>& centers){double maximum=0;for(size_t i=0;i<centers.size();++i)for(size_t j=0;j<i;++j)maximum=std::max(maximum,(centers[i]-centers[j]).norm());return maximum;};
    uint64_t totalRows=0;
    for(const auto& item:tracks){std::vector<Vec3> directions,centers;std::set<uint32_t> images;Vec3 mean{};uint64_t assigned=0;std::set<uint64_t> assignedRoots;
        for(const auto& row:item.second.rows){const auto& image=poses.images.at(row.image);const auto& camera=poses.cameras.at(image.camera_id);const auto direction=mul(transpose(image.pose.R),camera.bearing(row.xy)).normalized();directions.push_back(direction);mean=mean+direction;centers.push_back(cameraCenter(image.pose));images.insert(row.image);
            const auto roots=oldRoots.find({row.image,row.feature});if(roots!=oldRoots.end()){++assigned;assignedRoots.insert(roots->second.begin(),roots->second.end());}}
        mean=mean.normalized();uint64_t invalid=0,inliers=0;double maximum=0,finite=0;
        for(size_t i=0;i<item.second.rows.size();++i){const auto& row=item.second.rows[i];const auto& image=poses.images.at(row.image);const auto& camera=poses.cameras.at(image.camera_id);auto directionPose=image.pose;directionPose.t={};bool valid=true;
            const auto error=reprojErrorAt(camera,directionPose,row.xy,mean,&valid)/camera.pixel_scale;check(std::isfinite(error),"directional native projection is nonfinite");invalid+=!valid;inliers+=valid&&error<=3;if(valid){maximum=std::max(maximum,error);finite+=error<=3?error*error:3*(2*error-3);}
            rays<<item.first<<','<<item.second.key.source<<','<<item.second.key.localPoint<<','<<row.image<<','<<row.feature<<','<<row.xy.x<<','<<row.xy.y<<','<<centers[i].x<<','<<centers[i].y<<','<<centers[i].z<<','<<directions[i].x<<','<<directions[i].y<<','<<directions[i].z<<','<<valid<<','<<error<<','<<std::acos(std::clamp(directions[i].dot(mean),-1.,1.))*180/3.14159265358979323846<<','<<(mul(image.pose.R,item.second.xyz)+image.pose.t).z<<'\n';}
        summary<<item.first<<','<<item.second.rows.size()<<','<<images.size()<<','<<separation(centers)<<','<<angles(directions)<<','<<mean.x<<','<<mean.y<<','<<mean.z<<','<<invalid<<','<<inliers<<','<<maximum<<','<<finite<<','<<assigned<<','<<assignedRoots.size()<<",false\n";totalRows+=item.second.rows.size();
    }
    weakGeometry<<"weak_name,weak_point,candidate_root,original_raw,distinct_images,original_world_ray_angle_deg,original_center_separation,min_original_depth,max_original_depth,original_invalid,max_original_pixel_error\n";
    for(const auto& entry:fs::directory_iterator(root.parent_path()/"source-evidence-v1")){const auto name=entry.path().filename().u8string();if(name.rfind("local-region-",0)!=0||entry.path().extension()!=".bin")continue;
        const auto sourceHash=sha(entry.path());auto weak=model_store_detail::readModel(entry.path());for(const auto& camera:weak.cameras)cameraRow(name,camera.second);
        for(const auto& point:weak.points3D){std::set<uint64_t> mapped;for(const auto& row:point.second.track)for(const auto& track:tracks)for(const auto& actual:track.second.rows)if(row.image_id==actual.image&&row.point2D_idx==actual.feature)mapped.insert(track.first);
            if(mapped.empty())continue;std::set<uint32_t> images;std::vector<Vec3> directions,centers;double minimum=std::numeric_limits<double>::infinity(),maximum=-minimum,maxError=0;uint64_t invalid=0;
            for(const auto& row:point.second.track){const auto& image=weak.images.at(row.image_id);const auto& camera=weak.cameras.at(image.camera_id);const auto xy=image.points2D.at(row.point2D_idx);directions.push_back(mul(transpose(image.pose.R),camera.bearing(xy)).normalized());centers.push_back(cameraCenter(image.pose));images.insert(row.image_id);bool valid=true;const auto error=reprojErrorAt(camera,image.pose,xy,point.second.xyz,&valid)/camera.pixel_scale;invalid+=!valid;if(valid)maxError=std::max(maxError,error);const auto depth=(mul(image.pose.R,point.second.xyz)+image.pose.t).z;minimum=std::min(minimum,depth);maximum=std::max(maximum,depth);}
            for(auto mappedRoot:mapped)weakGeometry<<name<<','<<point.first<<','<<mappedRoot<<','<<point.second.track.size()<<','<<images.size()<<','<<angles(directions)<<','<<separation(centers)<<','<<minimum<<','<<maximum<<','<<invalid<<','<<maxError<<'\n';
        }check(sha(entry.path())==sourceHash,"readonly ray diagnostic changed original weak source");
    }
    check(sha(root/"scene/head.bin")==headHash&&sha(root/"immutable-raw/manifest.bin")==catalogHash,"ray diagnostic changed retained geometry or catalogue");
    std::ostringstream receipt;receipt<<"mode=readonly-native-selected-ray-diagnostic\nselected_points="<<tracks.size()<<"\nselected_incoming_raw="<<totalRows<<"\nold_feature_tables_read_for_binding_only=true\nold_raw_rescanned=0\nfinite_depth_certified=false\nno_large_finite_XYZ_constructed=true\nsource_snapshot="<<snapshot.identity<<"\noriginal_head_manifest_SHA_unchanged=true\nfull_capture_verified=false\n";write(output/"ray-diagnostic-proof.txt",receipt.str());
    cameras.flush();rays.flush();summary.flush();bindings.flush();weakGeometry.flush();check(bool(cameras)&&bool(rays)&&bool(summary)&&bool(bindings)&&bool(weakGeometry),"ray diagnostic output failed");std::cout<<"PASS read-only native directional diagnosis of "<<tracks.size()<<" points/"<<totalRows<<" raw rows; no finite point certified or changed\n";
}
static int test(int argc,char** argv){if(argc>1&&std::string(argv[1])=="--admission-rays"){check(argc==6,"usage: --admission-rays coordination retainedRejected priorInvalidProof newOutput");admissionRays(fs::u8path(argv[2]),fs::u8path(argv[3]),fs::u8path(argv[4]),fs::u8path(argv[5]));return 0;}if(argc>1&&std::string(argv[1])=="--admission-invalid"){check(argc==5,"usage: --admission-invalid coordination retainedRejected newOutput");admissionInvalid(fs::u8path(argv[2]),fs::u8path(argv[3]),fs::u8path(argv[4]));return 0;}const auto r=configuration(argc,argv);const std::string mode=argv[1];if(mode=="--audit-accepted"){auditAccepted(r);return 0;}if(mode=="--config"){std::cout<<"PASS actual official config/cache signature identical\n"<<r.signature;return 0;}if(mode=="--prepare")prepare(r);else if(mode=="--resume-boundary")boundary(r);else if(mode=="--stage-recovered"){check(argc==9,"two genuine recovered model paths required");stageRecovered(r,fs::u8path(argv[7]),fs::u8path(argv[8]));}else if(mode=="--weak-seed-diagnostic"){check(argc==9,"original weak source and isolated output required");weakSeedDiagnostic(r,fs::u8path(argv[7]),fs::u8path(argv[8]));}else if(mode=="--complete"||mode=="--complete-natural")complete(r,mode=="--complete-natural");else throw std::runtime_error("unknown formal replay mode");return 0;}
int main(int argc,char** argv){return sfmTestMain(argc,argv,test);}
