#include "sfm/map/CanonicalImageIdentity.h"
#include "sfm/map/CanonicalInitialization.h"
#include "sfm/tests/TestMain.h"
#include <iostream>

using namespace sfm; using namespace sfm::regional;
namespace fs=std::filesystem;
struct Input { std::string name,export_name; };
static void check(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
template<class Run> static void rejects(Run run,const char* reason){bool failed=false;try{run();}catch(const std::exception&){failed=true;}check(failed,reason);}
static Reconstruction model(bool incoming){
    Reconstruction value;value.cameras[0]=Camera::defaultFor(0,800,600,500,CamModel::Pinhole);value.cameras[0].pixel_scale=2;
    const uint64_t point=incoming?20:10;Point3D geometry;geometry.xyz={.2,.1,5};
    for(uint32_t image=0;image<(incoming?3u:2u);++image){Image view;view.id=image;view.camera_id=0;view.name="capture-"+std::to_string(image)+(incoming?"":".JPG");view.registered=true;view.exif_orientation=2;
        view.pose={mat3Identity(),{-.4*image,0,0}};view.points2D.push_back(value.cameras.at(0).project(geometry.xyz+view.pose.t));view.point3D_ids.push_back(point);value.images.emplace(image,std::move(view));geometry.track.push_back({image,0});}
    geometry.track.push_back({incoming?2u:0u,0});value.points3D.emplace(point,std::move(geometry));value.next_point3D_id=point+1;return value;
}
static int test(int argc,char** argv){
    const fs::path root=argc>1?fs::u8path(argv[1]):fs::path("canonical-image-identity-fixture");check(!fs::exists(root),"fixture requires a fresh owned directory");fs::create_directories(root);
    std::map<uint32_t,Input> inputs{{0,{"capture-0","capture-0.JPG"}},{1,{"capture-1","capture-1.JPG"}},{2,{"capture-2","capture-2.JPG"}}};
    auto old=model(false),held=model(true);const fs::path original=root/"original-held-stem.bin",before=root/"old-original.bin";
    model_store_detail::writeModel(held,original);model_store_detail::writeModel(old,before);const auto originalSHA=canonical_initial_detail::hash(original),oldSHA=canonical_initial_detail::hash(before);
    RegionalWorkingStore catalog(root/"catalog",64u<<20);catalog.importModel(before,old,"old-actual-names");
    canonicalizeSourceImageIdentity(held,inputs,catalog.poses());
    check(held.images.at(0).name=="capture-0.JPG"&&held.images.at(1).name=="capture-1.JPG"&&held.images.at(2).name=="capture-2.JPG","held stem did not bind the explicit old catalogue/new original file names");
    auto exact=held;for(auto& image:exact.images)image.second.name=inputs.at(image.first).name;
    check(canonical_initial_detail::nativeDigest(exact)==originalSHA,"source name adapter changed any P/K/XYZ/feature/raw/native field");
    const auto normalized=root/"normalized-held.bin";model_store_detail::writeModel(held,normalized);
    const auto readback=model_store_detail::readModel(normalized);check(canonical_initial_detail::nativeDigest(readback)==canonical_initial_detail::nativeDigest(held),"native source normalized serialization changed fields");
    catalog.importModel(normalized,readback,"incoming-explicit-names");catalog.verifyImmutable();
    check(catalog.sourceCount()==2&&catalog.sourceObservationCount(0)==3&&catalog.sourceObservationCount(1)==4&&catalog.poses().images.size()==3,"normalized source import lost duplicate raw or new input coverage");
    auto parameters=held;canonicalizeImageIdentity(parameters,inputs);check(parameters.images.at(0).name=="capture-0"&&parameters.images.at(2).name=="capture-2","Scene internal feature identity was confused with source file names");
    auto bad=model(true);bad.images.at(0).name="capture-0.png";rejects([&]{canonicalizeSourceImageIdentity(bad,inputs,old);},"unknown extension was silently guessed into a valid original photo");
    bad=model(true);bad.images.at(0).id=9;rejects([&]{canonicalizeSourceImageIdentity(bad,inputs,old);},"image ID mismatch silently adapted");
    auto missing=inputs;missing.erase(2);bad=model(true);rejects([&]{canonicalizeSourceImageIdentity(bad,missing,old);},"unknown new input ID silently adapted");
    auto unknownOld=old;unknownOld.images.at(1).name="unrelated.JPG";bad=model(true);rejects([&]{canonicalizeSourceImageIdentity(bad,inputs,unknownOld);},"unknown existing catalogue name silently replaced");
    auto missingExport=inputs;missingExport.at(2).export_name.clear();bad=model(true);rejects([&]{canonicalizeSourceImageIdentity(bad,missingExport,old);},"missing original filename guessed from stem");
    check(canonical_initial_detail::hash(original)==originalSHA&&canonical_initial_detail::hash(before)==oldSHA,"metadata adapter changed frozen original input bytes");
    std::cout<<"explicit source/Scene name identities, native import, duplicates, geometry byte equality and closed bad identity tests passed\n";return 0;
}
int main(int argc,char** argv){return sfmTestMain(argc,argv,test);}
