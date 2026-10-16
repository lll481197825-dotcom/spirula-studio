#pragma once

#include "sfm/map/RegionalQuality.h"
#include "sfm/map/RegionalWorkingStore.h"
#include "sfm/map/SceneTopology.h"
#include "sfm/map/CanonicalLandmarkGeometry.h"
#include "sfm/map/CanonicalParameterView.h"
#include "sfm/ba/CpuParallel.h"
#include "core/Sha256.h"
#include <iomanip>
#include <functional>

namespace sfm::regional {

struct ExportPhotoFailure { uint32_t image = UINT32_MAX; std::string path, reason; uint64_t occurrences=1; std::string count_unit="check"; };
struct VerifiedExportForecast {
    uint64_t raw=0,point_slots=0,features=0,framing_bytes=0,metadata_report_bytes=0,required_peak_bytes=0;
    std::map<std::string,uint64_t> phases;
};
inline VerifiedExportForecast verifiedExportForecast(uint64_t raw,uint64_t slots,uint64_t features,uint64_t framing,uint64_t metadata) {
    if(!metadata) throw std::runtime_error("formal export metadata/report capacity is unknown");
    VerifiedExportForecast out; out.raw=raw;out.point_slots=slots;out.features=features;out.framing_bytes=framing;out.metadata_report_bytes=metadata;
    const uint64_t nodes=storage::addBytes(storage::scaleBytes(slots,40),4079);
    const uint64_t fixed=storage::addBytes(nodes,storage::addBytes(framing,metadata));
    auto phase=[&](const char* name,uint64_t value){out.phases[name]=storage::addBytes(value,fixed);out.required_peak_bytes=std::max(out.required_peak_bytes,out.phases[name]);};
    phase("feature-sort",storage::scaleBytes(raw,112));
    phase("point-sort",storage::scaleBytes(raw,144));
    phase("point-write",storage::addBytes(storage::scaleBytes(raw,96),storage::scaleBytes(slots,51)));
    phase("median-sort",storage::addBytes(storage::scaleBytes(raw,80),storage::scaleBytes(slots,51)));
    phase("image-write",storage::addBytes(storage::scaleBytes(raw,56),storage::addBytes(storage::scaleBytes(slots,51),storage::scaleBytes(features,24))));
    phase("native-reader",storage::addBytes(storage::scaleBytes(raw,24),storage::addBytes(storage::scaleBytes(slots,51),storage::scaleBytes(features,24))));
    return out;
}
struct ExportInputPhoto { std::string path, name, export_name; };
struct ExportConstraintPhoto {
    uint32_t image = UINT32_MAX;
    std::string path;
    bool gps_verified = false, scale_verified = false, attitude_verified = false;
    double gps_error_metres = std::numeric_limits<double>::quiet_NaN();
    double scale_error_fraction = std::numeric_limits<double>::quiet_NaN();
    double attitude_error_degrees = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::string> reasons;
};
struct ExportFinalAcceptance {
    std::vector<ExportConstraintPhoto> photos;
    bool historical_evaluated = false, historical_required = false, historical_verified = false;
};
struct VerifiedExportReport {
    ExportStats stats;
    Quality raw;
    uint64_t lineage_rows = 0;
    uint64_t finite_raw = 0, directional_raw = 0, directional_points = 0, directional_unique = 0;
    uint64_t retained_raw = 0, retained_points = 0, retained_unique = 0;
    bool reader_verified = false, verified = false;
    std::map<uint32_t, PhotoQuality> raw_photos;
    std::map<uint32_t, PhotoQuality> unique_quality;
    std::map<uint32_t, uint64_t> unique_photos;
    std::map<uint32_t, uint64_t> unique_landmarks;
    std::map<uint32_t, uint64_t> retained_unique_photos;
    std::map<uint32_t, ExportConstraintPhoto> constraints;
    std::vector<ExportPhotoFailure> failures;
    VerifiedExportForecast forecast;
    uint64_t observed_owned_bytes_at_checkpoints=0;
    uint64_t reader_parameter_cache_bytes=0, reader_raw_batches=0;
};
struct VerifiedExportOptions {
    size_t cache_bytes = size_t(256) << 20;
    uint64_t reader_memory_bytes = 0;
    double max_error = 3;
    Quality accepted_raw;
    std::map<uint32_t, ExportInputPhoto> inputs;
    std::function<std::string(uint32_t)> input_path;
    bool historical_required = false;
    const std::vector<uint32_t>* owners = nullptr;
    std::function<void(const char*,uint64_t,uint64_t)> phase_bytes;
    int threads = 0;
    std::function<void(const char*,uint64_t,uint64_t)> reportphase;
};
using ExportFinalGate = std::function<ExportFinalAcceptance(const Reconstruction&, const Quality&,
    const std::map<uint32_t, PhotoQuality>&, const ExportStats&)>;

namespace verified_export_detail {
struct Node { uint64_t present = 0; Vec3 xyz; uint8_t rgb[3] = {128,128,128}; uint8_t padding[5]{}; };
static_assert(sizeof(Node)==40,"formal node disk forecast layout changed");
struct RawAlias { uint64_t row = 0, point = 0; };
static_assert(sizeof(RawAlias) == 16, "raw aliases must not duplicate the immutable raw catalogue");
struct FormalCounts { uint64_t cameras = 0, images = 0, features = 0, points = 0, tracks = 0; };

inline std::string csv(const std::string& text) {
    std::string out = "\""; for (char c : text) { if (c == '\"') out += '\"'; out += c; } return out + '\"';
}
inline std::string inputPath(uint32_t image,const Reconstruction& poses,const VerifiedExportOptions& options) {
    const auto input=options.inputs.find(image); if (input!=options.inputs.end()) return input->second.path;
    if (options.input_path) return options.input_path(image);
    const auto at=poses.images.find(image); return at==poses.images.end()?"unknown-input-"+std::to_string(image):at->second.name;
}
inline std::string exportName(uint32_t image,const Reconstruction& poses,const VerifiedExportOptions& options) {
    const auto input=options.inputs.find(image);
    if(input==options.inputs.end()) return poses.images.at(image).name;
    return input->second.export_name.empty()?input->second.name:input->second.export_name;
}
inline void failureReport(const fs::path& output, VerifiedExportReport& result, const std::string& binding,
                          const Reconstruction& poses, const VerifiedExportOptions& options) {
    using Reason=std::pair<std::string,std::string>;
    std::map<uint32_t,std::map<Reason,ExportPhotoFailure>> grouped;
    for(const auto& row:result.failures) {auto& at=grouped[row.image][{row.reason,row.count_unit}];
        if(at.reason.empty()) at=row; else at.occurrences=storage::addBytes(at.occurrences,row.occurrences);}
    result.failures.clear();for(const auto& photo:grouped)for(const auto& row:photo.second)result.failures.push_back(row.second);
    std::map<uint32_t,std::string> photoReasons; std::string globalReasons; uint64_t reportBytes=1u<<20;
    for(const auto& row:result.failures) {auto& reasons=row.image==UINT32_MAX?globalReasons:photoReasons[row.image];if(!reasons.empty())reasons+="; ";
        reasons+=row.reason;if(row.occurrences>1)reasons+=" [occurrences="+std::to_string(row.occurrences)+", unit="+row.count_unit+"]";
        reportBytes=storage::addBytes(reportBytes,256+2*row.path.size()+2*row.reason.size()+2*row.count_unit.size());}
    std::map<uint32_t,ExportInputPhoto> inventory=options.inputs;
    if (inventory.empty()) for(const auto& item:poses.images) inventory[item.first]={inputPath(item.first,poses,options),item.second.name};
    for(const auto& item:inventory) reportBytes=storage::addBytes(reportBytes,512+2*item.second.path.size()+2*globalReasons.size()+2*photoReasons[item.first].size());
    if(result.forecast.metadata_report_bytes && reportBytes>result.forecast.metadata_report_bytes)
        throw std::runtime_error("complete formal report exceeds its admitted metadata/report byte bound");
    std::ofstream rows(output / "export-photo-failures.csv"); rows << "scope,image_id,path,reason,occurrences,count_unit\n";
    for (const auto& row : result.failures) rows << (row.image==UINT32_MAX?"global":"photo")<<','<<row.image << ',' << csv(row.path) << ',' << csv(row.reason)<<','<<row.occurrences<<','<<csv(row.count_unit) << '\n';
    rows.flush(); if (!rows) throw std::runtime_error("cannot write isolated formal export failure report");
    std::ofstream photos(output / "export-per-photo.csv"); photos << std::setprecision(17)<<"image_id,path,registered,raw_observations,raw_inliers,raw_invalid,unique_visual_observations,unique_visual_landmarks,unique_inliers,unique_invalid,unique_mean_extraction_error,gps_error_metres,scale_error_fraction,attitude_error_degrees,verified,reason,constrained_raw_observations,retained_raw_observations,retained_unique_observations,total_unique_observations\n";
    for (const auto& item:inventory) {
        const auto at=result.raw_photos.find(item.first); const PhotoQuality q=at==result.raw_photos.end()?PhotoQuality{}:at->second;
        const auto unique=result.unique_photos.find(item.first); std::string reasons;
        const auto landmarks=result.unique_landmarks.find(item.first);
        const auto uq=result.unique_quality.find(item.first); const PhotoQuality uniqueQuality=uq==result.unique_quality.end()?PhotoQuality{}:uq->second;
        const auto constraint=result.constraints.find(item.first); const auto pose=poses.images.find(item.first);
        const ExportConstraintPhoto decision=constraint==result.constraints.end()?ExportConstraintPhoto{}:constraint->second;
        reasons=globalReasons; if(!photoReasons[item.first].empty()){if(!reasons.empty())reasons+="; ";reasons+=photoReasons[item.first];}
        const bool photoVerified=result.reader_verified && reasons.empty() && constraint!=result.constraints.end() &&
            decision.gps_verified && decision.scale_verified && decision.attitude_verified;
        const auto retained=result.retained_unique_photos.find(item.first);
        const uint64_t retainedUnique=retained==result.retained_unique_photos.end()?0:retained->second;
        photos<<item.first<<','<<csv(item.second.path)<<','<<(pose!=poses.images.end() && pose->second.registered)
            <<','<<storage::addBytes(q.observations,q.retained_observations)<<','<<q.inliers<<','<<q.invalid_projections<<','<<(unique==result.unique_photos.end()?0:unique->second)
            <<','<<(landmarks==result.unique_landmarks.end()?0:landmarks->second)
            <<','<<uniqueQuality.inliers<<','<<uniqueQuality.invalid_projections<<','<<(uniqueQuality.observations?uniqueQuality.error/uniqueQuality.observations:std::numeric_limits<double>::quiet_NaN())
            <<','<<decision.gps_error_metres<<','<<decision.scale_error_fraction<<','<<decision.attitude_error_degrees
            <<','<<photoVerified<<','<<csv(reasons.empty()?(photoVerified?"verified":"pending per-photo acceptance"):reasons)
            <<','<<q.observations<<','<<q.retained_observations<<','<<retainedUnique<<','<<storage::addBytes(uniqueQuality.observations,retainedUnique)<<'\n';
    }
    photos.flush(); if (!photos) throw std::runtime_error("cannot flush formal per-photo acceptance report");
    std::ofstream summary(output / "verified-export.txt"); summary << std::setprecision(17)
        << "schema=canonical-scene-export-v3\nscene_binding=" << binding << "\nfull_capture_verified=" << (result.verified ? "true" : "false")
        << "\nreader_verified=" << result.reader_verified << "\nraw_observations=" << result.raw.observations
        << "\nlineage_rows=" << result.lineage_rows << "\nraw_invalid=" << result.raw.invalid_projections
        << "\nfinite_raw_observations=" << result.finite_raw << "\ndirectional_raw_observations=" << result.directional_raw
        << "\nretained_raw_observations=" << result.retained_raw << "\nuntriangulated_raw_observations=" << result.retained_raw
        << "\nconstrained_raw_observations=" << storage::addBytes(result.finite_raw,result.directional_raw)
        << "\nunresolved_raw_observations=" << (result.reader_verified ? "0" : "unknown")
        << "\nraw_type_coverage_verified=" << result.reader_verified << "\ndirectional_landmarks=" << result.directional_points
        << "\ndirectional_unique_observations=" << result.directional_unique
        << "\nretained_landmarks=" << result.retained_points << "\nretained_unique_observations=" << result.retained_unique
        << "\nretained_scope=untriangulated_original_measurements_without_geometric_constraint"
        << "\ncolmap_scope=finite_metric_landmarks\ndirectional_scope=depth_unobservable_orientation_constraints\nmetric_landmark_minimum=12"
        << "\nraw_finite=" << result.raw.finite_cost << "\nraw_total=" << result.raw.cost
        << "\nunique_observations=" << result.stats.observations << "\ncanonical_points=" << result.stats.points
        << "\nconstrained_unique_observations=" << result.stats.observations
        << "\ntotal_unique_observations=" << storage::addBytes(result.stats.observations,result.retained_unique)
        << "\nphoto_components=" << result.stats.photo_components << "\nfailed_checks=" << result.failures.size()
        << "\nexport_features_from_VKFT="<<result.forecast.features<<"\nexport_required_peak_bytes="<<result.forecast.required_peak_bytes
        << "\nexport_metadata_report_bytes="<<result.forecast.metadata_report_bytes
        << "\nobserved_owned_bytes_at_checkpoints="<<result.observed_owned_bytes_at_checkpoints
        << "\nreader_parameter_cache_bytes="<<result.reader_parameter_cache_bytes<<"\nreader_raw_batches="<<result.reader_raw_batches<<'\n';
    for(const auto& phase:result.forecast.phases)summary<<"export_phase_peak_bytes."<<phase.first<<'='<<phase.second<<'\n';
    summary.flush(); if (!summary) throw std::runtime_error("cannot flush formal export proof");
}
inline double finiteDouble(model_store_detail::Reader& in) {
    const double value = in.scalar<double>(); if (!std::isfinite(value)) throw std::runtime_error("nonfinite formal COLMAP value"); return value;
}
inline std::string colmapName(model_store_detail::Reader& in) {
    std::string name; for (;;) { const char c = in.scalar<char>(); if (!c) return name; name += c;
        if (name.size() > (1u << 20)) throw std::runtime_error("unterminated formal COLMAP image name"); }
}

// Validate framing before the general native parser can allocate from a corrupt
// count. No automatic track repair is accepted; raw tracks are compared below.
inline FormalCounts formalStructure(const fs::path& output, uint64_t memory = UINT64_MAX) {
    using Reader = model_store_detail::Reader; FormalCounts n;
    std::map<uint32_t,std::vector<uint64_t>> featurePoints; uint64_t assignedFeatures=0;
    std::set<uint32_t> cameras;
#ifdef _WIN32
    MEMORYSTATUSEX host{}; host.dwLength=sizeof(host);
    if (!GlobalMemoryStatusEx(&host)) throw std::runtime_error("cannot query RAM for strict formal structure validation");
    const uint64_t available=std::min<uint64_t>(host.ullAvailPhys,host.ullAvailPageFile), reserve=uint64_t(512)<<20;
    if (available<=reserve) throw std::runtime_error("insufficient actual RAM/commit headroom for strict formal structure validation");
    memory=std::min(memory,available-reserve);
#endif
    { Reader in(output / "cameras.bin"); n.cameras = in.count(24);
      for (uint64_t i = 0; i < n.cameras; ++i) {
        if (!cameras.insert(in.scalar<uint32_t>()).second) throw std::runtime_error("duplicate formal camera identity");
        const int32_t model = in.scalar<int32_t>();
        const auto camera = camFromColmapId(model);
        if (camColmapId(camera) != model || !in.scalar<uint64_t>() || !in.scalar<uint64_t>())
            throw std::runtime_error("unsupported/invalid formal camera model or dimensions");
        for (int k = 0; k < camColmapParams(camera); ++k) finiteDouble(in);
      } in.finish(); }
    { Reader in(output / "images.bin"); n.images = in.count(73);
      for (uint64_t i = 0; i < n.images; ++i) {
        const auto image=in.scalar<uint32_t>();
        if (featurePoints.count(image)) throw std::runtime_error("duplicate formal image identity");
        double quaternionNorm=0; for (int k=0;k<4;++k) {const auto value=finiteDouble(in); quaternionNorm+=value*value;}
        if(std::abs(quaternionNorm-1)>1e-12) throw std::runtime_error("invalid formal unit rotation quaternion");
        for (int k=0;k<3;++k) finiteDouble(in);
        if (!cameras.count(in.scalar<uint32_t>())) throw std::runtime_error("formal image references unknown camera");
        colmapName(in);
        const uint64_t count = in.count(24); n.features = storage::addBytes(n.features, count);
        if (storage::addBytes(storage::scaleBytes(n.features,8),storage::scaleBytes(n.images,256))>memory)
            throw std::runtime_error("strict formal structural validation exceeds admitted RAM");
        auto& ids=featurePoints[image]; ids.resize(size_t(count));
        for (uint64_t k = 0; k < count; ++k) { finiteDouble(in); finiteDouble(in); ids[size_t(k)]=in.scalar<uint64_t>(); assignedFeatures+=ids[size_t(k)]!=kInvalidPoint3D; }
      } in.finish(); }
    { Reader in(output / "points3D.bin"); n.points = in.count(51);
      for (uint64_t i = 0; i < n.points; ++i) {
        const auto point=in.scalar<uint64_t>();
        if (point == kInvalidPoint3D) throw std::runtime_error("invalid formal point identity");
        for (int k = 0; k < 3; ++k) finiteDouble(in);
        for (int k = 0; k < 3; ++k) in.scalar<uint8_t>(); finiteDouble(in);
        const uint64_t count = in.count(8); n.tracks = storage::addBytes(n.tracks, count);
        std::set<uint32_t> views; std::set<uint64_t> pairs;
        for (uint64_t k = 0; k < count; ++k) {
            const auto image=in.scalar<uint32_t>(),feature=in.scalar<uint32_t>(); const auto at=featurePoints.find(image);
            if (at==featurePoints.end() || feature>=at->second.size() || at->second[feature]!=point || !pairs.insert((uint64_t(image)<<32)|feature).second)
                throw std::runtime_error("formal track/feature reciprocal structure is invalid; native auto-repair forbidden");
            views.insert(image);
        }
        if(views.size()<2) throw std::runtime_error("formal canonical point has fewer than two distinct supporting images");
      } in.finish(); }
    if (assignedFeatures!=n.tracks) throw std::runtime_error("formal image/track reciprocal counts disagree; native auto-repair forbidden");
    return n;
}
inline uint64_t readerEstimate(FormalCounts n) {
    uint64_t bytes = storage::scaleBytes(n.points, sizeof(Point3D) + 96);
    bytes = storage::addBytes(bytes, storage::scaleBytes(n.tracks, sizeof(TrackElement)));
    bytes = storage::addBytes(bytes, storage::scaleBytes(n.features, sizeof(Vec2) + sizeof(uint64_t)));
    return storage::addBytes(bytes, storage::scaleBytes(n.images, sizeof(Image) + 4096));
}
inline Reconstruction strictReadBack(const fs::path& output, uint64_t memory) {
    const auto n = formalStructure(output,memory);
    const auto estimate = readerEstimate(n);
    if (!memory || estimate > memory) throw std::runtime_error("formal native Reader requires " + std::to_string(estimate) + " bytes; explicit admitted memory is " + std::to_string(memory));
#ifdef _WIN32
    MEMORYSTATUSEX state{}; state.dwLength = sizeof(state);
    if (!GlobalMemoryStatusEx(&state) || std::min<uint64_t>(state.ullAvailPhys,state.ullAvailPageFile) < storage::addBytes(estimate, uint64_t(512) << 20))
        throw std::runtime_error("insufficient actual available RAM for complete formal native Reader");
#endif
    const auto before = spirula::sha256_file((output / "points3D.bin").u8string());
    Reconstruction rec = Reconstruction::readBinary(output.u8string());
    if (rec.cameras.size() != n.cameras || rec.images.size() != n.images || rec.points3D.size() != n.points)
        throw std::runtime_error("native formal Reader merged duplicate identities");
    model_store_detail::Reader in(output / "points3D.bin");
    if (in.scalar<uint64_t>() != n.points) throw std::runtime_error("formal point file changed during readback");
    for (uint64_t i = 0; i < n.points; ++i) {
        const auto id = in.scalar<uint64_t>(); const auto& point = rec.points3D.at(id);
        for (double expected : {point.xyz.x, point.xyz.y, point.xyz.z})
            if (in.scalar<double>() != expected) throw std::runtime_error("formal native Reader point changed");
        for (int k = 0; k < 3; ++k) if (in.scalar<uint8_t>() != point.rgb[k]) throw std::runtime_error("formal point color changed");
        if (in.scalar<double>() != point.error || in.scalar<uint64_t>() != point.track.size())
            throw std::runtime_error("formal native Reader repaired or changed raw track metadata");
        std::set<uint64_t> pairs;
        for (const auto& row : point.track) {
            if (in.scalar<uint32_t>() != row.image_id || in.scalar<uint32_t>() != row.point2D_idx ||
                !pairs.insert((uint64_t(row.image_id)<<32)|row.point2D_idx).second || rec.images.at(row.image_id).point3D_ids.at(row.point2D_idx) != id)
                throw std::runtime_error("formal native Reader repaired/nonreciprocal/conflicting track");
        }
    }
    in.finish();
    uint64_t reciprocal = 0;
    for (const auto& item : rec.images) for (size_t f = 0; f < item.second.point3D_ids.size(); ++f) {
        const auto id = item.second.point3D_ids[f]; if (id == kInvalidPoint3D) continue;
        if (!rec.points3D.count(id)) throw std::runtime_error("formal image feature references an unknown point");
        ++reciprocal;
    }
    if (reciprocal != n.tracks || before != spirula::sha256_file((output / "points3D.bin").u8string()))
        throw std::runtime_error("formal files/reciprocal counts changed during native readback");
    return rec;
}
struct DirectionalPoint { Vec3 direction; std::vector<TrackElement> track; };
inline std::map<uint64_t, DirectionalPoint> strictReadDirections(const fs::path& output, const Reconstruction& finite, uint64_t memory) {
    std::map<uint64_t, DirectionalPoint> result;
    model_store_detail::Reader in(output / "directional-points.bin");
    if (in.scalar<uint64_t>() != 0x315249444e414353ull || in.scalar<uint32_t>() != 1 || in.scalar<uint32_t>() != 1)
        throw std::runtime_error("unknown typed directional landmark export schema");
    const auto count = in.count(40); uint64_t bytes = storage::scaleBytes(count, sizeof(DirectionalPoint) + 96);
    std::set<uint64_t> features;
    for (uint64_t i = 0; i < count; ++i) {
        const auto id = in.scalar<uint64_t>(); DirectionalPoint point;
        point.direction = {finiteDouble(in), finiteDouble(in), finiteDouble(in)};
        if (!id || finite.points3D.count(id) || result.count(id) || std::abs(point.direction.norm() - 1) > 1e-10)
            throw std::runtime_error("directional export identity/type/unit norm invalid");
        const auto rows = in.count(8); bytes = storage::addBytes(bytes, storage::scaleBytes(rows, sizeof(TrackElement) + 48));
        if (!memory || bytes > memory) throw std::length_error("directional native Reader exceeds admitted memory");
        uint32_t firstPhoto=UINT32_MAX; bool secondPhoto=false; uint64_t previousFeature=0; point.track.reserve(size_t(rows));
        for (uint64_t r = 0; r < rows; ++r) {
            TrackElement row{in.scalar<uint32_t>(), in.scalar<uint32_t>()};
            const auto featureKey=(uint64_t(row.image_id)<<32)|row.point2D_idx;
            if(r && featureKey<=previousFeature) throw std::runtime_error("directional native Reader track order/uniqueness differs");
            previousFeature=featureKey;
            const auto image = finite.images.find(row.image_id);
            if (image == finite.images.end() || row.point2D_idx >= image->second.points2D.size() ||
                image->second.point3D_ids.at(row.point2D_idx) != kInvalidPoint3D ||
                !features.insert(featureKey).second)
                throw std::runtime_error("directional native Reader feature identity is conflicting/missing");
            point.track.push_back(row); if(firstPhoto==UINT32_MAX) firstPhoto=row.image_id; else secondPhoto=secondPhoto || row.image_id!=firstPhoto;
        }
        if (!secondPhoto) throw std::runtime_error("directional landmark lacks two distinct supporting photos");
        result.emplace(id, std::move(point));
    }
    in.finish(); return result;
}
inline std::map<uint64_t,std::vector<TrackElement>> strictReadRetained(const fs::path& output,const Reconstruction& finite,
    const std::map<uint64_t,DirectionalPoint>& directions,uint64_t memory) {
    std::map<uint64_t,std::vector<TrackElement>> result; std::set<uint64_t> features;
    uint64_t bytes=0;
    for(const auto& point:directions) for(const auto& row:point.second.track) {
        bytes=storage::addBytes(bytes,48); if(!memory || bytes>memory)throw std::length_error("retained Reader direction-disjointness exceeds admitted memory");
        features.insert((uint64_t(row.image_id)<<32)|row.point2D_idx);
    }
    model_store_detail::Reader in(output/"retained-observations.bin");
    if(in.scalar<uint64_t>()!=0x315445524e414353ull || in.scalar<uint32_t>()!=1 || in.scalar<uint32_t>()!=2)
        throw std::runtime_error("unknown retained original-measurement schema");
    const auto count=in.count(24); bytes=storage::addBytes(bytes,storage::scaleBytes(count,sizeof(std::vector<TrackElement>)+96));
    if(count && (!memory || bytes>memory))throw std::length_error("retained native Reader exceeds admitted memory");
    for(uint64_t i=0;i<count;++i) {
        const auto id=in.scalar<uint64_t>(),rows=in.count(8);
        if(!id || !rows || finite.points3D.count(id) || directions.count(id) || result.count(id))
            throw std::runtime_error("retained original-measurement identity/track invalid");
        bytes=storage::addBytes(bytes,storage::scaleBytes(rows,sizeof(TrackElement)+48));
        if(!memory || bytes>memory)throw std::length_error("retained native Reader exceeds admitted memory");
        auto& track=result[id]; track.reserve(size_t(rows)); uint32_t onlyPhoto=UINT32_MAX; uint64_t previous=0;
        for(uint64_t r=0;r<rows;++r) {
            TrackElement row{in.scalar<uint32_t>(),in.scalar<uint32_t>()}; const uint64_t key=(uint64_t(row.image_id)<<32)|row.point2D_idx;
            const auto image=finite.images.find(row.image_id);
            if((r && key<=previous) || image==finite.images.end() || row.point2D_idx>=image->second.points2D.size() ||
                image->second.point3D_ids.at(row.point2D_idx)!=kInvalidPoint3D || !features.insert(key).second ||
                (r && onlyPhoto!=row.image_id))throw std::runtime_error("retained original measurement gained geometry or conflicting/multiple views");
            previous=key; onlyPhoto=row.image_id; track.push_back(row);
        }
    }
    in.finish(); return result;
}
inline void sameParameters(const Reconstruction& expected, Reconstruction& actual,const VerifiedExportOptions& options) {
    if (expected.numRegistered() != actual.numRegistered() || expected.cameras.size() != actual.cameras.size())
        throw std::runtime_error("formal export changed camera/image coverage");
    for (const auto& item : expected.cameras) {
        const auto& a = item.second; auto& b = actual.cameras.at(item.first); double x[12]{}, y[12]{};
        packColmap(a,x); packColmap(b,y);
        if (a.id != b.id || a.model != b.model || a.width != b.width || a.height != b.height)
            throw std::runtime_error("formal export changed camera identity/model/dimensions");
        for (int k = 0; k < camColmapParams(a.model); ++k) if (x[k] != y[k]) throw std::runtime_error("formal export changed intrinsics");
        // Pixel scale is an immutable extraction-unit conversion, not an
        // additional camera parameter in the COLMAP projection model.
        if (!(a.pixel_scale > 0) || !std::isfinite(a.pixel_scale)) throw std::runtime_error("unknown raw extraction pixel scale");
        b.pixel_scale = a.pixel_scale;
    }
    for (const auto& item : expected.images) if (item.second.registered) {
        const auto& a = item.second; const auto& b = actual.images.at(item.first);
        if (a.id != b.id || a.camera_id != b.camera_id || exportName(item.first,expected,options) != b.name || a.pose.t.x != b.pose.t.x ||
            a.pose.t.y != b.pose.t.y || a.pose.t.z != b.pose.t.z)
            throw std::runtime_error("formal export changed image identity/translation");
        for (size_t k = 0; k < 9; ++k) if (std::abs(a.pose.R[k] - b.pose.R[k]) > 1e-12)
            throw std::runtime_error("formal pose quaternion serialization changed rotation");
    }
}
inline void projection(Quality& q, PhotoQuality& photo, const Camera& camera, const Image& image, Vec2 xy, Vec3 xyz, double gate, LandmarkKind kind = LandmarkKind::FiniteXYZ) {
    if(kind==LandmarkKind::RetainedObservation) {
        validateLandmark({xyz,kind}); ++q.observations; ++q.retained_observations; ++photo.retained_observations; return;
    }
    bool valid = true; const double e = reprojLandmarkError(camera,image.pose,xy,{xyz,kind},&valid) / camera.pixel_scale;
    if (!std::isfinite(e)) throw std::runtime_error("nonfinite original raw formal residual");
    const double value = e <= gate ? e * e : gate * (2 * e - gate);
    q.cost += value; q.finite_cost += valid ? value : 0; ++q.observations; q.invalid_projections += !valid;
    q.directional_observations += kind==LandmarkKind::UnitDirection;
    ++photo.observations; photo.inliers += valid && e <= gate; photo.invalid_projections += !valid; photo.error += e;
}
template<class Catalog> void verifyOriginalRawLedger(const Catalog& catalog, const CanonicalScene& scene, const fs::path& output,
    const Reconstruction& actual, const std::map<uint64_t, DirectionalPoint>& directions,
    const std::map<uint64_t, std::vector<TrackElement>>& retained, const VerifiedExportOptions& options,
    uint64_t workingBytes, VerifiedExportReport& report) {
    constexpr uint64_t margin = 2u << 16;
    if (workingBytes <= margin) throw std::length_error("formal raw Reader has no admitted working space");
    CanonicalParameterView parameters(scene, size_t(std::min<uint64_t>(workingBytes / 2, SIZE_MAX)));
    report.reader_parameter_cache_bytes = parameters.bytes();
    const uint64_t fixed = storage::addBytes(parameters.bytes(), margin);
    if (fixed >= workingBytes) throw std::length_error("formal raw Reader parameter cache exhausts working space");
    struct Entry { RegionalWorkingStore::RawObservation raw; RawAlias alias; CanonicalLandmark expected; };
    const uint64_t rowBytes = sizeof(Entry) + sizeof(PhotoQuality) + 160;
    const size_t capacity = size_t(std::min<uint64_t>({65536, scene.observations(), (workingBytes - fixed) / rowBytes}));
    if (!capacity) throw std::length_error("formal raw Reader batch exceeds admitted working space");
    std::vector<Entry> batch; batch.reserve(capacity);
    model_store_detail::Reader aliases(output / "raw-alias-map.bin", 65536);
    if (aliases.scalar<uint64_t>() != 0x3150414d57415253ull || aliases.scalar<uint64_t>() != scene.observations()) throw std::runtime_error("raw alias map header changed");
    auto& pool = bacpu::Pool::get(); const int workers = options.threads > 0 ? std::min(options.threads, pool.size()) : pool.size();
    report.raw = {}; report.raw.finite_cost = 0; report.raw_photos.clear(); report.finite_raw = report.directional_raw = report.retained_raw = 0; report.reader_raw_batches = 0;
    uint64_t readRows = 0;
    auto flush = [&] {
        if (batch.empty()) return;
        const int tasks = int((batch.size() + 2047) / 2048);
        struct Sum { Quality quality; std::map<uint32_t, PhotoQuality> photos; uint64_t finite = 0, directional = 0, retained = 0; };
        std::vector<Sum> sums; sums.resize(size_t(tasks)); for (auto& sum : sums) sum.quality.finite_cost = 0;
        pool.runChecked(tasks, workers, [&](int task, int) {
            cancel::check(); auto& sum = sums[size_t(task)]; const size_t end = std::min(batch.size(), (size_t(task) + 1) * 2048);
            for (size_t i = size_t(task) * 2048; i < end; ++i) {
                const auto& item = batch[i]; const auto& row = item.raw; const auto& expected = item.expected; Vec3 value{};
                if (expected.kind == LandmarkKind::FiniteXYZ) { value = actual.points3D.at(item.alias.point).xyz; ++sum.finite; }
                else if (expected.kind == LandmarkKind::UnitDirection) { value = directions.at(item.alias.point).direction; ++sum.directional; }
                else if (expected.kind == LandmarkKind::RetainedObservation) { (void)retained.at(item.alias.point); ++sum.retained; }
                else throw std::runtime_error("formal original raw references an unknown landmark state");
                if (value.x != expected.value.x || value.y != expected.value.y || value.z != expected.value.z)
                    throw std::runtime_error("formal export changed authoritative typed landmark parameters");
                const auto& image = actual.images.at(row.image);
                if (row.feature >= image.points2D.size() || image.points2D[row.feature].x != row.xy.x || image.points2D[row.feature].y != row.xy.y ||
                    (expected.kind == LandmarkKind::FiniteXYZ ? image.point3D_ids.at(row.feature) != item.alias.point : image.point3D_ids.at(row.feature) != kInvalidPoint3D))
                    throw std::runtime_error("original raw row lost formal feature/typed landmark association");
                if (expected.kind != LandmarkKind::FiniteXYZ) {
                    const auto& track = expected.kind == LandmarkKind::UnitDirection ? directions.at(item.alias.point).track : retained.at(item.alias.point);
                    const uint64_t key = (uint64_t(row.image) << 32) | row.feature;
                    const auto found = std::lower_bound(track.begin(), track.end(), key, [](const auto& entry, uint64_t value) { return ((uint64_t(entry.image_id) << 32) | entry.point2D_idx) < value; });
                    if (found == track.end() || found->image_id != row.image || found->point2D_idx != row.feature)
                        throw std::runtime_error("original raw row missing from formal typed feature association");
                }
                projection(sum.quality, sum.photos[row.image], actual.cameras.at(image.camera_id), image, row.xy, value, options.max_error, expected.kind);
            }
        });
        for (const auto& sum : sums) {
            accumulateQuality(report.raw, sum.quality); report.finite_raw += sum.finite; report.directional_raw += sum.directional; report.retained_raw += sum.retained;
            for (const auto& item : sum.photos) {
                auto& photo = report.raw_photos[item.first]; const auto& from = item.second;
                photo.observations += from.observations; photo.retained_observations += from.retained_observations; photo.inliers += from.inliers; photo.invalid_projections += from.invalid_projections; photo.error += from.error;
            }
        }
        ++report.reader_raw_batches; batch.clear();
        if (options.reportphase) options.reportphase("native-reader-raw", report.raw.observations, scene.observations());
    };
    if (options.reportphase) options.reportphase("native-reader-raw", 0, scene.observations());
    catalog.visitObservations([&](const auto& row) {
        if (!(readRows & 65535)) cancel::check(); const auto alias = aliases.scalar<RawAlias>(); const uint64_t root = parameters.canonical(row.point);
        if (alias.row != readRows++ || alias.point != root + 1) throw std::runtime_error("raw source/ordinal to formal point mapping changed");
        batch.push_back({row, alias, parameters.landmark(root)}); if (batch.size() == capacity) flush();
    });
    flush(); aliases.finish();
    if (readRows != scene.observations() || report.raw.observations != readRows || report.finite_raw + report.directional_raw + report.retained_raw != readRows)
        throw std::runtime_error("formal native Reader original raw/type count incomplete");
}
} // namespace verified_export_detail

// The exact same typed VKFT/header and metadata bounds are used by the initial
// cohort admission and by the serializer. No alias build or solver is needed.
template<class Catalog,class FeaturePath>
VerifiedExportForecast forecastVerifiedExport(const Catalog& catalog,const Reconstruction& poses,FeaturePath featurePath,
    const VerifiedExportOptions& options,const fs::path& scratch,const fs::path& output) {
    using namespace verified_export_detail;
    uint64_t raw=0,slots=0,features=0,framing=64,maxPath=0,maxLabel=0,sourceViews=0;
    for(const auto& input:options.inputs)maxPath=std::max<uint64_t>(maxPath,input.second.path.size());
    maxPath=std::max<uint64_t>(maxPath,std::max(scratch.u8string().size(),output.u8string().size()));
    for(const auto& image:poses.images)if(image.second.registered){uint32_t count=0;
        if(!peekFeatures(featurePath(image.first).string(),count))throw std::runtime_error("formal feature VKFT capacity header invalid/incomplete");
        features=storage::addBytes(features,count);framing=storage::addBytes(framing,73+exportName(image.first,poses,options).size());}
    for(const auto& camera:poses.cameras)framing=storage::addBytes(framing,24+8*camColmapParams(camera.second.model));
    for(size_t source=0;source<catalog.sourceCount();++source){
        raw=storage::addBytes(raw,catalog.sourceObservationCount(source));slots=storage::addBytes(slots,catalog.sourcePointCount(source));
        sourceViews=storage::addBytes(sourceViews,catalog.sourceViews(source).numRegistered());maxLabel=std::max<uint64_t>(maxLabel,catalog.sourceLabel(source).size());}
    const uint64_t perPhoto=storage::addBytes(1024,storage::scaleBytes(storage::addBytes(4096,2*maxPath+128),32));
    uint64_t metadata=storage::addBytes(uint64_t(1)<<20,storage::scaleBytes(options.inputs.empty()?poses.images.size():options.inputs.size(),perPhoto));
    metadata=storage::addBytes(metadata,storage::scaleBytes(sourceViews,2*maxLabel+2*maxPath+1024));
    metadata=storage::addBytes(metadata,storage::scaleBytes(catalog.sourceCount(),2*maxLabel+8192));
    return verifiedExportForecast(raw,slots,features,framing,metadata);
}

// Read-only export of one accepted canonical snapshot. The raw mapping refers
// back to the immutable catalogue's deterministic source/ordinal order; it does
// not create a second raw XY database or a second mutable XYZ authority.
template<class Scene, class Catalog, class FeaturePath>
VerifiedExportReport exportVerifiedScene(const Catalog& catalog, const Scene& scene, FeaturePath featurePath,
    const fs::path& scratch, const fs::path& output, const VerifiedExportOptions& options, const ExportFinalGate& finalGate) {
    using namespace verified_export_detail;
    VerifiedExportReport report; report.raw.finite_cost = 0;
    const auto& poses = scene.poses(); const auto snapshot = scene.acceptedSnapshot();
    auto allFail = [&](const std::string& reason) {
        for(auto& row:report.failures)if(row.image==UINT32_MAX && row.reason==reason){++row.occurrences;return;}
        report.failures.push_back({UINT32_MAX,"",reason});
    };
    auto checkpoint=[&](const char* phase){uint64_t observed=0;
        for(const auto& directory:{scratch,output})if(fs::exists(directory))for(const auto& file:fs::recursive_directory_iterator(directory))
            if(file.is_regular_file())observed=storage::addBytes(observed,file.file_size());
        report.observed_owned_bytes_at_checkpoints=std::max(report.observed_owned_bytes_at_checkpoints,observed);
        if(report.forecast.required_peak_bytes && observed>report.forecast.required_peak_bytes)throw std::runtime_error("observed generated formal files exceed export capacity forecast");
        if(options.phase_bytes)options.phase_bytes(phase,observed,report.forecast.required_peak_bytes);};
    if (fs::exists(output) && !fs::is_empty(output)) {
        allFail("formal export requires a new isolated output directory; existing artifact retained"); return report;
    }
    fs::create_directories(output);
    try {
        if (scene.editing() || snapshot.identity != scene.identity() || snapshot.generation != scene.generation() ||
            !options.cache_bytes || !(options.max_error > 0) || !std::isfinite(options.max_error))
            throw std::runtime_error("formal export requires a bound accepted canonical scene snapshot");
        if(!poses.rigs.empty() || !poses.rig_detached.empty())
            throw std::runtime_error("verified formal export rig metadata codec is unavailable; rig capture cannot be certified");
        scene.requireBinding(Scene::bindingFor(catalog, snapshot.binding.topology));
        if (!finalGate) allFail("formal export full acceptance gate is unknown");
        if (options.inputs.empty()) allFail("formal export complete original input inventory is unknown");
        if (fs::exists(scratch)) throw std::runtime_error("formal export scratch must be a new isolated stage directory");
        std::set<std::string> names,paths,formalNames;
        for(const auto& item:options.inputs) {
            if(item.second.path.empty() || item.second.name.empty() || !names.insert(item.second.name).second || !paths.insert(item.second.path).second)
                throw std::runtime_error("original formal export input names/paths are empty or duplicate");
            const auto formalName=exportName(item.first,poses,options);
            const auto relative=fs::u8path(formalName);
            bool traverses=false; for(const auto& part:relative) traverses=traverses || part=="..";
            if(formalName.empty() || relative.is_absolute() || traverses || !formalNames.insert(formalName).second ||
                relative.filename()!=fs::u8path(item.second.path).filename())
                throw std::runtime_error("formal original photo filename mapping is missing, conflicting or differs from the real input path");
            const auto at=poses.images.find(item.first);
            if(at==poses.images.end() || !at->second.registered) report.failures.push_back({item.first,item.second.path,"original input has no visually registered canonical pose"});
            else if(at->second.name!=item.second.name) report.failures.push_back({item.first,item.second.path,"canonical scene original input name changed"});
        }
        names.clear();
        for(const auto& item:poses.images) {
            if(!names.insert(item.second.name).second) throw std::runtime_error("formal scene contains duplicate-name inputs");
            if(!options.inputs.empty() && !options.inputs.count(item.first)) allFail("canonical scene contains an unknown original input ID");
        }
        report.forecast=forecastVerifiedExport(catalog,poses,featurePath,options,scratch,output);
        if(report.forecast.raw!=scene.observations() || report.forecast.point_slots!=scene.pointSlots())throw std::runtime_error("formal capacity source layout differs from accepted Scene");
        storage::requireSpace(scratch, report.forecast.required_peak_bytes, uint64_t(2) << 30);
        fs::create_directories(scratch);
        PagedArray<Node> nodes(scratch / "canonical-points.bin", options.cache_bytes / 2);
        for (uint64_t point = 0; point < scene.pointSlots(); ++point) { if (point % 65536 == 0) cancel::check(); nodes.append(Node{}); }
        uint64_t roots = 0; scene.visitCanonicalLandmarks([&](uint64_t id, CanonicalLandmark point) {
            validateLandmark(point);
            const auto xyz = point.value;
            if (id >= nodes.size() || nodes.get(id).present || !std::isfinite(xyz.norm())) throw std::runtime_error("invalid canonical export point identity");
            Node node{}; node.present = 1; node.xyz = xyz; node.padding[0] = uint8_t(point.kind); nodes.set(id,node); ++roots;
        });
        if (roots != scene.canonicalPoints()) throw std::runtime_error("canonical export point coverage changed");
        const auto records = scratch / "raw-feature-records.bin";
        std::ofstream rows(records,std::ios::binary), lineage(output / "raw-alias-map.bin",std::ios::binary);
        record(lineage,uint64_t(0x3150414d57415253ull)); record(lineage,scene.observations());
        uint64_t rawIndex = 0; std::vector<uint64_t> sourceRows(scene.sourceCount());
        catalog.visitObservations([&](const auto& row) {
            if (rawIndex % 65536 == 0) cancel::check();
            if (row.point.source >= sourceRows.size() || row.ordinal != sourceRows[row.point.source]++) throw std::runtime_error("immutable raw source/ordinal order changed");
            const uint64_t root = scene.canonical(row.point); auto node = nodes.get(root);
            if (!node.present) throw std::runtime_error("raw row maps to unknown canonical point");
            if (node.present==1) { const auto appearance=catalog.pointAppearance(row.point);
                for (int c=0;c<3;++c) node.rgb[c]=appearance.rgb[c]; node.present=2; nodes.set(root,node); }
            const auto& image = poses.images.at(row.image); const auto& camera = poses.cameras.at(image.camera_id);
            if (!image.registered || !std::isfinite(row.xy.x) || !std::isfinite(row.xy.y)) throw std::runtime_error("unknown raw image/XY");
            record(rows,Observation{(uint64_t(row.image)<<32)|row.feature,root,row.xy});
            record(lineage,RawAlias{rawIndex++,root+1});
        });
        rows.flush(); lineage.flush(); if (!rows || !lineage) throw std::runtime_error("cannot flush original raw alias map"); rows.close(); lineage.close();
        if (rawIndex != scene.observations()) throw std::runtime_error("immutable raw lineage full count changed");
        checkpoint("raw-generated");
        for (size_t s = 0; s < sourceRows.size(); ++s) if (sourceRows[s] != snapshot.binding.sources.at(s).observations)
            throw std::runtime_error("immutable raw source multiplicity changed");
        report.lineage_rows = rawIndex;
        std::ofstream binding(output / "raw-alias-binding.txt"); binding << "schema=immutable-raw-alias-v1\nscene=" << snapshot.identity
            << "\ngeneration=" << snapshot.generation << "\nraw_catalog=" << snapshot.binding.rawCatalog << "\ntopology=" << snapshot.binding.topology
            << "\nlandmark_kind_sha256=" << snapshot.kindHash << '\n';
        uint64_t sourceFirst = 0; for (size_t s = 0; s < sourceRows.size(); ++s) {
            const auto& source = snapshot.binding.sources.at(s); binding << "source=" << s << ",first=" << sourceFirst << ",count=" << source.observations
                << ",points=" << source.points << ",digest=" << source.digest << ",label=" << source.label << '\n'; sourceFirst += source.observations;
        }
        binding << "alias_sha256=" << spirula::sha256_file((output / "raw-alias-map.bin").u8string()) << '\n'; binding.flush();
        if (!binding) throw std::runtime_error("cannot flush immutable raw alias binding");
        const auto byFeature = sortObservations(records,scratch / "by-feature",options.cache_bytes / 2);
        checkpoint("feature-sorted");
        fs::remove(records); // Created and closed by this isolated export stage.
        const auto uniqueFeatures = scratch / "unique-feature-records.bin", uniqueTracks = scratch / "unique-track-records.bin";
        { std::ifstream in(byFeature,std::ios::binary); std::ofstream features(uniqueFeatures,std::ios::binary), tracks(uniqueTracks,std::ios::binary);
          Observation row{}, previous{}; bool have = false;
          while (nextRecord(in,row)) {
            if (have && row.key == previous.key) {
                if (row.point != previous.point || row.xy.x != previous.xy.x || row.xy.y != previous.xy.y)
                    throw std::runtime_error("one original feature has conflicting canonical point/XY aliases");
                continue;
            }
            record(features,row); record(tracks,Observation{row.point,row.key,row.xy}); previous=row; have=true;
          }
          features.flush(); tracks.flush(); if (!features || !tracks) throw std::runtime_error("cannot write unique canonical export tracks"); }
        checkpoint("unique-features-generated");
        fs::remove(byFeature);
        const auto byPoint = sortObservations(uniqueTracks,scratch / "by-point",options.cache_bytes / 2);
        checkpoint("point-sorted");
        fs::remove(uniqueTracks);
        report.stats.images = poses.numRegistered(); std::map<uint32_t,uint32_t> parent, regionParent;
        std::map<uint32_t,uint64_t> uniqueCounts,landmarkCounts,singleViewPoints;
        for (const auto& item : poses.images) { parent[item.first]=item.first; uniqueCounts[item.first]=0; landmarkCounts[item.first]=0; }
        auto component = [&](uint32_t image) { while (parent.at(image)!=image) { parent[image]=parent.at(parent.at(image)); image=parent.at(image); } return image; };
        if (options.owners) for (const auto& item:poses.images) {
            const uint32_t owner=options.owners->at(item.first);
            if (owner==UINT32_MAX) throw std::runtime_error("formal input photo has unknown region owner");
            regionParent[owner]=owner;
        }
        auto regionComponent=[&](uint32_t region) { while(regionParent.at(region)!=region) {
            regionParent[region]=regionParent.at(regionParent.at(region)); region=regionParent.at(region); } return region; };
        const auto residuals=scratch / "unique-residuals.bin";
        std::ofstream residualOut(residuals,std::ios::binary);
        { std::ifstream in(byPoint,std::ios::binary); std::ofstream out(output / "points3D.bin",std::ios::binary), directions(output / "directional-points.bin",std::ios::binary), retained(output / "retained-observations.bin",std::ios::binary); detail::wr<uint64_t>(out,0);
          detail::wr<uint64_t>(directions,0x315249444e414353ull); detail::wr<uint32_t>(directions,1); detail::wr<uint32_t>(directions,1); detail::wr<uint64_t>(directions,0);
          detail::wr<uint64_t>(retained,0x315445524e414353ull); detail::wr<uint32_t>(retained,1); detail::wr<uint32_t>(retained,2); detail::wr<uint64_t>(retained,0);
          Observation row{}; bool have=nextRecord(in,row);
          while (have) {
            cancel::check(); const uint64_t point=row.key; const auto node=nodes.get(point); std::vector<Observation> track;
            std::set<uint32_t> views; double sum=0, scaled=0;
            do { const uint32_t image=uint32_t(row.point>>32); views.insert(image);
                 track.push_back(row); have=nextRecord(in,row); } while (have && row.key==point);
            const auto kind = LandmarkKind(node.padding[0]);
            if(kind==LandmarkKind::RetainedObservation) {
                if(views.size()!=1 || node.xyz.x!=0 || node.xyz.y!=0 || node.xyz.z!=0)
                    throw std::runtime_error("retained original measurement has observable geometry or nonzero payload");
                detail::wr(retained,point+1); detail::wr<uint64_t>(retained,track.size());
                for(const auto& obs:track) {const uint32_t image=uint32_t(obs.point>>32);
                    detail::wr<uint32_t>(retained,image); detail::wr<uint32_t>(retained,uint32_t(obs.point));
                    ++report.retained_unique_photos[image];}
                ++report.retained_points; report.retained_unique=storage::addBytes(report.retained_unique,track.size()); continue;
            }
            if (kind != LandmarkKind::FiniteXYZ && kind != LandmarkKind::UnitDirection) throw std::runtime_error("unknown export landmark kind");
            if (views.size()<2) for(uint32_t image:views)++singleViewPoints[image];
            std::set<uint32_t> inlierViews;
            for (const auto& obs:track) {
                const uint32_t image=uint32_t(obs.point>>32); const auto& im=poses.images.at(image); const auto& cam=poses.cameras.at(im.camera_id);
                bool valid=true; const double error=reprojLandmarkError(cam,im.pose,obs.xy,{node.xyz,kind},&valid);
                if (!std::isfinite(error)) throw std::runtime_error("nonfinite canonical export projection");
                const double extractionError=error/cam.pixel_scale;
                auto& visual=report.unique_quality[image]; ++visual.observations; visual.inliers+=valid && extractionError<=options.max_error;
                visual.invalid_projections+=!valid; visual.error+=extractionError;
                record(residualOut,error);
                if(kind == LandmarkKind::FiniteXYZ && valid && extractionError<=options.max_error) inlierViews.insert(image);
                sum+=error; scaled+=extractionError; if(views.size()>=2) ++uniqueCounts[image];
            }
            if(inlierViews.size()>=2) {
                const uint32_t root=component(*inlierViews.begin());
                for(uint32_t image:inlierViews) {++landmarkCounts[image]; parent[component(image)]=root;}
            }
            if (options.owners && inlierViews.size()>=2) {
                const uint32_t first=options.owners->at(*inlierViews.begin()); bool bridge=false;
                for(uint32_t image:inlierViews) { const uint32_t owner=options.owners->at(image);
                    bridge=bridge || owner!=first; regionParent[regionComponent(owner)]=regionComponent(first); }
                report.stats.cross_region_points+=bridge;
            }
            if (kind == LandmarkKind::FiniteXYZ) {
                detail::wr(out,point+1); for (double value:{node.xyz.x,node.xyz.y,node.xyz.z}) detail::wr(out,value);
                for (int c=0;c<3;++c) detail::wr(out,node.rgb[c]); detail::wr(out,sum/track.size()); detail::wr<uint64_t>(out,track.size());
                for (const auto& obs:track) { detail::wr<uint32_t>(out,uint32_t(obs.point>>32)); detail::wr<uint32_t>(out,uint32_t(obs.point)); }
                ++report.stats.points;
            } else {
                detail::wr(directions,point+1); for (double value:{node.xyz.x,node.xyz.y,node.xyz.z}) detail::wr(directions,value);
                detail::wr<uint64_t>(directions,track.size());
                for (const auto& obs:track) { detail::wr<uint32_t>(directions,uint32_t(obs.point>>32)); detail::wr<uint32_t>(directions,uint32_t(obs.point)); }
                ++report.directional_points; report.directional_unique += track.size();
            }
            report.stats.observations+=track.size(); report.stats.error+=sum; report.stats.extraction_error+=scaled;
          }
          out.seekp(0); detail::wr(out,report.stats.points); out.flush(); if (!out) throw std::runtime_error("cannot flush canonical points3D.bin");
          directions.seekp(16); detail::wr(directions,report.directional_points); directions.flush(); if (!directions) throw std::runtime_error("cannot flush typed directional-points.bin");
          retained.seekp(16); detail::wr(retained,report.retained_points); retained.flush(); if(!retained)throw std::runtime_error("cannot flush retained-observations.bin"); }
        checkpoint("points-written");
        for(const auto& item:singleViewPoints)report.failures.push_back({item.first,inputPath(item.first,poses,options),"canonical point lacks two distinct supporting views",item.second,"point"});
        fs::remove(byPoint);
        residualOut.flush(); if(!residualOut) throw std::runtime_error("cannot flush unique formal residuals"); residualOut.close();
        if(report.stats.observations) {
            const auto ordered=scene_index_detail::sortFile<double>(residuals,scratch / "residual-sort",
                std::max<size_t>(8u<<20,options.cache_bytes/2),[](double a,double b){return a<b;});
            { model_store_detail::Reader values(ordered);
              const uint64_t middle=report.stats.observations/2;
              values.seek((middle-(report.stats.observations%2==0))*sizeof(double));
              const double left=values.scalar<double>();
              report.stats.median=report.stats.observations%2?left:(left+values.scalar<double>())/2;
              values.seek(fs::file_size(ordered)); values.finish(); }
            checkpoint("median-sorted");
            fs::remove(ordered);
        }
        fs::remove(residuals);
        { std::ifstream in(uniqueFeatures,std::ios::binary); Observation row{}; bool have=nextRecord(in,row);
          std::ofstream out(output / "images.bin",std::ios::binary); detail::wr(out,report.stats.images);
          for (const auto& item:poses.images) {
            cancel::check(); const auto& im=item.second; if(!im.registered) continue;
            const auto f=readFeatures(featurePath(im.id).string(),false);
            const auto& camera=poses.cameras.at(im.camera_id);
            if(f.width!=camera.width || f.height!=camera.height ||
                std::abs(f.pixelScale()-camera.pixel_scale)>std::max(1e-12,std::abs(camera.pixel_scale)*1e-12))
                throw std::runtime_error("formal source image dimensions/extraction pixel scale differ from accepted camera metadata");
            std::vector<uint64_t> ids(f.count(),kInvalidPoint3D);
            while (have && uint32_t(row.key>>32)==im.id) {
                const uint32_t key=uint32_t(row.key);
                if (key>=ids.size() || row.xy.x!=f.keypoints[key].x || row.xy.y!=f.keypoints[key].y)
                    throw std::runtime_error("formal export source feature identity/XY changed");
                if (nodes.get(row.point).padding[0] == uint8_t(LandmarkKind::FiniteXYZ)) ids[key]=row.point+1;
                have=nextRecord(in,row);
            }
            detail::wr(out,im.id); const auto q=rotationToQuaternion(im.pose.R);
            for (double v:{q[0],q[1],q[2],q[3],im.pose.t.x,im.pose.t.y,im.pose.t.z}) detail::wr(out,v);
            const auto formalName=exportName(im.id,poses,options);
            detail::wr(out,im.camera_id); out.write(formalName.c_str(),formalName.size()+1); detail::wr<uint64_t>(out,f.count());
            for (size_t k=0;k<f.count();++k) { detail::wr<double>(out,f.keypoints[k].x); detail::wr<double>(out,f.keypoints[k].y); detail::wr(out,ids[k]); }
          }
          if (have) throw std::runtime_error("formal export raw feature references unknown input"); out.flush(); if (!out) throw std::runtime_error("cannot flush canonical images.bin"); }
        checkpoint("images-written");
        fs::remove(uniqueFeatures);
        { std::ofstream out(output / "cameras.bin",std::ios::binary); detail::wr<uint64_t>(out,poses.cameras.size());
          for (const auto& item:poses.cameras) { const auto& camera=item.second; detail::wr(out,camera.id); detail::wr<int32_t>(out,camColmapId(camera.model));
            detail::wr<uint64_t>(out,camera.width); detail::wr<uint64_t>(out,camera.height); double values[12]{}; packColmap(camera,values);
            for (int k=0;k<camColmapParams(camera.model);++k) detail::wr(out,values[k]); }
          out.flush(); if (!out) throw std::runtime_error("cannot flush canonical cameras.bin"); }
        nodes.flush();
        if (report.stats.points + report.directional_points + report.retained_points != roots) allFail("canonical landmarks without original raw support remain");
        report.stats.min_photo_observations=poses.images.empty()?0:UINT64_MAX;
        for (const auto& item:uniqueCounts) { report.stats.photo_components+=component(item.first)==item.first;
            report.stats.photos_without_points+=landmarkCounts.at(item.first)==0; report.stats.min_photo_observations=std::min(report.stats.min_photo_observations,item.second);
            if (landmarkCounts.at(item.first)<12) report.failures.push_back({item.first,inputPath(item.first,poses,options),"insufficient distinct canonical visual landmarks: "+std::to_string(landmarkCounts.at(item.first))}); }
        for(const auto& item:poses.images) if(!photoQualityAccepted(report.unique_quality[item.first],options.max_error))
            report.failures.push_back({item.first,inputPath(item.first,poses,options),"unique original-feature per-photo projection quality failed"});
        if(options.owners) for(const auto& item:regionParent) report.stats.components+=regionComponent(item.first)==item.first;
        else report.stats.components=report.stats.photo_components;
        report.stats.error=report.stats.observations?report.stats.error/report.stats.observations:0;
        report.stats.extraction_error=report.stats.observations?report.stats.extraction_error/report.stats.observations:0;
        report.unique_photos=uniqueCounts;
        report.unique_landmarks=landmarkCounts;
        std::map<std::string,std::string> formalHashes;
        uint64_t hashedFiles = 0; if (options.reportphase) options.reportphase("native-reader-file-sha", 0, 6);
        for (const auto& name:{"cameras.bin","images.bin","points3D.bin","raw-alias-map.bin","directional-points.bin","retained-observations.bin"}) {
            formalHashes[name]=spirula::sha256_file((output/name).u8string());
            if (options.reportphase) options.reportphase("native-reader-file-sha", ++hashedFiles, 6);
        }
        if (options.reportphase) options.reportphase("native-reader-load", 0, 3);
        Reconstruction actual=strictReadBack(output,options.reader_memory_bytes); sameParameters(poses,actual,options);
        if (options.reportphase) options.reportphase("native-reader-load", 1, 3);
        const auto finiteBytes = modelResidentBytes(actual);
        const auto directions = strictReadDirections(output,actual,options.reader_memory_bytes > finiteBytes ? options.reader_memory_bytes - finiteBytes : 0);
        if (options.reportphase) options.reportphase("native-reader-load", 2, 3);
        if (directions.size() != report.directional_points) throw std::runtime_error("directional Reader landmark coverage differs");
        uint64_t directionBytes=storage::scaleBytes(directions.size(),sizeof(DirectionalPoint)+96);
        for(const auto& point:directions)directionBytes=storage::addBytes(directionBytes,storage::scaleBytes(point.second.track.capacity(),sizeof(TrackElement)));
        const auto typedBytes=storage::addBytes(finiteBytes,directionBytes);
        const auto retained=strictReadRetained(output,actual,directions,options.reader_memory_bytes>typedBytes?options.reader_memory_bytes-typedBytes:0);
        if (options.reportphase) options.reportphase("native-reader-load", 3, 3);
        if(retained.size()!=report.retained_points)throw std::runtime_error("retained Reader root coverage differs");
        uint64_t finiteUnique=0,directionalUnique=0,retainedUnique=0;
        for(const auto& point:actual.points3D) finiteUnique=storage::addBytes(finiteUnique,point.second.track.size());
        for(const auto& point:directions) directionalUnique=storage::addBytes(directionalUnique,point.second.track.size());
        for(const auto& point:retained)retainedUnique=storage::addBytes(retainedUnique,point.second.size());
        if(actual.points3D.size()!=report.stats.points || directionalUnique!=report.directional_unique ||
            storage::addBytes(finiteUnique,directionalUnique)!=report.stats.observations || retainedUnique!=report.retained_unique)
            throw std::runtime_error("formal native Reader invented/lost unique finite, directional or retained tracks");
        uint64_t retainedBytes = storage::scaleBytes(retained.size(), sizeof(std::vector<TrackElement>) + 96);
        for (const auto& point : retained) retainedBytes = storage::addBytes(retainedBytes, storage::scaleBytes(point.second.capacity(), sizeof(TrackElement)));
        const uint64_t loadedBytes = storage::addBytes(typedBytes, retainedBytes);
        verifyOriginalRawLedger(catalog, scene, output, actual, directions, retained, options,
            options.reader_memory_bytes > loadedBytes ? options.reader_memory_bytes - loadedBytes : 0, report);
        auto& photos = report.raw_photos;
        const auto& expected=options.accepted_raw;
        if (!expected.observations || expected.observations!=report.raw.observations || !std::isfinite(expected.cost) || !std::isfinite(expected.finite_cost) ||
            expected.retained_observations!=report.raw.retained_observations || expected.directional_observations!=report.raw.directional_observations ||
            expected.invalid_projections!=report.raw.invalid_projections || std::abs(expected.cost-report.raw.cost)>std::max(1e-8,expected.cost*1e-8) ||
            std::abs(expected.finite_cost-report.raw.finite_cost)>std::max(1e-8,expected.finite_cost*1e-8))
            throw std::runtime_error("formal native Reader differs from complete accepted raw objective");
        report.reader_verified=true;
        if (report.raw.invalid_projections) allFail("original raw invalid projection remains");
        for (const auto& item:poses.images) if (!photoQualityAccepted(photos[item.first],options.max_error))
            report.failures.push_back({item.first,inputPath(item.first,poses,options),"complete original raw per-photo projection quality failed"});
        if (report.stats.photo_components!=1 || report.stats.photos_without_points) allFail("formal visual track connectivity incomplete");
        if (finalGate) {
            const auto external=finalGate(actual,report.raw,photos,report.stats); std::map<uint32_t,ExportConstraintPhoto> decisions;
            for (const auto& decision:external.photos) {
                if ((!options.inputs.empty()? !options.inputs.count(decision.image):!poses.images.count(decision.image)) || !decisions.emplace(decision.image,decision).second)
                    throw std::runtime_error("formal external acceptance has unknown/duplicate input photo identities");
            }
            report.constraints=decisions;
            std::map<uint32_t,ExportInputPhoto> gateInventory=options.inputs;
            if(gateInventory.empty()) for(const auto& item:poses.images) gateInventory[item.first]={inputPath(item.first,poses,options),item.second.name};
            for (const auto& item:gateInventory) {
                const auto at=decisions.find(item.first); const auto path=item.second.path;
                if (at==decisions.end()) { report.failures.push_back({item.first,path,"GPS/scale/attitude acceptance unknown for this input"}); continue; }
                const auto& decision=at->second;
                if (decision.path!=path) report.failures.push_back({item.first,path,"external acceptance original input path differs"});
                if (!decision.gps_verified || !std::isfinite(decision.gps_error_metres) || decision.gps_error_metres<0 || decision.gps_error_metres>5)
                    report.failures.push_back({item.first,path,"GPS <= 5m not verified"});
                if (!decision.scale_verified || !std::isfinite(decision.scale_error_fraction) || decision.scale_error_fraction<0 || decision.scale_error_fraction>.01)
                    report.failures.push_back({item.first,path,"scale <= 1% not verified"});
                if (!decision.attitude_verified || !std::isfinite(decision.attitude_error_degrees) || decision.attitude_error_degrees<0 || decision.attitude_error_degrees>10)
                    report.failures.push_back({item.first,path,"original attitude <= 10deg not verified"});
                for (const auto& reason:decision.reasons) report.failures.push_back({item.first,path,reason});
            }
            if (!external.historical_evaluated || external.historical_required!=options.historical_required ||
                (options.historical_required && !external.historical_verified)) allFail("required historical source certification is unknown/failed");
        }
        report.verified=report.failures.empty();
        if (snapshot.identity!=scene.identity() || snapshot.generation!=scene.generation() || scene.editing())
            throw std::runtime_error("accepted canonical snapshot changed during export acceptance");
        for (const auto& item:formalHashes) if (item.second!=spirula::sha256_file((output/item.first).u8string()))
            throw std::runtime_error("formal file changed after its native Reader acceptance");
        std::ofstream hashes(output / "formal-file-sha256.txt");
        for (const auto& name:{"cameras.bin","images.bin","points3D.bin","raw-alias-map.bin","directional-points.bin","retained-observations.bin"}) hashes<<name<<'='<<spirula::sha256_file((output/name).u8string())<<'\n';
        hashes.flush(); if (!hashes) throw std::runtime_error("cannot flush actual formal file digests");
        checkpoint("native-reader-complete");
    } catch (const std::exception& error) { allFail(error.what()); report.verified=false; report.reader_verified=false; }
    failureReport(output,report,snapshot.identity,poses,options);
    return report;
}
} // namespace sfm::regional
