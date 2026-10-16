#include "sfm/core/FeatureIdentity.h"
#include "sfm/core/Resume.h"
#include "sfm/Pipeline.h"
#include "sfm/tests/TestMain.h"
#include "external/stb_image_write.h"
#include <chrono>
#include <fstream>
#ifdef _WIN32
#include <winioctl.h>
#endif

namespace fs = std::filesystem;
static void require(bool good, const char* message) { if (!good) throw std::runtime_error(message); }

static void requiredPipeline(const fs::path& root) {
    const auto photos = root / "photos", workspace = root / "pipeline";
    fs::create_directories(photos);
    fs::create_directories(workspace / "features");
    std::vector<unsigned char> pixels(32 * 32 * 3, 128);
    for (const char* name : {"a", "b"}) {
        const auto image = photos / (std::string(name) + ".png");
        require(stbi_write_png(image.string().c_str(), 32, 32, 3, pixels.data(), 32 * 3) != 0, "cannot write fixture image");
        sfm::FeatureSet features;
        features.width = features.extract_width = features.height = features.extract_height = 32;
        features.dim = 8; features.dtype = sfm::DType::U8;
        features.keypoints.resize(1); features.descriptors.resize(8);
        sfm::writeFeatures((workspace / "features" / (std::string(name) + ".bin")).string(), features);
    }
    sfm::SfmConfig cfg;
    cfg.quiet = true;
    cfg.reuse_features = cfg.reuse_matches = "require";
    cfg.finalize(sfm::CMD_AUTO);
    const auto extract = sfm::stageSignature(cfg, sfm::CMD_EXTRACT) + sfm::resume::kSignedImages + photos.string() + "\n";
    const auto resume = sfm::resume::dir(workspace.string());
    sfm::resume::store(resume / sfm::resume::kExtractSig, extract);
    sfm::resume::store(resume / sfm::resume::kMatchSig, "incompatible matching identity");
    std::ofstream(workspace / "matches.bin") << "preserved matches";
    std::ofstream(workspace / "features/stale.bin") << "preserved extra features";
    sfm::AutoInputs input; input.image_dir = photos.string(); input.workspace = workspace.string();
    const auto before = sfm::resume::featureDirectoryIdentity(workspace / "features");
    bool rejected = false;
    try { sfm::run_auto(cfg, input); }
    catch (const std::runtime_error& e) { rejected = std::string(e.what()).find("required matching reuse") != std::string::npos; }
    require(rejected, "required matching reuse executed matching after an incompatible signature");
    require(fs::file_size(workspace / "matches.bin") == 17 &&
        before == sfm::resume::featureDirectoryIdentity(workspace / "features"), "required matching reuse changed cached files");
    fs::last_write_time(photos / "a.png", fs::last_write_time(workspace / "features/a.bin") + std::chrono::seconds(1));
    rejected = false;
    try { sfm::run_auto(cfg, input); }
    catch (const std::runtime_error& e) { rejected = std::string(e.what()).find("required feature reuse") != std::string::npos; }
    require(rejected && before == sfm::resume::featureDirectoryIdentity(workspace / "features"),
        "changed image silently replaced or retimed required features");
}

static void aliasDirectory(const fs::path& target, const fs::path& alias) {
#ifdef _WIN32
    fs::create_directory(alias);
    const std::wstring substitute = L"\\??\\" + target.native(), print = target.native();
    struct Junction {
        DWORD tag; WORD length, reserved, substitute_offset, substitute_length, print_offset, print_length;
        wchar_t names[4096];
    } buffer{};
    buffer.tag = IO_REPARSE_TAG_MOUNT_POINT;
    buffer.substitute_length = WORD(substitute.size() * sizeof(wchar_t));
    buffer.print_offset = WORD(buffer.substitute_length + sizeof(wchar_t));
    buffer.print_length = WORD(print.size() * sizeof(wchar_t));
    buffer.length = WORD(8 + buffer.print_offset + buffer.print_length + sizeof(wchar_t));
    require(buffer.length < sizeof(buffer.names), "junction fixture is too large");
    std::copy(substitute.begin(), substitute.end(), buffer.names);
    std::copy(print.begin(), print.end(), buffer.names + substitute.size() + 1);
    HANDLE handle = CreateFileW(alias.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    require(handle != INVALID_HANDLE_VALUE, "cannot open junction fixture");
    DWORD returned = 0;
    const bool good = DeviceIoControl(handle, FSCTL_SET_REPARSE_POINT, &buffer, buffer.length + 8,
        nullptr, 0, &returned, nullptr) != 0;
    CloseHandle(handle);
    require(good, "cannot create junction fixture");
#else
    fs::create_directory_symlink(target, alias);
#endif
}

static int testIdentity(int argc, char** argv) {
    if (argc == 3) {
        const auto original = sfm::resume::featureDirectoryIdentity(fs::u8path(argv[1]));
        const auto alias = sfm::resume::featureDirectoryIdentity(fs::u8path(argv[2]));
        require(original == alias, "real capture directory alias changed feature identity");
        std::printf("original=%s alias=%s\n", original.c_str(), alias.c_str());
        return 0;
    }
    const auto root = fs::temp_directory_path() / ("sfm-feature-identity-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto source = root / fs::u8path(u8"原特征"), alias = root / "linked-features";
    fs::create_directories(source / "flight");
    const auto feature = source / "flight/image.bin";
    std::ofstream(feature, std::ios::binary) << "keypoint identities";
    aliasDirectory(source, alias);
    require(fs::equivalent(source, alias), "fixture is not the same physical directory");
    const auto first = sfm::resume::featureDirectoryIdentity(source);
    require(sfm::resume::featureIdentityRoot(source) == source, "ordinary directory lost legacy spelling");
    require(first == sfm::resume::featureDirectoryIdentity(alias), "junction/symlink invalidated matches and mapping");
    fs::last_write_time(feature, fs::last_write_time(feature) + std::chrono::seconds(1));
    require(first != sfm::resume::featureDirectoryIdentity(alias), "re-extracted features were accepted as unchanged");
    const auto changed = sfm::resume::featureDirectoryIdentity(source);
    std::ofstream(source / "added.bin") << "new photo";
    require(changed != sfm::resume::featureDirectoryIdentity(alias), "added photo failed to invalidate matches");
    bool rejected = false;
    try { sfm::resume::featureDirectoryIdentity(root / "missing"); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, "failed feature enumeration generated a partial identity");
    sfm::resume::requireReusable("require", true, "matching");
    rejected = false;
    try { sfm::resume::requireReusable("require", false, "matching"); }
    catch (const std::runtime_error& e) {
        rejected = std::string(e.what()).find("no automatic recomputation") != std::string::npos;
    }
    require(rejected && fs::file_size(feature) == 19, "required reuse mutated original features or silently recomputed");
    sfm::resume::requireReusable("auto", false, "matching");
    requiredPipeline(root);
    fs::remove(alias);
    fs::remove_all(root);
    std::puts("PASS");
    return 0;
}
int main(int argc, char** argv) { return sfmTestMain(argc, argv, testIdentity); }
