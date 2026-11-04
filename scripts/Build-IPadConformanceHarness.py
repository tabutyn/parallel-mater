#!/usr/bin/env python3
"""Build a signed iPad command harness around frozen candidate libraries."""

import argparse
import os
import pathlib
import shutil
import subprocess


def run(*command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-build", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--team", required=True)
    args = parser.parse_args()

    source = args.candidate_source.resolve()
    products = (args.candidate_build / "Release-iphoneos").resolve()
    gallery = products / "ParallelMater Gallery.app"
    if not gallery.is_dir():
        parser.error(f"candidate Gallery is missing: {gallery}")
    for name in ("ParallelMater.framework", "ParallelMaterMetal.framework",
                 "ParallelMaterValidation.framework"):
        if not (products / name).is_dir():
            parser.error(f"candidate framework is missing: {products / name}")

    output = args.output.resolve()
    project = output / "project"
    build = output / "build"
    shutil.rmtree(output, ignore_errors=True)
    project.mkdir(parents=True)
    harness = pathlib.Path(__file__).resolve().parents[1] / \
        "examples/ios/conformance_harness_main.mm"
    producer = pathlib.Path(__file__).resolve().parents[1] / \
        "conformance/v1/src/realtime_workload.cpp"
    info = project / "Info.plist.in"
    cmake = project / "CMakeLists.txt"
    xcconfig = project / "NoSDKStatCache.xcconfig"
    xcconfig.write_text("SDK_STAT_CACHE_ENABLE = NO\n", encoding="utf-8")
    os.environ["XCODE_XCCONFIG_FILE"] = str(xcconfig)
    info.write_text("""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\"><dict>
<key>CFBundleDevelopmentRegion</key><string>en</string>
<key>CFBundleExecutable</key><string>ParallelMaterConformance</string>
<key>CFBundleIdentifier</key><string>com.ballrollergames.parallelmater.gallery</string>
<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
<key>CFBundleName</key><string>ParallelMaterConformance</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>LSRequiresIPhoneOS</key><true/>
<key>UILaunchScreen</key><dict/>
<key>UISupportedInterfaceOrientations~ipad</key><array><string>UIInterfaceOrientationLandscapeLeft</string></array>
</dict></plist>
""", encoding="utf-8")
    libraries = [
        "libparallel-mater-conformance-runner.a",
        "libparallel-mater-conformance-registry.a",
        "libparallel-mater-scene.a",
        "libparallel-mater-host-preparation.a",
    ]
    links = "\n".join(f'  "{products / name}"' for name in libraries)
    cmake.write_text(f"""cmake_minimum_required(VERSION 3.27)
project(ParallelMaterIPadConformance LANGUAGES CXX OBJCXX)
add_executable(ParallelMaterConformance MACOSX_BUNDLE
  "{harness}" "{producer}")
target_compile_features(ParallelMaterConformance PRIVATE cxx_std_20)
target_include_directories(ParallelMaterConformance PRIVATE
  "{source / 'conformance/v1/include'}"
  "{source / 'include'}"
  "{producer.parent}")
target_link_libraries(ParallelMaterConformance PRIVATE
{links}
  "{products / 'ParallelMater.framework/ParallelMater'}"
  "-framework UIKit" "-framework Foundation" "-framework Metal")
set_target_properties(ParallelMaterConformance PROPERTIES
  MACOSX_BUNDLE_INFO_PLIST "{info}"
  XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
  XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "{args.team}"
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER
    "com.ballrollergames.parallelmater.gallery"
  XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "2"
  XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "26.0"
  XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@executable_path/Frameworks")
add_custom_command(TARGET ParallelMaterConformance POST_BUILD
  COMMAND ${{CMAKE_COMMAND}} -E copy_directory
    "{gallery / 'Frameworks'}" "$<TARGET_BUNDLE_DIR:ParallelMaterConformance>/Frameworks"
  COMMAND ${{CMAKE_COMMAND}} -E copy_directory
    "{gallery / 'manifests'}" "$<TARGET_BUNDLE_DIR:ParallelMaterConformance>/manifests"
  COMMAND ${{CMAKE_COMMAND}} -E copy_directory
    "{gallery / 'config'}" "$<TARGET_BUNDLE_DIR:ParallelMaterConformance>/config"
  COMMAND ${{CMAKE_COMMAND}} -E copy_directory
    "{gallery / 'assets'}" "$<TARGET_BUNDLE_DIR:ParallelMaterConformance>/assets"
  COMMAND ${{CMAKE_COMMAND}} -E copy_directory
    "{source / 'conformance/v1'}" "$<TARGET_BUNDLE_DIR:ParallelMaterConformance>/conformance/v1")
""", encoding="utf-8")

    run("cmake", "-S", str(project), "-B", str(build), "-G", "Xcode",
        "-DCMAKE_SYSTEM_NAME=iOS", "-DCMAKE_OSX_SYSROOT=iphoneos",
        "-DCMAKE_OSX_ARCHITECTURES=arm64")
    run("cmake", "--build", str(build), "--config", "Release", "--target",
        "ParallelMaterConformance", "--", "-allowProvisioningUpdates")
    app = build / "Release-iphoneos/ParallelMaterConformance.app"
    if not app.is_dir():
        raise RuntimeError(f"signed harness was not produced: {app}")
    print(app)


if __name__ == "__main__":
    main()
