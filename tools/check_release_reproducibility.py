#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build twice and record comparable Parallel Mater release manifests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys


SCHEMA = "parallel-mater-release-reproducibility/v1"
MANIFEST_SCHEMA = "parallel-mater-release-manifest/v1"
TEXT_RUNTIME_SUFFIXES = {
    ".cmake", ".conf", ".cfg", ".ini", ".json", ".manifest", ".pc",
    ".plist", ".txt", ".xml", ".yaml", ".yml",
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_output(command, cwd=None):
    return subprocess.check_output(
        [str(part) for part in command], cwd=cwd, text=True,
        stderr=subprocess.STDOUT).strip()


def executable_identity(path, version_command=None):
    resolved = Path(path).resolve()
    result = {"name": resolved.name, "sha256": sha256_file(resolved)}
    if version_command:
        result["version"] = run_output([resolved, *version_command]).splitlines()[0]
    return result


def source_identity(root):
    revision = run_output(["git", "rev-parse", "HEAD"], root)
    status = run_output(["git", "status", "--porcelain", "--untracked-files=all"], root)
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\0")
    digest = hashlib.sha256()
    count = 0
    for relative in sorted(path for path in tracked if path):
        data = (root / relative).read_bytes()
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
        count += 1
    return {
        "revision": revision,
        "dirty": bool(status),
        "tracked_file_count": count,
        "tracked_tree_sha256": digest.hexdigest(),
    }


def parse_cache(build_dir):
    values = {}
    cache = build_dir / "CMakeCache.txt"
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        left, value = line.split("=", 1)
        if ":" not in left:
            continue
        key, kind = left.rsplit(":", 1)
        values[key] = {"type": kind, "value": value}
    return values


def replace_path(value, path, marker):
    for spelling in (str(path.resolve()), path.resolve().as_posix()):
        value = value.replace(spelling, marker)
        value = value.replace(spelling.lower(), marker)
    return value


def portable_value(value, root, build_dir, install_dir):
    normalized = value
    for path, marker in ((build_dir, "<build>"), (install_dir, "<install>"),
                         (root, "<source>")):
        normalized = replace_path(normalized, path, marker)
    # External absolute tool paths are inputs by identity, not runtime paths.
    parts = normalized.split(";")
    for index, part in enumerate(parts):
        candidate = Path(part)
        if (candidate.is_absolute() or re.match(r"^[A-Za-z]:[/\\]", part)) and "<" not in part:
            parts[index] = f"<external>/{candidate.name}"
    return ";".join(parts).replace("\\", "/")


def build_options(cache, root, build_dir, install_dir):
    exact = {
        "BUILD_TESTING", "CMAKE_BUILD_TYPE", "CMAKE_GENERATOR",
        "CMAKE_GENERATOR_PLATFORM", "CMAKE_GENERATOR_TOOLSET",
        "CMAKE_INSTALL_LIBDIR", "CMAKE_MSVC_RUNTIME_LIBRARY",
    }
    flag_prefixes = (
        "CMAKE_CXX_FLAGS", "CMAKE_CUDA_FLAGS", "CMAKE_EXE_LINKER_FLAGS",
        "CMAKE_SHARED_LINKER_FLAGS", "CMAKE_MODULE_LINKER_FLAGS",
    )
    selected = {}
    for key, record in sorted(cache.items()):
        if (key.startswith("PARALLEL_MATER_") or key in exact or
                key.startswith(flag_prefixes)):
            selected[key] = {
                "type": record["type"],
                "value": portable_value(record["value"], root, build_dir, install_dir),
            }
    return selected


def compiler_metadata(build_dir):
    results = []
    seen = set()
    pattern = re.compile(r'^set\((CMAKE_[A-Z]+_COMPILER(?:_[A-Z_]+)?) "(.*)"\)$')
    for path in sorted((build_dir / "CMakeFiles").glob("*/CMake*Compiler.cmake")):
        language_match = re.match(r"CMake([A-Z]+)Compiler\.cmake", path.name)
        if not language_match or language_match.group(1) in seen:
            continue
        language = language_match.group(1)
        values = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line)
            if match:
                values[match.group(1)] = match.group(2)
        prefix = f"CMAKE_{language}_COMPILER"
        compiler = values.get(prefix)
        if not compiler or not Path(compiler).is_file():
            continue
        record = {
            "language": language,
            "id": values.get(prefix + "_ID", "unknown"),
            "version": values.get(prefix + "_VERSION", "unknown"),
            "executable": executable_identity(compiler),
        }
        for suffix, name in (("_ARCHITECTURE_ID", "architecture"),
                             ("_TARGET", "target")):
            if values.get(prefix + suffix):
                record[name] = values[prefix + suffix]
        results.append(record)
        seen.add(language)
    if not results:
        raise RuntimeError("no configured compiler metadata found")
    return results


def toolchain_programs(cache):
    programs = {}
    for key in ("CMAKE_AR", "CMAKE_LINKER", "CMAKE_MT", "CMAKE_RC_COMPILER",
                "CMAKE_RANLIB"):
        value = cache.get(key, {}).get("value")
        if value and Path(value).is_file():
            programs[key] = executable_identity(value)
    return programs


def platform_sdk_metadata(cache):
    sdks = {}
    resource_compiler = cache.get("CMAKE_RC_COMPILER", {}).get("value", "")
    windows_match = re.search(
        r"[\\/]Windows Kits[\\/]10[\\/]bin[\\/]([^\\/]+)[\\/]",
        resource_compiler, re.IGNORECASE)
    if windows_match:
        sdks["windows"] = {"version": windows_match.group(1)}
    osx_sysroot = cache.get("CMAKE_OSX_SYSROOT", {}).get("value")
    if osx_sysroot:
        sdk = {"name": Path(osx_sysroot).name}
        try:
            sdk["version"] = run_output(["xcrun", "--sdk", osx_sysroot,
                                         "--show-sdk-version"])
        except (OSError, subprocess.CalledProcessError):
            pass
        sdks["apple"] = sdk
    return sdks


def resolved_dependency_versions(cache):
    prefixes = ("CUDAToolkit", "Vulkan", "glfw", "OptiX")
    versions = {}
    for key, record in sorted(cache.items()):
        if key.endswith(("_VERSION", "_VERSION_STRING")) and key.startswith(prefixes):
            versions[key] = record["value"]
    return versions


def declared_dependencies(root, cache):
    slang_text = (root / "cmake/Slang.cmake").read_text(encoding="utf-8")
    version_match = re.search(
        r'set\(PARALLEL_MATER_SLANG_VERSION\s+"([^"]+)"', slang_text)
    if not version_match:
        raise RuntimeError("could not read the pinned Slang version")
    archives = []
    archive_pattern = re.compile(
        r'"slang-\$\{PARALLEL_MATER_SLANG_VERSION\}-([^"]+)"\)\s*'
        r'set\(PARALLEL_MATER_SLANG_SHA256\s*([0-9a-f]{64})\)', re.MULTILINE)
    for archive, digest in archive_pattern.findall(slang_text):
        archives.append({"archive_suffix": archive, "sha256": digest})
    if not archives:
        raise RuntimeError("could not read the pinned Slang archive checksums")

    cgltf_readme = (root / "third_party/cgltf/README.parallel-mater.md").read_text(
        encoding="utf-8")
    cgltf_match = re.search(r"commit\s*`([0-9a-f]{40})`", cgltf_readme)
    if not cgltf_match:
        raise RuntimeError("could not read the pinned cgltf revision")

    cmake_text = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    fetch_content = []
    for match in re.finditer(r"FetchContent_Declare\(\s*([A-Za-z0-9_]+)(.*?)\)",
                             cmake_text, re.DOTALL):
        body = match.group(2)
        repository = re.search(r"GIT_REPOSITORY\s+(\S+)", body)
        revision = re.search(r"GIT_TAG\s+(\S+)", body)
        if repository and revision:
            fetch_content.append({
                "name": match.group(1),
                "repository": repository.group(1),
                "revision": revision.group(1),
            })

    dependencies = [
        {
            "name": "cgltf",
            "kind": "vendored",
            "revision": cgltf_match.group(1),
            "files": {
                "cgltf.h": sha256_file(root / "third_party/cgltf/cgltf.h"),
                "LICENSE": sha256_file(root / "third_party/cgltf/LICENSE"),
            },
        },
        {
            "name": "Slang",
            "kind": "pinned-release-archives",
            "version": version_match.group(1),
            "archives": archives,
            "selected": bool(cache.get("PARALLEL_MATER_SLANGC_EXECUTABLE", {}).get("value")),
        },
    ]
    dependencies.extend({"kind": "git", **item} for item in fetch_content)

    slang_path = cache.get("PARALLEL_MATER_SLANGC_EXECUTABLE", {}).get("value")
    if slang_path and Path(slang_path).is_file():
        dependencies[1]["compiler"] = executable_identity(slang_path, ["-version"])
    return dependencies


def artifact_kind(data, path):
    if data.startswith(b"MZ"):
        return "pe"
    if data.startswith(b"\x7fELF"):
        return "elf"
    if data.startswith(b"!<arch>\n"):
        return "archive"
    if data[:4] in {b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf",
                    b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}:
        return "mach-o"
    if data.startswith(b"Microsoft C/C++ MSF") or path.suffix.lower() == ".pdb":
        return "pdb"
    if data.startswith(b"PK\x03\x04"):
        return "zip"
    if is_text(data):
        return "text"
    return "binary"


def is_text(data):
    if b"\0" in data[:8192]:
        return False
    try:
        text = data[:8192].decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not text:
        return True
    printable = sum(character.isprintable() or character in "\r\n\t" for character in text)
    return printable / len(text) >= 0.95


def inventory(root):
    if not root.is_dir():
        return []
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        data = path.read_bytes()
        records.append({
            "path": path.relative_to(root).as_posix(),
            "size": len(data),
            "sha256": sha256_bytes(data),
            "kind": artifact_kind(data, path),
        })
    return records


def generated_inventory(build_dir, configuration):
    records = {}
    roots = [(build_dir / "generated", "generated")]
    for candidate in (build_dir / "manifests", build_dir / configuration / "manifests"):
        roots.append((candidate, candidate.relative_to(build_dir).as_posix()))
    for root, prefix in roots:
        for record in inventory(root):
            record["path"] = f"{prefix}/{record['path']}"
            records[record["path"]] = record
    return [records[key] for key in sorted(records)]


def normalize_pe_timestamp(data):
    if len(data) < 0x40 or not data.startswith(b"MZ"):
        return data
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if offset + 12 > len(data) or data[offset:offset + 4] != b"PE\0\0":
        return data
    normalized = bytearray(data)
    normalized[offset + 8:offset + 12] = b"\0\0\0\0"
    return bytes(normalized)


def normalize_archive_headers(data):
    if not data.startswith(b"!<arch>\n"):
        return data
    normalized = bytearray(data)
    offset = 8
    while offset + 60 <= len(normalized):
        header = normalized[offset:offset + 60]
        if header[58:60] != b"`\n":
            return data
        try:
            size = int(header[48:58].decode("ascii").strip())
        except ValueError:
            return data
        normalized[offset + 16:offset + 48] = b" " * 32
        offset += 60 + size + (size % 2)
    return bytes(normalized)


def normalize_embedded_paths(data, paths):
    normalized = data
    for path in paths:
        for spelling in {str(path.resolve()), path.resolve().as_posix()}:
            normalized = normalized.replace(spelling.encode("utf-8"), b"<build-path>")
            normalized = normalized.replace(
                spelling.encode("utf-16-le"), "<build-path>".encode("utf-16-le"))
    return normalized


def classify_difference(first_path, second_path, kind, path_pairs=None):
    first = first_path.read_bytes()
    second = second_path.read_bytes()
    path_pairs = path_pairs or []
    normalized_first = normalize_embedded_paths(first, [pair[0] for pair in path_pairs])
    normalized_second = normalize_embedded_paths(second, [pair[1] for pair in path_pairs])
    paths_changed = normalized_first != first or normalized_second != second
    if normalized_first == normalized_second:
        return "embedded_build_path"
    if (kind == "pe" and
            normalize_pe_timestamp(normalized_first) == normalize_pe_timestamp(normalized_second)):
        return ("embedded_build_path_and_pe_coff_timestamp" if paths_changed
                else "pe_coff_timestamp")
    if (kind == "archive" and
            normalize_archive_headers(normalized_first) ==
            normalize_archive_headers(normalized_second)):
        return ("embedded_build_path_and_archive_member_metadata" if paths_changed
                else "archive_member_metadata")
    if paths_changed:
        return f"embedded_build_path_and_{kind}_content"
    if kind == "text":
        return "text_content"
    return f"{kind}_content"


def compare_inventory(first, second, first_root, second_root, path_pairs=None):
    first_records = {record["path"]: record for record in first}
    second_records = {record["path"]: record for record in second}
    added = sorted(set(second_records) - set(first_records))
    removed = sorted(set(first_records) - set(second_records))
    differences = []
    for relative in sorted(set(first_records) & set(second_records)):
        left = first_records[relative]
        right = second_records[relative]
        if left["sha256"] == right["sha256"]:
            continue
        kind = left["kind"] if left["kind"] == right["kind"] else "binary"
        differences.append({
            "path": relative,
            "classification": classify_difference(
                first_root / relative, second_root / relative, kind, path_pairs),
            "first": {"sha256": left["sha256"], "size": left["size"], "kind": left["kind"]},
            "second": {"sha256": right["sha256"], "size": right["size"], "kind": right["kind"]},
        })
    return {
        "artifact_sets_match": not added and not removed,
        "bit_identical": not added and not removed and not differences,
        "added": added,
        "removed": removed,
        "differences": differences,
    }


def audit_runtime_paths(install_dir, forbidden):
    required_hits = []
    binary_mentions = []
    encodings = []
    for value in forbidden:
        for spelling in {str(value.resolve()), value.resolve().as_posix()}:
            encodings.append((spelling, spelling.encode("utf-8")))
            encodings.append((spelling, spelling.encode("utf-16-le")))
    for path in sorted(item for item in install_dir.rglob("*") if item.is_file()):
        data = path.read_bytes()
        matches = sorted({spelling for spelling, encoded in encodings if encoded in data})
        if not matches:
            continue
        relative = path.relative_to(install_dir).as_posix()
        record = {"path": relative, "matched_path_kinds": len(matches)}
        if path.suffix.lower() in TEXT_RUNTIME_SUFFIXES or is_text(data):
            required_hits.append(record)
        else:
            binary_mentions.append(record)
    return {
        "status": "passed" if not required_hits else "failed",
        "required_path_hits": required_hits,
        "binary_path_mentions_diagnostic_only": binary_mentions,
    }


def capture_manifest(root, build_dir, install_dir, source, preset, configuration,
                     cmake_arguments):
    cache = parse_cache(build_dir)
    make_program = cache.get("CMAKE_MAKE_PROGRAM", {}).get("value")
    tools = {
        "cmake": executable_identity(shutil.which("cmake"), ["--version"]),
        "python": executable_identity(sys.executable, ["--version"]),
    }
    if make_program and Path(make_program).is_file():
        tools["build_program"] = executable_identity(make_program, ["--version"])
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source": source,
        "product": {
            "name": cache.get("CMAKE_PROJECT_NAME", {}).get("value", "ParallelMater"),
            "version": cache.get("CMAKE_PROJECT_VERSION", {}).get("value", "unknown"),
        },
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "configuration": {
            "preset": preset,
            "build_configuration": configuration,
            "cmake_arguments": cmake_arguments,
            "options": build_options(cache, root, build_dir, install_dir),
        },
        "tools": tools,
        "compilers": compiler_metadata(build_dir),
        "toolchain_programs": toolchain_programs(cache),
        "platform_sdks": platform_sdk_metadata(cache),
        "dependencies": declared_dependencies(root, cache),
        "resolved_dependency_versions": resolved_dependency_versions(cache),
        "artifacts": {
            "installed": inventory(install_dir),
            "generated": generated_inventory(build_dir, configuration),
        },
        "runtime_path_audit": audit_runtime_paths(
            install_dir, [root, build_dir, install_dir]),
    }
    serialized = json.dumps(manifest, sort_keys=True)
    for forbidden in (str(root.resolve()), root.resolve().as_posix(),
                      str(build_dir.resolve()), build_dir.resolve().as_posix(),
                      str(install_dir.resolve()), install_dir.resolve().as_posix()):
        if forbidden in serialized:
            raise RuntimeError("release manifest retained a workstation path")
    return manifest


def run_logged(name, command, root, output):
    log = output / f"{name}.log"
    with log.open("w", encoding="utf-8") as stream:
        stream.write("command: " + " ".join(str(part) for part in command) + "\n")
        process = subprocess.run(
            [str(part) for part in command], cwd=root, stdout=stream,
            stderr=subprocess.STDOUT, check=False)
    print(f"{'PASS' if process.returncode == 0 else 'FAIL'} {name}", flush=True)
    if process.returncode:
        raise RuntimeError(f"{name}: see {log}")


def build_once(label, args, root, output, source):
    build_dir = output / f"build-{label}"
    install_dir = output / f"install-{label}"
    configure = ["cmake", "--preset", args.preset, "-B", build_dir]
    recorded_arguments = []
    if args.generator:
        configure.extend(["-G", args.generator])
        recorded_arguments.append(f"-G{args.generator}")
    if args.architecture:
        configure.extend(["-A", args.architecture])
        recorded_arguments.append(f"-A{args.architecture}")
    configure.extend(args.cmake_arg)
    recorded_arguments.extend(args.cmake_arg)
    run_logged(f"configure-{label}", configure, root, output)
    run_logged(f"build-{label}", [
        "cmake", "--build", build_dir, "--config", args.config,
        "--parallel", str(args.jobs)], root, output)
    run_logged(f"install-{label}", [
        "cmake", "--install", build_dir, "--config", args.config,
        "--prefix", install_dir], root, output)
    manifest = capture_manifest(
        root, build_dir, install_dir, source, args.preset, args.config,
        recorded_arguments)
    (output / f"manifest-{label}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest, build_dir, install_dir


def comparable_inputs(manifest):
    return {key: manifest[key] for key in (
        "source", "product", "host", "configuration", "tools", "compilers",
        "toolchain_programs", "platform_sdks", "dependencies",
        "resolved_dependency_versions")}


def main():
    parser = argparse.ArgumentParser(
        description="Run two isolated builds and classify release artifact differences")
    parser.add_argument("--preset", default="sdk-ci")
    parser.add_argument("--output", type=Path, required=True,
                        help="New evidence directory (never overwritten)")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--generator",
                        help="Override the preset generator (for example Visual Studio)")
    parser.add_argument("--architecture",
                        help="Generator architecture used with --generator")
    parser.add_argument("--cmake-arg", action="append", default=[])
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Diagnostic only; clean source is required for release evidence")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = source_identity(root)
    if source["dirty"] and not args.allow_dirty:
        print("source tree is dirty; commit inputs or use --allow-dirty for diagnostics",
              file=sys.stderr)
        return 2
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": SCHEMA,
        "status": "failed",
        "scope": "release_inputs_and_artifact_reproducibility",
        "source": source,
    }
    try:
        first, first_build, first_install = build_once(
            "a", args, root, output, source)
        second, second_build, second_install = build_once(
            "b", args, root, output, source)
        installed = compare_inventory(
            first["artifacts"]["installed"], second["artifacts"]["installed"],
            first_install, second_install,
            [(first_build, second_build), (first_install, second_install)])
        generated = compare_inventory(
            first["artifacts"]["generated"], second["artifacts"]["generated"],
            first_build, second_build,
            [(first_build, second_build), (first_install, second_install)])
        inputs_match = comparable_inputs(first) == comparable_inputs(second)
        path_audits_pass = all(
            manifest["runtime_path_audit"]["status"] == "passed"
            for manifest in (first, second))
        bit_identical = installed["bit_identical"] and generated["bit_identical"]
        report.update({
            "status": "passed" if inputs_match and path_audits_pass else "failed",
            "inputs_match": inputs_match,
            "runtime_path_audits_pass": path_audits_pass,
            "installed_artifacts": installed,
            "generated_artifacts": generated,
            "bit_identical": bit_identical,
            "claim": (
                "Bit-identical installed and generated artifacts were observed for "
                "these two builds and this exact host/toolchain/input tuple."
                if bit_identical else
                "Bit-identical output is not established; all observed artifact "
                "differences are classified in this report."
            ),
        })
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        report["error"] = str(error)
        print(error, file=sys.stderr)
    finally:
        (output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] == "passed":
        print(report["claim"])
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
