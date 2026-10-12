"""Compile the C contract, audit runtime definitions, and inventory legacy APIs."""
import json
import os
import re

from api_model import declarations, public_types
from content import SECTIONS


def check_contract(root, destination, directory, run, require, skip_vulkan=False):
    commands = [cmd for cmd in declarations() if not (skip_vulkan and cmd["extension"])]
    source = '#include "parallel_mater/parallel_mater.h"\n'
    if not skip_vulkan:
        source += '#include "parallel_mater/interop_vulkan.h"\n'
    types = {name for section in SECTIONS for card in section.cards
             if not (skip_vulkan and card.extension) for name in public_types(card.types)}
    # Compile complete host structures/enums, aliases, callbacks and opaque
    # handle values. Opaque Pm*_T internals intentionally have no public sizeof.
    source += "\n".join(f'typedef char check_{name}[(sizeof({name}) > 0) ? 1 : -1];'
                        for name in sorted(types))
    source += '\nint main(void) {\n'
    for cmd in commands:
        name = cmd["name"]
        source += f'    PFN_{name} check_{name} = {name};\n'
        source += f'    (void)check_{name};\n'
    source += '    return 0;\n}\n'
    test = directory / "all_commands.c"
    test.write_text(source)
    for compiler, language, standard in (
            (os.environ.get("CC", "cc"), "c", "c11"),
            (os.environ.get("CXX", "c++"), "c++", "c++17")):
        run([compiler, "-x", language, f"-std={standard}", "-Wall", "-Wextra",
             "-Werror", "-pedantic", "-Iinclude", "-fsyntax-only", str(test)])

    definitions = {}
    for path in sorted((root / "src/api").glob("*.cpp")):
        text = path.read_text()
        for match in re.finditer(
                r'extern\s+"C"\s+PMAPI_ATTR\s+[^\n]*?\b(pm\w+)\s*\(', text):
            definitions.setdefault(match.group(1), []).append(str(path.relative_to(root)))
    expected = {command["name"] for command in commands}
    require(expected <= definitions.keys(),
            f"Runtime definitions missing: {sorted(expected - definitions.keys())}")
    require(all(len(definitions[name]) == 1 for name in expected),
            "Runtime commands must have exactly one source definition")
    print(f"PASS: {len(commands)} C/C++ declarations have typed pointers and one runtime definition; "
          f"{len(types)} complete public types")


def legacy_inventory(root):
    """Name-level source audit, NOT backend compilation or overload conformance."""
    backends = {
        "cuda": ("parallel_mater.hpp", ["src/world.cu"]),
        "metal": ("metal.hpp", ["src/metal/world.mm", "src/metal/systems.mm"]),
        "d3d12": ("d3d12.hpp", ["src/d3d12/world.cpp", "src/d3d12/stubs.cpp"]),
    }
    inventory = {}
    for backend, (header, paths) in backends.items():
        text = (root / "include/parallel_mater" / header).read_text()
        world = text.split("class World {", 1)[1].split("private:", 1)[0]
        methods = sorted(set(re.findall(r"\[\[nodiscard\]\][^;{}]*?\b(\w+)\s*\(", world)))
        sources = [(path, (root/path).read_text()) for path in paths]
        records = []
        for method in methods:
            locations = []
            for path, code in sources:
                matches = list(re.finditer(r"\bWorld::" + method + r"\s*\(", code))
                matches += list(re.finditer(r"\bPM_(?:ADD|REMOVE|UPDATE)_STUB\(\s*" + method + r"\s*,", code))
                matches += list(re.finditer(r"\bPM_SYSTEM_(?:(?:UNREVISED_)?FORWARD_[123]|(?:REVISION_)?VIEW)\(\s*"
                                           + method + r"\s*,", code))
                locations.extend(f"{path}:{code.count(chr(10), 0, match.start())+1}" for match in matches)
            records.append({"method": method, "definition_candidates": locations,
                            "unsupported_stub": bool(locations) and
                            all(location.startswith("src/d3d12/stubs.cpp:") for location in locations)})
        inventory[backend] = {"header": "include/parallel_mater/"+header,
                              "world_methods": records,
                              "backend_types": sorted(set(re.findall(
                                  r"^(?:struct|enum class|class) (\w+)", text, re.M)))}
        frame = text.split("class FrameToken {", 1)[1].split("private:", 1)[0]
        frame_names = sorted(set(re.findall(r"\[\[nodiscard\]\][^;{}]*?\b(\w+)\s*\(", frame)))
        inventory[backend]["frame_methods"] = [{"method": name,
            "definition_candidates": [f"{path}:{code.count(chr(10), 0, match.start())+1}"
                for path, code in sources for match in re.finditer(r"\bFrameToken::"+name+r"\s*\(", code)]}
            for name in frame_names]
        helpers = sorted(set(re.findall(r"^\[\[nodiscard\]\] Status (\w+)\(", text, re.M)))
        helper_sources = [(path, (root/path).read_text()) for path in
                          ("src/rope_geometry.cpp", "src/soft_body_geometry.cpp",
                           "src/fluid_geometry.cpp", "src/fluid_source.cpp")]
        inventory[backend]["host_utilities"] = [{"function": name,
            "definition_candidates": [f"{path}:{code.count(chr(10), 0, match.start())+1}"
                for path, code in helper_sources for match in re.finditer(r"\bStatus "+name+r"\s*\(", code)]}
            for name in helpers]
    inventory["shared_types"] = sorted(set(re.findall(r"^(?:struct|enum class) (\w+)",
        (root/"include/parallel_mater/types.hpp").read_text(), re.M)))
    inventory["scope"] = ("World/FrameToken method-name, host utility and public-type inventory; overload signatures, "
                           "constructors/operators, native SDK execution and conformance are not inferred.")
    return inventory


if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(legacy_inventory(root), indent=2))
