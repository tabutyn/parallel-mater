"""Compile the public C contract and audit loader entry-point coverage."""
import os
import re

from api_model import declarations, public_types
from content import SECTIONS

LOADER_SPECIAL = {
    "pmGetInstanceProcAddr", "pmGetDeviceProcAddr",
    "pmEnumerateInstanceVersion", "pmCreateInstance", "pmDestroyInstance",
    "pmEnumeratePhysicalDevices", "pmCreateDevice", "pmDestroyDevice",
    "pmGetDeviceQueue", "pmAllocateCommandBuffers", "pmFreeCommandBuffers",
    "pmQueueSubmit", "pmImportNativeDevice", "pmImportVulkanDeviceEXT",
    "pmEnumerateInstanceLayerProperties",
    "pmEnumerateInstanceExtensionProperties",
    "pmEnumerateDeviceExtensionProperties",
}


def check_contract(root, destination, directory, run, require, compile_source,
                   skip_vulkan=False):
    del destination
    commands = [command for command in declarations()
                if not (skip_vulkan and command["extension"])]
    source = '#include "parallel_mater/parallel_mater.h"\n'
    if not skip_vulkan:
        source += '#include "parallel_mater/interop_vulkan.h"\n'
    types = {name for section in SECTIONS for card in section.cards
             if not (skip_vulkan and card.extension)
             for name in public_types(card.types)}
    source += "\n".join(
        f'typedef char check_{name}[(sizeof({name}) > 0) ? 1 : -1];'
        for name in sorted(types))
    source += '\nint main(void) {\n'
    for command in commands:
        name = command["name"]
        source += f'    PFN_{name} check_{name} = {name};\n'
        source += f'    (void)check_{name};\n'
    source += '    return 0;\n}\n'
    test = directory / "all_commands.c"
    test.write_text(source, encoding="utf-8", newline="\n")
    defaults = ("cl", "cl") if os.name == "nt" else ("cc", "c++")
    for compiler, language, standard in ((defaults[0], "c", "c11"),
                                          (defaults[1], "c++", "c++17")):
        compiler = os.environ.get("CC" if language == "c" else "CXX", compiler)
        compile_source(compiler, language, standard, test, "include")

    definitions = {}
    for path in sorted((root / "src/loader").glob("*.cpp")):
        text = path.read_text()
        for match in re.finditer(
                r'extern\s+"C"\s+PMAPI_ATTR\s+[\w*]+\s+PMAPI_CALL\s+'
                r'(pm\w+)\s*\(', text):
            definitions.setdefault(match.group(1), []).append(
                str(path.relative_to(root)))
    for command in commands:
        if command["name"] not in LOADER_SPECIAL:
            definitions.setdefault(command["name"], []).append(
                "generated/api/loader_forward.inc")
    expected = {command["name"] for command in commands}
    require(expected <= definitions.keys(),
            f"Runtime definitions missing: "
            f"{sorted(expected - definitions.keys())}")
    require(all(len(definitions[name]) == 1 for name in expected),
            "Runtime commands must have exactly one source definition")
    print(f"PASS: {len(commands)} C/C++ declarations have typed pointers and "
          f"one loader definition; {len(types)} complete public types")
