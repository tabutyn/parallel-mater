#!/usr/bin/env python3
"""Generate loader and driver glue from the Parallel Mater API registry."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "reference"))

from api_model import declarations  # noqa: E402


SPECIAL = {
    "pmGetInstanceProcAddr", "pmGetDeviceProcAddr",
    "pmEnumerateInstanceVersion", "pmCreateInstance", "pmDestroyInstance",
    "pmEnumeratePhysicalDevices", "pmCreateDevice", "pmDestroyDevice",
    "pmGetDeviceQueue", "pmAllocateCommandBuffers", "pmFreeCommandBuffers",
    "pmQueueSubmit", "pmImportNativeDevice", "pmImportVulkanDeviceEXT",
    "pmEnumerateInstanceLayerProperties",
    "pmEnumerateInstanceExtensionProperties",
    "pmEnumerateDeviceExtensionProperties",
}

LOADER_ONLY = {
    "pmEnumerateInstanceLayerProperties",
    "pmEnumerateInstanceExtensionProperties",
}

DISPATCHABLE = {
    "PmInstance", "PmPhysicalDevice", "PmDevice", "PmQueue", "PmCommandBuffer"
}


def guarded(command, body):
    if command["extension"]:
        return "#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)\n" + body + "#endif\n"
    return body


def emit(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    commands = declarations()

    remap = ["// Generated from registry/pm.xml.\n#pragma once\n"]
    unmap = ["// Generated from registry/pm.xml.\n"]
    lookup = ["// Generated from registry/pm.xml.\n"]
    loader_lookup = ["// Generated from registry/pm.xml.\n"]
    forwards = ["// Generated from registry/pm.xml.\n"]

    for command in commands:
        name = command["name"]
        remap.append(f"#define {name} pm_driver_impl_{name}\n")
        unmap.append(f"#undef {name}\n")
        if name not in LOADER_ONLY:
            lookup.append(guarded(command,
                f'PM_DRIVER_LOOKUP({name}, pm_driver_impl_{name});\n'))
        loader_lookup.append(guarded(command,
            f'PM_LOADER_LOOKUP({name});\n'))

        if name in SPECIAL:
            continue
        first_type, first_name = command["params"][0]
        if first_type not in DISPATCHABLE:
            raise RuntimeError(f"{name}: no dispatchable first parameter")
        params = command["args"]
        args = [parameter[1] for parameter in command["params"]]
        native_args = [
            f"reinterpret_cast<{first_type}>(route.native)", *args[1:]
        ]
        if command["returns"] == "PmResult":
            invocation = f'''    const PmResult validation = parallel_mater::loader::pre_call(route, "{name}");
    if (validation != PM_SUCCESS) return validation;
    const PmResult result = function({", ".join(native_args)});
    parallel_mater::loader::post_call(route, "{name}", result);
    return result;'''
        elif command["returns"] == "void":
            invocation = f'''    if (parallel_mater::loader::pre_call(route, "{name}") != PM_SUCCESS)
        return;
    function({", ".join(native_args)});
    parallel_mater::loader::post_call(route, "{name}", PM_SUCCESS);
    return;'''
        else:
            invocation = f'''    return function({", ".join(native_args)});'''
        body = f'''extern "C" PMAPI_ATTR {command["returns"]} PMAPI_CALL {name}(
    {params}) {{
    const auto route = parallel_mater::loader::route({first_name});
    if (!route) return parallel_mater::loader::failure<{command["returns"]}>();
    auto function = reinterpret_cast<PFN_{name}>(
        route.driver->resolve(route, "{name}"));
    if (function == nullptr)
        return parallel_mater::loader::unsupported<{command["returns"]}>();
{invocation}
}}
'''
        forwards.append(guarded(command, body))

    (output / "driver_remap.h").write_text("".join(remap))
    (output / "driver_unmap.h").write_text("".join(unmap))
    (output / "driver_lookup.inc").write_text("".join(lookup))
    (output / "loader_lookup.inc").write_text("".join(loader_lookup))
    (output / "loader_forward.inc").write_text("".join(forwards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    emit(args.output)


if __name__ == "__main__":
    main()
