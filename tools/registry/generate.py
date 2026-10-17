#!/usr/bin/env python3
"""Generate loader and driver glue from the Parallel Mater API registry."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_registry import declarations, interaction_capabilities  # noqa: E402


MANUAL_LAYER_INTERCEPTED = {
    "pmEnumerateDeviceExtensionProperties": "physical-device extension dispatch",
    "pmDestroyInstance": "aggregate instance teardown",
    "pmEnumeratePhysicalDevices": "aggregate physical-device enumeration",
    "pmCreateDevice": "device wrapper creation",
    "pmDestroyDevice": "device wrapper teardown",
    "pmGetDeviceQueue": "queue wrapper lookup and creation",
    "pmAllocateCommandBuffers": "command-buffer wrapper creation",
    "pmFreeCommandBuffers": "command-buffer translation and teardown",
    "pmQueueSubmit": "command-buffer translation before submission",
    "pmImportNativeDevice": "native-device wrapper creation",
    "pmImportVulkanDeviceEXT": "Vulkan-device wrapper creation",
}

MANUAL_LAYER_EXCLUDED = {
    "pmGetInstanceProcAddr": "loader symbol discovery, not command execution",
    "pmGetDeviceProcAddr": "loader symbol discovery, not command execution",
    "pmEnumerateInstanceVersion": "global call with no enabled layer chain",
    "pmEnumerateInstanceLayerProperties": "global call with no enabled layer chain",
    "pmEnumerateInstanceExtensionProperties": "global call with no enabled layer chain",
    "pmCreateInstance": "selects the layer chain that later calls use",
}

SPECIAL = set(MANUAL_LAYER_INTERCEPTED) | set(MANUAL_LAYER_EXCLUDED)

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


def write_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def emit(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    commands = declarations()

    remap = ["// Generated from registry/pm.xml.\n#pragma once\n"]
    unmap = ["// Generated from registry/pm.xml.\n"]
    lookup = ["// Generated from registry/pm.xml.\n"]
    loader_lookup = ["// Generated from registry/pm.xml.\n"]
    layer_args = ['// Generated from registry/pm.xml. Do not edit.\n'
                  '#ifndef PARALLEL_MATER_LAYER_ARGS_H\n'
                  '#define PARALLEL_MATER_LAYER_ARGS_H\n'
                  '#include <parallel_mater/parallel_mater.h>\n'
                  '#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)\n'
                  '#include <parallel_mater/interop_vulkan.h>\n#endif\n']
    layer_dispatch = ['// Generated typed validation dispatch.\n']
    manual_layer_inventory = [
        "// Generated manual-entrypoint validation inventory. Do not edit.\n"]
    forwards = ["// Generated from registry/pm.xml.\n"]
    interaction_inventory = ["// Generated from registry/pm.xml.\n"]

    for command in commands:
        name = command["name"]
        remap.append(f"#define {name} pm_driver_impl_{name}\n")
        unmap.append(f"#undef {name}\n")
        if name not in LOADER_ONLY:
            lookup.append(guarded(command,
                f'PM_DRIVER_LOOKUP({name}, pm_driver_impl_{name});\n'))
        loader_lookup.append(guarded(command,
            f'PM_LOADER_LOOKUP({name});\n'))

        fields = "".join(f"    {kind} {param};\n" for kind, param in command["params"])
        layer_args.append(guarded(command,
            f"typedef struct PmLayerArgs_{name} {{\n{fields}}} PmLayerArgs_{name};\n"))
        layer_dispatch.append(guarded(command, f'''    if (std::strcmp(call.pCommandName, "{name}") == 0) {{
        if (call.argumentSize != sizeof(PmLayerArgs_{name}))
            return PM_ERROR_INVALID_ARGUMENT;
        return function(*static_cast<const PmLayerArgs_{name} *>(call.pArguments));
    }}
'''))
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
            invocation = f'''    const PmResult validation = parallel_mater::loader::pre_call(route, "{name}", arguments);
    if (validation != PM_SUCCESS) return validation;
    const PmResult result = function({", ".join(native_args)});
    parallel_mater::loader::post_call(route, "{name}", result, arguments);
    return result;'''
        elif command["returns"] == "void":
            invocation = f'''    if (parallel_mater::loader::pre_call(route, "{name}", arguments) != PM_SUCCESS)
        return;
    function({", ".join(native_args)});
    parallel_mater::loader::post_call(route, "{name}", PM_SUCCESS, arguments);
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
    const PmLayerArgs_{name} arguments{{{", ".join(args)}}};
{invocation}
}}
'''
        forwards.append(guarded(command, body))

    command_by_name = {command["name"]: command for command in commands}
    unknown_manual = SPECIAL - command_by_name.keys()
    if unknown_manual:
        raise RuntimeError(
            f"manual loader inventory contains unknown commands: "
            f"{sorted(unknown_manual)}")
    for index, (name, reason) in enumerate(MANUAL_LAYER_INTERCEPTED.items()):
        manual_layer_inventory.append(guarded(
            command_by_name[name],
            f'PM_MANUAL_LAYER_INTERCEPTED({name}, {index}, "{reason}")\n'))
    for name, reason in MANUAL_LAYER_EXCLUDED.items():
        manual_layer_inventory.append(guarded(
            command_by_name[name],
            f'PM_MANUAL_LAYER_EXCLUDED({name}, "{reason}")\n'))

    for capability in interaction_capabilities():
        requirements = " | ".join(
            f"require_{name}" for name in capability["requires"])
        maximum = ("PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE"
                   if capability["graph"] == "none" else
                   "PM_INTERACTION_GRAPH_EDGES_SIMULATION_CAPACITY")
        interaction_inventory.append(
            "PM_INTERACTION_CAPABILITY("
            f"PM_INTERACTION_FAMILY_{capability['first']}, "
            f"PM_INTERACTION_FAMILY_{capability['second']}, "
            f"PM_INTERACTION_SCOPE_{capability['scope']}, "
            f"PM_INTERACTION_MODE_{capability['mode']}, "
            f"PM_INTERACTION_DIRECTION_{capability['direction']}, "
            f"PM_INTERACTION_MATERIAL_{capability['profile']}, "
            f"{maximum}, {requirements})\n")

    layer_args.append("#endif\n")
    (output / "parallel_mater").mkdir(exist_ok=True)
    write_changed(output / "parallel_mater/pm_layer_args.h", "".join(layer_args))
    write_changed(output / "layer_dispatch.inc", "".join(layer_dispatch))
    write_changed(output / "driver_remap.h", "".join(remap))
    write_changed(output / "driver_unmap.h", "".join(unmap))
    write_changed(output / "driver_lookup.inc", "".join(lookup))
    write_changed(output / "loader_lookup.inc", "".join(loader_lookup))
    write_changed(output / "loader_forward.inc", "".join(forwards))
    write_changed(output / "manual_layer_inventory.inc",
                  "".join(manual_layer_inventory))
    write_changed(output / "interaction_capabilities.inc",
                  "".join(interaction_inventory))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    emit(args.output)


if __name__ == "__main__":
    main()
