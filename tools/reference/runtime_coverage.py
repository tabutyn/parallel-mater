"""Generate command/type coverage without modifying runtime source files."""
import json
import re

from api_model import declarations, public_types
from content import SECTIONS
from model import SPEC_NUMBERS

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


def generate_coverage(root, destination, provenance):
    commands = declarations()
    definitions = {}
    for source in sorted((root / "src/loader").glob("*.cpp")):
        text = source.read_text()
        for match in re.finditer(
                r'extern\s+"C"\s+PMAPI_ATTR\s+[\w*]+\s+PMAPI_CALL\s+(pm\w+)\s*\(',
                text):
            definitions.setdefault(match.group(1), []).append(
                f"{source.relative_to(root)}:{text.count(chr(10), 0, match.start()) + 1}")
    for command in commands:
        if command["name"] not in LOADER_SPECIAL:
            definitions.setdefault(command["name"], []).append(
                "generated/api/loader_forward.inc")

    missing = sorted({command["name"] for command in commands} - definitions.keys())
    duplicates = {name: locations for name, locations in definitions.items()
                  if len(locations) != 1}
    if missing or duplicates:
        raise ValueError(
            f"Runtime command coverage mismatch; missing={missing}, duplicates={duplicates}")

    inventory = {
        "registry": provenance,
        "contract": "Parallel Mater command API 0.2",
        "implementation": "src/loader public entry points routed to installable drivers",
        "statusMeaning": {
            "implemented": "owned and executed by the loader",
            "driver-routed": ("the loader forwards the command; support is "
                              "reported by the selected driver's features, "
                              "limits, formats, and extensions"),
        },
        "functions": [{
            "name": command["name"],
            "section": SPEC_NUMBERS[command["card"]],
            "card": command["card"],
            "header": "interop_vulkan.h" if command["extension"] else "parallel_mater.h",
            "source": definitions[command["name"]][0],
            "status": ("driver-routed"
                       if definitions[command["name"]][0] ==
                          "generated/api/loader_forward.inc"
                       else "implemented"),
        } for command in commands],
        "types": [{"name": name, "section": SPEC_NUMBERS[card.key],
                   "card": card.key}
                  for section in SECTIONS for card in section.cards
                  for name in sorted(public_types(card.types))],
        "derived_function_pointer_types": ["PFN_" + command["name"]
                                           for command in commands],
        "derived_type_rule": ("Each PFN_pmName has pmName's exact signature and "
                              "PMAPI_CALL convention; section " +
                              SPEC_NUMBERS["base_types"] + "."),
    }
    (destination / "api-coverage.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8", newline="\n")
