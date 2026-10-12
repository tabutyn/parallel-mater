"""Generate command/type coverage without modifying runtime source files."""
import json
import re

from api_model import declarations, public_types
from content import SECTIONS
from model import SPEC_NUMBERS
from check_api import legacy_inventory


def generate_coverage(root, destination):
    commands = declarations()
    definitions = {}
    for source in sorted((root / "src/api").glob("*.cpp")):
        text = source.read_text()
        for match in re.finditer(
                r'extern\s+"C"\s+PMAPI_ATTR\s+[^\n]*?\b(pm\w+)\s*\(', text):
            definitions.setdefault(match.group(1), []).append(
                f"{source.relative_to(root)}:{text.count(chr(10), 0, match.start()) + 1}")

    missing = sorted({command["name"] for command in commands} - definitions.keys())
    duplicates = {name: locations for name, locations in definitions.items()
                  if len(locations) != 1}
    if missing or duplicates:
        raise ValueError(
            f"Runtime command coverage mismatch; missing={missing}, duplicates={duplicates}")

    capability_gated = {
        "pmCreatePhysicsPipeline", "pmCmdBindPhysicsPipeline",
        "pmCreateSimulation", "pmBindSimulationMemory", "pmCmdBuildGeometry",
        "pmCmdWriteRigidBodies", "pmCmdApplyRigidInputs", "pmCmdSetRigidStates",
        "pmCmdWriteRigidConstraints", "pmCmdResetSimulation",
        "pmCmdSetFluidParticles", "pmCmdStepSimulation",
        "pmCmdCopyRigidStates", "pmCmdCopyFluidParticles",
    }
    inventory = {
        "contract": "Parallel Mater command API 0.2",
        "implementation": "src/api runtime entry points; no generated runtime stubs",
        "functions": [{
            "name": command["name"],
            "section": SPEC_NUMBERS[command["card"]],
            "card": command["card"],
            "header": "interop_vulkan.h" if command["extension"] else "parallel_mater.h",
            "source": definitions[command["name"]][0],
            "status": ("capability-gated until the Slang simulation schedule is wired"
                       if command["name"] in capability_gated else "implemented"),
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
        json.dumps(inventory, indent=2) + "\n")
    (destination / "legacy-api-inventory.json").write_text(
        json.dumps(legacy_inventory(root), indent=2) + "\n")
