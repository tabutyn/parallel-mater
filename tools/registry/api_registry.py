"""Canonical Parallel Mater command inventory from registry/pm.xml."""

from pathlib import Path
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "registry" / "pm.xml"

INTERACTION_FAMILIES = (
    "FLUID", "CLOTH", "SOFT_BODY", "ROPE", "SMOKE", "RIGID_BODY")
INTERACTION_SCOPES = (
    "SELF", "SAME_DOMAIN", "SEPARATE_SYSTEM", "DISTINCT_FAMILY")
INTERACTION_MODES = (
    "CONTACT", "ATTACHMENT", "DRAG", "CONTAINMENT", "PRESSURE",
    "VISCOSITY", "THERMAL", "PHASE", "INTERFACE")
INTERACTION_DIRECTIONS = (
    "BIDIRECTIONAL", "FIRST_TO_SECOND", "SECOND_TO_FIRST",
    "FIRST_TO_SECOND_TRANSFER", "SECOND_TO_FIRST_TRANSFER",
    "EXTERNAL_TO_FIRST", "EXTERNAL_TO_SECOND")
INTERACTION_PROFILES = (
    "ENDPOINT_NATIVE", "SMOKE_PARTICLE", "SMOKE_GRID", "EQUAL",
    "MISCIBLE", "IMMISCIBLE_INTERFACE", "THERMAL_PHASE",
    "SMOKE_PARTICLE_GRID")
INTERACTION_FEATURES = (
    "rigidBodies", "rigidConstraints", "particleLiquids",
    "liquidRigidCoupling", "liquidLiquidCoupling", "cloth", "rope",
    "softBodies", "smoke", "domainCouplings")


def declarations():
    result = []
    root = ET.parse(REGISTRY).getroot()
    for card in root.findall("./cards/card"):
        source = (card.findtext("commands") or "").strip()
        matches = list(re.finditer(
            r"^(\w+) (pm\w+)\((.*?)\);", source, re.M | re.S))
        if source and re.sub(
                r"^(\w+) (pm\w+)\((.*?)\);", "", source,
                flags=re.M | re.S).strip():
            raise ValueError(
                f"unparsed command declarations in registry card "
                f"{card.attrib['key']}")
        for match in matches:
            returns, name, args = match.groups()
            params = []
            for argument in args.split(","):
                argument = " ".join(argument.split())
                part = re.fullmatch(r"(.+?[ *])(\w+)", argument)
                if not part:
                    raise ValueError(f"unparsed argument: {name}: {argument}")
                params.append((part[1].strip(), part[2]))
            result.append({
                "name": name,
                "returns": returns,
                "args": args,
                "params": params,
                "card": card.attrib["key"],
                "section": card.attrib["section"],
                "extension": card.attrib.get("extension") == "true",
            })
    return result


def interaction_capabilities():
    """Return and validate the reference driver's exact interaction inventory."""
    root = ET.parse(REGISTRY).getroot()
    result = []
    seen = set()
    families = {name: index for index, name in enumerate(INTERACTION_FAMILIES)}
    allowed = {
        "first": set(INTERACTION_FAMILIES),
        "second": set(INTERACTION_FAMILIES),
        "scope": set(INTERACTION_SCOPES),
        "mode": set(INTERACTION_MODES),
        "direction": set(INTERACTION_DIRECTIONS),
        "profile": set(INTERACTION_PROFILES),
        "graph": {"none", "simulation-capacity"},
    }
    for node in root.findall("./interaction-capabilities/capability"):
        capability = {name: node.attrib.get(name) for name in allowed}
        for name, values in allowed.items():
            if capability[name] not in values:
                raise ValueError(
                    f"invalid interaction capability {name}: {capability[name]!r}")
        requirements = tuple(node.attrib.get("requires", "").split())
        if not requirements or any(value not in INTERACTION_FEATURES
                                   for value in requirements):
            raise ValueError(
                f"invalid interaction requirements: {requirements!r}")
        first = families[capability["first"]]
        second = families[capability["second"]]
        distinct = capability["scope"] == "DISTINCT_FAMILY"
        if (distinct and first >= second) or (not distinct and first != second):
            raise ValueError(f"non-canonical interaction pair: {capability!r}")
        key = tuple(capability[name] for name in (
            "first", "second", "scope", "mode", "direction", "profile"))
        if key in seen:
            raise ValueError(f"duplicate interaction capability: {key!r}")
        seen.add(key)
        capability["requires"] = requirements
        result.append(capability)
    if not result:
        raise ValueError("interaction capability inventory is empty")
    return result
