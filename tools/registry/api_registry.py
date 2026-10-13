"""Canonical Parallel Mater command inventory from registry/pm.xml."""

from pathlib import Path
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "registry" / "pm.xml"


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
