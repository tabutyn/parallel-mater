"""Small public-type inventory used by reference coverage checks.

The command grammar is parsed by tools/registry/api_registry.py. Public type
matching remains deliberately limited to the C forms emitted by the registry.
"""
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "registry"))

from api_registry import declarations  # noqa: E402,F401


def public_types(source):
    names = set(re.findall(r"\btypedef\s+(?:struct|enum|union)\s+\w+\s*\{.*?\}\s*(\w+)\s*;",
                           source, re.S))
    names.update(re.findall(r"\btypedef\s+[^;{}()]+?\s+(Pm\w+)\s*;", source))
    names.update(re.findall(r"\*\s*((?:Pm|PFN_pm)\w+)\)\s*\(", source))
    return names
