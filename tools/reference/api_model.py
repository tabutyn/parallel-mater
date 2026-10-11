"""Small declaration inventory shared by generation and coverage checks.

This accepts the deliberately limited C grammar used in content.py, not arbitrary
C++. Unknown signatures/typedef forms fail validation instead of being skipped.
"""
import re

from content import SECTIONS


def public_types(source):
    names = set(re.findall(r"\btypedef\s+(?:struct|enum|union)\s+\w+\s*\{.*?\}\s*(\w+)\s*;",
                           source, re.S))
    names.update(re.findall(r"\btypedef\s+[^;{}()]+?\s+(Pm\w+)\s*;", source))
    names.update(re.findall(r"\*\s*((?:Pm|PFN_pm)\w+)\)\s*\(", source))
    return names


def declarations():
    result = []
    for section in SECTIONS:
        for card in section.cards:
            matches = list(re.finditer(r"^(\w+) (pm\w+)\((.*?)\);", card.functions, re.M | re.S))
            if card.functions and re.sub(r"^(\w+) (pm\w+)\((.*?)\);", "", card.functions,
                                         flags=re.M | re.S).strip():
                raise ValueError(f"Unparsed declarations in {card.key}")
            for match in matches:
                returns, name, args = match.groups()
                params = []
                for arg in args.split(","):
                    arg = " ".join(arg.split())
                    part = re.fullmatch(r"(.+?[ *])(\w+)", arg)
                    if not part:
                        raise ValueError(f"Unparsed argument: {name}: {arg}")
                    params.append((part[1].strip(), part[2]))
                result.append(dict(name=name, returns=returns, args=args, params=params,
                                   card=card.key, section=section.key, extension=card.extension))
    return result
