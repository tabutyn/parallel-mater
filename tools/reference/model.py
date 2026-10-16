"""Shared numbering and the deliberately smaller quick-reference view."""
from dataclasses import replace

from content import SECTIONS

SPEC_NUMBERS = {}
for chapter, section in enumerate(SECTIONS, 1):
    SPEC_NUMBERS[section.key] = str(chapter)
    for subsection, card in enumerate(section.cards, 1):
        SPEC_NUMBERS[card.key] = f"{chapter}.{subsection}"

REFERENCE_NOTES = {
    "results": "Return codes are reported via PmResult return values.",
}
REFERENCE_TABLES = {"constraint_dofs", "lifecycle", "stage_mapping", "native_mapping", "format_layouts", "feature_matrix"}


def reference_cards(section):
    if section.key == "overview":
        return [card for card in section.cards if card.key in ("intent", "legend")]
    return [card for card in section.cards
            if card.types or card.functions or card.key in REFERENCE_TABLES]


def reference_sections():
    return [replace(section, cards=reference_cards(section)) for section in SECTIONS
            if reference_cards(section)]


def example_fragment(key):
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2] / "docs/reference/examples/record_frame.c").read_text()
    return source.split(f"/* REF_BEGIN: {key} */\n")[1].split(f"/* REF_END: {key} */")[0].strip()


def symbols():
    import re
    from api_model import public_types
    found = {}
    for section in SECTIONS:
        for card in section.cards:
            for name in re.findall(r"\b(pm\w+)\s*\(", card.functions):
                found[name] = card.key
            for name in public_types(card.types):
                found[name] = card.key
    return sorted(found.items(), key=lambda item: item[0].lower())
