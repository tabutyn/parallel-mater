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
