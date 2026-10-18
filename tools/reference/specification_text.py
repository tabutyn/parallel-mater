"""Searchable specification generated without PDF/font dependencies."""
import html
from content import SECTIONS
from model import SPEC_NUMBERS, example_fragment, symbols
from provenance import registry_notice

NUMBERS = {**SPEC_NUMBERS, "symbol_index": "A"}
SCOPE = (
    "This specification defines the Parallel Mater 0.2 command API: object ownership, "
    "device discovery, explicit storage, physics pipelines, command execution, and "
    "synchronization. It is the behavioral contract for implementations and applications. "
    "The companion reference guide lists declarations and points here by section number."
)
NORMATIVE = (
    "Requirements stated as must, must not, required, or an imperative define valid usage "
    "and required implementation behavior. Examples illustrate those requirements and "
    "do not override them. The core domains and optional extension boundaries are "
    f"specified in section {SPEC_NUMBERS['not_capabilities']}; the API-family overview is not a claim that every device "
    "supports every physics domain."
)


def plain(text):
    return html.unescape(text).replace("<br/><br/>", "\n\n").replace("<br/>", "\n")


def render_markdown():
    markdown = [f"<!-- Generated from {registry_notice()} -->",
                "# Parallel Mater 0.2 — API Specification", SCOPE, NORMATIVE,
                "## Contents"]
    markdown.extend(f"- [{NUMBERS[s.key]} {s.title}](#spec-{s.key})" for s in SECTIONS)
    markdown.append("- [A Symbol index](#spec-symbol_index)")
    for section in SECTIONS:
        markdown.extend([f'<a id="spec-{section.key}"></a>',
                         f"## {NUMBERS[section.key]} {section.title}"])
        if section.subtitle:
            markdown.append(section.subtitle)
        for card in section.cards:
            markdown.extend([f'<a id="spec-{card.key}"></a>',
                             f"### {NUMBERS[card.key]} {card.title}"])
            for source in (card.types, card.functions):
                if source:
                    markdown.append(f"```c\n{source}\n```")
            if card.rows:
                rows = ["| Item | Definition |", "| --- | --- |"]
                rows.extend("| " + a.replace("|", r"\|") + " | " + b.replace("|", r"\|") + " |"
                            for a, b in card.rows)
                markdown.append("\n".join(rows))
            if card.text:
                markdown.append(plain(card.text))
            if card.code:
                source = example_fragment(card.code[1:]) if card.code.startswith("@") else card.code
                markdown.extend(["Example (informative):", f"```c\n{source}\n```"])
            if card.links:
                markdown.append("Related sections: " + ", ".join(
                    f"[{NUMBERS[key]}](#spec-{key})" for key in card.links) + ".")
    markdown.extend(['<a id="spec-symbol_index"></a>', "## A Symbol index"])
    rows = ["| Name | Section |", "| --- | --- |"]
    rows.extend(f"| {name} | [{NUMBERS[key]}](#spec-{key}) |" for name, key in symbols())
    markdown.append("\n".join(rows))
    return "\n\n".join(markdown) + "\n"
