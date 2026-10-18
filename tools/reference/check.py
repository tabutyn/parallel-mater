#!/usr/bin/env python3
"""Contract/document checks only: these are NOT physics conformance tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from collections import Counter

import pymupdf as fitz
from PIL import Image

from build import DEST, ROOT, RELEASE_ARTIFACTS, all_cards
from content import SECTIONS
from layout import BOTTOM, COLUMNS, MARGIN, TOP, SPEC_PURPLE, symbol_color
from model import REFERENCE_NOTES, SPEC_NUMBERS, reference_sections
from specification_text import NUMBERS, plain
from api_model import public_types
from check_api import check_contract
from provenance import file_sha256, registry_provenance


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(command, **kwargs):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, **kwargs)
    require(result.returncode == 0,
            f"Command failed: {' '.join(map(str, command))}\n{result.stdout}{result.stderr}")
    return result


def compiler_matrix():
    defaults = ("cl", "cl") if os.name == "nt" else ("cc", "c++")
    return ((os.environ.get("CC", defaults[0]), "c", "c11"),
            (os.environ.get("CXX", defaults[1]), "c++", "c++17"))


def compile_source(compiler, language, standard, source, include=None):
    if Path(compiler).name.lower() in ("cl", "cl.exe"):
        command = [compiler, "/nologo", "/TC" if language == "c" else "/TP",
                   f"/std:{standard}", "/W4", "/WX", "/wd4232", "/permissive-", "/Zs"]
        if include:
            command.append(f"/I{include}")
        command.append(str(source))
    else:
        command = [compiler, "-x", language, f"-std={standard}", "-Wall",
                   "-Wextra", "-Werror", "-pedantic"]
        if include:
            command += [f"-I{include}"]
        command += ["-fsyntax-only", str(source)]
    run(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--staging-root", type=Path,
                        help="Keep the clean regenerated artifact tree at this path")
    parser.add_argument("--skip-vulkan-header", action="store_true",
                        help="Only if the optional Vulkan SDK is not installed")
    args = parser.parse_args()
    examples = ["record_frame.c", "abi.c", "load_instance.c"]
    if not args.skip_vulkan_header:
        examples.append("vulkan_header.c")
    compile_count = 0
    for compiler, language, standard in compiler_matrix():
        for example in examples:
            compile_source(compiler, language, standard, DEST / "examples" / example)
            compile_count += 1
    print(f"PASS: {compile_count} C11/C++17 syntax and ABI checks (no linking or runtime claims)")

    cards = all_cards()
    quick_sections = reference_sections()
    quick_cards = [card for section in quick_sections for card in section.cards]
    require(len({c.key for c in cards}) == len(cards), "Duplicate panel keys")
    headers = [DEST / "api/parallel_mater.h", DEST / "api/interop_vulkan.h"]
    installed_headers = [ROOT / "include/parallel_mater/parallel_mater.h",
                         ROOT / "include/parallel_mater/interop_vulkan.h"]
    for draft, installed in zip(headers, installed_headers):
        require(draft.read_text() == installed.read_text(),
                f"Installed header differs from reference: {installed}")
    pdf_path = DEST / "parallel-mater-reference.pdf"
    manifest = json.loads(pdf_path.with_suffix(".layout.json").read_text())
    require({c.key for c in quick_cards} == {p["key"] for p in manifest["panels"]}, "Missing or extra reference sections")
    declared = set(re.findall(r"\b(pm\w+)\(", "\n".join(p.read_text() for p in headers)))
    typed = set(re.findall(r"\*PFN_(pm\w+)\)\(", "\n".join(p.read_text() for p in headers))) - {"pmVoidFunction"}
    require(typed == declared, "Every command needs a matching typed function pointer")
    documented = {s for s in manifest["symbols"] if s.startswith("pm")}
    require(declared == documented, f"Header / reference symbol mismatch: {declared ^ documented}")
    declared_types = public_types("\n".join(p.read_text() for p in headers)) - {"PFN_"+name for name in declared}
    require(declared_types == set(manifest["symbols"]) - declared,
            "Public structs, enums, aliases or callback types are missing from the reference")
    coverage = json.loads((DEST / "api-coverage.json").read_text())
    provenance = registry_provenance()
    require(coverage["registry"] == provenance,
            "Coverage does not identify the current registry revision")
    require({cmd["name"] for cmd in coverage["functions"]} == declared, "Stale command inventory")
    require({typ["name"] for typ in coverage["types"]} == declared_types, "Stale type inventory")
    keys = {c.key for c in cards} | {s.key for s in SECTIONS}
    require(all(link in keys for c in cards for link in c.links), "Unresolved panel reference")
    print(f"PASS: {len(declared)} functions share their declarations with the PDF")
    checked_headers = headers[:1] if args.skip_vulkan_header else headers
    checked_names = sorted(set(re.findall(r"\b(pm\w+)\(",
                                         "\n".join(p.read_text() for p in checked_headers))))
    pointer_check = f'#include "{checked_headers[-1]}"\n' + "\n".join(
        f"PFN_{name} checked_{name} = {name};" for name in checked_names)
    with tempfile.TemporaryDirectory(prefix="pm-reference-pointers-") as directory:
        for compiler, language, standard in compiler_matrix():
            suffix = ".c" if language == "c" else ".cpp"
            source = Path(directory) / ("pointers" + suffix)
            source.write_text(pointer_check, encoding="utf-8", newline="\n")
            compile_source(compiler, language, standard, source)
    print(f"PASS: {len(checked_names)} typed function pointers match their C/C++ declarations")

    doc = fitz.open(pdf_path)
    spec_path = DEST / "parallel-mater-specification.pdf"
    specification = json.loads(spec_path.with_suffix(".layout.json").read_text())
    spec_doc = fitz.open(spec_path)
    spec_markdown = spec_path.with_suffix(".md").read_text()
    require(manifest["registry"] == provenance,
            "Reference layout does not identify the current registry revision")
    require(specification["registry"] == provenance,
            "Specification layout does not identify the current registry revision")
    require(all(provenance["sha256"] in path.read_text(encoding="utf-8")
                for path in headers + installed_headers),
            "Generated headers do not identify the current registry revision")
    require(provenance["sha256"] in spec_markdown,
            "Specification text does not identify the current registry revision")
    spec_text = "\n".join(page.get_text() for page in spec_doc)
    spec_body = "\n".join(page.get_text(clip=fitz.Rect(30, 40, page.rect.width-30,
                                                    page.rect.height-35)) for page in spec_doc)
    spec_normalized = re.sub(r"\s+", "", spec_body)
    require(len(spec_doc) == specification["page_count"], "Specification page count mismatch")
    require(specification["numbers"] == NUMBERS, "Specification numbering drift")
    require(set(specification["page_map"]) == set(NUMBERS), "Missing specification section")
    require("API Specification" in spec_doc.metadata["title"], "Missing specification title")
    require("behavior required of conforming implementations" in " ".join(spec_text.split()),
            "Missing API contract scope in specification")
    for card in cards:
        for explanation in card.text.split("<br/><br/>"):
            if explanation:
                require(plain(explanation) in spec_markdown,
                        f"Explanation missing from specification Markdown: {card.key}")
                require(re.sub(r"\s+", "", plain(explanation)) in spec_normalized,
                        f"Explanation missing from specification PDF: {card.key}")
        require(f'<a id="spec-{card.key}"></a>' in spec_markdown, f"Missing Markdown anchor: {card.key}")
    for name in declared:
        require(name in spec_text, f"Missing specification declaration: {name}")
    require(set(specification["symbols"]) == set(manifest["symbols"]), "Guide / specification type mismatch")
    for number, page in enumerate(spec_doc, 1):
        for word in page.get_text("words"):
            x0, y0, x1, y1 = word[:4]
            require(x0 >= 30 and y0 >= 15 and x1 <= page.rect.width-30 and y1 <= page.rect.height-15,
                    f"Specification text outside page bounds on page {number}: {word}")
        for link in page.get_links():
            require(link["kind"] == fitz.LINK_GOTO and 0 <= link["page"] < len(spec_doc),
                    f"Invalid specification cross-reference: {link}")
    require("\ufffd" not in spec_text, "Specification contains replacement glyphs")
    print(f"PASS: {len(spec_doc)}-page specification preserves every explanation and declaration")
    require(len(doc) == manifest["page_count"], "Page count mismatch")
    require(len(doc.get_toc()) == len(quick_sections) + 1, "Missing PDF outline entries")
    require([entry[1] for entry in doc.get_toc()[:3]] ==
            ["Parallel Mater", "Return codes", "Function Pointers and Instancing"],
            "Opening topic order changed")
    require("API Reference" in doc.metadata["title"], "Missing reference title in metadata")
    all_text = "\n".join(page.get_text() for page in doc)
    normalized = " ".join(all_text.split())
    require(REFERENCE_NOTES["results"] in normalized, "Incorrect return-code caption")
    for card in cards:
        if card.key in ("intent", "legend") or not card.text:
            continue
        require(re.sub(r"\s+", "", plain(card.text)) not in re.sub(r"\s+", "", all_text),
                f"Explanatory paragraph leaked into reference: {card.key}")
    require(all(item["card"] in ("intent", "legend", "results")
                for item in manifest["items"] if item["kind"] == "text"),
            "Reference body must contain declarations, not explanatory text")
    for author_note in ("runtime not implemented", "Read this draft correctly", "MIGRATION.md",
                        "DESIGN.md", "Rebuild and verify", "syntax-checked", "World::",
                        "worktree", "This deliverable", "Design Draft"):
        require(author_note.lower() not in normalized.lower(),
                f"Author/implementation-progress text in consumer reference: {author_note}")
    require("\ufffd" not in all_text, "PDF contains replacement glyphs")
    for symbol in manifest["symbols"]:
        require(symbol in all_text, f"Symbol missing from searchable text: {symbol}")
    link_count, remote_count, marker_count, image_digests, colored_names = 0, 0, 0, set(), set()
    for number, page in enumerate(doc, 1):
        for word in page.get_text("words"):
            rect = fitz.Rect(word[:4])
            require(rect.x0 >= MARGIN - 1 and rect.y0 >= MARGIN - 1
                    and rect.x1 <= page.rect.width - 2 and rect.y1 <= page.rect.height - 1,
                    f"Text outside page safe area on page {number}: {word}")
        for link in page.get_links():
            if link["kind"] == fitz.LINK_GOTO:
                require(0 <= link["page"] < len(doc), f"Broken page link: {link}")
                link_count += 1
            elif link["kind"] == fitz.LINK_GOTOR:
                require(link["file"] == spec_path.name and 0 <= link["page"] < len(spec_doc),
                        f"Broken specification link: {link}")
                remote_count += 1
                linked_words = [word[4] for word in page.get_text("words")
                                if link["from"].contains(fitz.Point((word[0]+word[2])/2,
                                                                  (word[1]+word[3])/2))]
                require(len(linked_words) == 1 and re.fullmatch(r"\[\d+(?:\.\d+)*\]", linked_words[0]),
                        f"Specification link covers a title instead of its purple marker: {linked_words}")
        for item in page.get_image_info(hashes=True):
            image_digests.add(item["digest"])
        for font in page.get_fonts():
            # Canvas defaults can leave an unused Helvetica resource. Every
            # actual drawn span below must use an embedded project font.
            if font[3] != "Helvetica":
                require(bool(doc.extract_font(font[0])[3]), f"Unembedded font: {font}")
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    require("Lato" in span["font"] or "DejaVu" in span["font"],
                            f"Unexpected drawn font {span['font']}")
                    name = span["text"].strip()
                    if re.fullmatch(r"\[(?:\d+(?:\.\d+)*|n\.n)\]", name):
                        require(span["color"] == int(SPEC_PURPLE.hexval(), 16),
                                f"Specification marker is not purple: {name}")
                        marker_count += 1
                    expected_color = symbol_color(name)
                    if expected_color:
                        require(span["color"] == int(expected_color.hexval(), 16),
                                f"Wrong name color for {name} on page {number}")
                        colored_names.add(name)
    require(len(image_digests) == 1, "Only the wordmark may be a raster image")
    require(remote_count == len(manifest["spec_links"]) and marker_count >= remote_count,
            "Specification markers or links are missing")
    for target in manifest["spec_links"]:
        require(target["number"] == SPEC_NUMBERS[target["key"]]
                and target["target_page"] == specification["page_map"][target["key"]],
                f"Specification link mismatch: {target}")
        actual = [link for link in doc[target["page"]-1].get_links() if link["kind"] == fitz.LINK_GOTOR]
        x0, y0, x1, y1 = target["rect"]
        expected_rect = fitz.Rect(x0, doc[0].rect.height-y1, x1, doc[0].rect.height-y0)
        require(any(link["page"] == target["target_page"]-1 and
                    max(abs(a-b) for a, b in zip(link["from"], expected_rect)) < .01
                    for link in actual), f"Incorrect clickable specification target: {target}")
    require({"pmCreateInstance", "PmInstanceCreateInfo", "PFN_pmVoidFunction", "PM_SUCCESS"}
            <= colored_names, "Function/type/constant color key is incomplete")
    bar = next(shape for shape in manifest["shapes"] if shape["section"] == "overview")
    bx0, by0, bx1, by1 = bar["rects"][0]
    require(bar["page"] == 1 and bx0 == MARGIN and abs(bx1 - (doc[0].rect.width-MARGIN)) < .01,
            "Opening bar must span the page")
    for item in manifest["items"]:
        if item["page"] != 1:
            continue
        x0, y0, x1, y1 = item["rect"]
        if item["section"] == "overview":
            require(bx0 <= x0 < x1 <= bx1 and by0 <= y0 < y1 <= by1,
                    f"Overview content outside top bar: {item}")
        else:
            require(y1 < by0, f"Body overlaps opening bar: {item}")
    legend = next(item for item in manifest["items"] if item["card"] == "legend")
    require(legend["page"] == 1, "Legend missing from opening bar")

    card_text = {card.key: [] for card in cards}
    for panel in manifest["panels"]:
        page = doc[panel["page"] - 1]
        x0, y0, x1, y1 = panel["rect"]
        require(x0 >= MARGIN and y0 >= BOTTOM and x1 <= page.rect.width - MARGIN + .01
                and y1 <= TOP + .01, f"Fragment outside content area: {panel}")
        rect = fitz.Rect(x0, page.rect.height - y1, x1, page.rect.height - y0)
        card_text[panel["key"]].append(page.get_text(clip=rect))
        for other in manifest["panels"]:
            if other["key"] == panel["key"] or other["page"] != panel["page"]:
                continue
            a, b, c, d = other["rect"]
            overlap = min(x1, c) - max(x0, a) > .01 and min(y1, d) - max(y0, b) > .01
            require(not overlap, f"Overlapping panels: {panel['key']} / {other['key']}")
    for card in cards:
        visible = "\n".join(card_text[card.key])
        for symbol in re.findall(r"\b(pm\w+)\s*\(", card.functions):
            require(symbol in visible, f"Declaration missing from its section: {symbol}")
        if card.types:
            tokens = lambda source: Counter(re.findall(r"\b[A-Za-z_]\w*|\b\d+\b", source))
            required_tokens = tokens(card.types)
            require(not (required_tokens - tokens(visible)),
                    f"Incomplete reference type/fields: {card.key}: {required_tokens - tokens(visible)}")
            require(not (required_tokens - tokens(spec_text)), f"Incomplete specification type: {card.key}")
    titles = [item["section"] for item in manifest["items"] if item["kind"] == "section"]
    require(len(titles) == len(set(titles)), "Category title is repeated across columns/pages")
    require("Embedding in a Vulkan pipeline" not in all_text + spec_text, "Vulkan-only integration section remains")
    for backend in ("CUDA", "Metal", "DirectX 12", "Vulkan"):
        require(backend in " ".join(card_text["native_mapping"]), f"Missing native backend: {backend}")
    require(manifest["columns"] == COLUMNS == 3, "Reference must flow through three columns")
    require(len({shape["color"] for shape in manifest["shapes"]}) >= 10,
            "Topic colors are missing")
    stepped = sum(len(contour) > 4 for shape in manifest["shapes"] for contour in shape["contours"])
    # A dense symbol-index spread may fill adjacent columns exactly; every
    # other page must retain the connected, stepped topic treatment.
    require(stepped >= len(doc) - 1,
            "Connected, stepped topic boundaries are missing")
    require(min(manifest["page_fill"]) >= .85, "Reference leaves too much page area unused")
    for i, item in enumerate(manifest["items"]):
        x0, y0, x1, y1 = item["rect"]
        require(x0 >= MARGIN and y0 >= BOTTOM and x1 <= manifest["page_size"][0] - MARGIN + .01
                and y1 <= TOP + .01, f"Content outside safe area: {item}")
        if item["kind"] in ("section", "heading"):
            following = manifest["items"][i+1]
            require(following["page"] == item["page"] and abs(following["rect"][0] - x0) < .01,
                    f"Orphan heading at column end: {item}")
    logo = Image.open(DEST / "assets/parallel-mater-wordmark.png")
    require(logo.mode == "RGBA" and logo.getextrema()[3][0] == 0, "Wordmark lost transparency")
    print(f"PASS: {len(doc)} pages, {len(manifest['panels'])} section fragments, {link_count} internal links; one raster asset")
    print(f"PASS: {stepped} stepped contours, full-page flow, consumer-facing text, no orphan headings")
    print("PASS: full-width opening bar, requested topic order, consistent function/type colors")
    print(f"PASS: declaration-only reference body; {remote_count} purple specification links")
    print("PASS: searchable declarations, embedded fonts, safe page bounds, no overlapping sections")

    release_path = DEST / "release-artifacts.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    require(release["registry"] == provenance,
            "Release artifact manifest does not identify the current registry revision")
    require(release["known_nondeterminism"] == [],
            "Reference artifacts claim unresolved nondeterminism")
    require(set(release["artifacts"]) == set(RELEASE_ARTIFACTS),
            "Release artifact manifest inventory is stale")
    for relative, expected in release["artifacts"].items():
        artifact = ROOT / relative
        require(artifact.stat().st_size == expected["size"],
                f"Release artifact size mismatch: {relative}")
        require(file_sha256(artifact) == expected["sha256"],
                f"Release artifact hash mismatch: {relative}")
    print(f"PASS: {len(RELEASE_ARTIFACTS)} release artifacts share registry "
          f"{provenance['sha256'][:12]}")

    with tempfile.TemporaryDirectory(prefix="pm-reference-check-") as directory:
        check_contract(ROOT, DEST, Path(directory), run, require,
                       compile_source, args.skip_vulkan_header)
        artifact_root = (args.staging_root.resolve() if args.staging_root
                         else Path(directory) / "generated")
        command = [sys.executable, "tools/reference/build.py",
                   "--artifact-root", str(artifact_root)]
        if args.font_dir:
            command += ["--font-dir", str(args.font_dir)]
        run(command)
        generated = list(RELEASE_ARTIFACTS) + ["docs/reference/release-artifacts.json"]
        mismatches = [relative for relative in generated
                      if (artifact_root / relative).read_bytes() != (ROOT / relative).read_bytes()]
        require(not mismatches,
                "Clean reference regeneration differs: " + ", ".join(mismatches))
    require(hashlib.sha256(pdf_path.read_bytes()).hexdigest() == manifest["pdf_sha256"],
            "PDF hash differs from manifest")
    require(hashlib.sha256(spec_path.read_bytes()).hexdigest() == specification["pdf_sha256"],
            "Specification hash differs from manifest")
    print("PASS: deterministic clean regeneration of headers, text, coverage, PDFs and layouts")


if __name__ == "__main__":
    main()
