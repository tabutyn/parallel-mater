"""Readable, numbered API specification, separate from the declaration sheet."""
import hashlib
import io
import re
import textwrap

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, PageBreak,
                               PageTemplate, Paragraph, Spacer, Table, TableStyle,
                               XPreformatted)
from reportlab.platypus.tableofcontents import TableOfContents

from content import SECTIONS
from layout import (DEST, DIAGRAM_HEIGHTS, H, INNER, INK, MUTED, SPEC_PURPLE, W,
                    draw_diagram, example_fragment, paragraph, symbols, text_markup)

from specification_text import NUMBERS, SCOPE, NORMATIVE, render_markdown

MARGIN = 42
WIDTH = W - 2 * MARGIN
CODE_SIZE, CODE_LEADING = 8.2, 10.5


def code_block(source):
    limit = int((WIDTH-16) / pdfmetrics.stringWidth("M", "Code", CODE_SIZE))
    lines = []
    for line in source.splitlines():
        indent = len(line) - len(line.lstrip())
        lines.extend(textwrap.wrap(line, width=limit, subsequent_indent=" " * (indent+2),
            break_long_words=False, break_on_hyphens=False, replace_whitespace=False) or [""])
    for line in lines:
        if pdfmetrics.stringWidth(line, "Code", CODE_SIZE) > WIDTH-16:
            raise ValueError(f"Specification code exceeds text area: {line}")
    return XPreformatted(text_markup("\n".join(lines), section_markers=False), ParagraphStyle("spec-code",
        fontName="Code", fontSize=CODE_SIZE, leading=CODE_LEADING, textColor=INK,
        backColor=colors.HexColor("#F1F4F4"), borderPadding=5, spaceAfter=10))


def prose(text, size=10.2, leading=14.3):
    result = paragraph(text, size, leading)
    result.style.spaceAfter = 9
    return result


def heading(key, title, level=0):
    result = paragraph(f"{NUMBERS[key]}  {title}", 15 if level == 0 else 11.7,
                       19 if level == 0 else 15, True,
                       INK if level == 0 else SPEC_PURPLE)
    result.style.keepWithNext = True
    result.style.spaceBefore = 13 if level == 0 else 8
    result.style.spaceAfter = 8
    result.spec_key, result.spec_level = key, level
    result.spec_title = f"{NUMBERS[key]} {title}"
    return result


def small_heading(title):
    result = paragraph(title, 9, 12, True, MUTED)
    result.style.keepWithNext = True
    result.style.spaceAfter = 4
    return result


def table(rows):
    data = [[paragraph(a, 9, 12, True), paragraph(b, 9, 12)] for a, b in rows]
    result = Table(data, colWidths=[WIDTH*.37, WIDTH*.63], hAlign="LEFT")
    result.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#F0F4F4"), colors.white]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), .25, colors.HexColor("#C7D5D5")),
    ]))
    result.spaceAfter = 10
    return result


class Diagram(Flowable):
    def __init__(self, kind):
        super().__init__()
        self.kind, self.width = kind, WIDTH
        self.height = DIAGRAM_HEIGHTS[kind] + 10

    def draw(self):
        draw_diagram(self.canv, self.kind, (WIDTH-INNER)/2, self.height-4,
                     self.height-10, compact=False)


class SpecificationDoc(BaseDocTemplate):
    def __init__(self, stream):
        super().__init__(stream, pagesize=(W, H), leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=45, bottomMargin=40, invariant=1, pageCompression=1,
            title="Parallel Mater 0.2 — API Specification", author="Parallel Mater",
            subject="Required API behavior, valid usage, ownership, and synchronization",
            creator="Parallel Mater specification generator")
        self.addPageTemplates(PageTemplate(id="body",
            frames=[Frame(MARGIN, 40, WIDTH, H-85, leftPadding=0, rightPadding=0,
                          topPadding=0, bottomPadding=0)], onPage=self.page_furniture))
        self.page_map = {}
        self.page_count = 0

    def beforeDocument(self):
        self.page_map = {}

    def page_furniture(self, pdf, doc):
        pdf.saveState()
        pdf.setFillColor(MUTED)
        pdf.setFont("Body", 8)
        pdf.drawString(MARGIN, H-27, "Parallel Mater 0.2 | API Specification")
        pdf.drawRightString(W-MARGIN, 23, str(doc.page))
        pdf.restoreState()

    def afterFlowable(self, flowable):
        self.page_count = self.page
        if not hasattr(flowable, "spec_key"):
            return
        key, level = flowable.spec_key, flowable.spec_level
        self.page_map[key] = self.page
        self.canv.bookmarkPage("spec_" + key)
        self.canv.addOutlineEntry(flowable.spec_title, "spec_" + key, level=level)
        if level == 0:
            self.notify("TOCEntry", (0, flowable.spec_title, self.page, "spec_" + key))


def build_specification(output):
    story = [prose("Parallel Mater", 25, 29), prose("API Specification · 0.2", 16, 21),
             prose(SCOPE), prose(NORMATIVE), Spacer(1, 12), small_heading("Contents")]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("contents", fontName="Body", fontSize=9.5,
        leading=13, textColor=INK, spaceBefore=2, leftIndent=0, firstLineIndent=0)]
    story.extend([toc, PageBreak()])
    source_code = {}
    for section in SECTIONS:
        story.append(heading(section.key, section.title))
        if section.subtitle:
            story.append(prose(section.subtitle))
        if section.diagram:
            story.append(Diagram(section.diagram))
        for card in section.cards:
            story.append(heading(card.key, card.title, 1))
            for source in (card.types, card.functions):
                if source:
                    story.append(code_block(source))
            if card.rows:
                story.append(table(card.rows))
            if card.text:
                for text in card.text.split("<br/><br/>"):
                    story.append(prose(text))
            if card.code:
                source = example_fragment(card.code[1:]) if card.code.startswith("@") else card.code
                source_code[card.key] = source
                story.extend([small_heading("Example (informative)"), code_block(source)])
            if card.links:
                links = ", ".join(f'<link href="#spec_{key}" color="{SPEC_PURPLE.hexval()}">'
                                  f'[{NUMBERS[key]}]</link>' for key in card.links)
                story.append(Paragraph("Related sections: " + links, ParagraphStyle("related",
                    fontName="Body", fontSize=9, leading=12, spaceAfter=10)))

    story.append(heading("symbol_index", "Symbol index"))
    for name, key in symbols():
        story.append(Paragraph(text_markup(name) +
            f' — <link href="#spec_{key}" color="{SPEC_PURPLE.hexval()}">[{NUMBERS[key]}]</link>',
            ParagraphStyle("symbol", fontName="Body", fontSize=9, leading=12)))

    stream = io.BytesIO()
    doc = SpecificationDoc(stream)
    doc.multiBuild(story)
    output.write_bytes(stream.getvalue())
    output.with_suffix(".md").write_text(
        render_markdown(), encoding="utf-8", newline="\n")
    return {"version": "0.2", "page_count": doc.page_count, "page_map": doc.page_map,
            "numbers": NUMBERS, "symbols": dict(symbols()), "example_keys": list(source_code),
            "pdf_sha256": hashlib.sha256(stream.getvalue()).hexdigest()}
