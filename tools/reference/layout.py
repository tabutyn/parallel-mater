"""Dense, column-flow reference sheets with connected topic contours."""
import hashlib
import html
import io
import math
from pathlib import Path
import re
import textwrap

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfdoc import PDFArray, PDFDictionary, PDFName, PDFString
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from content import SECTIONS, Card, Section
from api_model import public_types
from model import REFERENCE_NOTES, SPEC_NUMBERS, reference_sections, example_fragment, symbols

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "docs/reference"
W, H = A4
MARGIN, GUTTER, PAD = 8.0, 4.0, 5.5
COLUMNS = 3
COL = (W - 2 * MARGIN - (COLUMNS - 1) * GUTTER) / COLUMNS
INNER = COL - 2 * PAD
TOP, BOTTOM = H - MARGIN, 12.0
CODE_SIZE, CODE_LEADING = 6.7, 8.0
LONG_TOKEN_SIZE = 6.4  # Full structure identifiers, never truncated or split.
BODY_SIZE, BODY_LEADING = 8.0, 9.4
INDEX_SIZE, INDEX_LEADING = 8.0, 16.0
DIAGRAM_HEIGHTS = {"ownership": 169, "allocation": 192, "pipeline": 239,
                   "lifecycle": 207, "handoff": 211}
INK = colors.HexColor("#24323C")
BLUE = colors.HexColor("#135A91")
GREEN = colors.HexColor("#176246")
PLUM = colors.HexColor("#703747")
SPEC_PURPLE = colors.HexColor("#7542A5")
TEAL = colors.HexColor("#256D73")
MUTED = colors.HexColor("#526069")
BORDER = colors.HexColor("#69434D")
PAPER = colors.HexColor("#FBFAF5")
PALETTE = {
    "overview": "#EAF0EF", "return_codes": "#F6EAC1", "instancing": "#E9E1F1",
    "discovery": "#DDEBF4", "memory": "#F4E3AC", "resources": "#DCEBCF",
    "simulation": "#F2D9C4", "constraints": "#E6D8ED", "pipelines": "#F2E69F",
    "domains": "#D8E9D1",
    "commands": "#DCE1F4", "transfer": "#CEE7E3", "sync": "#F0D1D4",
    "interop": "#E1DCF0", "queries": "#D4E8D5", "capabilities": "#EDE0B8",
    "example": "#DAE5EC", "fundamentals": "#DDE7ED", "index": "#E8E4D9",
}


def symbol_color(token):
    if re.fullmatch(r"pm[A-Z]\w*", token):
        return BLUE
    if re.fullmatch(r"(?:(?:Pm|Vk)[A-Z]\w*|PFN_pm\w+)", token):
        return GREEN
    if re.fullmatch(r"PM_\w+", token):
        return PLUM
    return None


def register_fonts(font_dir=None):
    roots = [Path(font_dir)] if font_dir else [
        Path("/usr/share/fonts/truetype/lato"),
        Path("/usr/share/fonts/truetype/dejavu"),
    ]
    for name, filename in (("Body", "Lato-Regular.ttf"), ("Bold", "Lato-Bold.ttf"),
                           ("Italic", "Lato-Italic.ttf"), ("Code", "DejaVuSansMono.ttf")):
        found = next((p / filename for p in roots if (p / filename).is_file()), None)
        if not found:
            raise SystemExit(f"Missing {filename}; install fonts or use --font-dir")
        pdfmetrics.registerFont(TTFont(name, str(found)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold", italic="Italic", boldItalic="Bold")


def text_markup(text, section_markers=True):
    markup = html.escape(html.unescape(text)).replace("&lt;br/&gt;", "<br/>")
    markup = re.sub(r"\b(?:PFN_pm\w+|pm[A-Z]\w*|(?:Pm|Vk)[A-Z]\w*|PM_\w+)\b",
                    lambda m: f'<font color="{symbol_color(m[0]).hexval()}">{m[0]}</font>', markup)
    if section_markers:
        markup = re.sub(r"\[(?:\d+(?:\.\d+)*|n\.n)\]",
                        lambda m: f'<font color="{SPEC_PURPLE.hexval()}">{m[0]}</font>', markup)
    return markup


def paragraph(text, size=BODY_SIZE, leading=BODY_LEADING, bold=False, color=INK):
    markup = text_markup(text)
    return Paragraph(markup, ParagraphStyle("reference", fontName="Bold" if bold else "Body",
        fontSize=size, leading=leading, textColor=color, spaceAfter=0,
        allowWidows=0, allowOrphans=0, splitLongWords=True))


def code_lines(source):
    max_chars = int(INNER / pdfmetrics.stringWidth("M", "Code", CODE_SIZE))
    lines = []
    for line in source.splitlines():
        indent = len(line) - len(line.lstrip())
        longest = max(map(len, line.split()), default=0)
        continuation = max(0, min(indent + 2, max_chars - longest))
        wrapped = textwrap.wrap(line, width=max_chars,
            subsequent_indent=" " * continuation,
            break_long_words=False, break_on_hyphens=False,
            replace_whitespace=False, drop_whitespace=True) or [""]
        for value in wrapped:
            if pdfmetrics.stringWidth(value, "Code", LONG_TOKEN_SIZE) > INNER + .01:
                raise ValueError(f"Code token exceeds column: {value}")
        lines.extend(wrapped)
    return lines


def components(card):
    result = []
    if card.key in REFERENCE_NOTES:
        result.append(("text", paragraph(REFERENCE_NOTES[card.key])))
    for source in (card.types, card.functions):
        if source:
            source = example_fragment(source[1:]) if source.startswith("@") else source
            result.append(("code", code_lines(source)))
    for left, right in card.rows:
        ratio = .60 if card.key == "format_layouts" else .42
        a = paragraph(left, 7.5, 8.9, True)
        b = paragraph(right, 7.5, 8.9)
        a.wrap(INNER * ratio - 4, H)
        b.wrap(INNER * (1 - ratio) - 3, H)
        result.append(("row", (a, b, max(a.height, b.height) + 4, ratio)))
    return result


class Flow:
    """Keep reference order; split long declarations/prose instead of wasting columns."""
    def __init__(self):
        self.pages = []
        self.page_map = {}
        self.section = None
        self.region = None
        self.titled_sections = set()
        self.new_page()

    def new_page(self):
        self.page = {"number": len(self.pages) + 1, "regions": [], "items": [], "fragments": [],
                     "column_top": TOP}
        self.pages.append(self.page)
        self.column, self.y, self.region = 0, TOP, None

    @property
    def x(self):
        return MARGIN + self.column * (COL + GUTTER)

    def next_column(self):
        self.finish_region(extend=True)
        self.column += 1
        if self.column == COLUMNS:
            self.new_page()
        self.y, self.region = self.page["column_top"], None

    def top_bar(self, section):
        """One full-width opening region; all later columns begin below it."""
        cards = {card.key: card for card in section.cards}
        self.page_map[section.key] = self.page["number"]
        left = MARGIN + PAD
        right = MARGIN + 2 * (COL + GUTTER) + PAD
        top = TOP - PAD
        intro_width = 2 * COL + GUTTER - 2 * PAD

        def place(kind, value, x, y, width, height, key=None):
            self.page["items"].append({"kind": kind, "value": value, "x": x,
                "top": y, "width": width, "height": height, "card": key,
                "section": section.key})
            if key:
                self.page_map[key] = self.page["number"]

        intro = paragraph(cards["intent"].text, 8.6, 10.2)
        intro.wrap(intro_width, H)
        place("text", intro, left, top, intro_width, intro.height, "intent")
        place("wordmark", None, right, top, INNER, 29)
        legend = paragraph(cards["legend"].text, 7.2, 8.5)
        legend.wrap(INNER, H)
        place("text", legend, right, top-31, INNER, legend.height, "legend")
        bottom = top - max(intro.height, 31 + legend.height) - PAD
        self.page["regions"].append({"key": section.key, "column": -1, "x": MARGIN,
            "width": W-2*MARGIN, "top": TOP, "bottom": bottom, "color": PALETTE[section.key]})
        self.page["column_top"] = bottom - GUTTER
        self.y = self.page["column_top"]

    def finish_region(self, extend=False):
        if self.region:
            self.region["bottom"] = BOTTOM if extend else max(BOTTOM, self.y - PAD)
        self.region = None

    def start_region(self, section):
        if self.region and self.region["key"] == section.key:
            return
        if self.region:
            self.finish_region()
            self.y -= PAD + GUTTER
        number = f" [{SPEC_NUMBERS[section.key]}]" if section.key in SPEC_NUMBERS else ""
        heading = paragraph(section.title + number, 10.0, 11.0, True)
        heading.wrap(INNER, H)
        first = section.key not in self.titled_sections
        if self.y - (PAD + (heading.height if first else 0) + 60) < BOTTOM:
            self.next_column()
        self.section = section
        self.region = {"key": section.key, "column": self.column, "x": self.x,
                       "top": self.y, "bottom": self.y, "color": PALETTE[section.key]}
        self.page["regions"].append(self.region)
        self.y -= PAD
        if first:
            self.add("section", heading, heading.height + 3, None)
            self.titled_sections.add(section.key)
        self.page_map.setdefault(section.key, self.page["number"])

    def available(self):
        return self.y - BOTTOM - PAD

    def add(self, kind, value, height, card):
        if height > self.available() + .01:
            raise ValueError(f"Layout overflow in {kind}: {height} > {self.available()}")
        self.page["items"].append({"kind": kind, "value": value,
            "x": self.x + PAD, "top": self.y, "height": height,
            "card": card.key if card else None, "section": self.section.key})
        self.y -= height
        self.region["bottom"] = self.y - PAD

    def continue_card(self, section, card):
        self.next_column()
        self.start_region(section)
        self.card_heading(card, continued=True)

    def card_heading(self, card, continued=False):
        title = card.title + (" (continued)" if continued else "")
        heading = paragraph(title + f" [{SPEC_NUMBERS[card.key]}]", 8.6, 10.0, True, PLUM)
        heading.wrap(INNER, H)
        if self.available() < heading.height + 30:
            section = self.section
            self.next_column()
            self.start_region(section)
        self.add("heading", heading, heading.height + 2, card)
        self.page_map.setdefault(card.key, self.page["number"])

    def card(self, section, card):
        self.start_region(section)
        self.card_heading(card)
        for kind, value in components(card):
            if kind == "code":
                remaining = value[:]
                while remaining:
                    count = int((self.available() - 2) / CODE_LEADING)
                    if count < min(3, len(remaining)):
                        self.continue_card(section, card)
                        continue
                    # Avoid a one-line declaration tail at the next column top.
                    if len(remaining) - count == 1 and count > 3:
                        count -= 1
                    lines = remaining[:count]
                    self.add("code", lines, len(lines) * CODE_LEADING + 2, card)
                    remaining = remaining[count:]
                    if remaining:
                        self.continue_card(section, card)
            elif kind == "text":
                remaining = value
                while remaining:
                    remaining.wrap(INNER, H)
                    if remaining.height + 3 <= self.available():
                        self.add("text", remaining, remaining.height + 3, card)
                        break
                    pieces = remaining.split(INNER, max(0, self.available() - 3))
                    if pieces:
                        first = pieces[0]
                        first.wrap(INNER, H)
                        self.add("text", first, first.height + 3, card)
                        remaining = pieces[1] if len(pieces) > 1 else None
                    self.continue_card(section, card)
            else:
                height = value[2] if kind == "row" else 9
                if height > self.available():
                    self.continue_card(section, card)
                self.add(kind, value, height, card)
        # Only a small separation between related subsections, not a card border.
        self.y -= min(4, self.available())

    def illustration(self, section):
        height = DIAGRAM_HEIGHTS[section.diagram]
        self.start_region(section)
        if self.available() < height + 35:
            self.next_column()
            self.start_region(section)
        self.add("diagram", section.diagram, height, None)


def layout():
    flow = Flow()
    for section in reference_sections():
        if section.key == "overview":
            flow.top_bar(section)
            continue
        flow.start_region(section)
        pending_diagram = section.diagram
        if pending_diagram and flow.available() >= DIAGRAM_HEIGHTS[pending_diagram] + 35:
            flow.illustration(section)
            pending_diagram = None
        for card in section.cards:
            flow.card(section, card)
            if pending_diagram and flow.available() >= DIAGRAM_HEIGHTS[pending_diagram] + 35:
                flow.illustration(section)
                pending_diagram = None
        if pending_diagram:
            flow.illustration(section)
    index_section = Section("index", "Symbol index", "", [])
    flow.start_region(index_section)
    # Balance the index through complete remaining sheets. A fixed leading can
    # strand a short list in the last column of an otherwise empty final sheet.
    entries = symbols()
    capacities = [flow.available()] + [TOP-BOTTOM-2*PAD-14] * (COLUMNS-flow.column-1)
    while sum(int(space / 9) for space in capacities) < len(entries):
        capacities.extend([TOP-BOTTOM-2*PAD] * COLUMNS)
    low, high = 9.0, 24.0
    for _ in range(40):
        middle = (low + high) / 2
        if sum(int(space / middle) for space in capacities) >= len(entries):
            low = middle
        else:
            high = middle
    index_leading = low - 1e-6
    for symbol, key in entries:
        if flow.available() < index_leading:
            flow.next_column()
            flow.start_region(index_section)
        if pdfmetrics.stringWidth(symbol, "Body", INDEX_SIZE) > INNER - 13:
            raise ValueError(f"Index label too wide: {symbol}")
        flow.add("index", (symbol, key), index_leading, None)
    flow.finish_region()
    return flow.pages, flow.page_map


def union_contours(rects):
    """Exact rectilinear union, including cross-column bridges; no raster masks."""
    rects = [tuple(round(v, 4) for v in r) for r in rects]
    xs = sorted({v for r in rects for v in (r[0], r[2])})
    ys = sorted({v for r in rects for v in (r[1], r[3])})
    cells = set()
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x, y = (xs[i] + xs[i+1]) / 2, (ys[j] + ys[j+1]) / 2
            if any(a < x < c and b < y < d for a, b, c, d in rects):
                cells.add((i, j))
    edges = {}
    for i, j in sorted(cells):
        a, b, c, d = xs[i], ys[j], xs[i+1], ys[j+1]
        for neighbor, p, q in (((i, j-1), (a, b), (c, b)), ((i+1, j), (c, b), (c, d)),
                               ((i, j+1), (c, d), (a, d)), ((i-1, j), (a, d), (a, b))):
            if neighbor not in cells:
                edges.setdefault(p, []).append(q)
    loops = []
    while edges:
        start = min(edges)
        p, points = start, []
        while True:
            points.append(p)
            q = edges[p].pop()
            if not edges[p]: del edges[p]
            p = q
            if p == start: break
        simplified = []
        for i, p in enumerate(points):
            a, b = points[i-1], points[(i+1) % len(points)]
            if not (a[0] == p[0] == b[0] or a[1] == p[1] == b[1]):
                simplified.append(p)
        loops.append(simplified)
    return loops


def topic_shapes(page):
    by_key = {}
    for region in page["regions"]:
        by_key.setdefault(region["key"], []).append(region)
    shapes = []
    for key, regions in by_key.items():
        rects = [(r["x"], r["bottom"], r["x"] + r.get("width", COL), r["top"]) for r in regions]
        for a in regions:
            for b in regions:
                if b["column"] != a["column"] + 1: continue
                low, high = max(a["bottom"], b["bottom"]), min(a["top"], b["top"])
                if high - low > 3:
                    rects.append((a["x"] + COL, low, b["x"], high))
        shapes.append({"section": key, "color": PALETTE[key], "rects": rects,
                       "contours": union_contours(rects)})
    return shapes


def draw_contour(pdf, points):
    path = pdf.beginPath()
    for i, p in enumerate(points):
        a, b = points[i-1], points[(i+1) % len(points)]
        before, after = math.dist(a, p), math.dist(b, p)
        radius = min(3, before / 2, after / 2)
        entry = (p[0] + (a[0] - p[0]) * radius / before, p[1] + (a[1] - p[1]) * radius / before)
        leave = (p[0] + (b[0] - p[0]) * radius / after, p[1] + (b[1] - p[1]) * radius / after)
        if i == 0: path.moveTo(*entry)
        else: path.lineTo(*entry)
        path.curveTo(p[0], p[1], p[0], p[1], *leave)
    path.close()
    pdf.drawPath(path, fill=1, stroke=1)


def arrow(pdf, x1, y1, x2, y2, dashed=False):
    pdf.saveState()
    pdf.setStrokeColor(TEAL)
    pdf.setFillColor(TEAL)
    pdf.setLineWidth(.65)
    if dashed: pdf.setDash(2, 2)
    pdf.line(x1, y1, x2, y2)
    angle = math.atan2(y2-y1, x2-x1)
    p = pdf.beginPath()
    p.moveTo(x2, y2)
    for offset in (2.65, -2.65):
        p.lineTo(x2 + 3.5 * math.cos(angle+offset), y2 + 3.5 * math.sin(angle+offset))
    p.close()
    pdf.drawPath(p, fill=1, stroke=0)
    pdf.restoreState()


def label(pdf, text, x, y, size=7.1, bold=False, color=INK, centered=True):
    pdf.setFont("Bold" if bold else "Body", size)
    pdf.setFillColor(color)
    (pdf.drawCentredString if centered else pdf.drawString)(x, y, text)


def node(pdf, text, x, y, width, height=23):
    pdf.setFillColor(colors.Color(1, 1, 1, .65))
    pdf.setStrokeColor(TEAL)
    pdf.setLineWidth(.4)
    pdf.roundRect(x, y, width, height, 2, fill=1, stroke=1)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        size = 7.3 if i == 0 else 6.9
        if pdfmetrics.stringWidth(line, "Bold" if i == 0 else "Body", size) > width - 5:
            raise ValueError(f"Diagram label too wide: {line}")
        label(pdf, line, x + width/2,
              y + height/2 + (len(lines)-1)*4.2 - i*8.4 - 2.5, size, i == 0)


def draw_diagram(pdf, kind, x, top, height, compact=True):
    # Coordinates relative to the column's top; diagrams are native PDF paths.
    center = x + INNER / 2
    if kind == "ownership":
        for i, text in enumerate(("Instance", "Physical device", "Device / queues")):
            yy = top - 24 - i*33
            node(pdf, text, x + 25, yy, INNER - 50)
            if i < 2: arrow(pdf, center, yy, center, yy-10)
        yy = top - 121
        node(pdf, "Memory", x, yy, 63)
        node(pdf, "Resources / state", x + 76, yy, INNER-76)
        arrow(pdf, x + 63, yy+11, x+76, yy+11)
        arrow(pdf, center, top-91, center, yy+23)
        node(pdf, "Pipeline + command buffer", x+15, top-156, INNER-30)
        arrow(pdf, x+128, yy, x+128, top-133)
        label(pdf, "allocate → bind → record → submit", center, top-166, 6.9)
    elif kind == "allocation":
        for i, text in enumerate(("Create resource", "Query requirements", "Allocate memory", "Bind aligned range", "Initialize / use")):
            yy = top - 24 - i*29
            node(pdf, text, x+23, yy, INNER-46)
            if i < 4: arrow(pdf, center, yy, center, yy-6)
        label(pdf, "Allocation ranges" if compact else "One allocation; distinct aligned ranges", center, top-157, 6.9)
        widths = (58, 62, INNER-120)
        xx = x
        for name, width in zip(("geometry", "state", "scratch"), widths):
            node(pdf, name, xx, top-183, width, 20)
            xx += width
    elif kind == "pipeline":
        names = ("Inputs / loads", "Integrate / predict", "Contacts / neighbors", "Solve rigid AVBD + liquids", "Reconstruct velocity / sleep", "Export to output buffers")
        for i, text in enumerate(names):
            yy = top - 26 - i*33
            node(pdf, text, x+12, yy, INNER-24)
            if i < 5: arrow(pdf, center, yy, center, yy-10)
        # Bracket substep work, not the application-level output copy.
        pdf.setStrokeColor(TEAL)
        pdf.setLineWidth(.6)
        pdf.lines([(x+7, top-36, x+2, top-36), (x+2, top-36, x+2, top-158),
                   (x+2, top-158, x+7, top-158)])
        label(pdf, "Substep loop" if compact else "Bracketed work repeats per substep.", center, top-207, 7.0)
        if not compact:
            label(pdf, "Contacts + joints share the rigid solve.", center, top-219, 7.0)
            label(pdf, "Couplings connect declared domains.", center, top-231, 7.0)
    elif kind == "lifecycle":
        xx, nw = x+29, INNER-62
        for i, name in enumerate(("Initial", "Recording", "Executable", "Pending", "Invalid")):
            yy = top - 25 - i*37
            node(pdf, name, xx, yy, nw)
            if i < 4:
                arrow(pdf, center-2, yy, center-2, yy-14)
                label(pdf, ("Begin", "End", "Submit", "one-time completes")[i], center+5, yy-9, 6.6, centered=False)
        right = x+INNER-7
        pdf.setStrokeColor(TEAL)
        pdf.setDash(2, 2)
        pdf.lines([(xx+nw, top-124, right, top-124), (right, top-124, right, top-88)])
        pdf.setDash()
        arrow(pdf, right, top-88, xx+nw, top-88, True)
        pdf.setStrokeColor(TEAL)
        pdf.setDash(2, 2)
        pdf.lines([(xx, top-162, x+9, top-162), (x+9, top-162, x+9, top-14)])
        pdf.setDash()
        arrow(pdf, x+9, top-14, xx, top-14, True)
        label(pdf, "Reusable completion → Executable", center, top-190, 7)
        label(pdf, "Reset → Initial; never while Pending", center, top-202, 7)
    elif kind == "handoff":
        for i, (name, detail) in enumerate((("Native work", "read / release / signal"), ("Physics", "wait / acquire"),
                ("Physics", "solve / copy / release / signal"), ("Native work", "wait / acquire / read"))):
            yy = top - 38 - i*44
            node(pdf, name+"\n"+detail, x+8, yy, INNER-16, 32)
            if i < 3: arrow(pdf, center, yy, center, yy-12)
        label(pdf, "Timeline + resource handoff" if compact else "Timeline signals connect GPU queues.", center, top-189, 7)
        if not compact:
            label(pdf, "Native API sets visibility and resource state.", center, top-201, 7)


def draw_wordmark(pdf, x, top):
    path = DEST / "assets/parallel-mater-wordmark.png"
    with Image.open(path) as img:
        # Ignore near-transparent edge noise when positioning the unchanged asset.
        a, b, c, d = img.getchannel("A").point(lambda v: 255 if v > 128 else 0).getbbox()
        width, height = img.size
    scale = min(INNER / (c-a), 23 / (d-b))
    pdf.saveState()
    clip = pdf.beginPath()
    clip.rect(x, top-29, INNER, 29)
    pdf.clipPath(clip, stroke=0)
    pdf.drawImage(str(path), x+(INNER-(c-a)*scale)/2-a*scale, top-3+(b-height)*scale,
                  width=width*scale, height=height*scale, mask="auto")
    pdf.restoreState()


def draw_code(pdf, lines, x, top):
    for i, line in enumerate(lines):
        xx = x
        for token in re.findall(r"\w+|[^\w]+", line):
            color = symbol_color(token) or INK
            pdf.setFillColor(color)
            size = (CODE_SIZE if pdfmetrics.stringWidth(line, "Code", CODE_SIZE) <= INNER
                    else LONG_TOKEN_SIZE)
            pdf.setFont("Code", size)
            pdf.drawString(xx, top-CODE_SIZE-i*CODE_LEADING, token)
            xx += pdfmetrics.stringWidth(token, "Code", size)


def draw_item(pdf, item, page_map):
    kind, value, x, top, height = (item[k] for k in ("kind", "value", "x", "top", "height"))
    if kind in ("section", "heading", "text"):
        value.drawOn(pdf, x, top-value.height)
    elif kind == "code":
        draw_code(pdf, value, x, top)
    elif kind == "row":
        a, b, _, ratio = value
        a.drawOn(pdf, x, top-a.height)
        b.drawOn(pdf, x+INNER*ratio, top-b.height)
        pdf.setStrokeColor(colors.Color(.3, .3, .3, .2))
        pdf.setLineWidth(.25)
        pdf.line(x, top-height+1, x+INNER, top-height+1)
    elif kind == "links":
        xx = x
        label(pdf, "See", xx, top-6.8, 6.7, color=MUTED, centered=False)
        xx += 13
        seen = set()
        for key in value:
            page = page_map[key]
            if page in seen: continue
            seen.add(page)
            title = f"p.{page}"
            width = pdfmetrics.stringWidth(title, "Body", 6.7)
            label(pdf, title, xx, top-6.8, 6.7, color=BLUE, centered=False)
            pdf.linkRect("", key, (xx, top-8, xx+width, top), thickness=0)
            xx += width+6
    elif kind == "diagram":
        draw_diagram(pdf, value, x, top, height)
    elif kind == "wordmark":
        draw_wordmark(pdf, x, top)
    elif kind == "index":
        symbol, key = value
        label(pdf, symbol, x, top-9, INDEX_SIZE, color=BLUE if symbol.startswith("pm") else GREEN, centered=False)
        pdf.setFont("Body", INDEX_SIZE)
        pdf.setFillColor(MUTED)
        pdf.drawRightString(x+INNER, top-9, str(page_map[key]))
        pdf.linkRect("", key, (x, top-height, x+INNER, top), thickness=0)


def specification_link(pdf, rect, page, filename):
    action = PDFDictionary({"S": PDFName("GoToR"), "F": PDFString(filename),
                            "D": PDFArray([page - 1, PDFName("Fit")])})
    annotation = PDFDictionary({"Type": PDFName("Annot"), "Subtype": PDFName("Link"),
        "Rect": PDFArray(rect), "Border": PDFArray([0, 0, 0]), "A": action})
    pdf._addAnnotation(annotation)


def build_pdf(output, specification):
    pages, page_map = layout()
    stream = io.BytesIO()
    pdf = canvas.Canvas(stream, pagesize=A4, invariant=1, pageCompression=1)
    pdf.setTitle("Parallel Mater 0.2 — API Reference")
    pdf.setAuthor("Parallel Mater")
    pdf.setSubject("Declarations and specification-section references")
    pdf.setCreator("Parallel Mater reference generator")
    manifest = {"version": "0.2", "page_size": [W, H], "columns": COLUMNS,
                "min_code_pt": LONG_TOKEN_SIZE, "min_body_pt": BODY_SIZE,
                "page_map": page_map, "symbols": dict(symbols()), "panels": [],
                "shapes": [], "items": [], "page_fill": [], "spec_links": [],
                "spec_numbers": SPEC_NUMBERS, "specification": "parallel-mater-specification.pdf"}
    seen_cards, seen_sections = set(), set()
    for page in pages:
        pdf.setFillColor(PAPER)
        pdf.rect(0, 0, W, H, fill=1, stroke=0)
        shapes = topic_shapes(page)
        for shape in shapes:
            pdf.setFillColor(colors.HexColor(shape["color"]))
            pdf.setStrokeColor(BORDER)
            pdf.setLineWidth(.8)
            for contour in shape["contours"]:
                draw_contour(pdf, contour)
            manifest["shapes"].append({**shape, "page": page["number"]})
        fragments = {}
        for item in page["items"]:
            kind, key = item["kind"], item["card"]
            if item["section"] not in seen_sections:
                section = next((s for s in SECTIONS if s.key == item["section"]), None)
                destination = "section_" + item["section"]
                pdf.bookmarkHorizontalAbsolute(destination, item["top"])
                pdf.addOutlineEntry(section.title if section else "Symbol index", destination, level=0)
                if item["section"] not in {c.key for s in SECTIONS for c in s.cards}:
                    pdf.bookmarkHorizontalAbsolute(item["section"], item["top"])
                seen_sections.add(item["section"])
            if key and key not in seen_cards:
                pdf.bookmarkHorizontalAbsolute(key, item["top"])
                seen_cards.add(key)
            draw_item(pdf, item, page_map)
            rect = [item["x"], item["top"]-item["height"], item["x"]+item.get("width", INNER), item["top"]]
            spec_key = item["section"] if kind == "section" else key if kind == "heading" else None
            if spec_key in SPEC_NUMBERS:
                target_page = specification["page_map"][spec_key]
                # Only the trailing purple section marker is a link. Category
                # and subsection names remain ordinary selectable text.
                heading = item["value"]
                end = item["x"] + heading.getActualLineWidths0()[-1]
                marker_width = pdfmetrics.stringWidth(
                    f"[{SPEC_NUMBERS[spec_key]}]", "Bold", heading.style.fontSize)
                bottom = item["top"] - heading.height
                marker_rect = [end-marker_width, bottom, end, bottom+heading.style.leading]
                specification_link(pdf, marker_rect, target_page, manifest["specification"])
                manifest["spec_links"].append({"key": spec_key, "number": SPEC_NUMBERS[spec_key],
                    "page": page["number"], "target_page": target_page, "rect": marker_rect})
            manifest["items"].append({"page": page["number"], "kind": kind, "card": key,
                                       "section": item["section"], "rect": rect})
            if key:
                group = (key, item["x"])
                if group not in fragments:
                    fragments[group] = {"key": key, "page": page["number"], "rect": rect[:]}
                else:
                    r = fragments[group]["rect"]
                    r[1], r[3] = min(r[1], rect[1]), max(r[3], rect[3])
        manifest["panels"].extend(fragments.values())
        occupied = sum(item["height"] * item.get("width", INNER) / INNER for item in page["items"])
        manifest["page_fill"].append(round(occupied / (COLUMNS * (TOP-BOTTOM)), 4))
        # Only a tiny folio, outside the content area; no running banners or status footer.
        label(pdf, str(page["number"]), W-MARGIN, 3.3, 6.0, color=MUTED)
        pdf.showPage()
    pdf.save()
    output.write_bytes(stream.getvalue())
    manifest["page_count"] = len(pages)
    manifest["pdf_sha256"] = hashlib.sha256(stream.getvalue()).hexdigest()
    return manifest
