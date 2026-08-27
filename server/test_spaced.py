"""Tests for the `interlinear_spaced` overlay layout (server.interlinear).

The compact `interlinear` layout is tested in test_interlinear.py; what this
one promises is different and is what these tests hold it to: the page is
opened up so that EVERY gloss is drawn, at its full proportional size, and the
original comes through unscaled and untorn.

Run from the repo root:

    pytest server/test_spaced.py
"""

import json
from io import BytesIO

import pymupdf
import pytest

from server import interlinear
from server.conftest import TOKEN

STYLE = "interlinear"
PAGE = (595.0, 842.0)

AR_ONE = "تغيير البرمجيات أمر لا مفر منه"
AR_TWO = "تظهر متطلبات جديدة عند استخدام البرنامج"
AR_LONG = ("منتجات البرمجيات هي أنظمة برمجية عامة توفر وظائف مفيدة لمجموعة "
           "واسعة من العملاء في مجالات مختلفة")


@pytest.fixture(scope="module", autouse=True)
def gloss_font():
    """The real font, fetched once — shaping is what these tests are about."""
    interlinear._font_path()


class _Builder:
    """A page under construction, and the sidecar blocks that describe it.

    Written in DISPLAY coordinates (y down), which is how a PDF page reads and
    how every assertion below is phrased; the flip into the sidecar's PDF space
    happens once, in {@link sidecar}.
    """

    def __init__(self, size: tuple[float, float] = PAGE) -> None:
        self.size = size
        self.doc = pymupdf.open()
        self.page = self.doc.new_page(width=size[0], height=size[1])
        self.blocks: list[dict] = []

    def text(self, x0: float, top: float, text: str, size: float = 11.0,
             *, gloss: str | None = None) -> pymupdf.Rect:
        """One line of Latin text with its baseline placed from `top`."""
        self.page.insert_text((x0, top + size * 0.8), text, fontname="helv",
                              fontsize=size)
        width = pymupdf.get_text_length(text, fontname="helv", fontsize=size)
        rect = pymupdf.Rect(x0, top, x0 + width, top + size)

        if gloss is not None:
            self.paragraph([rect], gloss, size)

        return rect

    def paragraph(self, lines: list[pymupdf.Rect], gloss: str,
                  size: float = 11.0) -> None:
        box = pymupdf.Rect(lines[0])

        for line in lines[1:]:
            box |= line

        self.blocks.append({"box": box, "lines": list(lines), "target": gloss,
                            "font_size": size})

    def fill(self, rect: pymupdf.Rect, color=(0.9, 0.9, 0.9)) -> None:
        self.page.draw_rect(rect, fill=color, color=None)

    def image(self, rect: pymupdf.Rect) -> None:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 60), False)
        pixmap.clear_with(90)
        self.page.insert_image(rect, pixmap=pixmap)

    def pdf(self, rotation: int = 0) -> bytes:
        if rotation:
            self.page.set_rotation(rotation)

        out = BytesIO()
        self.doc.save(out)

        return out.getvalue()

    def sidecar(self, *, lang_out: str = "ar", pages: int = 1) -> dict:
        height = self.size[1]

        return {
            "version": 1,
            "lang_in": "en",
            "lang_out": lang_out,
            "total_pages": pages,
            "pages": [{
                "page_number": 0,
                "mediabox": [0.0, 0.0, self.size[0], height],
                "blocks": [{
                    "box": [block["box"].x0, height - block["box"].y1,
                            block["box"].x1, height - block["box"].y0],
                    "lines": [[line.x0, height - line.y1, line.x1,
                               height - line.y0] for line in block["lines"]],
                    "source": "source",
                    "target": block["target"],
                    "font_size": block["font_size"],
                    "label": "plain text",
                } for block in self.blocks],
                "obstacles": [],
            }],
        }


def _render(builder: _Builder, *, rotation: int = 0, **kwargs):
    options = (interlinear.OverlayOptions.defaults(STYLE) if not kwargs
               else interlinear.OverlayOptions(**kwargs))

    return interlinear.render_overlay(builder.pdf(rotation), builder.sidecar(),
                                      style=STYLE, options=options)


def _is_arabic(text: str) -> bool:
    return any("؀" <= ch <= "ۿ" or "ﭐ" <= ch <= "﻿"
               for ch in text)


def _spans(pdf_bytes: bytes, index: int = 0) -> list[tuple[pymupdf.Rect, str]]:
    """Every span on a page, boxed by the ink it leaves rather than its em band.

    The padded box overlaps its neighbours by design — that is the whole
    reason the layout measures the way it does — so asserting on it would
    report a collision between a gloss and a line that have clear air between
    them.
    """
    doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    pymupdf.TOOLS.set_small_glyph_heights(True)

    try:
        return [(pymupdf.Rect(span["bbox"]), span["text"])
                for block in doc[index].get_text("dict")["blocks"]
                if block["type"] == 0
                for line in block["lines"] for span in line["spans"]
                if span["text"].strip()]
    finally:
        pymupdf.TOOLS.set_small_glyph_heights(False)
        doc.close()


def _gloss_above(line: pymupdf.Rect, glosses: list[pymupdf.Rect],
                 ceiling: float) -> list[pymupdf.Rect]:
    """The gloss lines standing over `line` and under whatever is above it."""
    return [gloss for gloss in glosses
            if gloss.y1 <= line.y0 + 0.5 and gloss.y0 >= ceiling - 0.5
            and gloss.x1 > line.x0 and gloss.x0 < line.x1 + 40]


def _arabic(pdf_bytes: bytes) -> list[pymupdf.Rect]:
    return [rect for rect, text in _spans(pdf_bytes) if _is_arabic(text)]


def _gloss_lines(pdf_bytes: bytes) -> list[pymupdf.Rect]:
    """The Arabic on the page as VISUAL lines, not as spans.

    The Story engine emits a span per direction run, so one Arabic line with a
    Latin term in it arrives as three; counting spans would count the term.
    """
    lines: list[pymupdf.Rect] = []

    for rect in sorted(_arabic(pdf_bytes), key=lambda item: (item.y0, item.x0)):
        for line in lines:
            if rect.y0 < line.y1 - 1 and rect.y1 > line.y0 + 1:
                line |= rect
                break
        else:
            lines.append(pymupdf.Rect(rect))

    return sorted(lines, key=lambda item: item.y0)


def _latin(pdf_bytes: bytes) -> list[tuple[pymupdf.Rect, str]]:
    return [(rect, text) for rect, text in _spans(pdf_bytes)
            if not _is_arabic(text)]


def _page_size(pdf_bytes: bytes, index: int = 0) -> tuple[float, float]:
    doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")

    try:
        return doc[index].rect.width, doc[index].rect.height
    finally:
        doc.close()


def _tight_list(size: float = 20.0, leading: float = 24.0,
                count: int = 5) -> _Builder:
    """A bullet list with barely a point of air between its lines.

    This is the shape the compact layout cannot serve — the band above each
    bullet is a fraction of the type size — and the one this layout exists for.
    """
    builder = _Builder()

    for index in range(count):
        builder.text(60, 100 + index * leading, f"Bullet number {index} of the list",
                     size, gloss=AR_ONE)

    return builder


# --------------------------------------------------------------------------
# the promise
# --------------------------------------------------------------------------

def test_a_tight_list_gets_every_gloss_and_the_page_grows():
    builder = _tight_list()

    pdf, report = _render(builder)

    assert report == {"pages": 1, "drawn": 5, "skipped": 0,
                      "raster_drawn": 0, "raster_skipped": 0,
                      "vocab_pages": 0}
    assert _page_size(pdf)[1] > PAGE[1]
    assert len(_gloss_lines(pdf)) == 5


def test_the_same_list_defeats_the_compact_layout():
    """The comparison that justifies the layout, so it cannot rot silently."""
    builder = _tight_list()

    _pdf, report = interlinear.render_overlay(builder.pdf(), builder.sidecar(),
                                              style=interlinear.COMPACT_STYLE)

    assert report["skipped"] > 0


def test_the_original_is_moved_but_never_scaled():
    builder = _tight_list()

    before = {text: rect for rect, text in _latin(builder.pdf())}
    pdf, _ = _render(builder)
    after = {text: rect for rect, text in _latin(pdf)}

    assert set(before) == set(after)

    for text, rect in before.items():
        assert after[text].width == pytest.approx(rect.width, abs=0.1)
        assert after[text].height == pytest.approx(rect.height, abs=0.1)
        assert after[text].x0 == pytest.approx(rect.x0, abs=0.1)
        # Everything moved DOWN, never up, and never past the taller page.
        assert after[text].y0 >= rect.y0 - 0.1


def test_the_original_lines_keep_their_order():
    builder = _tight_list()

    pdf, _ = _render(builder)
    order = [text for _rect, text in
             sorted(_latin(pdf), key=lambda item: item[0].y0)]

    assert order == [f"Bullet number {index} of the list" for index in range(5)]


def test_no_gloss_is_drawn_over_the_original():
    builder = _tight_list()

    pdf, _ = _render(builder)
    latin = [rect for rect, _text in _latin(pdf)]

    for gloss in _arabic(pdf):
        for line in latin:
            overlap = gloss & line
            assert not overlap.is_valid or overlap.is_empty, \
                f"gloss {gloss} sits on {line}"


def test_each_line_of_a_paragraph_is_glossed_above_itself():
    builder = _Builder()
    lines = [builder.text(60, 100 + index * 16,
                          "Software products are generic software systems", 11.0)
             for index in range(3)]
    builder.paragraph(lines, AR_LONG, 11.0)

    pdf, report = _render(builder)

    assert report["drawn"] == 1
    latin = sorted((rect for rect, _text in _latin(pdf)), key=lambda r: r.y0)
    arabic = _gloss_lines(pdf)

    assert len(latin) == 3

    # Strictly alternating: gloss, line, gloss, line, gloss, line.
    ceiling = 0.0

    for line in latin:
        assert _gloss_above(line, arabic, ceiling), f"nothing glosses {line}"
        ceiling = line.y1


def test_a_gloss_is_proportional_to_the_type_it_glosses():
    builder = _Builder()
    builder.text(60, 80, "A heading", 32.0, gloss=AR_ONE)
    builder.text(60, 300, "Body text on the same page", 9.0, gloss=AR_TWO)

    pdf, _ = _render(builder)
    arabic = _gloss_lines(pdf)

    assert arabic[0].height > 2 * arabic[-1].height


# --------------------------------------------------------------------------
# what a cut may not damage
# --------------------------------------------------------------------------

def test_lines_whose_font_boxes_overlap_are_still_told_apart():
    """Big type, tight leading: the reported boxes overlap, the glyphs do not.

    Reading the font's own em band as ink would report a solid wall down the
    page and open no band at all — see {@link interlinear._tight_span}.
    """
    builder = _Builder()
    builder.text(60, 100, "Introduction to", 28.0, gloss=AR_ONE)
    builder.text(60, 129, "NumPy arrays", 28.0, gloss=AR_TWO)

    pdf, report = _render(builder)

    assert report == {"pages": 1, "drawn": 2, "skipped": 0,
                      "raster_drawn": 0, "raster_skipped": 0,
                      "vocab_pages": 0}

    latin = sorted((rect for rect, _text in _latin(pdf)), key=lambda r: r.y0)
    arabic = _gloss_lines(pdf)
    ceiling = 0.0

    for line in latin:
        assert _gloss_above(line, arabic, ceiling), f"nothing glosses {line}"
        ceiling = line.y1


def test_an_image_beside_the_text_is_not_sliced():
    builder = _Builder()
    builder.image(pymupdf.Rect(360, 80, 520, 500))

    for index in range(6):
        builder.text(60, 100 + index * 24, f"Line {index} of the bullet list",
                     20.0, gloss=AR_ONE)

    pdf, report = _render(builder)

    assert report["skipped"] == 0
    assert _page_size(pdf)[1] > PAGE[1]

    # Measured on the rendered page rather than on the image list: every slice
    # of the page refers back to the same embedded page object, so the list
    # reports the image once per slice whether or not any of them is visible.
    doc = pymupdf.open(stream=BytesIO(pdf), filetype="pdf")

    try:
        pixmap = doc[0].get_pixmap(dpi=36)
        column = round(440 * 36 / 72)
        rows = [y for y in range(pixmap.height)
                if pixmap.pixel(column, y)[0] < 200]
    finally:
        doc.close()

    # One unbroken run of grey, the height the image was placed at.
    assert rows
    assert rows == list(range(rows[0], rows[-1] + 1)), "the image was torn apart"
    assert (rows[-1] - rows[0]) * 72 / 36 == pytest.approx(240, abs=6.0)


def test_a_filled_panel_is_opened_up_rather_than_stepped_around():
    """A cut through the middle of a solid box reopens as more of the box."""
    builder = _Builder()
    builder.fill(pymupdf.Rect(50, 90, 545, 260))

    for index in range(4):
        builder.text(70, 110 + index * 24, f"print('line {index}')", 18.0,
                     gloss=AR_ONE)

    pdf, report = _render(builder)

    assert report == {"pages": 1, "drawn": 4, "skipped": 0,
                      "raster_drawn": 0, "raster_skipped": 0,
                      "vocab_pages": 0}

    doc = pymupdf.open(stream=BytesIO(pdf), filetype="pdf")

    try:
        # The panel is still one filled shape, and it grew with its contents.
        panels = [pymupdf.Rect(drawing["rect"]) for drawing in doc[0].get_drawings()
                  if drawing["rect"].width > 400]
        assert panels
        assert max(panel.height for panel in panels) > 170
    finally:
        doc.close()


def test_a_page_number_in_the_margin_does_not_become_its_own_column():
    """Otherwise it floats where it was while the footer beside it moves down."""
    builder = _Builder()

    for index in range(5):
        builder.text(60, 100 + index * 24, f"Body line {index} of the slide",
                     20.0, gloss=AR_ONE)

    footer = builder.text(60, 700, "Software Products", 9.0, gloss=AR_TWO)
    number = builder.text(560, 700, "7", 9.0, gloss="٧")

    pdf, report = _render(builder)

    assert report["skipped"] == 0

    after = {text: rect for rect, text in _latin(pdf)}
    assert after["7"].y0 == pytest.approx(after["Software Products"].y0, abs=1.0)
    assert after["7"].y0 > number.y0, "the footer moved down with the page"
    assert after["Software Products"].y0 > footer.y0


# --------------------------------------------------------------------------
# pages this layout hands back untouched
# --------------------------------------------------------------------------

def test_a_page_the_run_said_nothing_about_is_carried_through():
    builder = _Builder()
    builder.text(60, 100, "Untranslated page", 11.0)
    original = builder.pdf()
    sidecar = builder.sidecar()
    # A second page the sidecar does not mention at all.
    doc = pymupdf.open(stream=BytesIO(original), filetype="pdf")
    doc.new_page(width=PAGE[0], height=PAGE[1])
    doc[1].insert_text((60, 120), "Second page", fontname="helv", fontsize=11)
    sidecar["pages"][0]["blocks"] = [{
        "box": [60.0, PAGE[1] - 111.0, 300.0, PAGE[1] - 100.0],
        "lines": [[60.0, PAGE[1] - 111.0, 300.0, PAGE[1] - 100.0]],
        "source": "Untranslated page", "target": AR_ONE,
        "font_size": 11.0, "label": "plain text",
    }]
    out = BytesIO()
    doc.save(out)
    doc.close()

    pdf, report = interlinear.render_overlay(out.getvalue(), sidecar, style=STYLE)

    assert report["pages"] == 1
    assert _page_size(pdf, 1) == PAGE
    assert "Second page" in " ".join(text for _rect, text in _spans(pdf, 1))


def test_a_rotated_page_keeps_its_rotation_and_is_still_glossed():
    builder = _Builder()
    builder.text(60, 100, "Software change is inevitable", 11.0, gloss=AR_ONE)

    pdf, report = _render(builder, rotation=90)

    assert report["drawn"] == 1

    doc = pymupdf.open(stream=BytesIO(pdf), filetype="pdf")

    try:
        assert doc[0].rotation == 90
        assert doc[0].rect == pymupdf.Rect(0, 0, PAGE[1], PAGE[0])
    finally:
        doc.close()

    assert _arabic(pdf)


# --------------------------------------------------------------------------
# options, plumbing and refusals
# --------------------------------------------------------------------------

def test_each_style_has_its_own_tuned_defaults():
    compact = interlinear.OverlayOptions.defaults(interlinear.COMPACT_STYLE)
    spaced = interlinear.OverlayOptions.defaults(STYLE)

    assert spaced.scale > compact.scale
    assert spaced.gap > compact.gap


def test_the_output_is_not_a_copy_of_the_page_per_slice():
    """Every slice refers back to one embedded page, so size stays flat."""
    builder = _tight_list(count=12, leading=22.0, size=16.0)

    pdf, _ = _render(builder)

    assert len(pdf) < 400_000


def test_a_sidecar_from_another_document_is_refused():
    builder = _Builder()
    builder.text(60, 100, "Software change", 11.0, gloss=AR_ONE)
    sidecar = builder.sidecar()
    sidecar["pages"][0]["page_number"] = 4

    with pytest.raises(interlinear.OverlayError, match="not from the same run"):
        interlinear.render_overlay(builder.pdf(), sidecar, style=STYLE)


def test_the_endpoint_builds_the_spaced_style(client):
    builder = _Builder()
    builder.text(60, 100, "Software change is inevitable", 11.0, gloss=AR_ONE)

    response = client.post(
        "/v1/overlay",
        headers={"X-Internal-Token": TOKEN},
        files={"original": ("doc.pdf", builder.pdf(), "application/pdf"),
               "sidecar": ("sidecar.json",
                           json.dumps(builder.sidecar()).encode(),
                           "application/json")},
        data={"style": STYLE})

    assert response.status_code == 200
    assert response.headers["X-Overlay-Drawn"] == "1"
    assert response.headers["X-Overlay-Skipped"] == "0"
    assert response.content.startswith(b"%PDF-")


def test_the_endpoint_uses_the_style_tuning_when_nothing_is_asked_for(client):
    """An unset option must not silently arrive as the OTHER style's value."""
    builder = _Builder()
    builder.text(60, 100, "Software change is inevitable", 20.0, gloss=AR_ONE)

    heights = {}

    for style in (interlinear.COMPACT_STYLE, STYLE):
        response = client.post(
            "/v1/overlay",
            headers={"X-Internal-Token": TOKEN},
            files={"original": ("doc.pdf", builder.pdf(), "application/pdf"),
                   "sidecar": ("sidecar.json",
                               json.dumps(builder.sidecar()).encode(),
                               "application/json")},
            data={"style": style})

        assert response.status_code == 200
        heights[style] = max(rect.height for rect in _arabic(response.content))

    assert heights[STYLE] > heights[interlinear.COMPACT_STYLE]


# --------------------------------------------------------------------------
# raster glosses — a label inside an embedded image, on a page that is opened
# --------------------------------------------------------------------------

def test_a_raster_label_rides_its_image_down_the_opened_page():
    """Opening the page cannot help a label inside a raster: it keeps its
    plate, and the plate lands wherever the cuts moved the image to."""
    builder = _Builder()
    builder.text(100, 90, "Software change is inevitable", gloss=AR_ONE)

    image = pymupdf.Rect(100, 300, 400, 540)
    label = pymupdf.Rect(200, 400, 300, 412)

    # The stand-in for a diagram: a flat light fill with a dark bar where the
    # label sits, so the pixel judge has quiet air above it.
    scale = 4
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(
        0, 0, int(image.width * scale), int(image.height * scale)))
    pix.set_rect(pix.irect, (205, 210, 215))
    pix.set_rect(pymupdf.IRect(int((label.x0 - image.x0) * scale),
                               int((label.y0 - image.y0) * scale),
                               int((label.x1 - image.x0) * scale),
                               int((label.y1 - image.y0) * scale)),
                 (40, 40, 60))
    builder.page.insert_image(image, pixmap=pix)

    height = PAGE[1]
    sidecar = builder.sidecar()
    sidecar["pages"][0]["blocks"].append({
        "box": [label.x0, height - label.y1, label.x1, height - label.y0],
        "lines": [[label.x0, height - label.y1, label.x1, height - label.y0]],
        "source": "label",
        "target": AR_TWO,
        "font_size": 10.0,
        "label": "plain text",
        "on_raster": True,
        "region": [image.x0, height - image.y1, image.x1, height - image.y0],
    })

    pdf, report = interlinear.render_overlay(builder.pdf(), sidecar,
                                             style=STYLE)

    assert report["drawn"] == 1 and report["skipped"] == 0
    assert report["raster_drawn"] == 1 and report["raster_skipped"] == 0

    doc = pymupdf.open(stream=BytesIO(pdf), filetype="pdf")

    try:
        growth = doc[0].rect.height - height
        placed = [pymupdf.Rect(info["bbox"]) for info in doc[0].get_image_info()]
        plates = [pymupdf.Rect(drawing["rect"])
                  for drawing in doc[0].get_drawings()
                  if drawing.get("fill") is not None
                  and (drawing.get("fill_opacity") or 1.0) < 1.0]
    finally:
        doc.close()

    # The band opened above the glossed line is what moved the image.
    assert growth > 0

    moved = image + (0, growth, 0, growth)
    moved_label = label + (0, growth, 0, growth)

    # The image really is at its shifted place on the taller page...
    assert any(abs(rect.y0 - moved.y0) < 1.0 and abs(rect.y1 - moved.y1) < 1.0
               for rect in placed)

    # ...and the plate and its gloss sit inside it, clear of the label bar.
    assert plates, "no plate was drawn"

    for plate in plates:
        assert moved.contains(plate)
        assert plate.y1 <= moved_label.y0 or plate.y0 >= moved_label.y1

    inside = [rect for rect in _arabic(pdf) if moved.contains(rect)]
    assert inside, "no Arabic was drawn inside the moved image"

    for rect in inside:
        assert any(plate.contains(rect) for plate in plates)
