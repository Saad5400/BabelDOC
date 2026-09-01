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
             *, gloss: str | None = None,
             source: str | None = None) -> pymupdf.Rect:
        """One line of Latin text with its baseline placed from `top`."""
        self.page.insert_text((x0, top + size * 0.8), text, fontname="helv",
                              fontsize=size)
        width = pymupdf.get_text_length(text, fontname="helv", fontsize=size)
        rect = pymupdf.Rect(x0, top, x0 + width, top + size)

        if gloss is not None:
            self.paragraph([rect], gloss, size, source=source)

        return rect

    def paragraph(self, lines: list[pymupdf.Rect], gloss: str,
                  size: float = 11.0, *, source: str | None = None) -> None:
        box = pymupdf.Rect(lines[0])

        for line in lines[1:]:
            box |= line

        # The source text matters to the layout in one way: a target that
        # repeats it is not a gloss at all ({@link interlinear._says_nothing}).
        self.blocks.append({"box": box, "lines": list(lines), "target": gloss,
                            "font_size": size, "source": source or "source"})

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
                    "source": block["source"],
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
    """The comparison that justifies the layout, so it cannot rot silently.

    On the real corpus the compact style skips 68% of its glosses and 96% on a
    2-slides-per-A4 handout, which is why the product no longer offers it. The
    engine still builds it — cached artifacts have to stay downloadable, and
    it is what glosses a rotated page — so this stays as the record of why.
    """
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
# what is not worth a gloss
# --------------------------------------------------------------------------

def test_a_target_that_repeats_its_source_is_not_glossed():
    """A slide number, a truth table cell, a keyword: 37 % of the corpus.

    The translator returns a target for every block it is handed, and on a
    real deck a great many of those targets are the source back again. Drawn,
    each one is a gloss restating the line under it — and in this layout the
    page is opened up to make room for it first.
    """
    builder = _Builder()
    builder.text(60, 100, "Change is inevitable", 11.0, gloss=AR_ONE,
                 source="Change is inevitable")
    builder.text(60, 300, "12", 11.0, gloss="12", source="12")
    builder.text(60, 500, "byte", 11.0, gloss="  byte  ", source="byte")

    pdf, report = _render(builder)

    assert report["drawn"] == 1
    # Every gloss on the page belongs to the first line; the "12" and the
    # "byte" were left to speak for themselves.
    assert _gloss_lines(pdf)
    assert all(gloss.y1 < 200 for gloss in _gloss_lines(pdf))
    # And nothing had to be opened up for the other two.
    assert _page_size(pdf)[1] < PAGE[1] + 40


def test_a_page_of_nothing_but_repeated_sources_is_carried_through_whole():
    builder = _Builder()

    for index in range(5):
        builder.text(60, 100 + index * 24, f"item {index}", 11.0,
                     gloss=f"item {index}", source=f"item {index}")

    pdf, report = _render(builder)

    assert report["drawn"] == 0
    assert report["skipped"] == 0
    assert _page_size(pdf) == PAGE
    assert not _arabic(pdf)


def test_a_merged_list_is_split_at_its_own_markers():
    """run44 p3: each item glossed with the wrong item's translation.

    BabelDOC merges a run of same-styled bullets into one paragraph, so a
    numbered list arrives as a single block. Cut by line-width weights, every
    gloss carried the tail of the previous item and the head of the next, and
    the last was the fragment «0 = 2».
    """
    source = ("a) The Moon is made of green cheese. b) Makkah is the Holy "
              "City of Islam. c) Madina is the capital of Saudi Arabia. "
              "d) 1+ 0= 1 e) 0+ 0= 2")
    target = ("أ) القمر مصنوع من الجبن الأخضر. ب) مكة هي المدينة المقدسة في "
              "الإسلام. ج) المدينة المنورة هي عاصمة المملكة العربية السعودية. "
              "د) 1 + 0 = 1 هـ) 0 + 0 = 2")

    chunks = interlinear._split_proportionally(target, [1.0] * 5, source)

    assert len(chunks) == 5
    assert chunks[0] == "أ) القمر مصنوع من الجبن الأخضر."
    assert chunks[4] == "هـ) 0 + 0 = 2"

    for marker, chunk in zip("أ ب ج د".split() + ["هـ"], chunks, strict=True):
        assert chunk.startswith(f"{marker})")


def test_prose_is_still_split_by_weight():
    """The split is proportional wherever the paragraph is not a list.

    One flowing sentence has no word that belongs to a particular source line,
    and nothing in it looks like a marker — so nothing changes for it.
    """
    chunks = interlinear._split_proportionally(
        "one two three four five six", [1.0, 1.0],
        "a flowing sentence with no markers in it at all")

    assert chunks == ["one two three", "four five six"]

    # A parenthesised term is not a list marker.
    assert interlinear._split_proportionally(
        "alpha beta gamma delta", [1.0, 1.0],
        "the (proposition) is a statement (either) true or false") == [
            "alpha beta", "gamma delta"]


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


def test_the_page_is_asked_where_it_may_be_cut_not_only_the_font():
    """The pixel check, on its own: ink the object model cannot see.

    A cut is a place the page is stretched, so the only question that matters
    is whether the page CHANGES from one row to the next there. Object bounds
    answer it from font and path metrics, and those can be wrong — which is
    the whole reason this check exists.

    It is a SECOND line of defence, not the only one: a strip taken wholly
    inside a thick mark is uniform and passes here, and is refused where it
    always was, by {@link interlinear._cut_items} keeping horizontal rules out
    of the gaps in the first place.
    """
    builder = _Builder()
    builder.page.draw_line(pymupdf.Point(50, 300), pymupdf.Point(545, 300),
                           color=(0, 0, 0), width=2.0)
    doc = pymupdf.open(stream=BytesIO(builder.pdf()), filetype="pdf")

    try:
        rows = interlinear._CutRows(doc[0])

        # Blank paper: nothing changes down it, so it may be cut anywhere.
        assert rows.constant(50, 545, 150, 150.6)
        # Onto the rule's edge: the page changes there, so it may not.
        assert not rows.constant(50, 545, 298.8, 299.4)
        # Beside the rule, where it does not reach.
        assert rows.constant(10, 45, 298.8, 299.4)
        # Off the page is not a place to cut either.
        assert not rows.constant(50, 545, -20, -19.4)
    finally:
        doc.close()


def test_a_cut_is_walked_up_out_of_ink_the_gaps_did_not_report():
    """The run44 defect, reproduced through the seam it came through.

    `_cut_above` works in gaps built from object bounds. run44's title face
    reports its 28 pt title as starting at y 63.0 while the glyphs reach up to
    y 47.2, so the gap above the title ran 16 pt down INSIDE the letters and
    the cut landed there: the sliced ascenders stayed above the band, the
    0.6 pt strip smeared them down it as gold bars, and the gloss was drawn on
    the wreckage — 16 of that document's 17 pages.

    Here the same lie is told directly: a title whose ink starts at y 303, a
    `line` claiming to start at 318, and a gap claiming clear air all the way
    down to it.
    """
    builder = _Builder()
    builder.text(60, 300, "Propositions", 28.0)
    doc = pymupdf.open(stream=BytesIO(builder.pdf()), filetype="pdf")

    try:
        rows = interlinear._CutRows(doc[0])

        def clean(cut, strip):
            return rows.constant(50, 545, cut - strip / 2, cut + strip / 2)

        # Where the glyphs really are, so the test cannot pass by accident if
        # the font's metrics change under it.
        ink = doc[0].get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"]
        assert ink[1] < 318 < ink[3]

        line = pymupdf.Rect(60, 318, 400, 340)
        gaps = [(200.0, 318.0)]

        trusting = interlinear._cut_above(line, gaps, [], 40.0, False)
        checked = interlinear._cut_above(line, gaps, [], 40.0, False, clean)

        # What the gaps alone say: cut at the foot of the gap, on a strip the
        # page itself reports as ink.
        assert trusting is not None
        assert not clean(*trusting)
        # What the page says: a strip that really is one flat slice of
        # background, well clear of the letters the other cut was inside.
        assert checked is not None
        assert clean(*checked)
        assert checked[0] < trusting[0] - 10
    finally:
        doc.close()


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


def test_a_paragraph_does_not_adopt_the_scenery_it_sits_on():
    """`_ink_bounds` decides ownership by centre, which scenery can exploit.

    A page-centred decorative ring, a full-height bracket, a tall arrow: any
    of them whose middle happens to fall inside a paragraph's band was adopted
    by that paragraph, and its anchor became most of the page. run22 p3 turned
    an 18.6 pt paragraph into a 405.7 pt one — and the anchor is what the
    reading order is sorted on.
    """
    box = pymupdf.Rect(60, 300, 300, 318)
    line = pymupdf.Rect(60, 302, 300, 316)
    ring = pymupdf.Rect(40, 100, 500, 520)

    assert (ring.y0 + ring.y1) / 2 == pytest.approx((box.y0 + box.y1) / 2,
                                                    abs=10)

    grown = interlinear._ink_bounds(box, [line, ring], 18.0)

    assert grown.height < 3 * box.height, f"the ring was adopted: {grown}"
    # And the ascent correction this exists for still happens.
    assert interlinear._ink_bounds(box, [pymupdf.Rect(60, 292, 300, 316)],
                                   18.0).y0 == pytest.approx(292, abs=0.1)


def test_a_deck_whose_decoration_is_forty_shapes_still_gets_line_glosses():
    """One big backdrop is recognised; forty small ones were not.

    run22's slides carry a concentric vignette drawn as ~40 curves. None is
    big enough to be a backdrop on its own, every one of them is a blocker,
    and between them they left the slide's main region 4.3 pt of free air in
    476 pt — so no line could find clear air above it, every paragraph fell
    through to "gloss above the whole region", and 415 of that document's 486
    glosses landed piled at the top of the page, out of reading order.

    A vignette is faint, which is the other half of why this is safe to
    ignore: the page's own pixels still have the last word on the cut, and
    they report a soft tone as the flat background it reads as.
    """
    builder = _Builder()

    for step in range(30):
        builder.page.draw_circle(pymupdf.Point(300, 300), 30 + step * 5,
                                 color=(0.97, 0.97, 0.98), width=0.8)

    lines = [builder.text(60, 150 + index * 40, f"Bullet {index} of the slide",
                          16.0, gloss=AR_ONE)
             for index in range(4)]

    pdf, report = _render(builder)

    assert report["drawn"] == 4
    assert report["skipped"] == 0

    arabic = _gloss_lines(pdf)
    latin = sorted((rect for rect, _text in _latin(pdf)), key=lambda r: r.y0)

    assert len(latin) == len(lines)

    # Each gloss stands over its own line, in order — not all four stacked
    # above the first.
    ceiling = 0.0

    for line in latin:
        above = _gloss_above(line, arabic, ceiling)
        assert above, f"nothing glosses {line}"
        assert min(gloss.y0 for gloss in above) > line.y0 - 60, \
            f"the gloss for {line} was hoisted away from it"
        ceiling = line.y1


def test_a_missing_font_size_is_guessed_from_the_line_not_the_box():
    """`font_size` is null on 431 corpus blocks, 243 of them taller than wide.

    The box HEIGHT used to stand in — and for a ROTATED block that is not the
    type size, it is the length of the line. run67 p6's vertical label
    "focus of this course" has a 94.5 pt tall box, so its gloss was asked for
    at 0.78 x 94.5 = 73.7 pt and clamped to the 28 pt ceiling.
    """
    rotated = pymupdf.Rect(500, 240, 513, 334)
    block = {"box": [500, 240, 513, 334], "lines": [[500, 240, 513, 334]],
             "target": AR_ONE, "font_size": None}

    # A line of type is as thick as its size whichever way round it is drawn.
    assert interlinear._source_size(block, rotated, 11.0) == pytest.approx(13.0)
    # The page's own body size is the ceiling on a guess, not a suggestion.
    assert interlinear._source_size(block, rotated, 5.0) == pytest.approx(10.0)
    # A recorded size is always believed, however odd the box.
    assert interlinear._source_size({**block, "font_size": 40.0}, rotated,
                                    11.0) == pytest.approx(40.0)
    # And the median is taken from what the page actually recorded.
    assert interlinear._typical_size(
        [{"font_size": 9.0}, {"font_size": 11.0}, {"font_size": 40.0},
         {"font_size": None}]) == pytest.approx(11.0)


def test_a_rotated_label_does_not_blow_the_page_open():
    """The same fault end to end: what the bad guess cost the page."""
    builder = _Builder()

    for index in range(3):
        builder.text(60, 100 + index * 20, "Body text on the slide", 11.0,
                     gloss=AR_ONE)

    label = pymupdf.Rect(500, 240, 513, 334)
    builder.blocks.append({"box": label, "lines": [label], "target": AR_TWO,
                           "font_size": None, "source": "focus of this course"})

    pdf, report = _render(builder)

    assert report["drawn"] == 4
    # Sized from the label, the page opens by ~116 pt; sized from the rotated
    # box it opened by 239.
    assert _page_size(pdf)[1] - PAGE[1] < 160


def test_a_caption_and_a_footnote_at_opposite_corners_are_not_a_column():
    """run8 p10: one full-page diagram, split in half and torn.

    The page is a single figure with a caption across the top
    (x 58.9-363.9) and a copyright line at the bottom right
    (x 371.8-470.7). The two never coexist vertically, but between them they
    leave a clear x range — so the page was read as two columns and split at
    x = 367.8. The two leaves then grew by different amounts and the diagram
    that spans both was sliced: its outline two mismatched halves 30 pt apart,
    the word "Partial" broken into "P" and "artial". The figure is a backdrop,
    so the splitter could not see the thing it was cutting.
    """
    rect = pymupdf.Rect(0, 0, 842, 595)
    caption = pymupdf.Rect(59, 30, 364, 44)
    footnote = pymupdf.Rect(372, 560, 471, 572)

    # Opposite corners: no height at all has marks on both sides.
    assert interlinear._coexist(368.0, [caption, footnote], rect) < 1.0

    # Two real columns running beside each other: most of the height does.
    left = pymupdf.Rect(59, 60, 364, 540)
    right = pymupdf.Rect(400, 60, 780, 540)

    assert (interlinear._coexist(382.0, [left, right], rect)
            > interlinear._MIN_COLUMN_COEXIST * rect.height)


def test_the_splitter_refuses_the_imagined_corridor():
    """The same page at the splitter: one region, so nothing can shear.

    Two leaves open their bands at different places, and whatever spans them
    is left in two halves at two heights — which on run8 p10 is the whole
    figure.
    """
    rect = pymupdf.Rect(0, 0, 842, 595)
    caption = pymupdf.Rect(59, 30, 364, 44)
    footnote = pymupdf.Rect(372, 560, 471, 572)

    imagined = interlinear._split_region(rect, [caption, footnote], 0, rect)

    assert imagined.axis != "columns", "the page was split at a coincidence"

    # And a page that really does have two columns is still split into them.
    columns = interlinear._split_region(
        rect, [pymupdf.Rect(59, 60, 364, 540),
               pymupdf.Rect(400, 60, 780, 540)], 0, rect)

    assert columns.axis == "columns"
    assert len(columns.children) == 2


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
# the text layer under the gloss
# --------------------------------------------------------------------------

def test_the_arabic_text_layer_repair_never_costs_the_overlay():
    """The repair lands with the engine-text fix; this file must not need it.

    An unguarded call on a build without that fix took compose's whole
    finishing pass down with it — every dual came back with no vocab strips
    and none of the compacting save. A text layer is never worth the page it
    is written on.
    """
    builder = _tight_list()

    def explode(_doc):
        raise RuntimeError("no repair on this build")

    from server import page_fonts

    before = getattr(page_fonts, "repair_arabic_text_layer", None)
    page_fonts.repair_arabic_text_layer = explode

    try:
        pdf, report = _render(builder)
    finally:
        if before is None:
            del page_fonts.repair_arabic_text_layer
        else:
            page_fonts.repair_arabic_text_layer = before

    assert report["drawn"] == 5
    assert len(_gloss_lines(pdf)) == 5


def test_the_arabic_text_layer_repair_is_offered_the_finished_document():
    builder = _tight_list()
    seen = []

    def record(doc):
        seen.append(doc.page_count)

        return 0

    from server import page_fonts

    before = getattr(page_fonts, "repair_arabic_text_layer", None)
    page_fonts.repair_arabic_text_layer = record

    try:
        _render(builder)
    finally:
        if before is None:
            del page_fonts.repair_arabic_text_layer
        else:
            page_fonts.repair_arabic_text_layer = before

    # Called once, after the pages are built and before the save.
    assert seen == [1]


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


def test_a_label_boxed_in_by_table_rules_is_still_glossed():
    """The pixel veto can never be cleared by a label in a table cell.

    There is a rule a couple of points above it and another a couple below,
    so neither band is ever quiet — and 738 of the corpus's 1058 in-image
    blocks were skipped for it (run59 96.6%, run39 93.2%). run39 p6 is a
    physics slide whose entire content is two rendered tables of SI units,
    delivered untranslated.

    A translucent plate clipping a table rule is a far better outcome for the
    reader than an untranslated table.
    """
    builder = _Builder()
    image = pymupdf.Rect(100, 200, 400, 440)
    label = pymupdf.Rect(150, 300, 260, 312)

    scale = 4
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(
        0, 0, int(image.width * scale), int(image.height * scale)))
    pix.set_rect(pix.irect, (250, 250, 250))

    def bar(rect, color):
        pix.set_rect(pymupdf.IRect(int((rect.x0 - image.x0) * scale),
                                   int((rect.y0 - image.y0) * scale),
                                   int((rect.x1 - image.x0) * scale),
                                   int((rect.y1 - image.y0) * scale)), color)

    # A row of the table: the label, with a rule two points above and below.
    bar(label, (40, 40, 60))
    bar(pymupdf.Rect(image.x0, label.y0 - 3, image.x1, label.y0 - 2),
        (30, 30, 30))
    bar(pymupdf.Rect(image.x0, label.y1 + 2, image.x1, label.y1 + 3),
        (30, 30, 30))
    builder.page.insert_image(image, pixmap=pix)

    height = PAGE[1]
    sidecar = builder.sidecar()
    sidecar["pages"][0]["blocks"].append({
        "box": [label.x0, height - label.y1, label.x1, height - label.y0],
        "lines": [[label.x0, height - label.y1, label.x1, height - label.y0]],
        "source": "candela",
        "target": AR_TWO,
        "font_size": 10.0,
        "label": "plain text",
        "on_raster": True,
        "region": [image.x0, height - image.y1, image.x1, height - image.y0],
    })

    pdf, report = interlinear.render_overlay(builder.pdf(), sidecar,
                                             style=STYLE)

    assert report["raster_drawn"] == 1, "the boxed-in label got no gloss"
    assert report["raster_skipped"] == 0

    # Drawn on the artwork, but never on the label it glosses.
    inside = [rect for rect in _arabic(pdf) if image.contains(rect)]
    assert inside

    for rect in inside:
        overlap = rect & label
        assert not overlap.is_valid or overlap.is_empty


def test_a_crowded_plate_never_covers_another_label():
    """What the crowded retry may still not do."""
    builder = _Builder()
    image = pymupdf.Rect(100, 200, 400, 440)
    rows = [pymupdf.Rect(150, 290 + index * 16, 260, 302 + index * 16)
            for index in range(3)]

    scale = 4
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(
        0, 0, int(image.width * scale), int(image.height * scale)))
    pix.set_rect(pix.irect, (250, 250, 250))

    for row in rows:
        pix.set_rect(pymupdf.IRect(int((row.x0 - image.x0) * scale),
                                   int((row.y0 - image.y0) * scale),
                                   int((row.x1 - image.x0) * scale),
                                   int((row.y1 - image.y0) * scale)),
                     (40, 40, 60))

    builder.page.insert_image(image, pixmap=pix)

    height = PAGE[1]
    sidecar = builder.sidecar()
    sidecar["pages"][0]["blocks"] += [{
        "box": [row.x0, height - row.y1, row.x1, height - row.y0],
        "lines": [[row.x0, height - row.y1, row.x1, height - row.y0]],
        "source": f"row {index}",
        "target": AR_TWO,
        "font_size": 10.0,
        "label": "plain text",
        "on_raster": True,
        "region": [image.x0, height - image.y1, image.x1, height - image.y0],
    } for index, row in enumerate(rows)]

    pdf, _report = interlinear.render_overlay(builder.pdf(), sidecar,
                                              style=STYLE)

    for rect in _arabic(pdf):
        for row in rows:
            overlap = rect & row
            assert not overlap.is_valid or overlap.is_empty, \
                f"a gloss was drawn over the label at {row}"
