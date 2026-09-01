"""Tests for server/page_fonts.py — the appendix pages' fonts and text layer.

Run from the repo root (pyproject's testpaths only covers tests/):

    pytest server/test_page_fonts.py
"""

import pymupdf
import pytest
from babeldoc.format.pdf.document_il.backend import pdf_creater

from server import page_fonts

# The vocabulary strip's own heading, and a gloss line shaped like the ones the
# strip really draws. Both are written the way a person types them — which is
# the whole point: this is what has to come back out.
HEADING = "كلمات هذه الصفحة"
GLOSS = "التجريد — non-functional"

PRESENTATION = range(0xFB50, 0xFF00)


@pytest.fixture(scope="module", autouse=True)
def fonts_cached():
    """Fetch the faces once — rendering real Arabic is what these tests do."""
    page_fonts._font_path(page_fonts.FONT_FILE)
    page_fonts._font_path(page_fonts.BOLD_FONT_FILE)


_FONT_CACHE: dict[tuple[str, ...], page_fonts.PageFonts] = {}


def _strip_page(texts=(HEADING, GLOSS)):
    """A page set the way an appendix renderer sets one: Story, subset face.

    The subset is cached per text set: taking it means walking two 15 MB
    faces, and every test here wants the same handful of pages.
    """
    key = tuple(texts)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = page_fonts.PageFonts(list(texts))
    fonts = _FONT_CACHE[key]
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=120)
    body = "".join(f'<div dir="rtl" style="font-size:14px">{text}</div>'
                   for text in texts)
    page.insert_htmlbox(pymupdf.Rect(10, 10, 410, 110), body,
                        css=fonts.css, archive=fonts.archive)
    return doc


def _reopened(doc):
    """`doc` as a reader sees it — saved, folded, and parsed back."""
    return pymupdf.open(stream=doc.tobytes(garbage=4, deflate=True),
                        filetype="pdf")


def _text(doc, index=0):
    return doc[index].get_text()


# --------------------------------------------------------------------------
# The defect: what an appendix page's text layer says it is
# --------------------------------------------------------------------------

def test_an_unrepaired_strip_extracts_as_neither_letters_nor_search_hits():
    # Pins the bug this module exists to fix, so the fix cannot quietly stop
    # mattering: the page renders «كلمات هذه الصفحة» and the text under it is
    # something else entirely.
    doc = _reopened(_strip_page())
    try:
        text = _text(doc)
        assert HEADING not in text
        assert not doc[0].search_for(HEADING)
        assert any(ord(char) in PRESENTATION for char in text)
    finally:
        doc.close()


def test_repair_makes_the_strip_extract_as_the_arabic_it_renders():
    doc = _strip_page()
    try:
        assert page_fonts.repair_arabic_text_layer(doc) >= 1
        saved = _reopened(doc)
        try:
            text = _text(saved)
            assert HEADING in text
            assert "التجريد" in text
            assert saved[0].search_for(HEADING)
            assert not any(ord(char) in PRESENTATION for char in text)
        finally:
            saved.close()
    finally:
        doc.close()


def test_repair_recovers_the_glyphs_that_have_no_cmap_entry_at_all():
    # Every «ل» and the hyphen inside an Arabic run are reached through GSUB,
    # so they have nothing to reverse-map from and a viewer reads their glyph
    # id as a codepoint: «كلمات» came out «ﻛɅﻤﺎت», and the junk differed per
    # document. Nothing may extract from the Latin Extended / Cyrillic
    # supplements that fallback lands in.
    doc = _strip_page()
    try:
        page_fonts.repair_arabic_text_layer(doc)
        saved = _reopened(doc)
        try:
            text = _text(saved)
            junk = {char for char in text
                    if 0x0180 <= ord(char) <= 0x024F
                    or 0x0370 <= ord(char) <= 0x052F}
            assert not junk
            assert "non-functional" in text
        finally:
            saved.close()
    finally:
        doc.close()


def test_repair_spells_the_lam_alef_ligature_as_two_letters():
    # One glyph, two characters. Dropping either one is what would make a
    # word containing «لا» unsearchable while looking perfectly correct.
    doc = _strip_page(["الطالب لا يقرأ"])
    try:
        page_fonts.repair_arabic_text_layer(doc)
        saved = _reopened(doc)
        try:
            assert "الطالب لا يقرأ" in _text(saved)
        finally:
            saved.close()
    finally:
        doc.close()


def test_repair_does_not_touch_the_pixels():
    # The risk of this fix, named: it rewrites what a glyph CLAIMS to be and
    # must not touch which glyph is drawn.
    doc = _strip_page()
    try:
        before = _reopened(doc)[0].get_pixmap(dpi=96)
        page_fonts.repair_arabic_text_layer(doc)
        after = _reopened(doc)[0].get_pixmap(dpi=96)
        assert bytes(after.samples) == bytes(before.samples)
    finally:
        doc.close()


def _objects(doc):
    """`xref -> (definition, stream)` — a whole-file fingerprint that skips
    the /ID, which pymupdf rewrites on every save."""
    return {xref: (doc.xref_object(xref),
                   doc.xref_stream(xref) if doc.xref_is_stream(xref) else None)
            for xref in range(1, doc.xref_length())}


def test_repair_leaves_a_document_with_no_arabic_exactly_as_it_was():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Discrete Structures", fontsize=18)
    try:
        before = _objects(doc)
        assert page_fonts.repair_arabic_text_layer(doc) == 0
        assert _objects(doc) == before
    finally:
        doc.close()


def test_repair_only_rewrites_the_fonts_that_carry_arabic():
    # A translated page keeps the original document's own fonts alongside the
    # Arabic one. Those must come through untouched.
    doc = _strip_page()
    doc[0].insert_text((20, 110), "Discrete Structures", fontsize=9)
    try:
        before = _objects(doc)
        page_fonts.repair_arabic_text_layer(doc)
        changed = [xref for xref, obj in _objects(doc).items()
                   if obj != before[xref]]
        assert changed
        # Every object we touched is a ToUnicode CMap. Nothing else in the
        # file — not a content stream, not the original document's own fonts.
        for xref in changed:
            assert b"begincmap" in (doc.xref_stream(xref) or b"")
    finally:
        doc.close()


def test_repair_is_idempotent():
    # babeldoc already ran this pass over its own output before the appendix
    # renderer opens the file, so the second call sees a document that is
    # half repaired: it must finish the job and touch nothing else.
    doc = _strip_page()
    try:
        assert page_fonts.repair_arabic_text_layer(doc) >= 1
        settled = _objects(doc)
        assert page_fonts.repair_arabic_text_layer(doc) == 0
        assert _objects(doc) == settled
    finally:
        doc.close()


def test_repair_never_raises_at_its_caller():
    # It is called from inside a guard whose job is to keep the run alive, so
    # a failure here has to cost the text layer and nothing else.
    class Hostile:
        def xref_length(self):
            raise RuntimeError("no")

    assert page_fonts.repair_arabic_text_layer(Hostile()) == 0


# --------------------------------------------------------------------------
# The pieces underneath
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("form", "letters"), [
    ("ﻟ", "ل"),        # initial lam
    ("ﻠ", "ل"),        # medial lam
    ("ﺎ", "ا"),        # final alef
    ("ﻻ", "لا"),       # lam-alef: one glyph, two letters
    ("ﻷ", "لأ"),       # lam-alef with hamza
    ("ﹶ", "َ"),   # a mark, without the space it decomposes with
])
def test_arabic_base_letters(form, letters):
    assert pdf_creater.arabic_base_letters(ord(form)) == letters


def test_normalize_leaves_everything_that_is_not_a_presentation_form():
    mixed = "الـ API 3.14 — ﺍﻟﻘﻀﻴﺔ"
    assert pdf_creater.normalize_arabic_presentation_forms(mixed) == \
        "الـ API 3.14 — القضية"


def test_a_cmap_it_cannot_read_whole_is_refused_rather_than_rewritten():
    # All-or-nothing on purpose: rewriting a CMap we only half parsed would
    # silently drop whatever we failed to read.
    assert pdf_creater.parse_tounicode_map(
        b"beginbfchar <0003> /notdef endbfchar") is None


def test_ranges_arrays_and_scalar_destinations_all_parse():
    parsed = pdf_creater.parse_tounicode_map(
        b"1 beginbfrange <0003> <0005> <0041> endbfrange\n"
        b"1 beginbfrange <0010> <0011> [<0628> <06280629>] endbfrange\n"
        b"1 beginbfchar <0020> <10780> endbfchar")
    assert parsed is not None
    mapping, nibbles = parsed
    assert nibbles == 4
    assert {code: text for code, (text, _) in mapping.items()} == {
        0x03: "A", 0x04: "B", 0x05: "C",
        0x10: "ب", 0x11: "بة",
        0x20: "\U00010780",
    }


def test_a_rewritten_cmap_round_trips_through_its_own_parser():
    # The rewriter keeps ranges where the destinations still count up, which
    # is what holds the delivered file's size flat. Nothing may shift on the
    # way through: a range that counted one code too far would silently
    # rename a glyph.
    mapping = {}
    mapping.update({code: (chr(0x41 + i), f"{0x41 + i:04x}")
                    for i, code in enumerate(range(3, 90))})       # a long run
    mapping.update({0x100: ("ل", "0644"), 0x101: ("ل", "0644"),    # collapsed
                    0x102: ("لا", "06440627")})                    # two letters
    mapping[0x200] = ("\U00010780", "10780")                       # astral
    # A run that would cross a 256 boundary if the last byte kept counting.
    mapping.update({0x300 + i: (chr(0x24F0 + i), f"{0x24F0 + i:04x}")
                    for i in range(24)})

    body = pdf_creater.render_tounicode_body(mapping, 4)
    reparsed = pdf_creater.parse_tounicode_map(
        b"endcodespacerange\n" + body.encode("ascii") + b"\nendcmap")

    assert reparsed is not None
    assert {code: text for code, (text, _) in reparsed[0].items()} == \
        {code: text for code, (text, _) in mapping.items()}
    assert b"beginbfrange" in body.encode("ascii")
