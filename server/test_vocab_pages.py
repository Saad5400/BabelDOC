"""Tests for server/vocab_pages.py — the «كلمات هذه الصفحة» renderer.

Run from the repo root (pyproject's testpaths only covers tests/):

    pytest server/test_vocab_pages.py
"""

import unicodedata

import pymupdf
import pytest

from server import page_fonts
from server import vocab_pages

PAGE = (720.0, 540.0)  # slide-shaped, like the decks this was built for
WIDESCREEN = (960.0, 540.0)  # a 16:9 deck at PowerPoint's 13.33"x7.5"


@pytest.fixture(scope="module", autouse=True)
def fonts_cached():
    """Fetch the faces once — rendering real Arabic is what these tests do."""
    page_fonts._font_path(page_fonts.FONT_FILE)
    page_fonts._font_path(page_fonts.BOLD_FONT_FILE)


def _doc(pages=2, size=PAGE):
    doc = pymupdf.open()
    for index in range(pages):
        page = doc.new_page(width=size[0], height=size[1])
        page.insert_text((72, 72), f"Slide {chr(97 + index)}", fontsize=24)
    return doc


def _rows(count):
    # Letter suffixes, not digits: a digit shaped inside an RTL run can be
    # drawn via a GSUB variant glyph with no cmap entry, which extracts as
    # junk (a characteristic of the GoNotoKurrent face).
    return [{"w": f"word{chr(97 + i // 26)}{chr(97 + i % 26)}",
             "ar": "معنى الكلمة", "note": "ملاحظة قصيرة"}
            for i in range(count)]


def _text(doc, index):
    return unicodedata.normalize("NFKC", doc[index].get_text())


# --------------------------------------------------------------------------
# sanitize_vocab
# --------------------------------------------------------------------------

def test_sanitize_keeps_clean_rows_and_drops_junk():
    cleaned = vocab_pages.sanitize_vocab({
        "0": [{"w": "declared", "ar": "يُعرَّف"}, "junk", {"w": "", "ar": "x"},
              {"w": "scope", "ar": "نطاق", "note": "توضيح"}],
        "banana": [{"w": "lost", "ar": "ضائع"}],
        "-1": [{"w": "lost", "ar": "ضائع"}],
        "2": "not a list",
    })

    assert cleaned == {0: [{"w": "declared", "ar": "يُعرَّف"},
                           {"w": "scope", "ar": "نطاق", "note": "توضيح"}]}


def test_sanitize_reimposes_the_caps():
    over = {"0": [{"w": f"w{chr(97 + i // 26)}{chr(97 + i % 26)}", "ar": "معنى"}
                  for i in range(30)]}
    many_pages = {str(n): [{"w": f"p{n}w{chr(97 + i)}", "ar": "معنى"}
                           for i in range(20)] for n in range(25)}

    assert len(vocab_pages.sanitize_vocab(over)[0]) == 20
    total = vocab_pages.sanitize_vocab(many_pages)
    assert sum(len(rows) for rows in total.values()) == 400
    assert 0 in total  # ascending pages: the earliest words survive
    assert 24 not in total


def test_sanitize_clips_oversized_strings():
    cleaned = vocab_pages.sanitize_vocab(
        {"0": [{"w": "x" * 500, "ar": "y" * 500, "note": "z" * 5000}]})

    row = cleaned[0][0]
    assert len(row["w"]) == 80
    assert len(row["ar"]) == 120
    assert len(row["note"]) == 200


# --------------------------------------------------------------------------
# interleave_vocab: rendering and placement
# --------------------------------------------------------------------------

def test_a_pages_vocab_is_inserted_right_after_it():
    doc = _doc(pages=2)

    added = vocab_pages.interleave_vocab(doc, {"0": _rows(4)}, {0: 0})

    assert added == {0: 1}
    assert doc.page_count == 3
    assert doc[1].rect.width == PAGE[0]
    assert doc[1].rect.height == PAGE[1]
    text = _text(doc, 1)
    assert "worda" in text
    # The title is there; the shaped lam can extract via a GSUB variant glyph
    # with no cmap entry (a characteristic of the GoNotoKurrent face), so
    # assert on the words that survive extraction rather than the exact string.
    assert "هذه" in text
    assert "معنى" in text
    # The neighbours are untouched content pages.
    assert "Slide a" in _text(doc, 0)
    assert "Slide b" in _text(doc, 2)
    doc.close()


def test_multiple_pages_interleave_back_to_front():
    doc = _doc(pages=3)

    added = vocab_pages.interleave_vocab(
        doc, {"0": _rows(2), "2": _rows(3)}, {0: 0, 2: 2})

    # c0 v0 c1 c2 v2 — page 1 has no vocab and gets nothing.
    assert added == {0: 1, 2: 1}
    assert doc.page_count == 5
    assert "worda" in _text(doc, 1)
    assert "Slide b" in _text(doc, 2)
    assert "Slide c" in _text(doc, 3)
    assert "worda" in _text(doc, 4)
    doc.close()


def test_more_than_six_entries_flow_into_two_columns():
    doc = _doc(pages=1)
    vocab_pages.interleave_vocab(doc, {"0": _rows(8)}, {0: 0})

    words = doc[1].get_text("words")
    xs = [word[0] for word in words if word[4].startswith("word")]
    middle = PAGE[0] / 2

    # Rows landed in BOTH halves of the page (the RTL right column first).
    assert any(x > middle for x in xs)
    assert any(x < middle for x in xs)
    doc.close()


def test_six_entries_stay_in_one_full_width_column():
    doc = _doc(pages=1)
    vocab_pages.interleave_vocab(doc, {"0": _rows(6)}, {0: 0})

    words = doc[1].get_text("words")
    xs = [word[0] for word in words if word[4].startswith("word")]

    # A single RTL column: every row starts in the right half.
    assert xs
    assert all(x > PAGE[0] / 2 for x in xs)
    doc.close()


def test_a_full_twenty_word_page_stays_on_one_widescreen_slide():
    # The extraction's per-page cap (20 rows, vocab.MAX_PER_PAGE) on a 16:9
    # deck page: two balanced columns of ten, ONE inserted page, every row
    # inside the margins.
    for size in (WIDESCREEN, (720.0, 405.0)):  # both common 16:9 exports
        doc = _doc(pages=1, size=size)
        rows = _rows(20)

        added = vocab_pages.interleave_vocab(doc, {"0": rows}, {0: 0})

        assert added == {0: 1}
        assert doc.page_count == 2
        words = doc[1].get_text("words")
        drawn = [word for word in words if word[4].startswith("word")]
        assert len(drawn) == 20
        middle = size[0] / 2
        assert sum(word[0] > middle for word in drawn) == 10  # right column
        assert sum(word[0] < middle for word in drawn) == 10  # left column
        assert all(word[3] <= size[1] - 40.0 + 1 for word in drawn)  # margin
        doc.close()


def test_an_overfull_page_continues_onto_a_second():
    doc = _doc(pages=1, size=(300.0, 160.0))

    added = vocab_pages.interleave_vocab(doc, {"0": _rows(12)}, {0: 0})

    assert added[0] > 1
    assert doc.page_count == 1 + added[0]
    # Every row made it somewhere.
    text = "".join(_text(doc, i) for i in range(1, doc.page_count))
    for row in _rows(12):
        assert row["w"] in text
    doc.close()


def test_junk_vocab_inserts_nothing():
    doc = _doc(pages=1)
    before = doc.page_count

    assert vocab_pages.interleave_vocab(doc, {"0": ["junk", 7]}, {0: 0}) == {}
    assert vocab_pages.interleave_vocab(doc, "not a dict", {0: 0}) == {}
    assert vocab_pages.interleave_vocab(doc, {}, {0: 0}) == {}
    assert doc.page_count == before
    doc.close()


def test_an_anchor_outside_the_document_is_skipped():
    doc = _doc(pages=1)

    added = vocab_pages.interleave_vocab(
        doc, {"0": _rows(2), "7": _rows(2)}, {0: 0, 7: 7})

    assert added == {0: 1}
    assert doc.page_count == 2
    doc.close()


def test_a_page_too_small_for_the_layout_is_skipped_not_raised():
    doc = _doc(pages=1, size=(40.0, 40.0))

    assert vocab_pages.interleave_vocab(doc, {"0": _rows(2)}, {0: 0}) == {}
    assert doc.page_count == 1
    doc.close()


# --------------------------------------------------------------------------
# attach_vocab: the bottom-strip layout (the production path)
# --------------------------------------------------------------------------

def test_a_strip_grows_the_page_and_keeps_its_content():
    doc = _doc(pages=2)
    base = doc[0].rect.height

    added = vocab_pages.attach_vocab(doc, {"0": _rows(4)}, {0: 0})

    assert set(added) == {0}
    assert added[0] > 0  # a strip, not a fallback insertion
    assert doc.page_count == 2  # nothing was inserted
    assert doc[0].rect.height == pytest.approx(base + added[0])
    assert doc[1].rect.height == base  # the stripless neighbour is untouched
    text = _text(doc, 0)
    assert "Slide a" in text  # the content survived, on the same page
    assert "worda" in text
    assert "معنى" in text
    doc.close()


def test_the_strip_sits_entirely_below_the_original_content():
    doc = _doc(pages=1)
    base = doc[0].rect.height

    vocab_pages.attach_vocab(doc, {"0": _rows(6)}, {0: 0})

    drawn = [word for word in doc[0].get_text("words")
             if word[4].startswith("word") or "معنى" in word[4]]
    assert drawn
    assert all(word[1] > base for word in drawn)  # every row is in the band
    doc.close()


def test_a_wide_page_flows_the_strip_into_several_columns():
    doc = _doc(pages=1, size=WIDESCREEN)

    added = vocab_pages.attach_vocab(doc, {"0": _rows(9)}, {0: 0})

    assert added[0] > 0
    xs = [word[0] for word in doc[0].get_text("words")
          if word[4].startswith("word")]
    middle = WIDESCREEN[0] / 2
    # Rows landed in BOTH halves of the band (the RTL right column first).
    assert any(x > middle for x in xs)
    assert any(x < middle for x in xs)
    doc.close()


def test_a_rotated_page_falls_back_to_an_inserted_page():
    doc = _doc(pages=1)
    doc[0].set_rotation(90)

    added = vocab_pages.attach_vocab(doc, {"0": _rows(3)}, {0: 0})

    assert added == {0: -1.0}  # -N = N fallback pages
    assert doc.page_count == 2
    assert "worda" in _text(doc, 1)
    doc.close()


def test_rows_that_would_dwarf_the_page_fall_back_too():
    doc = _doc(pages=1, size=(300.0, 160.0))

    added = vocab_pages.attach_vocab(doc, {"0": _rows(12)}, {0: 0})

    assert added[0] < 0  # inserted pages, not a strip taller than the slide
    assert doc.page_count == 1 + int(-added[0])
    assert doc[0].rect.height == 160.0  # the content page was left alone
    doc.close()


def test_a_band_over_half_the_page_falls_back_rather_than_growing_it():
    # run32's shape: 14 rows measured 277pt under a 595x335 slide — 83% of
    # the slide, 45% of the delivered sheet, and it passed the old 0.9 guard.
    short = pymupdf.open()
    short.new_page(width=595.0, height=335.0)

    added = vocab_pages.attach_vocab(short, {"0": _rows(14)}, {0: 0})

    assert added[0] < 0  # an inserted page, not a band the size of the slide
    assert short[0].rect.height == 335.0  # the slide keeps its own geometry
    short.close()

    # The same rows on a page tall enough to carry them still get their strip:
    # it is the SHARE of the page that decides, not the number of rows.
    tall = pymupdf.open()
    tall.new_page(width=595.0, height=842.0)

    added = vocab_pages.attach_vocab(tall, {"0": _rows(14)}, {0: 0})

    assert added[0] > 0
    assert added[0] < 0.5 * 842.0
    assert tall.page_count == 1
    tall.close()


def test_attach_junk_vocab_touches_nothing():
    doc = _doc(pages=1)
    before = doc[0].rect.height

    assert vocab_pages.attach_vocab(doc, {"0": ["junk", 7]}, {0: 0}) == {}
    assert vocab_pages.attach_vocab(doc, "not a dict", {0: 0}) == {}
    assert doc.page_count == 1
    assert doc[0].rect.height == before
    doc.close()


def test_a_page_with_no_anchor_is_dropped_out_loud(caplog):
    # A key past the end of the document — what a model that echoed a printed
    # slide number produces — used to take a whole page of words with it in
    # silence.
    doc = _doc(pages=2)

    with caplog.at_level("WARNING", logger="doctranslate.vocab_pages"):
        added = vocab_pages.attach_vocab(
            doc, {"0": _rows(2), "9": _rows(3)}, {0: 0, 1: 1})

    assert set(added) == {0}
    assert "9" in caplog.text
    assert "3 word(s)" in caplog.text
    doc.close()


def test_the_strip_lives_at_negative_pdf_y():
    # The compose restore contract: the strip occupies exactly
    # [mediabox.y0, y0 + height) below the original zero line, so cropping
    # the recorded height back off recovers the pristine page.
    doc = _doc(pages=1)

    added = vocab_pages.attach_vocab(doc, {"0": _rows(3)}, {0: 0})

    media = doc[0].mediabox
    assert media.y0 == pytest.approx(-added[0])
    assert media.y1 == PAGE[1]
    doc.close()


# --------------------------------------------------------------------------
# interleave_vocab: same-named conflicting subsets in the target
# --------------------------------------------------------------------------

def _conflicting_target(size=PAGE):
    """A raw saved-and-reopened doc that embeds its OWN GoNotoKurrent subsets.

    The page's Arabic is a different text set than the vocab rows', so its
    subsets carry a different glyph numbering under the very same face names —
    what a babeldoc-produced mono embeds. Vocab pages are drawn directly into
    the target ({@link vocab_pages.draw_vocab_pages}); this fixture pins that
    the drawn text keeps rendering from its own subsets, pixel-identical to a
    blank-document render, with the same-named strangers alongside.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=size[0], height=size[1])
    fonts = page_fonts.PageFonts(
        ["غضب الخط شيء آخر تماما وليس من كلمات الصفحة"],
        "div.x {font-family: glossbold; font-size: 14px;}")
    page.insert_htmlbox(
        pymupdf.Rect(48, 48, size[0] - 48, 200),
        '<div class="x" dir="rtl">غضب الخط شيء آخر تماما</div>',
        css=fonts.css, archive=fonts.archive)
    data = doc.tobytes(garbage=4, deflate=True)
    doc.close()

    return pymupdf.open(stream=data, filetype="pdf")


def _saved_page_pixmap(doc, index):
    """Page `index` as rendered from the SAVED file (garbage=4, reopened)."""
    data = doc.tobytes(garbage=4, deflate=True)
    saved = pymupdf.open(stream=data, filetype="pdf")
    try:
        return saved[index].get_pixmap(dpi=96)
    finally:
        saved.close()


def test_vocab_arabic_survives_a_conflicting_embedded_subset():
    rows = _rows(5)

    # The reference: the same rows into a blank document — the standalone
    # render, known-correct. Pixels, not extracted text: extraction reads
    # ToUnicode, which stayed correct even in the broken files.
    reference = _doc(pages=1)
    assert vocab_pages.interleave_vocab(reference, {"0": rows}, {0: 0})
    want = _saved_page_pixmap(reference, 1)
    reference.close()

    target = _conflicting_target()
    assert vocab_pages.interleave_vocab(target, {"0": rows}, {0: 0})
    got = _saved_page_pixmap(target, 1)
    target.close()

    assert (got.width, got.height) == (want.width, want.height)
    mismatch = sum(a != b for a, b in zip(got.samples, want.samples,
                                          strict=True))
    assert mismatch / len(want.samples) < 0.005


def _saved_band_pixmap(doc, index, band):
    """The band rect of page `index`, rendered from the SAVED file."""
    data = doc.tobytes(garbage=4, deflate=True)
    saved = pymupdf.open(stream=data, filetype="pdf")
    try:
        return saved[index].get_pixmap(dpi=96, clip=band)
    finally:
        saved.close()


def test_strip_arabic_survives_a_conflicting_embedded_subset():
    # The strip draws DIRECTLY onto a page that already embeds its own
    # same-named GoNotoKurrent subsets (every babeldoc mono does). Pin that
    # the band renders pixel-identical to the same band drawn on a blank
    # page — pixels, not extracted text, for the same ToUnicode reason as
    # the page-variant test above.
    rows = _rows(5)

    reference = pymupdf.open()
    reference.new_page(width=PAGE[0], height=PAGE[1])
    added = vocab_pages.attach_vocab(reference, {"0": rows}, {0: 0})
    assert added and added[0] > 0
    band = pymupdf.Rect(0, PAGE[1], PAGE[0], PAGE[1] + added[0])
    want = _saved_band_pixmap(reference, 0, band)
    reference.close()

    target = _conflicting_target()
    got_added = vocab_pages.attach_vocab(target, {"0": rows}, {0: 0})
    assert got_added[0] == pytest.approx(added[0])
    got = _saved_band_pixmap(target, 0, band)
    target.close()

    assert (got.width, got.height) == (want.width, want.height)
    mismatch = sum(a != b for a, b in zip(got.samples, want.samples,
                                          strict=True))
    assert mismatch / len(want.samples) < 0.005
