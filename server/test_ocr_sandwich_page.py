"""Tests for who the scanned lane is allowed to erase (fork side).

`--ocr-workaround` is turned on for a whole DOCUMENT, by majority vote: more
than half of record 151's 154 pages are scans, so the flag went on and every
page was treated as one. Two of the things the flag licenses are destructive:

    paragraph_finder.process_page
        add_text_fill_background(page)   # an opaque plate over every paragraph
        page.pdf_character = []          # every character no paragraph claimed

Both are safe on an ocrmypdf sandwich page, where the raster underneath still
says everything the text layer said — and only there. Record 151's first 29
pages are typeset, not scanned: on those the characters ARE the page, and the
document-wide flag deleted them. MEASURED on p.14 (0-based 13), rendering the
same page through the same fork with the flag on and off:

    with the flag   exercises 12 and 14-18 gone, 21-24 gone, 27-32 gone,
                    52-56 and 59 gone, and every value in both columns of
                    the radical/exponent table gone
    without it      all present

and on p.20 the whole worked solution (x^2+5x=24 ... x=3, x=-8) and the
"Check Your Answers" box came back.

So the licence is granted per page now, on the evidence that makes it safe:
the page's own text is invisible ink laid over its own picture. MEASURED
render-mode census on record 151 through the real OCR lane —

    raster pages   42/42, 1245/1245, 1049/1049 characters invisible
    digital pages  0/2431, 0/1709

Run from the repo root:

    pytest server/test_ocr_sandwich_page.py
"""

from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.utils import layout_helper
from babeldoc.format.pdf.document_il.utils.layout_helper import is_ocr_sandwich_page

INVISIBLE = layout_helper.INVISIBLE_TEXT_RENDER_MODE


def _box(x0, y0, x1, y1):
    return il_version_1.Box(x=x0, y=y0, x2=x1, y2=y1)


def _char(index, render_mode=None):
    char = il_version_1.PdfCharacter(
        box=_box(index * 5.0, 100.0, index * 5.0 + 5.0, 110.0),
        char_unicode="x",
    )
    if render_mode is not None:
        char.render_mode = render_mode
    return char


def _page(loose=(), paragraphs=()):
    return il_version_1.Page(
        mediabox=il_version_1.Mediabox(box=_box(0, 0, 612, 792)),
        cropbox=il_version_1.Cropbox(box=_box(0, 0, 612, 792)),
        page_number=0,
        unit="point",
        base_operations=il_version_1.BaseOperations(value=""),
        pdf_character=list(loose),
        pdf_paragraph=list(paragraphs),
    )


def _paragraph_of(chars):
    """One paragraph holding `chars` as a single line."""
    return il_version_1.PdfParagraph(
        box=_box(0, 100, 200, 110),
        pdf_paragraph_composition=[
            il_version_1.PdfParagraphComposition(
                pdf_line=il_version_1.PdfLine(
                    box=_box(0, 100, 200, 110), pdf_character=list(chars)
                )
            )
        ],
    )


# --------------------------------------------------- the verdict itself


def test_an_ocrmypdf_sandwich_page_is_recognised():
    # Every character invisible: the pixels are the content.
    page = _page(loose=[_char(i, INVISIBLE) for i in range(40)])
    assert is_ocr_sandwich_page(page) is True


def test_a_digital_page_is_not_a_sandwich():
    # Record 151 p.14: 2431 characters, none of them invisible.
    page = _page(loose=[_char(i) for i in range(40)])
    assert is_ocr_sandwich_page(page) is False


def test_the_census_reaches_characters_already_inside_paragraphs():
    # styles_and_formulas asks the same question one stage later, by which
    # time the characters have moved into paragraphs.
    page = _page(paragraphs=[_paragraph_of(_char(i, INVISIBLE) for i in range(40))])
    assert is_ocr_sandwich_page(page) is True

    page = _page(paragraphs=[_paragraph_of(_char(i) for i in range(40))])
    assert is_ocr_sandwich_page(page) is False


def test_a_handful_of_invisible_runs_does_not_condemn_a_digital_page():
    # The image-text lane injects invisible OCR runs over embedded rasters on
    # otherwise-digital pages. A few of those must not hand the whole page to
    # the eraser.
    chars = [_char(i) for i in range(40)] + [
        _char(i, INVISIBLE) for i in range(40, 44)
    ]
    assert is_ocr_sandwich_page(_page(loose=chars)) is False


def test_a_page_with_no_text_is_not_a_sandwich():
    # Nothing to erase and nothing to mask: both answers are a no-op, and
    # this is the one that can never destroy anything.
    assert is_ocr_sandwich_page(_page()) is False


# ------------------------------------------- what the verdict is used for


class _Config:
    """The settings paragraph_finder reads on this path."""

    ocr_workaround = True
    remove_non_formula_lines = False
    skip_formula_offset_calculation = True
    merge_alternating_line_numbers = False
    debug = False
    formular_font_pattern = None
    formular_char_pattern = None

    def image_text_regions_for_page(self, page_number):
        return None


def _finder():
    from babeldoc.format.pdf.document_il.midend.paragraph_finder import ParagraphFinder

    finder = ParagraphFinder.__new__(ParagraphFinder)
    finder.translation_config = _Config()
    return finder


def _run_wipe(page):
    """Exactly the block process_page guards, with the same predicate."""
    finder = _finder()
    ocr_page = finder.translation_config.ocr_workaround and is_ocr_sandwich_page(page)
    if ocr_page:
        page.pdf_character = []
    return ocr_page


def test_the_wipe_still_applies_to_a_real_sandwich_page():
    page = _page(loose=[_char(i, INVISIBLE) for i in range(40)])
    assert _run_wipe(page) is True
    assert page.pdf_character == []


def test_the_wipe_spares_a_digital_page_in_a_scanned_document():
    # The defect: 40 characters that no layout box claimed — record 151's
    # exercise numbers, table cells and stacked formulas — deleted because
    # OTHER pages of the same document were scans.
    page = _page(loose=[_char(i) for i in range(40)])
    assert _run_wipe(page) is False
    assert len(page.pdf_character) == 40
