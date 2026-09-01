"""Tests for the rotated-text guard (fork side).

Characters are clustered into lines by page-x. For a run set at 90 degrees
that coordinate is its reading COLUMN, so every glyph lands in its own
cluster, in reverse: run67's vertical chapter sidebar arrives as
'E D OC D N A SM E TSYS R E B M U N' — "NUMBER SYSTEMS AND CODE" read
bottom to top — on all 48 of its pages. The fragments the model recognises
come back translated and the rest are redrawn as a column of single letters
on top of the un-erased original.

Until rotated runs are carried in their own frame through extraction,
clustering and typesetting, the honest outcome is to leave them alone.

Run from the repo root:

    pytest server/test_rotated_text.py
"""

from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.midend.paragraph_finder import ParagraphFinder


def _box(x0, y0, x1, y1):
    return il_version_1.Box(x=x0, y=y0, x2=x1, y2=y1)


def _line(text, x0, y0, x1, y1):
    return il_version_1.PdfLine(
        box=_box(x0, y0, x1, y1),
        pdf_character=[
            il_version_1.PdfCharacter(box=_box(x0, y0, x1, y1), char_unicode=c)
            for c in text
        ],
    )


def _paragraph(lines, box=None):
    return il_version_1.PdfParagraph(
        box=box,
        pdf_paragraph_composition=[
            il_version_1.PdfParagraphComposition(pdf_line=line) for line in lines
        ],
    )


def _finder():
    return ParagraphFinder.__new__(ParagraphFinder)


def _page(paragraphs):
    return il_version_1.Page(
        mediabox=il_version_1.Mediabox(box=_box(0, 0, 720, 540)),
        cropbox=il_version_1.Cropbox(box=_box(0, 0, 720, 540)),
        pdf_paragraph=list(paragraphs),
        page_number=0,
        unit="point",
        base_operations=il_version_1.BaseOperations(value=""),
    )


# run67's sidebar: 20 lines of one glyph each in a 26 x 407 pt box.
SIDEBAR = _paragraph(
    [_line(c, 12.0, 97.0 + i * 20, 20.0, 107.0 + i * 20)
     for i, c in enumerate("EDOCDNASMETSYSREBMUN")],
    box=_box(12.0, 97.0, 38.1, 504.0),
)

# run67 p6: a rotated label whose glyphs did cluster into words, but whose
# lines are far too narrow to hold their characters horizontally.
ROTATED_LABEL = _paragraph([
    _line("rse", 5.0, 200.0, 10.3, 214.9),
    _line("f this ", 5.0, 150.0, 18.2, 193.6),
    _line("cou fo", 5.0, 110.0, 18.2, 145.2),
])


def test_a_glyph_column_is_recognised_as_rotated():
    assert _finder()._is_rotated_text_paragraph(SIDEBAR)


def test_a_rotated_label_is_recognised_by_its_impossible_aspect():
    assert _finder()._is_rotated_text_paragraph(ROTATED_LABEL)


def test_ordinary_horizontal_text_is_not_rotated():
    # run67 p6's own diagram labels, which must keep translating.
    for text, x1 in (("device drivers", 66.6), ("programs", 45.1),
                     ("transistors", 50.0), ("filters", 25.8)):
        paragraph = _paragraph([_line(text, 0.0, 100.0, x1, 110.9)])
        assert not _finder()._is_rotated_text_paragraph(paragraph), text


def test_a_wrapped_body_paragraph_is_not_rotated():
    paragraph = _paragraph([
        _line("Microprocessors have revolutionized", 0.0, 200.0, 471.0, 212.0),
        _line("our world over the past decades", 0.0, 186.0, 460.0, 198.0),
    ])
    assert not _finder()._is_rotated_text_paragraph(paragraph)


def test_a_short_stack_is_not_a_glyph_column():
    # Three initials in a column is not enough evidence to drop content.
    paragraph = _paragraph(
        [_line(c, 10.0, 100.0 + i * 20, 18.0, 110.0 + i * 20)
         for i, c in enumerate("ABC")],
        box=_box(10.0, 100.0, 18.0, 150.0),
    )
    assert not _finder()._is_rotated_text_paragraph(paragraph)


def test_restore_drops_only_the_rotated_paragraphs():
    good = _paragraph([_line("device drivers", 0.0, 100.0, 66.6, 110.9)])
    page = _page([SIDEBAR, good, ROTATED_LABEL])
    _finder().restore_rotated_text_paragraphs(page)
    assert page.pdf_paragraph == [good]
