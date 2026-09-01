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


def _column_line(text, x0, y0, x1, y1):
    """A line the extractor read DOWN a column: one glyph per baseline.

    This is what a rotated run really looks like in the IL — the shape the
    guard has to recognise — as opposed to _line, where every character
    shares the line's own box and therefore its baseline.
    """
    step = (y1 - y0) / max(len(text), 1)
    return il_version_1.PdfLine(
        box=_box(x0, y0, x1, y1),
        pdf_character=[
            il_version_1.PdfCharacter(
                box=_box(x0, y0 + i * step, x1, y0 + (i + 1) * step),
                char_unicode=c,
            )
            for i, c in enumerate(text)
        ],
    )


def _stacked_line(rows, x0, y0, x1, y1):
    """One line occupying several baselines — a fraction, a superscript.

    Each row is (text, row_x0, row_x1); the rows are stacked top to bottom
    inside the line's box, every character of a row on that row's baseline.
    """
    step = (y1 - y0) / max(len(rows), 1)
    characters = []
    for index, (text, row_x0, row_x1) in enumerate(rows):
        row_y = y1 - (index + 1) * step
        width = (row_x1 - row_x0) / max(len(text), 1)
        characters.extend(
            il_version_1.PdfCharacter(
                box=_box(
                    row_x0 + i * width,
                    row_y,
                    row_x0 + (i + 1) * width,
                    row_y + step,
                ),
                char_unicode=c,
            )
            for i, c in enumerate(text)
        )
    return il_version_1.PdfLine(
        box=_box(x0, y0, x1, y1), pdf_character=characters
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
    _column_line("rse", 5.0, 200.0, 10.3, 214.9),
    _column_line("f this ", 5.0, 150.0, 18.2, 193.6),
    _column_line("cou fo", 5.0, 110.0, 18.2, 145.2),
])

# run39 p15's third worked example: one line, three baselines, a whole row
# of characters on each. Tall for its width for a completely different
# reason than a rotated run, and the guard used to delete it.
STACKED_FORMULA = _paragraph([
    _stacked_line(
        [("1x106um", 356.2, 424.3),
         ("1 m", 376.8, 403.8),
         ("0.91 m x = 9.1 x 105 u m", 286.4, 542.3)],
        286.4, 82.8, 542.3, 123.9,
    )
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


def test_a_stacked_formula_is_not_rotated():
    # run39 p15: both worked examples are one line across three baselines.
    # Judged on aspect alone they read as rotated, and the guard deleted
    # them — the page was delivered with two empty parentheses and a
    # fraction rule where the arithmetic used to be.
    assert not _finder()._is_rotated_text_paragraph(STACKED_FORMULA)


def test_a_stacked_construct_is_told_from_a_column_by_its_baselines():
    # The measurement that separates them (run39 p15 / run67 p6):
    # a formula puts a row on each baseline, a column puts one glyph.
    formula_line = STACKED_FORMULA.pdf_paragraph_composition[0].pdf_line
    rotated_line = ROTATED_LABEL.pdf_paragraph_composition[2].pdf_line
    finder = _finder()
    formula_bands = finder._baseline_bands(formula_line)
    rotated_bands = finder._baseline_bands(rotated_line)
    assert len(formula_bands) == 3
    assert finder._line_char_count(formula_line) / len(formula_bands) > 2.0
    assert finder._line_char_count(rotated_line) / len(rotated_bands) <= 2.0


def test_restore_drops_only_the_rotated_paragraphs():
    good = _paragraph([_line("device drivers", 0.0, 100.0, 66.6, 110.9)])
    page = _page([SIDEBAR, good, ROTATED_LABEL, STACKED_FORMULA])
    _finder().restore_rotated_text_paragraphs(page)
    assert page.pdf_paragraph == [good, STACKED_FORMULA]
