"""Tests for re-joining the fragments of one justified line (fork side).

A justified line's inter-word gaps are stretched wide enough that the line
clusterer emits each word as its own line, each of which becomes its own
paragraph and its own translation unit. Isolated function words come back
from the model unchanged, so one cell of run9's FAQ table was delivered as
«What are the التكاليف of برمجيات الهندسة؟».

The trap is that a stretched word gap is WIDER than the gutter between two
table columns (measured on run9 p8: ~21 pt word gaps at a 15 pt font
against a 14 pt gutter), so no gap threshold separates them. What does is
that a column starts at the same x on many lines while a stretched word gap
starts nowhere in particular.

Run from the repo root:

    pytest server/test_line_fragments.py
"""

from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.midend.paragraph_finder import ParagraphFinder


def _box(x0, y0, x1, y1):
    return il_version_1.Box(x=x0, y=y0, x2=x1, y2=y1)


def _paragraph(*line_boxes):
    return il_version_1.PdfParagraph(
        pdf_paragraph_composition=[
            il_version_1.PdfParagraphComposition(
                pdf_line=il_version_1.PdfLine(
                    box=_box(*b),
                    pdf_character=[
                        il_version_1.PdfCharacter(box=_box(*b), char_unicode="x")
                    ],
                )
            )
            for b in line_boxes
        ]
    )


def _finder():
    return ParagraphFinder.__new__(ParagraphFinder)


# run9 p8: one justified line broken into five words, at a 15 pt font.
JUSTIFIED = [
    (56.6, 357.4, 91.1, 372.4),
    (111.8, 357.4, 172.9, 372.4),
    (194.1, 357.4, 227.8, 372.4),
    (248.9, 357.4, 261.3, 372.4),
    (281.3, 357.4, 336.5, 372.4),
]
# The Answer column of the same table: many rows start at x ~= 348.
ANSWER_COLUMN_X = 348.1


def _table_paragraphs():
    """The page as the finder sees it: a Question column, an Answer column
    that starts at the same x on many rows, and one justified question."""
    paragraphs = [_paragraph(b) for b in JUSTIFIED]
    for i in range(6):
        y = 200.0 + i * 20.0
        paragraphs.append(_paragraph((ANSWER_COLUMN_X, y, 723.0, y + 15.0)))
    return paragraphs


def test_a_column_is_detected_from_repeated_left_edges():
    starts = _finder()._page_column_starts(_table_paragraphs())
    assert any(abs(s - ANSWER_COLUMN_X) <= 2.0 for s in starts)
    # a stretched word gap starts nowhere in particular
    assert not any(abs(s - 111.8) <= 2.0 for s in starts)


def test_justified_word_fragments_merge():
    finder = _finder()
    starts = finder._page_column_starts(_table_paragraphs())
    a, b = _paragraph(JUSTIFIED[0]), _paragraph(JUSTIFIED[1])
    assert finder._should_merge_line_fragments(a, b, starts)


def test_a_fragment_that_begins_a_column_never_merges():
    # The Question cell ends at x=336.5 and the Answer cell starts at
    # x=348.1 on the SAME baseline. The 11.6 pt gutter is NARROWER than the
    # 20.7 pt word gap above, so only the column signal saves this.
    finder = _finder()
    starts = finder._page_column_starts(_table_paragraphs())
    question = _paragraph((56.6, 357.4, 336.5, 372.4))
    answer = _paragraph((ANSWER_COLUMN_X, 357.4, 723.0, 372.4))
    assert not finder._should_merge_line_fragments(question, answer, starts)


def test_an_absurdly_wide_gap_never_merges():
    finder = _finder()
    a = _paragraph((56.6, 357.4, 91.1, 372.4))
    b = _paragraph((400.0, 357.4, 460.0, 372.4))
    assert not finder._should_merge_line_fragments(a, b, [])


def test_a_tight_gap_still_merges_without_column_information():
    # The historic behaviour (a lost inter-word space) is unchanged.
    finder = _finder()
    a = _paragraph((56.6, 357.4, 91.1, 372.4))
    b = _paragraph((95.0, 357.4, 140.0, 372.4))
    assert finder._should_merge_line_fragments(a, b)


def test_fragments_on_different_baselines_never_merge():
    finder = _finder()
    a = _paragraph((56.6, 357.4, 91.1, 372.4))
    b = _paragraph((111.8, 300.0, 172.9, 315.0))
    assert not finder._should_merge_line_fragments(a, b, [])


def test_one_paragraphs_own_wrapped_lines_do_not_look_like_a_column():
    # Every line of a wrapped paragraph starts at the same x; that is not a
    # column, and counting lines rather than baselines would say it was.
    starts = _finder()._page_column_starts(
        [_paragraph((100.0, 300.0, 400.0, 315.0),
                    (100.0, 285.0, 400.0, 300.0),
                    (100.0, 270.0, 400.0, 285.0))]
    )
    assert not any(abs(s - 100.0) <= 2.0 for s in starts)
