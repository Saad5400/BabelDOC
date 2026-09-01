"""Tests for stacked constructs under the RTL mirror (fork side).

A fraction is not a sequence of runs. Its numerator, its rule and its
denominator are positioned against each other, so anything that lays them
out independently takes it apart: the RTL mirror reverses each part
against the others and the numerator lands beside the denominator instead
of above it. run39 p15, MEASURED against the source page:

    source      numerator x[356.2,424.3] y475.7
                denominator x[376.8,403.7] y494.9   19 pt apart, concentric
    delivered   numerator x[455.1,506.5] y465.7
                denominator x[416.7,438.7] y470.2   4.5 pt apart, DISJOINT

Two things keep a stack together: it is recognised as one rigid formula
(styles_and_formulas), and the vector paths that bracket it — the
parentheses, the fraction rule — are mirrored with it rather than with
whatever larger paragraph happens to contain them (typesetting).

Run from the repo root:

    pytest server/test_stacked_formula.py
"""

from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.midend.styles_and_formulas import StylesAndFormulas
from babeldoc.format.pdf.document_il.midend.typesetting import Typesetting


def _box(x0, y0, x1, y1):
    return il_version_1.Box(x=x0, y=y0, x2=x1, y2=y1)


def _row(text, x0, x1, y0, y1):
    """One row of characters, all on the same baseline."""
    width = (x1 - x0) / max(len(text), 1)
    return [
        il_version_1.PdfCharacter(
            box=_box(x0 + i * width, y0, x0 + (i + 1) * width, y1),
            visual_bbox=il_version_1.VisualBbox(
                box=_box(x0 + i * width, y0, x0 + (i + 1) * width, y1)
            ),
            char_unicode=c,
        )
        for i, c in enumerate(text)
    ]


def _line(*rows):
    characters = [char for row in rows for char in row]
    return il_version_1.PdfLine(
        box=_box(
            min(c.box.x for c in characters),
            min(c.box.y for c in characters),
            max(c.box.x2 for c in characters),
            max(c.box.y2 for c in characters),
        ),
        pdf_character=characters,
    )


class _Config:
    """The two settings process_page_formulas reads, at their defaults."""

    formular_font_pattern = None
    formular_char_pattern = None


def _styles():
    styles = StylesAndFormulas.__new__(StylesAndFormulas)
    styles.translation_config = _Config()
    styles._code_formula_ids = set()
    styles._stacked_formula_ids = set()
    return styles


# run39 p15's third worked example, at its source coordinates.
FRACTION = _line(
    _row("1x106um", 356.2, 424.3, 475.7, 493.3),
    _row("1 m", 376.8, 403.7, 494.9, 512.5),
    _row("0.91 m x", 285.8, 347.5, 486.1, 503.7),
    _row("= 9.1 x 105 u m", 436.6, 543.5, 486.1, 503.7),
)


# ------------------------------------------------------- what is a stack


def test_a_fraction_is_a_vertical_stack():
    assert _styles()._is_vertical_stack(FRACTION)


def test_a_superscript_is_not_a_vertical_stack():
    # Two baselines, but the raised row sits BESIDE its base, not over it.
    line = _line(
        _row("1 x 10", 494.3, 538.6, 313.9, 331.5),
        _row("-6", 538.6, 550.3, 313.1, 324.8),
    )
    assert not _styles()._is_vertical_stack(line)


def test_ordinary_text_is_not_a_vertical_stack():
    line = _line(_row("Microprocessors have", 0.0, 471.0, 200.0, 212.0))
    assert not _styles()._is_vertical_stack(line)


def test_a_lone_character_over_a_row_is_not_a_stack():
    # A single raised glyph is a mark on the row, not a row of its own.
    line = _line(
        _row("area", 100.0, 140.0, 300.0, 312.0),
        _row("2", 118.0, 122.0, 290.0, 298.0),
    )
    assert not _styles()._is_vertical_stack(line)


# ------------------------------------------ a stack survives as one block


def test_a_stacked_line_becomes_one_rigid_formula():
    paragraph = il_version_1.PdfParagraph(
        box=_box(285.8, 475.7, 543.5, 512.5),
        pdf_paragraph_composition=[
            il_version_1.PdfParagraphComposition(pdf_line=FRACTION)
        ],
    )
    page = il_version_1.Page(
        mediabox=il_version_1.Mediabox(box=_box(0, 0, 841.89, 595.28)),
        cropbox=il_version_1.Cropbox(box=_box(0, 0, 841.89, 595.28)),
        pdf_paragraph=[paragraph],
        page_number=0,
        unit="point",
        base_operations=il_version_1.BaseOperations(value=""),
    )
    styles = _styles()
    styles.process_page_formulas(page)

    compositions = paragraph.pdf_paragraph_composition
    assert len(compositions) == 1
    formula = compositions[0].pdf_formula
    assert formula is not None
    assert len(formula.pdf_character) == len(FRACTION.pdf_character)
    # And it is protected from the two paths that would take it apart
    # again: splitting at a comma, or being handed back as plain text.
    assert not styles.should_split_formula(formula)
    assert not styles.is_translatable_formula(formula)


# ------------------------------- the brackets travel with what they bracket


def _anchor(x0, y0, x1, y1, dx):
    return ((x0, y0, x1, y1), dx)


def test_a_bracket_outside_its_paragraph_follows_its_own_region():
    # run39 p15's second worked example. The parentheses reach 8.8 pt below
    # the text they enclose, so the paragraph box does not contain them and
    # the blue paragraph above — which does — used to capture them: the
    # text moved -414.58 pt and the brackets -229.11 pt, leaving an empty
    # pair of parentheses 180 pt from its fraction.
    page = il_version_1.Page(
        mediabox=il_version_1.Mediabox(box=_box(0, 0, 841.89, 595.28)),
        cropbox=il_version_1.Cropbox(box=_box(0, 0, 841.89, 595.28)),
        page_layout=[
            il_version_1.PageLayout(
                id=1, conf=0.35, class_name="isolate_formula",
                box=_box(491.0, 253.0, 765.0, 295.0),
            )
        ],
        page_number=0,
        unit="point",
        base_operations=il_version_1.BaseOperations(value=""),
    )
    formula_text = _anchor(494.3, 263.8, 762.2, 291.3, -414.58)
    blue_paragraph = _anchor(57.7, 142.3, 555.1, 416.2, -229.11)
    anchors = [formula_text, blue_paragraph]

    anchors.extend(
        Typesetting._layout_region_anchors(
            Typesetting.__new__(Typesetting), page, anchors
        )
    )
    bracket = _box(582.3, 255.9, 662.8, 290.6)
    assert Typesetting._find_anchor_dx(anchors, bracket) == -414.58


def test_a_region_spanning_two_groups_does_not_speak_for_either():
    page = il_version_1.Page(
        mediabox=il_version_1.Mediabox(box=_box(0, 0, 841.89, 595.28)),
        cropbox=il_version_1.Cropbox(box=_box(0, 0, 841.89, 595.28)),
        page_layout=[
            il_version_1.PageLayout(
                id=1, conf=0.4, class_name="plain text",
                box=_box(100.0, 100.0, 700.0, 200.0),
            )
        ],
        page_number=0,
        unit="point",
        base_operations=il_version_1.BaseOperations(value=""),
    )
    anchors = [
        _anchor(110.0, 110.0, 300.0, 190.0, -200.0),
        _anchor(400.0, 110.0, 690.0, 190.0, +200.0),
    ]
    assert (
        Typesetting._layout_region_anchors(
            Typesetting.__new__(Typesetting), page, anchors
        )
        == []
    )


def test_fallback_line_regions_are_not_anchors():
    # Every text line on the page has one; they are not groups.
    page = il_version_1.Page(
        mediabox=il_version_1.Mediabox(box=_box(0, 0, 841.89, 595.28)),
        cropbox=il_version_1.Cropbox(box=_box(0, 0, 841.89, 595.28)),
        page_layout=[
            il_version_1.PageLayout(
                id=1, conf=1.0, class_name="fallback_line",
                box=_box(100.0, 100.0, 300.0, 120.0),
            )
        ],
        page_number=0,
        unit="point",
        base_operations=il_version_1.BaseOperations(value=""),
    )
    anchors = [_anchor(110.0, 105.0, 290.0, 118.0, -50.0)]
    assert (
        Typesetting._layout_region_anchors(
            Typesetting.__new__(Typesetting), page, anchors
        )
        == []
    )


def test_a_stack_sharing_its_paragraph_with_prose_is_left_alone():
    # run39 p10: the same construct, but the paragraph also carries the
    # sentence that introduces it. An indivisible 250 pt unit cannot be
    # placed after a sentence, and the line filler drew the formula
    # straight over the Arabic — 2 span overlaps became 12. Only a
    # paragraph that IS the stack becomes one block.
    sentence = _line(
        _row("Converting 5.3 cm2 to m2 will be:", 49.0, 320.0, 263.0, 332.0)
    )
    alone = il_version_1.PdfParagraph(
        box=_box(285.8, 475.7, 543.5, 512.5),
        pdf_paragraph_composition=[
            il_version_1.PdfParagraphComposition(pdf_line=FRACTION)
        ],
    )
    with_prose = il_version_1.PdfParagraph(
        box=_box(49.0, 253.0, 543.5, 332.0),
        pdf_paragraph_composition=[
            il_version_1.PdfParagraphComposition(pdf_line=sentence),
            il_version_1.PdfParagraphComposition(pdf_line=FRACTION),
        ],
    )
    assert _styles()._paragraph_is_one_stack(alone)
    assert not _styles()._paragraph_is_one_stack(with_prose)
