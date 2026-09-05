"""Tests for the shrink-to-fit legibility floor (fork side).

Shrink-to-fit used to bottom out at scale 0.1 with no floor on the RESULTING
point size, so a 12.7 pt source line could be delivered at 1.27 pt. The
search now stops at whichever binds first — MIN_LEGIBLE_FONT_SIZE_PT
absolute, or MIN_LEGIBLE_SCALE of the paragraph's own dominant source size —
and a paragraph that still does not fit is laid out AT the floor and allowed
to overflow rather than being shrunk into unreadability or dropped.

Run from the repo root:

    pytest server/test_typesetting_floor.py
"""

from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.midend import typesetting as ts
from babeldoc.format.pdf.document_il.midend.typesetting import Typesetting


def _unit(font_size):
    unit = ts.TypesettingUnit.__new__(ts.TypesettingUnit)
    unit.font_size = font_size
    unit.char = None
    return unit


def _typesetting():
    return Typesetting.__new__(Typesetting)


def _floor(*font_sizes):
    return _typesetting()._legibility_min_scale([_unit(s) for s in font_sizes])


# ---------------------------------------------------------------- the floor


def test_large_text_may_shrink_only_to_the_relative_floor():
    # 40 pt: the absolute floor would allow 5.5/40 = 0.14, which is a
    # heading reduced to a seventh of its size. The relative floor binds.
    assert _floor(40.0) == ts.MIN_LEGIBLE_SCALE


def test_body_text_may_shrink_only_to_the_absolute_floor():
    # 11 pt body text: 45 % of it is 4.95 pt, below the legibility floor,
    # so the absolute floor binds instead.
    assert _floor(11.0) == ts.MIN_LEGIBLE_FONT_SIZE_PT / 11.0
    assert 11.0 * _floor(11.0) == ts.MIN_LEGIBLE_FONT_SIZE_PT


def test_a_12_7pt_line_can_no_longer_become_1_27pt():
    # The defect this exists for, stated as its own arithmetic.
    assert 12.7 * _floor(12.7) >= ts.MIN_LEGIBLE_FONT_SIZE_PT


def test_already_tiny_source_text_is_never_shrunk():
    # A 4.5 pt source line is smaller than the floor. Shrinking it further
    # is not allowed; scaling it UP is not either — the clamp is one-sided.
    assert _floor(4.5) == 1.0


def test_the_dominant_size_decides_not_an_outlier():
    # A superscript or a stray small glyph must not drag the whole
    # paragraph's floor down with it.
    assert _floor(12.0, 12.0, 12.0, 4.0) == _floor(12.0)


def test_a_paragraph_with_no_font_sizes_falls_back_to_the_relative_floor():
    assert _floor() == ts.MIN_LEGIBLE_SCALE


# ------------------------------------------------- settling at the floor


class _Recorder(Typesetting):
    """Captures what _settle_at_floor applies, without a real page."""

    def __init__(self):
        self.applied = None

    def _apply_typeset_units(self, paragraph, page, typeset_units, scale):
        self.applied = (typeset_units, scale)


def test_settle_at_floor_draws_the_overflowing_layout():
    # The old code returned min_scale with NOTHING laid out, which on the
    # apply pass left the composition empty and dropped the paragraph off
    # the page. Overflowing text a reader can see beats invisible text.
    recorder = _Recorder()
    paragraph = il_version_1.PdfParagraph(debug_id="test")
    units = [_unit(10.0)]
    scale, final = recorder._settle_at_floor(
        paragraph, None, (0.5, units), 0.5, None, apply_layout=True)
    assert scale == 0.5
    assert final is units
    assert recorder.applied == (units, 0.5)


def test_settle_at_floor_is_a_no_op_when_no_layout_was_ever_produced():
    recorder = _Recorder()
    paragraph = il_version_1.PdfParagraph(debug_id="test")
    scale, final = recorder._settle_at_floor(
        paragraph, None, None, 0.45, None, apply_layout=True)
    assert scale == 0.45
    assert final is None
    assert recorder.applied is None


# ------------------------------------ the paragraph is never drawn to nothing


class _NoLayout(Typesetting):
    """A typesetter whose layout pass always comes back with nothing.

    That is what `retypeset_with_precomputed_scale` really does when the
    paragraph has no box to lay out in, and what `_settle_at_floor` leaves
    behind when no candidate layout was ever produced. `render_paragraph`
    clears the composition BEFORE the rebuild, so on that path the paragraph
    used to leave typesetting empty — which is not a paragraph that moved,
    it is a paragraph deleted from the page.
    """

    def __init__(self):
        self.font_mapper = None

    def create_typesetting_units(self, paragraph, fonts):
        unit = ts.TypesettingUnit.__new__(ts.TypesettingUnit)
        unit.font_size = 10.0
        unit.char = None
        unit.formular = None
        unit.can_passthrough_cache = False  # a translated unit: never passed through
        return [unit]

    def _prefit_raster_label_box(self, paragraph, page, units):
        pass

    def retypeset_with_precomputed_scale(self, *args, **kwargs):
        pass  # no box, no layout: exactly the early return

    def _update_paragraph_render_order(self, paragraph):
        pass


def _char(x0, x1, text):
    return il_version_1.PdfCharacter(
        box=il_version_1.Box(x=x0, y=100.0, x2=x1, y2=110.0),
        char_unicode=text,
    )


def _page():
    return il_version_1.Page(
        page_number=7,
        unit="point",
        mediabox=il_version_1.Mediabox(
            box=il_version_1.Box(x=0, y=0, x2=612, y2=792)
        ),
        cropbox=il_version_1.Cropbox(box=il_version_1.Box(x=0, y=0, x2=612, y2=792)),
        base_operations=il_version_1.BaseOperations(value=""),
    )


def test_a_paragraph_that_cannot_be_laid_out_keeps_its_source_glyphs():
    chars = [_char(10.0, 20.0, "a"), _char(20.0, 30.0, "b")]
    paragraph = il_version_1.PdfParagraph(
        debug_id="keepme",
        box=None,
        pdf_paragraph_composition=[
            il_version_1.PdfParagraphComposition(
                pdf_line=il_version_1.PdfLine(
                    box=il_version_1.Box(x=10.0, y=100.0, x2=30.0, y2=110.0),
                    pdf_character=list(chars),
                )
            )
        ],
    )
    _NoLayout().render_paragraph(paragraph, _page(), {})

    # The backend can only draw pdf_character and pdf_formula compositions,
    # so the fallback has to hand it characters, not the line it started from.
    drawn = [c.pdf_character for c in paragraph.pdf_paragraph_composition]
    assert [c.char_unicode for c in drawn] == ["a", "b"]
    assert [c.box.x for c in drawn] == [10.0, 20.0]
    assert paragraph.scale == 1.0


def test_a_formula_survives_the_same_fallback():
    formula = il_version_1.PdfFormula(
        box=il_version_1.Box(x=10.0, y=100.0, x2=30.0, y2=110.0),
        pdf_character=[_char(10.0, 20.0, "x")],
    )
    paragraph = il_version_1.PdfParagraph(
        debug_id="keepme",
        box=None,
        pdf_paragraph_composition=[
            il_version_1.PdfParagraphComposition(pdf_formula=formula)
        ],
    )
    _NoLayout().render_paragraph(paragraph, _page(), {})
    assert [c.pdf_formula for c in paragraph.pdf_paragraph_composition] == [formula]


def test_a_translated_paragraph_with_no_glyphs_left_is_reported_not_hidden(caplog):
    # Its source characters were replaced by translated unicode, so there is
    # genuinely nothing to draw. That must be LOUD rather than silent.
    style = il_version_1.PdfStyle(font_id="base", font_size=10.0)
    paragraph = il_version_1.PdfParagraph(
        debug_id="gone",
        box=None,
        pdf_paragraph_composition=[
            il_version_1.PdfParagraphComposition(
                pdf_same_style_unicode_characters=(
                    il_version_1.PdfSameStyleUnicodeCharacters(
                        unicode="مرحبا", pdf_style=style
                    )
                )
            )
        ],
    )
    with caplog.at_level("WARNING"):
        _NoLayout().render_paragraph(paragraph, _page(), {})
    assert paragraph.pdf_paragraph_composition == []
    assert "no glyphs to fall back on" in caplog.text
