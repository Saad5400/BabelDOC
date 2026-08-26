"""Tests for the translation sidecar (babeldoc/.../midend/translation_sidecar.py).

Run from the repo root:

    pytest server/test_sidecar.py
"""

import json

import pytest
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.midend import translation_sidecar


def _box(x0, y0, x1, y1):
    return il_version_1.Box(x=x0, y=y0, x2=x1, y2=y1)


def _character(x0, y0, x1, y1):
    return il_version_1.PdfCharacter(box=_box(x0, y0, x1, y1), char_unicode="x")


def _line_composition(y_top, x0, x1, *, size=10.0, step=6.0):
    """One source line as a same-style character run (what the IL holds by the
    time the sidecar snapshot runs — StylesAndFormulas has regrouped the
    original pdfLine compositions by style)."""
    characters = [_character(x, y_top - size, min(x + step, x1), y_top)
                  for x in _frange(x0, x1, step)]

    return il_version_1.PdfParagraphComposition(
        pdf_same_style_characters=il_version_1.PdfSameStyleCharacters(
            box=_box(x0, y_top - size, x1, y_top),
            pdf_character=characters,
        ))


def _frange(start, stop, step):
    while start < stop:
        yield start
        start += step


def _paragraph(text, box, font_size=11.0, label="plain text", lines=()):
    return il_version_1.PdfParagraph(
        box=box,
        pdf_style=il_version_1.PdfStyle(font_size=font_size),
        unicode=text,
        layout_label=label,
        pdf_paragraph_composition=[_line_composition(*line) for line in lines],
    )


def _document(paragraphs, *, figures=(), mediabox=(0.0, 0.0, 595.0, 842.0)):
    page = il_version_1.Page(
        mediabox=il_version_1.Mediabox(box=_box(*mediabox)),
        cropbox=il_version_1.Cropbox(box=_box(*mediabox)),
        pdf_paragraph=list(paragraphs),
        pdf_figure=[il_version_1.PdfFigure(box=box) for box in figures],
        page_number=0,
        unit="point",
        base_operations=il_version_1.BaseOperations(value=""),
    )

    return il_version_1.Document(page=[page], total_pages=1)


def test_it_pairs_the_source_text_with_the_translation():
    docs = _document([_paragraph("Software change is inevitable",
                                 _box(60, 700, 400, 714))])

    # The real pipeline: snapshot, translate in place, then build.
    sources = translation_sidecar.snapshot_source(docs)
    docs.page[0].pdf_paragraph[0].unicode = "تغيير البرمجيات أمر لا مفر منه"

    sidecar = translation_sidecar.build_sidecar(docs, lang_in="en", lang_out="ar",
                                                sources=sources)

    block = sidecar["pages"][0]["blocks"][0]
    assert block["source"] == "Software change is inevitable"
    assert block["target"] == "تغيير البرمجيات أمر لا مفر منه"
    assert block["box"] == [60.0, 700.0, 400.0, 714.0]
    assert block["font_size"] == 11.0
    assert sidecar["version"] == translation_sidecar.SIDECAR_VERSION
    assert sidecar["lang_out"] == "ar"


def test_the_translation_is_captured_without_the_pipelines_scaffolding():
    """The defect a real run exposed.

    `paragraph.unicode` after translation is the translator's RAW output: it
    still carries `{v1}` placeholders standing in for formulas and inline
    glyphs, and `<style id='N'>` tags marking runs whose formatting must
    survive. Typesetting consumes those; a sidecar that captured them verbatim
    handed the overlay renderer literal «{v1}مقدمة إلى NumPy» to draw.
    """
    paragraph = _paragraph("• Introduction to NumPy", _box(60, 700, 400, 714))
    # What ILTranslator leaves behind: the raw string on `unicode`, and the
    # PARSED runs on the compositions — the bullet resolved back to a real
    # character, the style tags already consumed.
    paragraph.unicode = "{v1}<style id='1'>مقدمة إلى</style> NumPy"
    paragraph.pdf_paragraph_composition = [
        il_version_1.PdfParagraphComposition(
            pdf_formula=il_version_1.PdfFormula(
                pdf_character=[_character(60, 704, 66, 714)])),
        il_version_1.PdfParagraphComposition(
            pdf_same_style_unicode_characters=(
                il_version_1.PdfSameStyleUnicodeCharacters(
                    unicode="مقدمة إلى NumPy"))),
    ]
    docs = _document([paragraph])

    target = translation_sidecar.build_sidecar(
        docs, lang_in="en", lang_out="ar")["pages"][0]["blocks"][0]["target"]

    assert "{v1}" not in target
    assert "<style" not in target
    assert "مقدمة إلى NumPy" in target


def test_styled_runs_are_not_welded_into_one_word():
    """A styled run carries no space of its own — on the page the runs are
    POSITIONED, not concatenated — so flattening them welds every boundary."""
    paragraph = _paragraph("Lecture 9: Introduction to NumPy (Numerical Computing)",
                           _box(60, 700, 400, 714))
    paragraph.unicode = ("<style id='1'>المحاضرة 9:</style>مقدمة إلى"
                         "<style id='2'>NumPy</style> (الحوسبة"
                         "<style id='3'>العددية</style>)")
    paragraph.pdf_paragraph_composition = [
        il_version_1.PdfParagraphComposition(
            pdf_same_style_unicode_characters=(
                il_version_1.PdfSameStyleUnicodeCharacters(unicode=text)))
        for text in ("المحاضرة 9:", "مقدمة إلى", "NumPy", " (الحوسبة",
                     "العددية", ")")
    ]
    docs = _document([paragraph])

    target = translation_sidecar.build_sidecar(
        docs, lang_in="en", lang_out="ar")["pages"][0]["blocks"][0]["target"]

    assert target == "المحاضرة 9: مقدمة إلى NumPy (الحوسبة العددية)"


def test_an_untranslated_paragraph_falls_back_to_its_own_text():
    # Nothing to parse (skip_translation, or a paragraph the translator left
    # alone): its unicode never had scaffolding in it to begin with.
    docs = _document([_paragraph("Introduction to NumPy", _box(60, 700, 400, 714))])

    target = translation_sidecar.build_sidecar(
        docs, lang_in="en", lang_out="ar")["pages"][0]["blocks"][0]["target"]

    assert target == "Introduction to NumPy"


def test_a_run_without_a_snapshot_still_carries_its_translation():
    docs = _document([_paragraph("ترجمة", _box(60, 700, 400, 714))])

    sidecar = translation_sidecar.build_sidecar(docs, lang_in="en", lang_out="ar")

    block = sidecar["pages"][0]["blocks"][0]
    assert block["target"] == "ترجمة"
    assert block["source"] is None


def test_the_page_frame_is_carried_so_a_renderer_can_map_the_boxes():
    docs = _document([_paragraph("ترجمة", _box(60, 700, 400, 714))],
                     mediabox=(0.0, 0.0, 720.0, 540.0))

    page = translation_sidecar.build_sidecar(docs, lang_in="en",
                                             lang_out="ar")["pages"][0]

    assert page["mediabox"] == [0.0, 0.0, 720.0, 540.0]
    assert page["page_number"] == 0


def test_figures_are_carried_as_obstacles():
    docs = _document([_paragraph("ترجمة", _box(60, 300, 400, 314))],
                     figures=[_box(60, 400, 400, 700)])

    page = translation_sidecar.build_sidecar(docs, lang_in="en",
                                             lang_out="ar")["pages"][0]

    assert page["obstacles"] == [[60.0, 400.0, 400.0, 700.0]]


def test_empty_and_specky_paragraphs_are_left_out():
    docs = _document([
        _paragraph("   ", _box(60, 700, 400, 714)),          # nothing to say
        _paragraph("ترجمة", _box(60, 600, 61, 601)),          # layout noise
        _paragraph("ترجمة", None),                            # no box at all
        _paragraph("ترجمة", _box(60, 500, 400, 514)),         # the real one
    ])

    blocks = translation_sidecar.build_sidecar(
        docs, lang_in="en", lang_out="ar")["pages"][0]["blocks"]

    assert [block["box"] for block in blocks] == [[60.0, 500.0, 400.0, 514.0]]


def test_an_inverted_box_is_normalised():
    docs = _document([_paragraph("ترجمة", _box(400, 714, 60, 700))])

    block = translation_sidecar.build_sidecar(
        docs, lang_in="en", lang_out="ar")["pages"][0]["blocks"][0]

    assert block["box"] == [60.0, 700.0, 400.0, 714.0]


def test_a_page_without_a_frame_stays_in_place_but_empty():
    docs = _document([_paragraph("ترجمة", _box(60, 700, 400, 714))])
    docs.page[0].mediabox = None

    page = translation_sidecar.build_sidecar(docs, lang_in="en",
                                             lang_out="ar")["pages"][0]

    # Page indexes must keep lining up with the PDF this is paired with.
    assert page == {"page_number": 0, "mediabox": None, "blocks": [],
                    "obstacles": []}


def test_write_sidecar_lands_readable_utf8(tmp_path):
    docs = _document([_paragraph("تغيير البرمجيات", _box(60, 700, 400, 714))])
    path = tmp_path / "sidecar.json"

    translation_sidecar.write_sidecar(docs, path, lang_in="en", lang_out="ar")

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["pages"][0]["blocks"][0]["target"] == "تغيير البرمجيات"


def test_a_failed_sidecar_never_fails_the_run(tmp_path):
    docs = _document([_paragraph("ترجمة", _box(60, 700, 400, 714))])

    # An unwritable path: the run's PDF is already finished and paid for, so a
    # lost sidecar is logged, never raised.
    translation_sidecar.write_sidecar(docs, tmp_path / "no" / "such" / "dir.json",
                                      lang_in="en", lang_out="ar")

    assert not (tmp_path / "no").exists()


@pytest.mark.parametrize("length", [1, 3])
def test_the_snapshot_is_keyed_by_position(length):
    docs = _document([_paragraph(f"source {i}", _box(60, 700 - i * 20,
                                                     400, 714 - i * 20))
                      for i in range(length)])

    snapshot = translation_sidecar.snapshot_source(docs)

    assert {index: entry["text"] for index, entry in snapshot[0].items()} == {
        i: f"source {i}" for i in range(length)}


def test_source_lines_are_reconstructed_for_the_spread_fallback():
    # One paragraph, three stacked source lines — a merged bullet list.
    docs = _document([_paragraph(
        "one two three", _box(60, 660, 400, 714),
        lines=[(714, 60, 400), (696, 60, 340), (678, 60, 380)])])

    sources = translation_sidecar.snapshot_source(docs)
    docs.page[0].pdf_paragraph[0].unicode = "واحد اثنان ثلاثة"

    block = translation_sidecar.build_sidecar(
        docs, lang_in="en", lang_out="ar", sources=sources)["pages"][0]["blocks"][0]

    assert len(block["lines"]) == 3
    # Top line first, and each line no wider than the paragraph that holds it.
    tops = [line[3] for line in block["lines"]]
    assert tops == sorted(tops, reverse=True)
    for x0, _y0, x1, _y1 in block["lines"]:
        assert block["box"][0] <= x0 < x1 <= block["box"][2]


def test_a_paragraph_with_no_characters_reports_no_lines():
    docs = _document([_paragraph("ترجمة", _box(60, 700, 400, 714))])

    block = translation_sidecar.build_sidecar(
        docs, lang_in="en", lang_out="ar")["pages"][0]["blocks"][0]

    assert block["lines"] == []
