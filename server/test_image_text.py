"""Tests for the image-text lane (Contract 2, fork side).

Digital pages with embedded raster images get invisible OCR runs injected
over the in-image text (server/image_prep.py, Contract 1). The fork must
recognise those runs, keep them out of the digital-text paragraphs, mask
them with background-matched fills, ride the RTL mirror rigidly with their
image, and tag them in the sidecar.

Run from the repo root:

    pytest server/test_image_text.py
"""

import json

import pytest
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.midend import translation_sidecar
from babeldoc.format.pdf.document_il.midend.paragraph_finder import ParagraphFinder
from babeldoc.format.pdf.document_il.midend.typesetting import Typesetting
from babeldoc.format.pdf.translation_config import TranslationConfig


def _box(x0, y0, x1, y1):
    return il_version_1.Box(x=x0, y=y0, x2=x1, y2=y1)


def _char(x0, y0, x1, y1, *, text="x", render_mode=None, xobj_id=-1):
    return il_version_1.PdfCharacter(
        box=_box(x0, y0, x1, y1),
        visual_bbox=il_version_1.VisualBbox(box=_box(x0, y0, x1, y1)),
        char_unicode=text,
        pdf_style=il_version_1.PdfStyle(
            font_id="helv",
            font_size=y1 - y0,
            graphic_state=il_version_1.GraphicState(),
        ),
        render_mode=render_mode,
        render_order=7,
        sub_render_order=0,
        xobj_id=xobj_id,
        advance=x1 - x0,
        vertical=False,
    )


def _word_chars(text, x, y, *, size=10.0, render_mode=3, step=6.0):
    return [
        _char(x + i * step, y, x + (i + 1) * step, y + size,
              text=c, render_mode=render_mode)
        for i, c in enumerate(text)
    ]


def _page(chars, *, page_number=0, mediabox=(0.0, 0.0, 720.0, 540.0)):
    return il_version_1.Page(
        mediabox=il_version_1.Mediabox(box=_box(*mediabox)),
        cropbox=il_version_1.Cropbox(box=_box(*mediabox)),
        pdf_character=list(chars),
        page_number=page_number,
        unit="point",
        base_operations=il_version_1.BaseOperations(value=""),
    )


class _StubConfig:
    """Just the members the image-text passes read."""

    def __init__(self, regions=None, ocr_workaround=False):
        self.ocr_workaround = ocr_workaround
        self.image_text_regions = regions
        self.debug = False

    image_text_regions_for_page = TranslationConfig.image_text_regions_for_page


def _finder(regions=None, ocr_workaround=False):
    finder = ParagraphFinder.__new__(ParagraphFinder)
    finder.translation_config = _StubConfig(regions, ocr_workaround)
    finder._mask_color_sampler = None  # no original PDF in unit tests
    return finder


REGION = (400.0, 100.0, 660.0, 350.0)


# ---------------------------------------------------------------- config


def test_config_parses_a_regions_dict():
    parsed = TranslationConfig._parse_image_text_regions(
        {"version": 1, "pages": {"3": [{"image_bbox": [10, 20, 110, 220]}]}}
    )
    assert parsed == {3: [(10.0, 20.0, 110.0, 220.0)]}


def test_config_parses_a_regions_file(tmp_path):
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(
        {"version": 1, "pages": {"0": [{"image_bbox": [1, 2, 3, 4]}]}}))
    parsed = TranslationConfig._parse_image_text_regions(path)
    assert parsed == {0: [(1.0, 2.0, 3.0, 4.0)]}


def test_config_rejects_an_unknown_version():
    with pytest.raises(ValueError, match="version"):
        TranslationConfig._parse_image_text_regions(
            {"version": 2, "pages": {}})


def test_config_off_means_no_regions_for_any_page():
    config = _StubConfig(None)
    assert config.image_text_regions_for_page(0) == []


# ----------------------------------------------------------- recognition


def test_recognition_needs_both_invisible_mode_and_a_region():
    regions = [REGION]
    inside_invisible = _char(450, 200, 500, 212, render_mode=3)
    inside_visible = _char(450, 200, 500, 212, render_mode=None)
    outside_invisible = _char(100, 200, 150, 212, render_mode=3)

    index = ParagraphFinder._image_text_region_index
    assert index(inside_invisible, regions) == 0
    assert index(inside_visible, regions) is None
    assert index(outside_invisible, regions) is None


def test_recognition_tolerates_two_points_of_overhang():
    regions = [REGION]
    barely_out = _char(REGION[0] - 1.5, 200, 500, 212, render_mode=3)
    too_far_out = _char(REGION[0] - 3.0, 200, 500, 212, render_mode=3)

    index = ParagraphFinder._image_text_region_index
    assert index(barely_out, regions) == 0
    assert index(too_far_out, regions) is None


# ------------------------------------------------------------ extraction


def test_labels_become_their_own_paragraphs_and_leave_the_page():
    # Two side-by-side labels and one wrapped label, plus normal page text.
    label_a = _word_chars("Class", 420, 300)
    label_b = _word_chars("Object", 560, 300)
    wrapped = (_word_chars("REPEAT", 450, 200)
               + _word_chars("CODE", 452, 188))
    normal = _word_chars("Body", 60, 300, render_mode=None)
    page = _page(label_a + label_b + wrapped + normal)

    finder = _finder({0: [REGION]})
    paragraphs = finder._extract_image_text_paragraphs(page)

    assert page.pdf_character == normal  # digital text never touched
    # side-by-side labels split apart; the wrapped label merged back into one
    assert sorted(p.unicode for p in paragraphs) == [
        "Class", "Object", "REPEATCODE"]
    for paragraph in paragraphs:
        assert paragraph.raster_region == list(REGION)
        for comp in paragraph.pdf_paragraph_composition:
            for char in comp.pdf_line.pdf_character:
                assert char.render_order is None
                assert char.sub_render_order is None


def test_extraction_is_inert_without_regions_or_on_the_scanned_lane():
    chars = _word_chars("Class", 420, 300)
    for finder in (_finder(None), _finder({0: [REGION]}, ocr_workaround=True)):
        page = _page(list(chars))
        assert finder._extract_image_text_paragraphs(page) == []
        assert page.pdf_character == chars


def test_extraction_ignores_other_pages_regions():
    page = _page(_word_chars("Class", 420, 300), page_number=1)
    finder = _finder({0: [REGION]})
    assert finder._extract_image_text_paragraphs(page) == []


# ----------------------------------------------------------------- masks


def test_each_label_gets_a_mask_tagged_with_its_region():
    page = _page(_word_chars("Class", 420, 300))
    finder = _finder({0: [REGION]})
    paragraphs = finder._extract_image_text_paragraphs(page)

    finder.add_image_text_masks(page, paragraphs)

    assert len(page.pdf_rectangle) == 1
    mask = page.pdf_rectangle[0]
    assert mask.fill_background is True
    assert mask.raster_region == list(REGION)
    box = paragraphs[0].box
    # padded around the glyphs, clipped to the region
    assert mask.box.x == pytest.approx(box.x - finder.IMAGE_TEXT_MASK_PAD)
    assert mask.box.x2 == pytest.approx(box.x2 + finder.IMAGE_TEXT_MASK_PAD)
    assert mask.box.x >= REGION[0] and mask.box.x2 <= REGION[2]


# ---------------------------------------------------------------- mirror


def _typesetting():
    return Typesetting.__new__(Typesetting)


def test_labels_and_masks_ride_their_images_mirror_translation():
    page = _page([])
    page_width = 720.0
    pivot = page_width  # cropbox x + x2

    label = il_version_1.PdfParagraph(
        box=_box(420, 290, 460, 302), unicode="Class",
        raster_region=list(REGION),
    )
    mask = il_version_1.PdfRectangle(
        box=_box(418, 288, 462, 304), fill_background=True,
        graphic_state=il_version_1.GraphicState(),
        raster_region=list(REGION),
    )
    normal = il_version_1.PdfParagraph(
        box=_box(60, 290, 200, 302), unicode="Body",
    )
    page.pdf_paragraph = [label, normal]
    page.pdf_rectangle = [mask]

    _typesetting()._mirror_page_layout(page)

    image_dx = pivot - REGION[0] - REGION[2]  # dx = pivot - x - x2
    # The label and its mask moved by the IMAGE's dx, not their own.
    assert label.box.x == pytest.approx(420 + image_dx)
    assert mask.box.x == pytest.approx(418 + image_dx)
    # Their stored regions moved to the image's mirrored position.
    assert label.raster_region[0] == pytest.approx(REGION[0] + image_dx)
    assert mask.raster_region[2] == pytest.approx(REGION[2] + image_dx)
    # The normal paragraph mirrors independently as before.
    assert normal.box.x == pytest.approx(pivot - 200)


def test_a_label_is_not_at_its_own_mirror_position():
    # The point of the anchor: a label left-of-center inside a right-of-
    # center image must NOT land at its own mirrored spot.
    page = _page([])
    label = il_version_1.PdfParagraph(
        box=_box(420, 290, 460, 302), unicode="Class",
        raster_region=list(REGION),
    )
    page.pdf_paragraph = [label]

    _typesetting()._mirror_page_layout(page)

    own_dx = 720.0 - 420 - 460
    image_dx = 720.0 - REGION[0] - REGION[2]
    assert own_dx != pytest.approx(image_dx)
    assert label.box.x == pytest.approx(420 + image_dx)


# ---------------------------------------------------------------- prefit


class _FakeUnit:
    can_passthrough = False

    def __init__(self, width):
        self.width = width


def test_a_short_label_box_widens_around_its_center_for_the_translation():
    page = _page([])
    label = il_version_1.PdfParagraph(
        box=_box(500, 200, 540, 212), unicode="Object",
        raster_region=list(REGION),
    )
    page.pdf_paragraph = [label]

    _typesetting()._prefit_raster_label_box(
        label, page, [_FakeUnit(40.0), _FakeUnit(40.0)])

    needed = 80.0 * 1.05 + 2.0
    assert label.box.x2 - label.box.x == pytest.approx(needed)
    # symmetric growth: the center stays put
    assert (label.box.x + label.box.x2) / 2 == pytest.approx(520.0)


def test_prefit_never_leaves_the_region_or_covers_a_sibling_label():
    page = _page([])
    label = il_version_1.PdfParagraph(
        box=_box(410, 200, 450, 212), unicode="Object",
        raster_region=list(REGION),
    )
    sibling = il_version_1.PdfParagraph(
        box=_box(470, 198, 520, 214), unicode="Class",
        raster_region=list(REGION),
    )
    page.pdf_paragraph = [label, sibling]

    _typesetting()._prefit_raster_label_box(label, page, [_FakeUnit(300.0)])

    assert label.box.x >= REGION[0] + 1
    assert label.box.x2 <= sibling.box.x - 1


def test_prefit_leaves_wide_enough_boxes_and_plain_paragraphs_alone():
    page = _page([])
    wide = il_version_1.PdfParagraph(
        box=_box(410, 200, 560, 212), unicode="Object",
        raster_region=list(REGION),
    )
    plain = il_version_1.PdfParagraph(
        box=_box(60, 200, 100, 212), unicode="Body",
    )
    page.pdf_paragraph = [wide, plain]

    typesetting = _typesetting()
    typesetting._prefit_raster_label_box(wide, page, [_FakeUnit(40.0)])
    typesetting._prefit_raster_label_box(plain, page, [_FakeUnit(400.0)])

    assert (wide.box.x, wide.box.x2) == (410, 560)
    assert (plain.box.x, plain.box.x2) == (60, 100)


# --------------------------------------------------------------- sidecar


def test_sidecar_blocks_carry_on_raster_and_region():
    on_raster = il_version_1.PdfParagraph(
        box=_box(420, 290, 460, 302), unicode="فئة",
        pdf_style=il_version_1.PdfStyle(font_size=10.0),
        raster_region=list(REGION),
    )
    plain = il_version_1.PdfParagraph(
        box=_box(60, 290, 200, 302), unicode="نص",
        pdf_style=il_version_1.PdfStyle(font_size=10.0),
    )
    page = _page([])
    page.pdf_paragraph = [on_raster, plain]
    docs = il_version_1.Document(page=[page], total_pages=1)

    sidecar = translation_sidecar.build_sidecar(docs, lang_in="en",
                                                lang_out="ar")

    blocks = sidecar["pages"][0]["blocks"]
    assert sidecar["version"] == 1  # additive fields only
    assert blocks[0]["on_raster"] is True
    assert blocks[0]["region"] == list(REGION)
    assert "on_raster" not in blocks[1]
    assert "region" not in blocks[1]


# ------------------------------------------------------------- rendering


def test_image_text_masks_render_on_digital_pages():
    """The scanned lane gates mask rectangles on ocr_workaround; an
    image-text mask must render on a normal digital run too, below the
    orderless translated characters."""
    from babeldoc.format.pdf.document_il.backend.pdf_creater import PDFCreater

    class _RenderConfig:
        ocr_workaround = False
        debug = False
        skip_form_render = True
        skip_curve_render = True

    page = _page([])
    page.pdf_rectangle = [
        il_version_1.PdfRectangle(
            box=_box(418, 288, 462, 304), fill_background=True,
            graphic_state=il_version_1.GraphicState(
                passthrough_per_char_instruction="1 1 1 rg"),
            raster_region=list(REGION),
        ),
        il_version_1.PdfRectangle(  # a plain rect still does not render
            box=_box(10, 10, 20, 20), fill_background=True,
            graphic_state=il_version_1.GraphicState(
                passthrough_per_char_instruction="1 1 1 rg"),
        ),
    ]
    label_char = _char(420, 290, 460, 302, render_mode=3)
    label_char.render_order = None
    label_char.sub_render_order = None
    page.pdf_character = [label_char]

    creater = PDFCreater.__new__(PDFCreater)
    units = creater.create_render_units_for_page(page, _RenderConfig())

    kinds = [type(unit).__name__ for unit in units]
    assert kinds.count("RectangleRenderUnit") == 1
    units.sort(key=lambda unit: unit.get_sort_key())
    ordered = [type(unit).__name__ for unit in units]
    # mask below the translated (orderless) character
    assert ordered.index("RectangleRenderUnit") < ordered.index(
        "CharacterRenderUnit")


# --------------------------------------------------- one line, one label


def _page_with_regions(chars, layouts, *, page_number=0):
    page = _page(chars, page_number=page_number)
    page.page_layout = [
        il_version_1.PageLayout(
            id=index + 1, conf=conf, class_name=class_name, box=_box(*box)
        )
        for index, (class_name, conf, box) in enumerate(layouts)
    ]
    return page


# run67 p5's copyright footer, at its real coordinates: one sentence in a
# letter-spaced italic, so its word gaps run to 18.1 and 21.9 pt against a
# 8.2 pt line. The gap rule cut it into three labels, each translated on
# its own and set right aligned in its own box, and the sentence was
# delivered as three Arabic fragments with 90 pt of white between them —
# on 44 of that document's 48 pages.
FOOTER_REGION = (0.0, 60.0, 841.0, 534.0)
FOOTER_CAPTION = ("figure_caption", 0.57, (20.0, 67.0, 438.0, 85.0))


def _footer_chars():
    return (
        _word_chars("Digital Design and", 41.7, 70.9, size=8.2, step=6.4)
        + _word_chars("Computer Architecture,", 174.8, 70.9, size=8.2, step=6.9)
        + _word_chars("Edition, 2012", 349.2, 70.9, size=8.2, step=6.2)
    )


def test_one_detected_region_makes_one_label_of_a_spaced_out_line():
    page = _page_with_regions(_footer_chars(), [FOOTER_CAPTION])
    paragraphs = _finder({0: [FOOTER_REGION]})._extract_image_text_paragraphs(page)

    assert [p.unicode for p in paragraphs] == [
        "Digital Design and Computer Architecture, Edition, 2012"
    ]


def test_without_a_region_to_vouch_for_them_the_pieces_stay_apart():
    # The same characters with no layout region over them: the gap rule is
    # all the evidence there is, and it still separates side-by-side
    # labels sharing a row.
    page = _page(_footer_chars())
    paragraphs = _finder({0: [FOOTER_REGION]})._extract_image_text_paragraphs(page)

    assert len(paragraphs) == 3


def test_a_region_covering_only_one_side_of_the_gap_does_not_join_it():
    caption = ("figure_caption", 0.57, (20.0, 67.0, 170.0, 85.0))
    page = _page_with_regions(_footer_chars(), [caption])
    paragraphs = _finder({0: [FOOTER_REGION]})._extract_image_text_paragraphs(page)

    assert len(paragraphs) == 3
