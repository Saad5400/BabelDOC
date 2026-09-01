"""Tests for what a placed graphic is mirrored ABOUT (fork side).

A raster's placement box is not what the reader sees. run14 p22's figure
is placed at x[275.09,505.56] but its soft mask is empty past column 550
of 602, so it prints only to x=486.04 — 19.5 pt of nothing on the right.
Reflected as placed, the picture printed from x=89.2, which is 16.8 pt
outside the slide's own border at x=106.0, and its opaque background
painted over that border.

The layout regions were detected on the RENDERED page, so a figure's
region is its ink: reflecting that instead puts the picture back inside
the frame it was drawn in. MEASURED on the real page, ink runs in the
band y[560,660]:

    source   106.0-107.4 (border)  129.0 ... (content)
    before    89.2- 90.7 (figure)  ... and NO border run at all
    after    106.0-107.4 (border)  113.2 ... (figure, 5.8 pt inside it)

Run from the repo root:

    pytest server/test_mirror_figure_frame.py
"""

from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.midend.typesetting import Typesetting


def _box(x0, y0, x1, y1):
    return il_version_1.Box(x=x0, y=y0, x2=x1, y2=y1)


def _page(*regions):
    return il_version_1.Page(
        mediabox=il_version_1.Mediabox(box=_box(0, 0, 595.28, 841.89)),
        cropbox=il_version_1.Cropbox(box=_box(0, 0, 595.28, 841.89)),
        page_layout=[
            il_version_1.PageLayout(
                id=index + 1, conf=0.5, class_name=class_name, box=box
            )
            for index, (class_name, box) in enumerate(regions)
        ],
        page_number=0,
        unit="point",
        base_operations=il_version_1.BaseOperations(value=""),
    )


PIVOT = 595.2755737304688
# run14 p22, at its real coordinates (IL space, y up).
PLACEMENT = _box(275.0871887, 165.4133911, 505.5621948, 303.1165161)
FIGURE_REGION = ("figure", _box(271.0, 164.0, 486.0, 305.0))
SLIDE_REGION = ("figure", _box(102.0, 94.0, 490.0, 381.0))


def _dx(page, box):
    return Typesetting._graphic_dx(Typesetting.__new__(Typesetting), page, box, PIVOT)


def test_a_padded_raster_is_mirrored_about_its_ink_not_its_placement():
    dx = _dx(_page(FIGURE_REGION, SLIDE_REGION), PLACEMENT)
    # The placement lands at 113.36 rather than 89.71 — inside the slide
    # border at 106.0 instead of 16.8 pt outside it.
    assert round(PLACEMENT.x + dx, 2) == 113.36


def test_a_placement_with_no_matching_region_keeps_its_own_reflection():
    # run67 p5's full-page background raster: its only figure region
    # covers a third of it, which is not that placement's frame.
    page = _page(("figure", _box(146.0, 81.0, 681.0, 324.0)))
    box = _box(0.0, 60.86, 841.89, 534.42)
    assert _dx(page, box) == PIVOT - box.x - box.x2


def test_a_region_that_merely_surrounds_the_figure_is_not_its_frame():
    # The whole lower slide is also detected as a figure. It contains the
    # placement, but it is not its extent, so it must not speak for it.
    assert _dx(_page(SLIDE_REGION), PLACEMENT) == (
        PIVOT - PLACEMENT.x - PLACEMENT.x2
    )


def test_a_text_region_never_speaks_for_a_graphic():
    page = _page(("plain text", _box(271.0, 164.0, 486.0, 305.0)))
    assert _dx(page, PLACEMENT) == PIVOT - PLACEMENT.x - PLACEMENT.x2


def test_a_page_with_no_layout_regions_is_unaffected():
    page = _page()
    assert _dx(page, PLACEMENT) == PIVOT - PLACEMENT.x - PLACEMENT.x2


def test_an_ocr_label_rides_the_same_translation_as_its_raster():
    # The label's raster_region is the placement box, so resolving it the
    # same way keeps label and picture together to the point.
    page = _page(FIGURE_REGION, SLIDE_REGION)
    typesetting = Typesetting.__new__(Typesetting)
    region = [275.0871887, 165.4133911, 505.5621948, 303.1165161]
    label_dx = typesetting._graphic_dx(
        page, Typesetting._region_box(region), PIVOT
    )
    assert label_dx == _dx(page, PLACEMENT)
