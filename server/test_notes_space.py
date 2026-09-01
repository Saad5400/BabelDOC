"""Tests for the stateless /v1/notes-space endpoint and server.notes_space.

Run from the repo root (pyproject's testpaths only covers tests/, so pass
the path explicitly):

    pytest server/test_notes_space.py
"""

from io import BytesIO

import pymupdf
import pytest

from server import notes_space
from server.conftest import TOKEN

PAGE = (595.0, 842.0)


def _pdf(pages: int = 1, rotations: tuple[int, ...] = ()) -> bytes:
    doc = pymupdf.open()
    for index in range(pages):
        page = doc.new_page(width=PAGE[0], height=PAGE[1])
        page.insert_text((72, 100), f"MARKER{index}", fontsize=24)
        if index < len(rotations) and rotations[index]:
            page.set_rotation(rotations[index])
    try:
        return doc.tobytes()
    finally:
        doc.close()


def _plan(sides, size="md", page=PAGE):
    """The band plan the module will choose for `page` — the tests ask it
    rather than restating it, because the whole point of the sizing is that
    it depends on what the composed page will do on paper."""
    return notes_space.plan_bands([page], sides, size)


def _post(client, pdf, sides, size=None, token=TOKEN):
    data = {"sides": sides}
    if size is not None:
        data["size"] = size
    return client.post(
        "/v1/notes-space",
        headers={"X-Internal-Token": token} if token else {},
        files={"file": ("doc.pdf", pdf, "application/pdf")},
        data=data,
    )


def _mediaboxes(pdf_bytes: bytes) -> list[tuple[float, float, float, float]]:
    doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    try:
        return [tuple(doc[i].mediabox) for i in range(doc.page_count)]
    finally:
        doc.close()


def _first_span_bbox(pdf_bytes: bytes) -> pymupdf.Rect:
    """Where the marker text sits in PAGE SPACE — moves only if content did."""
    doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    try:
        for block in doc[0].get_text("dict")["blocks"]:
            if block["type"] == 0:
                return pymupdf.Rect(block["lines"][0]["spans"][0]["bbox"])
        raise AssertionError("no text on page 0")
    finally:
        doc.close()


def _text(pdf_bytes: bytes) -> str:
    doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    try:
        return " ".join(doc[i].get_text() for i in range(doc.page_count))
    finally:
        doc.close()


# --------------------------------------------------------------------------
# server.notes_space
# --------------------------------------------------------------------------

def test_parse_sides_tolerates_mess_and_orders_canonically():
    assert notes_space.parse_sides("bottom, right") == ["bottom", "right"]
    assert notes_space.parse_sides(" RIGHT ,bottom,,bottom") == ["bottom",
                                                                "right"]
    assert notes_space.parse_sides("top,bottom,left,right") == list(
        notes_space.SIDES)


@pytest.mark.parametrize("raw", ["", " , ", "diagonal", "bottom,diagonal"])
def test_parse_sides_refuses_junk(raw):
    with pytest.raises(notes_space.NotesSpaceError):
        notes_space.parse_sides(raw)


@pytest.mark.parametrize("side", ["bottom", "top", "left", "right"])
def test_each_side_grows_its_own_edge(client, side):
    # The mediabox grows outward on the visual side — bottom is y0 down, top
    # is y1 up, left is x0 leftward, right is x1 rightward (PDF x grows to
    # the right) — by the band this page's plan asks for.
    plan = _plan([side])
    band = plan.band_w if side in ("left", "right") else plan.band_h
    delta = {"bottom": (0.0, -band, 0.0, 0.0),
             "top": (0.0, 0.0, 0.0, band),
             "left": (-band, 0.0, 0.0, 0.0),
             "right": (0.0, 0.0, band, 0.0)}[side]

    pdf = _pdf()
    before = _mediaboxes(pdf)[0]

    resp = _post(client, pdf, side)

    assert resp.status_code == 200
    assert band > 0
    after = _mediaboxes(resp.content)[0]
    for axis in range(4):
        assert after[axis] == pytest.approx(before[axis] + delta[axis])
    assert "MARKER0" in _text(resp.content)  # the content survived


def test_content_stays_put_when_only_bottom_and_right_grow(client):
    # Page space hangs off the top-left corner, so bands that grow only the
    # bottom and right leave the marker's page-space position untouched.
    pdf = _pdf()
    before = _first_span_bbox(pdf)

    resp = _post(client, pdf, "bottom,right")

    assert resp.status_code == 200
    assert _first_span_bbox(resp.content) == before
    plan = _plan(["bottom", "right"])
    after = _mediaboxes(resp.content)[0]
    assert after[1] == pytest.approx(-plan.band_h)
    assert after[2] == pytest.approx(PAGE[0] + plan.band_w)


def test_a_top_band_shifts_the_content_down_in_page_space(client):
    pdf = _pdf()
    before = _first_span_bbox(pdf)

    resp = _post(client, pdf, "top")

    after = _first_span_bbox(resp.content)
    assert after.y0 == pytest.approx(before.y0 + _plan(["top"]).band_h)
    assert after.x0 == pytest.approx(before.x0)


def test_the_ruled_lines_stay_out_of_the_content(client):
    resp = _post(client, _pdf(), "bottom,right")

    assert resp.status_code == 200
    doc = pymupdf.open(stream=BytesIO(resp.content), filetype="pdf")
    try:
        drawings = doc[0].get_drawings()
        assert drawings  # the bands are ruled, not blank
        for drawing in drawings:
            rect = drawing["rect"]
            # Every rule lives wholly in a band: below the content (the
            # bottom band) or to its right (the right band) — never on it.
            assert rect.y0 > PAGE[1] or rect.x0 > PAGE[0]
    finally:
        doc.close()


def test_the_bottom_band_owns_the_corner(client):
    # bottom+right: the bottom band's rules span the FULL new width, running
    # under the right band; the right band's rules stop at the content's
    # height. So the widest rule is wider than the original page.
    resp = _post(client, _pdf(), "bottom,right")

    doc = pymupdf.open(stream=BytesIO(resp.content), filetype="pdf")
    try:
        widths = [drawing["rect"].width for drawing in doc[0].get_drawings()
                  if drawing["rect"].y0 > PAGE[1]]  # the bottom band's rules
        assert widths
        assert max(widths) > PAGE[0]  # reaches under the side band
    finally:
        doc.close()


def test_sizes_scale_the_band(client):
    pdf = _pdf()
    growth = {}
    for size in notes_space.SIZES:
        resp = _post(client, pdf, "bottom", size=size)
        assert resp.status_code == 200
        growth[size] = -_mediaboxes(resp.content)[0][1]
        assert growth[size] == pytest.approx(_plan(["bottom"], size).band_h)
    assert growth["sm"] < growth["md"] < growth["lg"]


def test_md_is_the_default_size(client):
    explicit = _post(client, _pdf(), "bottom", size="md")
    default = _post(client, _pdf(), "bottom")
    assert _mediaboxes(default.content) == _mediaboxes(explicit.content)


def test_a_rotated_page_is_left_untouched(client):
    pdf = _pdf(pages=2, rotations=(0, 90))
    before = _mediaboxes(pdf)

    resp = _post(client, pdf, "bottom")

    assert resp.status_code == 200
    after = _mediaboxes(resp.content)
    assert after[0][1] == pytest.approx(before[0][1] - _plan(["bottom"]).band_h)
    assert after[1] == pytest.approx(before[1])  # the rotated page kept its box


# --------------------------------------------------------------------------
# What the reader actually gets: a SHEET OF PAPER
# --------------------------------------------------------------------------

# Every side combination, not just the tidy ones — the defect lived in
# `top,bottom`, which is the combination nobody had looked at.
ALL_SIDE_SETS = [["top"], ["bottom"], ["left"], ["right"],
                 ["top", "bottom"], ["left", "right"],
                 ["bottom", "right"], ["top", "bottom", "left", "right"]]


@pytest.mark.parametrize("sides", ALL_SIDE_SETS)
@pytest.mark.parametrize("size", ["sm", "md", "lg"])
@pytest.mark.parametrize("page", [(595.0, 842.0),   # A4
                                  (546.0, 793.0),   # a real translated deck
                                  (1092.0, 793.0)])  # a side-by-side dual
def test_the_printed_rules_are_far_enough_apart_to_write_between(sides, size,
                                                                 page):
    """The one thing this feature is for is writing on a printout.

    A printer fits the whole page to the paper, so growing the page shrinks
    the rule spacing with it: `top,bottom` at `lg` used to take an A4 page to
    595x1600, which prints at 53% and puts the rules 4.8 mm apart. This
    asserts the number the reader's pen cares about, for every combination.
    """
    plan = notes_space.plan_bands([page], sides, size)
    width = page[0] + plan.band_w * sum(s in sides for s in ("left", "right"))
    height = page[1] + plan.band_h * sum(s in sides for s in ("top", "bottom"))

    printed = plan.spacing * notes_space.print_scale(width, height)

    assert printed >= notes_space._MIN_PRINTED_SPACING - 0.01, (
        f"{size} on {sides} prints its rules {printed / 72 * 25.4:.1f} mm "
        f"apart on a {width:.0f}x{height:.0f} page")


@pytest.mark.parametrize("sides", ALL_SIDE_SETS)
@pytest.mark.parametrize("size", ["sm", "md", "lg"])
def test_every_band_gets_at_least_one_rule(client, sides, size):
    """A band with no rules in it is just a bigger margin.

    Widening the spacing to keep the printed rules writable can outrun a
    thin band; this is the assertion that stops that trade going too far.
    """
    resp = _post(client, _pdf(), ",".join(sides), size=size)

    assert resp.status_code == 200
    doc = pymupdf.open(stream=BytesIO(resp.content), filetype="pdf")
    try:
        assert doc[0].get_drawings(), f"no rules drawn for {size} on {sides}"
    finally:
        doc.close()


def test_one_document_gets_one_band_on_every_page(client):
    """Pages of different heights must not get different writing room.

    The band used to be a fraction of each page's OWN height, so a mono whose
    vocab strips make some pages taller handed the reader 24% more room on
    some sheets than others — in one printout, of one document.
    """
    doc = pymupdf.open()
    for height in (600.0, 800.0, 1000.0):
        doc.new_page(width=PAGE[0], height=height)
    pdf = doc.tobytes()
    doc.close()

    resp = _post(client, pdf, "bottom", size="lg")

    assert resp.status_code == 200
    boxes = _mediaboxes(resp.content)
    bands = [round(-box[1], 3) for box in boxes]
    assert len(set(bands)) == 1, f"bands differ across pages: {bands}"


def test_a_password_protected_pdf_is_422_not_500(client):
    """An encrypted PDF OPENS, with a correct page_count, and only fails on
    the first page access — several frames down, as a 500. A student handing
    over the locked PDF their university published is the ordinary case."""
    doc = pymupdf.open()
    doc.new_page(width=PAGE[0], height=PAGE[1])
    locked = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                         user_pw="secret", owner_pw="owner")
    doc.close()

    resp = _post(client, locked, "bottom")

    assert resp.status_code == 422
    assert "password" in resp.json()["detail"]


@pytest.mark.parametrize("sides", ["", "diagonal", "bottom;right"])
def test_junk_sides_are_422(client, sides):
    resp = _post(client, _pdf(), sides)
    assert resp.status_code == 422


def test_a_junk_size_is_422(client):
    resp = _post(client, _pdf(), "bottom", size="xl")
    assert resp.status_code == 422


def test_a_missing_sides_field_is_422(client):
    resp = client.post(
        "/v1/notes-space",
        headers={"X-Internal-Token": TOKEN},
        files={"file": ("doc.pdf", _pdf(), "application/pdf")},
    )
    assert resp.status_code == 422


def test_a_non_pdf_is_422(client):
    resp = _post(client, b"not a pdf", "bottom")
    assert resp.status_code == 422


def test_missing_token_is_401(client):
    resp = _post(client, _pdf(), "bottom", token=None)
    assert resp.status_code == 401
