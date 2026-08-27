"""Tests for the stateless /v1/overlay endpoint and server.interlinear.

Run from the repo root (pyproject's testpaths only covers tests/, so pass the
path explicitly):

    pytest server/test_interlinear.py
"""

import json
from io import BytesIO

import pymupdf
import pytest

from server import interlinear
from server.conftest import TOKEN

PAGE = (595.0, 842.0)

AR_ONE = "تغيير البرمجيات أمر لا مفر منه"
AR_TWO = "تظهر متطلبات جديدة عند استخدام البرنامج"
# A Latin technical term inside an Arabic sentence — the glossary keeps these
# in Latin script on purpose, and bidi has to survive the round trip.
AR_MIXED = "الـ deadlock حالة جمود في CI/CD"


@pytest.fixture(scope="module", autouse=True)
def gloss_font():
    """The real font, fetched once — shaping is what these tests are about.

    babeldoc caches it under $HOME; the Docker image bakes that cache.
    """
    interlinear._font_path()


def _page_pdf(paragraphs: list[tuple[float, float, float, float, str]],
              size: tuple[float, float] = PAGE, rotation: int = 0) -> bytes:
    """A one-page PDF with Latin text at the given (x0, y_top, x1, y_bottom)."""
    doc = pymupdf.open()
    page = doc.new_page(width=size[0], height=size[1])

    for x0, y0, x1, _y1, text in paragraphs:
        page.insert_textbox(pymupdf.Rect(x0, y0, x1, y0 + 40), text,
                            fontname="helv", fontsize=11)

    if rotation:
        page.set_rotation(rotation)

    out = BytesIO()
    doc.save(out)
    doc.close()

    return out.getvalue()


def _sidecar(blocks: list[dict], *, size: tuple[float, float] = PAGE,
             lang_out: str = "ar", version: int = 1) -> dict:
    """A sidecar for one page. Block boxes are given in DISPLAY coordinates
    (y down, like the PDF is written above) and flipped here."""
    height = size[1]

    return {
        "version": version,
        "lang_in": "en",
        "lang_out": lang_out,
        "total_pages": 1,
        "pages": [{
            "page_number": 0,
            "mediabox": [0.0, 0.0, size[0], size[1]],
            "blocks": [{
                "box": [b["box"][0], height - b["box"][3],
                        b["box"][2], height - b["box"][1]],
                "source": b.get("source"),
                "lines": [[line[0], height - line[3], line[2], height - line[1]]
                          for line in b.get("lines", ())],
                "target": b["target"],
                "font_size": b.get("font_size", 11.0),
                "label": "plain text",
            } for b in blocks],
            "obstacles": [],
        }],
    }


def _render(original: bytes, sidecar: dict, **kwargs):
    options = interlinear.OverlayOptions(**kwargs) if kwargs else None

    return interlinear.render_overlay(original, sidecar, options=options)


def _is_arabic(text: str) -> bool:
    """True when a span carries Arabic — base letters or contextual forms."""
    return any("\u0600" <= ch <= "\u06ff" or "\ufb50" <= ch <= "\ufeff"
               for ch in text)


def _drawn_spans(pdf_bytes: bytes) -> list[tuple[pymupdf.Rect, str]]:
    """Every text span on page 1, as (bbox, text)."""
    doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")

    try:
        return [(pymupdf.Rect(span["bbox"]), span["text"])
                for block in doc[0].get_text("dict")["blocks"]
                if block["type"] == 0
                for line in block["lines"] for span in line["spans"]]
    finally:
        doc.close()


# --------------------------------------------------------------------------
# server.interlinear
# --------------------------------------------------------------------------

def test_gloss_lands_in_the_band_above_its_paragraph():
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}])

    pdf, report = _render(original, sidecar)

    assert report == {"pages": 1, "drawn": 1, "skipped": 0, "glossary_pages": 0,
                      "vocab_pages": 0}

    arabic = [rect for rect, text in _drawn_spans(pdf) if _is_arabic(text)]
    assert arabic, "no Arabic was drawn"
    # Above the paragraph it glosses, and clear of it.
    assert max(rect.y1 for rect in arabic) <= 300


def test_the_original_page_is_left_intact():
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}])

    pdf, _ = _render(original, sidecar)

    latin = " ".join(text for _rect, text in _drawn_spans(pdf))
    assert "Software change is inevitable" in latin


def test_arabic_is_shaped_not_left_as_isolated_letters():
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}])

    pdf, _ = _render(original, sidecar)

    drawn = "".join(text for _rect, text in _drawn_spans(pdf))
    # Contextual (presentation) forms are what a shaped Arabic run is made of;
    # an unshaped run would carry only the base letters of the U+06xx block.
    # That they come back out at all also means the font subset kept their cmap
    # entries — without those the finished PDF's Arabic extracts as mojibake.
    assert any("\ufb50" <= ch <= "\ufeff" for ch in drawn)


def test_a_latin_term_inside_an_arabic_gloss_survives():
    original = _page_pdf([(60, 300, 500, 340, "Deadlocks in CI/CD")])
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_MIXED}])

    pdf, report = _render(original, sidecar)

    assert report["drawn"] == 1
    drawn = "".join(text for _rect, text in _drawn_spans(pdf))
    assert "deadlock" in drawn


def test_a_paragraph_with_no_room_above_it_is_skipped_not_overdrawn():
    # Two paragraphs 2 pt apart: the lower one has nowhere to put a gloss.
    original = _page_pdf([(60, 300, 500, 316, "First paragraph"),
                          (60, 318, 500, 334, "Second paragraph")])
    sidecar = _sidecar([{"box": (60, 300, 500, 316), "target": AR_ONE},
                        {"box": (60, 318, 500, 334), "target": AR_TWO}])

    _pdf, report = _render(original, sidecar, min_font_size=8.0, squeeze=1.0)

    assert report["drawn"] == 1
    assert report["skipped"] == 1


def test_glosses_do_not_overlap_each_other():
    original = _page_pdf([(60, 200, 500, 240, "First paragraph"),
                          (60, 260, 500, 300, "Second paragraph")])
    sidecar = _sidecar([{"box": (60, 200, 500, 214), "target": AR_ONE},
                        {"box": (60, 260, 500, 274), "target": AR_TWO}])

    pdf, report = _render(original, sidecar)

    assert report["drawn"] == 2
    # One band per gloss: spans of the same line share a y range and must not
    # be read as two overlapping glosses.
    bands = sorted({(round(rect.y0, 1), round(rect.y1, 1))
                    for rect, text in _drawn_spans(pdf) if _is_arabic(text)})

    assert len(bands) == 2
    assert bands[0][1] <= bands[1][0] + 0.5


def test_align_left_and_right_put_the_gloss_on_opposite_sides():
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    sidecar = _sidecar([{"box": (60, 300, 500, 314), "target": AR_ONE}])

    def gloss_x0(align: str) -> float:
        pdf, _ = _render(original, sidecar, align=align)

        return min(rect.x0 for rect, text in _drawn_spans(pdf)
                   if _is_arabic(text))

    # 'left' hugs the paragraph's own left edge; 'right' is pushed away from it.
    assert gloss_x0("left") < gloss_x0("right")


def test_a_rotated_page_still_gets_its_gloss():
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")],
                         rotation=90)
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}])

    pdf, report = _render(original, sidecar)

    assert report["drawn"] == 1
    doc = pymupdf.open(stream=BytesIO(pdf), filetype="pdf")

    try:
        # The overlay must not leave the page turned the wrong way.
        assert doc[0].rotation == 90
    finally:
        doc.close()


def test_the_output_is_not_bloated_by_a_font_copy_per_gloss():
    blocks = [{"box": (60, 60 + i * 60, 500, 74 + i * 60), "target": AR_TWO}
              for i in range(10)]
    original = _page_pdf([(60, 60 + i * 60, 500, 74 + i * 60, f"Paragraph {i}")
                          for i in range(10)])

    pdf, report = _render(original, _sidecar(blocks))

    assert report["drawn"] == 10
    # The unsubsetted face alone is 15 MB, and the Story engine embeds one copy
    # per box it draws — 10 glosses used to mean ~75 MB.
    assert len(pdf) < 1_000_000


def test_a_gloss_is_never_drawn_over_the_original_text():
    """The overlay's one hard promise: the reader's own document stays legible.

    Lines 1 pt apart leave nowhere to put anything, and a band whose ceiling is
    computed loosely would hand the gloss the whole empty page above and land
    it straight through the line it belongs under.
    """
    lines = [(60, 300 + i * 12, 500, 311 + i * 12) for i in range(4)]
    original = _page_pdf([(60, 300, 500, 348, "A running paragraph " * 6)])
    sidecar = _sidecar([{"box": (60, 300, 500, 348), "target": AR_TWO,
                         "lines": list(lines)}])

    pdf, _report = _render(original, sidecar)

    glosses = [rect for rect, text in _drawn_spans(pdf) if _is_arabic(text)]
    source = [pymupdf.Rect(x0, y0, x1, y1) for x0, y0, x1, y1 in lines]

    for gloss in glosses:
        for line in source:
            assert not (gloss & line).is_valid or (gloss & line).is_empty, (
                f"gloss {gloss} overlaps source line {line}")


def test_a_merged_bullet_list_is_spread_over_its_own_lines():
    """The case the source lines exist for.

    BabelDOC merges a run of same-styled bullets into ONE paragraph, so a whole
    slide list arrives as a single block: the band above it can hold the gloss,
    but only as a crushed lump, while each bullet has an empty line above it
    going unused.
    """
    bullets = [(90, 160 + i * 40, 500, 176 + i * 40) for i in range(4)]
    original = _page_pdf([(x0, y0, x1, y1, f"Bullet number {i}")
                          for i, (x0, y0, x1, y1) in enumerate(bullets)])
    sidecar = _sidecar([{"box": (90, 160, 500, 336),
                         "target": " ".join([AR_TWO] * 4),
                         "lines": list(bullets)}])

    pdf, report = _render(original, sidecar)

    assert report == {"pages": 1, "drawn": 1, "skipped": 0, "glossary_pages": 0,
                      "vocab_pages": 0}
    # One gloss band per bullet, each above its own bullet — not one lump.
    bands = sorted({round(rect.y0, 1) for rect, text in _drawn_spans(pdf)
                    if _is_arabic(text)})
    assert len(bands) == 4


def test_a_tightly_leaded_paragraph_keeps_one_unbroken_gloss():
    """The other side of the same choice: a running paragraph's lines have no
    room, so breaking the sentence up would only make it smaller."""
    lines = [(60, 300 + i * 12, 500, 311 + i * 12) for i in range(4)]
    original = _page_pdf([(60, 300, 500, 348, "A running paragraph " * 6)])
    sidecar = _sidecar([{"box": (60, 300, 500, 348), "target": AR_TWO,
                         "lines": list(lines)}])

    pdf, report = _render(original, sidecar)

    assert report["drawn"] == 1
    bands = {round(rect.y0, 1) for rect, text in _drawn_spans(pdf)
             if _is_arabic(text)}
    # All of it above the paragraph, in one band — nothing wedged into the
    # 1 pt of leading between the source's own lines.
    assert len(bands) == 1
    assert max(bands) < 300


def _with_backdrop(pdf_bytes: bytes, kind: str) -> bytes:
    """The same page, with a full-bleed backdrop behind it."""
    doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    page = doc[0]

    if kind == "image":
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 64))
        pix.set_rect(pix.irect, (210, 225, 245))
        page.insert_image(page.rect, pixmap=pix, overlay=False)
    else:
        page.draw_rect(page.rect, color=None, fill=(0.82, 0.88, 0.96),
                       overlay=False)

    out = BytesIO()
    doc.save(out)
    doc.close()

    return out.getvalue()


@pytest.mark.parametrize("backdrop", ["image", "drawing"])
def test_a_full_bleed_backdrop_does_not_veto_every_gloss(backdrop):
    """A slide deck's background is drawn like anything else on the page.

    Whatever the page is drawn ON — a filled panel, a full-bleed photo —
    overlaps every paragraph on it, so counting it as something a gloss must
    avoid returns the document with nothing on it: a 200, a PDF identical to
    the original, and no error anywhere to say so.
    """
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}])

    _pdf, report = _render(_with_backdrop(original, backdrop), sidecar)

    assert report["drawn"] == 1
    assert report["skipped"] == 0


def test_a_full_bleed_sidecar_figure_does_not_veto_every_gloss():
    """The same page-sized backdrop, seen from the sidecar's side.

    BabelDOC records a background image as a figure, so guarding the page's own
    ink but not the sidecar's account of it would only move the veto.
    """
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}])
    sidecar["pages"][0]["obstacles"] = [[0.0, 0.0, PAGE[0], PAGE[1]]]

    _pdf, report = _render(original, sidecar)

    assert report["drawn"] == 1


def test_a_box_that_under_claims_by_more_than_a_font_size_still_misses_the_text():
    """`_cover_ink` raises an anchor's top by at most one source font size.

    That bound is what stops a tall neighbour bleeding into a paragraph from
    dragging its gloss up the page — but it also means a box that falls short
    by MORE than a font size is only partly corrected. The page's own ink is in
    the obstacle set precisely so the remainder is still caught: the gloss goes
    under the real glyphs rather than through them.
    """
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    # The sidecar claims the paragraph starts 40 pt below where its glyphs
    # really are — far past the one-font-size correction _cover_ink allows.
    sidecar = _sidecar([{"box": (60, 340, 500, 356), "target": AR_ONE,
                         "font_size": 11.0}])

    pdf, report = _render(original, sidecar)

    assert report["drawn"] + report["skipped"] == 1

    glosses = [rect for rect, text in _drawn_spans(pdf) if _is_arabic(text)]
    source = [rect for rect, text in _drawn_spans(pdf) if not _is_arabic(text)]

    for gloss in glosses:
        for line in source:
            overlap = gloss & line
            assert not overlap.is_valid or overlap.is_empty, (
                f"gloss {gloss} landed on the source text {line}")


def test_a_paragraph_at_the_very_top_is_skipped_not_pushed_off_the_page():
    """The other end of the same bound: nothing is ever drawn above the page.

    A paragraph flush against the top edge has no band, and growing its anchor
    to cover its own ascenders only takes more of a band that was not there —
    so it comes back as a skip, which the report tells the caller about.
    """
    original = _page_pdf([(60, 2, 500, 42, "Software change is inevitable")])
    sidecar = _sidecar([{"box": (60, 2, 500, 18), "target": AR_ONE,
                         "font_size": 11.0}])

    pdf, report = _render(original, sidecar)

    assert report == {"pages": 1, "drawn": 0, "skipped": 1, "glossary_pages": 0,
                      "vocab_pages": 0}
    assert not [rect for rect, text in _drawn_spans(pdf) if _is_arabic(text)]


def test_a_stroked_frame_frees_its_inside_but_keeps_its_edge():
    """A callout outline is one path whose bbox is the whole callout.

    Its inside is transparent — and is exactly where the text it frames sits —
    so reading the bbox as ink hands a slide's own furniture a veto over every
    gloss inside it. The four edges it really drew stay obstacles.
    """
    original = _page_pdf([(90, 200, 620, 240, "Bullet number one")])
    doc = pymupdf.open(stream=BytesIO(original), filetype="pdf")
    doc[0].draw_rect(pymupdf.Rect(70, 180, 520, 440), color=(0.2, 0.3, 0.6),
                     width=1.2)
    framed = BytesIO()
    doc.save(framed)
    doc.close()

    sidecar = _sidecar([{"box": (90, 200, 500, 216), "target": AR_ONE,
                         "font_size": 12.0}])

    pdf, report = _render(framed.getvalue(), sidecar)

    # Drawn — the inside of the frame was free all along…
    assert report == {"pages": 1, "drawn": 1, "skipped": 0, "glossary_pages": 0,
                      "vocab_pages": 0}

    # …and still clear of the edge the frame actually drew.
    glosses = [rect for rect, text in _drawn_spans(pdf) if _is_arabic(text)]
    assert glosses
    assert min(rect.y0 for rect in glosses) >= 180.6


def test_an_ltr_target_language_is_glossed_too():
    original = _page_pdf([(60, 300, 500, 340, "Le changement est inevitable")])
    sidecar = _sidecar([{"box": (60, 300, 400, 314),
                         "target": "Software change is inevitable"}],
                       lang_out="en")

    pdf, report = _render(original, sidecar)

    assert report["drawn"] == 1
    assert "Software change is inevitable" in "".join(
        text for _rect, text in _drawn_spans(pdf))


def _glossary_sidecar():
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}])
    sidecar["glossary"] = [{"term": "Wrapping", "arabic": "التغليف",
                            "explanation": "شرح ودّي قصير للمصطلح.",
                            "page": 1, "quote": "wrapper classes"}]

    return sidecar


def test_a_glossary_bearing_sidecar_appends_the_terms_pages():
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])

    pdf, report = _render(original, _glossary_sidecar())

    assert report["drawn"] == 1
    assert report["glossary_pages"] == 1

    doc = pymupdf.open(stream=BytesIO(pdf), filetype="pdf")
    try:
        assert doc.page_count == 2  # the original page + the appended glossary
        assert "Wrapping" in doc[1].get_text()
    finally:
        doc.close()


def test_the_glossary_kill_switch_covers_the_overlay(monkeypatch):
    from server import config

    monkeypatch.setattr(config, "GLOSSARY_PAGES", False)
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])

    pdf, report = _render(original, _glossary_sidecar())

    assert report["glossary_pages"] == 0
    doc = pymupdf.open(stream=BytesIO(pdf), filetype="pdf")
    try:
        assert doc.page_count == 1
    finally:
        doc.close()


def test_a_junk_glossary_never_fails_the_overlay():
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}])
    sidecar["glossary"] = "not even a list"

    pdf, report = _render(original, sidecar)

    assert report["drawn"] == 1
    assert report["glossary_pages"] == 0
    assert pdf.startswith(b"%PDF-")


def test_a_future_sidecar_version_is_refused():
    with pytest.raises(interlinear.OverlayError, match="unsupported sidecar"):
        _render(_page_pdf([(60, 300, 500, 340, "x")]),
                _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}],
                         version=99))


def test_a_sidecar_from_another_document_is_refused():
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}])
    sidecar["pages"][0]["page_number"] = 7

    with pytest.raises(interlinear.OverlayError, match="not from the same run"):
        _render(_page_pdf([(60, 300, 500, 340, "x")]), sidecar)


def test_a_sidecar_with_no_translated_text_is_refused():
    with pytest.raises(interlinear.OverlayError, match="no translated text"):
        _render(_page_pdf([(60, 300, 500, 340, "x")]), _sidecar([]))


def test_an_unreadable_original_is_refused():
    with pytest.raises(interlinear.OverlayError, match="not a readable PDF"):
        _render(b"not a pdf",
                _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}]))


@pytest.mark.parametrize("kwargs", [
    {"scale": 0.0},
    {"min_font_size": 20.0, "max_font_size": 10.0},
    {"align": "middle"},
    # Not a colour at all: it would otherwise be pasted into the gloss
    # stylesheet as written, closing the rule and opening another.
    {"color": "red} body {display:none"},
    {"squeeze": 0.0},
    {"line_height": 9.0},
])
def test_bad_options_are_refused(kwargs):
    with pytest.raises(interlinear.OverlayError):
        _render(_page_pdf([(60, 300, 500, 340, "x")]),
                _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}]),
                **kwargs)


# --------------------------------------------------------------------------
# /v1/overlay
# --------------------------------------------------------------------------

def _post(client, original, sidecar, *, style="interlinear", token=TOKEN, **data):
    body = sidecar if isinstance(sidecar, bytes) else json.dumps(sidecar).encode()

    return client.post(
        "/v1/overlay",
        headers={"X-Internal-Token": token} if token else {},
        files={"original": ("orig.pdf", original, "application/pdf"),
               "sidecar": ("sidecar.json", body, "application/json")},
        data={"style": style, **data},
    )


def test_endpoint_returns_a_pdf_and_reports_the_fit(client):
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    resp = _post(client, original,
                 _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}]))

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF-")
    assert resp.headers["x-overlay-drawn"] == "1"
    assert resp.headers["x-overlay-skipped"] == "0"


def test_endpoint_names_the_download_after_the_original(client):
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    resp = _post(client, original,
                 _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}]))

    # The uploaded stem plus the style: a reader saving three layouts of one
    # deck ends up with three distinguishable files.
    assert resp.headers["content-disposition"] == (
        'attachment; filename="orig.interlinear.pdf"')


def test_endpoint_refuses_an_unknown_style(client):
    original = _page_pdf([(60, 300, 500, 340, "x")])
    resp = _post(client, original,
                 _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}]),
                 style="marginal")

    assert resp.status_code == 422


def test_endpoint_refuses_a_non_pdf_original(client):
    resp = _post(client, b"not a pdf",
                 _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}]))

    assert resp.status_code == 422


def test_endpoint_refuses_a_malformed_sidecar(client):
    original = _page_pdf([(60, 300, 500, 340, "x")])
    resp = _post(client, original, b"{not json")

    assert resp.status_code == 422
    assert "valid JSON" in resp.json()["detail"]


def test_endpoint_refuses_bad_options(client):
    original = _page_pdf([(60, 300, 500, 340, "x")])
    resp = _post(client, original,
                 _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}]),
                 align="middle")

    assert resp.status_code == 422


def test_endpoint_requires_the_token(client):
    original = _page_pdf([(60, 300, 500, 340, "x")])
    resp = _post(client, original,
                 _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}]),
                 token=None)

    assert resp.status_code == 401


# --------------------------------------------------------------------------
# Vocab pages: «كلمات هذه الصفحة» interleaved into the overlay
# --------------------------------------------------------------------------

VOCAB = {"0": [{"w": "inevitable", "ar": "حتمي، لا مفر منه"}]}


def _page_texts(pdf_bytes: bytes) -> list[str]:
    doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")

    try:
        return [doc[i].get_text() for i in range(doc.page_count)]
    finally:
        doc.close()


def test_a_pages_vocab_page_follows_it():
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}])
    sidecar["vocab"] = VOCAB

    result, report = _render(original, sidecar)

    assert report["vocab_pages"] == 1
    texts = _page_texts(result)
    assert len(texts) == 2
    assert "inevitable" in texts[0]  # the original page, glossed
    assert "inevitable" in texts[1]  # its vocab page, right behind it


def test_the_glossary_tail_still_comes_after_the_vocab_pages():
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}])
    sidecar["vocab"] = VOCAB
    sidecar["glossary"] = [{"term": "Wrapping", "arabic": "التغليف",
                            "explanation": "شرح ودّي.", "page": 1,
                            "quote": None}]

    result, report = _render(original, sidecar)

    assert report["vocab_pages"] == 1
    assert report["glossary_pages"] == 1
    texts = _page_texts(result)
    assert len(texts) == 3
    assert "inevitable" in texts[1]  # vocab in the body
    assert "Wrapping" in texts[2]    # terms at the very end


def test_a_sidecar_without_vocab_adds_nothing():
    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}])

    result, report = _render(original, sidecar)

    assert report["vocab_pages"] == 0
    assert len(_page_texts(result)) == 1


def test_the_vocab_kill_switch_disables_the_insert(monkeypatch):
    from server import config
    monkeypatch.setattr(config, "VOCAB_PAGES", False)

    original = _page_pdf([(60, 300, 500, 340, "Software change is inevitable")])
    sidecar = _sidecar([{"box": (60, 300, 400, 314), "target": AR_ONE}])
    sidecar["vocab"] = VOCAB

    result, report = _render(original, sidecar)

    assert report["vocab_pages"] == 0
    assert len(_page_texts(result)) == 1
