"""Tests for the matching phrase highlights (server/phrase_highlights.py) at
both call sites: the /v1/compose pre-pass and the /v1/overlay chips + gloss
spans.

Run from the repo root:

    pytest server/test_phrase_highlights.py

Synthetic PDFs and uploader-shaped sidecars only — no LLM, no network.
"""

import json
import math
from io import BytesIO

import pymupdf
import pytest
from pypdf import PdfReader
from pypdf import PdfWriter

from server import interlinear
from server import phrase_highlights
from server.conftest import TOKEN

LETTER = (612.0, 792.0)   # original pages
A4 = (595.0, 842.0)       # translated pages

# One paragraph's pairs, rects in sidecar space (PDF user space, y up).
S_RECT = [100.0, 700.0, 250.0, 712.0]
T_RECT = [300.0, 650.0, 450.0, 662.0]
PAIRS = [
    {"s": "We can not", "t": "لا يمكننا",
     "s_rects": [S_RECT], "t_rects": [T_RECT]},
    {"s": "create", "t": "إنشاء",
     "s_rects": [[100.0, 680.0, 160.0, 692.0]],
     "t_rects": [[200.0, 630.0, 260.0, 642.0]]},
]


def _pdf(page_sizes: list[tuple[float, float]]) -> bytes:
    writer = PdfWriter()
    for w, h in page_sizes:
        writer.add_blank_page(width=w, height=h)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def _sidecar(pairs=None, *, total_pages=1, glossary=None) -> dict:
    data = {"version": 1, "lang_in": "en", "lang_out": "ar",
            "total_pages": total_pages,
            "pages": [{"page_number": i, "mediabox": [0, 0, *A4],
                       "blocks": [], "obstacles": []}
                      for i in range(total_pages)]}
    if pairs is not None:
        data["pages"][0]["blocks"] = [{
            "box": [60.0, 600.0, 500.0, 714.0], "source": "x", "lines": [],
            "target": "y", "font_size": 11.0, "label": "plain text",
            "pairs": pairs}]
    if glossary is not None:
        data["glossary"] = glossary
    return data


def _fills(pdf_bytes: bytes, page_index: int) -> list[tuple]:
    """Every filled drawing on the page: (bbox, fill RGB, fill opacity)."""
    doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    try:
        return [(pymupdf.Rect(d["rect"]), d["fill"], d.get("fill_opacity", 1))
                for d in doc[page_index].get_drawings() if d.get("fill")]
    finally:
        doc.close()


def _chips(pdf_bytes: bytes, page_index: int) -> list[tuple]:
    """The translucent fills — chips draw at FILL_OPACITY, all else at 1."""
    return [f for f in _fills(pdf_bytes, page_index) if f[2] < 0.99]


def _close(a, b) -> bool:
    return all(abs(x - y) < 0.02 for x, y in zip(a, b, strict=True))


# --------------------------------------------------------------------------
# The pure helpers
# --------------------------------------------------------------------------

def test_the_palette_cycles_per_paragraph():
    assert phrase_highlights.chip_color(0) == phrase_highlights.PALETTE[0]
    assert phrase_highlights.chip_color(4) == phrase_highlights.PALETTE[4]
    assert phrase_highlights.chip_color(5) == phrase_highlights.PALETTE[0]


@pytest.mark.parametrize("garbage", [
    None,
    "not a rect",
    [1, 2, 3],
    [1, 2, 3, 4, 5],
    ["x", 2, 3, 4],
    [None, 2, 3, 4],
    [True, 2, 3, 4],                 # bools are numbers to isinstance only
    [float("nan"), 0, 10, 10],
    [0, 0, float("inf"), 10],
    [10, 0, 10, 5],                  # zero width
    [0, 10, 5, 10],                  # zero height
    [10, 0, 5, 20],                  # negative width
])
def test_garbage_rects_are_refused_silently(garbage):
    assert phrase_highlights.sane_rect(garbage) is None


def test_a_sane_rect_comes_back_as_floats():
    assert phrase_highlights.sane_rect([1, 2, 3, 4]) == [1.0, 2.0, 3.0, 4.0]


def test_pair_chip_rects_keeps_the_side_it_has():
    chips = phrase_highlights.pair_chip_rects(PAIRS, "s_rects")
    assert [(rects, index) for rects, index in chips] == [
        ([S_RECT], 0), ([[100.0, 680.0, 160.0, 692.0]], 1)]

    one_sided = [{"s": "a", "t": "ب"},                       # no rects at all
                 {"s": "b", "t": "ت", "s_rects": [S_RECT]}]
    assert phrase_highlights.pair_chip_rects(one_sided, "s_rects") == [
        ([S_RECT], 1)]
    assert phrase_highlights.pair_chip_rects(one_sided, "t_rects") == []


def test_pair_chip_rects_survives_garbage_and_keeps_colour_positions():
    pairs = ["not a dict",
             {"s": "a", "t": "ب", "s_rects": "not a list"},
             {"s": "b", "t": "ت", "s_rects": [[10, 0, 5, 20], S_RECT]}]
    # The one good rect keeps its pair's colour index (2), garbage vanishes.
    assert phrase_highlights.pair_chip_rects(pairs, "s_rects") == [
        ([S_RECT], 2)]


def test_pair_chip_rects_caps_a_runaway_pairs_list():
    runaway = [{"s": "a", "t": "ب", "s_rects": [S_RECT]}] * 80
    assert len(phrase_highlights.pair_chip_rects(runaway, "s_rects")) == \
        phrase_highlights.MAX_PAIRS


def test_draw_phrase_rects_clamps_to_the_page_and_skips_the_hopeless():
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)

    drawn = phrase_highlights.draw_phrase_rects(
        page,
        [pymupdf.Rect(150, 150, 400, 400),    # clamped to the page corner
         pymupdf.Rect(300, 300, 400, 400),    # entirely off the page
         "not a rect at all"],                # survived, not drawn
        0)

    assert drawn == 1
    fills = [pymupdf.Rect(d["rect"]) for d in page.get_drawings()
             if d.get("fill")]
    assert len(fills) == 1
    assert fills[0].x1 <= 200.5 and fills[0].y1 <= 200.5
    doc.close()


def test_highlight_pairs_without_pairs_returns_the_very_same_bytes():
    pdf = _pdf([A4])
    assert phrase_highlights.highlight_pairs(pdf, _sidecar(), "s_rects") is pdf
    assert phrase_highlights.highlight_pairs(pdf, "junk", "s_rects") is pdf
    assert phrase_highlights.highlight_pairs(
        pdf, {"pages": "junk"}, "s_rects") is pdf


def test_highlight_pairs_survives_an_unreadable_pdf():
    sidecar = _sidecar(PAIRS)
    assert phrase_highlights.highlight_pairs(
        b"not a pdf", sidecar, "s_rects") == b"not a pdf"


def test_highlight_pairs_ignores_pages_the_pdf_does_not_have():
    pdf = _pdf([A4])
    sidecar = _sidecar(PAIRS)
    sidecar["pages"][0]["page_number"] = 7
    assert phrase_highlights.highlight_pairs(pdf, sidecar, "s_rects") is pdf


def test_highlight_pairs_bounds_the_rects_per_page():
    flood = [{"s": "a", "t": "ب", "s_rects": [S_RECT] * 20}] * 50  # 1000 rects
    out = phrase_highlights.highlight_pairs(_pdf([A4]), _sidecar(flood),
                                            "s_rects")
    assert len(_chips(out, 0)) == phrase_highlights.MAX_RECTS_PER_PAGE


# --------------------------------------------------------------------------
# The gloss highlighter (the overlay's Arabic side)
# --------------------------------------------------------------------------

AMBER = ('<span style="background-color:'
         f'{phrase_highlights.span_color(0)}">')
SKY = ('<span style="background-color:'
       f'{phrase_highlights.span_color(1)}">')


def test_the_gloss_splits_into_matching_coloured_spans():
    gh = phrase_highlights.gloss_highlighter("لا يمكننا إنشاء", PAIRS)

    assert gh is not None
    assert gh.html("لا يمكننا إنشاء") == (
        f"{AMBER}لا يمكننا</span> {SKY}إنشاء</span>")


def test_matching_is_whitespace_flexible():
    gh = phrase_highlights.gloss_highlighter(
        " لا   يمكننا إنشاء ",
        [{"t": "لا  يمكننا "}, {"t": " إنشاء"}])

    assert gh is not None
    assert gh.html("لا يمكننا   إنشاء") == (
        f"{AMBER}لا يمكننا</span> {SKY}إنشاء</span>")


def test_spread_chunks_are_consumed_in_order():
    gh = phrase_highlights.gloss_highlighter("لا يمكننا إنشاء", PAIRS)

    assert gh.html("لا يمكننا") == f"{AMBER}لا يمكننا</span>"
    assert gh.html("إنشاء") == f"{SKY}إنشاء</span>"


def test_a_chunk_straddling_a_phrase_boundary_gets_both_colours():
    gh = phrase_highlights.gloss_highlighter("لا يمكننا إنشاء", PAIRS)

    assert gh.html("يمكننا إنشاء") is None  # not the next piece
    assert gh.html("لا يمكننا إنشاء") == (
        f"{AMBER}لا يمكننا</span> {SKY}إنشاء</span>")


def test_a_mismatched_segmentation_means_no_highlighter_at_all():
    assert phrase_highlights.gloss_highlighter(
        "شيء آخر تماما", PAIRS) is None
    assert phrase_highlights.gloss_highlighter(
        "لا يمكننا إنشاء", "not a list") is None
    assert phrase_highlights.gloss_highlighter(
        "لا يمكننا إنشاء", [{"t": 42}, {"t": "إنشاء"}]) is None
    assert phrase_highlights.gloss_highlighter("لا يمكننا إنشاء", []) is None


def test_markup_inside_a_phrase_stays_text():
    gh = phrase_highlights.gloss_highlighter(
        "قبل <b> & بعد", [{"t": "قبل"}, {"t": "<b> & بعد"}])

    markup = gh.html("قبل <b> & بعد")

    assert "&lt;b&gt; &amp; بعد" in markup
    assert "<b>" not in markup


def test_consecutive_words_of_one_phrase_share_one_span():
    gh = phrase_highlights.gloss_highlighter(
        "واحد اثنان ثلاثة", [{"t": "واحد اثنان"}, {"t": "ثلاثة"}])

    assert gh.html("واحد اثنان ثلاثة").count("<span") == 2


# --------------------------------------------------------------------------
# /v1/compose: the PyMuPDF pre-pass
# --------------------------------------------------------------------------

def _post_compose(client, original, translated, fmt, sidecar):
    files = {"original": ("orig.pdf", original, "application/pdf"),
             "translated": ("trans.pdf", translated, "application/pdf")}
    if sidecar is not None:
        files["sidecar"] = ("sidecar.json", sidecar, "application/json")
    return client.post("/v1/compose", headers={"X-Internal-Token": TOKEN},
                       files=files, data={"format": fmt})


def test_alternating_highlights_both_sides_in_matching_colours(client):
    resp = _post_compose(client, _pdf([LETTER]), _pdf([A4]), "alternating",
                         json.dumps(_sidecar(PAIRS)))

    assert resp.status_code == 200
    original_chips = _chips(resp.content, 0)
    translated_chips = _chips(resp.content, 1)
    assert len(original_chips) == 2
    assert len(translated_chips) == 2

    # Phrase 0 amber on BOTH sides, phrase 1 sky on both sides. Phrase 0 sits
    # higher on the page (smaller display y) in both rect sets.
    for chips in (original_chips, translated_chips):
        fills = sorted(chips, key=lambda chip: chip[0].y0)
        assert _close(fills[0][1], phrase_highlights._chip_fill(0))
        assert _close(fills[1][1], phrase_highlights._chip_fill(1))
        assert all(math.isclose(chip[2], phrase_highlights.FILL_OPACITY,
                        abs_tol=0.01) for chip in chips)

    # And the chips sit where the sidecar's y-up rects say, flipped to
    # display space (plus the 1.5 pt padding): S_RECT on the LETTER page…
    s_chip = min(original_chips, key=lambda chip: chip[0].y0)[0]
    assert s_chip.contains(pymupdf.Rect(100, 792 - 712, 250, 792 - 700))
    # …and T_RECT on the A4 page.
    t_chip = min(translated_chips, key=lambda chip: chip[0].y0)[0]
    assert t_chip.contains(pymupdf.Rect(300, 842 - 662, 450, 842 - 650))


def test_side_by_side_scaling_carries_the_chips_into_each_half(client):
    resp = _post_compose(client, _pdf([LETTER]), _pdf([A4]), "side_by_side",
                         json.dumps(_sidecar(PAIRS)))

    assert resp.status_code == 200
    chips = _chips(resp.content, 0)
    assert len(chips) == 4  # two phrases, both sides, one wide page

    half = max(LETTER[0], A4[0])
    left = [chip for chip, _fill, _op in chips if chip.x1 <= half]
    right = [chip for chip, _fill, _op in chips if chip.x0 >= half]
    assert len(left) == 2   # the original's s_rects landed in its half
    assert len(right) == 2  # the translated's t_rects in the other


def test_without_pairs_the_dual_is_chipless(client):
    resp = _post_compose(client, _pdf([LETTER]), _pdf([A4]), "alternating",
                         json.dumps(_sidecar()))

    assert resp.status_code == 200
    assert not _fills(resp.content, 0)
    assert not _fills(resp.content, 1)


def test_the_kill_switch_stops_the_drawing(client, monkeypatch):
    from server import config
    monkeypatch.setattr(config, "PHRASE_HIGHLIGHTS", False)

    resp = _post_compose(client, _pdf([LETTER]), _pdf([A4]), "alternating",
                         json.dumps(_sidecar(PAIRS)))

    assert resp.status_code == 200
    assert not _fills(resp.content, 0)
    assert not _fills(resp.content, 1)


def test_garbage_pairs_are_survived_and_the_good_rect_still_draws(client):
    pairs = [
        {"s": "a", "t": "ب", "s_rects": [[10, 10, 5, 20]]},      # negative area
        {"s": "b", "t": "ت", "s_rects": [[0, 0, 0, 0]],
         "t_rects": "nope"},
        {"s": "c", "t": "ث", "s_rects": [["x", 1, 2, 3], [1, 2, 3], None]},
        {"s": "d", "t": "ج",
         "s_rects": [[float("nan"), 0, 10, 10], [1e400, 0, 10, 10]]},
        "not even a dict",
        {"s": "e", "t": "ح", "s_rects": [[True, 0, 10, 10]]},
        {"s": "f", "t": "خ", "s_rects": [S_RECT]},               # the one good rect
    ]
    # json.dumps writes NaN/Infinity and the endpoint's json.loads reads them
    # back, so the poison coordinates really reach sane_rect.
    resp = _post_compose(client, _pdf([LETTER]), _pdf([A4]), "alternating",
                         json.dumps(_sidecar(pairs)))

    assert resp.status_code == 200
    assert resp.content.startswith(b"%PDF-")
    assert len(_chips(resp.content, 0)) == 1
    assert not _fills(resp.content, 1)


def test_a_pair_missing_one_side_still_highlights_the_other(client):
    pairs = [{"s": "a", "t": "ب", "s_rects": [S_RECT]}]  # no t_rects anywhere
    resp = _post_compose(client, _pdf([LETTER]), _pdf([A4]), "alternating",
                         json.dumps(_sidecar(pairs)))

    assert resp.status_code == 200
    assert len(_chips(resp.content, 0)) == 1
    assert not _fills(resp.content, 1)


def test_the_pre_pass_leaves_the_glossary_tail_accounting_intact(client):
    """Chips + the baked-tail rule together: the translated input's own
    glossary tail is still dropped, the fresh appendix still appended, and the
    content pages still carry their chips."""
    glossary = [{"term": "Wrapping", "arabic": "التغليف",
                 "explanation": "كلمة wrapping تعني «التغليف».",
                 "page": 1, "quote": "wrapper classes"}]
    original = _pdf([LETTER] * 2)
    translated = _pdf([A4] * 2 + [(500.0, 500.0)])  # odd size marks the tail
    sidecar = _sidecar(PAIRS, total_pages=2, glossary=glossary)

    resp = _post_compose(client, original, translated, "alternating",
                         json.dumps(sidecar))

    assert resp.status_code == 200
    reader = PdfReader(BytesIO(resp.content))
    sizes = [(float(p.mediabox.width), float(p.mediabox.height))
             for p in reader.pages]
    assert (500.0, 500.0) not in sizes           # baked tail still ignored
    assert sizes[:4] == [LETTER, A4, LETTER, A4]
    assert len(sizes) > 4                        # fresh glossary still appended
    assert len(_chips(resp.content, 1)) == 2     # and the chips are there

    doc = pymupdf.open(stream=BytesIO(resp.content), filetype="pdf")
    try:
        tail = " ".join(doc[i].get_text() for i in range(4, doc.page_count))
    finally:
        doc.close()
    assert "Wrapping" in tail


# --------------------------------------------------------------------------
# /v1/overlay: chips over the source, coloured spans in the gloss
# --------------------------------------------------------------------------

PAGE = (595.0, 842.0)
AR_TARGET = "لا يمكننا إنشاء"


@pytest.fixture(scope="module", autouse=True)
def gloss_font():
    interlinear._font_path()


def _page_pdf(paragraphs) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE[0], height=PAGE[1])
    for x0, y0, x1, _y1, text in paragraphs:
        page.insert_textbox(pymupdf.Rect(x0, y0, x1, y0 + 40), text,
                            fontname="helv", fontsize=11)
    out = BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()


def _overlay_sidecar(blocks) -> dict:
    """Blocks with boxes/lines/s_rects in DISPLAY coordinates, flipped here."""
    height = PAGE[1]

    def flip(box):
        return [box[0], height - box[3], box[2], height - box[1]]

    return {
        "version": 1, "lang_in": "en", "lang_out": "ar", "total_pages": 1,
        "pages": [{
            "page_number": 0,
            "mediabox": [0.0, 0.0, PAGE[0], PAGE[1]],
            "blocks": [{
                "box": flip(b["box"]),
                "source": b.get("source"),
                "lines": [flip(line) for line in b.get("lines", ())],
                "target": b["target"],
                "font_size": b.get("font_size", 11.0),
                "label": "plain text",
                **({"pairs": [
                    {**pair, **({"s_rects": [flip(r) for r in pair["s_rects"]]}
                                if "s_rects" in pair else {})}
                    for pair in b["pairs"]]} if "pairs" in b else {}),
            } for b in blocks],
            "obstacles": [],
        }],
    }


@pytest.fixture()
def built_markup(monkeypatch):
    """Every gloss markup the render builds, captured at _gloss_html."""
    captured = []
    real = interlinear._gloss_html

    def spy(*args, **kwargs):
        markup = real(*args, **kwargs)
        captured.append(markup)
        return markup

    monkeypatch.setattr(interlinear, "_gloss_html", spy)
    return captured


def _overlay_block(pairs=PAIRS):
    return {"box": (60, 300, 400, 314), "target": AR_TARGET,
            "pairs": [{"s": p["s"], "t": p["t"],
                       "s_rects": [[60, 300, 200, 312]] if i == 0
                       else [[210, 300, 330, 312]]}
                      for i, p in enumerate(pairs)]}


def test_the_overlay_colours_the_gloss_and_chips_the_source(built_markup):
    original = _page_pdf([(60, 300, 500, 340, "We can not create")])
    sidecar = _overlay_sidecar([_overlay_block()])

    pdf, report = interlinear.render_overlay(original, sidecar)

    assert report["drawn"] == 1
    highlighted = [m for m in built_markup if "background-color" in m]
    assert highlighted
    assert AMBER in highlighted[-1] and SKY in highlighted[-1]

    # The source chips: translucent fills over the phrases' display rects.
    chips = _chips(pdf, 0)
    assert len(chips) == 2
    assert {phrase_highlights.chip_color(0), phrase_highlights.chip_color(1)} \
        == {"#" + "".join(f"{round(c * 255):02X}" for c in fill)
            for _rect, fill, _op in chips}
    assert min(chips, key=lambda c: c[0].x0)[0].contains(
        pymupdf.Rect(60, 300, 200, 312))


def test_a_spread_gloss_keeps_its_phrase_colours_chunk_by_chunk(built_markup):
    bullets = [(90, 160 + i * 40, 500, 176 + i * 40) for i in range(3)]
    phrases = ["تظهر متطلبات جديدة", "عند استخدام البرنامج", "في بيئة عمل حقيقية"]
    original = _page_pdf([(x0, y0, x1, y1, f"Bullet number {i}")
                          for i, (x0, y0, x1, y1) in enumerate(bullets)])
    sidecar = _overlay_sidecar([{
        "box": (90, 160, 500, 296), "target": " ".join(phrases),
        "lines": list(bullets),
        "pairs": [{"s": f"phrase {i}", "t": t} for i, t in enumerate(phrases)],
    }])

    _pdf_bytes, report = interlinear.render_overlay(original, sidecar)

    assert report["drawn"] == 1
    highlighted = [m for m in built_markup if "background-color" in m]
    # Three equal-weight lines, three equal phrases: each chunk is one phrase,
    # drawn in order with the palette's first three colours.
    assert len(highlighted) == 3
    for index, markup in enumerate(highlighted):
        assert phrase_highlights.span_color(index) in markup


def test_a_mismatch_falls_back_to_the_plain_gloss_but_keeps_the_chips(
        built_markup):
    original = _page_pdf([(60, 300, 500, 340, "We can not create")])
    block = _overlay_block()
    block["pairs"][0]["t"] = "شيء آخر"  # no longer segments the target
    sidecar = _overlay_sidecar([block])

    pdf, report = interlinear.render_overlay(original, sidecar)

    assert report["drawn"] == 1
    assert not [m for m in built_markup if "background-color" in m]
    assert len(_chips(pdf, 0)) == 2  # the source side is independent


def test_markup_in_a_highlighted_gloss_is_still_escaped(built_markup):
    target = "قبل <b> & بعد"
    original = _page_pdf([(60, 300, 500, 340, "Before and after")])
    sidecar = _overlay_sidecar([{
        "box": (60, 300, 400, 314), "target": target,
        "pairs": [{"s": "Before", "t": "قبل"},
                  {"s": "and after", "t": "<b> & بعد"}]}])

    _pdf_bytes, report = interlinear.render_overlay(original, sidecar)

    assert report["drawn"] == 1
    highlighted = [m for m in built_markup if "background-color" in m]
    assert highlighted
    assert "&lt;b&gt; &amp; بعد" in highlighted[-1]
    assert "<b>" not in highlighted[-1]


def test_the_kill_switch_covers_chips_and_spans_alike(built_markup,
                                                      monkeypatch):
    from server import config
    monkeypatch.setattr(config, "PHRASE_HIGHLIGHTS", False)

    original = _page_pdf([(60, 300, 500, 340, "We can not create")])
    pdf, report = interlinear.render_overlay(
        original, _overlay_sidecar([_overlay_block()]))

    assert report["drawn"] == 1
    assert not [m for m in built_markup if "background-color" in m]
    assert not _fills(pdf, 0)


def test_junk_pairs_never_fail_the_overlay(built_markup):
    original = _page_pdf([(60, 300, 500, 340, "We can not create")])
    block = {"box": (60, 300, 400, 314), "target": AR_TARGET}
    sidecar = _overlay_sidecar([block])
    sidecar["pages"][0]["blocks"][0]["pairs"] = "not even a list"

    pdf, report = interlinear.render_overlay(original, sidecar)

    assert report["drawn"] == 1
    assert not [m for m in built_markup if "background-color" in m]
    assert pdf.startswith(b"%PDF-")
