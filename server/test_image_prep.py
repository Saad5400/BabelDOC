"""Tests for the image-text OCR prep pass (server/image_prep.py).

Run from the repo root (pyproject's testpaths only covers tests/, so pass the
path explicitly):

    pytest server/test_image_prep.py

The end-to-end test needs the tesseract binary; it is skipped where the
binary is missing (the Docker image always has it).
"""

import json
import shutil

import pymupdf
import pytest

from server import image_prep
from server.ocr_prep import Line
from server.ocr_prep import Word

PAGE = (600.0, 400.0)


def _png_with_text(text: str, size=(300, 150), fontsize=40) -> bytes:
    """A raster image (PNG bytes) with black text on white — the embedded
    'diagram' the pass is supposed to OCR."""
    doc = pymupdf.open()
    page = doc.new_page(width=size[0], height=size[1])
    page.insert_text((20, size[1] / 2), text, fontname="helv",
                     fontsize=fontsize)
    png = page.get_pixmap(dpi=300).tobytes("png")
    doc.close()
    return png


def _pdf_with_images(placements, texts=(), size=PAGE) -> bytes:
    """A one-page PDF with an embedded raster per placement rect, plus
    optional real digital text spans (x, y, string)."""
    doc = pymupdf.open()
    page = doc.new_page(width=size[0], height=size[1])
    for rect, label in placements:
        page.insert_image(pymupdf.Rect(rect), stream=_png_with_text(label),
                          keep_proportion=False)
    for x, y, s in texts:
        page.insert_text((x, y), s, fontname="helv", fontsize=14)
    out = doc.tobytes()
    doc.close()
    return out


def _word(x, y, x2, y2, conf=95, text="word"):
    return Word(x, y, x2, y2, conf, text)


def _line(words, x_size=30.0):
    return Line(min(w.x for w in words), min(w.y for w in words),
                max(w.x2 for w in words), max(w.y2 for w in words),
                x_size, 0.0, words)


# ---------------------------------------------------------------------------
# region discovery


def test_gather_regions_skips_small_placements(tmp_path):
    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_images([
        ((100, 100, 400, 250), "BIG"),        # 300x150pt: qualifies
        ((450, 100, 490, 120), "TINY"),       # 40x20pt: below 50x30
    ]))
    doc = pymupdf.open(src)
    regions = image_prep.gather_regions(doc[0])
    assert len(regions) == 1
    clip, raw, _dpi = regions[0]
    assert clip == pymupdf.Rect(100, 100, 400, 250)


def test_gather_regions_keeps_a_wide_short_text_band(tmp_path):
    # run59's boxed DEFINITION panels are 348 x 29.5 pt — two dense lines of
    # body text, and the whole point of the slide. The old 50 x 30 pt box
    # test dropped them as decoration by half a point.
    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_images([
        ((100, 100, 448, 129.5), "DEFINITION"),
    ]))
    doc = pymupdf.open(src)
    assert len(image_prep.gather_regions(doc[0])) == 1


def test_gather_regions_keeps_a_narrow_tall_figure(tmp_path):
    # Portrait figures (run39's 43.8 x 52.1 pt icons) were rejected for
    # being narrower than MIN_W_PT even though they carry labels.
    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_images([
        ((100, 100, 144, 200), "FIG"),
    ]))
    doc = pymupdf.open(src)
    assert len(image_prep.gather_regions(doc[0])) == 1


def test_gather_regions_still_skips_a_decorative_rule(tmp_path):
    # run9's page rules are 698 x 9.4 pt hairlines. Wide, large in area,
    # and not text — the short side is what gives them away.
    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_images([
        ((10, 100, 590, 109.4), "RULE"),
    ]))
    doc = pymupdf.open(src)
    assert image_prep.gather_regions(doc[0]) == []


def test_gather_regions_merges_stacked_placements(tmp_path):
    # a figure drawn over its own drop shadow must OCR once, not twice
    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_images([
        ((100, 100, 400, 250), "FIGURE"),
        ((110, 110, 390, 240), "FIGURE"),     # almost the same placement
        ((450, 300, 590, 390), "OTHER"),      # separate: stays its own region
    ]))
    doc = pymupdf.open(src)
    regions = image_prep.gather_regions(doc[0])
    assert len(regions) == 2
    merged = next(r for r in regions if r[0].x0 < 200)
    assert merged[1] == pymupdf.Rect(100, 100, 400, 250)  # union of the pair


def test_gather_regions_clips_full_bleed_to_page():
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE[0], height=PAGE[1])
    page.insert_image(pymupdf.Rect(-50, -50, PAGE[0] + 80, PAGE[1] + 90),
                      stream=_png_with_text("BLEED"), keep_proportion=False)
    regions = image_prep.gather_regions(page)
    assert len(regions) == 1
    clip, raw, _dpi = regions[0]
    assert clip == page.rect                 # OCR renders only the page
    assert raw.x0 == -50                     # regions.json keeps the raw box


# ---------------------------------------------------------------------------
# digital-text overlap kill


def test_kill_text_overlaps_kills_only_covered_words():
    covered = _word(100, 100, 200, 130, text="duplicate")
    clear = _word(300, 100, 400, 130, text="fresh")
    lines = [_line([covered, clear])]
    # a digital span right on top of the first word
    image_prep.kill_text_overlaps(lines, [(95, 95, 205, 135)], pad_px=2.0)
    assert covered.dead and covered.reason == "dup"
    assert not clear.dead


def test_kill_text_overlaps_ignores_grazing_contact():
    # sharing an edge sliver is not duplication (overlap share stays small)
    w = _word(100, 100, 200, 130, text="label")
    image_prep.kill_text_overlaps([_line([w])], [(195, 128, 300, 160)],
                                  pad_px=0.0)
    assert not w.dead


# ---------------------------------------------------------------------------
# line segmentation


def test_split_segments_separates_distant_labels():
    # two arrow labels sharing one hOCR line, one glyph height ~30px:
    # the 200px gap splits them, the 10px word spacing does not
    a1, a2 = _word(0, 0, 80, 30, text="generates"), \
        _word(90, 0, 160, 30, text="fast")
    b = _word(360, 0, 470, 30, text="helps-with")
    segs = image_prep.split_segments([a1, a2, b])
    assert [[w.text for w in s] for s in segs] == \
        [["generates", "fast"], ["helps-with"]]


def test_split_segments_keeps_sentences_whole():
    ws = [_word(i * 100, 0, i * 100 + 80, 30, text=f"w{i}") for i in range(4)]
    assert len(image_prep.split_segments(ws)) == 1


# ---------------------------------------------------------------------------
# coordinate mapping


def test_rect_to_pdf_space_flips_y():
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)
    inv = ~page.transformation_matrix
    box = image_prep.rect_to_pdf_space(pymupdf.Rect(100, 50, 300, 150), inv)
    assert box == [100.0, 250.0, 300.0, 350.0]   # y-up: y0 = 400-150


# ---------------------------------------------------------------------------
# end to end


needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract not installed")


@needs_tesseract
def test_prep_document_injects_invisible_runs(tmp_path):
    src, dst = tmp_path / "in.pdf", tmp_path / "out.pdf"
    regions_path = tmp_path / "regions.json"
    img_rect = (100, 100, 400, 250)
    # mixed case: tesseract reliably garbles the last letters of an ALL-CAPS
    # trailing word on this synthetic render («WORLD» -> «WORLL»)
    src.write_bytes(_pdf_with_images(
        [(img_rect, "Sample Label")],
        texts=[(50, 350, "real digital text")],
    ))

    regions = image_prep.prep_document(str(src), str(dst), str(regions_path))

    # regions.json: version 1, the one placement, PDF space y-up
    assert regions["version"] == 1
    assert list(regions["pages"]) == ["0"]
    (entry,) = regions["pages"]["0"]
    x0, y0, x1, y1 = entry["image_bbox"]
    assert (x0, x1) == (100.0, 400.0)
    assert (y0, y1) == (PAGE[1] - 250.0, PAGE[1] - 100.0)
    assert json.loads(regions_path.read_text()) == regions

    # the recognized words landed inside the placement, in the text layer
    out = pymupdf.open(dst)
    words = {w[4].lower(): pymupdf.Rect(w[:4]) for w in out[0].get_text("words")}
    assert "sample" in words and "label" in words
    assert pymupdf.Rect(img_rect).contains(words["sample"])
    assert pymupdf.Rect(img_rect).contains(words["label"])
    assert "real" in words                    # digital layer untouched

    # invisible: the page renders exactly as before
    src_doc = pymupdf.open(src)
    pix_in = src_doc[0].get_pixmap(dpi=120)
    pix_out = out[0].get_pixmap(dpi=120)
    assert pix_in.samples == pix_out.samples


@needs_tesseract
def test_prep_document_never_duplicates_digital_text(tmp_path):
    # the same string exists as REAL text drawn over the image (the REPEAT
    # YOUR CODE case): the OCR word must die, not shadow the text layer
    src, dst = tmp_path / "in.pdf", tmp_path / "out.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE[0], height=PAGE[1])
    page.insert_image(pymupdf.Rect(100, 100, 400, 250),
                      stream=_png_with_text("SHARED"), keep_proportion=False)
    # digital text at the exact spot where the raster text renders
    ocr_box = None
    for w in _probe_ocr_words(page):
        if w[4] == "SHARED":
            ocr_box = pymupdf.Rect(w[:4])
    assert ocr_box is not None, "the probe render must see the raster text"
    page.insert_text((ocr_box.x0, ocr_box.y1), "SHARED", fontname="helv",
                     fontsize=ocr_box.height * 0.8)
    src.write_bytes(doc.tobytes())

    regions = image_prep.prep_document(str(src), str(dst),
                                       str(tmp_path / "regions.json"))
    assert regions["pages"] == {}             # nothing survived => no region
    out_words = [w[4] for w in pymupdf.open(dst)[0].get_text("words")]
    assert out_words.count("SHARED") == 1     # the digital one only


@needs_tesseract
def test_prep_document_leaves_a_corner_wordmark_as_pixels(tmp_path):
    """The gate, end to end: a small emblem in a page corner injects nothing.

    Job 136's Umm Al-Qura logo is a 1.5 %-of-slide raster in the top-right
    corner; it OCR'd as `ll` / `ola_cola` / `ol UMM AL-QURA UNIVERSITY`, was
    "translated" into `أولا _ كولا` and plated over the logo on 36 of that
    deck's 38 pages. Nothing must reach the text layer.
    """
    src, dst = tmp_path / "in.pdf", tmp_path / "out.pdf"
    # 60 x 60 pt of a 600 x 400 pt page (1.5 %), wholly in the top-right
    # corner band, exactly like the real placement.
    src.write_bytes(_pdf_with_images([((510, 20, 570, 80), "Logo")]))

    regions = image_prep.prep_document(str(src), str(dst),
                                       str(tmp_path / "regions.json"))

    assert regions["pages"] == {}
    assert pymupdf.open(dst)[0].get_text("words") == []


@needs_tesseract
def test_prep_document_still_injects_a_readable_diagram_label(tmp_path):
    """The other half of the gate: the same emblem-sized raster in the middle
    of the page, carrying a real word, is still the lane's whole point."""
    src, dst = tmp_path / "in.pdf", tmp_path / "out.pdf"
    src.write_bytes(_pdf_with_images([((100, 100, 400, 250), "Requirements")]))

    regions = image_prep.prep_document(str(src), str(dst),
                                       str(tmp_path / "regions.json"))

    assert list(regions["pages"]) == ["0"]
    words = [w[4].lower() for w in pymupdf.open(dst)[0].get_text("words")]
    assert "requirements" in words


def _probe_ocr_words(page):
    """Where the raster text of a synthetic page OCRs to, via a plain
    image_prep run against a copy — keeps the duplication test honest."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        probe_src = f"{td}/probe.pdf"
        probe_dst = f"{td}/probe_out.pdf"
        single = pymupdf.open()
        single.insert_pdf(page.parent)
        single.save(probe_src)
        image_prep.prep_document(probe_src, probe_dst, f"{td}/r.json")
        return pymupdf.open(probe_dst)[0].get_text("words")


# ---------------------------------------------------------------------------
# legibility floor


def _region_ops(size_px, px_per_pt=4.0):
    """build_region_ops output for one synthetic OCR line of `size_px`."""
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE[0], height=PAGE[1])
    font = pymupdf.Font("helv")
    words = [_word(0, 0, 120, size_px, text="Definition")]
    line = _line(words, x_size=size_px)
    clip = pymupdf.Rect(100, 100, 400, 250)
    segments = image_prep.region_segments([line], clip, px_per_pt)
    ops = image_prep.build_region_ops(
        segments, clip, px_per_pt, ~page.transformation_matrix, font)
    doc.close()
    return ops


def test_build_region_ops_skips_sub_legible_lines():
    # A diagram micro-label: recognised at 4 pt on the page. Replacing the
    # crisp raster with 4 pt vector Arabic is a smear over the artwork, so
    # the run is not injected and the source pixels stay.
    assert _region_ops(size_px=4.0 * 4.0) == []


def test_build_region_ops_keeps_legible_lines():
    # The same label at 10 pt is real content and must still be injected.
    ops = _region_ops(size_px=10.0 * 4.0)
    assert len(ops) == 1
    assert "Tf 3 Tr" in ops[0]          # still an invisible run
    assert "(Definition) Tj" in ops[0]


def test_build_region_ops_skips_zero_size_ocr_noise():
    # Every 0.00 pt "line" in the real corpus is a mis-recognised diagram
    # stroke ('=', '0+0=2'), never text.
    assert _region_ops(size_px=0.0) == []
