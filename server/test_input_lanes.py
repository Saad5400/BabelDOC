"""Tests for the input-handling decisions in server/pipeline.py.

Which lane a document takes, which of its pages the OCR lane is allowed to
touch, and whether the document is already written in the language we would
translate it into. No babeldoc run and no provider call: every one of these is
a decision made from the input bytes alone, before a penny is spent.

Run from the repo root (pyproject's testpaths only covers tests/):

    pytest server/test_input_lanes.py
"""

import pymupdf
import pytest

from server import ocr_prep
from server import pipeline

ARABIC = "الترجمة العربية للنص الأصلي في هذه الصفحة تحتاج مساحة كافية"
ENGLISH = ("A program consists of one or more classes. Typically each class "
           "is in a separate file and the compiler reads every one of them.")


def _pdf(tmp_path, name, pages):
    """`pages` is one entry per page: a string of text, or None for a blank
    (raster-shaped) page with nothing extractable on it.

    Everything goes in through the same Arabic-capable face — helv cannot
    encode Arabic and would silently drop it, which reads as "the detector
    found no Arabic" rather than as a broken fixture. The face is the one the
    gloss layers already use, cached by babeldoc and baked into the image.
    """
    from server.interlinear import _font_path

    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page(width=595, height=842)
        if not body:
            continue
        page.insert_font(fontname="noto", fontfile=str(_font_path()))
        for row, chunk in enumerate(body.split("\n")):
            page.insert_text((50, 100 + row * 20), chunk, fontsize=11,
                             fontname="noto")
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


def _text_page(copies=3):
    return "\n".join([ENGLISH] * copies)


def _arabic_page(copies=3):
    # insert_text with helv cannot draw Arabic, so the Arabic goes in as a
    # text object we only ever EXTRACT — which is exactly what the language
    # check reads.
    return "\n".join([ARABIC] * copies)


# --------------------------------------------------------------------------
# language_share: the real-text-layer discriminator
# --------------------------------------------------------------------------

def test_language_share_separates_prose_from_a_broken_cid_map():
    """The regression this metric exists for.

    A PDF whose /ToUnicode CMaps are gone renders perfectly and extracts as
    punctuation soup. The old "sensible character" count scored that soup 0.74
    against a 0.5 gate, so it passed as digital and babeldoc replaced a
    legible page with mojibake.
    """
    garbage = "!!\"#$%&’$()*!\"#$%&’\"$((&)&*+,-.\")/01)&*&)\"%&-+$)’"
    assert pipeline.language_share(garbage) < pipeline.MIN_LANGUAGE_SHARE
    assert pipeline.language_share(ENGLISH) > 0.8


def test_language_share_keeps_a_dense_conversion_table():
    """Digits count. On letters and spaces alone a hex/binary table scores
    0.43 — a perfectly good text layer a letters-only gate would have sent
    down the OCR lane."""
    table = "1010 0xFF 255 | 1011 0xAB 171 | 1100 0x0C 12 | 1101 0x5D 93"
    assert pipeline.language_share(table) >= pipeline.MIN_LANGUAGE_SHARE


# --------------------------------------------------------------------------
# classify_pages / has_real_text_layer
# --------------------------------------------------------------------------

def test_a_back_loaded_scan_is_not_called_digital(tmp_path):
    """The head-only sample got this exactly backwards.

    Eight good pages then twelve pixel pages: reading only the first eight
    filed the document as digital, so the scanned tail never reached the OCR
    lane at all.
    """
    path = _pdf(tmp_path, "digital8_then_scan12.pdf",
                [_text_page()] * 8 + [None] * 12)
    verdicts = pipeline.classify_pages(path)
    assert verdicts[:8] == [pipeline.PAGE_TEXT] * 8
    assert verdicts[8:] == [pipeline.PAGE_EMPTY] * 12
    assert pipeline.has_real_text_layer(path, verdicts) is False
    assert pipeline.pages_needing_ocr(verdicts) == list(range(8, 20))


def test_a_scanned_cover_does_not_condemn_the_pages_behind_it(tmp_path):
    """Five pixel cover pages in front of nine good slides.

    This used to flip the WHOLE document to --force-ocr: the nine good pages
    were rasterised and re-OCR'd, and a crisp keyword table came back as a
    smear. The document is mostly text and belongs in the digital lane.
    """
    path = _pdf(tmp_path, "scan_cover_then_slides.pdf",
                [None] * 5 + [_text_page()] * 9)
    verdicts = pipeline.classify_pages(path)
    assert pipeline.has_real_text_layer(path, verdicts) is True
    assert pipeline.pages_needing_ocr(verdicts) == [0, 1, 2, 3, 4]


def test_pages_needing_ocr_covers_garbage_as_well_as_blanks():
    verdicts = [pipeline.PAGE_TEXT, pipeline.PAGE_GARBAGE,
                pipeline.PAGE_EMPTY, pipeline.PAGE_TEXT]
    assert pipeline.pages_needing_ocr(verdicts) == [1, 2]


def test_page_ranges_are_one_based_and_collapsed():
    """ocrmypdf --pages is 1-based; the classifier is 0-based."""
    assert pipeline._page_ranges([0, 1, 2, 7]) == "1-3,8"
    assert pipeline._page_ranges([3]) == "4"
    assert pipeline._page_ranges([0, 1, 4, 5, 6]) == "1-2,5-7"
    assert pipeline._page_ranges([]) == ""


# --------------------------------------------------------------------------
# analyze_source_language
# --------------------------------------------------------------------------

def test_an_arabic_document_is_refused(tmp_path):
    """Nothing used to detect this. An Arabic PDF ran to `done`, was charged
    for, and came back as a strictly degraded Arabic-to-Arabic copy."""
    path = _pdf(tmp_path, "already_arabic.pdf", [_arabic_page()] * 3)
    analysis = pipeline.analyze_source_language(path)
    assert analysis["arabic_share"] >= pipeline.ARABIC_REFUSE_SHARE
    assert analysis["verdict"] == "refuse"


def test_an_english_document_passes(tmp_path):
    path = _pdf(tmp_path, "english.pdf", [_text_page()] * 3)
    analysis = pipeline.analyze_source_language(path)
    assert analysis["verdict"] == "ok"
    assert analysis["arabic_chars"] == 0


def test_a_bilingual_document_warns_rather_than_refusing(tmp_path):
    """A handout that is part Arabic is a legitimate thing to want
    translated. The direction is never swapped on the user's behalf and the
    run is never refused on their behalf either — the caller is told."""
    path = _pdf(tmp_path, "bilingual.pdf",
                [_arabic_page(1) + "\n" + _text_page(1)] * 4)
    analysis = pipeline.analyze_source_language(path)
    assert pipeline.ARABIC_WARN_SHARE <= analysis["arabic_share"] \
        < pipeline.ARABIC_REFUSE_SHARE
    assert analysis["verdict"] == "warn"


def test_a_scan_is_never_refused_on_language(tmp_path):
    """A scan has no text layer to read, so its Arabic share is unknowable
    here — and 'unknown' must never become 'refuse'."""
    path = _pdf(tmp_path, "scan.pdf", [None] * 4)
    verdicts = pipeline.classify_pages(path)
    analysis = pipeline.analyze_source_language(path, verdicts)
    assert analysis["pages_sampled"] == 0
    assert analysis["verdict"] == "ok"


def test_the_refusal_leads_with_its_code(tmp_path):
    """The caller only ever sees the flat `error` string on the job record,
    so the machine-readable part has to survive in it."""
    path = _pdf(tmp_path, "already_arabic.pdf", [_arabic_page()] * 3)
    analysis = pipeline.analyze_source_language(path)
    exc = pipeline.SourceLanguageRefused(analysis)
    assert str(exc).startswith("source_already_in_target_language:")
    assert exc.analysis is analysis


def test_analyze_input_answers_every_door_question(tmp_path):
    """One read of the document, shared by the lane decision, the OCR page
    selection and the language check — this is what /v1/jobs can call to
    refuse at the door instead of after the run."""
    path = _pdf(tmp_path, "mixed.pdf", [_text_page()] * 3 + [None] * 2)
    report = pipeline.analyze_input(path)
    assert report["pages"] == 5
    assert report["scanned"] is False
    assert report["ocr_pages"] == [3, 4]
    assert report["language"]["verdict"] == "ok"
    assert report["page_kinds"][0] == pipeline.PAGE_TEXT


# --------------------------------------------------------------------------
# OCR language selection
# --------------------------------------------------------------------------

def test_tesseract_lang_maps_the_job_languages(monkeypatch):
    """`-l eng` was hardcoded, so every Arabic run on a mixed scan came back
    as Latin noise that was then translated and drawn over the page."""
    monkeypatch.setattr(ocr_prep, "installed_languages",
                        lambda: frozenset({"eng", "ara", "osd"}))
    assert ocr_prep.tesseract_lang("en", "ar") == "eng+ara"
    assert ocr_prep.tesseract_lang("ar", "en") == "ara+eng"
    assert ocr_prep.tesseract_lang("en", "en") == "eng"


def test_tesseract_lang_never_names_a_model_that_is_not_installed(monkeypatch):
    """Naming a missing model does not degrade — tesseract exits non-zero and
    the whole OCR stage fails. An image built without tesseract-ocr-ara has to
    keep working exactly as it did."""
    monkeypatch.setattr(ocr_prep, "installed_languages",
                        lambda: frozenset({"eng", "osd"}))
    assert ocr_prep.tesseract_lang("en", "ar") == "eng"
    assert ocr_prep.tesseract_lang("ar") == "eng"


def test_parse_page_list_round_trips():
    assert ocr_prep.parse_page_list("0,3,4") == {0, 3, 4}
    assert ocr_prep.parse_page_list("") == set()
    assert ocr_prep.parse_page_list(None) == set()


@pytest.mark.parametrize("keep,ocr_calls,survives", [("", 3, False),
                                                    ("0,1,2", 0, True)])
def test_ocr_prep_leaves_kept_pages_alone(tmp_path, monkeypatch, keep,
                                          ocr_calls, survives):
    """This pass STRIPS the text of every page it touches and rewrites it
    from its own tesseract read — which is right for a scan and destroys a
    good slide. A mixed document has to be able to hand it only the pages
    that were really scanned.
    """
    src = _pdf(tmp_path, "in.pdf", [_text_page()] * 3)
    dst = tmp_path / "out.pdf"
    seen = []

    # A tesseract that finds nothing: parse_hocr is stubbed to [], so the
    # only thing left to observe is the strip.
    monkeypatch.setattr(ocr_prep.subprocess, "run",
                        lambda argv, **kw: seen.append(argv))
    monkeypatch.setattr(ocr_prep, "parse_hocr", lambda path: [])
    monkeypatch.setattr(ocr_prep.sys, "argv",
                        ["ocr_prep.py", "--keep-pages", keep,
                         str(src), str(dst)])
    ocr_prep.main()

    assert len(seen) == ocr_calls
    doc = pymupdf.open(str(dst))
    try:
        kept = ["program" in doc[i].get_text().lower() for i in range(3)]
    finally:
        doc.close()
    assert kept == [survives] * 3


def test_ocr_prep_passes_the_language_through(tmp_path, monkeypatch):
    src = _pdf(tmp_path, "in.pdf", [None])
    dst = tmp_path / "out.pdf"
    seen = []
    monkeypatch.setattr(ocr_prep.subprocess, "run",
                        lambda argv, **kw: seen.append(argv))
    monkeypatch.setattr(ocr_prep, "parse_hocr", lambda path: [])
    monkeypatch.setattr(ocr_prep.sys, "argv",
                        ["ocr_prep.py", "--lang", "eng+ara",
                         str(src), str(dst)])
    ocr_prep.main()
    assert seen and seen[0][seen[0].index("-l") + 1] == "eng+ara"
