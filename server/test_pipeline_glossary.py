"""Tests for the pipeline's glossary step (server/pipeline.py::_append_glossary).

No babeldoc run and no API call: the extraction is monkeypatched, and what is
under test is the seam — entries recorded in the sidecar, pages appended to the
finished result, and the never-lose-the-run failure posture.

Run from the repo root (pyproject's testpaths only covers tests/):

    pytest server/test_pipeline_glossary.py
"""

import json

import pymupdf

from server import pipeline
from server import terms

ENTRY = {"term": "Wrapping", "arabic": "التغليف",
         "explanation": "شرح ودّي قصير للمصطلح.", "page": 1,
         "quote": "wrapper classes"}


def _result_pdf(path, pages=2):
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page(width=595, height=842)
    doc.save(str(path))
    doc.close()


def _sidecar(path):
    path.write_text(json.dumps({
        "version": 1, "lang_in": "en", "lang_out": "ar", "total_pages": 2,
        "pages": [{"page_number": 0, "mediabox": [0, 0, 595, 842],
                   "blocks": [], "obstacles": []}],
    }), encoding="utf-8")


def test_entries_land_in_the_sidecar_and_pages_on_the_result(tmp_path,
                                                             monkeypatch):
    pdf, sidecar = tmp_path / "result.pdf", tmp_path / "sidecar.json"
    _result_pdf(pdf)
    _sidecar(sidecar)
    monkeypatch.setattr(terms, "extract_terms", lambda _data: [ENTRY])

    pipeline._append_glossary(pdf, sidecar)

    stored = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stored["glossary"] == [ENTRY]
    assert stored["version"] == 1  # the shape the overlay reads is untouched

    doc = pymupdf.open(str(pdf))
    try:
        assert doc.page_count == 3
        assert "Wrapping" in doc[2].get_text()
    finally:
        doc.close()


def test_zero_terms_is_recorded_but_appends_nothing(tmp_path, monkeypatch):
    pdf, sidecar = tmp_path / "result.pdf", tmp_path / "sidecar.json"
    _result_pdf(pdf)
    _sidecar(sidecar)
    before = pdf.read_bytes()
    monkeypatch.setattr(terms, "extract_terms", lambda _data: [])

    pipeline._append_glossary(pdf, sidecar)

    assert json.loads(sidecar.read_text(encoding="utf-8"))["glossary"] == []
    assert pdf.read_bytes() == before  # the result was never rewritten


def test_an_exploding_extraction_never_touches_the_result(tmp_path,
                                                          monkeypatch):
    pdf, sidecar = tmp_path / "result.pdf", tmp_path / "sidecar.json"
    _result_pdf(pdf)
    _sidecar(sidecar)
    before = pdf.read_bytes()

    def _boom(_data):
        raise RuntimeError("provider down")

    # extract_terms itself never raises in production; this guards the seam
    # anyway — nothing in the glossary step may lose the paid translation.
    monkeypatch.setattr(terms, "extract_terms", _boom)

    pipeline._append_glossary(pdf, sidecar)  # must not raise

    assert pdf.read_bytes() == before


def test_an_unreadable_sidecar_is_skipped(tmp_path):
    pdf, sidecar = tmp_path / "result.pdf", tmp_path / "sidecar.json"
    _result_pdf(pdf)
    sidecar.write_text("{not json", encoding="utf-8")
    before = pdf.read_bytes()

    pipeline._append_glossary(pdf, sidecar)  # must not raise

    assert pdf.read_bytes() == before
