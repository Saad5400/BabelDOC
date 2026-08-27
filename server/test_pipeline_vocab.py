"""Tests for the pipeline's vocab step (server/pipeline.py::_insert_vocab).

No babeldoc run and no API call: the extraction is monkeypatched, and what is
under test is the seam — entries recorded in the sidecar, pages interleaved
into the finished result, the artifact_layout record that lets /v1/compose
undo the interleaving, and the never-lose-the-run failure posture.

Run from the repo root (pyproject's testpaths only covers tests/):

    pytest server/test_pipeline_vocab.py
"""

import json

import pymupdf

from server import pipeline
from server import vocab

VOCAB = {"0": [{"w": "declared", "ar": "يُصرَّح عنه"}],
         "1": [{"w": "scope", "ar": "نطاق"}]}


def _result_pdf(path, content_pages=2, terms_tail=False):
    doc = pymupdf.open()
    for index in range(content_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"Content {chr(97 + index)}", fontsize=24)
    if terms_tail:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), "TERMSTAIL", fontsize=24)
    doc.save(str(path))
    doc.close()


def _sidecar(path, total_pages=2):
    path.write_text(json.dumps({
        "version": 1, "lang_in": "en", "lang_out": "ar",
        "total_pages": total_pages,
        "pages": [{"page_number": number, "mediabox": [0, 0, 595, 842],
                   "blocks": [], "obstacles": []}
                  for number in range(total_pages)],
        "glossary": [{"term": "Wrapping", "arabic": "التغليف",
                      "explanation": "شرح", "page": 1, "quote": None}],
    }), encoding="utf-8")


def _texts(path):
    doc = pymupdf.open(str(path))
    try:
        return [doc[i].get_text() for i in range(doc.page_count)]
    finally:
        doc.close()


def test_vocab_lands_in_the_sidecar_and_pages_interleave(tmp_path, monkeypatch):
    pdf, sidecar = tmp_path / "result.pdf", tmp_path / "sidecar.json"
    _result_pdf(pdf)
    _sidecar(sidecar)
    monkeypatch.setattr(vocab, "extract_vocab",
                        lambda _data, **_kwargs: VOCAB)

    pipeline._insert_vocab(pdf, sidecar)

    stored = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stored["vocab"] == VOCAB
    assert stored["version"] == 1  # the shape the overlay reads is untouched
    assert stored["artifact_layout"] == {"content_pages": [0, 2]}

    texts = _texts(pdf)
    assert len(texts) == 4  # c0 v0 c1 v1
    assert "Content a" in texts[0]
    assert "declared" in texts[1]
    assert "Content b" in texts[2]
    assert "scope" in texts[3]


def test_the_terms_tail_stays_at_the_very_end(tmp_path, monkeypatch):
    pdf, sidecar = tmp_path / "result.pdf", tmp_path / "sidecar.json"
    _result_pdf(pdf, terms_tail=True)  # the state _append_glossary leaves
    _sidecar(sidecar)
    monkeypatch.setattr(vocab, "extract_vocab",
                        lambda _data, **_kwargs: VOCAB)

    pipeline._insert_vocab(pdf, sidecar)

    texts = _texts(pdf)
    assert len(texts) == 5  # c0 v0 c1 v1 terms
    assert "declared" in texts[1]
    assert "scope" in texts[3]
    assert "TERMSTAIL" in texts[4]

    stored = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stored["artifact_layout"] == {"content_pages": [0, 2]}


def test_the_glossarys_terms_are_the_exclusion_list(tmp_path, monkeypatch):
    pdf, sidecar = tmp_path / "result.pdf", tmp_path / "sidecar.json"
    _result_pdf(pdf)
    _sidecar(sidecar)
    captured = {}

    def _capture(_data, exclude=()):
        captured["exclude"] = list(exclude)
        return {}

    monkeypatch.setattr(vocab, "extract_vocab", _capture)

    pipeline._insert_vocab(pdf, sidecar)

    assert captured["exclude"] == ["Wrapping"]


def test_no_vocab_is_recorded_but_inserts_nothing(tmp_path, monkeypatch):
    pdf, sidecar = tmp_path / "result.pdf", tmp_path / "sidecar.json"
    _result_pdf(pdf)
    _sidecar(sidecar)
    before = pdf.read_bytes()
    monkeypatch.setattr(vocab, "extract_vocab", lambda _data, **_kwargs: {})

    pipeline._insert_vocab(pdf, sidecar)

    stored = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stored["vocab"] == {}
    assert "artifact_layout" not in stored  # total_pages still tells the truth
    assert pdf.read_bytes() == before  # the result was never rewritten


def test_an_exploding_extraction_never_touches_the_result(tmp_path,
                                                          monkeypatch):
    pdf, sidecar = tmp_path / "result.pdf", tmp_path / "sidecar.json"
    _result_pdf(pdf)
    _sidecar(sidecar)
    before = pdf.read_bytes()

    def _boom(_data, **_kwargs):
        raise RuntimeError("provider down")

    # extract_vocab itself never raises in production; this guards the seam
    # anyway — nothing in the vocab step may lose the paid translation.
    monkeypatch.setattr(vocab, "extract_vocab", _boom)

    pipeline._insert_vocab(pdf, sidecar)  # must not raise

    assert pdf.read_bytes() == before
    assert "vocab" not in json.loads(sidecar.read_text(encoding="utf-8"))


def test_an_unreadable_sidecar_is_skipped(tmp_path):
    pdf, sidecar = tmp_path / "result.pdf", tmp_path / "sidecar.json"
    _result_pdf(pdf)
    sidecar.write_text("{not json", encoding="utf-8")
    before = pdf.read_bytes()

    pipeline._insert_vocab(pdf, sidecar)  # must not raise

    assert pdf.read_bytes() == before
