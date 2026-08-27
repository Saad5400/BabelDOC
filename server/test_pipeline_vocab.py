"""Tests for the pipeline's vocab step (server/pipeline.py::_insert_vocab).

No babeldoc run and no API call: the extraction is monkeypatched, and what is
under test is the seam — entries recorded in the sidecar, each page grown by
its own bottom strip in the finished result, the artifact_layout record that
lets /v1/compose undo the baking, and the never-lose-the-run failure posture.

Run from the repo root (pyproject's testpaths only covers tests/):

    pytest server/test_pipeline_vocab.py
"""

import json

import pymupdf

from server import pipeline
from server import vocab

VOCAB = {"0": [{"w": "declared", "ar": "يُصرَّح عنه"}],
         "1": [{"w": "scope", "ar": "نطاق"}]}


def _result_pdf(path, content_pages=2, appendix_tail=False):
    doc = pymupdf.open()
    for index in range(content_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"Content {chr(97 + index)}", fontsize=24)
    if appendix_tail:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), "TAILMARK", fontsize=24)
    doc.save(str(path))
    doc.close()


def _sidecar(path, total_pages=2):
    path.write_text(json.dumps({
        "version": 1, "lang_in": "en", "lang_out": "ar",
        "total_pages": total_pages,
        "pages": [{"page_number": number, "mediabox": [0, 0, 595, 842],
                   "blocks": [], "obstacles": []}
                  for number in range(total_pages)],
        # Nothing on this branch writes a "glossary" key; a sidecar carrying
        # one (a future deep-terms pass, a foreign artifact) is honoured as
        # the exclusion list, so the read stays pinned here.
        "glossary": [{"term": "Wrapping", "arabic": "التغليف",
                      "explanation": "شرح", "page": 1, "quote": None}],
    }), encoding="utf-8")


def _texts(path):
    doc = pymupdf.open(str(path))
    try:
        return [doc[i].get_text() for i in range(doc.page_count)]
    finally:
        doc.close()


def test_vocab_lands_in_the_sidecar_and_strips_the_pages(tmp_path,
                                                         monkeypatch):
    pdf, sidecar = tmp_path / "result.pdf", tmp_path / "sidecar.json"
    _result_pdf(pdf)
    _sidecar(sidecar)
    monkeypatch.setattr(vocab, "extract_vocab",
                        lambda _data, **_kwargs: VOCAB)

    pipeline._insert_vocab(pdf, sidecar)

    stored = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stored["vocab"] == VOCAB
    assert stored["version"] == 1  # the shape the overlay reads is untouched
    layout = stored["artifact_layout"]
    assert layout["content_pages"] == [0, 1]  # strips insert nothing
    assert set(layout["vocab_strips"]) == {"0", "1"}
    assert all(height > 0 for height in layout["vocab_strips"].values())

    texts = _texts(pdf)
    assert len(texts) == 2  # each page carries its own words
    assert "Content a" in texts[0]
    assert "declared" in texts[0]
    assert "Content b" in texts[1]
    assert "scope" in texts[1]

    doc = pymupdf.open(str(pdf))
    try:
        # The pages grew by exactly the recorded strip heights.
        for index in range(2):
            grown = 842 + layout["vocab_strips"][str(index)]
            assert abs(doc[index].rect.height - grown) < 0.1
    finally:
        doc.close()


def test_an_existing_appendix_tail_stays_at_the_very_end(tmp_path, monkeypatch):
    # A result that already ends with an appendix page past total_pages: the
    # strips draw on the content pages only and the tail keeps its place.
    pdf, sidecar = tmp_path / "result.pdf", tmp_path / "sidecar.json"
    _result_pdf(pdf, appendix_tail=True)
    _sidecar(sidecar)
    monkeypatch.setattr(vocab, "extract_vocab",
                        lambda _data, **_kwargs: VOCAB)

    pipeline._insert_vocab(pdf, sidecar)

    texts = _texts(pdf)
    assert len(texts) == 3  # c0 c1 tail
    assert "declared" in texts[0]
    assert "scope" in texts[1]
    assert "TAILMARK" in texts[2]

    stored = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stored["artifact_layout"]["content_pages"] == [0, 1]
    assert set(stored["artifact_layout"]["vocab_strips"]) == {"0", "1"}


def test_a_sidecars_glossary_terms_are_the_exclusion_list(tmp_path,
                                                          monkeypatch):
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
