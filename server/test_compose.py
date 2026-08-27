"""Tests for the stateless /v1/compose endpoint and server.compose.

Run from the repo root (pyproject's testpaths only covers tests/, so pass
the path explicitly):

    pytest server/test_compose.py
"""

import json
from io import BytesIO

import pytest
from pypdf import PdfReader
from pypdf import PdfWriter

from server.conftest import TOKEN

# Distinguishable page sizes (points).
LETTER = (612.0, 792.0)   # original: 3 pages
A4 = (595.0, 842.0)       # translated: 2 pages


def _pdf(page_sizes: list[tuple[float, float]]) -> bytes:
    writer = PdfWriter()
    for w, h in page_sizes:
        writer.add_blank_page(width=w, height=h)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def _text_pdf(marker: str, size: tuple[float, float]) -> bytes:
    """Minimal one-page PDF whose text layer contains `marker`."""
    w, h = size
    content = f"BT /F1 24 Tf 40 40 Td ({marker}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w:g} {h:g}] "
         f"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>").encode(),
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, xref_pos))
    return bytes(out)


@pytest.fixture(scope="module")
def original_pdf() -> bytes:
    return _pdf([LETTER] * 3)


@pytest.fixture(scope="module")
def translated_pdf() -> bytes:
    return _pdf([A4] * 2)


def _post(client, original, translated, fmt, token=TOKEN):
    return client.post(
        "/v1/compose",
        headers={"X-Internal-Token": token} if token else {},
        files={"original": ("orig.pdf", original, "application/pdf"),
               "translated": ("trans.pdf", translated, "application/pdf")},
        data={"format": fmt},
    )


def _pages(response) -> list:
    assert response.headers["content-type"] == "application/pdf"
    return list(PdfReader(BytesIO(response.content)).pages)


def _size(page) -> tuple[float, float]:
    return float(page.mediabox.width), float(page.mediabox.height)


def test_alternating_interleaves_and_pads(client, original_pdf, translated_pdf):
    resp = _post(client, original_pdf, translated_pdf, "alternating")
    assert resp.status_code == 200
    pages = _pages(resp)
    # 3 originals + 2 translated + 1 blank pad = 6, interleaved o,t,o,t,o,blank.
    assert len(pages) == 6
    assert [_size(p) for p in pages[:4]] == [LETTER, A4, LETTER, A4]
    assert _size(pages[4]) == LETTER
    assert _size(pages[5]) == LETTER  # pad blank sized like its original twin


def test_side_by_side_dimensions(client, original_pdf, translated_pdf):
    resp = _post(client, original_pdf, translated_pdf, "side_by_side")
    assert resp.status_code == 200
    pages = _pages(resp)
    assert len(pages) == 3  # max(3, 2)
    half = max(LETTER[0], A4[0])
    # Pages 1-2 pair letter + A4: width = two halves, height = max height.
    for page in pages[:2]:
        assert _size(page) == (2 * half, max(LETTER[1], A4[1]))
    # Page 3 has no translated twin: sized from the original alone.
    assert _size(pages[2]) == (2 * LETTER[0], LETTER[1])


def test_side_by_side_keeps_both_text_layers(client):
    resp = _post(client, _text_pdf("ORIGMARK", LETTER),
                 _text_pdf("TRANSMARK", A4), "side_by_side")
    assert resp.status_code == 200
    text = _pages(resp)[0].extract_text()
    assert "ORIGMARK" in text
    assert "TRANSMARK" in text


def test_alternating_keeps_page_order_of_text(client):
    resp = _post(client, _text_pdf("ORIGMARK", LETTER),
                 _text_pdf("TRANSMARK", A4), "alternating")
    assert resp.status_code == 200
    pages = _pages(resp)
    assert "ORIGMARK" in pages[0].extract_text()
    assert "TRANSMARK" in pages[1].extract_text()


def test_a_longer_translated_appends_its_tail_instead_of_pairing_blanks(
        client, original_pdf, translated_pdf):
    # Swap the inputs: original (2 pages) shorter than translated (3 pages).
    # The mono result may now end with appended glossary pages, so the tail is
    # kept whole at the end rather than interleaved against blanks.
    resp = _post(client, translated_pdf, original_pdf, "alternating")
    assert resp.status_code == 200
    pages = _pages(resp)
    assert len(pages) == 5  # 2 pairs + 1 tail page
    assert [_size(p) for p in pages[:4]] == [A4, LETTER, A4, LETTER]
    assert _size(pages[4]) == LETTER  # the translated tail page itself


def test_side_by_side_appends_the_translated_tail_full_width(
        client, original_pdf, translated_pdf):
    resp = _post(client, translated_pdf, original_pdf, "side_by_side")
    assert resp.status_code == 200
    pages = _pages(resp)
    assert len(pages) == 3  # 2 pairs + 1 tail page
    assert _size(pages[2]) == LETTER  # whole, at its own size


def test_bad_format_is_422(client, original_pdf, translated_pdf):
    resp = _post(client, original_pdf, translated_pdf, "mono")
    assert resp.status_code == 422


def test_missing_file_is_422(client, original_pdf):
    resp = client.post(
        "/v1/compose",
        headers={"X-Internal-Token": TOKEN},
        files={"original": ("orig.pdf", original_pdf, "application/pdf")},
        data={"format": "alternating"},
    )
    assert resp.status_code == 422


def test_non_pdf_is_422(client, original_pdf):
    resp = _post(client, original_pdf, b"not a pdf", "alternating")
    assert resp.status_code == 422


def test_missing_token_is_401(client, original_pdf, translated_pdf):
    resp = _post(client, original_pdf, translated_pdf, "alternating", token=None)
    assert resp.status_code == 401


# --------------------------------------------------------------------------
# The optional sidecar part: glossary pages on the composed dual
# --------------------------------------------------------------------------

GLOSSARY = [{"term": "Wrapping", "arabic": "التغليف",
             "explanation": "كلمة wrapping تعني «التغليف»، مأخوذة من wrap.",
             "page": 1, "quote": "wrapper classes"}]


def _sidecar_json(total_pages, glossary=None):
    data = {"version": 1, "lang_in": "en", "lang_out": "ar",
            "total_pages": total_pages,
            "pages": [{"page_number": i, "mediabox": [0, 0, *A4],
                       "blocks": [], "obstacles": []}
                      for i in range(total_pages)]}
    if glossary is not None:
        data["glossary"] = glossary
    return json.dumps(data)


def _tail_text(response, from_page):
    """Text of the response PDF's pages past `from_page`, via pymupdf.

    pymupdf, not pypdf: the glossary pages carry a subset font whose spans
    pypdf's extractor reads only partially.
    """
    import pymupdf

    doc = pymupdf.open(stream=BytesIO(response.content), filetype="pdf")
    try:
        return " ".join(doc[i].get_text()
                        for i in range(from_page, doc.page_count))
    finally:
        doc.close()


def _post_with_sidecar(client, original, translated, fmt, sidecar):
    return client.post(
        "/v1/compose",
        headers={"X-Internal-Token": TOKEN},
        files={"original": ("orig.pdf", original, "application/pdf"),
               "translated": ("trans.pdf", translated, "application/pdf"),
               "sidecar": ("sidecar.json", sidecar, "application/json")},
        data={"format": fmt},
    )


def test_a_glossary_sidecar_appends_the_terms_pages(client):
    original, translated = _pdf([LETTER] * 2), _pdf([A4] * 2)
    resp = _post_with_sidecar(client, original, translated, "alternating",
                              _sidecar_json(2, GLOSSARY))
    assert resp.status_code == 200
    pages = _pages(resp)
    assert len(pages) > 4  # 2 pairs + at least one glossary page
    assert "Wrapping" in _tail_text(resp, 4)


def test_the_translated_inputs_own_glossary_tail_is_ignored(client):
    # Translated = 2 content pages + 1 baked-in glossary page (3 total); the
    # sidecar records 2 content pages. The baked tail must not appear: the
    # entries are rendered fresh instead, exactly once.
    original = _pdf([LETTER] * 2)
    translated = _pdf([A4] * 2 + [(500.0, 500.0)])  # odd size marks the tail
    resp = _post_with_sidecar(client, original, translated, "alternating",
                              _sidecar_json(2, GLOSSARY))
    assert resp.status_code == 200
    pages = _pages(resp)
    assert (500.0, 500.0) not in [_size(p) for p in pages]
    assert [_size(p) for p in pages[:4]] == [LETTER, A4, LETTER, A4]
    assert "Wrapping" in _tail_text(resp, 4)


def test_a_sidecar_without_glossary_changes_nothing(client, original_pdf,
                                                    translated_pdf):
    plain = _post(client, original_pdf, translated_pdf, "alternating")
    with_sidecar = _post_with_sidecar(client, original_pdf, translated_pdf,
                                      "alternating", _sidecar_json(2))
    assert with_sidecar.status_code == 200
    assert len(_pages(with_sidecar)) == len(_pages(plain))


def test_an_empty_glossary_appends_nothing(client, original_pdf,
                                           translated_pdf):
    resp = _post_with_sidecar(client, original_pdf, translated_pdf,
                              "alternating", _sidecar_json(2, []))
    assert resp.status_code == 200
    assert len(_pages(resp)) == 6  # exactly the plain alternating output


def test_the_kill_switch_disables_the_append(client, monkeypatch):
    from server import config
    monkeypatch.setattr(config, "GLOSSARY_PAGES", False)

    original, translated = _pdf([LETTER] * 2), _pdf([A4] * 2)
    resp = _post_with_sidecar(client, original, translated, "alternating",
                              _sidecar_json(2, GLOSSARY))
    assert resp.status_code == 200
    assert len(_pages(resp)) == 4  # pairs only, no glossary pages


def test_a_malformed_sidecar_is_422(client, original_pdf, translated_pdf):
    resp = _post_with_sidecar(client, original_pdf, translated_pdf,
                              "alternating", "{not json")
    assert resp.status_code == 422
