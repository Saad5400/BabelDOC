"""Tests for the stateless /v1/compose endpoint and server.compose.

Run from the repo root (pyproject's testpaths only covers tests/, so pass
the path explicitly):

    pytest server/test_compose.py
"""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

TOKEN = "test-token"

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


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from server import app as app_module
    from server import config

    monkeypatch.setattr(config, "DOCTRANSLATE_TOKEN", TOKEN)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    # The compose endpoint is stateless; keep the job worker (and its heavy
    # babeldoc import) out of these tests.
    monkeypatch.setattr(app_module.jobs, "start_worker", lambda: None)
    with TestClient(app_module.app) as c:
        yield c


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


def test_shorter_original_is_padded(client, original_pdf, translated_pdf):
    # Swap the inputs: original (2 pages) shorter than translated (3 pages).
    resp = _post(client, translated_pdf, original_pdf, "alternating")
    assert resp.status_code == 200
    pages = _pages(resp)
    assert len(pages) == 6
    assert _size(pages[4]) == LETTER  # blank pad sized like the translated twin
    assert _size(pages[5]) == LETTER


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
