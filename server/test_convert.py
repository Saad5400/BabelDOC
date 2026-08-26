"""Tests for the stateless /v1/convert endpoint and server.convert.

Run from the repo root (pyproject's testpaths only covers tests/, so pass
the path explicitly):

    pytest server/test_convert.py

The tests that actually drive LibreOffice skip when it is absent, so this file
is runnable on a dev box without a 500 MB office suite; CI and the image both
have it.
"""

import shutil
import zipfile
from io import BytesIO

import pytest
from pypdf import PdfReader

from server import convert
from server.conftest import TOKEN

needs_soffice = pytest.mark.skipif(
    shutil.which("soffice") is None and shutil.which("libreoffice") is None,
    reason="LibreOffice is not installed on this machine",
)


def _pptx(slide_text: str) -> bytes:
    """A minimal but genuinely valid one-slide .pptx carrying `slide_text`."""
    presentation = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
        ' xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst>'
        '<p:sldSz cx="9144000" cy="6858000"/><p:notesSz cx="6858000" cy="9144000"/>'
        "</p:presentation>"
    )
    slide = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
        ' xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        "<p:cSld><p:spTree>"
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        "<p:grpSpPr/>"
        '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr><a:spLocks/>'
        "</p:cNvSpPr><p:nvPr/></p:nvSpPr>"
        '<p:spPr><a:xfrm><a:off x="685800" y="2130425"/>'
        '<a:ext cx="7772400" cy="1470025"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>'
        f"<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang=\"en-US\"/>"
        f"<a:t>{slide_text}</a:t></a:r></a:p></p:txBody></p:sp>"
        "</p:spTree></p:cSld></p:sld>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
        "</Relationships>"
    )
    pres_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>'
        "</Relationships>"
    )

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("ppt/presentation.xml", presentation)
        archive.writestr("ppt/_rels/presentation.xml.rels", pres_rels)
        archive.writestr("ppt/slides/slide1.xml", slide)
    return buffer.getvalue()


def test_extension_gate_rejects_what_it_cannot_open():
    assert convert.is_convertible("lecture.pptx")
    assert convert.is_convertible("NOTES.DOCX")
    # A PDF is not "convertible" — callers must not round-trip one through
    # LibreOffice and silently reflow the layout babeldoc is about to read.
    assert not convert.is_convertible("already.pdf")
    assert not convert.is_convertible("sheet.xlsx")
    assert not convert.is_convertible("noextension")


def test_unconvertible_type_refuses_before_touching_libreoffice():
    with pytest.raises(convert.ConvertError, match="cannot convert"):
        convert.office_to_pdf(b"%PDF-1.4 whatever", "already.pdf")


def test_empty_document_refuses():
    with pytest.raises(convert.ConvertError, match="empty"):
        convert.office_to_pdf(b"", "deck.pptx")


@needs_soffice
def test_converts_a_pptx_to_a_readable_pdf():
    pdf_bytes = convert.office_to_pdf(_pptx("Operating System Security"), "05-OS.pptx")

    assert pdf_bytes.startswith(b"%PDF-")

    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) == 1
    # The slide's words must survive into the PDF's text layer — that layer is
    # exactly what babeldoc translates, so a picture-perfect but textless
    # conversion would be a silent failure.
    assert "Operating System Security" in reader.pages[0].extract_text()


@needs_soffice
def test_a_corrupt_document_fails_loudly_rather_than_returning_junk():
    with pytest.raises(convert.ConvertError):
        convert.office_to_pdf(b"this is not a zip container at all", "broken.pptx")


def test_endpoint_requires_the_shared_secret(client):
    response = client.post("/v1/convert",
                           files={"file": ("deck.pptx", b"x", "application/octet-stream")})
    assert response.status_code == 401


def test_endpoint_422s_an_unsupported_type(client):
    response = client.post(
        "/v1/convert",
        headers={"X-Internal-Token": TOKEN},
        files={"file": ("sheet.xlsx", b"x", "application/octet-stream")},
    )
    assert response.status_code == 422
    assert "pptx" in response.json()["detail"]


@needs_soffice
def test_endpoint_returns_pdf_bytes(client):
    response = client.post(
        "/v1/convert",
        headers={"X-Internal-Token": TOKEN},
        files={"file": ("05-OS.pptx", _pptx("Access Control"),
                        "application/vnd.openxmlformats-officedocument.presentationml.presentation")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")
    assert "Access Control" in PdfReader(BytesIO(response.content)).pages[0].extract_text()
