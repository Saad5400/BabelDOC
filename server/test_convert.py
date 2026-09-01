"""Tests for the stateless /v1/convert endpoint and server.convert.

Run from the repo root (pyproject's testpaths only covers tests/, so pass
the path explicitly):

    pytest server/test_convert.py

Two layers, on purpose:

* Everything the client has to get right about Gotenberg — the multipart it
  sends, a full queue, a timeout, an unreachable service, a body that is not a
  PDF — runs against an `httpx.MockTransport` through the REAL client code.
  None of those are reproducible against a live service on demand.
* One end-to-end test converts an actual deck through an actual Gotenberg, and
  SKIPS when none is reachable, so this file is runnable on a dev box with
  nothing running. Start one with:

      docker run --rm -p 3000:3000 gotenberg/gotenberg:8.36.0-libreoffice
"""

import zipfile
from io import BytesIO

import httpx
import pytest
from pypdf import PdfReader

from server import convert
from server.conftest import TOKEN


def _gotenberg_is_up() -> bool:
    from server import config

    try:
        response = httpx.get(f"{config.GOTENBERG_URL.rstrip('/')}/health",
                             timeout=2.0)
        return (response.status_code == 200
                and response.json()["details"]["libreoffice"]["status"] == "up")
    except (httpx.HTTPError, ValueError, KeyError):
        return False


needs_gotenberg = pytest.mark.skipif(
    not _gotenberg_is_up(),
    reason="no Gotenberg reachable at GOTENBERG_URL",
)


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _replies(status: int, content: bytes = b"", *, record: list | None = None):
    """A Gotenberg that always answers the same way, remembering the requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        return httpx.Response(status, content=content)

    return _transport(handler)


def _pdf(text: str = "hello") -> bytes:
    """Byte string that passes the %PDF- gate — the gate is what is under test."""
    return b"%PDF-1.7\n" + text.encode() + b"\n%%EOF\n"


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


@pytest.mark.anyio
async def test_unconvertible_type_refuses_before_touching_the_service(monkeypatch):
    monkeypatch.setattr(convert, "TRANSPORT", _explodes())

    with pytest.raises(convert.ConvertError, match="cannot convert"):
        await convert.office_to_pdf(b"%PDF-1.4 whatever", "already.pdf")


@pytest.mark.anyio
async def test_empty_document_refuses_before_touching_the_service(monkeypatch):
    monkeypatch.setattr(convert, "TRANSPORT", _explodes())

    with pytest.raises(convert.ConvertError, match="empty"):
        await convert.office_to_pdf(b"", "deck.pptx")


def _explodes() -> httpx.MockTransport:
    """A Gotenberg that fails the test if it is called at all.

    The two refusals above are the reason the extension and emptiness checks
    exist: a request that cannot succeed must never occupy a slot in a bounded
    queue that other users' documents are waiting in.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"the service was called: {request.url}")

    return _transport(handler)


@pytest.mark.anyio
async def test_a_successful_conversion_returns_the_service_s_pdf(monkeypatch):
    seen: list[httpx.Request] = []
    monkeypatch.setattr(convert, "TRANSPORT",
                        _replies(200, _pdf("converted"), record=seen))

    result = await convert.office_to_pdf(b"PK\x03\x04 a real deck", "05-OS.pptx")

    assert result == _pdf("converted")

    request, = seen
    assert request.method == "POST"
    assert request.url.path == "/forms/libreoffice/convert"


@pytest.mark.anyio
async def test_the_upload_keeps_the_extension_and_carries_a_unique_name(monkeypatch):
    """The two things Gotenberg reads a filename for, both of them silent when wrong.

    The suffix picks the import filter — send a .pptx as `document` and the
    conversion fails — and duplicate names DROP documents rather than erroring,
    so the name must never be one a second part could collide with.
    """
    seen: list[httpx.Request] = []
    monkeypatch.setattr(convert, "TRANSPORT", _replies(200, _pdf(), record=seen))

    await convert.office_to_pdf(b"deck", "05-OS.pptx")
    await convert.office_to_pdf(b"deck", "05-OS.pptx")

    names = [_uploaded_name(request) for request in seen]

    assert all(name.endswith(".pptx") for name in names), names
    assert names[0] != names[1], names
    # The user's own name is not what travels: it is theirs, and it is not
    # guaranteed to be unique or even to be a legal filename.
    assert not any("05-OS" in name for name in names), names


def _uploaded_name(request: httpx.Request) -> str:
    """The `filename=` of the single `files` part in a multipart body."""
    body = request.content
    marker = b'name="files"; filename="'
    at = body.index(marker) + len(marker)
    return body[at:body.index(b'"', at)].decode()


# --------------------------------------------------------------------------
# The 503-vs-422 promise, asserted where the caller actually reads it
# --------------------------------------------------------------------------

#: (what Gotenberg does, the status /v1/convert must answer with).
#: The existing tests below assert the exception CLASS, which is a weaker
#: statement than the one catodemy relies on: its retry keys off the HTTP
#: status (NormalizeToPdf retries 502/503/504 and nothing else), so a fault
#: that reaches it as 422 is not merely mis-worded, it is un-retryable.
CONVERT_FAULTS = [
    # Reaching the service at all. ConnectTimeout and PoolTimeout are
    # subclasses of TimeoutException, so the old ordering caught them in the
    # "the render timed out" branch and blamed the document — which is what a
    # hibernated or restarting Gotenberg looks like from here.
    ("connect-timeout", httpx.ConnectTimeout, 503),
    ("pool-timeout", httpx.PoolTimeout, 503),
    ("connect-error", httpx.ConnectError, 503),
    ("protocol-error", httpx.RemoteProtocolError, 503),
    # Answers about the service rather than the document.
    ("429-rate-limited", 429, 503),
    ("502-bad-gateway", 502, 503),
    ("503-queue-full", 503, 503),
    ("504-gateway-timeout", 504, 503),
    # Answers about the document.
    ("400-bad-request", 400, 422),
    ("500-libreoffice-failed", 500, 422),
    ("200-but-not-a-pdf", (200, b"<html>error</html>"), 422),
]


@pytest.mark.parametrize(("name", "fault", "expected"),
                         CONVERT_FAULTS,
                         ids=[row[0] for row in CONVERT_FAULTS])
def test_the_endpoint_tells_a_busy_service_from_a_bad_document(
        client, monkeypatch, name, fault, expected):
    """What /v1/convert answers, for each way the render can go wrong.

    500 is deliberately a 422. MEASURED against the real Gotenberg 8.36.0, a
    genuinely corrupt .pptx answers 500 with prose about memory saying the
    request "may be retried" — the same status and the same words an actual
    resource failure produces. Neither separates them, so it stays the
    document's fault: a reader whose file really is broken has to be told to
    re-export it.
    """
    if isinstance(fault, tuple):
        transport = _replies(*fault)
    elif isinstance(fault, int):
        transport = _replies(fault, b"gotenberg says something")
    else:
        def handler(request: httpx.Request) -> httpx.Response:
            raise fault("simulated", request=request)

        transport = _transport(handler)

    monkeypatch.setattr(convert, "TRANSPORT", transport)

    response = client.post(
        "/v1/convert",
        headers={"X-Internal-Token": TOKEN},
        files={"file": ("05-OS.pptx", b"deck",
                        "application/vnd.openxmlformats-officedocument."
                        "presentationml.presentation")},
    )

    assert response.status_code == expected, (
        f"{name}: /v1/convert answered {response.status_code}, expected "
        f"{expected} — {response.text[:200]}")


def test_a_503_never_tells_the_reader_their_file_is_broken(client, monkeypatch):
    """The wording, not just the status: this is the lie the class exists to
    stop, and it is the sentence the user reads."""
    monkeypatch.setattr(convert, "TRANSPORT", _replies(502, b"bad gateway"))

    response = client.post(
        "/v1/convert", headers={"X-Internal-Token": TOKEN},
        files={"file": ("05-OS.pptx", b"deck", "application/octet-stream")})

    assert response.status_code == 503
    detail = response.json()["detail"].lower()
    assert "unavailable" in detail or "busy" in detail, detail
    for lie in ("corrupt", "re-export", "password"):
        assert lie not in detail, detail


@pytest.mark.anyio
async def test_a_full_queue_reads_as_busy_not_as_a_bad_document(monkeypatch):
    """503 is Gotenberg's bounded queue, and the wording has to say so.

    A 503 turned into "your file could not be converted" sends a user off to
    re-export a deck that was never the problem.
    """
    monkeypatch.setattr(convert, "TRANSPORT", _replies(503, b"queue is full"))

    with pytest.raises(convert.ConvertUnavailableError, match="busy"):
        await convert.office_to_pdf(b"deck", "05-OS.pptx")


@pytest.mark.anyio
async def test_a_timeout_is_reported_as_a_timeout(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    monkeypatch.setattr(convert, "TRANSPORT", _transport(handler))

    with pytest.raises(convert.ConvertError, match="timed out"):
        await convert.office_to_pdf(b"deck", "05-OS.pptx")


@pytest.mark.anyio
async def test_a_timeout_is_not_blamed_on_the_document(monkeypatch):
    """It stays a plain ConvertError — the caller's 422 — as it always was.

    A render that ran for five minutes and did not finish usually IS about the
    document, and the endpoint's contract for that has not changed.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    monkeypatch.setattr(convert, "TRANSPORT", _transport(handler))

    with pytest.raises(convert.ConvertError) as raised:
        await convert.office_to_pdf(b"deck", "05-OS.pptx")

    assert not isinstance(raised.value, convert.ConvertUnavailableError)


@pytest.mark.anyio
async def test_an_unreachable_service_is_not_blamed_on_the_document(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(convert, "TRANSPORT", _transport(handler))

    with pytest.raises(convert.ConvertUnavailableError, match="could not be reached"):
        await convert.office_to_pdf(b"deck", "05-OS.pptx")


@pytest.mark.anyio
async def test_a_conversion_failure_carries_the_service_s_own_reason(monkeypatch):
    monkeypatch.setattr(convert, "TRANSPORT",
                        _replies(500, b"could not open the document"))

    with pytest.raises(convert.ConvertError) as raised:
        await convert.office_to_pdf(b"not really a deck", "broken.pptx")

    assert "could not open the document" in str(raised.value)
    # A rendering failure IS the document's fault — it must not be reported as
    # the service being unavailable, or the caller retries forever.
    assert not isinstance(raised.value, convert.ConvertUnavailableError)


@pytest.mark.anyio
async def test_a_body_that_is_not_a_pdf_is_refused(monkeypatch):
    """The one invariant every downstream step depends on, checked here.

    Everything after this point — babeldoc, the dual compose, the caller's own
    belt-and-braces check — assumes these bytes are a PDF.
    """
    monkeypatch.setattr(convert, "TRANSPORT", _replies(200, b"<html>oops</html>"))

    with pytest.raises(convert.ConvertError, match="not a PDF"):
        await convert.office_to_pdf(b"deck", "05-OS.pptx")


@pytest.mark.anyio
@pytest.mark.parametrize(("status", "payload", "expected"), [
    (200, b'{"status":"up","details":{"libreoffice":{"status":"up"}}}', "up"),
    # Gotenberg answering while the thing it answers FOR is down converts
    # nothing, so it is not "up" as far as this engine is concerned.
    (200, b'{"status":"up","details":{"libreoffice":{"status":"down"}}}', "missing"),
    (200, b'{"status":"up","details":{}}', "missing"),
    (200, b"not json at all", "missing"),
    (503, b'{"status":"down"}', "missing"),
])
async def test_service_status_reports_the_libreoffice_component(
        monkeypatch, status, payload, expected):
    monkeypatch.setattr(convert, "TRANSPORT", _replies(status, payload))

    assert await convert.service_status() == expected


@pytest.mark.anyio
async def test_service_status_survives_an_unreachable_service(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(convert, "TRANSPORT", _transport(handler))

    assert await convert.service_status() == "missing"


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


def test_endpoint_returns_the_converted_pdf(client, monkeypatch):
    monkeypatch.setattr(convert, "TRANSPORT", _replies(200, _pdf("converted")))

    response = client.post(
        "/v1/convert",
        headers={"X-Internal-Token": TOKEN},
        files={"file": ("05-OS.pptx", b"PK\x03\x04 deck",
                        "application/vnd.openxmlformats-officedocument.presentationml.presentation")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")


def test_endpoint_503s_a_busy_service_rather_than_422ing_the_document(client, monkeypatch):
    """The status code is the whole point: 422 means "fix your file".

    The caller logs and shows the engine's failure; a saturated queue told as a
    422 turns into "your document may be corrupt or password-protected" in
    front of a user whose document is fine.
    """
    monkeypatch.setattr(convert, "TRANSPORT", _replies(503, b"queue is full"))

    response = client.post(
        "/v1/convert",
        headers={"X-Internal-Token": TOKEN},
        files={"file": ("05-OS.pptx", b"PK\x03\x04 deck", "application/octet-stream")},
    )

    assert response.status_code == 503
    assert "busy" in response.json()["detail"]


def test_endpoint_422s_a_document_the_service_could_not_render(client, monkeypatch):
    monkeypatch.setattr(convert, "TRANSPORT", _replies(500, b"could not open the document"))

    response = client.post(
        "/v1/convert",
        headers={"X-Internal-Token": TOKEN},
        files={"file": ("broken.pptx", b"not a zip", "application/octet-stream")},
    )

    assert response.status_code == 422
    assert "could not open the document" in response.json()["detail"]


def test_healthz_reports_gotenberg_as_a_component(client, monkeypatch):
    monkeypatch.setattr(
        convert, "TRANSPORT",
        _replies(200, b'{"status":"up","details":{"libreoffice":{"status":"up"}}}'))

    versions = client.get("/healthz").json()["versions"]

    assert versions["gotenberg"] == "up"
    # The binary it replaced is gone from the image, so it must be gone from
    # the health report too — a "libreoffice: missing" line would park this
    # engine at 503 degraded forever.
    assert "libreoffice" not in versions


def test_healthz_is_degraded_when_the_conversion_service_is_gone(client, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(convert, "TRANSPORT", _transport(handler))

    response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json()["versions"]["gotenberg"] == "missing"


@needs_gotenberg
@pytest.mark.anyio
async def test_a_real_deck_converts_to_a_readable_pdf():
    pdf_bytes = await convert.office_to_pdf(_pptx("Operating System Security"),
                                            "05-OS.pptx")

    assert pdf_bytes.startswith(b"%PDF-")

    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) == 1
    # The slide's words must survive into the PDF's text layer — that layer is
    # exactly what babeldoc translates, so a picture-perfect but textless
    # conversion would be a silent failure.
    assert "Operating System Security" in reader.pages[0].extract_text()


@needs_gotenberg
@pytest.mark.anyio
async def test_a_real_corrupt_document_fails_loudly_rather_than_returning_junk():
    """A .pptx that is not an OOXML package at all.

    Note what is NOT asserted here: a file of PLAIN TEXT named .pptx converts
    happily, because LibreOffice falls back to its text filter on a suffix it
    cannot otherwise honour. That was true of the soffice this replaced too —
    it is the office suite's behaviour, not the transport's — and a one-page
    PDF of the user's bytes is a harmless outcome for input nobody sends.
    """
    package = BytesIO()
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("junk.txt", "not a presentation")

    with pytest.raises(convert.ConvertError) as raised:
        await convert.office_to_pdf(package.getvalue(), "broken.pptx")

    # The service's failure names the USER's file, not the scratch name the
    # upload travelled under.
    assert "broken.pptx" in str(raised.value)
    assert not isinstance(raised.value, convert.ConvertUnavailableError)


@needs_gotenberg
def test_the_real_service_reports_itself_healthy_through_the_endpoint(client):
    response = client.get("/healthz")

    assert response.json()["versions"]["gotenberg"] == "up"
