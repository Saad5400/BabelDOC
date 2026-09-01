"""Invariants that hold at EVERY endpoint taking a PDF or a sidecar.

Run from the repo root (pyproject's testpaths only covers tests/, so pass
the path explicitly):

    pytest server/test_endpoint_boundary.py

server/ has one test file per module and, until this one, no test that
crossed them. That gap is why a set of defects survived a green suite: each
module was tested against the inputs its own author had in mind, and nothing
asserted that the endpoints agree with each other. They did not. The same
unopenable bytes were a clean 422 through `interlinear_compact` and a 500
through `interlinear`; a password-protected PDF — the ordinary case of a
student uploading the locked file their university published — was a 500
from three endpoints and a 422 from two; a sidecar belonging to a different
run was refused by one endpoint, half-refused by another and silently obeyed
by a third.

So the invariants are stated once, here, and swept across the cross product:

1. No input produces a 5xx. Ever. A bad upload is the caller's mistake and
   comes back as 422.
2. A PDF part is either opened or refused — never opened halfway.
3. A sidecar and a PDF that are not from the same run are refused, at every
   endpoint that takes both.
"""

import json
from io import BytesIO

import pymupdf
import pytest
from pypdf import PdfWriter

from server.conftest import TOKEN

A4 = (595.0, 842.0)
OTHER = (842.0, 1190.0)  # a different page size entirely


# --------------------------------------------------------------------------
# fixtures: the PDFs a real caller manages to send
# --------------------------------------------------------------------------

def _pdf(pages: int = 2, size: tuple[float, float] = A4) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=size[0], height=size[1])
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def _encrypted_pdf(user_pw: str = "secret") -> bytes:
    """A PDF that asks for a password to open.

    The suite had no such fixture, which is precisely why this was invisible:
    an encrypted document OPENS in pymupdf, reports the right page_count, and
    sails past every 0-page and MAX_PAGES gate before failing on the first
    page access — deep inside a builder, as a 500.
    """
    doc = pymupdf.open()
    doc.new_page(width=A4[0], height=A4[1])
    try:
        return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                           user_pw=user_pw, owner_pw="owner")
    finally:
        doc.close()


def _owner_locked_pdf() -> bytes:
    """Restricted, but not password-to-open — this one must still WORK."""
    doc = pymupdf.open()
    doc.new_page(width=A4[0], height=A4[1])
    try:
        return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                           owner_pw="owner",
                           permissions=int(pymupdf.PDF_PERM_PRINT))
    finally:
        doc.close()


def _empty_pdf() -> bytes:
    out = BytesIO()
    PdfWriter().write(out)
    return out.getvalue()


BAD_PDFS = {
    "encrypted": _encrypted_pdf(),
    "magic-then-garbage": b"%PDF-1.4\n" + b"not a pdf at all\n" * 30,
    "truncated": _pdf()[:120],
    "no-pages": _empty_pdf(),
    "not-a-pdf": b"this is a text file",
}


# --------------------------------------------------------------------------
# fixtures: sidecars
# --------------------------------------------------------------------------

def _sidecar(pages: int = 2, size: tuple[float, float] = A4,
             artifact_layout: dict | None = None) -> dict:
    data = {
        "version": 1, "lang_in": "en", "lang_out": "ar",
        "total_pages": pages,
        "pages": [{"page_number": number, "mediabox": [0.0, 0.0, *size],
                   "blocks": [], "obstacles": []}
                  for number in range(pages)],
    }
    if artifact_layout is not None:
        data["artifact_layout"] = artifact_layout
    return data


def _json(data) -> bytes:
    return json.dumps(data).encode()


# --------------------------------------------------------------------------
# the endpoints, described once
# --------------------------------------------------------------------------

def _post(client, path, files, form=None, token=TOKEN):
    return client.post(path,
                       headers={"X-Internal-Token": token} if token else {},
                       files=files, data=form or {})


#: name -> (path, form, which parts are PDFs, builder for a good request)
def _requests(pdf: bytes, sidecar: bytes) -> dict:
    return {
        "compose": ("/v1/compose",
                    {"format": "alternating"},
                    ["original", "translated"],
                    {"original": ("o.pdf", pdf, "application/pdf"),
                     "translated": ("t.pdf", pdf, "application/pdf")}),
        "overlay-interlinear": ("/v1/overlay", {"style": "interlinear"},
                                ["original"],
                                {"original": ("o.pdf", pdf, "application/pdf"),
                                 "sidecar": ("s.json", sidecar,
                                             "application/json")}),
        "overlay-compact": ("/v1/overlay",
                            {"style": "interlinear_compact"}, ["original"],
                            {"original": ("o.pdf", pdf, "application/pdf"),
                             "sidecar": ("s.json", sidecar,
                                         "application/json")}),
        "strip-vocab": ("/v1/strip-vocab", {}, ["translated"],
                        {"translated": ("t.pdf", pdf, "application/pdf"),
                         "sidecar": ("s.json", sidecar, "application/json")}),
        "notes-space": ("/v1/notes-space", {"sides": "bottom", "size": "md"},
                        ["file"],
                        {"file": ("f.pdf", pdf, "application/pdf")}),
    }


ENDPOINTS = list(_requests(b"", b""))


# --------------------------------------------------------------------------
# 1 + 2 — no input produces a 5xx; a PDF part is opened or refused
# --------------------------------------------------------------------------

@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("flavour", list(BAD_PDFS))
def test_a_pdf_the_engine_cannot_use_is_refused_not_crashed(client, endpoint,
                                                            flavour):
    """422, never 500, for every bad PDF at every part of every endpoint.

    This single assertion over the cross product is the one nobody wrote.
    `interlinear_compact` guarded its `pymupdf.open` and `interlinear` did
    not; `notes_space` guarded the open but not `needs_pass`.
    """
    bad = BAD_PDFS[flavour]
    good_sidecar = _json(_sidecar())
    path, form, pdf_parts, files = _requests(bad, good_sidecar)[endpoint]

    for part in pdf_parts:
        payload = dict(files)
        payload[part] = (f"{part}.pdf", bad, "application/pdf")

        response = _post(client, path, payload, form)

        assert response.status_code < 500, (
            f"{endpoint}: a {flavour} PDF in `{part}` returned "
            f"{response.status_code}\n{response.text[:300]}")
        assert response.status_code == 422, (
            f"{endpoint}: a {flavour} PDF in `{part}` returned "
            f"{response.status_code}, expected 422")


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_a_password_protected_pdf_says_so(client, endpoint):
    """The message has to name the cause: this file needs unlocking.

    "not a readable PDF" sends the reader off to re-export a file that is
    perfectly fine and merely locked.
    """
    locked = _encrypted_pdf()
    path, form, pdf_parts, files = _requests(locked, _json(_sidecar()))[endpoint]

    response = _post(client, path, files, form)

    assert response.status_code == 422
    assert "password" in response.json()["detail"].lower(), \
        response.json()["detail"]


@pytest.mark.parametrize("endpoint", ["compose", "notes-space"])
def test_an_owner_locked_pdf_still_works(client, endpoint):
    """Restricted-but-openable is the common case and must NOT be refused."""
    pdf = _owner_locked_pdf()
    path, form, _parts, files = _requests(pdf, _json(_sidecar(1)))[endpoint]

    response = _post(client, path, files, form)

    assert response.status_code == 200, response.text[:300]


# --------------------------------------------------------------------------
# 3 — a sidecar from another run is refused, at every endpoint that takes one
# --------------------------------------------------------------------------

SIDECAR_ENDPOINTS = ["overlay-interlinear", "overlay-compact", "strip-vocab"]


@pytest.mark.parametrize("endpoint", SIDECAR_ENDPOINTS)
@pytest.mark.parametrize("foreign", [
    pytest.param(_sidecar(9), id="longer"),
    pytest.param(_sidecar(2, size=OTHER), id="same-length-different-pages"),
])
def test_a_sidecar_from_another_run_is_refused(client, endpoint, foreign):
    path, form, _parts, files = _requests(_pdf(2), _json(foreign))[endpoint]

    response = _post(client, path, files, form)

    assert response.status_code == 422, (
        f"{endpoint}: a foreign sidecar returned {response.status_code}")
    assert "same run" in response.json()["detail"], response.json()["detail"]


@pytest.mark.parametrize("endpoint", ["overlay-interlinear", "overlay-compact"])
def test_a_shorter_sidecar_is_refused_by_the_overlay(client, endpoint):
    """The half of the guard that was missing.

    interlinear only ever refused a sidecar naming a page the original does
    NOT have. A shorter one names only pages that exist, so it passed —
    MEASURED on the corpus, run23's 4-page sidecar on run30's 25-page
    original drew 63 of another course's glosses onto it and answered 200.

    The overlay can be strict because its sidecar describes the original page
    for page: any difference in the count is conclusive. strip-vocab cannot
    (see the next test).
    """
    path, form, _parts, files = _requests(_pdf(25), _json(_sidecar(4)))[endpoint]

    response = _post(client, path, files, form)

    assert response.status_code == 422
    assert "same run" in response.json()["detail"]


def test_strip_vocab_refuses_the_foreign_sidecar_that_mutilated_run30(client):
    """F4, in the shape it actually took.

    MEASURED on the real corpus before the fix: run30's 25-page mono with
    run23's 4-page sidecar came back 200 with FOUR pages, cropped 546x793 to
    546x519 — a slide sliced in half — and the caller cached that as the
    reader's "without vocabulary" download. The two runs' pages are different
    sizes, which is what makes it provable, and is the case every pair in the
    14-run corpus falls into.

    THE LIMIT, stated plainly: a sidecar that is a strict PREFIX of this
    document and whose pages are the same size cannot be told apart from a
    legitimate one. A baked mono is content pages plus vocabulary pages plus
    an appendix tail, so pages the layout does not name are expected, and
    "fewer pages than the file" is a normal reading of a correct sidecar.
    Page geometry and page counts cannot close that; only a run identifier in
    both artifacts could, and neither carries one.
    """
    mono = _pdf(25, size=A4)
    foreign = _sidecar(4, size=OTHER,
                       artifact_layout={"content_pages": [0, 1, 2, 3]})

    response = _post(client, "/v1/strip-vocab",
                     {"translated": ("t.pdf", mono, "application/pdf"),
                      "sidecar": ("s.json", _json(foreign),
                                  "application/json")})

    assert response.status_code == 422
    assert "same run" in response.json()["detail"]


def test_strip_vocab_keeps_every_content_page_of_its_own_document(client):
    """The control: the right sidecar still strips, and loses nothing."""
    mono = _pdf(25)
    own = _sidecar(25, artifact_layout={"content_pages": list(range(25))})

    response = _post(client, "/v1/strip-vocab",
                     {"translated": ("t.pdf", mono, "application/pdf"),
                      "sidecar": ("s.json", _json(own), "application/json")})

    assert response.status_code == 200
    doc = pymupdf.open(stream=BytesIO(response.content), filetype="pdf")
    try:
        assert doc.page_count == 25
    finally:
        doc.close()


@pytest.mark.parametrize("endpoint", SIDECAR_ENDPOINTS)
@pytest.mark.parametrize("junk", [
    pytest.param(b"[1, 2, 3]", id="a-json-list"),
    pytest.param(b"null", id="null"),
    pytest.param(b'{"artifact_layout": {"content_pages": "nope"}}',
                 id="no-pages-recorded"),
])
def test_a_sidecar_that_is_not_a_sidecar_is_refused(client, endpoint, junk):
    """Returning the input unchanged is indistinguishable from success.

    strip-vocab used to answer 200-with-the-input for all three of these, and
    the caller cached that as the reader's vocabulary-free download — which
    still had every vocabulary strip in it.
    """
    path, form, _parts, files = _requests(_pdf(2), junk)[endpoint]

    response = _post(client, path, files, form)

    assert response.status_code == 422, response.text[:200]


# --------------------------------------------------------------------------
# a lone surrogate: one bad character must not cost the run every rebuild
# --------------------------------------------------------------------------

def _surrogate_sidecar(escaped: bool) -> bytes:
    """The two ways half an emoji reaches `json.loads`.

    `ensure_ascii` writes it as a `\\ud800` escape; without it the surrogate
    goes out as three raw bytes, and `json.loads` decodes a BYTES argument
    with `surrogatepass` — so that form parses CLEANLY and hands the lone
    surrogate straight to the builder.
    """
    data = _sidecar(1)
    data["pages"][0]["blocks"] = [{
        "box": [60.0, 700.0, 520.0, 714.0],
        "source": "a paragraph", "lines": [[60.0, 700.0, 520.0, 714.0]],
        "target": "نص فيه \ud800 حرف تالف", "font_size": 10.0,
        "label": "plain text",
    }]
    if escaped:
        return json.dumps(data).encode()
    return json.dumps(data, ensure_ascii=False).encode("utf-8", "surrogatepass")


@pytest.mark.parametrize("endpoint", SIDECAR_ENDPOINTS)
@pytest.mark.parametrize("escaped", [True, False], ids=["escaped", "raw"])
def test_a_lone_surrogate_does_not_break_the_run_for_ever(client, endpoint,
                                                          escaped):
    """A sidecar's targets are LLM output; half an emoji is an ordinary way
    for that to go wrong. One of them used to make every future free rebuild
    of that run a 500 — permanently, with no way back short of paying for the
    translation again."""
    path, form, _parts, files = _requests(
        _pdf(1), _surrogate_sidecar(escaped))[endpoint]

    response = _post(client, path, files, form)

    assert response.status_code == 200, response.text[:300]


def test_the_surrogate_scrub_leaves_real_characters_alone():
    """Well-formed astral characters are not surrogates once parsed.

    `json.loads` joins a surrogate PAIR into the single character it spells,
    so an emoji that survived the round trip must survive the scrub too —
    otherwise the fix for the broken character breaks the working ones.
    """
    from server.app import _scrub_surrogates

    parsed = json.loads('{"a": "\\ud83d\\ude00 ok", "b": "x\\ud800y"}')

    assert _scrub_surrogates(parsed) == {"a": "😀 ok", "b": "xy"}
