"""Office-document → PDF normalisation, via the shared Gotenberg service.

BabelDOC translates PDFs and nothing else, but the material people actually
have is a slide deck: university lectures are shipped as .pptx far more often
than as .pdf. Before this module a .pptx was a dead end at the very first step
— the caller had no PDF to submit, so the user was told to go convert the file
themselves.

The conversion is still the ENGINE's job to arrange, for the same reason OCR
is: it is document plumbing the callers must not each have to carry. What
changed is WHERE the office suite lives. It used to be a headless LibreOffice
baked into this image — ~450 MB of Writer and Impress that the translation path
never touches, plus a soffice process forked per request inside a container
capped at 3 GB. Now it is one HTTP call to the shared Gotenberg service, which
runs LibreOffice for every app on the box behind a bounded queue.

The endpoint's contract is unchanged: it is deliberately STATELESS, like
/v1/compose — bytes in, bytes out, no job, no cache, no /data. The caller keeps
the converted PDF and treats it as the original from then on, which is what
makes the downstream dual-format compose work: compose pairs the ORIGINAL PDF
with the translated one, and a .pptx can never be half of that pair.

The call is ASYNC. It is awaited from inside FastAPI's event loop, where the
old `subprocess.run` blocked every other request for the whole render — a
media-heavy deck could stall a health check or a job poll for a minute. An
`httpx.AsyncClient` yields instead.

Gotenberg serialises its own LibreOffice around one profile, so the per-call
user-profile dance this module used to perform is gone with the binary. What
survives is the FILENAME discipline: Gotenberg, like soffice, picks its import
filter from the suffix, and duplicate part filenames silently drop documents —
so each upload gets an explicit, unique name carrying the real extension.
"""

import uuid

import httpx

from server import config

# What the office suite is asked to open. Kept to the document formats a caller
# could plausibly want translated — a spreadsheet's grid survives neither the
# PDF conversion nor the translation in any useful shape, so it stays out.
CONVERTIBLE_EXTENSIONS = (
    "docx", "doc", "odt", "rtf",
    "pptx", "ppt", "odp",
)


# The transport every call is made over. `None` means httpx's own — the only
# value production ever sees. It exists as a seam so the tests can drive a
# mock Gotenberg (httpx.MockTransport) through the REAL client code: timeouts,
# a full queue and a refused connection are all things this module has to get
# right and none of them are reproducible against a live service on demand.
TRANSPORT: httpx.AsyncBaseTransport | None = None


class ConvertError(Exception):
    """The document could not be turned into a PDF."""


class ConvertUnavailableError(ConvertError):
    """The conversion SERVICE could not do the work — nothing to do with the file.

    Separate from its parent because the two mean opposite things to a user:
    a plain ConvertError says "this document is the problem, re-export it",
    while this one says "the machinery is busy or down, the same file will work
    in a minute". Telling a reader to re-export a perfectly good deck because
    a queue was full is the failure this class exists to prevent.
    """


def is_convertible(filename: str) -> bool:
    return extension_of(filename) in CONVERTIBLE_EXTENSIONS


def extension_of(filename: str) -> str:
    return (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""


async def office_to_pdf(data: bytes, filename: str) -> bytes:
    """Convert one office document to PDF bytes.

    `filename` is only read for its EXTENSION: LibreOffice picks its import
    filter from the file suffix, so the name sent to Gotenberg has to keep it.
    The user's actual name never leaves this process.
    """
    extension = extension_of(filename)

    if extension not in CONVERTIBLE_EXTENSIONS:
        raise ConvertError(f"cannot convert .{extension or '?'} to PDF")

    if not data:
        raise ConvertError("the document is empty")

    # Unique per call, and never the user's own name: Gotenberg keys the parts
    # of a multipart upload by filename, and two parts sharing one name lose a
    # document silently. Single-file here, but the invariant is cheap to hold
    # and expensive to rediscover.
    upload_name = f"{uuid.uuid4().hex}.{extension}"

    try:
        async with _client(_timeout()) as client:
            response = await client.post(
                f"{config.GOTENBERG_URL.rstrip('/')}/forms/libreoffice/convert",
                files={"files": (upload_name, data,
                                 "application/octet-stream")},
            )
    except httpx.TimeoutException as exc:
        # Same wording the soffice ceiling used to produce: from the caller's
        # side a render that never finished is one story, wherever it ran.
        raise ConvertError("converting the document to PDF timed out") from exc
    except httpx.RequestError as exc:
        raise ConvertUnavailableError(
            f"the conversion service could not be reached: {exc}") from exc

    if response.status_code == 503:
        # Gotenberg's bounded queue, full. The document is fine; the service is
        # saturated — so this must never read as "your file is broken".
        raise ConvertUnavailableError(
            "the conversion service is busy; try again in a moment")

    if response.status_code != 200:
        raise ConvertError(
            "the conversion service could not render the document"
            + _detail_of(response, upload_name, filename))

    pdf_bytes = response.content

    # Kept from the soffice era for the same reason it was written: the ONE
    # thing every downstream step assumes is that these bytes are a PDF.
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ConvertError("the conversion service returned a file that is not a PDF")

    return pdf_bytes


async def service_status() -> str:
    """Gotenberg's LibreOffice, as /healthz reports it: "up" or "missing".

    Deliberately a probe of the COMPONENT that matters, not of the service as a
    whole: Gotenberg answering while its LibreOffice is down would convert
    nothing, and /healthz exists to say so before a user finds out.
    """
    try:
        async with _client(_health_timeout()) as client:
            response = await client.get(f"{config.GOTENBERG_URL.rstrip('/')}/health")

        if response.status_code != 200:
            return "missing"

        status = (response.json()
                  .get("details", {})
                  .get("libreoffice", {})
                  .get("status"))
    except (httpx.HTTPError, ValueError, AttributeError):
        return "missing"

    return "up" if status == "up" else "missing"


def _client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    """The one place the HTTP client is built (see TRANSPORT)."""
    return httpx.AsyncClient(timeout=timeout, transport=TRANSPORT)


def _timeout() -> httpx.Timeout:
    """Read ceiling above Gotenberg's own, connect ceiling far below it.

    Gotenberg gives up on a conversion at 300 s and answers with an error; the
    client must outlast that, or a wedged render comes back as a bare timeout
    instead of the service's own account of what went wrong. Connecting, by
    contrast, is a container on the same docker network: seconds, or it is not
    there at all.
    """
    return httpx.Timeout(config.GOTENBERG_TIMEOUT_SECONDS, connect=10.0)


def _health_timeout() -> httpx.Timeout:
    """A health check must answer fast or count as a failure."""
    return httpx.Timeout(5.0, connect=2.0)


def _detail_of(response: httpx.Response, upload_name: str, filename: str) -> str:
    """Gotenberg's own explanation, when it sent one.

    Its error bodies are short plain text and name the file they failed on —
    by the scratch name WE gave it, which means nothing to anyone reading a
    log. Swapping the user's own name back in is the difference between
    "failed to convert '9f3c…c98.pptx'" and a line an operator can act on.

    Trimmed to one line and 300 characters: this rides into the caller's log
    through an HTTP error detail, and a service having a bad day can produce a
    great deal of prose.
    """
    try:
        body = response.text.replace(upload_name, filename).strip().splitlines()
    except UnicodeDecodeError:
        return ""

    return f": {body[-1][:300]}" if body else ""
