"""doctranslate engine — internal HTTP wrapper around the BabelDOC arabic-rtl fork.

Run: uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import dataclasses
import functools
import hmac
import json
import subprocess
from contextlib import asynccontextmanager
from io import BytesIO

import anyio.to_thread
import pymupdf
from fastapi import Depends
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import Header
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import Response

from server import compose
from server import config
from server import convert
from server import interlinear
from server import jobs
from server import notes_space

MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# How far a page may differ from the size the sidecar recorded for it and
# still count as the same page. A point of slack absorbs the round trip
# through JSON and PDF number formatting; it is far tighter than the gap
# between any two real documents. MEASURED across the 14 production runs the
# sweep collected: every correct (document, sidecar) pair agrees exactly.
GEOMETRY_TOLERANCE = 1.0


def _blocking(func, /, *args, **kwargs):
    """Run a CPU-bound builder off the event loop.

    Every stateless builder here is synchronous and spends seconds to a minute
    inside pymupdf. Called directly from an `async def` handler it owns the
    event loop for that whole time: no other request is even READ, so a free
    layout download stalls every job poll and every health check in the
    process — and an orchestrator that gets no answer from /healthz restarts
    the container, which fails whatever paid translation was running
    (server/jobs.py marks it "server restarted while job was running").

    server/convert.py's docstring diagnosed exactly this for /v1/convert and
    fixed it there; this is the same fix for the other four. The handlers stay
    `async def` because each has to `await` its uploads first, so they cannot
    take Starlette's threadpool for free the way a plain `def` endpoint does —
    they hand the CPU work over explicitly instead.
    """
    return anyio.to_thread.run_sync(functools.partial(func, *args, **kwargs))


def _open_pdf(data: bytes, part: str) -> pymupdf.Document:
    """The uploaded bytes as a document, or a 422 saying why not.

    One place, so every endpoint refuses an unreadable upload the same way.
    Two things go wrong here and only one of them raises: a file pymupdf
    cannot parse throws at open, while a PASSWORD-PROTECTED one opens
    happily — `page_count` is even correct — and only explodes on the first
    page access, deep inside a builder, as a 500. `needs_pass` is the missing
    half, and a password-protected upload is the ordinary case: a student
    hands over the locked PDF the university published.

    The caller owns the returned document and must close it.
    """
    try:
        doc = pymupdf.open(stream=BytesIO(data), filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - pymupdf raises many error types
        raise HTTPException(
            status_code=422,
            detail=f"{part} is not a readable PDF: {exc}") from exc

    if doc.needs_pass:
        doc.close()
        raise HTTPException(
            status_code=422,
            detail=f"{part} is password-protected; it has to be unlocked first")

    if doc.page_count == 0:
        doc.close()
        raise HTTPException(status_code=422, detail=f"{part} has no pages")

    return doc


def _scrub_surrogates(value):
    """A parsed sidecar with its unpaired surrogates dropped.

    A sidecar's `target` strings are LLM output that was serialised to JSON
    and stored. An LLM that emits half an emoji puts a lone `\\ud83d` in
    there, `json.loads` accepts it without complaint (it pairs well-formed
    surrogates into real characters and leaves the odd one alone), and then
    the first `.encode("utf-8")` anywhere downstream raises — so ONE bad
    character means every future free rebuild of that run is a 500, for ever,
    with no way back short of paying for the translation again.

    Dropped rather than replaced: what is left is exactly the text minus a
    character that was never a character.
    """
    if isinstance(value, str):
        return "".join(char for char in value
                       if not 0xD800 <= ord(char) <= 0xDFFF)
    if isinstance(value, list):
        return [_scrub_surrogates(item) for item in value]
    if isinstance(value, dict):
        return {_scrub_surrogates(key): _scrub_surrogates(item)
                for key, item in value.items()}
    return value


def _parse_sidecar(raw: bytes, part: str = "sidecar"):
    """The sidecar part as data: JSON, then scrubbed of lone surrogates.

    The scrub walks the whole structure (MEASURED: 7 ms on run30's 158 KB
    sidecar, 11 ms on run20's 322 KB), so it is skipped unless the raw bytes
    could possibly carry a surrogate. There are exactly two ways one reaches
    `json.loads`, and each leaves its own fingerprint in the bytes:

    * a `\\uD…` escape — the ensure_ascii spelling;
    * a 0xED lead byte — `json.loads` decodes a BYTES argument with
      `surrogatepass`, so the three-byte form sails straight through and
      comes back out as a lone surrogate. That is the one that bites: it does
      not look like an encoding error to anything upstream.

    Neither fingerprint occurs in any of the 14 production sidecars the sweep
    collected, so in practice this costs one substring scan.
    """
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422,
                            detail=f"{part} is not valid JSON: {exc}") from exc

    if b"\\ud" in raw.lower() or b"\xed" in raw:
        return _scrub_surrogates(parsed)

    return parsed


def _sidecar_page_count(sidecar: dict) -> int | None:
    """How many pages the run had, per the sidecar's own record of itself."""
    count = sidecar.get("total_pages")

    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        return count

    pages = sidecar.get("pages")

    return len(pages) if isinstance(pages, list) and pages else None


def _sidecar_page_sizes(sidecar: dict) -> dict[int, tuple[float, float]]:
    """(width, height) per page number, for the entries that record a box.

    Uploader-supplied, so an entry that is not a four-number mediabox is
    simply not a size we can check against — dropped, not fatal.
    """
    sizes: dict[int, tuple[float, float]] = {}

    for entry in sidecar.get("pages") or ():
        if not isinstance(entry, dict):
            continue

        number = entry.get("page_number")
        box = entry.get("mediabox")

        if not isinstance(number, int) or isinstance(number, bool):
            continue
        if not isinstance(box, list) or len(box) != 4:
            continue

        try:
            x0, y0, x1, y1 = (float(value) for value in box)
        except (TypeError, ValueError):
            continue

        sizes[number] = (abs(x1 - x0), abs(y1 - y0))

    return sizes


def _content_positions(sidecar: dict) -> list[int] | None:
    """The baked mono's content-page positions, or None for "not recorded".

    Reads `artifact_layout.content_pages` by exactly the rule compose.py
    reads it by (strictly increasing non-negative ints, else the layout is
    junk and the mono is treated as pre-vocab), so the identity check below
    and the strip that follows it agree on what the sidecar says.
    """
    layout = sidecar.get("artifact_layout")
    positions = layout.get("content_pages") if isinstance(layout, dict) else None

    if not isinstance(positions, list) or not positions:
        return None

    for index, position in enumerate(positions):
        if not isinstance(position, int) or isinstance(position, bool):
            return None
        if position < 0 or (index and position <= positions[index - 1]):
            return None

    return positions


def _strip_heights(sidecar: dict) -> dict[int, float]:
    """Per-content-page baked vocab strip heights, read compose.py's way."""
    layout = sidecar.get("artifact_layout")
    strips = layout.get("vocab_strips") if isinstance(layout, dict) else None
    out: dict[int, float] = {}

    if not isinstance(strips, dict):
        return out

    for key, value in strips.items():
        try:
            index, height = int(key), float(value)
        except (TypeError, ValueError):
            continue

        if index >= 0 and 0 < height <= 14400:  # NaN fails both comparisons
            out[index] = height

    return out


def _reject_foreign_sidecar(detail: str):
    raise HTTPException(
        status_code=422,
        detail=f"{detail} — the sidecar and the PDF are not from the same run")


def _require_a_sidecar(sidecar) -> None:
    """Refuse a sidecar that is not demonstrably FOR this document.

    A sidecar and a PDF arrive as two independent uploads, and nothing in
    either one says which run it came from. Get the pairing wrong and the
    builders do not fail — they succeed on the wrong document: a foreign
    sidecar strips a 25-page translation down to four cropped pages, or draws
    another course's glosses over this one, and answers 200 with a
    plausible-looking PDF that the caller then caches as the user's download.

    The pairing is checked the only way the data allows: the run's page count
    and every page's size have to be the document's own. That is not a proof
    of identity, but a mismatch is conclusive, and MEASURED across the 14
    production runs the sweep collected, every correct pair agrees exactly
    and every mismatched pair is caught.

    The gap it cannot close, said plainly: at strip-vocab, a sidecar that is
    a strict PREFIX of this document and whose pages are the same size. A
    baked mono is content pages plus vocabulary pages plus an appendix tail,
    so pages the layout does not name are EXPECTED — "fewer pages than the
    file" is a normal reading of a correct sidecar, and nothing in either
    artifact says which run it came from. Closing it needs a run identifier
    written into both; that is a change to what the pipeline emits, not
    something this boundary can derive.
    """
    if not isinstance(sidecar, dict):
        raise HTTPException(status_code=422,
                            detail="sidecar is not a translation sidecar")

    if _sidecar_page_count(sidecar) is None:
        raise HTTPException(
            status_code=422,
            detail="sidecar records no pages, so it cannot be checked "
                   "against the PDF it was sent with")


def _require_sidecar_matches_original(sidecar, doc: pymupdf.Document) -> dict:
    """The overlay's check: the sidecar describes THIS original, page for page.

    interlinear.py already refuses a sidecar that names a page the original
    does not have, but that guard is one-sided: a SHORTER sidecar from another
    run names only pages that exist, so it sails through and its paragraphs
    are drawn onto a document they have nothing to do with.
    """
    _require_a_sidecar(sidecar)
    declared = _sidecar_page_count(sidecar)

    if declared != doc.page_count:
        _reject_foreign_sidecar(
            f"the sidecar is for a {declared}-page document and the original "
            f"has {doc.page_count}")

    for number, (width, height) in _sidecar_page_sizes(sidecar).items():
        if not 0 <= number < doc.page_count:
            continue  # interlinear reports this one, in its own words

        box = doc[number].mediabox

        if (abs(box.width - width) > GEOMETRY_TOLERANCE
                or abs(box.height - height) > GEOMETRY_TOLERANCE):
            _reject_foreign_sidecar(
                f"the sidecar's page {number + 1} is {width:.0f}x{height:.0f} "
                f"and the original's is {box.width:.0f}x{box.height:.0f}")

    return sidecar


def _require_sidecar_matches_mono(sidecar, doc: pymupdf.Document) -> dict:
    """The strip's check: the sidecar's layout describes THIS baked mono.

    strip-vocab acts on positions the sidecar hands it — keep these pages,
    crop this many points off those. Against the wrong document those are
    instructions to delete and crop a stranger, carried out silently: 25 pages
    in, four cropped ones out, HTTP 200.

    Two shapes to check. A mono with a recorded `artifact_layout` says where
    its content pages sit and how tall a vocab strip each carries, so each of
    those pages must be its sidecar page's size PLUS its strip. A pre-vocab
    mono records no layout, and is simply the translation page for page,
    possibly with a baked appendix tail after it.
    """
    _require_a_sidecar(sidecar)
    declared = _sidecar_page_count(sidecar)
    positions = _content_positions(sidecar)
    strips: dict[int, float] = {}

    if positions is None:
        if declared > doc.page_count:
            _reject_foreign_sidecar(
                f"the sidecar is for a {declared}-page translation and the "
                f"file has {doc.page_count} pages")
        positions = list(range(declared))
    else:
        if len(positions) != declared:
            _reject_foreign_sidecar(
                f"the sidecar's layout names {len(positions)} content pages "
                f"for a {declared}-page translation")
        if positions[-1] >= doc.page_count:
            _reject_foreign_sidecar(
                f"the sidecar's layout puts content on page "
                f"{positions[-1] + 1} and the file has {doc.page_count} pages")
        strips = _strip_heights(sidecar)

    sizes = _sidecar_page_sizes(sidecar)

    for index, position in enumerate(positions):
        size = sizes.get(index)

        if size is None:
            continue

        width, height = size
        box = doc[position].mediabox
        expected = height + strips.get(index, 0.0)

        if (abs(box.width - width) > GEOMETRY_TOLERANCE
                or abs(box.height - expected) > GEOMETRY_TOLERANCE):
            _reject_foreign_sidecar(
                f"the sidecar's page {index + 1} is {width:.0f}x{expected:.0f} "
                f"and the file's is {box.width:.0f}x{box.height:.0f}")

    return sidecar


def _parse_flag(value: str, name: str) -> bool:
    """A form field carrying a boolean: "1"/"0" (or "true"/"false").

    The Laravel caller sends "1"/"0"; the tolerant spellings cost nothing.
    Anything else is a typo'd request, refused rather than guessed at.
    """
    lowered = (value or "").strip().lower()

    if lowered in ("1", "true"):
        return True
    if lowered in ("0", "false"):
        return False

    raise HTTPException(status_code=422,
                        detail=f"{name} must be 1 or 0")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    jobs.start_worker()
    yield


app = FastAPI(title="doctranslate engine", lifespan=lifespan)


def require_token(x_internal_token: str | None = Header(default=None)) -> None:
    if not config.DOCTRANSLATE_TOKEN:
        raise HTTPException(status_code=503,
                            detail="DOCTRANSLATE_TOKEN is not configured")
    if not x_internal_token or not hmac.compare_digest(
            x_internal_token, config.DOCTRANSLATE_TOKEN):
        raise HTTPException(status_code=401, detail="invalid X-Internal-Token")


def _cmd_version(argv: list[str]) -> str:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        first = (out.stdout or out.stderr).strip().splitlines()
        return first[0] if first else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "missing"


@app.get("/healthz")
async def healthz():
    from babeldoc.const import __version__ as babeldoc_version

    # The office suite is no longer a binary in this image but a service on the
    # network, so its line is a REACHABILITY probe rather than a `which`. It
    # reports Gotenberg's own libreoffice component: a Gotenberg that answers
    # while its LibreOffice is down would convert nothing.
    #
    # Async because that probe is: the version calls are subprocesses and would
    # block the loop for as long as they run, so they go to a worker thread —
    # which is what an ordinary `def` endpoint got for free and this one has to
    # ask for.
    versions = {
        "babeldoc": babeldoc_version,
        "tesseract": await anyio.to_thread.run_sync(
            _cmd_version, ["tesseract", "--version"]),
        "ocrmypdf": await anyio.to_thread.run_sync(
            _cmd_version, ["ocrmypdf", "--version"]),
        "gotenberg": await convert.service_status(),
    }
    ok = all(v != "missing" for v in versions.values())
    return JSONResponse(status_code=200 if ok else 503,
                        content={"status": "ok" if ok else "degraded",
                                 "versions": versions})


@app.post("/v1/jobs", status_code=202, dependencies=[Depends(require_token)])
async def create_job(
    file: UploadFile = File(...),
    lang_in: str = Form("en"),
    lang_out: str = Form("ar"),
    format: str = Form("translated"),
    title: str | None = Form(None),
):
    if format not in config.VALID_FORMATS:
        raise HTTPException(status_code=422,
                            detail=f"format must be one of {config.VALID_FORMATS}")
    pdf_bytes = await file.read()
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="file is not a PDF")

    jobs.cleanup_expired()
    job = jobs.create_job(pdf_bytes, filename=file.filename or "document.pdf",
                          lang_in=lang_in, lang_out=lang_out, fmt=format,
                          title=title)
    return {"job_id": job["job_id"], "status": job["status"]}


def _pdf_attachment(filename: str | None, suffix: str = "") -> dict[str, str]:
    """The `Content-Disposition` for a PDF built out of `filename`.

    Shared by the three stateless builders, which differ only in what they add
    to the stem. The header value is written raw, so it has to stay latin-1: a
    stem that is not plain ASCII (or that carries a quote of its own) is
    REPLACED rather than escaped. Nothing depends on it — every caller stores
    the bytes under its own name — so this only decides what a browser save or
    a curl would call the file.
    """
    stem = (filename or "document.pdf").rsplit(".", 1)[0]

    if not stem.isascii() or '"' in stem:
        stem = "document"

    name = ".".join(part for part in (stem, suffix, "pdf") if part)

    return {"Content-Disposition": f'attachment; filename="{name}"'}


@app.post("/v1/compose", dependencies=[Depends(require_token)])
async def compose_dual(
    original: UploadFile = File(...),
    translated: UploadFile = File(...),
    format: str = Form(...),
    sidecar: UploadFile | None = File(None),
    vocab: str = Form("1"),
):
    """Stateless dual builder: original + translated (mono) in, dual PDF out.

    `sidecar` is optional: send the run's sidecar JSON and, when it carries
    the "vocab" entries the mono run selected, each page's «كلمات هذه الصفحة»
    page is rendered into the dual right after that page's pair (and the
    translated input's own baked-in vocab pages are taken back out via the
    sidecar's artifact_layout, so nothing appears twice). Without it,
    behavior is exactly as before.

    `vocab=0` opts this download out of the vocab layer: the baked-in vocab
    still comes back out (that is undoing what the input carries, not adding
    a feature), but no fresh strips go in.
    """
    want_vocab = _parse_flag(vocab, "vocab")
    if format not in compose.COMPOSE_FORMATS:
        raise HTTPException(status_code=422,
                            detail=f"format must be one of {compose.COMPOSE_FORMATS}")
    parts = {"original": await original.read(),
             "translated": await translated.read()}
    for name, pdf_bytes in parts.items():
        if len(pdf_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{name} too large")
        if not pdf_bytes.startswith(b"%PDF-"):
            raise HTTPException(status_code=422, detail=f"{name} is not a PDF")
        # compose reached 422 for a locked PDF only because pymupdf happened
        # to throw later, with "File has not been decrypted" — true, and no
        # use to a reader deciding what to do about it. Opened here for the
        # same reason and in the same words as everywhere else.
        _open_pdf(pdf_bytes, name).close()

    sidecar_data = None
    if sidecar is not None:
        sidecar_bytes = await sidecar.read()
        if len(sidecar_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="sidecar too large")
        sidecar_data = _parse_sidecar(sidecar_bytes)

    try:
        result = await _blocking(compose.compose_dual,
                                 parts["original"], parts["translated"],
                                 format, sidecar=sidecar_data,
                                 vocab=want_vocab)
    except compose.ComposeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(content=result, media_type="application/pdf",
                    headers=_pdf_attachment(original.filename, format))


@app.post("/v1/overlay", dependencies=[Depends(require_token)])
async def overlay(
    original: UploadFile = File(...),
    sidecar: UploadFile = File(...),
    style: str = Form("interlinear"),
    # Unset means "whatever this style is tuned for" — the two layouts want
    # different type sizes and clearances (interlinear.OverlayOptions.defaults),
    # so a shared literal default here would quietly impose one on the other.
    scale: float | None = Form(None),
    min_font_size: float | None = Form(None),
    max_font_size: float | None = Form(None),
    gap: float | None = Form(None),
    color: str | None = Form(None),
    align: str | None = Form(None),
    plate_color: str | None = Form(None),
    plate_opacity: float | None = Form(None),
    plate_padding: float | None = Form(None),
    vocab: str = Form("1"),
):
    """Stateless layout builder for the layouts that need the translated TEXT.

    /v1/compose rebuilds the duals by shuffling two finished PDFs; this rebuilds
    the ones that have to know what each paragraph says and where it belongs,
    from the run's sidecar. Like compose, it is stateless, LLM-free and
    therefore free: one paid translation, any number of layouts.

    `vocab=0` opts this render out of the «كلمات هذه الصفحة» strips the
    sidecar's "vocab" would otherwise add to the overlay pages.
    """
    want_vocab = _parse_flag(vocab, "vocab")
    if style not in interlinear.OVERLAY_STYLES:
        raise HTTPException(
            status_code=422,
            detail=f"style must be one of {interlinear.OVERLAY_STYLES}")

    original_bytes = await original.read()
    sidecar_bytes = await sidecar.read()

    for name, data in (("original", original_bytes), ("sidecar", sidecar_bytes)):
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{name} too large")

    if not original_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="original is not a PDF")

    parsed = _parse_sidecar(sidecar_bytes)

    # Opened HERE, once, rather than inside whichever style was asked for:
    # the two styles open it independently and only one of them guarded the
    # call, so the same unopenable upload was a clean 422 through
    # interlinear_compact and a 500 through interlinear. Refusing it at the
    # door also gives the identity check below the page sizes it needs.
    source = _open_pdf(original_bytes, "original")
    try:
        _require_sidecar_matches_original(parsed, source)
    finally:
        source.close()

    try:
        options = dataclasses.replace(
            interlinear.OverlayOptions.defaults(style),
            **{key: value for key, value in
               (("scale", scale), ("min_font_size", min_font_size),
                ("max_font_size", max_font_size), ("gap", gap),
                ("color", color), ("align", align),
                ("plate_color", plate_color),
                ("plate_opacity", plate_opacity),
                ("plate_padding", plate_padding))
               if value is not None})
        result, report = await _blocking(interlinear.render_overlay,
                                         original_bytes, parsed, style=style,
                                         options=options, vocab=want_vocab)
    except interlinear.OverlayError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(
        content=result, media_type="application/pdf",
        headers={
            **_pdf_attachment(original.filename, style),
            # The fit is not all-or-nothing: a caller that sees most of a
            # document skipped knows the layout did not suit it, instead of
            # shipping a near-empty overlay as a success.
            "X-Overlay-Pages": str(report["pages"]),
            "X-Overlay-Drawn": str(report["drawn"]),
            "X-Overlay-Skipped": str(report["skipped"]),
            # The raster lane's fit, on its own headers: a deck whose diagrams
            # all came back unglossed should say so even when every ordinary
            # paragraph fitted.
            "X-Overlay-Raster-Drawn": str(report["raster_drawn"]),
            "X-Overlay-Raster-Skipped": str(report["raster_skipped"]),
        })


@app.post("/v1/convert", dependencies=[Depends(require_token)])
async def convert_to_pdf(file: UploadFile = File(...)):
    """Stateless normaliser: an office document in, its PDF rendering out.

    The translation pipeline is PDF-only, and so is the dual-format compose
    that pairs an original with its translation — so a .pptx has to become a
    PDF before anything else can touch it. The caller keeps what comes back and
    uses it as the original from that point on.

    The render itself happens on the shared Gotenberg service (see
    server/convert.py), awaited rather than forked: this handler used to run a
    blocking soffice inside the event loop and hold up every other request —
    a job poll, a health check — for the length of a deck's conversion.
    """
    filename = file.filename or "document"

    if not convert.is_convertible(filename):
        raise HTTPException(
            status_code=422,
            detail=f"cannot convert this file type; supported: "
                   f"{', '.join(convert.CONVERTIBLE_EXTENSIONS)}")

    data = await file.read()

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large")

    try:
        pdf_bytes = await convert.office_to_pdf(data, filename)
    # Order matters: the unavailable case is a ConvertError too, and it is the
    # one that must NOT come back as 422. A 422 tells the caller the document
    # is unacceptable — re-export it and try again — which is a lie when the
    # conversion service was merely busy or down, and it is a lie the user
    # reads as "my file is broken".
    except convert.ConvertUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except convert.ConvertError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers=_pdf_attachment(filename))


@app.post("/v1/strip-vocab", dependencies=[Depends(require_token)])
async def strip_vocab(
    translated: UploadFile = File(...),
    sidecar: UploadFile = File(...),
):
    """Stateless un-baker: the stored mono in, the mono WITHOUT its vocab out.

    The pipeline bakes the «كلمات هذه الصفحة» layer into the mono it stores;
    a reader who wants the translation clean gets it back here, exactly, via
    the sidecar's artifact_layout — baked bottom strips are physically
    redacted off and inserted fallback pages dropped (compose.strip_vocab, the
    same undo /v1/compose performs before pairing). A sidecar without a usable
    layout means a pre-vocab mono: nothing to strip, the bytes come back
    unchanged.
    """
    translated_bytes = await translated.read()
    sidecar_bytes = await sidecar.read()

    for name, data in (("translated", translated_bytes),
                       ("sidecar", sidecar_bytes)):
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{name} too large")

    if not translated_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="translated is not a PDF")

    sidecar_data = _parse_sidecar(sidecar_bytes)

    mono = _open_pdf(translated_bytes, "translated")
    try:
        _require_sidecar_matches_mono(sidecar_data, mono)
    finally:
        mono.close()

    try:
        result = await _blocking(compose.strip_vocab,
                                 translated_bytes, sidecar_data)
    except compose.ComposeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(content=result, media_type="application/pdf",
                    headers=_pdf_attachment(translated.filename))


@app.post("/v1/notes-space", dependencies=[Depends(require_token)])
async def add_notes_space(
    file: UploadFile = File(...),
    sides: str = Form(...),
    size: str = Form("md"),
):
    """Stateless margin builder: any PDF in, the same PDF with ruled note
    space out.

    Every page grows a band of blank, faintly-ruled writing space (أسطر) on
    each requested side — `sides` is a comma-separated subset of top, bottom,
    left, right; `size` is sm/md/lg. The content never moves (the mediabox
    grows outward, vocab-strip style), so this composes with whatever the PDF
    already is: a mono with baked strips, a dual, an overlay. Rotated pages
    are left untouched.
    """
    data = await file.read()

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large")

    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="file is not a PDF")

    try:
        result = await _blocking(notes_space.add_notes_space, data,
                                 notes_space.parse_sides(sides), size=size)
    except notes_space.NotesSpaceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(content=result, media_type="application/pdf",
                    headers=_pdf_attachment(file.filename, "notes"))


def _get_job_or_404(job_id: str) -> dict:
    job = jobs.read_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_token)])
def get_job(job_id: str):
    job = _get_job_or_404(job_id)
    body = {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "pages": job["pages"],
        "format": job["format"],
    }
    if job["status"] == "done":
        body["usage"] = job["usage"]
    if job["status"] == "failed":
        body["error"] = job["error"]
    # The source-language analysis, when the run has got far enough to have
    # one. This body is an explicit allowlist rather than the job record, so
    # a field the pipeline starts recording is invisible to the caller until
    # it is named here — and this one is the difference between a reader
    # being told their Arabic document was already Arabic and their being
    # charged for a degraded Arabic-to-Arabic copy of it. Conditional because
    # a job from before the analysis existed simply has none.
    if job.get("language") is not None:
        body["language"] = job["language"]
    return body


@app.get("/v1/jobs/{job_id}/sidecar", dependencies=[Depends(require_token)])
def get_sidecar(job_id: str):
    """The run's translated text as data — the input to /v1/overlay.

    404 rather than an empty document when a run produced none (a native dual
    run, or one from before sidecars existed): the caller has to be able to
    tell "no sidecar" from "a sidecar with nothing in it", because only the
    first means the layouts built from it are simply unavailable for this run.
    """
    job = _get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409,
                            detail=f"job is {job['status']}, not done")
    path = jobs.sidecar_path(job_id)
    if not path.is_file():
        raise HTTPException(status_code=404,
                            detail="this run produced no translation sidecar")
    return FileResponse(path, media_type="application/json",
                        filename=f"{job_id}.sidecar.json")


@app.get("/v1/jobs/{job_id}/result", dependencies=[Depends(require_token)])
def get_result(job_id: str):
    job = _get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409,
                            detail=f"job is {job['status']}, not done")
    path = jobs.result_path(job_id)
    if not path.is_file():
        raise HTTPException(status_code=409, detail="result no longer available")
    stem = (job.get("filename") or "document.pdf").rsplit(".", 1)[0]
    return FileResponse(path, media_type="application/pdf",
                        filename=f"{stem}.{job['lang_out']}.pdf")
