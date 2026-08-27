"""doctranslate engine — internal HTTP wrapper around the BabelDOC arabic-rtl fork.

Run: uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import dataclasses
import hmac
import json
import shutil
import subprocess
from contextlib import asynccontextmanager

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

MAX_UPLOAD_BYTES = 100 * 1024 * 1024


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
def healthz():
    from babeldoc.const import __version__ as babeldoc_version

    versions = {
        "babeldoc": babeldoc_version,
        "tesseract": _cmd_version(["tesseract", "--version"]),
        "ocrmypdf": _cmd_version(["ocrmypdf", "--version"]),
        # Presence, not `--version`: soffice builds its user profile on the
        # first run, which can outlast a health check's patience — and a slow
        # answer here would report the whole engine degraded over a binary the
        # translation path does not even use.
        "libreoffice": shutil.which("soffice") or "missing",
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

    sidecar_data = None
    if sidecar is not None:
        sidecar_bytes = await sidecar.read()
        if len(sidecar_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="sidecar too large")
        try:
            sidecar_data = json.loads(sidecar_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422,
                                detail=f"sidecar is not valid JSON: {exc}") from exc

    try:
        result = compose.compose_dual(parts["original"], parts["translated"],
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

    try:
        parsed = json.loads(sidecar_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422,
                            detail=f"sidecar is not valid JSON: {exc}") from exc

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
        result, report = interlinear.render_overlay(original_bytes, parsed,
                                                    style=style, options=options,
                                                    vocab=want_vocab)
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
        pdf_bytes = convert.office_to_pdf(data, filename)
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

    try:
        sidecar_data = json.loads(sidecar_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422,
                            detail=f"sidecar is not valid JSON: {exc}") from exc

    try:
        result = compose.strip_vocab(translated_bytes, sidecar_data)
    except compose.ComposeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(content=result, media_type="application/pdf",
                    headers=_pdf_attachment(translated.filename))


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
