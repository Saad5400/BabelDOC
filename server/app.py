"""doctranslate engine — internal HTTP wrapper around the BabelDOC arabic-rtl fork.

Run: uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import hmac
import subprocess
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from server import config, jobs

MAX_UPLOAD_BYTES = 100 * 1024 * 1024


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
