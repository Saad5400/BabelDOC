"""Disk-backed job store + single sequential background worker."""

import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

from server import config

logger = logging.getLogger("doctranslate.jobs")

_LOCK = threading.Lock()
_QUEUE: "queue.SimpleQueue[str]" = queue.SimpleQueue()
_WORKER: threading.Thread | None = None
_CURRENT_JOB_ID: str | None = None


def jobs_root() -> Path:
    return config.DATA_DIR / "jobs"


def job_dir(job_id: str) -> Path:
    return jobs_root() / job_id


def _job_json_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def read_job(job_id: str) -> dict | None:
    path = _job_json_path(job_id)
    with _LOCK:
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None


def write_job(job: dict) -> None:
    job["updated_at"] = time.time()
    path = _job_json_path(job["job_id"])
    with _LOCK:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job, indent=2))
        os.replace(tmp, path)


def update_job(job_id: str, **fields) -> dict | None:
    job = read_job(job_id)
    if job is None:
        return None
    job.update(fields)
    write_job(job)
    return job


def create_job(pdf_bytes: bytes, *, filename: str, lang_in: str, lang_out: str,
               fmt: str, title: str | None) -> dict:
    job_id = uuid.uuid4().hex[:16]
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "input.pdf").write_bytes(pdf_bytes)
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "filename": filename,
        "lang_in": lang_in,
        "lang_out": lang_out,
        "format": fmt,
        "title": title,
        "pages": None,
        "progress": {"percent": 0.0, "stage": "queued"},
        "usage": None,
        "error": None,
    }
    write_job(job)
    _QUEUE.put(job_id)
    return job


def result_path(job_id: str) -> Path:
    return job_dir(job_id) / "output.pdf"


def cleanup_expired() -> int:
    """Delete job dirs older than JOB_TTL_HOURS. Never touches the running job."""
    cutoff = time.time() - config.JOB_TTL_HOURS * 3600
    removed = 0
    root = jobs_root()
    if not root.is_dir():
        return 0
    for d in root.iterdir():
        if not d.is_dir() or d.name == _CURRENT_JOB_ID:
            continue
        try:
            mtime = _job_json_path(d.name).stat().st_mtime if _job_json_path(d.name).is_file() else d.stat().st_mtime
            if mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info("TTL cleanup removed %d job(s)", removed)
    return removed


def _worker_loop() -> None:
    global _CURRENT_JOB_ID
    # Imported here so the app can start (and /healthz answer) even while
    # heavy babeldoc imports/asset downloads are still warming up.
    from server import pipeline

    pipeline.warmup()
    while True:
        job_id = _QUEUE.get()
        job = read_job(job_id)
        if job is None or job["status"] != "queued":
            continue
        _CURRENT_JOB_ID = job_id
        try:
            update_job(job_id, status="running",
                       progress={"percent": 0.0, "stage": "starting"})
            pipeline.run_job(job_id)
        except Exception as exc:  # noqa: BLE001 - job isolation boundary
            logger.exception("job %s failed", job_id)
            update_job(job_id, status="failed", error=str(exc)[:2000])
        finally:
            _CURRENT_JOB_ID = None


def start_worker() -> None:
    global _WORKER
    if _WORKER is not None:
        return
    jobs_root().mkdir(parents=True, exist_ok=True)
    # Recover state from a previous process.
    for d in sorted(jobs_root().iterdir(), key=lambda p: p.stat().st_mtime):
        job = read_job(d.name)
        if job is None:
            continue
        if job["status"] == "running":
            update_job(job["job_id"], status="failed",
                       error="server restarted while job was running")
        elif job["status"] == "queued":
            _QUEUE.put(job["job_id"])
    _WORKER = threading.Thread(target=_worker_loop, name="doctranslate-worker",
                               daemon=True)
    _WORKER.start()
