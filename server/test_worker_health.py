"""A dead job worker must fail /healthz, not hide behind a serving uvicorn.

bitstring 5.0 once made babeldoc's PDF writer unimportable: the worker thread
died in `pipeline.init()` while every endpoint, /healthz included, kept
answering "ok", and no translation would ever run.
"""

import threading

import pytest

from server import app as app_module
from server import jobs
from server import pipeline


@pytest.fixture()
def healthy_binaries(monkeypatch):
    """/healthz's other checks pass, so the worker is the only variable."""
    monkeypatch.setattr(app_module, "_cmd_version", lambda _argv: "1.0")

    async def gotenberg_up():
        return "8.36.0"

    monkeypatch.setattr(app_module.convert, "service_status", gotenberg_up)


def _run_worker_until_it_stops(monkeypatch) -> None:
    """The real jobs._worker_loop on a thread registered as THE worker."""
    def run():
        try:
            jobs._worker_loop()
        except BaseException:  # noqa: BLE001 - the thread dying is the point
            pass

    worker = threading.Thread(target=run, daemon=True)
    monkeypatch.setattr(jobs, "_WORKER", worker)
    monkeypatch.setattr(jobs, "_WORKER_ERROR", None)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()


def test_no_worker_started_is_not_a_problem(monkeypatch):
    monkeypatch.setattr(jobs, "_WORKER", None)
    assert jobs.worker_problem() is None


def test_a_live_worker_is_not_a_problem(monkeypatch):
    stop = threading.Event()
    worker = threading.Thread(target=stop.wait, daemon=True)
    worker.start()
    monkeypatch.setattr(jobs, "_WORKER", worker)
    try:
        assert jobs.worker_problem() is None
    finally:
        stop.set()


def test_a_worker_whose_init_fails_records_why(monkeypatch):
    def broken_init():
        raise ImportError("cannot import name 'BitStream' from 'bitstring'")

    monkeypatch.setattr(pipeline, "init", broken_init)
    _run_worker_until_it_stops(monkeypatch)

    problem = jobs.worker_problem()
    assert problem.startswith("worker init failed: ImportError")
    assert "BitStream" in problem


def test_a_dead_worker_without_a_recorded_reason_is_still_a_problem(monkeypatch):
    worker = threading.Thread(target=lambda: None)
    worker.start()
    worker.join()
    monkeypatch.setattr(jobs, "_WORKER", worker)
    monkeypatch.setattr(jobs, "_WORKER_ERROR", None)

    assert jobs.worker_problem() == "job worker thread is not running"


@pytest.mark.usefixtures("healthy_binaries")
def test_healthz_fails_while_the_worker_is_dead(client, monkeypatch):
    monkeypatch.setattr(jobs, "_WORKER", None)
    healthy = client.get("/healthz")
    assert healthy.status_code == 200
    assert healthy.json()["worker"] == "running"

    def broken_init():
        raise ImportError("cannot import name 'BitStream' from 'bitstring'")

    monkeypatch.setattr(pipeline, "init", broken_init)
    _run_worker_until_it_stops(monkeypatch)

    dead = client.get("/healthz")
    assert dead.status_code == 503
    assert dead.json()["status"] == "degraded"
    assert "BitStream" in dead.json()["worker"]
