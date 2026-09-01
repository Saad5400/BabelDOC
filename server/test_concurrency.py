"""The event loop must stay free while a stateless builder is working.

Run from the repo root (pyproject's testpaths only covers tests/, so pass
the path explicitly):

    pytest server/test_concurrency.py

This file exists because every other test in `server/` drives the app through
FastAPI's `TestClient`, which is synchronous: one request at a time, each
waited out before the next. That can never show the defect these tests are
about. A handler that does seconds of CPU-bound work inside the event loop
answers a `TestClient` perfectly well and stalls every OTHER request in the
process for the whole time.

The production consequence is not a slow download. `/healthz` is what the
orchestrator polls; a health check that times out gets the container
restarted, and a restart marks the running job "server restarted while job
was running" — so ONE reader's free layout rebuild kills a DIFFERENT reader's
paid translation. Commit 9ee2d14 diagnosed exactly that and fixed it for
`/v1/convert` alone; the other four handlers went on doing it and the suite
stayed green, because nothing in it could see it.

So: the real app, under a real uvicorn, on an ephemeral port, with a slow
build genuinely in flight — and then the question the orchestrator asks.

The builders are STUBBED with a deliberately slow function rather than fed a
huge document. What is under test is whether the handler hands its work to a
thread, and that is a property of the handler, not of how expensive a fixture
happens to be — a blank 600-page PDF composes in 0.13 s (MEASURED), far too
fast to tell a blocked loop from a busy one. The stub sleeps, which is what
the real work does to the GIL: pymupdf releases it in C.
"""

import json
import socket
import threading
import time
from contextlib import closing
from io import BytesIO

import httpx
import pytest
import uvicorn
from pypdf import PdfWriter

from server.conftest import TOKEN

PAGE = (595.0, 842.0)
BUILD_SECONDS = 3.0     # how long the stubbed build takes
HEALTHZ_DEADLINE = 1.5  # a health check answers in this, or the loop is blocked


def _free_port() -> int:
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _pdf(pages: int = 2) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=PAGE[0], height=PAGE[1])
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def _sidecar_bytes(pages: int = 2) -> bytes:
    """A sidecar that PASSES the identity check, so the build is reached."""
    return json.dumps({
        "version": 1, "lang_in": "en", "lang_out": "ar",
        "total_pages": pages,
        "pages": [{"page_number": number,
                   "mediabox": [0.0, 0.0, *PAGE],
                   "blocks": [], "obstacles": []}
                  for number in range(pages)],
    }).encode()


@pytest.fixture(scope="module")
def live_server():
    """The real app under a real uvicorn, on its own port, in a thread."""
    from server import app as app_module
    from server import config

    port = _free_port()
    config.DOCTRANSLATE_TOKEN = TOKEN
    app_module.jobs.start_worker = lambda: None

    server = uvicorn.Server(uvicorn.Config(app_module.app, host="127.0.0.1",
                                           port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            httpx.get(f"{base}/healthz", timeout=2.0)
            break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:  # pragma: no cover - the server failing to start is a broken env
        pytest.fail("the test server never came up")

    yield base

    server.should_exit = True
    thread.join(timeout=10)


def _slow(result):
    """A builder that takes its time and returns what the handler expects."""
    def build(*_args, **_kwargs):
        time.sleep(BUILD_SECONDS)
        return result
    return build


# (module, attribute, stub return value, path, files, form)
BUILDERS = {
    "compose": ("compose", "compose_dual", b"%PDF-1.7\nstub\n%%EOF\n",
                "/v1/compose",
                lambda: {"original": ("o.pdf", _pdf(), "application/pdf"),
                         "translated": ("t.pdf", _pdf(), "application/pdf")},
                {"format": "alternating"}),
    "overlay": ("interlinear", "render_overlay",
                (b"%PDF-1.7\nstub\n%%EOF\n",
                 {"pages": 2, "drawn": 1, "skipped": 0,
                  "raster_drawn": 0, "raster_skipped": 0}),
                "/v1/overlay",
                lambda: {"original": ("o.pdf", _pdf(), "application/pdf"),
                         "sidecar": ("s.json", _sidecar_bytes(),
                                     "application/json")},
                {"style": "interlinear"}),
    "strip-vocab": ("compose", "strip_vocab", b"%PDF-1.7\nstub\n%%EOF\n",
                    "/v1/strip-vocab",
                    lambda: {"translated": ("t.pdf", _pdf(),
                                            "application/pdf"),
                             "sidecar": ("s.json", _sidecar_bytes(),
                                         "application/json")},
                    {}),
    "notes-space": ("notes_space", "add_notes_space",
                    b"%PDF-1.7\nstub\n%%EOF\n", "/v1/notes-space",
                    lambda: {"file": ("f.pdf", _pdf(), "application/pdf")},
                    {"sides": "bottom", "size": "md"}),
}


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_a_build_in_flight_does_not_stall_the_health_check(live_server,
                                                           monkeypatch, kind):
    """/healthz answers WHILE a build is running, not after it.

    Fails on all four before the handlers were moved off the loop: the health
    check does not come back until the build has finished.
    """
    import server

    module_name, attribute, result, path, files, form = BUILDERS[kind]
    monkeypatch.setattr(getattr(server, module_name), attribute, _slow(result))

    headers = {"X-Internal-Token": TOKEN}
    outcome: dict = {}

    def build():
        start = time.monotonic()
        response = httpx.post(live_server + path, files=files(), data=form,
                              headers=headers, timeout=120.0)
        outcome["status"] = response.status_code
        outcome["seconds"] = time.monotonic() - start

    worker = threading.Thread(target=build)
    worker.start()
    try:
        time.sleep(0.3)  # let the build reach the handler

        probes = []
        while worker.is_alive() and len(probes) < 4:
            start = time.monotonic()
            httpx.get(live_server + "/healthz",
                      timeout=BUILD_SECONDS + HEALTHZ_DEADLINE + 5)
            probes.append(time.monotonic() - start)
    finally:
        worker.join(timeout=120)

    assert outcome.get("status") == 200, outcome
    # The stub really did hold the handler open, so the probes above really
    # were taken while a build was in flight.
    assert outcome["seconds"] >= BUILD_SECONDS, outcome
    assert probes, "the build finished before a single health check was tried"
    assert max(probes) < HEALTHZ_DEADLINE, (
        f"/healthz took {max(probes):.2f}s while a {kind} build was in flight "
        f"(the build itself took {outcome['seconds']:.2f}s) — the handler is "
        f"holding the event loop")


def test_two_builds_run_at_once_rather_than_one_after_the_other(live_server,
                                                                monkeypatch):
    """Two concurrent builds overlap.

    The health-check assertion above would also pass a server that answered
    /healthz promptly but still serialised its builders. Off the loop they
    run side by side, so two BUILD_SECONDS builds finish in well under two
    of them.
    """
    import server

    monkeypatch.setattr(server.notes_space, "add_notes_space",
                        _slow(b"%PDF-1.7\nstub\n%%EOF\n"))

    headers = {"X-Internal-Token": TOKEN}
    elapsed: list[float] = []

    def build():
        start = time.monotonic()
        httpx.post(live_server + "/v1/notes-space",
                   files={"file": ("f.pdf", _pdf(), "application/pdf")},
                   data={"sides": "bottom", "size": "md"},
                   headers=headers, timeout=120.0)
        elapsed.append(time.monotonic() - start)

    started = time.monotonic()
    workers = [threading.Thread(target=build) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=120)
    wall = time.monotonic() - started

    assert len(elapsed) == 2
    assert wall < BUILD_SECONDS * 1.8, (
        f"two {BUILD_SECONDS}s builds took {wall:.2f}s of wall clock — they "
        f"were serialised, not run side by side")
