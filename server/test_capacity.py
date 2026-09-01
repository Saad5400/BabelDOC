"""The global admission gate: how many documents may be in flight at once.

    pytest server/test_capacity.py

The defect these tests exist for is not a wrong answer, it is a dead process.
The engine runs under a 3 GiB cgroup; the job worker was sequential but the
five stateless builders were bounded by NOTHING, and (MEASURED on this branch,
peak RSS of the uvicorn process) one 73-page translation with six builders
alongside it reached 3618 MB — 546 MB past the cap the kernel enforces, which
is the OOM kill the sweep found in the kernel log.

So the assertions come in three layers, because each can pass while the next
fails:

*   the gate itself — a unit, no HTTP;
*   the handlers really enter it, and say 503 rather than hanging when they
    cannot — through `TestClient`;
*   and, under a real uvicorn with real concurrency, the builders are really
    bounded and `/healthz` and the job-status GETs really are not. A
    `TestClient` is synchronous and can never show either.
"""

import asyncio
import threading
import time

import httpx
import pytest

from server import capacity
from server import config
from server.conftest import TOKEN

# The live-server harness, and the fixtures that feed it, already exist for
# the event-loop tests next door. Reused rather than copied: a second uvicorn
# fixture would be a second thing to keep true.
from server.test_concurrency import _pdf
from server.test_concurrency import _slow

# Renamed on import so what the tests below ask for reads as what it is. An
# imported fixture is always a name a test parameter then shadows, hence the
# F811 waivers at the two tests that use it.
from server.test_concurrency import live_server as engine  # noqa: F401


@pytest.fixture(autouse=True)
def fresh_gate():
    """A gate with nobody in it, for every test in this file.

    The gate is process-global on purpose, so without this one test's leaked
    slot is the next test's hang — which reads exactly like the defect under
    test and is not it.
    """
    capacity.reset()
    yield
    capacity.reset()


@pytest.fixture()
def limit(monkeypatch):
    """Set the gate's limit for one test."""
    def _set(value: int):
        monkeypatch.setattr(config, "MAX_CONCURRENT_HEAVY", value)
    return _set


# --- the gate itself -------------------------------------------------------


@pytest.mark.anyio
async def test_no_more_than_the_limit_run_at_once(limit):
    limit(2)
    concurrent = 0
    high_water = 0

    async def op():
        nonlocal concurrent, high_water
        async with capacity.slot("op"):
            concurrent += 1
            high_water = max(high_water, concurrent)
            await asyncio.sleep(0.05)
            concurrent -= 1

    await asyncio.gather(*(op() for _ in range(8)))

    assert high_water == 2, (
        f"{high_water} operations were in flight at once against a limit of 2")
    assert capacity.stats() == {"limit": 2, "in_flight": 0, "queued": 0}


@pytest.mark.anyio
async def test_slots_are_granted_in_arrival_order(limit):
    """FIFO, so a wait is bounded by the queue rather than by luck.

    A plain counting semaphore would let a stream of cheap rebuilds walk past
    a paid translation for as long as the stream lasted.
    """
    limit(1)
    admitted: list[int] = []

    async def op(number: int):
        async with capacity.slot(f"op{number}"):
            admitted.append(number)
            await asyncio.sleep(0.02)

    first = asyncio.create_task(op(0))
    await asyncio.sleep(0.01)  # let it take the only slot
    rest = [asyncio.create_task(op(n)) for n in range(1, 5)]
    # Started in order, one event-loop tick apart, so arrival order is known.
    for _ in rest:
        await asyncio.sleep(0.005)
    await asyncio.gather(first, *rest)

    assert admitted == [0, 1, 2, 3, 4]


@pytest.mark.anyio
async def test_a_waiter_that_runs_out_of_patience_is_told_it_is_busy(
        limit):
    limit(1)

    async with capacity.slot("holder"):
        started = time.monotonic()
        with pytest.raises(capacity.BusyError) as caught:
            async with capacity.slot("waiter", wait_seconds=0.2):
                pytest.fail("a slot was granted while the only one was held")
        waited = time.monotonic() - started

    assert 0.2 <= waited < 2.0, waited
    assert caught.value.retry_after >= 30
    assert "busy" in str(caught.value)
    # And the failed waiter left no trace behind it.
    assert capacity.stats()["queued"] == 0


@pytest.mark.anyio
async def test_a_waiter_that_gives_up_releases_the_one_behind_it(
        limit):
    """The queue is FIFO, so an abandoned ticket has to be removed from it.

    Left in place it would be a permanent head that never takes its slot, and
    everything behind it would wait out its own deadline for nothing —
    one client hanging up would close the engine.
    """
    limit(1)
    admitted = asyncio.Event()

    async def patient():
        async with capacity.slot("patient", wait_seconds=10):
            admitted.set()

    async with capacity.slot("holder"):
        impatient = asyncio.create_task(
            _expect_busy(capacity.slot("impatient", wait_seconds=0.2)))
        await asyncio.sleep(0.02)
        behind = asyncio.create_task(patient())
        await asyncio.sleep(0.02)
        await impatient
        assert not admitted.is_set()

    await asyncio.wait_for(behind, timeout=5)
    assert admitted.is_set()


async def _expect_busy(manager):
    with pytest.raises(capacity.BusyError):
        async with manager:
            pytest.fail("a slot was granted that should not have been")


def test_a_translation_job_and_a_rebuild_share_one_pool(limit):
    """The job worker is a plain thread and the builders are on the loop.

    They still count against the same limit, because the memory ceiling they
    are both spending is one process's. A gate per endpoint would have let
    exactly the measured OOM happen: a translation at its peak plus every
    builder that happened to arrive.
    """
    limit(1)
    inside = threading.Event()
    finish = threading.Event()

    def job():
        with capacity.slot_sync("job"):
            inside.set()
            finish.wait(timeout=10)

    worker = threading.Thread(target=job)
    worker.start()
    try:
        assert inside.wait(timeout=5)
        assert capacity.stats()["in_flight"] == 1

        async def rebuild():
            with pytest.raises(capacity.BusyError):
                async with capacity.slot("overlay", wait_seconds=0.2):
                    pytest.fail("a rebuild ran while a translation held "
                                "the only slot")

        asyncio.run(rebuild())
    finally:
        finish.set()
        worker.join(timeout=10)

    assert capacity.stats()["in_flight"] == 0


def test_a_translation_job_waits_rather_than_being_refused(limit):
    """No deadline on the worker's door.

    A queued translation is already on disk and already polled for; refusing
    it would throw away work the reader paid for, so it waits however long the
    rebuilds ahead of it take.
    """
    limit(1)
    ran = threading.Event()

    async def hold_then_release():
        async with capacity.slot("overlay"):
            await asyncio.sleep(0.4)

    def job():
        with capacity.slot_sync("job"):
            ran.set()

    holder = threading.Thread(target=lambda: asyncio.run(hold_then_release()))
    holder.start()
    time.sleep(0.05)
    worker = threading.Thread(target=job)
    worker.start()
    worker.join(timeout=10)
    holder.join(timeout=10)

    assert ran.is_set(), "the translation was never let in"


def test_the_last_one_out_hands_the_memory_back(limit, monkeypatch):
    """glibc keeps every high-water mark unless it is told not to.

    MEASURED: one 73-page overlay takes the process from 421 MB to 1001 MB and
    LEAVES it there; `malloc_trim(0)` returns 464 MB of that. Without this the
    engine's floor climbs with every document it has ever seen, which is how
    documents that each fit inside the cap add up to an OOM.
    """
    limit(2)
    trims: list[int] = []
    monkeypatch.setattr(capacity, "trim", lambda: trims.append(1))

    async def two_then_none():
        first = capacity.slot("a")
        await first.__aenter__()
        second = capacity.slot("b")
        await second.__aenter__()
        await first.__aexit__(None, None, None)
        assert not trims, "trimmed while work was still in flight"
        await second.__aexit__(None, None, None)

    asyncio.run(two_then_none())
    assert trims == [1], "the last release did not trim"


def test_trim_is_harmless_where_there_is_no_malloc_trim(monkeypatch):
    monkeypatch.setattr(capacity, "_MALLOC_TRIM", None)
    capacity.trim()  # must not raise


def test_a_nonsense_limit_is_refused_rather_than_obeyed(monkeypatch):
    """`ENGINE_MAX_CONCURRENT=0` would not be a cautious engine.

    It would be one that queues every document for ever, and the only place
    anyone would find out is production.
    """
    monkeypatch.setenv("ENGINE_MAX_CONCURRENT", "0")
    assert config._positive("ENGINE_MAX_CONCURRENT", "2", int) == 2
    monkeypatch.setenv("ENGINE_MAX_CONCURRENT", "two")
    assert config._positive("ENGINE_MAX_CONCURRENT", "2", int) == 2
    monkeypatch.setenv("ENGINE_MAX_CONCURRENT", "4")
    assert config._positive("ENGINE_MAX_CONCURRENT", "2", int) == 4


def test_the_limit_never_drops_below_one(limit):
    limit(0)
    assert capacity.GATE.limit == 1


# --- the handlers really enter it ------------------------------------------


def test_a_busy_engine_says_so_with_a_retry_after(client, limit):
    """503, not a hang and not a 422.

    422 would tell catodemy the document is unacceptable — re-export it and
    try again — which is a lie about a file that is perfectly fine, and one
    the reader sees as "my file is broken".
    """
    limit(1)

    async def fill():
        manager = capacity.slot("holder")
        await manager.__aenter__()
        return manager

    loop = asyncio.new_event_loop()
    manager = loop.run_until_complete(fill())
    try:
        response = client.post(
            "/v1/notes-space",
            headers={"X-Internal-Token": TOKEN},
            files={"file": ("f.pdf", _pdf(), "application/pdf")},
            data={"sides": "bottom", "size": "md"},
        )
    finally:
        loop.run_until_complete(manager.__aexit__(None, None, None))
        loop.close()

    assert response.status_code == 503, response.text
    assert int(response.headers["Retry-After"]) >= 30
    assert "busy" in response.json()["detail"]


def test_a_typo_is_refused_at_the_door_rather_than_queued(client, limit):
    """Validation that needs no request body happens BEFORE the gate.

    A `sides` value that names no side is a broken request, and a broken
    request should not spend the engine's queue to be told so.
    """
    limit(1)

    async def fill():
        manager = capacity.slot("holder")
        await manager.__aenter__()
        return manager

    loop = asyncio.new_event_loop()
    manager = loop.run_until_complete(fill())
    try:
        response = client.post(
            "/v1/notes-space",
            headers={"X-Internal-Token": TOKEN},
            files={"file": ("f.pdf", _pdf(), "application/pdf")},
            data={"sides": "diagonally", "size": "md"},
        )
    finally:
        loop.run_until_complete(manager.__aexit__(None, None, None))
        loop.close()

    assert response.status_code == 422, response.text


def test_healthz_reports_the_queue_without_entering_it(client, limit):
    limit(1)

    async def fill():
        manager = capacity.slot("holder")
        await manager.__aenter__()
        return manager

    loop = asyncio.new_event_loop()
    manager = loop.run_until_complete(fill())
    try:
        response = client.get("/healthz")
    finally:
        loop.run_until_complete(manager.__aexit__(None, None, None))
        loop.close()

    body = response.json()
    assert body["capacity"] == {"limit": 1, "in_flight": 1, "queued": 0}
    # A busy engine is not a broken one. Reporting "degraded" here would get
    # the container restarted, and a restart fails the paid translation the
    # engine was busy WITH.
    assert body["status"] == "ok"


# --- and under real concurrency --------------------------------------------


def _fire(base_url, path, files, form, out: list, timeout=180.0):
    headers = {"X-Internal-Token": TOKEN}
    start = time.monotonic()
    response = httpx.post(base_url + path, files=files, data=form,
                          headers=headers, timeout=timeout)
    out.append((response.status_code, start - _fire.t0,
                time.monotonic() - _fire.t0))


def test_four_concurrent_builds_are_bounded_to_the_limit(engine,  # noqa: F811
                                                         monkeypatch):
    """The test that would have caught an unbounded builder.

    Four notes-space requests, each stubbed to take BUILD_SECONDS, against a
    limit of 2: two run, two wait, and the wall clock is two rounds rather
    than one. Before the gate all four ran side by side — which is precisely
    the pile-up the kernel killed the process for, and every existing test
    passed while it did, because `TestClient` sends one request at a time.
    """
    import server

    monkeypatch.setattr(config, "MAX_CONCURRENT_HEAVY", 2)
    monkeypatch.setattr(config, "ADMISSION_WAIT_SECONDS", 120.0)
    monkeypatch.setattr(server.notes_space, "add_notes_space",
                        _slow(b"%PDF-1.7\nstub\n%%EOF\n"))

    from server.test_concurrency import BUILD_SECONDS

    results: list = []
    _fire.t0 = time.monotonic()
    threads = [threading.Thread(
        target=_fire,
        args=(engine, "/v1/notes-space",
              {"file": ("f.pdf", _pdf(), "application/pdf")},
              {"sides": "bottom", "size": "md"}, results))
        for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)
    wall = time.monotonic() - _fire.t0

    assert len(results) == 4, results
    assert {status for status, _, _ in results} == {200}, results
    assert wall >= BUILD_SECONDS * 1.8, (
        f"four {BUILD_SECONDS}s builds finished in {wall:.1f}s — more than "
        f"two of them ran at once, so nothing is bounding them")
    # ...and not so serialised that the limit is being ignored downward.
    assert wall < BUILD_SECONDS * 3.5, (
        f"four builds took {wall:.1f}s against a limit of 2 — they were "
        f"serialised rather than run two at a time")


def test_healthz_and_job_status_answer_while_the_gate_is_saturated(
        engine, monkeypatch, tmp_path):  # noqa: F811
    """The two endpoints that must never queue.

    /healthz is what the platform restarts the container on, and the job-status
    GET is what catodemy polls — and catodemy retries a failed poll exactly
    once, 200 ms later, before abandoning a translation that is still running.
    Either of them waiting behind a build turns a busy engine into a dead one.
    """
    import server
    from server import jobs

    monkeypatch.setattr(config, "MAX_CONCURRENT_HEAVY", 2)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server.notes_space, "add_notes_space",
                        _slow(b"%PDF-1.7\nstub\n%%EOF\n"))

    jobs.jobs_root().mkdir(parents=True, exist_ok=True)
    job = jobs.create_job(b"%PDF-1.7\n", filename="d.pdf", lang_in="en",
                          lang_out="ar", fmt="translated", title=None)

    results: list = []
    _fire.t0 = time.monotonic()
    threads = [threading.Thread(
        target=_fire,
        args=(engine, "/v1/notes-space",
              {"file": ("f.pdf", _pdf(), "application/pdf")},
              {"sides": "bottom", "size": "md"}, results))
        for _ in range(4)]
    for thread in threads:
        thread.start()

    headers = {"X-Internal-Token": TOKEN}
    health: list[float] = []
    status: list[float] = []
    try:
        time.sleep(0.3)
        while any(thread.is_alive() for thread in threads) and len(health) < 8:
            for probe, url, hdrs in (
                    (health, engine + "/healthz", {}),
                    (status, f"{engine}/v1/jobs/{job['job_id']}",
                     headers)):
                start = time.monotonic()
                response = httpx.get(url, headers=hdrs, timeout=30.0)
                assert response.status_code == 200, response.text
                probe.append(time.monotonic() - start)
    finally:
        for thread in threads:
            thread.join(timeout=180)

    assert health and status
    # `fix/engine-api` measured /healthz at a 1.09 s worst case under two
    # concurrent overlays. The gate must not make that worse — and with four
    # requests in and two of them queueing, it does not.
    assert max(health) < 1.5, f"/healthz worst case {max(health):.2f}s"
    assert max(status) < 1.5, f"job status worst case {max(status):.2f}s"
