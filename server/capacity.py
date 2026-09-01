"""One global admission gate in front of every heavy document operation.

The engine dies of memory, not of CPU. It runs under a 3 GiB cgroup and on
2026-09-01 the kernel killed it there:

    uvicorn invoked oom-killer
    Memory cgroup out of memory: Killed process 1777219 (uvicorn)
        anon-rss:3127868kB   constraint=CONSTRAINT_MEMCG

Five of the seven production failures that week are that kill. Raising the cap
is not available — the host has 7.7 GB total and ~2.3 GB free — so the only
lever left is how much work is allowed to be in flight at once.

MEASURED on this branch, against the 14-document production corpus, as peak
RSS of the uvicorn process (`/proc/<pid>/status` VmRSS, sampled at 4 Hz):

    resident floor, model loaded, nothing running        620 MB
    one 17-page SCANNED job, through the OCR lane      +1884 MB
    one 41-page translation job, on its own            +1482 MB
    one 73-page translation job, on its own            +1088 MB
    one 73-page interlinear overlay                     +593 MB
    one 25-page alternating compose                     +169 MB
    one 39-page side-by-side compose                     +82 MB
    one 41-page notes-space, four sides, lg              +24 MB
    one 73-page strip-vocab                              +20 MB

The scanned document is the expensive one, and by a margin — page count is not
the cost, raster payload is. It also has a second peak nothing else has: while
`ocrmypdf --force-ocr` runs it forks 17 workers, and the process TREE reaches
1692 MB while uvicorn itself is still at 625 MB. Those workers are in the same
cgroup, so they spend the same 3 GiB. It happens early, before babeldoc's own
peak, which is the only reason the two do not add up.

Nothing bounded the second column. The job worker has always been sequential —
one translation at a time — but the five stateless builders were not bounded at
ALL, and `fix/engine-api` (this branch's base) has just moved all of them onto
threads so they no longer block the event loop. That fix was necessary and it
raised the ceiling: four concurrent overlays alone took the process from 620 MB
to 2044 MB, and one translation job with six builders alongside it peaked at
**3618 MB** — 546 MB past the cap the kernel actually enforces. That run is the
production crash, reproduced.

So: one gate, `ENGINE_MAX_CONCURRENT` slots, in front of the translation worker
AND all five builders. Not one gate per endpoint — the memory ceiling is a
property of the process, so the bound has to be too.

Three properties this has to have, and each is a decision:

*   **A caller that cannot get a slot queues; it does not fail.** The engine
    being busy is not the caller's document being bad. Past
    `ENGINE_ADMISSION_WAIT_SECONDS` it gets a 503 and a `Retry-After`, which is
    honest and which catodemy already retries; what it must never get is a hang
    with no answer.

*   **FIFO, so waiting is bounded by the queue and not by luck.** A plain
    counting semaphore lets a stream of cheap builders starve the paid
    translation behind them indefinitely. Every caller takes a ticket and slots
    are granted from the head, so the job that has been waiting longest goes
    next and a 503 means "there are genuinely N of these ahead of you".

*   **`/healthz` and the job-status GETs never touch this gate.** They are how
    catodemy and the platform's health check learn the engine is alive; a health
    check that queues behind a 60-second overlay is how a busy engine becomes a
    restarted one, and a restart fails whatever paid translation was running.

The gate is also where the process gives memory BACK. glibc does not return a
freed arena to the OS on its own, and the measurement is stark: an overlay that
takes RSS from 421 MB to 1001 MB leaves it at 1001 MB, and `malloc_trim(0)`
brings it to 537 MB — 464 MB of the 580 MB was allocator slack, not live data.
So every release trims; see `Gate.release` for why "only when the gate falls
idle" is not enough.
"""

import asyncio
import collections
import contextlib
import ctypes
import gc
import logging
import threading
import time

from server import config

logger = logging.getLogger("doctranslate.capacity")

try:
    _LIBC = ctypes.CDLL("libc.so.6")
    _MALLOC_TRIM = _LIBC.malloc_trim
except (OSError, AttributeError):  # pragma: no cover - non-glibc platform
    _MALLOC_TRIM = None


class BusyError(RuntimeError):
    """No slot came free inside the caller's deadline.

    Carries `retry_after` so the handler can say how long to wait rather than
    leaving the caller to guess, and `queued`/`limit` so the reason is in the
    message a human eventually reads.
    """

    def __init__(self, label: str, waited: float, queued: int, limit: int,
                 retry_after: int):
        self.label = label
        self.waited = waited
        self.queued = queued
        self.limit = limit
        self.retry_after = retry_after
        super().__init__(
            f"the engine is busy: {limit} document operation(s) are already "
            f"running and {queued} more are queued ahead of this one, which "
            f"waited {waited:.1f}s for a slot. Retry in {retry_after}s.")


class _Ticket:
    """One caller's place in the queue.

    `event` and `loop` are set only for an async waiter: a release signals it
    through `loop.call_soon_threadsafe`, because releases happen on whichever
    thread finished — the job worker, or one of anyio's — and never on the
    event loop.
    """

    __slots__ = ("label", "event", "loop")

    def __init__(self, label: str, event=None, loop=None):
        self.label = label
        self.event = event
        self.loop = loop


class Gate:
    """The admission gate. One instance, `GATE`, is the global one."""

    def __init__(self, limit: int | None = None):
        self._lock = threading.Lock()
        self._free = threading.Condition(self._lock)
        self._queue: collections.deque[_Ticket] = collections.deque()
        self._in_flight = 0
        self._limit_override = limit

    @property
    def limit(self) -> int:
        """Read through to config every time, so a test (or an operator
        editing the env and restarting) does not have to rebuild the gate.
        Floored at 1: a limit of 0 would not be a cautious engine, it would be
        an engine that refuses every document."""
        if self._limit_override is not None:
            return max(1, self._limit_override)
        return max(1, config.MAX_CONCURRENT_HEAVY)

    def stats(self) -> dict:
        """What the gate is doing right now — for /healthz, which must not
        queue to find out."""
        with self._lock:
            return {"limit": self.limit, "in_flight": self._in_flight,
                    "queued": len(self._queue)}

    # -- internals ---------------------------------------------------------

    def _grant_locked(self, ticket: _Ticket) -> bool:
        """Take a slot if one is free AND this ticket is at the head."""
        if self._in_flight >= self.limit:
            return False
        if not self._queue or self._queue[0] is not ticket:
            return False
        self._queue.popleft()
        self._in_flight += 1
        return True

    def _wake_locked(self) -> list[_Ticket]:
        """Wake every waiter: the head of the queue may have changed, so who
        can proceed is not knowable from here. Sync waiters re-check under the
        condition; async ones are signalled by the caller, outside the lock."""
        self._free.notify_all()
        return [t for t in self._queue if t.event is not None]

    @staticmethod
    def _signal(tickets: list[_Ticket]) -> None:
        for ticket in tickets:
            try:
                ticket.loop.call_soon_threadsafe(ticket.event.set)
            except RuntimeError:  # pragma: no cover - loop already closed
                pass

    # -- acquire / release -------------------------------------------------

    def acquire(self, label: str) -> None:
        """Take a slot, waiting as long as it takes. The job worker's door.

        No deadline on purpose: a queued translation is already durable on
        disk, its caller polls for status rather than holding a connection
        open, and refusing it would throw away work someone paid for. A
        stateless rebuild is the opposite on every count, which is why the
        async door below has a deadline.
        """
        ticket = _Ticket(label)
        with self._lock:
            self._queue.append(ticket)
            try:
                while not self._grant_locked(ticket):
                    self._free.wait()
            except BaseException:
                with contextlib.suppress(ValueError):
                    self._queue.remove(ticket)
                # Signalling under the lock is safe: `call_soon_threadsafe`
                # only appends to the loop's ready queue and pokes its
                # self-pipe, and takes nothing this thread could be holding.
                self._signal(self._wake_locked())
                raise

    async def acquire_async(self, label: str,
                            wait_seconds: float | None = None) -> None:
        """Take a slot without blocking the event loop, or raise `BusyError`.

        The wait is a real await: a request queueing here holds no thread, so
        a hundred of them cost the process a coroutine each and nothing else.
        That matters — waiting inside anyio's worker pool instead would burn a
        thread per waiter and, past the pool's limit, would stall the plain
        `def` endpoints (the job-status GETs) that share it.
        """
        if wait_seconds is None:
            wait_seconds = config.ADMISSION_WAIT_SECONDS
        started = time.monotonic()
        deadline = started + wait_seconds
        ticket = _Ticket(label, asyncio.Event(), asyncio.get_running_loop())

        with self._lock:
            self._queue.append(ticket)
            if self._grant_locked(ticket):
                return
            queued, in_flight = len(self._queue), self._in_flight

        logger.info("%s is waiting for a slot (%d in flight, %d queued)",
                    label, in_flight, queued)

        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._busy(label, started)
                try:
                    await asyncio.wait_for(ticket.event.wait(), remaining)
                except (asyncio.TimeoutError, TimeoutError):
                    raise self._busy(label, started) from None
                ticket.event.clear()
                with self._lock:
                    if self._grant_locked(ticket):
                        return
        except BaseException:
            # Timed out, or the client hung up and the handler was cancelled.
            # Either way this ticket leaves the queue and whoever is behind it
            # is now the head, so they have to be told.
            with self._lock:
                with contextlib.suppress(ValueError):
                    self._queue.remove(ticket)
                woken = self._wake_locked()
            self._signal(woken)
            raise

    def _busy(self, label: str, started: float) -> BusyError:
        with self._lock:
            queued = max(0, len(self._queue) - 1)
            in_flight = self._in_flight
        waited = time.monotonic() - started
        # Long enough that a retry is not simply the same 503 again: the
        # cheapest thing ahead of the caller is a compose, the dearest a
        # translation, so half the wait it already spent is a better guess
        # than any constant, and it never asks for less than 30s.
        retry_after = max(30, int(waited / 2))
        logger.warning("%s gave up after %.1fs waiting for a slot "
                       "(%d in flight, %d queued)", label, waited,
                       in_flight, queued)
        return BusyError(label, waited, queued, self.limit, retry_after)

    def release(self, label: str) -> None:
        """Give the slot back, and give the MEMORY back with it.

        Every release trims, not only the last one out. Trimming only when the
        gate fell idle was the first shape of this and it measured badly: a
        translation job holds its slot for minutes, so while one is running the
        gate is never idle, and the slack left behind by each finished rebuild
        simply accumulated underneath it. On the job-plus-six-rebuilds run,
        trimming on idle alone peaked at 2957 MB; trimming on every release
        peaked at 2265 and 2275 MB over two runs, and ENDED at 973 MB rather
        than 2853 MB (MEASURED, same documents, same order).

        It costs 20-250 ms, against operations measured in seconds, and it is
        safe with work still in flight — `malloc_trim` takes the arena locks
        rather than needing the process to be quiet.
        """
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            woken = self._wake_locked()
        self._signal(woken)
        trim()

    # -- context managers --------------------------------------------------

    @contextlib.contextmanager
    def slot_sync(self, label: str):
        self.acquire(label)
        try:
            yield
        finally:
            self.release(label)

    @contextlib.asynccontextmanager
    async def slot(self, label: str, wait_seconds: float | None = None):
        await self.acquire_async(label, wait_seconds)
        try:
            yield
        finally:
            self.release(label)


def trim() -> None:
    """Hand the allocator's free arenas back to the OS.

    Without this the process keeps every high-water mark it ever reached: an
    overlay peaks at 1001 MB and STAYS at 1001 MB, so the next document starts
    from there and the one after that from higher still — which is how a
    3 GiB cap is reached by documents that individually fit inside it many
    times over. MEASURED: 464 MB returned after one 73-page overlay, 171 MB
    after loading the layout model, at a cost of 20-250 ms.

    Best-effort: not glibc, or a libc without `malloc_trim`, simply means the
    `gc.collect()` and nothing more.
    """
    gc.collect()
    if _MALLOC_TRIM is None:  # pragma: no cover - non-glibc platform
        return
    try:
        _MALLOC_TRIM(0)
    except Exception:  # noqa: BLE001 - pragma: no cover - never worth raising
        logger.debug("malloc_trim failed", exc_info=True)


#: The gate. One per process, because the memory ceiling is one per process.
GATE = Gate()


# Thin module-level doors rather than aliases bound to the instance: they look
# up `GATE` at CALL time, so `reset()` below really does replace the gate the
# handlers use. An alias captured at import would leave every caller talking to
# the old one, which is the kind of thing that makes a test suite pass while
# testing nothing.
def slot(label: str, wait_seconds: float | None = None):
    return GATE.slot(label, wait_seconds)


def slot_sync(label: str):
    return GATE.slot_sync(label)


def stats() -> dict:
    return GATE.stats()


def reset() -> None:
    """A fresh gate. For tests, which must not inherit each other's counters."""
    global GATE
    GATE = Gate()
