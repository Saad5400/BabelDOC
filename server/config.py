"""Environment-driven configuration for the doctranslate engine service."""

import os
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
JOB_TTL_HOURS = float(os.environ.get("JOB_TTL_HOURS", "24"))


def _positive(name: str, default: str, cast):
    """An env knob that is refused rather than obeyed when it is nonsense.

    A capacity limit read as 0 (or as "two") would not be a cautious engine,
    it would be one that never runs anything — the sort of typo that is only
    ever discovered in production, in the dark.
    """
    raw = os.environ.get(name, default).strip()
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        value = cast(default)
    return value if value > 0 else cast(default)


# How many HEAVY document operations may be in flight at once — translation
# jobs and the stateless builders together, in ONE global pool. See
# server/capacity.py for the measurements; the short version is that the engine
# runs under a 3 GiB cgroup, a single translation job costs up to 1.5 GB of
# peak RSS and a 73-page overlay 0.6 GB, so the arithmetic for the default is
#
#     620 MB floor + 1482 MB (worst job) + 593 MB (worst overlay) = 2695 MB
#
# which fits, and a third concurrent operation does not. Raise it only
# alongside the cgroup limit, and only with a measurement.
MAX_CONCURRENT_HEAVY = _positive("ENGINE_MAX_CONCURRENT", "2", int)

# How long a stateless rebuild waits for a slot before it is told the engine is
# busy (503 + Retry-After) instead of being left hanging.
#
# Sized against the CALLER's patience, not ours. Waiting past the client's own
# timeout buys nothing: the client has already given up, and the slot is then
# spent building a document nobody will read. catodemy's real budgets, READ
# from app/Services/DocTranslate/DocTranslateClient.php rather than assumed:
#
#     POST /v1/compose        60 s     POST /v1/overlay       120 s
#     POST /v1/strip-vocab    60 s     POST /v1/notes-space   120 s
#     POST /v1/convert       360 s (240 s interactive; nginx cuts it at 300 s)
#
# and NONE of those POSTs is retried — the file parts are streamed, so a
# client-side retry would re-send empty ones, which means every 503 the gate
# emits is a real error in front of a real reader. So the budget is not one
# number: it is the caller's own timeout, minus the time the build itself
# still needs after it is let in (MEASURED: ~4 s compose, ~1 s strip-vocab,
# ~2 s notes-space, up to ~32 s for a 73-page overlay), minus margin.
#
# The numbers are HALF the caller's budget, not most of it, and that is the
# one surprising thing here. A deadline is only as punctual as the event loop
# that fires it, and under two concurrent real overlays this loop is not
# punctual: MEASURED, a /healthz that normally answers in 8 ms took as long as
# 15 s, and a compose given a 40 s budget was refused after 63.6 s — 23.6 s
# late, and by then past the 60 s at which its caller had already given up. A
# 503 that arrives after the client has stopped listening is not an honest
# answer, it is a bare read error with extra steps. So each budget leaves room
# for that lateness: 30 s + ~24 s worst observed still lands inside 60 s, and
# 60 s + ~24 s inside 120 s.
#
# (The lateness itself is GIL contention between the two CPU-bound builders
# and the loop, it predates this gate — the same measurement on the base
# branch is worse, 30.8 s worst case — and it is not something a queue can
# fix. It is written down here because it is what sizes these numbers.)
#
# /v1/convert keeps the default despite its far longer client budget: its own
# render can take Gotenberg's full 300 s afterwards, and nginx cuts the
# synchronous path at 300 s regardless, so time spent queueing is time the
# render will not get.
#
# Two things are deliberately NOT subject to any of this:
#   * translation JOBS — durable on disk, polled rather than held open, so
#     they wait as long as it takes (server/capacity.py);
#   * /healthz and the job-status GETs — they never enter the gate at all.
#     catodemy retries a status poll exactly once, 200 ms later, and then
#     abandons a translation that is still running, so a status poll that
#     queued behind an overlay would throw away paid work.
ADMISSION_WAIT_SECONDS = _positive("ENGINE_ADMISSION_WAIT_SECONDS", "30",
                                   float)

#: For /v1/overlay and /v1/notes-space, whose caller allows 120 s.
LONG_ADMISSION_WAIT_SECONDS = _positive("ENGINE_LONG_ADMISSION_WAIT_SECONDS",
                                        "60", float)

# The shared Gotenberg service — the office suite /v1/convert renders through.
#
# Infra, so it is env: on prod the engine's container sits on the same docker
# network as gotenberg-int and reaches it by container name; a dev box runs its
# own on localhost (`docker run --rm -p 3000:3000
# gotenberg/gotenberg:8.36.0-libreoffice`), which is the default here so a
# fresh checkout converts without any env at all.
GOTENBERG_URL = os.environ.get("GOTENBERG_URL", "http://localhost:3000")

# Above Gotenberg's OWN 300 s conversion ceiling on purpose. Below it, a slow
# render would come back as a client timeout — the least informative failure
# available — instead of the service's own account of what went wrong.
GOTENBERG_TIMEOUT_SECONDS = 330.0

# Shared secret for the /v1 API. Empty/unset => all /v1 requests are refused.
DOCTRANSLATE_TOKEN = os.environ.get("DOCTRANSLATE_TOKEN", "")

# LLM endpoint (OpenAI-compatible; OpenRouter by default).
#
# gemini-3.1-flash-lite is the benchmarked pick (2026-08-25, 36/36 segments over
# the engine's own prompt): it is the only candidate besides 3.5-flash-lite that
# never drops a {v1} formula or <style id> run — the failure that corrupts a
# rebuilt PDF — and gemini-2.5-flash, which this replaces, lost style runs on
# 4/36. It is also ~33% cheaper per segment than 2.5-flash and reads better in
# Arabic ("علاقة التكرار", "مبرهنة الماستر" vs 3.5-flash-lite's "للالقائمة" and
# its translation of JVM). 2.5-flash-lite is 3x cheaper again but lost a
# placeholder on 9/36 — not a trade worth making on a deliverable a user paid for.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "google/gemini-3.1-flash-lite")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# No pricing knobs. Cost is whatever the provider says it was — see server/cost.py.
# The rate table that used to live here computed a number nobody could reconcile
# (and added a $0.001/page surcharge no provider charges), and that number went
# straight to a user's credit balance.

# «كلمات هذه الصفحة» — the per-page vocabulary layer: each page's NEW English
# words and short phrases, explained in Arabic on a compact page inserted right
# after it (first occurrence only). On by default; VOCAB_PAGES=0 (or
# false/no/off) kills the whole feature: the extraction pass after a mono run
# AND every insertion (mono result, /v1/overlay, /v1/compose). See
# server/vocab.py and server/vocab_pages.py.
VOCAB_PAGES = (os.environ.get("VOCAB_PAGES", "1").strip().lower()
               not in ("0", "false", "no", "off"))

GLOSSARY_PATH = SERVER_DIR / "glossary_ar_cs.csv"
OCR_PREP_SCRIPT = SERVER_DIR / "ocr_prep.py"
IMAGE_PREP_SCRIPT = SERVER_DIR / "image_prep.py"
FIX_LAYER_ORDER_SCRIPT = SERVER_DIR / "fix_layer_order.py"

VALID_FORMATS = ("translated", "alternating", "side_by_side")

# Every mono run also emits a translation SIDECAR (its translated text as
# data), from which layouts that need to know what the translation SAYS are
# rebuilt later for free — see server/interlinear.py. Off for native dual
# runs, whose IL boxes are the dual page's, not the original's.
SIDECAR_FORMATS = ("translated",)
