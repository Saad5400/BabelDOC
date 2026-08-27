"""Environment-driven configuration for the doctranslate engine service."""

import os
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
JOB_TTL_HOURS = float(os.environ.get("JOB_TTL_HOURS", "24"))

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
