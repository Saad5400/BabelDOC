"""Environment-driven configuration for the doctranslate engine service."""

import os
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
JOB_TTL_HOURS = float(os.environ.get("JOB_TTL_HOURS", "24"))

# Shared secret for the /v1 API. Empty/unset => all /v1 requests are refused.
DOCTRANSLATE_TOKEN = os.environ.get("DOCTRANSLATE_TOKEN", "")

# LLM endpoint (OpenAI-compatible; OpenRouter by default).
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "google/gemini-2.5-flash")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Pricing knobs (USD).
PROMPT_USD_PER_1M = float(os.environ.get("PROMPT_USD_PER_1M", "0.30"))
COMPLETION_USD_PER_1M = float(os.environ.get("COMPLETION_USD_PER_1M", "2.50"))
PAGE_OVERHEAD_USD = float(os.environ.get("PAGE_OVERHEAD_USD", "0.001"))

GLOSSARY_PATH = SERVER_DIR / "glossary_ar_cs.csv"
OCR_PREP_SCRIPT = SERVER_DIR / "ocr_prep.py"
FIX_LAYER_ORDER_SCRIPT = SERVER_DIR / "fix_layer_order.py"

VALID_FORMATS = ("translated", "alternating", "side_by_side")
