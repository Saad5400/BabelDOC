# doctranslate engine

Internal HTTP wrapper around this BabelDOC fork (`arabic-rtl`): layout-preserving
PDF translation (EN→AR primary), including the scanned-PDF OCR pipeline. Built to
sit behind catodemy as an arms-length service (AGPL boundary: this whole repo,
wrapper included, is public).

## Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/v1/jobs` | token | Submit a PDF. Multipart: `file` (pdf, ≤100 MB); fields `lang_in` (default `en`), `lang_out` (default `ar`), `format` = `translated` \| `alternating` \| `side_by_side` (default `translated`), optional `title` (stamped into the result's PDF metadata). Returns `202 {"job_id", "status": "queued"}`. |
| `GET` | `/v1/jobs/{id}` | token | `{"status": "queued|running|done|failed", "progress": {"percent", "stage"}, "pages", "format", "usage": {...} (when done), "error" (when failed)}` |
| `GET` | `/v1/jobs/{id}/result` | token | The output PDF (`409` until `done`). |
| `GET` | `/healthz` | open | `200 {"status":"ok","versions":{babeldoc,tesseract,ocrmypdf}}`; `503 degraded` if a binary is missing. Deploy health check. |

Auth: header `X-Internal-Token` must equal env `DOCTRANSLATE_TOKEN`
(`401` otherwise; `503` if the env is unset — the service refuses to run open).

Jobs run sequentially on a single background worker (bounded memory/CPU); state
lives as JSON under `DATA_DIR/jobs/{id}/` beside the input/output PDFs. Jobs older
than `JOB_TTL_HOURS` are deleted on each submit.

## Pipeline

- **Digital PDFs** → babeldoc directly (no-watermark, CS glossary
  `server/glossary_ar_cs.csv`, no auto glossary extraction).
- **Scanned / garbage-text-layer PDFs** (detected via pypdf text extraction over
  the first pages) → `ocrmypdf --force-ocr -l eng` → `ocr_prep.py` (sandwich
  rebuild) → babeldoc `--skip-scanned-detection --ocr-workaround` →
  `fix_layer_order.py` (mono output only; a no-op safety net after `ocr_prep.py`).
- **Formats**: `translated` = babeldoc mono; `alternating` and `side_by_side` =
  babeldoc's two native dual modes.
- **Usage**: real token counters from the translator;
  `cost_usd = prompt·PROMPT_USD_PER_1M/1e6 + completion·COMPLETION_USD_PER_1M/1e6 + pages·PAGE_OVERHEAD_USD`.
  Note: babeldoc caches translations — resubmitting identical content reports
  near-zero tokens (that is real spend, not a bug).

## Environment

| Env | Default | Meaning |
|---|---|---|
| `DOCTRANSLATE_TOKEN` | — (required) | Shared secret for `/v1` |
| `OPENAI_API_KEY` | — (required) | LLM key (OpenRouter) |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint |
| `OPENAI_MODEL` | `google/gemini-2.5-flash` | Translation model |
| `DATA_DIR` | `/data` | Job storage (volume) |
| `JOB_TTL_HOURS` | `24` | Job retention |
| `PROMPT_USD_PER_1M` | `0.30` | Pricing knob |
| `COMPLETION_USD_PER_1M` | `2.50` | Pricing knob |
| `PAGE_OVERHEAD_USD` | `0.001` | Per-page overhead in `cost_usd` |

## Run locally

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e . -r server/requirements.txt   # from the repo root
DOCTRANSLATE_TOKEN=dev OPENAI_API_KEY=... uvicorn server.app:app --port 8000
```

Requires `tesseract` (+eng) and ghostscript on PATH for the scanned path.
First run downloads babeldoc assets (fonts + onnx layout model) to
`~/.cache/babeldoc`.

## Docker / Coolify

- **Build context: the repo root**, dockerfile `server/Dockerfile`
  (`docker build -f server/Dockerfile .`). The image bakes tesseract,
  ghostscript, fonts and the babeldoc assets, so boot needs no downloads.
- Port `8000`; persistent volume on `/data`; health check `GET /healthz`.
- Coolify: new app from this repo/branch, Build Pack = Dockerfile,
  "Dockerfile Location" = `/server/Dockerfile`, base directory `/`. Keep it on
  the internal network only (no public domain) and set `DOCTRANSLATE_TOKEN` +
  `OPENAI_API_KEY` in both this app and catodemy.
