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
| `GET` | `/v1/jobs/{id}/sidecar` | token | The run's **translation sidecar** — its translated text as data (`409` until `done`, `404` when the run produced none). See below. |
| `POST` | `/v1/compose` | token | Stateless dual builder. Multipart `original` + `translated` PDFs, field `format` = `alternating` \| `side_by_side`. Returns the composed PDF. Pure page shuffling — no LLM, no job, no charge. |
| `POST` | `/v1/overlay` | token | Stateless **overlay** builder. Multipart `original` PDF + `sidecar` JSON, field `style` = `interlinear`, optional `scale`, `min_font_size`, `max_font_size`, `color` (hex), `align`, `plate_color` (hex), `plate_opacity`, `plate_padding`. Returns the overlaid PDF plus `X-Overlay-Pages` / `X-Overlay-Drawn` / `X-Overlay-Skipped` (and `X-Overlay-Raster-Drawn` / `X-Overlay-Raster-Skipped` for glosses inside embedded images). No LLM, no job, no charge. |
| `POST` | `/v1/convert` | token | Stateless office-to-PDF normaliser (LibreOffice). Multipart `file` (docx/pptx/…). Returns the PDF. |
| `GET` | `/healthz` | open | `200 {"status":"ok","versions":{babeldoc,tesseract,ocrmypdf,libreoffice}}`; `503 degraded` if a binary is missing. Deploy health check. |

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
- **Usage**: `cost_usd` is the **provider's own reported cost**, summed over
  every call of the job (`usage: {include: true}` on each request; see
  `server/cost.py`), never a locally computed figure. It is `null` when the
  endpoint reported no cost at all — the caller treats that as "charge nothing,
  log loudly", which is the right failure for money that cannot be
  substantiated. `generation_ids` lets a finished job be reconciled against the
  provider afterwards, and `priced_calls`/`calls` is the coverage of `cost_usd`,
  so a partially-priced run reads as partial rather than as cheap.
  Note: babeldoc caches translations — resubmitting identical content reports
  near-zero tokens and near-zero cost (that is real spend, not a bug).

## One translation, many layouts

A translation costs money; a layout does not. Two of the three ways to rebuild a
layout after the fact are already here — `/v1/compose` shuffles pages, and
`/v1/overlay` needs to know what the translation actually *says*, which is what
the sidecar is for.

- **Sidecar** (`babeldoc/format/pdf/document_il/midend/translation_sidecar.py`):
  every **mono** run writes `DATA_DIR/jobs/{id}/sidecar.json` — per page, the
  page frame; per paragraph, its box in the ORIGINAL page, its source text, its
  own source LINE boxes, the translation, and the source font size. It is
  captured in the one window where all of that is true at once: after
  `ILTranslator` (which overwrites the source text in place) and before
  `Typesetting` (which starts moving the boxes). Coordinates are unrotated PDF
  user space, so they map straight onto the untouched original.
  Fetch it with the result and keep it: it is what makes every later layout
  free. Native dual runs write none (their geometry is the dual page's).
- **`interlinear`** (`server/interlinear.py`): the original page untouched, with
  each paragraph's translation drawn small in the whitespace directly above it.
  Sized to the band it is given, spread down the paragraph's own source lines
  when one band cannot hold it legibly (which is what makes a slide's merged
  bullet list work), and SKIPPED rather than drawn over the reader's document
  when there is no room — `X-Overlay-Skipped` reports how often that happened.
  Arabic is shaped and bidi-resolved by PyMuPDF's Story engine; the gloss font
  is BabelDOC's own, subset per document (a deck would otherwise carry one
  15 MB font copy per gloss).
  Sidecar blocks marked `on_raster` (text the engine found INSIDE an embedded
  image, with the image's `region`) are glossed inside that image instead: on a
  rounded translucent plate in the band directly above the label, placed by
  reading the region's PIXELS — a plate may dim flat artwork but never cover a
  mark — falling back below the label, else skipped. Their fit comes back on
  `X-Overlay-Raster-Drawn` / `X-Overlay-Raster-Skipped`, and the plate is
  tunable via `plate_color` / `plate_opacity` / `plate_padding` options.

## Environment

| Env | Default | Meaning |
|---|---|---|
| `DOCTRANSLATE_TOKEN` | — (required) | Shared secret for `/v1` |
| `OPENAI_API_KEY` | — (required) | LLM key (OpenRouter) |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint |
| `OPENAI_MODEL` | `google/gemini-3.1-flash-lite` | Translation model |
| `DATA_DIR` | `/data` | Job storage (volume) |
| `JOB_TTL_HOURS` | `24` | Job retention |

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
