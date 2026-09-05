# doctranslate engine

Internal HTTP wrapper around this BabelDOC fork (`arabic-rtl`): layout-preserving
PDF translation (EN→AR primary), including the scanned-PDF OCR pipeline. Built to
sit behind catodemy as an arms-length service (AGPL boundary: this whole repo,
wrapper included, is public).

## Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/v1/jobs` | token | Submit a PDF. Multipart: `file` (pdf, ≤100 MB); fields `lang_in` (default `en`), `lang_out` (default `ar`), `format` = `translated` \| `alternating` \| `side_by_side` (default `translated`), optional `title` (stamped into the result's PDF metadata). Returns `202 {"job_id", "status": "queued"}`. |
| `GET` | `/v1/jobs/{id}` | token | `{"status": "queued|running|done|failed", "progress": {"percent", "stage"}, "pages", "format", "usage": {...} (when done), "error" + "coverage" (when failed)}` |
| `GET` | `/v1/jobs/{id}/result` | token | The output PDF (`409` until `done`). |
| `GET` | `/v1/jobs/{id}/sidecar` | token | The run's **translation sidecar** — its translated text as data (`409` until `done`, `404` when the run produced none). See below. |
| `POST` | `/v1/compose` | token | Stateless dual builder. Multipart `original` + `translated` PDFs, field `format` = `alternating` \| `side_by_side`, optional `sidecar` JSON (see **Vocab pages**). Returns the composed PDF. Pure page shuffling (plus the vocab render when the sidecar carries one) — no LLM, no job, no charge. |
| `POST` | `/v1/overlay` | token | Stateless **overlay** builder. Multipart `original` PDF + `sidecar` JSON, field `style` = `interlinear` \| `interlinear_compact`, optional `scale`, `min_font_size`, `max_font_size`, `gap`, `color` (hex), `align`, `plate_color` (hex), `plate_opacity`, `plate_padding` (each defaults to the tuning of the style asked for). Returns the overlaid PDF plus `X-Overlay-Pages` / `X-Overlay-Drawn` / `X-Overlay-Skipped` (and `X-Overlay-Raster-Drawn` / `X-Overlay-Raster-Skipped` for glosses inside embedded images). No LLM, no job, no charge. |
| `POST` | `/v1/convert` | token | Stateless office-to-PDF normaliser, rendered by the shared **Gotenberg** service (`GOTENBERG_URL`). Multipart `file` (docx/pptx/…). Returns the PDF. `422` when the document cannot be rendered (the service's own reason rides along); **`503` when the conversion service is busy or unreachable** — that one is not the document's fault and must not be shown as one. |
| `GET` | `/healthz` | open | `200 {"status":"ok","versions":{babeldoc,tesseract,ocrmypdf,gotenberg}}`; `503 degraded` if a binary is missing or the conversion service is unreachable. The `gotenberg` line is a live probe of that service's own `libreoffice` component, not a `which` — it is no longer a binary in this image. Deploy health check. |

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
- **Coverage**: `usage` also carries `paragraphs_total` / `paragraphs_translated`
  / `paragraphs_untranslated` — babeldoc's own count, not a second one — so the
  caller can see what it is charging for. A run that leaves more than
  `MAX_UNTRANSLATED_RATIO` of them untranslated is `failed`
  (`translation incomplete: N of M paragraphs untranslated`) with no `usage` at
  all, and a failed job reports how far it got under `coverage`. The partial PDF
  stays on disk for debugging; `/result` still `409`s.
- **Hard provider errors** (401/402/403, or a 429 that outlives the retries)
  cancel the run on the FIRST one and fail the job as
  `provider: <what the provider said> (key limit exceeded, HTTP 403)`. Before
  this, a spent key was retried 3x per paragraph and then shipped as a
  full-English PDF the caller charged for.

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
- **`interlinear`** (`server/interlinear.py`): the original page, opened up so
  that each paragraph's translation stands directly above the line it renders.
  The page is cut along lines nothing crosses and the parts below each cut
  slide down, making a band exactly as tall as the gloss needs; the freed space
  is painted with a hairline of the background it was opened in, so a slide's
  panel simply grows. A page is first divided into regions — columns at
  full-height corridors, rows at full-width gaps — and each is opened
  independently, so a screenshot beside the bullets is never sliced by them.
  Nothing is scaled and nothing is reflowed: the original comes through at
  1:1, further down a taller page. On two real 26-page decks this drew every
  gloss (0 skipped) where the compact layout skipped a fifth of them.
- **`interlinear_compact`**: the same idea with the page size held fixed — each
  gloss drawn in whatever whitespace the author left above its paragraph,
  sized to that band, spread down the paragraph's own source lines when one
  band cannot hold it legibly, and SKIPPED rather than drawn over the reader's
  document when there is no room (`X-Overlay-Skipped` counts those). For a
  reader who needs the original pagination — and the fallback for a page whose
  /Rotate the spaced layout cannot open along.

  Arabic is shaped and bidi-resolved by PyMuPDF's Story engine in both; the
  gloss font is BabelDOC's own, subset per document (a deck would otherwise
  carry one 15 MB font copy per gloss).

  Sidecar blocks marked `on_raster` (text the engine found INSIDE an embedded
  image, with the image's `region`) are glossed inside that image in both
  styles: on a rounded translucent plate in the band directly above the label,
  placed by reading the region's PIXELS — a plate may dim flat artwork but
  never cover a mark — falling back below the label, else skipped. Opening the
  page cannot help a label that lives inside a raster, so under `interlinear`
  the plates simply ride the image to wherever the cuts moved it. Their fit
  comes back on `X-Overlay-Raster-Drawn` / `X-Overlay-Raster-Skipped`, and the
  plate is tunable via `plate_color` / `plate_opacity` / `plate_padding`.

## Vocab pages («كلمات هذه الصفحة»)

The readers are not native English speakers, and most of what stops them is
ordinary English ("declared", "scope", "evolved", "custom"), not just deep
technical terms. After a **mono** run, one extra LLM pass over the run's
sidecar (`server/vocab.py`) picks each page's NEW English words and short
phrases — a deliberately generous bar (when in doubt, include; only trivial
function words and pure code identifiers stay out), first occurrence only (a
word introduced on page 3 never comes back on page 7), ≤20 words/page and
≤400/document; a document over ~30k source words is split into page-aligned
chunk calls that carry the already-introduced list. The entries land in the
sidecar under a top-level **`"vocab"`** key
(`{"<page_number>": [{w, ar, note?}]}`, string keys matching the sidecar's
0-based `page_number`; `{}` when nothing made the cut; absent on runs from
before this feature — consumers must tolerate both), and
`server/vocab_pages.py` renders each page's words as one compact RTL page —
tight rows of bold English word — Arabic meaning — optional muted note, two
columns past six entries — **inserted IMMEDIATELY AFTER that page's content**,
never deferred to the end. Pages without new words get nothing.

Because the mono result is now interleaved, the pipeline also records where
the content pages ended up, as a top-level
**`"artifact_layout": {"content_pages": [i0, i1, ...]}`** (index of content
page N in the baked mono; written only when pages were really inserted). The
same insertion rule rides the free layouts:

- **`/v1/overlay`** inserts each page's vocab page right after that original
  page whenever the sidecar it received carries `"vocab"` (the report gains a
  `vocab_pages` count).
- **`/v1/compose`** accepts an optional multipart `sidecar` part. It first
  takes the translated CONTENT pages back out — via `artifact_layout` when
  the sidecar carries one (exact: the baked-in vocab pages and any appendix
  tail are dropped), else by trimming everything past the sidecar's
  `total_pages` — assembles the dual, then inserts each vocab page after its
  pair: after the (original, translated) pair of page N for `alternating`,
  after wide page N for `side_by_side`. Without a sidecar, compose is also
  tolerant of a translated PDF that is *longer* than the original: the extra
  tail pages are appended whole at the end (alternating and side-by-side
  alike) instead of being paired against blanks.

Everything is best-effort: any failure logs and the artifact ships without
the vocab layer, never degraded and never failed. Kill switch:
`VOCAB_PAGES=0` disables the extraction call and every insertion on all three
paths — behavior is byte-identical to before the feature.

## Environment

| Env | Default | Meaning |
|---|---|---|
| `DOCTRANSLATE_TOKEN` | — (required) | Shared secret for `/v1` |
| `OPENAI_API_KEY` | — (required) | LLM key (OpenRouter) |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint |
| `OPENAI_MODEL` | `google/gemini-3.1-flash-lite` | Translation model |
| `GOTENBERG_URL` | `http://localhost:3000` | The shared Gotenberg service `/v1/convert` renders office documents through. On prod: `http://gotenberg-int:3000` (same docker network) |
| `DATA_DIR` | `/data` | Job storage (volume) |
| `JOB_TTL_HOURS` | `24` | Job retention |
| `MAX_UNTRANSLATED_RATIO` | `0.05` | Share of paragraphs a run may leave untranslated before it is `failed` instead of delivered (capped at `1.0` = off) |
| `VOCAB_PAGES` | `1` | «كلمات هذه الصفحة» per-page vocabulary pages; `0` disables extraction and every insertion |

## Run locally

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e . -r server/requirements.txt   # from the repo root
DOCTRANSLATE_TOKEN=dev OPENAI_API_KEY=... uvicorn server.app:app --port 8000
```

Requires `tesseract` (+eng) and ghostscript on PATH for the scanned path.
First run downloads babeldoc assets (fonts + onnx layout model) to
`~/.cache/babeldoc`.

`/v1/convert` needs a Gotenberg to talk to — the default `GOTENBERG_URL` points
at one on localhost:

```bash
docker run --rm -p 3000:3000 gotenberg/gotenberg:8.36.0-libreoffice
```

Without it `/healthz` reads `degraded` and `/v1/convert` answers `503`;
everything else (translation, compose, overlay, notes-space) is unaffected —
the office suite was never on the translation path.

## Docker / Coolify

- **Build context: the repo root**, dockerfile `server/Dockerfile`
  (`docker build -f server/Dockerfile .`). The image bakes tesseract,
  ghostscript, fonts and the babeldoc assets, so boot needs no downloads. It
  carries **no office suite**: `/v1/convert` renders over HTTP against the
  shared Gotenberg service, which is what keeps this image ~520 MB smaller
  (3.4 GB → 2.88 GB, measured) and
  keeps a forked soffice out of a container capped at 3 GB.
- Port `8000`; persistent volume on `/data`; health check `GET /healthz`.
- Coolify: new app from this repo/branch, Build Pack = Dockerfile,
  "Dockerfile Location" = `/server/Dockerfile`, base directory `/`. Keep it on
  the internal network only (no public domain) and set `DOCTRANSLATE_TOKEN` +
  `OPENAI_API_KEY` in both this app and catodemy.
- Set `GOTENBERG_URL=http://gotenberg-int:3000`. Both containers sit on the
  `coolify` docker network, so the name resolves directly — no domain, no
  exposed port. If it is unset the engine falls back to `localhost:3000`,
  finds nothing, and `/healthz` reports `degraded` until it is fixed.
