"""The proven EN->AR translation pipeline behind each job.

Digital PDFs:   babeldoc (arabic-rtl fork) directly.
Scanned PDFs / garbage text layers:
    ocrmypdf --force-ocr -l eng  ->  ocr_prep.py  ->  babeldoc
    (--skip-scanned-detection --ocr-workaround)  ->  fix_layer_order.py
Formats: translated (mono), alternating / side_by_side (babeldoc native dual).
A mono run also emits a translation sidecar (jobs.sidecar_path) — the
translated text as data, which /v1/overlay turns into further layouts
without paying for the translation twice.
"""

import asyncio
import json
import logging
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from pypdf import PdfReader

from server import config
from server import jobs

logger = logging.getLogger("doctranslate.pipeline")

_warmup_lock = threading.Lock()
_doc_layout_model = None

# Overall progress bands (percent).
_P_DETECT = 2.0
_P_OCR = 8.0
_P_PREP = 12.0
_P_BABELDOC_END = 97.0


def warmup() -> None:
    """One-time heavy init: cache folders + layout model (downloads on first run)."""
    global _doc_layout_model
    with _warmup_lock:
        if _doc_layout_model is not None:
            return
        import babeldoc.format.pdf.high_level as high_level
        from babeldoc.docvision.doclayout import DocLayoutModel

        high_level.init()
        _doc_layout_model = DocLayoutModel.load_onnx()
        logger.info("warmup complete (doc layout model loaded)")


def _set_progress(job_id: str, percent: float, stage: str) -> None:
    jobs.update_job(job_id, progress={"percent": round(min(percent, 100.0), 2),
                                      "stage": stage})


def count_pages(pdf_path: Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)


def has_real_text_layer(pdf_path: Path, sample_pages: int = 8) -> bool:
    """True when the PDF carries a usable digital text layer.

    Scanned PDFs extract as empty; broken CID layers (no ToUnicode) extract as
    garbage. Heuristic: a page counts as "real" when it yields >= 40 chars of
    which at least half are letters/digits/common punctuation.
    """
    reader = PdfReader(str(pdf_path))
    pages = reader.pages[:sample_pages]
    real_pages = 0
    for page in pages:
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001 - malformed page = no text
            text = ""
        if len(text) < 40:
            continue
        sensible = sum(1 for c in text if c.isalnum() or c.isspace()
                       or c in ".,;:!?()-'\"/%+=")
        if sensible / len(text) >= 0.5:
            real_pages += 1
    return real_pages >= max(1, len(pages) // 2)


def _run_cmd(argv: list[str], job_id: str, stage: str) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise RuntimeError(f"{stage} failed (exit {proc.returncode}): {tail}")


def _run_babeldoc(job_id: str, input_pdf: Path, out_dir: Path, *, lang_in: str,
                  lang_out: str, fmt: str, scanned: bool, progress_base: float,
                  sidecar_path: Path | None = None):
    """Run the fork through its Python API; returns (TranslateResult, translator)."""
    from babeldoc.format.pdf.high_level import async_translate
    from babeldoc.format.pdf.translation_config import TranslationConfig
    from babeldoc.format.pdf.translation_config import WatermarkOutputMode
    from babeldoc.glossary import Glossary
    from babeldoc.translator.translator import set_translate_rate_limiter

    from server.cost import CostTrackingTranslator

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    # CostTrackingTranslator is a plain OpenAITranslator that also asks the
    # provider what each call cost, so the job's usage is the provider's own
    # figure rather than a locally computed one.
    translator = CostTrackingTranslator(
        lang_in=lang_in,
        lang_out=lang_out,
        model=config.OPENAI_MODEL,
        base_url=config.OPENAI_BASE_URL,
        api_key=config.OPENAI_API_KEY,
    )
    set_translate_rate_limiter(4)

    glossaries = []
    if config.GLOSSARY_PATH.is_file():
        glossary = Glossary.from_csv(config.GLOSSARY_PATH, lang_out)
        if glossary.entries:
            glossaries.append(glossary)

    tc = TranslationConfig(
        translator=translator,
        input_file=str(input_pdf),
        lang_in=lang_in,
        lang_out=lang_out,
        doc_layout_model=_doc_layout_model,
        output_dir=str(out_dir),
        no_dual=(fmt == "translated"),
        no_mono=(fmt != "translated"),
        use_alternating_pages_dual=(fmt == "alternating"),
        watermark_output_mode=WatermarkOutputMode.NoWatermark,
        skip_scanned_detection=scanned,
        ocr_workaround=scanned,
        glossaries=glossaries,
        auto_extract_glossary=False,
        translation_sidecar_path=sidecar_path,
    )

    span = _P_BABELDOC_END - progress_base
    result_holder: dict = {}

    async def consume() -> None:
        async for event in async_translate(tc):
            etype = event.get("type")
            if etype in ("progress_start", "progress_update", "progress_end"):
                overall = float(event.get("overall_progress") or 0.0)
                _set_progress(job_id, progress_base + overall * span / 100.0,
                              str(event.get("stage") or "translating"))
            elif etype == "finish":
                result_holder["result"] = event["translate_result"]
                break  # the event stream does not terminate on its own
            elif etype == "error":
                result_holder["error"] = str(event.get("error"))
                break

    asyncio.run(consume())

    if "error" in result_holder:
        raise RuntimeError(f"babeldoc failed: {result_holder['error']}")
    if "result" not in result_holder:
        raise RuntimeError("babeldoc finished without producing a result")
    return result_holder["result"], translator


def _usage(translator) -> dict:
    """This job's REAL spend, as reported by the provider.

    Three outcomes, kept distinct because they mean different things to the
    caller's money path:

    - `calls == 0` — babeldoc served the whole document from its translation
      cache, so no request was ever made and the run cost NOTHING. That is
      certain knowledge, not missing information, so `cost_usd` is 0.0 with
      `cost_source: "cache"`. Reporting null here (as this did at first) filed a
      genuinely free run under the same signal as billing drift and logged a
      warning for a completely normal event.
    - some call reported a cost — `cost_usd` is the provider's own summed
      figure, `cost_source: "provider"`. `priced_calls`/`calls` is its coverage,
      so a partially-priced run is visibly partial rather than quietly cheap.
    - calls were made but NONE reported a cost — genuinely unknown. `cost_usd`
      is None and this is the one case that warrants a warning: the caller
      (catodemy's RunDocumentTranslation) treats a missing cost as "charge
      nothing, log loudly", which is the right failure for money we cannot
      substantiate. The old behaviour — computing a plausible number from a
      hardcoded rate table — turned that same situation into a silent,
      unreconcilable charge on a user's balance.

    `generation_ids` is what makes a finished job auditable against the provider
    afterwards.
    """
    spend = translator.spend()
    calls, priced = spend["calls"], spend["priced_calls"]

    if calls == 0:
        # Fully cached: free, and known to be free.
        cost_usd, source = 0.0, "cache"
        logger.info("no provider calls (translation cache hit); this run was free")
    elif priced == 0:
        cost_usd, source = None, None
        logger.warning(
            "none of %s call(s) reported a cost; usage.cost_usd is null and "
            "the caller will not charge for this run",
            calls,
        )
    else:
        cost_usd, source = spend["cost_usd"], "provider"

        if priced != calls:
            logger.warning(
                "only %s of %s call(s) reported a cost; usage.cost_usd understates the run",
                priced, calls,
            )

    return {
        "prompt_tokens": int(translator.prompt_token_count.value),
        "completion_tokens": int(translator.completion_token_count.value),
        "cost_usd": cost_usd,
        "cost_source": source,
        "calls": calls,
        "priced_calls": priced,
        "generation_ids": spend["generation_ids"],
    }


def _set_pdf_title(pdf_path: Path, title: str) -> None:
    try:
        import pymupdf

        doc = pymupdf.open(str(pdf_path))
        meta = doc.metadata or {}
        meta["title"] = title
        doc.set_metadata(meta)
        doc.saveIncr()
        doc.close()
    except Exception:  # noqa: BLE001 - cosmetic only
        logger.warning("could not set PDF title on %s", pdf_path, exc_info=True)


def _append_glossary(output_pdf: Path, sidecar_path: Path) -> None:
    """The «شرح المصطلحات» step: pick the run's hard terms, record them in the
    sidecar, append the styled pages to the result.

    Best-effort at every seam — the paid translation is already on disk and
    NOTHING here may lose it. A failure logs and the job finishes exactly as it
    did before this feature existed. The entries are written into the sidecar
    (top-level "glossary" key, [] when nothing made the cut) even when no pages
    can be appended, so /v1/overlay and /v1/compose can render the same pages
    later without a second LLM call.
    """
    from server import terms

    try:
        sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no sidecar, no glossary
        logger.exception("glossary: could not read the sidecar; skipping")
        return

    try:
        entries = terms.extract_terms(sidecar_data)  # [] on failure, by contract
    except Exception:  # noqa: BLE001 - belt and braces on that contract
        logger.exception("glossary: extract_terms broke its never-raise "
                         "contract; skipping")
        return

    sidecar_data["glossary"] = entries
    try:
        sidecar_path.write_text(json.dumps(sidecar_data, ensure_ascii=False),
                                encoding="utf-8")
    except Exception:  # noqa: BLE001
        logger.exception("glossary: could not write entries into the sidecar")

    if not entries:
        return

    try:
        import pymupdf

        from server import glossary_pages

        doc = pymupdf.open(str(output_pdf))
        try:
            added = glossary_pages.append_glossary_pages(doc, entries)
            if not added:
                return
            # A full save (not incremental): garbage=4 folds the per-box
            # copies of the subset card font into one. Via a sibling temp
            # file — pymupdf cannot rewrite the file it has open.
            tmp = output_pdf.with_name(output_pdf.name + ".glossary.tmp")
            doc.save(str(tmp), garbage=4, deflate=True)
        finally:
            doc.close()
        tmp.replace(output_pdf)
    except Exception:  # noqa: BLE001 - the appendix must never lose the run
        logger.exception("glossary: appending pages failed; the result ships "
                         "without them")


def run_job(job_id: str) -> None:
    job = jobs.read_job(job_id)
    if job is None:
        raise RuntimeError("job vanished")
    d = jobs.job_dir(job_id)
    input_pdf = d / "input.pdf"
    work = d / "work"
    work.mkdir(exist_ok=True)
    out_dir = work / "out"
    out_dir.mkdir(exist_ok=True)
    fmt = job["format"]
    lang_in, lang_out = job["lang_in"], job["lang_out"]

    _set_progress(job_id, 0.5, "analyzing")
    pages = count_pages(input_pdf)
    jobs.update_job(job_id, pages=pages)
    scanned = not has_real_text_layer(input_pdf)
    _set_progress(job_id, _P_DETECT, "scanned" if scanned else "digital")

    babeldoc_input = input_pdf
    if scanned:
        ocr_pdf = work / "ocr.pdf"
        prep_pdf = work / "prep.pdf"
        _set_progress(job_id, _P_DETECT, "ocr")
        _run_cmd(["ocrmypdf", "--force-ocr", "-l", "eng",
                  str(input_pdf), str(ocr_pdf)], job_id, "ocrmypdf")
        _set_progress(job_id, _P_OCR, "ocr_prep")
        _run_cmd([sys.executable, str(config.OCR_PREP_SCRIPT),
                  str(ocr_pdf), str(prep_pdf)], job_id, "ocr_prep")
        babeldoc_input = prep_pdf
        _set_progress(job_id, _P_PREP, "translating")
        progress_base = _P_PREP
    else:
        progress_base = _P_DETECT

    # The sidecar rides along with mono runs (SIDECAR_FORMATS): the run that is
    # already being paid for is the only place the translated text and the
    # ORIGINAL page's geometry are both in hand, and writing them down is what
    # lets later layouts (server/interlinear.py) be rebuilt for free instead of
    # re-translating. It is written straight into the job dir so it survives
    # the work-dir cleanup below and can be fetched like the result.
    sidecar = jobs.sidecar_path(job_id) if fmt in config.SIDECAR_FORMATS else None

    result, translator = _run_babeldoc(
        job_id, babeldoc_input, out_dir, lang_in=lang_in, lang_out=lang_out,
        fmt=fmt, scanned=scanned, progress_base=progress_base,
        sidecar_path=sidecar)

    _set_progress(job_id, _P_BABELDOC_END, "finalizing")
    if fmt == "translated":
        produced = result.no_watermark_mono_pdf_path or result.mono_pdf_path
    else:
        produced = result.no_watermark_dual_pdf_path or result.dual_pdf_path
    if not produced or not Path(produced).is_file():
        raise RuntimeError("babeldoc produced no output PDF")
    produced = Path(produced)

    output_pdf = jobs.result_path(job_id)
    if scanned and fmt == "translated":
        # Move the translated /OCR-* layer above the page raster (belt-and-braces;
        # a no-op after ocr_prep.py rebuilt the sandwich in the right order).
        _run_cmd([sys.executable, str(config.FIX_LAYER_ORDER_SCRIPT),
                  str(produced), str(output_pdf)], job_id, "fix_layer_order")
    else:
        shutil.copyfile(produced, output_pdf)

    # «شرح المصطلحات»: one LLM pass over the sidecar picks the document's
    # genuinely hard terms, the entries land in the sidecar ("glossary" key)
    # and the styled pages are appended to the result. Mono runs only — the
    # sidecar is the input, and only mono runs have one.
    if sidecar is not None and sidecar.is_file() and config.GLOSSARY_PAGES:
        _set_progress(job_id, 98.0, "glossary")
        _append_glossary(output_pdf, sidecar)

    if job.get("title"):
        _set_pdf_title(output_pdf, job["title"])

    usage = _usage(translator)
    shutil.rmtree(work, ignore_errors=True)
    jobs.update_job(job_id, status="done", usage=usage,
                    progress={"percent": 100.0, "stage": "done"})
