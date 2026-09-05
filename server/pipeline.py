"""The proven EN->AR translation pipeline behind each job.

Digital PDFs:   babeldoc (arabic-rtl fork) directly.
Scanned PDFs / garbage text layers:
    ocrmypdf --force-ocr --pages <the bad ones>  ->  ocr_prep.py  ->  babeldoc
    (--skip-scanned-detection --ocr-workaround)  ->  fix_layer_order.py
Formats: translated (mono), alternating / side_by_side (babeldoc native dual).
A mono run also emits a translation sidecar (jobs.sidecar_path) — the
translated text as data, which /v1/overlay turns into further layouts
without paying for the translation twice.

The lane is chosen by `classify_pages`, which reads EVERY page and files each
one as text / garbage / empty. Two things hang off it: which lane the document
takes (a majority of real text pages = digital), and — inside the OCR lane —
WHICH pages are OCR'd. A page that already carries a good text layer is never
rasterised, whichever lane the document as a whole ends up in.
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


#: Per-page verdicts from `classify_pages`.
PAGE_TEXT = "text"        # a usable digital text layer: translate it as text
PAGE_GARBAGE = "garbage"  # characters come out, but they are not language
PAGE_EMPTY = "empty"      # nothing worth translating came out at all

MIN_PAGE_CHARS = 40
#: Gate on `language_share`; see its docstring for where the number comes from.
MIN_LANGUAGE_SHARE = 0.5


def language_share(text: str) -> float:
    """Share of `text` that is letters, digits or whitespace.

    This is the discriminator between a real text layer and a broken one, and
    it is deliberately NOT the "sensible character" count it replaces. A PDF
    whose /ToUnicode CMaps are missing or wrong (a LaTeX subset, an old
    scanner driver, a re-exported deck) renders perfectly and extracts roughly
    the right NUMBER of characters — they are simply the wrong ones, and what
    they land on is overwhelmingly punctuation: `!!"#$%&'$()*!"#$%&'"$((&)&*`.
    Counting punctuation as sensible scored exactly that file 0.74 and waved
    it through as digital, so the OCR lane was never entered and babeldoc
    replaced a legible page with its own mojibake.

    MEASURED over the 469 text-bearing pages of the 14-document production
    corpus, this share never drops below 0.808. The same measurement on a
    corpus document with its five CMaps deleted: 0.190-0.206. The gate sits at
    0.5, roughly midway, with 0.3 of margin on either side.

    Digits are counted deliberately. On letters and spaces alone a hex/binary
    conversion table (run67 p.48) scores 0.433 and a formula-dense logic slide
    0.660 — both perfectly good text layers that a letters-only gate would
    have shipped down the OCR lane.
    """
    if not text:
        return 0.0
    good = sum(1 for c in text if c.isalpha() or c.isdigit() or c.isspace())
    return good / len(text)


def classify_pages(pdf_path: Path) -> list[str]:
    """One verdict per page, over the WHOLE document.

    This used to sample the first 8 pages only, which gets the very common
    "digital front, scanned back" document exactly wrong: the head is clean,
    the document is filed as digital, and the scanned tail never reaches the
    OCR lane. Reading every page costs ~5 ms each (0.4 s for an 83-page deck,
    MEASURED on the corpus) against a translation run measured in minutes.
    """
    reader = PdfReader(str(pdf_path))
    verdicts = []
    for page in reader.pages:
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001 - malformed page = no text
            text = ""
        if len(text) < MIN_PAGE_CHARS:
            verdicts.append(PAGE_EMPTY)
        elif language_share(text) < MIN_LANGUAGE_SHARE:
            verdicts.append(PAGE_GARBAGE)
        else:
            verdicts.append(PAGE_TEXT)
    return verdicts


def has_real_text_layer(pdf_path: Path,
                        verdicts: list[str] | None = None) -> bool:
    """True when the PDF carries a usable digital text layer.

    Pass `verdicts` when the caller has already classified the pages, so a
    document is only ever read once.
    """
    if verdicts is None:
        verdicts = classify_pages(pdf_path)
    real_pages = sum(1 for verdict in verdicts if verdict == PAGE_TEXT)
    return real_pages >= max(1, len(verdicts) // 2)


def pages_needing_ocr(verdicts: list[str]) -> list[int]:
    """0-based indices of the pages the OCR lane must actually process.

    Everything that is not a real text layer: a raster scan (empty) and a
    broken CID map (garbage) both need the pixels read. A page filed as
    `text` is left exactly as it is — see `_ocr_lane`.
    """
    return [number for number, verdict in enumerate(verdicts)
            if verdict != PAGE_TEXT]


#: At or above this Arabic share the document IS the target language: refuse.
ARABIC_REFUSE_SHARE = 0.60
#: At or above this, say so and let the caller decide: a bilingual handout is
#: a legitimate thing to want translated, a wholly Arabic one is not.
ARABIC_WARN_SHARE = 0.20

_ARABIC_RANGES = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)


def _is_arabic(char: str) -> bool:
    point = ord(char)
    return any(low <= point <= high for low, high in _ARABIC_RANGES)


class SourceLanguageRefused(RuntimeError):
    """The document is already written in the language we would translate to.

    Raised before a single provider call is made, so the run costs nothing.
    `code` is the stable identifier the caller maps to its own message —
    `str(exc)` leads with it so the code survives even a caller that only ever
    sees the flat `error` string on the job record.
    """

    code = "source_already_in_target_language"

    def __init__(self, analysis: dict):
        self.analysis = analysis
        super().__init__(
            f"{self.code}: the document is already "
            f"{analysis['arabic_share'] * 100:.0f}% Arabic "
            f"({analysis['arabic_chars']} Arabic letters against "
            f"{analysis['latin_chars']} Latin) — there is nothing to "
            f"translate into Arabic.")


def analyze_source_language(pdf_path: Path,
                            verdicts: list[str] | None = None) -> dict:
    """What language the document's own text layer is actually in.

    Nothing anywhere used to ask. `lang_in` is a constant that travels from
    catodemy's config to `TranslationConfig` without a single caller ever
    setting it, so an Arabic document ran to `done` and came back as a
    strictly degraded Arabic-to-Arabic copy — reprocessed glyphs punched full
    of `(cid:580)` holes, a second vocabulary layer stacked on the first — and
    was charged for in full. The only thing that ever stopped it was
    babeldoc's refusal to re-translate its OWN output, which does not fire on
    a file anyone else produced.

    Counting letters, not characters: digits, punctuation and whitespace are
    shared between the two scripts and only dilute the signal. Presentation
    forms count as Arabic because that is how our own delivered files (the
    most likely accidental re-upload) extract.

    Returns a dict that is also written onto the job record, so the caller can
    see the number rather than only the verdict:

        {"arabic_chars": int, "latin_chars": int, "arabic_share": float,
         "verdict": "ok" | "warn" | "refuse", "pages_sampled": int}
    """
    reader = PdfReader(str(pdf_path))
    arabic = latin = 0
    sampled = 0
    for number, page in enumerate(reader.pages):
        if verdicts is not None and verdicts[number] != PAGE_TEXT:
            continue
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - malformed page = no text
            continue
        sampled += 1
        for char in text:
            if not char.isalpha():
                continue
            if _is_arabic(char):
                arabic += 1
            elif char.isascii():
                latin += 1

    letters = arabic + latin
    share = (arabic / letters) if letters else 0.0
    if share >= ARABIC_REFUSE_SHARE:
        verdict = "refuse"
    elif share >= ARABIC_WARN_SHARE:
        verdict = "warn"
    else:
        verdict = "ok"
    return {"arabic_chars": arabic, "latin_chars": latin,
            "arabic_share": round(share, 4), "verdict": verdict,
            "pages_sampled": sampled}


def analyze_input(pdf_path: Path) -> dict:
    """Everything the input tells us before a penny is spent.

    One read of the document, shared by the lane decision, the OCR page
    selection and the source-language check. `POST /v1/jobs` can call this on
    the uploaded bytes and refuse a `language.verdict == "refuse"` document
    with a 422 at the door; `run_job` calls it anyway, so a caller that does
    not is still never charged for an Arabic-to-Arabic run.
    """
    verdicts = classify_pages(pdf_path)
    return {
        "pages": len(verdicts),
        "page_kinds": verdicts,
        "scanned": not has_real_text_layer(pdf_path, verdicts),
        "ocr_pages": pages_needing_ocr(verdicts),
        "language": analyze_source_language(pdf_path, verdicts),
    }


class _ScannedAfterAll(RuntimeError):
    """babeldoc's scanned detector disagreed with ours, mid-run.

    Internal to this module: `run_job` catches it and re-runs the document
    down the OCR lane. It never reaches the caller, because a disagreement
    between our two detectors is our problem, not the user's.
    """


def _run_cmd(argv: list[str], job_id: str, stage: str) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise RuntimeError(f"{stage} failed (exit {proc.returncode}): {tail}")


def _run_babeldoc(job_id: str, input_pdf: Path, out_dir: Path, *, lang_in: str,
                  lang_out: str, fmt: str, scanned: bool, progress_base: float,
                  sidecar_path: Path | None = None,
                  image_text_regions: Path | None = None):
    """Run the fork through its Python API.

    Returns (TranslateResult, translator, coverage) — coverage being how many
    paragraphs the run actually translated, which is the difference between a
    translation and a copy.
    """
    from babeldoc.format.pdf.high_level import async_translate
    from babeldoc.format.pdf.translation_config import TranslationConfig
    from babeldoc.format.pdf.translation_config import WatermarkOutputMode
    from babeldoc.glossary import Glossary
    from babeldoc.translator.translator import set_translate_rate_limiter

    from server.cost import CostTrackingTranslator
    from server.cost import HardProviderError

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
        image_text_regions=image_text_regions,
    )
    # So the first terminal provider refusal can cancel the run in flight
    # rather than being discovered afterwards (server/cost.py).
    translator.translation_config = tc

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
                # The event carries the exception OBJECT (its docstring says
                # str; progress_monitor.translate_error passes it through), and
                # the type is what tells a recoverable disagreement apart from
                # a real failure. Keep both.
                result_holder["error"] = event.get("error")
                break

    asyncio.run(consume())

    # Recorded before anything can raise, so a failed job still reports how
    # far it got — GET /v1/jobs/{id} surfaces it as "coverage".
    coverage = _coverage(tc)
    jobs.update_job(job_id, coverage=coverage)
    logger.info(
        "job %s translated %d of %d paragraphs (%d untranslated)",
        job_id,
        coverage["paragraphs_translated"],
        coverage["paragraphs_total"],
        coverage["paragraphs_untranslated"],
    )

    # Ahead of every other verdict: a spent or rejected key is not a scanned
    # document and not a babeldoc bug, and saying so is the whole point.
    hard = translator.hard_error()
    if hard:
        raise HardProviderError(hard)

    if "error" in result_holder:
        from babeldoc.babeldoc_exception.BabelDOCException import ScannedPDFError

        error = result_holder["error"]
        if isinstance(error, ScannedPDFError) and not scanned:
            raise _ScannedAfterAll(str(error))
        raise RuntimeError(f"babeldoc failed: {error}")
    if "result" not in result_holder:
        raise RuntimeError("babeldoc finished without producing a result")
    return result_holder["result"], translator, coverage


def _coverage(tc) -> dict:
    """How many of the document's paragraphs came back translated.

    babeldoc already counts this — it logs "Translation completed. Total: ...,
    Untranslated: ..." and then forgets it. `record_translation_coverage`
    banks the same two numbers on the config (and a split run accumulates
    into it), so this is a read, not a second counter.
    """
    counts = getattr(tc, "translation_coverage", None) or {}
    total = int(counts.get("total") or 0)
    untranslated = max(0, min(int(counts.get("untranslated") or 0), total))
    return {"paragraphs_total": total,
            "paragraphs_translated": total - untranslated,
            "paragraphs_untranslated": untranslated}


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


def _write_sidecar(sidecar_path: Path, data: dict) -> None:
    try:
        sidecar_path.write_text(json.dumps(data, ensure_ascii=False),
                                encoding="utf-8")
    except Exception:  # noqa: BLE001 - the sidecar update is best-effort
        logger.exception("could not write the updated sidecar")


def _insert_vocab(output_pdf: Path, sidecar_path: Path) -> None:
    """The «كلمات هذه الصفحة» step: pick each page's new English words, record
    them in the sidecar, and draw each page's words as a compact strip at the
    BOTTOM of that page (the page grows by exactly the strip's height) — never
    deferred to the end. A page the strip cannot serve falls back to the
    classic inserted vocab page right after it.

    A sidecar may carry a deep-terms list under its "glossary" key (nothing on
    this branch writes one, but a sidecar that has it is honoured); those
    terms are this pass's exclusion list, so no word is ever explained twice.
    An appendix tail the result may already carry simply shifts back as body
    pages are inserted, staying at the very end.

    The baked layout is recorded in the sidecar as
    `"artifact_layout": {"content_pages": [...], "vocab_strips": {...}}` —
    where each translated content page ended up in the mono file, plus each
    stripped page's strip height in points — which is what lets /v1/compose
    later recover the pristine content pages EXACTLY (drop any inserted
    fallback pages, crop the strips back off) instead of tail-trimming.
    Written only when the file was really changed: an untouched mono is still
    described perfectly by `total_pages`.

    Best-effort at every seam: the paid translation is already on disk and
    NOTHING here may lose it. A failure logs and the job finishes exactly as
    it did before this feature existed.
    """
    from server import vocab

    try:
        sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no sidecar, no vocab
        logger.exception("vocab: could not read the sidecar; skipping")
        return

    exclude = [entry.get("term")
               for entry in sidecar_data.get("glossary") or []
               if isinstance(entry, dict) and entry.get("term")]

    try:
        entries = vocab.extract_vocab(sidecar_data, exclude=exclude)
    except Exception:  # noqa: BLE001 - belt and braces on the {} contract
        logger.exception("vocab: extract_vocab broke its never-raise "
                         "contract; skipping")
        return

    sidecar_data["vocab"] = entries
    _write_sidecar(sidecar_path, sidecar_data)

    if not entries:
        return

    content = sidecar_data.get("total_pages")

    if not isinstance(content, int) or isinstance(content, bool) or content <= 0:
        content = len(sidecar_data.get("pages") or [])

    try:
        import pymupdf

        from server import page_fonts
        from server import vocab_pages

        doc = pymupdf.open(str(output_pdf))
        try:
            content = min(content, doc.page_count) or doc.page_count
            added = vocab_pages.attach_vocab(
                doc, entries, {number: number for number in range(content)})
            if not added:
                return
            # The strip pages are written HERE, after babeldoc's own cmap fix
            # has already run over the document, so this is the only place
            # their text layer can be repaired: without it they store Arabic
            # presentation forms instead of letters and Ctrl+F, copy-paste,
            # screen readers and catodemy's corpus ingestion all get glyph
            # soup. A no-op on a document that has no presentation forms.
            #
            # Looked up rather than called directly: this whole block sits
            # inside a never-lose-the-run guard, so on a build where the
            # repair is not present an AttributeError would be swallowed and
            # would silently cost the entire vocab layer — a much larger
            # regression than the one the repair fixes. Absent, we ship the
            # strips unrepaired, which is exactly what shipped before it
            # existed.
            repair = getattr(page_fonts, "repair_arabic_text_layer", None)
            if repair is None:
                logger.warning("vocab: page_fonts carries no Arabic text-layer "
                               "repair; the strips ship as presentation forms")
            else:
                repair(doc)
            # A full save (not incremental): garbage=4 folds the per-box
            # copies of the subset row font into one. Via a sibling temp
            # file — pymupdf cannot rewrite the file it has open.
            tmp = output_pdf.with_name(output_pdf.name + ".vocab.tmp")
            doc.save(str(tmp), garbage=4, deflate=True)
        finally:
            doc.close()
        tmp.replace(output_pdf)
    except Exception:  # noqa: BLE001 - the vocab layer must never lose the run
        logger.exception("vocab: attaching strips failed; the result ships "
                         "without them")
        return

    # Only now — the mono on disk really carries the baked layout. A strip is
    # reported as its height (+h points), a fallback insertion as -n pages.
    positions, strips, shift = [], {}, 0
    for number in range(content):
        positions.append(number + shift)
        value = added.get(number, 0)
        if value < 0:
            shift += int(-value)
        elif value:
            strips[str(number)] = round(value, 2)
    layout: dict = {"content_pages": positions}
    if strips:
        layout["vocab_strips"] = strips
    sidecar_data["artifact_layout"] = layout
    _write_sidecar(sidecar_path, sidecar_data)


def _page_ranges(numbers: list[int]) -> str:
    """[0,1,2,7] -> "1-3,8" — ocrmypdf's --pages argument, which is 1-based."""
    ranges, start, previous = [], None, None
    for number in numbers:
        if start is None:
            start = previous = number
        elif number == previous + 1:
            previous = number
        else:
            ranges.append((start, previous))
            start = previous = number
    if start is not None:
        ranges.append((start, previous))
    return ",".join(f"{a + 1}" if a == b else f"{a + 1}-{b + 1}"
                    for a, b in ranges)


def _ocr_lane(job_id: str, input_pdf: Path, work: Path, *, targets: list[int],
              lang_in: str, lang_out: str) -> Path:
    """Give the pixel pages a text layer, and touch NOTHING else.

    `targets` is the 0-based pages that need reading. Both stages are told
    about it. That is the whole fix for the case where five rasterised cover
    pages in front of nine good slides flipped the entire document to
    `--force-ocr`: the nine were rasterised, re-OCR'd and re-typeset, and a
    crisp Java keyword table came back as a smear of overlapping Arabic and
    Latin. `--force-ocr` is still right for the pages we DO hand over — a
    broken CID map has to be thrown away, not skipped — it just no longer
    gets to decide that on the whole document's behalf.

    MEASURED on a 20-page mixed specimen: with `--pages`, the untouched pages
    keep their text byte for byte (486/646/897 characters) and their vector
    drawings, while the named pages gain an OCR layer.
    """
    from server.ocr_prep import tesseract_lang

    ocr_pdf = work / "ocr.pdf"
    prep_pdf = work / "prep.pdf"
    lang = tesseract_lang(lang_in, lang_out)
    keep = ",".join(str(number) for number in range(count_pages(input_pdf))
                    if number not in set(targets))

    _set_progress(job_id, _P_DETECT, "ocr")
    _run_cmd(["ocrmypdf", "--force-ocr", "-l", lang,
              "--pages", _page_ranges(targets),
              str(input_pdf), str(ocr_pdf)], job_id, "ocrmypdf")
    _set_progress(job_id, _P_OCR, "ocr_prep")
    _run_cmd([sys.executable, str(config.OCR_PREP_SCRIPT),
              "--lang", lang, "--keep-pages", keep,
              str(ocr_pdf), str(prep_pdf)], job_id, "ocr_prep")
    return prep_pdf


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
    verdicts = classify_pages(input_pdf)
    pages = len(verdicts)
    scanned = not has_real_text_layer(input_pdf, verdicts)
    ocr_targets = pages_needing_ocr(verdicts)
    language = analyze_source_language(input_pdf, verdicts)
    jobs.update_job(job_id, pages=pages, language=language)

    # The document is already in the language we would translate it into.
    # Refuse HERE, before any provider call, so the run costs nothing — and
    # say the number, because the caller has to decide what to tell the user.
    # `warn` is deliberately not a refusal: a bilingual handout is a
    # legitimate thing to want translated, and the direction is never swapped
    # on the user's behalf.
    if language["verdict"] == "refuse" and lang_out.lower().startswith("ar") \
            and not lang_in.lower().startswith("ar"):
        raise SourceLanguageRefused(language)
    if language["verdict"] == "warn":
        logger.warning("job %s: %.0f%% of the source is already Arabic",
                       job_id, language["arabic_share"] * 100)

    _set_progress(job_id, _P_DETECT, "scanned" if scanned else "digital")

    babeldoc_input = input_pdf
    image_text_regions: Path | None = None
    if not scanned:
        # Text inside embedded raster images (diagram labels, figure scans on
        # otherwise-digital slides): OCR it into invisible runs so babeldoc
        # translates it in place. Best-effort — a deck whose diagrams cannot
        # be prepped still deserves its text-layer translation, so a prep
        # failure degrades to the old behaviour instead of failing the job.
        prep_pdf = work / "image_prep.pdf"
        regions_json = work / "image_regions.json"
        _set_progress(job_id, _P_DETECT, "image_prep")
        try:
            _run_cmd([sys.executable, str(config.IMAGE_PREP_SCRIPT),
                      str(input_pdf), str(prep_pdf), str(regions_json)],
                     job_id, "image_prep")
            regions = json.loads(regions_json.read_text())
            if regions.get("pages"):
                babeldoc_input = prep_pdf
                image_text_regions = regions_json
        except Exception:  # noqa: BLE001 - degrade, loudly, to text-only
            logger.exception("image_prep failed; translating without image text")
    if scanned:
        babeldoc_input = _ocr_lane(job_id, input_pdf, work,
                                   targets=ocr_targets, lang_in=lang_in,
                                   lang_out=lang_out)
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

    try:
        result, translator, coverage = _run_babeldoc(
            job_id, babeldoc_input, out_dir, lang_in=lang_in, lang_out=lang_out,
            fmt=fmt, scanned=scanned, progress_base=progress_base,
            sidecar_path=sidecar, image_text_regions=image_text_regions)
    except _ScannedAfterAll:
        # Two scanned detectors that could disagree, and the disagreement used
        # to end the job: ours reads the text layer, babeldoc's strips it and
        # compares renders, and a deck of full-slide screenshots with 5 pt
        # captions passes the first and is killed by the second. The user got
        # no file at all and an English sentence about it.
        #
        # babeldoc is the better judge here — "the text layer contributes
        # nothing visible" is precisely what its SSIM check measures and ours
        # cannot see — so believe it and re-run down the lane we would have
        # taken had we agreed, rather than failing. Every page goes, because
        # what babeldoc just told us is that none of them really has text.
        logger.warning("job %s: babeldoc detected a scanned document our text "
                       "check passed; re-running through the OCR lane", job_id)
        scanned = True
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        babeldoc_input = _ocr_lane(job_id, input_pdf, work,
                                   targets=list(range(pages)),
                                   lang_in=lang_in, lang_out=lang_out)
        result, translator, coverage = _run_babeldoc(
            job_id, babeldoc_input, out_dir, lang_in=lang_in, lang_out=lang_out,
            fmt=fmt, scanned=True, progress_base=_P_PREP,
            sidecar_path=sidecar, image_text_regions=None)

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

    # Refused as soon as there is something to keep, and before the vocab
    # pass: a run this incomplete must not buy one more LLM call on its way
    # out. The partial PDF stays on disk for debugging — /v1/jobs/{id}/result
    # 409s like any other failed job — and `usage`, the caller's signal to
    # charge, is never written.
    shutil.rmtree(work, ignore_errors=True)
    _refuse_poor_coverage(coverage)

    # «كلمات هذه الصفحة»: one lighter LLM pass over the sidecar picks each
    # page's new general-English words, records them in the sidecar ("vocab"
    # key) and interleaves the compact vocab pages into the result. Mono runs
    # only — the sidecar is the input, and only mono runs have one.
    if sidecar is not None and sidecar.is_file() and config.VOCAB_PAGES:
        _set_progress(job_id, 98.0, "vocab")
        _insert_vocab(output_pdf, sidecar)

    if job.get("title"):
        _set_pdf_title(output_pdf, job["title"])

    usage = _usage(translator) | coverage
    jobs.update_job(job_id, status="done", usage=usage,
                    progress={"percent": 100.0, "stage": "done"})


def _refuse_poor_coverage(coverage: dict) -> None:
    """Never hand back a mostly-English 'translation' as a finished job."""
    total = coverage["paragraphs_total"]
    untranslated = coverage["paragraphs_untranslated"]
    if total > 0 and untranslated / total > config.MAX_UNTRANSLATED_RATIO:
        raise RuntimeError(f"translation incomplete: {untranslated} of "
                           f"{total} paragraphs untranslated")
