"""Per-page vocabulary extraction for the «كلمات هذه الصفحة» pages.

The document's readers are not native English speakers, and the words that
stop them are mostly not the deep technical terms — they are ordinary English
("declared", "scope", "evolved", "custom", "range of customers") met for the
first time. This pass reads the run's sidecar and picks, for EVERY page, the
English words and short phrases a non-native CS student might not be fully
comfortable with, each with a concise Arabic meaning, rendered by
`server/vocab_pages.py` as a strip under the page the words appeared on. The
bar is deliberately LOW: when in doubt a word is included — only the trivial
function words (the, is, of, and…) and pure code identifiers stay out.

WHICH PAGE a word lands on is DERIVED here, never taken from the model: every
entry is placed on the page where its word FIRST OCCURS in the sidecar's own
source text ({@link _page_index}). Slides that print their own page number as
their first text block ("41 | Abstraction (Cont.)") used to make the model echo
the printed number instead of the marker, which slid a whole document's
vocabulary one page late and threw the last page's words away; deriving the
page from the text is immune to that, and it enforces the first-occurrence rule
— and drops a word that occurs nowhere — for free. The caller may pass an
exclusion list of terms already explained elsewhere (a sidecar's deep-terms
"glossary", when one exists), so no word is ever explained twice.

The model is shown each page's SOURCE and the run's OWN Arabic rendering of it,
and told to reuse the document's wording for a term it already translated: the
strip must never contradict the body four centimetres above it. Arabic comes
back undiacritised — the body carries no tashkeel, and the same
`postprocess_arabic_translation` the body goes through is applied here, so the
prompt's request is backed by a guarantee.

One JSON-mode chat call covers the whole document; a document whose text
exceeds {@link CHUNK_WORDS} words is split into page-aligned chunks, each call
carrying the words already introduced so far so later chunks keep the
first-occurrence rule. A chunk that fails mid-run logs and keeps what earlier
chunks produced.

Failure posture: a translation must NEVER fail — or even degrade — because of
this pass. Every failure mode logs and returns `{}`, which every caller reads
as "no vocab pages".
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from server import config

logger = logging.getLogger("doctranslate.vocab")

# Caps, whatever the model says. Twenty rows still sit comfortably on one
# two-column compact page; past ~400 words the layer stops being "this page's
# words" and starts being a dictionary of the language.
MAX_PER_PAGE = 20
MAX_TOTAL = 400

# One call covers this many words of text — source AND the page's translation,
# both of which travel — so a longer document is split into page-aligned
# chunks and no single call carries a maximum-context bill.
CHUNK_WORDS = 30_000

_SYSTEM_PROMPT = """\
أنت مساعد لغوي لطلاب علوم الحاسب العرب — غير ناطقين بالإنجليزية. ستقرأ نص \
مستند إنجليزي صفحةً صفحة، ومعه ترجمة المستند العربية لكل صفحة، وتختار من كل \
صفحة الكلمات والعبارات الإنجليزية الجديدة التي قد لا يرتاح لها الطالب \
ارتياحًا كاملًا — مثل declared أو scope أو evolved أو custom أو \
"range of customers" — لتُشرح له بالعربية بجانب الصفحة نفسها.

قواعد الاختيار:
- كن كريمًا في الاختيار: أي كلمة أو عبارة إنجليزية قد يتردد عندها طالب غير \
ناطق بالإنجليزية — ولو قليلًا — تُدرَج. إذا شككت هل يعرفها الطالب تمامًا، \
أدرِجها.
- يُستثنى فقط: الكلمات الوظيفية البديهية (the, is, of, and, a, this, use) — \
هذه لا تُذكر إطلاقًا — ومعرّفات الكود الصِّرفة (أسماء متغيرات ودوال كما وردت \
في الشيفرة، مثل boolean_expression).
- لا تشرح الكلمات الواردة في قائمة «شُرحت سابقًا» — شُرحت في مكان آخر من \
المستند.
- الكلمة تُشرح عند أول ظهورها فقط: كلمة ظهرت في صفحة أسبق (أو في قائمة \
«شُرحت سابقًا») لا تُعاد في الصفحات التالية، ولا تُعاد باشتقاق آخر منها \
(emerging بعد emerged مثلًا).
- الكلمة أو العبارة تُكتب كما وردت في نص الصفحة حرفيًا؛ لا تخترع عبارة ليست \
في النص.
- حتى 20 كلمة للصفحة الواحدة؛ صفحة بلا كلمات جديدة تُحذف من الجواب كليًا. \
صفحة نصية عادية فيها كلمات جديدة يُتوقع أن تُخرج 8 كلمات أو أكثر — \
إن وجدت أقل من ذلك فراجع النص مرة أخرى بحثًا عن كلمات فاتتك.

قواعد المعنى العربي:
- إن كانت ترجمة المستند لهذه الصفحة قد ترجمت المصطلح نفسه، فاستعمل ترجمتها \
حرفيًا: الشرح يُقرأ على الصفحة ذاتها، فلا يجوز أن يخالف نصها.
- بلا تشكيل: اكتب «يصرح عنه» لا «يُصرَّح عنه».
- إن اتفق معنيان عربيان لكلمتين في الصفحة نفسها، فالـ note واجبة على كل \
منهما لتمييزهما.

لكل كلمة:
- w: الكلمة أو العبارة كما وردت في النص.
- ar: معناها بالعربية في كلمة إلى خمس كلمات، بلا تشكيل.
- note (اختياري): عبارة توضيحية قصيرة واحدة بالعربية — مثلًا لتمييز المعنى \
المقصود هنا عن المعنى الشائع.

أرجع JSON فقط بهذا الشكل، ومفاتيح الصفحات هي أرقام سطور «— الصفحة N —» — \
تجاهل أي أرقام صفحات مطبوعة داخل نص الصفحة نفسها:
{"vocab": {"3": [{"w": "declared", "ar": "يصرح عنه",
                  "note": "في البرمجة: تعريف المتغير قبل استخدامه"}]}}
"""

# Matching a word against the source text: whitespace is normalised on both
# sides (a phrase may straddle a line break in the sidecar's blocks) and the
# comparison is casefolded.
_WHITESPACE_RE = re.compile(r"\s+")

# The spellings one entry stands for, so a word family is not explained twice
# ("emerging" on page 14 and "emerged" on page 24 are one word). Only stems
# long enough to still be a word are formed.
_SUFFIXES = ("ing", "ed", "es", "s")
_MIN_STEM = 4


def _fold(text: object) -> str:
    """`text` casefolded with its whitespace collapsed to single spaces."""
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip().casefold()


def _family(term: str) -> set[str]:
    """The casefolded spellings `term` counts as, for the dedup check.

    "declared" also answers to "declar" and "declare", so a later "declare"
    (or an earlier one) collides with it; "bug" answers only to itself,
    because nothing shorter than {@link _MIN_STEM} letters is a stem.
    """
    words = _fold(term).split()

    if not words:
        return set()

    keys = {" ".join(words)}
    last = words[-1]

    for suffix in _SUFFIXES:
        if last.endswith(suffix) and len(last) - len(suffix) >= _MIN_STEM:
            stem = last[:-len(suffix)]
            keys.add(" ".join([*words[:-1], stem]))

            # ...and the dropped-e spelling, for the suffixes that really
            # drop one: "ranges" → "rang" → "range", "declared" → "declare".
            # NOT for a plain plural, where it would merge "cars" into
            # "care".
            if suffix != "s" and not stem.endswith("e"):
                keys.add(" ".join([*words[:-1], stem + "e"]))

    return keys


def _is_code_identifier(term: str) -> bool:
    """`boolean_expression`, `main()` — a name out of the code, not a word.

    Deliberately narrow: one token carrying an underscore or a call's
    parentheses. Hyphenated and multi-word picks ("non-functional",
    "just-in-time (JIT)") are ordinary English and stay.
    """
    stripped = term.strip()

    return bool(stripped) and " " not in stripped and (
        "_" in stripped or stripped.endswith("()"))


def _page_index(pages: list[tuple[int, str]], term: str) -> int | None:
    """The page `term` first occurs on, or None when it occurs on none.

    Whole-word occurrences win wherever they are ("bug" belongs to the page
    that says *bug*, not to page 0's "debugging"); a substring occurrence is
    the fallback, which is what lets a picked "instruction" ride the page that
    prints "instructions".
    """
    needle = _fold(term)

    if not needle:
        return None

    pattern = re.escape(needle)

    if needle[0].isalnum():
        pattern = r"(?<![0-9a-z])" + pattern

    if needle[-1].isalnum():
        pattern = pattern + r"(?![0-9a-z])"

    whole = re.compile(pattern)
    fallback = None

    for number, text in pages:
        if whole.search(text):
            return number

        if fallback is None and needle in text:
            fallback = number

    return fallback


def _undiacritise(text: str) -> str:
    """`text` through the body translation's own Arabic post-processor.

    The document body carries no tashkeel — babeldoc strips it from every
    translated paragraph — so a strip that does carries it alone. This is the
    same function, not a second rule: the prompt asks, this guarantees.
    """
    try:
        from babeldoc.format.pdf.document_il.midend.il_translator import (
            postprocess_arabic_translation)
    except Exception:  # noqa: BLE001 - a gloss with tashkeel beats no gloss
        logger.exception("vocab: the Arabic post-processor is unavailable; "
                         "meanings ship as the model wrote them")
        return text

    # The source is English, so it never carries diacritics: the strip is
    # unconditional, exactly as it is for a body paragraph.
    return postprocess_arabic_translation("", text)


def _page_texts(sidecar: dict) -> list[tuple[int, str, str]]:
    """Each page as (page_number, SOURCE text, the run's ARABIC rendering).

    Ascending by page number, because the first-occurrence placement walks
    them in reading order. The source is what the words are picked from; the
    translation travels so the model can reuse the document's own wording for
    a term instead of inventing a second one for the same page.
    """
    out: list[tuple[int, str, str]] = []

    for page in sidecar.get("pages") or []:
        if not isinstance(page, dict):
            continue

        number = page.get("page_number")

        if not isinstance(number, int) or isinstance(number, bool) or number < 0:
            continue

        blocks = [block for block in (page.get("blocks") or [])
                  if isinstance(block, dict)]
        body = "\n".join(text for text in
                         (str(block.get("source") or "").strip()
                          for block in blocks) if text)
        rendered = "\n".join(text for text in
                             (str(block.get("target") or "").strip()
                              for block in blocks) if text)

        if body:
            out.append((number, body, rendered))

    out.sort()

    return out


def _chunks(pages: list[tuple[int, str, str]]
            ) -> list[list[tuple[int, str, str]]]:
    """`pages` grouped into calls: whole pages, ~{@link CHUNK_WORDS} each.

    Pages are never split — a page's words belong to one call — so a single
    pathological page larger than the budget still travels, alone. Both the
    source and the translation count against the budget: both are sent.
    """
    chunks: list[list[tuple[int, str, str]]] = []
    current: list[tuple[int, str, str]] = []
    words = 0

    for page in pages:
        count = len(page[1].split()) + len(page[2].split())

        if current and words + count > CHUNK_WORDS:
            chunks.append(current)
            current, words = [], 0

        current.append(page)
        words += count

    if current:
        chunks.append(current)

    return chunks


def build_prompt(chunk: list[tuple[int, str, str]],
                 introduced: list[str]) -> str:
    """One chunk's user message: the already-introduced words, then the pages.

    Each page carries its English text and, when the run produced one, the
    document's own Arabic for that page — the anchor that keeps the strip's
    terminology identical to the body it sits under.

    `introduced` is what keeps the first-occurrence rule across chunks (and
    what excludes the caller's already-explained terms): the model is told,
    not trusted to remember — and the caller deduplicates again anyway.
    """
    parts = []

    if introduced:
        parts.append("شُرحت سابقًا (لا تُعِد أيًّا منها): "
                     + "، ".join(introduced))

    pages = []

    for number, source, rendered in chunk:
        page = [f"— الصفحة {number} —", "الإنجليزية:", source]

        if rendered:
            pages.append("\n".join([*page, "ترجمة المستند لهذه الصفحة:",
                                    rendered]))
        else:
            pages.append("\n".join(page))

    parts.append("نص المستند:\n\n" + "\n\n".join(pages))

    return "\n\n".join(parts)


def _clean_entry(item: Any) -> dict | None:
    """One model item as a stored entry, or None when it is not usable."""
    if not isinstance(item, dict):
        return None

    word = str(item.get("w") or "").strip()
    arabic = _undiacritise(str(item.get("ar") or "").strip()).strip()

    if not word or not arabic:
        return None

    entry = {"w": word, "ar": arabic}
    note = _undiacritise(str(item.get("note") or "").strip()).strip()

    if note:
        entry["note"] = note

    return entry


def parse_response(content: str) -> dict[int, list[dict]]:
    """The model's reply as per-page entries: validated, junk dropped.

    Tolerant of the shapes JSON mode still produces — the pages dict bare
    instead of under "vocab", numeric keys as strings or ints — and strict
    about what is kept: an entry without a word or a meaning is dropped, not
    padded. A key that is not a page number at all takes its entries with it,
    and says so in the log: a page of vocabulary must never disappear quietly.
    Deduplication, placement and the caps are the caller's job, because they
    run ACROSS chunks.
    """
    data = json.loads(content)

    if isinstance(data, dict) and isinstance(data.get("vocab"), dict):
        data = data["vocab"]

    if not isinstance(data, dict):
        raise ValueError(f"response carries no vocab pages: {type(data).__name__}")

    out: dict[int, list[dict]] = {}

    for key, items in data.items():
        try:
            number = int(key)
        except (TypeError, ValueError):
            number = -1

        if number < 0 or not isinstance(items, list):
            logger.warning("vocab: dropping %s item(s) under the unusable "
                           "page key %r", len(items) if isinstance(items, list)
                           else 0, key)
            continue

        entries = [entry for entry in map(_clean_entry, items)
                   if entry is not None]

        if entries:
            out[number] = entries

    return out


def _client():
    from openai import OpenAI

    return OpenAI(base_url=config.OPENAI_BASE_URL,
                  api_key=config.OPENAI_API_KEY, timeout=180, max_retries=1)


def _log_shared_meanings(result: dict[str, list[dict]]) -> None:
    """Say when one page explains two words with the identical Arabic.

    The prompt makes `note` mandatory in exactly this case; when it comes back
    without one the reader is left with two rows that read the same, so the
    gap is worth a line in the log even though it is not worth dropping a word
    over.
    """
    for number, rows in result.items():
        meanings: dict[str, list[str]] = {}

        for row in rows:
            if not row.get("note"):
                meanings.setdefault(row["ar"], []).append(row["w"])

        for arabic, words in meanings.items():
            if len(words) > 1:
                logger.warning("vocab: page %s explains %s with the same "
                               "unqualified meaning %r", number,
                               " / ".join(words), arabic)


def extract_vocab(sidecar: dict, exclude: list[str] | tuple[str, ...] = (),
                  client=None) -> dict[str, list[dict]]:
    """Each page's new vocabulary, explained — or `{}` and a log line.

    Returns `{"<page_number>": [{w, ar, note?}, ...]}` with STRING keys — the
    exact shape stored in the sidecar's "vocab" key — where the page number is
    the one the word FIRST OCCURS on in the sidecar's source text, not the key
    the model returned. `exclude` is the caller's already-explained terms (a
    sidecar's deep-terms glossary, when one exists): they have their own pages,
    so they never appear here. `{}` is BOTH the "nothing new to explain" answer
    and the failure answer, deliberately — the callers insert no pages either
    way, and the distinction lives in the log, never in a failed job.
    """
    if not config.VOCAB_PAGES:
        return {}

    try:
        if not config.OPENAI_API_KEY and client is None:
            logger.warning("vocab: OPENAI_API_KEY is not configured; skipping")
            return {}

        pages = _page_texts(sidecar)

        if not pages:
            return {}

        sources = [(number, _fold(source)) for number, source, _t in pages]
        client = client or _client()
        excluded = [str(term).strip() for term in exclude if str(term).strip()]
        seen: set[str] = set()

        for term in excluded:
            seen |= _family(term)

        introduced = list(excluded)
        result: dict[str, list[dict]] = {}
        total = moved = absent = code = 0

        for chunk in _chunks(pages):
            if total >= MAX_TOTAL:
                break

            try:
                response = client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",
                         "content": build_prompt(chunk, introduced)},
                    ],
                    response_format={"type": "json_object"},
                )
                parsed = parse_response(
                    response.choices[0].message.content or "")
            except Exception:  # noqa: BLE001 - keep what earlier chunks made
                logger.exception("vocab: a chunk's extraction failed; keeping "
                                 "the %s word(s) already selected", total)
                break

            # Placed by the text, not by the key the model returned — then
            # ascending, so an over-budget document keeps its EARLIEST words.
            placed: list[tuple[int, dict]] = []

            for number in sorted(parsed):
                for entry in parsed[number]:
                    if _is_code_identifier(entry["w"]):
                        logger.info("vocab: %r is a code identifier; dropped",
                                    entry["w"])
                        code += 1
                        continue

                    page = _page_index(sources, entry["w"])

                    if page is None:
                        logger.info("vocab: %r occurs nowhere in the source; "
                                    "dropped (model said page %s)",
                                    entry["w"], number)
                        absent += 1
                        continue

                    if page != number:
                        moved += 1

                    placed.append((page, entry))

            placed.sort(key=lambda item: item[0])

            for page, entry in placed:
                if total >= MAX_TOTAL:
                    break

                kept = result.setdefault(str(page), [])

                if len(kept) >= MAX_PER_PAGE:
                    continue

                keys = _family(entry["w"])

                if keys & seen:
                    continue

                seen |= keys
                introduced.append(entry["w"])
                kept.append(entry)
                total += 1

        result = {number: rows for number, rows in result.items() if rows}

        if moved or absent or code:
            logger.info("vocab: %s word(s) placed on the page they first "
                        "occur on rather than the key returned; %s dropped as "
                        "absent from the document, %s as code identifiers",
                        moved, absent, code)

        _log_shared_meanings(result)
        logger.info("vocab: %s word(s) over %s page(s)", total, len(result))

        return result
    except Exception:  # noqa: BLE001 - vocab must never fail the run
        logger.exception("vocab: extraction failed; the document ships "
                         "without vocab pages")
        return {}
