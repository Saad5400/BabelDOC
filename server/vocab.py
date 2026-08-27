"""Per-page vocabulary extraction for the «كلمات هذه الصفحة» pages.

The document's readers are not native English speakers, and the words that
stop them are mostly not the deep technical terms — they are ordinary English
("declared", "scope", "evolved", "custom", "range of customers") met for the
first time. This pass reads the run's sidecar and picks, for EVERY page, the
English words and short phrases a non-native CS student might not be fully
comfortable with, each with a concise Arabic meaning, rendered by
`server/vocab_pages.py` as a compact page inserted right after the page the
words appeared on. The bar is deliberately LOW: when in doubt a word is
included — only the trivial function words (the, is, of, and…) and pure code
identifiers stay out.

A word is explained at its FIRST occurrence only — a word introduced on page 3
never comes back on page 7 — and the caller may pass an exclusion list of
terms already explained elsewhere (a sidecar's deep-terms "glossary", when one
exists), so no word is ever explained twice.

One JSON-mode chat call covers the whole document; a document whose source text
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
from typing import Any

from server import config

logger = logging.getLogger("doctranslate.vocab")

# Caps, whatever the model says. Twenty rows still sit comfortably on one
# two-column compact page; past ~400 words the layer stops being "this page's
# words" and starts being a dictionary of the language.
MAX_PER_PAGE = 20
MAX_TOTAL = 400

# One call covers this many words of source text; a longer document is split
# into page-aligned chunks so no single call carries a maximum-context bill.
CHUNK_WORDS = 30_000

_SYSTEM_PROMPT = """\
أنت مساعد لغوي لطلاب علوم الحاسب العرب — غير ناطقين بالإنجليزية. ستقرأ نص \
مستند إنجليزي صفحةً صفحة، وتختار من كل صفحة الكلمات والعبارات الإنجليزية \
الجديدة التي قد لا يرتاح لها الطالب ارتياحًا كاملًا — مثل declared أو scope \
أو evolved أو custom أو "range of customers" — لتُشرح له بالعربية بجانب \
الصفحة نفسها.

قواعد الاختيار:
- كن كريمًا في الاختيار: أي كلمة أو عبارة إنجليزية قد يتردد عندها طالب غير \
ناطق بالإنجليزية — ولو قليلًا — تُدرَج. إذا شككت هل يعرفها الطالب تمامًا، \
أدرِجها.
- يُستثنى فقط: الكلمات الوظيفية البديهية (the, is, of, and, a, this, use) — \
هذه لا تُذكر إطلاقًا — ومعرّفات الكود الصِّرفة (أسماء متغيرات ودوال كما وردت \
في الشيفرة).
- لا تشرح الكلمات الواردة في قائمة «شُرحت سابقًا» — شُرحت في مكان آخر من \
المستند.
- الكلمة تُشرح عند أول ظهورها فقط: كلمة ظهرت في صفحة أسبق (أو في قائمة \
«شُرحت سابقًا») لا تُعاد في الصفحات التالية.
- حتى 20 كلمة للصفحة الواحدة؛ صفحة بلا كلمات جديدة تُحذف من الجواب كليًا. \
صفحة نصية عادية فيها كلمات جديدة يُتوقع أن تُخرج 8 كلمات أو أكثر — \
إن وجدت أقل من ذلك فراجع النص مرة أخرى بحثًا عن كلمات فاتتك.

لكل كلمة:
- w: الكلمة أو العبارة كما وردت في النص.
- ar: معناها بالعربية في كلمة إلى خمس كلمات.
- note (اختياري): عبارة توضيحية قصيرة واحدة بالعربية — مثلًا لتمييز المعنى \
المقصود هنا عن المعنى الشائع.

أرجع JSON فقط بهذا الشكل، ومفاتيح الصفحات هي الأرقام المعطاة في النص حرفيًا:
{"vocab": {"3": [{"w": "declared", "ar": "يُصرَّح عنه",
                  "note": "في البرمجة: تعريف المتغير قبل استخدامه"}]}}
"""


def _page_texts(sidecar: dict) -> list[tuple[int, str]]:
    """The sidecar's per-page SOURCE text, keyed by its 0-based page_number.

    Source text, not the translation: the English words are what this pass
    reads. The 0-based numbers are what the prompt shows and what the model
    must echo back as keys — they are the sidecar's own page identifiers, so
    the entries land back on the right page without any off-by-one bookkeeping.
    """
    out: list[tuple[int, str]] = []

    for page in sidecar.get("pages") or []:
        if not isinstance(page, dict):
            continue

        number = page.get("page_number")

        if not isinstance(number, int) or isinstance(number, bool) or number < 0:
            continue

        texts = [str(block.get("source") or "").strip()
                 for block in (page.get("blocks") or [])
                 if isinstance(block, dict)]
        body = "\n".join(text for text in texts if text)

        if body:
            out.append((number, body))

    return out


def _chunks(pages: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """`pages` grouped into calls: whole pages, ~{@link CHUNK_WORDS} each.

    Pages are never split — a page's words belong to one call — so a single
    pathological page larger than the budget still travels, alone.
    """
    chunks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    words = 0

    for number, body in pages:
        count = len(body.split())

        if current and words + count > CHUNK_WORDS:
            chunks.append(current)
            current, words = [], 0

        current.append((number, body))
        words += count

    if current:
        chunks.append(current)

    return chunks


def build_prompt(chunk: list[tuple[int, str]], introduced: list[str]) -> str:
    """One chunk's user message: the already-introduced words, then the pages.

    `introduced` is what keeps the first-occurrence rule across chunks (and
    what excludes the caller's already-explained terms): the model is told,
    not trusted to remember — and the caller deduplicates again anyway.
    """
    parts = []

    if introduced:
        parts.append("شُرحت سابقًا (لا تُعِد أيًّا منها): "
                     + "، ".join(introduced))

    body = "\n\n".join(f"— الصفحة {number} —\n{text}"
                       for number, text in chunk)
    parts.append(f"نص المستند:\n\n{body}")

    return "\n\n".join(parts)


def _clean_entry(item: Any) -> dict | None:
    """One model item as a stored entry, or None when it is not usable."""
    if not isinstance(item, dict):
        return None

    word = str(item.get("w") or "").strip()
    arabic = str(item.get("ar") or "").strip()

    if not word or not arabic:
        return None

    entry = {"w": word, "ar": arabic}
    note = str(item.get("note") or "").strip()

    if note:
        entry["note"] = note

    return entry


def parse_response(content: str) -> dict[int, list[dict]]:
    """The model's reply as per-page entries: validated, junk dropped.

    Tolerant of the shapes JSON mode still produces — the pages dict bare
    instead of under "vocab", numeric keys as strings or ints — and strict
    about what is kept: an entry without a word or a meaning is dropped, not
    padded. Deduplication and the caps are the caller's job, because they run
    ACROSS chunks.
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
            continue

        if number < 0 or not isinstance(items, list):
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


def extract_vocab(sidecar: dict, exclude: list[str] | tuple[str, ...] = (),
                  client=None) -> dict[str, list[dict]]:
    """Each page's new vocabulary, explained — or `{}` and a log line.

    Returns `{"<page_number>": [{w, ar, note?}, ...]}` with STRING keys — the
    exact shape stored in the sidecar's "vocab" key. `exclude` is the caller's
    already-explained terms (a sidecar's deep-terms glossary, when one
    exists): they have their own pages, so they never appear here. `{}` is
    BOTH the "nothing new to explain" answer and the failure answer,
    deliberately — the callers insert no pages either way, and the distinction
    lives in the log, never in a failed job.
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

        client = client or _client()
        excluded = [str(term).strip() for term in exclude if str(term).strip()]
        seen = {term.casefold() for term in excluded}
        introduced = list(excluded)
        result: dict[str, list[dict]] = {}
        total = 0

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

            for number in sorted(parsed):
                kept = result.setdefault(str(number), [])

                for entry in parsed[number]:
                    if total >= MAX_TOTAL or len(kept) >= MAX_PER_PAGE:
                        break

                    key = entry["w"].casefold()

                    if key in seen:
                        continue

                    seen.add(key)
                    introduced.append(entry["w"])
                    kept.append(entry)
                    total += 1

                if not kept:
                    del result[str(number)]

        logger.info("vocab: %s word(s) over %s page(s)", total, len(result))

        return result
    except Exception:  # noqa: BLE001 - vocab must never fail the run
        logger.exception("vocab: extraction failed; the document ships "
                         "without vocab pages")
        return {}
