"""Term extraction for the «شرح المصطلحات» pages.

After a mono run finishes, ONE chat call over the run's sidecar picks the
genuinely difficult English terms in the document — the ones an Arabic-speaking
CS student would stumble on ("Wrapping", "Overloading"), never every technical
word — and writes a short, friendly Arabic explanation for each. The entries
land in the sidecar under a top-level "glossary" key, and
`server/glossary_pages.py` renders them as styled pages appended to the output.

This is NOT babeldoc's automatic_term_extractor: that builds a
translation-consistency glossary (term -> fixed rendering) to keep a long
document consistent. This pass produces EXPLANATIONS for a reader, chosen for
difficulty, capped at {@link MAX_TERMS}, and allowed to come back empty.

Failure posture: a translation must NEVER fail — or even degrade — because of
this pass. Every failure mode (endpoint down, malformed JSON, a response that
is not the shape asked for) logs and returns [], which every caller reads as
"no glossary pages".
"""

from __future__ import annotations

import json
import logging
from typing import Any

from server import config

logger = logging.getLogger("doctranslate.terms")

# Hard cap on entries, whatever the model says. Ten cards fill roughly one
# appended page; more stops being "the hard words" and starts being an index.
MAX_TERMS = 10

# The prompt carries the document's source text; a 200-page deck must not turn
# one extraction call into a maximum-context bill. Pages are dropped WHOLE past
# this budget (never mid-paragraph) and the cut is logged.
MAX_PROMPT_CHARS = 60_000

_SYSTEM_PROMPT = """\
أنت مساعد تعليمي لطلاب علوم الحاسب العرب في السعودية. ستقرأ نص مستند إنجليزي \
تُرجم إلى العربية، وتختار منه المصطلحات الإنجليزية الصعبة فعلًا — الكلمات التي \
يتعثر عندها طالب عربي حتى بعد الترجمة، لأن الكلمة غريبة أو استعمالها في \
البرمجة بعيد عن معناها اليومي (مثل Wrapping أو Overloading).

قواعد الاختيار:
- لا تشرح كل كلمة تقنية؛ الكلمات المألوفة (class, function, loop, variable) لا \
تُذكر إطلاقًا.
- من صفر إلى عشرة مصطلحات. إذا لم يكن في المستند شيء يستحق الشرح فأرجع قائمة \
فارغة — هذا جواب صحيح تمامًا.
- لكل مصطلح اكتب شرحًا ودّيًا من جملة إلى ثلاث جمل بعربية سعودية مبسطة: من أين \
جاءت الكلمة (أصلها أو تشبيه من الحياة اليومية) ثم ماذا تعني في سياق هذا \
المستند بالذات. بدون تشكيل إلا إذا أزال لبسًا.

مثال على الأسلوب المطلوب (لمصطلح Wrapping):
«كلمة wrapping تعني «التغليف»، مأخوذة من wrap أي يُغلِّف — مثل ساندوتش الراب \
المغلَّف. في البرمجة: نغلِّف قيمة بدائية داخل كائن، مثل وضع int داخل Integer.»

أرجع JSON فقط بهذا الشكل:
{"terms": [{"term": "المصطلح بالإنجليزية",
            "arabic": "مقابله العربي القصير",
            "explanation": "الشرح",
            "page": رقم الصفحة (يبدأ من 1) التي ورد فيها,
            "quote": "عبارة قصيرة من النص الأصلي ورد فيها المصطلح"}]}
"""


def build_prompt(sidecar: dict) -> str:
    """The user message: the document's source text, page by page.

    Source text (not the translation) is what the terms live in; page numbers
    are 1-based because that is how the entries cite them («الشريحة 12»).
    """
    parts: list[str] = []
    used = 0

    for page in sidecar.get("pages") or []:
        if not isinstance(page, dict):
            continue

        texts = [str(block.get("source") or "").strip()
                 for block in (page.get("blocks") or [])
                 if isinstance(block, dict)]
        body = "\n".join(text for text in texts if text)

        if not body:
            continue

        number = page.get("page_number")
        label = (number + 1) if isinstance(number, int) else "?"
        part = f"— الصفحة {label} —\n{body}"

        if used + len(part) > MAX_PROMPT_CHARS:
            logger.info("terms: prompt truncated at page %s (%s char budget)",
                        label, MAX_PROMPT_CHARS)
            break

        parts.append(part)
        used += len(part)

    return "نص المستند:\n\n" + "\n\n".join(parts)


def _clean_entry(item: Any) -> dict | None:
    """One model item as a stored entry, or None when it is not usable."""
    if not isinstance(item, dict):
        return None

    term = str(item.get("term") or "").strip()
    arabic = str(item.get("arabic") or "").strip()
    explanation = str(item.get("explanation") or "").strip()

    if not term or not explanation:
        return None

    page = item.get("page")
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        page = None

    quote = str(item.get("quote") or "").strip() or None

    return {"term": term, "arabic": arabic, "explanation": explanation,
            "page": page, "quote": quote}


def parse_response(content: str) -> list[dict]:
    """The model's reply as clean entries: deduplicated, capped, validated.

    Tolerant of the shapes JSON mode still produces in the wild — a bare list
    instead of {"terms": [...]}, junk items in the list — and strict about
    what is kept: an entry without a term or an explanation is dropped, not
    padded.
    """
    data = json.loads(content)

    if isinstance(data, dict):
        items = data.get("terms")
    elif isinstance(data, list):
        items = data
    else:
        items = None

    if not isinstance(items, list):
        raise ValueError(f"response carries no terms list: {type(data).__name__}")

    entries: list[dict] = []
    seen: set[str] = set()

    for item in items:
        entry = _clean_entry(item)

        if entry is None:
            continue

        key = entry["term"].casefold()
        if key in seen:
            continue

        seen.add(key)
        entries.append(entry)

        if len(entries) >= MAX_TERMS:
            break

    return entries


def _client():
    from openai import OpenAI

    return OpenAI(base_url=config.OPENAI_BASE_URL,
                  api_key=config.OPENAI_API_KEY, timeout=180, max_retries=1)


def extract_terms(sidecar: dict, client=None) -> list[dict]:
    """The document's hard terms, explained — or [] and a log line.

    [] is BOTH the "nothing worth explaining" answer and the failure answer,
    deliberately: the callers append no pages either way, and the distinction
    lives in the log, never in a failed job.
    """
    if not config.GLOSSARY_PAGES:
        return []

    try:
        if not config.OPENAI_API_KEY and client is None:
            logger.warning("terms: OPENAI_API_KEY is not configured; skipping")
            return []

        prompt = build_prompt(sidecar)

        if len(prompt) < 80:  # a sidecar with no source text to speak of
            return []

        client = client or _client()
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        entries = parse_response(content)
        logger.info("terms: %s term(s) selected", len(entries))

        return entries
    except Exception:  # noqa: BLE001 - the glossary must never fail the run
        logger.exception("terms: extraction failed; the document ships without "
                         "glossary pages")
        return []
