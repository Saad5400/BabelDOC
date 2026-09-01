from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from string import Template

import tiktoken
from tqdm import tqdm

import babeldoc.format.pdf.document_il.il_version_1 as il_version_1
from babeldoc.babeldoc_exception.BabelDOCException import ContentFilterError
from babeldoc.format.pdf.document_il import Document
from babeldoc.format.pdf.document_il import GraphicState
from babeldoc.format.pdf.document_il import Page
from babeldoc.format.pdf.document_il import PdfFont
from babeldoc.format.pdf.document_il import PdfFormula
from babeldoc.format.pdf.document_il import PdfParagraph
from babeldoc.format.pdf.document_il import PdfParagraphComposition
from babeldoc.format.pdf.document_il import PdfSameStyleCharacters
from babeldoc.format.pdf.document_il import PdfSameStyleUnicodeCharacters
from babeldoc.format.pdf.document_il import PdfStyle
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.layout_helper import get_char_unicode_string
from babeldoc.format.pdf.document_il.utils.layout_helper import get_paragraph_unicode
from babeldoc.format.pdf.document_il.utils.layout_helper import is_same_style
from babeldoc.format.pdf.document_il.utils.layout_helper import (
    is_same_style_except_font,
)
from babeldoc.format.pdf.document_il.utils.layout_helper import (
    is_same_style_except_size,
)
from babeldoc.format.pdf.document_il.utils.paragraph_helper import (
    is_placeholder_only_paragraph,
)
from babeldoc.format.pdf.document_il.utils.paragraph_helper import (
    is_pure_numeric_paragraph,
)
from babeldoc.format.pdf.document_il.utils.style_helper import GRAY80
from babeldoc.format.pdf.translation_config import TitleContextSnapshot
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.translator.translator import BaseTranslator
from babeldoc.utils.priority_thread_pool_executor import PriorityThreadPoolExecutor

logger = logging.getLogger(__name__)


ARABIC_STYLE_ADDENDUM = """
### Arabic Style Rules (target = Arabic)
- CONSISTENCY IS THE FIRST PRIORITY. Within one document a given English term MUST map to exactly ONE Arabic rendering. If a term appears in the Context block or the Glossary, reuse that exact wording character-for-character; never re-word, re-order, or re-inflect a heading, running title, footer, chapter label or citation you have already seen.
- Write clear, modern technical Arabic (فصحى معاصرة) in the register of a good university textbook: direct and natural, never bureaucratic, archaic, or stiffly literal.
- ABSOLUTELY NO tashkeel/diacritics (فتحة، ضمة، كسرة، سكون، شدة، تنوين) unless the source text itself is diacritized. Write «أنظمة التشغيل - امتحان نهائي», NEVER «أَنْظِمَةُ التَّشْغِيلِ - اِمْتِحَانٌ نِهَائِيٌّ». This applies to titles, headers, and short labels too.
- For EVERY technical term, apply this decision procedure (the word examples below are illustrations of the procedure, not an enumeration):
  (a) If a widely-used, natural Arabic term exists, use it (e.g. algorithm → «خوارزمية», array → «مصفوفة», variable → «متغير»).
  (b) If the best Arabic rendering would be unnatural, ambiguous, or would change the meaning, TRANSLITERATE the English term into Arabic script following how Arabic-speaking practitioners of the field actually say it (e.g. method → «ميثود», plural «ميثودات»; class → «كلاس»; object → «أوبجكت»; constructor → «كونستركتور»). NEVER force a semantically wrong Arabic word just to avoid transliteration (e.g. «أساليب» for a Java method is WRONG; «ميثود» is right). These transliterations apply ONLY when the word carries its programming sense — a Java method, an OOP object, a class definition. When the SAME English word carries its ordinary sense, translate it normally: research «methods» → «مناهج/أساليب», physical «objects» in a simulation → «أجسام/عناصر», a school «class» → «صف», a data «abstraction» → «تجريد للبيانات» (never «بياني», which means graphical). Read the sentence before deciding.
  (c) Keep Latin script ONLY for: acronyms (API, SGD, CI/CD), code identifiers and version strings, and product/tool/proper names (Git, GitHub, Docker, DevOps). Names are NEVER transliterated and NEVER translated: person names (Ian Sommerville, Silberschatz, Zelle), publisher and company names (ELSEVIER, MIT Press, Sun Microsystems), product/tool/language names (Java, Python, Git, Docker, Blackboard), and the TITLES of cited books, papers and courses. A bibliography or reference-list entry keeps its author, title, publisher and edition exactly as printed — the reader must be able to search for it. Write «Java», never «جافا»; «Git», never «جيت»; «DevOps», never «ديف أوبس»; «Java Virtual Machine» or «آلة Java الافتراضية», never «جافا فيرتشوال ماشين» and never «بيئة تشغيل Java» (that is the JRE, not the JVM). Transliteration under (b) is for common-noun jargon used inside Arabic sentences, not for names.
- When the source sentence introduces or defines a term that stays in Latin script under (c), let the surrounding Arabic explanation carry the meaning, e.g. «الـ API هو الواجهة التي تتخاطب عبرها البرامج».
- The detached «الـ X» form (definite article + tatweel + space) is ONLY correct before a LATIN-script word: «الـ API», «الـ JVM», «الـ CPU». Before an ARABIC word the article is ATTACHED with no space and no tatweel: write «الخوارزمية», «البرنامج», «الكلاس», «المخرجات» — NEVER «الـ خوارزمية». The same holds for the prefixes بـ / لـ / كـ / فـ / وـ: write «بالبنية», «لترجمة», «للقضايا» — never «بـ البنية», «لـ ترجمة».
- Translate short standalone headers and labels into Arabic (CONTENTS → المحتويات, TOOL → الأداة, JOB → المهمة, VS → مقابل); do not leave small English UI-style labels untranslated unless they are proper nouns, code, or established technical terms as above. A single ordinary English word on a diagram or in a table cell is still content and MUST be translated: "and" → «و», "one" → «واحد», "more" → «المزيد», "file" → «ملف», "disk" → «قرص», "done" → «تم», "user" → «مستخدم», "idle" → «خامل», "Mass" → «الكتلة», "Time" → «الزمن», "code" → «شيفرة». Returning the input unchanged is only for genuine code: identifiers with dots/underscores/parentheses/camelCase, language keywords, file names, and acronyms.
- Inside ARABIC PROSE, keep one space on each side of an inline symbol that connects Arabic words: write «التعلم = تحسين الأداء», never «التعلم=تحسين الأداء». This rule NEVER applies inside a URL, a file path, a code fragment, an identifier, or a mathematical expression: leave `?from_search=16`, `boolean_expression`, `aload_0`, `String[] args`, `SET>java`, `~calvanese` and `[-(2^(N-1)-1)]` character-for-character unchanged. A URL is copied verbatim — never localise its path (`…-misses-the-point.en.html` stays `.en.html`).
- Prefer natural technical wording over dictionary-literal calques: pitfalls → «أخطاء شائعة» (not «مآزق»), overview → «نظرة عامة», best practices → «أفضل الممارسات», garbage collection → «جمع المهملات» (not «جمع القمامة»).
- Numbers, code, identifiers, and version strings stay exactly as in the source.
- Mathematical and scientific notation is COPIED, never re-typeset. Reproduce it byte-for-byte: numbers and their decimal points (5.98 stays «5.98», never «5. 98»), signs bound to their operand (-5 stays «-5», never «- 5»), exponents and subscripts, unit SYMBOLS (m, cm, kg, ns, m/s, km/h stay Latin — do NOT write «م/سم/كجم»; the unit NAME in running prose may be translated, "one metre" → «متر واحد»), variable letters (p, q, r, A, a, i, n, N stay Latin — never render them as «أ ي ن»), logical and mathematical operators (∧ ∨ ¬ ⊕ → ↔ ⟹ Σ × ÷ ± stay exactly as written — never swap ∧ for ∨), and slide or page fractions such as (5/23), which keep their original digit order and are NEVER reversed to (23/5). If a formula in the source is garbled, OCR-damaged or unreadable, COPY IT UNCHANGED — do not guess what it was meant to be.
- Add the English original in parentheses at most ONCE per paragraph, and only for a term the paragraph itself introduces or defines: «ميثود (method)». Do not re-gloss a term you have already glossed; the English is available to the reader in the side-by-side and interlinear layouts.
- The output may contain ONLY Arabic script, Latin script, digits, and standard punctuation. NEVER emit Chinese/CJK or any other script.
"""

# Characters stripped by the Arabic post-processor when the source has no diacritics:
# Arabic tashkeel and Quranic annotation marks.
_ARABIC_DIACRITICS_RE = re.compile(
    "[\u064b-\u0655\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed]"
)
# Normalize spacing around inline connectors (= + arrows) that touch Arabic text.
_ARABIC_OP_CANDIDATE_RE = re.compile("[ \t]*([=+\u2192\u2190])[ \t]*")
_OP_CHARS = set("=+<>-\u2192\u2190")
# Literal HTML tag pairs occasionally emitted by the model as text (e.g.
# <code>...</code> wrapping a translated fragment, stray <p> tags). The name
# list deliberately excludes placeholder shapes: <style id='N'>...</style> is
# not listed, and <b1>/<b12> placeholders fail the (?![0-9A-Za-z]) guard.
_STRAY_HTML_TAG_RE = re.compile(
    r"</?(?:strong|span|code|pre|sub|sup|em|br|b|i|u|p)(?![0-9A-Za-z])\s*/?>",
    re.IGNORECASE,
)
# CJK detection for guarding Arabic output against script leakage.
CJK_CHARS_RE = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def _is_arabic_char(ch: str) -> bool:
    return bool(ch) and "\u0600" <= ch <= "\u06ff"


def _is_arabic_lang(lang: str | None) -> bool:
    if not lang:
        return False
    lang = lang.lower()
    return lang.startswith("ar") or "arab" in lang


def _has_arabic_diacritics(text: str) -> bool:
    return bool(_ARABIC_DIACRITICS_RE.search(text or ""))


def _fix_arabic_operator_spacing(text: str) -> str:
    def repl(m: "re.Match[str]") -> str:
        op = m.group(1)
        before = text[m.start() - 1] if m.start() > 0 else ""
        after = text[m.end()] if m.end() < len(text) else ""
        # Leave multi-character operators (==, <=, ->, etc.) alone.
        if before in _OP_CHARS or after in _OP_CHARS:
            return m.group(0)
        if _is_arabic_char(before) or _is_arabic_char(after):
            left = " " if before else ""
            right = " " if after else ""
            return f"{left}{op}{right}"
        return m.group(0)

    return _ARABIC_OP_CANDIDATE_RE.sub(repl, text)


# A parenthetical whose content is pure Latin: the terminology rule's
# «مصطلح (English)» gloss. Wanted in running text, ruinous on a diagram
# label — the label box is the drawn shape it sits in, and the suffix can
# triple its width.
_LATIN_GLOSS_RE = re.compile(r"\s*[(（]\s*[A-Za-z][A-Za-z0-9 .&/'’-]*\s*[)）]")
# Same shape, capturing the English inside, for the run-scoped de-duplicator.
_LATIN_GLOSS_CAPTURE_RE = re.compile(
    r"\s*[(（]\s*([A-Za-z][A-Za-z0-9 .&/'’-]*?)\s*[)）]"
)

# «الـ خوارزمية» -> «الخوارزمية». The tatweel-detached article/prefix form is
# correct ONLY before a Latin word («الـ API»), which the Arabic-letter
# lookahead excludes by construction. The captured group is the single letter
# carrying the tatweel: the ل of «الـ», or a bare بـ/لـ/كـ/فـ/وـ prefix.
_DETACHED_ARABIC_PREFIX_RE = re.compile(
    "([الوفبك])ـ[ \t\u00a0]+"
    "(?=[ء-غف-ي])"
)

# Characters that attract stray whitespace when the model re-typesets an
# identifier, URL, decimal or signed number as if it were Arabic prose.
_SPACING_MAGNETS = set("=_~<>[](){}/\\|+-.,:;*^&#!?'\"@$%")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'()؀-ۿ]+")
_BRACKET_FRACTION_RE = re.compile(r"\((\d+)/(\d+)\)")


def _fix_arabic_detached_prefix(text: str) -> str:
    """Re-attach «الـ / بـ / لـ / كـ / فـ / وـ» to a following ARABIC word.

    62 blocks in the shipped corpus read «الـ خوارزمية» / «بـ البنية» — a
    spelling error, and the single most visible one on a rendered page. The
    Latin form («الـ API») is untouched: the lookahead requires an Arabic
    letter.
    """
    return _DETACHED_ARABIC_PREFIX_RE.sub(r"\1", text)


def _protected_source_tokens(source_text: str) -> list[str]:
    """Whitespace-separated source tokens that must survive character-for-character.

    A token qualifies when it mixes alphanumerics with at least one character
    that attracts stray spacing (``_ . - = ~ [] > /``): decimals (``5.98``),
    signed numbers (``-6``), identifiers (``boolean_expression``, ``aload_0``),
    query strings (``?from_search=16``) and shell fragments (``SET>java``).
    """
    tokens = []
    for token in source_text.split():
        if len(token) < 2:
            continue
        if not any(ch.isalnum() for ch in token):
            continue
        if not any(ch in _SPACING_MAGNETS for ch in token):
            continue
        tokens.append(token)
    # Longest first: restoring «boolean_expression» before «_expression»
    # avoids a partial repair leaving the rest mangled.
    return sorted(set(tokens), key=len, reverse=True)


def _spaced_variant_pattern(token: str) -> "re.Pattern[str] | None":
    """Regex matching ``token`` with optional whitespace injected at its seams.

    Whitespace is allowed only BETWEEN characters and only where one side is a
    spacing magnet, so the pattern can never eat the space that follows a list
    marker like ``1.``.
    """
    parts = []
    for i, ch in enumerate(token):
        parts.append(re.escape(ch))
        if i + 1 < len(token) and (
            ch in _SPACING_MAGNETS or token[i + 1] in _SPACING_MAGNETS
        ):
            parts.append("[ \t\u00a0]*")
    if "[ \t\u00a0]*" not in parts:
        return None
    try:
        return re.compile("".join(parts))
    except re.error:  # pragma: no cover - defensive
        return None


def restore_source_fidelity(source_text: str, translated_text: str) -> str:
    """Undo re-typesetting of things the model was supposed to copy verbatim.

    Three shipped defect classes, all fixable without asking the model again:

    * ``5.98`` -> «5. 98», ``-6`` -> «- 6», ``?from_search=16`` ->
      «?from _ search =16», ``String[] args`` -> «String [] args»: the source
      token is restored wherever a whitespace-injected variant of it appears.
    * ``(5/23)`` -> «(23/5)»: a bracketed slide fraction whose digits were
      swapped is put back in the source's order.
    * a URL whose path was *localised* (``…-misses-the-point.en.html`` ->
      ``.ar.html``) is replaced by the URL as printed in the source.
    """
    if not source_text or not translated_text:
        return translated_text

    for token in _protected_source_tokens(source_text):
        if token in translated_text:
            continue
        pattern = _spaced_variant_pattern(token)
        if pattern is None:
            continue
        translated_text = pattern.sub(lambda _m, t=token: t, translated_text)

    for match in _BRACKET_FRACTION_RE.finditer(source_text):
        original = match.group(0)
        if original in translated_text:
            continue
        reversed_form = f"({match.group(2)}/{match.group(1)})"
        if reversed_form != original and reversed_form in translated_text:
            translated_text = translated_text.replace(reversed_form, original, 1)

    source_urls = _URL_RE.findall(source_text)
    if source_urls:
        for target_url in _URL_RE.findall(translated_text):
            if target_url in source_urls:
                continue
            best = max(
                source_urls,
                key=lambda candidate: len(
                    os.path.commonprefix([candidate, target_url])
                ),
            )
            if len(os.path.commonprefix([best, target_url])) >= 12:
                translated_text = translated_text.replace(target_url, best, 1)

    return translated_text


def dedupe_latin_gloss_parentheticals(text: str, seen: set[str]) -> str:
    """Drop «(English)» glosses whose English term was already glossed this run.

    The style rule asks for the English original on the FIRST occurrence of a
    term, but paragraphs are translated independently so "first" is unknowable
    to the model: «متمم الاثنين (two's complement)» shipped 15 times in one
    document. ``seen`` is the run-scoped memory that makes the rule real. It is
    only updated when the result is actually used, so a refused strip does not
    silently consume a term's one allowed gloss.
    """
    if not text:
        return text
    local_seen = set(seen)
    newly_seen: set[str] = set()

    def repl(match: "re.Match[str]") -> str:
        key = re.sub(r"\s+", " ", match.group(1)).strip().lower()
        if key in local_seen:
            return ""
        local_seen.add(key)
        newly_seen.add(key)
        return match.group(0)

    stripped = re.sub(r"\s{2,}", " ", _LATIN_GLOSS_CAPTURE_RE.sub(repl, text)).strip()
    if stripped and any(_is_arabic_char(ch) for ch in stripped):
        seen |= newly_seen
        return stripped
    return text


def strip_latin_gloss_parentheticals(text: str) -> str:
    """«التغليف (Encapsulation)» -> «التغليف», for image-text labels only.

    The English original is not lost to the reader: it is right there in the
    side-by-side and interlinear layouts, and on the raster itself everywhere
    else. Stripping is refused when it would leave no Arabic (a label the
    model legitimately kept Latin must survive whole).
    """
    stripped = re.sub(r"\s{2,}", " ", _LATIN_GLOSS_RE.sub("", text)).strip()
    if stripped and any(_is_arabic_char(ch) for ch in stripped):
        return stripped
    return text


# Operators whose count must survive translation. Getting one of these wrong
# does not read as a clumsy sentence, it prints a false law: a shipped
# distributive-law slide turned `(p∧r)` into «(p ∨ r)».
_MATH_OPERATORS = "∧∨¬⊕↔⇒⟹≡≠≤≥⊂⊆∈∉∀∃∑∏√±×÷"
_DECIMAL_RE = re.compile(r"\d+\.\d+")


def arabic_math_fidelity_error(source_text: str, translated_text: str) -> str | None:
    """Reason to reject a translation that altered its source's mathematics.

    Checked AFTER :func:`restore_source_fidelity` has had its chance, so this
    only fires on damage no deterministic repair can undo — an operator the
    model resolved to the wrong one, or digits it actually rewrote. Returns
    None when the translation is faithful.
    """
    if not source_text or not translated_text:
        return None
    for operator in _MATH_OPERATORS:
        expected = source_text.count(operator)
        if expected and source_text.count(operator) != translated_text.count(operator):
            return (
                f"math operator '{operator}' count changed "
                f"{expected} -> {translated_text.count(operator)}"
            )
    for decimal in set(_DECIMAL_RE.findall(source_text)):
        if decimal not in translated_text:
            return f"decimal number '{decimal}' lost or rewritten"
    for match in _BRACKET_FRACTION_RE.finditer(source_text):
        if match.group(0) not in translated_text:
            return f"fraction '{match.group(0)}' lost or reordered"
    return None


# Characters that make a token look like code rather than prose.
_CODE_TOKEN_MAGNETS = set("._()[]{}<>/\\=;:#$@&|*+-%\"'`")
# Reserved words: a keyword slide is a table of these, and «عام» for `public`
# would be a worse defect than leaving them in Latin.
_PROGRAMMING_KEYWORDS = frozenset(
    """abstract assert boolean break byte case catch char class const continue
    default do double elif else enum except extends final finally float for from
    goto if implements import instanceof int interface lambda long native new
    none nonlocal null package pass print private protected public raise return
    self short static strictfp super switch synchronized this throw throws
    transient true false try var void volatile while with yield""".split()
)
# Unit symbols: `kg` is not the English word "kg" waiting to be translated.
_UNIT_SYMBOLS = frozenset(
    """m cm mm km nm kg mg s ms ns h min mol rad deg Hz kHz MHz GHz
    J W V K L mL Pa N C F""".split()
)


def is_code_shaped_input(text: str) -> bool:
    """True when the model echoing `text` back unchanged is a legitimate result.

    `Hello.java`, `aload_0`, `String[] args`, `JVM` and `camelCase` are code:
    an identical output is correct. A bare `disk`, `Mass`, `byte` or `and` on a
    diagram is not — those shipped untranslated 80 times because the echo guard
    accepted ANY input of <= 10 tokens without a retry.
    """
    if not any(ch.isalpha() for ch in text):
        # Pure digits and symbols: there is nothing to translate.
        return True
    if any(ch in _MATH_OPERATORS for ch in text):
        # A logic or maths fragment («p ∧ ¬q», «→r»). Retrying one of these is
        # how `A a i n N` became «أ ي ن» on a shipped slide.
        return True
    cores = [token.strip("،,;:.!?«»…") for token in text.split()]
    cores = [core for core in cores if core]
    if not cores:
        return True

    def _is_code_token(core: str) -> bool:
        if any(ch in _CODE_TOKEN_MAGNETS for ch in core):
            return True
        if any(ch.isdigit() for ch in core):
            return True
        if core.isupper():
            # Acronym: API, JVM, CPU. A LONG all-caps word is a shouted header
            # («CONTENTS»), which the style rules say to translate.
            return len(core) <= 6
        if core != core.lower() and core != core.capitalize():
            # camelCase / PascalCase with an inner capital.
            return True
        return False

    if any(_is_code_token(core) for core in cores):
        return True
    if len(cores) > 1 and all(core[:1].isupper() for core in cores):
        # A run of capitalised words is a proper name («Ian Sommerville»,
        # «Sun Microsystems»), which rule (c) keeps in Latin.
        return True
    # A plain English word. Echoing it back is a surrender, not a decision.
    return False


def postprocess_arabic_translation(source_text: str, translated_text: str) -> str:
    """Deterministic safety net for Arabic output quality.

    - Strips tashkeel unless the source itself was diacritized.
    - Re-attaches a detached «الـ / بـ / لـ» to a following Arabic word.
    - Normalizes spacing around inline operators touching Arabic letters.
    - Removes literal HTML tags (<code>, <p>, <b>, ...) occasionally emitted
      by the model as visible text; placeholder shapes are preserved.
    - Restores numbers, identifiers, URLs and slide fractions the model
      re-typeset instead of copying (runs last, so it also undoes any damage
      the operator re-spacer above could have done).
    """
    if not translated_text:
        return translated_text
    if not _has_arabic_diacritics(source_text):
        translated_text = _ARABIC_DIACRITICS_RE.sub("", translated_text)
    translated_text = _fix_arabic_detached_prefix(translated_text)
    translated_text = _fix_arabic_operator_spacing(translated_text)
    translated_text = _STRAY_HTML_TAG_RE.sub("", translated_text)
    translated_text = restore_source_fidelity(source_text, translated_text)
    return translated_text


PROMPT_TEMPLATE = Template(
    """$role_block

## Rules

1. Keep the structure exactly unchanged: do NOT add/remove/reorder any tags, placeholders, or tokens.
2. Keep all tags unchanged (e.g., <style>, <b>, </style>).
   - Translate human-readable text inside tags.
   - Do NOT translate text inside <code>…</code>.
3. Do NOT translate or alter placeholders: {v1}, {name}, %s, %d, [[...]], %%...%%.
4. If the entire input is pure code/identifiers, return it unchanged.
5. Translate ALL human-readable content into $lang_out.
$style_addendum_block
$glossary_block

$context_block

## Output

Output ONLY the translated $lang_out text. No explanations, no backticks, no extra text.

Now translate the following text:

$text_to_translate"""
)


class RichTextPlaceholder:
    def __init__(
        self,
        placeholder_id: int,
        composition: PdfSameStyleCharacters,
        left_placeholder: str,
        right_placeholder: str,
        left_regex_pattern: str = None,
        right_regex_pattern: str = None,
    ):
        self.id = placeholder_id
        self.composition = composition
        self.left_placeholder = left_placeholder
        self.right_placeholder = right_placeholder
        self.left_regex_pattern = left_regex_pattern
        self.right_regex_pattern = right_regex_pattern

    def to_dict(self) -> dict:
        return {
            "type": "rich_text",
            "id": self.id,
            "left_placeholder": self.left_placeholder,
            "right_placeholder": self.right_placeholder,
            "left_regex_pattern": self.left_regex_pattern,
            "right_regex_pattern": self.right_regex_pattern,
            "composition_chars": get_char_unicode_string(self.composition.pdf_character)
            if self.composition and self.composition.pdf_character
            else None,
        }


class FormulaPlaceholder:
    def __init__(
        self,
        placeholder_id: int,
        formula: PdfFormula,
        placeholder: str,
        regex_pattern: str,
    ):
        self.id = placeholder_id
        self.formula = formula
        self.placeholder = placeholder
        self.regex_pattern = regex_pattern

    def to_dict(self) -> dict:
        return {
            "type": "formula",
            "id": self.id,
            "placeholder": self.placeholder,
            "regex_pattern": self.regex_pattern,
            "formula_chars": get_char_unicode_string(self.formula.pdf_character)
            if self.formula and self.formula.pdf_character
            else None,
        }


class PbarContext:
    def __init__(self, pbar):
        self.pbar = pbar

    def __enter__(self):
        return self.pbar

    def __exit__(self, exc_type, exc_value, traceback):
        self.pbar.advance()


class DocumentTranslateTracker:
    def __init__(self):
        self.page = []
        self.cross_page = []
        # Track paragraphs that are combined due to cross-column detection within the same page
        self.cross_column = []

    def new_page(self):
        page = PageTranslateTracker()
        self.page.append(page)
        return page

    def new_cross_page(self):
        page = PageTranslateTracker()
        self.cross_page.append(page)
        return page

    def new_cross_column(self):
        """Create and return a new PageTranslateTracker dedicated to cross-column merging."""
        page = PageTranslateTracker()
        self.cross_column.append(page)
        return page

    def to_json(self):
        pages = []
        for page in self.page:
            paragraphs = self.convert_paragraph(page)
            pages.append({"paragraph": paragraphs})
        cross_page = []
        for page in self.cross_page:
            paragraphs = self.convert_paragraph(page)
            cross_page.append({"paragraph": paragraphs})
        cross_column = []
        for page in self.cross_column:
            paragraphs = self.convert_paragraph(page)
            cross_column.append({"paragraph": paragraphs})
        return json.dumps(
            {
                "cross_page": cross_page,
                "cross_column": cross_column,
                "page": pages,
            },
            ensure_ascii=False,
            indent=2,
        )

    def convert_paragraph(self, page):
        paragraphs = []
        for para in page.paragraph:
            i_str = getattr(para, "input", None)
            o_str = getattr(para, "output", None)
            pdf_unicode = getattr(para, "pdf_unicode", None)
            llm_translate_trackers = getattr(para, "llm_translate_trackers", None)
            placeholders = getattr(para, "placeholders", None)
            original_placeholders = getattr(para, "original_placeholders", None)
            removed_hallucinated_placeholders = getattr(
                para,
                "removed_hallucinated_placeholders",
                None,
            )

            llm_translate_trackers_json = []
            if llm_translate_trackers:
                for tracker in llm_translate_trackers:
                    llm_translate_trackers_json.append(tracker.to_dict())

            placeholders_json = []
            if placeholders:
                for placeholder in placeholders:
                    placeholders_json.append(placeholder.to_dict())

            if pdf_unicode is None or i_str is None:
                continue
            paragraph_json = {
                "input": i_str,
                "output": o_str,
                "pdf_unicode": pdf_unicode,
                "llm_translate_trackers": llm_translate_trackers_json,
                "placeholders": placeholders_json,
                "multi_paragraph_id": getattr(para, "multi_paragraph_id", None),
                "multi_paragraph_index": getattr(para, "multi_paragraph_index", None),
                "original_placeholders": original_placeholders,
                "removed_hallucinated_placeholders": removed_hallucinated_placeholders,
            }
            paragraphs.append(
                paragraph_json,
            )
        return paragraphs


class PageTranslateTracker:
    def __init__(self):
        self.paragraph = []

    def new_paragraph(self):
        paragraph = ParagraphTranslateTracker()
        self.paragraph.append(paragraph)
        return paragraph


class ParagraphTranslateTracker:
    def __init__(self):
        self.llm_translate_trackers = []
        self.original_placeholders: dict[str, int] = {}
        self.removed_hallucinated_placeholders: dict[str, int] = {}

    def set_pdf_unicode(self, unicode: str):
        self.pdf_unicode = unicode

    def set_input(self, input_text: str):
        self.input = input_text

    def set_placeholders(
        self, placeholders: list[RichTextPlaceholder | FormulaPlaceholder]
    ):
        self.placeholders = placeholders

    def set_original_placeholders(self, placeholders: dict[str, int] | None):
        """Record original placeholder-like tokens from the source text."""
        self.original_placeholders = placeholders or {}

    def record_multi_paragraph_id(self, mid):
        self.multi_paragraph_id = mid

    def record_multi_paragraph_index(self, index):
        self.multi_paragraph_index = index

    def set_output(self, output: str):
        self.output = output

    def record_removed_hallucinated_placeholder(self, token: str):
        """Record placeholder-like tokens removed from translated text."""
        if not token:
            return
        self.removed_hallucinated_placeholders[token] = (
            self.removed_hallucinated_placeholders.get(token, 0) + 1
        )

    def new_llm_translate_tracker(self) -> LLMTranslateTracker:
        tracker = LLMTranslateTracker()
        self.llm_translate_trackers.append(tracker)
        return tracker

    def last_llm_translate_tracker(self) -> LLMTranslateTracker | None:
        if self.llm_translate_trackers:
            return self.llm_translate_trackers[-1]
        return None


class LLMTranslateTracker:
    def __init__(self):
        self.input = ""
        self.output = ""
        self.has_error = False
        self.error_message = ""
        self.placeholder_full_match = False
        self.fallback_to_translate = False

    def set_input(self, input_text: str):
        self.input = input_text

    def set_output(self, output_text: str):
        self.output = output_text

    def set_error_message(self, error_message: str):
        self.has_error = True
        self.error_message = error_message

    def set_placeholder_full_match(self):
        self.placeholder_full_match = True

    def set_fallback_to_translate(self):
        self.fallback_to_translate = True

    def to_dict(self):
        return {
            "input": self.input,
            "output": self.output,
            "has_error": self.has_error,
            "error_message": self.error_message,
            "placeholder_full_match": self.placeholder_full_match,
            "fallback_to_translate": self.fallback_to_translate,
        }


class ILTranslator:
    stage_name = "Translate Paragraphs"

    def __init__(
        self,
        translate_engine: BaseTranslator,
        translation_config: TranslationConfig,
        tokenizer=None,
    ):
        self.translate_engine = translate_engine
        self.translation_config = translation_config
        self.font_mapper = FontMapper(translation_config)
        self.shared_context_cross_split_part = (
            translation_config.shared_context_cross_split_part
        )
        if tokenizer is None:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        else:
            self.tokenizer = tokenizer

        # Cache glossaries at initialization
        self._cached_glossaries = (
            self.shared_context_cross_split_part.get_glossaries_for_translation(
                self.translation_config.auto_extract_glossary
            )
        )

        self.support_llm_translate = False
        try:
            if translate_engine and hasattr(translate_engine, "do_llm_translate"):
                translate_engine.do_llm_translate(None)
                self.support_llm_translate = True
        except NotImplementedError:
            self.support_llm_translate = False

        self.use_as_fallback = False
        self.add_content_filter_hint_lock = threading.Lock()
        self.docs = None

        # Final-tier retry configuration: after the primary attempt fails,
        # retry up to 2 more times (with exponential backoff; the last retry
        # uses a simplified prompt).
        self.max_translate_attempts = 3
        # Observability: paragraphs that remain untranslated after ALL tiers,
        # keyed by 0-based page number.
        self.untranslated_by_page: dict[int, int] = {}
        self.untranslated_lock = threading.Lock()

        # Run-scoped consistency memo. babeldoc's own TranslationCache is keyed
        # on the WHOLE prompt (context block, glossary block and all), so the
        # same header on two pages is a cache miss and gets translated twice —
        # 30% of source strings that repeat inside a document came back with
        # more than one Arabic rendering, and one real handout printed the same
        # slide header as «(23/5)» and «(5/23)» side by side. This memo is keyed
        # on the exact source text instead, so a repeat is byte-identical to its
        # first occurrence and, on the fallback path, free.
        self.translation_memo: dict[tuple[str, bool], str] = {}
        self.translation_memo_hits = 0
        self.translation_memo_lock = threading.Lock()
        # Run-scoped memory of «(English)» glosses already emitted, so the
        # style rule's "on the FIRST occurrence" is enforceable at all.
        self.seen_latin_glosses: set[str] = set()

        # Pre-compile patterns for placeholder-like tokens that may be hallucinated by LLM.
        # We only consider the same shapes as our own formula & rich-text placeholders.
        self._formula_placeholder_pattern = re.compile(
            self.translate_engine.get_formular_placeholder(r"\d+")[1], re.IGNORECASE
        )
        self._style_left_placeholder_pattern = re.compile(
            self.translate_engine.get_rich_text_left_placeholder(r"\d+")[1],
            re.IGNORECASE,
        )
        self._style_right_placeholder_pattern = re.compile(
            self.translate_engine.get_rich_text_right_placeholder(r"\d+")[1],
            re.IGNORECASE,
        )

    def calc_token_count(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        except Exception:
            return 0

    def translate(self, docs: Document):
        self.docs = docs
        tracker = DocumentTranslateTracker()

        if not self.translation_config.shared_context_cross_split_part.first_paragraph:
            # Try to find the first title paragraph
            title_paragraph = self.find_title_paragraph(docs)
            self.translation_config.shared_context_cross_split_part.first_paragraph = (
                self.shared_context_cross_split_part.snapshot_title_paragraph(
                    title_paragraph
                )
            )
            self.translation_config.shared_context_cross_split_part.recent_title_paragraph = self.shared_context_cross_split_part.snapshot_title_paragraph(
                title_paragraph
            )
            if title_paragraph:
                logger.info(f"Found first title paragraph: {title_paragraph.unicode}")

        # count total paragraph
        total = sum(len(page.pdf_paragraph) for page in docs.page)
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            total,
        ) as pbar:
            with PriorityThreadPoolExecutor(
                max_workers=self.translation_config.pool_max_workers,
            ) as executor:
                for page in docs.page:
                    self.process_page(page, executor, pbar, tracker.new_page())

        path = self.translation_config.get_working_file_path("translate_tracking.json")

        if (
            self.translation_config.debug
            or self.translation_config.working_dir is not None
        ):
            logger.debug(f"save translate tracking to {path}")
            with Path(path).open("w", encoding="utf-8") as f:
                f.write(tracker.to_json())

        if not self.use_as_fallback:
            untranslated_total = self.report_untranslated()
            if untranslated_total:
                logger.warning(
                    f"Translation completed with {untranslated_total} "
                    f"untranslated paragraph(s)."
                )

    def find_title_paragraph(self, docs: Document) -> PdfParagraph | None:
        """Find the first paragraph with layout_label 'title' in the document.

        Args:
            docs: The document to search in

        Returns:
            The first title paragraph found, or None if no title paragraph exists
        """
        for page in docs.page:
            for paragraph in page.pdf_paragraph:
                if paragraph.layout_label == "title":
                    logger.info(f"Found title paragraph: {paragraph.unicode}")
                    return paragraph
        return None

    def process_page(
        self,
        page: Page,
        executor: PriorityThreadPoolExecutor,
        pbar: tqdm | None = None,
        tracker: PageTranslateTracker = None,
    ):
        self.translation_config.raise_if_cancelled()
        for paragraph in page.pdf_paragraph:
            page_font_map = {}
            for font in page.pdf_font:
                page_font_map[font.font_id] = font
            page_xobj_font_map = {}
            for xobj in page.pdf_xobject:
                page_xobj_font_map[xobj.xobj_id] = page_font_map.copy()
                for font in xobj.pdf_font:
                    page_xobj_font_map[xobj.xobj_id][font.font_id] = font
            # self.translate_paragraph(paragraph, pbar,tracker.new_paragraph(), page_font_map, page_xobj_font_map)
            paragraph_token_count = self.calc_token_count(paragraph.unicode)
            if paragraph.layout_label == "title":
                self.shared_context_cross_split_part.recent_title_paragraph = (
                    self.shared_context_cross_split_part.snapshot_title_paragraph(
                        paragraph
                    )
                )
            executor.submit(
                self.translate_paragraph,
                paragraph,
                page,
                pbar,
                tracker.new_paragraph(),
                page_font_map,
                page_xobj_font_map,
                priority=1048576 - paragraph_token_count,
                paragraph_token_count=paragraph_token_count,
                title_paragraph=self.translation_config.shared_context_cross_split_part.first_paragraph,
                local_title_paragraph=self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
            )

    class TranslateInput:
        def __init__(
            self,
            unicode: str,
            placeholders: list[RichTextPlaceholder | FormulaPlaceholder],
            base_style: PdfStyle = None,
        ):
            self.unicode = unicode
            self.placeholders = placeholders
            self.base_style = base_style
            # Original placeholder-like tokens extracted from the source text.
            # Key: exact matched token string; Value: occurrence count.
            self.original_placeholder_tokens: dict[str, int] = {}

        def set_original_placeholder_tokens(self, tokens: dict[str, int] | None):
            """Attach original placeholder-like tokens from source text."""
            self.original_placeholder_tokens = tokens or {}

        def get_placeholders_hint(self) -> dict[str, str] | None:
            hint = {}
            for placeholder in self.placeholders:
                if isinstance(placeholder, FormulaPlaceholder):
                    cid_count = 0
                    for char in placeholder.formula.pdf_character:
                        if re.match(r"^\(cid:\d+\)$", char.char_unicode):
                            cid_count += 1
                    if cid_count > len(placeholder.formula.pdf_character) * 0.8:
                        continue

                    hint[placeholder.placeholder] = get_char_unicode_string(
                        placeholder.formula.pdf_character
                    )
            if hint:
                return hint
            return None

    def create_formula_placeholder(
        self,
        formula: PdfFormula,
        formula_id: int,
        paragraph: PdfParagraph,
    ):
        placeholder = self.translate_engine.get_formular_placeholder(formula_id)
        if isinstance(placeholder, tuple):
            placeholder, regex_pattern = placeholder
        else:
            regex_pattern = re.escape(placeholder)
        if re.match(regex_pattern, paragraph.unicode, re.IGNORECASE):
            return self.create_formula_placeholder(formula, formula_id + 1, paragraph)

        return FormulaPlaceholder(formula_id, formula, placeholder, regex_pattern)

    def create_rich_text_placeholder(
        self,
        composition: PdfSameStyleCharacters,
        composition_id: int,
        paragraph: PdfParagraph,
    ):
        left_placeholder = self.translate_engine.get_rich_text_left_placeholder(
            composition_id,
        )
        right_placeholder = self.translate_engine.get_rich_text_right_placeholder(
            composition_id,
        )
        if isinstance(left_placeholder, tuple):
            left_placeholder, left_placeholder_regex_pattern = left_placeholder
        else:
            left_placeholder_regex_pattern = re.escape(left_placeholder)
        if isinstance(right_placeholder, tuple):
            right_placeholder, right_placeholder_regex_pattern = right_placeholder
        else:
            right_placeholder_regex_pattern = re.escape(right_placeholder)
        if re.match(
            f"{left_placeholder_regex_pattern}|{right_placeholder_regex_pattern}",
            paragraph.unicode,
            re.IGNORECASE,
        ):
            return self.create_rich_text_placeholder(
                composition,
                composition_id + 1,
                paragraph,
            )

        return RichTextPlaceholder(
            composition_id,
            composition,
            left_placeholder,
            right_placeholder,
            left_placeholder_regex_pattern,
            right_placeholder_regex_pattern,
        )

    def get_translate_input(
        self,
        paragraph: PdfParagraph,
        page_font_map: dict[str, PdfFont] = None,
        disable_rich_text_translate: bool | None = None,
    ):
        if not paragraph.pdf_paragraph_composition:
            return

        # Skip pure numeric paragraphs
        if is_pure_numeric_paragraph(paragraph):
            return None

        # Skip paragraphs with only placeholders
        if is_placeholder_only_paragraph(paragraph):
            return None

        # Extract original placeholder-like tokens from the raw paragraph text
        original_placeholder_tokens: dict[str, int] = {}

        def scan_placeholder_tokens(text: str, tokens: dict[str, int]):
            for pattern in (
                self._formula_placeholder_pattern,
                self._style_left_placeholder_pattern,
                self._style_right_placeholder_pattern,
            ):
                for match in pattern.finditer(text):
                    token = match.group(0)
                    tokens[token] = tokens.get(token, 0) + 1

        if paragraph.unicode:
            scan_placeholder_tokens(paragraph.unicode, original_placeholder_tokens)
        if len(paragraph.pdf_paragraph_composition) == 1:
            # 如果整个段落只有一个组成部分，那么直接返回，不需要套占位符等
            composition = paragraph.pdf_paragraph_composition[0]
            if (
                composition.pdf_line
                or composition.pdf_same_style_characters
                or composition.pdf_character
            ):
                translate_input = self.TranslateInput(
                    paragraph.unicode,
                    [],
                    paragraph.pdf_style,
                )
                translate_input.set_original_placeholder_tokens(
                    original_placeholder_tokens,
                )
                return translate_input
            elif composition.pdf_formula:
                # 不需要翻译纯公式
                return None
            elif composition.pdf_same_style_unicode_characters:
                # DEBUG INSERT CHAR, NOT TRANSLATE
                return None
            else:
                logger.error(
                    f"Unknown composition type. "
                    f"Composition: {composition}. "
                    f"Paragraph: {paragraph}. ",
                )
                return None

        # 如果没有指定 disable_rich_text_translate，使用配置中的值
        if disable_rich_text_translate is None:
            disable_rich_text_translate = (
                self.translation_config.disable_rich_text_translate
            )

        placeholder_id = 1
        placeholders = []
        chars = []
        for composition in paragraph.pdf_paragraph_composition:
            if composition.pdf_line:
                chars.extend(composition.pdf_line.pdf_character)
            elif composition.pdf_formula:
                formula_placeholder = self.create_formula_placeholder(
                    composition.pdf_formula,
                    placeholder_id,
                    paragraph,
                )
                placeholders.append(formula_placeholder)
                # 公式只需要一个占位符，所以 id+1
                placeholder_id = formula_placeholder.id + 1
                chars.extend(formula_placeholder.placeholder)
            elif composition.pdf_character:
                chars.append(composition.pdf_character)
            elif composition.pdf_same_style_characters:
                if disable_rich_text_translate:
                    # 如果禁用富文本翻译，直接添加字符
                    chars.extend(composition.pdf_same_style_characters.pdf_character)
                    continue

                fonta = self.font_mapper.map(
                    page_font_map[
                        composition.pdf_same_style_characters.pdf_style.font_id
                    ],
                    "1",
                )
                fontb = self.font_mapper.map(
                    page_font_map[paragraph.pdf_style.font_id],
                    "1",
                )
                if (
                    # 样式和段落基准样式一致，无需占位符
                    is_same_style(
                        composition.pdf_same_style_characters.pdf_style,
                        paragraph.pdf_style,
                    )
                    # 字号差异在 0.7-1.3 之间，可能是首字母变大效果，无需占位符
                    or is_same_style_except_size(
                        composition.pdf_same_style_characters.pdf_style,
                        paragraph.pdf_style,
                    )
                    or (
                        # 除了字体以外样式都和基准一样，并且字体都映射到同一个字体。无需占位符
                        is_same_style_except_font(
                            composition.pdf_same_style_characters.pdf_style,
                            paragraph.pdf_style,
                        )
                        and fonta
                        and fontb
                        and fonta.font_id == fontb.font_id
                    )
                    # or len(composition.pdf_same_style_characters.pdf_character) == 1
                ):
                    chars.extend(composition.pdf_same_style_characters.pdf_character)
                    continue
                placeholder = self.create_rich_text_placeholder(
                    composition.pdf_same_style_characters,
                    placeholder_id,
                    paragraph,
                )
                placeholders.append(placeholder)
                # 样式需要一左一右两个占位符，所以 id+2
                placeholder_id = placeholder.id + 2
                chars.append(placeholder.left_placeholder)
                chars.extend(composition.pdf_same_style_characters.pdf_character)
                chars.append(placeholder.right_placeholder)
            else:
                logger.error(
                    "Unexpected PdfParagraphComposition type "
                    "in PdfParagraph during translation. "
                    f"Composition: {composition}. "
                    f"Paragraph: {paragraph}. ",
                )
                return None

            # 如果占位符数量超过阈值，且未禁用富文本翻译，则递归调用并禁用富文本翻译
            if len(placeholders) > 40 and not disable_rich_text_translate:
                logger.warning(
                    f"Too many placeholders ({len(placeholders)}) in paragraph[{paragraph.debug_id}], "
                    "disabling rich text translation for this paragraph",
                )
                return self.get_translate_input(paragraph, page_font_map, True)

        text = get_char_unicode_string(chars)
        translate_input = self.TranslateInput(text, placeholders, paragraph.pdf_style)
        translate_input.set_original_placeholder_tokens(original_placeholder_tokens)
        return translate_input

    def process_formula(
        self,
        formula: PdfFormula,
        formula_id: int,
        paragraph: PdfParagraph,
    ):
        placeholder = self.create_formula_placeholder(formula, formula_id, paragraph)
        if placeholder.placeholder in paragraph.unicode:
            return self.process_formula(formula, formula_id + 1, paragraph)

        return placeholder

    def process_composition(
        self,
        composition: PdfSameStyleCharacters,
        composition_id: int,
        paragraph: PdfParagraph,
    ):
        placeholder = self.create_rich_text_placeholder(
            composition,
            composition_id,
            paragraph,
        )
        if (
            placeholder.left_placeholder in paragraph.unicode
            or placeholder.right_placeholder in paragraph.unicode
        ):
            return self.process_composition(
                composition,
                composition_id + 1,
                paragraph,
            )

        return placeholder

    def parse_translate_output(
        self,
        input_text: TranslateInput,
        output: str,
        tracker: ParagraphTranslateTracker | None = None,
        llm_translate_tracker: LLMTranslateTracker | None = None,
    ) -> [PdfParagraphComposition]:
        result = []

        # 如果没有占位符，直接返回整个文本
        if not input_text.placeholders:
            comp = PdfParagraphComposition()
            comp.pdf_same_style_unicode_characters = PdfSameStyleUnicodeCharacters()
            comp.pdf_same_style_unicode_characters.unicode = output
            comp.pdf_same_style_unicode_characters.pdf_style = input_text.base_style
            if llm_translate_tracker:
                llm_translate_tracker.set_placeholder_full_match()
            return [comp]

        # 构建正则表达式模式
        patterns = []
        placeholder_patterns = []
        placeholder_map = {}

        for placeholder in input_text.placeholders:
            if isinstance(placeholder, FormulaPlaceholder):
                # 转义特殊字符
                # pattern = re.escape(placeholder.placeholder)
                pattern = placeholder.regex_pattern
                patterns.append(f"({pattern})")
                placeholder_patterns.append(f"({pattern})")
                placeholder_map[placeholder.placeholder] = placeholder
            else:
                left = placeholder.left_regex_pattern
                right = placeholder.right_regex_pattern
                patterns.append(f"({left}.*?{right})")
                placeholder_patterns.append(f"({left})")
                placeholder_patterns.append(f"({right})")
                placeholder_map[placeholder.left_placeholder] = placeholder
        all_match = True
        for pattern in patterns:
            if not re.search(pattern, output, flags=re.IGNORECASE):
                all_match = False
                break
        if all_match:
            if llm_translate_tracker:
                llm_translate_tracker.set_placeholder_full_match()
        else:
            logger.debug(f"Failed to match all placeholder for {input_text.unicode}")
        # 合并所有模式
        combined_pattern = "|".join(patterns)
        combined_placeholder_pattern = "|".join(placeholder_patterns)
        # Build allowed placeholder tokens: originals from source + placeholders we injected.
        allowed_placeholder_tokens: set[str] = set()
        if getattr(input_text, "original_placeholder_tokens", None):
            allowed_placeholder_tokens.update(input_text.original_placeholder_tokens)
        for placeholder in input_text.placeholders:
            if isinstance(placeholder, FormulaPlaceholder):
                allowed_placeholder_tokens.add(placeholder.placeholder)
            else:
                allowed_placeholder_tokens.add(placeholder.left_placeholder)
                allowed_placeholder_tokens.add(placeholder.right_placeholder)

        def remove_placeholder(text: str):
            """Remove placeholder artifacts and hallucinated placeholder-like tokens."""
            # First, remove any leftover placeholders built from our own regex patterns.
            if combined_placeholder_pattern:
                text = re.sub(
                    combined_placeholder_pattern,
                    "",
                    text,
                    flags=re.IGNORECASE,
                )

            # Then, detect placeholder-like tokens of the same shapes as our own
            # formula and rich-text placeholders. Only keep those in the allowed set.
            def _replace_token(match: re.Match) -> str:
                token = match.group(0)
                if token in allowed_placeholder_tokens:
                    return token
                if tracker is not None:
                    tracker.record_removed_hallucinated_placeholder(token)
                return ""

            text = self._formula_placeholder_pattern.sub(_replace_token, text)
            text = self._style_left_placeholder_pattern.sub(_replace_token, text)
            text = self._style_right_placeholder_pattern.sub(_replace_token, text)
            return text

        # 找到所有匹配
        last_end = 0
        for match in re.finditer(combined_pattern, output, flags=re.IGNORECASE):
            # 处理匹配之前的普通文本
            if match.start() > last_end:
                text = output[last_end : match.start()]
                if text:
                    comp = PdfParagraphComposition()
                    comp.pdf_same_style_unicode_characters = (
                        PdfSameStyleUnicodeCharacters()
                    )
                    comp.pdf_same_style_unicode_characters.unicode = remove_placeholder(
                        text,
                    )
                    comp.pdf_same_style_unicode_characters.pdf_style = (
                        input_text.base_style
                    )
                    result.append(comp)

            matched_text = match.group(0)

            # 处理占位符
            if any(
                isinstance(p, FormulaPlaceholder)
                and re.match(f"^{p.regex_pattern}$", matched_text, re.IGNORECASE)
                for p in input_text.placeholders
            ):
                # 处理公式占位符
                placeholder = next(
                    p
                    for p in input_text.placeholders
                    if isinstance(p, FormulaPlaceholder)
                    and re.match(f"^{p.regex_pattern}$", matched_text, re.IGNORECASE)
                )
                comp = PdfParagraphComposition()
                comp.pdf_formula = placeholder.formula
                result.append(comp)
            else:
                # 处理富文本占位符
                placeholder = next(
                    p
                    for p in input_text.placeholders
                    if not isinstance(p, FormulaPlaceholder)
                    and re.match(
                        f"^{p.left_regex_pattern}", matched_text, re.IGNORECASE
                    )
                )
                text = re.match(
                    f"^{placeholder.left_regex_pattern}(.*){placeholder.right_regex_pattern}$",
                    matched_text,
                    re.IGNORECASE,
                ).group(1)

                if isinstance(
                    placeholder.composition,
                    PdfSameStyleCharacters,
                ) and text.replace(" ", "") == "".join(
                    x.char_unicode for x in placeholder.composition.pdf_character
                ).replace(
                    " ",
                    "",
                ):
                    comp = PdfParagraphComposition(
                        pdf_same_style_characters=placeholder.composition,
                    )
                else:
                    comp = PdfParagraphComposition()
                    comp.pdf_same_style_unicode_characters = (
                        PdfSameStyleUnicodeCharacters()
                    )
                    comp.pdf_same_style_unicode_characters.pdf_style = (
                        placeholder.composition.pdf_style
                    )
                    comp.pdf_same_style_unicode_characters.unicode = remove_placeholder(
                        text,
                    )
                result.append(comp)

            last_end = match.end()

        # 处理最后的普通文本
        if last_end < len(output):
            text = output[last_end:]
            if text:
                comp = PdfParagraphComposition()
                comp.pdf_same_style_unicode_characters = PdfSameStyleUnicodeCharacters()
                comp.pdf_same_style_unicode_characters.unicode = remove_placeholder(
                    text,
                )
                comp.pdf_same_style_unicode_characters.pdf_style = input_text.base_style
                result.append(comp)

        return result

    def pre_translate_paragraph(
        self,
        paragraph: PdfParagraph,
        tracker: ParagraphTranslateTracker,
        page_font_map: dict[str, PdfFont],
        xobj_font_map: dict[int, dict[str, PdfFont]],
    ):
        """Pre-translation processing: prepare text for translation."""
        if paragraph.vertical:
            return None, None
        tracker.set_pdf_unicode(paragraph.unicode)
        if paragraph.xobj_id in xobj_font_map:
            page_font_map = xobj_font_map[paragraph.xobj_id]
        disable_rich_text_translate = (
            self.translation_config.disable_rich_text_translate
        )
        if not self.support_llm_translate:
            disable_rich_text_translate = True

        translate_input = self.get_translate_input(
            paragraph, page_font_map, disable_rich_text_translate
        )
        if not translate_input:
            return None, None
        tracker.set_input(translate_input.unicode)
        tracker.set_placeholders(translate_input.placeholders)
        tracker.set_original_placeholders(
            getattr(translate_input, "original_placeholder_tokens", None),
        )
        text = translate_input.unicode
        if len(text) < self.translation_config.min_text_length:
            logger.debug(
                f"Text too short to translate, skip. Text: {text}. Paragraph id: {paragraph.debug_id}."
            )
            return None, None
        return text, translate_input

    @staticmethod
    def _source_text_of(translate_input) -> str:
        """The exact text that was sent for translation, whatever wraps it."""
        if isinstance(translate_input, str):
            return translate_input
        return getattr(translate_input, "unicode", "") or ""

    def lookup_translation_memo(
        self, source_text: str, is_raster: bool = False
    ) -> str | None:
        """The translation this run already accepted for this exact source.

        Callers may consult this BEFORE spending an LLM call: a repeated
        header, footer or citation is answered for free and, more importantly,
        identically.
        """
        if not source_text:
            return None
        with self.translation_memo_lock:
            return self.translation_memo.get((source_text, bool(is_raster)))

    def _store_translation_memo(
        self, source_text: str, is_raster: bool, translated_text: str
    ) -> None:
        if not source_text or not translated_text:
            return
        with self.translation_memo_lock:
            self.translation_memo.setdefault(
                (source_text, bool(is_raster)), translated_text
            )

    def post_translate_paragraph(
        self,
        paragraph: PdfParagraph,
        tracker: ParagraphTranslateTracker,
        translate_input,
        translated_text: str,
    ):
        """Post-translation processing: update paragraph with translated text.

        The source text is read off the TranslateInput. It used to be read as
        ``translate_input if isinstance(translate_input, str) else ""`` — but
        every caller passes a TranslateInput, so the Arabic post-processor was
        always handed an empty source and the echo check below compared a str
        against an object and could never be true.
        """
        source_text = self._source_text_of(translate_input)
        is_raster = bool(getattr(paragraph, "raster_region", None))

        memoized = self.lookup_translation_memo(source_text, is_raster)
        if memoized is not None:
            # This exact source has already been translated in this run. Reuse
            # that wording verbatim so the document says one thing once.
            translated_text = memoized
            with self.translation_memo_lock:
                self.translation_memo_hits += 1
        else:
            if _is_arabic_lang(self.translation_config.lang_out):
                translated_text = postprocess_arabic_translation(
                    source_text,
                    translated_text,
                )
            if is_raster:
                translated_text = strip_latin_gloss_parentheticals(translated_text)
            elif _is_arabic_lang(self.translation_config.lang_out):
                with self.translation_memo_lock:
                    translated_text = dedupe_latin_gloss_parentheticals(
                        translated_text, self.seen_latin_glosses
                    )
        tracker.set_output(translated_text)
        if translated_text == source_text:
            if llm_translate_tracker := tracker.last_llm_translate_tracker():
                llm_translate_tracker.set_placeholder_full_match()
            return False
        self._store_translation_memo(source_text, is_raster, translated_text)
        paragraph.unicode = translated_text
        paragraph.pdf_paragraph_composition = self.parse_translate_output(
            translate_input,
            translated_text,
            tracker,
            tracker.last_llm_translate_tracker(),
        )
        for composition in paragraph.pdf_paragraph_composition:
            if (
                composition.pdf_same_style_unicode_characters
                and composition.pdf_same_style_unicode_characters.pdf_style is None
            ):
                composition.pdf_same_style_unicode_characters.pdf_style = (
                    paragraph.pdf_style
                )
        return True

    def _build_role_block(self) -> str:
        """Build the role block for LLM prompt.

        Returns:
            Role block string with custom_system_prompt or default role description.
        """
        custom_prompt = getattr(self.translation_config, "custom_system_prompt", None)
        if custom_prompt:
            role_block = custom_prompt.strip()
            if "Follow all rules strictly." not in role_block:
                if not role_block.endswith("\n"):
                    role_block += "\n"
                role_block += "Follow all rules strictly."
        else:
            role_block = (
                f"You are a professional {self.translation_config.lang_out} native translator who needs to fluently translate text "
                f"into {self.translation_config.lang_out}.\n\n"
                "Follow all rules strictly."
            )
        return role_block

    def _build_context_block(
        self,
        title_paragraph: TitleContextSnapshot | None = None,
        local_title_paragraph: TitleContextSnapshot | None = None,
        translate_input: TranslateInput | None = None,
    ) -> str:
        """Build the context/hints block for LLM prompt.

        Args:
            title_paragraph: First title paragraph in the document
            local_title_paragraph: Most recent title paragraph
            translate_input: TranslateInput containing placeholder hints

        Returns:
            Context block string, empty if no context hints available
        """
        context_lines: list[str] = []
        hint_idx = 1

        if title_paragraph:
            context_lines.append(
                f"{hint_idx}. First title in the full text: {title_paragraph.unicode}"
            )
            hint_idx += 1

        if local_title_paragraph:
            is_different_from_global = True
            if title_paragraph:
                if local_title_paragraph.debug_id == title_paragraph.debug_id:
                    is_different_from_global = False

            if is_different_from_global:
                context_lines.append(
                    f"{hint_idx}. The most recent title is: {local_title_paragraph.unicode}"
                )
                hint_idx += 1

        if translate_input and self.translation_config.add_formula_placehold_hint:
            placeholders_hint = translate_input.get_placeholders_hint()
            if placeholders_hint:
                context_lines.append(
                    f"{hint_idx}. Formula placeholder hint:\n{placeholders_hint}"
                )

        if context_lines:
            return "## Context / Hints\n" + "\n".join(context_lines) + "\n"
        return ""

    def _build_glossary_block(self, text: str) -> str:
        """Build the glossary block for LLM prompt.

        Args:
            text: Text to match against glossary entries

        Returns:
            Glossary block string with tables, empty if no active glossary entries
        """
        if not self._cached_glossaries:
            return ""

        glossary_entries_per_glossary: dict[str, list[tuple[str, str]]] = {}

        for glossary in self._cached_glossaries:
            active_entries = glossary.get_active_entries_for_text(text)
            if active_entries:
                glossary_entries_per_glossary[glossary.name] = sorted(active_entries)

        if not glossary_entries_per_glossary:
            return ""

        glossary_block_lines: list[str] = [
            "## Glossary",
            "",
            "Always use the glossary's **Target Term** for any occurrence of its **Source Term** "
            "(including variants, inside tags, or broken across lines).",
            "",
            "Unlisted terms are translated naturally.",
            "",
        ]

        for glossary_name, entries in glossary_entries_per_glossary.items():
            glossary_block_lines.append(f"### Glossary: {glossary_name}")
            glossary_block_lines.append("")
            glossary_block_lines.append(
                "| Source Term | Target Term |\n|-------------|-------------|"
            )
            for original_source, target_text in entries:
                glossary_block_lines.append(f"| {original_source} | {target_text} |")
            glossary_block_lines.append("")

        return "\n".join(glossary_block_lines)

    def generate_prompt_for_llm(
        self,
        text: str,
        title_paragraph: TitleContextSnapshot | None = None,
        local_title_paragraph: TitleContextSnapshot | None = None,
        translate_input: TranslateInput | None = None,
    ):
        """Generate LLM prompt using template-based approach.

        Args:
            text: Text to be translated
            title_paragraph: First title paragraph in the document
            local_title_paragraph: Most recent title paragraph
            translate_input: TranslateInput containing placeholder information

        Returns:
            Final LLM prompt string
        """
        role_block = self._build_role_block()
        context_block = self._build_context_block(
            title_paragraph, local_title_paragraph, translate_input
        )
        glossary_block = self._build_glossary_block(text)

        style_addendum_block = ""
        if _is_arabic_lang(self.translation_config.lang_out):
            style_addendum_block = ARABIC_STYLE_ADDENDUM

        return PROMPT_TEMPLATE.substitute(
            role_block=role_block,
            glossary_block=glossary_block,
            context_block=context_block,
            style_addendum_block=style_addendum_block,
            lang_out=self.translation_config.lang_out,
            text_to_translate=text,
        )

    def add_content_filter_hint(self, page: Page, paragraph: PdfParagraph):
        with self.add_content_filter_hint_lock:
            new_box = il_version_1.Box(
                x=paragraph.box.x,
                y=paragraph.box.y2,
                x2=paragraph.box.x2,
                y2=paragraph.box.y2 + 1.1,
            )
            page.pdf_paragraph.append(
                self._create_text(
                    "翻译服务检测到内容可能包含不安全或敏感内容，请您避免翻译敏感内容，感谢您的配合。",
                    GRAY80,
                    new_box,
                    1,
                )
            )
            logger.info("success add content filter hint")

    def _create_text(
        self,
        text: str,
        color: GraphicState,
        box: il_version_1.Box,
        font_size: float = 4,
    ):
        style = il_version_1.PdfStyle(
            font_id="base",
            font_size=font_size,
            graphic_state=color,
        )
        return il_version_1.PdfParagraph(
            first_line_indent=False,
            box=box,
            vertical=False,
            pdf_style=style,
            unicode=text,
            pdf_paragraph_composition=[
                il_version_1.PdfParagraphComposition(
                    pdf_same_style_unicode_characters=il_version_1.PdfSameStyleUnicodeCharacters(
                        unicode=text,
                        pdf_style=style,
                        debug_info=True,
                    ),
                ),
            ],
            xobj_id=-1,
        )

    def _build_simple_retry_prompt(self, text: str) -> str:
        """Minimal last-resort prompt used on the final retry tier.

        Drops all structure/rules/glossary blocks: some failures are caused by
        the model choking on the long rule-heavy prompt, so the last attempt
        asks for a bare translation only.
        """
        script_rule = ""
        if _is_arabic_lang(self.translation_config.lang_out):
            script_rule = (
                " The output may contain only Arabic script, Latin script, "
                "digits, and punctuation - never Chinese/CJK characters."
                " Keep person, product and publisher names, code identifiers,"
                " URLs, numbers and mathematical notation in Latin script,"
                " exactly as written - never transliterate or re-space them."
            )
        return (
            f"Translate the following text into {self.translation_config.lang_out}. "
            "Keep any placeholders or tags (e.g. {v1}, <style id='1'>...</style>) "
            f"exactly unchanged.{script_rule} "
            "Output only the translation, nothing else.\n\n"
            f"{text}"
        )

    def _record_untranslated(self, page: Page | None):
        """Count a paragraph that remains untranslated after all retry tiers."""
        page_number = getattr(page, "page_number", None) if page else None
        key = page_number if page_number is not None else -1
        with self.untranslated_lock:
            self.untranslated_by_page[key] = self.untranslated_by_page.get(key, 0) + 1

    def report_untranslated(self) -> int:
        """Log one WARNING per page with untranslated paragraphs.

        Returns the total number of untranslated paragraphs so callers can
        fold it into their final summary line.
        """
        total = 0
        with self.untranslated_lock:
            items = sorted(self.untranslated_by_page.items())
        for page_number, count in items:
            total += count
            page_label = page_number + 1 if page_number >= 0 else "unknown"
            logger.warning(f"UNTRANSLATED page={page_label} paragraphs={count}")
        return total

    def translate_paragraph(
        self,
        paragraph: PdfParagraph,
        page: Page,
        pbar: tqdm | None = None,
        tracker: ParagraphTranslateTracker = None,
        page_font_map: dict[str, PdfFont] = None,
        xobj_font_map: dict[int, dict[str, PdfFont]] = None,
        paragraph_token_count: int = 0,
        title_paragraph: TitleContextSnapshot | None = None,
        local_title_paragraph: TitleContextSnapshot | None = None,
    ):
        """Translate a paragraph using pre and post processing functions.

        Retry ladder: attempt 1 uses the full prompt; on failure (exception or
        the model echoing the source back), retry up to 2 more times with
        exponential backoff; the final retry uses a simplified prompt. If all
        attempts fail, the paragraph is recorded as untranslated so the run
        summary can surface it (instead of silently keeping the source text).
        """
        self.translation_config.raise_if_cancelled()
        with PbarContext(pbar):
            try:
                if self.use_as_fallback:
                    # il translator llm only modifies unicode in some situations
                    paragraph.unicode = get_paragraph_unicode(paragraph)
                # Pre-translation processing
                text, translate_input = self.pre_translate_paragraph(
                    paragraph, tracker, page_font_map, xobj_font_map
                )
                if text is None:
                    return

                # Consistency memo: this exact source was already translated in
                # this run. Reuse it — free, and identical by construction.
                memo_hit = self.lookup_translation_memo(
                    text, bool(getattr(paragraph, "raster_region", None))
                )
                if memo_hit is not None and self.post_translate_paragraph(
                    paragraph, tracker, translate_input, memo_hit
                ):
                    return

                last_error: str | None = None

                for attempt in range(self.max_translate_attempts):
                    self.translation_config.raise_if_cancelled()
                    if attempt > 0:
                        # Exponential backoff with jitter: ~1s, ~2s.
                        time.sleep(2 ** (attempt - 1) + random.uniform(0, 0.5))

                    llm_translate_tracker = tracker.new_llm_translate_tracker()
                    try:
                        # Perform translation
                        if self.support_llm_translate:
                            if attempt >= self.max_translate_attempts - 1:
                                # Last tier: simplified prompt.
                                llm_prompt = self._build_simple_retry_prompt(text)
                            else:
                                llm_prompt = self.generate_prompt_for_llm(
                                    text,
                                    title_paragraph,
                                    local_title_paragraph,
                                    translate_input,
                                )
                            llm_translate_tracker.set_input(llm_prompt)
                            translated_text = self.translate_engine.llm_translate(
                                llm_prompt,
                                rate_limit_params={
                                    "paragraph_token_count": paragraph_token_count
                                },
                            )
                            llm_translate_tracker.set_output(translated_text)
                        else:
                            translated_text = self.translate_engine.translate(
                                text,
                                rate_limit_params={
                                    "paragraph_token_count": paragraph_token_count
                                },
                            )
                        translated_text = re.sub(
                            r"[. 。…，]{20,}", ".", translated_text
                        )

                        # CJK guard (mirrors the batch-path check): never
                        # apply Arabic output contaminated with CJK glyphs;
                        # treat it as a failed attempt so the next tier runs.
                        if _is_arabic_lang(
                            self.translation_config.lang_out
                        ) and CJK_CHARS_RE.search(translated_text):
                            last_error = (
                                "CJK characters leaked into Arabic translation"
                            )
                            llm_translate_tracker.set_error_message(last_error)
                            logger.warning(
                                f"Fallback translation attempt {attempt + 1}/"
                                f"{self.max_translate_attempts}: CJK leak for "
                                f"paragraph {paragraph.debug_id}, retrying."
                            )
                            continue

                        # Mathematics guard: a translation that resolved an
                        # operator to the wrong one, or rewrote a number, is
                        # not a clumsy sentence — it prints a false law. Reject
                        # it while there is still a tier left to try. Checked
                        # against the deterministically repaired text so the
                        # repairable damage (split decimals, reversed slide
                        # fractions) does not burn a retry.
                        if (
                            _is_arabic_lang(self.translation_config.lang_out)
                            and attempt < self.max_translate_attempts - 1
                        ):
                            math_error = arabic_math_fidelity_error(
                                text,
                                postprocess_arabic_translation(text, translated_text),
                            )
                            if math_error:
                                last_error = math_error
                                llm_translate_tracker.set_error_message(last_error)
                                logger.warning(
                                    f"Fallback translation attempt {attempt + 1}/"
                                    f"{self.max_translate_attempts}: {math_error} "
                                    f"for paragraph {paragraph.debug_id}, retrying."
                                )
                                continue

                        # Post-translation processing
                        applied = self.post_translate_paragraph(
                            paragraph, tracker, translate_input, translated_text
                        )
                        if applied:
                            return
                        # Model echoed the source back. That is a legitimate
                        # result only when the input really is code — an
                        # identifier, a keyword, an acronym, a file name. A
                        # bare `disk` or `Mass` on a diagram is a surrender,
                        # and used to be accepted because ANY input of <= 10
                        # tokens skipped the retry.
                        if is_code_shaped_input(
                            text
                        ) or self.translation_config.disable_same_text_fallback:
                            return
                        last_error = "translation result identical to input"
                        llm_translate_tracker.set_error_message(last_error)
                        logger.warning(
                            f"Fallback translation attempt {attempt + 1}/"
                            f"{self.max_translate_attempts} returned source "
                            f"unchanged for paragraph {paragraph.debug_id}."
                        )
                    except ContentFilterError as e:
                        logger.warning(f"ContentFilterError: {e.message}")
                        self.add_content_filter_hint(page, paragraph)
                        return
                    except Exception as e:
                        last_error = str(e)
                        llm_translate_tracker.set_error_message(last_error)
                        logger.warning(
                            f"Fallback translation attempt {attempt + 1}/"
                            f"{self.max_translate_attempts} failed for paragraph "
                            f"{paragraph.debug_id}: {e}"
                        )

                # All retry tiers exhausted: the paragraph keeps its source
                # text. Record it so production output surfaces the loss.
                self._record_untranslated(page)
                logger.warning(
                    f"Paragraph {paragraph.debug_id} remains untranslated after "
                    f"{self.max_translate_attempts} attempts. Last error: {last_error}"
                )
            except Exception as e:
                logger.exception(
                    f"Error translating paragraph. Paragraph: {paragraph.debug_id} ({paragraph.unicode}). Error: {e}. ",
                )
                self._record_untranslated(page)
                # ignore error and continue
                return
