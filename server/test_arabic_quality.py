"""Arabic output quality: the deterministic half of the translation contract.

Every assertion here is a verbatim source→target pair taken from a REAL
production run in the sweep corpus (13 sidecars, 6,106 translated blocks). The
model is not involved: these cover the prompt-independent guarantees — the
post-processor's repairs, the echo/code classifier, the run-scoped consistency
memo, and the glossary file itself — so they run for free and never flake on a
sampling temperature.
"""

import json
import threading
from types import SimpleNamespace

import pytest
from babeldoc.format.pdf.document_il.midend.il_translator import ARABIC_STYLE_ADDENDUM
from babeldoc.format.pdf.document_il.midend.il_translator import ILTranslator
from babeldoc.format.pdf.document_il.midend.il_translator import (
    ParagraphTranslateTracker,
)
from babeldoc.format.pdf.document_il.midend.il_translator import (
    arabic_math_fidelity_error,
)
from babeldoc.format.pdf.document_il.midend.il_translator import (
    count_translation_coverage,
)
from babeldoc.format.pdf.document_il.midend.il_translator import (
    dedupe_latin_gloss_parentheticals,
)
from babeldoc.format.pdf.document_il.midend.il_translator import is_code_shaped_input
from babeldoc.format.pdf.document_il.midend.il_translator import is_untranslated_prose
from babeldoc.format.pdf.document_il.midend.il_translator import (
    postprocess_arabic_translation,
)

# --------------------------------------------------------------------------
# F8 — a detached article is a spelling error the reader sees on every page
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "shipped", "expected"),
    [
        # run14 p1 / p6 / p7 / p27, run14 p8, run44 p9, run38 p12.
        (
            "An algorithm is a set of finite steps",
            "الـ خوارزمية هي مجموعة من الخطوات المحدودة",
            "الخوارزمية هي مجموعة من الخطوات المحدودة",
        ),
        (
            "A program is a set of instructions",
            "الـ برنامج هو مجموعة من التعليمات",
            "البرنامج هو مجموعة من التعليمات",
        ),
        (
            "is called the syntax of the language",
            "تسمى بـ البنية النحوية للغة",
            "تسمى بالبنية النحوية للغة",
        ),
        (
            "The disjunction of propositions p and q",
            "يرمز لـ فصل القضايا p و q",
            "يرمز لفصل القضايا p و q",
        ),
    ],
)
def test_detached_arabic_article_is_reattached(source, shipped, expected):
    assert postprocess_arabic_translation(source, shipped) == expected


@pytest.mark.parametrize(
    "text",
    [
        "الـ API هو الواجهة التي تتخاطب عبرها البرامج",
        "لـ ترجمة الـ bytecode إلى تعليمات",
        "الـ JVM ينفذ الـ bytecode",
    ],
)
def test_detached_article_before_latin_is_left_alone(text):
    """«الـ API» is the CORRECT form — the rule is Arabic-only by construction."""
    out = postprocess_arabic_translation("some source", text)
    assert "الـ API" in out or "الـ bytecode" in out or "الـ JVM" in out
    assert "الـ ترجمة" not in out


# --------------------------------------------------------------------------
# F4 / F9 — things the model was told to copy and re-typeset instead
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "shipped", "expected"),
    [
        # F4c — run39 physics, 12 blocks. «5. 98» is two numbers, not one.
        ("The mass is 5.98 kg", "الكتلة هي 5. 98 كجم", "الكتلة هي 5.98 كجم"),
        ("1.66054 units", "1. 66054 وحدة", "1.66054 وحدة"),
        # F4d — a sign detached from its operand, 16 blocks.
        ("-6 + 6 equals zero", "- 6 + 6 يساوي صفرا", "-6 + 6 يساوي صفرا"),
        # F4b — run59, 11 blocks: the slide number read backwards.
        (
            "Compound Propositions (5/23)",
            "القضايا المركبة (23/5)",
            "القضايا المركبة (5/23)",
        ),
        ("Propositional Logic (10/13)", "منطق القضايا (13/10)", "منطق القضايا (10/13)"),
        # F9 — identifiers, query strings and shell fragments.
        ("see ?from_search=16", "انظر ?from _ search =16", "انظر ?from_search=16"),
        (
            "the boolean_expression is evaluated",
            "يتم تقييم boolean _ expression",
            "يتم تقييم boolean_expression",
        ),
        (
            "aload_0 then iload_2",
            "aload _0 ثم iload _2",
            "aload_0 ثم iload_2",
        ),
        (
            "public static void main(String[] args)",
            "public static void main(String [] args)",
            "public static void main(String[] args)",
        ),
        ("type SET>java Hello", "اكتب SET > java Hello", "اكتب SET>java Hello"),
        ("visit ~calvanese for notes", "زر ~ calvanese للملاحظات", "زر ~calvanese للملاحظات"),
    ],
)
def test_source_fidelity_is_restored(source, shipped, expected):
    assert postprocess_arabic_translation(source, shipped) == expected


def test_localised_url_is_put_back():
    """run20 p66 shipped a dead link: the model translated a URL PATH."""
    source = (
        "Read https://www.gnu.org/philosophy/open-source-misses-the-point.en.html today"
    )
    shipped = (
        "اقرأ https://www.gnu.org/philosophy/open-source-misses-the-point.ar.html اليوم"
    )
    out = postprocess_arabic_translation(source, shipped)
    assert "open-source-misses-the-point.en.html" in out
    assert ".ar.html" not in out


@pytest.mark.parametrize(
    ("source", "translated"),
    [
        # A numbered list marker must keep the space that follows it.
        ("1. Introduction", "1. مقدمة"),
        ("2. Compound propositions", "2. القضايا المركبة"),
        # Ordinary prose with no protected token must pass through untouched.
        ("Software engineering is an engineering discipline", "هندسة البرمجيات تخصص هندسي"),
    ],
)
def test_repairs_do_not_fire_on_healthy_output(source, translated):
    assert postprocess_arabic_translation(source, translated) == translated


def test_tashkeel_policy_still_holds_and_now_reads_the_real_source():
    """0/6106 body blocks carry tashkeel — keep it that way.

    The source argument used to arrive empty (post_translate_paragraph passed
    ``translate_input if isinstance(translate_input, str) else ""``), so the
    "unless the source itself is diacritized" half of the rule never ran.
    """
    assert postprocess_arabic_translation("Operating Systems", "أَنْظِمَة") == "أنظمة"
    assert postprocess_arabic_translation("قُرْآن", "أَنْظِمَة") == "أَنْظِمَة"


# --------------------------------------------------------------------------
# F4a — mathematics the post-processor cannot repair, only reject
# --------------------------------------------------------------------------


def test_flipped_logic_operator_is_rejected():
    """run22 p35: a distributive-law slide shipped `(p∧r)` as «(p ∨ r)»."""
    assert arabic_math_fidelity_error("(p∧r) ≡ (r∧p)", "(p ∨ r) ≡ (r ∧ p)")


@pytest.mark.parametrize(
    ("source", "translated"),
    [
        ("(p∧r) ≡ (r∧p)", "(p∧r) ≡ (r∧p)"),
        ("The mass is 5.98 kg", "الكتلة هي 5.98 كجم"),
        ("Compound Propositions (5/23)", "القضايا المركبة (5/23)"),
        ("Software engineering", "هندسة البرمجيات"),
    ],
)
def test_faithful_mathematics_is_not_rejected(source, translated):
    assert arabic_math_fidelity_error(source, translated) is None


# --------------------------------------------------------------------------
# F10 — an echoed English word is a surrender, not a decision
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    ["disk", "Mass", "Time", "and", "one", "file", "done", "idle", "CONTENTS"],
)
def test_plain_english_labels_are_not_code(label):
    assert is_code_shaped_input(label) is False


@pytest.mark.parametrize("word", ["byte", "class", "int", "public", "while"])
def test_reserved_words_keep_the_benefit_of_the_doubt(word):
    """A deliberate trade, not an oversight.

    `byte` is both a Java keyword and a diagram label. run14/run38 each carry a
    keyword-table slide of ~50 such blocks; «بايت» there would be a worse and
    far more visible defect than one untranslated label, and leaving it Latin is
    what already ships. The prompt still asks for a translation on attempt 1 —
    this only decides whether an ECHO is worth paying to retry.
    """
    assert is_code_shaped_input(word) is True


@pytest.mark.parametrize(
    "code",
    [
        "Hello.java",
        "aload_0",
        "String[] args",
        "JVM",
        "CI/CD",
        "camelCase",
        "boolean_expression",
        "https://example.org/a",
        "2024",
        "x = y + 1",
        "Ian Sommerville",
        "Sun Microsystems",
        "p",
        "p ∧ ¬q",
        "→r",
        "kg",
    ],
)
def test_real_code_may_legitimately_echo(code):
    assert is_code_shaped_input(code) is True


# --------------------------------------------------------------------------
# F11 — «(two's complement)» fifteen times in one document
# --------------------------------------------------------------------------


def test_latin_gloss_is_kept_once_then_dropped():
    seen: set[str] = set()
    first = dedupe_latin_gloss_parentheticals(
        "المتمم الثنائي (two's complement) يستخدم للأعداد السالبة", seen
    )
    assert "(two's complement)" in first
    second = dedupe_latin_gloss_parentheticals(
        "المتمم الثنائي (two's complement) مرة أخرى", seen
    )
    assert "(two's complement)" not in second
    assert second.startswith("المتمم الثنائي")


def test_gloss_dedup_refuses_to_empty_a_latin_only_label():
    seen = {"jvm"}
    assert dedupe_latin_gloss_parentheticals("(JVM)", seen) == "(JVM)"


# --------------------------------------------------------------------------
# F1 — one source string, one Arabic rendering, for the whole run
# --------------------------------------------------------------------------


class _CountingEngine:
    """Records every prompt it is asked to translate."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def llm_translate(self, prompt, rate_limit_params=None):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


class _Pbar:
    def advance(self, n=1):
        pass


def _bare_translator(engine, source_text):
    """An ILTranslator with only the collaborators these paths actually touch.

    Building a real one would drag in a FontMapper, a TranslationConfig and a
    PDF; none of that is involved in the memo.
    """
    translator = ILTranslator.__new__(ILTranslator)
    translator.translate_engine = engine
    translator.translation_config = SimpleNamespace(
        lang_out="ar",
        raise_if_cancelled=lambda: None,
        disable_same_text_fallback=False,
    )
    translator.support_llm_translate = True
    translator.use_as_fallback = False
    translator.max_translate_attempts = 3
    translator.untranslated_by_page = {}
    translator.untranslated_debug_ids = set()
    translator.untranslated_lock = threading.Lock()
    translator.translation_memo = {}
    translator.translation_memo_hits = 0
    translator.translation_memo_lock = threading.Lock()
    translator.seen_latin_glosses = set()

    translate_input = ILTranslator.TranslateInput(source_text, [])
    translator.pre_translate_paragraph = lambda *a, **k: (source_text, translate_input)
    translator.generate_prompt_for_llm = lambda *a, **k: f"PROMPT::{source_text}"
    translator.parse_translate_output = lambda *a, **k: []
    return translator


def _paragraph():
    return SimpleNamespace(
        debug_id="p",
        unicode="",
        pdf_paragraph_composition=[],
        pdf_style=None,
        raster_region=None,
    )


def test_repeated_source_is_translated_once_and_identically():
    """run59 p8 printed the SAME slide header two ways on one sheet.

    The engine is primed with two DIFFERENT translations of the same string —
    exactly what a non-zero sampling temperature produces. Only the first may
    ever be spent, and only the first may ever be printed.
    """
    header = "Compound Propositions"
    engine = _CountingEngine(["القضايا المركبة", "العبارات المركبة"])
    translator = _bare_translator(engine, header)

    first, second = _paragraph(), _paragraph()
    for paragraph in (first, second):
        translator.translate_paragraph(
            paragraph, page=SimpleNamespace(page_number=0), pbar=_Pbar(),
            tracker=ParagraphTranslateTracker(),
        )

    assert len(engine.prompts) == 1, "the repeat should not have cost a call"
    assert first.unicode == second.unicode == "القضايا المركبة"
    assert translator.translation_memo_hits == 1


def test_memo_does_not_collapse_different_sources():
    engine = _CountingEngine(["الأول"])
    translator = _bare_translator(engine, "Introduction")
    paragraph = _paragraph()
    translator.translate_paragraph(
        paragraph, page=SimpleNamespace(page_number=0), pbar=_Pbar(),
        tracker=ParagraphTranslateTracker(),
    )
    assert translator.lookup_translation_memo("Introduction") == "الأول"
    assert translator.lookup_translation_memo("Conclusion") is None


# --------------------------------------------------------------------------
# F2 / F3 / F6 / F7 — the glossary the corpus actually needs
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cs_glossary():
    from pathlib import Path

    from babeldoc.glossary import Glossary

    from server import config

    return Glossary.from_csv(Path(config.GLOSSARY_PATH), "ar")


@pytest.mark.parametrize(
    ("text", "source_term", "target"),
    [
        # F2 — «بيئة تشغيل Java» is the JRE, shipped as the JVM on a slide
        # teaching the JDK/JRE/JVM distinction.
        (
            "JIT compilers interact with the Java Virtual Machine",
            "Java Virtual Machine",
            "آلة Java الافتراضية (JVM)",
        ),
        # F7 — compile and assemble both shipped as «تجميع».
        ("Compiling and Running a Java Program", "compiling", "ترجمة"),
        ("such as Assembly Language", "assembly language", "لغة التجميع"),
        # F3b — converse / inverse / contrapositive collapsed onto «العكس».
        ("CONVERSE, CONTRAPOSITIVE, AND INVERSE", "converse", "العكس"),
        ("CONVERSE, CONTRAPOSITIVE, AND INVERSE", "inverse", "النقيض"),
        ("CONVERSE, CONTRAPOSITIVE, AND INVERSE", "contrapositive", "عكس النقيض"),
        # F3a — «المنطق القضاياي» and «المنطق القضي» are not Arabic words.
        ("Introduction to Propositional Logic", "propositional logic", "منطق القضايا"),
        # F11 — «(two's complement)» ×15 in run67.
        ("the two's complement representation", "two's complement", "المتمم الثنائي"),
        # F13 — «تأمل اللوحة الأم» for `Consider the desktop`.
        ("Consider the desktop", "desktop", "حاسب مكتبي"),
    ],
)
def test_glossary_pins_the_terms_that_broke(cs_glossary, text, source_term, target):
    entries = dict(cs_glossary.get_active_entries_for_text(text))
    assert entries.get(source_term) == target


@pytest.mark.parametrize(
    ("text", "source_term", "bad_target"),
    [
        # F6 — a Latin target inside Arabic produced «تشغيل عدة thread».
        ("run multiple threads", "thread", "thread"),
        ("Incremental and agile development", "agile", "Agile"),
    ],
)
def test_glossary_rows_no_longer_emit_bare_latin(cs_glossary, text, source_term, bad_target):
    entries = dict(cs_glossary.get_active_entries_for_text(text))
    assert entries[source_term] != bad_target
    assert any("؀" <= ch <= "ۿ" for ch in entries[source_term])


def test_glossary_has_no_duplicate_sources(cs_glossary):
    from pathlib import Path

    from server import config

    rows = Path(config.GLOSSARY_PATH).read_text(encoding="utf-8").splitlines()[1:]
    sources = [row.split(",")[0].strip().lower() for row in rows if row.strip()]
    assert len(sources) == len(set(sources))


# --------------------------------------------------------------------------
# The style addendum's own promises
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "CONSISTENCY IS THE FIRST PRIORITY",  # F1
        "programming sense",  # F5 — method/object/class scoped
        "Names are NEVER transliterated",  # F2 / F12
        "never «جافا»",  # F2
        "NEVER «الـ خوارزمية»",  # F8
        "never re-typeset",  # F4
        "never localise its path",  # F9
        "MUST be translated",  # F10
        "at most ONCE per paragraph",  # F11
    ],
)
def test_style_addendum_carries_each_rule(phrase):
    assert phrase in ARABIC_STYLE_ADDENDUM


def test_gloss_dedup_never_deletes_a_parenthetical_the_source_printed():
    """run38 p29/p30 both print `Java Runtime Environment (JRE)`.

    «(JRE)» there is the author's own text. Treating it as the model's gloss
    and dropping it on the second slide would delete source content.
    """
    seen: set[str] = set()
    source = "Java Runtime Environment (JRE)"
    first = dedupe_latin_gloss_parentheticals("بيئة تشغيل Java (JRE)", seen, source)
    second = dedupe_latin_gloss_parentheticals("بيئة تشغيل Java (JRE)", seen, source)
    assert first == second == "بيئة تشغيل Java (JRE)"


def test_gloss_dedup_still_strips_a_gloss_the_model_invented():
    seen: set[str] = set()
    source = "Bytecodes are portable"
    assert "(bytecode)" in dedupe_latin_gloss_parentheticals(
        "البايت كود (bytecode) قابل للنقل", seen, source
    )
    assert "(bytecode)" not in dedupe_latin_gloss_parentheticals(
        "البايت كود (bytecode) مستقل عن المنصة", seen, source
    )


# --------------------------------------------------------------------------
# F14 — the model echoes the English back, and the engine ships it
# --------------------------------------------------------------------------
#
# On 30 production decks 13-36% of the English prose was delivered
# untranslated, spread over almost every page. Two mechanisms, both here:
#
#   * `is_code_shaped_input` called a whole paragraph "code" as soon as ANY
#     one of its tokens carried a digit, a hyphen or was a lone letter, so the
#     echo of a full English bullet was accepted with no retry and no log;
#   * the batch path skipped the echo guard for anything of <= 10 tokens and
#     threw away `post_translate_paragraph`'s return value, so a short echoed
#     sentence was marked translated.
#
# Every sentence below is verbatim from record 156 or record 138.


ECHOED_PROSE = [
    # 138 p17 — "c++" was the code token that bought the whole sentence.
    "He was unhappy using c++ programming language, so he developed java.",
    # "long-lifetime" and "(10" did it here.
    "Custom software usually has a long-lifetime (10 years or more).",
    # 156 p5 — the bullet glyph alone was enough.
    "• Efficiency is getting the most output from the least amount of input.",
    # 156 p10 — the parenthesised initials.
    "– president, chief executive officer (C E O), managing director, chancellor",
    # 156 p6 — a figure number.
    "Exhibit 1.1 Efficiency, Effectiveness, and Performance in Student Meetings",
    # 138 p26 — a title with a digit glued to a word and a broken hyphen.
    "Step2: Compiling a Java Program into Byte- codes",
    # 138 p30 — prose ABOUT code is still prose.
    "• Unlike the normal compiler, the JIT compiler compiles the code only "
    "when required.",
    # 156 p19/p20 — Title Case long enough to be a heading, not a name.
    "What Factors Are Reshaping and Redefining Management?",
]


@pytest.mark.parametrize("sentence", ECHOED_PROSE)
def test_prose_with_a_stray_code_token_is_still_prose(sentence):
    assert is_code_shaped_input(sentence) is False


@pytest.mark.parametrize(
    "line",
    [
        "iload_2",
        "this.arr;",
        "p ∧ q",
        "https://www.slideshare.net/wso2.org/java-performance-and-profiling",
        "SET>java Hello",
        "istore 2goto A iload_2 ireturn",
        # 138 p24 — two product names and two URLs, nothing to translate.
        "• NetBeans (www.netbeans.org) • IntelliJ IDEA (www.jetbrains.com)",
        "String[] args",
    ],
)
def test_a_line_that_is_mostly_code_still_reads_as_code(line):
    assert is_code_shaped_input(line) is True


def _echo_run(source, outputs):
    """Run the single-paragraph ladder against an engine with these outputs."""
    engine = _CountingEngine(outputs)
    translator = _bare_translator(engine, source)
    paragraph = _paragraph()
    translator.translate_paragraph(
        paragraph, page=SimpleNamespace(page_number=0), pbar=_Pbar(),
        tracker=ParagraphTranslateTracker(),
    )
    return translator, engine, paragraph


def test_an_echoed_sentence_is_retried_and_the_retry_says_so():
    source = "He was unhappy using c++ programming language, so he developed java."
    arabic = "لم يكن سعيدا باستخدام لغة c++، لذلك طور java."
    translator, engine, paragraph = _echo_run(source, [source, arabic])

    assert len(engine.prompts) == 2, "the echo must have cost a retry"
    assert "UNCHANGED" in engine.prompts[1], (
        "the retry must name the failure it is retrying"
    )
    assert "ar" in engine.prompts[1], "the retry must name the target language"
    assert paragraph.unicode == arabic
    assert translator.report_untranslated() == 0


def test_a_sentence_that_only_ever_echoes_is_counted_untranslated():
    source = "• Managers must create customer-responsive organizations and teams."
    translator, engine, paragraph = _echo_run(source, [source, source, source])

    assert len(engine.prompts) == 3, "every tier must have been spent"
    assert paragraph.unicode == "", "an echo must never be applied as a translation"
    assert translator.report_untranslated() == 1, (
        "an English paragraph in an Arabic document is untranslated, and the "
        "coverage numbers are what the caller bills on"
    )


def test_a_code_echo_costs_no_retry_and_is_not_untranslated():
    """Otherwise MAX_UNTRANSLATED_RATIO would fail every code-heavy deck."""
    translator, engine, _ = _echo_run("aload_0", ["aload_0"])
    assert len(engine.prompts) == 1
    assert translator.report_untranslated() == 0


# ---------------------------------------------------- the batch path

class _BatchEngine:
    """Returns one canned JSON batch response, then records nothing else."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.prompts: list[str] = []

    def llm_translate(self, prompt, rate_limit_params=None):
        self.prompts.append(prompt)
        return json.dumps(
            [{"id": i, "output": text} for i, text in enumerate(self.outputs)],
            ensure_ascii=False,
        )


class _RecordingExecutor:
    def __init__(self):
        self.submissions: list[dict] = []

    def submit(self, fn, *args, **kwargs):
        self.submissions.append({"fn": fn, "args": args, "kwargs": kwargs})


def _batch_run(source, output):
    """One paragraph through the BATCH loop, with everything else stubbed."""
    from babeldoc.format.pdf.document_il.midend.il_translator import (
        PageTranslateTracker,
    )
    from babeldoc.format.pdf.document_il.midend.il_translator_llm_only import (
        BatchParagraph,
    )
    from babeldoc.format.pdf.document_il.midend.il_translator_llm_only import (
        ILTranslatorLLMOnly,
    )

    engine = _BatchEngine([output])
    il = _bare_translator(_CountingEngine([]), source)

    batch = ILTranslatorLLMOnly.__new__(ILTranslatorLLMOnly)
    batch.translate_engine = engine
    batch.il_translator = il
    batch.translation_config = il.translation_config
    batch.translation_config.add_formula_placehold_hint = False
    batch.total_count = batch.ok_count = batch.fallback_count = 0
    batch.calc_token_count = lambda text: len(text.split())
    batch._build_llm_prompt = lambda **kwargs: "BATCH PROMPT"

    paragraph = _paragraph()
    paragraph.layout_label = "plain text"
    page_tracker = PageTranslateTracker()
    batch_paragraph = BatchParagraph([paragraph], [SimpleNamespace(page_number=0)],
                                     page_tracker)
    executor = _RecordingExecutor()
    batch.translate_paragraph(batch_paragraph, pbar=_Pbar(), executor=executor)
    return batch, executor, paragraph


def test_a_short_echoed_sentence_leaves_the_batch_for_the_retry_ladder():
    """Seven tokens: under the old > 10 gate this shipped as English."""
    source = "What Factors Are Reshaping and Redefining Management?"
    batch, executor, paragraph = _batch_run(source, source)

    assert batch.fallback_count == 1, "the echo must fall back, not count as ok"
    assert batch.ok_count == 0
    assert len(executor.submissions) == 1, "the retry ladder must have been queued"
    assert executor.submissions[0]["kwargs"]["previous_output_echoed"] is True, (
        "the ladder must be told the source was echoed, or it repeats the "
        "prompt that produced the echo"
    )


def test_a_code_echo_stays_in_the_batch():
    batch, executor, _ = _batch_run("System.out.println", "System.out.println")
    assert batch.fallback_count == 0
    assert batch.ok_count == 1
    assert executor.submissions == []


# --------------------------------------------------------------------------
# F15 — "100% translated" for a document a third of which shipped in English
# --------------------------------------------------------------------------
#
# Deck 156 (23 slides) delivered 38 English paragraphs and reported
# `paragraphs_untranslated: 0`. None of them was an echo: 21 died in
# `pre_translate_paragraph` (their text had become formula placeholders
# upstream, so there was "nothing to translate") and 17 in `process_page`'s
# placeholder-only filter. No counter in the engine watches those doors, so
# coverage is taken from the FINISHED DOCUMENT instead: whatever the pipeline
# believes it did, an English paragraph in an Arabic document is untranslated.


@pytest.mark.parametrize(
    "text",
    [
        "• Understanding management offers insights into many organizational aspects.",
        "A manager is someone who works with and through other people.",
        "Exhibit 1.1 Efficiency, Effectiveness, and Performance in Student Meetings",
    ],
)
def test_english_prose_left_in_the_document_is_untranslated(text):
    assert is_untranslated_prose(text, "ar") is True


@pytest.mark.parametrize(
    "text",
    [
        "الكفاءة هي الحصول على أكبر قدر من المخرجات",  # translated
        "مزيج من العربية and some English",  # partly Arabic: the model answered
        "aload_0",  # code
        "System.out.println(args[0]);",
        "1 - 14",  # page furniture
        "JVM",  # too short to carry a verdict
        "",
    ],
)
def test_what_the_census_must_not_call_untranslated(text):
    assert is_untranslated_prose(text, "ar") is False


def test_the_census_abstains_for_a_non_arabic_target():
    """The script test is Arabic-specific; for other targets it says nothing."""
    assert is_untranslated_prose("A manager is someone who works with people.", "fr") is False


def _doc(*texts):
    paragraphs = [
        SimpleNamespace(debug_id=f"p{i}", unicode=text) for i, text in enumerate(texts)
    ]
    return SimpleNamespace(page=[SimpleNamespace(pdf_paragraph=paragraphs)])


def test_coverage_counts_the_paragraphs_that_shipped_in_english():
    docs = _doc(
        "الكفاءة هي الحصول على أكبر قدر من المخرجات",
        "• Understanding management offers insights into many aspects.",
        "System.out.println(args[0]);",
        "الفعالية هي إنجاز الأنشطة",
    )
    assert count_translation_coverage(docs, "ar") == (4, 1)


def test_a_failed_paragraph_is_counted_once_not_twice():
    """The retry ladder's own record and the census are the same paragraph."""
    docs = _doc("• Managers must create customer-responsive teams.")
    assert count_translation_coverage(docs, "ar", {"p0"}) == (1, 1)


def test_a_short_paragraph_the_ladder_gave_up_on_still_counts():
    """Too short for the script test to judge, but the ladder saw it fail."""
    docs = _doc("disk")
    assert count_translation_coverage(docs, "ar") == (1, 0)
    assert count_translation_coverage(docs, "ar", {"p0"}) == (1, 1)
