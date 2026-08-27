"""Tests for phrase-pair alignment (babeldoc/.../midend/phrase_pairs.py and its
capture into the translation sidecar).

Run from the repo root:

    pytest server/test_phrase_pairs.py

No LLM anywhere: the pairs under test are what a model WOULD have returned,
including the shapes a model returns when it misbehaves.
"""

import json
from types import SimpleNamespace

from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.midend import phrase_pairs
from babeldoc.format.pdf.document_il.midend import translation_sidecar
from babeldoc.format.pdf.document_il.midend.il_translator import (
    FormulaPlaceholder,
    ILTranslator,
    RichTextPlaceholder,
)
from babeldoc.format.pdf.document_il.midend.il_translator_llm_only import (
    ILTranslatorLLMOnly,
    PHRASE_PAIRS_PROMPT_BLOCK,
    PROMPT_TEMPLATE,
    pairs_eligible,
    strip_style_placeholders,
)
from babeldoc.format.pdf.document_il.midend.typesetting import reshape_rtl_text

SOURCE = "We can not create two local variables with the same name"
TARGET = "لا يمكننا إنشاء متغيرين محليين بالاسم نفسه"
PAIRS = [
    {"s": "We can not", "t": "لا يمكننا"},
    {"s": "create", "t": "إنشاء"},
    {"s": "two local variables", "t": "متغيرين محليين"},
    {"s": "with the same name", "t": "بالاسم نفسه"},
]
# A monotonic segmentation: each pair sits at the same position in both texts.
IDENTITY = [0, 1, 2, 3]

# The owner's real e2e case: Arabic legitimately REORDERS the sentence — the
# verb phrase «يجب الإعلان عن» comes FIRST in the translation although its
# source phrase is the sentence's second. Pairs are aligned by MEANING and
# listed in SOURCE order; the permutation says where each landed.
REORDER_SOURCE = "A local variable must be declared before it is used."
REORDER_TARGET = "يجب الإعلان عن المتغير المحلي قبل استخدامه."
REORDER_PAIRS = [
    {"s": "A local variable", "t": "المتغير المحلي"},
    {"s": "must be declared", "t": "يجب الإعلان عن"},
    {"s": "before it is used.", "t": "قبل استخدامه."},
]
REORDER_PERM = [1, 0, 2]  # target tile k belongs to pair REORDER_PERM[k]


# --- the prompt asks for pairs, in the shape the parser accepts back ---------


def test_the_prompt_template_carries_the_pairs_block_slot():
    assert "$phrase_pairs_block" in PROMPT_TEMPLATE.template


def test_the_pairs_instruction_teaches_the_contract_with_a_worked_example():
    assert '"want_pairs": true' in PHRASE_PAIRS_PROMPT_BLOCK
    assert "NEVER split inside a word" in PHRASE_PAIRS_PROMPT_BLOCK
    # Alignment is by MEANING, and the worked example is the owner's real
    # reordering case: every phrase of both sides appears verbatim.
    assert "MEANING" in PHRASE_PAIRS_PROMPT_BLOCK
    for pair in REORDER_PAIRS:
        assert pair["s"] in PHRASE_PAIRS_PROMPT_BLOCK
        assert pair["t"] in PHRASE_PAIRS_PROMPT_BLOCK


def test_the_pairs_instruction_covers_the_plain_text_of_styled_inputs():
    # Styled paragraphs carry <style id='N'> wrappers in their input; the
    # phrases must cover the text with those removed, content kept in place.
    assert "PLAIN text" in PHRASE_PAIRS_PROMPT_BLOCK
    assert "<style id='N'>...</style>" in PHRASE_PAIRS_PROMPT_BLOCK
    assert "keeping each tag's inner text in place" in PHRASE_PAIRS_PROMPT_BLOCK


# --- parsing: items with and without pairs -----------------------------------


def test_an_item_with_pairs_yields_them():
    item = {"id": 0, "output": TARGET, "pairs": PAIRS}

    assert phrase_pairs.pairs_from_item(item) == PAIRS


def test_an_item_without_pairs_is_simply_pairless():
    assert phrase_pairs.pairs_from_item({"id": 0, "output": TARGET}) is None


def test_malformed_pairs_are_rejected_at_the_door():
    for bad in (
        "not a list",
        [],
        ["not a dict"],
        [{"s": "hello"}],                     # missing t
        [{"s": "hello", "t": 42}],            # wrong type
        [{"s": "  ", "t": "مرحبا"}],          # blank phrase
        [{"s": "a", "t": "ب"}] * (phrase_pairs.MAX_PAIRS + 1),  # runaway
    ):
        assert phrase_pairs.pairs_from_item({"id": 0, "pairs": bad}) is None


# --- the tiler: target phrases located wherever the translation put them -----


def test_a_monotonic_segmentation_tiles_as_the_identity():
    assert phrase_pairs.tile_permutation(
        [p["t"].split() for p in PAIRS], TARGET.split()) == IDENTITY


def test_a_reordered_translation_tiles_by_meaning():
    assert phrase_pairs.tile_permutation(
        [p["t"].split() for p in REORDER_PAIRS],
        REORDER_TARGET.split()) == REORDER_PERM


def test_phrases_that_do_not_tile_the_text_answer_none():
    for phrases, full in (
        ([["a"], ["b"]], ["a", "c"]),          # wrong token
        ([["a"], ["b"]], ["a", "b", "c"]),     # leftover text
        ([["a", "b"], ["c"]], ["a", "c"]),     # token counts differ
        ([["a"], []], ["a"]),                  # an empty phrase
        ([], ["a"]),                           # no phrases at all
        ([["a"]] * (phrase_pairs.MAX_PAIRS + 1),
         ["a"] * (phrase_pairs.MAX_PAIRS + 1)),  # runaway
    ):
        assert phrase_pairs.tile_permutation(phrases, full) is None


def test_identical_duplicate_phrases_are_assigned_deterministically():
    # Two interchangeable «نعم» phrases: the tiler pins the lowest pair index
    # to the earliest target position, every time.
    assert phrase_pairs.tile_permutation(
        [["نعم"], ["لا"], ["نعم"]],
        ["نعم", "نعم", "لا"]) == [0, 2, 1]


def test_a_pathological_duplicate_heavy_tiling_fails_closed():
    # Distinct all-"a" phrases of every length 1..10 against a text whose LAST
    # token can never match: every complete placement fails at the end, so the
    # naive search would grind through factorially many orders. The step
    # budget answers None instead of spinning.
    phrases = [["a"] * n for n in range(1, 11)]
    full = ["a"] * (sum(range(1, 11)) - 1) + ["b"]

    assert phrase_pairs.tile_permutation(phrases, full) is None


# --- validation: the strict segmentation contract ----------------------------


def test_a_complete_ordered_segmentation_passes():
    assert phrase_pairs.validate_pairs(PAIRS, SOURCE, TARGET) == (
        PAIRS, IDENTITY)


def test_a_reordered_translation_validates_with_its_permutation():
    # The owner's ruling: matching colours mean matching MEANING. The pairs
    # stay in source order; the permutation records the Arabic's own order.
    assert phrase_pairs.validate_pairs(
        REORDER_PAIRS, REORDER_SOURCE, REORDER_TARGET) == (
        REORDER_PAIRS, REORDER_PERM)


def test_validation_normalises_whitespace_but_nothing_else():
    sloppy = [{"s": "  We   can not", "t": "لا  يمكننا "}] + PAIRS[1:]

    assert phrase_pairs.validate_pairs(sloppy, f"  {SOURCE} ", TARGET) == (
        PAIRS, IDENTITY)


def test_pairs_that_do_not_reproduce_the_source_are_discarded():
    # "create" went missing: concatenation no longer equals the input.
    assert phrase_pairs.validate_pairs(
        [PAIRS[0], PAIRS[2], PAIRS[3]], SOURCE, TARGET) is None


def test_pairs_that_do_not_reproduce_the_target_are_discarded():
    broken = [dict(PAIRS[0], t="لا"), *PAIRS[1:]]

    assert phrase_pairs.validate_pairs(broken, SOURCE, TARGET) is None


def test_a_mid_word_split_is_discarded():
    # "variables" split as "varia"+"bles": the words re-tokenize differently,
    # which is exactly how a renderer would re-join them wrongly later.
    broken = [
        PAIRS[0], PAIRS[1],
        {"s": "two local varia", "t": "متغيرين"},
        {"s": "bles with the same name", "t": "محليين بالاسم نفسه"},
    ]

    assert phrase_pairs.validate_pairs(broken, SOURCE, TARGET) is None


def test_a_mid_word_split_inside_an_arabic_word_is_discarded():
    broken = [
        {"s": "We can not", "t": "لا يمكن"},
        {"s": "create", "t": "نا إنشاء"},
        PAIRS[2], PAIRS[3],
    ]

    assert phrase_pairs.validate_pairs(broken, SOURCE, TARGET) is None


def test_diacritics_the_postprocessor_stripped_are_absorbed():
    # The LLM paired against ITS OWN diacritized output; the paragraph ends up
    # holding the post-processor's stripped text. The stored "t" must be the
    # stripped form — the one the sidecar's target actually says.
    diacritized = [{"s": "We can not", "t": "لا يُمكننا"}, *PAIRS[1:]]

    assert phrase_pairs.validate_pairs(diacritized, SOURCE, TARGET) == (
        PAIRS, IDENTITY)


def test_empty_inputs_never_validate():
    assert phrase_pairs.validate_pairs(None, SOURCE, TARGET) is None
    assert phrase_pairs.validate_pairs(PAIRS, "", TARGET) is None
    assert phrase_pairs.validate_pairs(PAIRS, SOURCE, "") is None


# --- rect resolution: phrases onto characters, per visual line ---------------


def _chars(text, y_top, x0, *, step=6.0, size=10.0):
    """One line of (text, box) tuples, one character each, left to right."""
    return [(ch, [x0 + i * step, y_top - size, x0 + (i + 1) * step, y_top])
            for i, ch in enumerate(text)]


def test_phrases_map_to_one_rect_per_line():
    chars = _chars("We can not", 714, 60) + _chars("create", 696, 60)

    rects = phrase_pairs.match_phrases_to_rects(
        chars, ["We can not", "create"])

    assert rects == [
        [[60.0, 704.0, 120.0, 714.0]],   # up to "t", the 10th character cell
        [[60.0, 686.0, 96.0, 696.0]],    # 6 chars on the next line
    ]


def test_a_phrase_wrapping_lines_yields_one_rect_per_line():
    # "can not create" starts on line 1 and finishes on line 2.
    chars = _chars("We can not", 714, 60) + _chars("create", 696, 60)

    rects = phrase_pairs.match_phrases_to_rects(chars, ["We", "can not create"])

    assert len(rects) == 2
    assert len(rects[1]) == 2                      # the wrap: two lines
    assert rects[1][0][3] > rects[1][1][3]         # top line first


def test_typeset_arabic_glyphs_match_their_logical_phrases():
    # What Typesetting actually stores for a translated RTL paragraph: Arabic
    # PRESENTATION forms (arabic_reshaper), one glyph per character, in
    # LOGICAL order — including the lam-alef ligature, ONE glyph that folds
    # back to TWO logical letters.
    reshaped = reshape_rtl_text("لا يمكننا إنشاء")
    assert reshaped != "لا يمكننا إنشاء"  # the premise: forms did change

    chars = _chars(reshaped, 714, 60)

    rects = phrase_pairs.match_phrases_to_rects(chars, ["لا يمكننا", "إنشاء"])

    assert rects is not None
    assert len(rects) == 2
    assert all(len(phrase_rects) == 1 for phrase_rects in rects)


def test_bidi_mirrored_brackets_still_match():
    # Typesetting stores "(" as ")" inside an RTL visual run.
    chars = _chars("قائمة )٣(", 714, 60)

    assert phrase_pairs.match_phrases_to_rects(
        chars, ["قائمة", "(٣)"]) is not None


def test_leftover_characters_mean_no_rects_at_all():
    chars = _chars("We can not create extra", 714, 60)

    assert phrase_pairs.match_phrases_to_rects(
        chars, ["We can not", "create"]) is None


def test_characters_that_spell_a_different_text_mean_no_rects():
    chars = _chars("Something else entirely!!", 714, 60)

    assert phrase_pairs.match_phrases_to_rects(
        chars, ["We can not", "create"]) is None


def test_a_phrase_whose_characters_have_no_boxes_means_no_rects():
    chars = _chars("We can not", 714, 60) + [(ch, None) for ch in "create"]

    assert phrase_pairs.match_phrases_to_rects(
        chars, ["We can not", "create"]) is None


# --- the sidecar: pairs captured, both rect sides, and never a regression ----


def _box(x0, y0, x1, y1):
    return il_version_1.Box(x=x0, y=y0, x2=x1, y2=y1)


def _char_run(text, y_top, x0, *, step=6.0, size=10.0):
    characters = [
        il_version_1.PdfCharacter(box=_box(x0 + i * step, y_top - size,
                                           x0 + (i + 1) * step, y_top),
                                  char_unicode=ch)
        for i, ch in enumerate(text)
    ]
    return il_version_1.PdfParagraphComposition(
        pdf_same_style_characters=il_version_1.PdfSameStyleCharacters(
            box=_box(x0, y_top - size, x0 + len(text) * step, y_top),
            pdf_character=characters,
        ))


def _paragraph(text, box, compositions=()):
    return il_version_1.PdfParagraph(
        box=box,
        pdf_style=il_version_1.PdfStyle(font_size=11.0),
        unicode=text,
        layout_label="plain text",
        pdf_paragraph_composition=list(compositions),
    )


def _document(paragraphs):
    page = il_version_1.Page(
        mediabox=il_version_1.Mediabox(box=_box(0.0, 0.0, 595.0, 842.0)),
        cropbox=il_version_1.Cropbox(box=_box(0.0, 0.0, 595.0, 842.0)),
        pdf_paragraph=list(paragraphs),
        page_number=0,
        unit="point",
        base_operations=il_version_1.BaseOperations(value=""),
    )
    return il_version_1.Document(page=[page], total_pages=1)


def _translated_document():
    """One paragraph through the real motions: snapshot, translate in place."""
    paragraph = _paragraph(
        "We can not create", _box(60, 680, 400, 714),
        compositions=[_char_run("We can not", 714, 60),
                      _char_run("create", 696, 60)])
    docs = _document([paragraph])
    sources = translation_sidecar.snapshot_source(docs)
    paragraph.unicode = "لا يمكننا إنشاء"
    pair_store = {id(paragraph): {
        "pairs": [{"s": "We can not", "t": "لا يمكننا"},
                  {"s": "create", "t": "إنشاء"}],
        "perm": [0, 1]}}
    return docs, sources, pair_store, paragraph


def _typeset(paragraph, *, y_top=714):
    """What Typesetting leaves behind: per-glyph compositions, reshaped forms
    in logical order, boxes in the (mirrored) translated page's space."""
    reshaped = reshape_rtl_text(paragraph.unicode)
    step = 6.0
    paragraph.pdf_paragraph_composition = [
        il_version_1.PdfParagraphComposition(
            pdf_character=il_version_1.PdfCharacter(
                box=_box(500 - (i + 1) * step, y_top - 10, 500 - i * step, y_top),
                char_unicode=ch))
        for i, ch in enumerate(reshaped)
    ]


def test_validated_pairs_land_in_the_block_with_source_rects():
    docs, sources, pair_store, _paragraph_ = _translated_document()

    block = translation_sidecar.build_sidecar(
        docs, lang_in="en", lang_out="ar", sources=sources,
        pair_store=pair_store)["pages"][0]["blocks"][0]

    assert [(p["s"], p["t"]) for p in block["pairs"]] == [
        ("We can not", "لا يمكننا"), ("create", "إنشاء")]
    # One rect per phrase per source line, in the original page's space.
    assert block["pairs"][0]["s_rects"] == [[60.0, 704.0, 120.0, 714.0]]
    assert block["pairs"][1]["s_rects"] == [[60.0, 686.0, 96.0, 696.0]]
    assert "t_rects" not in block["pairs"][0]  # typesetting has not run yet


def test_target_rects_are_resolved_after_typesetting(tmp_path):
    docs, sources, pair_store, paragraph = _translated_document()
    path = tmp_path / "sidecar.json"

    state = translation_sidecar.write_sidecar(
        docs, path, lang_in="en", lang_out="ar", sources=sources,
        pair_store=pair_store)
    assert state is not None

    _typeset(paragraph)  # the translated characters get their boxes NOW
    translation_sidecar.attach_target_rects(state)

    written = json.loads(path.read_text(encoding="utf-8"))
    pairs = written["pages"][0]["blocks"][0]["pairs"]
    assert all(pair["t_rects"] for pair in pairs)
    # Both phrases sit on the one typeset line; the first phrase (لا يمكننا)
    # is RTL-first, i.e. to the RIGHT of the second.
    assert pairs[0]["t_rects"][0][0] > pairs[1]["t_rects"][0][0]
    # And the source side survived the rewrite untouched.
    assert pairs[0]["s_rects"] == [[60.0, 704.0, 120.0, 714.0]]


def test_a_reordered_translation_gets_its_target_rects_on_the_right_pairs(
        tmp_path):
    """The owner's e2e defect, end to end: the Arabic puts «يجب الإعلان عن»
    FIRST although its pair is listed second, and every pair's t_rects must
    land on ITS OWN phrase — wherever the translation put it."""
    paragraph = _paragraph(
        REORDER_SOURCE, _box(60, 680, 400, 714),
        compositions=[_char_run(REORDER_SOURCE, 714, 60, step=5.0)])
    docs = _document([paragraph])
    sources = translation_sidecar.snapshot_source(docs)
    paragraph.unicode = REORDER_TARGET
    pairs, perm = phrase_pairs.validate_pairs(
        REORDER_PAIRS, REORDER_SOURCE, REORDER_TARGET)
    pair_store = {id(paragraph): {"pairs": pairs, "perm": perm}}
    path = tmp_path / "sidecar.json"

    state = translation_sidecar.write_sidecar(
        docs, path, lang_in="en", lang_out="ar", sources=sources,
        pair_store=pair_store)
    assert state is not None

    _typeset(paragraph)
    translation_sidecar.attach_target_rects(state)

    written = json.loads(path.read_text(encoding="utf-8"))["pages"][0][
        "blocks"][0]["pairs"]
    # Entries stayed in SOURCE order…
    assert [(p["s"], p["t"]) for p in written] == [
        (p["s"], p["t"]) for p in REORDER_PAIRS]
    assert all(pair["t_rects"] for pair in written)
    # …while the rects follow the TRANSLATION's order. The typeset line runs
    # right-to-left, so the pair whose phrase comes FIRST in the Arabic —
    # «يجب الإعلان عن», listed second — sits furthest RIGHT, then «المتغير
    # المحلي» (listed first), then «قبل استخدامه.» (listed third).
    assert (written[1]["t_rects"][0][0] > written[0]["t_rects"][0][0]
            > written[2]["t_rects"][0][0])


def test_unmatchable_typeset_text_keeps_pairs_but_omits_target_rects(tmp_path):
    docs, sources, pair_store, paragraph = _translated_document()
    path = tmp_path / "sidecar.json"

    state = translation_sidecar.write_sidecar(
        docs, path, lang_in="en", lang_out="ar", sources=sources,
        pair_store=pair_store)

    paragraph.unicode = "شيء آخر تماما"  # the page now says something else
    _typeset(paragraph)
    translation_sidecar.attach_target_rects(state)  # must not raise

    pairs = json.loads(path.read_text(encoding="utf-8"))[
        "pages"][0]["blocks"][0]["pairs"]
    assert all("t_rects" not in pair for pair in pairs)
    assert all(pair["s_rects"] for pair in pairs)


def test_a_run_without_pairs_writes_the_sidecar_exactly_as_before(tmp_path):
    def build(**kwargs):
        docs, sources, _store, _p = _translated_document()
        return translation_sidecar.build_sidecar(
            docs, lang_in="en", lang_out="ar", sources=sources, **kwargs)

    today = json.dumps(build(), ensure_ascii=False, sort_keys=True)

    # No store, an empty store, and a store about OTHER paragraphs must all
    # produce the identical document — and never a "pairs" key.
    assert json.dumps(build(pair_store={}), ensure_ascii=False,
                      sort_keys=True) == today
    assert json.dumps(build(pair_store={12345: {"pairs": PAIRS,
                                                "perm": IDENTITY}}),
                      ensure_ascii=False, sort_keys=True) == today
    assert '"pairs"' not in today

    # And the write path reports nothing to finalize.
    path = tmp_path / "sidecar.json"
    docs, sources, _store, _p = _translated_document()
    assert translation_sidecar.write_sidecar(
        docs, path, lang_in="en", lang_out="ar", sources=sources) is None


def test_a_source_that_cannot_be_mapped_keeps_the_pairs_textual():
    # The snapshot's characters spell a DIFFERENT text than the phrases (a
    # hyphenation repair, a dropped ligature): the alignment is still true,
    # the source boxes are not — so the entries carry no s_rects at all.
    paragraph = _paragraph("We can not create", _box(60, 680, 400, 714),
                           compositions=[_char_run("Mismatched chars!", 714, 60)])
    docs = _document([paragraph])
    sources = translation_sidecar.snapshot_source(docs)
    paragraph.unicode = "لا يمكننا إنشاء"
    pair_store = {id(paragraph): {
        "pairs": [{"s": "We can not", "t": "لا يمكننا"},
                  {"s": "create", "t": "إنشاء"}],
        "perm": [0, 1]}}

    block = translation_sidecar.build_sidecar(
        docs, lang_in="en", lang_out="ar", sources=sources,
        pair_store=pair_store)["pages"][0]["blocks"][0]

    assert [(p["s"], p["t"]) for p in block["pairs"]] == [
        ("We can not", "لا يمكننا"), ("create", "إنشاء")]
    assert all("s_rects" not in pair for pair in block["pairs"])


def test_attach_target_rects_swallows_a_broken_state(tmp_path):
    translation_sidecar.attach_target_rects(None)
    translation_sidecar.attach_target_rects(
        {"path": tmp_path / "nope.json", "sidecar": {}, "pending": [
            (["not-an-entry"], object(), [0])]})  # garbage in, silence out
    translation_sidecar.attach_target_rects(
        {"path": tmp_path / "nope.json", "sidecar": {}, "pending": [
            ("wrong", "shape")]})  # even the tuple arity is untrusted


# --- styled paragraphs: style wrappers no longer exclude a paragraph ---------
#
# On real lecture slides nearly every body bullet is a styled paragraph — the
# bullet glyph «•» alone is a different font run and becomes a <style id='N'>
# wrapper — and under the old all-placeholders-excluded rule only titles and
# footers ever got pairs. Style tags merely WRAP text the model sees, so such
# paragraphs are eligible; formula placeholders ({vN}) REPLACE text the model
# never sees, so those stay excluded.


def _style_placeholder(pid=1):
    """A RichTextPlaceholder exactly as the OpenAI-shaped translator mints it."""
    return RichTextPlaceholder(
        pid, None,
        f"<style id='{pid}'>", "</style>",
        f"<\\s*style\\s*id\\s*=\\s*'\\s*{pid}\\s*'\\s*>", r"<\s*\/\s*style\s*>",
    )


def _formula_placeholder(pid=1):
    return FormulaPlaceholder(
        pid, None, "{v" + str(pid) + "}", f"{{\\s*v\\s*{pid}\\s*}}")


def _capture_translator():
    """An ILTranslatorLLMOnly reduced to its capture seam: no engine, no LLM."""
    translator = object.__new__(ILTranslatorLLMOnly)
    translator.capture_phrase_pairs = True
    translator.pairs_kept = 0
    translator.pairs_discarded = 0
    translator.translation_config = SimpleNamespace(phrase_pair_store={})
    return translator


def test_style_placeholders_no_longer_disqualify_but_formulas_still_do():
    assert pairs_eligible(ILTranslator.TranslateInput("plain text", []))
    assert pairs_eligible(ILTranslator.TranslateInput(
        "<style id='1'>•</style> body", [_style_placeholder(1)]))
    assert not pairs_eligible(ILTranslator.TranslateInput(
        "{v1} energy", [_formula_placeholder(1)]))
    assert not pairs_eligible(ILTranslator.TranslateInput(
        "<style id='1'>E</style> = {v2}",
        [_style_placeholder(1), _formula_placeholder(2)]))


def test_stripping_style_wrappers_leaves_the_plain_text():
    placeholders = [_style_placeholder(1)]

    assert strip_style_placeholders(
        "<style id='1'>•</style> We can not create", placeholders,
    ) == "• We can not create"
    # A mid-sentence styled run (a bolded term) strips in place.
    assert strip_style_placeholders(
        "We can <style id='1'>not</style> create", placeholders,
    ) == "We can not create"
    # The model's echoed tags are matched the way parse_translate_output
    # matches them: case-insensitive and whitespace-tolerant.
    assert strip_style_placeholders(
        "< STYLE id = ' 1 ' >•< / style > We can", placeholders,
    ) == "• We can"
    # Formula tokens are NOT text and are never stripped into it.
    assert strip_style_placeholders(
        "{v1} stays", [_formula_placeholder(1)]) == "{v1} stays"


def test_a_mid_sentence_styled_run_validates_over_the_plain_text():
    plain = strip_style_placeholders(
        "We can <style id='1'>not</style> create", [_style_placeholder(1)])

    pairs = [{"s": "We can not", "t": "لا يمكننا"}, {"s": "create", "t": "إنشاء"}]
    assert phrase_pairs.validate_pairs(pairs, plain, "لا يمكننا إنشاء") == (
        pairs, [0, 1])


def test_a_styled_bullet_paragraph_is_captured_end_to_end(tmp_path):
    # The real deck shape: the bullet glyph is its own font run, so the model
    # was sent "<style id='1'>•</style> We can not create" and echoed the tag
    # around the untranslated bullet.
    paragraph = _paragraph(
        "• We can not create", _box(60, 680, 400, 714),
        compositions=[_char_run("• We can not", 714, 60),
                      _char_run("create", 696, 60)])
    docs = _document([paragraph])
    sources = translation_sidecar.snapshot_source(docs)

    # What ILTranslator leaves after APPLYING the parsed output: the styled
    # run resolved back to its own characters, the translation as a unicode
    # run — while paragraph.unicode keeps the RAW tagged output.
    paragraph.unicode = "<style id='1'>•</style> لا يمكننا إنشاء"
    paragraph.pdf_paragraph_composition = [
        _char_run("•", 714, 60),
        il_version_1.PdfParagraphComposition(
            pdf_same_style_unicode_characters=(
                il_version_1.PdfSameStyleUnicodeCharacters(
                    unicode=" لا يمكننا إنشاء"))),
    ]

    translator = _capture_translator()
    translate_input = ILTranslator.TranslateInput(
        "<style id='1'>•</style> We can not create", [_style_placeholder(1)])
    translator._capture_phrase_pairs(
        paragraph=paragraph,
        translate_input=translate_input,
        source_text=translate_input.unicode,
        raw_pairs=[{"s": "•", "t": "•"},
                   {"s": "We can not", "t": "لا يمكننا"},
                   {"s": "create", "t": "إنشاء"}])

    # The pairs validated over the PLAIN texts on both sides.
    assert translator.pairs_kept == 1
    store = translator.translation_config.phrase_pair_store
    assert [(p["s"], p["t"]) for p in store[id(paragraph)]["pairs"]] == [
        ("•", "•"), ("We can not", "لا يمكننا"), ("create", "إنشاء")]

    # s_rects resolve against the snapshotted source characters…
    path = tmp_path / "sidecar.json"
    state = translation_sidecar.write_sidecar(
        docs, path, lang_in="en", lang_out="ar", sources=sources,
        pair_store=store)
    assert state is not None
    block = json.loads(path.read_text(encoding="utf-8"))[
        "pages"][0]["blocks"][0]
    # …and the block's target — what the "t" phrases were validated
    # against — is the PLAIN reader text, never the tagged raw output.
    assert block["target"] == "• لا يمكننا إنشاء"
    assert all(pair["s_rects"] for pair in block["pairs"])

    # …and t_rects resolve against the typeset characters.
    paragraph.unicode = "• لا يمكننا إنشاء"
    _typeset(paragraph)
    translation_sidecar.attach_target_rects(state)
    written = json.loads(path.read_text(encoding="utf-8"))[
        "pages"][0]["blocks"][0]["pairs"]
    assert all(pair["t_rects"] for pair in written)


def test_a_formula_paragraph_is_still_excluded_at_the_capture_seam():
    translator = _capture_translator()
    paragraph = _paragraph("طاقة {v1}", _box(60, 680, 400, 714))

    translator._capture_phrase_pairs(
        paragraph=paragraph,
        translate_input=ILTranslator.TranslateInput(
            "{v1} energy", [_formula_placeholder(1)]),
        source_text="{v1} energy",
        raw_pairs=[{"s": "{v1} energy", "t": "طاقة {v1}"}])

    # Not discarded — never in the game: no store entry, no counter moved.
    assert translator.translation_config.phrase_pair_store == {}
    assert translator.pairs_kept == 0
    assert translator.pairs_discarded == 0
