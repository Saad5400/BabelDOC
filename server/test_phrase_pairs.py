"""Tests for phrase-pair alignment (babeldoc/.../midend/phrase_pairs.py and its
capture into the translation sidecar).

Run from the repo root:

    pytest server/test_phrase_pairs.py

No LLM anywhere: the pairs under test are what a model WOULD have returned,
including the shapes a model returns when it misbehaves.
"""

import json

from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.midend import phrase_pairs
from babeldoc.format.pdf.document_il.midend import translation_sidecar
from babeldoc.format.pdf.document_il.midend.il_translator_llm_only import (
    PHRASE_PAIRS_PROMPT_BLOCK,
    PROMPT_TEMPLATE,
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


# --- the prompt asks for pairs, in the shape the parser accepts back ---------


def test_the_prompt_template_carries_the_pairs_block_slot():
    assert "$phrase_pairs_block" in PROMPT_TEMPLATE.template


def test_the_pairs_instruction_teaches_the_contract_with_a_worked_example():
    assert '"want_pairs": true' in PHRASE_PAIRS_PROMPT_BLOCK
    assert "NEVER split inside a word" in PHRASE_PAIRS_PROMPT_BLOCK
    # The owner's worked example, whole: every phrase of both sides.
    for pair in PAIRS:
        assert pair["s"] in PHRASE_PAIRS_PROMPT_BLOCK
        assert pair["t"] in PHRASE_PAIRS_PROMPT_BLOCK


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


# --- validation: the strict segmentation contract ----------------------------


def test_a_complete_ordered_segmentation_passes():
    assert phrase_pairs.validate_pairs(PAIRS, SOURCE, TARGET) == PAIRS


def test_validation_normalises_whitespace_but_nothing_else():
    sloppy = [{"s": "  We   can not", "t": "لا  يمكننا "}] + PAIRS[1:]

    assert phrase_pairs.validate_pairs(sloppy, f"  {SOURCE} ", TARGET) == PAIRS


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

    assert phrase_pairs.validate_pairs(diacritized, SOURCE, TARGET) == PAIRS


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
    pair_store = {id(paragraph): [{"s": "We can not", "t": "لا يمكننا"},
                                  {"s": "create", "t": "إنشاء"}]}
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
    assert json.dumps(build(pair_store={12345: PAIRS}), ensure_ascii=False,
                      sort_keys=True) == today
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
    pair_store = {id(paragraph): [{"s": "We can not", "t": "لا يمكننا"},
                                  {"s": "create", "t": "إنشاء"}]}

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
            (["not-an-entry"], object())]})  # garbage in, silence out
