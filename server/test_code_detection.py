"""Tests for "is this paragraph code?" (fork side).

Two independent mechanisms used to turn a monospace HEADING into an
untranslatable formula:

  * styles_and_formulas' own code-paragraph rule, which above
    CODE_PARAGRAPH_PURE_MONO_RATIO asked for no evidence of code at all; and
  * babeldoc's formula-FONT pattern, whose default matches `.*Mono`, so
    every character of a monospace face is classified as formula.

A deck whose heading face is NotoMono tripped both, and 46 headings on 28 of
its 83 pages were delivered in English inside an otherwise Arabic document.

Run from the repo root:

    pytest server/test_code_detection.py
"""

from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
    StylesAndFormulas,
)
from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
    looks_like_prose,
)

# Real headings from the deck this defect was found on (run3).
HEADINGS = [
    "Exercise: Classify the Scenario",
    "Hardware & Software",
    "Assets",
    "Data Assets",
    "Values of Assets & Replaceability",
    "Relationship Among Vulnerabilities, Threats, and Controls",
    "Beyond the CIA Triad",
    "CIA Scenario Exercise",
    "Sources of Threats",
    "Intent Is Not Always Clear",
    "Types of Attacks",
]

# Single source lines that carry no ";{}=" and must still read as code.
CODE_LINES = [
    "aload_0",
    "istore 2goto A iload_2 ireturn",
    "SET>java Hello",
    "System.out.println",
    "this.arr",
    "#10",
]


def test_real_headings_read_as_prose():
    for heading in HEADINGS:
        assert looks_like_prose(heading), heading


def test_code_lines_do_not_read_as_prose():
    for line in CODE_LINES:
        assert not looks_like_prose(line), line


def test_statement_punctuation_is_not_prose():
    assert not looks_like_prose("let S = 0;")
    assert not looks_like_prose("x += 1")
    assert not looks_like_prose("}")


def test_empty_and_symbol_only_text_is_not_prose():
    assert not looks_like_prose("")
    assert not looks_like_prose("   ")
    assert not looks_like_prose("&")


def test_a_number_inside_a_heading_is_still_prose():
    assert looks_like_prose("Chapter 3")


# ------------------------------------------------- the formula-font rule


def _char(text, font_id="mono", formula_layout_id=None):
    return il_version_1.PdfCharacter(
        box=il_version_1.Box(0, 0, 1, 1),
        visual_bbox=il_version_1.VisualBbox(box=il_version_1.Box(0, 0, 1, 1)),
        char_unicode=text,
        formula_layout_id=formula_layout_id,
        pdf_style=il_version_1.PdfStyle(
            font_id=font_id,
            font_size=12.0,
            graphic_state=il_version_1.GraphicState(),
        ),
    )


def _paragraph(*line_texts, font_id="mono"):
    return il_version_1.PdfParagraph(
        pdf_paragraph_composition=[
            il_version_1.PdfParagraphComposition(
                pdf_line=il_version_1.PdfLine(
                    pdf_character=[_char(c, font_id) for c in text]
                )
            )
            for text in line_texts
        ]
    )


def _sf():
    return StylesAndFormulas.__new__(StylesAndFormulas)


def test_monospace_heading_withdraws_the_formula_font_signal():
    paragraph = _paragraph("Exercise: Classify the Scenario")
    assert _sf()._is_monospace_prose(paragraph, {"mono"})


def test_multi_line_monospace_block_keeps_the_formula_font_signal():
    # Two lines of monospace is a code block even if each line is words.
    paragraph = _paragraph("public class Foo", "return bar")
    assert not _sf()._is_monospace_prose(paragraph, {"mono"})


def test_monospace_code_line_keeps_the_formula_font_signal():
    paragraph = _paragraph("istore 2goto A iload_2 ireturn")
    assert not _sf()._is_monospace_prose(paragraph, {"mono"})


def test_a_line_already_in_a_formula_layout_is_left_alone():
    paragraph = il_version_1.PdfParagraph(
        pdf_paragraph_composition=[
            il_version_1.PdfParagraphComposition(
                pdf_line=il_version_1.PdfLine(
                    pdf_character=[
                        _char(c, "mono", formula_layout_id=3) for c in "Assets"
                    ]
                )
            )
        ]
    )
    assert not _sf()._is_monospace_prose(paragraph, {"mono"})


def test_a_non_monospace_heading_is_not_affected():
    paragraph = _paragraph("Exercise: Classify the Scenario", font_id="serif")
    assert not _sf()._is_monospace_prose(paragraph, {"mono"})


# ====================================================================
# Prose that arrived at the translator as a formula and was never sent
# ====================================================================
#
# MEASURED, no LLM, with the deterministic repro harness over two whole
# production decks (record 156, "Fundamentals of Management", 23 pages;
# record 138, "Algorithms and Introduction to Java", 35 pages):
#
#   156   157 paragraphs, 37 untranslated   38 paragraphs made into one
#                                           rigid formula, 37 of them prose
#   138   342 paragraphs, 39 untranslated   40 made into one rigid formula,
#                                           39 of them prose
#
# Every one of them came from the SAME place: `_paragraph_is_one_stack`.
# The paragraph finder had put all the visual lines of a wrapped bullet
# into one pdf_line, so `_is_vertical_stack` saw rows sitting one above
# another with ~100% horizontal overlap -- a numerator over a denominator
# -- and process_page_formulas replaced the whole paragraph with a single
# pdf_formula composition. `is_placeholder_only_paragraph` then skipped it,
# and the reader got an English bullet in an Arabic deck.
#
# Geometry cannot tell these apart. The text can.

# Real leftovers, copied from the sidecars of those two runs (blocks whose
# target came back identical to their source).
PROSE_LEFTOVERS = [
    # the four named in the defect report
    "• Understanding management offers insights into many organizational"
    " aspects.",
    "• Custom software usually has a long-lifetime (10 years or more) and"
    " must be maintained.",
    "• Programming languages are a set of instructions written in a way that"
    " a computer could understand",
    "He was unhappy using c++ programming language, so he developed java.",
    # ten more, verbatim from probe-156.sidecar.json and 138's sidecar
    "• One does not need to become a manager to benefit from the study of"
    " management.",
    "– Management is a learned talent rather than something that comes"
    " naturally.",
    "• Effectiveness is completing activities so that organizational goals"
    " are attained; often described as “doing the right things.”",
    "A manager is someone who works with and through other people by"
    " coordinating their work activities in order to accomplish"
    " organizational goals.",
    "– president, chief executive officer (C E O), managing director,"
    " chancellor",
    "– Even not-for-profit organizations need to make money to continue"
    " operating.",
    "Exhibit 1.1 Efficiency, Effectiveness, and Performance in Student"
    " Meetings",
    "• The Java compiler (javac) translates Java source code (Hello.java)"
    " into a special representation called bytecode (object program)",
    "• Java has removed many complicated and rarely-used features, for"
    " example, explicit pointers, operator overloading, etc.",
    "• We use programs almost daily (email, word processors, video games,"
    " bank ATMs, etc.).",
    "Why Are Customers Important to the Manager ’s Job?",
]

# Text that must keep its verbatim-preservation privilege.
CODE_BLOCKS = [
    "public static void main(String[] args)",
    "this.arr;",
    "iload_2",
    "def f(x):",
    "SET>java Hello",
    "let S = 0;",
    "System.out.println(x);",
]


def test_real_leftovers_are_not_code():
    from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
        is_code_block_text,
    )

    for text in PROSE_LEFTOVERS:
        assert not is_code_block_text(text), text


def test_code_blocks_are_still_code():
    from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
        is_code_block_text,
    )

    for text in CODE_BLOCKS:
        assert is_code_block_text(text), text


def _wrapped(text, rows=2, x0=36.0, width=560.0, top=460.0, leading=28.8):
    """A paragraph whose visual lines all sit inside ONE pdf_line.

    This is the shape the paragraph finder produced for every bullet on
    record 156: `rows` baselines, one above another, each spanning almost
    the full text column, so their horizontal overlap is ~100%.
    """
    per_row = max(1, len(text) // rows + 1)
    characters = []
    for row in range(rows):
        chunk = text[row * per_row : (row + 1) * per_row]
        if not chunk:
            continue
        y = top - row * leading
        step = width / max(len(chunk), 1)
        for i, char in enumerate(chunk):
            box = il_version_1.Box(
                x=x0 + i * step, y=y, x2=x0 + (i + 1) * step, y2=y + 17.6
            )
            characters.append(
                il_version_1.PdfCharacter(
                    box=box,
                    visual_bbox=il_version_1.VisualBbox(box=box),
                    char_unicode=char,
                    pdf_style=il_version_1.PdfStyle(
                        font_id="body",
                        font_size=17.6,
                        graphic_state=il_version_1.GraphicState(),
                    ),
                )
            )
    return il_version_1.PdfParagraph(
        unicode=text,
        pdf_paragraph_composition=[
            il_version_1.PdfParagraphComposition(
                pdf_line=il_version_1.PdfLine(pdf_character=characters)
            )
        ],
    )


def test_a_wrapped_prose_bullet_is_not_a_stack():
    for text in PROSE_LEFTOVERS:
        paragraph = _wrapped(text)
        # The geometry really is a stack -- that is the whole trap.
        assert _sf()._is_vertical_stack(
            paragraph.pdf_paragraph_composition[0].pdf_line
        ), text
        assert not _sf()._paragraph_is_one_stack(paragraph), text


def test_a_wrapped_prose_bullet_reaches_the_translator():
    # End of the chain: a paragraph left as a pdf_line composition is one
    # `is_placeholder_only_paragraph` does NOT skip.
    from babeldoc.format.pdf.document_il.utils.paragraph_helper import (
        is_placeholder_only_paragraph,
    )

    paragraph = _wrapped(PROSE_LEFTOVERS[0], rows=3)
    page = il_version_1.Page(
        mediabox=il_version_1.Mediabox(
            box=il_version_1.Box(0, 0, 720.0, 540.0)
        ),
        cropbox=il_version_1.Cropbox(box=il_version_1.Box(0, 0, 720.0, 540.0)),
        pdf_paragraph=[paragraph],
        page_number=0,
        unit="point",
        base_operations=il_version_1.BaseOperations(value=""),
    )
    styles = _full_sf()
    styles.process_page_formulas(page)

    assert not is_placeholder_only_paragraph(paragraph)


def test_a_stacked_code_line_is_still_one_block():
    # Two rows of Java, one above the other: still preserved verbatim.
    paragraph = _wrapped("public static void main(String[] args)", rows=2)
    assert _sf()._paragraph_is_one_stack(paragraph)


class _StackConfig:
    """The two settings process_page_formulas reads, at their defaults."""

    formular_font_pattern = None
    formular_char_pattern = None


class _FontMapperStub:
    """Every character of these tests exists in the font."""

    @staticmethod
    def has_char(_char):
        return True


def _full_sf():
    """A StylesAndFormulas wired up enough to run a whole page."""
    styles = StylesAndFormulas.__new__(StylesAndFormulas)
    styles.translation_config = _StackConfig()
    styles.font_mapper = _FontMapperStub()
    styles._code_formula_ids = set()
    styles._stacked_formula_ids = set()
    return styles


# -------------------------------------- the code-paragraph rule itself


def _code_page(*paragraphs, font_name="ABCDEF+CourierNewPSMT"):
    return il_version_1.Page(
        mediabox=il_version_1.Mediabox(
            box=il_version_1.Box(0, 0, 720.0, 540.0)
        ),
        cropbox=il_version_1.Cropbox(box=il_version_1.Box(0, 0, 720.0, 540.0)),
        pdf_font=[
            il_version_1.PdfFont(
                font_id="mono", name=font_name, monospace=True, xref_id=1
            )
        ],
        pdf_paragraph=list(paragraphs),
        page_number=0,
        unit="point",
        base_operations=il_version_1.BaseOperations(value=""),
    )


def _mono_paragraph(*line_texts):
    paragraph = _paragraph(*line_texts)
    for composition in paragraph.pdf_paragraph_composition:
        for i, char in enumerate(composition.pdf_line.pdf_character):
            box = il_version_1.Box(x=i, y=0, x2=i + 1, y2=10)
            char.box = box
            char.visual_bbox = il_version_1.VisualBbox(box=box)
    paragraph.unicode = " ".join(line_texts)
    return paragraph


def _is_one_formula(paragraph):
    compositions = paragraph.pdf_paragraph_composition
    return len(compositions) == 1 and compositions[0].pdf_formula is not None


def test_monospace_code_becomes_one_preserved_block():
    for text in ("String[] args;", "this.arr;", "iload_2", "def f(x):"):
        paragraph = _mono_paragraph(text)
        _full_sf().detect_code_paragraphs(_code_page(paragraph))
        assert _is_one_formula(paragraph), text


def test_a_monospace_pseudocode_box_becomes_one_preserved_block():
    paragraph = _mono_paragraph(
        "SET S = 0;", "FOR i = 1 TO n DO", "S = S + i;", "END FOR"
    )
    _full_sf().detect_code_paragraphs(_code_page(paragraph))
    assert _is_one_formula(paragraph)


def test_monospace_prose_is_never_converted_however_many_lines():
    # The line count was the old escape hatch: one line of words was spared,
    # two lines of words were not. A deck whose body face is monospace is
    # still a deck.
    paragraph = _mono_paragraph(
        "Understanding management offers insights",
        "into many organizational aspects.",
    )
    _full_sf().detect_code_paragraphs(_code_page(paragraph))
    assert not _is_one_formula(paragraph)


def test_a_monospace_heading_is_never_converted():
    paragraph = _mono_paragraph("Exercise: Classify the Scenario")
    _full_sf().detect_code_paragraphs(_code_page(paragraph))
    assert not _is_one_formula(paragraph)
