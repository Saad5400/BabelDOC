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
