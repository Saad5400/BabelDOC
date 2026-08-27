"""Tests for server/glossary_pages.py — the «شرح المصطلحات» page renderer.

Run from the repo root (pyproject's testpaths only covers tests/):

    pytest server/test_glossary_pages.py
"""

import unicodedata

import pymupdf
import pytest

from server import glossary_pages

PAGE = (720.0, 540.0)  # slide-shaped, like the decks this was built for

EXPLANATION = ("كلمة wrapping تعني «التغليف»، مأخوذة من wrap أي يُغلِّف — مثل "
               "ساندوتش الراب المغلَّف. في البرمجة: نغلِّف قيمة بدائية داخل "
               "كائن، مثل وضع int داخل Integer.")


@pytest.fixture(scope="module", autouse=True)
def fonts_cached():
    """Fetch the faces once — rendering real Arabic is what these tests do."""
    glossary_pages._font_path(glossary_pages.FONT_FILE)
    glossary_pages._font_path(glossary_pages.BOLD_FONT_FILE)


def _doc(pages=2, size=PAGE):
    doc = pymupdf.open()
    for index in range(pages):
        page = doc.new_page(width=size[0], height=size[1])
        page.insert_text((72, 72), f"Slide {index + 1}", fontsize=24)
    return doc


def _entry(term="Wrapping", **overrides):
    entry = {"term": term, "arabic": "التغليف", "explanation": EXPLANATION,
             "page": 12, "quote": "wrapper classes wrap primitive values"}
    entry.update(overrides)
    return entry


def _text(doc, from_page):
    text = "".join(doc[i].get_text() for i in range(from_page, doc.page_count))
    # Shaped Arabic extracts as presentation forms; fold them back so tests
    # can look for logical strings.
    return unicodedata.normalize("NFKC", text)


def test_entries_become_appended_pages_of_the_document_size():
    doc = _doc()

    added = glossary_pages.append_glossary_pages(doc, [_entry()])

    assert added == 1
    assert doc.page_count == 3
    assert doc[2].rect.width == PAGE[0]
    assert doc[2].rect.height == PAGE[1]
    doc.close()


def test_the_page_carries_the_term_the_arabic_and_the_source():
    doc = _doc()
    glossary_pages.append_glossary_pages(doc, [_entry()])

    text = _text(doc, 2)

    assert "Wrapping" in text
    assert "wrapper classes wrap primitive values" in text  # the quote
    assert "12" in text  # the slide number
    # Shaped Arabic came back out: the cmap of the subset survived.
    assert any("؀" <= ch <= "ۿ" for ch in text)
    doc.close()


def test_ten_long_entries_overflow_onto_further_pages():
    # Letter suffixes, not digits: a digit shaped inside an RTL run can be
    # drawn via a GSUB variant glyph with no cmap entry, which extracts as
    # junk (a face characteristic the interlinear overlay shares) — and this
    # test is about overflow, not about that.
    names = [f"Term{letter}" for letter in "ABCDEFGHIJ"]
    doc = _doc()
    entries = [_entry(name, explanation=EXPLANATION * 3) for name in names]

    added = glossary_pages.append_glossary_pages(doc, entries)

    assert added > 1
    assert doc.page_count == 2 + added
    text = _text(doc, 2)
    for name in names:
        assert name in text  # nothing fell off the end
    doc.close()


def test_no_entries_means_no_pages():
    doc = _doc()

    assert glossary_pages.append_glossary_pages(doc, []) == 0
    assert glossary_pages.append_glossary_pages(doc, None) == 0
    assert doc.page_count == 2
    doc.close()


def test_unusable_entries_are_skipped_and_alone_add_nothing():
    doc = _doc()
    junk = ["not a dict", {"term": "", "explanation": "x"},
            {"term": "NoBody", "explanation": "  "}]

    assert glossary_pages.append_glossary_pages(doc, junk) == 0
    assert doc.page_count == 2

    added = glossary_pages.append_glossary_pages(doc, [*junk, _entry()])

    assert added == 1
    doc.close()


def test_an_entry_without_page_or_quote_renders_without_a_source_line():
    doc = _doc()

    added = glossary_pages.append_glossary_pages(
        doc, [_entry(page=None, quote=None)])

    assert added == 1
    assert "وردت في" not in _text(doc, 2)
    doc.close()


def test_an_empty_document_still_gets_a_page():
    doc = pymupdf.open()

    assert glossary_pages.append_glossary_pages(doc, [_entry()]) == 1
    assert doc.page_count == 1
    doc.close()


def test_the_output_stays_small_after_a_garbage_save(tmp_path):
    """The 15 MB faces must arrive subset, or every result balloons."""
    doc = _doc()
    glossary_pages.append_glossary_pages(doc, [_entry(), _entry("Casting")])

    out = tmp_path / "out.pdf"
    doc.save(str(out), garbage=4, deflate=True)
    doc.close()

    assert out.stat().st_size < 2_000_000
