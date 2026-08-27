"""Tests for server/terms.py — the «شرح المصطلحات» extraction pass.

No real API calls anywhere here: the client is a stub. What is under test is
the prompt built from a sidecar, the parsing/validation of what a model sends
back, and — above all — that NOTHING a model can say turns into an exception
for the caller.

Run from the repo root (pyproject's testpaths only covers tests/):

    pytest server/test_terms.py
"""

import json

import pytest

from server import config
from server import terms


def _sidecar(pages=None):
    return {
        "version": 1,
        "lang_in": "en",
        "lang_out": "ar",
        "total_pages": len(pages or []),
        "pages": pages if pages is not None else [{
            "page_number": 0,
            "mediabox": [0, 0, 595, 842],
            "blocks": [{"box": [50, 700, 500, 720],
                        "source": "Wrapper classes wrap primitive values "
                                  "inside objects for the collections API.",
                        "lines": [], "target": "نغلف القيم", "font_size": 12,
                        "label": "plain text"}],
            "obstacles": [],
        }],
    }


class _StubClient:
    """An OpenAI-shaped client that returns a canned message content."""

    def __init__(self, content):
        self.requests = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.requests.append(kwargs)

                class _Msg:
                    pass

                msg = _Msg()
                msg.content = content
                choice = _Msg()
                choice.message = msg
                response = _Msg()
                response.choices = [choice]
                return response

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _entry(term="Wrapping", **overrides):
    entry = {"term": term, "arabic": "التغليف", "explanation": "شرح ودّي.",
             "page": 3, "quote": "wrapper classes"}
    entry.update(overrides)
    return entry


# --------------------------------------------------------------------------
# build_prompt
# --------------------------------------------------------------------------

def test_the_prompt_carries_the_source_text_with_1_based_pages():
    prompt = terms.build_prompt(_sidecar())

    assert "Wrapper classes wrap primitive values" in prompt
    assert "الصفحة 1" in prompt  # page_number 0 is the reader's page 1


def test_the_prompt_is_truncated_whole_pages_at_a_time():
    pages = [{"page_number": i, "blocks": [{"source": "x" * 10_000}]}
             for i in range(20)]

    prompt = terms.build_prompt(_sidecar(pages))

    assert len(prompt) <= terms.MAX_PROMPT_CHARS + 200
    # The cut is between pages: whatever made it in is a complete page.
    assert prompt.count("x" * 10_000) == prompt.count("— الصفحة")


def test_pages_without_source_text_do_not_pad_the_prompt():
    pages = [{"page_number": 0, "blocks": [{"source": None}, {"source": "  "}]}]

    prompt = terms.build_prompt(_sidecar(pages))

    assert "الصفحة" not in prompt


# --------------------------------------------------------------------------
# parse_response
# --------------------------------------------------------------------------

def test_a_valid_response_parses_into_clean_entries():
    content = json.dumps({"terms": [_entry()]})

    entries = terms.parse_response(content)

    assert entries == [_entry()]


def test_a_bare_list_is_accepted_too():
    entries = terms.parse_response(json.dumps([_entry()]))

    assert len(entries) == 1


def test_duplicates_are_dropped_case_insensitively():
    content = json.dumps({"terms": [_entry("Wrapping"), _entry("wrapping")]})

    entries = terms.parse_response(content)

    assert [e["term"] for e in entries] == ["Wrapping"]


def test_more_than_the_cap_is_cut_to_the_cap():
    content = json.dumps({"terms": [_entry(f"Term{i}") for i in range(25)]})

    assert len(terms.parse_response(content)) == terms.MAX_TERMS


def test_junk_items_are_dropped_not_padded():
    content = json.dumps({"terms": [
        "not a dict",
        {"term": "", "explanation": "no term"},
        {"term": "NoExplanation", "explanation": "  "},
        _entry(),
    ]})

    entries = terms.parse_response(content)

    assert [e["term"] for e in entries] == ["Wrapping"]


def test_a_bogus_page_or_quote_degrades_to_none():
    content = json.dumps({"terms": [_entry(page="twelve", quote="")]})

    (entry,) = terms.parse_response(content)

    assert entry["page"] is None
    assert entry["quote"] is None


@pytest.mark.parametrize("content", ["", "not json at all",
                                     '{"terms": "not a list"}', '"a string"',
                                     "42"])
def test_malformed_content_raises_for_extract_terms_to_catch(content):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        terms.parse_response(content)


# --------------------------------------------------------------------------
# extract_terms — the never-fail contract
# --------------------------------------------------------------------------

def test_a_valid_run_returns_the_entries():
    client = _StubClient(json.dumps({"terms": [_entry()]}))

    entries = terms.extract_terms(_sidecar(), client=client)

    assert len(entries) == 1
    (request,) = client.requests
    assert request["response_format"] == {"type": "json_object"}
    assert request["model"] == config.OPENAI_MODEL


def test_zero_terms_is_a_valid_answer():
    client = _StubClient(json.dumps({"terms": []}))

    assert terms.extract_terms(_sidecar(), client=client) == []


@pytest.mark.parametrize("content", ["garbage", '{"nope": 1}', "", None])
def test_malformed_model_output_returns_empty_never_raises(content):
    client = _StubClient(content)

    assert terms.extract_terms(_sidecar(), client=client) == []


def test_a_client_that_explodes_returns_empty_never_raises():
    class _Bomb:
        @property
        def chat(self):
            raise RuntimeError("provider down")

    assert terms.extract_terms(_sidecar(), client=_Bomb()) == []


def test_the_kill_switch_makes_no_call_at_all(monkeypatch):
    monkeypatch.setattr(config, "GLOSSARY_PAGES", False)
    client = _StubClient(json.dumps({"terms": [_entry()]}))

    assert terms.extract_terms(_sidecar(), client=client) == []
    assert client.requests == []


def test_a_sidecar_with_no_text_makes_no_call():
    client = _StubClient(json.dumps({"terms": [_entry()]}))

    assert terms.extract_terms(_sidecar(pages=[]), client=client) == []
    assert client.requests == []
