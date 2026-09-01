"""Tests for server/vocab.py — the per-page vocabulary extraction.

No real LLM call anywhere: a fake client returns canned JSON-mode replies and
records the prompts it was sent.

Run from the repo root (pyproject's testpaths only covers tests/):

    pytest server/test_vocab.py
"""

import json
from types import SimpleNamespace

import pytest

from server import config
from server import vocab


class _FakeClient:
    """Canned chat.completions client: one reply (or exception) per call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.prompts.append(kwargs["messages"][-1]["content"])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=reply))])


def _sidecar(pages: dict[int, str]) -> dict:
    return {"version": 1, "lang_in": "en", "lang_out": "ar",
            "total_pages": len(pages),
            "pages": [{"page_number": number, "mediabox": [0, 0, 595, 842],
                       "blocks": [{"source": text}], "obstacles": []}
                      for number, text in pages.items()]}


def _reply(pages: dict) -> str:
    return json.dumps({"vocab": pages}, ensure_ascii=False)


ENTRY = {"w": "declared", "ar": "يصرح عنه"}


# --------------------------------------------------------------------------
# parse_response
# --------------------------------------------------------------------------

def test_parse_reads_the_wrapped_and_the_bare_shape():
    wrapped = vocab.parse_response(_reply({"0": [ENTRY]}))
    bare = vocab.parse_response(json.dumps({"0": [ENTRY]}))

    assert wrapped == {0: [ENTRY]} == bare


def test_parse_drops_junk_pages_and_junk_entries():
    parsed = vocab.parse_response(json.dumps({"vocab": {
        "0": [ENTRY, "junk", {"w": "", "ar": "x"}, {"w": "scope"}, 7],
        "banana": [ENTRY],
        "-3": [ENTRY],
        "2": "not a list",
    }}))

    assert parsed == {0: [ENTRY]}


def test_parse_keeps_an_optional_note_and_strips_whitespace():
    parsed = vocab.parse_response(json.dumps({"1": [
        {"w": " scope ", "ar": " نطاق ", "note": " مدى صلاحية المتغير "},
        {"w": "custom", "ar": "مخصص", "note": ""},
    ]}, ensure_ascii=False))

    assert parsed == {1: [
        {"w": "scope", "ar": "نطاق", "note": "مدى صلاحية المتغير"},
        {"w": "custom", "ar": "مخصص"},
    ]}


def test_parse_refuses_a_non_object_reply():
    with pytest.raises(ValueError, match="no vocab"):
        vocab.parse_response(json.dumps([ENTRY]))


# --------------------------------------------------------------------------
# The Arabic: no tashkeel, ever
# --------------------------------------------------------------------------

def test_tashkeel_is_stripped_from_the_meaning_and_the_note():
    # The body of every translation is undiacritised by babeldoc's own
    # post-processor; a strip that carries tashkeel carries it alone.
    parsed = vocab.parse_response(json.dumps({"0": [
        {"w": "declared", "ar": "يُصرَّح عنه",
         "note": "تعريف المتغيِّر قبل استخدامه"}]}, ensure_ascii=False))

    assert parsed == {0: [{"w": "declared", "ar": "يصرح عنه",
                           "note": "تعريف المتغير قبل استخدامه"}]}


def test_the_prompts_own_worked_example_carries_no_tashkeel():
    # The example is the strongest instruction in the prompt: a diacritised
    # one is why 10% of the delivered rows came back diacritised.
    diacritics = set("ًٌٍَُِّْٰ")
    example = vocab._SYSTEM_PROMPT.split('{"vocab"')[-1]

    assert '"ar": "يصرح عنه"' in example
    assert not diacritics & set(example)


# --------------------------------------------------------------------------
# extract_vocab: selection rules
# --------------------------------------------------------------------------

def test_entries_come_back_string_keyed_per_page():
    client = _FakeClient([_reply({"0": [ENTRY],
                                  "1": [{"w": "scope", "ar": "نطاق"}]})])

    result = vocab.extract_vocab(
        _sidecar({0: "declared once", 1: "the scope"}), client=client)

    assert result == {"0": [ENTRY], "1": [{"w": "scope", "ar": "نطاق"}]}


def test_a_word_repeated_on_a_later_page_keeps_its_first_occurrence_only():
    client = _FakeClient([_reply({
        "0": [ENTRY],
        "3": [{"w": "Declared", "ar": "آخر"}, {"w": "scope", "ar": "نطاق"}],
    })])

    result = vocab.extract_vocab(
        _sidecar({0: "declared", 3: "declared scope"}), client=client)

    assert result == {"0": [ENTRY], "3": [{"w": "scope", "ar": "نطاق"}]}


def test_the_deep_glossary_terms_are_excluded_and_named_in_the_prompt():
    client = _FakeClient([_reply({"0": [
        {"w": "wrapping", "ar": "تغليف"}, ENTRY]})])

    result = vocab.extract_vocab(_sidecar({0: "wrapping declared"}),
                                 exclude=["Wrapping"], client=client)

    assert result == {"0": [ENTRY]}
    assert "Wrapping" in client.prompts[0]  # the شُرحت سابقًا list


def test_a_page_is_capped_at_twenty_words():
    entries = [{"w": f"word{chr(97 + i // 26)}{chr(97 + i % 26)}", "ar": "معنى"}
               for i in range(30)]
    client = _FakeClient([_reply({"0": entries})])

    result = vocab.extract_vocab(_sidecar({0: "text"}), client=client)

    assert len(result["0"]) == vocab.MAX_PER_PAGE == 20
    assert result["0"] == entries[:20]


def test_the_document_is_capped_at_four_hundred_words():
    pages = {n: [{"w": f"w{n}x{chr(97 + i)}", "ar": "معنى"}
                 for i in range(20)] for n in range(25)}
    client = _FakeClient([_reply({str(n): v for n, v in pages.items()})])

    result = vocab.extract_vocab(
        _sidecar(dict.fromkeys(range(25), "text")), client=client)

    assert sum(len(v) for v in result.values()) == vocab.MAX_TOTAL == 400
    # Ascending pages: the earliest words survive the cap.
    assert "0" in result and "24" not in result  # noqa: PT018


def test_an_empty_document_makes_no_call():
    client = _FakeClient([])

    assert vocab.extract_vocab(_sidecar({}), client=client) == {}
    assert client.prompts == []


# --------------------------------------------------------------------------
# extract_vocab: chunking
# --------------------------------------------------------------------------

def test_a_long_document_is_chunked_and_later_chunks_know_the_earlier_words():
    long_page = "lorem " * 20_000  # two of these exceed CHUNK_WORDS together
    client = _FakeClient([
        _reply({"0": [ENTRY]}),
        _reply({"1": [{"w": "declared", "ar": "آخر"},
                      {"w": "evolved", "ar": "تطور"}]}),
    ])

    result = vocab.extract_vocab(_sidecar({0: long_page, 1: long_page}),
                                 client=client)

    assert len(client.prompts) == 2
    # The second call is told what the first introduced...
    assert "declared" in client.prompts[1]
    assert "declared" not in client.prompts[0]
    # ...and the dedupe holds even if the model repeats it anyway.
    assert result == {"0": [ENTRY], "1": [{"w": "evolved", "ar": "تطور"}]}


def test_a_failing_later_chunk_keeps_the_earlier_chunks_words():
    long_page = "lorem " * 20_000
    client = _FakeClient([_reply({"0": [ENTRY]}), RuntimeError("boom")])

    result = vocab.extract_vocab(_sidecar({0: long_page, 1: long_page}),
                                 client=client)

    assert result == {"0": [ENTRY]}


# --------------------------------------------------------------------------
# extract_vocab: failure posture and the kill switch
# --------------------------------------------------------------------------

def test_a_failing_call_returns_empty():
    client = _FakeClient([RuntimeError("provider down")])

    assert vocab.extract_vocab(_sidecar({0: "text"}), client=client) == {}


def test_a_malformed_reply_returns_empty():
    client = _FakeClient(["{not json"])

    assert vocab.extract_vocab(_sidecar({0: "text"}), client=client) == {}


def test_the_kill_switch_skips_the_call_entirely(monkeypatch):
    monkeypatch.setattr(config, "VOCAB_PAGES", False)
    client = _FakeClient([_reply({"0": [ENTRY]})])

    assert vocab.extract_vocab(_sidecar({0: "text"}), client=client) == {}
    assert client.prompts == []


def test_no_api_key_and_no_client_skips_quietly(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")

    assert vocab.extract_vocab(_sidecar({0: "text"})) == {}
