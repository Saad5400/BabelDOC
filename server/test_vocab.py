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


def _sidecar(pages: dict[int, str], targets: dict[int, str] | None = None
             ) -> dict:
    """A sidecar whose pages carry `pages` as source — and, where given, the
    run's own Arabic for that page."""
    targets = targets or {}

    return {"version": 1, "lang_in": "en", "lang_out": "ar",
            "total_pages": len(pages),
            "pages": [{"page_number": number, "mediabox": [0, 0, 595, 842],
                       "blocks": [{"source": text,
                                   "target": targets.get(number, "")}],
                       "obstacles": []}
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


def test_parse_says_out_loud_which_entries_a_bad_key_took_with_it(caplog):
    with caplog.at_level("WARNING", logger="doctranslate.vocab"):
        vocab.parse_response(json.dumps({"banana": [ENTRY, ENTRY]}))

    assert "banana" in caplog.text
    assert "2 item(s)" in caplog.text


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
# Placement: the page is DERIVED from the text, never the model's key
# --------------------------------------------------------------------------

def test_an_entry_lands_on_the_page_its_word_first_occurs_on():
    # The run14 shape: slides that print their own number as the first text
    # block, and a model that echoed the printed number instead of the marker.
    client = _FakeClient([_reply({
        "2": [{"w": "subtasks", "ar": "مهام فرعية"}],
        "3": [{"w": "Parallelogram", "ar": "متوازي أضلاع"}],
    })])

    result = vocab.extract_vocab(_sidecar({
        0: "1 | Title",
        1: "2 | breaking a task into smaller subtasks",
        2: "3 | The Parallelogram denotes input",
    }), client=client)

    assert result == {"1": [{"w": "subtasks", "ar": "مهام فرعية"}],
                      "2": [{"w": "Parallelogram", "ar": "متوازي أضلاع"}]}


def test_a_key_past_the_end_of_the_document_keeps_its_words():
    # `vocab["41"]` on a 41-page document used to be dropped without a log.
    client = _FakeClient([_reply({"3": [{"w": "postcondition", "ar": "شرط لاحق"}]})])

    result = vocab.extract_vocab(
        _sidecar({0: "one", 1: "two", 2: "a postcondition holds after"}),
        client=client)

    assert result == {"2": [{"w": "postcondition", "ar": "شرط لاحق"}]}


def test_a_whole_word_elsewhere_beats_a_substring_earlier():
    client = _FakeClient([_reply({"0": [{"w": "bug", "ar": "خلل"}]})])

    result = vocab.extract_vocab(
        _sidecar({0: "debugging the program", 1: "a bug in the code"}),
        client=client)

    assert result == {"1": [{"w": "bug", "ar": "خلل"}]}


def test_a_word_that_occurs_nowhere_is_dropped_and_logged(caplog):
    client = _FakeClient([_reply({"0": [ENTRY,
                                        {"w": "violates", "ar": "ينتهك"}]})])

    with caplog.at_level("INFO", logger="doctranslate.vocab"):
        result = vocab.extract_vocab(_sidecar({0: "declared here"}),
                                     client=client)

    assert result == {"0": [ENTRY]}
    assert "violates" in caplog.text


def test_a_code_identifier_is_dropped():
    client = _FakeClient([_reply({"0": [
        {"w": "boolean_expression", "ar": "تعبير منطقي"},
        {"w": "non-functional", "ar": "غير وظيفي"}]})])

    result = vocab.extract_vocab(
        _sidecar({0: "if boolean_expression: non-functional requirements"}),
        client=client)

    assert result == {"0": [{"w": "non-functional", "ar": "غير وظيفي"}]}


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


def test_one_word_family_is_explained_once():
    # "emerging" on page 14 and "emerged" on page 24 are one word.
    client = _FakeClient([_reply({
        "0": [{"w": "emerging", "ar": "ناشئة"}],
        "1": [{"w": "emerged", "ar": "ظهرت"}],
    })])

    result = vocab.extract_vocab(
        _sidecar({0: "emerging fields", 1: "it emerged later"}), client=client)

    assert result == {"0": [{"w": "emerging", "ar": "ناشئة"}]}


def test_the_deep_glossary_terms_are_excluded_and_named_in_the_prompt():
    client = _FakeClient([_reply({"0": [
        {"w": "wrapping", "ar": "تغليف"}, ENTRY]})])

    result = vocab.extract_vocab(_sidecar({0: "wrapping declared"}),
                                 exclude=["Wrapping"], client=client)

    assert result == {"0": [ENTRY]}
    assert "Wrapping" in client.prompts[0]  # the شُرحت سابقًا list


def test_a_page_is_capped_at_twenty_words():
    words = [f"word{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(30)]
    entries = [{"w": word, "ar": "معنى"} for word in words]
    client = _FakeClient([_reply({"0": entries})])

    result = vocab.extract_vocab(_sidecar({0: " ".join(words)}), client=client)

    assert len(result["0"]) == vocab.MAX_PER_PAGE == 20
    assert result["0"] == entries[:20]


def test_the_document_is_capped_at_four_hundred_words():
    words = {n: [f"w{n}x{chr(97 + i)}" for i in range(20)] for n in range(25)}
    pages = {n: [{"w": word, "ar": "معنى"} for word in row]
             for n, row in words.items()}
    client = _FakeClient([_reply({str(n): v for n, v in pages.items()})])

    result = vocab.extract_vocab(
        _sidecar({n: " ".join(row) for n, row in words.items()}), client=client)

    assert sum(len(v) for v in result.values()) == vocab.MAX_TOTAL == 400
    # Ascending pages: the earliest words survive the cap.
    assert "0" in result and "24" not in result  # noqa: PT018


def test_an_empty_document_makes_no_call():
    client = _FakeClient([])

    assert vocab.extract_vocab(_sidecar({}), client=client) == {}
    assert client.prompts == []


# --------------------------------------------------------------------------
# The prompt: the page's own translation travels with its source
# --------------------------------------------------------------------------

def test_the_prompt_carries_the_runs_own_arabic_for_the_page():
    # Without it the strip says «نمذجة البرمجيات» four centimetres under a
    # body that says «النمطية».
    client = _FakeClient([_reply({"0": [{"w": "Modularity", "ar": "النمطية"}]})])

    vocab.extract_vocab(_sidecar({0: "Modularity is the process"},
                                 {0: "النمطية (Modularity) هي عملية"}),
                        client=client)

    prompt = client.prompts[0]

    assert "النمطية (Modularity) هي عملية" in prompt
    assert "— الصفحة 0 —" in prompt


def test_a_page_without_a_translation_still_travels():
    client = _FakeClient([_reply({"0": [ENTRY]})])

    vocab.extract_vocab(_sidecar({0: "declared once"}), client=client)

    assert "declared once" in client.prompts[0]


def test_two_words_sharing_a_meaning_without_a_note_are_logged(caplog):
    client = _FakeClient([_reply({"0": [{"w": "implementing", "ar": "تنفيذ"},
                                        {"w": "execute", "ar": "تنفيذ"}]})])

    with caplog.at_level("WARNING", logger="doctranslate.vocab"):
        result = vocab.extract_vocab(
            _sidecar({0: "implementing and execute"}), client=client)

    assert len(result["0"]) == 2
    assert "تنفيذ" in caplog.text


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

    result = vocab.extract_vocab(
        _sidecar({0: f"declared {long_page}", 1: f"evolved {long_page}"}),
        client=client)

    assert len(client.prompts) == 2
    # The second call is told what the first introduced...
    assert "declared" in client.prompts[1]
    assert "declared" not in client.prompts[0].split("نص المستند:")[0]
    # ...and the dedupe holds even if the model repeats it anyway.
    assert result == {"0": [ENTRY], "1": [{"w": "evolved", "ar": "تطور"}]}


def test_the_page_translations_count_against_the_chunk_budget():
    half = "lorem " * 16_000  # under CHUNK_WORDS alone, over it with its twin
    client = _FakeClient([_reply({}), _reply({})])

    vocab.extract_vocab(_sidecar({0: half}, {0: half}), client=client)

    assert len(vocab._chunks(vocab._page_texts(
        _sidecar({0: half, 1: half}, {0: half, 1: half})))) == 2


def test_a_failing_later_chunk_keeps_the_earlier_chunks_words():
    long_page = "lorem " * 20_000
    client = _FakeClient([_reply({"0": [ENTRY]}), RuntimeError("boom")])

    result = vocab.extract_vocab(
        _sidecar({0: f"declared {long_page}", 1: long_page}), client=client)

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
