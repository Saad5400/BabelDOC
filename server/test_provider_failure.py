"""What the engine does when the PROVIDER is the thing that failed.

The incident this pins down: the OpenRouter key hit its monthly limit, every
call came back `403 {"error": {"message": "Key limit exceeded (monthly
limit)..."}}`, babeldoc's per-paragraph handler caught each one and left the
paragraph in English, and the job finished `done` with a full-English PDF that
the caller then charged a user for. Three things had to be true for that, and
each has a test here:

1. nothing counted how much of the document was actually translated;
2. a refusal no retry can fix was retried three times per paragraph and then
   swallowed;
3. "finished" and "translated" were the same word.

Run from the repo root (pyproject's testpaths only covers tests/):

    pytest server/test_provider_failure.py
"""

import copy
import json
import threading
from pathlib import Path

import httpx
import openai
import pymupdf
import pytest
from babeldoc.format.pdf.translation_config import TranslationConfig
from tenacity import RetryError

from server import config
from server import cost
from server import jobs
from server import pipeline
from server.conftest import TOKEN
from server.cost import CostTrackingTranslator
from server.cost import HardProviderError
from server.cost import hard_provider_error

KEY_LIMIT = ("Key limit exceeded (monthly limit). Add credits or wait for the "
             "limit to reset.")


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _openai_error(cls, status: int, message: str):
    """The real SDK exception a provider refusal actually arrives as."""
    body = {"error": {"message": message, "code": status}}
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/x")
    response = httpx.Response(status, request=request, json=body)
    return cls(f"Error code: {status} - {body}", response=response, body=body)


def _text_pdf(path: Path, pages: int = 2) -> None:
    """A digital PDF with enough English on each page to classify as text."""
    doc = pymupdf.open()
    for index in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"The quick brown fox jumps over page "
                                   f"number {index} of this document.",
                         fontsize=14)
    doc.save(str(path))
    doc.close()


class _StubConfig:
    """A stand-in for TranslationConfig — the real one loads a layout model.

    It borrows the real `record_translation_coverage`, because that accumulator
    (and the fact that a split part shares the dict) is part of what is under
    test here.
    """

    record_translation_coverage = TranslationConfig.record_translation_coverage

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.translation_coverage = {"total": 0, "untranslated": 0}
        self.cancelled = False

    def cancel_translation(self):
        self.cancelled = True


class _StubTranslator:
    """Enough of CostTrackingTranslator for the pipeline's two uses of it."""

    def __init__(self, *_args, **_kwargs):
        self.translation_config = None
        self._hard = None
        self.prompt_token_count = type("V", (), {"value": 120})()
        self.completion_token_count = type("V", (), {"value": 90})()

    def hard_error(self):
        return self._hard

    def spend(self):
        return {"cost_usd": 0.0123, "calls": 4, "priced_calls": 4,
                "generation_ids": ["gen-1"]}


def _bare_translator(translation_config=None) -> CostTrackingTranslator:
    """A CostTrackingTranslator without its OpenAI client — `_call` only."""
    translator = object.__new__(CostTrackingTranslator)
    translator._lock = threading.Lock()
    translator._hard_error = None
    translator.translation_config = translation_config
    return translator


# --------------------------------------------------------------------------
# 1. recognising a refusal that retrying cannot fix
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("cls", "status", "message"), [
    (openai.PermissionDeniedError, 403, KEY_LIMIT),
    (openai.AuthenticationError, 401, "No auth credentials found"),
    (openai.APIStatusError, 402, "Insufficient credits. Add more to continue."),
    (openai.RateLimitError, 429, "Rate limit exceeded"),
])
def test_a_terminal_provider_refusal_is_recognised(cls, status, message):
    reason = hard_provider_error(_openai_error(cls, status, message))

    assert reason is not None
    assert reason.startswith("provider: ")
    assert message[:30] in reason
    assert f"HTTP {status}" in reason


def test_the_monthly_key_limit_reads_as_itself():
    reason = hard_provider_error(
        _openai_error(openai.PermissionDeniedError, 403, KEY_LIMIT))

    assert "Key limit exceeded (monthly limit)" in reason
    assert "key limit exceeded" in reason  # the label, alongside the words


def test_tenacity_giving_up_is_unwrapped():
    """A 429 that survived the SDK's retry ladder arrives inside a RetryError."""
    inner = _openai_error(openai.RateLimitError, 429, "Rate limit exceeded")
    attempt = type("Attempt", (), {"exception": lambda _self: inner})()

    assert hard_provider_error(RetryError(attempt)) is not None


def test_a_refusal_disguised_as_something_else_is_still_caught():
    """Some gateways relay the upstream's words under a status of their own."""
    assert hard_provider_error(
        RuntimeError("upstream said: Insufficient credits")) is not None


@pytest.mark.parametrize("exc", [
    ValueError("translation result identical to input"),
    _openai_error(openai.BadRequestError, 400, "context length exceeded"),
    _openai_error(openai.InternalServerError, 500, "upstream hiccup"),
])
def test_an_ordinary_failure_is_not_terminal(exc):
    """Only the key's problems end a run; a bad paragraph is still retried."""
    assert hard_provider_error(exc) is None


# --------------------------------------------------------------------------
# 2. one hard refusal ends the run
# --------------------------------------------------------------------------

def test_the_first_hard_refusal_cancels_the_translation():
    translation_config = _StubConfig()
    translator = _bare_translator(translation_config)

    def _refuse(_text, _params):
        raise _openai_error(openai.PermissionDeniedError, 403, KEY_LIMIT)

    with pytest.raises(HardProviderError) as caught:
        translator._call(_refuse, "hello", None)

    assert "Key limit exceeded" in str(caught.value)
    assert translation_config.cancelled
    assert translator.hard_error() == str(caught.value)


def test_later_calls_do_not_reach_the_provider_at_all():
    """3 attempts x every remaining paragraph, for an answer already known."""
    translator = _bare_translator(_StubConfig())
    translator._hard_error = "provider: key limit exceeded"
    calls = []

    with pytest.raises(HardProviderError):
        translator._call(lambda *a: calls.append(a), "hello", None)

    assert calls == []


def test_both_provider_call_paths_go_through_the_guard():
    translator = _bare_translator(_StubConfig())

    assert translator.do_llm_translate(None) is None  # passes through

    translator._hard_error = "provider: key limit exceeded"
    for call in (lambda: translator.do_llm_translate("x", {}),
                 lambda: translator.do_translate("x", {})):
        with pytest.raises(HardProviderError):
            call()  # no client on this instance: it never got that far


def test_an_ordinary_failure_passes_straight_through():
    translator = _bare_translator(_StubConfig())

    def _boom(_text, _params):
        raise ValueError("model returned the source unchanged")

    with pytest.raises(ValueError):
        translator._call(_boom, "hello", None)

    assert translator.hard_error() is None  # the next paragraph still runs


# --------------------------------------------------------------------------
# 3. coverage: babeldoc's own count, kept
# --------------------------------------------------------------------------

def test_coverage_accumulates_across_split_parts():
    """Split parts are shallow copies, so they share the one dict."""
    translation_config = _StubConfig()

    translation_config.record_translation_coverage(100, 3)
    translation_config.record_translation_coverage(80, 1)

    assert pipeline._coverage(translation_config) == {
        "paragraphs_total": 180,
        "paragraphs_translated": 176,
        "paragraphs_untranslated": 4,
    }


def test_a_split_part_counts_into_its_parent():
    """babeldoc makes each part a `copy.copy` of the config: one shared dict."""
    parent = _StubConfig()

    copy.copy(parent).record_translation_coverage(10, 2)

    assert parent.translation_coverage == {"total": 10, "untranslated": 2}


def test_coverage_of_a_run_that_never_recorded_anything():
    assert pipeline._coverage(object()) == {
        "paragraphs_total": 0,
        "paragraphs_translated": 0,
        "paragraphs_untranslated": 0,
    }


# --------------------------------------------------------------------------
# 4. _run_babeldoc: coverage banked, hard refusal raised ahead of everything
# --------------------------------------------------------------------------

@pytest.fixture()
def babeldoc_stub(monkeypatch, tmp_path):
    """`_run_babeldoc` with its babeldoc imports replaced, not its own logic."""
    import babeldoc.format.pdf.high_level as high_level
    import babeldoc.format.pdf.translation_config as translation_config_module

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(config, "GLOSSARY_PATH", tmp_path / "no-glossary.csv")
    monkeypatch.setattr(translation_config_module, "TranslationConfig",
                        _StubConfig)

    state: dict = {"events": [{"type": "finish", "translate_result": "RESULT"}]}

    def _translator(*args, **kwargs):
        translator = _StubTranslator(*args, **kwargs)
        translator._hard = state.get("hard")
        return translator

    monkeypatch.setattr(cost, "CostTrackingTranslator", _translator)

    async def _fake_async_translate(tc):
        tc.record_translation_coverage(*state.get("counts", (40, 1)))
        for event in state["events"]:
            yield event

    monkeypatch.setattr(high_level, "async_translate", _fake_async_translate)
    return state


def _call_run_babeldoc(job_id, tmp_path):
    return pipeline._run_babeldoc(job_id, tmp_path / "in.pdf",
                                  tmp_path / "out", lang_in="en", lang_out="ar",
                                  fmt="translated", scanned=False,
                                  progress_base=2.0)


def test_a_finished_run_banks_its_coverage_on_the_job(babeldoc_stub, tmp_path):
    job = jobs.create_job(b"%PDF-1.4\n", filename="x.pdf", lang_in="en",
                          lang_out="ar", fmt="translated", title=None)

    result, _translator, coverage = _call_run_babeldoc(job["job_id"], tmp_path)

    assert result == "RESULT"
    assert coverage == {"paragraphs_total": 40, "paragraphs_translated": 39,
                        "paragraphs_untranslated": 1}
    assert jobs.read_job(job["job_id"])["coverage"] == coverage


def test_a_hard_refusal_beats_every_other_verdict(babeldoc_stub, tmp_path):
    """Even a run babeldoc calls finished: the key is why nothing translated."""
    job = jobs.create_job(b"%PDF-1.4\n", filename="x.pdf", lang_in="en",
                          lang_out="ar", fmt="translated", title=None)
    babeldoc_stub["counts"] = (40, 40)
    babeldoc_stub["hard"] = f"provider: {KEY_LIMIT} (key limit exceeded, HTTP 403)"

    with pytest.raises(HardProviderError) as caught:
        _call_run_babeldoc(job["job_id"], tmp_path)

    assert "Key limit exceeded (monthly limit)" in str(caught.value)
    # Recorded before the raise: a failed job still says how far it got.
    assert jobs.read_job(job["job_id"])["coverage"]["paragraphs_translated"] == 0


# --------------------------------------------------------------------------
# 5. run_job: what the caller is finally told
# --------------------------------------------------------------------------

@pytest.fixture()
def run_job_stub(monkeypatch, tmp_path):
    """A job that reaches the translate step, with babeldoc itself stubbed."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")

    def _fake_run_cmd(argv, _job_id, stage):
        if stage == "image_prep":
            Path(argv[-1]).write_text(json.dumps({"pages": []}))

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_run_cmd)

    state: dict = {"counts": (40, 1)}

    def _fake_run_babeldoc(job_id, _input_pdf, out_dir, **_kwargs):
        if state.get("raise"):
            raise state["raise"]
        produced = Path(out_dir) / "mono.pdf"
        _text_pdf(produced, pages=1)
        translation_config = _StubConfig()
        translation_config.record_translation_coverage(*state["counts"])
        coverage = pipeline._coverage(translation_config)
        jobs.update_job(job_id, coverage=coverage)
        result = type("R", (), {"no_watermark_mono_pdf_path": produced,
                                "mono_pdf_path": produced})()
        return result, _StubTranslator(), coverage

    monkeypatch.setattr(pipeline, "_run_babeldoc", _fake_run_babeldoc)
    return state


def _submit(tmp_path) -> str:
    source = tmp_path / "source.pdf"
    _text_pdf(source)
    job = jobs.create_job(source.read_bytes(), filename="source.pdf",
                          lang_in="en", lang_out="ar", fmt="translated",
                          title=None)
    return job["job_id"]


def _run_as_worker(job_id: str) -> dict:
    """jobs._worker_loop's error boundary, so the test sees a real verdict."""
    try:
        pipeline.run_job(job_id)
    except Exception as exc:  # noqa: BLE001 - mirrors the worker
        jobs.update_job(job_id, status="failed", error=str(exc)[:2000])
    return jobs.read_job(job_id)


def test_a_good_run_reports_its_coverage_alongside_its_cost(run_job_stub,
                                                            tmp_path):
    job = _run_as_worker(_submit(tmp_path))

    assert job["status"] == "done"
    assert job["usage"]["paragraphs_total"] == 40
    assert job["usage"]["paragraphs_translated"] == 39
    assert job["usage"]["paragraphs_untranslated"] == 1
    assert job["usage"]["cost_usd"] == 0.0123  # the spend fields are untouched
    assert jobs.result_path(job["job_id"]).is_file()


def test_a_mostly_untranslated_run_is_refused(run_job_stub, tmp_path):
    run_job_stub["counts"] = (40, 9)  # 22.5%, well past the 5% default

    job = _run_as_worker(_submit(tmp_path))

    assert job["status"] == "failed"
    assert job["error"] == "translation incomplete: 9 of 40 paragraphs untranslated"
    assert job["usage"] is None  # nothing for the caller to charge against
    assert job["coverage"]["paragraphs_untranslated"] == 9
    # The partial output stays on disk for debugging; /result 409s regardless.
    assert jobs.result_path(job["job_id"]).is_file()


def test_a_loss_inside_the_threshold_still_ships(run_job_stub, tmp_path):
    run_job_stub["counts"] = (100, 5)  # exactly 5%: at the line, not over it

    assert _run_as_worker(_submit(tmp_path))["status"] == "done"


def test_the_threshold_is_tunable(run_job_stub, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_UNTRANSLATED_RATIO", 0.5)
    run_job_stub["counts"] = (40, 9)

    assert _run_as_worker(_submit(tmp_path))["status"] == "done"


def test_a_spent_key_fails_the_job_and_bills_nothing(run_job_stub, tmp_path):
    reason = f"provider: {KEY_LIMIT} (key limit exceeded, HTTP 403)"
    run_job_stub["raise"] = HardProviderError(reason)

    job = _run_as_worker(_submit(tmp_path))

    assert job["status"] == "failed"
    assert job["error"] == reason
    assert "Key limit exceeded (monthly limit)" in job["error"]
    assert job["usage"] is None


# --------------------------------------------------------------------------
# 6. what GET /v1/jobs/{id} shows (the body is an explicit allowlist)
# --------------------------------------------------------------------------

def _store(**fields) -> str:
    job = jobs.create_job(b"%PDF-1.4\n", filename="x.pdf", lang_in="en",
                          lang_out="ar", fmt="translated", title=None)
    jobs.update_job(job["job_id"], **fields)
    return job["job_id"]


def test_a_done_job_shows_coverage_inside_usage(client):
    job_id = _store(status="done",
                    usage={"cost_usd": 0.02, "paragraphs_total": 40,
                           "paragraphs_translated": 39,
                           "paragraphs_untranslated": 1})

    body = client.get(f"/v1/jobs/{job_id}",
                      headers={"X-Internal-Token": TOKEN}).json()

    assert body["usage"]["paragraphs_translated"] == 39
    assert "coverage" not in body  # `usage` already carries it


def test_a_failed_job_shows_how_far_it_got(client):
    job_id = _store(status="failed",
                    error="provider: Key limit exceeded (monthly limit).",
                    coverage={"paragraphs_total": 412,
                              "paragraphs_translated": 0,
                              "paragraphs_untranslated": 412})

    body = client.get(f"/v1/jobs/{job_id}",
                      headers={"X-Internal-Token": TOKEN}).json()

    assert body["error"].startswith("provider: Key limit exceeded")
    assert body["coverage"]["paragraphs_translated"] == 0
    assert "usage" not in body  # the charge signal is never on a failed job


def test_a_failed_job_from_before_coverage_existed_still_answers(client):
    job_id = _store(status="failed", error="server restarted")

    body = client.get(f"/v1/jobs/{job_id}",
                      headers={"X-Internal-Token": TOKEN}).json()

    assert body["error"] == "server restarted"
    assert "coverage" not in body


def test_the_result_of_a_refused_run_is_409(client):
    job_id = _store(status="failed",
                    error="translation incomplete: 40 of 40 paragraphs "
                          "untranslated")
    jobs.result_path(job_id).write_bytes(b"%PDF-1.4\n")  # kept for debugging

    response = client.get(f"/v1/jobs/{job_id}/result",
                          headers={"X-Internal-Token": TOKEN})

    assert response.status_code == 409
