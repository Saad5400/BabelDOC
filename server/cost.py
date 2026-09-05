"""Real provider cost capture for the doctranslate engine.

The engine used to INVENT its cost: it multiplied BabelDOC's own token counters
by a hardcoded rate table and added a flat per-page surcharge. That number was
wrong in three separate ways — it was list price rather than what the provider
actually billed, it was pinned to rates that did not move when OPENAI_MODEL did,
and the per-page term corresponded to no provider charge at all (on a 41-page
job it was 25% of the reported "cost"). It was also unauditable: nothing recorded
which generations a job produced, so a charge could never be reconciled after the
fact.

OpenRouter already returns the exact figure. Asking for it costs one flag —
`usage: {"include": true}` on the request — and the response then carries
`usage.cost` (the real amount billed, after provider routing and any cache
discount) plus the generation id.

Both of OpenAITranslator's call paths (`do_translate` and `do_llm_translate`)
already pass `extra_body=self.extra_body` and already funnel their response
through `update_token_count`, so hooking those two points captures every call
without duplicating the retry/rate-limit/error handling that wraps them.

Those same two points are also where a HARD provider failure is caught — see
`hard_provider_error`.
"""

import logging
import threading

import openai
from babeldoc.translator.translator import OpenAITranslator

logger = logging.getLogger(__name__)


class HardProviderError(RuntimeError):
    """A provider refusal no retry can fix. Its str() is the job's `error`."""


#: HTTP statuses that mean the key, not the paragraph, is the problem.
#: 429 only reaches us after the SDK's own retry ladder has given up, so by
#: then it is not a blip either.
_HARD_STATUS = {
    401: "authentication rejected",
    402: "insufficient credits",
    403: "key limit exceeded",
    429: "rate limited (retries exhausted)",
}

#: Belt and braces for gateways that dress a terminal refusal up as something
#: else (a 200 with an error body, a 500 relaying the upstream's words).
_HARD_TEXT = ("key limit exceeded", "insufficient credits", "insufficient_quota",
              "invalid api key", "no auth credentials")


def _provider_message(exc) -> str:
    """The provider's OWN sentence, not the SDK's repr of the whole body."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return str(getattr(exc, "message", None) or exc)


def hard_provider_error(exc) -> str | None:
    """The job `error` string when `exc` is terminal, else None.

    401/402/403 are terminal by definition — the key is invalid, out of money,
    or over its limit — and no amount of retrying a paragraph changes that.
    This is what turned a monthly key limit into a full-English PDF the caller
    then charged for: every call 403'd, every paragraph burned three attempts,
    and the run finished "successfully" in the source language.
    """
    # tenacity gives up by raising RetryError around the real one.
    attempt = getattr(exc, "last_attempt", None)
    if attempt is not None:
        try:
            exc = attempt.exception() or exc
        except Exception:  # noqa: BLE001 - a malformed RetryError is not fatal
            pass

    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        # The SDK's own classes, for the rare build that does not carry the
        # status on the exception itself.
        for cls, code in ((openai.AuthenticationError, 401),
                          (openai.PermissionDeniedError, 403),
                          (openai.RateLimitError, 429)):
            if isinstance(exc, cls):
                status = code
                break

    detail = _provider_message(exc).strip().replace("\n", " ")
    label = _HARD_STATUS.get(status)

    if label is None and not any(text in detail.lower() for text in _HARD_TEXT):
        return None

    label = label or "call refused"
    if status is not None:
        label = f"{label}, HTTP {status}"
    return f"provider: {detail[:280]} ({label})" if detail else f"provider: {label}"


def _extract(usage, field):
    """Read a field the OpenAI SDK does not declare (OpenRouter's extras)."""
    value = getattr(usage, field, None)

    if value is None:
        extra = getattr(usage, "model_extra", None) or {}
        value = extra.get(field)

    return value


class CostTrackingTranslator(OpenAITranslator):
    """An OpenAITranslator that knows what it actually spent.

    `cost_usd` is the sum of the provider's own reported per-call cost. It is
    None when NO call reported one — the honest answer to "what did this cost"
    when the endpoint does not say, and the signal the app already handles by
    charging nothing and logging loudly. It is never a locally computed guess:
    a number we cannot reconcile against the provider is worse than no number,
    because it silently becomes someone's credit balance.

    `priced_calls` vs `calls` is the coverage of that figure, reported so a
    partial answer is visibly partial rather than quietly low.

    It also fails the run FAST on a provider refusal no retry can fix
    (`hard_provider_error`): the first one cancels the translation and every
    later call short-circuits, so a spent key ends the job in seconds with a
    specific error instead of quietly degrading to a full-English PDF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #: Set by the pipeline once the config exists, so the first hard error
        #: can cancel the run rather than merely being remembered.
        self.translation_config = None
        self._hard_error: str | None = None

        # Ask OpenRouter to bill-stamp every response. Harmless on endpoints
        # that do not understand it — they simply return no `cost`, which the
        # coverage counters below then report as unpriced.
        self.extra_body["usage"] = {"include": True}

        self._lock = threading.Lock()
        self._cost_usd = 0.0
        self._calls = 0
        self._priced_calls = 0
        self._generation_ids: list[str] = []

    def hard_error(self) -> str | None:
        """The first terminal provider refusal of this run, if there was one."""
        with self._lock:
            return self._hard_error

    def _call(self, method, *args, **kwargs):
        """Every provider call of the run goes through here.

        babeldoc's per-paragraph handler catches Exception and leaves the
        paragraph in its source language, which is the right posture for a
        model that garbled one paragraph and the wrong one for a key that is
        out of quota. So the terminal class is recognised here, once, and it
        cancels the whole run.
        """
        hard = self.hard_error()
        if hard:
            # Not a network round-trip: this call would fail identically, and
            # 3 attempts x every remaining paragraph of a spent key is minutes
            # of waiting for an answer we already have.
            raise HardProviderError(hard)

        try:
            return method(*args, **kwargs)
        except Exception as exc:
            reason = hard_provider_error(exc)
            if reason is None:
                raise

            with self._lock:
                first = self._hard_error is None
                if first:
                    self._hard_error = reason

            if first:
                logger.error("aborting the run — %s", reason)
                if self.translation_config is not None:
                    self.translation_config.cancel_translation()

            raise HardProviderError(reason) from exc

    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        return self._call(super().do_translate, text, rate_limit_params)

    def do_llm_translate(self, text, rate_limit_params: dict = None):
        return self._call(super().do_llm_translate, text, rate_limit_params)

    def update_token_count(self, response):
        super().update_token_count(response)

        try:
            cost = _extract(response.usage, "cost") if response.usage else None
        except Exception:
            logger.exception("could not read usage.cost from the response")
            cost = None

        generation_id = getattr(response, "id", None)

        with self._lock:
            self._calls += 1

            if cost is not None:
                self._cost_usd += float(cost)
                self._priced_calls += 1

            # Keep the ids so a finished job can be reconciled against the
            # provider's own dashboard later. Bounded: a big deck is thousands
            # of calls and the whole list rides in the job record.
            if generation_id and len(self._generation_ids) < 2000:
                self._generation_ids.append(str(generation_id))

    def spend(self) -> dict:
        """This run's real spend, for the job record."""
        with self._lock:
            return {
                "cost_usd": round(self._cost_usd, 6) if self._priced_calls else None,
                "calls": self._calls,
                "priced_calls": self._priced_calls,
                "generation_ids": list(self._generation_ids),
            }
