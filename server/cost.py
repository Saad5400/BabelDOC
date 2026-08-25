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
"""

import logging
import threading

from babeldoc.translator.translator import OpenAITranslator

logger = logging.getLogger(__name__)


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
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ask OpenRouter to bill-stamp every response. Harmless on endpoints
        # that do not understand it — they simply return no `cost`, which the
        # coverage counters below then report as unpriced.
        self.extra_body["usage"] = {"include": True}

        self._lock = threading.Lock()
        self._cost_usd = 0.0
        self._calls = 0
        self._priced_calls = 0
        self._generation_ids: list[str] = []

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
