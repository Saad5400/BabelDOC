"""Phrase-pair alignment: the translation, segmented into aligned phrases.

The LLM that translates a paragraph is also asked — for paragraphs whose input
carries no placeholder scaffolding — to segment its own work into an ordered
list of aligned phrase pairs: «We can not»↔«لا يمكننا», «create»↔«إنشاء», and so
on. Captured into the sidecar (with the page rectangles of each phrase on both
sides), those pairs are what lets a later layout draw matching soft highlights
over the source and its translation.

This module is the data discipline around that idea, in three parts:

- {@link pairs_from_item}: pull a response item's raw "pairs" without trusting
  its shape.
- {@link validate_pairs}: the strict per-paragraph contract. The pairs are
  listed in SOURCE order and aligned by MEANING, so the two sides obey
  different disciplines. The "s" phrases must be a COMPLETE, ORDERED
  segmentation of the source: their word tokens, concatenated, equal the
  source's word tokens. The "t" phrases must be a COMPLETE segmentation of the
  translation IN WHATEVER ORDER the translation actually uses — Arabic
  legitimately reorders a sentence's phrases — which {@link tile_permutation}
  checks by tiling the translation's word tokens with the phrases. Both sides
  split only at word boundaries — an Arabic word split mid-way would be
  re-joined wrongly by any renderer. Anything less is discarded for that
  paragraph only.
- {@link tile_permutation}: the target-order oracle — which pair each
  successive stretch of a text belongs to. validate_pairs derives it once for
  the capture pipeline (and returns it with the pairs); the server's gloss
  highlighter re-derives it from the sidecar's pairs at render time, which is
  safe because the tiler is deterministic.
- {@link match_phrases_to_rects}: map validated phrases onto a paragraph's
  characters and union their boxes per visual line. Works on both sides: the
  ORIGINAL characters (snapshotted before translation) and the TYPESET
  characters (which carry Arabic presentation forms, stripped bidi controls
  and bidi-mirrored brackets — {@link _fold} makes both worlds comparable).

Everything here returns None instead of raising on bad data: pairs are a
bonus on top of a paid translation, and wrong boxes are worse than none.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# How many phrases a single paragraph may reasonably split into. The prompt
# asks for 2-8; anything past this bound is a model runaway, not a segmentation.
MAX_PAIRS = 50

# The tiling search's work budget. Pairs are uploader-influenced at the server
# endpoints, and a duplicate-heavy phrase list can make the backtracking
# explore factorially many placements; past this many candidate probes the
# search fails closed — no pairs rather than an unbounded worker.
MAX_TILING_STEPS = 5000

# Same set the Arabic post-processor strips (il_translator._ARABIC_DIACRITICS_RE):
# tashkeel and Quranic annotation marks. Stripped from BOTH sides before any
# comparison, so a phrase that kept its harakat still matches a post-processed
# paragraph that lost them.
_DIACRITICS_RE = re.compile(
    "[\u064b-\u0655\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed]"
)

# Invisible bidi controls / zero-width characters (typesetting strips these
# from translated text before rendering — see typesetting.BIDI_CONTROL_REGEX).
_CONTROLS_RE = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f"  # ZWSP ZWNJ ZWJ LRM RLM
    "\u061c"  # Arabic Letter Mark
    "\u202a-\u202e"  # LRE RLE PDF LRO RLO
    "\u2066-\u2069"  # LRI RLI FSI PDI
    "\u2060\ufeff]"  # word joiner, BOM/ZWNBSP
)

# Typesetting bidi-mirrors bracket-like punctuation inside RTL visual runs
# ("(" is stored as ")"), so each mirror pair is folded to one canonical
# member before comparison. Derived from typesetting.BIDI_MIRROR_MAP.
_MIRROR_CANON = {
    ")": "(",
    "]": "[",
    "}": "{",
    ">": "<",
    "»": "«",
    "›": "‹",
}


def _fold(text: str) -> str:
    """One comparable form for logical text AND typeset glyphs.

    NFKC turns Arabic presentation forms (what the typeset characters carry
    after arabic_reshaper) back into their logical letters — a lam-alef
    ligature folds to its two letters — and normalizes compatibility
    characters (NBSP, ligated «fi») identically on both sides. Controls,
    diacritics and mirror asymmetry are erased as documented above.
    """
    if not text:
        return ""
    text = _CONTROLS_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS_RE.sub("", text)
    return "".join(_MIRROR_CANON.get(ch, ch) for ch in text)


def _words(text: str) -> list[str]:
    return text.split()


class _SearchBudgetError(Exception):
    """The tiling search hit MAX_TILING_STEPS; the answer is fail-closed."""


def tile_permutation(
    phrases_tokens: list[list[str]],
    full_tokens: list[str],
) -> list[int] | None:
    """How the phrases tile the full text, or None.

    `phrases_tokens` are the pairs' target phrases as word-token lists, in the
    pairs' listed (SOURCE) order; `full_tokens` the full translation's word
    tokens. When the phrases partition the full text in SOME order, the return
    value has one entry per successive tile of the full text: the index of the
    pair whose phrase sits there. A monotonic segmentation answers the identity
    permutation; a reordered one answers where each pair actually landed.

    DETERMINISTIC on purpose, so it can be re-derived downstream from the same
    inputs: at every position the lowest-indexed unused phrase that matches is
    tried first, which makes the answer the lexicographically smallest valid
    permutation — two IDENTICAL phrases are assigned to their target positions
    in ascending pair-index order (identical phrases are interchangeable, so
    any tie-break is equally true; this one is stable).

    Backtracking, bounded two ways: at each position only DISTINCT phrase
    contents are probed (an identical twin of a failed branch fails
    identically), and the whole search stops at {@link MAX_TILING_STEPS}
    probes — pairs are uploader-influenced at the server, and a crafted
    duplicate-heavy list must fail closed, never spin.
    """
    count = len(phrases_tokens)

    if not 1 <= count <= MAX_PAIRS or any(not p for p in phrases_tokens):
        return None

    if sum(len(p) for p in phrases_tokens) != len(full_tokens):
        return None

    used = [False] * count
    permutation: list[int] = []
    steps = 0

    def _place(position: int) -> bool:
        nonlocal steps
        if position == len(full_tokens):
            return True

        tried: set[tuple[str, ...]] = set()

        for index in range(count):
            if used[index]:
                continue

            tokens = phrases_tokens[index]
            content = tuple(tokens)
            if content in tried:
                continue
            tried.add(content)

            steps += 1
            if steps > MAX_TILING_STEPS:
                raise _SearchBudgetError

            if full_tokens[position:position + len(tokens)] != tokens:
                continue

            used[index] = True
            permutation.append(index)
            if _place(position + len(tokens)):
                return True
            used[index] = False
            permutation.pop()

        return False

    try:
        return permutation if _place(0) else None
    except _SearchBudgetError:
        logger.warning(
            "phrase pairs: tiling search exhausted its budget; failing closed"
        )
        return None


def pairs_from_item(item) -> list[dict] | None:
    """A response item's raw pairs, shape-checked but not yet validated.

    Accepts only a non-empty list of {"s": str, "t": str} with non-blank
    values; anything else (missing, wrong type, empty strings, runaway
    length) is None.
    """
    if not isinstance(item, dict):
        return None

    raw = item.get("pairs")

    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_PAIRS:
        return None

    pairs = []
    for entry in raw:
        if not isinstance(entry, dict):
            return None
        s, t = entry.get("s"), entry.get("t")
        if not isinstance(s, str) or not isinstance(t, str):
            return None
        if not s.strip() or not t.strip():
            return None
        pairs.append({"s": s.strip(), "t": t.strip()})

    return pairs


def validate_pairs(
    raw_pairs: list[dict] | None,
    source_text: str,
    target_text: str,
) -> tuple[list[dict], list[int]] | None:
    """The strict segmentation contract — (pairs, permutation) — or None.

    The pairs are listed in SOURCE order and aligned by meaning, so the two
    sides are checked differently. SOURCE: the flattened word tokens of the
    "s" phrases must equal the source's word tokens — completeness, order and
    word-boundary splits in one check (a mid-word split, "can"+"not" for
    "cannot", changes the token list and fails). TARGET: the "t" phrases must
    TILE the translation's word tokens in some permutation
    ({@link tile_permutation}) — the translation may put its phrases in a
    different order than the source, and the pair's list position never
    implies its position in the translation. Word boundaries bind exactly the
    same way: a tile match is token-by-token.

    The target is compared against the text as it was APPLIED to the
    paragraph, which the Arabic post-processor may have altered after the LLM
    produced the pairs; diacritic stripping is absorbed (the stored "t" is
    then the stripped form, matching what the sidecar's target says), any
    other divergence discards the pairs.

    On success the permutation rides along with the pairs: for each
    successive stretch of the target text, the index of the pair sitting
    there. It is derived here — against the exact text the pairs were
    validated on — and carried, not recomputed, to every consumer that must
    walk the translation in ITS order (`attach_target_rects`).
    """
    if not raw_pairs or not source_text or not target_text:
        return None

    if [w for p in raw_pairs for w in _words(p["s"])] != _words(source_text):
        return None

    t_phrases = [p["t"] for p in raw_pairs]

    permutation = tile_permutation(
        [_words(p) for p in t_phrases], _words(target_text)
    )

    if permutation is None:
        t_phrases = [_DIACRITICS_RE.sub("", p).strip() for p in t_phrases]
        stripped_target = _DIACRITICS_RE.sub("", target_text)
        if any(not p for p in t_phrases):
            return None
        permutation = tile_permutation(
            [_words(p) for p in t_phrases], _words(stripped_target)
        )
        if permutation is None:
            return None

    return [
        {"s": " ".join(_words(pair["s"])), "t": " ".join(_words(t))}
        for pair, t in zip(raw_pairs, t_phrases, strict=True)
    ], permutation


def _line_rects(chars: list[tuple[str, tuple | None]]) -> list[list[float]] | None:
    """One phrase's characters unioned per visual line, top line first.

    Same geometric break rule as the sidecar's source lines: a character
    whose TOP sits below the current line's bottom starts a new line. Only
    the y axis decides, so RTL lines (x decreasing) group exactly like LTR
    ones. A phrase whose characters carry no usable boxes yields None — the
    caller treats the whole side as unresolved rather than emitting a pair
    that silently cannot be drawn.
    """
    rects: list[list[float]] = []
    current: list[float] | None = None

    for _text, box in chars:
        if box is None:
            continue
        x0, y0, x1, y1 = box
        if current is None or y1 < current[1]:
            current = [x0, y0, x1, y1]
            rects.append(current)
        else:
            current[0] = min(current[0], x0)
            current[1] = min(current[1], y0)
            current[2] = max(current[2], x1)
            current[3] = max(current[3], y1)

    rects = [r for r in rects if r[2] > r[0] and r[3] > r[1]]
    return rects or None


def match_phrases_to_rects(
    chars: list[tuple[str, tuple | list | None]],
    phrases: list[str],
) -> list[list[list[float]]] | None:
    """Each phrase's page rectangles, or None when the mapping is not exact.

    `chars` is the paragraph's characters in the order the IL stores them —
    (text, [x0, y0, x1, y1]) with the box in the same space the characters
    came from — and `phrases` its validated segmentation in the same order.

    The mapping is whitespace-blind on purpose: the typeset character list
    drops line-leading spaces and the source list interleaves synthesized
    dummy spaces, so word boundaries cannot be trusted in either. Instead the
    NON-SPACE folded characters of the concatenated phrases must consume the
    stream exactly, one to one, start to end. (One typeset glyph may fold to
    several logical characters — a lam-alef ligature — so the stream is built
    per folded character, each remembering its glyph's box.) Any leftover on
    either side means the text on the page is not the text the phrases
    segment, and the answer is None — never a guessed box.
    """
    if not phrases:
        return None

    stream: list[tuple[str, tuple | list | None]] = []
    for text, box in chars:
        for ch in _fold(text or ""):
            if not ch.isspace():
                stream.append((ch, box))

    position = 0
    rects_per_phrase: list[list[list[float]]] = []

    for phrase in phrases:
        wanted = [ch for ch in _fold(phrase) if not ch.isspace()]
        if not wanted:
            return None
        end = position + len(wanted)
        if [ch for ch, _box in stream[position:end]] != wanted:
            return None
        line_rects = _line_rects(stream[position:end])
        if line_rects is None:
            return None
        rects_per_phrase.append(line_rects)
        position = end

    if position != len(stream):
        return None

    return rects_per_phrase
