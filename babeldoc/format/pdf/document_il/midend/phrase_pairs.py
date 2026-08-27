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
- {@link validate_pairs}: the strict per-paragraph contract. The phrases must
  be a COMPLETE, ORDERED segmentation of both texts, split only at word
  boundaries — an Arabic word split mid-way would be re-joined wrongly by any
  renderer. Formally: the word tokens of the phrases, concatenated, must equal
  the word tokens of the full text, on both sides. Anything less is discarded
  for that paragraph only.
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
) -> list[dict] | None:
    """The strict segmentation contract, or None.

    Both sides must reproduce their full text when the phrases are
    concatenated with single spaces (after whitespace normalization), and
    every split must fall on a whitespace word boundary. Both requirements
    are one check: the flattened word tokens of the phrases must equal the
    word tokens of the full text — a mid-word split ("can"+"not" for
    "cannot") changes the token list and fails.

    The target is compared against the text as it was APPLIED to the
    paragraph, which the Arabic post-processor may have altered after the LLM
    produced the pairs; diacritic stripping is absorbed (the stored "t" is
    then the stripped form, matching what the sidecar's target says), any
    other divergence discards the pairs.
    """
    if not raw_pairs or not source_text or not target_text:
        return None

    if [w for p in raw_pairs for w in _words(p["s"])] != _words(source_text):
        return None

    t_phrases = [p["t"] for p in raw_pairs]

    if [w for p in t_phrases for w in _words(p)] != _words(target_text):
        t_phrases = [_DIACRITICS_RE.sub("", p).strip() for p in t_phrases]
        stripped_target = _DIACRITICS_RE.sub("", target_text)
        if any(not p for p in t_phrases) or [
            w for p in t_phrases for w in _words(p)
        ] != _words(stripped_target):
            return None

    return [
        {"s": " ".join(_words(pair["s"])), "t": " ".join(_words(t))}
        for pair, t in zip(raw_pairs, t_phrases, strict=True)
    ]


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
