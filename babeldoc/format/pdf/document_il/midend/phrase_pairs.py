"""Phrase-pair alignment: the translation, segmented into aligned phrases.

The LLM that translates a paragraph is also asked to segment its own work into
an ordered list of aligned phrase pairs: «We can not»↔«لا يمكننا»,
«create»↔«إنشاء», and so on. Style tags are stripped before anything is
compared — the phrases cover the plain text — and formula placeholders ride
along as OPAQUE WORDS: a `{vN}` token counts as one word of both texts (babeldoc
classifies a slide bullet's «•» as a formula, so on real decks most body
paragraphs carry one), is validated exactly like any other word, and is
expanded back to the formula's own text afterwards
({@link expand_formula_tokens}) so the sidecar stores only real page text.
Captured into the sidecar (with the page rectangles of each phrase on both
sides), those pairs are what lets a later layout draw matching soft highlights
over the source and its translation.

This module is the data discipline around that idea, in three parts:

- {@link pairs_from_item}: pull a response item's raw "pairs" without trusting
  its shape.
- {@link validate_pairs}: the strict per-paragraph contract, preceded by a
  deterministic NORMALIZATION pass over the raw pairs. Word-level granularity
  makes the model produce two systematic, mechanically recoverable shapes:
  consecutive source words that each repeat the same "t" (a fertility group —
  "software"/"systems" both answering «أنظمة برمجية») are merged into one
  multi-word pair, and — only as a retry when strict validation still fails —
  a "t" that is nothing but an Arabic proclitic («و», «لـ») is fused into the
  FOLLOWING pair, the way the output actually spells it («ومختلفة»). The
  merged pairs are what validation checks and what the sidecar stores.
- The strict contract itself: the pairs are
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

# A formula placeholder token, in the exact canonical form the translator
# mints ({@link translator.OpenAITranslator.get_formular_placeholder}) and the
# prompt orders the model to echo verbatim. Tokenization treats each one as a
# word of its own even when the source welded it to a neighbour — a bulleted
# slide line arrives as "{v1}Software products…" — because the model is asked
# to place the token in a phrase as a separate word. A sloppy echo ("{ v1 }")
# simply fails validation's token equality, which is the usual fate of any
# altered word.
FORMULA_TOKEN_RE = re.compile(r"\{v\d+\}")

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
    """Word tokens, with each `{vN}` formula token split out as its own word.

    The isolation matters on both sides: the SOURCE welds a bullet's token to
    the first word ("{v1}Software"), and the model may weld or space its echo
    in the output; either way the token itself is atomic — it stands for
    characters no word boundary can enter — and everything around it splits
    normally.
    """
    return FORMULA_TOKEN_RE.sub(lambda m: f" {m.group(0)} ", text).split()


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


# Arabic tatweel (kashida): a typographic filler the model uses to write a
# clitic "as attached" («لـ»). Meaningful only to the clitic merge below —
# real outputs may legitimately contain it («الـ API» is house style), so it
# is never blanket-stripped from phrases or targets.
_TATWEEL = "ـ"

# Arabic proclitic prefixes the model sometimes emits as a "t" of their own
# while its actual output fuses them into the following word. Conjunctions
# (و ف), prepositions (ب ل ك), future س, and the definite article with its
# fused preposition/conjunction forms.
_ARABIC_PROCLITICS = frozenset(
    ["و", "ف", "ب", "ل", "ك", "س", "ال", "لل", "وال", "بال", "كال", "فال"]
)


def _merge_repeated_targets(pairs: list[dict]) -> list[dict]:
    """Consecutive pairs sharing one "t" merged into one multi-word pair.

    Word-level granularity invites fertility repeats: the model answers
    "software"→«أنظمة برمجية» and "systems"→«أنظمة برمجية» although the target
    contains «أنظمة برمجية» once. While pair i and i+1 have the same "t"
    (string equality after whitespace normalization), they are one pair:
    "s" values joined with a space, "t" kept once. A single left-to-right
    pass with accumulation reaches the fixed point — a run of any length
    collapses into its head.
    """
    merged: list[dict] = []
    for pair in pairs:
        if merged and merged[-1]["t"].split() == pair["t"].split():
            merged[-1] = {"s": f"{merged[-1]['s']} {pair['s']}",
                          "t": merged[-1]["t"]}
        else:
            merged.append(dict(pair))
    return merged


def _merge_clitic_targets(pairs: list[dict]) -> list[dict] | None:
    """Proclitic-only "t" values fused into the FOLLOWING pair, or None.

    The model sometimes splits an Arabic clitic off as its own pair —
    "and"→«و», "to"→«لـ» — while its actual output fuses the clitic into the
    next word («ومختلفة», «لمجموعة»). Any pair whose whole "t" is one of
    {@link _ARABIC_PROCLITICS} (after stripping tatweel and whitespace) is
    merged into its successor: "s" values joined with a space, "t" values
    joined with NO space using the tatweel-stripped clitic (the successor's
    own "t" is kept verbatim — it may carry a legitimate tatweel). Walking
    right to left lets chained clitics fold into an already-merged successor.

    A trailing clitic has no successor and is left alone — that candidate
    simply fails validation, which is the point: this is a RETRY shape, and
    None (nothing merged) tells the caller there is no retry to run.
    """
    merged: list[dict] = []
    changed = False
    for pair in reversed(pairs):
        clitic = pair["t"].replace(_TATWEEL, "").strip()
        if merged and clitic in _ARABIC_PROCLITICS:
            following = merged[0]
            merged[0] = {"s": f"{pair['s']} {following['s']}",
                         "t": f"{clitic}{following['t']}"}
            changed = True
        else:
            merged.insert(0, dict(pair))
    return merged if changed else None


def validate_pairs(
    raw_pairs: list[dict] | None,
    source_text: str,
    target_text: str,
) -> tuple[list[dict], list[int]] | None:
    """Normalization, then the strict contract — (pairs, permutation) — or None.

    The raw pairs first pass through a deterministic, purely mechanical
    normalization ({@link _merge_repeated_targets}); the strict validation
    below then runs UNCHANGED on the result. If it fails, one retry candidate
    is built ({@link _merge_clitic_targets}) and validated the same way; if
    both fail, the paragraph's pairs are discarded exactly as before. No
    model recall, no fuzzy matching — every merge is forced by the pairs
    themselves, and a merged pair behaves downstream exactly like a
    hand-written multi-word pair.

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

    candidate = _merge_repeated_targets(raw_pairs)
    validated = _validate_candidate(candidate, source_text, target_text)
    if validated is not None:
        return validated

    retry = _merge_clitic_targets(candidate)
    if retry is None:
        return None
    return _validate_candidate(retry, source_text, target_text)


def _validate_candidate(
    pairs: list[dict],
    source_text: str,
    target_text: str,
) -> tuple[list[dict], list[int]] | None:
    """{@link validate_pairs}' strict contract, checked on ONE candidate."""
    if [w for p in pairs for w in _words(p["s"])] != _words(source_text):
        return None

    t_phrases = [p["t"] for p in pairs]

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
        for pair, t in zip(pairs, t_phrases, strict=True)
    ], permutation


def expand_formula_tokens(
    pairs: list[dict],
    expansions: dict[str, str],
) -> list[dict] | None:
    """VALIDATED pairs with every `{vN}` replaced by its formula's text, or None.

    Validation runs over the tokenized texts, where a `{vN}` is an opaque
    word; the sidecar must store only real page text — the snapshotted source
    characters and the typeset target characters both spell the formula's
    actual glyphs (the bullet «•», the inline symbol), and every downstream
    consumer (rect matching, gloss highlighting, compose) reads the phrases
    against those streams. So the LAST step of a capture is this substitution,
    with `expansions` mapping each canonical token to the text its formula's
    own characters spell (the same characters {@link
    il_translator.ILTranslator.parse_translate_output} resolves the token back
    to).

    Fail-closed rules, in the module's spirit (wrong text is worse than none —
    any violation discards the whole paragraph's pairs):

    - Every token must be a known key, and the "s" phrases and "t" phrases
      must carry the SAME token multiset. Validation already pins the "s" side
      to the source's tokens; a token the model dropped from its translation —
      or invented on one side only — leaves the sides unequal.
    - A token whose expansion is empty or whitespace-only is dropped from its
      phrase; a phrase left EMPTY by that is not merged into a neighbour (the
      two sides could disagree on which neighbour) — the pairs are discarded.

    Pairs without tokens pass through unchanged.
    """
    s_tokens: list[str] = []
    t_tokens: list[str] = []
    expanded: list[dict] = []

    for pair in pairs:
        new_pair: dict[str, str] = {}
        for side, seen in (("s", s_tokens), ("t", t_tokens)):
            words: list[str] = []
            for word in _words(pair[side]):
                if FORMULA_TOKEN_RE.fullmatch(word):
                    if word not in expansions:
                        return None
                    seen.append(word)
                    replacement = expansions[word].strip()
                    if replacement:
                        words.append(replacement)
                else:
                    words.append(word)
            if not words:
                return None
            new_pair[side] = " ".join(words)
        expanded.append(new_pair)

    if sorted(s_tokens) != sorted(t_tokens):
        return None

    return expanded


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
