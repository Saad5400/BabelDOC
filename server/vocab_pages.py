"""The «كلمات هذه الصفحة» pages: per-page vocabulary rendered and interleaved.

`server/vocab.py` picks each page's NEW English words and short phrases and
gives each a concise Arabic meaning; this module is the one renderer and the
one inserter of those entries, shared by every artifact they ride in — the
mono result (server/pipeline.py), the recomposed duals (server/compose.py) and
the interlinear overlay (server/interlinear.py). A page's vocab page goes
IMMEDIATELY AFTER that page's content, never to the end of the document — the
reader meets the words while the page is still in front of them.

DESIGN: a quick word list, not a set of explanations. A blue #1d4ed8 accent on
the English word, #111827 body, #64748b muted title and notes, and rows rather
than anything heavier: a small muted title («كلمات هذه الصفحة»), then tight
RTL rows of bold English word — Arabic meaning — optional muted note, with a
soft #F1F5F9 stripe on alternate rows. More than six entries flow into two
columns (right column first — the page is RTL), which is what keeps a full
20-word page (the extraction's per-page cap) on one inserted slide-sized page;
a page that still overflows continues onto a second inserted page.

TEXT and FONTS are the shared page-fonts machinery (server/page_fonts.py):
`PageFonts` subsets the regular + bold GoNotoKurrent faces once per document,
`measure` runs the Story.place pass that decides row heights and page breaks,
and every caller saves with `garbage=4` for the same per-box-font-copy reason.
"""

from __future__ import annotations

import logging

import pymupdf

from server.page_fonts import PageFonts
from server.page_fonts import _escape
from server.page_fonts import measure

logger = logging.getLogger("doctranslate.vocab_pages")

_TITLE_COLOR = "#64748b"   # the title is deliberately subtle
_WORD_COLOR = "#1d4ed8"
_BODY_COLOR = "#111827"
_NOTE_COLOR = "#64748b"
_STRIPE_BG = (0xF1 / 255, 0xF5 / 255, 0xF9 / 255)  # #F1F5F9 as pymupdf floats

# Page geometry (points). The page is sized like the content page it follows;
# these are the metrics inside it.
_MARGIN_X = 48.0
_MARGIN_TOP = 40.0
_MARGIN_BOTTOM = 40.0
_TITLE_GAP = 12.0
_ROW_PAD_X = 8.0
# Vertical rhythm sized so a FULL page — 20 single-line rows, the extraction's
# per-page cap, as two columns of ten — still fits the shortest common slide
# (16:9 at 720x405 pt) on one inserted page.
_ROW_PAD_Y = 3.0
_ROW_GAP = 1.5
_COLUMN_GAP = 20.0
_TWO_COLUMNS_ABOVE = 6      # >6 entries = two columns

_TITLE = "كلمات هذه الصفحة"

# The pipeline's own entries are already clean and capped (vocab._clean_entry,
# vocab.MAX_PER_PAGE / MAX_TOTAL), but the compose/overlay sidecar is
# uploader-supplied, so the same bounds are re-imposed here before anything is
# measured or subset — a crafted sidecar must not buy unbounded pages or CPU
# out of one request.
_MAX_ROWS = 20    # mirrors vocab.MAX_PER_PAGE
_MAX_TOTAL = 400  # mirrors vocab.MAX_TOTAL
_CLIP = {"w": 80, "ar": 120, "note": 200}

_CSS = (
    f"div.vtitle {{font-family: glossbold; font-size: 14px;"
    f" color: {_TITLE_COLOR};}}"
    f"div.row {{font-size: 13px; color: {_BODY_COLOR}; line-height: 1.5;}}"
    f"span.w {{font-family: glossbold; font-size: 14px;"
    f" color: {_WORD_COLOR};}}"
    f"span.note {{font-size: 11px; color: {_NOTE_COLOR};}}"
)


def _usable(entry: object) -> dict | None:
    """One sidecar item as a clipped renderable row, or None."""
    if not isinstance(entry, dict):
        return None

    word = str(entry.get("w") or "").strip()[:_CLIP["w"]]
    arabic = str(entry.get("ar") or "").strip()[:_CLIP["ar"]]

    if not word or not arabic:
        return None

    row = {"w": word, "ar": arabic}
    note = str(entry.get("note") or "").strip()[:_CLIP["note"]]

    if note:
        row["note"] = note

    return row


def sanitize_vocab(vocab: object) -> dict[int, list[dict]]:
    """The sidecar's "vocab" value as clean int-keyed pages of rows.

    Junk keys and junk items are dropped silently; the per-page and
    per-document caps are re-imposed (ascending page order, so an over-budget
    document keeps its EARLIEST words — the first-occurrence rule's spirit).
    """
    if not isinstance(vocab, dict):
        return {}

    pages: dict[int, list[dict]] = {}

    for key, items in vocab.items():
        try:
            number = int(key)
        except (TypeError, ValueError):
            continue

        if number < 0 or number in pages or not isinstance(items, list):
            continue

        rows = [row for row in map(_usable, items)
                if row is not None][:_MAX_ROWS]

        if rows:
            pages[number] = rows

    total = 0
    out: dict[int, list[dict]] = {}

    for number in sorted(pages):
        if total >= _MAX_TOTAL:
            break

        rows = pages[number][:_MAX_TOTAL - total]
        out[number] = rows
        total += len(rows)

    return out


def _row_html(row: dict) -> str:
    parts = [f'<span class="w">{_escape(row["w"])}</span>',
             f" — {_escape(row['ar'])}"]

    if row.get("note"):
        parts.append(f' <span class="note">· {_escape(row["note"])}</span>')

    return f'<div class="row" dir="rtl">{"".join(parts)}</div>'


def _title_html() -> str:
    return f'<div class="vtitle" dir="rtl">{_TITLE}</div>'


def make_fonts(pages: dict[int, list[dict]]) -> PageFonts:
    """One subset of the two faces covering every row in the document."""
    texts = [_TITLE + " —·"]
    texts += [str(value) for rows in pages.values()
              for row in rows for value in row.values()]

    return PageFonts(texts, _CSS)


def draw_vocab_pages(doc: pymupdf.Document, at_index: int, rows: list[dict],
                     page_size: tuple[float, float], fonts: PageFonts) -> int:
    """One content page's rows drawn as 1+ new pages in `doc` at `at_index`;
    returns pages added.

    The pages are created and drawn DIRECTLY in `doc` (new_page +
    insert_htmlbox), never rendered into a scratch document and grafted in
    with insert_pdf: the target usually already embeds its own
    identically-named GoNotoKurrent subsets (every babeldoc output does), and
    drawing in place keeps the vocab pages' subsets out of the page-grafting
    machinery instead of trusting it to keep same-named, differently-numbered
    subsets apart.

    0 when nothing can be drawn — no usable rows, or a page too small for
    the layout (the deliberate skip, not an exception); `doc` is untouched
    then.
    """
    width, height = float(page_size[0]), float(page_size[1])
    content_width = width - 2 * _MARGIN_X
    bottom = height - _MARGIN_BOTTOM
    two_columns = len(rows) > _TWO_COLUMNS_ABOVE
    column_width = ((content_width - _COLUMN_GAP) / 2 if two_columns
                    else content_width)
    text_width = column_width - 2 * _ROW_PAD_X

    if not rows or text_width <= 0 or bottom <= _MARGIN_TOP:
        if rows:
            logger.warning("vocab: page %sx%s too small for rows; skipped",
                           width, height)
        return 0

    page = doc.new_page(pno=at_index, width=width, height=height)
    added = 1

    title = _title_html()
    title_height = measure(title, fonts, content_width)
    page.insert_htmlbox(
        pymupdf.Rect(_MARGIN_X, _MARGIN_TOP, width - _MARGIN_X,
                     _MARGIN_TOP + title_height + 2),
        title, css=fonts.css, archive=fonts.archive)

    column_top = _MARGIN_TOP + title_height + _TITLE_GAP
    # An RTL page fills its right column first.
    columns = ([width - _MARGIN_X - column_width, _MARGIN_X]
               if two_columns else [_MARGIN_X])
    # Balanced, not overflow-driven: with room to spare the rows still split
    # roughly half and half, so eight entries read as two short columns
    # rather than one long one beside an empty half.
    split = (len(rows) + 1) // 2 if two_columns else None
    column = 0
    y = column_top

    for index, row in enumerate(rows):
        html = _row_html(row)
        text_height = measure(html, fonts, text_width)
        row_height = text_height + 2 * _ROW_PAD_Y

        if ((split is not None and index == split and column == 0)
                or (y + row_height > bottom and y > column_top)):
            column += 1

            if column < len(columns):
                y = column_top
            else:
                # Overflow: continue onto a further page (no repeated title),
                # created right after the one that just filled up.
                page = doc.new_page(pno=at_index + added,
                                    width=width, height=height)
                added += 1
                column = 0
                column_top = _MARGIN_TOP
                split = None  # continuation pages just flow
                y = column_top

        x0 = columns[column]
        row_rect = pymupdf.Rect(x0, y, x0 + column_width,
                                min(y + row_height, bottom))

        if index % 2 == 1:
            page.draw_rect(row_rect, color=None, fill=_STRIPE_BG,
                           radius=min(0.5, 3.0 / max(row_rect.height, 1.0)))

        # scale_low=0 lets a row taller than a whole column shrink to fit
        # rather than truncate; a normal row is measured to fit exactly.
        page.insert_htmlbox(
            pymupdf.Rect(row_rect.x0 + _ROW_PAD_X, row_rect.y0 + _ROW_PAD_Y,
                         row_rect.x1 - _ROW_PAD_X, row_rect.y1 - _ROW_PAD_Y),
            html, css=fonts.css, archive=fonts.archive, scale_low=0)
        y = row_rect.y1 + _ROW_GAP

    return added


def interleave_vocab(doc: pymupdf.Document, vocab: object,
                     anchors: dict[int, int]) -> dict[int, int]:
    """Insert each page's vocab page right after its anchor; pages added per
    content page.

    `anchors` maps a content page number (the vocab dict's key) to the index
    IN `doc` of the page its vocab must follow — the caller owns that mapping,
    because it differs per artifact (page N itself in the mono and the
    overlay, the end of page N's pair in a dual). Anchors outside `doc` are
    skipped, insertion runs back-to-front so the pre-insertion indices stay
    true, each inserted page is sized like its anchor page, and the pages are
    drawn directly into `doc` ({@link draw_vocab_pages}) — never grafted from
    a scratch document. The caller owns the save (`garbage=4`, as everywhere
    the Story engine draws).
    """
    pages = sanitize_vocab(vocab)
    plan = [(anchors[number], number) for number in pages
            if number in anchors and 0 <= anchors[number] < doc.page_count]

    if not plan:
        return {}

    fonts = make_fonts({number: pages[number] for _anchor, number in plan})
    added: dict[int, int] = {}

    for anchor, number in sorted(plan, reverse=True):
        rect = doc[anchor].rect
        count = draw_vocab_pages(doc, anchor + 1, pages[number],
                                 (rect.width, rect.height), fonts)

        if count:
            added[number] = count

    if added:
        logger.info("vocab: %s page(s) inserted for %s content page(s)",
                    sum(added.values()), len(added))

    return added
