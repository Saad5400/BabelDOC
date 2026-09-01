#!/usr/bin/env python
"""Rebuild a clean OCR sandwich PDF for scanned slides (wave-2 OCR-prep).

Replaces the ocrmypdf text layer entirely: renders each raster page, runs
tesseract hOCR, then writes ONE invisible text line per surviving hOCR line
directly into the page content stream AFTER the page image.

Fixes fed to BabelDOC (--ocr-workaround):
1. Garbage glyphs: words with low confidence, symbol-junk tokens, and leading
   bullet-marker misreads («¢», «e», «*») are stripped before translation.
   Wave-3: 1-2 char tokens need conf>=75 (thin brace/border strokes read as
   «|», «l», «I», «1»), and extreme-aspect single chars die outright.
2. Logo/graphic regions (wave-3, raster-sampled): every surviving word's bbox
   is sampled in the 300-dpi render. Words sitting on a non-white background
   (colored slide, photo, infographic card) die; words whose neighbourhood is
   dense with ink (crest emblem next to a wordmark) die; short colored lines
   in the page corner bands (logo lockups) die. Lines mostly dead from these
   rules die entirely. No text => no white mask => pixels stay pristine.
   Contagion is now sourced ONLY from graphic/corner kills and only spreads
   into short lines, so junk strokes (braces, borders) no longer eat the
   leading words of real sentences (the wave-2 "scattered labels" bug).
   Wave-4: clean high-confidence words whose surrounding ring is one solid
   NON-white color (title text on a teal card) are rescued from the graphic
   kills — the fork now paints bg-matched masks, so they can be translated.
3. Staircase alignment: every bullet line on a page is stretched (via text
   matrix x-scale) to a common right margin, so BabelDOC's per-paragraph
   in-box RTL alignment lands every translated line on the same right edge.
   Wave-5: wrapped continuation lines are wired to their parent item (tight
   vertical gap, contained span) and share the group's font size; markers
   the OCR missed are inferred from siblings, and every marker is re-emitted
   as a textual "• " prefix so the translated RTL paragraph keeps its bullet
   (BabelDOC's ocr_workaround regroup pass re-merges each group into ONE
   paragraph that wraps as a unit).
4. Sparse look: font-size hint is derived from real glyph heights
   (min(x_size, 1.35 * median word height)) instead of ocrmypdf's smaller
   guess, so translated text is typeset near the original size.

Usage: ocr_prep.py in.pdf out.pdf [--dpi 300] [--lang eng+ara]
                                  [--keep-pages 0,1,2] [--debug out_dbg.pdf]

`--keep-pages` names the 0-based pages that already have a real text layer.
This pass rewrites the text of every page it touches, so a mixed document —
a scanned cover in front of nine good slides — must hand it only the pages
that were actually scanned.
"""

import functools
import html
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

logger = logging.getLogger("doctranslate.ocr_prep")

DPI = 300
CONF_KILL = 60          # words below this confidence die
SHORT_CONF_KILL = 75    # 1-2 char tokens need this much confidence
JUNK_ZONE_PAD_PT = 15   # padding (pt) around killed words for contagion
CORNER_FRAC_X = 0.18    # corner zone width fraction
CORNER_FRAC_Y = 0.22    # corner zone height fraction
TINY_LINE_PT = 8.0      # corner lines shorter than this (x_size, pt) die
# raster-sampling thresholds (wave-3 logo/graphic protection)
WHITE_LVL = 200         # r,g,b all >= this => "white" (paper)
SAT_LVL = 45            # max-min channel spread >= this => "colored"
BG_INK_KILL = 0.55      # word bbox non-white fraction above this => on graphic
NEIGH_INK_KILL = 0.50   # expanded-bbox non-white fraction above this => beside graphic
CORNER_X = 0.20         # corner band (fraction of page) for colored-line kill
CORNER_Y = 0.22
CORNER_COLOR_RATIO = 0.40   # colored/ink ratio above this in corner => logo text
# wave-4 solid-background rescue: clean, confident text sitting on a FLAT
# colored fill (teal title card, banner) survives the graphic kills — BabelDOC
# now paints a background-matched mask instead of a white box. Logos, photos
# and diagrams stay protected: the ring around their words is never one solid
# non-white color (and corner/wordmark kills still apply after the rescue).
SOLID_CONF = 80         # rescue needs at least this word confidence
SOLID_BG_FRAC = 0.80    # ring dominant-color fraction to call the bg "solid"
STROKE_CHARS = set("Il1|!i/\\()[]{}jJtf")
MARKER_SET = {"¢", "e", "o", "©", "*", "•", "·", "-", "–", "—", "="}
SINGLE_CHAR_KEEP = {"&", "#", "+", "%", "@", "$", "=", "?", "!", "7"}
JUNK_ONLY = re.compile(r"^[|<>_\-~^`*.,;:'\"()\[\]{}\\/¢©®™°·—–\s]+$")

# ---------------------------------------------------------------------------
# Which language tesseract is told to read
# ---------------------------------------------------------------------------
# Hardcoding `-l eng` made tesseract read every Arabic run on a mixed scan as
# Latin noise — a real scanned page came back as `algo Gill`, `sow Josi`,
# `Pxé` — and that noise was then dutifully translated and drawn over the
# reader's own page. The realistic scanned input for this product is not an
# English-only handout; it is an English handout with an Arabic university
# header, or an instructor's Arabic annotations.
#
# ISO 639-1 (what the job carries) -> the ISO 639-2/T code tesseract names its
# model files by. Only the languages this product plausibly meets; anything
# already 3 letters is passed through as a tesseract code.
ISO1_TO_TESSERACT = {
    "ar": "ara", "de": "deu", "en": "eng", "es": "spa", "fa": "fas",
    "fr": "fra", "he": "heb", "hi": "hin", "id": "ind", "it": "ita",
    "ja": "jpn", "ko": "kor", "ms": "msa", "nl": "nld", "pt": "por",
    "ru": "rus", "tr": "tur", "ur": "urd", "zh": "chi_sim",
}
DEFAULT_TESSERACT_LANG = "eng"


@functools.lru_cache(maxsize=1)
def installed_languages() -> frozenset[str]:
    """The models this image actually carries.

    Asked rather than assumed, because naming a model that is not installed
    does not degrade — tesseract exits non-zero and the whole OCR stage fails.
    An image built without `tesseract-ocr-ara` therefore keeps working exactly
    as it did, one line quieter in the log.
    """
    try:
        proc = subprocess.run(["tesseract", "--list-langs"],
                              capture_output=True, text=True, timeout=30,
                              check=False)
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    # First line is a header ("List of available languages ...").
    return frozenset(line.strip() for line in proc.stdout.splitlines()[1:]
                     if line.strip())


def tesseract_lang(*wanted: str) -> str:
    """A `-l` argument covering `wanted`, restricted to what is installed.

    Order is preserved (tesseract weights the first language) and duplicates
    collapse, so `tesseract_lang("en", "ar")` is `"eng+ara"` on the engine
    image and `"eng"` on a box that only has English.
    """
    available = installed_languages()
    codes: list[str] = []
    missing: list[str] = []
    for name in wanted:
        if not name:
            continue
        code = ISO1_TO_TESSERACT.get(name.split("-")[0].lower(), name.lower())
        if code in codes:
            continue
        if available and code not in available:
            missing.append(code)
            continue
        codes.append(code)
    if missing:
        logger.warning("tesseract model(s) %s are not installed; OCR will run "
                       "as %s", "+".join(missing), "+".join(codes) or
                       DEFAULT_TESSERACT_LANG)
    return "+".join(codes) or DEFAULT_TESSERACT_LANG


class Word:
    __slots__ = ("x", "y", "x2", "y2", "conf", "text", "dead", "contagion",
                 "reason", "solid")

    def __init__(self, x, y, x2, y2, conf, text):
        self.x, self.y, self.x2, self.y2 = x, y, x2, y2
        self.conf, self.text, self.dead = conf, text, False
        self.contagion = True  # kept for marker bookkeeping
        self.reason = None     # why the word died (conf/junk/graphic/corner/...)
        self.solid = False     # wave-4: clean text on a solid colored card

    def kill(self, reason):
        self.dead = True
        self.reason = reason


class Raster:
    """Pixel sampler over the 300-dpi page render."""

    def __init__(self, pix):
        self.buf = pix.samples
        self.w, self.h = pix.width, pix.height
        self.n, self.stride = pix.n, pix.stride

    def stats(self, x0, y0, x1, y1, max_samples=1600):
        """(ink_frac, colored_ink_ratio) inside the clipped box."""
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.w, int(x1)), min(self.h, int(y1))
        if x1 <= x0 or y1 <= y0:
            return 0.0, 0.0
        area = (x1 - x0) * (y1 - y0)
        step = max(1, int((area / max_samples) ** 0.5))
        total = ink = colored = 0
        buf, n, stride = self.buf, self.n, self.stride
        for y in range(y0, y1, step):
            row = y * stride
            for x in range(x0, x1, step):
                i = row + x * n
                r, g, b = buf[i], buf[i + 1], buf[i + 2]
                total += 1
                if r >= WHITE_LVL and g >= WHITE_LVL and b >= WHITE_LVL:
                    continue
                ink += 1
                if max(r, g, b) - min(r, g, b) >= SAT_LVL:
                    colored += 1
        if not total:
            return 0.0, 0.0
        return ink / total, (colored / ink if ink else 0.0)

    def ring_profile(self, x0, y0, x1, y1, pad, max_samples=1200):
        """(dominant_frac, dominant_rgb) of the ring just outside the box."""
        ox0, oy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        ox1, oy1 = min(self.w, int(x1 + pad)), min(self.h, int(y1 + pad))
        if ox1 <= ox0 or oy1 <= oy0:
            return 0.0, (255, 255, 255)
        area = (ox1 - ox0) * (oy1 - oy0)
        step = max(1, int((area / max_samples) ** 0.5))
        buckets = {}
        buf, n, stride = self.buf, self.n, self.stride
        total = 0
        for y in range(oy0, oy1, step):
            row = y * stride
            inside_y = y0 <= y < y1
            for x in range(ox0, ox1, step):
                if inside_y and x0 <= x < x1:
                    continue        # ring only: skip the glyph box itself
                i = row + x * n
                r, g, b = buf[i], buf[i + 1], buf[i + 2]
                key = (r // 32, g // 32, b // 32)
                cnt, sr, sg, sb = buckets.get(key, (0, 0, 0, 0))
                buckets[key] = (cnt + 1, sr + r, sg + g, sb + b)
                total += 1
        if not total:
            return 0.0, (255, 255, 255)
        cnt, sr, sg, sb = max(buckets.values())
        return cnt / total, (sr // cnt, sg // cnt, sb // cnt)


class Line:
    def __init__(self, x, y, x2, y2, x_size, baseline_off, words):
        self.x, self.y, self.x2, self.y2 = x, y, x2, y2
        self.x_size, self.baseline_off = x_size, baseline_off
        self.words = words
        self.is_list = False
        self.stretch_right = None
        self.dead = False
        self.marker_left = None     # x of a stripped bullet marker (mask must cover it)
        self.group_root = None      # wrapped-paragraph root this line continues
        self.size_override = None   # cluster-normalized font size (pt)

    def alive_words(self):
        return [w for w in self.words if not w.dead]


def parse_hocr(path):
    """Minimal hOCR parser -> list of Line (careas/pars flattened)."""
    txt = Path(path).read_text(encoding="utf-8")
    lines = []
    # split on line-level spans; tesseract emits ocr_line/ocr_header/ocr_caption/ocr_textfloat
    line_re = re.compile(
        r"<span class='ocr_(?:line|header|caption|textfloat)' [^>]*title=\"([^\"]*)\"[^>]*>(.*?)</span>\s*</span>",
        re.S,
    )
    word_re = re.compile(
        r"<span class='ocrx_word' [^>]*title=['\"]([^'\"]*)['\"][^>]*>(.*?)</span>", re.S
    )
    tag_re = re.compile(r"<[^>]+>")
    for m in line_re.finditer(txt):
        title, body = m.group(1), m.group(2) + "</span>"
        bb = re.search(r"bbox (\d+) (\d+) (\d+) (\d+)", title)
        xs = re.search(r"x_size ([\d.]+)", title)
        bl = re.search(r"baseline ([-\d.]+) ([-\d.]+)", title)
        words = []
        for wm in word_re.finditer(body):
            wt, wbody = wm.group(1), wm.group(2)
            wb = re.search(r"bbox (\d+) (\d+) (\d+) (\d+)", wt)
            wc = re.search(r"x_wconf (\d+)", wt)
            text = html.unescape(tag_re.sub("", wbody)).strip()
            if not text or not wb:
                continue
            words.append(Word(*map(int, wb.groups()),
                              int(wc.group(1)) if wc else 0, text))
        if not words or not bb:
            continue
        lines.append(Line(*map(int, bb.groups()),
                          float(xs.group(1)) if xs else 0.0,
                          float(bl.group(2)) if bl else 0.0, words))
    return lines


def boxes_overlap(a, b, pad=0.0):
    return not (a.x2 + pad < b.x or b.x2 + pad < a.x
                or a.y2 + pad < b.y or b.y2 + pad < a.y)


def on_solid_card(raster, w):
    """Wave-4 rescue: is this word clean text on a flat colored fill?

    White-dominant rings never rescue: dense ink surrounded by paper is a
    graphic (crest, chart), and words on paper were not bg-killed anyway.
    """
    if w.conf < SOLID_CONF or len(w.text) < 3:
        return False
    if not any(c.isalnum() for c in w.text):
        return False
    # tight ring: big title glyphs must not drag neighbor words' ink into it
    h = max(w.y2 - w.y, 1)
    pad = max(8, min(40, 0.5 * h))
    frac, (r, g, b) = raster.ring_profile(w.x, w.y, w.x2, w.y2, pad)
    if frac < SOLID_BG_FRAC:
        return False
    return not (r >= WHITE_LVL and g >= WHITE_LVL and b >= WHITE_LVL)


def filter_words(lines, page_w_px, page_h_px, px_per_pt, raster):
    # pass 1: confidence + junk-token + marker strip
    for ln in lines:
        for i, w in enumerate(ln.words):
            wpx, hpx = w.x2 - w.x, w.y2 - w.y
            if w.conf < CONF_KILL:
                w.kill("conf")
            elif JUNK_ONLY.match(w.text):
                w.kill("junk")
            elif len(w.text) == 1 and not w.text.isalnum() \
                    and w.text not in SINGLE_CHAR_KEEP:
                w.kill("junk")
            # wave-3: thin strokes (braces, table borders) read as chars
            elif len(w.text) == 1 and w.text in STROKE_CHARS \
                    and hpx > 3 * max(wpx, 1):
                w.kill("stroke")
            elif len(w.text) <= 2 and w.conf < SHORT_CONF_KILL \
                    and not (w.text.isdigit() and w.conf >= 65):
                w.kill("shortconf")
        # leading bullet-marker strip (independent of confidence)
        alive = ln.alive_words()
        if alive:
            first = alive[0]
            rest = [w for w in alive[1:]]
            if (first.text in MARKER_SET
                    and (first.x2 - first.x) <= 0.015 * page_w_px
                    and rest and rest[0].x - first.x2 >= 0.5 * (first.x2 - first.x)):
                first.kill("marker")
                first.contagion = False
                ln.is_list = True
                ln.marker_left = first.x
        # bullets sometimes die by conf/junk before the marker check:
        # a killed single-char first word with the same geometry also marks a list line
        if not ln.is_list and ln.words:
            f = ln.words[0]
            if f.dead and len(f.text) == 1 and (f.x2 - f.x) <= 0.015 * page_w_px:
                nxt = next((w for w in ln.words[1:] if not w.dead), None)
                if nxt and nxt.x - f.x2 >= 0.5 * (f.x2 - f.x):
                    ln.is_list = True
                    f.contagion = False
                    ln.marker_left = f.x

    # pass 2: tiny corner lines (logo lockups in page corners)
    for ln in lines:
        in_corner_x = ln.x2 < CORNER_FRAC_X * page_w_px or ln.x > (1 - CORNER_FRAC_X) * page_w_px
        in_corner_y = ln.y2 < CORNER_FRAC_Y * page_h_px or ln.y > (1 - CORNER_FRAC_Y) * page_h_px
        if in_corner_x and in_corner_y and ln.x_size < TINY_LINE_PT * px_per_pt:
            for w in ln.words:
                w.kill("corner")

    # pass 2.5 (wave-3): raster-sampled graphic/logo protection
    for ln in lines:
        n_words = len(ln.words)
        # the WHOLE line must sit inside a corner band (logo lockups do;
        # page-wide headers that merely start in a corner don't)
        line_in_bx = (ln.x > (1 - CORNER_X) * page_w_px
                      or ln.x2 < CORNER_X * page_w_px)
        line_in_by = (ln.y2 < CORNER_Y * page_h_px
                      or ln.y > (1 - CORNER_Y) * page_h_px)
        cratios = {}
        for w in ln.words:
            if w.dead:
                continue
            h = max(w.y2 - w.y, 1)
            ink, cratio = raster.stats(w.x, w.y, w.x2, w.y2)
            cratios[id(w)] = cratio
            rescued = None                   # lazy: on_solid_card scans the raster
            if ink > BG_INK_KILL:            # colored slide / photo / filled card
                rescued = on_solid_card(raster, w)
                if not rescued:
                    w.kill("graphic")
                    continue
            nink, _ = raster.stats(w.x - h, w.y - h, w.x2 + h, w.y2 + h)
            if nink > NEIGH_INK_KILL:        # dense graphic right next door (crest)
                if rescued is None:
                    rescued = on_solid_card(raster, w)
                if not rescued:
                    w.kill("graphic")
                    continue
            w.solid = bool(rescued)
            if line_in_bx and line_in_by and n_words <= 6 \
                    and cratio > CORNER_COLOR_RATIO:
                w.kill("corner")             # colored short corner line = logo text

        alive = ln.alive_words()
        # brand wordmark anywhere on the page: short ALL-CAPS colored line in
        # a SMALL face («UMM AL-QURA UNIVERSITY» sits mid-page on some title
        # layouts). Large gold headings (SE2202) must survive: cap the size.
        # Solid-rescued lines get a wider cap: the centered logo lockup on
        # full-teal divider pages is bigger (~14pt) but still no title.
        caps_max = 16.0 if alive and all(w.solid for w in alive) else 12.0
        if alive and len(alive) <= 4 \
                and 0 < ln.x_size < caps_max * px_per_pt:
            if all(len(w.text) >= 2 and w.text.upper() == w.text
                   and any(c.isalpha() for c in w.text) for w in alive):
                mean_cr = sum(cratios.get(id(w), 0.0) for w in alive) / len(alive)
                if mean_cr >= 0.5:
                    for w in alive:
                        w.kill("corner")
        # isolated colored fragment inside a text line: calligraphy /
        # graphic shards OCR'd as letters far away from the real sentence
        alive = ln.alive_words()
        if len(alive) >= 2:
            alive.sort(key=lambda w: w.x)
            for i, w in enumerate(alive):
                h = max(w.y2 - w.y, 1)
                gaps = []
                if i > 0:
                    gaps.append(w.x - alive[i - 1].x2)
                if i < len(alive) - 1:
                    gaps.append(alive[i + 1].x - w.x2)
                if gaps and min(gaps) > 3 * h \
                        and cratios.get(id(w), 0.0) >= 0.6 \
                        and not w.solid:
                    w.kill("graphic")
        # a lone surviving single-character is almost always a stray stroke
        # (non-alphanumeric survivors like «&» are calligraphy shards no
        # matter how confident tesseract is about them)
        alive = ln.alive_words()
        if len(alive) == 1 and len(alive[0].text) == 1 \
                and (alive[0].conf < 90 or not alive[0].text.isalnum()):
            alive[0].kill("stroke")

    # if a line lost half its words to graphic/corner kills, it IS the graphic
    # (solid-card words are exempt: high conf + flat ring is strong evidence)
    for ln in lines:
        gdead = sum(1 for w in ln.words if w.reason in ("graphic", "corner"))
        if gdead and gdead >= 0.5 * len(ln.words):
            for w in ln.words:
                if not w.dead and not w.solid:
                    w.kill("graphic")

    # a rescued (solid) word inside a line that is mostly junk/low-conf is a
    # misread calligraphy/emblem shard (logo lockups), not card text
    for ln in lines:
        ndead = sum(1 for w in ln.words
                    if w.reason in ("conf", "junk", "stroke", "shortconf"))
        if ndead >= 0.5 * len(ln.words):
            for w in ln.words:
                if not w.dead and w.solid:
                    w.kill("graphic")

    # pass 3: contagion — ONLY graphic/corner kills spread (junk strokes must
    # not eat sentence prefixes), and only into short / mostly-dead lines
    pad = JUNK_ZONE_PAD_PT * px_per_pt
    for _ in range(2):
        dead = [w for ln in lines for w in ln.words
                if w.reason in ("graphic", "corner", "contagion")]
        changed = False
        for ln in lines:
            alive_n = len(ln.alive_words())
            dead_frac = 1 - alive_n / max(len(ln.words), 1)
            if alive_n > 4 and dead_frac < 0.4:
                continue    # healthy long line: immune to contagion
            for w in ln.words:
                if w.dead or w.solid:
                    continue
                for dw in dead:
                    if boxes_overlap(w, dw, pad):
                        w.kill("contagion")
                        changed = True
                        break
        if not changed:
            break

    # pass 4 (wave-5): raster bullet detection — a compact ink dot in the
    # whitespace strip just left of a line is a bullet marker even when
    # tesseract missed or mangled the glyph (one item of a list otherwise
    # loses its bullet while all its siblings keep theirs)
    for ln in lines:
        ws = ln.alive_words()
        if ln.is_list or not ws:
            continue
        text_left = min(w.x for w in ws)
        wy = min(w.y for w in ws)
        wy2 = max(w.y2 for w in ws)
        mx = find_marker_dot(raster, text_left, wy, wy2, ln.x_size)
        if mx is not None:
            ln.is_list = True
            ln.marker_left = mx

    for ln in lines:
        if not ln.alive_words():
            ln.dead = True


def find_marker_dot(raster, text_left, wy, wy2, x_size):
    """x of a compact ink dot in the whitespace strip left of a line.

    The strip reaches ~1.35 glyph heights left of the text, so page-margin
    whitespace stays empty for continuation lines while a missed «▪»/«•»
    shows up as a small, solid, well-separated ink blob. Anything large,
    sparse or bleeding through the strip edges (braces, ornaments, colored
    cards) is not a marker.
    """
    h = max(wy2 - wy, 8)
    ref = max(x_size, 0.8 * h, 20)
    x1 = max(0, int(text_left - 1.35 * ref))
    x2 = max(0, int(text_left - 0.15 * ref))
    y1 = max(0, int(wy - 0.15 * h))
    y2 = min(raster.h, int(wy2 + 0.15 * h))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    buf, n, stride = raster.buf, raster.n, raster.stride
    cols = {}  # x -> [min_y, max_y, count]
    for y in range(y1, y2, 2):
        row = y * stride
        for x in range(x1, min(x2, raster.w), 2):
            i = row + x * n
            if buf[i] >= WHITE_LVL and buf[i + 1] >= WHITE_LVL \
                    and buf[i + 2] >= WHITE_LVL:
                continue
            col = cols.setdefault(x, [y, y, 0])
            col[0], col[1] = min(col[0], y), max(col[1], y)
            col[2] += 1
    if not cols:
        return None
    # the marker is the RIGHTMOST connected ink run: braces/ornaments left
    # of the list are separated from the dot by whitespace
    xs = sorted(cols)
    run = [xs[-1]]
    for x in reversed(xs[:-1]):
        if run[0] - x > 6:
            break
        run.insert(0, x)
    bx, bx2 = run[0], run[-1]
    by = min(cols[x][0] for x in run)
    by2 = max(cols[x][1] for x in run)
    count = sum(cols[x][2] for x in run)
    if count < 3:
        return None
    dot_w, dot_h = bx2 - bx + 2, by2 - by + 2
    if dot_w > 0.8 * h or dot_h > 0.8 * h or dot_w < 3 or dot_h < 3:
        return None
    # must be separated from the strip edges (edge contact = graphic bleed
    # or the first glyph of the text itself)
    if bx <= x1 + 2 or bx2 >= x2 - 3 or by <= y1 + 2 or by2 >= y2 - 3:
        return None
    if count * 4 / (dot_w * dot_h) < 0.15:  # sparse speckle, not a glyph
        return None
    center = (by + by2) / 2
    if not (wy + 0.1 * h <= center <= wy2 - 0.1 * h):
        return None
    return bx


def line_metrics(ln, px_per_pt):
    """(left_px, right_px, baseline_px, size_pt) from surviving words."""
    ws = ln.alive_words()
    left = min(w.x for w in ws)
    right = max(w.x2 for w in ws)
    baseline = ln.y2 + ln.baseline_off
    heights = sorted(w.y2 - w.y for w in ws)
    med_h = heights[len(heights) // 2]
    size_px = min(ln.x_size, 1.35 * med_h) if ln.x_size > 0 else 1.35 * med_h
    return left, right, baseline, size_px / px_per_pt


def assign_stretch(lines, px_per_pt):
    """Wire wrapped continuation lines to their parent line, infer bullet
    markers the OCR missed from siblings, then give every list group on the
    page a common right margin and font size per left-x cluster.

    Continuation lines stay where they are (one Tj each): BabelDOC's
    ocr_workaround regroup pass re-merges each group into ONE paragraph that
    wraps as a unit. The job here is consistent geometry only: group-uniform
    size, marker prefixes, and a shared right edge for sibling items.
    """
    alive = sorted((ln for ln in lines if not ln.dead), key=lambda ln: ln.y)

    # word-extent vertical bounds: tesseract's line boxes routinely bleed
    # into the neighbouring line, word boxes don't
    def wbounds(ln):
        ws = ln.alive_words()
        return min(w.y for w in ws), max(w.y2 for w in ws)

    # continuations: the next line of a wrapped paragraph sits closer to its
    # parent (<= one glyph height between ink extents) than a following
    # separate block does, starts inside the parent's horizontal span, is
    # not itself a list item, and matches the parent's size
    for ln in alive:
        if ln.is_list:
            continue
        top, _ = wbounds(ln)
        parent = best_bot = None
        for prev in alive:
            if prev is ln:
                continue
            _, pbot = wbounds(prev)
            if pbot > top:
                continue
            if best_bot is None or pbot > best_bot:
                if prev.x - 40 <= ln.x <= prev.x2:
                    parent, best_bot = prev, pbot
        if parent is None:
            continue
        lh = max(parent.x_size, ln.x_size, 1)
        if min(parent.x_size, ln.x_size) <= 0.6 * lh:
            continue
        if top - best_bot <= 1.0 * lh:
            ln.group_root = parent.group_root or parent

    def group_of(root):
        return [root] + [ln for ln in alive if ln.group_root is root]

    # cluster list-item group roots by their marker x (fallback: text left,
    # within 25 pt), then give each cluster a shared right margin and one
    # font size (groups included)
    tol = 25 * px_per_pt
    clusters = []
    for ln in alive:
        if ln.group_root is not None or not ln.is_list:
            continue
        left = ln.marker_left if ln.marker_left is not None \
            else min(w.x for w in ln.alive_words())
        placed = False
        for cl in clusters:
            if abs(cl["left"] - left) <= tol:
                cl["roots"].append(ln)
                placed = True
                break
        if not placed:
            clusters.append({"left": left, "roots": [ln]})
    for cl in clusters:
        members = [ln for r in cl["roots"] for ln in group_of(r)]
        right = max(max(w.x2 for w in ln.alive_words()) for ln in members)
        sizes = sorted(line_metrics(ln, px_per_pt)[3] for ln in members)
        med_size = sizes[len(sizes) // 2]
        for ln in members:
            ln.size_override = med_size
        if len(cl["roots"]) >= 2:
            for r in cl["roots"]:
                r.stretch_right = right


def strip_existing_text(doc, page):
    """Remove the ocrmypdf text layer: drop top-level q..Q blocks that invoke
    an /OCR-* form xobject, and any inline BT..ET blocks."""
    page.clean_contents()
    xref = page.get_contents()[0]
    s = doc.xref_stream(xref).decode("latin-1")
    toks = re.split(r"(\s+)", s)
    blocks, cur, depth = [], [], 0
    for t in toks:
        cur.append(t)
        if t == "q":
            depth += 1
        elif t == "Q":
            depth -= 1
            if depth == 0:
                blocks.append("".join(cur))
                cur = []
    if "".join(cur).strip():
        blocks.append("".join(cur))
    kept = [b for b in blocks if "/OCR-" not in b]
    new = re.sub(r"\bBT\b.*?\bET\b", "", "\n".join(kept), flags=re.S)
    doc.update_stream(xref, new.encode("latin-1"))
    return xref


def esc_pdf(s):
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_page_ops(lines, page_h_pt, px_per_pt, font, visible=False):
    ops = []
    for ln in lines:
        if ln.dead:
            continue
        left, right, baseline, size_pt = line_metrics(ln, px_per_pt)
        if ln.size_override:
            size_pt = ln.size_override
        if ln.marker_left is not None:
            left = min(left, ln.marker_left)  # mask must cover the raster dot
        target_right = ln.stretch_right if ln.stretch_right else right
        text = " ".join(w.text for w in ln.alive_words())
        # keep latin-1 only (junk already stripped; be safe for Tj encoding)
        text = text.encode("latin-1", "ignore").decode("latin-1")
        if not text.strip():
            continue
        if ln.is_list:
            # re-emit the stripped marker: the translated RTL paragraph then
            # carries its bullet (visually on the right)
            text = "• " + text
        natural = font.text_length(text, fontsize=size_pt)
        if natural <= 0:
            continue
        target_w_pt = (target_right - left) / px_per_pt
        sx = min(3.0, max(0.2, target_w_pt / natural))
        x_pt = left / px_per_pt
        y_pt = page_h_pt - baseline / px_per_pt
        mode = 0 if visible else 3
        color = "1 0 0 rg " if visible else ""
        # helv is WinAnsi-encoded: the bullet glyph lives at 0x95
        payload = text.replace("•", "\x95")
        ops.append(
            f"BT {color}/helv {size_pt:.2f} Tf {mode} Tr "
            f"{sx:.4f} 0 0 1 {x_pt:.2f} {y_pt:.2f} Tm ({esc_pdf(payload)}) Tj ET"
        )
    return "\n".join(ops)


def _take_option(args, flag, default=None):
    if flag not in args:
        return default
    i = args.index(flag)
    value = args[i + 1]
    del args[i:i + 2]
    return value


def parse_page_list(value):
    """`--keep-pages 0,3,4` -> {0, 3, 4}. Empty or absent -> empty set."""
    if not value:
        return set()
    return {int(part) for part in value.split(",") if part.strip()}


def main():
    args = sys.argv[1:]
    dbg_path = _take_option(args, "--debug")
    dpi = int(_take_option(args, "--dpi", DPI))
    # Which language(s) tesseract reads. The caller passes the job's own
    # languages; an unknown or uninstalled one degrades to English rather
    # than failing the stage.
    lang = _take_option(args, "--lang") or DEFAULT_TESSERACT_LANG
    # Pages that already carry a good text layer, 0-based. They keep it: this
    # pass strips and rewrites the text of every page it touches, so running
    # it over a page that was never scanned destroys perfectly good text.
    keep = parse_page_list(_take_option(args, "--keep-pages"))
    src, dst = args
    px_per_pt = dpi / 72.0

    doc = pymupdf.open(src)
    dbg = pymupdf.open(src) if dbg_path else None
    font = pymupdf.Font("helv")

    with tempfile.TemporaryDirectory() as td:
        for pno, page in enumerate(doc):
            if pno in keep:
                continue
            pix = page.get_pixmap(dpi=dpi)
            img = f"{td}/p{pno}.png"
            pix.save(img)
            subprocess.run(
                ["tesseract", img, f"{td}/p{pno}", "-l", lang, "hocr"],
                check=True, capture_output=True,
            )
            lines = parse_hocr(f"{td}/p{pno}.hocr")
            filter_words(lines, pix.width, pix.height, px_per_pt, Raster(pix))
            assign_stretch(lines, px_per_pt)

            for target_doc, vis in ((doc, False), (dbg, True)) if dbg else ((doc, False),):
                tp = target_doc[pno]
                xref = strip_existing_text(target_doc, tp)
                tp.insert_font(fontname="helv")
                ops = build_page_ops(lines, tp.rect.height, px_per_pt, font, visible=vis)
                if not ops:
                    continue
                old = target_doc.xref_stream(xref)
                new = old + b"\nq\n" + ops.encode("latin-1") + b"\nQ\n"
                target_doc.update_stream(xref, new)

    doc.save(dst, garbage=3, deflate=True)
    if dbg:
        dbg.save(dbg_path, garbage=3, deflate=True)
    print(f"wrote {dst}" + (f" and {dbg_path}" if dbg else ""))


if __name__ == "__main__":
    main()
