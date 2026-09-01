#!/usr/bin/env python
"""OCR-prep for text living INSIDE embedded raster images on DIGITAL pages.

The scanned-page lane (ocr_prep.py) rebuilds a whole-page text layer. This
pass is its sibling for digital decks: the page's own text layer is
authoritative and untouched — but diagrams, screenshots and figure scans
embedded as raster images carry English text BabelDOC cannot see. For every
qualifying image placement we render just that clip, run tesseract hOCR on
it, filter the words with ocr_prep's junk / graphic / logo heuristics, and
inject one INVISIBLE (Tr 3) text run per surviving OCR line over the
recognized words. A regions.json sidecar lists each image placement bbox
(PDF user space, y-up, unrotated) that received at least one run, so the
fork can apply the recognition rule of the image-text contract:

    a char is image-OCR text iff (render mode == 3) AND (its box lies
    inside a declared image_bbox, +-2pt).

Differences from ocr_prep.py:
- The existing content stream is only APPENDED to, never stripped: out.pdf
  must render pixel-identical to in.pdf (invisible runs change nothing).
- OCR runs per image PLACEMENT region (page.get_image_info bboxes, clipped
  to the page), not per page; hOCR coordinates map back through the clip.
- Any OCR word overlapping a real digital text span dies (never duplicate
  the text layer).
- No bullet/stretch/staircase machinery: the fork groups image-OCR lines
  per region itself; here one plain Tj per line is the whole story.

Usage: image_prep.py in.pdf out.pdf regions.json [--dpi 300] [--debug dbg.pdf]
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf
from PIL import Image  # a hard transitive dep of babeldoc (pdfminer.six)

if __package__:
    from server.ocr_prep import CONF_KILL
    from server.ocr_prep import JUNK_ONLY
    from server.ocr_prep import JUNK_ZONE_PAD_PT
    from server.ocr_prep import SHORT_CONF_KILL
    from server.ocr_prep import SINGLE_CHAR_KEEP
    from server.ocr_prep import STROKE_CHARS
    from server.ocr_prep import Raster
    from server.ocr_prep import boxes_overlap
    from server.ocr_prep import esc_pdf
    from server.ocr_prep import parse_hocr
else:  # script mode: python server/image_prep.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ocr_prep import CONF_KILL
    from ocr_prep import JUNK_ONLY
    from ocr_prep import JUNK_ZONE_PAD_PT
    from ocr_prep import SHORT_CONF_KILL
    from ocr_prep import SINGLE_CHAR_KEEP
    from ocr_prep import STROKE_CHARS
    from ocr_prep import Raster
    from ocr_prep import boxes_overlap
    from ocr_prep import esc_pdf
    from ocr_prep import parse_hocr

DPI = 300
MAX_DPI = 600            # render at native image resolution up to this
MIN_W_PT = 50.0          # placements smaller than this are decorations
MIN_H_PT = 30.0
MERGE_OVERLAP = 0.5      # placements overlapping this much (of the smaller
                         # one) OCR as ONE region (stacked shadow + figure)
MIN_LEGIBLE_PT = 5.5     # OCR lines smaller than this are left as pixels
TEXT_OVERLAP_KILL = 0.25 # OCR-word area share covered by digital text => dup
TEXT_PAD_PT = 1.0        # slack around digital spans for the overlap test
FULL_PAGE_FRAC = 0.60    # region covering this much of the page gets the
                         # page-corner logo rules (full-bleed figure scans)
# raster-sampling thresholds, same roles as ocr_prep
BG_INK_KILL = 0.55
NEIGH_INK_KILL = 0.50
CORNER_X = 0.20
CORNER_Y = 0.22
CORNER_COLOR_RATIO = 0.40
# flat-ring rescue: unlike ocr_prep's on_solid_card (which refuses white
# rings, because on a SCANNED PAGE dense ink on paper is a crest/chart),
# inside an image region a flat ring of ANY color — white included — means
# diagram fill or figure background, and diagram labels are exactly what
# this lane exists to keep (Sommerville labels sit beside dense arrow ink
# on white; hexagon labels sit on solid or GRADIENT color fills, which a
# bucketed dominant-color test misjudges at bucket edges). Flatness is the
# ring's mean absolute deviation instead: measured on the test decks, real
# labels score MAD <= 7 while car-photo neighbourhoods score >= 29 — the
# threshold of 16 sits in the middle of that gap.
RESCUE_CONF = 80
RING_FLAT_MAD = 16.0

SX_MIN, SX_MAX = 0.2, 3.0
SEG_GAP_H = 1.0          # split a line at word gaps wider than this many
                         # median glyph heights (sentence spacing is ~0.3)
TILE_MIN_PX = 2000       # renders at least this tall AND wide also get a
                         # TILED sparse pass: tesseract drops labels inside
                         # closed shapes (the Fig 1.2 «Software» ellipse) on
                         # a full page, but reads them once a tile edge cuts
                         # the shape open
TILE_TARGET_PX = 1200    # approximate tile size (20% overlap between tiles)
# glyphs that reach below the baseline: a segment containing none of these
# has its ink bottom ON the baseline, so the hOCR baseline can be clamped to
# the word boxes (tesseract floats all-caps labels ~5pt high otherwise —
# their tight line box lacks the descender room its baseline offset assumes)
DESCENDER_CHARS = set("gjpqyQ_;,()[]{}@")


# --------------------------------------------------------------------------
# region discovery


def gather_regions(page):
    """Qualifying image placements as (clip_rect, raw_rect, native_dpi).

    clip_rect is the placement intersected with the page (that is what we
    render and OCR); raw_rect is the union of the raw placement bboxes (what
    regions.json declares, so the fork's own view of the image placement
    matches). Overlapping placements (drop shadow under a figure, an image
    drawn twice) merge into one region.
    """
    regions = []
    for info in page.get_image_info(xrefs=True):
        raw = pymupdf.Rect(info["bbox"])
        clip = raw & page.rect
        if clip.is_empty or clip.width < MIN_W_PT or clip.height < MIN_H_PT:
            continue
        if info["width"] < 8 or info["height"] < 8:
            continue  # decorative strip / spacer
        native = 72.0 * info["width"] / max(raw.width, 1e-6)
        regions.append([clip, raw, native])

    merged = True
    while merged:
        merged = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                a, b = regions[i], regions[j]
                inter = a[0] & b[0]
                if inter.is_empty:
                    continue
                smaller = min(a[0].get_area(), b[0].get_area())
                if inter.get_area() / max(smaller, 1e-6) >= MERGE_OVERLAP:
                    a[0] |= b[0]
                    a[1] |= b[1]
                    a[2] = max(a[2], b[2])
                    del regions[j]
                    merged = True
                    break
            if merged:
                break
    return [(clip, raw, dpi) for clip, raw, dpi in regions]


# --------------------------------------------------------------------------
# OCR (two passes) and filtering (ocr_prep passes, trimmed for image regions)


def _run_tesseract(img_path, out_stem, psm=None):
    cmd = ["tesseract", img_path, out_stem]
    if psm:
        cmd += ["--psm", psm]
    cmd += ["-l", "eng", "hocr"]
    subprocess.run(cmd, check=True, capture_output=True)
    return parse_hocr(out_stem + ".hocr")


def ocr_region(img_path, out_stem):
    """hOCR lines for one region render, from three tesseract passes.

    Measured on the test decks:
    - default layout analysis on the color render reads a diagram of
      scattered short labels as artwork and returns almost nothing, but is
      the best reader of real text blocks — it goes first and wins wherever
      it saw something;
    - the same layout analysis on a GRAYSCALE render reads some words the
      color render garbles («Requirements» inside a stroked ellipse comes
      back as conf-0 noise in color, conf 96 in gray);
    - a grayscale --psm 11 (sparse) pass finds dark-on-light labels layout
      analysis dropped (the red Polymorphism hexagon);
    - white-on-color labels vanish in grayscale (white over yellow is flat
      gray) but are high-contrast in the SATURATION channel (text has none,
      the fill is saturated) — a sat --psm 11 pass finds Object/Class/
      Inheritance.
    Later-pass words overlapping an earlier pass's confident words are
    dropped.
    """
    lines = _run_tesseract(img_path, out_stem)
    im = Image.open(img_path).convert("RGB")
    gray = im.convert("L")
    gray_path = out_stem + "_gray.png"
    gray.save(gray_path)
    sat_path = out_stem + "_sat.png"
    im.convert("HSV").split()[1].save(sat_path)

    passes = [("gray3", gray_path, None),
              ("gray11", gray_path, "11"),
              ("sat11", sat_path, "11")]

    for tag, path, psm in passes:
        extra = _run_tesseract(path, out_stem + "_" + tag, psm=psm)
        _merge_lines(lines, extra)
    if gray.width >= TILE_MIN_PX and gray.height >= TILE_MIN_PX:
        _merge_lines(lines, _ocr_tiled(gray, out_stem))
    return lines


def _merge_lines(lines, extra):
    """Append extra hOCR lines, dropping words an earlier pass already read.

    Only CONFIDENT existing words block a discovery: a garbage box an
    earlier pass hallucinated over a navy hexagon must not shadow the
    «Class» the saturation pass reads there.
    """
    seen = [w for ln in lines for w in ln.words if w.conf >= CONF_KILL]
    for ln in extra:
        ln.words = [w for w in ln.words
                    if not any(boxes_overlap(w, b) for b in seen)]
        if ln.words:
            ln.x = min(w.x for w in ln.words)
            ln.x2 = max(w.x2 for w in ln.words)
            ln.y = min(w.y for w in ln.words)
            ln.y2 = max(w.y2 for w in ln.words)
            lines.append(ln)


def _ocr_tiled(gray, out_stem):
    """Sparse pass over overlapping tiles of a large region render.

    Tile edges cut closed shapes open, so tesseract reads the labels it
    refuses on the whole page. Overlap means a word can appear in two tiles
    (whole in one, truncated in the other): keep the longest reading of any
    overlapping cluster.
    """
    nx = max(1, round(gray.width / TILE_TARGET_PX))
    ny = max(1, round(gray.height / TILE_TARGET_PX))
    tw, th = gray.width // nx, gray.height // ny
    tiled = []
    for iy in range(ny):
        for ix in range(nx):
            x0, y0 = ix * tw, iy * th
            x1 = min(gray.width, x0 + tw + tw // 5)
            y1 = min(gray.height, y0 + th + th // 5)
            stem = f"{out_stem}_tile{ix}_{iy}"
            gray.crop((x0, y0, x1, y1)).save(stem + ".png")
            for ln in _run_tesseract(stem + ".png", stem, psm="11"):
                for w in ln.words:
                    w.x, w.x2 = w.x + x0, w.x2 + x0
                    w.y, w.y2 = w.y + y0, w.y2 + y0
                ln.x, ln.x2 = ln.x + x0, ln.x2 + x0
                ln.y, ln.y2 = ln.y + y0, ln.y2 + y0
                tiled.append(ln)
    # intra-pass dedupe, longest text first («engineering» beats the
    # tile-edge fragment «engin» covering the same pixels)
    kept = []
    for ln in sorted(tiled,
                     key=lambda ln: -max(len(w.text) for w in ln.words)):
        ln.words = [w for w in ln.words
                    if w.conf >= CONF_KILL
                    and not any(boxes_overlap(w, k) for k in kept)]
        kept.extend(ln.words)
    return [ln for ln in tiled if ln.words]


def ring_mad(raster, x0, y0, x1, y1, pad, max_samples=1200):
    """Mean absolute deviation of the ring just outside the box."""
    ox0, oy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    ox1, oy1 = min(raster.w, int(x1 + pad)), min(raster.h, int(y1 + pad))
    if ox1 <= ox0 or oy1 <= oy0:
        return 999.0
    area = (ox1 - ox0) * (oy1 - oy0)
    step = max(1, int((area / max_samples) ** 0.5))
    buf, n, stride = raster.buf, raster.n, raster.stride
    px = []
    for y in range(oy0, oy1, step):
        row = y * stride
        inside_y = y0 <= y < y1
        for x in range(ox0, ox1, step):
            if inside_y and x0 <= x < x1:
                continue  # ring only
            i = row + x * n
            px.append((buf[i], buf[i + 1], buf[i + 2]))
    if not px:
        return 999.0
    means = [sum(p[c] for p in px) / len(px) for c in range(3)]
    return sum(abs(p[c] - means[c]) for p in px for c in range(3)) / (3 * len(px))


def flat_ring_rescue(raster, w):
    """A confident word whose surrounding ring is one flat color (any color,
    gradients included) is a diagram/figure label, not photo texture."""
    if w.conf < RESCUE_CONF or len(w.text) < 3:
        return False
    if not any(c.isalnum() for c in w.text):
        return False
    h = max(w.y2 - w.y, 1)
    pad = max(8, min(40, 0.25 * h))
    return ring_mad(raster, w.x, w.y, w.x2, w.y2, pad) <= RING_FLAT_MAD


def kill_text_overlaps(lines, text_rects_px, pad_px):
    """Words overlapping the page's real digital text die: the digital layer
    is authoritative and must never be duplicated by OCR runs."""
    for ln in lines:
        for w in ln.words:
            if w.dead:
                continue
            area = max((w.x2 - w.x) * (w.y2 - w.y), 1.0)
            for tx0, ty0, tx1, ty1 in text_rects_px:
                ix = min(w.x2, tx1 + pad_px) - max(w.x, tx0 - pad_px)
                iy = min(w.y2, ty1 + pad_px) - max(w.y, ty0 - pad_px)
                if ix > 0 and iy > 0 and ix * iy / area > TEXT_OVERLAP_KILL:
                    w.kill("dup")
                    break


def filter_region_words(lines, raster, page_bands_px):
    """ocr_prep's junk/graphic/logo passes, applied to one region render.

    page_bands_px: None, or (x_lo, x_hi, y_lo, y_hi) corner-band edges of
    the PAGE mapped into region pixels — only full-bleed regions get the
    corner logo rules (a small diagram's corners are not page corners).
    """
    # pass 1: confidence + junk tokens (ocr_prep rules, markers dropped —
    # bullets inside figures are artwork, not list furniture)
    for ln in lines:
        for w in ln.words:
            if w.dead:
                continue  # dup-killed by the digital-text overlap pass
            wpx, hpx = w.x2 - w.x, w.y2 - w.y
            if w.conf < CONF_KILL:
                w.kill("conf")
            elif JUNK_ONLY.match(w.text):
                w.kill("junk")
            elif len(w.text) == 1 and not w.text.isalnum() \
                    and w.text not in SINGLE_CHAR_KEEP:
                w.kill("junk")
            elif len(w.text) == 1 and w.text in STROKE_CHARS \
                    and hpx > 3 * max(wpx, 1):
                w.kill("stroke")
            elif len(w.text) <= 2 and w.conf < SHORT_CONF_KILL \
                    and not (w.text.isdigit() and w.conf >= 65):
                w.kill("shortconf")

    # pass 2: raster-sampled graphic/photo protection with solid-card rescue
    for ln in lines:
        cratios = {}
        for w in ln.words:
            if w.dead:
                continue
            h = max(w.y2 - w.y, 1)
            ink, cratio = raster.stats(w.x, w.y, w.x2, w.y2)
            cratios[id(w)] = cratio
            rescued = None
            if ink > BG_INK_KILL:            # photo / dense colored artwork
                rescued = flat_ring_rescue(raster, w)
                if not rescued:
                    w.kill("graphic")
                    continue
            nink, _ = raster.stats(w.x - h, w.y - h, w.x2 + h, w.y2 + h)
            if nink > NEIGH_INK_KILL:        # dense ink right next door
                if rescued is None:
                    rescued = flat_ring_rescue(raster, w)
                if not rescued:
                    w.kill("graphic")
                    continue
            w.solid = bool(rescued)

        # corner logo rules only when the region IS effectively the page
        if page_bands_px is not None:
            x_lo, x_hi, y_lo, y_hi = page_bands_px
            alive = ln.alive_words()
            in_bx = ln.x > x_hi or ln.x2 < x_lo
            in_by = ln.y2 < y_lo or ln.y > y_hi
            if in_bx and in_by and len(ln.words) <= 6:
                for w in alive:
                    if cratios.get(id(w), 0.0) > CORNER_COLOR_RATIO:
                        w.kill("corner")

        # isolated colored fragment inside a line: graphic shards OCR'd as
        # letters far away from the real words. Confident words are exempt:
        # a diagram's parallel arrow labels («generates» … «helps-with»)
        # legitimately share one hOCR line with a wide gap between them.
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
                        and w.conf < 85 \
                        and not w.solid:
                    w.kill("graphic")
        # a lone surviving single character is a stray stroke or arc shard —
        # unconditionally here (labels that matter are 2+ chars; ocr_prep's
        # conf>=90 escape hatch let «C» shards from circled figures through)
        alive = ln.alive_words()
        if len(alive) == 1 and len(alive[0].text) == 1:
            alive[0].kill("stroke")

    # a line half-dead from graphic kills IS the graphic
    for ln in lines:
        gdead = sum(1 for w in ln.words if w.reason in ("graphic", "corner"))
        if gdead and gdead >= 0.5 * len(ln.words):
            for w in ln.words:
                if not w.dead and not w.solid:
                    w.kill("graphic")

    # a rescued word inside a mostly-junk line is a misread emblem shard —
    # but one low-conf neighbour must not sink a good label («or Property»),
    # so it takes at least two junk kills to condemn the line
    for ln in lines:
        ndead = sum(1 for w in ln.words
                    if w.reason in ("conf", "junk", "stroke", "shortconf"))
        if ndead >= 2 and ndead >= 0.5 * len(ln.words):
            for w in ln.words:
                if not w.dead and w.solid:
                    w.kill("graphic")

    # contagion: graphic/corner kills spread into short / mostly-dead lines
    pad = JUNK_ZONE_PAD_PT * raster_px_per_pt(raster)
    for _ in range(2):
        dead = [w for ln in lines for w in ln.words
                if w.reason in ("graphic", "corner", "contagion")]
        changed = False
        for ln in lines:
            alive_n = len(ln.alive_words())
            dead_frac = 1 - alive_n / max(len(ln.words), 1)
            if alive_n > 4 and dead_frac < 0.4:
                continue
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

    for ln in lines:
        if not ln.alive_words():
            ln.dead = True


def raster_px_per_pt(raster):
    """The region raster does not know its dpi; the caller stashes it."""
    return getattr(raster, "px_per_pt", DPI / 72.0)


# --------------------------------------------------------------------------
# injection


def split_segments(ws):
    """Split a line's surviving words at gaps wider than one glyph height:
    two labels of parallel arrows («generates» … «helps-with») share one
    hOCR line, and ONE stretched run would strand the second label's glyphs
    mid-air between them (and make the fork treat both as one paragraph —
    the contract wants separate labels to stay separate)."""
    ws = sorted(ws, key=lambda w: w.x)
    heights = sorted(w.y2 - w.y for w in ws)
    med_h = max(heights[len(heights) // 2], 1)
    segs, cur = [], [ws[0]]
    for w in ws[1:]:
        if w.x - cur[-1].x2 > SEG_GAP_H * med_h:
            segs.append(cur)
            cur = [w]
        else:
            cur.append(w)
    segs.append(cur)
    return segs


def seg_metrics(ln, ws, px_per_pt):
    """(left_px, right_px, baseline_px, size_pt) for one word segment."""
    left = min(w.x for w in ws)
    right = max(w.x2 for w in ws)
    baseline = ln.y2 + ln.baseline_off
    heights = sorted(w.y2 - w.y for w in ws)
    med_h = heights[len(heights) // 2]
    size_px = min(ln.x_size, 1.35 * med_h) if ln.x_size > 0 else 1.35 * med_h
    return left, right, baseline, size_px / px_per_pt


def build_region_ops(lines, clip, px_per_pt, inv_ptm, font, visible=False):
    """Content-stream ops for one region: one Tj per surviving hOCR line
    segment, x-scaled onto the recognized extent, positioned in PDF user
    space."""
    ops = []
    for ln in lines:
        if ln.dead:
            continue
        for ws in split_segments(ln.alive_words()):
            left, right, baseline, size_pt = seg_metrics(ln, ws, px_per_pt)
            text = " ".join(w.text for w in ws)
            text = text.encode("latin-1", "ignore").decode("latin-1")
            if not text.strip():
                continue
            if not set(text) & DESCENDER_CHARS:
                # descender-less text: ink bottom IS the baseline; clamp
                # DOWN only (a correct hOCR baseline equals the ink bottom
                # already, a floated all-caps one sits ~5pt above it)
                baseline = max(baseline, max(w.y2 for w in ws))
            if size_pt < MIN_LEGIBLE_PT:
                # Below the legibility floor, translating this label makes
                # the page WORSE. The original is a crisp raster label the
                # reader can zoom into; what replaces it is vector Arabic at
                # the same tiny size, which is a smear — and, because these
                # micro-labels cluster inside diagrams, several of them
                # overprint each other and the artwork underneath.
                #
                # Measured over the sweep corpus's 1,058 real image-OCR
                # blocks: 8.4 % fall below 5.5 pt, and they are concentrated
                # in exactly the two decks whose figures came out illegible
                # (run14 44/100 blocks, run30 31/65) while the OCR lane's
                # real earners are untouched (run59 1/328, run67 0/129,
                # run38 0/71, run39 0/59, run8 0/49). The floor also
                # collects the lane's pure noise: every 0.00 pt "line" in
                # the corpus is a mis-recognised diagram stroke ('=',
                # '0+0=2').
                #
                # Skipping the run leaves the source pixels showing, which
                # is honest: we do not claim to have translated what we
                # cannot draw legibly.
                continue
            natural = font.text_length(text, fontsize=size_pt)
            if natural <= 0:
                continue
            target_w_pt = (right - left) / px_per_pt
            sx = min(SX_MAX, max(SX_MIN, target_w_pt / natural))
            page_pt = pymupdf.Point(clip.x0 + left / px_per_pt,
                                    clip.y0 + baseline / px_per_pt)
            pdf_pt = page_pt * inv_ptm
            mode = 0 if visible else 3
            color = "1 0 0 rg " if visible else ""
            ops.append(
                f"BT {color}/helv {size_pt:.2f} Tf {mode} Tr "
                f"{sx:.4f} 0 0 1 {pdf_pt.x:.2f} {pdf_pt.y:.2f} Tm "
                f"({esc_pdf(text)}) Tj ET"
            )
    return ops


def append_ops(doc, page, ops):
    """Append invisible runs AFTER the existing content — nothing existing
    is touched, so the page keeps rendering exactly as before."""
    page.clean_contents()  # balances q/Q so appended ops see base user space
    page.insert_font(fontname="helv")
    xref = page.get_contents()[0]
    old = doc.xref_stream(xref)
    payload = "\n".join(ops)
    doc.update_stream(xref, old + b"\nq\n" + payload.encode("latin-1") + b"\nQ\n")


def rect_to_pdf_space(rect, inv_ptm):
    """Page-space rect -> [x0, y0, x1, y1] in PDF user space, y-UP."""
    p0 = pymupdf.Point(rect.x0, rect.y0) * inv_ptm
    p1 = pymupdf.Point(rect.x1, rect.y1) * inv_ptm
    xs, ys = sorted((p0.x, p1.x)), sorted((p0.y, p1.y))
    return [round(xs[0], 2), round(ys[0], 2), round(xs[1], 2), round(ys[1], 2)]


# --------------------------------------------------------------------------
# driver


# One line of a code screenshot, as OCR sees it: statement punctuation,
# comment markers, declaration keywords, empty call parens, or a UML
# attribute's `name: type` suffix.
CODE_LINE = re.compile(
    r"[;{}]"
    r"|\(\s*\)"
    r"|//|/\*|\*/"
    r"|^\s*(public|private|protected|static|void|int|double|float|class"
    r"|return|new|import|package)\b"
    r"|\b\w+\s*:\s*(int|double|float|void|string|char|bool|boolean)\b",
    re.IGNORECASE,
)
CODE_MIN_LINES = 2       # fewer code-ish lines than this never kills a region
CODE_LINE_FRAC = 0.4     # ...nor a region where they are a minority


def region_is_code(lines):
    """Is this region a code screenshot / UML card rather than a diagram?

    Judged on the lines that SURVIVED filtering, so a photo caption next to
    junk does not tip the scale. Killing the whole region (not just the code
    lines) is deliberate: translating a screenshot's prose comments while
    masking half its statements produces exactly the shredded hybrid this
    guard exists to prevent.
    """
    texts = [" ".join(w.text for w in ln.alive_words())
             for ln in lines if not ln.dead]
    texts = [t for t in texts if t.strip()]
    if len(texts) < CODE_MIN_LINES:
        return False
    hits = sum(1 for t in texts if CODE_LINE.search(t))
    return hits >= CODE_MIN_LINES and hits / len(texts) >= CODE_LINE_FRAC


def prep_document(src, dst, regions_path, dpi=DPI, dbg_path=None):
    doc = pymupdf.open(src)
    dbg = pymupdf.open(src) if dbg_path else None
    font = pymupdf.Font("helv")
    regions_out = {"version": 1, "pages": {}}

    with tempfile.TemporaryDirectory() as td:
        for pno, page in enumerate(doc):
            regions = gather_regions(page)
            if not regions:
                continue
            inv_ptm = ~page.transformation_matrix
            text_rects = [pymupdf.Rect(w[:4]) for w in page.get_text("words")]
            page_ops, dbg_ops, page_regions = [], [], []

            for rno, (clip, raw, native_dpi) in enumerate(regions):
                use_dpi = int(min(MAX_DPI, max(dpi, native_dpi)))
                px_per_pt = use_dpi / 72.0
                pix = page.get_pixmap(dpi=use_dpi, clip=clip)
                img = f"{td}/p{pno}r{rno}.png"
                pix.save(img)
                lines = ocr_region(img, f"{td}/p{pno}r{rno}")
                if not lines:
                    continue

                raster = Raster(pix)
                raster.px_per_pt = px_per_pt

                # digital text spans, mapped into region pixels
                text_px = [
                    ((r.x0 - clip.x0) * px_per_pt, (r.y0 - clip.y0) * px_per_pt,
                     (r.x1 - clip.x0) * px_per_pt, (r.y1 - clip.y0) * px_per_pt)
                    for r in text_rects if not (r & clip).is_empty
                ]
                kill_text_overlaps(lines, text_px, TEXT_PAD_PT * px_per_pt)

                bands = None
                if clip.get_area() >= FULL_PAGE_FRAC * page.rect.get_area():
                    bands = (
                        (page.rect.x0 + CORNER_X * page.rect.width - clip.x0) * px_per_pt,
                        (page.rect.x1 - CORNER_X * page.rect.width - clip.x0) * px_per_pt,
                        (page.rect.y0 + CORNER_Y * page.rect.height - clip.y0) * px_per_pt,
                        (page.rect.y1 - CORNER_Y * page.rect.height - clip.y0) * px_per_pt,
                    )
                filter_region_words(lines, raster, bands)

                if region_is_code(lines):
                    # A code screenshot (or a UML card, which is identifier
                    # soup). Policy says code stays verbatim, and a masked,
                    # half-translated screenshot is strictly worse than the
                    # untouched original — so the whole region opts out.
                    continue

                ops = build_region_ops(lines, clip, px_per_pt, inv_ptm, font)
                if not ops:
                    continue
                page_ops.extend(ops)
                if dbg:
                    dbg_ops.extend(build_region_ops(
                        lines, clip, px_per_pt, inv_ptm, font, visible=True))
                page_regions.append({"image_bbox": rect_to_pdf_space(raw, inv_ptm)})

            if page_ops:
                append_ops(doc, page, page_ops)
                regions_out["pages"][str(pno)] = page_regions
            if dbg and dbg_ops:
                append_ops(dbg, dbg[pno], dbg_ops)

    doc.save(dst, garbage=3, deflate=True)
    if dbg:
        dbg.save(dbg_path, garbage=3, deflate=True)
    Path(regions_path).write_text(json.dumps(regions_out, indent=1))
    return regions_out


def main():
    args = sys.argv[1:]
    dbg_path = None
    if "--debug" in args:
        i = args.index("--debug")
        dbg_path = args[i + 1]
        del args[i:i + 2]
    dpi = DPI
    if "--dpi" in args:
        i = args.index("--dpi")
        dpi = int(args[i + 1])
        del args[i:i + 2]
    src, dst, regions_path = args
    regions = prep_document(src, dst, regions_path, dpi=dpi, dbg_path=dbg_path)
    n = sum(len(v) for v in regions["pages"].values())
    print(f"wrote {dst} and {regions_path} ({n} regions)"
          + (f" and {dbg_path}" if dbg_path else ""))


if __name__ == "__main__":
    main()
