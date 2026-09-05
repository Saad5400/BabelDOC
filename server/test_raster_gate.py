"""Tests for the in-image gloss gate (server/raster_gate.py).

Every case below is a reading (or a whole image) taken from the production
run that exposed the defect, with its real geometry in points:

  * job 136 p1 — the Umm Al-Qura bilingual wordmark, OCR'd as `ll`,
    `ola_cola` and `ol UMM AL-QURA UNIVERSITY`, "translated" into
    `أولا _ كولا`, and plated over the logo on 36 of 38 pages;
  * job 154 p1 — a blurred decorative stock photo of code, read as
    `FalseFalse` -> `خطأخطأ`;
  * job 154 pp.33/37 — code screenshots whose masks and glosses shredded
    each other;
  * job 143 p6 — two screenshots of the SAME Windows dialog on one page:
    the small dense one came out fully unreadable, the larger sparse one
    rendered correctly. Both must keep doing what they now do.

Run from the repo root:

    pytest server/test_raster_gate.py
"""

import pytest

from server import raster_gate
from server.raster_gate import RasterText


def _item(text, x0, y0, x1, y1, conf=None):
    return RasterText(text=text, box=(x0, y0, x1, y1), conf=conf)


# The five images above, as (readings, region, page). Boxes are page points,
# y down, exactly as image_prep measured them.
LOGO_REGION = (869.8, 0.0, 957.9, 88.1)
SLIDE_PAGE = (0.0, 0.0, 960.0, 540.0)
LOGO = [
    _item("ol", 906.7, 63.1, 915.1, 73.0, conf=76.0),
    _item("UMM AL-QURA UNIVERSITY", 871.9, 75.1, 956.4, 79.9, conf=93.7),
    _item("ola_cola", 917.1, 64.6, 956.4, 72.7, conf=74.0),
    _item("ll", 906.3, 26.6, 922.3, 53.5, conf=81.0),
]

BLUR_REGION = (414.0, 0.0, 960.0, 540.0)
BLURRED_PHOTO = [
    _item("False", 721.4, 182.4, 750.2, 191.5, conf=96.0),
    _item("-use_y", 666.7, 191.8, 703.4, 201.4, conf=89.0),
    _item("False", 721.4, 193.9, 750.2, 202.6, conf=96.0),
    _item("not", 752.2, 475.9, 768.2, 497.0, conf=92.0),
]

CODE_REGION = (81.0, 221.2, 518.2, 293.2)
CODE_SHOT = [
    _item("text", 94.4, 243.8, 129.7, 254.4, conf=96.0),
    _item('input("Type anything...', 159.2, 242.6, 362.8, 257.5, conf=90.0),
    _item("print(5*text)", 94.0, 270.9, 208.7, 285.8, conf=90.0),
]

# 143 p6's dense dialog and its sparse twin, both on a 612 x 792 page.
DOC_PAGE = (0.0, 0.0, 612.0, 792.0)
DENSE_REGION = (211.4, 56.7, 400.5, 237.0)
DENSE_DIALOG = [
    _item("Client\\", 274.8, 88.1, 288.2, 96.5, conf=80.0),
    _item("C:\\Program", 221.3, 125.1, 235.9, 129.9, conf=80.0),
    _item("Engine C", 317.0, 124.1, 334.6, 132.3, conf=88.0),
    _item("Management", 279.8, 132.1, 305.3, 136.9, conf=96.0),
    _item("C:\\Program", 221.3, 138.8, 235.9, 143.6, conf=79.0),
    _item("Engine", 316.1, 135.9, 329.8, 144.5, conf=89.0),
    _item("Move Up", 372.0, 135.7, 390.5, 145.7, conf=87.0),
    _item("Engine C", 306.5, 145.7, 323.0, 150.5, conf=87.0),
    _item("C:\\Program Files", 221.3, 147.4, 252.7, 158.5, conf=89.5),
    _item("Move Down", 370.3, 148.9, 391.9, 159.2, conf=70.5),
    _item("text...", 379.7, 170.0, 391.0, 178.9, conf=87.0),
    _item("System32\\OpenSSH\\", 256.3, 197.8, 297.1, 207.2, conf=79.0),
]

# 154 p37 — a code screenshot beside its terminal output. Its code lines are
# a minority of the readings, so the code pattern alone does not condemn it;
# 80 % of its rows have no room for a gloss, which does. It came out with
# masks over the code and Arabic printed on top of Arabic.
DENSE_CODE_REGION = (45.0, 204.0, 929.2, 379.5)
DENSE_CODE_SHOT = [
    _item('ni = float(input("enter', 75.7, 232.8, 346.0, 252.5, conf=92),
    _item("first", 361.3, 232.8, 416.3, 248.4, conf=96),
    _item("number:", 431.6, 237.1, 509.2, 248.4, conf=93),
    _item("enter first number:", 618.8, 230.2, 828.6, 245.5, conf=95.3),
    _item("4.5", 844.0, 231.6, 876.4, 245.5, conf=96),
    _item('n2 = float(input("enter second', 75.7, 262.3, 427.6, 281.8, conf=92.8),
    _item("number:", 444.4, 262.3, 520.4, 277.7, conf=92),
    _item("enter second number:", 618.8, 252.5, 839.9, 267.8, conf=95.7),
    _item("the result=", 618.8, 274.8, 741.2, 290.2, conf=92),
    _item("11.5", 766.7, 276.2, 808.9, 290.2, conf=91),
    _item("sum", 77.2, 295.9, 109.3, 307.0, conf=84),
    _item("ni", 147.5, 293.0, 168.6, 307.0, conf=84),
    _item("n2", 206.5, 293.0, 226.2, 307.0, conf=87),
    _item("result=", 208.0, 321.1, 286.7, 336.5, conf=89),
    _item("sum)", 337.3, 321.1, 379.6, 337.7, conf=96),
    _item('orint("the', 77.2, 321.1, 190.9, 337.7, conf=90),
]

SPARSE_REGION = (183.0, 309.5, 428.5, 587.8)
SPARSE_DIALOG = [
    _item("Edit environment variable", 187.6, 318.2, 253.8, 323.2, conf=95.3),
    _item("New", 397.3, 343.6, 408.6, 347.9, conf=96.0),
    _item("Edit", 398.0, 364.0, 407.6, 368.6, conf=96.0),
    _item("Delete", 394.9, 405.1, 411.0, 409.9, conf=96.0),
    _item("Move Up", 391.1, 436.0, 414.8, 441.8, conf=96.0),
    _item("Move Down", 387.2, 456.4, 418.7, 461.0, conf=96.0),
    _item("Edit text...", 390.6, 487.1, 415.3, 491.7, conf=94.0),
]


def _fresh(items):
    """The module-level fixtures are mutated by the gate (each carries its
    own verdict), so every test judges its own copies."""
    return [RasterText(text=i.text, box=i.box, conf=i.conf) for i in items]


# ---------------------------------------------------------------------------
# one reading at a time


@pytest.mark.parametrize("text", ["ll", "ol", "Il", "wm", "x", "aaaa", "bcdfg"])
def test_gibberish_is_not_language(text):
    assert raster_gate.text_reject_reason(text) == "gibberish"


@pytest.mark.parametrize("text", [
    "UMM AL-QURA UNIVERSITY",       # a wordmark is still English
    "ola_cola",                     # a misread one, but it reads as words
    "Requirements engineering",
    "Move Down",
    "Type anything...",
    "88888",                        # a terminal's output: digits translate to
    "4.5",                          # themselves, and are never WRONG
])
def test_real_readings_survive(text):
    assert raster_gate.text_reject_reason(text) is None


def test_text_already_in_the_target_language_is_refused():
    # The Arabic half of a bilingual wordmark, round-tripped back into Arabic.
    assert raster_gate.text_reject_reason("جامعة أم القرى") == "target-language"


def test_a_token_written_in_two_scripts_is_refused():
    assert raster_gate.text_reject_reason("olأم") == "mixed-script"


def test_a_low_confidence_reading_is_refused():
    # 143 p6's '‘%SYSTEMROOT' came back at 65: over ocr_prep's per-word floor,
    # under anything that reads as a word somebody typed.
    assert raster_gate.text_reject_reason("%SYSTEMROOT", conf=65.0) == "conf"
    assert raster_gate.text_reject_reason("Management", conf=96.0) is None


def test_unknown_text_and_confidence_stand_the_rules_down():
    # An old sidecar records no source for some blocks. Unknown is not empty.
    assert raster_gate.text_reject_reason(None) is None
    assert raster_gate.text_reject_reason(None, conf=None) is None
    assert raster_gate.text_reject_reason("") == "empty"


# ---------------------------------------------------------------------------
# one image at a time


def test_a_corner_wordmark_is_left_as_pixels():
    plan = raster_gate.gloss_plan(_fresh(LOGO), LOGO_REGION, SLIDE_PAGE)

    assert plan.reason == "logo"
    assert plan.keep == []


def test_a_wordmark_in_the_middle_of_the_slide_is_still_refused():
    """Not by the corner rule — by the company its readings keep.

    Half of what OCR found on the logo (`ll`, `ol`) is not language, and a
    reading that shares an image with that much junk is a misreading of the
    same emblem.
    """
    middle = (440.0, 220.0, 528.1, 308.1)
    shift = (middle[0] - LOGO_REGION[0], middle[1] - LOGO_REGION[1])
    moved = [RasterText(text=i.text, conf=i.conf,
                        box=(i.box[0] + shift[0], i.box[1] + shift[1],
                             i.box[2] + shift[0], i.box[3] + shift[1]))
             for i in LOGO]

    plan = raster_gate.gloss_plan(moved, middle, SLIDE_PAGE)

    assert plan.reason == "junk"


def test_a_blurred_image_is_left_as_pixels():
    # 154 p1's decorative photo scores 37; every image in the corpus that
    # carries real text scores 232 or more.
    plan = raster_gate.gloss_plan(_fresh(BLURRED_PHOTO), BLUR_REGION,
                                  SLIDE_PAGE, sharpness=37.3)

    assert plan.reason == "blurred"


def test_sharpness_is_optional_not_assumed():
    # A lane that cannot measure the pixels must not have the rule guess.
    plan = raster_gate.gloss_plan(_fresh(SPARSE_DIALOG), SPARSE_REGION, DOC_PAGE)

    assert plan.reason is None


def test_a_code_screenshot_is_left_as_pixels():
    plan = raster_gate.gloss_plan(_fresh(CODE_SHOT), CODE_REGION, SLIDE_PAGE)

    assert plan.reason == "code"


@pytest.mark.parametrize("line", [
    'input("Type anything...',      # 154 p33
    "print(5*text)",
    'ni = float(input("enter',      # 154 p37
    "C:\\Program Files",            # 143 p6
    "%SYSTEMROOT",
])
def test_the_code_pattern_recognises_a_screenshot_line(line):
    assert raster_gate.GATE_CODE_LINE.search(line)


@pytest.mark.parametrize("line", [
    "Requirements engineering",
    "Edit environment variable",
    "Type anything...",
    "Move Down",
    "Software Design and Implementation",
])
def test_the_code_pattern_leaves_a_diagram_label_alone(line):
    assert not raster_gate.GATE_CODE_LINE.search(line)


def test_a_dense_screenshot_is_skipped_entirely():
    """143 p6's small dialog: 57 % of its rows have no room for a gloss.

    A terminal or a path list left untouched reads far better than one made
    illegible, so the whole image opts out rather than half of it.
    """
    plan = raster_gate.gloss_plan(_fresh(DENSE_DIALOG), DENSE_REGION, DOC_PAGE)

    assert plan.reason in ("dense", "code")
    assert plan.keep == []


def test_density_alone_condemns_a_screenshot_the_code_pattern_misses():
    """154 p37, where the rows are the whole argument.

    Only three of its readings look like code, well under the pattern's
    minority bar — but 80 % of its rows have no room to put anything, and
    the page came out with Arabic printed over Arabic.
    """
    items = _fresh(DENSE_CODE_SHOT)
    packed, rows = raster_gate.packed_row_fraction(items)

    assert not raster_gate.region_is_code([i.text for i in items])
    assert rows >= raster_gate.GATE_DENSITY_MIN_ROWS
    assert packed >= raster_gate.GATE_MAX_PACKED_ROW_FRACTION
    assert raster_gate.gloss_plan(
        _fresh(DENSE_CODE_SHOT), DENSE_CODE_REGION, SLIDE_PAGE).reason == "dense"


def test_the_same_dialog_at_a_readable_pitch_is_still_glossed():
    """The sparse twin of the dense dialog, on the same page, renders fine —
    and the gate must not take it down with its neighbour."""
    plan = raster_gate.gloss_plan(_fresh(SPARSE_DIALOG), SPARSE_REGION, DOC_PAGE)
    packed, rows = raster_gate.packed_row_fraction(_fresh(SPARSE_DIALOG))

    assert plan.reason is None
    assert packed == 0.0
    assert [item.text for item in plan.keep] == [i.text for i in SPARSE_DIALOG]


def test_a_two_line_caption_is_never_called_dense():
    """Two readings is one coin flip, not a measurement of density.

    154 p33's terminal output pane is two lines in a small box; it comes out
    legible and must stay glossed.
    """
    pane = (525.0, 243.8, 749.2, 293.2)
    items = [_item("Type anything...", 533.6, 250.5, 692.5, 267.0, conf=95.5),
             _item("88888", 533.6, 274.2, 583.3, 286.2, conf=95.0)]

    assert raster_gate.gloss_plan(items, pane, SLIDE_PAGE).reason is None


def test_an_image_that_is_wall_to_wall_text_is_refused():
    # The backstop behind the row arithmetic: whatever the pitch, a mask
    # this big has eaten the image rather than annotated it.
    region = (250.0, 350.0, 350.0, 390.0)      # mid-page, so not a logo
    items = [_item("Requirements engineering",
                   252.0, 352.0, 348.0, 388.0, conf=95.0)]

    assert raster_gate.gloss_plan(items, region, DOC_PAGE).reason == "crowded"


def test_an_image_with_no_readings_is_not_blamed_on_a_rule():
    assert raster_gate.gloss_plan([], SPARSE_REGION, DOC_PAGE).reason == "empty"


# ---------------------------------------------------------------------------
# collisions


def test_a_mask_may_not_cover_a_neighbouring_reading():
    """136 p1's two wordmark readings, at their real spacing.

    `ola_cola` and the line under it are 2.4 pt apart; the mask that would
    hide the first reaches 2.8 pt down, onto the second's glyphs. Both came
    out illegible, one printed over the other.
    """
    items = [_item("ola_cola", 917.1, 64.6, 956.4, 72.7, conf=74.0),
             _item("UMM AL-QURA UNIVERSITY", 871.9, 75.1, 956.4, 79.9,
                   conf=93.7)]

    kept, dropped = raster_gate.drop_collisions(items)

    assert len(kept) == 1
    # The longer reading holds the space: it is what carries the meaning.
    assert kept[0].text == "UMM AL-QURA UNIVERSITY"
    assert [(d.text, d.reason) for d in dropped] == [("ola_cola", "collision")]


def test_readings_that_clear_each_other_all_survive():
    items = [_item("Requirements", 100.0, 100.0, 200.0, 112.0, conf=95.0),
             _item("Design", 100.0, 140.0, 160.0, 152.0, conf=95.0)]

    kept, dropped = raster_gate.drop_collisions(items)

    assert len(kept) == 2
    assert dropped == []


def test_the_survivors_masks_never_intersect_another_survivors_glyphs():
    """The invariant, over a whole crowded image: whatever the gate keeps,
    no kept mask lands on another kept reading."""
    items = _fresh(SPARSE_DIALOG) + [
        _item("Browse...", 391.0, 379.1, 420.0, 389.9, conf=93.0),
        # a shard sitting right on top of «Delete»
        _item("Deleto", 395.5, 404.0, 412.0, 410.5, conf=71.0),
    ]

    kept, _dropped = raster_gate.drop_collisions(items)

    for index, item in enumerate(kept):
        mask = item.padded()
        for other in kept[index + 1:]:
            assert not (mask[0] < other.rect[2] and other.rect[0] < mask[2]
                        and mask[1] < other.rect[3] and other.rect[1] < mask[3]), (
                f"{item.text!r}'s mask covers {other.text!r}")


# ---------------------------------------------------------------------------
# rows


def test_side_by_side_fragments_are_one_row_not_two():
    """A screenshot line read as three fragments must be judged as one row,
    or every screenshot looks infinitely dense."""
    items = [_item("num", 91.1, 352.3, 118.9, 361.2),
             _item('int(input( "Type', 150.6, 348.5, 294.4, 364.3),
             _item("number...", 325.3, 349.2, 408.4, 361.2),
             _item("print(5*num)", 89.9, 379.2, 203.9, 395.0)]

    assert len(raster_gate.rows(items)) == 2
