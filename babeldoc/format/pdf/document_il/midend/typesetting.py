from __future__ import annotations

import copy
import logging
import re
import statistics
import unicodedata
from functools import cache

import pymupdf
import regex
from rtree import index

from babeldoc.const import WATERMARK_VERSION
from babeldoc.format.pdf.document_il import Box
from babeldoc.format.pdf.document_il import PdfCharacter
from babeldoc.format.pdf.document_il import PdfCurve
from babeldoc.format.pdf.document_il import PdfForm
from babeldoc.format.pdf.document_il import PdfFormula
from babeldoc.format.pdf.document_il import PdfParagraphComposition
from babeldoc.format.pdf.document_il import PdfStyle
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.formular_helper import update_formula_data
from babeldoc.format.pdf.document_il.utils.layout_helper import BULLET_POINT_PATTERN
from babeldoc.format.pdf.document_il.utils.layout_helper import box_to_tuple
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import WatermarkOutputMode

logger = logging.getLogger(__name__)

# --- Legibility floor -------------------------------------------------
# Shrink-to-fit used to bottom out at scale 0.1, with no floor on the
# RESULTING point size, so a 12.7 pt source line could legally be delivered
# at 1.27 pt. Measured over the 14 real production source documents in the
# sweep corpus (13,177 text spans): the smallest size any author actually
# used is 4.56 pt, only 0.4 % of spans are under 5 pt and 1.0 % under
# 5.5 pt. Anything below that is not "small print", it is unreadable — and
# it is never what the source looked like.
#
# So the search stops at whichever comes first:
#   * MIN_LEGIBLE_FONT_SIZE_PT absolute, or
#   * MIN_LEGIBLE_SCALE of the paragraph's own source size,
# and a paragraph that still does not fit is laid out AT that floor and
# allowed to overflow its box. Overflow is honest — a reader can see it and
# read it; a 1.3 pt paragraph is silently lost.
#
# A source paragraph that is already at or below the floor is simply never
# shrunk (its min_scale clamps to 1.0) rather than being scaled UP.
MIN_LEGIBLE_FONT_SIZE_PT = 5.5
MIN_LEGIBLE_SCALE = 0.45

LINE_BREAK_REGEX = regex.compile(
    r"^["
    r"a-z"
    r"A-Z"
    r"0-9"
    r"\u00C0-\u00FF"  # Latin-1 Supplement
    r"\u0100-\u017F"  # Latin Extended A
    r"\u0180-\u024F"  # Latin Extended B
    r"\u1E00-\u1EFF"  # Latin Extended Additional
    r"\u2C60-\u2C7F"  # Latin Extended C
    r"\uA720-\uA7FF"  # Latin Extended D
    r"\uAB30-\uAB6F"  # Latin Extended E
    r"\u0250-\u02A0"  # IPA Extensions
    r"\u0400-\u04FF"  # Cyrillic
    r"\u0300-\u036F"  # Combining Diacritical Marks
    r"\u0500-\u052F"  # Cyrillic Supplement
    r"\u0370-\u03FF"  # Greek and Coptic
    r"\u2DE0-\u2DFF"  # Cyrillic Extended-A
    r"\uA650-\uA69F"  # Cyrillic Extended-B
    r"\u1200-\u137F"  # Ethiopic
    r"\u1380-\u139F"  # Ethiopic Supplement
    r"\u2D80-\u2DDF"  # Ethiopic Extended
    r"\uAB00-\uAB2F"  # Ethiopic Extended-A
    r"\U0001E7E0-\U0001E7FF"  # Ethiopic Extended-B
    r"\u0E80-\u0EFF"  # Lao
    r"\u0D00-\u0D7F"  # Malayalam
    r"\u0A80-\u0AFF"  # Gujarati
    r"\u0E00-\u0E7F"  # Thai
    r"\u1000-\u109F"  # Myanmar
    r"\uAA60-\uAA7F"  # Myanmar Extended-A
    r"\uA9E0-\uA9FF"  # Myanmar Extended-B
    r"\U000116D0-\U000116FF"  # Myanmar Extended-C
    r"\u0B80-\u0BFF"  # Tamil
    r"\u0C00-\u0C7F"  # Telugu
    r"\u0B00-\u0B7F"  # Oriya
    r"\u0530-\u058F"  # Armenian
    r"\u10A0-\u10FF"  # Georgian
    r"\u1C90-\u1CBF"  # Georgian Extended
    r"\u2D00-\u2D2F"  # Georgian Supplement
    r"\u1780-\u17FF"  # Khmer
    r"\u19E0-\u19FF"  # Khmer Symbols
    r"\U00010B00-\U00010B3F"  # Avestan
    r"\u1D00-\u1D7F"  # Phonetic Extensions
    r"\u1400-\u167F"  # Unified Canadian Aboriginal Syllabics
    r"\u0B00-\u0B7F"  # Oriya
    r"\u0780-\u07BF"  # Thaana
    r"\U0001E900-\U0001E95F"  # Adlam
    r"\u1C80-\u1C8F"  # Cyrillic Extended-C
    r"\U0001E030-\U0001E08F"  # Cyrillic Extended-D
    r"\uA000-\uA48F"  # Yi Syllables
    r"\uA490-\uA4CF"  # Yi Radicals
    r"\u0600-\u06FF"  # Arabic
    r"\u0750-\u077F"  # Arabic Supplement
    r"\u0870-\u089F"  # Arabic Extended-B
    r"\u08A0-\u08FF"  # Arabic Extended-A
    r"\uFB50-\uFDFF"  # Arabic Presentation Forms-A
    r"\uFE70-\uFEFF"  # Arabic Presentation Forms-B
    r"\u0590-\u05FF"  # Hebrew
    r"'"
    r"-"  # Hyphen
    r"·"  # Middle Dot (U+00B7) For Català
    r"ʻ"  # Spacing Modifier Letters U+02BB
    r"]+$"
)

# --- RTL (Arabic / Hebrew) support ------------------------------------------
RTL_CHAR_REGEX = regex.compile(
    r"[\u0590-\u05FF"  # Hebrew
    r"\u0600-\u06FF"  # Arabic
    r"\u0750-\u077F"  # Arabic Supplement
    r"\u0870-\u089F"  # Arabic Extended-B
    r"\u08A0-\u08FF"  # Arabic Extended-A
    r"\uFB1D-\uFB4F"  # Hebrew Presentation Forms
    r"\uFB50-\uFDFF"  # Arabic Presentation Forms-A
    r"\uFE70-\uFEFF"  # Arabic Presentation Forms-B
    r"]"
)

# Invisible bidi control / zero-width characters. Our per-character renderer
# gives every code point a real advance width (the font mapper falls back to a
# visible-width glyph), so control characters that should be zero-width instead
# create spurious gaps ("justified"-looking lines). They carry no information
# for our explicit bidi algorithm, so strip them from translated text.
BIDI_CONTROL_REGEX = regex.compile(
    "[\u200b\u200c\u200d\u200e\u200f"  # ZWSP ZWNJ ZWJ LRM RLM
    "\u061c"  # Arabic Letter Mark
    "\u202a-\u202e"  # LRE RLE PDF LRO RLO
    "\u2066-\u2069"  # LRI RLI FSI PDI
    "\u2060\ufeff]"  # word joiner, BOM/ZWNBSP
)

# Collapse runs of plain spaces left behind by control-char stripping
# (translators sometimes emit "word ‏ word", which would otherwise
# render as a double-width gap).
MULTI_SPACE_REGEX = regex.compile(r"[  ]{2,}")

# Neutral operator characters that bind to a following digit run
# (e.g. "> 10", "= 5", "±3"): keeping them with the digits preserves the
# left-to-right reading of the math snippet inside an RTL sentence.
NEUTRAL_OPERATOR_CHARS = set("=<>+±×÷≈≠≤≥~∼*/-−")

# Bracket pairs that attach to a Latin identifier: an opening bracket
# immediately following a strong-LTR character (identifier or digit run)
# belongs to that run — "myCircle.getArea()", "Clone()", "list[0]" must
# travel as ONE left-to-right run, brackets staying on the identifier's
# logical side instead of drifting to the RTL edge of the line.
LTR_ATTACHED_BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}

# Characters replaced by their bidi-mirrored counterpart when they are part of
# a right-to-left visual run.
BIDI_MIRROR_MAP = {
    "(": ")",
    ")": "(",
    "[": "]",
    "]": "[",
    "{": "}",
    "}": "{",
    "<": ">",
    ">": "<",
    "«": "»",
    "»": "«",
    "‹": "›",
    "›": "‹",
}

RTL_LANG_PREFIXES = ("AR", "FA", "HE", "IW", "UR", "PS", "SD", "UG", "YI", "CKB", "DV")

# A glyph's visual bbox larger than this many times its advance box is
# considered corrupt (typical offender: color-emoji fonts whose glyph bbox
# covers the whole font matrix). Such boxes inflate formula and paragraph
# boxes and end up teleporting content (emoji rendered on top of headings,
# paragraphs anchored inside banners).
VISUAL_BBOX_INFLATION_RATIO = 3.0


def char_bidi_class(ch: str) -> str:
    """Simplified bidi class of one character: 'R', 'L' or 'N' (neutral).

    Glyphs without a usable unicode mapping ("(cid:NN)" placeholders) are
    treated as strong LTR: they come from preserved source content (math
    italics, symbol fonts) whose internal left-to-right order must never be
    disturbed.
    """
    if not ch:
        return "N"
    if "(cid" in ch:
        return "L"
    ch = ch[0]
    if RTL_CHAR_REGEX.match(ch):
        return "R"
    if unicodedata.bidirectional(ch) in ("L", "EN", "AN"):
        return "L"
    return "N"


def is_rtl_lang(lang_code: str) -> bool:
    code = lang_code.upper().replace("_", "-").split("-")[0]
    return code in RTL_LANG_PREFIXES


@cache
def _get_arabic_reshaper():
    try:
        from arabic_reshaper import ArabicReshaper

        return ArabicReshaper(configuration={"delete_harakat": False})
    except Exception:
        logger.warning(
            "arabic_reshaper is not available; "
            "Arabic text will be rendered with isolated letter forms.",
        )
        return None


def reshape_rtl_text(text: str) -> str:
    """Substitute Arabic letters with their contextual presentation forms.

    The string stays in logical order; only the joining forms (and lam-alef
    ligatures) are substituted so that every character can still be rendered
    as an independent glyph by the per-character renderer.
    """
    if not text or not RTL_CHAR_REGEX.search(text):
        return text
    reshaper = _get_arabic_reshaper()
    if reshaper is None:
        return text
    try:
        return reshaper.reshape(text)
    except Exception:
        logger.exception("Failed to reshape RTL text")
        return text


class TypesettingUnit:
    def __str__(self):
        return self.try_get_unicode() or ""

    def __init__(
        self,
        char: PdfCharacter | None = None,
        formular: PdfFormula | None = None,
        unicode: str | None = None,
        font: pymupdf.Font | None = None,
        original_font: il_version_1.PdfFont | None = None,
        font_size: float | None = None,
        style: PdfStyle | None = None,
        xobj_id: int | None = None,
        debug_info: bool = False,
    ):
        assert (char is not None) + (formular is not None) + (
            unicode is not None
        ) == 1, "Only one of chars and formular can be not None"
        self.char = char
        self.formular = formular
        self.unicode = unicode
        self.x = None
        self.y = None
        self.scale = None
        self.debug_info = debug_info

        # Cache variables
        self.box_cache: Box | None = None
        self.can_break_line_cache: bool | None = None
        self.is_cjk_char_cache: bool | None = None
        self.mixed_character_blacklist_cache: bool | None = None
        self.is_space_cache: bool | None = None
        self.is_hung_punctuation_cache: bool | None = None
        self.is_cannot_appear_in_line_end_punctuation_cache: bool | None = None
        self.can_passthrough_cache: bool | None = None
        self.width_cache: float | None = None
        self.height_cache: float | None = None

        self.font_size: float | None = None

        if unicode:
            assert font_size, "Font size must be provided when unicode is provided"
            assert style, "Style must be provided when unicode is provided"
            assert len(unicode) == 1, "Unicode must be a single character"
            assert xobj_id is not None, (
                "Xobj id must be provided when unicode is provided"
            )

            self.font = font
            if font is not None and hasattr(font, "font_id"):
                self.font_id = font.font_id
            else:
                self.font_id = "base"
            if original_font:
                self.original_font = original_font
            else:
                self.original_font = None

            self.font_size = font_size
            self.style = style
            self.xobj_id = xobj_id

    def try_resue_cache(self, old_tu: TypesettingUnit):
        if old_tu.is_cjk_char_cache is not None:
            self.is_cjk_char_cache = old_tu.is_cjk_char_cache

        if old_tu.can_break_line_cache is not None:
            self.can_break_line_cache = old_tu.can_break_line_cache

        if old_tu.is_space_cache is not None:
            self.is_space_cache = old_tu.is_space_cache

        if old_tu.is_hung_punctuation_cache is not None:
            self.is_hung_punctuation_cache = old_tu.is_hung_punctuation_cache

        if old_tu.is_cannot_appear_in_line_end_punctuation_cache is not None:
            self.is_cannot_appear_in_line_end_punctuation_cache = (
                old_tu.is_cannot_appear_in_line_end_punctuation_cache
            )

        if old_tu.can_passthrough_cache is not None:
            self.can_passthrough_cache = old_tu.can_passthrough_cache

        if old_tu.mixed_character_blacklist_cache is not None:
            self.mixed_character_blacklist_cache = (
                old_tu.mixed_character_blacklist_cache
            )

    def try_get_unicode(self) -> str | None:
        if self.char:
            return self.char.char_unicode
        elif self.formular:
            return None
        elif self.unicode:
            return self.unicode

    @property
    def mixed_character_blacklist(self):
        if self.mixed_character_blacklist_cache is None:
            self.mixed_character_blacklist_cache = self.calc_mixed_character_blacklist()

        return self.mixed_character_blacklist_cache

    def calc_mixed_character_blacklist(self):
        unicode = self.try_get_unicode()
        if unicode:
            return unicode in [
                "。",
                "，",
                "：",
                "？",
                "！",
            ]
        return False

    @property
    def can_break_line(self):
        if self.can_break_line_cache is None:
            self.can_break_line_cache = self.calc_can_break_line()

        return self.can_break_line_cache

    def calc_can_break_line(self):
        unicode = self.try_get_unicode()
        if not unicode:
            return True
        if LINE_BREAK_REGEX.match(unicode):
            return False
        return True

    @property
    def bidi_class(self):
        """Simplified bidi class: 'R' strong RTL, 'L' strong LTR, 'N' neutral.

        Formulas (preserved source content, usually Latin/digits) count as
        strong LTR when they contain at least one strong/numeric character, so
        they keep their internal left-to-right order when embedded in an RTL
        line. A formula made only of neutrals (an arrow, "= ", "(" ...) is
        itself neutral: it must join the surrounding RTL flow instead of being
        pinned as a left-to-right island.
        """
        if self.formular:
            classes = [
                char_bidi_class(c.char_unicode)
                for c in self.formular.pdf_character
                if c.char_unicode
            ]
            if not classes or any(cls == "L" for cls in classes):
                # No mapped characters at all: preserved source content,
                # keep the old strong-LTR behaviour.
                return "L"
            if any(cls == "R" for cls in classes):
                return "R"
            return "N"
        s = self.try_get_unicode()
        if not s:
            return "N"
        # Pass the full string: char_bidi_class must see "(cid:NN)"
        # placeholders whole, not their first character "(".
        return char_bidi_class(s)

    def horizontal_shift(self, dx: float):
        """Shift an already relocated unit horizontally, in place."""
        if abs(dx) < 1e-9:
            return

        def _shift_box(box):
            if box is not None:
                box.x += dx
                box.x2 += dx

        if self.char:
            _shift_box(self.char.box)
            if self.char.visual_bbox:
                _shift_box(self.char.visual_bbox.box)
        elif self.formular:
            _shift_box(self.formular.box)
            for char in self.formular.pdf_character:
                _shift_box(char.box)
                if char.visual_bbox:
                    _shift_box(char.visual_bbox.box)
            for curve in self.formular.pdf_curve:
                _shift_box(curve.box)
                if curve.relocation_transform and len(curve.relocation_transform) == 6:
                    curve.relocation_transform[4] += dx
            for form in self.formular.pdf_form:
                _shift_box(form.box)
                if form.relocation_transform and len(form.relocation_transform) == 6:
                    form.relocation_transform[4] += dx
        elif self.unicode:
            if self.x is not None:
                self.x += dx
        self.box_cache = None

    def apply_bidi_mirror(self):
        """Swap mirrorable punctuation when rendered in an RTL visual run."""
        if self.unicode and self.unicode in BIDI_MIRROR_MAP:
            self.unicode = BIDI_MIRROR_MAP[self.unicode]

    @property
    def is_cjk_char(self):
        if self.is_cjk_char_cache is None:
            self.is_cjk_char_cache = self.calc_is_cjk_char()

        return self.is_cjk_char_cache

    def calc_is_cjk_char(self):
        if self.formular:
            return False
        unicode = self.try_get_unicode()
        if not unicode:
            return False
        if "(cid" in unicode:
            return False
        if len(unicode) > 1:
            return False
        assert len(unicode) == 1, "Unicode must be a single character"
        if unicode in [
            "（",
            "）",
            "【",
            "】",
            "《",
            "》",
            "〔",
            "〕",
            "〈",
            "〉",
            "〖",
            "〗",
            "「",
            "」",
            "『",
            "』",
            "、",
            "。",
            "：",
            "？",
            "！",
            "，",
        ]:
            return True
        if unicode:
            if re.match(
                r"^["
                r"\u3000-\u303f"  # CJK Symbols and Punctuation
                r"\u3040-\u309f"  # Hiragana
                r"\u30a0-\u30ff"  # Katakana
                r"\u3100-\u312f"  # Bopomofo
                r"\uac00-\ud7af"  # Hangul Syllables
                r"\u1100-\u11ff"  # Hangul Jamo
                r"\u3130-\u318f"  # Hangul Compatibility Jamo
                r"\ua960-\ua97f"  # Hangul Jamo Extended-A
                r"\ud7b0-\ud7ff"  # Hangul Jamo Extended-B
                r"\u3190-\u319f"  # Kanbun
                r"\u3200-\u32ff"  # Enclosed CJK Letters and Months
                r"\u3300-\u33ff"  # CJK Compatibility
                r"\ufe30-\ufe4f"  # CJK Compatibility Forms
                r"\u4e00-\u9fff"  # CJK Unified Ideographs
                r"\u2e80-\u2eff"  # CJK Radicals Supplement
                r"\u31c0-\u31ef"  # CJK Strokes
                r"\u2f00-\u2fdf"  # Kangxi Radicals
                r"\ufe10-\ufe1f"  # Vertical Forms
                r"]+$",
                unicode,
            ):
                return True
            try:
                unicodedata_name = unicodedata.name(unicode)
                return (
                    "CJK UNIFIED IDEOGRAPH" in unicodedata_name
                    or "FULLWIDTH" in unicodedata_name
                )
            except ValueError:
                return False
        return False

    @property
    def is_space(self):
        if self.is_space_cache is None:
            self.is_space_cache = self.calc_is_space()

        return self.is_space_cache

    def calc_is_space(self):
        if self.formular:
            return False
        unicode = self.try_get_unicode()
        return unicode == " "

    @property
    def is_hung_punctuation(self):
        if self.is_hung_punctuation_cache is None:
            self.is_hung_punctuation_cache = self.calc_is_hung_punctuation()

        return self.is_hung_punctuation_cache

    def calc_is_hung_punctuation(self):
        if self.formular:
            return False
        unicode = self.try_get_unicode()

        if unicode:
            return unicode in [
                # 英文标点
                ",",
                ".",
                ":",
                ";",
                "?",
                "!",
                # 中文点号
                "，",  # 逗号
                "。",  # 句号
                "．",  # 全角句号
                "、",  # 顿号
                "：",  # 冒号
                "；",  # 分号
                "！",  # 叹号
                "‼",  # 双叹号
                "？",  # 问号
                "⁇",  # 双问号
                # 结束引号
                "”",  # 右双引号
                "’",  # 右单引号
                "」",  # 右直角单引号
                "』",  # 右直角双引号
                # 结束括号
                ")",  # 右圆括号
                "]",  # 右方括号
                "}",  # 右花括号
                "）",  # 右圆括号
                "〕",  # 右龟甲括号
                "〉",  # 右单书名号
                "】",  # 右黑色方头括号
                "〗",  # 右空白方头括号
                "］",  # 全角右方括号
                "｝",  # 全角右花括号
                # 结束双书名号
                "》",  # 右双书名号
                # 连接号
                "～",  # 全角波浪号
                "-",  # 连字符减号
                "–",  # 短破折号 (EN DASH)
                "—",  # 长破折号 (EM DASH)
                # 间隔号
                "·",  # 中间点
                "・",  # 片假名中间点
                "‧",  # 连字点
                # 分隔号
                "/",  # 斜杠
                "／",  # 全角斜杠
                "⁄",  # 分数斜杠
            ]
        return False

    @property
    def is_cannot_appear_in_line_end_punctuation(self):
        if self.is_cannot_appear_in_line_end_punctuation_cache is None:
            self.is_cannot_appear_in_line_end_punctuation_cache = (
                self.calc_is_cannot_appear_in_line_end_punctuation()
            )

        return self.is_cannot_appear_in_line_end_punctuation_cache

    def calc_is_cannot_appear_in_line_end_punctuation(self):
        if self.formular:
            return False
        unicode = self.try_get_unicode()
        if not unicode:
            return False
        return unicode in [
            # 开始引号
            "“",  # 左双引号
            "‘",  # 左单引号
            "「",  # 左直角单引号
            "『",  # 左直角双引号
            # 开始括号
            "(",  # 左圆括号
            "[",  # 左方括号
            "{",  # 左花括号
            "（",  # 左圆括号
            "〔",  # 左龟甲括号
            "〈",  # 左单书名号
            "《",  # 左双书名号
            # 开始单双书名号
            "〖",  # 左空白方头括号
            "〘",  # 左黑色方头括号
            "〚",  # 左单书名号
        ]

    def passthrough(
        self,
    ) -> tuple[list[PdfCharacter], list[PdfCurve], list[PdfForm]]:
        if self.char:
            return [self.char], [], []
        elif self.formular:
            return (
                self.formular.pdf_character,
                self.formular.pdf_curve,
                self.formular.pdf_form,
            )
        elif self.unicode:
            logger.error(f"Cannot passthrough unicode. TypesettingUnit: {self}. ")
            logger.error(f"Cannot passthrough unicode. TypesettingUnit: {self}. ")
            return [], [], []

    @property
    def can_passthrough(self):
        if self.can_passthrough_cache is None:
            self.can_passthrough_cache = self.calc_can_passthrough()

        return self.can_passthrough_cache

    def calc_can_passthrough(self):
        return self.unicode is None

    def calculate_box(self):
        if self.char:
            box = copy.deepcopy(self.char.box)
            if self.char.visual_bbox and self.char.visual_bbox.box:
                box.y = self.char.visual_bbox.box.y
                box.y2 = self.char.visual_bbox.box.y2
                # return self.char.visual_bbox.box

            return box
        elif self.formular:
            return self.formular.box
            # if self.formular.x_offset <= 0.5:
            #     return self.formular.box
            # formular_box = copy.copy(self.formular.box)
            # formular_box.x2 += self.formular.x_advance
            # return formular_box
        elif self.unicode:
            char_width = self.font.char_lengths(self.unicode, self.font_size)[0]
            if self.x is None or self.y is None or self.scale is None:
                return Box(0, 0, char_width, self.font_size)
            return Box(self.x, self.y, self.x + char_width, self.y + self.font_size)

    @property
    def box(self):
        if not self.box_cache:
            self.box_cache = self.calculate_box()

        return self.box_cache

    @property
    def width(self):
        if self.width_cache is None:
            self.width_cache = self.calc_width()

        return self.width_cache

    def calc_width(self):
        box = self.box
        return box.x2 - box.x

    @property
    def height(self):
        if self.height_cache is None:
            self.height_cache = self.calc_height()

        return self.height_cache

    def calc_height(self):
        box = self.box
        return box.y2 - box.y

    def relocate(
        self,
        x: float,
        y: float,
        scale: float,
    ) -> TypesettingUnit:
        """重定位并缩放排版单元

        Args:
            x: 新的 x 坐标
            y: 新的 y 坐标
            scale: 缩放因子

        Returns:
            新的排版单元
        """
        if self.char:
            # 创建新的字符对象
            new_char = PdfCharacter(
                pdf_character_id=self.char.pdf_character_id,
                char_unicode=self.char.char_unicode,
                box=Box(
                    x=x,
                    y=y,
                    x2=x + self.width * scale,
                    y2=y + self.height * scale,
                ),
                pdf_style=PdfStyle(
                    font_id=self.char.pdf_style.font_id,
                    font_size=self.char.pdf_style.font_size * scale,
                    graphic_state=self.char.pdf_style.graphic_state,
                ),
                scale=scale,
                vertical=self.char.vertical,
                advance=self.char.advance * scale if self.char.advance else None,
                debug_info=self.debug_info,
                xobj_id=self.char.xobj_id,
            )
            new_tu = TypesettingUnit(char=new_char)
            new_tu.try_resue_cache(self)
            return new_tu

        elif self.formular:
            # 创建新的公式对象，保持内部字符的相对位置
            new_chars = []
            min_x = self.formular.box.x
            min_y = self.formular.box.y

            for char in self.formular.pdf_character:
                # 计算相对位置
                rel_x = char.box.x - min_x
                rel_y = char.box.y - min_y

                visual_rel_x = char.visual_bbox.box.x - min_x
                visual_rel_y = char.visual_bbox.box.y - min_y

                # 创建新的字符对象
                new_char = PdfCharacter(
                    pdf_character_id=char.pdf_character_id,
                    char_unicode=char.char_unicode,
                    box=Box(
                        x=x + (rel_x + self.formular.x_offset) * scale,
                        y=y + (rel_y + self.formular.y_offset) * scale,
                        x2=x
                        + (rel_x + (char.box.x2 - char.box.x) + self.formular.x_offset)
                        * scale,
                        y2=y
                        + (rel_y + (char.box.y2 - char.box.y) + self.formular.y_offset)
                        * scale,
                    ),
                    visual_bbox=il_version_1.VisualBbox(
                        box=Box(
                            x=x + (visual_rel_x + self.formular.x_offset) * scale,
                            y=y + (visual_rel_y + self.formular.y_offset) * scale,
                            x2=x
                            + (
                                visual_rel_x
                                + (char.visual_bbox.box.x2 - char.visual_bbox.box.x)
                                + self.formular.x_offset
                            )
                            * scale,
                            y2=y
                            + (
                                visual_rel_y
                                + (char.visual_bbox.box.y2 - char.visual_bbox.box.y)
                                + self.formular.y_offset
                            )
                            * scale,
                        ),
                    ),
                    pdf_style=PdfStyle(
                        font_id=char.pdf_style.font_id,
                        font_size=char.pdf_style.font_size * scale,
                        graphic_state=char.pdf_style.graphic_state,
                    ),
                    scale=scale,
                    vertical=char.vertical,
                    advance=char.advance * scale if char.advance else None,
                    xobj_id=char.xobj_id,
                )
                new_chars.append(new_char)

            # Calculate bounding box from new_chars
            min_x = min(char.visual_bbox.box.x for char in new_chars)
            min_y = min(char.visual_bbox.box.y for char in new_chars)
            max_x = max(char.visual_bbox.box.x2 for char in new_chars)
            max_y = max(char.visual_bbox.box.y2 for char in new_chars)

            new_formula = PdfFormula(
                box=Box(
                    x=min_x,
                    y=min_y,
                    x2=max_x,
                    y2=max_y,
                ),
                pdf_character=new_chars,
                x_offset=self.formular.x_offset * scale,
                y_offset=self.formular.y_offset * scale,
                x_advance=self.formular.x_advance * scale,
            )

            # Handle contained curves
            new_curves = []
            for curve in self.formular.pdf_curve:
                new_curve = self._transform_curve_for_relocation(
                    curve,
                    self.formular.box.x,
                    self.formular.box.y,
                    x,
                    y,
                    scale,
                )
                new_curves.append(new_curve)
            new_formula.pdf_curve = new_curves

            # Handle contained forms
            new_forms = []
            for form in self.formular.pdf_form:
                new_form = self._transform_form_for_relocation(
                    form, self.formular.box.x, self.formular.box.y, x, y, scale
                )
                new_forms.append(new_form)
            new_formula.pdf_form = new_forms

            update_formula_data(new_formula)

            new_tu = TypesettingUnit(formular=new_formula)
            new_tu.try_resue_cache(self)
            return new_tu

        elif self.unicode:
            # 对于 Unicode 字符，我们存储新的位置信息
            new_unit = TypesettingUnit(
                unicode=self.unicode,
                font=self.font,
                original_font=self.original_font,
                font_size=self.font_size * scale,
                style=self.style,
                xobj_id=self.xobj_id,
                debug_info=self.debug_info,
            )
            new_unit.x = x
            new_unit.y = y
            new_unit.scale = scale
            new_unit.try_resue_cache(self)
            return new_unit

    def _transform_curve_for_relocation(
        self,
        curve,
        original_formula_x: float,
        original_formula_y: float,
        new_x: float,
        new_y: float,
        scale: float,
    ):
        """Transform a curve for formula relocation."""
        new_curve = PdfCurve(
            box=curve.box,
            graphic_state=curve.graphic_state,
            pdf_path=list(curve.pdf_path),
            pdf_original_path=list(curve.pdf_original_path),
            pdf_original_path_primitive=curve.pdf_original_path_primitive,
            debug_info=curve.debug_info,
            fill_background=curve.fill_background,
            stroke_path=curve.stroke_path,
            evenodd=curve.evenodd,
            passthrough_paint=curve.passthrough_paint,
            xobj_id=curve.xobj_id,
            render_order=curve.render_order,
            ctm=list(curve.ctm),
            relocation_transform=list(curve.relocation_transform),
        )

        if new_curve.box:
            # Calculate relative position to formula's original position (same as chars)
            rel_x = new_curve.box.x - original_formula_x
            rel_y = new_curve.box.y - original_formula_y

            # Apply same transformation as characters
            new_curve.box = Box(
                x=new_x + (rel_x + self.formular.x_offset) * scale,
                y=new_y + (rel_y + self.formular.y_offset) * scale,
                x2=new_x
                + (
                    rel_x
                    + (new_curve.box.x2 - new_curve.box.x)
                    + self.formular.x_offset
                )
                * scale,
                y2=new_y
                + (
                    rel_y
                    + (new_curve.box.y2 - new_curve.box.y)
                    + self.formular.y_offset
                )
                * scale,
            )

        # Set relocation transform instead of modifying original CTM
        translation_x = (
            new_x + self.formular.x_offset * scale - original_formula_x * scale
        )
        translation_y = (
            new_y + self.formular.y_offset * scale - original_formula_y * scale
        )

        # Create relocation transformation matrix
        from babeldoc.format.pdf.document_il.utils.matrix_helper import (
            create_translation_and_scale_matrix,
        )

        relocation_matrix = create_translation_and_scale_matrix(
            translation_x, translation_y, scale
        )
        new_curve.relocation_transform = list(relocation_matrix)

        return new_curve

    def _transform_form_for_relocation(
        self,
        form,
        original_formula_x: float,
        original_formula_y: float,
        new_x: float,
        new_y: float,
        scale: float,
    ):
        """Transform a form for formula relocation."""
        new_form = PdfForm(
            box=form.box,
            graphic_state=form.graphic_state,
            pdf_matrix=form.pdf_matrix,
            pdf_affine_transform=form.pdf_affine_transform,
            pdf_form_subtype=form.pdf_form_subtype,
            xobj_id=form.xobj_id,
            ctm=list(form.ctm),
            relocation_transform=list(form.relocation_transform),
            render_order=form.render_order,
            form_type=form.form_type,
        )

        if new_form.box:
            # Calculate relative position to formula's original position (same as chars)
            rel_x = new_form.box.x - original_formula_x
            rel_y = new_form.box.y - original_formula_y

            # Apply same transformation as characters
            new_form.box = Box(
                x=new_x + (rel_x + self.formular.x_offset) * scale,
                y=new_y + (rel_y + self.formular.y_offset) * scale,
                x2=new_x
                + (rel_x + (new_form.box.x2 - new_form.box.x) + self.formular.x_offset)
                * scale,
                y2=new_y
                + (rel_y + (new_form.box.y2 - new_form.box.y) + self.formular.y_offset)
                * scale,
            )

        # Set relocation transform instead of modifying original matrices
        translation_x = (
            new_x + self.formular.x_offset * scale - original_formula_x * scale
        )
        translation_y = (
            new_y + self.formular.y_offset * scale - original_formula_y * scale
        )

        # Create relocation transformation matrix
        from babeldoc.format.pdf.document_il.utils.matrix_helper import (
            create_translation_and_scale_matrix,
        )

        relocation_matrix = create_translation_and_scale_matrix(
            translation_x, translation_y, scale
        )
        new_form.relocation_transform = list(relocation_matrix)

        return new_form

    def render(
        self,
    ) -> tuple[list[PdfCharacter], list[PdfCurve], list[PdfForm]]:
        """渲染排版单元为 PdfCharacter 列表

        Returns:
            PdfCharacter 列表
        """
        if self.can_passthrough:
            return self.passthrough()
        elif self.unicode:
            assert self.x is not None, (
                "x position must be set, should be set by `relocate`"
            )
            assert self.y is not None, (
                "y position must be set, should be set by `relocate`"
            )
            assert self.scale is not None, (
                "scale must be set, should be set by `relocate`"
            )
            x = self.x
            y = self.y
            # if self.original_font and self.font and hasattr(self.original_font, "descent") and hasattr(self.font, "descent_fontmap"):
            #     original_descent = self.original_font.descent
            #     new_descent = self.font.descent_fontmap
            #     y -= (original_descent - new_descent) * self.font_size / 1000

            # 计算字符宽度
            char_width = self.width

            new_char = PdfCharacter(
                pdf_character_id=self.font.has_glyph(ord(self.unicode)),
                char_unicode=self.unicode,
                box=Box(
                    x=x,  # 使用存储的位置
                    y=y,
                    x2=x + char_width,
                    y2=y + self.font_size,
                ),
                pdf_style=PdfStyle(
                    font_id=self.font_id,
                    font_size=self.font_size,
                    graphic_state=self.style.graphic_state,
                ),
                scale=self.scale,
                vertical=False,
                advance=char_width,
                xobj_id=self.xobj_id,
                debug_info=self.debug_info,
            )
            return [new_char], [], []
        else:
            logger.error(f"Unknown typesetting unit. TypesettingUnit: {self}. ")
            logger.error(f"Unknown typesetting unit. TypesettingUnit: {self}. ")
            return [], [], []


class _RtlBidiElement:
    """One bidi-resolvable element of a finished RTL line.

    Either a whole TypesettingUnit, or a single character split out of a
    formula block (so that the formula's edge neutrals can be resolved
    independently of its strong LTR core).
    """

    __slots__ = ("unit", "char", "cls", "text")

    def __init__(self, unit, char, cls, text):
        self.unit = unit
        self.char = char
        self.cls = cls
        self.text = text

    @classmethod
    def from_unit(cls, unit: TypesettingUnit) -> "_RtlBidiElement":
        text = unit.try_get_unicode()
        if text is None and unit.formular:
            text = "".join(
                c.char_unicode
                for c in unit.formular.pdf_character
                if c.char_unicode
            )
        return cls(unit, None, unit.bidi_class, text or "")

    @classmethod
    def from_formula_char(cls, char) -> "_RtlBidiElement":
        text = char.char_unicode or ""
        if not text:
            # Glyph with no unicode at all: preserved source content, keep it
            # anchored in the formula's LTR island.
            return cls(None, char, "L", text)
        return cls(None, char, char_bidi_class(text), text)

    @property
    def x(self) -> float:
        if self.unit is not None:
            return self.unit.box.x
        return self.char.box.x

    @property
    def x2(self) -> float:
        if self.unit is not None:
            return self.unit.box.x2
        return self.char.box.x2

    def shift(self, dx: float) -> None:
        if abs(dx) < 1e-9:
            return
        if self.unit is not None:
            self.unit.horizontal_shift(dx)
            return
        char_box = self.char.box
        char_box.x += dx
        char_box.x2 += dx
        if self.char.visual_bbox and self.char.visual_bbox.box:
            visual_box = self.char.visual_bbox.box
            visual_box.x += dx
            visual_box.x2 += dx

    def apply_bidi_mirror(self) -> None:
        # Formula characters render by glyph id from the original font, so
        # their glyphs cannot be swapped here; only unicode units mirror.
        if self.unit is not None:
            self.unit.apply_bidi_mirror()

    @property
    def is_neutral_operator(self) -> bool:
        return len(self.text) == 1 and self.text in NEUTRAL_OPERATOR_CHARS

    @property
    def starts_with_digit(self) -> bool:
        return bool(self.text) and self.text[0].isdigit()

    @property
    def is_space_like(self) -> bool:
        return bool(self.text) and self.text.isspace()


class Typesetting:
    stage_name = "Typesetting"

    # How far past its source width an image-OCR label box may widen
    # (see _prefit_raster_label_box).
    RASTER_LABEL_MAX_GROWTH = 2.2

    def __init__(self, translation_config: TranslationConfig):
        self.font_mapper = FontMapper(translation_config)
        self.translation_config = translation_config
        self.lang_code = self.translation_config.lang_out.upper()
        self.is_cjk = (
            # Why zh-CN/zh-HK/zh-TW here but not zh-Hans and so on?
            # See https://funstory-ai.github.io/BabelDOC/supported_languages/
            ("ZH" in self.lang_code)  # C
            or ("JA" in self.lang_code)
            or ("JP" in self.lang_code)  # J
            or ("KR" in self.lang_code)  # K
            or ("CN" in self.lang_code)
            or ("HK" in self.lang_code)
            or ("TW" in self.lang_code)
        )
        self.is_rtl = is_rtl_lang(self.lang_code)

    def preprocess_document(self, document: il_version_1.Document, pbar):
        """预处理文档，获取每个段落的最优缩放因子，不执行实际排版"""
        all_scales: list[float] = []
        all_paragraphs: list[il_version_1.PdfParagraph] = []

        for page in document.page:
            pbar.advance()
            # 准备字体信息（复制自 render_page 的逻辑）
            fonts: dict[
                str | int,
                il_version_1.PdfFont | dict[str, il_version_1.PdfFont],
            ] = {f.font_id: f for f in page.pdf_font if f.font_id}
            page_fonts = {f.font_id: f for f in page.pdf_font if f.font_id}
            for k, v in self.font_mapper.fontid2font.items():
                fonts[k] = v
            for xobj in page.pdf_xobject:
                if xobj.xobj_id is not None:
                    fonts[xobj.xobj_id] = page_fonts.copy()
                    for font in xobj.pdf_font:
                        if (
                            xobj.xobj_id in fonts
                            and isinstance(fonts[xobj.xobj_id], dict)
                            and font.font_id
                        ):
                            fonts[xobj.xobj_id][font.font_id] = font

            # 处理每个段落
            for paragraph in page.pdf_paragraph:
                all_paragraphs.append(paragraph)
                unit_count = 0
                try:
                    typesetting_units = self.create_typesetting_units(paragraph, fonts)
                    unit_count = len(typesetting_units)
                    for unit in typesetting_units:
                        if unit.formular:
                            unit_count += len(unit.formular.pdf_character) - 1

                    # 如果所有单元都可以直接传递，则 scale = 1.0
                    if all(unit.can_passthrough for unit in typesetting_units):
                        paragraph.optimal_scale = 1.0
                    else:
                        # A raster label's box is widened before its scale is
                        # computed, so the scale reflects the fitted box.
                        self._prefit_raster_label_box(
                            paragraph, page, typesetting_units
                        )
                        # 获取最优缩放因子
                        optimal_scale = self._get_optimal_scale(
                            paragraph, page, typesetting_units
                        )
                        paragraph.optimal_scale = optimal_scale
                except Exception as e:
                    # 如果预处理出错，默认使用 1.0 缩放因子
                    logger.warning(f"预处理段落时出错：{e}")
                    paragraph.optimal_scale = 1.0

                if paragraph.optimal_scale is not None:
                    all_scales.extend([paragraph.optimal_scale] * unit_count)

        # 获取缩放因子的众数
        if all_scales:
            try:
                modes = statistics.multimode(all_scales)
                mode_scale = min(modes)
            except statistics.StatisticsError:
                logger.warning(
                    "Could not find a mode for paragraph scales. Falling back to median."
                )
                mode_scale = statistics.median(all_scales)
            # 将所有大于众数的值修改为众数
            for paragraph in all_paragraphs:
                if (
                    paragraph.optimal_scale is not None
                    and paragraph.optimal_scale > mode_scale
                ):
                    paragraph.optimal_scale = mode_scale
        else:
            logger.error(
                "document_scales is empty, there seems no paragraph in this PDF"
            )

    def _find_optimal_scale_and_layout(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        typesetting_units: list[TypesettingUnit],
        initial_scale: float = 1.0,
        use_english_line_break: bool = True,
        apply_layout: bool = False,
    ) -> tuple[float, list[TypesettingUnit] | None]:
        """查找最优缩放因子并可选择性地执行布局

        Args:
            paragraph: 段落对象
            page: 页面对象
            typesetting_units: 排版单元列表
            initial_scale: 初始缩放因子
            use_english_line_break: 是否使用英文换行规则
            apply_layout: 是否应用布局到 paragraph（True 时执行实际排版）

        Returns:
            tuple[float, list[TypesettingUnit] | None]: (最终缩放因子，排版后的单元列表或 None)
        """
        if not paragraph.box:
            return initial_scale, None

        box = paragraph.box
        scale = initial_scale
        line_skip = 1.50 if self.is_cjk else 1.3
        min_scale = self._legibility_min_scale(typesetting_units)
        expand_space_flag = 0
        final_typeset_units = None
        # Smallest-scale layout seen that did NOT fit, kept so the paragraph
        # can be drawn overflowing at the floor instead of vanishing.
        overflow_candidate: tuple[float, list[TypesettingUnit]] | None = None
        rtl_align = self._rtl_ocr_paragraph_align(paragraph, page)

        while scale >= min_scale:
            try:
                # 尝试布局排版单元
                typeset_units, all_units_fit = self._layout_typesetting_units(
                    typesetting_units,
                    box,
                    scale,
                    line_skip,
                    paragraph,
                    use_english_line_break,
                    rtl_align=rtl_align,
                )

                # 如果所有单元都放得下
                if all_units_fit:
                    if apply_layout:
                        self._apply_typeset_units(paragraph, page, typeset_units, scale)
                        final_typeset_units = typeset_units
                    return scale, final_typeset_units
                # Does not fit. Keep the tightest layout seen: if the search
                # runs out of room above the legibility floor this is what
                # gets drawn (overflowing) rather than nothing at all.
                if overflow_candidate is None or scale <= overflow_candidate[0]:
                    overflow_candidate = (scale, typeset_units)
            except Exception:
                # 如果布局检查出错，继续尝试下一个缩放因子
                pass

            # 添加与原 retypeset 一致的逻辑检查
            if not hasattr(paragraph, "debug_id") or not paragraph.debug_id:
                return self._settle_at_floor(
                    paragraph, page, overflow_candidate, min_scale,
                    final_typeset_units, apply_layout,
                )

            # 减小缩放因子
            if scale > 0.6:
                scale -= 0.05
            else:
                scale -= 0.1

            if scale < 0.7:
                space_expanded = False  # 标记是否成功扩展了空间

                raster_region = self._raster_region(paragraph)
                if expand_space_flag == 0:
                    # 尝试向下扩展
                    try:
                        min_y = self.get_max_bottom_space(box, page) + 2
                        if raster_region is not None:
                            # An image-OCR label may only grow inside its
                            # own raster image (Contract 2).
                            min_y = max(min_y, float(raster_region[1]) + 1)
                        if min_y < box.y:
                            expanded_box = Box(x=box.x, y=min_y, x2=box.x2, y2=box.y2)
                            box = expanded_box
                            if apply_layout:
                                # 更新段落的边界框
                                paragraph.box = expanded_box
                            space_expanded = True
                    except Exception:
                        pass
                    expand_space_flag = 1

                    # 只有成功扩展空间时才 continue，否则继续减小 scale
                    if space_expanded:
                        continue

                elif expand_space_flag == 1:
                    # 尝试向右扩展
                    try:
                        max_x = self.get_max_right_space(box, page) - 5
                        if raster_region is not None:
                            max_x = min(max_x, float(raster_region[2]) - 1)
                        if max_x > box.x2:
                            expanded_box = Box(x=box.x, y=box.y, x2=max_x, y2=box.y2)
                            box = expanded_box
                            if apply_layout:
                                # 更新段落的边界框
                                paragraph.box = expanded_box
                            space_expanded = True
                    except Exception:
                        pass
                    expand_space_flag = 2

                    # 只有成功扩展空间时才 continue，否则继续减小 scale
                    if space_expanded:
                        continue

                # 只有在扩展尝试阶段 (expand_space_flag < 2) 且扩展失败时才重置 scale
                # 当 expand_space_flag >= 2 时，说明已经尝试过所有扩展，应该继续正常的 scale 减小
                if expand_space_flag < 2:
                    # 如果无法扩展空间，重置 scale 并继续循环
                    scale = 1.0

        # 如果仍然放不下，尝试去除英文换行限制
        if use_english_line_break:
            return self._find_optimal_scale_and_layout(
                paragraph,
                page,
                typesetting_units,
                initial_scale,
                use_english_line_break=False,
                apply_layout=apply_layout,
            )

        # Nothing fits, even at the legibility floor and without the English
        # line-break rules. Draw it at the floor and let it overflow.
        return self._settle_at_floor(
            paragraph, page, overflow_candidate, min_scale,
            final_typeset_units, apply_layout,
        )

    def _legibility_min_scale(
        self, typesetting_units: list["TypesettingUnit"]
    ) -> float:
        """Smallest scale this paragraph may be shrunk to.

        Floors the RESULTING point size, not just the ratio: the source's own
        dominant size decides how much of a shrink `MIN_LEGIBLE_FONT_SIZE_PT`
        allows. Source text that is already at or below the floor is never
        shrunk at all (and never enlarged — the clamp is one-sided).
        """
        font_sizes = [
            size
            for unit in typesetting_units
            for size in (
                unit.font_size,
                unit.char.pdf_style.font_size
                if unit.char is not None and unit.char.pdf_style is not None
                else None,
            )
            if size
        ]
        if not font_sizes:
            return MIN_LEGIBLE_SCALE
        try:
            base_size = statistics.mode(font_sizes)
        except statistics.StatisticsError:
            base_size = statistics.median(font_sizes)
        if base_size <= 0:
            return MIN_LEGIBLE_SCALE
        return min(1.0, max(MIN_LEGIBLE_FONT_SIZE_PT / base_size, MIN_LEGIBLE_SCALE))

    def _apply_typeset_units(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        typeset_units: list["TypesettingUnit"],
        scale: float,
    ) -> None:
        """Write a finished layout onto the paragraph and page."""
        paragraph.scale = scale
        paragraph.pdf_paragraph_composition = []
        for unit in typeset_units:
            chars, curves, forms = unit.render()
            for char in chars:
                paragraph.pdf_paragraph_composition.append(
                    PdfParagraphComposition(pdf_character=char),
                )
            for curve in curves:
                page.pdf_curve.append(curve)
            for form in forms:
                page.pdf_form.append(form)

    def _settle_at_floor(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        overflow_candidate: tuple[float, list["TypesettingUnit"]] | None,
        min_scale: float,
        final_typeset_units: list["TypesettingUnit"] | None,
        apply_layout: bool,
    ) -> tuple[float, list["TypesettingUnit"] | None]:
        """Give up on fitting and draw the paragraph at the legibility floor.

        The old code returned `min_scale` with nothing laid out, which — on the
        apply pass — left `pdf_paragraph_composition` empty and dropped the
        paragraph from the page entirely. Overflowing text the reader can see
        and read beats text that is either invisible or 1.3 pt tall.
        """
        if overflow_candidate is None:
            return min_scale, final_typeset_units
        scale, typeset_units = overflow_candidate
        if apply_layout:
            self._apply_typeset_units(paragraph, page, typeset_units, scale)
            final_typeset_units = typeset_units
        return scale, final_typeset_units

    def _rtl_ocr_paragraph_align(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
    ) -> str:
        """Line alignment ("left"/"center"/"right") for an RTL paragraph.

        Non-OCR RTL pages are mirrored, so in-box right alignment is always
        correct there. OCR-workaround pages sit on an unmirrorable raster:
        a short left-anchored original (a slide title) whose translation is
        right-aligned inside the same tight box looks like it floats. Anchor
        such paragraphs to the edge the original used, inferred from the
        paragraph box position on the page. List items (they carry a textual
        bullet marker) and full-width body text keep the natural RTL right
        margin.
        """
        if not self.is_rtl:
            return "right"
        shared_context = getattr(
            self.translation_config, "shared_context_cross_split_part", None
        )
        auto_ocr = getattr(
            self.translation_config,
            "auto_enabled_ocr_workaround",
            getattr(shared_context, "auto_enabled_ocr_workaround", False),
        )
        if not (getattr(self.translation_config, "ocr_workaround", False) or auto_ocr):
            return "right"
        box = paragraph.box
        page_box = page.cropbox.box if page.cropbox else None
        if not (self._box_is_valid(box) and self._box_is_valid(page_box)):
            return "right"
        text = (paragraph.unicode or "").lstrip()
        if text and BULLET_POINT_PATTERN.match(text[0]):
            return "right"
        page_width = page_box.x2 - page_box.x
        if page_width <= 0 or box.x2 - box.x > 0.62 * page_width:
            return "right"
        left_gap = box.x - page_box.x
        right_gap = page_box.x2 - box.x2
        if abs(left_gap - right_gap) <= 0.08 * page_width:
            return "center"
        return "left" if left_gap < right_gap else "right"

    def _get_optimal_scale(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        typesetting_units: list[TypesettingUnit],
        use_english_line_break: bool = True,
    ) -> float:
        """获取段落的最优缩放因子，不执行实际排版"""
        scale, _ = self._find_optimal_scale_and_layout(
            paragraph,
            page,
            typesetting_units,
            1.0,
            use_english_line_break,
            apply_layout=False,
        )
        return scale

    def retypeset_with_precomputed_scale(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        typesetting_units: list[TypesettingUnit],
        precomputed_scale: float,
        use_english_line_break: bool = True,
    ):
        """使用预计算的缩放因子进行排版"""
        if not paragraph.box:
            return

        # 使用通用方法进行排版，传入预计算的缩放因子作为初始值
        self._find_optimal_scale_and_layout(
            paragraph,
            page,
            typesetting_units,
            precomputed_scale,
            use_english_line_break,
            apply_layout=True,
        )

    def typesetting_document(self, document: il_version_1.Document):
        # Corrupt (emoji) visual bboxes must be repaired before mirroring or
        # layout: they inflate formula and paragraph boxes, which corrupts the
        # mirror translation and the line layout alike.
        if self.is_rtl:
            for page in document.page:
                for paragraph in page.pdf_paragraph:
                    try:
                        self._sanitize_paragraph_visual_boxes(page, paragraph)
                    except Exception:
                        logger.exception(
                            "Failed to sanitize visual boxes for paragraph "
                            f"{getattr(paragraph, 'debug_id', None)}"
                        )
        # RTL page mirroring must run before any layout: paragraphs are then
        # typeset directly into their final (mirrored) boxes, so the per-line
        # RTL logic never double-mirrors.
        # OCR-workaround pages are backed by a full-page raster that cannot be
        # mirrored, so masks and translated text must stay at their original
        # positions; only in-box RTL alignment applies there.
        shared_context = getattr(
            self.translation_config, "shared_context_cross_split_part", None
        )
        auto_ocr = getattr(
            self.translation_config,
            "auto_enabled_ocr_workaround",
            getattr(shared_context, "auto_enabled_ocr_workaround", False),
        )
        if self.is_rtl and not (
            getattr(self.translation_config, "ocr_workaround", False) or auto_ocr
        ):
            for page in document.page:
                try:
                    self._mirror_page_layout(page)
                except Exception:
                    logger.exception(
                        f"Failed to mirror RTL page layout for page "
                        f"{page.page_number}"
                    )
        # 原有的排版逻辑
        if self.translation_config.progress_monitor:
            with self.translation_config.progress_monitor.stage_start(
                self.stage_name,
                len(document.page) * 2,
            ) as pbar:
                # 预处理：获取所有段落的最优缩放因子
                self.preprocess_document(document, pbar)

                for page in document.page:
                    self.translation_config.raise_if_cancelled()
                    self.render_page(page)
                    pbar.advance()
        else:
            for page in document.page:
                self.translation_config.raise_if_cancelled()
                self.render_page(page)

    # ------------------------------------------------------------------
    # Visual-bbox sanitization (corrupt emoji glyph boxes)
    # ------------------------------------------------------------------
    def _sanitize_paragraph_visual_boxes(
        self, page: il_version_1.Page, paragraph: il_version_1.PdfParagraph
    ) -> None:
        """Repair paragraphs whose boxes were inflated by corrupt glyph boxes.

        Color-emoji glyphs report visual bboxes covering the whole font
        matrix (10x the advance box). Those propagate into formula boxes and
        the paragraph box, so the paragraph anchors far away from its real
        text line (legend lines rendered inside the banner above, emoji
        mirrored onto headings). Clamp such visual bboxes to the advance box,
        recompute the affected formula boxes, and pull the paragraph box back
        to its surviving content.
        """
        changed = False
        content_boxes: list[Box] = []
        inflated_pre_x2 = None

        def collect(chars) -> None:
            for char in chars:
                if self._box_is_valid(char.box):
                    content_boxes.append(char.box)

        for composition in paragraph.pdf_paragraph_composition or []:
            if composition is None:
                continue
            if composition.pdf_formula:
                formula = composition.pdf_formula
                formula_changed = False
                for char in formula.pdf_character:
                    box = char.box
                    visual = char.visual_bbox.box if char.visual_bbox else None
                    if not (self._box_is_valid(box) and self._box_is_valid(visual)):
                        continue
                    char_w = box.x2 - box.x
                    char_h = box.y2 - box.y
                    if char_w <= 0 or char_h <= 0:
                        continue
                    if (
                        visual.x2 - visual.x
                        > VISUAL_BBOX_INFLATION_RATIO * char_w + 2
                        or visual.y2 - visual.y
                        > VISUAL_BBOX_INFLATION_RATIO * char_h + 2
                    ):
                        if inflated_pre_x2 is None or visual.x2 > inflated_pre_x2:
                            inflated_pre_x2 = visual.x2
                        char.visual_bbox.box = copy.deepcopy(box)
                        formula_changed = True
                if formula_changed and formula.pdf_character:
                    update_formula_data(formula)
                    changed = True
                    # Unrelated page graphics (chip/pill backgrounds) can get
                    # absorbed into the formula through the corrupt visual
                    # bbox. Members that do not even touch the repaired
                    # formula box belong to the page: return them so they are
                    # mirrored/rendered like their untouched siblings.
                    self._evict_foreign_formula_members(page, formula)
                collect(formula.pdf_character)
            elif composition.pdf_line:
                collect(composition.pdf_line.pdf_character)
            elif composition.pdf_same_style_characters:
                collect(composition.pdf_same_style_characters.pdf_character)
            elif composition.pdf_character:
                collect([composition.pdf_character])

        if not changed or not content_boxes:
            return
        box = paragraph.box
        if not self._box_is_valid(box):
            return
        union_x = min(b.x for b in content_boxes)
        union_y = min(b.y for b in content_boxes)
        union_x2 = max(b.x2 for b in content_boxes)
        union_y2 = max(b.y2 for b in content_boxes)

        font_sizes = [
            comp.pdf_same_style_unicode_characters.pdf_style.font_size
            for comp in paragraph.pdf_paragraph_composition or []
            if comp is not None
            and comp.pdf_same_style_unicode_characters is not None
            and comp.pdf_same_style_unicode_characters.pdf_style is not None
            and comp.pdf_same_style_unicode_characters.pdf_style.font_size
        ]
        if paragraph.pdf_style is not None and paragraph.pdf_style.font_size:
            font_sizes.append(paragraph.pdf_style.font_size)
        line_height = max([*font_sizes, union_y2 - union_y, 1.0])

        if box.y > union_y2 or box.y2 < union_y:
            # The whole stored box is displaced away from its real content
            # (single-formula emoji paragraphs): trust the content.
            box.x, box.y = union_x - 0.5, union_y - 0.5
            box.x2, box.y2 = union_x2 + 0.5, union_y2 + 0.5
            return
        if box.y2 - union_y2 > 2.5 * line_height:
            box.y2 = union_y2 + 0.2 * line_height
        if union_y - box.y > 2.5 * line_height:
            box.y = union_y - 0.2 * line_height
        # Only reclaim the right edge when it was set by an inflated visual
        # box; translated text (which carries no boxes) may legitimately
        # extend past the surviving content on either side.
        if (
            inflated_pre_x2 is not None
            and inflated_pre_x2 >= box.x2 - 1
            and box.x2 - union_x2 > 2.5 * line_height
        ):
            box.x2 = union_x2 + 0.2 * line_height

    def _evict_foreign_formula_members(
        self, page: il_version_1.Page, formula: PdfFormula
    ) -> None:
        """Move formula curves/forms that don't touch the formula box back to
        the page level."""
        fbox = formula.box
        if not self._box_is_valid(fbox):
            return

        def is_foreign(box) -> bool:
            if not self._box_is_valid(box):
                return False
            return (
                box.x2 < fbox.x - 1
                or box.x > fbox.x2 + 1
                or box.y2 < fbox.y - 1
                or box.y > fbox.y2 + 1
            )

        kept_curves = []
        for curve in formula.pdf_curve:
            if is_foreign(curve.box):
                page.pdf_curve.append(curve)
            else:
                kept_curves.append(curve)
        formula.pdf_curve = kept_curves

        kept_forms = []
        for form in formula.pdf_form:
            if is_foreign(form.box):
                page.pdf_form.append(form)
            else:
                kept_forms.append(form)
        formula.pdf_form = kept_forms

    # ------------------------------------------------------------------
    # RTL page mirroring
    # ------------------------------------------------------------------
    @staticmethod
    def _box_is_valid(box) -> bool:
        return box is not None and None not in (box.x, box.y, box.x2, box.y2)

    @staticmethod
    def _shift_box_x(box, dx: float) -> None:
        box.x += dx
        box.x2 += dx

    @staticmethod
    def _shift_char_x(char, dx: float) -> None:
        if char.box is not None and char.box.x is not None:
            char.box.x += dx
            char.box.x2 += dx
        if (
            char.visual_bbox
            and char.visual_bbox.box
            and char.visual_bbox.box.x is not None
        ):
            char.visual_bbox.box.x += dx
            char.visual_bbox.box.x2 += dx

    @staticmethod
    def _compose_device_translation(obj, dx: float) -> None:
        """Add a device-space horizontal translation to a curve/form.

        The relocation transform is emitted first in the content stream, so it
        is the outermost matrix: composing a translation reduces to adding dx
        to its e-component. Content is only translated, never flipped.
        """
        transform = getattr(obj, "relocation_transform", None)
        if transform is not None and len(transform) == 6:
            transform[4] = float(transform[4]) + dx
        else:
            obj.relocation_transform = [1.0, 0.0, 0.0, 1.0, dx, 0.0]

    @staticmethod
    def _paragraph_will_retypeset(paragraph: il_version_1.PdfParagraph) -> bool:
        """Mirror of render_paragraph's passthrough decision.

        A paragraph is retypeset when it contains at least one translated
        unicode composition (those cannot pass through). Retypeset paragraphs
        only need their box mirrored; passthrough paragraphs render at their
        stored character positions and need their content shifted too.
        """
        for composition in paragraph.pdf_paragraph_composition or []:
            unicode_comp = composition.pdf_same_style_unicode_characters
            if (
                unicode_comp is not None
                and unicode_comp.unicode
                and unicode_comp.pdf_style is not None
                and unicode_comp.pdf_style.font_id is not None
            ):
                return True
        return False

    def _shift_paragraph_content(
        self, paragraph: il_version_1.PdfParagraph, dx: float
    ) -> None:
        for composition in paragraph.pdf_paragraph_composition or []:
            if composition is None:
                continue
            if composition.pdf_line:
                line = composition.pdf_line
                if self._box_is_valid(line.box):
                    self._shift_box_x(line.box, dx)
                for char in line.pdf_character:
                    self._shift_char_x(char, dx)
            elif composition.pdf_character:
                self._shift_char_x(composition.pdf_character, dx)
            elif composition.pdf_same_style_characters:
                same_style = composition.pdf_same_style_characters
                if self._box_is_valid(getattr(same_style, "box", None)):
                    self._shift_box_x(same_style.box, dx)
                for char in same_style.pdf_character:
                    self._shift_char_x(char, dx)
            elif composition.pdf_formula:
                formula = composition.pdf_formula
                if self._box_is_valid(formula.box):
                    self._shift_box_x(formula.box, dx)
                for char in formula.pdf_character:
                    self._shift_char_x(char, dx)
                for curve in formula.pdf_curve:
                    if self._box_is_valid(curve.box):
                        self._shift_box_x(curve.box, dx)
                    self._compose_device_translation(curve, dx)
                for form in formula.pdf_form:
                    if self._box_is_valid(form.box):
                        self._shift_box_x(form.box, dx)
                    self._compose_device_translation(form, dx)

    def _mirror_loose_characters(
        self, page, pivot: float, in_scope
    ) -> list[tuple[tuple[float, float, float, float], float]]:
        """Mirror characters that never joined a paragraph.

        Characters are clustered into visual runs (same line, small gaps) and
        each run is translated rigidly, so multi-character labels keep their
        internal order instead of being scrambled per glyph.

        Returns the clusters as ``(bbox, dx)`` anchors so that page-level
        curves/forms living inside a cluster (fraction bars, radical rules of
        a preserved equation) can follow the same rigid translation.
        """
        loose = [
            char
            for char in page.pdf_character
            if in_scope(char.xobj_id) and self._box_is_valid(char.box)
        ]
        if not loose:
            return []
        # Union-find over generously expanded boxes: sub/superscripts,
        # fraction parts and operator spacing of a formula must all land in
        # ONE cluster, otherwise the mirror scrambles the formula.
        loose.sort(key=lambda c: c.box.y)
        count = len(loose)
        parent = list(range(count))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        boxes = [(c.box.x, c.box.y, c.box.x2, c.box.y2) for c in loose]
        for i in range(count):
            x1, y1, x2, y2 = boxes[i]
            height_i = y2 - y1
            for j in range(i + 1, count):
                a1, b1, a2, b2 = boxes[j]
                if b1 - y2 > 40:
                    break  # sorted by y: nothing below can connect anymore
                height = max(height_i, b2 - b1, 5.0)
                x_gap = max(a1 - x2, x1 - a2)
                y_gap = max(b1 - y2, y1 - b2)
                if x_gap < 1.5 * height and y_gap < 0.5 * height:
                    root_i, root_j = find(i), find(j)
                    if root_i != root_j:
                        parent[root_j] = root_i
        clusters_by_root: dict[int, list] = {}
        for i, char in enumerate(loose):
            clusters_by_root.setdefault(find(i), []).append(char)
        anchors: list[tuple[tuple[float, float, float, float], float]] = []
        for cluster in clusters_by_root.values():
            left = min(c.box.x for c in cluster)
            right = max(c.box.x2 for c in cluster)
            bottom = min(c.box.y for c in cluster)
            top = max(c.box.y2 for c in cluster)
            dx = pivot - left - right
            anchors.append(((left, bottom, right, top), dx))
            if abs(dx) < 1e-6:
                continue
            for char in cluster:
                self._shift_char_x(char, dx)
        return anchors

    def _mirror_page_layout(self, page: il_version_1.Page) -> None:
        """Mirror the page skeleton around its vertical midline for RTL.

        Every page-space layout element's bounding box is mirrored
        (new_x = page_right + page_left - old_x2) by translating the element
        rigidly — content is never flipped: images keep their pixels, formulas
        keep their glyph order, vector paths are translated via their
        relocation transform. Elements living inside form XObjects use the
        XObject's own coordinate space and are left untouched.
        """
        ref_box = None
        if page.cropbox is not None and self._box_is_valid(page.cropbox.box):
            ref_box = page.cropbox.box
        elif page.mediabox is not None and self._box_is_valid(page.mediabox.box):
            ref_box = page.mediabox.box
        if ref_box is None:
            return
        pivot = ref_box.x + ref_box.x2
        xobj_ids = {
            xobj.xobj_id for xobj in page.pdf_xobject if xobj.xobj_id is not None
        }

        # Form XObjects placed at page level via a single pure-translation
        # matrix (typically a full-page "EmbeddedPdfPage" wrapper) act as an
        # alternate page coordinate space: mirror their CONTENT around the
        # equivalent local pivot and leave their Do-form untouched. Any other
        # XObject placement is opaque: the Do-form is translated to the
        # mirrored position and the content (images included) is never
        # touched, so nothing is ever flipped.
        xref_to_xobj = {
            xobj.xref_id: xobj for xobj in page.pdf_xobject if xobj.xref_id
        }
        placements: dict[int, list] = {}
        for form in page.pdf_form:
            if form.xobj_id in xobj_ids:
                continue  # nested placement: moves with its parent scope
            subtype = form.pdf_form_subtype
            xobj_form = subtype.pdf_xobj_form if subtype is not None else None
            if xobj_form is not None and xobj_form.xref_id in xref_to_xobj:
                placements.setdefault(xobj_form.xref_id, []).append(form)

        scope_pivots: dict[int | None, float] = {None: pivot}
        wrapper_form_ids: set[int] = set()
        for xref_id, forms in placements.items():
            offsets = []
            pure_translation = True
            for form in forms:
                matrix = form.pdf_matrix
                if (
                    matrix is None
                    or abs((matrix.a or 0.0) - 1.0) > 1e-3
                    or abs((matrix.d or 0.0) - 1.0) > 1e-3
                    or abs(matrix.b or 0.0) > 1e-3
                    or abs(matrix.c or 0.0) > 1e-3
                ):
                    pure_translation = False
                    break
                offsets.append(matrix.e or 0.0)
            if (
                pure_translation
                and offsets
                and max(offsets) - min(offsets) < 1e-3
            ):
                xobj = xref_to_xobj[xref_id]
                scope_pivots[xobj.xobj_id] = pivot - 2.0 * offsets[0]
                wrapper_form_ids.update(id(form) for form in forms)

        unmirrored_xobjs = xobj_ids - set(scope_pivots)
        if unmirrored_xobjs:
            logger.info(
                f"RTL mirror: leaving XObject scopes {sorted(unmirrored_xobjs)} "
                f"as opaque blocks on page {page.page_number}"
            )

        for scope_id, scope_pivot in scope_pivots.items():
            if scope_id is None:
                def in_scope(xid, _ids=xobj_ids):
                    return xid not in _ids
            else:
                def in_scope(xid, _sid=scope_id):
                    return xid == _sid
            self._mirror_scope(
                page, scope_pivot, in_scope, wrapper_form_ids
            )

    @staticmethod
    def _raster_region(obj) -> list[float] | None:
        """The image-text region an element is anchored to, if any.

        Image-OCR label paragraphs and their masks carry the containing
        raster image's placement box (Contract 2): under the RTL mirror they
        must ride that image's rigid translation, never mirror on their own.
        """
        region = getattr(obj, "raster_region", None)
        if region and len(region) == 4 and None not in region:
            return region
        return None

    @classmethod
    def _shift_raster_region(cls, obj, dx: float) -> None:
        region = cls._raster_region(obj)
        if region is not None:
            region[0] = float(region[0]) + dx
            region[2] = float(region[2]) + dx

    @staticmethod
    def _find_anchor_dx(anchors, box) -> float | None:
        """dx of the smallest paragraph box containing `box`, if any."""
        best = None
        best_area = None
        for (ax, ay, ax2, ay2), dx in anchors:
            if (
                ax - 1.0 <= box.x
                and ax2 + 1.0 >= box.x2
                and ay - 1.0 <= box.y
                and ay2 + 1.0 >= box.y2
            ):
                area = (ax2 - ax) * (ay2 - ay)
                if best_area is None or area < best_area:
                    best_area = area
                    best = dx
        return best

    # A layout region is only allowed to stand for its members when they
    # agree: this much of a member must lie inside the region, and their
    # translations must not differ by more than this many points.
    REGION_MEMBER_COVERAGE = 0.6
    REGION_MEMBER_DX_TOLERANCE = 0.5

    def _layout_region_anchors(
        self,
        page: il_version_1.Page,
        member_anchors: list[tuple[tuple[float, float, float, float], float]],
    ) -> list[tuple[tuple[float, float, float, float], float]]:
        """Anchors from the page's own layout regions.

        A vector path that BRACKETS a group — a parenthesis around a
        fraction, a rule under a heading — reaches outside the box of the
        text it belongs to, so containment in a paragraph box misses it and
        the first larger paragraph that does contain it captures it
        instead. run39 p15's second worked example lost its parentheses
        that way: its own text moved by -414.58 pt while the parentheses
        took the -229.11 pt of the blue paragraph above, and the delivered
        page showed an empty pair of brackets 180 pt from its fraction.

        The regions the page was parsed into are the grouping the document
        itself declares, so let a region stand as an anchor for the
        elements inside it — but only when the members it contains all
        move together, since a region that spans two groups cannot speak
        for either.
        """
        anchors: list[tuple[tuple[float, float, float, float], float]] = []
        for layout in page.page_layout or []:
            if layout.class_name == "fallback_line":
                continue
            box = layout.box
            if box is None or None in (box.x, box.y, box.x2, box.y2):
                continue
            members = [
                (member_box, dx)
                for member_box, dx in member_anchors
                if self._box_coverage(member_box, box)
                >= self.REGION_MEMBER_COVERAGE
            ]
            if not members:
                continue
            dxs = [dx for _member_box, dx in members]
            if max(dxs) - min(dxs) > self.REGION_MEMBER_DX_TOLERANCE:
                continue
            anchors.append(
                (
                    (
                        min(box.x, *(m[0][0] for m in members)),
                        min(box.y, *(m[0][1] for m in members)),
                        max(box.x2, *(m[0][2] for m in members)),
                        max(box.y2, *(m[0][3] for m in members)),
                    ),
                    dxs[0],
                )
            )
        return anchors

    # Layout classes that stand for a piece of GRAPHIC content, i.e. the
    # regions that can speak for a raster placement's visible extent.
    GRAPHIC_REGION_CLASSES = frozenset({"figure", "table"})
    # A region may stand for a placement's ink only if it is essentially
    # INSIDE that placement: padding makes the ink smaller than the box,
    # never larger. The allowance covers the detector's own quantisation
    # (integer pixel boxes, expanded by one pixel) — run14 p22's figure
    # region reaches 4.09 pt past its placement on the left for that
    # reason, while run67 p6's region reaches 74.8 pt past its raster
    # because it is drawn around the raster AND its label column.
    GRAPHIC_REGION_SLOP_PT = 6.0
    # And it must actually be that placement rather than a piece of it:
    # run67 p5's only figure region covers a third of the full-page
    # background raster it sits on.
    GRAPHIC_REGION_MIN_COVERAGE = 0.7
    # Below this the correction is indistinguishable from detector noise,
    # so the placement keeps its own reflection. run14 p22's real padding
    # asks for 23.65 pt.
    GRAPHIC_REGION_MIN_CORRECTION_PT = 10.0

    def _graphic_dx(self, page: il_version_1.Page, box, pivot: float) -> float:
        """Mirror translation for a placed graphic.

        A raster's PLACEMENT box is not what the reader sees: an image
        whose alpha mask is empty down one side is placed wider than it
        prints, and reflecting the placement therefore moves the visible
        picture by the width of the padding. run14 p22, MEASURED: the
        figure is placed at x[275.09,505.56] but its soft mask is empty
        past column 550 of 602, so it prints to x=486.04 — 19.5 pt of
        nothing on the right. Reflected as placed it printed from x=89.2,
        which is 16.8 pt OUTSIDE the slide's own border at x=106.0, and its
        opaque background painted over that border.

        The layout regions were detected on the RENDERED page, so where a
        region sits INSIDE a placement and fills most of it, the region is
        that placement's ink and reflecting it puts the picture back where
        it belongs. A region that reaches outside the placement is
        something else — a figure drawn around the raster and its labels —
        and speaking for the raster with it would drag the picture away
        from the text that stayed put.
        """
        own = pivot - box.x - box.x2
        best_coverage = self.GRAPHIC_REGION_MIN_COVERAGE
        best = None
        for layout in page.page_layout or []:
            if layout.class_name not in self.GRAPHIC_REGION_CLASSES:
                continue
            region = layout.box
            if region is None or None in (
                region.x, region.y, region.x2, region.y2
            ):
                continue
            slop = self.GRAPHIC_REGION_SLOP_PT
            if (
                region.x < box.x - slop
                or region.y < box.y - slop
                or region.x2 > box.x2 + slop
                or region.y2 > box.y2 + slop
            ):
                continue  # reaches outside the placement: not its ink
            coverage = self._box_coverage(
                (box.x, box.y, box.x2, box.y2), region
            )
            if coverage > best_coverage:
                best_coverage = coverage
                best = region
        if best is None:
            return own
        corrected = pivot - best.x - best.x2
        if abs(corrected - own) < self.GRAPHIC_REGION_MIN_CORRECTION_PT:
            return own
        return corrected

    @staticmethod
    def _region_box(region: list[float]) -> il_version_1.Box:
        return il_version_1.Box(
            x=float(region[0]),
            y=float(region[1]),
            x2=float(region[2]),
            y2=float(region[3]),
        )

    @staticmethod
    def _box_coverage(
        box: tuple[float, float, float, float], container
    ) -> float:
        """Fraction of `box`'s area that lies inside `container`."""
        x, y, x2, y2 = box
        area = (x2 - x) * (y2 - y)
        if area <= 0:
            return 0.0
        overlap_x = min(x2, container.x2) - max(x, container.x)
        overlap_y = min(y2, container.y2) - max(y, container.y)
        if overlap_x <= 0 or overlap_y <= 0:
            return 0.0
        return (overlap_x * overlap_y) / area

    def _mirror_scope(
        self,
        page: il_version_1.Page,
        pivot: float,
        in_scope,
        wrapper_form_ids: set[int],
    ) -> None:
        # A page-level curve/form fully inside a paragraph box is part of that
        # paragraph's content (fraction bars, inline rules): it must follow
        # the paragraph's translation instead of being mirrored around its own
        # center, or it detaches from the glyphs it belongs to.
        paragraph_anchors: list[tuple[tuple[float, float, float, float], float]] = []
        for paragraph in page.pdf_paragraph:
            if not in_scope(paragraph.xobj_id):
                continue
            if not self._box_is_valid(paragraph.box):
                continue
            box = paragraph.box
            region = self._raster_region(paragraph)
            if region is not None:
                # An image-OCR label rides its raster image: the image
                # mirrors as a rigid translation (content never flips), so
                # the label gets exactly the image's dx, not its own.
                dx = self._graphic_dx(page, self._region_box(region), pivot)
                self._shift_raster_region(paragraph, dx)
            else:
                dx = pivot - box.x - box.x2
            paragraph_anchors.append(((box.x, box.y, box.x2, box.y2), dx))
            if abs(dx) < 1e-6:
                continue
            self._shift_box_x(paragraph.box, dx)
            if not self._paragraph_will_retypeset(paragraph):
                self._shift_paragraph_content(paragraph, dx)

        cluster_anchors = self._mirror_loose_characters(page, pivot, in_scope)
        # Paragraph boxes are the more precise anchors; cluster bboxes only
        # apply when no paragraph contains the element (smallest wins).
        paragraph_anchors.extend(cluster_anchors)
        paragraph_anchors.extend(
            self._layout_region_anchors(page, paragraph_anchors)
        )

        for curve in page.pdf_curve:
            if not in_scope(curve.xobj_id) or not self._box_is_valid(curve.box):
                continue
            dx = self._find_anchor_dx(paragraph_anchors, curve.box)
            if dx is None:
                dx = pivot - curve.box.x - curve.box.x2
            if abs(dx) < 1e-6:
                continue
            self._shift_box_x(curve.box, dx)
            self._compose_device_translation(curve, dx)

        for form in page.pdf_form:
            if id(form) in wrapper_form_ids:
                continue
            if not in_scope(form.xobj_id) or not self._box_is_valid(form.box):
                continue
            dx = self._find_anchor_dx(paragraph_anchors, form.box)
            if dx is None:
                dx = self._graphic_dx(page, form.box, pivot)
            if abs(dx) < 1e-6:
                continue
            self._shift_box_x(form.box, dx)
            self._compose_device_translation(form, dx)

        for rect in page.pdf_rectangle:
            if not in_scope(rect.xobj_id) or not self._box_is_valid(rect.box):
                continue
            region = self._raster_region(rect)
            if region is not None:
                # An image-OCR mask rides its raster image, like its label.
                dx = self._graphic_dx(page, self._region_box(region), pivot)
                self._shift_raster_region(rect, dx)
            else:
                dx = pivot - rect.box.x - rect.box.x2
            if abs(dx) >= 1e-6:
                self._shift_box_x(rect.box, dx)

        for figure in page.pdf_figure:
            if not in_scope(getattr(figure, "xobj_id", None)):
                continue
            if self._box_is_valid(figure.box):
                dx = self._graphic_dx(page, figure.box, pivot)
                if abs(dx) >= 1e-6:
                    self._shift_box_x(figure.box, dx)

    def render_page(self, page: il_version_1.Page):
        fonts: dict[
            str | int,
            il_version_1.PdfFont | dict[str, il_version_1.PdfFont],
        ] = {f.font_id: f for f in page.pdf_font if f.font_id}
        page_fonts = {f.font_id: f for f in page.pdf_font if f.font_id}
        for k, v in self.font_mapper.fontid2font.items():
            fonts[k] = v
        for xobj in page.pdf_xobject:
            if xobj.xobj_id is not None:
                fonts[xobj.xobj_id] = page_fonts.copy()
                for font in xobj.pdf_font:
                    if font.font_id:
                        fonts[xobj.xobj_id][font.font_id] = font
        if (
            page.page_number == 0
            and self.translation_config.watermark_output_mode
            == WatermarkOutputMode.Watermarked
        ):
            self.add_watermark(page)
        try:
            para_index = index.Index()
            para_map = {}
            #
            valid_paras = [
                p
                for p in page.pdf_paragraph
                if p.box
                and all(c is not None for c in [p.box.x, p.box.y, p.box.x2, p.box.y2])
            ]

            for i, para in enumerate(valid_paras):
                para_map[i] = para
                para_index.insert(i, box_to_tuple(para.box))

            for i, p_upper in para_map.items():
                if not (p_upper.box and p_upper.box.y is not None):
                    continue

                # Calculate paragraph height and set required gap accordingly
                para_height = p_upper.box.y2 - p_upper.box.y
                required_gap = 0.5 if para_height < 36 else 3

                check_area = il_version_1.Box(
                    x=p_upper.box.x,
                    y=p_upper.box.y - required_gap,
                    x2=p_upper.box.x2,
                    y2=p_upper.box.y,
                )

                candidate_ids = list(para_index.intersection(box_to_tuple(check_area)))

                conflicting_paras = []
                for para_id in candidate_ids:
                    if para_id == i:
                        continue
                    p_lower = para_map[para_id]
                    if not (
                        p_lower.box
                        and p_upper.box
                        and p_lower.box.x2 < p_upper.box.x
                        or p_lower.box.x > p_upper.box.x2
                    ):
                        conflicting_paras.append(p_lower)

                if conflicting_paras:
                    max_y2 = max(
                        p.box.y2
                        for p in conflicting_paras
                        if p.box and p.box.y2 is not None
                    )

                    new_y = max_y2 + required_gap
                    if p_upper.box and new_y < p_upper.box.y2:
                        p_upper.box.y = new_y
        except Exception as e:
            logger.warning(
                f"Failed to adjust paragraph positions on page {page.page_number}: {e}"
            )
        # 开始实际的渲染过程
        for paragraph in page.pdf_paragraph:
            self.render_paragraph(paragraph, page, fonts)

    def add_watermark(self, page: il_version_1.Page):
        page_width = page.cropbox.box.x2 - page.cropbox.box.x
        page_height = page.cropbox.box.y2 - page.cropbox.box.y
        style = il_version_1.PdfStyle(
            font_id="base",
            font_size=6,
            graphic_state=il_version_1.GraphicState(),
        )
        text = f"本文档由 funstory.ai 的开源 PDF 翻译库 BabelDOC {WATERMARK_VERSION} (https://github.com/funstory-ai/BabelDOC) 翻译，本仓库正在积极的建设当中，欢迎 star 和关注。"
        if self.translation_config.debug:
            text += "\n 当前为 DEBUG 模式，将显示更多辅助信息。请注意，部分框的位置对应原文，但在译文中可能不正确。"
        page.pdf_paragraph.append(
            il_version_1.PdfParagraph(
                first_line_indent=False,
                box=il_version_1.Box(
                    x=page.cropbox.box.x + page_width * 0.05,
                    y=page.cropbox.box.y,
                    x2=page.cropbox.box.x2,
                    y2=page.cropbox.box.y2 - page_height * 0.05,
                ),
                vertical=False,
                pdf_style=style,
                pdf_paragraph_composition=[
                    il_version_1.PdfParagraphComposition(
                        pdf_same_style_unicode_characters=il_version_1.PdfSameStyleUnicodeCharacters(
                            unicode=text,
                            pdf_style=style,
                        ),
                    ),
                ],
                xobj_id=-1,
            ),
        )

    def _prefit_raster_label_box(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        typesetting_units: list[TypesettingUnit],
    ) -> None:
        """Widen a raster label's box around its center to fit one line.

        An image-OCR label box is glyph-tight around a SHORT source word
        ("Object"), so a longer translation would wrap letter-by-letter down
        the artwork. The label sits centered on its shape, so growth is
        symmetric about the center — clamped to the containing image region
        and stopping short of the other labels on the same raster
        (Contract 2). Idempotent: a box that is already wide enough is left
        alone.
        """
        region = self._raster_region(paragraph)
        box = paragraph.box
        if region is None or box is None:
            return
        natural_width = sum(unit.width for unit in typesetting_units)
        needed = natural_width * 1.05 + 2.0
        # The label usually sits on a shape (a hexagon, an ellipse) the IL
        # cannot see: growing to the translation's full natural width would
        # walk the text off the shape onto whatever is behind it. Grow at
        # most this factor past the source label and let the normal
        # shrink-to-fit scale handle the rest.
        needed = min(needed, (box.x2 - box.x) * self.RASTER_LABEL_MAX_GROWTH)
        if needed <= box.x2 - box.x:
            return
        center = (box.x + box.x2) / 2
        x1 = max(center - needed / 2, float(region[0]) + 1.0)
        x2 = min(center + needed / 2, float(region[2]) - 1.0)
        for other in page.pdf_paragraph:
            if other is paragraph or self._raster_region(other) is None:
                continue
            other_box = other.box
            if not self._box_is_valid(other_box):
                continue
            if other_box.y >= box.y2 or other_box.y2 <= box.y:
                continue  # no vertical overlap: not in the way
            if other_box.x2 <= box.x:
                x1 = max(x1, other_box.x2 + 1.0)
            if other_box.x >= box.x2:
                x2 = min(x2, other_box.x - 1.0)
        if x2 - x1 > box.x2 - box.x:
            box.x, box.x2 = x1, x2

    def render_paragraph(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        fonts: dict[
            str | int,
            il_version_1.PdfFont | dict[str, il_version_1.PdfFont],
        ],
    ):
        typesetting_units = self.create_typesetting_units(paragraph, fonts)
        # 如果所有单元都可以直接传递，则直接传递
        if all(unit.can_passthrough for unit in typesetting_units):
            paragraph.scale = 1.0
            paragraph.pdf_paragraph_composition = self.create_passthrough_composition(
                typesetting_units,
            )
        else:
            self._prefit_raster_label_box(paragraph, page, typesetting_units)
            # 使用预计算的缩放因子进行重排版
            precomputed_scale = (
                paragraph.optimal_scale if paragraph.optimal_scale is not None else 1.0
            )

            # 如果有单元无法直接传递，则进行重排版
            paragraph.pdf_paragraph_composition = []
            self.retypeset_with_precomputed_scale(
                paragraph, page, typesetting_units, precomputed_scale
            )

            # 重排版后，重新设置段落各字符的 render order
            self._update_paragraph_render_order(paragraph)

    def _get_width_before_next_break_point(
        self, typesetting_units: list[TypesettingUnit], scale: float
    ) -> float:
        if not typesetting_units:
            return 0
        if typesetting_units[0].can_break_line:
            return 0

        total_width = 0
        for unit in typesetting_units:
            if unit.can_break_line:
                return total_width * scale
            total_width += unit.width
        return total_width * scale

    def _layout_typesetting_units(
        self,
        typesetting_units: list[TypesettingUnit],
        box: Box,
        scale: float,
        line_skip: float,
        paragraph: il_version_1.PdfParagraph,
        use_english_line_break: bool = True,
        rtl_align: str = "right",
    ) -> tuple[list[TypesettingUnit], bool]:
        """布局排版单元。

        Args:
            typesetting_units: 要布局的排版单元列表
            box: 布局边界框
            scale: 缩放因子

        Returns:
            tuple[list[TypesettingUnit], bool]: (已布局的排版单元列表，是否所有单元都放得下)
        """
        # 计算字号众数
        font_sizes = []
        for unit in typesetting_units:
            if unit.font_size:
                font_sizes.append(unit.font_size)
            if unit.char and unit.char.pdf_style and unit.char.pdf_style.font_size:
                font_sizes.append(unit.char.pdf_style.font_size)
        font_sizes.sort()
        font_size = statistics.mode(font_sizes)

        space_width = (
            self.font_mapper.base_font.char_lengths("你", font_size * scale)[0] * 0.5
        )

        # 计算行高（使用众数）
        unit_heights = (
            [unit.height for unit in typesetting_units] if typesetting_units else []
        )
        if not unit_heights:
            avg_height = 0
        elif len(unit_heights) == 1:
            avg_height = unit_heights[0] * scale
        else:
            try:
                avg_height = statistics.mode(unit_heights) * scale
            except statistics.StatisticsError:
                # 如果没有众数（所有值都出现相同次数），则使用平均值
                avg_height = sum(unit_heights) / len(unit_heights) * scale

        # 初始化位置为右上角，并减去一个平均行高
        current_x = box.x
        current_y = box.y2 - avg_height
        box = copy.deepcopy(box)
        # box.y -= avg_height * (line_spacing - 1.01) # line_spacing 已被替换为 line_skip
        line_height = 0
        current_line_heights = []  # 存储当前行所有元素的高度

        # 存储已排版的单元
        typeset_units = []
        all_units_fit = True
        last_unit: TypesettingUnit | None = None
        line_ys = [current_y]
        # RTL: lines are filled left-to-right in logical order, then each
        # finished line is mirrored into visual (right-to-left) order.
        rtl_layout = self.is_rtl and any(
            unit.bidi_class == "R" for unit in typesetting_units
        )
        line_start_index = 0
        if paragraph.first_line_indent:
            current_x += space_width * 4
        # 遍历所有排版单元
        for i, unit in enumerate(typesetting_units):
            # 计算当前单元在当前缩放下的尺寸
            unit_width = unit.width * scale
            unit_height = unit.height * scale

            # 跳过行首的空格
            if current_x == box.x and unit.is_space:
                continue

            if (
                last_unit  # 有上一个单元
                and last_unit.is_cjk_char ^ unit.is_cjk_char  # 中英文交界处
                and (
                    last_unit.box
                    and last_unit.box.y
                    and current_y - 0.1
                    <= last_unit.box.y2
                    <= current_y + line_height + 0.1
                )  # 在同一行，且有垂直重叠
                and not last_unit.mixed_character_blacklist  # 不是混排空格黑名单字符
                and not unit.mixed_character_blacklist  # 同上
                and current_x > box.x  # 不是行首
                and unit.try_get_unicode() != " "  # 不是空格
                and last_unit.try_get_unicode() != " "  # 不是空格
                and last_unit.try_get_unicode()
                not in [
                    "。",
                    "！",
                    "？",
                    "；",
                    "：",
                    "，",
                ]
            ):
                current_x += space_width * 0.5
            if use_english_line_break:
                width_before_next_break_point = self._get_width_before_next_break_point(
                    typesetting_units[i:], scale
                )
            else:
                width_before_next_break_point = 0

            # 如果当前行放不下这个元素，换行
            if not unit.is_hung_punctuation and (
                (current_x + unit_width > box.x2)
                or (
                    # width_before_next_break_point already includes the
                    # current unit's width, so it must not be added again:
                    # double counting made words wrap mid-word whenever the
                    # remainder of the word only just fit on the line.
                    use_english_line_break
                    and width_before_next_break_point > 0
                    and current_x + width_before_next_break_point > box.x2
                )
                or (
                    unit.is_cannot_appear_in_line_end_punctuation
                    and current_x + unit_width * 2 > box.x2
                )
            ):
                # 换行
                if rtl_layout:
                    self._finalize_rtl_line(
                        typeset_units, line_start_index, box, rtl_align
                    )
                line_start_index = len(typeset_units)
                current_x = box.x
                if not current_line_heights:
                    return [], False
                max_height = max(current_line_heights)
                mode_height = statistics.mode(current_line_heights)

                # Line advance must not collapse below the dominant em height.
                # current_line_heights are per-glyph bounding-box heights, which
                # for an all-Latin line (e.g. a citation run) are far shorter than
                # the CJK em, shrinking the advance to ~half a line and letting the
                # next CJK line overlap it. Floor the advance with font_size (the
                # paragraph's dominant em) so script-mix never undersizes the gap.
                current_y -= max(
                    font_size * scale * line_skip,
                    mode_height * line_skip,
                    max_height * 1.05,
                )
                line_ys.append(current_y)
                line_height = 0.0
                current_line_heights = []  # 清空当前行高度列表

                # 检查是否超出底部边界
                # if current_y - unit_height < box.y:
                if current_y < box.y:
                    all_units_fit = False
                    # 这里不要 break，继续排版剩余内容

                if unit.is_space:
                    line_height = max(line_height, unit_height)
                    continue

            # 放置当前单元
            relocated_unit = unit.relocate(current_x, current_y, scale)
            typeset_units.append(relocated_unit)

            # 添加当前单元的高度到当前行高度列表
            if not unit.is_space:
                current_line_heights.append(unit_height)

            prev_x = current_x
            # 更新 x 坐标
            current_x = relocated_unit.box.x2
            if prev_x > current_x:
                logger.warning(f"坐标回绕！！！TypesettingUnit: {unit.box}, ")

            last_unit = relocated_unit

        if rtl_layout:
            self._finalize_rtl_line(typeset_units, line_start_index, box, rtl_align)

        return typeset_units, all_units_fit

    def _finalize_rtl_line(
        self,
        typeset_units: list[TypesettingUnit],
        line_start_index: int,
        box: Box,
        align: str = "right",
    ) -> None:
        """Mirror one finished line into right-to-left visual order.

        The line was filled left-to-right in logical order. Mirroring every
        unit around the paragraph box converts it to RTL visual order and
        right-aligns it against the box. Embedded LTR runs (Latin letters,
        digits, formulas) are then reversed back so they keep their internal
        left-to-right order, and mirrorable punctuation participating in the
        RTL flow is swapped (e.g. "(" renders as ")").
        """
        line_units = typeset_units[line_start_index:]
        if not line_units:
            return
        pivot = box.x + box.x2

        # Build bidi elements. Most units map 1:1, but a formula without
        # curves/forms is split into per-character elements: styled runs such
        # as "80-20) →", "x =" or "> 10" arrive as single formula blocks, and
        # only per-character resolution lets their edge neutrals join the
        # surrounding RTL flow while the strong LTR core (digits/latin) keeps
        # its internal order.
        #
        # The line mirror MUST be applied at the same (element) granularity as
        # the later LTR-run restore: mirror then restore cancel into a rigid
        # translation for LTR runs, which is only true when both act on the
        # same pieces. (Mirroring whole formula blocks rigidly and then
        # restoring per character would reverse formula-internal layout.)
        elements = self._build_rtl_line_elements(line_units)
        for element in elements:
            element.shift(pivot - element.x - element.x2)

        classes = [element.cls for element in elements]
        resolved = self._resolve_rtl_neutrals(elements, classes)

        count = len(elements)
        index_ = 0
        while index_ < count:
            if resolved[index_] != "L":
                elements[index_].apply_bidi_mirror()
                index_ += 1
                continue
            run_end = index_
            while run_end < count and resolved[run_end] == "L":
                run_end += 1
            run = elements[index_:run_end]
            run_left = min(element.x for element in run)
            run_right = max(element.x2 for element in run)
            for element in run:
                element.shift(run_left + run_right - element.x - element.x2)
            index_ = run_end

        # Anchored OCR paragraphs: the mirror right-aligned the line against
        # box.x2; shift it back so the translated text hugs the edge the
        # original was anchored to.
        if align != "right" and elements:
            dx = box.x - min(element.x for element in elements)
            if align == "center":
                dx /= 2
            if abs(dx) > 0.01:
                for element in elements:
                    element.shift(dx)

        # Formula boxes may be stale after per-character shifts; refresh them
        # so debug rendering / later consumers see the true extent.
        for unit in line_units:
            if unit.formular and unit.formular.pdf_character:
                update_formula_data(unit.formular)
            unit.box_cache = None

    def _build_rtl_line_elements(
        self,
        line_units: list[TypesettingUnit],
    ) -> list["_RtlBidiElement"]:
        elements: list[_RtlBidiElement] = []
        for unit in line_units:
            formular = unit.formular
            splittable = (
                formular is not None
                and formular.pdf_character
                and not formular.pdf_curve
                and not formular.pdf_form
            )
            if not splittable:
                elements.append(_RtlBidiElement.from_unit(unit))
                continue
            for char in formular.pdf_character:
                elements.append(_RtlBidiElement.from_formula_char(char))
        return elements

    def _resolve_rtl_neutrals(
        self,
        elements: list["_RtlBidiElement"],
        classes: list[str],
    ) -> list[str]:
        """Resolve neutral elements against their strong neighbours.

        Rules (simplified UAX#9, paragraph direction RTL):
        - a neutral between two strong runs of the same direction takes that
          direction;
        - a neutral operator (=, >, <, +, -, ...) directly preceding a digit
          run stays with it, so math snippets like "> 10" read left-to-right
          as one cluster;
        - a bracket group whose opening bracket directly follows a strong-LTR
          character is claimed by that LTR run ("getArea()", "list[0]"), so
          edge brackets never migrate to the RTL side of the identifier;
        - any other neutral (RTL/LTR boundaries, line edges) takes the
          paragraph direction (RTL).
        """
        count = len(classes)
        resolved = list(classes)
        # Pre-pass: bracket groups attached to a Latin identifier / digit run.
        index_ = 0
        while index_ < count:
            open_ch = elements[index_].text
            close_ch = LTR_ATTACHED_BRACKET_PAIRS.get(open_ch)
            if close_ch is None or index_ == 0 or resolved[index_ - 1] != "L":
                index_ += 1
                continue
            depth = 0
            match_ = index_
            while match_ < count:
                text = elements[match_].text
                if text == open_ch:
                    depth += 1
                elif text == close_ch:
                    depth -= 1
                    if depth == 0:
                        break
                match_ += 1
            if match_ < count and all(
                classes[k] != "R" for k in range(index_ + 1, match_)
            ):
                # The whole group (brackets and interior neutrals) joins the
                # identifier's LTR run; interior strong-L stays L anyway.
                for k in range(index_, match_ + 1):
                    if resolved[k] == "N":
                        resolved[k] = "L"
                index_ = match_ + 1
            else:
                index_ += 1
        for i, cls in enumerate(classes):
            if cls != "N":
                continue
            if resolved[i] != "N":
                # Already claimed by an operator+digit cluster.
                continue
            prev_strong = next(
                (classes[j] for j in range(i - 1, -1, -1) if classes[j] != "N"),
                "R",
            )
            next_index = next(
                (j for j in range(i + 1, count) if classes[j] != "N"),
                None,
            )
            next_strong = classes[next_index] if next_index is not None else "R"
            if prev_strong == next_strong:
                resolved[i] = prev_strong
                continue
            if (
                next_strong == "L"
                and next_index is not None
                and elements[i].is_neutral_operator
                and elements[next_index].starts_with_digit
                and all(
                    elements[j].is_space_like for j in range(i + 1, next_index)
                )
            ):
                # Operator + digit cluster: "> 10", "= 5", "±3" ...
                for j in range(i, next_index):
                    resolved[j] = "L"
                continue
            resolved[i] = "R"
        return resolved

    def create_typesetting_units(
        self,
        paragraph: il_version_1.PdfParagraph,
        fonts: dict[str, il_version_1.PdfFont],
    ) -> list[TypesettingUnit]:
        if not paragraph.pdf_paragraph_composition:
            return []
        result = []

        @cache
        def get_font(font_id: str, xobj_id: int | None):
            if xobj_id in fonts:
                font = fonts[xobj_id][font_id]
            else:
                font = fonts[font_id]
            return font

        for composition in paragraph.pdf_paragraph_composition:
            if composition is None:
                continue
            if composition.pdf_line:
                result.extend(
                    [
                        TypesettingUnit(char=char)
                        for char in composition.pdf_line.pdf_character
                    ],
                )
            elif composition.pdf_character:
                result.append(
                    TypesettingUnit(
                        char=composition.pdf_character,
                        debug_info=paragraph.debug_info,
                    ),
                )
            elif composition.pdf_same_style_characters:
                result.extend(
                    [
                        TypesettingUnit(char=char)
                        for char in composition.pdf_same_style_characters.pdf_character
                    ],
                )
            elif composition.pdf_same_style_unicode_characters:
                style = composition.pdf_same_style_unicode_characters.pdf_style
                if style is None:
                    logger.warning(
                        f"Style is None. "
                        f"Composition: {composition}. "
                        f"Paragraph: {paragraph}. ",
                    )
                    continue
                font_id = style.font_id
                if font_id is None:
                    logger.warning(
                        f"Font ID is None. "
                        f"Composition: {composition}. "
                        f"Paragraph: {paragraph}. ",
                    )
                    continue
                font = get_font(font_id, paragraph.xobj_id)
                unicode_text = composition.pdf_same_style_unicode_characters.unicode
                if unicode_text and self.is_rtl:
                    # Bidi control characters would render as visible-width
                    # glyphs (spurious gaps) and our explicit bidi algorithm
                    # ignores them anyway; drop them and collapse the space
                    # runs they leave behind.
                    unicode_text = BIDI_CONTROL_REGEX.sub("", unicode_text)
                    unicode_text = MULTI_SPACE_REGEX.sub(" ", unicode_text)
                    # Substitute Arabic contextual (presentation) forms so the
                    # per-character renderer produces joined glyphs.
                    unicode_text = reshape_rtl_text(unicode_text)
                if unicode_text:
                    result.extend(
                        [
                            TypesettingUnit(
                                unicode=char_unicode,
                                font=self.font_mapper.map(
                                    font,
                                    char_unicode,
                                ),
                                original_font=font,
                                font_size=style.font_size,
                                style=style,
                                xobj_id=paragraph.xobj_id,
                                debug_info=composition.pdf_same_style_unicode_characters.debug_info
                                or False,
                            )
                            for char_unicode in unicode_text
                            if char_unicode not in ("\n",)
                        ],
                    )
            elif composition.pdf_formula:
                result.extend([TypesettingUnit(formular=composition.pdf_formula)])
            else:
                logger.error(
                    f"Unknown composition type. "
                    f"Composition: {composition}. "
                    f"Paragraph: {paragraph}. ",
                )
                continue
        result = list(
            filter(
                lambda x: x.unicode is None or x.font is not None,
                result,
            ),
        )

        if any(x.width < 0 for x in result):
            logger.warning("有排版单元宽度小于 0，请检查字体映射是否正确。")
        return result

    def create_passthrough_composition(
        self,
        typesetting_units: list[TypesettingUnit],
    ) -> list[PdfParagraphComposition]:
        """从排版单元创建直接传递的段落组合。

        Args:
            typesetting_units: 排版单元列表

        Returns:
            段落组合列表
        """
        composition = []
        for unit in typesetting_units:
            if unit.formular:
                # 对于公式单元，直接创建包含完整公式的组合
                composition.append(PdfParagraphComposition(pdf_formula=unit.formular))
            else:
                # 对于字符单元，使用原有逻辑
                chars, curves, forms = unit.passthrough()
                composition.extend(
                    [PdfParagraphComposition(pdf_character=char) for char in chars],
                )
        return composition

    def get_max_right_space(self, current_box: Box, page) -> float:
        """获取段落右侧最大可用空间

        Args:
            current_box: 当前段落的边界框
            page: 当前页面

        Returns:
            可以扩展到的最大 x 坐标
        """
        # 获取页面的裁剪框作为初始最大限制
        max_x = page.cropbox.box.x2 * 0.9

        # 检查所有可能的阻挡元素
        for para in page.pdf_paragraph:
            if para.box == current_box or para.box is None:  # 跳过当前段落
                continue
            # 只考虑在当前段落右侧且有垂直重叠的元素
            if para.box.x > current_box.x and not (
                para.box.y >= current_box.y2 or para.box.y2 <= current_box.y
            ):
                max_x = min(max_x, para.box.x)
        for char in page.pdf_character:
            if char.box.x > current_box.x and not (
                char.box.y >= current_box.y2 or char.box.y2 <= current_box.y
            ):
                max_x = min(max_x, char.box.x)
        # 检查图形
        for figure in page.pdf_figure:
            if figure.box.x > current_box.x and not (
                figure.box.y >= current_box.y2 or figure.box.y2 <= current_box.y
            ):
                max_x = min(max_x, figure.box.x)

        # A paragraph drawn inside a vector shape (chip/pill/card background)
        # must not expand across the shape's border: its text would cross the
        # visible outline and collide with neighbouring shapes.
        for elem_box in self._containing_graphic_boxes(current_box, page):
            max_x = min(max_x, elem_box.x2 - 1)

        return max_x

    def _containing_graphic_boxes(self, current_box: Box, page: il_version_1.Page):
        """Yield boxes of page-level curves/forms that contain current_box."""
        for elems in (page.pdf_curve, page.pdf_form):
            for elem in elems:
                box = elem.box
                if not self._box_is_valid(box):
                    continue
                if (
                    box.x - 1 <= current_box.x
                    and box.x2 + 1 >= current_box.x2
                    and box.y - 1 <= current_box.y
                    and box.y2 + 1 >= current_box.y2
                    and (box.x2 - box.x) * (box.y2 - box.y)
                    > (current_box.x2 - current_box.x)
                    * (current_box.y2 - current_box.y)
                ):
                    yield box

    def get_max_bottom_space(self, current_box: Box, page: il_version_1.Page) -> float:
        """获取段落下方最大可用空间

        Args:
            current_box: 当前段落的边界框
            page: 当前页面

        Returns:
            可以扩展到的最小 y 坐标
        """
        # 获取页面的裁剪框作为初始最小限制
        min_y = page.cropbox.box.y * 1.1

        # 检查所有可能的阻挡元素
        for para in page.pdf_paragraph:
            if para.box == current_box or para.box is None:  # 跳过当前段落
                continue
            # 只考虑在当前段落下方且有水平重叠的元素
            if para.box.y2 < current_box.y and not (
                para.box.x >= current_box.x2 or para.box.x2 <= current_box.x
            ):
                min_y = max(min_y, para.box.y2)
        for char in page.pdf_character:
            if char.box.y2 < current_box.y and not (
                char.box.x >= current_box.x2 or char.box.x2 <= current_box.x
            ):
                min_y = max(min_y, char.box.y2)
        # 检查图形
        for figure in page.pdf_figure:
            if figure.box.y2 < current_box.y and not (
                figure.box.x >= current_box.x2 or figure.box.x2 <= current_box.x
            ):
                min_y = max(min_y, figure.box.y2)

        # Stay inside any vector shape that visually contains this paragraph
        # (see get_max_right_space).
        for elem_box in self._containing_graphic_boxes(current_box, page):
            min_y = max(min_y, elem_box.y + 1)

        return min_y

    def _update_paragraph_render_order(self, paragraph: il_version_1.PdfParagraph):
        """
        重新设置段落各字符的 render order
        主 render order 等于 paragraph 的 renderorder，sub render order 从 1 开始自增
        """
        if not hasattr(paragraph, "render_order") or paragraph.render_order is None:
            return

        main_render_order = paragraph.render_order
        sub_render_order = 1

        # 遍历段落的所有组成部分
        for composition in paragraph.pdf_paragraph_composition:
            # 检查单个字符
            if composition.pdf_character:
                char = composition.pdf_character
                char.render_order = main_render_order
                char.sub_render_order = sub_render_order
                sub_render_order += 1
