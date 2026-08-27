import logging
import random
import re
from pathlib import Path as FsPath

import numpy as np
import pymupdf

from babeldoc.babeldoc_exception.BabelDOCException import ExtractTextError
from babeldoc.format.pdf.document_il import Box
from babeldoc.format.pdf.document_il import Document
from babeldoc.format.pdf.document_il import GraphicState
from babeldoc.format.pdf.document_il import Page
from babeldoc.format.pdf.document_il import PdfCharacter
from babeldoc.format.pdf.document_il import PdfLine
from babeldoc.format.pdf.document_il import PdfParagraph
from babeldoc.format.pdf.document_il import PdfParagraphComposition
from babeldoc.format.pdf.document_il import PdfRectangle
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.formular_helper import (
    collect_page_formula_font_ids,
)
from babeldoc.format.pdf.document_il.utils.layout_helper import (
    HEIGHT_NOT_USFUL_CHAR_IN_CHAR,
)
from babeldoc.format.pdf.document_il.utils.layout_helper import SPACE_REGEX
from babeldoc.format.pdf.document_il.utils.layout_helper import Layout
from babeldoc.format.pdf.document_il.utils.layout_helper import add_space_dummy_chars
from babeldoc.format.pdf.document_il.utils.layout_helper import build_layout_index
from babeldoc.format.pdf.document_il.utils.layout_helper import calculate_iou_for_boxes
from babeldoc.format.pdf.document_il.utils.layout_helper import get_char_unicode_string
from babeldoc.format.pdf.document_il.utils.layout_helper import get_character_layout
from babeldoc.format.pdf.document_il.utils.layout_helper import BULLET_POINT_PATTERN
from babeldoc.format.pdf.document_il.utils.layout_helper import is_bullet_point
from babeldoc.format.pdf.document_il.utils.layout_helper import (
    is_character_in_formula_layout,
)
from babeldoc.format.pdf.document_il.utils.layout_helper import is_text_layout
from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
    build_code_font_ids,
)
from babeldoc.format.pdf.document_il.utils.paragraph_helper import is_cid_paragraph
from babeldoc.format.pdf.document_il.utils.style_helper import BLACK
from babeldoc.format.pdf.document_il.utils.style_helper import INDIGO
from babeldoc.format.pdf.document_il.utils.style_helper import WHITE
from babeldoc.format.pdf.translation_config import TranslationConfig

logger = logging.getLogger(__name__)

# Base58 alphabet (Bitcoin style, without numbers 0, O, I, l)
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# Matches a numbered list item prefix at the start of a line, e.g. "1. ", "12) ".
# The negative lookahead avoids matching decimals like "3.14".
NUMBERED_ITEM_PATTERN = re.compile(r"\s*(\d{1,2})[.)](?!\d)")


def generate_base58_id(length: int = 5) -> str:
    """Generate a random base58 ID of specified length."""
    return "".join(random.choice(BASE58_ALPHABET) for _ in range(length))


class MaskColorSampler:
    """Sample OCR-workaround mask colors from the original page raster.

    Plain white masks leave ugly white boxes on scanned slides where the
    text sits on a colored fill (teal title cards, banners). Instead, fill
    each mask with the dominant color of a thin ring just outside the mask
    box: solid backgrounds match seamlessly, and white paper still yields
    the paper's own near-white. If the ring disagrees with itself
    (gradient / photo / box poking out of its card), fall back to the
    dominant color INSIDE the box — for a text mask that is the local
    background too, glyphs being the minority. White is never forced.

    Also samples the dominant glyph ("ink") color inside the box so the
    translated text can keep the original color (gold on teal) instead of
    the flat OCR black. On near-white backgrounds only a clearly SATURATED
    ink retints (teal/gold headings); anti-aliased grays stay black.
    """

    ZOOM = 1.5           # 108-dpi render is plenty for flat fills
    RING_GAP_PT = 1.0    # gap between box edge and ring (skip AA halo)
    RING_WIDTH_PT = 2.5  # sampled ring thickness
    RING_SOLID_FRAC = 0.6  # ring agreement below this => use inner fallback
    INK_DIST = 60        # channel distance from bg to count a pixel as ink
    INK_MIN_FRAC = 0.02  # min ink pixel fraction to trust an ink color
    INK_SAT_MIN = 40     # channel spread for an ink color to count as colored
    NEAR_WHITE = 240     # near-white bg: only saturated ink retints the text

    def __init__(self, pdf_path: str):
        self.pdf = pymupdf.open(pdf_path)
        self._cache: tuple | None = None  # (page_number, img, matrix)

    def _page_raster(self, page_number: int):
        if self._cache and self._cache[0] == page_number:
            return self._cache[1], self._cache[2]
        page = self.pdf[page_number]
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(self.ZOOM, self.ZOOM),
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )[:, :, :3]
        matrix = page.transformation_matrix * pymupdf.Matrix(self.ZOOM, self.ZOOM)
        self._cache = (page_number, img, matrix)
        return img, matrix

    @staticmethod
    def _dominant(pixels: np.ndarray):
        """Modal 32-level color bucket as (fraction, mean rgb 0-255)."""
        if pixels.size == 0:
            return 0.0, None
        q = (pixels // 32).astype(np.int32)
        keys = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
        counts = np.bincount(keys, minlength=512)
        key = int(counts.argmax())
        rgb = pixels[keys == key].mean(axis=0)
        return counts[key] / len(pixels), tuple(float(c) for c in rgb)

    def sample(self, page_number: int, x1: float, y1: float, x2: float, y2: float):
        """(bg_rgb, ink_rgb | None) for a mask box in PDF user space."""
        img, matrix = self._page_raster(page_number)
        h, w = img.shape[:2]
        p1 = pymupdf.Point(x1, y1) * matrix
        p2 = pymupdf.Point(x2, y2) * matrix
        px1, px2 = sorted((p1.x, p2.x))
        py1, py2 = sorted((p1.y, p2.y))
        gap = self.RING_GAP_PT * self.ZOOM
        ring = (self.RING_GAP_PT + self.RING_WIDTH_PT) * self.ZOOM
        ix1, iy1 = int(px1 - gap), int(py1 - gap)
        ix2, iy2 = int(px2 + gap) + 1, int(py2 + gap) + 1
        ox1, oy1 = max(0, int(px1 - ring)), max(0, int(py1 - ring))
        ox2, oy2 = min(w, int(px2 + ring) + 1), min(h, int(py2 + ring) + 1)
        inner = img[
            max(0, int(py1)) : min(h, int(py2) + 1),
            max(0, int(px1)) : min(w, int(px2) + 1),
        ].reshape(-1, 3)
        if ox2 <= ox1 or oy2 <= oy1:
            return None, None
        bands = [
            img[oy1 : max(oy1, min(iy1, oy2)), ox1:ox2],  # above
            img[min(max(iy2, oy1), oy2) : oy2, ox1:ox2],  # below
            img[oy1:oy2, ox1 : max(ox1, min(ix1, ox2))],  # left
            img[oy1:oy2, min(max(ix2, ox1), ox2) : ox2],  # right
        ]
        ring_px = np.concatenate([b.reshape(-1, 3) for b in bands])
        frac, bg = self._dominant(ring_px)
        if bg is None or frac < self.RING_SOLID_FRAC:
            inner_frac, inner_bg = self._dominant(inner)
            if inner_bg is not None and (bg is None or inner_frac > frac):
                bg = inner_bg
        if bg is None:
            return None, None
        ink = None
        if len(inner):
            dist = (
                np.abs(inner.astype(np.int16) - np.array(bg, dtype=np.int16))
                .max(axis=1)
            )
            ink_px = inner[dist > self.INK_DIST]
            if len(ink_px) >= max(16, self.INK_MIN_FRAC * len(inner)):
                _, candidate = self._dominant(ink_px)
                if candidate is not None:
                    saturated = (
                        max(candidate) - min(candidate) >= self.INK_SAT_MIN
                    )
                    if saturated or min(bg) < self.NEAR_WHITE:
                        ink = candidate
        return bg, ink


def solid_graphic_state(rgb) -> GraphicState:
    """Fill+stroke GraphicState for an 0-255 rgb triple."""
    r, g, b = (max(0.0, min(1.0, c / 255.0)) for c in rgb)
    return GraphicState(
        passthrough_per_char_instruction=(
            f"{r:.4f} {g:.4f} {b:.4f} rg {r:.4f} {g:.4f} {b:.4f} RG"
        ),
    )


class ParagraphFinder:
    stage_name = "Parse Paragraphs"

    # 定义项目符号的正则表达式模式

    def __init__(self, translation_config: TranslationConfig):
        self.translation_config = translation_config
        self.font_mapper = FontMapper(translation_config)

    def _preprocess_formula_layouts(self, page: Page):
        """
        Identifies 'formula' layouts that do not significantly overlap with any text layouts
        and re-labels them as 'isolate_formula'.
        """
        # Use a simplified Layout object for is_text_layout check
        text_layouts = [
            layout
            for layout in page.page_layout
            if is_text_layout(Layout(layout.id, layout.class_name))
        ]
        formula_layouts = [
            layout for layout in page.page_layout if layout.class_name == "formula"
        ]

        if not text_layouts or not formula_layouts:
            return

        for formula_layout in formula_layouts:
            is_isolated = True
            for text_layout in text_layouts:
                iou = calculate_iou_for_boxes(formula_layout.box, text_layout.box)
                if iou >= 0.5:
                    is_isolated = False
                    break

            if is_isolated:
                formula_layout.class_name = "isolate_formula"

    def _get_mask_color_sampler(self) -> MaskColorSampler | None:
        """Lazily open the original input PDF for mask-color sampling.

        Only used for OCR-workaround masks; any failure falls back to the
        historical plain-white masks.
        """
        if not hasattr(self, "_mask_color_sampler"):
            self._mask_color_sampler = None
            try:
                path = self.translation_config.get_working_file_path("input.pdf")
                if FsPath(path).exists():
                    self._mask_color_sampler = MaskColorSampler(str(path))
            except Exception:
                logger.exception(
                    "Failed to open original PDF for mask-color sampling; "
                    "OCR-workaround masks fall back to white."
                )
        return self._mask_color_sampler

    @staticmethod
    def _tint_paragraph_text(paragraph: PdfParagraph, ink_rgb) -> None:
        """Give the paragraph's (future translated) text the sampled glyph
        color instead of the flat OCR-workaround black."""
        state = solid_graphic_state(ink_rgb)
        if paragraph.pdf_style is not None:
            paragraph.pdf_style.graphic_state = state
        for composition in paragraph.pdf_paragraph_composition or []:
            chars = []
            if composition.pdf_line:
                chars = composition.pdf_line.pdf_character
            elif composition.pdf_formula:
                chars = composition.pdf_formula.pdf_character
            elif composition.pdf_character:
                chars = [composition.pdf_character]
            elif composition.pdf_same_style_characters:
                chars = composition.pdf_same_style_characters.pdf_character
            for char in chars:
                if char.pdf_style is not None:
                    char.pdf_style.graphic_state = state

    def add_text_fill_background(self, page: Page):
        layout_map = {layout.id: layout for layout in page.page_layout}
        sampler = self._get_mask_color_sampler()
        for paragraph in page.pdf_paragraph:
            layout_id = paragraph.layout_id
            if layout_id is None:
                continue
            layout = layout_map[layout_id]
            if paragraph.box is None:
                continue
            x1, y1, x2, y2 = (
                paragraph.box.x,
                paragraph.box.y,
                paragraph.box.x2,
                paragraph.box.y2,
            )
            layout_box = layout.box
            if layout_box.x < x1:
                x1 = layout_box.x
            if layout_box.y < y1:
                y1 = layout_box.y
            if layout_box.x2 > x2:
                x2 = layout_box.x2
            if layout_box.y2 > y2:
                y2 = layout_box.y2
            assert x2 > x1 and y2 > y1
            # OCR-workaround masks used to be plain white, which leaves white
            # boxes on colored slide backgrounds. Sample the surrounding
            # background color from the original raster instead; the sampled
            # color of white paper stays (near-)white on its own.
            fill_state = WHITE
            if sampler is not None:
                try:
                    bg, ink = sampler.sample(page.page_number, x1, y1, x2, y2)
                    if bg is not None:
                        fill_state = solid_graphic_state(bg)
                    if ink is not None:
                        self._tint_paragraph_text(paragraph, ink)
                except Exception:
                    logger.exception(
                        "Mask-color sampling failed for paragraph "
                        f"{getattr(paragraph, 'debug_id', None)} on page "
                        f"{page.page_number}; using a white mask."
                    )
            page.pdf_rectangle.append(
                PdfRectangle(
                    box=Box(x1, y1, x2, y2),
                    fill_background=True,
                    graphic_state=fill_state,
                    debug_info=False,
                    xobj_id=paragraph.xobj_id,
                )
            )

    def update_paragraph_data(self, paragraph: PdfParagraph, update_unicode=False):
        if not paragraph.pdf_paragraph_composition:
            return

        chars = []
        for composition in paragraph.pdf_paragraph_composition:
            if composition.pdf_line:
                chars.extend(composition.pdf_line.pdf_character)
            elif composition.pdf_formula:
                chars.extend(composition.pdf_formula.pdf_character)
            elif composition.pdf_character:
                chars.append(composition.pdf_character)
            elif composition.pdf_same_style_unicode_characters:
                continue
            else:
                logger.error(
                    "Unexpected composition type"
                    " in PdfParagraphComposition. "
                    "This type only appears in the IL "
                    "after the translation is completed.",
                )
                continue

        if update_unicode and chars:
            paragraph.unicode = get_char_unicode_string(chars)
        if not chars:
            return
        # 更新边界框
        min_x = min(char.visual_bbox.box.x for char in chars)
        min_y = min(char.visual_bbox.box.y for char in chars)
        max_x = max(char.visual_bbox.box.x2 for char in chars)
        max_y = max(char.visual_bbox.box.y2 for char in chars)
        paragraph.box = Box(min_x, min_y, max_x, max_y)
        paragraph.vertical = chars[0].vertical
        paragraph.xobj_id = chars[0].xobj_id

        paragraph.first_line_indent = False
        if (
            paragraph.pdf_paragraph_composition
            and paragraph.pdf_paragraph_composition[0].pdf_line
            and paragraph.pdf_paragraph_composition[0]
            .pdf_line.pdf_character[0]
            .visual_bbox.box.x
            - paragraph.box.x
            > 1
        ):
            paragraph.first_line_indent = True

    def update_line_data(self, line: PdfLine):
        min_x = min(char.visual_bbox.box.x for char in line.pdf_character)
        min_y = min(char.visual_bbox.box.y for char in line.pdf_character)
        max_x = max(char.visual_bbox.box.x2 for char in line.pdf_character)
        max_y = max(char.visual_bbox.box.y2 for char in line.pdf_character)
        line.box = Box(min_x, min_y, max_x, max_y)

    def add_debug_info(self, page: Page):
        if not self.translation_config.debug:
            return
        for paragraph in page.pdf_paragraph:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_line:
                    line = composition.pdf_line
                    page.pdf_rectangle.append(
                        PdfRectangle(
                            box=line.box,
                            fill_background=False,
                            graphic_state=INDIGO,
                            debug_info=True,
                            line_width=0.2,
                        )
                    )

    def process(self, document):
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            len(document.page),
        ) as pbar:
            if not document.page:
                return
            for page in document.page:
                self.translation_config.raise_if_cancelled()
                self.process_page(page)
                pbar.advance()

            total_paragraph_count = 0
            for page in document.page:
                total_paragraph_count += len(page.pdf_paragraph)
            if total_paragraph_count == 0:
                raise ExtractTextError("The document contains no paragraphs.")

            if self.check_cid_paragraph(document):
                raise ExtractTextError("The document contains too many CID paragraphs.")

    def check_cid_paragraph(self, doc: Document):
        cid_para_count = 0
        para_total = 0
        for page in doc.page:
            para_total += len(page.pdf_paragraph)
            for para in page.pdf_paragraph:
                if is_cid_paragraph(para):
                    cid_para_count += 1
        return cid_para_count / para_total > 0.8

    def bbox_overlap(self, bbox1: Box, bbox2: Box) -> bool:
        return (
            bbox1.x < bbox2.x2
            and bbox1.x2 > bbox2.x
            and bbox1.y < bbox2.y2
            and bbox1.y2 > bbox2.y
        )

    def process_page(self, page: Page):
        layout_index, layout_map = build_layout_index(page)
        # 预处理公式布局的标签
        self._preprocess_formula_layouts(page)

        # Image-text lane (digital pages with injected invisible OCR runs
        # over embedded raster images): pull those characters out FIRST so
        # they can never merge with normal digital-text paragraphs, and turn
        # them into their own per-region label paragraphs.
        image_text_paragraphs = self._extract_image_text_paragraphs(page)

        # 第一步：根据 layout 创建 paragraphs
        # 在这一步中，page.pdf_character 中的字符会被移除
        paragraphs = self._group_characters_into_paragraphs(
            page, layout_index, layout_map
        )
        page.pdf_paragraph = paragraphs

        page_level_formula_font_ids, xobj_specific_formula_font_ids = (
            collect_page_formula_font_ids(
                page, self.translation_config.formular_font_pattern
            )
        )

        # for para in paragraphs:
        #     if not para.debug_id:
        #         continue
        #     new_line = PdfLine(
        #         pdf_character=[x.pdf_character for x in para.pdf_paragraph_composition]
        #     )
        #     self.update_line_data(new_line)
        #     para.pdf_paragraph_composition = [
        #         PdfParagraphComposition(pdf_line=new_line)
        #     ]

        # 第二步：将段落内的字符拆分为行
        for paragraph in paragraphs:
            if (
                paragraph.xobj_id
                and paragraph.xobj_id in xobj_specific_formula_font_ids
            ):
                current_formula_font_ids = xobj_specific_formula_font_ids[
                    paragraph.xobj_id
                ]
            else:
                current_formula_font_ids = page_level_formula_font_ids
            self._split_paragraph_into_lines(paragraph, current_formula_font_ids)

        # 第 2.5 步：合并同一视觉行上被布局框切开的段落碎片
        # （例如加粗的 "AI3" 与 "011 — ..."，或 "The" / 标题主体 / ")"）
        self.merge_same_line_fragment_paragraphs(paragraphs)

        # 第三步：处理段落中的空格
        for paragraph in paragraphs:
            add_space_dummy_chars(paragraph)
            self.process_paragraph_spacing(paragraph)
            self.update_paragraph_data(paragraph)

        # 第四步：计算所有行宽度的中位数
        median_width = self.calculate_median_line_width(paragraphs)

        # 第五步：处理独立段落
        self.process_independent_paragraphs(paragraphs, median_width)

        # 第 5.5 步：按列表项拆分段落
        # 项目符号经常是矢量曲线（小圆点），不在文本流中；
        # 编号列表（"1."、"2."…）也不会被布局模型分开。
        self.split_list_item_paragraphs(page, paragraphs)

        # 新增后处理：合并带行号交替的正文段落（a 正文、b 行号、c 正文 -> 合并 a 与 c，保留 b）
        if getattr(self.translation_config, "merge_alternating_line_numbers", True):
            self.merge_alternating_line_number_paragraphs(paragraphs)

        # Wave 5b (digital decks): a bold label line, an indented monospace
        # code line and an italic "e.g." line often share one layout box and
        # would be translated as one blob and re-wrapped, scrambling the
        # label -> code hierarchy. Keep style-distinct lines as separate
        # paragraphs. The OCR path has its own regroup pass and a flat text
        # layer without style structure, so this is digital-only.
        if not self.translation_config.ocr_workaround:
            self.split_style_boundary_paragraphs(page, paragraphs)

        # OCR sandwich pages: the injected text layer carries textual bullet
        # markers and clean per-line geometry. Split items apart at those
        # markers, then pull wrapped continuation lines back into their item
        # (or heading/body block) so each logical paragraph translates and
        # wraps as ONE unit — layout detection on raster slides is otherwise
        # per-line and erratic.
        if self.translation_config.ocr_workaround:
            self.split_text_bullet_paragraphs(paragraphs)
            self.merge_ocr_continuation_paragraphs(paragraphs)

        for paragraph in paragraphs:
            self.update_paragraph_data(paragraph, update_unicode=True)

        if self.translation_config.ocr_workaround:
            self.add_text_fill_background(page)
            # since this is ocr file,
            # image characters are not needed
            page.pdf_character = []

        self.fix_overlapping_paragraphs(page)

        # Image-text lane: the label paragraphs join the page only after all
        # digital-text passes (merging, splitting, overlap fixing) have run,
        # so none of those ever touches them. Their masks ride along.
        if image_text_paragraphs:
            page.pdf_paragraph.extend(image_text_paragraphs)
            self.add_image_text_masks(page, image_text_paragraphs)

        # 第六步：对每一行的字符进行排序
        # self._sort_characters_in_lines(page)

        self.add_debug_info(page)

        # 新阶段：设置段落的 renderorder 为所有组成部分中 renderorder 最小的
        self._set_paragraph_render_order(page)

    def _set_paragraph_render_order(self, page: Page):
        """
        设置段落的 renderorder 为段落所有组成部分中 renderorder 最小的值
        """
        for paragraph in page.pdf_paragraph:
            min_render_order = 9999999999999999

            # 遍历段落的所有组成部分
            for composition in paragraph.pdf_paragraph_composition:
                # 检查 PdfLine 中的字符
                if composition.pdf_line:
                    for char in composition.pdf_line.pdf_character:
                        if (
                            hasattr(char, "render_order")
                            and char.render_order is not None
                        ):
                            min_render_order = min(min_render_order, char.render_order)

                # 检查单个字符
                elif composition.pdf_character:
                    char = composition.pdf_character
                    if hasattr(char, "render_order") and char.render_order is not None:
                        min_render_order = min(min_render_order, char.render_order)

                # 检查公式中的字符
                elif composition.pdf_formula:
                    for char in composition.pdf_formula.pdf_character:
                        if (
                            hasattr(char, "render_order")
                            and char.render_order is not None
                        ):
                            min_render_order = min(min_render_order, char.render_order)

            # 如果找到了有效的 renderorder，设置段落的 renderorder
            if min_render_order != 9999999999999999:
                paragraph.render_order = min_render_order

    def is_isolated_formula(self, char: PdfCharacter):
        return char.char_unicode in (
            "(cid:122)",
            "(cid:123)",
            "(cid:124)",
            "(cid:125)",
        )

    def _paragraph_text_ascii(self, p: PdfParagraph) -> str:
        parts: list[str] = []
        for comp in p.pdf_paragraph_composition or []:
            if comp.pdf_line:
                for ch in comp.pdf_line.pdf_character or []:
                    if ch.char_unicode is not None:
                        parts.append(ch.char_unicode)
            elif comp.pdf_character and comp.pdf_character.char_unicode is not None:
                parts.append(comp.pdf_character.char_unicode)
        return "".join(parts)

    def _is_ascii_digit_or_space_paragraph(self, p: PdfParagraph) -> bool:
        text = self._paragraph_text_ascii(p)
        if not text:
            return True
        has_digit = False
        for c in text:
            if c.isdigit() and ord(c) < 128:
                has_digit = True
                continue
            if c.isspace():
                continue
            return False
        return True if has_digit or text.strip() == "" else False

    @staticmethod
    def _same_layout_and_xobj(a: PdfParagraph, c: PdfParagraph) -> bool:
        return (
            a.layout_id is not None
            and c.layout_id is not None
            and a.layout_id == c.layout_id
            and a.xobj_id is not None
            and c.xobj_id is not None
            and a.xobj_id == c.xobj_id
        )

    def merge_alternating_line_number_paragraphs(self, paragraphs: list[PdfParagraph]):
        # a 代表正文
        # l 代表行号
        if not paragraphs or len(paragraphs) < 3:
            return
        i = 0
        while i < len(paragraphs) - 2:
            a = paragraphs[i]
            # 吞掉一个或多个连续的行号段 l
            j = i + 1
            saw_l = False
            while j < len(paragraphs) and self._is_ascii_digit_or_space_paragraph(
                paragraphs[j]
            ):
                saw_l = True
                j += 1
            # 现在 j 指向候选的 c
            if saw_l and j < len(paragraphs):
                c = paragraphs[j]
                if self._same_layout_and_xobj(a, c):
                    a.pdf_paragraph_composition.extend(c.pdf_paragraph_composition)
                    self.update_paragraph_data(a)
                    del paragraphs[j]
                    # 不移动 i，继续尝试把更多正文接到 a，实现 a l+ a l+ a ... 链式合并
                    continue
            i += 1

    # ------------------------------------------------------------------
    # Same-line fragment merging (fixes stranded styled prefixes such as
    # a bold "AI3" split from "011 — ..." by overlapping layout boxes).
    # ------------------------------------------------------------------

    def merge_same_line_fragment_paragraphs(self, paragraphs: list[PdfParagraph]):
        """Merge consecutive paragraphs that are fragments of one visual line.

        Overlapping layout detections often split a heading into several
        paragraphs on the same visual line (e.g. a bold first token). The
        fragments end up as separate paragraphs which are translated (or not)
        independently and typeset in place, leaving the prefix stranded.
        Merging them lets the whole heading translate and move as one unit.
        """
        if len(paragraphs) < 2:
            return
        i = 0
        while i < len(paragraphs) - 1:
            a = paragraphs[i]
            b = paragraphs[i + 1]
            if self._should_merge_line_fragments(a, b):
                self._merge_fragment_paragraph(a, b)
                del paragraphs[i + 1]
                # stay on i: allows chains like "The" + <title body> + ")"
            else:
                i += 1

    @staticmethod
    def _paragraph_lines(paragraph: PdfParagraph) -> list[PdfLine]:
        return [
            comp.pdf_line
            for comp in paragraph.pdf_paragraph_composition
            if comp.pdf_line
        ]

    def _should_merge_line_fragments(
        self, a: PdfParagraph, b: PdfParagraph
    ) -> bool:
        if not a.pdf_paragraph_composition or not b.pdf_paragraph_composition:
            return False
        if a.xobj_id != b.xobj_id:
            return False
        a_lines = self._paragraph_lines(a)
        b_lines = self._paragraph_lines(b)
        if not a_lines or not b_lines:
            return False
        # Only merge when at least one side is a single-line fragment;
        # two multi-line paragraphs are never a split heading.
        if len(a_lines) > 1 and len(b_lines) > 1:
            return False
        last_a = a_lines[-1]
        first_b = b_lines[0]
        if last_a.box is None or first_b.box is None:
            return False
        # Must be on the same visual line: strong y-overlap.
        inter = min(last_a.box.y2, first_b.box.y2) - max(
            last_a.box.y, first_b.box.y
        )
        min_height = min(
            last_a.box.y2 - last_a.box.y, first_b.box.y2 - first_b.box.y
        )
        if min_height <= 0 or inter / min_height < 0.5:
            return False
        # Must be horizontally adjacent (a lost inter-word space at most).
        line_height = max(
            last_a.box.y2 - last_a.box.y, first_b.box.y2 - first_b.box.y
        )
        gap = first_b.box.x - last_a.box.x2
        return -2.0 <= gap <= max(2.0, line_height * 0.6)

    def _merge_fragment_paragraph(self, a: PdfParagraph, b: PdfParagraph):
        """Merge paragraph b into a; b's first line joins a's last line."""
        a_line_indices = [
            idx
            for idx, comp in enumerate(a.pdf_paragraph_composition)
            if comp.pdf_line
        ]
        b_line_indices = [
            idx
            for idx, comp in enumerate(b.pdf_paragraph_composition)
            if comp.pdf_line
        ]
        a_last_line = a.pdf_paragraph_composition[a_line_indices[-1]].pdf_line
        b_first_idx = b_line_indices[0]
        b_first_line = b.pdf_paragraph_composition[b_first_idx].pdf_line

        a_last_line.pdf_character.extend(b_first_line.pdf_character)
        self.update_line_data(a_last_line)

        for idx, comp in enumerate(b.pdf_paragraph_composition):
            if idx == b_first_idx:
                continue
            a.pdf_paragraph_composition.append(comp)

        # Keep the layout of the wider fragment (usually the real heading).
        if (
            a.box is not None
            and b.box is not None
            and (b.box.x2 - b.box.x) > (a.box.x2 - a.box.x)
        ):
            a.layout_id = b.layout_id
            a.layout_label = b.layout_label

        self.update_paragraph_data(a)

    # ------------------------------------------------------------------
    # List item splitting (fixes bullet/numbered items collapsing into a
    # single paragraph blob with orphaned vector bullet markers).
    # ------------------------------------------------------------------

    def split_list_item_paragraphs(
        self, page: Page, paragraphs: list[PdfParagraph]
    ):
        """Split paragraphs so each list item is its own paragraph.

        Bullet markers are frequently drawn as small vector curves (dots)
        rather than characters, so the paragraph flow has no textual signal
        for item boundaries. Numbered items ("1.", "2.", ...) are text but
        the layout model often puts a whole list into one layout box.
        A paragraph is split before any line that has a bullet-marker curve
        just to its left, or that starts the next number in a sequence.
        """
        marker_boxes = self._collect_bullet_marker_boxes(page)
        i = 0
        while i < len(paragraphs):
            paragraph = paragraphs[i]
            comps = paragraph.pdf_paragraph_composition
            if comps and len(comps) > 1:
                split_at = self._find_list_item_split_index(comps, marker_boxes)
                if split_at is not None:
                    new_paragraph = PdfParagraph(
                        box=Box(0, 0, 0, 0),
                        pdf_paragraph_composition=comps[split_at:],
                        unicode="",
                        debug_id=generate_base58_id(),
                        layout_label=paragraph.layout_label,
                        layout_id=paragraph.layout_id,
                    )
                    paragraph.pdf_paragraph_composition = comps[:split_at]
                    self.update_paragraph_data(paragraph)
                    self.update_paragraph_data(new_paragraph)
                    paragraphs.insert(i + 1, new_paragraph)
                    # The remainder is re-examined on the next iteration,
                    # so chains of items split one by one.
            i += 1

    @staticmethod
    def _collect_bullet_marker_boxes(page: Page) -> list[Box]:
        """Small vector curves (dots/squares) that can act as bullet markers."""
        markers = []
        for curve in page.pdf_curve or []:
            box = curve.box
            if box is None:
                continue
            width = box.x2 - box.x
            height = box.y2 - box.y
            if 0 < width <= 8 and 0 < height <= 8:
                markers.append(box)
        return markers

    @staticmethod
    def _line_text(line: PdfLine) -> str:
        return "".join(
            char.char_unicode
            for char in line.pdf_character
            if char.char_unicode is not None
        )

    def _find_list_item_split_index(
        self,
        comps: list[PdfParagraphComposition],
        marker_boxes: list[Box],
    ) -> int | None:
        # Numbered-sequence context comes from the paragraph's first line.
        expected_number = None
        first_line_x = None
        first_comp_line = comps[0].pdf_line
        if first_comp_line is not None and first_comp_line.box is not None:
            match = NUMBERED_ITEM_PATTERN.match(self._line_text(first_comp_line))
            if match:
                expected_number = int(match.group(1)) + 1
                first_line_x = first_comp_line.box.x

        for j in range(1, len(comps)):
            line = comps[j].pdf_line
            if line is None or line.box is None:
                continue
            if marker_boxes and self._line_has_marker_to_left(line, marker_boxes):
                return j
            if expected_number is not None and first_line_x is not None:
                match = NUMBERED_ITEM_PATTERN.match(self._line_text(line))
                if (
                    match
                    and int(match.group(1)) == expected_number
                    and abs(line.box.x - first_line_x) < 3.0
                ):
                    return j
        return None

    @staticmethod
    def _line_has_marker_to_left(line: PdfLine, marker_boxes: list[Box]) -> bool:
        line_box = line.box
        line_height = line_box.y2 - line_box.y
        for marker in marker_boxes:
            marker_center_y = (marker.y + marker.y2) / 2
            if not (line_box.y - 2 <= marker_center_y <= line_box.y2 + 2):
                continue
            gap = line_box.x - marker.x2
            if 0 <= gap <= max(20.0, line_height * 1.5):
                return True
        return False

    # ------------------------------------------------------------------
    # Style-boundary splitting (wave 5b, digital decks only): keep lines
    # whose dominant style (font family / bold / italic / mono) differs
    # from the previous line's as separate paragraphs, so a label line, a
    # code line and a note line each typeset on their own line in the
    # original vertical order instead of being merged into one wrapped
    # (and, for RTL, scrambled) paragraph.
    # ------------------------------------------------------------------

    # A line's dominant font must cover at least this share of its
    # non-space characters to count as "the" style of the line.
    STYLE_SPLIT_DOMINANCE = 0.6
    # Both-lines-mono threshold above which a style change is code-internal
    # (bold keywords inside a code block) and must NOT split: the block has
    # to stay whole for the verbatim code-passthrough to catch it.
    STYLE_SPLIT_PURE_MONO = 0.9
    # A line at least this mono-heavy marks a code context, where a leading
    # styled run ("e.g.", a label) starts a new logical line.
    STYLE_SPLIT_CODE_CONTEXT = 0.6

    @staticmethod
    def _line_style_signature(
        line: PdfLine, code_font_ids: set
    ) -> tuple[str | None, str | None, float] | None:
        """(dominant_font_id | None, first_char_font_id, mono_ratio).

        The dominant font is None when no single font reaches
        STYLE_SPLIT_DOMINANCE. Returns None for lines without styled text.
        """
        counts: dict[str, int] = {}
        total = 0
        mono = 0
        prefix_font: str | None = None
        for char in line.pdf_character:
            unicode_ = char.char_unicode
            if unicode_ is None or unicode_.isspace():
                continue
            style = char.pdf_style
            font_id = style.font_id if style else None
            if font_id is None:
                continue
            if prefix_font is None:
                prefix_font = font_id
            counts[font_id] = counts.get(font_id, 0) + 1
            total += 1
            if font_id in code_font_ids:
                mono += 1
        if not total:
            return None
        dominant, count = max(counts.items(), key=lambda kv: kv[1])
        if count / total < ParagraphFinder.STYLE_SPLIT_DOMINANCE:
            dominant = None
        return dominant, prefix_font, mono / total

    @classmethod
    def _is_style_boundary(cls, sig_a, sig_b) -> bool:
        """Do consecutive lines a and b belong to different logical lines?"""
        if sig_a is None or sig_b is None:
            return False
        dom_a, prefix_a, mono_a = sig_a
        dom_b, prefix_b, mono_b = sig_b
        if mono_a >= cls.STYLE_SPLIT_PURE_MONO and mono_b >= cls.STYLE_SPLIT_PURE_MONO:
            # Solid code block (a bold keyword line is still code): keep it
            # whole so the code-paragraph passthrough preserves it verbatim.
            return False
        if dom_a is not None and dom_b is not None and dom_a != dom_b:
            # Bold label line vs monospace code line, italic vs regular ...
            return True
        if (
            mono_a >= cls.STYLE_SPLIT_CODE_CONTEXT
            or mono_b >= cls.STYLE_SPLIT_CODE_CONTEXT
        ) and prefix_a != prefix_b:
            # Code context: a leading styled run ("e.g.," before an inline
            # snippet) starts a new logical line even when the dominant
            # font matches the code line above.
            return True
        return False

    def split_style_boundary_paragraphs(
        self, page: Page, paragraphs: list[PdfParagraph]
    ):
        page_code_font_ids, xobj_code_font_ids = build_code_font_ids(page)
        i = 0
        while i < len(paragraphs):
            paragraph = paragraphs[i]
            comps = paragraph.pdf_paragraph_composition
            if comps and len(comps) > 1:
                code_font_ids = page_code_font_ids
                if (
                    paragraph.xobj_id is not None
                    and paragraph.xobj_id in xobj_code_font_ids
                ):
                    code_font_ids = xobj_code_font_ids[paragraph.xobj_id]
                split_at = None
                prev_sig = None
                for j, comp in enumerate(comps):
                    line = comp.pdf_line
                    if line is None:
                        continue
                    sig = self._line_style_signature(line, code_font_ids)
                    if (
                        j > 0
                        and prev_sig is not None
                        and self._is_style_boundary(prev_sig, sig)
                    ):
                        split_at = j
                        break
                    prev_sig = sig
                if split_at is not None:
                    new_paragraph = PdfParagraph(
                        box=Box(0, 0, 0, 0),
                        pdf_paragraph_composition=comps[split_at:],
                        unicode="",
                        debug_id=generate_base58_id(),
                        layout_label=paragraph.layout_label,
                        layout_id=paragraph.layout_id,
                    )
                    paragraph.pdf_paragraph_composition = comps[:split_at]
                    self.update_paragraph_data(paragraph)
                    self.update_paragraph_data(new_paragraph)
                    paragraphs.insert(i + 1, new_paragraph)
                    # The remainder is re-examined on the next iteration,
                    # so chains of styled lines split one by one.
            i += 1

    # ------------------------------------------------------------------
    # OCR-workaround regrouping (scanned slides with an injected hOCR text
    # layer): textual bullet markers split items apart, and wrapped
    # continuation lines merge back into their logical paragraph.
    # ------------------------------------------------------------------

    @classmethod
    def _line_starts_with_marker(cls, line: PdfLine) -> bool:
        text = cls._line_text(line).lstrip()
        if not text:
            return False
        if BULLET_POINT_PATTERN.match(text[0]):
            return True
        return bool(NUMBERED_ITEM_PATTERN.match(text))

    def split_text_bullet_paragraphs(self, paragraphs: list[PdfParagraph]):
        """Split paragraphs before every line that starts a new list item.

        The OCR prep pass re-emits stripped bullet glyphs as text ("• ..."),
        so on scanned slides item boundaries ARE textual — unlike vector
        bullets, which split_list_item_paragraphs handles.
        """
        i = 0
        while i < len(paragraphs):
            paragraph = paragraphs[i]
            comps = paragraph.pdf_paragraph_composition
            if comps and len(comps) > 1:
                split_at = None
                for j in range(1, len(comps)):
                    line = comps[j].pdf_line
                    if line is not None and self._line_starts_with_marker(line):
                        split_at = j
                        break
                if split_at is not None:
                    new_paragraph = PdfParagraph(
                        box=Box(0, 0, 0, 0),
                        pdf_paragraph_composition=comps[split_at:],
                        unicode="",
                        debug_id=generate_base58_id(),
                        layout_label=paragraph.layout_label,
                        layout_id=paragraph.layout_id,
                    )
                    paragraph.pdf_paragraph_composition = comps[:split_at]
                    self.update_paragraph_data(paragraph)
                    self.update_paragraph_data(new_paragraph)
                    paragraphs.insert(i + 1, new_paragraph)
                    # The remainder is re-examined on the next iteration.
            i += 1

    def merge_ocr_continuation_paragraphs(self, paragraphs: list[PdfParagraph]):
        """Merge wrapped continuation lines back into their paragraph.

        Layout detection sees a raster slide, so a wrapped list item (or a
        multi-line heading/body block) often lands in several per-line
        regions and would be translated and typeset line by line — two font
        sizes and two margins per item. Geometry decides instead: a line
        much closer to the block above than sibling items are, that does not
        start its own item, continues that block.
        """
        changed = True
        while changed:
            changed = False
            for i, a in enumerate(paragraphs):
                for j, b in enumerate(paragraphs):
                    if i == j or not self._is_ocr_continuation(a, b):
                        continue
                    a.pdf_paragraph_composition.extend(
                        b.pdf_paragraph_composition
                    )
                    self.update_paragraph_data(a)
                    del paragraphs[j]
                    changed = True
                    break
                if changed:
                    break

    def _is_ocr_continuation(self, a: PdfParagraph, b: PdfParagraph) -> bool:
        if a.xobj_id != b.xobj_id:
            return False
        a_lines = self._paragraph_lines(a)
        b_lines = self._paragraph_lines(b)
        if not a_lines or not b_lines:
            return False
        if self._line_starts_with_marker(b_lines[0]):
            return False
        la, lb, fa = a_lines[-1].box, b_lines[0].box, a_lines[0].box
        if any(
            box is None or None in (box.x, box.y, box.x2, box.y2)
            for box in (la, lb, fa)
        ):
            return False
        ha, hb = la.y2 - la.y, lb.y2 - lb.y
        if ha <= 0 or hb <= 0 or max(ha, hb) > 1.45 * min(ha, hb):
            return False
        # b directly below a: unrelated stacked blocks sit >= ~1.1 line
        # heights apart, wrapped lines <= ~0.8 (headings may even overlap)
        gap = la.y - lb.y2
        if not (-0.8 * min(ha, hb) <= gap <= 0.85 * max(ha, hb)):
            return False
        overlap = min(la.x2, lb.x2) - max(la.x, lb.x)
        if overlap < 0.5 * min(la.x2 - la.x, lb.x2 - lb.x):
            return False
        if self._line_starts_with_marker(a_lines[0]):
            # a wrapped item line hangs under the item text, never under
            # the marker itself (the marker is roughly a third of a line
            # height wide)
            return lb.x >= fa.x + 0.3 * hb
        # plain blocks: shared left margin, centered stack, or an indented
        # continuation
        if abs(lb.x - fa.x) <= max(2.0, 0.25 * hb):
            return True
        mid_delta = abs((lb.x + lb.x2) - (fa.x + fa.x2)) / 2
        if mid_delta <= 0.12 * max(la.x2 - la.x, lb.x2 - lb.x, fa.x2 - fa.x):
            return True
        return lb.x >= fa.x + 0.5 * hb

    # ------------------------------------------------------------------
    # Image-text lane (digital pages): recognise injected invisible OCR
    # runs over embedded raster images, group them into per-region label
    # paragraphs, and give each one a background-matched mask. This is the
    # per-REGION equivalent of the scanned lane (ocr_workaround), on pages
    # whose real text layer stays fully digital. Inert unless the run
    # declares image_text_regions.
    # ------------------------------------------------------------------

    # Contract 1: a character is an image-OCR character iff its render mode
    # is invisible (Tr 3) AND its box lies inside a declared image_bbox
    # with this tolerance (pt).
    IMAGE_TEXT_BBOX_TOLERANCE = 2.0
    # Two runs on one visual line further apart than this many line heights
    # are separate labels (hexagon labels sharing a row), never one line.
    IMAGE_TEXT_GAP_FACTOR = 1.2
    IMAGE_TEXT_GAP_MIN = 8.0
    # Mask padding beyond the label's glyph box, to swallow the raster
    # glyphs' anti-aliasing halo and OCR-tight ascender undershoot (pt).
    IMAGE_TEXT_MASK_PAD = 2.0

    def _extract_image_text_paragraphs(self, page: Page) -> list[PdfParagraph]:
        """Pull image-OCR characters off the page into label paragraphs.

        Recognised characters leave page.pdf_character before the normal
        layout-based grouping ever sees them, so they can never merge with
        digital-text paragraphs. Returns the label paragraphs (not yet added
        to the page); [] leaves the page untouched.
        """
        if self.translation_config.ocr_workaround:
            return []
        regions = self.translation_config.image_text_regions_for_page(
            page.page_number
        )
        if not regions:
            return []

        per_region: list[list[PdfCharacter]] = [[] for _ in regions]
        remaining: list[PdfCharacter] = []
        for char in page.pdf_character:
            index = self._image_text_region_index(char, regions)
            if index is None:
                remaining.append(char)
            else:
                per_region[index].append(char)
        if not any(per_region):
            return []
        page.pdf_character = remaining

        paragraphs: list[PdfParagraph] = []
        for region, chars in zip(regions, per_region, strict=True):
            if chars:
                paragraphs.extend(
                    self._build_image_text_region_paragraphs(region, chars)
                )
        return paragraphs

    @classmethod
    def _image_text_region_index(
        cls,
        char: PdfCharacter,
        regions: list[tuple[float, float, float, float]],
    ) -> int | None:
        """Contract 1 recognition rule: invisible AND inside a region."""
        if char.render_mode != 3:
            return None
        box = char.box
        if box is None or None in (box.x, box.y, box.x2, box.y2):
            return None
        tol = cls.IMAGE_TEXT_BBOX_TOLERANCE
        for index, (x0, y0, x1, y1) in enumerate(regions):
            if (
                box.x >= x0 - tol
                and box.x2 <= x1 + tol
                and box.y >= y0 - tol
                and box.y2 <= y1 + tol
            ):
                return index
        return None

    def _build_image_text_region_paragraphs(
        self,
        region: tuple[float, float, float, float],
        chars: list[PdfCharacter],
    ) -> list[PdfParagraph]:
        """One region's characters -> its label paragraphs.

        Line-thread the region's characters into visual lines, split lines at
        large horizontal gaps (side-by-side labels), then merge adjacent
        lines that geometrically continue each other (a wrapped label) using
        the scanned lane's continuation rule. Each resulting paragraph is
        tagged with its region and stripped of render order so it layers
        exactly like the scanned lane: mask below, translated text on top.
        """
        carrier = PdfParagraph(
            box=Box(0, 0, 0, 0),
            pdf_paragraph_composition=[
                PdfParagraphComposition(pdf_character=char) for char in chars
            ],
            unicode="",
            debug_id=generate_base58_id(),
        )
        self._split_paragraph_into_lines(carrier, set())

        paragraphs: list[PdfParagraph] = []
        for composition in carrier.pdf_paragraph_composition:
            line = composition.pdf_line
            if line is None or not line.pdf_character:
                continue
            for piece in self._split_image_text_line_at_gaps(line):
                paragraph = PdfParagraph(
                    box=Box(0, 0, 0, 0),
                    pdf_paragraph_composition=[piece],
                    unicode="",
                    debug_id=generate_base58_id(),
                    layout_label="image_text",
                )
                self.update_paragraph_data(paragraph)
                paragraphs.append(paragraph)

        # Wrapped labels: pull continuation lines back into their paragraph.
        self.merge_ocr_continuation_paragraphs(paragraphs)

        for paragraph in paragraphs:
            paragraph.raster_region = list(region)
            self.update_paragraph_data(paragraph, update_unicode=True)
            for composition in paragraph.pdf_paragraph_composition:
                if not composition.pdf_line:
                    continue
                for char in composition.pdf_line.pdf_character:
                    # Layering (scanned-lane convention): orderless elements
                    # render last, masks (finite sub-order) below the
                    # translated text (orderless). Also drop the invisible
                    # run's own graphic state; the mask pass may retint.
                    char.render_order = None
                    char.sub_render_order = None
                    if char.pdf_style is not None:
                        char.pdf_style.graphic_state = BLACK
        return paragraphs

    def _split_image_text_line_at_gaps(
        self, line: PdfLine
    ) -> list[PdfParagraphComposition]:
        """Split one threaded line into label pieces at big horizontal gaps."""
        chars = sorted(
            line.pdf_character,
            key=lambda c: (c.visual_bbox.box.x if c.visual_bbox else c.box.x),
        )
        height = max(
            (c.visual_bbox.box.y2 - c.visual_bbox.box.y for c in chars),
            default=0.0,
        )
        threshold = max(
            self.IMAGE_TEXT_GAP_MIN, self.IMAGE_TEXT_GAP_FACTOR * height
        )
        pieces: list[list[PdfCharacter]] = [[chars[0]]]
        for prev, char in zip(chars, chars[1:], strict=False):
            prev_box = prev.visual_bbox.box if prev.visual_bbox else prev.box
            char_box = char.visual_bbox.box if char.visual_bbox else char.box
            if char_box.x - prev_box.x2 > threshold:
                pieces.append([])
            pieces[-1].append(char)
        return [self.create_line(piece) for piece in pieces if piece]

    def add_image_text_masks(
        self, page: Page, paragraphs: list[PdfParagraph]
    ) -> None:
        """A background-matched mask under each image-OCR label paragraph.

        The mask covers the label's source pixels inside the raster image
        (plus an anti-aliasing pad, clipped to the region) and carries the
        region so the RTL mirror can move it rigidly with the image. The
        sampled ink color retints the translated text (white-on-teal labels
        stay white).
        """
        sampler = self._get_mask_color_sampler()
        pad = self.IMAGE_TEXT_MASK_PAD
        for paragraph in paragraphs:
            if paragraph.box is None or not paragraph.raster_region:
                continue
            rx0, ry0, rx1, ry1 = paragraph.raster_region
            # The paragraph box is built from visual bboxes (descent-shifted);
            # the raster glyphs underneath also overshoot the OCR-tight run
            # by their ascenders. Union both char boxes so the mask covers
            # the full glyph band, not just the shifted visual one.
            gx1, gy1, gx2, gy2 = (
                paragraph.box.x,
                paragraph.box.y,
                paragraph.box.x2,
                paragraph.box.y2,
            )
            for composition in paragraph.pdf_paragraph_composition or []:
                if not composition.pdf_line:
                    continue
                for char in composition.pdf_line.pdf_character:
                    for char_box in (
                        char.box,
                        char.visual_bbox.box if char.visual_bbox else None,
                    ):
                        if char_box is None or None in (
                            char_box.x, char_box.y, char_box.x2, char_box.y2
                        ):
                            continue
                        gx1 = min(gx1, char_box.x)
                        gy1 = min(gy1, char_box.y)
                        gx2 = max(gx2, char_box.x2)
                        gy2 = max(gy2, char_box.y2)
            x1 = max(gx1 - pad, rx0)
            y1 = max(gy1 - pad, ry0)
            x2 = min(gx2 + pad, rx1)
            y2 = min(gy2 + pad, ry1)
            if x2 <= x1 or y2 <= y1:
                continue
            fill_state = WHITE
            if sampler is not None:
                try:
                    bg, ink = sampler.sample(page.page_number, x1, y1, x2, y2)
                    if bg is not None:
                        fill_state = solid_graphic_state(bg)
                    if ink is not None:
                        self._tint_paragraph_text(paragraph, ink)
                except Exception:
                    logger.exception(
                        "Mask-color sampling failed for image-text paragraph "
                        f"{getattr(paragraph, 'debug_id', None)} on page "
                        f"{page.page_number}; using a white mask."
                    )
            page.pdf_rectangle.append(
                PdfRectangle(
                    box=Box(x1, y1, x2, y2),
                    fill_background=True,
                    graphic_state=fill_state,
                    debug_info=False,
                    xobj_id=paragraph.xobj_id,
                    raster_region=list(paragraph.raster_region),
                )
            )

    def _group_characters_into_paragraphs(
        self, page: Page, layout_index, layout_map
    ) -> list[PdfParagraph]:
        paragraphs: list[PdfParagraph] = []
        if page.pdf_paragraph:
            paragraphs.extend(page.pdf_paragraph)
            page.pdf_paragraph = []

        char_areas = [
            (char.visual_bbox.box.x2 - char.visual_bbox.box.x)
            * (char.visual_bbox.box.y2 - char.visual_bbox.box.y)
            for char in page.pdf_character
        ]
        median_char_area = 0.0
        if char_areas:
            char_areas.sort()
            mid = len(char_areas) // 2
            median_char_area = (
                char_areas[mid]
                if len(char_areas) % 2 == 1
                else (char_areas[mid - 1] + char_areas[mid]) / 2
            )

        current_paragraph: PdfParagraph | None = None
        current_layout: Layout | None = None
        skip_chars = []

        for char in page.pdf_character:
            char_layout = get_character_layout(char, layout_index, layout_map)
            # Check if character is in any formula layout and set formula_layout_id
            char.formula_layout_id = is_character_in_formula_layout(
                char, page, layout_index, layout_map
            )

            if not is_text_layout(char_layout) or self.is_isolated_formula(char):
                skip_chars.append(char)
                continue

            char_box = char.visual_bbox.box
            # char_pdf_box = char.box
            # if calculate_iou_for_boxes(char_box, char_pdf_box) < 0.2:
            #     char_box = char_pdf_box
            char_area = (char_box.x2 - char_box.x) * (char_box.y2 - char_box.y)
            is_small_char = char_area < median_char_area * 0.05

            is_new_paragraph = False
            if current_paragraph is None:
                is_new_paragraph = True
            elif (
                not (
                    is_small_char
                    and current_paragraph.pdf_paragraph_composition
                    and char_layout.id == current_layout.id
                )
                and char.char_unicode not in HEIGHT_NOT_USFUL_CHAR_IN_CHAR
            ):
                if (
                    (
                        char_layout.id != current_layout.id
                        and not SPACE_REGEX.match(char.char_unicode)
                    )
                    or (  # not same xobject
                        current_paragraph.pdf_paragraph_composition
                        and current_paragraph.pdf_paragraph_composition[
                            -1
                        ].pdf_character.xobj_id
                        != char.xobj_id
                    )
                    or (
                        is_bullet_point(char)
                        and not current_paragraph.pdf_paragraph_composition
                    )
                ):
                    is_new_paragraph = True

            if is_new_paragraph:
                current_layout = char_layout
                current_paragraph = PdfParagraph(
                    pdf_paragraph_composition=[],
                    layout_id=current_layout.id,
                    debug_id=generate_base58_id(),
                    layout_label=current_layout.name,
                )
                paragraphs.append(current_paragraph)

            current_paragraph.pdf_paragraph_composition.append(
                PdfParagraphComposition(pdf_character=char)
            )

        page.pdf_character = skip_chars
        for para in paragraphs:
            self.update_paragraph_data(para)
        return paragraphs

    def _merge_overlapping_clusters(
        self, lines: dict[int, list[PdfCharacter]], char_height_average: float
    ) -> dict[int, list[PdfCharacter]]:
        """
        Merge clusters that have significant y-axis overlap.
        If y_intersection / min_height > 0.5 or the distance between y-midlines is less than char_height_average, merge the two clusters.
        """
        if len(lines) <= 1:
            return lines

        # Calculate y-axis ranges for each cluster
        cluster_ranges = {}
        cluster_midlines = {}
        for label, chars in lines.items():
            y_values = [char.visual_bbox.box.y for char in chars] + [
                char.visual_bbox.box.y2 for char in chars
            ]
            y_min, y_max = min(y_values), max(y_values)
            cluster_ranges[label] = (y_min, y_max)
            cluster_midlines[label] = (y_min + y_max) / 2

        # Keep merging until no more merges are possible
        changed = True
        while changed:
            changed = False
            labels_to_check = list(lines.keys())

            for i in range(len(labels_to_check)):
                if not changed:  # Only continue if no merge happened in this iteration
                    for j in range(i + 1, len(labels_to_check)):
                        label1, label2 = labels_to_check[i], labels_to_check[j]

                        # Skip if either label has been merged away
                        if label1 not in lines or label2 not in lines:
                            continue

                        y1_min, y1_max = cluster_ranges[label1]
                        y2_min, y2_max = cluster_ranges[label2]

                        # Calculate intersection
                        intersection_start = max(y1_min, y2_min)
                        intersection_end = min(y1_max, y2_max)

                        # Calculate midline distance
                        midline_distance = abs(
                            cluster_midlines[label1] - cluster_midlines[label2]
                        )

                        should_merge = False
                        if (
                            intersection_end > intersection_start
                        ):  # There is intersection
                            intersection_height = intersection_end - intersection_start
                            height1 = y1_max - y1_min
                            height2 = y2_max - y2_min
                            min_height = min(height1, height2)

                            # Check if intersection ratio exceeds threshold
                            if (
                                min_height > 0
                                and intersection_height / min_height > 0.3
                            ):
                                should_merge = True

                        # Check if midline distance is less than char_height_average
                        if midline_distance < char_height_average:
                            should_merge = True

                        if should_merge:
                            # Merge label2 into label1
                            lines[label1].extend(lines[label2])
                            del lines[label2]

                            # Update cluster range and midline for the merged cluster
                            new_y_min = min(y1_min, y2_min)
                            new_y_max = max(y1_max, y2_max)
                            cluster_ranges[label1] = (new_y_min, new_y_max)
                            cluster_midlines[label1] = (new_y_min + new_y_max) / 2
                            del cluster_ranges[label2]
                            del cluster_midlines[label2]

                            changed = True
                            break

        return lines

    def _get_effective_y_bounds(self, char: PdfCharacter) -> tuple[float, float]:
        """
        Determines the effective vertical boundaries (y1, y2) for a character.

        It prioritizes the visual bounding box if its Intersection over Union (IoU)
        with the PDF bounding box is high (>= 0.5), otherwise, it falls back to the
        PDF bounding box. This helps use more accurate layout information when available.
        """
        visual_box = char.visual_bbox.box
        return visual_box.y, visual_box.y2
        pdf_box = char.box
        if calculate_iou_for_boxes(visual_box, pdf_box) >= 0.5:
            return visual_box.y, visual_box.y2
        return pdf_box.y, pdf_box.y2

    @staticmethod
    def _compute_collision_counts_histogram(
        y1_arr: np.ndarray,
        y2_arr: np.ndarray,
        para_y_min: float,
        para_y_max: float,
        step: float,
    ) -> np.ndarray:
        """Compute overlap counts at each scan line using a difference-array histogram.

        Args:
            y1_arr: 1-D array with lower y bounds of characters (inclusive).
            y2_arr: 1-D array with upper y bounds of characters (exclusive).
            para_y_min: Minimum y of the paragraph.
            para_y_max: Maximum y of the paragraph.
            step: Scan step size.

        Returns:
            1-D NumPy int32 array where index i corresponds to y = para_y_max - i × step.
        """
        # Number of scan positions
        m = int(np.ceil((para_y_max - para_y_min) / step))
        if m <= 0:
            return np.array([], dtype=np.int32)

        # Map character bounds to discrete indices (top inclusive, bottom exclusive)
        starts = np.floor((para_y_max - y2_arr) / step).astype(np.int32)
        ends = np.floor((para_y_max - y1_arr) / step).astype(np.int32) + 1
        # Clip ends to the valid range [0, m]
        np.clip(ends, 0, m, out=ends)

        hist = np.zeros(m + 1, dtype=np.int32)
        np.add.at(hist, starts, 1)
        np.add.at(hist, ends, -1)

        return np.cumsum(hist[:-1])

    def _split_paragraph_into_lines(
        self, paragraph: PdfParagraph, formula_font_ids: set[str]
    ):
        """
        Splits a paragraph into lines using a "line-threading" method.

        This method works by scanning vertically across the paragraph's bounding
        box and counting how many characters intersect with a horizontal line
        at each y-coordinate. The regions with a low number of intersections
        (less than 2) are identified as gaps between lines. The characters
        are then partitioned into lines based on these identified gaps.
        """
        if not paragraph.pdf_paragraph_composition:
            return

        # 1. Extract all characters and other compositions from the paragraph.
        all_chars: list[PdfCharacter] = []
        other_compositions: list[PdfParagraphComposition] = []
        for comp in paragraph.pdf_paragraph_composition:
            if comp.pdf_character:
                all_chars.append(comp.pdf_character)
            else:
                other_compositions.append(comp)

        if not all_chars:
            return

        # 2. Determine effective y-bounds for each character and the paragraph's total vertical range.
        char_y_bounds = [
            {"char": char, "y1": y1, "y2": y2}
            for char in all_chars
            for y1, y2 in [self._get_effective_y_bounds(char)]
        ]

        if not char_y_bounds:
            paragraph.pdf_paragraph_composition = other_compositions
            self.update_paragraph_data(paragraph)
            return

        para_y_min = min(b["y1"] for b in char_y_bounds)
        para_y_max = max(b["y2"] for b in char_y_bounds)

        # If the paragraph is vertically flat, treat it as a single line.
        if (para_y_max - para_y_min) < 5:  # Using a small threshold
            # all_chars.sort(key=lambda c: c.visual_bbox.box.x)
            single_line_composition = self.create_line(all_chars)
            paragraph.pdf_paragraph_composition = [
                single_line_composition
            ] + other_compositions
            self.update_paragraph_data(paragraph)
            return

        # 3. Perform "threading" scan to create a collision histogram.
        # Scan from top (max y) to bottom (min y) with a step of 0.5.
        scan_y_min = para_y_min
        scan_y_max = para_y_max
        step = 0.25

        y_coordinates = np.arange(scan_y_max, scan_y_min, -step)

        # Compute collision counts using NumPy histogram (O(m + n))
        y1_arr = np.array([b["y1"] for b in char_y_bounds], dtype=np.float32)
        y2_arr = np.array([b["y2"] for b in char_y_bounds], dtype=np.float32)
        collision_counts = self._compute_collision_counts_histogram(
            y1_arr,
            y2_arr,
            scan_y_min,
            scan_y_max,
            step,
        )

        # 4. Find gaps (regions with low collision count) from the histogram.
        gaps = []
        in_gap = False
        for i, count in enumerate(collision_counts):
            if count < 1 and not in_gap:
                in_gap = True
                gap_start_index = i
            elif count >= 1 and in_gap:
                in_gap = False
                gaps.append((gap_start_index, i - 1))
        if in_gap:
            gaps.append((gap_start_index, len(collision_counts) - 1))

        # If no significant gaps are found, treat it as a single line.
        if not gaps:
            # all_chars.sort(key=lambda c: c.visual_bbox.box.x)
            single_line_composition = self.create_line(all_chars)
            paragraph.pdf_paragraph_composition = [
                single_line_composition
            ] + other_compositions
            self.update_paragraph_data(paragraph)
            return

        # 5. Assign characters to lines based on the identified gaps.
        # Calculate separator y-coordinates from the midpoints of the gaps.
        separator_y_coords = sorted(
            [y_coordinates[start_idx] for start_idx, end_idx in gaps],
            reverse=True,
        )

        lines: list[list[PdfCharacter]] = [
            [] for _ in range(len(separator_y_coords) + 1)
        ]

        for b in char_y_bounds:
            char_y_center = (b["y1"] + b["y2"]) / 2
            line_idx = 0
            # Find which line bucket the character belongs to.
            for sep_y in separator_y_coords:
                if char_y_center > sep_y:
                    break
                line_idx += 1
            lines[line_idx].append(b["char"])

        # 6. Rebuild the paragraph's composition list from the new lines.
        new_line_compositions = []
        for line_chars in lines:
            if line_chars:
                # Sort characters within each line by x-coordinate (left-to-right).
                # line_chars.sort(key=lambda c: c.visual_bbox.box.x)
                new_line_compositions.append(self.create_line(line_chars))

        # The lines are already sorted vertically due to the scanning process.
        paragraph.pdf_paragraph_composition = new_line_compositions + other_compositions
        self.update_paragraph_data(paragraph)

    def process_paragraph_spacing(self, paragraph: PdfParagraph):
        if not paragraph.pdf_paragraph_composition:
            return

        # 处理行级别的空格
        processed_lines = []
        for composition in paragraph.pdf_paragraph_composition:
            if not composition.pdf_line:
                processed_lines.append(composition)
                continue

            line = composition.pdf_line
            if not "".join(
                x.char_unicode for x in line.pdf_character
            ).strip():  # 跳过完全空白的行
                continue

            # 处理行内字符的尾随空格
            processed_chars = []
            for char in line.pdf_character:
                if not char.char_unicode.isspace():
                    processed_chars = processed_chars + [char]
                elif processed_chars:  # 只有在有非空格字符后才考虑保留空格
                    processed_chars.append(char)

            # 移除尾随空格
            while processed_chars and processed_chars[-1].char_unicode.isspace():
                processed_chars.pop()

            if processed_chars:  # 如果行内还有字符
                line = self.create_line(processed_chars)
                processed_lines.append(line)

        paragraph.pdf_paragraph_composition = processed_lines
        self.update_paragraph_data(paragraph)

    def create_line(self, chars: list[PdfCharacter]) -> PdfParagraphComposition:
        assert chars

        line = PdfLine(pdf_character=chars)
        self.update_line_data(line)
        return PdfParagraphComposition(pdf_line=line)

    def calculate_median_line_width(self, paragraphs: list[PdfParagraph]) -> float:
        # 收集所有行的宽度
        line_widths = []
        for paragraph in paragraphs:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_line:
                    line = composition.pdf_line
                    line_widths.append(line.box.x2 - line.box.x)

        if not line_widths:
            return 0.0

        # 计算中位数
        line_widths.sort()
        mid = len(line_widths) // 2
        if len(line_widths) % 2 == 0:
            return (line_widths[mid - 1] + line_widths[mid]) / 2
        return line_widths[mid]

    def process_independent_paragraphs(
        self,
        paragraphs: list[PdfParagraph],
        median_width: float,
    ):
        i = 0
        while i < len(paragraphs):
            paragraph = paragraphs[i]
            if len(paragraph.pdf_paragraph_composition) <= 1:  # 跳过只有一行的段落
                i += 1
                continue

            j = 1
            while j < len(paragraph.pdf_paragraph_composition):
                prev_composition = paragraph.pdf_paragraph_composition[j - 1]
                if not prev_composition.pdf_line:
                    j += 1
                    continue

                prev_line = prev_composition.pdf_line
                prev_width = prev_line.box.x2 - prev_line.box.x
                prev_text = "".join([c.char_unicode for c in prev_line.pdf_character])

                # 检查是否包含连续的点（至少 20 个）
                # 如果有至少连续 20 个点，则代表这是目录条目
                if re.search(r"\.{20,}", prev_text):
                    # 创建新的段落
                    new_paragraph = PdfParagraph(
                        box=Box(0, 0, 0, 0),  # 临时边界框
                        pdf_paragraph_composition=(
                            paragraph.pdf_paragraph_composition[j:]
                        ),
                        unicode="",
                        debug_id=generate_base58_id(),
                        layout_label=paragraph.layout_label,
                        layout_id=paragraph.layout_id,
                    )
                    # 更新原段落
                    paragraph.pdf_paragraph_composition = (
                        paragraph.pdf_paragraph_composition[:j]
                    )

                    # 更新两个段落的数据
                    self.update_paragraph_data(paragraph)
                    self.update_paragraph_data(new_paragraph)

                    # 在原段落后插入新段落
                    paragraphs.insert(i + 1, new_paragraph)
                    break

                # 如果前一行宽度小于中位数的一半，将当前行及后续行分割成新段落
                if (
                    self.translation_config.split_short_lines
                    and prev_width
                    < median_width * self.translation_config.short_line_split_factor
                ) or (
                    paragraph.pdf_paragraph_composition
                    and (current_line := paragraph.pdf_paragraph_composition[j])
                    and (line := current_line.pdf_line)
                    and (chars := line.pdf_character)
                    and (char := chars[0])
                    and is_bullet_point(char)
                ):
                    # 创建新的段落
                    new_paragraph = PdfParagraph(
                        box=Box(0, 0, 0, 0),  # 临时边界框
                        pdf_paragraph_composition=(
                            paragraph.pdf_paragraph_composition[j:]
                        ),
                        unicode="",
                        debug_id=generate_base58_id(),
                        layout_label=paragraph.layout_label,
                        layout_id=paragraph.layout_id,
                    )
                    # 更新原段落
                    paragraph.pdf_paragraph_composition = (
                        paragraph.pdf_paragraph_composition[:j]
                    )

                    # 更新两个段落的数据
                    self.update_paragraph_data(paragraph)
                    self.update_paragraph_data(new_paragraph)

                    # 在原段落后插入新段落
                    paragraphs.insert(i + 1, new_paragraph)
                    break
                j += 1
            i += 1

    @staticmethod
    def is_bbox_contain_in_vertical(bbox1: Box, bbox2: Box) -> bool:
        """Check if one bounding box is completely contained within the other."""
        # Check if bbox1 is contained in bbox2
        bbox1_in_bbox2 = bbox1.y >= bbox2.y and bbox1.y2 <= bbox2.y2
        # Check if bbox2 is contained in bbox1
        bbox2_in_bbox1 = bbox2.y >= bbox1.y and bbox2.y2 <= bbox1.y2
        return bbox1_in_bbox2 or bbox2_in_bbox1

    def fix_overlapping_paragraphs(self, page: Page):
        """
        Adjusts the bounding boxes of paragraphs on a page to resolve vertical overlaps.

        Iteratively checks pairs of paragraphs and adjusts their vertical boundaries
        (y and y2) if they overlap, aiming to place the boundary at the midpoint
        of the vertical overlap.
        """
        paragraphs = page.pdf_paragraph
        if not paragraphs or len(paragraphs) < 2:
            return

        max_iterations = len(paragraphs) * len(paragraphs)  # Safety break
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            overlap_found_in_pass = False

            for i in range(len(paragraphs)):
                for j in range(i + 1, len(paragraphs)):
                    para1 = paragraphs[i]
                    para2 = paragraphs[j]

                    if para1.box is None or para2.box is None:
                        continue

                    if para1.xobj_id != para2.xobj_id:
                        continue

                    # Check for overlap using the existing method
                    if self.bbox_overlap(para1.box, para2.box):
                        if self.is_bbox_contain_in_vertical(para1.box, para2.box):
                            continue
                        # Calculate vertical overlap details
                        overlap_y_start = max(para1.box.y, para2.box.y)
                        overlap_y_end = min(para1.box.y2, para2.box.y2)
                        overlap_height = overlap_y_end - overlap_y_start

                        # Calculate horizontal overlap details
                        overlap_x_start = max(para1.box.x, para2.box.x)
                        overlap_x_end = min(para1.box.x2, para2.box.x2)
                        overlap_width = overlap_x_end - overlap_x_start

                        # Ensure there's a real 2D overlap, focusing on vertical adjustment
                        if overlap_height > 1e-6 and overlap_width > 1e-6:
                            overlap_found_in_pass = True

                            # Determine which paragraph is visually higher
                            if para1.box.y2 > para2.box.y and para1.box.y < para2.box.y:
                                lower_para = para1
                                higher_para = para2
                            # Handle cases where y values are identical (or very close)
                            # Prefer the one with smaller y2 as the higher one, or break tie arbitrarily
                            elif para1.box.y2 < para2.box.y2:
                                lower_para = para1
                                higher_para = para2
                            else:
                                lower_para = para2
                                higher_para = para1

                            # Calculate the midpoint of the vertical overlap
                            mid_y = overlap_y_start + overlap_height / 2

                            # Adjust boxes, ensuring they remain valid (y2 > y)
                            if mid_y > higher_para.box.y and mid_y < lower_para.box.y2:
                                higher_para.box.y = mid_y + 1
                                lower_para.box.y2 = mid_y - 1
                            else:
                                # This might happen if one box is fully contained vertically
                                # within another, or due to floating point issues.
                                # Log a warning and skip adjustment for this pair in this iteration.
                                # A more complex strategy might be needed for full containment.
                                logger.warning(
                                    "Could not resolve overlap between paragraphs"
                                    f" {higher_para.debug_id} and {lower_para.debug_id}"
                                    " using simple midpoint strategy."
                                    f" Midpoint: {mid_y},"
                                    f" Higher Box: {higher_para.box},"
                                    f" Lower Box: {lower_para.box}"
                                )

            # If no overlaps were found and adjusted in this pass, we're done.
            if not overlap_found_in_pass:
                break

        if iterations == max_iterations:
            logger.warning(
                f"Maximum iterations ({max_iterations}) reached in"
                f" fix_overlapping_paragraphs for page {page.page_number}."
                " Some overlaps might remain."
            )

    def _sort_characters_in_lines(self, page: Page):
        """Sort characters in each line from left to right, top to bottom."""
        for paragraph in page.pdf_paragraph:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_line:
                    line = composition.pdf_line
                    line.pdf_character.sort(key=self._get_char_sort_key)

    def _get_char_sort_key(self, char: PdfCharacter):
        """Get sort key for character positioning (top to bottom, left to right)."""
        visual_box = char.visual_bbox.box
        pdf_box = char.box

        # Use visual box if IoU with bbox is >= 0.1, otherwise use bbox
        if calculate_iou_for_boxes(visual_box, pdf_box) >= 0.1:
            box = visual_box
        else:
            box = pdf_box

        # Sort by y coordinate first (top to bottom), then x coordinate (left to right)
        # Note: In PDF coordinate system, y increases upward, so we negate y for top-to-bottom sorting
        return (box.x, -box.y)
