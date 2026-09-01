import io
import itertools
import logging
import math
import os
import re
import time
import unicodedata
from abc import ABC
from abc import abstractmethod
from multiprocessing import Process
from pathlib import Path

import freetype
import pymupdf
from bitstring import BitStream

from babeldoc.assets.embedding_assets_metadata import FONT_NAMES
from babeldoc.format.pdf.document_il import PdfOriginalPath
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.matrix_helper import matrix_to_bytes
from babeldoc.format.pdf.document_il.utils.type3_font_metrics import (
    inverse_type3_font_size_for_tf,
)
from babeldoc.format.pdf.document_il.utils.zstd_helper import zstd_decompress
from babeldoc.format.pdf.new_parser.pdf_token_serializer import serialize_pdf_token
from babeldoc.format.pdf.translation_config import TranslateResult
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import WatermarkOutputMode

logger = logging.getLogger(__name__)

SUBSET_FONT_STAGE_NAME = "Subset font"
SAVE_PDF_STAGE_NAME = "Save PDF"
_EXTGSTATE_USAGE_RE = re.compile(rb"/([!#$%&'*+,\-.0-9:;=?@A-Z\\^_`a-z{|}~]+)\s+gs\b")
_SHADING_USAGE_RE = re.compile(rb"/([!#$%&'*+,\-.0-9:;=?@A-Z\\^_`a-z{|}~]+)\s+sh\b")


class RenderUnit(ABC):
    """Abstract base class for all renderable units."""

    def __init__(
        self,
        render_order: int,
        sub_render_order: int = 0,
        xobj_id: str | None = None,
    ):
        self.render_order = render_order
        self.sub_render_order = sub_render_order
        self.xobj_id = xobj_id
        if self.render_order is None:
            self.render_order = 9999999999999999
        if self.sub_render_order is None:
            self.sub_render_order = 9999999999999999

    @abstractmethod
    def render(
        self,
        draw_op: BitStream,
        context: "RenderContext",
    ) -> None:
        """Render this unit to the draw_op BitStream."""
        pass

    def get_sort_key(self) -> tuple[int, int]:
        """Get the sort key for ordering render units."""
        return (self.render_order, self.sub_render_order)


class CharacterRenderUnit(RenderUnit):
    """Render unit for PDF characters."""

    def __init__(
        self,
        char: il_version_1.PdfCharacter,
        render_order: int,
        sub_render_order: int = 0,
    ):
        super().__init__(render_order, sub_render_order, char.xobj_id)
        self.char = char

    def render(self, draw_op: BitStream, context: "RenderContext") -> None:
        char = self.char
        if char.char_unicode == "\n":
            return
        if char.pdf_character_id is None:
            return

        char_size = char.pdf_style.font_size
        font_id = char.pdf_style.font_id
        tf_font_size = self._font_size_for_pdf_tf(char_size, font_id, context)

        # Get encoding length map based on xobj_id
        if self.xobj_id in context.xobj_encoding_length_map:
            encoding_length_map = context.xobj_encoding_length_map[self.xobj_id]
        else:
            encoding_length_map = context.page_encoding_length_map

        # Check font exists if needed
        if context.check_font_exists:
            if self.xobj_id in context.xobj_available_fonts:
                if font_id not in context.xobj_available_fonts[self.xobj_id]:
                    return
            elif font_id not in context.available_font_list:
                return

        draw_op.append(b"q ")
        context.pdf_creator.render_graphic_state(draw_op, char.pdf_style.graphic_state)

        if char.vertical:
            draw_op.append(
                f"BT /{font_id} {tf_font_size:f} Tf 0 1 -1 0 {char.box.x2:f} {char.box.y:f} Tm ".encode(),
            )
        else:
            text_draw_op = (
                f"BT /{font_id} {tf_font_size:f} Tf "
                f"1 0 0 1 {char.box.x:f} {char.box.y:f} Tm "
            )
            draw_op.append(text_draw_op.encode())

        encoding_length = encoding_length_map.get(font_id, None)
        if encoding_length is None:
            if font_id in context.all_encoding_length_map:
                encoding_length = context.all_encoding_length_map[font_id]
            else:
                logger.debug(
                    f"Font {font_id} not found in encoding length map for page {context.page.page_number}"
                )
                return

        draw_op.append(
            f"<{char.pdf_character_id:0{encoding_length * 2}x}>".upper().encode(),
        )
        draw_op.append(b" Tj ET Q \n")

    def _font_size_for_pdf_tf(
        self,
        char_size: float,
        font_id: str,
        context: "RenderContext",
    ) -> float:
        get_scoped_font = getattr(context, "get_scoped_font", None)
        font = (
            get_scoped_font(font_id, self.xobj_id)
            if callable(get_scoped_font)
            else None
        )
        if font is None:
            xobj_font_map = getattr(context, "xobj_font_map", {})
            if self.xobj_id in xobj_font_map:
                font = xobj_font_map[self.xobj_id].get(font_id)
        if font is None:
            font = getattr(context, "page_font_map", {}).get(font_id)
        if font is None:
            return char_size
        return inverse_type3_font_size_for_tf(font, char_size)


class FormRenderUnit(RenderUnit):
    """Render unit for PDF forms."""

    def __init__(
        self,
        form: il_version_1.PdfForm,
        render_order: int,
        sub_render_order: int = 0,
    ):
        super().__init__(render_order, sub_render_order, form.xobj_id)
        self.form = form

    def render(self, draw_op: BitStream, context: "RenderContext") -> None:
        form = self.form
        draw_op.append(b"q ")

        # Apply relocation transform first if present (before passthrough instructions)
        # This ensures masks in passthrough_per_char_instruction use the correct coordinate system
        assert form.pdf_matrix is not None
        if form.relocation_transform and len(form.relocation_transform) == 6:
            try:
                relocation_matrix = tuple(float(x) for x in form.relocation_transform)
                draw_op.append(matrix_to_bytes(relocation_matrix))
            except (ValueError, TypeError):
                # If relocation transform conversion fails, skip it and use original matrix later
                pass

        draw_op.append(matrix_to_bytes(form.pdf_matrix))

        draw_op.append(b" ")

        draw_op.append(
            form.graphic_state.passthrough_per_char_instruction.encode(),
        )

        draw_op.append(b" ")

        assert form.pdf_form_subtype is not None
        if form.pdf_form_subtype.pdf_xobj_form:
            draw_op.append(
                f" /{form.pdf_form_subtype.pdf_xobj_form.do_args} Do ".encode()
            )
        elif form.pdf_form_subtype.pdf_inline_form:
            # Handle inline form (inline image)
            inline_form = form.pdf_form_subtype.pdf_inline_form

            # Start inline image
            draw_op.append(b" BI ")

            # Add image parameters if available
            if inline_form.image_parameters:
                import json

                try:
                    params = json.loads(inline_form.image_parameters)
                    for key, value in params.items():
                        if key.startswith("/"):
                            key = key[1:]  # Remove leading slash
                        draw_op.append(f"/{key} {serialize_pdf_token(value)} ".encode())
                except json.JSONDecodeError:
                    pass

            # Start image data
            draw_op.append(b"ID ")

            # Add image data if available (base64 decode it first)
            if inline_form.form_data:
                import base64

                try:
                    image_data = base64.b64decode(inline_form.form_data)
                    draw_op.append(image_data)
                except Exception:
                    pass

            # End inline image
            draw_op.append(b" EI ")
        draw_op.append(b" Q\n")


class RectangleRenderUnit(RenderUnit):
    """Render unit for PDF rectangles."""

    def __init__(
        self,
        rectangle: il_version_1.PdfRectangle,
        render_order: int,
        sub_render_order: int = 0,
        line_width: float = 0.4,
    ):
        super().__init__(render_order, sub_render_order, rectangle.xobj_id)
        self.rectangle = rectangle
        self.line_width = line_width

    def render(self, draw_op: BitStream, context: "RenderContext") -> None:
        rectangle = self.rectangle
        x1 = rectangle.box.x
        y1 = rectangle.box.y
        x2 = rectangle.box.x2
        y2 = rectangle.box.y2
        width = x2 - x1
        height = y2 - y1

        draw_op.append(b"q n ")
        draw_op.append(
            rectangle.graphic_state.passthrough_per_char_instruction.encode(),
        )

        line_width = self.line_width
        if rectangle.line_width is not None:
            line_width = rectangle.line_width
        if line_width > 0:
            draw_op.append(f" {line_width:.6f} w ".encode())

        draw_op.append(f"{x1:.6f} {y1:.6f} {width:.6f} {height:.6f} re ".encode())
        if rectangle.fill_background:
            draw_op.append(b" f ")
        else:
            draw_op.append(b" S ")

        draw_op.append(b"Q\n")


class CurveRenderUnit(RenderUnit):
    """Render unit for PDF curves."""

    def __init__(
        self,
        curve: il_version_1.PdfCurve,
        render_order: int,
        sub_render_order: int = 0,
    ):
        super().__init__(render_order, sub_render_order, curve.xobj_id)
        self.curve = curve

    def render(self, draw_op: BitStream, context: "RenderContext") -> None:
        curve = self.curve
        draw_op.append(b"q n ")

        # Apply relocation transform first if present (before passthrough instructions)
        # This ensures masks in passthrough_per_char_instruction use the correct coordinate system
        if curve.relocation_transform and len(curve.relocation_transform) == 6:
            try:
                relocation_matrix = tuple(float(x) for x in curve.relocation_transform)
                draw_op.append(matrix_to_bytes(relocation_matrix))
            except (ValueError, TypeError):
                # If relocation transform conversion fails, skip it and use original CTM later
                pass

        draw_op.append(b" ")

        # Apply original CTM if present
        if curve.ctm and len(curve.ctm) == 6:
            ctm = curve.ctm
            draw_op.append(
                f"{ctm[0]:.6f} {ctm[1]:.6f} {ctm[2]:.6f} {ctm[3]:.6f} {ctm[4]:.6f} {ctm[5]:.6f} cm ".encode()
            )

        draw_op.append(b" ")

        draw_op.append(
            curve.graphic_state.passthrough_per_char_instruction.encode(),
        )

        if curve.passthrough_paint:
            draw_op.append(b" Q\n")
            return

        draw_op.append(b" ")
        path_op = BitStream(b" ")

        # Use original path if available, otherwise fall back to transformed path
        path_to_use = (
            curve.pdf_original_path
            if curve.pdf_original_path is not None
            else curve.pdf_path
        )
        if not self._append_original_primitive_path(path_op):
            for path in path_to_use:
                if isinstance(path, PdfOriginalPath):
                    path = path.pdf_path
                if path.has_xy:
                    path_op.append(f"{path.x:F} {path.y:F} {path.op} ".encode())
                else:
                    path_op.append(f"{path.op} ".encode())

        if curve.fill_background:
            draw_op.append(path_op)
            draw_op.append(b" f")
        if curve.evenodd:
            draw_op.append(b"* ")
        else:
            draw_op.append(b" ")
        if curve.stroke_path:
            draw_op.append(path_op)
            draw_op.append(b"S ")

        # final_op = b' B '

        draw_op.append(b" n Q\n")

    def _append_original_primitive_path(self, path_op: BitStream) -> bool:
        primitive = self.curve.pdf_original_path_primitive
        if primitive is None or primitive.op != "re" or len(primitive.args) != 4:
            return False
        try:
            x, y, width, height = (float(arg) for arg in primitive.args)
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(arg) for arg in (x, y, width, height)):
            return False
        path_op.append(f"{x:F} {y:F} {width:F} {height:F} re ".encode())
        return True


class RenderContext:
    """Context object containing shared state for rendering."""

    def __init__(
        self,
        pdf_creator: "PDFCreater",
        page: il_version_1.Page,
        available_font_list: set[str],
        page_encoding_length_map: dict[str, int],
        all_encoding_length_map: dict[str, int],
        xobj_available_fonts: dict[str, set[str]],
        xobj_encoding_length_map: dict[str, dict[str, int]],
        page_font_map: dict[str, il_version_1.PdfFont],
        xobj_font_map: dict[str, dict[str, il_version_1.PdfFont]],
        ctm_for_ops: bytes,
        check_font_exists: bool = False,
    ):
        self.pdf_creator = pdf_creator
        self.page = page
        self.available_font_list = available_font_list
        self.page_encoding_length_map = page_encoding_length_map
        self.all_encoding_length_map = all_encoding_length_map
        self.xobj_available_fonts = xobj_available_fonts
        self.xobj_encoding_length_map = xobj_encoding_length_map
        self.page_font_map = page_font_map
        self.xobj_font_map = xobj_font_map
        self.ctm_for_ops = ctm_for_ops
        self.check_font_exists = check_font_exists

    def get_scoped_font(
        self,
        font_id: str,
        xobj_id: str | None = None,
    ) -> il_version_1.PdfFont | None:
        if xobj_id in self.xobj_font_map:
            scoped_font = self.xobj_font_map[xobj_id].get(font_id)
            if scoped_font is not None:
                return scoped_font
        return self.page_font_map.get(font_id)


def to_int(src):
    return int(re.search(r"\d+", src).group(0))


def parse_mapping(text):
    mapping = []
    for x in re.finditer(rb"<(?P<num>[a-fA-F0-9]+)>", text):
        mapping.append(int(x.group("num"), 16))
    return mapping


def apply_normalization(cmap, gid, code):
    need = False
    if 0x2F00 <= code <= 0x2FD5:  # Kangxi Radicals
        need = True
    if 0xF900 <= code <= 0xFAFF:  # CJK Compatibility Ideographs
        need = True
    if need:
        norm = unicodedata.normalize("NFD", chr(code))
        cmap[gid] = ord(norm)
    else:
        cmap[gid] = code


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def update_tounicode_cmap_pair(cmap, data):
    for start, stop, value in batched(data, 3):
        for gid in range(start, stop + 1):
            code = value + gid - start
            apply_normalization(cmap, gid, code)


def update_tounicode_cmap_code(cmap, data):
    for gid, code in batched(data, 2):
        apply_normalization(cmap, gid, code)


def parse_tounicode_cmap(data):
    cmap = {}
    for x in re.finditer(
        rb"\s+beginbfrange\s*(?P<r>(<[0-9a-fA-F]+>\s*)+)endbfrange\s+", data
    ):
        update_tounicode_cmap_pair(cmap, parse_mapping(x.group("r")))
    for x in re.finditer(
        rb"\s+beginbfchar\s*(?P<c>(<[0-9a-fA-F]+>\s*)+)endbfchar", data
    ):
        update_tounicode_cmap_code(cmap, parse_mapping(x.group("c")))
    return cmap


def parse_truetype_data(data):
    glyph_in_use = []
    face = freetype.Face(io.BytesIO(data))
    for i in range(face.num_glyphs):
        face.load_glyph(i)
        if face.glyph.outline.contours:
            glyph_in_use.append(i)
    return glyph_in_use


TOUNICODE_HEAD = """\
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo <</Registry(Adobe)/Ordering(UCS)/Supplement 0>> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange"""
TOUNICODE_TAIL = """\
endcmap
CMapName currentdict /CMap defineresource pop
end
end"""


def make_tounicode(cmap, used):
    short = []
    for x in used:
        if x in cmap:
            short.append((x, cmap[x]))
    line = [TOUNICODE_HEAD]
    for block in batched(short, 100):
        line.append(f"{len(block)} beginbfchar")
        for glyph, code in block:
            if code < 0x10000:
                line.append(f"<{glyph:04x}><{code:04x}>")
            else:
                code -= 0x10000
                high = 0xD800 + (code >> 10)
                low = 0xDC00 + (code & 0b1111111111)
                line.append(f"<{glyph:04x}><{high:04x}{low:04x}>")
        line.append("endbfchar")
    line.append(TOUNICODE_TAIL)
    return "\n".join(line)


def reproduce_one_font(doc, index):
    m = doc.xref_get_key(index, "ToUnicode")
    f = doc.xref_get_key(index, "DescendantFonts")
    if m[0] == "xref" and f[0] == "array":
        mi = to_int(m[1])
        fi = to_int(f[1])
        ff = doc.xref_get_key(fi, "FontDescriptor/FontFile2")
        ms = doc.xref_stream(mi)
        fs = doc.xref_stream(to_int(ff[1]))
        cmap = parse_tounicode_cmap(ms)
        used = parse_truetype_data(fs)
        text = make_tounicode(cmap, used)
        doc.update_stream(mi, bytes(text, "U8"))


def reproduce_cmap(doc):
    assert doc
    font_set = set()
    for page in doc:
        try:
            font_list = page.get_fonts()
            for font in font_list:
                if font[1] == "ttf" and font[3] in FONT_NAMES and ".ttf" in font[4]:
                    font_set.add(font)
        except Exception as e:
            logger.error(f"Error in getting page fonts: {e}")
    for font in font_set:
        reproduce_one_font(doc, font[0])
    normalize_arabic_text_layer(doc)
    return doc


# --- The Arabic text layer ---------------------------------------------------
#
# Arabic is DRAWN as presentation forms: the typesetter substitutes each letter
# with its contextual shape (U+FEDF for an initial lam, U+FEFB for the lam-alef
# ligature) so the per-character renderer can place one glyph at a time. That is
# a rendering decision, and it must not leak into the text layer — but it does,
# because the ToUnicode CMap the viewer builds is a reverse of the font's cmap,
# and the font's cmap is what we looked the shape up in. The delivered file then
# LOOKS right and IS unsearchable: Ctrl+F for a normally typed word matches
# nothing, a copy-paste pastes shapes that will not re-shape anywhere else, a
# screen reader gets glyph soup, and `pdftotext` — which is what indexes these
# documents downstream — sees the same.
#
# So after the pages are drawn we walk every ToUnicode CMap and put the letters
# back: each presentation form becomes the letter(s) a human would type, which
# for a ligature means TWO characters out of one glyph (U+FEFB -> "لا"). Nothing
# about the drawing changes; only what the glyph claims to be.
#
# The second half is glyphs the CMap never mentioned at all. A shaper reaches
# them through GSUB, and GSUB output has no cmap entry to be reversed from — so
# the viewer falls back to using the glyph id as if it were a codepoint, and a
# lam extracts as "Ʌ". Those we recover from the font program: the substitution
# tables say which glyph a glyph came FROM, and the letter travels along that
# edge.

# Arabic Presentation Forms-A and -B. U+FEFD..U+FEFF are excluded on purpose:
# FEFF is the byte-order mark, not a letter shape.
_ARABIC_PRESENTATION_RANGES = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFC))

# A CMap destination is UTF-16BE hex; a range's destination may also be an array
# of them, one per code in the range.
_CMAP_TOKEN_RE = re.compile(rb"<([0-9A-Fa-f]*)>|(\[)|(\])")
_BFCHAR_RE = re.compile(rb"beginbfchar(.*?)endbfchar", re.DOTALL)
_BFRANGE_RE = re.compile(rb"beginbfrange(.*?)endbfrange", re.DOTALL)
_CMAP_BODY_RE = re.compile(
    rb"(endcodespacerange)(.*?)(endcmap)",
    re.DOTALL,
)

# Filling CMap gaps from the font program only makes sense for a real subset.
# A retain-gids subset of a 64k-glyph face carries every glyph the face ever
# had, and blindly naming all of them would add tens of thousands of entries
# for glyphs the page never draws.
_MAX_SUBSET_GLYPHS_FOR_GAP_FILL = 8192


def is_arabic_presentation_form(code: int) -> bool:
    return any(low <= code <= high for low, high in _ARABIC_PRESENTATION_RANGES)


def arabic_base_letters(code: int) -> str:
    """The letter(s) a human types for the presentation form `code`.

    One glyph may be two characters: U+FEFB (lam-alef) is "لا". The marks in
    U+FE70..U+FE7F decompose to a space plus the mark — the space is the
    placeholder the shape is drawn on, not text, so it goes.
    """
    char = chr(code)
    folded = unicodedata.normalize("NFKC", char)
    if folded.startswith(" ") and len(folded) > 1:
        folded = folded[1:]
    return folded or char


def normalize_arabic_presentation_forms(text: str) -> str:
    """`text` with every Arabic presentation form replaced by its letters."""
    if not any(is_arabic_presentation_form(ord(char)) for char in text):
        return text
    return "".join(
        arabic_base_letters(ord(char)) if is_arabic_presentation_form(ord(char))
        else char
        for char in text
    )


def _cmap_tokens(section: bytes):
    """`(kind, value)` for each `<hex>` / `[` / `]` in a CMap section."""
    for match in _CMAP_TOKEN_RE.finditer(section):
        if match.group(1) is not None:
            yield "hex", match.group(1).decode("ascii").lower()
        elif match.group(2) is not None:
            yield "[", None
        else:
            yield "]", None


def _hex_to_text(raw: str) -> str | None:
    """A CMap destination as a string, or None if it is not readable as one.

    Whole 16-bit units are UTF-16BE, which is what the spec asks for. Anything
    else is a writer that put the scalar codepoint in directly (MuPDF emits
    `<10780>` for U+10780); read it that way rather than refusing the font.
    """
    if not raw:
        return None
    try:
        if len(raw) % 4 == 0:
            return bytes.fromhex(raw).decode("utf-16-be")
        code = int(raw, 16)
    except (ValueError, UnicodeDecodeError):
        return None
    if code > 0x10FFFF or 0xD800 <= code <= 0xDFFF:
        return None
    return chr(code)


def parse_tounicode_map(data: bytes) -> tuple[dict[int, tuple[str, str]], int] | None:
    """`(code -> (text, destination hex), code width)`, or None if unreadable.

    Deliberately all-or-nothing. A CMap we only half understand is one we must
    not rewrite, because rewriting it would drop whatever we failed to parse.
    The raw destination is carried along so an entry we do not change can be
    written back exactly as it was found.
    """
    mapping: dict[int, tuple[str, str]] = {}
    nibbles = 4

    def record(code: int, dst: str) -> bool:
        text = _hex_to_text(dst)
        if text is None:
            return False
        mapping[code] = (text, dst)
        return True

    for match in _BFCHAR_RE.finditer(data):
        tokens = list(_cmap_tokens(match.group(1)))
        if len(tokens) % 2 or any(kind != "hex" for kind, _ in tokens):
            return None
        for (_, src), (_, dst) in batched(tokens, 2):
            nibbles = max(nibbles, len(src))
            if not record(int(src, 16), dst):
                return None

    for match in _BFRANGE_RE.finditer(data):
        tokens = list(_cmap_tokens(match.group(1)))
        index = 0
        while index < len(tokens):
            if len(tokens) - index < 3:
                return None
            (lo_kind, lo), (hi_kind, hi) = tokens[index], tokens[index + 1]
            if lo_kind != "hex" or hi_kind != "hex":
                return None
            low, high = int(lo, 16), int(hi, 16)
            if high < low or high - low > 0xFFFF:
                return None
            nibbles = max(nibbles, len(lo), len(hi))
            index += 2
            kind, value = tokens[index]
            if kind == "hex":
                # A range counts its destination up with the code: the last
                # unit increments, which for a hex string is the whole number.
                start = int(value, 16)
                for code in range(low, high + 1):
                    if not record(code, f"{start + code - low:0{len(value)}x}"):
                        return None
                index += 1
            elif kind == "[":
                index += 1
                code = low
                while index < len(tokens) and tokens[index][0] == "hex":
                    if code <= high and not record(code, tokens[index][1]):
                        return None
                    code += 1
                    index += 1
                if index >= len(tokens) or tokens[index][0] != "]":
                    return None
                index += 1
            else:
                return None

    return mapping, nibbles


def _tounicode_runs(mapping: dict[int, tuple[str, str]]):
    """`mapping` split into `(code, destination)` singles and countable runs.

    A CMap that spelled a whole font out one entry at a time would be several
    times the size of the one we read, and most of a font's map is Latin or
    CJK that still counts up perfectly — it is only the Arabic that stops
    doing so, because many shapes collapse onto the same letter. So keep the
    ranges where they still hold. Per the spec only the LAST byte of a
    destination counts up, so a run never crosses a 256 boundary.
    """
    items = sorted(mapping.items())
    index = 0
    while index < len(items):
        code, (text, dst) = items[index]
        length = 1
        if len(text) == 1 and ord(text) < 0x10000:
            start = int(dst, 16)
            limit = 0xFF - (start & 0xFF)
            while (index + length < len(items)
                   and length <= limit
                   and items[index + length][0] == code + length
                   and len(items[index + length][1][0]) == 1
                   and int(items[index + length][1][1], 16) == start + length
                   and len(items[index + length][1][1]) == len(dst)):
                length += 1
        yield code, dst, length
        index += length


def render_tounicode_body(mapping: dict[int, tuple[str, str]], nibbles: int) -> str:
    """`mapping` as CMap sections — the body between codespace and endcmap."""
    singles, ranges = [], []
    for code, dst, length in _tounicode_runs(mapping):
        if length > 1:
            ranges.append(
                f"<{code:0{nibbles}x}><{code + length - 1:0{nibbles}x}><{dst}>"
                .upper())
        else:
            singles.append(f"<{code:0{nibbles}x}><{dst}>".upper())

    lines = []
    for keyword, entries in (("bfchar", singles), ("bfrange", ranges)):
        for block in batched(entries, 100):
            lines.append(f"{len(block)} begin{keyword}")
            lines.extend(block)
            lines.append(f"end{keyword}")
    return "\n".join(lines)


def _glyph_unicode_from_font_program(data: bytes) -> dict[int, str]:
    """`glyph id -> the text it stands for`, read out of an embedded font.

    Two passes, in the order a viewer would want them. The cmap gives the
    codepoint each glyph is reached by directly (the highest one, which is how
    the reverse map a viewer builds resolves a tie). Then the substitution
    tables carry that text along every single-substitution and ligature edge,
    which is the only way to name a glyph a shaper produced — those have no
    cmap entry at all, and are exactly the glyphs that extract as mojibake.
    """
    from fontTools.ttLib import TTFont

    font = TTFont(io.BytesIO(data), lazy=True, fontNumber=0)
    try:
        order = font.getGlyphOrder()
        by_name: dict[str, str] = {}
        if "cmap" in font:
            highest: dict[str, int] = {}
            for code, name in font.getBestCmap().items():
                if highest.get(name, -1) < code:
                    highest[name] = code
            by_name = {
                name: normalize_arabic_presentation_forms(chr(code))
                for name, code in highest.items()
            }

        singles, ligatures = [], []
        if "GSUB" in font and font["GSUB"].table.LookupList:
            for lookup in font["GSUB"].table.LookupList.Lookup:
                for subtable in lookup.SubTable or ():
                    kind = lookup.LookupType
                    if kind == 7:  # extension: the real subtable is inside
                        kind = subtable.ExtensionLookupType
                        subtable = subtable.ExtSubTable
                    if kind == 1 and getattr(subtable, "mapping", None):
                        singles.append(subtable.mapping)
                    elif kind == 4 and getattr(subtable, "ligatures", None):
                        ligatures.append(subtable.ligatures)

        # A substitution chain can be several hops long (isolated -> initial ->
        # a stylistic alternate), so run to a fixpoint rather than once.
        for _ in range(8):
            changed = False
            for mapping in singles:
                for source, target in mapping.items():
                    if target not in by_name and source in by_name:
                        by_name[target] = by_name[source]
                        changed = True
            for ligature_sets in ligatures:
                for first, records in ligature_sets.items():
                    if first not in by_name:
                        continue
                    for record in records:
                        parts = [by_name[first]]
                        parts.extend(
                            by_name.get(component, "")
                            for component in record.Component
                        )
                        if record.LigGlyph not in by_name and all(parts):
                            by_name[record.LigGlyph] = "".join(parts)
                            changed = True
            if not changed:
                break

        return {
            gid: by_name[name]
            for gid, name in enumerate(order)
            if name in by_name and by_name[name]
        }
    finally:
        font.close()


def _font_program_bytes(doc, font_xref: int) -> bytes | None:
    """The TrueType program behind a Type0 font, if it is embedded plainly."""
    descendants = doc.xref_get_key(font_xref, "DescendantFonts")
    if descendants[0] != "array":
        return None
    try:
        descendant = to_int(descendants[1])
        # A CIDToGIDMap stream means the code in the stream is not the glyph
        # id, and every gap we filled would land on the wrong glyph.
        cid_to_gid = doc.xref_get_key(descendant, "CIDToGIDMap")
        if cid_to_gid[0] not in ("null", "name") or (
            cid_to_gid[0] == "name" and cid_to_gid[1] != "/Identity"
        ):
            return None
        font_file = doc.xref_get_key(descendant, "FontDescriptor/FontFile2")
        if font_file[0] != "xref":
            return None
        return doc.xref_stream(to_int(font_file[1]))
    except Exception:  # noqa: BLE001 - an unreadable font is simply skipped
        return None


def _normalize_one_font(doc, font_xref: int) -> bool:
    """Put the letters back in one font's ToUnicode. True if it changed."""
    to_unicode = doc.xref_get_key(font_xref, "ToUnicode")
    if to_unicode[0] != "xref":
        return False
    stream_xref = to_int(to_unicode[1])
    data = doc.xref_stream(stream_xref)
    if not data:
        return False

    parsed = parse_tounicode_map(data)
    if parsed is None:
        logger.debug("cmap: font %s has a ToUnicode we cannot rewrite", font_xref)
        return False
    mapping, nibbles = parsed

    # The pass exists for Arabic. A CMap with no presentation form in it is
    # left byte-identical, which is what keeps this safe to run over the
    # original document's own fonts.
    if not any(
        is_arabic_presentation_form(ord(char))
        for text, _ in mapping.values()
        for char in text
    ):
        return False

    updated: dict[int, tuple[str, str]] = {}
    for code, (text, dst) in mapping.items():
        folded = normalize_arabic_presentation_forms(text)
        updated[code] = (
            (text, dst) if folded == text
            else (folded, folded.encode("utf-16-be").hex())
        )

    program = _font_program_bytes(doc, font_xref)
    if program:
        try:
            glyphs = _glyph_unicode_from_font_program(program)
        except Exception:  # noqa: BLE001 - a font we cannot read is not fatal
            logger.debug("cmap: font %s program is unreadable", font_xref,
                         exc_info=True)
            glyphs = {}
        # Only a genuine subset: see _MAX_SUBSET_GLYPHS_FOR_GAP_FILL.
        if glyphs and len(glyphs) <= _MAX_SUBSET_GLYPHS_FOR_GAP_FILL:
            for gid, text in glyphs.items():
                if gid not in updated:
                    updated[gid] = (text, text.encode("utf-16-be").hex())

    if updated == mapping:
        return False

    body = render_tounicode_body(updated, nibbles)
    replaced, count = _CMAP_BODY_RE.subn(
        lambda match: match.group(1) + b"\n" + body.encode("ascii") + b"\n"
        + match.group(3),
        data,
        count=1,
    )
    if not count:
        return False
    doc.update_stream(stream_xref, replaced)
    return True


def normalize_arabic_text_layer(doc) -> int:
    """Rewrite every ToUnicode CMap in `doc` so Arabic extracts as letters.

    Returns how many fonts were rewritten. A document with no Arabic
    presentation forms in any CMap comes out untouched.
    """
    repaired = 0
    for xref in range(1, doc.xref_length()):
        try:
            if doc.xref_get_key(xref, "Type")[1] != "/Font":
                continue
            repaired += _normalize_one_font(doc, xref)
        except Exception:  # noqa: BLE001 - never lose a document over its text layer
            logger.debug("cmap: could not normalize font %s", xref, exc_info=True)
    if repaired:
        logger.debug("cmap: rewrote %d Arabic ToUnicode map(s)", repaired)
    return repaired


def _subset_fonts_process(pdf_path, output_path):
    """Function to run in subprocess for font subsetting.

    Args:
        pdf_path: Path to the PDF file to subset
        output_path: Path where to save the result
    """
    try:
        pdf = pymupdf.open(pdf_path)
        pdf.subset_fonts(fallback=False)
        pdf.save(output_path)
        # 返回 0 表示成功
        os._exit(0)
    except Exception as e:
        logger.error(f"Error in font subsetting subprocess: {e}")
        # 返回 1 表示失败
        os._exit(1)


def _save_pdf_clean_process(
    pdf_path,
    output_path,
    garbage=1,
    deflate=True,
    clean=True,
    deflate_fonts=True,
    linear=False,
):
    """Function to run in subprocess for saving PDF with clean=True which can be time-consuming.

    Args:
        pdf_path: Path to the PDF file to save
        output_path: Path where to save the result
        garbage: Garbage collection level (0, 1, 2, 3, 4)
        deflate: Whether to deflate the PDF
        clean: Whether to clean the PDF
        deflate_fonts: Whether to deflate fonts
        linear: Whether to linearize the PDF
    """
    try:
        pdf = pymupdf.open(pdf_path)
        pdf.save(
            output_path,
            garbage=garbage,
            deflate=deflate,
            clean=clean,
            deflate_fonts=deflate_fonts,
            linear=linear,
        )
        # 返回 0 表示成功
        os._exit(0)
    except Exception as e:
        logger.error(f"Error in save PDF with clean=True subprocess: {e}")
        # 返回 1 表示失败
        os._exit(1)


class PDFCreater:
    stage_name = "Generate drawing instructions"

    def __init__(
        self,
        original_pdf_path: str,
        document: il_version_1.Document,
        translation_config: TranslationConfig,
        mediabox_data: dict,
    ):
        self.original_pdf_path = original_pdf_path
        self.docs = document
        self.font_path = translation_config.font
        self.font_mapper = FontMapper(translation_config)
        self.translation_config = translation_config
        self.mediabox_data = mediabox_data

    def render_graphic_state(
        self,
        draw_op: BitStream,
        graphic_state: il_version_1.GraphicState,
    ):
        if graphic_state is None:
            return
        # if graphic_state.stroking_color_space_name:
        #     draw_op.append(
        #         f"/{graphic_state.stroking_color_space_name} CS \n".encode()
        #     )
        # if graphic_state.non_stroking_color_space_name:
        #     draw_op.append(
        #         f"/{graphic_state.non_stroking_color_space_name}"
        #         f" cs \n".encode()
        #     )
        # if graphic_state.ncolor is not None:
        #     if len(graphic_state.ncolor) == 1:
        #         draw_op.append(f"{graphic_state.ncolor[0]} g \n".encode())
        #     elif len(graphic_state.ncolor) == 3:
        #         draw_op.append(
        #             f"{' '.join((str(x) for x in graphic_state.ncolor))} sc \n".encode()
        #         )
        # if graphic_state.scolor is not None:
        #     if len(graphic_state.scolor) == 1:
        #         draw_op.append(f"{graphic_state.scolor[0]} G \n".encode())
        #     elif len(graphic_state.scolor) == 3:
        #         draw_op.append(
        #             f"{' '.join((str(x) for x in graphic_state.scolor))} SC \n".encode()
        #         )

        if graphic_state.passthrough_per_char_instruction:
            draw_op.append(
                f"{graphic_state.passthrough_per_char_instruction} \n".encode(),
            )

    @staticmethod
    def _parse_xref_ref(value: str) -> int | None:
        match = re.match(r"^\s*(\d+)\s+0\s+R\s*$", value)
        if match is None:
            return None
        return int(match.group(1))

    @classmethod
    def _find_resource_ref(
        cls,
        pdf: pymupdf.Document,
        candidate_resource_xrefs: list[int],
        resource_kind: str,
        name: str,
    ) -> str | None:
        for candidate_xref in candidate_resource_xrefs:
            try:
                value_type, value = pdf.xref_get_key(
                    candidate_xref,
                    f"Resources/{resource_kind}/{name}",
                )
            except Exception:
                continue
            if value_type != "null" and value != "null":
                return value
        return None

    @classmethod
    def _set_resource_ref(
        cls,
        pdf: pymupdf.Document,
        target_xref: int,
        resource_kind: str,
        name: str,
        resource_ref: str,
    ) -> None:
        resource_key = f"Resources/{resource_kind}/{name}"
        try:
            pdf.xref_set_key(target_xref, resource_key, resource_ref)
            return
        except Exception as exc:
            if "has indirects" not in str(exc):
                raise
            direct_set_error = exc

        resources_type, resources_value = pdf.xref_get_key(
            target_xref,
            "Resources",
        )
        resources_xref = (
            cls._parse_xref_ref(resources_value) if resources_type == "xref" else None
        )
        if resources_xref is None:
            raise direct_set_error

        kind_type, kind_value = pdf.xref_get_key(resources_xref, resource_kind)
        kind_xref = cls._parse_xref_ref(kind_value) if kind_type == "xref" else None
        if kind_xref is not None:
            pdf.xref_set_key(kind_xref, name, resource_ref)
            return

        pdf.xref_set_key(
            resources_xref,
            f"{resource_kind}/{name}",
            resource_ref,
        )

    @classmethod
    def _ensure_stream_named_resources(
        cls,
        pdf: pymupdf.Document,
        target_xref: int,
        stream: bytes,
        candidate_resource_xrefs: list[int],
        *,
        resource_kind: str,
        usage_re: re.Pattern[bytes],
    ) -> None:
        used_names = {
            match.group(1).decode("latin-1") for match in usage_re.finditer(stream)
        }
        if not used_names:
            return

        for name in sorted(used_names):
            try:
                current_type, current_value = pdf.xref_get_key(
                    target_xref,
                    f"Resources/{resource_kind}/{name}",
                )
            except Exception:
                current_type, current_value = "null", "null"
            if current_type != "null" and current_value != "null":
                continue

            resource_ref = cls._find_resource_ref(
                pdf,
                candidate_resource_xrefs,
                resource_kind,
                name,
            )
            if resource_ref is None:
                continue
            cls._set_resource_ref(
                pdf,
                target_xref,
                resource_kind,
                name,
                resource_ref,
            )

    @classmethod
    def _ensure_stream_extgstate_resources(
        cls,
        pdf: pymupdf.Document,
        target_xref: int,
        stream: bytes,
        candidate_resource_xrefs: list[int],
    ) -> None:
        cls._ensure_stream_named_resources(
            pdf,
            target_xref,
            stream,
            candidate_resource_xrefs,
            resource_kind="ExtGState",
            usage_re=_EXTGSTATE_USAGE_RE,
        )

    @classmethod
    def _ensure_stream_shading_resources(
        cls,
        pdf: pymupdf.Document,
        target_xref: int,
        stream: bytes,
        candidate_resource_xrefs: list[int],
    ) -> None:
        cls._ensure_stream_named_resources(
            pdf,
            target_xref,
            stream,
            candidate_resource_xrefs,
            resource_kind="Shading",
            usage_re=_SHADING_USAGE_RE,
        )

    def render_paragraph_to_char(
        self,
        paragraph: il_version_1.PdfParagraph,
    ) -> list[il_version_1.PdfCharacter]:
        chars = []
        for composition in paragraph.pdf_paragraph_composition:
            if composition.pdf_character:
                chars.append(composition.pdf_character)
            elif composition.pdf_formula:
                # Flatten formula: extract all characters from the formula
                chars.extend(composition.pdf_formula.pdf_character)
            else:
                logger.error(
                    f"Unknown composition type. "
                    f"This type only appears in the IL "
                    f"after the translation is completed."
                    f"During pdf rendering, this type is not supported."
                    f"Composition: {composition}. "
                    f"Paragraph: {paragraph}. ",
                )
                continue
        if not chars and paragraph.unicode and paragraph.debug_id:
            logger.error(
                f"Unable to export paragraphs that have "
                f"not yet been formatted: {paragraph}",
            )
            return chars
        return chars

    def create_render_units_for_page(
        self,
        page: il_version_1.Page,
        translation_config: TranslationConfig,
    ) -> list[RenderUnit]:
        """Convert all renderable objects in a page to render units."""
        render_units = []

        # Collect all characters (from page and paragraphs)
        chars = []
        if page.pdf_character:
            chars.extend(page.pdf_character)
        for paragraph in page.pdf_paragraph:
            chars.extend(self.render_paragraph_to_char(paragraph))

        # Convert characters to render units
        for i, char in enumerate(chars):
            render_order = getattr(char, "render_order", 100)  # Default render order
            sub_render_order = getattr(char, "sub_render_order", i)
            render_units.append(
                CharacterRenderUnit(char, render_order, sub_render_order)
            )

        # Collect forms from formulas within paragraphs
        formula_forms = []
        for paragraph in page.pdf_paragraph:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_formula:
                    formula_forms.extend(composition.pdf_formula.pdf_form)

        # Convert forms to render units (page-level forms + forms from formulas)
        if not translation_config.skip_form_render:
            all_forms = list(page.pdf_form) + formula_forms
            for i, form in enumerate(all_forms):
                render_order = getattr(
                    form, "render_order", 50
                )  # Forms render before characters
                sub_render_order = getattr(form, "sub_render_order", i)
                render_units.append(
                    FormRenderUnit(form, render_order, sub_render_order)
                )

        # Convert rectangles to render units (only for OCR workaround, image
        # -text masks, or debug). An image-text mask (it carries its raster
        # region) must render on digital pages too: it covers the source
        # label pixels inside an embedded raster image.
        for i, rect in enumerate(page.pdf_rectangle):
            is_image_text_mask = bool(getattr(rect, "raster_region", None))
            if (
                (translation_config.ocr_workaround or is_image_text_mask)
                and not rect.debug_info
                and rect.fill_background
            ) or (translation_config.debug and rect.debug_info):
                render_order = getattr(
                    rect, "render_order", 10
                )  # Rectangles render first
                sub_render_order = getattr(rect, "sub_render_order", i)
                line_width = 0.1 if translation_config.ocr_workaround else 0.4
                render_units.append(
                    RectangleRenderUnit(
                        rect, render_order, sub_render_order, line_width
                    )
                )

        # Collect curves from formulas within paragraphs
        formula_curves = []
        for paragraph in page.pdf_paragraph:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_formula:
                    formula_curves.extend(composition.pdf_formula.pdf_curve)

        # Convert curves to render units (page-level curves + curves from formulas, only for debug)
        if not translation_config.skip_curve_render:
            all_curves = list(page.pdf_curve) + formula_curves
            for i, curve in enumerate(all_curves):
                if (
                    curve.passthrough_paint
                    or curve.debug_info
                    or translation_config.debug
                ):
                    render_order = getattr(
                        curve, "render_order", 20
                    )  # Curves render after rectangles
                    sub_render_order = getattr(curve, "sub_render_order", i)
                    render_units.append(
                        CurveRenderUnit(curve, render_order, sub_render_order)
                    )

        return render_units

    def render_units_to_stream(
        self,
        render_units: list[RenderUnit],
        context: RenderContext,
        page_op: BitStream,
        xobj_draw_ops: dict[str, BitStream],
    ) -> None:
        """Render sorted render units to appropriate draw streams."""
        # Sort render units by (render_order, sub_render_order)
        sorted_units = sorted(render_units, key=lambda unit: unit.get_sort_key())

        for unit in sorted_units:
            # Determine which draw_op to use based on xobj_id
            if unit.xobj_id in xobj_draw_ops:
                draw_op = xobj_draw_ops[unit.xobj_id]
            else:
                draw_op = page_op

            # Render the unit
            unit.render(draw_op, context)

    def get_available_font_list(self, pdf, page):
        page_xref_id = pdf[page.page_number].xref
        return self.get_xobj_available_fonts(page_xref_id, pdf)

    def get_xobj_available_fonts(self, page_xref_id, pdf):
        try:
            resources_type, r_id = pdf.xref_get_key(page_xref_id, "Resources")
            if resources_type == "xref":
                resource_xref_id = re.search("(\\d+) 0 R", r_id).group(1)
                r_id = pdf.xref_object(int(resource_xref_id))
                resources_type = "dict"
            if resources_type == "dict":
                xref_id = re.search("/Font (\\d+) 0 R", r_id)
                if xref_id is not None:
                    xref_id = xref_id.group(1)
                    font_dict = pdf.xref_object(int(xref_id))
                else:
                    search = re.search("/Font *<<(.+?)>>", r_id.replace("\n", " "))
                    if search is None:
                        # Have resources but no fonts
                        return set()
                    font_dict = search.group(1)
            else:
                r_id = int(r_id.split(" ")[0])
                _, font_dict = pdf.xref_get_key(r_id, "Font")
            fonts = re.findall("/([^ ]+?) ", font_dict)
            return set(fonts)
        except Exception:
            return set()

    def _render_rectangle(
        self,
        draw_op: BitStream,
        rectangle: il_version_1.PdfRectangle,
        line_width: float = 0.4,
    ):
        """Draw a rectangle in PDF for visualization purposes.

        Args:
            draw_op: BitStream to append PDF drawing operations
            rectangle: Rectangle object containing position information
            line_width: Line width
        """
        x1 = rectangle.box.x
        y1 = rectangle.box.y
        x2 = rectangle.box.x2
        y2 = rectangle.box.y2
        width = x2 - x1
        height = y2 - y1
        # Save graphics state
        draw_op.append(b"q ")

        # Set green color for debug visibility
        draw_op.append(
            rectangle.graphic_state.passthrough_per_char_instruction.encode(),
        )  # Green stroke
        if rectangle.line_width is not None:
            line_width = rectangle.line_width
        if line_width > 0:
            draw_op.append(f" {line_width:.6f} w ".encode())  # Line width
        draw_op.append(f"{x1:.6f} {y1:.6f} {width:.6f} {height:.6f} re ".encode())
        if rectangle.fill_background:
            draw_op.append(b" f ")
        else:
            draw_op.append(b" S ")

        # Restore graphics state
        draw_op.append(b" n Q\n")

    def create_side_by_side_dual_pdf(
        self,
        original_pdf: pymupdf.Document,
        translated_pdf: pymupdf.Document,
        dual_out_path: str,
        translation_config: TranslationConfig,
    ) -> pymupdf.Document:
        """Create a dual PDF with side-by-side pages (original and translation).

        Args:
            original_pdf: Original PDF document
            translated_pdf: Translated PDF document
            dual_out_path: Output path for the dual PDF
            translation_config: Translation configuration

        Returns:
            The created dual PDF document
        """
        # Create a new PDF for side-by-side pages
        dual = pymupdf.open()
        page_count = min(original_pdf.page_count, translated_pdf.page_count)

        for page_id in range(page_count):
            # Get pages from both PDFs
            orig_page = original_pdf[page_id]
            trans_page = translated_pdf[page_id]
            rotate_angle = orig_page.rotation
            total_width = orig_page.rect.width + trans_page.rect.width
            max_height = max(orig_page.rect.height, trans_page.rect.height)
            left_width = (
                orig_page.rect.width
                if not translation_config.dual_translate_first
                else trans_page.rect.width
            )

            orig_page.set_rotation(0)
            trans_page.set_rotation(0)

            # Create new page with combined width
            dual_page = dual.new_page(width=total_width, height=max_height)

            # Define rectangles for left and right sides
            rect_left = pymupdf.Rect(0, 0, left_width, max_height)
            rect_right = pymupdf.Rect(left_width, 0, total_width, max_height)

            # Show pages according to dual_translate_first setting
            if translation_config.dual_translate_first:
                # Show translated page on left and original on right
                rect_left, rect_right = rect_right, rect_left
            try:
                # Show original page on left and translated on right (default)
                dual_page.show_pdf_page(
                    rect_left,
                    original_pdf,
                    page_id,
                    keep_proportion=True,
                    rotate=-rotate_angle,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to show original page on left and translated on right (default). "
                    f"Page ID: {page_id}. "
                    f"Original PDF: {self.original_pdf_path}. "
                    f"Translated PDF: {translation_config.input_file}. ",
                    exc_info=e,
                )
            try:
                dual_page.show_pdf_page(
                    rect_right,
                    translated_pdf,
                    page_id,
                    keep_proportion=True,
                    rotate=-rotate_angle,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to show translated page on left and original on right. "
                    f"Page ID: {page_id}. "
                    f"Original PDF: {self.original_pdf_path}. "
                    f"Translated PDF: {translation_config.input_file}. ",
                    exc_info=e,
                )
        return dual

    def create_alternating_pages_dual_pdf(
        self,
        original_pdf: pymupdf.Document,
        translated_pdf: pymupdf.Document,
        translation_config: TranslationConfig,
    ) -> pymupdf.Document:
        """Create a dual PDF with alternating pages (original and translation).

        Args:
            original_pdf_path: Path to the original PDF
            translated_pdf: Translated PDF document
            translation_config: Translation configuration

        Returns:
            The created dual PDF document
        """
        # Open the original PDF and insert translated PDF
        dual = original_pdf
        dual.insert_file(translated_pdf)

        # Rearrange pages to alternate between original and translated
        page_count = translated_pdf.page_count
        for page_id in range(page_count):
            if translation_config.dual_translate_first:
                dual.move_page(page_count + page_id, page_id * 2)
            else:
                dual.move_page(page_count + page_id, page_id * 2 + 1)

        return dual

    def write_debug_info(
        self,
        pdf: pymupdf.Document,
        translation_config: TranslationConfig,
    ):
        self.font_mapper.add_font(pdf, self.docs)

        for page in self.docs.page:
            _, r_id = pdf.xref_get_key(pdf[page.page_number].xref, "Contents")
            resource_xref_id = re.search("(\\d+) 0 R", r_id).group(1)
            base_op = pdf.xref_stream(int(resource_xref_id))
            translation_config.raise_if_cancelled()
            xobj_available_fonts = {}
            xobj_draw_ops = {}
            xobj_encoding_length_map = {}
            available_font_list = self.get_available_font_list(pdf, page)

            page_encoding_length_map = {
                f.font_id: f.encoding_length for f in page.pdf_font
            }
            page_op = BitStream()
            # q {ops_base}Q 1 0 0 1 {x0} {y0} cm {ops_new}
            page_op.append(b"q ")
            if base_op is not None:
                page_op.append(base_op)
            page_op.append(b" Q ")
            page_op.append(
                f"q Q 1 0 0 1 {page.cropbox.box.x:.6f} {page.cropbox.box.y:.6f} cm \n".encode(),
            )
            # 收集所有字符
            chars = []
            # 首先添加页面级别的字符
            if page.pdf_character:
                chars.extend(page.pdf_character)
            # 然后添加段落中的字符
            for paragraph in page.pdf_paragraph:
                chars.extend(self.render_paragraph_to_char(paragraph))

            # 渲染所有字符
            for char in chars:
                if not getattr(char, "debug_info", False):
                    continue
                if char.char_unicode == "\n":
                    continue
                if char.pdf_character_id is None:
                    # dummy char
                    continue
                char_size = char.pdf_style.font_size
                font_id = char.pdf_style.font_id

                if font_id not in available_font_list:
                    continue
                draw_op = page_op
                encoding_length_map = page_encoding_length_map

                draw_op.append(b"q ")
                self.render_graphic_state(draw_op, char.pdf_style.graphic_state)
                if char.vertical:
                    draw_op.append(
                        f"BT /{font_id} {char_size:f} Tf 0 1 -1 0 {char.box.x2:f} {char.box.y:f} Tm ".encode(),
                    )
                else:
                    draw_op.append(
                        f"BT /{font_id} {char_size:f} Tf 1 0 0 1 {char.box.x:f} {char.box.y:f} Tm ".encode(),
                    )

                encoding_length = encoding_length_map[font_id]
                # pdf32000-2008 page14:
                # As hexadecimal data enclosed in angle brackets < >
                # see 7.3.4.3, "Hexadecimal Strings."
                draw_op.append(
                    f"<{char.pdf_character_id:0{encoding_length * 2}x}>".upper().encode(),
                )

                draw_op.append(b" Tj ET Q \n")
            for rect in page.pdf_rectangle:
                if not rect.debug_info:
                    continue
                self._render_rectangle(page_op, rect)
            draw_op = page_op
            # Since this is a draw instruction container,
            # no additional information is needed
            pdf.update_stream(int(resource_xref_id), draw_op.tobytes())
        translation_config.raise_if_cancelled()

        # 使用子进程进行字体子集化
        if not translation_config.skip_clean:
            pdf = self.subset_fonts_in_subprocess(pdf, translation_config, tag="debug")
        return pdf

    @staticmethod
    def subset_fonts_in_subprocess(
        pdf: pymupdf.Document, translation_config: TranslationConfig, tag: str
    ) -> pymupdf.Document:
        """Run font subsetting in a subprocess with timeout.

        Args:
            pdf: The PDF document object
            translation_config: Translation configuration

        Returns:
            Path to the PDF with subsetted fonts, or original path if subsetting failed or timed out
        """
        original_pdf = pdf
        # Create temporary file paths
        temp_input = str(
            translation_config.get_working_file_path(f"temp_subset_input_{tag}.pdf")
        )
        temp_output = str(
            translation_config.get_working_file_path(f"temp_subset_output_{tag}.pdf")
        )

        # Save PDF to temporary file without subsetting
        pdf.save(temp_input)

        # Create and start subprocess
        process = Process(target=_subset_fonts_process, args=(temp_input, temp_output))
        process.start()

        # Wait for subprocess with timeout (1 minute)
        timeout = 60  # 1 minutes in seconds
        start_time = time.time()

        while process.is_alive():
            if time.time() - start_time > timeout:
                logger.warning(
                    f"Font subsetting timeout after {timeout} seconds, terminating subprocess"
                )
                process.terminate()
                try:
                    process.join(5)  # Give it 5 seconds to clean up
                    if process.is_alive():
                        logger.warning("Subprocess did not terminate, killing it")
                        process.kill()
                        process.terminate()
                        process.kill()
                        process.terminate()
                        process.kill()
                        process.terminate()
                except Exception as e:
                    logger.error(f"Error terminating font subsetting process: {e}")

                return original_pdf

            time.sleep(0.5)  # Check every half second

        # Process completed, check exit code
        exit_code = process.exitcode
        success = exit_code == 0

        # Check if subsetting was successful
        if (
            success
            and Path(temp_output).exists()
            and Path(temp_output).stat().st_size > 0
        ):
            logger.info("Font subsetting completed successfully")
            return pymupdf.open(temp_output)
        else:
            logger.warning(
                f"Font subsetting failed with exit code {exit_code} or produced empty file"
            )
            return original_pdf

    @staticmethod
    def save_pdf_with_timeout(
        pdf: pymupdf.Document,
        output_path: str,
        translation_config: TranslationConfig,
        garbage: int = 1,
        deflate: bool = True,
        clean: bool = True,
        deflate_fonts: bool = True,
        linear: bool = False,
        timeout: int = 120,
        tag: str = "",
    ) -> bool:
        """Save a PDF document with a timeout for the clean=True operation.

        Args:
            pdf: The PDF document object
            output_path: Path where to save the PDF
            translation_config: Translation configuration
            garbage: Garbage collection level (0, 1, 2, 3, 4)
            deflate: Whether to deflate the PDF
            clean: Whether to clean the PDF
            deflate_fonts: Whether to deflate fonts
            linear: Whether to linearize the PDF
            timeout: Timeout in seconds (default: 2 minutes)

        Returns:
            True if saved with clean=True successfully, False if fallback to clean=False was used
        """
        # Create temporary file paths
        temp_input = str(
            translation_config.get_working_file_path(f"temp_save_input_{tag}.pdf")
        )
        temp_output = str(
            translation_config.get_working_file_path(f"temp_save_output_{tag}.pdf")
        )

        # Save PDF to temporary file first
        pdf.save(temp_input)

        # Try to save with clean=True in a subprocess
        process = Process(
            target=_save_pdf_clean_process,
            args=(
                temp_input,
                temp_output,
                garbage,
                deflate,
                clean,
                deflate_fonts,
                linear,
            ),
        )
        process.start()

        # Wait for subprocess with timeout
        start_time = time.time()

        while process.is_alive():
            if time.time() - start_time > timeout:
                logger.warning(
                    f"PDF save with clean={clean} timeout after {timeout} seconds, terminating subprocess"
                )
                process.terminate()
                try:
                    process.join(5)  # Give it 5 seconds to clean up
                    if process.is_alive():
                        logger.warning("Subprocess did not terminate, killing it")
                        process.kill()
                        process.terminate()
                        process.kill()
                        process.terminate()
                        process.kill()
                        process.terminate()
                except Exception as e:
                    logger.error(f"Error terminating PDF save process: {e}")

                # Fallback to save without clean parameter
                logger.info("Falling back to save with clean=False")
                try:
                    pdf.save(
                        output_path,
                        garbage=garbage,
                        deflate=deflate,
                        clean=False,
                        deflate_fonts=deflate_fonts,
                        linear=linear,
                    )
                    return False
                except Exception as e:
                    logger.error(f"Error in fallback save: {e}")
                    # Last resort: basic save
                    pdf.save(output_path)
                    return False

            time.sleep(0.5)  # Check every half second

        # Process completed, check exit code
        exit_code = process.exitcode
        success = exit_code == 0

        # Check if save was successful
        if (
            success
            and Path(temp_output).exists()
            and Path(temp_output).stat().st_size > 0
        ):
            logger.info(f"PDF save with clean={clean} completed successfully")
            # Copy the successfully created file to the target path
            try:
                import shutil

                shutil.copy2(temp_output, output_path)
                return True
            except Exception as e:
                logger.error(f"Error copying saved PDF: {e}")
                pdf.save(output_path)  # Fallback to direct save
                return False
            finally:
                Path(temp_input).unlink()
                Path(temp_output).unlink()
        else:
            logger.warning(
                f"PDF save with clean={clean} failed with exit code {exit_code} or produced empty file"
            )
            # Fallback to save without clean parameter
            try:
                pdf.save(
                    output_path,
                    garbage=garbage,
                    deflate=deflate,
                    clean=False,
                    deflate_fonts=deflate_fonts,
                    linear=linear,
                )
            except Exception as e:
                logger.error(f"Error in fallback save: {e}")
                # Last resort: basic save
                pdf.save(output_path)

            return False

    def restore_media_box(self, doc: pymupdf.Document, mediabox_data: dict) -> None:
        for xref, page_box_data in mediabox_data.items():
            for name, box in page_box_data.items():
                try:
                    doc.xref_set_key(xref, name, box)
                except Exception:
                    logger.debug(f"Error restoring media box {name} from PDF")

    def write(
        self,
        translation_config: TranslationConfig,
        check_font_exists: bool = False,
    ) -> TranslateResult:
        try:
            basename = Path(translation_config.input_file).stem
            debug_suffix = ".debug" if translation_config.debug else ""
            if (
                translation_config.watermark_output_mode
                != WatermarkOutputMode.Watermarked
            ):
                debug_suffix += ".no_watermark"
            mono_out_path = translation_config.get_output_file_path(
                f"{basename}{debug_suffix}.{translation_config.lang_out}.mono.pdf",
            )
            pdf = pymupdf.open(self.original_pdf_path)
            self.font_mapper.add_font(pdf, self.docs)
            with self.translation_config.progress_monitor.stage_start(
                self.stage_name,
                len(self.docs.page),
            ) as pbar:
                for page in self.docs.page:
                    self.update_page_content_stream(
                        check_font_exists, page, pdf, translation_config
                    )
                    pbar.advance()
            translation_config.raise_if_cancelled()
            gc_level = 1
            if self.translation_config.ocr_workaround:
                gc_level = 4
            with self.translation_config.progress_monitor.stage_start(
                SUBSET_FONT_STAGE_NAME,
                1,
            ) as pbar:
                if not translation_config.skip_clean:
                    pdf = self.subset_fonts_in_subprocess(
                        pdf, translation_config, tag="mono"
                    )

                pbar.advance()
            try:
                self.restore_media_box(pdf, self.mediabox_data)
            except Exception:
                logger.exception("restore media box failed")

            try:
                from babeldoc.format.pdf.document_il.midend.typesetting import (
                    is_rtl_lang,
                )

                if is_rtl_lang(translation_config.lang_out or ""):
                    pdf.xref_set_key(
                        pdf.pdf_catalog(),
                        "ViewerPreferences/Direction",
                        "/R2L",
                    )
            except Exception:
                logger.exception("failed to set RTL viewer preference")

            if translation_config.only_include_translated_page:
                total_page = set(range(0, len(pdf)))

                pages_to_translate = {
                    page.page_number
                    for page in self.docs.page
                    if self.translation_config.should_translate_page(
                        page.page_number + 1
                    )
                }

                should_removed_page = list(total_page - pages_to_translate)

                pdf.delete_pages(should_removed_page)

            with self.translation_config.progress_monitor.stage_start(
                SAVE_PDF_STAGE_NAME,
                2,
            ) as pbar:
                if not translation_config.no_mono:
                    if translation_config.debug:
                        translation_config.raise_if_cancelled()
                        pdf.save(
                            f"{mono_out_path}.decompressed.pdf",
                            expand=True,
                            pretty=True,
                        )
                    translation_config.raise_if_cancelled()
                    self.save_pdf_with_timeout(
                        pdf,
                        mono_out_path,
                        translation_config,
                        garbage=gc_level,
                        deflate=True,
                        clean=not translation_config.skip_clean,
                        deflate_fonts=True,
                        linear=False,
                        tag="mono",
                    )
                pbar.advance()
                dual_out_path = None
                if not translation_config.no_dual:
                    dual_out_path = translation_config.get_output_file_path(
                        f"{basename}{debug_suffix}.{translation_config.lang_out}.dual.pdf",
                    )
                    translation_config.raise_if_cancelled()
                    original_pdf = pymupdf.open(self.original_pdf_path)

                    if translation_config.debug:
                        translation_config.raise_if_cancelled()
                        try:
                            original_pdf = self.write_debug_info(
                                original_pdf, translation_config
                            )
                        except Exception:
                            logger.warning(
                                "Failed to write debug info to dual PDF",
                                exc_info=True,
                            )

                    if (
                        self.translation_config.only_include_translated_page
                        and should_removed_page
                    ):
                        original_pdf.delete_pages(should_removed_page)
                    translated_pdf = pdf

                    # Choose between alternating pages and side-by-side format
                    # Default to side-by-side if not specified
                    use_alternating_pages = (
                        translation_config.use_alternating_pages_dual
                    )

                    if use_alternating_pages:
                        # Create a dual PDF with alternating pages (original and translation)
                        dual = self.create_alternating_pages_dual_pdf(
                            original_pdf,
                            translated_pdf,
                            translation_config,
                        )
                    else:
                        # Create a dual PDF with side-by-side pages (original and translation)
                        dual = self.create_side_by_side_dual_pdf(
                            original_pdf,
                            translated_pdf,
                            dual_out_path,
                            translation_config,
                        )

                    self.save_pdf_with_timeout(
                        dual,
                        dual_out_path,
                        translation_config,
                        garbage=gc_level,
                        deflate=True,
                        clean=not translation_config.skip_clean,
                        deflate_fonts=True,
                        linear=False,
                        tag="dual",
                    )
                    if translation_config.debug:
                        translation_config.raise_if_cancelled()
                        dual.save(
                            f"{dual_out_path}.decompressed.pdf",
                            expand=True,
                            pretty=True,
                        )
                pbar.advance()
            if self.translation_config.no_mono:
                mono_out_path = None
            if self.translation_config.no_dual:
                dual_out_path = None
            auto_extracted_glossary_path = None
            if (
                self.translation_config.save_auto_extracted_glossary
                and self.translation_config.shared_context_cross_split_part.auto_extracted_glossary
            ):
                auto_extracted_glossary_path = self.translation_config.get_output_file_path(
                    f"{basename}{debug_suffix}.{translation_config.lang_out}.glossary.csv"
                )
                with auto_extracted_glossary_path.open("w", encoding="utf-8-sig") as f:
                    logger.info(
                        f"save auto extracted glossary to {auto_extracted_glossary_path}"
                    )
                    f.write(
                        self.translation_config.shared_context_cross_split_part.auto_extracted_glossary.to_csv()
                    )

            return TranslateResult(
                mono_out_path, dual_out_path, auto_extracted_glossary_path
            )
        except Exception:
            logger.exception(
                "Failed to create PDF: %s",
                translation_config.input_file,
            )
            if not check_font_exists:
                return self.write(translation_config, True)
            raise

    def update_page_content_stream(
        self, check_font_exists, page, pdf, translation_config, skip_char: bool = False
    ):
        assert page.cropbox is not None and page.cropbox.box is not None
        page_crop_box = page.cropbox.box
        ctm_for_ops = (
            1,
            0,
            0,
            1,
            -page_crop_box.x,
            -page_crop_box.y,
        )
        ctm_for_ops = f" {' '.join(f'{x:f}' for x in ctm_for_ops)} cm ".encode()
        translation_config.raise_if_cancelled()
        xobj_available_fonts = {}
        xobj_draw_ops = {}
        xobj_encoding_length_map = {}
        available_font_list = self.get_available_font_list(pdf, page)
        page_font_map = {f.font_id: f for f in page.pdf_font}
        xobj_font_map = {}
        page_encoding_length_map: dict[str | None, int | None] = {
            f.font_id: f.encoding_length for f in page.pdf_font
        }
        all_encoding_length_map = page_encoding_length_map.copy()
        for xobj in page.pdf_xobject:
            xobj_available_fonts[xobj.xobj_id] = available_font_list.copy()
            try:
                xobj_available_fonts[xobj.xobj_id].update(
                    self.get_xobj_available_fonts(xobj.xref_id, pdf),
                )
            except Exception:
                pass
            xobj_font_map[xobj.xobj_id] = page_font_map.copy()
            xobj_font_map[xobj.xobj_id].update({f.font_id: f for f in xobj.pdf_font})
            xobj_encoding_length_map[xobj.xobj_id] = {
                f.font_id: f.encoding_length for f in xobj.pdf_font
            }
            all_encoding_length_map.update(xobj_encoding_length_map[xobj.xobj_id])
            xobj_encoding_length_map[xobj.xobj_id].update(page_encoding_length_map)
            xobj_op = BitStream()
            base_op = xobj.base_operations.value
            base_op = zstd_decompress(base_op)
            xobj_op.append(base_op.encode())
            xobj_draw_ops[xobj.xobj_id] = xobj_op
        page_op = BitStream()
        # q {ops_base}Q 1 0 0 1 {x0} {y0} cm {ops_new}
        # page_op.append(b"q ")
        # base_op = page.base_operations.value
        # base_op = zstd_decompress(base_op)
        # page_op.append(base_op.encode())
        # page_op.append(b" \n")
        page_op.append(ctm_for_ops)
        page_op.append(b" \n")
        # Create render context
        context = RenderContext(
            pdf_creator=self,
            page=page,
            available_font_list=available_font_list,
            page_encoding_length_map=page_encoding_length_map,
            all_encoding_length_map=all_encoding_length_map,
            xobj_available_fonts=xobj_available_fonts,
            xobj_encoding_length_map=xobj_encoding_length_map,
            page_font_map=page_font_map,
            xobj_font_map=xobj_font_map,
            ctm_for_ops=ctm_for_ops,
            check_font_exists=check_font_exists,
        )
        # Create render units for all renderable objects
        render_units = self.create_render_units_for_page(page, translation_config)
        if skip_char:
            render_units = [
                unit
                for unit in render_units
                if not isinstance(unit, CharacterRenderUnit)
            ]
        # Render all units to their appropriate streams
        self.render_units_to_stream(render_units, context, page_op, xobj_draw_ops)
        candidate_resource_xrefs = [
            pdf[page.page_number].xref,
            *(xobj.xref_id for xobj in page.pdf_xobject),
        ]
        # Update xobject streams
        for xobj in page.pdf_xobject:
            draw_op = xobj_draw_ops[xobj.xobj_id]
            try:
                stream = draw_op.tobytes()
                self._ensure_stream_extgstate_resources(
                    pdf,
                    xobj.xref_id,
                    stream,
                    candidate_resource_xrefs,
                )
                self._ensure_stream_shading_resources(
                    pdf,
                    xobj.xref_id,
                    stream,
                    candidate_resource_xrefs,
                )
                pdf.update_stream(xobj.xref_id, stream)
            except Exception:
                logger.warning(f"update xref {xobj.xref_id} stream fail, continue")
        draw_op = page_op
        op_container = pdf.get_new_xref()
        # Since this is a draw instruction container,
        # no additional information is needed
        pdf.update_object(op_container, "<<>>")
        stream = draw_op.tobytes()
        self._ensure_stream_extgstate_resources(
            pdf,
            pdf[page.page_number].xref,
            stream,
            candidate_resource_xrefs,
        )
        self._ensure_stream_shading_resources(
            pdf,
            pdf[page.page_number].xref,
            stream,
            candidate_resource_xrefs,
        )
        pdf.update_stream(op_container, stream)
        pdf[page.page_number].set_contents(op_container)
