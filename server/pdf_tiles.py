"""Consolidate pathological repeated tiny image backgrounds without changing text.

Some slide exports paint a 5x5 texture tens of thousands of times per page.
Every placement becomes a PDF intermediate object. Replace only a contiguous,
page-covering run, and accept it only after an exact rendered comparison.
"""

import re

import pymupdf

_NUMBER = rb"[-+]?(?:\d*\.\d+|\d+\.?\d*)"
_MIN_TILES = 1000
_DPI = 216


def _outside_literal(prefix: bytes) -> bool:
    """Conservatively refuse candidates inside strings or inline-image data."""
    depth = 0
    index = 0
    while index < len(prefix):
        char = prefix[index]
        if depth:
            if char == 92:
                index += 2
                continue
            if char == 40:
                depth += 1
            elif char == 41:
                depth -= 1
        elif char == 37:
            end = prefix.find(b"\n", index)
            if end < 0:
                return False
            index = end
        elif char == 40:
            depth = 1
        elif char == 60 and prefix[index : index + 2] != b"<<":
            end = prefix.find(b">", index + 1)
            if end < 0:
                return False
            index = end
        elif prefix[index : index + 2] == b"<<":
            index += 1
        index += 1
    return depth == 0 and re.search(rb"(?<!\S)BI(?!\S)", prefix) is None


def _replace(page, content: bytes, start: int, end: int, pixmap) -> None:
    doc = page.parent
    original_xref = page.get_contents()[0]
    image_xref = page.insert_image(page.rect, pixmap=pixmap, overlay=True)
    # This comes only from tiny decorative tiles. Do not turn a texture
    # compaction into a full-page OCR request.
    doc.xref_set_key(image_xref, "CatodemyTexture", "true")
    inserted = doc.xref_stream(page.get_contents()[-1])
    doc.update_stream(original_xref, content[:start] + inserted + content[end:])
    page.set_contents(original_xref)


def _prune_unused_images(doc) -> None:
    # Slide exports share one Resources dictionary across every page. Adding
    # a replacement background to it would make every page advertise ALL
    # backgrounds, and parsers can eagerly decode those unused images.
    # Give each page its own dictionary and retain every image actually named
    # by a Do operator. Non-image resources and unusual names remain intact.
    for page in doc:
        # A form can refer to images through inherited page resources. Leave
        # such pages alone rather than guessing its resource dependencies.
        if page.get_xobjects():
            continue
        content = b"\n".join(doc.xref_stream(x) for x in page.get_contents())
        names = re.findall(
            rb"/([^\s<>(){}\[\]/%]+)(?:\s|%[^\r\n]*(?:[\r\n]|$))+Do(?![^\s<>(){}\[\]/%])",
            content,
        )
        used = {
            re.sub(
                rb"#([0-9a-fA-F]{2})", lambda m: bytes([int(m[1], 16)]), name
            ).decode("latin1")
            for name in names
        }
        node, seen = page.xref, set()
        while node not in seen:
            seen.add(node)
            kind, value = doc.xref_get_key(node, "Resources")
            if kind != "null":
                break
            parent_kind, parent = doc.xref_get_key(node, "Parent")
            if parent_kind != "xref":
                break
            node = int(parent.split()[0])
        if kind not in ("dict", "xref"):
            continue
        resource = doc.get_new_xref()
        doc.update_object(
            resource,
            doc.xref_object(int(value.split()[0])) if kind == "xref" else value,
        )
        kind, value = doc.xref_get_key(resource, "XObject")
        if kind not in ("dict", "xref"):
            continue
        objects = doc.get_new_xref()
        doc.update_object(
            objects, doc.xref_object(int(value.split()[0])) if kind == "xref" else value
        )
        for name in doc.xref_get_keys(objects):
            if name in used or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                continue
            kind, value = doc.xref_get_key(objects, name)
            if (
                kind == "xref"
                and doc.xref_get_key(int(value.split()[0]), "Subtype")[1] == "/Image"
            ):
                doc.xref_set_key(objects, name, "null")
        doc.xref_set_key(resource, "XObject", f"{objects} 0 R")
        doc.xref_set_key(page.xref, "Resources", f"{resource} 0 R")


def compact(data: bytes) -> tuple[bytes, dict]:
    report = {"pages": 0, "placements_removed": 0}
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            if page.rotation or len(page.get_contents()) != 1:
                continue
            media = page.mediabox
            if media.x0 != 0 or media.y0 != 0 or page.cropbox != media:
                continue
            tiny = [
                (image[0], image[7])
                for image in page.get_images()
                if 0 < image[2] <= 8 and 0 < image[3] <= 8
            ]
            if not tiny:
                continue
            content = doc.xref_stream(page.get_contents()[0])
            for _xref, name in tiny:
                pattern = re.compile(
                    rb"(?<!\S)q\s+"
                    + rb"\s+".join([rb"(" + _NUMBER + rb")"] * 6)
                    + rb"\s+cm\s+/"
                    + re.escape(name.encode())
                    + rb"\s+Do\s+Q(?!\S)"
                )
                matches = list(pattern.finditer(content))
                if len(matches) < _MIN_TILES:
                    continue
                start, end = matches[0].start(), matches[-1].end()
                if not _outside_literal(content[:start]):
                    continue
                if any(
                    content[a.end() : b.start()].strip()
                    for a, b in zip(matches, matches[1:], strict=False)
                ):
                    continue
                matrices = [tuple(map(float, match.groups())) for match in matches]
                if any(b or c or a <= 0 or d <= 0 for a, b, c, d, _e, _f in matrices):
                    continue
                boxes = [
                    pymupdf.Rect(e, f, e + a, f + d) for a, _b, _c, d, e, f in matrices
                ]
                union = pymupdf.Rect(boxes[0])
                for box in boxes[1:]:
                    union |= box
                if (union & media).get_area() < 0.9 * media.get_area():
                    continue
                # All matched operators form one uninterrupted paint layer.
                # Render only that layer, then check the whole page before
                # touching the original document's stream or resource table.
                with pymupdf.open() as texture, pymupdf.open() as candidate:
                    texture.insert_pdf(doc, from_page=page.number, to_page=page.number)
                    tile_page = texture[0]
                    tile_xref = tile_page.get_contents()[0]
                    texture.update_stream(tile_xref, content[start:end])
                    tile_page.set_contents(tile_xref)
                    pixels = tile_page.get_pixmap(dpi=_DPI, alpha=False)
                    candidate.insert_pdf(
                        doc, from_page=page.number, to_page=page.number
                    )
                    test_page = candidate[0]
                    text = test_page.get_text()
                    before = test_page.get_pixmap(dpi=_DPI, alpha=False).samples
                    _replace(test_page, content, start, end, pixels)
                    after = test_page.get_pixmap(dpi=_DPI, alpha=False).samples
                    if before != after or test_page.get_text() != text:
                        continue
                    _replace(page, content, start, end, pixels)
                    report["pages"] += 1
                    report["placements_removed"] += len(matches) - 1
                pymupdf.TOOLS.store_shrink(100)
                break
        if not report["pages"]:
            return data, report
        _prune_unused_images(doc)
        return doc.tobytes(garbage=3, deflate=True), report
