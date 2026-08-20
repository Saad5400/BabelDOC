#!/usr/bin/env python
"""Post-process BabelDOC output of an ocrmypdf'd (sandwich) PDF.

ocrmypdf places the OCR text layer BEFORE the page image in the content
stream (invisible text, so order doesn't matter visually). BabelDOC's
--ocr-workaround replaces that layer in place with white-masking rects +
visible translated text, which then gets covered by the page raster.
This script moves BabelDOC's /OCR-* form XObject invocation to the end of
each page's content stream so the translation draws on top.

Usage: fix_layer_order.py in.pdf out.pdf
"""
import re
import sys

import pymupdf


def toplevel_blocks(s: str):
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
    return blocks


def main(src, dst):
    d = pymupdf.open(src)
    for p in d:
        cont = b"".join(d.xref_stream(x) for x in p.get_contents())
        blocks = toplevel_blocks(cont.decode("latin1"))
        ocr = [b for b in blocks if "/OCR-" in b]
        other = [b for b in blocks if "/OCR-" not in b]
        new = "\n".join(b.strip() for b in other + ocr)
        xref = p.get_contents()[0]
        p.set_contents(xref)
        d.update_stream(xref, new.encode("latin1"))
    d.save(dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
