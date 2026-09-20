import pymupdf
import pytest

from server import pdf_tiles


def tiled_pdf(size=40, interrupted=False):
    doc = pymupdf.open()
    page = doc.new_page(width=size * 5, height=size * 5)
    tile = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 5, 5))
    tile.clear_with(240)
    tile.set_pixel(2, 2, (225, 225, 225))
    page.insert_image(pymupdf.Rect(0, 0, 5, 5), pixmap=tile)
    name = page.get_images()[0][7]
    blocks = [
        f"q 5 0 0 5 {x * 5} {y * 5} cm /{name} Do Q\n".encode()
        for y in range(size)
        for x in range(size)
    ]
    if interrupted:
        blocks.insert(len(blocks) // 2, b"q 1 0 0 RG 0 0 m 100 100 l S Q\n")
    xref = page.get_contents()[0]
    doc.update_stream(xref, b"".join(blocks))
    page.insert_text((10, 25), "Preserve this text", fontsize=12)
    content = b"\n".join(doc.xref_stream(x) for x in page.get_contents())
    doc.update_stream(xref, content)
    page.set_contents(xref)
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data


def test_tiled_background_becomes_one_image_with_identical_text_and_pixels():
    source = tiled_pdf()
    result, report = pdf_tiles.compact(source)
    assert report == {"pages": 1, "placements_removed": 1599}
    with pymupdf.open(stream=source) as old, pymupdf.open(stream=result) as new:
        assert old[0].get_text() == new[0].get_text()
        assert old[0].get_pixmap(dpi=216).samples == new[0].get_pixmap(dpi=216).samples
        assert len(new[0].get_image_info()) == 1


@pytest.mark.parametrize("source", [tiled_pdf(4), tiled_pdf(interrupted=True)])
def test_small_or_interrupted_patterns_are_unchanged(source):
    result, report = pdf_tiles.compact(source)
    assert result == source
    assert report["pages"] == 0


def test_changed_candidate_is_rejected_before_original_is_modified(monkeypatch):
    source = tiled_pdf()
    replace = pdf_tiles._replace

    def bad_candidate(page, *args):
        replace(page, *args)
        page.insert_text((10, 50), "Must be rejected")

    monkeypatch.setattr(pdf_tiles, "_replace", bad_candidate)
    result, report = pdf_tiles.compact(source)
    assert result == source and report["pages"] == 0


@pytest.mark.parametrize(
    "prefix,expected",
    [
        (b"q\n", True),
        (b"(unclosed ", False),
        (b"<ffff", False),
        (b"% still a comment ", False),
        (b"% done\n", True),
        (b"(escaped \\) still inside ", False),
        (b"BI /W 5 ID raw", False),
        (b"<< /Name (closed) >>\n", True),
    ],
)
def test_patterns_inside_pdf_literals_are_not_rewritten(prefix, expected):
    assert pdf_tiles._outside_literal(prefix) is expected


def test_unused_images_are_pruned_per_page_without_mutating_shared_resources():
    with pymupdf.open(stream=tiled_pdf(4)) as doc:
        page = doc[0]
        original_contents = page.get_contents()[0]
        image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 6, 6))
        image.clear_with(150)
        page.insert_image(page.rect, pixmap=image)
        second_name = page.get_images()[-1][7]
        page.set_contents(original_contents)
        resources = doc.xref_get_key(page.xref, "Resources")[1]
        other = doc.new_page(width=20, height=20)
        stream = doc.get_new_xref()
        doc.update_object(stream, "<<>>")
        doc.update_stream(stream, f"q 20 0 0 20 0 0 cm /{second_name} Do Q".encode())
        other.set_contents(stream)
        doc.xref_set_key(other.xref, "Resources", resources)
        before = [p.get_pixmap().samples for p in doc]
        pdf_tiles._prune_unused_images(doc)
        assert [len(p.get_images()) for p in doc] == [1, 1]
        assert [p.get_pixmap().samples for p in doc] == before


def test_resource_name_escapes_and_comments_do_not_lose_images():
    with pymupdf.open(stream=tiled_pdf(4)) as doc:
        page = doc[0]
        xref = page.get_contents()[0]
        content = doc.xref_stream(xref).replace(b"/fzImg0 Do", b"/fzImg#30 % comment\n Do")
        doc.update_stream(xref, content)
        before = page.get_pixmap().samples
        pdf_tiles._prune_unused_images(doc)
        assert len(page.get_images()) == 1
        assert page.get_pixmap().samples == before


def test_compaction_does_not_create_an_ocr_region_for_decorative_texture():
    from server.image_prep import gather_regions
    source = tiled_pdf()
    result, _ = pdf_tiles.compact(source)
    with pymupdf.open(stream=source) as old, pymupdf.open(stream=result) as new:
        assert gather_regions(old[0]) == []
        assert gather_regions(new[0]) == []
