"""Real child-process progress must be visible before OCR finishes."""

import json
import subprocess
import sys

import pymupdf
import pytest

from server import image_prep
from server import pipeline


def test_ocr_child_progress_is_reported_while_the_process_is_running(
    tmp_path, monkeypatch
):
    progress = tmp_path / "progress.json"
    finished = tmp_path / "finished"
    child = tmp_path / "child.py"
    child.write_text("""import json,sys,time
from pathlib import Path
progress,finished=map(Path,sys.argv[1:])
progress.write_text(json.dumps({'done':1,'total':2}))
time.sleep(1.3)
progress.write_text(json.dumps({'done':2,'total':2}))
finished.touch()
""")
    seen = []
    monkeypatch.setattr(
        pipeline,
        "_set_progress",
        lambda _job, percent, _stage: seen.append((percent, finished.exists())),
    )
    pipeline._run_cmd(
        [sys.executable, str(child), str(progress), str(finished)],
        "test",
        "image_prep",
        progress_file=progress,
    )
    assert seen == [(7.0, False), (12.0, True)]


def test_reporting_preserves_child_timeout_and_error(tmp_path):
    progress = tmp_path / "missing.json"
    with pytest.raises(subprocess.TimeoutExpired):
        pipeline._run_cmd(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            "test",
            "image_prep",
            progress_file=progress,
            timeout=0.15,
        )
    with pytest.raises(RuntimeError, match="exit 3.*specific failure"):
        pipeline._run_cmd(
            [
                sys.executable,
                "-c",
                'import sys; print("specific failure",file=sys.stderr); sys.exit(3)',
            ],
            "test",
            "image_prep",
            progress_file=progress,
        )


def test_pages_without_images_still_advance_and_do_not_change_text(
    tmp_path, monkeypatch
):
    source, result, regions, progress = (
        tmp_path / name
        for name in ("in.pdf", "out.pdf", "regions.json", "progress.json")
    )
    with pymupdf.open() as doc:
        for number in range(3):
            doc.new_page().insert_text((50, 70), f"Original page {number}")
        doc.save(source)
    seen = []
    write = image_prep._write_progress

    def capture(path, done, total):
        write(path, done, total)
        seen.append(json.loads(progress.read_text()))

    monkeypatch.setattr(image_prep, "_write_progress", capture)
    image_prep.prep_document(source, result, regions, progress_path=progress)
    assert seen == [{"done": number, "total": 3} for number in range(4)]
    with pymupdf.open(source) as old, pymupdf.open(result) as new:
        assert [p.get_text() for p in old] == [p.get_text() for p in new]


def test_unavailable_progress_destination_does_not_fail_image_preparation(tmp_path):
    image_prep._write_progress(tmp_path / "missing-dir" / "progress.json", 1, 3)


def test_standalone_cli_publishes_final_page_progress(tmp_path):
    from pathlib import Path

    source, result, regions, progress = (
        tmp_path / name
        for name in ("in.pdf", "out.pdf", "regions.json", "progress.json")
    )
    with pymupdf.open() as doc:
        doc.new_page().insert_text((50, 70), "A real digital page")
        doc.save(source)
    subprocess.run(  # noqa: S603 - fixed module and test-owned paths
        [
            sys.executable,
            str(Path(image_prep.__file__)),
            str(source),
            str(result),
            str(regions),
            "--progress-file",
            str(progress),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    assert json.loads(progress.read_text()) == {"done": 1, "total": 1}
    assert result.is_file() and regions.is_file()
