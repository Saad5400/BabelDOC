"""The layout model's lifecycle: loaded per translation job, released after it.

The DocLayout-YOLO ONNX session used to be loaded by the worker at startup and
held for the life of the process, so an idle service sat on it (and on the ORT
arena behind it) for weeks although only translation jobs ever use it.
"""

import queue
import threading

import pytest

from server import capacity
from server import config
from server import jobs
from server import pipeline


class _FakeModel:
    pass


class _StopWorkerError(Exception):
    pass


@pytest.fixture()
def model_loads(monkeypatch):
    """Count real-looking loads without touching onnxruntime."""
    from babeldoc.docvision.doclayout import DocLayoutModel

    loads: list[_FakeModel] = []

    def load_onnx():
        model = _FakeModel()
        loads.append(model)
        return model

    monkeypatch.setattr(DocLayoutModel, "load_onnx", staticmethod(load_onnx))
    monkeypatch.setattr(pipeline, "_doc_layout_model", None)
    return loads


@pytest.fixture()
def trims(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(capacity, "trim", lambda: calls.append(1))
    return calls


def test_init_does_not_load_the_layout_model(model_loads, monkeypatch):
    monkeypatch.setattr(pipeline, "_initialised", False)

    pipeline.init()
    pipeline.init()

    assert model_loads == []
    assert pipeline._doc_layout_model is None


def test_a_job_loads_the_model_once_however_many_babeldoc_runs_it_makes(model_loads):
    first = pipeline._layout_model()
    second = pipeline._layout_model()  # the scanned-after-all retry

    assert first is second
    assert len(model_loads) == 1


def test_release_drops_the_model_and_trims(model_loads, trims):
    pipeline._layout_model()

    assert pipeline.release_layout_model() is True
    assert pipeline._doc_layout_model is None
    assert trims == [1]

    # Nothing held: nothing to release and no trim on its account.
    assert pipeline.release_layout_model() is False
    assert trims == [1]

    pipeline._layout_model()
    assert len(model_loads) == 2  # the next job loads afresh


def _drive_worker(monkeypatch, tmp_path, run_job) -> str:
    """Run one job through the real jobs._worker_loop and wait for its verdict."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(jobs, "_QUEUE", queue.SimpleQueue())
    monkeypatch.setattr(pipeline, "init", lambda: None)
    monkeypatch.setattr(pipeline, "run_job", run_job)

    # The loop never returns on its own; a sentinel id ends the thread so it
    # cannot outlive the test and pick up another test's jobs.
    real_read_job = jobs.read_job

    def read_job(job_id):
        if job_id == "stop":
            raise _StopWorkerError
        return real_read_job(job_id)

    monkeypatch.setattr(jobs, "read_job", read_job)

    job = jobs.create_job(b"%PDF-1.4 stub", filename="a.pdf", lang_in="en",
                          lang_out="ar", fmt="translated", title=None)
    jobs._QUEUE.put("stop")
    def run():
        try:
            jobs._worker_loop()
        except _StopWorkerError:
            pass

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive(), "worker did not finish the job"
    assert jobs._CURRENT_JOB_ID is None
    return job["job_id"]


@pytest.mark.parametrize("fails", [False, True])
def test_the_worker_releases_the_model_when_a_job_ends(model_loads, trims, monkeypatch,
                                                       tmp_path, fails):
    held_during_job = []

    def run_job(job_id):
        held_during_job.append(pipeline._layout_model())
        if fails:
            raise RuntimeError("babeldoc failed: boom")
        jobs.update_job(job_id, status="done")

    job_id = _drive_worker(monkeypatch, tmp_path, run_job)

    assert held_during_job and held_during_job[0] is model_loads[0]
    assert jobs.read_job(job_id)["status"] == ("failed" if fails else "done")
    assert pipeline._doc_layout_model is None
    assert trims  # the release trimmed (and so did the slot)


@pytest.mark.usefixtures("trims")
def test_a_job_that_never_reaches_babeldoc_loads_nothing(model_loads, monkeypatch, tmp_path):
    def run_job(job_id):
        jobs.update_job(job_id, status="failed", error="not a PDF")

    _drive_worker(monkeypatch, tmp_path, run_job)

    assert model_loads == []


def test_trim_empties_the_mupdf_store(monkeypatch):
    import pymupdf

    shrinks: list[int] = []
    monkeypatch.setattr(pymupdf.TOOLS, "store_shrink", lambda pct: shrinks.append(pct))

    capacity.trim()

    assert shrinks == [100]


def test_trim_survives_a_store_shrink_failure(monkeypatch):
    import pymupdf

    def boom(_pct):
        raise RuntimeError("no store")

    monkeypatch.setattr(pymupdf.TOOLS, "store_shrink", boom)

    capacity.trim()  # must not raise


def test_the_onnx_session_does_not_keep_an_arena(monkeypatch):
    """The real model, if its asset is present: arena and memory pattern off."""
    from babeldoc.assets.assets import get_doclayout_onnx_model_path
    from babeldoc.docvision import doclayout

    try:
        path = get_doclayout_onnx_model_path()
    except Exception:  # noqa: BLE001 - no asset and no network: nothing to load
        pytest.skip("layout model asset not available")

    monkeypatch.setenv("BABELDOC_ORT_THREADS", "2")
    options = doclayout.OnnxModel(str(path)).model.get_session_options()

    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False
    assert options.intra_op_num_threads == 2


def test_ort_threads_fall_back_to_two(monkeypatch):
    from babeldoc.docvision import doclayout

    monkeypatch.setenv("BABELDOC_ORT_THREADS", "not-a-number")
    assert doclayout._ort_intra_op_threads() == 2
    monkeypatch.setenv("BABELDOC_ORT_THREADS", "0")
    assert doclayout._ort_intra_op_threads() == 1
    monkeypatch.delenv("BABELDOC_ORT_THREADS")
    assert doclayout._ort_intra_op_threads() == 2
