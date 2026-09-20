import concurrent.futures
import threading

from server import config
from server import interlinear
from server import overlay_cache


def test_retries_reuse_completed_bytes_and_distinguish_options(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    calls = []
    def build(*_args, **kwargs):
        calls.append(kwargs)
        return b"%PDF-cached", {"pages": 1, "drawn": 2}
    monkeypatch.setattr(interlinear, "render_overlay", build)
    options = interlinear.OverlayOptions.defaults("interlinear")
    def render(vocab=True):
        return overlay_cache.render(b"%PDF-source", {"target": "نص"},
                                    style="interlinear", options=options, vocab=vocab)
    assert render() == render()
    assert len(calls) == 1
    render(False)
    assert len(calls) == 2
    # Expired results rebuild instead of surviving forever.
    monkeypatch.setattr(overlay_cache, "TTL_SECONDS", -1)
    render()
    assert len(calls) == 3


def test_simultaneous_retries_only_render_once(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    calls = []
    started, finish = threading.Event(), threading.Event()
    def build(*_args, **_kwargs):
        calls.append(1)
        started.set()
        assert finish.wait(5)
        return b"%PDF-once", {"pages": 1}
    monkeypatch.setattr(interlinear, "render_overlay", build)
    def render():
        return overlay_cache.render(b"%PDF-source", {}, style="interlinear",
            options=interlinear.OverlayOptions.defaults("interlinear"), vocab=False)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        first = pool.submit(render)
        assert started.wait(5)
        second = pool.submit(render)
        finish.set()
        assert first.result() == second.result()
    assert len(calls) == 1


def test_failed_build_is_retryable_and_cache_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(overlay_cache, "MAX_BYTES", 50)
    calls = []
    def build(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise interlinear.OverlayError("invalid")
        return b"%PDF-" + b"x" * 20, {"pages": 1}
    monkeypatch.setattr(interlinear, "render_overlay", build)
    import pytest
    def render(source):
        return overlay_cache.render(source, {}, style="interlinear",
            options=interlinear.OverlayOptions.defaults("interlinear"), vocab=False)
    with pytest.raises(interlinear.OverlayError):
        render(b"one")
    render(b"one")
    render(b"two")
    assert sum(p.stat().st_size for p in (tmp_path / "overlay-cache").glob("*.cache")) <= 50
    assert len(calls) == 3
