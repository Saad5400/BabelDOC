"""Bounded disk cache for expensive, deterministic PDF overlays.

A client can time out while MuPDF is still rendering. Keep that finished work
so a retry does not render it again. Striped locks coalesce concurrent retries
without an unbounded lock registry. Inputs never appear in filenames or logs.
"""
import dataclasses
import hashlib
import json
import logging
import tempfile
import threading
import time
from pathlib import Path

from server import compose
from server import config
from server import interlinear

logger = logging.getLogger("doctranslate.overlay_cache")
_LOCKS = [threading.Lock() for _ in range(64)]
_PRUNE_LOCK = threading.Lock()
MAX_BYTES = 256 * 1024 * 1024
TTL_SECONDS = 24 * 3600
# Rendering changes get a fresh namespace even if an operator forgets to bump
# a version. Cover the renderers and font/text repair code, not user filenames.
_SOURCE_FILES = (
    "server/interlinear.py", "server/page_fonts.py", "server/vocab_pages.py",
    "server/raster_gate.py", "server/compose.py", "server/pdf_tiles.py",
    "babeldoc/format/pdf/document_il/backend/pdf_creater.py",
)
_ROOT = Path(__file__).resolve().parent.parent
_REVISION = hashlib.sha256(b"".join((_ROOT / name).read_bytes()
                                   for name in _SOURCE_FILES)).hexdigest()


def _prune(root: Path) -> None:
    with _PRUNE_LOCK:
        files = sorted(root.glob("*.cache"), key=lambda p: p.stat().st_mtime,
                       reverse=True)
        total = 0
        cutoff = time.time() - TTL_SECONDS
        for path in files:
            stat = path.stat()
            total += stat.st_size
            if stat.st_mtime < cutoff or total > MAX_BYTES:
                path.unlink(missing_ok=True)


def render(original: bytes, sidecar: dict, *, style: str,
           options: interlinear.OverlayOptions, vocab: bool):
    options = options.validated()
    digest = hashlib.sha256(original)
    digest.update(json.dumps([_REVISION, sidecar, style,
                              dataclasses.asdict(options), vocab,
                              config.VOCAB_PAGES], sort_keys=True,
                             ensure_ascii=True).encode())
    return _cached(digest.hexdigest(), lambda: interlinear.render_overlay(
        original, sidecar, style=style, options=options, vocab=vocab))


def render_dual(original: bytes, translated: bytes, format: str, *,
                sidecar: dict | None = None, vocab: bool = True) -> bytes:
    digest = hashlib.sha256(original)
    digest.update(hashlib.sha256(translated).digest())
    digest.update(json.dumps(["compose", _REVISION, sidecar, format, vocab,
                              config.VOCAB_PAGES], sort_keys=True,
                             ensure_ascii=True).encode())
    result, _ = _cached(digest.hexdigest(), lambda: (
        compose.compose_dual(original, translated, format, sidecar=sidecar, vocab=vocab),
        {},
    ))
    return result


def _cached(key: str, build):
    root = config.DATA_DIR / "overlay-cache"
    path = root / (key + ".cache")
    started = time.monotonic()
    with _LOCKS[int(key[:2], 16) % len(_LOCKS)]:
        try:
            if path.is_file() and time.time() - path.stat().st_mtime < TTL_SECONDS:
                with path.open("rb") as cached:
                    report = json.loads(cached.readline())
                    result = cached.read()
                if result.startswith(b"%PDF-"):
                    logger.info("overlay cache hit: %.3fs, %d bytes",
                                time.monotonic() - started, len(result))
                    return result, report
        except (OSError, ValueError):
            logger.warning("overlay cache read failed; rebuilding", exc_info=True)

        result, report = build()
        temporary = None
        try:
            header = json.dumps(report).encode() + b"\n"
            if len(header) + len(result) <= MAX_BYTES:
                root.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=root, delete=False) as out:
                    temporary = Path(out.name)
                    out.write(header)
                    out.write(result)
                temporary.replace(path)
                _prune(root)
        except OSError:
            logger.warning("overlay cache write failed; serving result", exc_info=True)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove cache temporary file", exc_info=True)
        logger.info("PDF built: %.3fs, %d bytes",
                    time.monotonic() - started, len(result))
        return result, report
