"""Build-time smoke test: fail the IMAGE BUILD, not the running worker.

Run by server/Dockerfile after the dependencies and the baked assets are in
place. It imports every server module (the subprocess scripts included; they
guard their entry points) and babeldoc's whole PDF stack, which is what the
job worker's `pipeline.init()` and a translation job import lazily, then loads
the layout model and runs it once.

Why it exists: a fresh build once resolved `bitstring` 5.0, which removed
`BitStream`. Every endpoint still answered and /healthz said "ok", but the
worker thread died importing babeldoc's PDF writer and no translation could
ever run. A build that cannot import what a job imports must not ship.

    python -m server.smoke_imports   # from the repo root (/app in the image)
"""

import importlib
import pkgutil
import sys
from pathlib import Path

#: babeldoc modules the worker reaches outside the PDF package walk below.
BABELDOC_MODULES = [
    "babeldoc.format.pdf.high_level",  # pipeline.init(); imports the PDF writer
    "babeldoc.format.pdf.translation_config",
    "babeldoc.glossary",
    "babeldoc.translator.translator",
    "babeldoc.docvision.doclayout",
    "babeldoc.babeldoc_exception.BabelDOCException",
    "babeldoc.assets.assets",
]


def server_modules() -> list[str]:
    here = Path(__file__).resolve().parent
    return sorted(
        f"server.{path.stem}"
        for path in here.glob("*.py")
        if not path.stem.startswith("test_") and path.stem not in ("conftest", "smoke_imports")
    )


def pdf_backend_modules() -> list[str]:
    """Every module of babeldoc's PDF intermediate-language stack, the part the
    pipeline drives (bitstring broke `document_il.backend.pdf_creater`)."""
    import babeldoc.format.pdf.document_il as il

    return sorted(
        info.name
        for info in pkgutil.walk_packages(il.__path__, prefix=f"{il.__name__}.")
    )


def check_layout_model() -> None:
    """Load the baked ONNX model and run one inference: onnxruntime, numpy and
    cv2 have to agree with each other, not only import."""
    import numpy as np
    from babeldoc.docvision.doclayout import DocLayoutModel

    model = DocLayoutModel.load_onnx()
    model.predict(np.full((800, 600, 3), 255, dtype=np.uint8))


def main() -> int:
    failures = []
    names = BABELDOC_MODULES + server_modules()
    try:
        names += pdf_backend_modules()
    except Exception as exc:  # noqa: BLE001 - reported like any other failure
        failures.append(f"babeldoc.format.pdf.document_il: {type(exc).__name__}: {exc}")
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - collect them all, then fail
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if not failures:
        try:
            check_layout_model()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"layout model: {type(exc).__name__}: {exc}")
    if failures:
        print("smoke test FAILED:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    print(f"smoke test ok: {len(names)} modules imported, layout model ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())
