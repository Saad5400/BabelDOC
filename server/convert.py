"""Office-document → PDF normalisation, via headless LibreOffice.

BabelDOC translates PDFs and nothing else, but the material people actually
have is a slide deck: university lectures are shipped as .pptx far more often
than as .pdf. Before this module a .pptx was a dead end at the very first step
— the caller had no PDF to submit, so the user was told to go convert the file
themselves.

The conversion lives HERE, in the engine, for the same reason OCR does: it is
document plumbing that needs a heavyweight binary (LibreOffice) which the
callers must not each have to carry. It is deliberately STATELESS, like
/v1/compose — bytes in, bytes out, no job, no cache, no /data. The caller keeps
the converted PDF and treats it as the original from then on, which is what
makes the downstream dual-format compose work: compose pairs the ORIGINAL PDF
with the translated one, and a .pptx can never be half of that pair.

Each conversion gets its OWN LibreOffice user profile. soffice serialises
itself around a shared profile directory — two concurrent requests against one
profile is the classic "second call silently produces nothing" failure — so the
profile is a per-call temp dir that dies with the request.
"""

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

# What LibreOffice is asked to open. Kept to the document formats a caller
# could plausibly want translated — a spreadsheet's grid survives neither the
# PDF conversion nor the translation in any useful shape, so it stays out.
CONVERTIBLE_EXTENSIONS = (
    "docx", "doc", "odt", "rtf",
    "pptx", "ppt", "odp",
)

# A big deck with embedded media genuinely takes a while to render; the ceiling
# exists to bound a WEDGED soffice (it hangs rather than fails when a profile
# is contended or a font lookup stalls), not to rush an honest conversion.
CONVERT_TIMEOUT_SECONDS = 300


class ConvertError(Exception):
    """The document could not be turned into a PDF."""


def is_convertible(filename: str) -> bool:
    return extension_of(filename) in CONVERTIBLE_EXTENSIONS


def extension_of(filename: str) -> str:
    return (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""


def office_to_pdf(data: bytes, filename: str) -> bytes:
    """Convert one office document to PDF bytes.

    `filename` is only read for its EXTENSION: LibreOffice picks its import
    filter from the file suffix, so the scratch copy has to keep it. The user's
    actual name never reaches the filesystem here.
    """
    extension = extension_of(filename)

    if extension not in CONVERTIBLE_EXTENSIONS:
        raise ConvertError(f"cannot convert .{extension or '?'} to PDF")

    if not data:
        raise ConvertError("the document is empty")

    with tempfile.TemporaryDirectory(prefix="doctranslate-convert-") as workdir:
        work = Path(workdir)
        source = work / f"source.{extension}"
        source.write_bytes(data)

        outdir = work / "out"
        outdir.mkdir()

        # A private profile per call (see the module docstring) — inside the
        # same temp dir, so it is cleaned up even when soffice leaves lock
        # files behind.
        profile = work / f"profile-{uuid.uuid4().hex}"

        argv = [
            _soffice(),
            "--headless",
            "--norestore",
            # Not merely cosmetic: a conversion that pops any dialog (a repair
            # prompt on a slightly malformed deck, a macro warning) would
            # otherwise block until the timeout.
            "--nolockcheck",
            "--nodefault",
            "--nofirststartwizard",
            f"-env:UserInstallation=file://{profile}",
            "--convert-to", "pdf",
            "--outdir", str(outdir),
            str(source),
        ]

        try:
            result = subprocess.run(
                argv, capture_output=True, text=True,
                timeout=CONVERT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise ConvertError("converting the document to PDF timed out") from exc
        except OSError as exc:
            raise ConvertError(f"LibreOffice could not be run: {exc}") from exc

        produced = next(iter(outdir.glob("*.pdf")), None)

        # soffice exits 0 on failures often enough that the exit code is not
        # evidence; the produced FILE is. Its stderr is the only useful thing
        # to report, so it rides along when there is nothing to hand back.
        if produced is None:
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            raise ConvertError(
                "LibreOffice produced no PDF"
                + (f": {detail[-1]}" if detail else "")
            )

        pdf_bytes = produced.read_bytes()

        if not pdf_bytes.startswith(b"%PDF-"):
            raise ConvertError("LibreOffice produced a file that is not a PDF")

        return pdf_bytes


def _soffice() -> str:
    path = shutil.which("soffice") or shutil.which("libreoffice")

    if path is None:
        raise ConvertError("LibreOffice is not installed in this image")

    return path
