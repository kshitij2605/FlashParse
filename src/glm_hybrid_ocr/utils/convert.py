"""Convert non-PDF documents to PDF using LibreOffice headless."""

import logging
import shutil
import subprocess
from pathlib import Path

from .extract import DIRECT_EXTRACT_EXTENSIONS

logger = logging.getLogger(__name__)

# Visual document formats that need PDF conversion then OCR
VISUAL_EXTENSIONS = {
    ".pdf",
    # Office documents (have visual layout, fonts, images)
    ".docx", ".doc", ".odt", ".rtf",
    # Presentations (slides with mixed content)
    ".pptx", ".ppt", ".odp",
    # Images (scanned documents, photos of text)
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp",
    # Ebooks
    ".epub",
}

# All supported file types
SUPPORTED_EXTENSIONS = VISUAL_EXTENSIONS | DIRECT_EXTRACT_EXTENSIONS


def is_supported(file_path: str | Path) -> bool:
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS


def ensure_pdf(file_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Convert a document to PDF if needed. Returns the path to the PDF.

    If the file is already a PDF, returns the original path unchanged.
    Otherwise, converts using LibreOffice headless and returns the new PDF path.

    Args:
        file_path: Path to the input document.
        output_dir: Directory to write the converted PDF. Defaults to same directory as input.

    Returns:
        Path to the PDF file.

    Raises:
        ValueError: If the file extension is not supported.
        RuntimeError: If LibreOffice is not installed or conversion fails.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return file_path

    if suffix not in VISUAL_EXTENSIONS:
        raise ValueError(
            f"File type {suffix} does not need PDF conversion. "
            f"Use direct extraction instead."
        )

    if shutil.which("libreoffice") is None:
        raise RuntimeError(
            "LibreOffice is required to convert non-PDF documents. "
            "Install it with: apt install libreoffice-core (Debian/Ubuntu) "
            "or brew install --cask libreoffice (macOS)"
        )

    out_dir = Path(output_dir) if output_dir else file_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Converting %s to PDF...", file_path.name)

    result = subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--norestore",
            "--convert-to", "pdf",
            "--outdir", str(out_dir),
            str(file_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice conversion failed (exit {result.returncode}): {result.stderr}"
        )

    pdf_path = out_dir / f"{file_path.stem}.pdf"
    if not pdf_path.exists():
        raise RuntimeError(
            f"Conversion completed but PDF not found at {pdf_path}. "
            f"LibreOffice output: {result.stdout}"
        )

    logger.info("Converted to %s", pdf_path.name)
    return pdf_path
