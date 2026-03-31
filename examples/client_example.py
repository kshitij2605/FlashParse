#!/usr/bin/env python3
"""
GLM Hybrid OCR API Client Example

Usage:
    python client_example.py /path/to/document.pdf [output_dir]
    python client_example.py /path/to/pdf_folder [output_dir]

API Endpoints:
    POST /api/v1/pdf/process - Process a PDF document (returns ZIP file)
    POST /api/v1/pdf/process-with-progress - Process with streaming progress (SSE)
    GET  /health             - Health check

Output ZIP contents:
    - {pdf_name}_with_captions.mmd - Final markdown with captions and tables
    - metadata.json - Simple metadata
    - parsing_metrics.json - Detailed processing metrics
    - images/ - Extracted image files
    - tables/ - Extracted table image files
"""

import argparse
import base64
import json
import os
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import requests

API_BASE_URL = os.getenv("GLM_HYBRID_OCR_API_URL", "http://localhost:8000")


def check_health() -> dict:
    response = requests.get(f"{API_BASE_URL}/health", timeout=10)
    response.raise_for_status()
    return response.json()


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def process_pdf_with_progress(
    pdf_path: str,
    output_dir: str,
    skip_captions: bool = False,
    dpi: int = 200,
) -> dict:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    SUPPORTED = {
        # Visual documents (OCR pipeline)
        ".pdf", ".docx", ".doc", ".odt", ".rtf", ".pptx", ".ppt", ".odp",
        ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".epub",
        # Text-based (direct extraction)
        ".txt", ".csv", ".xlsx", ".xls", ".ods", ".html", ".htm",
    }
    if pdf_path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {pdf_path.suffix}")

    pdf_name = pdf_path.stem
    output_path = Path(output_dir) / pdf_name
    output_path.mkdir(parents=True, exist_ok=True)

    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        data = {
            "skip_captions": str(skip_captions).lower(),
            "dpi": str(dpi),
        }
        response = requests.post(
            f"{API_BASE_URL}/api/v1/pdf/process-with-progress",
            files=files,
            data=data,
            timeout=1800,
            stream=True,
        )

    response.raise_for_status()

    result = {
        "pages_processed": 0,
        "images_extracted": 0,
        "tables_extracted": 0,
        "processing_time": 0,
        "output_dir": str(output_path),
        "files": [],
    }

    last_phase = None
    start_time = time.time()
    total_chunks = 0
    chunks = []
    received_chunks = 0

    for line in response.iter_lines():
        if not line:
            continue

        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue

        try:
            event_data = json.loads(line[6:])
        except json.JSONDecodeError:
            continue

        event_type = event_data.get("type")

        if event_type == "progress":
            phase = event_data.get("phase", "")
            current = event_data.get("current", 0)
            total = event_data.get("total", 0)
            message = event_data.get("message", "")
            elapsed = event_data.get("elapsed", 0)
            eta = event_data.get("eta")

            if phase != last_phase:
                if last_phase is not None:
                    print()
                last_phase = phase

            if phase == "caption_generation" and total > 0:
                progress_pct = (current / total) * 100
                bar_width = 30
                filled = int(bar_width * current / total)
                bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
                elapsed_str = format_time(elapsed)
                print(
                    f"\r  Captions: [{bar}] {current}/{total} ({progress_pct:.1f}%) | Elapsed: {elapsed_str}    ",
                    end="",
                    flush=True,
                )
            else:
                elapsed_str = format_time(elapsed)
                print(f"\r  {message} [{elapsed_str}]    ", end="", flush=True)

        elif event_type == "complete":
            print()
            result["pages_processed"] = event_data.get("pages", 0)
            result["images_extracted"] = event_data.get("images", 0)
            result["tables_extracted"] = event_data.get("tables", 0)
            result["processing_time"] = time.time() - start_time

            zip_base64 = event_data.get("zip_base64", "")
            if zip_base64:
                zip_data = base64.b64decode(zip_base64)
                with zipfile.ZipFile(BytesIO(zip_data), "r") as zf:
                    zf.extractall(output_path)
                    result["files"] = zf.namelist()

        elif event_type == "complete_start":
            print()
            result["pages_processed"] = event_data.get("pages", 0)
            result["images_extracted"] = event_data.get("images", 0)
            result["tables_extracted"] = event_data.get("tables", 0)
            result["processing_time"] = time.time() - start_time
            total_chunks = event_data.get("total_chunks", 0)
            total_size = event_data.get("total_size", 0)
            chunks = [""] * total_chunks
            received_chunks = 0
            print(f"  Receiving ZIP data: {total_size / 1024 / 1024:.1f} MB in {total_chunks} chunks...")

        elif event_type == "chunk":
            chunk_index = event_data.get("index", 0)
            chunks[chunk_index] = event_data.get("data", "")
            received_chunks += 1
            progress = (received_chunks / total_chunks) * 100
            print(f"\r  Downloading: {received_chunks}/{total_chunks} chunks ({progress:.0f}%)    ", end="", flush=True)

        elif event_type == "complete_end":
            print()
            print("  Extracting ZIP...")
            zip_base64 = "".join(chunks)
            zip_data = base64.b64decode(zip_base64)
            with zipfile.ZipFile(BytesIO(zip_data), "r") as zf:
                zf.extractall(output_path)
                result["files"] = zf.namelist()

        elif event_type == "error":
            print()
            raise RuntimeError(f"Server error: {event_data.get('message', 'Unknown error')}")

    return result


def process_pdf(
    pdf_path: str,
    output_dir: str,
    skip_captions: bool = False,
    dpi: int = 200,
) -> dict:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pdf_name = pdf_path.stem
    output_path = Path(output_dir) / pdf_name
    output_path.mkdir(parents=True, exist_ok=True)

    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        data = {
            "skip_captions": str(skip_captions).lower(),
            "dpi": str(dpi),
        }
        response = requests.post(
            f"{API_BASE_URL}/api/v1/pdf/process",
            files=files,
            data=data,
            timeout=1800,
            stream=True,
        )

    response.raise_for_status()

    result = {
        "pages_processed": int(response.headers.get("X-Pages-Processed", 0)),
        "images_extracted": int(response.headers.get("X-Images-Extracted", 0)),
        "tables_extracted": int(response.headers.get("X-Tables-Extracted", 0)),
        "processing_time": float(response.headers.get("X-Processing-Time", 0)),
        "output_dir": str(output_path),
        "files": [],
    }

    content = response.content
    if len(content) == 0:
        raise ValueError("Server returned empty response")

    with zipfile.ZipFile(BytesIO(content), "r") as zf:
        zf.extractall(output_path)
        result["files"] = zf.namelist()

    return result


def find_pdfs_in_folder(folder_path: str) -> list:
    folder = Path(folder_path)
    pdfs = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
    return sorted(set(pdfs))


def process_folder(
    folder_path: str,
    output_dir: str,
    skip_captions: bool = False,
    dpi: int = 200,
) -> dict:
    pdfs = find_pdfs_in_folder(folder_path)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {folder_path}")

    total_pdfs = len(pdfs)
    results = {
        "total_pdfs": total_pdfs,
        "successful": 0,
        "failed": 0,
        "pdf_results": [],
        "errors": [],
    }

    folder_start_time = time.time()
    print(f"\n  Found {total_pdfs} PDF file(s) to process")
    print("=" * 60)

    for idx, pdf_path in enumerate(pdfs, 1):
        folder_elapsed = time.time() - folder_start_time
        print(f"\n  [{idx}/{total_pdfs}] Processing: {pdf_path.name} [Elapsed: {format_time(folder_elapsed)}]")
        print("  " + "-" * 50)

        try:
            result = process_pdf_with_progress(
                str(pdf_path), output_dir,
                skip_captions=skip_captions, dpi=dpi,
            )
            results["successful"] += 1
            results["pdf_results"].append({"pdf": pdf_path.name, "status": "success", "result": result})
            print(f"    Pages: {result['pages_processed']}, Images: {result['images_extracted']}, Tables: {result['tables_extracted']}")
            print(f"    Time: {result['processing_time']:.2f}s")
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({"pdf": pdf_path.name, "error": str(e)})
            print(f"    ERROR: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="GLM Hybrid OCR API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python client_example.py document.pdf
    python client_example.py /path/to/pdf_folder
    python client_example.py document.pdf /path/to/output
    python client_example.py document.pdf --skip-captions

Environment Variables:
    GLM_HYBRID_OCR_API_URL - API base URL (default: http://localhost:8000)
        """,
    )
    parser.add_argument("pdf_path", nargs="?", default=None, help="Path to PDF file or folder")
    parser.add_argument("output_dir", nargs="?", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--skip-captions", action="store_true", help="Skip caption generation")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF rendering (default: 200)")
    parser.add_argument("--health-only", action="store_true", help="Only check API health")

    args = parser.parse_args()

    print("GLM Hybrid OCR API Client")
    print(f"API URL: {API_BASE_URL}")
    print("=" * 50)

    print("\n[1] Checking API health...")
    try:
        health = check_health()
        print(f"  Status: {health['status']}")
        print(f"  Version: {health['version']}")
        print(f"  VLM Available: {health['vlm_available']}")

        if args.health_only:
            return 0
        if args.pdf_path is None:
            print("\n  No PDF path provided. Use --health-only or provide a PDF path.")
            return 1
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot connect to API at {API_BASE_URL}")
        return 1
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1

    input_path = Path(args.pdf_path)
    print(f"\n[2] Input: {input_path}")
    print(f"  DPI: {args.dpi}")
    print(f"  Skip captions: {args.skip_captions}")

    try:
        if input_path.is_dir():
            folder_result = process_folder(
                args.pdf_path, args.output_dir,
                skip_captions=args.skip_captions, dpi=args.dpi,
            )
            print("\n" + "=" * 50)
            print(f"  Successful: {folder_result['successful']}, Failed: {folder_result['failed']}")
            return 0 if folder_result["failed"] == 0 else 1
        else:
            result = process_pdf_with_progress(
                args.pdf_path, args.output_dir,
                skip_captions=args.skip_captions, dpi=args.dpi,
            )
            print(f"\n[3] Processing complete!")
            print(f"  Pages: {result['pages_processed']}, Images: {result['images_extracted']}, Tables: {result['tables_extracted']}")
            print(f"  Time: {result['processing_time']:.2f}s")
            print(f"  Output: {result['output_dir']}")

            # Show files
            images = [f for f in result["files"] if f.startswith("images/")]
            tables = [f for f in result["files"] if f.startswith("tables/")]
            other = [f for f in result["files"] if not f.startswith(("images/", "tables/"))]
            print("\n  Files:")
            for f in other:
                print(f"    - {f}")
            if images:
                print(f"  Images: {len(images)} files")
            if tables:
                print(f"  Tables: {len(tables)} files")
            print("\nDone!")
            return 0

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
