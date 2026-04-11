import asyncio
import base64
import json
import io
import logging
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from ...pipeline.orchestrator import AsyncPDFPipeline
from ...utils.convert import is_supported

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pdf", tags=["pdf"])

# Set by main.py during lifespan
pipeline: AsyncPDFPipeline | None = None


def _content_disposition(original_filename: str) -> str:
    stem = Path(original_filename).stem
    safe_name = f"{stem}_output.zip"
    try:
        safe_name.encode("latin-1")
        return f'attachment; filename="{safe_name}"'
    except UnicodeEncodeError:
        encoded = quote(f"{stem}_output.zip")
        return f"attachment; filename=\"output.zip\"; filename*=UTF-8''{encoded}"


def _create_zip(output_dir: str, include_layouts: bool = True) -> bytes:
    buffer = io.BytesIO()
    output_path = Path(output_dir)
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(output_path.rglob("*")):
            if file_path.is_file():
                if not include_layouts and file_path.name.endswith("_layouts.pdf"):
                    continue
                arcname = file_path.relative_to(output_path)
                zf.write(file_path, arcname)
    return buffer.getvalue()


@router.post("/process")
async def process_pdf(
    file: UploadFile = File(...),
    skip_captions: str = Form("false"),
    skip_layouts: str = Form("false"),
    dpi: str = Form("200"),
):
    if pipeline is None:
        return JSONResponse(status_code=503, content={"detail": "Pipeline not initialized"})

    skip_captions_bool = skip_captions.lower() in ("true", "1", "yes")
    skip_layouts_bool = skip_layouts.lower() in ("true", "1", "yes")
    dpi_int = int(dpi)

    if not is_supported(file.filename):
        return JSONResponse(
            status_code=400,
            content={"detail": f"Unsupported file type: {Path(file.filename).suffix}"},
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / file.filename
        content = await file.read()
        pdf_path.write_bytes(content)

        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        t0 = time.time()
        result = await pipeline.process(
            str(pdf_path),
            str(output_dir),
            skip_captions=skip_captions_bool,
            dpi=dpi_int,
        )
        processing_time = time.time() - t0

        zip_data = _create_zip(str(output_dir), include_layouts=not skip_layouts_bool)

        return StreamingResponse(
            io.BytesIO(zip_data),
            media_type="application/zip",
            headers={
                "Content-Disposition": _content_disposition(file.filename),
                "X-Pages-Processed": str(result.pages_processed),
                "X-Images-Extracted": str(result.images_extracted),
                "X-Tables-Extracted": str(result.tables_extracted),
                "X-Processing-Time": f"{processing_time:.2f}",
            },
        )


@router.post("/process-with-progress")
async def process_pdf_with_progress(
    file: UploadFile = File(...),
    skip_captions: str = Form("false"),
    skip_layouts: str = Form("false"),
    dpi: str = Form("200"),
):
    if pipeline is None:
        return JSONResponse(status_code=503, content={"detail": "Pipeline not initialized"})

    skip_captions_bool = skip_captions.lower() in ("true", "1", "yes")
    skip_layouts_bool = skip_layouts.lower() in ("true", "1", "yes")
    dpi_int = int(dpi)

    if not is_supported(file.filename):
        return JSONResponse(
            status_code=400,
            content={"detail": f"Unsupported file type: {Path(file.filename).suffix}"},
        )

    file_content = await file.read()
    file_filename = file.filename

    async def event_stream():
        start_time = time.time()
        event_queue: asyncio.Queue = asyncio.Queue()

        async def progress_callback(phase, current, total, message):
            elapsed = time.time() - start_time
            eta = None
            if current > 0 and total > 0:
                eta = (elapsed / current) * (total - current)
            event = {
                "type": "progress",
                "phase": phase,
                "current": current,
                "total": total,
                "message": message,
                "elapsed": round(elapsed, 1),
                "eta": round(eta, 1) if eta is not None else None,
            }
            await event_queue.put(event)

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / file_filename
            pdf_path.write_bytes(file_content)

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            # Run pipeline in background task
            async def run_pipeline():
                try:
                    result = await pipeline.process(
                        str(pdf_path),
                        str(output_dir),
                        skip_captions=skip_captions_bool,
                        dpi=dpi_int,
                        progress_callback=progress_callback,
                    )
                    await event_queue.put(("done", result))
                except Exception as e:
                    await event_queue.put(("error", e))

            task = asyncio.create_task(run_pipeline())

            # Yield events as they come
            while True:
                event = await event_queue.get()

                if isinstance(event, tuple):
                    status, payload = event
                    if status == "error":
                        error_event = {"type": "error", "message": str(payload)}
                        yield f"data: {json.dumps(error_event)}\n\n"
                        break

                    # status == "done"
                    result = payload
                    zip_data = _create_zip(str(output_dir), include_layouts=not skip_layouts_bool)
                    zip_b64 = base64.b64encode(zip_data).decode("utf-8")

                    CHUNK_SIZE = 1024 * 1024
                    if len(zip_b64) <= CHUNK_SIZE:
                        complete_event = {
                            "type": "complete",
                            "pages": result.pages_processed,
                            "images": result.images_extracted,
                            "tables": result.tables_extracted,
                            "zip_base64": zip_b64,
                        }
                        yield f"data: {json.dumps(complete_event)}\n\n"
                    else:
                        chunks = [
                            zip_b64[i : i + CHUNK_SIZE]
                            for i in range(0, len(zip_b64), CHUNK_SIZE)
                        ]
                        start_event = {
                            "type": "complete_start",
                            "pages": result.pages_processed,
                            "images": result.images_extracted,
                            "tables": result.tables_extracted,
                            "total_chunks": len(chunks),
                            "total_size": len(zip_data),
                        }
                        yield f"data: {json.dumps(start_event)}\n\n"

                        for idx, chunk in enumerate(chunks):
                            yield f"data: {json.dumps({'type': 'chunk', 'index': idx, 'data': chunk})}\n\n"

                        yield f"data: {json.dumps({'type': 'complete_end'})}\n\n"
                    break
                else:
                    # Progress event dict
                    yield f"data: {json.dumps(event)}\n\n"

            await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
