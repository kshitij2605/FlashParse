"""Main async pipeline coordinator.

Bridges glmocr pipeline (threaded) with async VLM post-processing.
Uses per-page callbacks to overlap OCR with image captioning.
"""

import asyncio
import io
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import httpx
import img2pdf

from glmocr.config import load_config
from glmocr.pipeline import Pipeline

from ..clients.vlm_client import AsyncVLMClient
from ..config.settings import Settings
from ..markdown.assembler import assemble_markdown
from ..models.types import ImageInfo, PipelineResult
from ..utils.convert import ensure_pdf
from ..utils.extract import extract_to_markdown, is_direct_extract
from ..utils.image_utils import crop_region_from_page, image_to_base64
from ..vlm.classify_and_caption import AsyncClassifyAndCaption

logger = logging.getLogger(__name__)


class AsyncPDFPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._process_semaphore = asyncio.Semaphore(2)

        # Initialize glmocr pipeline
        glmocr_config = load_config(settings.glmocr.config_path)
        self.glmocr_pipeline = Pipeline(glmocr_config.pipeline)
        self.glmocr_pipeline.start()

        # GLM-OCR vLLM endpoint for deferred text extraction on misc images
        ocr_api = glmocr_config.pipeline.ocr_api
        self._ocr_url = ocr_api.api_url or f"http://{ocr_api.api_host}:{ocr_api.api_port}/v1/chat/completions"
        self._ocr_model = ocr_api.model
        self._ocr_client = httpx.AsyncClient(
            timeout=httpx.Timeout(ocr_api.request_timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
            proxy=None,
        )

        # Initialize async VLM components
        self.vlm_client = AsyncVLMClient(settings.vlm)
        self.classify_and_caption = AsyncClassifyAndCaption(self.vlm_client, settings.vlm)

    async def process(
        self,
        pdf_path: str,
        output_dir: str,
        skip_captions: bool = False,
        dpi: int | None = None,
        progress_callback: Callable | None = None,
    ) -> PipelineResult:
        async with self._process_semaphore:
            return await self._process(
                pdf_path, output_dir, skip_captions, dpi, progress_callback
            )

    async def _process(
        self,
        pdf_path: str,
        output_dir: str,
        skip_captions: bool = False,
        dpi: int | None = None,
        progress_callback: Callable | None = None,
    ) -> PipelineResult:
        timings = {}
        timing_ranges: dict[str, tuple[float, float]] = {}  # step -> (start, end)
        t0 = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Direct extraction for text-based formats (txt, csv, xlsx, html, etc.)
        if is_direct_extract(pdf_path):
            return await self._process_direct_extract(pdf_path, output_path, t0)

        # Convert non-PDF visual documents to PDF
        converted_pdf = await asyncio.get_event_loop().run_in_executor(
            None, ensure_pdf, pdf_path, output_path
        )
        pdf_path = str(converted_pdf)

        images_dir = Path(output_dir) / "images"
        tables_dir = Path(output_dir) / "tables"
        image_infos: list[ImageInfo] = []
        table_count = 0
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        layout_vis_dir = str(output_path / "_layout_vis_tmp")

        if progress_callback:
            await progress_callback("glmocr_processing", 0, 1, "Running GLM-OCR pipeline...")

        t1 = time.time()

        if skip_captions:
            # No overlap needed — run glmocr then extract regions
            glmocr_result = await asyncio.get_event_loop().run_in_executor(
                None, self._run_glmocr, pdf_path, None, layout_vis_dir
            )
            t1_end = time.time()
            timings["glmocr_pipeline"] = t1_end - t1
            timing_ranges["glmocr_pipeline"] = (t1, t1_end)

            page_images_dict = glmocr_result.page_images or {}
            page_images = [page_images_dict[i] for i in sorted(page_images_dict.keys())]
            raw = glmocr_result.json_result
            json_result = json.loads(raw) if isinstance(raw, str) else raw

            self._extract_regions(
                json_result, page_images, image_infos, images_dir, tables_dir
            )
            table_count = sum(
                1 for page in json_result for r in page
                if r.get("label") == "table" and r.get("bbox_2d")
            )
        else:
            # Overlap classify+caption with glmocr; start misc OCR as soon as glmocr frees port 8080
            loop = asyncio.get_event_loop()
            page_queue: asyncio.Queue = asyncio.Queue()
            caption_tasks: list[asyncio.Task] = []
            misc_ocr_pending: list[tuple[ImageInfo, str]] = []
            t_caption_start = [None]  # mutable for closure
            completed_captions = [0]

            def page_callback(page_idx, page_regions, page_image):
                """Bridge from recognition thread to asyncio."""
                loop.call_soon_threadsafe(
                    page_queue.put_nowait, (page_idx, page_regions, page_image)
                )

            async def do_classify_and_caption(
                info: ImageInfo, img_b64: str, page_img, page_b64: str,
            ) -> None:
                """Classify first, then caption only non-misc images."""
                category = await self.classify_and_caption.classify_only(
                    info.cropped, image_b64=img_b64,
                )
                info.category = category
                if category != "miscellaneous":
                    cat, caption = await self.classify_and_caption.classify_and_caption(
                        info.cropped, page_image=page_img,
                        image_b64=img_b64, page_image_b64=page_b64,
                    )
                    info.category = cat
                    info.caption = caption
                else:
                    misc_ocr_pending.append((info, img_b64))
                    info.caption = ""
                completed_captions[0] += 1
                if progress_callback:
                    await progress_callback(
                        "caption_generation", completed_captions[0], -1,
                        f"Captioned {completed_captions[0]} images",
                    )

            async def process_incoming_pages():
                """Consume page results from callback queue, start captioning immediately."""
                nonlocal table_count
                page_b64_cache: dict[int, str] = {}

                while True:
                    page_idx, page_regions, page_image = await page_queue.get()
                    if page_idx is None:  # sentinel — all pages done
                        break

                    for region in page_regions:
                        region_idx = region.get("index", 0)
                        bbox = region.get("bbox_2d")

                        # Image regions have content=None (task_type "skip")
                        if region.get("content") is None and bbox and page_image is not None:
                            cropped = crop_region_from_page(page_image, bbox)
                            filename = f"{page_idx}_{region_idx}.jpg"
                            info = ImageInfo(
                                page_idx=page_idx,
                                region_idx=region_idx,
                                bbox_2d=bbox,
                                cropped=cropped,
                                label="image",
                                image_filename=filename,
                            )
                            image_infos.append(info)

                            images_dir.mkdir(parents=True, exist_ok=True)
                            if cropped.mode != "RGB":
                                cropped = cropped.convert("RGB")
                            cropped.save(images_dir / filename, "JPEG", quality=95)

                            # Start classify+caption immediately (overlaps with glmocr)
                            if t_caption_start[0] is None:
                                t_caption_start[0] = time.time()
                                logger.info("First image arrived at %.3f (page %d)", t_caption_start[0], page_idx)
                            img_b64 = image_to_base64(cropped)
                            if page_idx not in page_b64_cache:
                                page_b64_cache[page_idx] = image_to_base64(page_image)
                            task = asyncio.create_task(
                                do_classify_and_caption(
                                    info, img_b64, page_image, page_b64_cache[page_idx]
                                )
                            )
                            caption_tasks.append(task)

                        elif region.get("label") == "table" and bbox and page_image is not None:
                            table_count += 1
                            cropped = crop_region_from_page(page_image, bbox)
                            tables_dir.mkdir(parents=True, exist_ok=True)
                            filename = f"{page_idx}_{region_idx}.jpg"
                            if cropped.mode != "RGB":
                                cropped = cropped.convert("RGB")
                            cropped.save(tables_dir / filename, "JPEG", quality=95)

            # Run page processing concurrently with glmocr
            page_processor = asyncio.create_task(process_incoming_pages())

            glmocr_result = await loop.run_in_executor(
                None, self._run_glmocr, pdf_path, page_callback, layout_vis_dir
            )
            t1_end = time.time()
            timings["glmocr_pipeline"] = t1_end - t1
            timing_ranges["glmocr_pipeline"] = (t1, t1_end)

            # Wait for all page callbacks to be processed (all tasks created)
            await page_processor

            # glmocr done → port 8080 is free. Start misc OCR immediately.
            # Most classify_only calls are done (they started 30+ seconds ago).
            # Caption tasks continue running on external VLM — no contention.
            misc_ocr_tasks: list[asyncio.Task] = []
            if misc_ocr_pending and self.settings.misc_ocr_enabled:
                t_ocr_misc = time.time()
                misc_ocr_tasks = [
                    asyncio.create_task(self._ocr_extract_text_into(info, b64))
                    for info, b64 in misc_ocr_pending
                ]
                logger.info("Started misc OCR for %d images (caption tasks still running)", len(misc_ocr_pending))

            # Wait for BOTH remaining caption tasks AND misc OCR to finish
            all_tasks = caption_tasks + misc_ocr_tasks
            if all_tasks:
                await asyncio.gather(*all_tasks)

            if t_caption_start[0] is not None:
                t_caption_end = time.time()
                timings["caption_generation"] = t_caption_end - t_caption_start[0]
                timing_ranges["caption_generation"] = (t_caption_start[0], t_caption_end)

            if misc_ocr_tasks:
                t_ocr_misc_end = time.time()
                timings["misc_ocr_extraction"] = t_ocr_misc_end - t_ocr_misc
                timing_ranges["misc_ocr_extraction"] = (t_ocr_misc, t_ocr_misc_end)
                logger.info(
                    "Extracted text from %d misc images in %.1fs",
                    len(misc_ocr_pending), timings["misc_ocr_extraction"],
                )

            raw = glmocr_result.json_result
            json_result = json.loads(raw) if isinstance(raw, str) else raw

        num_pages = len(json_result)

        if progress_callback:
            await progress_callback(
                "glmocr_processing", 1, 1,
                f"Complete: {num_pages} pages, {len(image_infos)} images captioned"
            )

        # Assemble final markdown
        t4 = time.time()
        final_markdown = assemble_markdown(json_result, image_infos)
        t4_end = time.time()
        timings["markdown_assembly"] = t4_end - t4
        timing_ranges["markdown_assembly"] = (t4, t4_end)
        t_total_end = time.time()
        timings["total"] = t_total_end - t0
        timing_ranges["overall"] = (t0, t_total_end)

        # Generate layouts PDF from visualization images
        pdf_name = Path(pdf_path).stem
        _generate_layouts_pdf(layout_vis_dir, output_path / f"{pdf_name}_layouts.pdf")

        mmd_path = output_path / f"{pdf_name}_with_captions.mmd"
        mmd_path.write_text(final_markdown, encoding="utf-8")

        # Build image stats
        category_counts = {}
        for info in image_infos:
            cat = info.category or "unknown"
            category_counts[cat] = category_counts.get(cat, 0) + 1

        image_stats = {
            "total": len(image_infos),
            "num_chart_images": category_counts.get("chart", 0),
            "num_figure_images": category_counts.get("figure", 0),
            "num_scanned_text_images": category_counts.get("scanned_text", 0),
            "num_miscellaneous_images": category_counts.get("miscellaneous", 0),
        }

        num_captions = sum(1 for img in image_infos if img.caption)

        # metadata.json
        metadata = {
            "pdf_title": Path(pdf_path).name,
            "num_pages": num_pages,
            "num_images": image_stats,
            "num_tables": table_count,
            "num_captions_generated": num_captions,
        }
        (output_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # parsing_metrics.json
        avg_per_page = timings["total"] / max(num_pages, 1)

        parsing_metrics = {
            "pdf_title": Path(pdf_path).name,
            "pdf_path": str(Path(pdf_path).resolve()),
            "output_path": str(output_path.resolve()),
            "run_timestamp": datetime.now().isoformat(),
            "model": "glm-ocr (vLLM) + PP-DocLayoutV3",
            "config": {
                "glmocr_config": self.settings.glmocr.config_path,
                "vlm_model": self.settings.vlm.model,
                "vlm_endpoint": self.settings.vlm.endpoint,
            },
            "timing": {
                step: {
                    "start": datetime.fromtimestamp(start).strftime("%H:%M:%S.%f")[:-3],
                    "end": datetime.fromtimestamp(end).strftime("%H:%M:%S.%f")[:-3],
                    "duration_seconds": round(end - start, 3),
                    "duration_formatted": _format_time(end - start),
                }
                for step, (start, end) in timing_ranges.items()
            } | {
                "average_per_page_seconds": round(avg_per_page, 3),
                "average_per_page_formatted": _format_time(avg_per_page),
            },
            "statistics": {
                "num_pages": num_pages,
                "num_images": image_stats,
                "num_tables": table_count,
                "num_captions_generated": num_captions,
            },
            "images": [
                {
                    "name": img.image_filename,
                    "page_index": img.page_idx,
                    "category": img.category,
                    "bbox": img.bbox_2d,
                    "has_caption": img.caption is not None,
                }
                for img in image_infos
            ],
            "tables": [
                {
                    "name": f"{page_idx}_{region.get('index', 0)}.jpg",
                    "page_index": page_idx,
                    "bbox": region.get("bbox_2d"),
                }
                for page_idx, page_regions in enumerate(json_result)
                for region in page_regions
                if region.get("label") == "table" and region.get("bbox_2d")
            ],
        }
        (output_path / "parsing_metrics.json").write_text(
            json.dumps(parsing_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return PipelineResult(
            markdown=final_markdown,
            pages_processed=num_pages,
            images_extracted=len(image_infos),
            tables_extracted=table_count,
            image_infos=image_infos,
            processing_times=timings,
        )

    async def _process_direct_extract(
        self, file_path: str, output_path: Path, t0: float
    ) -> PipelineResult:
        """Handle text-based files (txt, csv, xlsx, html) via direct extraction."""
        loop = asyncio.get_event_loop()
        file_name = Path(file_path).name

        t1 = time.time()
        markdown = await loop.run_in_executor(None, extract_to_markdown, file_path)
        extract_time = time.time() - t1

        total_time = time.time() - t0
        timings = {"direct_extraction": extract_time, "total": total_time}

        # Write output files
        stem = Path(file_path).stem
        mmd_path = output_path / f"{stem}_extracted.mmd"
        mmd_path.write_text(markdown, encoding="utf-8")

        metadata = {
            "pdf_title": file_name,
            "num_pages": 1,
            "num_images": {"total": 0},
            "num_tables": 0,
            "num_captions_generated": 0,
            "extraction_method": "direct",
        }
        (output_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        parsing_metrics = {
            "pdf_title": file_name,
            "pdf_path": str(Path(file_path).resolve()),
            "output_path": str(output_path.resolve()),
            "run_timestamp": datetime.now().isoformat(),
            "model": "direct extraction (no OCR)",
            "extraction_method": "direct",
            "timing": {
                "direct_extraction_seconds": round(extract_time, 3),
                "direct_extraction_formatted": _format_time(extract_time),
                "overall_total_seconds": round(total_time, 3),
                "overall_formatted": _format_time(total_time),
            },
            "statistics": {
                "num_pages": 1,
                "num_images": {"total": 0},
                "num_tables": 0,
                "num_captions_generated": 0,
            },
            "images": [],
            "tables": [],
        }
        (output_path / "parsing_metrics.json").write_text(
            json.dumps(parsing_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return PipelineResult(
            markdown=markdown,
            pages_processed=1,
            images_extracted=0,
            tables_extracted=0,
            processing_times=timings,
        )

    def _extract_regions(self, json_result, page_images, image_infos, images_dir, tables_dir):
        """Extract image and table regions from json_result (used in skip_captions mode)."""
        for page_idx, page_regions in enumerate(json_result):
            for region in page_regions:
                label = region.get("label", "text")
                region_idx = region.get("index", 0)
                bbox = region.get("bbox_2d")

                if label == "image" and bbox and page_idx < len(page_images):
                    cropped = crop_region_from_page(page_images[page_idx], bbox)
                    filename = f"{page_idx}_{region_idx}.jpg"
                    info = ImageInfo(
                        page_idx=page_idx,
                        region_idx=region_idx,
                        bbox_2d=bbox,
                        cropped=cropped,
                        label=label,
                        image_filename=filename,
                    )
                    image_infos.append(info)
                    images_dir.mkdir(parents=True, exist_ok=True)
                    if cropped.mode != "RGB":
                        cropped = cropped.convert("RGB")
                    cropped.save(images_dir / filename, "JPEG", quality=95)

                elif label == "table" and bbox and page_idx < len(page_images):
                    cropped = crop_region_from_page(page_images[page_idx], bbox)
                    tables_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{page_idx}_{region_idx}.jpg"
                    if cropped.mode != "RGB":
                        cropped = cropped.convert("RGB")
                    cropped.save(tables_dir / filename, "JPEG", quality=95)

    def _run_glmocr(self, pdf_path: str, page_callback=None, layout_vis_dir: str | None = None):
        """Run glmocr pipeline synchronously (called in executor)."""
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"file://{pdf_path}"},
                        }
                    ],
                }
            ]
        }
        results = list(self.glmocr_pipeline.process(
            request_data,
            page_callback=page_callback,
            save_layout_visualization=layout_vis_dir is not None,
            layout_vis_output_dir=layout_vis_dir,
        ))
        if not results:
            raise RuntimeError("GLM-OCR pipeline returned no results")
        return results[0]

    async def _ocr_extract_text_into(self, info: ImageInfo, image_b64: str) -> None:
        """Extract text from a misc image and set it on the ImageInfo."""
        info.caption = await self._ocr_extract_text(image_b64)

    async def _ocr_extract_text(self, image_b64: str) -> str:
        """Use GLM-OCR vLLM to extract text from a misc image region."""
        payload = {
            "model": self._ocr_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }],
            "max_tokens": 4096,
            "temperature": 0.01,
            "top_p": 0.00001,
        }
        for attempt in range(3):
            try:
                response = await self._ocr_client.post(self._ocr_url, json=payload)
                response.raise_for_status()
                text = response.json()["choices"][0]["message"]["content"].strip()
                return text
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                logger.warning("GLM-OCR text extraction failed after 3 attempts: %s", e)
                return ""

    async def close(self):
        await self.vlm_client.close()
        await self._ocr_client.aclose()
        self.glmocr_pipeline.stop()


def _generate_layouts_pdf(layout_vis_dir: str, output_pdf: Path) -> None:
    """Collect layout visualization JPEGs and combine into a single PDF."""
    vis_dir = Path(layout_vis_dir)
    if not vis_dir.exists():
        return
    # Sort by page number
    jpg_files = sorted(
        vis_dir.glob("layout_page*.jpg"),
        key=lambda p: int(p.stem.replace("layout_page", "")),
    )
    if not jpg_files:
        return
    image_bytes_list = []
    for jpg in jpg_files:
        image_bytes_list.append(jpg.read_bytes())
    pdf_bytes = img2pdf.convert(image_bytes_list)
    output_pdf.write_bytes(pdf_bytes)
    # Clean up temp visualization directory
    import shutil
    shutil.rmtree(vis_dir, ignore_errors=True)


def _format_time(seconds: float) -> str:
    if seconds < 0:
        return "0.0s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs:.1f}s")
    return " ".join(parts)
