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

import img2pdf

from glmocr.config import load_config
from glmocr.pipeline import Pipeline

from ..clients.vlm_client import AsyncVLMClient
from ..config.settings import Settings
from ..markdown.assembler import assemble_markdown
from ..models.types import ImageInfo, PipelineResult
from ..utils.image_utils import crop_region_from_page, image_to_base64
from ..vlm.classify_and_caption import AsyncClassifyAndCaption

logger = logging.getLogger(__name__)


class AsyncPDFPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings

        # Initialize glmocr pipeline
        glmocr_config = load_config(settings.glmocr.config_path)
        self.glmocr_pipeline = Pipeline(glmocr_config.pipeline)
        self.glmocr_pipeline.start()

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
        timings = {}
        t0 = time.time()

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
            timings["glmocr_pipeline"] = time.time() - t1

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
            # Overlap OCR with captioning using per-page callbacks
            loop = asyncio.get_event_loop()
            page_queue: asyncio.Queue = asyncio.Queue()
            caption_tasks: list[asyncio.Task] = []
            t_caption_start = [None]  # mutable for closure
            completed_captions = [0]

            def page_callback(page_idx, page_regions, page_image):
                """Bridge from recognition thread to asyncio."""
                loop.call_soon_threadsafe(
                    page_queue.put_nowait, (page_idx, page_regions, page_image)
                )

            async def do_classify_and_caption(info: ImageInfo, img_b64: str, page_img, page_b64: str) -> None:
                info.category, info.caption = await self.classify_and_caption.classify_and_caption(
                    info.cropped,
                    page_image=page_img,
                    image_b64=img_b64,
                    page_image_b64=page_b64,
                )
                completed_captions[0] += 1
                if progress_callback:
                    await progress_callback(
                        "caption_generation", completed_captions[0], -1,
                        f"Captioned {completed_captions[0]} images"
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
                            filename = f"page{page_idx}_{region_idx}.jpg"
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

                            # Start captioning immediately
                            if t_caption_start[0] is None:
                                t_caption_start[0] = time.time()
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
                            filename = f"page{page_idx}_{region_idx}.jpg"
                            if cropped.mode != "RGB":
                                cropped = cropped.convert("RGB")
                            cropped.save(tables_dir / filename, "JPEG", quality=95)

            # Run page processing concurrently with glmocr
            page_processor = asyncio.create_task(process_incoming_pages())

            glmocr_result = await loop.run_in_executor(
                None, self._run_glmocr, pdf_path, page_callback, layout_vis_dir
            )
            timings["glmocr_pipeline"] = time.time() - t1

            # Wait for all page callbacks to be processed
            await page_processor

            # Wait for all caption tasks to complete
            if caption_tasks:
                await asyncio.gather(*caption_tasks)

            if t_caption_start[0] is not None:
                timings["caption_generation"] = time.time() - t_caption_start[0]

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
        timings["markdown_assembly"] = time.time() - t4
        timings["total"] = time.time() - t0

        # Generate layouts PDF from visualization images
        pdf_name = Path(pdf_path).stem
        _generate_layouts_pdf(layout_vis_dir, output_path / f"{pdf_name}_layouts.pdf")

        mmd_path = output_path / f"{pdf_name}_with_captions.mmd"
        mmd_path.write_text(final_markdown, encoding="utf-8")

        # Build image stats (matching deepseek-ocr-api format)
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

        # metadata.json (matching deepseek-ocr-api format)
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

        # parsing_metrics.json (matching deepseek-ocr-api format + glm-ocr specifics)
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
                "glmocr_pipeline_seconds": round(timings.get("glmocr_pipeline", 0), 3),
                "glmocr_pipeline_formatted": _format_time(timings.get("glmocr_pipeline", 0)),
                "caption_generation_seconds": round(timings.get("caption_generation", 0), 3),
                "caption_generation_formatted": _format_time(timings.get("caption_generation", 0)),
                "markdown_assembly_seconds": round(timings.get("markdown_assembly", 0), 3),
                "markdown_assembly_formatted": _format_time(timings.get("markdown_assembly", 0)),
                "overall_total_seconds": round(timings["total"], 3),
                "overall_formatted": _format_time(timings["total"]),
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
                    "name": f"page{page_idx}_{region.get('index', 0)}.jpg",
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

    def _extract_regions(self, json_result, page_images, image_infos, images_dir, tables_dir):
        """Extract image and table regions from json_result (used in skip_captions mode)."""
        for page_idx, page_regions in enumerate(json_result):
            for region in page_regions:
                label = region.get("label", "text")
                region_idx = region.get("index", 0)
                bbox = region.get("bbox_2d")

                if label == "image" and bbox and page_idx < len(page_images):
                    cropped = crop_region_from_page(page_images[page_idx], bbox)
                    filename = f"page{page_idx}_{region_idx}.jpg"
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
                    filename = f"page{page_idx}_{region_idx}.jpg"
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

    async def close(self):
        await self.vlm_client.close()
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
