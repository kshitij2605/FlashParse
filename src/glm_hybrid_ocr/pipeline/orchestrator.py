"""Main async pipeline coordinator.

Bridges glmocr pipeline (threaded) with async VLM post-processing.
Uses per-page callbacks to overlap OCR with image captioning.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Callable

from glmocr.config import load_config
from glmocr.pipeline import Pipeline

from ..clients.vlm_client import AsyncVLMClient
from ..config.settings import Settings
from ..markdown.assembler import assemble_markdown
from ..models.types import ImageInfo, PipelineResult
from ..utils.image_utils import crop_region_from_page, image_to_base64
from ..vlm.captioner import AsyncCaptioner
from ..vlm.classifier import AsyncImageClassifier

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
        self.classifier = AsyncImageClassifier(self.vlm_client, settings.vlm)
        self.captioner = AsyncCaptioner(self.vlm_client, settings.vlm)

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

        if progress_callback:
            await progress_callback("glmocr_processing", 0, 1, "Running GLM-OCR pipeline...")

        t1 = time.time()

        if skip_captions:
            # No overlap needed — run glmocr then extract regions
            glmocr_result = await asyncio.get_event_loop().run_in_executor(
                None, self._run_glmocr, pdf_path
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

            async def classify_and_caption(info: ImageInfo, img_b64: str, page_img, page_b64: str) -> None:
                cls_result = await self.classifier.classify(info.cropped)
                info.category = cls_result.category
                info.caption = await self.captioner.caption(
                    info.cropped,
                    info.category,
                    page_img,
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
                                classify_and_caption(
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
                None, self._run_glmocr, pdf_path, page_callback
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

        # Save outputs
        pdf_name = Path(pdf_path).stem
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        mmd_path = output_path / f"{pdf_name}_with_captions.mmd"
        mmd_path.write_text(final_markdown, encoding="utf-8")

        timings["total"] = time.time() - t0
        metadata = {
            "pdf_name": pdf_name,
            "pages_processed": num_pages,
            "images_extracted": len(image_infos),
            "tables_extracted": table_count,
        }
        (output_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        category_counts = {}
        for info in image_infos:
            cat = info.category or "unknown"
            category_counts[cat] = category_counts.get(cat, 0) + 1

        metrics = {
            "timing": {k: f"{v:.2f}s" for k, v in timings.items()},
            "statistics": {
                "num_pages": num_pages,
                "num_images": category_counts,
                "num_tables": table_count,
            },
        }
        (output_path / "parsing_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
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

    def _run_glmocr(self, pdf_path: str, page_callback=None):
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
            request_data, page_callback=page_callback
        ))
        if not results:
            raise RuntimeError("GLM-OCR pipeline returned no results")
        return results[0]

    async def close(self):
        await self.vlm_client.close()
        self.glmocr_pipeline.stop()
