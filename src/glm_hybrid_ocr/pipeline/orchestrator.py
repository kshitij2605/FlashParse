"""Main async pipeline coordinator.

Bridges glmocr pipeline (threaded) with async VLM post-processing.
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
from ..utils.image_utils import crop_region_from_page
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

        # Phase 1: Run glmocr pipeline (page images are captured internally)
        if progress_callback:
            await progress_callback("glmocr_processing", 0, 1, "Running GLM-OCR pipeline...")

        t1 = time.time()
        glmocr_result = await asyncio.get_event_loop().run_in_executor(
            None, self._run_glmocr, pdf_path
        )
        timings["glmocr_pipeline"] = time.time() - t1

        # Reuse page images rendered by glmocr (no redundant PDF rendering)
        page_images_dict = glmocr_result.page_images or {}
        page_images = [page_images_dict[i] for i in sorted(page_images_dict.keys())]

        raw = glmocr_result.json_result
        json_result = json.loads(raw) if isinstance(raw, str) else raw
        num_pages = len(json_result)

        if progress_callback:
            await progress_callback("glmocr_processing", 1, 1, f"GLM-OCR complete: {num_pages} pages")

        image_infos: list[ImageInfo] = []
        table_count = 0
        images_dir = Path(output_dir) / "images"
        tables_dir = Path(output_dir) / "tables"

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

                    # Save cropped image
                    images_dir.mkdir(parents=True, exist_ok=True)
                    if cropped.mode != "RGB":
                        cropped = cropped.convert("RGB")
                    cropped.save(images_dir / filename, "JPEG", quality=95)

                elif label == "table" and bbox and page_idx < len(page_images):
                    table_count += 1
                    cropped = crop_region_from_page(page_images[page_idx], bbox)
                    tables_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"page{page_idx}_{region_idx}.jpg"
                    if cropped.mode != "RGB":
                        cropped = cropped.convert("RGB")
                    cropped.save(tables_dir / filename, "JPEG", quality=95)

        # Phase 3: Image classification + captioning (pipelined parallel)
        # Each image: classify → caption runs as one task; all images run concurrently.
        # Pre-encode base64 to avoid redundant page image encoding.
        if not skip_captions and image_infos:
            t3 = time.time()
            total_images = len(image_infos)
            completed = 0

            if progress_callback:
                await progress_callback(
                    "caption_generation", 0, total_images,
                    f"Processing {total_images} images (classify+caption)..."
                )

            # Pre-encode all cropped images and page images to base64
            from ..utils.image_utils import image_to_base64

            page_b64_cache: dict[int, str] = {}
            image_b64_list: list[str] = []
            for info in image_infos:
                image_b64_list.append(image_to_base64(info.cropped))
                if info.page_idx not in page_b64_cache and info.page_idx < len(page_images):
                    page_b64_cache[info.page_idx] = image_to_base64(page_images[info.page_idx])

            async def classify_and_caption(info: ImageInfo, img_b64: str) -> None:
                nonlocal completed
                page_b64 = page_b64_cache.get(info.page_idx)
                page_img = page_images[info.page_idx] if info.page_idx < len(page_images) else None

                cls_result = await self.classifier.classify(info.cropped)
                info.category = cls_result.category

                info.caption = await self.captioner.caption(
                    info.cropped,
                    info.category,
                    page_img,
                    image_b64=img_b64,
                    page_image_b64=page_b64,
                )

                completed += 1
                if progress_callback:
                    await progress_callback(
                        "caption_generation", completed, total_images,
                        f"Captioned {completed}/{total_images} images"
                    )

            await asyncio.gather(*[
                classify_and_caption(info, b64)
                for info, b64 in zip(image_infos, image_b64_list)
            ])

            timings["caption_generation"] = time.time() - t3

        # Phase 4: Assemble final markdown
        t4 = time.time()
        final_markdown = assemble_markdown(json_result, image_infos)
        timings["markdown_assembly"] = time.time() - t4

        # Phase 5: Save outputs
        pdf_name = Path(pdf_path).stem
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save markdown
        mmd_path = output_path / f"{pdf_name}_with_captions.mmd"
        mmd_path.write_text(final_markdown, encoding="utf-8")

        # Save metadata
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

        # Save parsing metrics
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

    def _run_glmocr(self, pdf_path: str):
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
        results = list(self.glmocr_pipeline.process(request_data))
        if not results:
            raise RuntimeError("GLM-OCR pipeline returned no results")
        return results[0]

    async def close(self):
        await self.vlm_client.close()
        self.glmocr_pipeline.stop()
