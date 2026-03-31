#!/usr/bin/env python3
"""Quick test script to run the pipeline directly."""
import asyncio
import sys
import time

sys.path.insert(0, "src")

from glm_hybrid_ocr.config.settings import Settings
from glm_hybrid_ocr.pipeline.orchestrator import AsyncPDFPipeline


async def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/test_pipeline.py <pdf_path> [output_dir]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./test_output"

    settings = Settings.load()
    print(f"VLM endpoint: {settings.vlm.endpoint}")
    print(f"VLM model: {settings.vlm.model}")
    print(f"GLMOCR config: {settings.glmocr.config_path}")
    print()

    print("Initializing pipeline...")
    pipeline = AsyncPDFPipeline(settings)
    print("Pipeline ready.\n")

    async def progress_callback(phase, current, total, message):
        print(f"  [{phase}] {current}/{total} - {message}")

    try:
        print(f"Processing: {pdf_path}")
        print(f"Output: {output_dir}")
        print()

        t0 = time.time()
        result = await pipeline.process(
            pdf_path,
            output_dir,
            skip_captions=False,
            progress_callback=progress_callback,
        )
        elapsed = time.time() - t0

        print(f"\n{'='*60}")
        print(f"DONE in {elapsed:.1f}s")
        print(f"  Pages: {result.pages_processed}")
        print(f"  Images: {result.images_extracted}")
        print(f"  Tables: {result.tables_extracted}")
        print(f"  Timings: {result.processing_times}")
        print(f"\nMarkdown preview (first 2000 chars):")
        print("-" * 60)
        print(result.markdown[:2000])
        print("-" * 60)
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
