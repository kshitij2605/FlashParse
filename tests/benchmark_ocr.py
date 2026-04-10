"""Benchmark pipeline speed with warm/cold runs."""
import sys
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glm_hybrid_ocr.pipeline.orchestrator import AsyncPDFPipeline
from glm_hybrid_ocr.config.settings import Settings


async def run_benchmark(pdf_path: str, output_dir: str, runs: int = 2, skip_captions: bool = True):
    settings = Settings()
    pipeline = AsyncPDFPipeline(settings)

    for i in range(runs):
        out = Path(output_dir) / f"run_{i}"
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        result = await pipeline.process(
            pdf_path=pdf_path,
            output_dir=str(out),
            skip_captions=skip_captions,
        )
        elapsed = time.time() - t0
        pages = result.pages_processed
        imgs = result.images_extracted
        label = "COLD" if i == 0 else "WARM"
        print(f"[{label}] Run {i+1}: {elapsed:.2f}s total, {pages} pages, {imgs} images, "
              f"{elapsed/pages:.2f}s/page, {pages/elapsed:.2f} pages/sec")
        print(f"  Timings: {result.processing_times}")

    pipeline.glmocr_pipeline.stop()


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "/home/mac/25gitlab/notebooklm/notebook_data/uploads/2022統合報告書.pdf"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/benchmark_ocr"
    skip = "--skip-captions" in sys.argv or "--ocr-only" in sys.argv
    asyncio.run(run_benchmark(pdf, out, skip_captions=skip))
