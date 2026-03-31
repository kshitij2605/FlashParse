#!/usr/bin/env python3
"""Timing breakdown for caption generation."""
import asyncio
import sys
import time

sys.path.insert(0, "src")

from PIL import Image
from glm_hybrid_ocr.config.settings import Settings
from glm_hybrid_ocr.clients.vlm_client import AsyncVLMClient
from glm_hybrid_ocr.vlm.classifier import AsyncImageClassifier
from glm_hybrid_ocr.vlm.captioner import AsyncCaptioner
from glm_hybrid_ocr.utils.image_utils import image_to_base64


async def main():
    settings = Settings.load()
    client = AsyncVLMClient(settings.vlm)
    classifier = AsyncImageClassifier(client, settings.vlm)
    captioner = AsyncCaptioner(client, settings.vlm)

    # Load test images from previous run
    from pathlib import Path
    img_dir = Path("./test_output/2022統合報告書-1-12/images")
    images = []
    for f in sorted(img_dir.glob("*.jpg"))[:5]:
        images.append(Image.open(f))

    print(f"Testing with {len(images)} images")

    # Test base64 encoding time
    t0 = time.time()
    for img in images:
        image_to_base64(img)
    t_encode = time.time() - t0
    print(f"Base64 encoding {len(images)} images: {t_encode:.3f}s ({t_encode/len(images)*1000:.1f}ms each)")

    # Test 1: Sequential classification
    t0 = time.time()
    for img in images:
        await classifier.classify(img)
    t_seq_cls = time.time() - t0
    print(f"Sequential classify {len(images)}: {t_seq_cls:.3f}s ({t_seq_cls/len(images):.3f}s each)")

    # Test 2: Parallel classification
    t0 = time.time()
    results = await asyncio.gather(*[classifier.classify(img) for img in images])
    t_par_cls = time.time() - t0
    print(f"Parallel classify {len(images)}: {t_par_cls:.3f}s")
    categories = [r.category for r in results]
    print(f"  Categories: {categories}")

    # Test 3: Sequential captioning
    t0 = time.time()
    for img, cat in zip(images, categories):
        await captioner.caption(img, cat, None)
    t_seq_cap = time.time() - t0
    print(f"Sequential caption {len(images)}: {t_seq_cap:.3f}s ({t_seq_cap/len(images):.3f}s each)")

    # Test 4: Parallel captioning
    t0 = time.time()
    captions = await asyncio.gather(*[
        captioner.caption(img, cat, None)
        for img, cat in zip(images, categories)
    ])
    t_par_cap = time.time() - t0
    print(f"Parallel caption {len(images)}: {t_par_cap:.3f}s")

    # Test 5: Combined classify+caption per image (parallel)
    async def classify_and_caption(img):
        cls_result = await classifier.classify(img)
        caption = await captioner.caption(img, cls_result.category, None)
        return cls_result.category, caption

    t0 = time.time()
    combined = await asyncio.gather(*[classify_and_caption(img) for img in images])
    t_combined = time.time() - t0
    print(f"Combined classify+caption parallel {len(images)}: {t_combined:.3f}s")

    print(f"\nSpeedup from combining: {(t_par_cls + t_par_cap) / t_combined:.2f}x")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
