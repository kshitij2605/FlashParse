#!/usr/bin/env python3
"""Test to verify caption generation concurrency."""
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
from pathlib import Path


async def main():
    settings = Settings.load()
    client = AsyncVLMClient(settings.vlm)
    classifier = AsyncImageClassifier(client, settings.vlm)
    captioner = AsyncCaptioner(client, settings.vlm)

    img_dir = Path("./test_output/2022統合報告書-1-12/images")
    images = [Image.open(f) for f in sorted(img_dir.glob("*.jpg"))]
    n = len(images)
    print(f"Testing concurrency with {n} images")
    print(f"Semaphore limit: {settings.vlm.max_concurrency}")
    print()

    # Pre-encode
    b64s = [image_to_base64(img) for img in images]

    # Track per-image timeline
    events = []

    async def classify_and_caption(idx, img, b64):
        t0 = time.time()
        events.append((idx, "classify_start", t0))
        cls = await classifier.classify(img)
        t1 = time.time()
        events.append((idx, "classify_end", t1))
        events.append((idx, "caption_start", t1))
        cap = await captioner.caption(img, cls.category, None, image_b64=b64)
        t2 = time.time()
        events.append((idx, "caption_end", t2))
        return idx, cls.category, t2 - t0

    t_start = time.time()
    results = await asyncio.gather(*[
        classify_and_caption(i, img, b64)
        for i, (img, b64) in enumerate(zip(images, b64s))
    ])
    t_total = time.time() - t_start

    # Print timeline
    print(f"{'Img':>3}  {'Event':<16}  {'T(s)':>6}  Timeline")
    print("-" * 70)
    for idx, event, t in sorted(events, key=lambda x: x[2]):
        elapsed = t - t_start
        bar_pos = int(elapsed * 2)  # 2 chars per second
        bar = " " * bar_pos
        marker = "C" if "classify" in event else "P"  # C=classify, P=caption(Produce)
        fill = ">" if "start" in event else "<"
        print(f"{idx:3d}  {event:<16}  {elapsed:6.2f}  |{bar}{marker}{fill}")

    print()
    print(f"Total wall time: {t_total:.2f}s")
    sum_individual = sum(r[2] for r in results)
    print(f"Sum of individual times: {sum_individual:.2f}s")
    print(f"Parallelism factor: {sum_individual / t_total:.2f}x")
    print()

    # Check overlap: were multiple requests in-flight at the same time?
    # Build intervals
    intervals = []
    for idx, event, t in events:
        if "start" in event:
            kind = "classify" if "classify" in event else "caption"
            intervals.append((idx, kind, t, None))
    for i, (idx, kind, start, _) in enumerate(intervals):
        for idx2, event2, t2 in events:
            if idx2 == idx and kind in event2 and "end" in event2:
                intervals[i] = (idx, kind, start, t2)
                break

    # Count max concurrent at any point
    all_times = []
    for idx, kind, start, end in intervals:
        if end:
            all_times.append((start, +1, idx, kind))
            all_times.append((end, -1, idx, kind))
    all_times.sort()
    concurrent = 0
    max_concurrent = 0
    for t, delta, idx, kind in all_times:
        concurrent += delta
        max_concurrent = max(max_concurrent, concurrent)

    print(f"Max concurrent VLM requests: {max_concurrent}")
    print()

    # Per-image summary
    print(f"{'Img':>3}  {'Category':<15}  {'Classify':>8}  {'Caption':>8}  {'Total':>8}")
    print("-" * 55)
    for idx, cat, total in sorted(results):
        cls_time = None
        cap_time = None
        for idx2, event, t in events:
            if idx2 == idx and event == "classify_start":
                cs = t
            if idx2 == idx and event == "classify_end":
                cls_time = t - cs
            if idx2 == idx and event == "caption_start":
                cps = t
            if idx2 == idx and event == "caption_end":
                cap_time = t - cps
        print(f"{idx:3d}  {cat:<15}  {cls_time:7.2f}s  {cap_time:7.2f}s  {total:7.2f}s")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
