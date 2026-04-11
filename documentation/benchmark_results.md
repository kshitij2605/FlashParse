# Benchmark Results

Raw benchmark data from performance testing on NVIDIA RTX A6000 (49.1 GiB).

**Test PDF:** 2022統合報告書.pdf (70 pages, 149 images, ~95 misc)
**Date:** 2026-04-11
**GLM-OCR:** 0.9B via vLLM, MTP-1 speculative decoding, prefix caching enabled
**VLM:** Qwen3.5-35B-A3B on difgpu01 (shared server, 10-20s variability between runs)
**Layout:** PP-DocLayoutV3, batch_size=4, DPI 200

---

## 1. gpu-memory-utilization Sweep

Tests different vLLM memory allocation levels with `max_workers=64`. All runs use `skip_layouts=true`.

### VRAM Breakdown

| Component | Memory |
|-----------|--------|
| GLM-OCR model weights + MTP | ~2.5 GiB |
| KV cache (at 0.70) | ~32 GiB |
| PP-DocLayoutV3 idle | ~2.1 GiB |
| PP-DocLayoutV3 peak (batch_size=4) | ~5.7 GiB |

### Results (warm runs)

| Utilization | vLLM VRAM | KV Cache Tokens | glmocr | Overall | Peak VRAM | Free at Peak |
|-------------|-----------|-----------------|--------|---------|-----------|--------------|
| 0.50 | 24.8 GiB | 317,664 | 35.6s | 57.8s | 29.9 GiB | 18.8 GiB |
| 0.60 | 29.6 GiB | ~390K | ~24s | 49.8s | ~31.6 GiB | 13.5 GiB |
| **0.70** | **34.5 GiB** | **463K** | **~24s** | **48.2s** | **~36.6 GiB** | **8.5 GiB** |
| 0.82 | 40.3 GiB | 551,712 | ~24s | 51.0s | ~42.4 GiB | 2.7 GiB |
| 0.85 | 41.7 GiB | — | 248.5s | 253.8s | — | <2 GiB (thrashing) |
| 0.90 | — | — | — | — | — | (failed to load) |

### Key Observations

- **0.50 degrades OCR by ~50%** (35.6s vs 24s) — not KV exhaustion but CUDA memory pressure during forward passes
- **0.70 = 0.82 in OCR speed** but with 3x more VRAM headroom
- **0.85 causes thrashing** — layout detector + vLLM leaves <2 GiB free, GPU memory contention slows everything 5x
- **0.90 fails to load** entirely when layout detector is co-located
- KV cache is massively over-provisioned: each OCR request uses only **~181 tokens** (measured via vLLM metrics: 312,791 prompt tokens / 3,022 requests)
- At 0.50 (317K tokens), the KV cache supports 1,755 concurrent requests — never the bottleneck

### Selected: `--gpu-memory-utilization 0.70`

---

## 2. max_workers (OCR Concurrency) Sweep

Tests different concurrent OCR request counts with `gpu-memory-utilization=0.70`. The glmocr pipeline uses PP-DocLayoutV3 to detect regions, crops each region, and sends individual region images to vLLM via `max_workers` concurrent HTTP connections.

### Results (warm runs)

| max_workers | connection_pool_size | Wall | Overall | glmocr | Caption |
|-------------|---------------------|------|---------|--------|---------|
| 8 | 128 | 92.8s | 87.4s | 70.9s | 85.7s |
| 16 | 128 | 73.3s | 68.1s | 51.3s | 66.3s |
| 32 | 128 | 64.5s | 59.1s | 40.5s | 57.1s |
| **64** | **256** | **57.1s** | **51.9s** | **35.2s** | **49.9s** |
| 96 | 384 | 60.3s | 55.0s | 34.9s | 52.7s |
| 128 | 512 | 56.1s | 50.6s | 33.7s | 48.7s |
| 256 | 1024 | 54.2s | 48.7s | 34.1s | 46.7s |
| 512 | 2048 | 62.6s | 57.2s | 34.8s | 55.2s |

### Key Observations

- **8→64 workers halves glmocr time** (70.9s → 35.2s) by letting vLLM batch more regions per GPU forward pass
- **glmocr plateaus at ~34s from 64 onwards** — GPU is saturated, more concurrency doesn't help
- Variations in overall/caption time (46-55s) are VLM server variability, not max_workers impact
- **512 workers is slower** (62.6s) — overhead of 2048 HTTP connections and thread management hurts
- Connection pool size should be ≥ 2x max_workers to avoid connection exhaustion under burst loads

### Selected: `max_workers: 64`, `connection_pool_size: 256`

---

## 3. Concurrent PDF Load Testing

Tests how many simultaneous PDF requests the API can handle.

### Phase 1: Without locks (Semaphore only)

Initial testing without concurrency locks in the glmocr Pipeline:

| Concurrent Requests | PDF | Result |
|---------------------|-----|--------|
| 1 | 70-page | ~51s |
| 2 | 70-page | CUDA OOM — two layout batches spike +7.2 GiB simultaneously |
| 3 | 70-page | PDFium data corruption — `RuntimeError: DataLoadingThread: Failed to load document` |

**Failure modes:**
- **CUDA OOM** (2 concurrent): Two layout detection batches (batch_size=4) running simultaneously spike ~7.2 GiB
- **PDFium corruption** (3 concurrent): The PDFium C library is not thread-safe even with separate `PdfDocument` handles. Crashes with `free(): invalid size` or `"Already borrowed"`

### Phase 2: With `_pdf_lock` + `_layout_lock` in glmocr Pipeline

Added two `threading.Lock()` instances to `Pipeline.__init__()`:
- `_pdf_lock`: Serializes pypdfium2 page rendering in `data_loading_thread`
- `_layout_lock`: Serializes `layout_detector.process()` in `_stream_process_layout_batch`

OCR recognition (the bulk of processing time) runs fully concurrently across PDFs.

### Results (70-page PDF, skip_captions, batch_size=2)

| Concurrent PDFs | Wall Time | Per-PDF Time | Throughput | Notes |
|---|---|---|---|---|
| 1 | ~51s | 51s | 1.4 pages/s | Baseline |
| 2 | 66.5s | ~66s each | 2.1 pages/s | +50% throughput |
| 3 | 100.4s | 68-100s | 2.1 pages/s | Longer tail latency |
| 4 | 123.2s | 65-123s | 2.3 pages/s | Still stable |
| 5 | 165.6s | 70-166s | 2.1 pages/s | Still stable |

No CUDA OOM or PDFium crashes in any test. Some `"Already borrowed"` 400 responses from vLLM during high concurrency, but the OCR client retries and all PDFs completed successfully.

### Selected: `asyncio.Semaphore(2)` with pipeline locks — 2 concurrent PDFs

---

## 4. vLLM Server Configuration

Tested with `gpu-memory-utilization=0.90` (before layout detector co-location was understood), OCR-only (`skip_captions=True`), 70-page PDF.

| Config | Cold | Warm | Pages/sec |
|--------|------|------|-----------|
| No MTP, no prefix cache | 44.3s | 35.0s | 2.00 |
| MTP only | 45.0s | 36.5s | 1.92 |
| Prefix cache only | 43.8s | 33.1s | 2.11 |
| **MTP + prefix cache** | 45.5s | **27.4s** | **2.55** |

**Note:** These numbers were measured at 0.90 without the layout detector loaded. With layout detector at 0.70, warm OCR time is ~35s (35.2s at max_workers=64).

### Prefix Caching Details

vLLM prefix caching uses positional block hashing. Same token blocks at the same positions = cache hit. OCR regions sharing "Text Recognition:" or "Table Recognition:" prefixes benefit from this. Measured 50.8% cache hit rate (158,896 / 312,791 prompt tokens from `local_cache_hit`).

### Selected: MTP-1 + prefix caching (default vLLM behavior)

---

## 5. skip_layouts Impact on Response Size

The layouts PDF (`*_layouts.pdf`) is a debug visualization showing detected regions on each page.

| Content | 70-page PDF |
|---------|-------------|
| Layouts PDF | ~41 MB (87% of ZIP) |
| Other output (mmd, images, tables, metrics) | ~6 MB |
| Total ZIP with layouts | ~47 MB |
| Total ZIP without layouts | ~6 MB |
| HTTP transfer time saved | ~15-20s |

Added `skip_layouts` parameter to both `/process` and `/process-with-progress` endpoints. Default: `"false"` (layouts included).

---

## 6. Pipeline Queue Configuration

The glmocr pipeline uses bounded queues between its 3 threads to control memory usage:

```
Data Loading → [page_queue, max 100] → Layout Detection → [region_queue, max 800] → VLM Recognition (max_workers concurrent)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `page_maxsize` | 100 | Max page images buffered between loader and layout threads |
| `region_maxsize` | 800 | Max cropped regions buffered between layout and recognition threads |
| `max_workers` | 64 | Concurrent OCR requests to vLLM |

For a 1000-page PDF: at most ~100 pages in memory at once (~0.5-2 GiB RAM). The data loading thread blocks when the queue is full (backpressure). No data loss — all pages are processed sequentially through the queue.

---

## 7. Layout batch_size Sweep

Tests different PP-DocLayoutV3 batch sizes. Larger batches = fewer GPU forward passes for layout detection, but delay the first `page_callback` (less overlap with captioning). Each config run 3 times to average out VLM server variability.

### Results (3-run averages, gpu-memory-utilization=0.70, max_workers=64)

| batch_size | Avg Overall | Avg glmocr | Avg Caption | Avg Overlap |
|------------|------------|------------|-------------|-------------|
| 1 | 51.1s | 36.0s | 50.6s | 35.5s |
| **2** | **51.7s** | **34.0s** | **50.9s** | **33.3s** |
| 4 | 58.9s | 34.9s | 56.9s | 32.9s |
| 8 | 53.5s | 36.0s | 50.3s | 32.8s |
| 16 | crashed | — | — | — |

### Individual Runs

| batch_size | Run | Overall | glmocr | Caption | Overlap |
|------------|-----|---------|--------|---------|---------|
| 1 | 1 | 54.4s | 37.7s | 53.9s | 37.2s |
| 1 | 2 | 49.4s | 35.2s | 48.9s | 34.7s |
| 1 | 3 | 49.4s | 35.0s | 49.0s | 34.5s |
| 2 | 1 | 48.7s | 35.4s | 47.9s | 34.6s |
| 2 | 2 | 47.0s | 33.1s | 46.2s | 32.4s |
| 2 | 3 | 59.3s | 33.6s | 58.5s | 32.8s |
| 4 | 1 | 59.1s | 37.2s | 57.1s | 35.3s |
| 4 | 2 | 59.3s | 33.8s | 57.3s | 31.8s |
| 4 | 3 | 58.4s | 33.7s | 56.3s | 31.6s |
| 8 | 1 | 59.9s | 36.5s | 56.7s | 33.3s |
| 8 | 2 | 50.0s | 36.4s | 46.7s | 33.2s |
| 8 | 3 | 50.7s | 35.0s | 47.6s | 31.9s |

### Key Observations

- **glmocr plateaus at ~34s for batch_size 2-8** — layout detection is a small fraction of total OCR time
- **batch_size=1 is ~2s slower** on glmocr (36.0s) because of more GPU forward passes
- **Overlap decreases with larger batch sizes** (~2.5s from 1→8) — larger batches delay the first `page_callback`
- **batch_size=16 crashes** — exceeds available VRAM headroom
- **VLM server variability (10-12s)** dominates overall time; individual run range is 47-59s across all configs
- Overlap = `glmocr + caption - overall` (time saved by running captioning in parallel with OCR)

### Selected: `batch_size: 2`

Best glmocr average (34.0s) with moderate VRAM usage (~4 GiB peak) and good captioning overlap (33.3s).
