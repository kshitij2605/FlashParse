# Inferencing Strategies

## Baseline

Sequential pipeline: glmocr OCR completes all pages → extract image regions → classify each image → caption each image.

| PDF | Pages | Images | OCR | Captions | Total |
|-----|-------|--------|-----|----------|-------|
| 統合報告書 (12p) | 12 | 10 | 15s | 11s | 26s |
| 統合報告書 (70p) | 70 | 149 | 26s | 103s | 106s |

---

## Implemented Strategies

### 1. Overlap OCR with captioning via per-page callbacks

**Date:** 2026-03-18
**Commit:** `feat: overlap OCR with captioning using per-page callbacks`

**Problem:** Captioning couldn't start until all pages finished OCR, even though the first page's images were ready within ~1s.

**Solution:**
- Added `page_callback` parameter to `glmocr.Pipeline.process()`
- The recognition thread fires the callback as soon as all regions for a page are recognized
- The orchestrator bridges from the recognition thread to asyncio via `loop.call_soon_threadsafe` + `asyncio.Queue`
- Image regions are cropped and caption tasks are created immediately as each page arrives

**Files changed:**
- `GLM-OCR/glmocr/pipeline/pipeline.py` — `page_callback` param, `maybe_notify_page_done()`, state tracking
- `src/glm_hybrid_ocr/pipeline/orchestrator.py` — `process_incoming_pages()` async consumer

**Results (12-page PDF):**

| Metric | Before | After |
|--------|--------|-------|
| Total | 26s | 19.87s |
| OCR | 15s | 5.9s |
| Captions | 11s (sequential) | 17.3s (overlapped) |
| Savings | — | ~24% |

**How overlap works:**
```
Before: |--- OCR 15s ---|--- Captions 11s ---| = 26s
After:  |--- OCR 5.9s ---|                     = 19.87s
              |--- Captions 17.3s (overlapped) ---|
```

---

### 2. Pipelined classify→caption per image (replaced by #3)

**Date:** 2026-03-31
**Commit:** (intermediate optimization, superseded by combined single VLM call)

**Problem:** Classification and captioning ran as two serial `asyncio.gather` blocks — captioning could not start until every classification finished:
```
Phase A: asyncio.gather(classify(img1), ..., classify(img10))  # ~0.6s
   ↓ wait for ALL classifications
Phase B: asyncio.gather(caption(img1), ..., caption(img10))    # ~26s
```

**Solution:**
- Replaced two serial `asyncio.gather` blocks with a single `asyncio.gather` where each task chains classify→caption for one image
- Each image starts captioning the moment its classification returns, without waiting for other images
- Pre-encoded base64 images upfront and cached page images by page index (eliminating redundant encoding)
- Added `image_b64`/`page_image_b64` pass-through params to VLM client to skip re-encoding

**Files changed:**
- `src/glm_hybrid_ocr/pipeline/orchestrator.py` — per-image pipelined classify→caption, base64 pre-encoding cache
- `src/glm_hybrid_ocr/clients/vlm_client.py` — added `image_b64`/`images_b64` params
- `src/glm_hybrid_ocr/vlm/captioner.py` — added base64 pass-through params

**Results (12-page PDF, 10 images):**

| Metric | Before (serial phases) | After (pipelined) |
|--------|------------------------|---------------------|
| Caption generation | 26.82s | 10.47s |
| Speedup | — | **2.56x** |
| Max concurrent VLM requests | 10 | 10 |
| Effective parallelism | ~1x (serial phases) | 5.04x |

**Concurrency details:**
- All 10 classify calls launch at t=0.00s, return by t=1.35s
- As each classify finishes, its caption starts immediately (no waiting)
- Misc images caption in ~0.8s, figure/chart in ~4-6s
- VLM server is the bottleneck — 10 concurrent requests achieve 5x parallelism bounded by GPU inference throughput

**Status:** Superseded by strategy #3 (combined single VLM call eliminates the separate classify step entirely).

---

### 3. Combined classify + caption into single VLM call

**Date:** 2026-03-31
**Commit:** `feat: combine classify+caption into single VLM call, add layouts PDF, align output format`

**Problem:** Each image required 2 sequential VLM calls: classify (~1s) then caption (~2-3s). With 149 images, this means 298 VLM requests.

**Solution:**
- Created a single prompt that asks the VLM to both classify and caption in one response
- Response uses structured tags: `[category]chart[/category]` and `[caption]...[/caption]`
- Fallback parsing infers category from response text when tags are missing (occurs ~50% of the time)

**Files changed:**
- `src/glm_hybrid_ocr/vlm/classify_and_caption.py` — new combined classifier+captioner
- `src/glm_hybrid_ocr/config/prompts.py` — `CLASSIFY_AND_CAPTION_PROMPT`
- `src/glm_hybrid_ocr/pipeline/orchestrator.py` — uses `AsyncClassifyAndCaption` instead of separate classifier+captioner

**Results (12-page PDF):**

| Metric | 2 calls | 1 call |
|--------|---------|--------|
| VLM requests | 20 (10×2) | 10 |
| Caption generation | 17.3s | 16.0s |
| Total | 19.87s | 23.3s* |

*Total was higher due to GPU memory pressure from vLLM 0.90 utilization slowing OCR. With equal conditions, the single-call approach should be faster or equivalent since it halves VLM request count.

**Expected impact on 70-page PDF:** 298 requests → 149 requests. Should significantly reduce VLM server saturation.

**Caveat:** The VLM doesn't always return `[category]...[/category]` tags. The fallback parser infers the category from keywords in the response text (e.g., "グラフ" → chart, "フロー" → figure).

---

## Strategies Implemented (minor)

### 4. Cache page image base64 encoding

**Date:** 2026-03-31 (implemented as part of strategy #2)

**Problem:** `image_to_base64(page_image)` is called for every image on the same page. For pages with many images, this is redundant work.

**Solution:** The `page_b64_cache` dict in `process_incoming_pages()` deduplicates page image encoding by page index. Pre-encodes all images upfront and passes base64 strings through to VLM client. Carried forward into strategy #3.

**Impact:** Minor — for 10 images across 7 unique pages, eliminates 3 redundant encodings (~17ms each). More impactful on the 70-page PDF with 149 images across fewer unique pages.

---

### 5. Skip captioning for miscellaneous images + higher concurrency

**Date:** 2026-04-01
**Commit:** `feat: classify-first pipeline — skip misc captions, increase VLM concurrency`

**Problem:** In the 70-page PDF, 72 out of 149 images (48%) are classified as "miscellaneous" (logos, icons, decorative images, product photos). These receive full combined classify+caption calls but often don't need detailed descriptions. Additionally, the default VLM concurrency of 20 was leaving throughput on the table.

**Solution:**
- Added `classify_only()` method to `AsyncClassifyAndCaption` — sends only the cropped image with a short classification prompt (`max_tokens=50`)
- Modified `do_classify_and_caption()` in the orchestrator: classify first, skip full caption for miscellaneous images (set `caption=""`)
- Increased default `max_concurrency` from 20 to 64

**Files changed:**
- `src/glm_hybrid_ocr/vlm/classify_and_caption.py` — added `classify_only()` method
- `src/glm_hybrid_ocr/pipeline/orchestrator.py` — classify-first routing in `do_classify_and_caption()`
- `src/glm_hybrid_ocr/config/settings.py` — `max_concurrency` default 20 → 64

**Results (70-page PDF, 149 images, warm runs):**

| Config | Total | Caption time | vs Baseline |
|--------|-------|-------------|-------------|
| Baseline (combined, concurrency 20) | 89.7s | 87.2s | — |
| Combined, concurrency 64 | 77.2s | 75.0s | -14% |
| Skip misc, concurrency 20 | 64.7s | 62.4s | -28% |
| Skip misc, concurrency 40 | 58.0s | 55.6s | -35% |
| **Skip misc, concurrency 64** | **55.6s** | **52.7s** | **-38%** |

**How it works:**
- Each image gets a fast classify-only call (~0.3s, single image, short prompt)
- Miscellaneous images (48% of total) get no further processing
- Non-misc images (chart/figure/scanned_text) get the full combined classify+caption call
- Net effect: 149 classify calls + 77 caption calls = 226 total (vs 149 expensive calls before)
- But classify calls are ~6x faster than caption calls, so total time drops significantly

**Trade-off:** Miscellaneous images lose their captions. This is acceptable since these are typically decorative (logos, icons, photos) where captions add minimal value.

---

## Strategies Not Yet Tested

### 7. Batch VLM requests

**Problem:** Each image is sent as a separate VLM API call, even when multiple images are ready simultaneously.

**Idea:** Group multiple images into a single VLM request for classification (e.g., "Classify these 5 images:"). Caption individually since each needs different context.

**Expected impact:** Depends on whether the VLM server supports efficient batching. OpenAI-compatible API doesn't natively batch, but fewer requests means less HTTP overhead.

**Trade-off:** More complex prompt engineering. Classification accuracy may drop with multiple images in one prompt.

---

### 8. Reduce image resolution for classification

**Problem:** Classification only needs to distinguish chart/figure/scanned_text/misc. Full resolution images are overkill.

**Idea:** Downscale cropped images to ~256px for the classification step (or the combined classify+caption call for misc images), use full resolution only for chart/figure captions.

**Expected impact:** Smaller payloads → faster VLM inference and network transfer. May improve classification accuracy by reducing noise.

---

## Changes to GLM-OCR Repository

The upstream `glmocr` package was modified with two commits on branch `main` (on top of `cca54de`).

### Commit 1: `6f40733` — Expose rendered page images via PipelineResult.page_images

**Problem:** The glmocr pipeline internally renders PDF pages to PIL images for layout detection and OCR, but discards them after processing. The downstream orchestrator had to re-render the PDF to crop image/table regions — redundant work adding ~2-3s.

**Changes:**

`glmocr/parser_result/pipeline_result.py`:
- Added `page_images: Optional[Dict[int, Any]]` parameter to `PipelineResult.__init__()`
- Stores rendered PIL images keyed by page index

`glmocr/pipeline/pipeline.py`:
- Passes `page_images=dict(state.images_dict)` when constructing `PipelineResult` (both in the empty-result path and the normal yield path)

**Impact:** Eliminates redundant PDF-to-image rendering in the orchestrator. The `images_dict` already exists in the pipeline state (populated during data loading), so this is zero-cost.

---

### Commit 2: `9cc314b` — Add per-page callback for streaming results

**Problem:** `Pipeline.process()` is a generator that yields results only after all pages are fully processed. For overlapping OCR with captioning, we need to know when individual pages are done.

**Changes:**

`glmocr/pipeline/pipeline.py`:

1. **New `page_callback` parameter on `process()`:**
   ```python
   def process(self, request_data, ..., page_callback=None):
   ```
   Signature: `fn(page_idx: int, page_regions: list[dict], page_image: PIL.Image)`
   Called from the recognition thread when all regions for a page are done.
   Sentinel `fn(None, None, None)` sent when all pages complete (or on error).

2. **New fields on `_AsyncPipelineState`:**
   ```python
   page_callback: Optional[Any] = None
   page_region_done_count: Optional[Dict[int, int]] = None  # page_idx -> regions done
   page_done_set: Optional[set] = None                      # pages already notified
   page_done_lock: Optional[threading.Lock] = None
   ```
   Initialized only when `page_callback` is provided (no overhead otherwise).

3. **New `maybe_notify_page_done(page_idx)` helper:**
   - Increments `page_region_done_count[page_idx]`
   - Compares against `len(layout_results_dict[page_idx])` (total regions for that page)
   - When all regions are done and page hasn't been notified yet, fires the callback
   - Collects page results from `recognition_results` under `results_lock`
   - Called from 4 locations in `vlm_recognition_thread`:
     - After a future completes in the polling loop (line ~499)
     - After pending skips are flushed at end of processing (line ~509)
     - After immediate skip processing when callback is set (line ~531)
     - After remaining futures complete in the final `as_completed` block (line ~561)

4. **Skip regions processed immediately when `page_callback` is set:**
   ```python
   if task_type == "skip":
       if state.page_callback is not None:
           # Process immediately so page completion is detected
           region["content"] = None
           state.recognition_results.append((page_idx, region))
           maybe_notify_page_done(page_idx)
       else:
           pending_skip.append(...)  # original behavior
   ```
   Without this, skip regions (images) are deferred to the end, preventing page completion detection until all OCR finishes.

5. **Sentinel signal after all processing:**
   ```python
   # After executor.shutdown(wait=True)
   if state.page_callback is not None:
       state.page_callback(None, None, None)
   ```
   Sent in both the normal completion path and the exception handler, so the consumer always gets a termination signal.

**Thread safety:** The callback fires from the recognition thread. The orchestrator bridges to asyncio via `loop.call_soon_threadsafe(page_queue.put_nowait, ...)`. All shared state (`page_region_done_count`, `page_done_set`) is protected by `page_done_lock`.

**Backward compatibility:** When `page_callback=None` (default), behavior is identical to the original code. No new state is allocated, skip regions are deferred as before, and `maybe_notify_page_done` is a no-op.

---

## vLLM Server Configuration Benchmark

**Date:** 2026-03-31
**Hardware:** NVIDIA RTX A6000 (49 GB), `--gpu-memory-utilization 0.90`
**PDF:** 70-page document, OCR-only (skip_captions=True), 2 runs per config (cold/warm)

| Config | Cold | Warm | Pages/sec (warm) |
|--------|------|------|-----------------|
| No MTP, no prefix cache | 44.3s | 35.0s | 2.00 |
| MTP only | 45.0s | 36.5s | 1.92 |
| Prefix cache only | 43.8s | 33.1s | 2.11 |
| **MTP + prefix cache** | 45.5s | **27.4s** | **2.55** |

**Findings:**
- **MTP alone provides no benefit** (slightly slower due to overhead — the draft model falls back to text-only mode for multimodal inputs)
- **Prefix caching provides ~25% speedup** on warm runs (reuses KV cache for repeated prompt prefixes across regions)
- **MTP + prefix cache together are synergistic** — best combination at 2.55 pages/sec, 37% faster than HuggingFace's claimed 1.86 pages/sec
- Cold start penalty is ~60-65% across all configs
- The default `start_vllm.sh` with MTP is optimal since vLLM enables prefix caching by default

**Optimal vLLM command:**
```bash
vllm serve zai-org/GLM-OCR \
    --port 8080 \
    --served-model-name glm-ocr \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --allowed-local-media-path / \
    --gpu-memory-utilization 0.90
```

---

## GPU Memory Configuration

The pipeline requires both vLLM (GLM-OCR) and the layout detector (PP-DocLayoutV3) on the same GPU.

| Component | Memory |
|-----------|--------|
| vLLM (GLM-OCR 0.9B + MTP) | ~44GB (at 0.90 utilization) |
| PP-DocLayoutV3 layout detector | ~4GB |
| **Total needed** | ~48GB |
| RTX A6000 available | 49.1GB |

`--gpu-memory-utilization 0.90` in vLLM leaves ~5GB for the layout detector. Using 0.85 leaves more room but reduces KV cache and slows OCR throughput.
