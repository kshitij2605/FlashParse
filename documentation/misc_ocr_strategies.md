# Misc Image OCR Strategies

## Problem

GLM-OCR skips image regions (content=None) during layout-based processing. The classify-first pipeline (strategy #5 in inferencing_strategies.md) further skips VLM captioning for miscellaneous images. This means text embedded in misc images (logos with text, decorative banners, promotional images) is lost entirely.

## Goal

Extract text from miscellaneous images without significantly increasing total processing time.

## Constraints

- GLM-OCR vLLM (port 8080) is busy during the glmocr pipeline phase -- cannot send misc OCR requests concurrently
- VLM captioning (external server, difgpu01:9000) runs in parallel with glmocr -- no contention
- The 70-page PDF has 149 images, ~83-95 classified as miscellaneous (varies by classify method)
- VLM server is shared, causing 10-20s variability between runs

## Test Setup

- **PDF:** 2022統合報告書.pdf (70 pages, 149 images)
- **Hardware:** NVIDIA RTX A6000 49GB
- **vLLM:** GLM-OCR 0.9B, MTP-1, gpu-memory-utilization 0.82, prefix caching enabled (default)
- **VLM:** Qwen3.5-122B-A10B on difgpu01:9000 (shared, load varies)
- **Protocol:** 2 runs per strategy, results from both runs shown
- **Date:** 2026-04-10

---

## Strategies

### A. No misc OCR (baseline)

Classify-first pipeline as committed. Misc images get `caption=""`. Non-misc get `classify_only()` + `classify_and_caption()` (2 VLM calls per image, redundant classification).

```
glmocr (8080)    ████████████████████████████████████  35s
classify+caption   ░░████████████████████████████████████████  42s (overlaps)
total:           |──────────────── 44.3s ─────────────────|
```

| Run | glmocr | caption_gen | misc_ocr | total | captions |
|-----|--------|-------------|----------|-------|----------|
| 1 | 36.2s | 41.8s | N/A | 44.4s | 66 |
| 2 (warm) | **35.5s** | **42.3s** | **N/A** | **44.3s** | **68** |

---

### B. Deferred sequential misc OCR

Same as A, plus: after glmocr + captions finish, OCR all misc images via GLM-OCR (port 8080 is free). Uses `classify_only()` + `classify_and_caption()` for non-misc (2 VLM calls, redundant).

```
glmocr (8080)    ████████████████████████████████████  35s
classify+caption   ░░████████████████████████████████████████████████████████████  60s
                                                                                  misc OCR ████████████████  16s
total:           |──────────────────────── 78.8s ─────────────────────────────────|
```

| Run | glmocr | caption_gen | misc_ocr | total | captions |
|-----|--------|-------------|----------|-------|----------|
| 1 | 37.1s | 42.5s | 15.3s | 60.6s | 147 |
| 2 (warm) | **35.0s** | **59.9s** | **15.5s** | **78.8s** | **147** |

**Note:** Run 2 caption_gen was slower (59.9s vs 42.5s) due to VLM server load. The misc OCR time is consistent at ~15-16s.

---

### C. Two-phase parallel (classify during glmocr, then caption + misc OCR in parallel)

Phase 1: `classify_only()` overlaps with glmocr. Phase 2: after glmocr finishes, run `caption_only()` (VLM) + misc OCR (GLM-OCR) in parallel. Uses `caption_only()` (1 VLM call per non-misc, no redundant classification).

```
glmocr (8080)    ████████████████████████████████████  35s
classify (VLM)     ░░████░░░░░░░░░░░░░░░░░░░░░░░░░░░
                                                      caption_only (VLM) ████████████████████████████████████████  41s
                                                      misc OCR (8080)    ████████████████████████████████████████  41s (parallel)
total:           |──────────────────────── 78.1s ─────────────────────────────────|
```

| Run | glmocr | caption+misc (parallel) | total | captions |
|-----|--------|-------------------------|-------|----------|
| 1 | 35.5s | 40.9s | 76.3s | 147 |
| 2 (warm) | **37.1s** | **40.9s** | **78.1s** | **147** |

**Observation:** Misc OCR (16s) is fully hidden behind caption time (41s). But caption-glmocr overlap is lost — captioning only starts after glmocr finishes. Total ends up similar to B because the overlap savings (~19s) are offset by not paying sequential misc OCR (~16s).

---

### D. Deferred sequential + caption_only

Same flow as B (caption overlaps glmocr, misc OCR deferred after captions), but uses `classify_only()` + `caption_only()` instead of `classify_only()` + `classify_and_caption()`. Eliminates redundant second classification.

```
glmocr (8080)    ████████████████████████████████████  35s
classify+caption_only  ░░████████████████████████████████████████████████████████  57s
                                                                                  misc OCR ████████████████  17s
total:           |──────────────────────── 76.3s ────────────────────────────────|
```

| Run | glmocr | caption_gen | misc_ocr | total | captions |
|-----|--------|-------------|----------|-------|----------|
| 1 | 37.8s | 55.9s | 15.9s | 74.5s | 147 |
| 2 (warm) | **35.1s** | **57.3s** | **16.6s** | **76.3s** | **147** |

**Observation:** Categories differ from B (83 misc vs 95 misc) because classify_only produces different categories than classify_and_caption's internal classification. Slight improvement over B (~2.5s) but within VLM noise.

---

### E. Early misc OCR + overlapped captioning (FINAL)

Hybrid of B and C: caption tasks start during glmocr via page_callback (like B) for overlap, AND misc OCR starts immediately when glmocr finishes (like C), not waiting for captions. Uses `classify_only()` + `classify_and_caption()`.

```
glmocr (8080)    ████████████████████████████████████  35s
classify+caption   ░░████████████████████████████████████████████████  48s (33s overlap with glmocr)
                                                      misc OCR (8080) ████████████████  15s (parallel with remaining captions)
total:           |──────────────────── 50.6s ────────────────────|
```

| Run | glmocr | caption_gen | misc_ocr | total | captions |
|-----|--------|-------------|----------|-------|----------|
| 1 | 35.8s | 61.3s | 27.5s (parallel) | 63.4s | 146 |
| 2 (warm) | **35.1s** | **48.3s** | **15.5s (parallel)** | **50.6s** | **145** |

**Observation:** Full caption-glmocr overlap restored (~33s). Misc OCR (15.5s) fully hidden behind remaining caption time (~15s after glmocr ends). This is the best strategy — only 6.3s overhead vs baseline A (50.6s vs 44.3s), while extracting text from all 95 misc images.

---

## Results Summary

| Strategy | Run 1 | Run 2 (warm) | Misc OCR time | Captions | Misc images |
|----------|-------|-------------|---------------|----------|-------------|
| **A. No misc OCR** | 44.4s | **44.3s** | N/A | 68 | 95 (empty) |
| B. Deferred sequential | 60.6s | 78.8s | 15.5s | 147 | 95 |
| C. Two-phase parallel | 76.3s | 78.1s | (parallel) | 147 | 83 |
| D. Deferred + caption_only | 74.5s | 76.3s | 16.6s | 147 | 83 |
| **E. Early misc OCR + overlap** | 63.4s | **50.6s** | 15.5s (parallel) | 145 | 95 |

### Key Findings

1. **Strategy E is the clear winner at 50.6s** — only 6.3s overhead over baseline A (44.3s) while extracting text from all 95 misc images.

2. **Caption-glmocr overlap is critical**: Strategy E preserves ~33s of overlap (captions start during glmocr via page_callback). Strategies B-D lost this overlap, costing 31-35s total.

3. **Misc OCR is fully hidden**: The 15.5s misc OCR runs in parallel with remaining caption tasks after glmocr finishes. Since caption_gen takes ~15s longer than glmocr, misc OCR adds zero additional wall time.

4. **VLM server variability**: caption_generation varies 40-63s between runs (shared server). Run 1 was slower (63.4s total) due to VLM load.

5. **Misc OCR consistently ~16s** for 95 misc images. This is the GLM-OCR processing time through the vision encoder.

6. **Category classification**: `classify_only()` + `classify_and_caption()` finds 95 misc images. The second call occasionally reclassifies.

## Recommendation

**Strategy E (early misc OCR + overlapped captioning)** is the winner:
- **50.6s total** — only 14% overhead vs no-misc-OCR baseline (44.3s)
- Full caption-glmocr overlap preserved (~33s overlap)
- Misc OCR fully hidden behind remaining caption time (zero wall-time cost)
- Extracts text from all 95 misc images (145 total captions vs 68 baseline)
- `MISC_OCR_ENABLED` toggle available to disable if needed

If misc OCR is not needed, **Strategy A** (no misc OCR, 44.3s) remains the fastest.

The `caption_only()` optimization (D) saves ~1 redundant VLM call per non-misc image but changes category distributions. This can be adopted independently as it's orthogonal to the misc OCR strategy choice.
