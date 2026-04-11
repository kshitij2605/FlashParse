# Performance Tuning Guide

How to configure FlashParse for optimal throughput depending on your GPU hardware.

---

## Quick Reference

| Parameter | File | Default | What it controls |
|-----------|------|---------|-----------------|
| `--gpu-memory-utilization` | `scripts/start_vllm.sh` | 0.70 | vLLM GPU memory allocation (model + KV cache) |
| `max_workers` | `glmocr_config.yaml` | 64 | Concurrent OCR region requests to vLLM (not PDFs — see below) |
| `connection_pool_size` | `glmocr_config.yaml` | 256 | HTTP connection pool for OCR region requests |
| `layout.batch_size` | `glmocr_config.yaml` | 2 | Pages per GPU forward pass for layout detection |
| `VLM_MAX_CONCURRENCY` | `.env` | 64 | Concurrent captioning requests to VLM server |
| `page_maxsize` | `glmocr_config.yaml` | 100 | Page image buffer between loader and layout threads |
| `region_maxsize` | `glmocr_config.yaml` | 800 | Cropped region buffer between layout and recognition threads |

---

## Understanding the Pipeline

The system processes **one PDF at a time**. Within that PDF, pages are split into regions (text blocks, tables, formulas, images) which are processed concurrently. All concurrency settings in this guide refer to **concurrent region processing within a single PDF**, not concurrent PDF processing. See [Concurrent PDF Processing](#concurrent-pdf-processing) for multi-PDF scaling.

Three components share the GPU:

```
+------------------------------------------+
| vLLM: GLM-OCR 0.9B + MTP                |  Model weights: ~2.5 GiB (fixed)
|   KV cache: varies with utilization      |  KV cache: scales with utilization
+------------------------------------------+
| PP-DocLayoutV3 layout detector           |  Idle: ~2.1 GiB, Peak: ~5.7 GiB
+------------------------------------------+
| Free headroom                            |  Needed for CUDA temp allocations
+------------------------------------------+
```

The layout detector's peak VRAM scales with `batch_size`:
- batch_size=1: ~3.0 GiB peak
- batch_size=2: ~4.0 GiB peak
- batch_size=4: ~5.7 GiB peak
- batch_size=8: ~7.5 GiB peak

---

## GPU Memory Sizing

### Single GPU Configurations

The key constraint: `vLLM allocation + layout detector peak + headroom ≤ total GPU VRAM`.

| GPU | VRAM | Recommended `gpu-memory-utilization` | Layout `batch_size` | Expected headroom |
|-----|------|--------------------------------------|---------------------|-------------------|
| RTX 3090 / 4090 | 24 GiB | 0.50 | 1 | ~9 GiB |
| RTX A5000 | 24 GiB | 0.50 | 1 | ~9 GiB |
| A30 | 24 GiB | 0.50 | 1 | ~9 GiB |
| A10G | 24 GiB | 0.50 | 1 | ~9 GiB |
| L40S | 48 GiB | 0.70 | 2 | ~9.5 GiB |
| **RTX A6000** | **49 GiB** | **0.70** | **2** | **~8.5 GiB** |
| A100-40GB | 40 GiB | 0.60 | 2 | ~10 GiB |
| A100-80GB | 80 GiB | 0.70 | 4 | ~20 GiB |
| H100 | 80 GiB | 0.70 | 4 | ~20 GiB |

**How to calculate for your GPU:**

```
vLLM allocation = GPU VRAM × utilization
layout peak     = ~3.0 GiB (batch_size=1) to ~5.7 GiB (batch_size=4)
headroom needed = ~3-4 GiB minimum (for CUDA temp allocations)

utilization = (GPU VRAM - layout peak - headroom) / GPU VRAM
```

### Worked examples

**RTX 3090 / RTX 4090 (24 GiB):**
```
Available for vLLM = 24 - 3.0 (layout, bs=1) - 4 (headroom) = 17 GiB
Max utilization = 17 / 24 = 0.71
→ Use 0.50 to be safe (vLLM takes 12 GiB, 12 GiB free for layout + headroom)
→ OCR will be ~50% slower than on a 49 GiB GPU due to CUDA memory pressure
```

**RTX A6000 (49 GiB):**
```
Available for vLLM = 49 - 4.0 (layout, bs=2) - 4 (headroom) = 41 GiB
Max utilization = 41 / 49 = 0.84
→ Use 0.70 (vLLM takes 34.5 GiB, 14.5 GiB free — comfortable headroom)
→ 0.82 works but leaves only 2.7 GiB free at peak
```

**A100-40GB (40 GiB):**
```
Available for vLLM = 40 - 4.0 (layout, bs=2) - 4 (headroom) = 32 GiB
Max utilization = 32 / 40 = 0.80
→ Use 0.60 (vLLM takes 24 GiB, 16 GiB free for layout + headroom)
→ 0.70 also works (28 GiB vLLM, 12 GiB free) — test which is stable
```

**A100-80GB / H100 (80 GiB):**
```
Available for vLLM = 80 - 5.7 (layout, bs=4) - 4 (headroom) = 70.3 GiB
Max utilization = 70.3 / 80 = 0.88
→ Use 0.70 (vLLM takes 56 GiB, 24 GiB free — very comfortable)
→ Could go to 0.80 if you want maximum KV cache, but 0.70 is more than enough
```

**L4 (24 GiB, lower bandwidth):**
```
Same calculation as RTX 3090: use 0.50, batch_size=1
→ L4 has lower memory bandwidth than A6000, so OCR will be slower regardless
→ max_workers=32 is sufficient (GPU compute bottleneck hits earlier)
```

### What happens if utilization is too high?

| Symptom | Cause | Fix |
|---------|-------|-----|
| 5-10x slower OCR (not OOM crash) | GPU memory thrashing — vLLM + layout detector leave <2 GiB free | Lower `gpu-memory-utilization` by 0.05 |
| vLLM fails to start / hangs at "Asynchronous scheduling" | Not enough VRAM to load model + allocate KV cache | Lower `gpu-memory-utilization` significantly |
| CUDA OOM during layout detection | Layout batch too large for remaining VRAM | Lower `layout.batch_size` |

### What happens if utilization is too low?

OCR gets slower. On a 49 GiB A6000:

| Utilization | vLLM VRAM | glmocr time | Notes |
|-------------|-----------|-------------|-------|
| 0.50 | 24.8 GiB | 35.6s | ~50% slower — CUDA memory pressure |
| 0.60 | 29.6 GiB | ~24s | Good |
| 0.70 | 34.5 GiB | ~24s | Optimal for 49 GiB |
| 0.82 | 40.3 GiB | ~24s | Works but only 2.7 GiB headroom |

The slowdown at 0.50 is **not** KV cache exhaustion — each OCR request uses only ~181 tokens, so even 0.50 supports 1,700+ concurrent requests. The slowdown comes from CUDA memory pressure during model forward passes.

---

## OCR Concurrency (max_workers)

**Important:** `max_workers` controls concurrent **region** requests, not concurrent PDFs. The pipeline processes one PDF at a time (see [Concurrent PDF Processing](#concurrent-pdf-processing)). Within that single PDF, the layout detector splits each page into ~15 regions (text blocks, tables, formulas, images). A 70-page PDF produces ~1,000 regions. `max_workers` controls how many of these cropped regions are sent to vLLM simultaneously for OCR.

### Benchmarked on RTX A6000 (49 GiB, utilization=0.70)

| max_workers | glmocr time | Notes |
|-------------|-------------|-------|
| 8 | 70.9s | GPU underutilized |
| 16 | 51.3s | |
| 32 | 40.5s | |
| **64** | **35.2s** | **GPU saturated** |
| 128 | 33.7s | +1s gain (noise) |
| 256 | 34.1s | No improvement |
| 512 | 34.8s | Slower — connection overhead |

**Why it plateaus at 64:** vLLM uses continuous batching — as each region request finishes, the next one fills its slot immediately. At 64 concurrent region requests, the GPU has zero idle time. More requests just queue inside vLLM.

### Recommended max_workers by GPU

| GPU class | max_workers | connection_pool_size |
|-----------|-------------|---------------------|
| 24 GiB (3090, 4090, A5000) | 32 | 128 |
| 40-49 GiB (A6000, A100-40GB) | 64 | 256 |
| 80 GiB (A100-80GB, H100) | 64-128 | 256-512 |

Higher VRAM doesn't require more workers — the GPU compute saturates at 64 for a 0.9B model. Only increase if using a faster GPU architecture (higher FLOPS/memory bandwidth).

`connection_pool_size` should be ≥ 2× `max_workers`. Higher values waste memory for no benefit.

---

## Layout Detection (batch_size)

`layout.batch_size` controls how many pages the PP-DocLayoutV3 detector processes per GPU forward pass.

### Benchmarked on RTX A6000 (70-page PDF, 3 runs averaged)

| batch_size | Avg glmocr | Avg VRAM spike | Overlap with captioning |
|------------|-----------|----------------|------------------------|
| 1 | 36.0s | ~3.0 GiB | 35.5s (best) |
| **2** | **34.0s** | **~4.0 GiB** | **33.3s** |
| 4 | 34.9s | ~5.7 GiB | 32.9s |
| 8 | 36.0s | ~7.5 GiB | 32.8s |
| 16 | crashed | >10 GiB | — |

**Trade-off:** Larger batches are marginally faster for layout detection alone, but delay the first `page_callback` (pages aren't available for captioning until the whole batch finishes layout). batch_size=2 is the sweet spot — fast layout with minimal overlap loss.

### Recommended batch_size by VRAM headroom

| Available headroom | batch_size |
|-------------------|------------|
| < 5 GiB | 1 |
| 5-8 GiB | 2 |
| 8-15 GiB | 2-4 |
| > 15 GiB | 4 |

---

## Concurrent PDF Processing

The pipeline currently processes **one PDF at a time**, enforced by `asyncio.Semaphore(1)`. Additional requests queue in the semaphore (not rejected).

### Why only 1?

Two hard constraints prevent concurrent PDF processing:

1. **PDFium thread safety** — the glmocr pipeline's PDF loader (pypdfium2) is not thread-safe. Two concurrent `Pipeline.process()` calls corrupt PDFium's internal state:
   ```
   RuntimeError: DataLoadingThread: Failed to load document (PDFium: Data format error)
   ```

2. **Layout detector VRAM spikes** — each pipeline instance runs its own layout batches. Two concurrent pipelines spike ~7.2 GiB simultaneously (2 × 3.6 GiB), causing CUDA OOM on most GPUs.

### Multi-GPU scaling

To process multiple PDFs concurrently, run **separate pipeline instances on separate GPUs**:

```
GPU 0: vLLM (port 8080) + Layout detector → API instance 1 (port 8000)
GPU 1: vLLM (port 8081) + Layout detector → API instance 2 (port 8001)
Load balancer → distributes requests across instances
```

Each GPU runs its own independent vLLM + layout detector + API server. This avoids both the PDFium and VRAM constraints.

**Per-GPU throughput:** ~1 page/second for a 70-page PDF with captioning enabled.

### Scaling without additional OCR GPUs

If you only have one GPU for GLM-OCR but multiple VLM GPUs for captioning:
- OCR throughput is fixed at 1 PDF at a time
- Caption throughput scales with VLM concurrency (`VLM_MAX_CONCURRENCY`)
- Increasing VLM capacity reduces per-PDF time but not concurrent PDF count

---

## VLM Captioning Concurrency

`VLM_MAX_CONCURRENCY` (in `.env`) controls how many concurrent image captioning requests are sent to the VLM server. This is independent of OCR concurrency.

| VLM setup | Recommended VLM_MAX_CONCURRENCY |
|-----------|--------------------------------|
| Shared VLM server | 20-40 |
| Dedicated VLM (single GPU) | 40-64 |
| Dedicated VLM (multi-GPU / high-end) | 64-128 |

Caption time is usually the bottleneck for image-heavy documents. The VLM server's GPU compute is the limit, not the concurrency setting. Setting this too high won't improve speed — it just queues more requests on the VLM server.

---

## Queue Buffer Sizes

`page_maxsize` and `region_maxsize` control memory usage for the internal pipeline queues. These are **blocking bounded queues** — when full, the producer thread waits until the consumer frees a slot. No data is lost.

```
Data Loading → [page_queue, max 100] → Layout Detection → [region_queue, max 800] → Recognition
```

| Parameter | Default | RAM usage | When to change |
|-----------|---------|-----------|---------------|
| `page_maxsize` | 100 | ~0.5-2 GiB (100 page images) | Lower to 20-50 on low-RAM systems |
| `region_maxsize` | 800 | ~0.2-0.5 GiB (800 cropped regions) | Rarely needs changing |

For a 1000-page PDF, at most 100 pages are in memory at once. The data loading thread blocks until the layout thread consumes pages.

**When to lower:** If your system has < 16 GiB RAM and you're processing high-DPI PDFs. Set `page_maxsize: 20` to reduce peak RAM.

**When to raise:** Almost never. The recognition thread (vLLM) is always slower than layout detection, so the region_queue is usually full. A larger buffer just wastes RAM.

---

## Example Configurations

### Budget: RTX 3090 (24 GiB VRAM, 32 GiB RAM)

```yaml
# glmocr_config.yaml
pipeline:
  max_workers: 32
  page_maxsize: 50
  region_maxsize: 400
  ocr_api:
    connection_pool_size: 128
  layout:
    batch_size: 1
```

```bash
# start_vllm.sh
vllm serve zai-org/GLM-OCR \
    --gpu-memory-utilization 0.50 \
    ...
```

```bash
# .env
VLM_MAX_CONCURRENCY=40
```

Expected: ~50s per 70-page PDF (OCR ~45s due to lower utilization).

### Standard: RTX A6000 (49 GiB VRAM, 64+ GiB RAM)

```yaml
# glmocr_config.yaml
pipeline:
  max_workers: 64
  page_maxsize: 100
  region_maxsize: 800
  ocr_api:
    connection_pool_size: 256
  layout:
    batch_size: 2
```

```bash
# start_vllm.sh
vllm serve zai-org/GLM-OCR \
    --gpu-memory-utilization 0.70 \
    ...
```

```bash
# .env
VLM_MAX_CONCURRENCY=64
```

Expected: ~52s per 70-page PDF.

### High-end: A100-80GB (80 GiB VRAM, 128+ GiB RAM)

```yaml
# glmocr_config.yaml
pipeline:
  max_workers: 64
  page_maxsize: 100
  region_maxsize: 800
  ocr_api:
    connection_pool_size: 256
  layout:
    batch_size: 4
```

```bash
# start_vllm.sh
vllm serve zai-org/GLM-OCR \
    --gpu-memory-utilization 0.70 \
    ...
```

```bash
# .env
VLM_MAX_CONCURRENCY=128
```

Expected: ~50s per 70-page PDF (slightly faster layout from batch_size=4, more VLM concurrency).

### Multi-GPU cluster (2× A6000)

```
GPU 0: vLLM + layout → API on port 8000
GPU 1: vLLM + layout → API on port 8001
Nginx/HAProxy load balancer → round-robin across ports
```

Each GPU uses the Standard config above. Throughput: 2 concurrent PDFs, ~2 pages/second aggregate.
