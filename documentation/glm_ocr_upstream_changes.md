# GLM-OCR Upstream Changes

Changes made to the GLM-OCR repository (`/home/mac/25gitlab/GLM-OCR`) to support FlashParse features. These modifications are on top of the upstream `zai-org/GLM-OCR` package and must be re-applied if the package is updated.

**Repository:** `/home/mac/25gitlab/GLM-OCR`
**Files modified:** `glmocr/pipeline/pipeline.py`, `glmocr/parser_result/pipeline_result.py`
**Installed to:** `/home/mac/.local/lib/python3.10/site-packages/glmocr/` (copied manually after each change)

---

## Change 1: Expose rendered page images via PipelineResult

**Commit:** `6f40733` — 2026-03-18
**Files:** `glmocr/parser_result/pipeline_result.py`, `glmocr/pipeline/pipeline.py`

### Problem

The orchestrator needs page images to crop detected image/table regions for captioning. Without this change, it had to re-render the entire PDF with pypdfium2 after glmocr finished — wasting ~1.5s on a 70-page PDF and duplicating work the pipeline already did internally.

### Change

Add a `page_images` field to `PipelineResult` — a `Dict[int, PIL.Image]` keyed by page index, containing the page images already rendered by the pipeline's data loading thread.

**`glmocr/parser_result/pipeline_result.py`:**
```python
class PipelineResult(BaseParserResult):
    def __init__(
        self,
        json_result,
        markdown_result,
        original_images: List[str],
        layout_vis_dir: Optional[str] = None,
        layout_image_indices: Optional[List[int]] = None,
        page_images: Optional[Dict[int, Any]] = None,   # ← added
    ):
        ...
        self.page_images = page_images                    # ← added
```

**`glmocr/pipeline/pipeline.py`:**
```python
# In the yield statements (2 locations), pass the rendered images:
yield PipelineResult(
    ...
    page_images=dict(state.images_dict),  # ← added
)
```

### Impact

- Eliminates redundant PDF rendering in the orchestrator
- Saves ~1.5s on a 70-page PDF
- Backward compatible — `page_images` defaults to `None`

---

## Change 2: Per-page callback for streaming results

**Commit:** `9cc314b` — 2026-03-18
**Files:** `glmocr/pipeline/pipeline.py`

### Problem

The pipeline yields results only after **all pages** are processed. For a 70-page PDF, this means image captioning can't start until ~35s of OCR completes. But images on page 0 are detected after ~1s — they could be sent to the VLM immediately.

### Change

Add a `page_callback` parameter to `Pipeline.process()` that fires from the recognition thread as soon as all regions for a page are recognised. This enables the orchestrator to start captioning images while OCR continues on remaining pages.

**New parameter:**
```python
def process(
    self,
    request_data,
    ...,
    page_callback=None,  # ← added: fn(page_idx, page_regions, page_image)
):
```

**New state fields in `_AsyncPipelineState`:**
```python
page_callback: Optional[Any] = None
page_region_done_count: Optional[Dict[int, int]] = None  # tracks regions completed per page
page_done_set: Optional[set] = None                       # pages that have fired callback
page_done_lock: Optional[threading.Lock] = None           # protects the above
```

**New function `maybe_notify_page_done(page_idx)`:**
- Called from the recognition thread after each region completes
- Increments `page_region_done_count[page_idx]`
- When count equals total regions for that page (from `layout_results_dict`), fires `page_callback(page_idx, page_results, page_image)` exactly once
- Protected by `page_done_lock` to prevent duplicate callbacks
- Sends sentinel `page_callback(None, None, None)` when all pages are done (or on error)

**Skip region handling change:**
When `page_callback` is set, skip regions (images — `task_type == "skip"`) are processed immediately instead of being deferred to `pending_skip`. This ensures page completion is detected promptly, since skip regions contribute to the page's region count.

### Impact

- 24% reduction in total time on a 12-page PDF (26s → 19.87s)
- Larger documents benefit more — captioning overlaps with OCR for ~30s on a 70-page PDF
- Zero overhead when `page_callback=None` (backward compatible)
- The orchestrator uses this via `loop.call_soon_threadsafe` + `asyncio.Queue` to bridge the sync callback to async captioning tasks

---

## Change 3: Concurrency locks for concurrent Pipeline.process() calls

**Commit:** `2539879` — 2026-04-11
**Files:** `glmocr/pipeline/pipeline.py`

### Problem

The `Pipeline` class was designed for one-at-a-time processing. Calling `process()` concurrently from multiple threads caused two types of crashes:

1. **PDFium C library crash** — pypdfium2 is not thread-safe at the C level. Even with completely separate `PdfDocument` handles for different PDFs, concurrent page rendering corrupts global state. Errors: `free(): invalid size` (fatal), `"Already borrowed"` (400), `"Failed to load document"`.

2. **PyTorch layout detector crash** — The PP-DocLayoutV3 model cannot run concurrent GPU forward passes on the same instance. Internal CUDA state gets corrupted.

However, per-invocation state (queues, dicts, counters, locks) was already isolated — `_create_async_pipeline_state()` creates a fresh `_AsyncPipelineState` for each `process()` call. Only the shared components needed serialization.

### Change

Add two `threading.Lock()` instances to `Pipeline.__init__()`:

```python
# In __init__:
self._pdf_lock = threading.Lock()
self._layout_lock = threading.Lock()
```

**`_pdf_lock`** — wraps the page iteration loop in `data_loading_thread`:
```python
def data_loading_thread() -> None:
    try:
        img_idx = 0
        unit_indices_list: List[int] = []
        with self._pdf_lock:                              # ← added
            for page, unit_idx in self.page_loader.iter_pages_with_unit_indices(
                image_urls
            ):
                state.images_dict[img_idx] = page
                state.page_queue.put(("image", img_idx, page))
                ...
        state.page_queue.put(("done", None, None))
```

The lock is held for the full iteration because the pypdfium2 generator keeps a `PdfDocument` open the entire time.

**`_layout_lock`** — wraps the layout detector call in `_stream_process_layout_batch`:
```python
def _stream_process_layout_batch(self, ...):
    with self._layout_lock:                               # ← added
        layout_results = self.layout_detector.process(
            batch_images, ...
        )
    # Region queue population runs outside the lock
    for img_idx, image, layout_result in zip(...):
        ...
```

### Concurrency diagram

```
PDF A: [load pages]──[layout]──[────────OCR recognition────────]
PDF B:               [load pages]──[layout]──[────────OCR recognition────────]
                     ↑ waits for   ↑ waits    ↑ runs concurrently
                       _pdf_lock    _layout_lock  with PDF A's OCR
```

### What is serialized vs concurrent

| Component | Serialized? | Why |
|---|---|---|
| PDF page rendering (pypdfium2) | Yes (`_pdf_lock`) | C library global state not thread-safe |
| Layout detection (PP-DocLayoutV3) | Yes (`_layout_lock`) | PyTorch model not safe for concurrent forward passes |
| OCR recognition (vLLM HTTP) | **No — fully concurrent** | Thread-safe HTTP connection pool, vLLM handles batching |
| Per-invocation state (queues, dicts) | **No — fully isolated** | Fresh `_AsyncPipelineState` per call |

### Benchmark results (RTX A6000, 70-page PDF, skip_captions)

| Concurrent PDFs | Wall Time | Throughput |
|---|---|---|
| 1 | ~51s | 1.4 pages/s |
| 2 | 66.5s | 2.1 pages/s |
| 3 | 100.4s | 2.1 pages/s |
| 4 | 123.2s | 2.3 pages/s |
| 5 | 165.6s | 2.1 pages/s |

### Impact

- Enables 2+ concurrent PDFs on a single GPU (~50% throughput improvement with 2)
- No additional VRAM — reuses the single layout detector instance
- Backward compatible — locks have zero overhead for single-PDF usage
- FlashParse orchestrator changed `asyncio.Semaphore(1)` → `Semaphore(2)` to allow 2 concurrent PDFs

---

## Summary

| Change | Date | Files | Purpose | Performance Impact |
|---|---|---|---|---|
| Expose page_images | 2026-03-18 | pipeline_result.py, pipeline.py | Eliminate redundant PDF rendering | -1.5s per document |
| Per-page callback | 2026-03-18 | pipeline.py | Overlap captioning with OCR | -24% total time |
| Concurrency locks | 2026-04-11 | pipeline.py | Enable concurrent PDF processing | +50% throughput (2 PDFs) |

All changes are backward compatible and have zero overhead when the new features are not used.
