# Architectural Decision Records

## ADR-001: Two-Model Architecture (GLM-OCR + Qwen VLM)

**Date:** 2026-03-18
**Status:** Accepted

**Context:** We need to process PDF documents into markdown with image captions. GLM-OCR handles text/table/formula recognition well but skips image regions entirely (returns `content=null`). We need a separate model for image understanding.

**Decision:** Use GLM-OCR (0.9B via vLLM) for text/table/formula OCR and a separate VLM (any OpenAI-compatible vision model) for image classification and captioning. The two models can run on the same or separate GPU servers.

**Consequences:**
- (+) Each model is optimized for its task (small fast OCR model + large capable vision model)
- (+) Models can be scaled independently
- (+) GLM-OCR runs locally with low latency; VLM can be local, remote, or a cloud API
- (-) Two servers to manage and monitor
- (-) Network latency for VLM calls
- (-) More complex orchestration to coordinate the two models

---

## ADR-002: Per-Page Callback for Overlapping OCR with Captioning

**Date:** 2026-03-18
**Status:** Accepted

**Context:** The original pipeline was strictly sequential: run all OCR, then extract images, then caption all images. For a 12-page PDF, OCR takes ~15s and captioning ~11s, totaling 26s. But page 0's images are ready after ~1s of OCR.

**Decision:** Add a `page_callback` parameter to `glmocr.Pipeline.process()` that fires from the recognition thread as each page completes. The orchestrator bridges this to the async event loop via `loop.call_soon_threadsafe` + `asyncio.Queue`, starting image captioning immediately.

**Alternatives considered:**
- **Polling glmocr state**: Would require exposing internal state and complex synchronization. Rejected for tight coupling.
- **Streaming generator**: The pipeline already yields results, but only after all pages complete. Modifying the generator to yield partial results would break the existing API contract.
- **Shared memory / message queue**: Over-engineered for a single-process pipeline.

**Consequences:**
- (+) 24% reduction in total time (26s -> 19.87s on 12-page PDF)
- (+) Captioning and OCR overlap naturally — bigger documents benefit more
- (+) Zero overhead when `page_callback=None` (backward compatible)
- (-) Required modifying the upstream `glmocr` package (2 commits)
- (-) Thread-to-async bridging adds complexity
- (-) Must handle sentinel signal and error cases in both threads

---

## ADR-003: Combined Classify+Caption in Single VLM Call

**Date:** 2026-03-31
**Status:** Accepted

**Context:** Originally each image required 2 sequential VLM calls: classify (chart/figure/scanned_text/misc, ~1s) then caption with a category-specific prompt (~2-3s). For 149 images, this means 298 VLM requests.

**Decision:** Use a single prompt that asks the VLM to both classify and caption in one response, using structured tags `[category]...[/category]` and `[caption]...[/caption]`.

**Alternatives considered:**
- **Keep two calls but pipeline them**: Implemented as an intermediate step (ADR-002.5). Per-image chaining reduced caption time from 26.8s to 10.5s, but still 2x the VLM requests.
- **Classify first, then caption only non-misc images**: Would skip ~57% of captions but still requires separate classify calls. Planned as a future optimization.
- **Use a different model for classification**: A smaller/faster model could classify. Rejected because the single-call approach is simpler and the VLM handles both tasks well.

**Consequences:**
- (+) Halves VLM request count (149 vs 298 for 70-page PDF)
- (+) Simpler pipeline — one async task per image instead of two
- (+) Bigger impact on large documents where VLM server saturates
- (-) VLM doesn't always return structured tags (~50% of responses missing `[category]` tags)
- (-) Requires fallback parsing (keyword matching in response text)
- (-) Category-specific prompt tuning is lost (the combined prompt is more generic)

---

## ADR-004: OpenAI-Compatible API for Both Models

**Date:** 2026-03-18
**Status:** Accepted

**Context:** Both GLM-OCR (via vLLM) and Qwen VLM expose OpenAI-compatible `/v1/chat/completions` endpoints.

**Decision:** Use the OpenAI chat completions API format for all model interactions. The VLM client (`AsyncVLMClient`) sends standard chat messages with base64-encoded images.

**Consequences:**
- (+) Standard API — can swap models without changing client code
- (+) vLLM, SGLang, TGI, and other serving frameworks all support this format
- (+) Well-documented, widely understood format
- (-) No native batching support (each image is a separate HTTP request)
- (-) Base64 encoding increases payload size by ~33%

---

## ADR-005: FastAPI with SSE Progress Streaming

**Date:** 2026-03-18
**Status:** Accepted

**Context:** PDF processing can take 20-100+ seconds. Clients need progress feedback, and the result (ZIP with markdown, images, tables) can be large.

**Decision:** Provide two endpoints:
1. `/api/v1/pdf/process` — simple POST, returns ZIP directly
2. `/api/v1/pdf/process-with-progress` — SSE stream with progress events, then base64-encoded ZIP in the final event (chunked for large outputs)

**Consequences:**
- (+) Simple endpoint for scripts/automation
- (+) SSE endpoint for UIs that need progress bars
- (+) Works through HTTP proxies (SSE is just `text/event-stream`)
- (-) SSE with base64 ZIP is not the most efficient transfer method
- (-) Large ZIPs must be chunked into multiple SSE events

---

## ADR-006: Category-Specific VLM Prompts (Japanese)

**Date:** 2026-03-18
**Status:** Accepted

**Context:** Different image types need different captioning approaches. A chart needs data extraction (numbers, axes, trends). A figure needs relationship description. Scanned text needs exact OCR. The target documents are Japanese.

**Decision:** All VLM prompts are in Japanese. Each image category has a specialized prompt:
- **chart**: Extract all numerical data, axis labels, legends, trends
- **figure**: Describe structure, relationships, and flow
- **scanned_text**: Extract exact text preserving structure
- **miscellaneous**: Concise 1-3 sentence summary

The VLM system message enforces Japanese-only responses.

**Consequences:**
- (+) Higher quality captions tailored to each image type
- (+) Scanned text captions are usable as OCR fallback
- (+) Chart captions preserve numerical data that would be lost in generic descriptions
- (-) Prompts are Japanese-only — would need translation/adaptation for other languages
- (-) More prompts to maintain

---

## ADR-007: deepseek-ocr-api Compatible Output Format

**Date:** 2026-03-31
**Status:** Accepted

**Context:** The existing `deepseek-ocr-api` project produces output ZIPs with specific `metadata.json` and `parsing_metrics.json` formats. Downstream tools consume this format.

**Decision:** Match the deepseek-ocr-api output format exactly:
- `metadata.json` with `pdf_title`, `num_pages`, `num_images` (broken down by category), `num_tables`, `num_captions_generated`
- `parsing_metrics.json` with `timing` (seconds + formatted), `statistics`, `images[]` array (per-image details), `tables[]` array
- `.mmd` extension for markdown files
- `images/` and `tables/` subdirectories

**Consequences:**
- (+) Drop-in replacement for deepseek-ocr-api consumers
- (+) Easier comparison between the two pipelines
- (-) Some fields are redundant between metadata.json and parsing_metrics.json
- (-) Format is now coupled to deepseek-ocr-api conventions

---

## ADR-008: Layout Visualization via glmocr Built-in + img2pdf

**Date:** 2026-03-31
**Status:** Accepted

**Context:** We want a PDF showing detected layout regions superimposed on pages, matching deepseek-ocr-api's output. GLM-OCR has a built-in `save_layout_visualization()` that draws polygonal bounding boxes on page images.

**Decision:** Use GLM-OCR's built-in visualization by passing `save_layout_visualization=True` and a temp directory to the pipeline. After processing, combine the resulting JPEGs into a single PDF using `img2pdf`, then clean up the temp directory.

**Alternatives considered:**
- **Custom rendering with PIL/matplotlib**: Full control over visualization, but duplicates existing functionality.
- **Use reportlab or fpdf2**: More control over PDF generation, but adds a heavy dependency for simple image-to-PDF conversion.

**Consequences:**
- (+) Reuses existing GLM-OCR visualization code — accurate polygonal bounding boxes with color-coded labels
- (+) `img2pdf` is lightweight and lossless (wraps JPEGs directly, no re-encoding)
- (-) Visualization style is controlled by glmocr, not configurable from this project
- (-) Requires temp directory for intermediate JPEGs

---

## ADR-009: Pydantic Settings with Environment Variable Overrides

**Date:** 2026-03-18
**Status:** Accepted

**Context:** The pipeline needs configuration for VLM endpoint, GLM-OCR config path, API host/port, and various tuning parameters. Configuration should work for both development and deployment.

**Decision:** Use `pydantic-settings` with `BaseSettings` classes. Each settings group has an environment variable prefix (`VLM_`, `GLMOCR_`, `API_`). Defaults are provided for all settings.

**Consequences:**
- (+) Type-safe configuration with validation
- (+) Environment variables work naturally in Docker/Kubernetes
- (+) Sensible defaults for local development
- (-) Two configuration systems: Pydantic settings for this project + YAML for glmocr pipeline

---

## ADR-010: Async httpx for VLM Client

**Date:** 2026-03-18
**Status:** Accepted

**Context:** The VLM server handles concurrent requests. We need to fire many captioning requests in parallel (up to 149 for a large document).

**Decision:** Use `httpx.AsyncClient` with connection pooling for all VLM requests. Concurrency is limited by an `asyncio.Semaphore` (default: 20 concurrent requests).

**Alternatives considered:**
- **aiohttp**: Similar capability, but httpx has a cleaner API and better typing.
- **requests + thread pool**: Would work but loses the benefits of async/await for coordinating with the rest of the pipeline.
- **OpenAI Python SDK**: Would handle the API format natively, but adds a heavy dependency and doesn't support custom concurrency control as cleanly.

**Consequences:**
- (+) True async I/O — no threads wasted waiting for HTTP responses
- (+) Semaphore prevents overwhelming the VLM server
- (+) Connection pooling reduces TCP handshake overhead
- (-) Must manually construct OpenAI-compatible request payloads
