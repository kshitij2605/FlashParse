# System Architecture

## Overview

GLM Hybrid OCR is a two-model document processing pipeline:

1. **GLM-OCR** (0.9B parameter model via vLLM) handles text, table, and formula recognition
2. **VLM** (any OpenAI-compatible vision model) handles image classification and captioning

These two models can run on the same or separate GPU servers and are coordinated by an async Python orchestrator that overlaps their work for maximum throughput.

## System Diagram

```
                           +-----------+
                           |  PDF File |
                           +-----------+
                                 |
                                 v
+================================================================+
|                     FastAPI Server (:8000)                      |
|  POST /api/v1/pdf/process          (returns ZIP)               |
|  POST /api/v1/pdf/process-with-progress  (SSE + ZIP)           |
+================================================================+
                                 |
                                 v
+================================================================+
|                   AsyncPDFPipeline (orchestrator.py)            |
|                                                                |
|  +---------------------------+  +----------------------------+ |
|  | Thread Pool (executor)    |  | Async Event Loop           | |
|  |                           |  |                            | |
|  | glmocr.Pipeline.process() |  | process_incoming_pages()   | |
|  |   - Data loading          |  |   - Crop image regions     | |
|  |   - Layout detection      |  |   - Save to disk           | |
|  |   - VLM recognition       |  |   - Fire caption tasks     | |
|  |                           |  |                            | |
|  | page_callback(idx, -------|->| asyncio.Queue              | |
|  |   regions, page_image)    |  |   (thread-safe bridge)     | |
|  +---------------------------+  +----------------------------+ |
|                                          |                     |
|                                          v                     |
|                                 +------------------+           |
|                                 | asyncio.Tasks    |           |
|                                 | (caption per img)|           |
|                                 +------------------+           |
+================================================================+
         |                                  |
         v                                  v
+------------------+              +------------------+
| vLLM Server      |              | VLM Server       |
| localhost:8080   |              | (configurable)   |
|                  |              |                  |
| GLM-OCR 0.9B    |              | Any vision model |
| + MTP decoding   |              | (OpenAI-compat)  |
| + PP-DocLayoutV3 |              |                  |
|                  |              | /v1/chat/complete |
| OpenAI-compat API|              +------------------+
| /v1/chat/complete|
+------------------+
         |                                  |
         v                                  v
+================================================================+
|                    Output Assembly                              |
|                                                                |
|  assemble_markdown(json_result, image_infos)                   |
|    - Merge OCR text with captions                              |
|    - Format per category (chart/figure/scanned_text/misc)      |
|    - Generate metadata.json, parsing_metrics.json              |
|    - Generate layouts PDF from visualization images             |
+================================================================+
         |
         v
+------------------+
| Output ZIP       |
|  .mmd            |
|  metadata.json   |
|  metrics.json    |
|  images/         |
|  tables/         |
|  _layouts.pdf    |
+------------------+
```

## Component Details

### 1. GLM-OCR Pipeline (glmocr package)

The `glmocr` package runs a three-thread pipeline:

```
Thread 1: Data Loading
  - Renders PDF pages to PIL images (pypdfium2)
  - Queues page images for layout detection

Thread 2: Layout Detection
  - PP-DocLayoutV3 (PaddlePaddle) detects 25 region types
  - Classifies regions: text, table, formula, image, header, footer, etc.
  - Outputs bounding boxes (normalized 0-1000 scale)

Thread 3: VLM Recognition
  - Sends each region to GLM-OCR vLLM server
  - Text regions → OCR text
  - Table regions → markdown table
  - Formula regions → LaTeX
  - Image regions → skipped (content=None)
  - Fires page_callback when all regions for a page complete
```

**Key data flow**: The pipeline yields a `PipelineResult` containing:
- `json_result`: `list[list[dict]]` — pages of regions, each with `{index, label, content, bbox_2d, native_label, polygon}`
- `page_images`: `dict[int, PIL.Image]` — rendered page images keyed by page index

### 2. Async/Sync Bridge

GLM-OCR is threaded; VLM captioning is async. The bridge works as follows:

```python
# In GLM-OCR recognition thread (Thread 3):
page_callback(page_idx, page_regions, page_image)

# Bridge to async (thread-safe):
loop.call_soon_threadsafe(page_queue.put_nowait, (page_idx, regions, image))

# In async event loop:
async def process_incoming_pages():
    while True:
        page_idx, regions, image = await page_queue.get()
        if page_idx is None:  # sentinel
            break
        for region in regions:
            if is_image_region(region):
                task = asyncio.create_task(classify_and_caption(...))
```

This achieves overlap: while OCR processes page N+1, VLM captions page N's images.

### 3. VLM Classification and Captioning

Each image region goes through a single VLM call that returns both classification and caption:

**Categories:**
| Category | Description | Caption Style |
|----------|-------------|---------------|
| `chart` | Bar charts, line graphs, pie charts | Detailed data extraction (numbers, axes, trends) |
| `figure` | Diagrams, flowcharts, illustrations | Relationship and structure description |
| `scanned_text` | Photographs of text, screenshots | Exact text extraction preserving structure |
| `miscellaneous` | Logos, icons, decorative, photos | Concise 1-3 sentence summary |

**Response parsing:**
1. Look for `[category]...[/category]` and `[caption]...[/caption]` tags
2. If tags are missing (~50% of responses), infer category from keywords in the response text (e.g., "グラフ" -> chart, "フロー" -> figure, "テキスト" -> scanned_text)
3. Full response text becomes the caption if `[caption]` tags are absent

### 4. Markdown Assembly

The assembler merges GLM-OCR's structured output with VLM captions:

```
For each page:
  For each region (ordered by index):
    - text/formula: insert content directly
    - table: insert markdown table
    - image: insert image reference + formatted caption
      - chart: blockquote with "Chart Description:" prefix
      - figure: italic text
      - scanned_text: code block
      - misc: italic text
  Insert page separator (---)
```

### 5. Layout Visualization

GLM-OCR's built-in `save_layout_visualization()` renders detected bounding boxes (polygonal) on page images. The orchestrator:
1. Passes a temp directory to GLM-OCR for saving visualization JPEGs
2. After processing, combines all JPEGs into a single PDF using `img2pdf`
3. Cleans up the temp directory

## Data Models

```python
@dataclass
class ImageInfo:
    page_idx: int               # Page number (0-indexed)
    region_idx: int             # Region index within page
    bbox_2d: list[int]          # [x1, y1, x2, y2] normalized 0-1000
    cropped: PIL.Image          # Cropped region image
    label: str                  # "image"
    category: ImageCategory     # "chart" | "figure" | "scanned_text" | "miscellaneous"
    caption: str                # VLM-generated caption text
    image_filename: str         # "page{idx}_{region_idx}.jpg"

@dataclass
class PipelineResult:
    markdown: str               # Final assembled markdown
    pages_processed: int
    images_extracted: int
    tables_extracted: int
    image_infos: list[ImageInfo]
    processing_times: dict      # Phase timings in seconds
```

## Configuration Architecture

```
Settings (Pydantic BaseSettings)
  ├── VLMSettings          # VLM endpoint, model, API key, concurrency, timeout
  │     env prefix: VLM_
  ├── GLMOCRPipelineSettings  # Path to glmocr_config.yaml
  │     env prefix: GLMOCR_
  └── APISettings          # FastAPI host and port
        env prefix: API_

glmocr_config.yaml
  ├── pipeline
  │     ├── max_workers: 64
  │     ├── ocr_api          # vLLM connection (localhost:8080, glm-ocr model)
  │     ├── page_loader      # Sampling params, DPI
  │     ├── layout            # PP-DocLayoutV3 config, 25 label classes
  │     └── result_formatter  # Label-to-task mapping, bbox merging
  └── logging
```

## Concurrency Model

```
+------------------------------------------------------------------+
|                        Process Timeline                           |
+------------------------------------------------------------------+

Thread Pool (1 thread):
  |== Data Load ==|== Layout Detect ==|== VLM Recognition ==|
  |    (render)   |  (PP-DocLayoutV3) |  (GLM-OCR vLLM)    |
                                      |                     |
                              page_callback fires per page  |
                                      |                     |
Async Event Loop:                     v                     |
  |              |== Crop + Caption (page 0) ==|            |
  |              |     |== Caption (page 1) ===|            |
  |              |     |   |== Caption (page 2) ==|         |
  |              |     |   |   ...                          |
  |              |     |   |         |== Caption (page N) ==|
  |                                                         |
  |== Markdown Assembly ==|                                 |
  |== Output Files ==|                                      |

VLM Server (remote):
  Handles up to 20 concurrent requests (semaphore-limited)
  Each request: classify + caption in ~2-5s depending on category
```

## GPU Memory Layout

Both GLM-OCR components share a single GPU:

```
RTX A6000 (49.1 GB)
+--------------------------------------------------+
| vLLM: GLM-OCR 0.9B + MTP speculative decoding   |
| gpu-memory-utilization = 0.90                    |
| ~44 GB (model weights + KV cache)                |
+--------------------------------------------------+
| PP-DocLayoutV3 layout detector                    |
| ~4 GB                                             |
+--------------------------------------------------+
| Free headroom: ~1 GB                              |
+--------------------------------------------------+
```

The VLM for captioning can run on a separate GPU server or be a cloud API (any OpenAI-compatible endpoint).

## Output File Specifications

### metadata.json

```json
{
  "pdf_title": "document.pdf",
  "num_pages": 12,
  "num_images": {
    "total": 10,
    "num_chart_images": 1,
    "num_figure_images": 3,
    "num_scanned_text_images": 0,
    "num_miscellaneous_images": 6
  },
  "num_tables": 1,
  "num_captions_generated": 10
}
```

### parsing_metrics.json

```json
{
  "pdf_title": "document.pdf",
  "pdf_path": "/absolute/path/to/document.pdf",
  "output_path": "/absolute/path/to/output",
  "run_timestamp": "2026-03-31T14:30:00.000000",
  "model": "glm-ocr (vLLM) + PP-DocLayoutV3",
  "config": {
    "glmocr_config": "glmocr_config.yaml",
    "vlm_model": "your-model-name",
    "vlm_endpoint": "http://your-vlm-server/v1/chat/completions"
  },
  "timing": {
    "glmocr_pipeline_seconds": 23.98,
    "glmocr_pipeline_formatted": "23.9s",
    "caption_generation_seconds": 16.0,
    "caption_generation_formatted": "16.0s",
    "markdown_assembly_seconds": 0.01,
    "markdown_assembly_formatted": "0.0s",
    "overall_total_seconds": 23.3,
    "overall_formatted": "23.3s",
    "average_per_page_seconds": 1.94,
    "average_per_page_formatted": "1.9s"
  },
  "statistics": {
    "num_pages": 12,
    "num_images": { "total": 10, "..." : "..." },
    "num_tables": 1,
    "num_captions_generated": 10
  },
  "images": [
    {
      "name": "page0_3.jpg",
      "page_index": 0,
      "category": "chart",
      "bbox": [100, 200, 900, 800],
      "has_caption": true
    }
  ],
  "tables": [
    {
      "name": "page1_2.jpg",
      "page_index": 1,
      "bbox": [50, 100, 950, 600]
    }
  ]
}
```

## GLM-OCR JSON Result Format

The GLM-OCR pipeline returns a list of pages, each containing a list of region dicts:

```json
[
  [
    {
      "index": 0,
      "label": "text",
      "content": "Recognized text content...",
      "bbox_2d": [x1, y1, x2, y2],
      "native_label": "paragraph_text",
      "polygon": [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    },
    {
      "index": 1,
      "label": "table",
      "content": "| Header | ... |\n|--------|-----|\n| Cell | ... |",
      "bbox_2d": [x1, y1, x2, y2]
    },
    {
      "index": 2,
      "label": "image",
      "content": null,
      "bbox_2d": [x1, y1, x2, y2]
    }
  ]
]
```

- `bbox_2d` values are normalized to 0-1000 scale
- `label` is one of: `text`, `table`, `formula`, `image`
- `content` is `null` for image regions (skipped by OCR)
- `polygon` provides precise polygonal boundaries for layout visualization

## Label Mapping

PP-DocLayoutV3 detects 25 native labels. These are mapped to 4 task types:

| Task | Native Labels |
|------|--------------|
| **text** (OCR) | text, content, paragraph_title, reference_content, title, abstract, author_or_affiliation, content_block |
| **table** (Table Recognition) | table |
| **formula** (Formula Recognition) | display_formula, inline_formula |
| **skip** (Image extraction) | chart, image |
| **abandon** (Filtered out) | header, footer, footnote, page_number, sidebar, code_block, toc, etc. |
