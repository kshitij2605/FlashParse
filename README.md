# GLM Hybrid OCR

The fastest open-source end-to-end document parsing system. Extracts structured text, tables, and formulas via OCR while simultaneously classifying and captioning images — all in a single pipeline pass at **~1.0s per page**.

Combines **GLM-OCR** (via vLLM) for text/table/formula recognition with any **OpenAI-compatible Vision Language Model** for image classification and captioning. OCR and captioning run in parallel — captioning starts as each page completes, not after the entire document finishes.

### Per-Element Speed

| Element | Average Speed |
|---------|--------------|
| Text OCR (per page) | ~0.6s |
| Table recognition (per table) | ~0.6s |
| Image classify + caption (per image) | ~0.9s |
| **Total (per page, overlapped)** | **~1.0s** |

OCR and captioning overlap, so the total per-page time is less than the sum of individual elements.

*Benchmarked with GLM-OCR 0.9B (vLLM, MTP-1 speculative decoding) for OCR and Qwen/Qwen3.5-35B-A3B (vLLM, MTP-1 speculative decoding) for image captioning, each on a single NVIDIA RTX A6000 (49 GB) for high throughput.*

## Architecture Overview

The system handles any document type through two processing paths:

- **Visually structured documents** (PDF, DOCX, PPTX, images, etc.) are converted to PDF format and processed through the full OCR + captioning pipeline
- **Already structured / text-based documents** (TXT, CSV, XLSX, HTML) are routed to a separate direct extraction pipeline that converts them to markdown without OCR

```
Any Document (PDF, DOCX, PPTX, images, TXT, CSV, XLSX, HTML, ...)
    |
    v
+---------------------------------------------------+
|              File Format Router                    |
+---------------------------------------------------+
    |                              |
    v (visual)                     v (text-based)
Convert to PDF                 Direct Extraction
(LibreOffice headless)         (no OCR needed)
    |                              |
    v                              v
+-------------------+       +-------------------+
|   GLM-OCR (vLLM)  |       |  Text Extractors   |
|   Text/Table/     |       |  TXT → raw text    |
|   Formula OCR     |       |  CSV → md table    |
|   + Layout Detect |       |  XLSX → md tables  |
+-------------------+       |  HTML → cleaned md  |
    |                       +-------------------+
    | per-page callback            |
    v                              v
+---------------------------------------------------+
|           AsyncPDFPipeline (orchestrator)          |
|  - Overlaps OCR with captioning                    |
|  - Bridges sync GLM-OCR threads to async VLM       |
+---------------------------------------------------+
    |                              |
    v                              v
+-------------------+       +-------------------+
|  VLM Server       |       |  Markdown Assembly |
|  Classify+Caption  |       |  + Output Files    |
|  (async, batched)  |       |                   |
+-------------------+       +-------------------+
    |
    v
Output: .mmd + metadata.json + parsing_metrics.json + images/ + tables/ + layouts.pdf
```

## Features

- **Multi-format input**: Accepts any document type — visual formats (PDF, DOCX, PPTX, images) go through OCR; text-based formats (TXT, CSV, XLSX, HTML) are extracted directly
- **Flexible VLM backend**: Works with any OpenAI-compatible vision model (vLLM, SGLang, TGI, OpenAI, etc.)
- **Overlapped processing**: Captioning starts as pages complete OCR, not after the entire document finishes
- **Combined classify+caption**: Single VLM call per image (halves request count vs separate calls)
- **Category-specific captioning**: Different prompts for charts (data extraction), figures (relationship description), scanned text (OCR), and miscellaneous images
- **Layout visualization**: PDF output with detected regions superimposed on pages
- **SSE progress streaming**: Real-time progress updates via Server-Sent Events

## Prerequisites

### Hardware

- **GPU memory**: ~48 GB for the GLM-OCR vLLM server (tested on NVIDIA RTX A6000 49 GB)
- **LibreOffice** (optional): Required only for non-PDF input (DOCX, PPTX, etc.). Install with `apt install libreoffice-core`

### External Services

1. **vLLM server** for GLM-OCR (localhost:8080)
2. **VLM server** for image captioning — any OpenAI-compatible endpoint with vision support

## Quick Start

### 1. Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Configure VLM

Copy the example environment file and set your VLM endpoint:

```bash
cp .env.example .env
```

Edit `.env` with your VLM provider:

```bash
# Using a local vLLM/SGLang server
VLM_ENDPOINT=http://localhost:8008/v1/chat/completions
VLM_API_KEY=your-api-key
VLM_MODEL=Qwen/Qwen2.5-VL-72B-Instruct

# Or using OpenAI
VLM_ENDPOINT=https://api.openai.com/v1/chat/completions
VLM_API_KEY=sk-...
VLM_MODEL=gpt-4o

# Or any other OpenAI-compatible provider
VLM_ENDPOINT=https://your-provider.com/v1/chat/completions
VLM_API_KEY=your-key
VLM_MODEL=your-model-name
```

### 3. Start vLLM server for GLM-OCR

```bash
./scripts/start_vllm.sh
```

This starts vLLM with GLM-OCR, MTP speculative decoding, and 90% GPU memory utilization (leaving room for the layout detector).

### 4. Start API server

```bash
python -m glm_hybrid_ocr.api.main
```

The API starts on `http://0.0.0.0:8000` by default.

### 5. Process a PDF

```bash
# Any document format
python examples/client_example.py document.pdf /path/to/output
python examples/client_example.py report.docx /path/to/output
python examples/client_example.py data.xlsx /path/to/output
python examples/client_example.py slides.pptx /path/to/output

# Folder of mixed documents
python examples/client_example.py /path/to/doc_folder /path/to/output

# Skip captioning (faster, OCR only)
python examples/client_example.py document.pdf output --skip-captions

# Custom DPI
python examples/client_example.py document.pdf output --dpi 300
```

### 6. Direct Python usage

```python
import asyncio
from glm_hybrid_ocr.config.settings import Settings
from glm_hybrid_ocr.pipeline.orchestrator import AsyncPDFPipeline

async def main():
    settings = Settings.load()
    pipeline = AsyncPDFPipeline(settings)

    result = await pipeline.process(
        pdf_path="document.pdf",
        output_dir="./output",
    )
    print(f"{result.pages_processed} pages, {result.images_extracted} images")
    await pipeline.close()

asyncio.run(main())
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/health` | Health check (versioned) |
| `POST` | `/api/v1/pdf/process` | Process PDF, returns ZIP |
| `POST` | `/api/v1/pdf/process-with-progress` | Process PDF with SSE progress streaming |

### POST /api/v1/pdf/process

**Parameters** (multipart form):
- `file` (required): PDF file
- `skip_captions` (optional): `"true"` to skip VLM captioning
- `dpi` (optional): rendering DPI (default: `"200"`)

**Response**: ZIP file with response headers `X-Pages-Processed`, `X-Images-Extracted`, `X-Tables-Extracted`, `X-Processing-Time`.

### POST /api/v1/pdf/process-with-progress

Same parameters. Returns `text/event-stream` with progress events followed by the ZIP as base64 in the final event.

## Output Format

```
output/
  {pdf_name}_with_captions.mmd    # Markdown with embedded captions
  {pdf_name}_layouts.pdf          # Layout visualization
  metadata.json                   # Page/image/table counts
  parsing_metrics.json            # Timing, config, per-image details
  images/                         # Extracted image crops (JPEG)
  tables/                         # Extracted table crops (JPEG)
```

## Configuration

All settings support environment variable overrides (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_ENDPOINT` | `http://localhost:8008/v1/chat/completions` | OpenAI-compatible VLM endpoint |
| `VLM_API_KEY` | (empty) | API key for VLM provider |
| `VLM_MODEL` | **(required)** | Model name (e.g., `gpt-4o`, `Qwen/Qwen2.5-VL-72B-Instruct`) |
| `VLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `VLM_TOP_P` | `0.1` | Top-p sampling |
| `VLM_MAX_TOKENS_CAPTION` | `2048` | Max tokens for caption generation |
| `VLM_MAX_CONCURRENCY` | `20` | Max concurrent VLM requests |
| `VLM_TIMEOUT` | `120.0` | Request timeout in seconds |
| `GLMOCR_CONFIG_PATH` | `glmocr_config.yaml` | Path to GLM-OCR config |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |

See `glmocr_config.yaml` for GLM-OCR pipeline settings (layout detection, OCR API, worker concurrency, label mapping).

## Supported VLM Providers

Any OpenAI-compatible `/v1/chat/completions` endpoint with vision (image) support:

| Provider | Example Endpoint | Example Model |
|----------|-----------------|---------------|
| vLLM | `http://localhost:8008/v1/chat/completions` | `Qwen/Qwen2.5-VL-72B-Instruct` |
| SGLang | `http://localhost:30000/v1/chat/completions` | `Qwen/Qwen2.5-VL-72B-Instruct` |
| OpenAI | `https://api.openai.com/v1/chat/completions` | `gpt-4o` |
| Together AI | `https://api.together.xyz/v1/chat/completions` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` |
| Any OpenAI-compatible | `https://your-provider/v1/chat/completions` | Your model name |

## Project Structure

```
src/glm_hybrid_ocr/
  api/
    main.py                  # FastAPI app with lifespan
    routes/pdf.py            # PDF processing endpoints
  clients/
    vlm_client.py            # Async HTTP client (OpenAI-compatible)
  config/
    settings.py              # Pydantic settings (VLM, GLM-OCR, API)
    prompts.py               # VLM prompts (Japanese, category-specific)
    constants.py             # Image category definitions
  models/
    types.py                 # ImageInfo, PipelineResult, etc.
  pipeline/
    orchestrator.py          # Main coordinator (async/sync bridge)
  vlm/
    classify_and_caption.py  # Combined classify+caption (current)
    classifier.py            # Standalone classifier
    captioner.py             # Standalone captioner
  markdown/
    assembler.py             # Builds markdown from OCR JSON + captions
  utils/
    convert.py               # Visual format → PDF conversion (LibreOffice)
    extract.py               # Direct text extraction (TXT, CSV, XLSX, HTML)
    image_utils.py           # Crop, base64, PDF rendering
    text_utils.py            # Dedup, markdown cleanup
```

## Documentation

- [`documentation/inferencing_strategies.md`](documentation/inferencing_strategies.md) - Performance optimization strategies with benchmarks
- [`documentation/system_architecture.md`](documentation/system_architecture.md) - Detailed system design and data flow
- [`documentation/ADR.md`](documentation/ADR.md) - Architectural Decision Records
- [`documentation/CONTRIBUTING.md`](documentation/CONTRIBUTING.md) - Development setup and contribution guidelines

## License

MIT
