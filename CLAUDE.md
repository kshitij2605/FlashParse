# CLAUDE.md

## Project Overview

FlashParse (`glm_hybrid_ocr`) — a document parsing system combining GLM-OCR (via vLLM) for text/table/formula OCR with any OpenAI-compatible VLM for image classification and captioning.

## Tech Stack

- Python 3.10+, FastAPI, uvicorn, httpx (async)
- GLM-OCR 0.9B via vLLM (MTP-1 speculative decoding) for OCR
- Any OpenAI-compatible VLM for image captioning (currently Qwen3.5-122B-A10B)
- PP-DocLayoutV3 for layout detection
- Package manager: `uv` (venv at `.venv/`)
- Build system: hatchling

## Key Architecture

- **Two processing paths**: visual documents (PDF, DOCX, PPTX, images) go through OCR pipeline; text-based (TXT, CSV, XLSX, HTML) use direct extraction
- **Orchestrator** (`src/glm_hybrid_ocr/pipeline/orchestrator.py`): bridges sync threaded GLM-OCR to async VLM captioning with per-page callbacks
- **Classify-first pipeline**: fast classify_only() call, skip captioning for miscellaneous images
- **Output**: ZIP with .mmd, metadata.json, parsing_metrics.json, images/, tables/

## Development Commands

```bash
# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Start GLM-OCR vLLM server
./scripts/start_vllm.sh

# Start API server
uvicorn src.glm_hybrid_ocr.api.main:app --host 0.0.0.0 --port 8000

# Run tests
pytest tests/

# Run benchmark
python tests/benchmark_ocr.py
```

## Configuration

- `.env` — VLM endpoint, API key, model name, server config
- `glmocr_config.yaml` — GLM-OCR pipeline config (layout detection, vLLM connection)
- Settings loaded via pydantic-settings (`src/glm_hybrid_ocr/config/settings.py`)

## Code Conventions

- Async throughout the VLM/API layer; GLM-OCR pipeline is sync/threaded
- Image filenames: `{page_idx}_{region_idx}.jpg` (no 'page' prefix)
- Commit messages: conventional commits style (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`)
- Never add `Co-Authored-By: Claude` or any Claude Code references to commits
- `.env` is gitignored — never commit secrets

## Important Paths

- `src/glm_hybrid_ocr/pipeline/orchestrator.py` — main processing coordinator
- `src/glm_hybrid_ocr/vlm/classify_and_caption.py` — VLM classify + caption logic
- `src/glm_hybrid_ocr/markdown/assembler.py` — markdown assembly from OCR JSON + captions
- `src/glm_hybrid_ocr/api/routes/pdf.py` — PDF processing endpoints
- `src/glm_hybrid_ocr/config/prompts.py` — VLM prompts (Japanese, category-specific)

## Servers

- GLM-OCR vLLM: port 8080
- API server: port 8000
- VLM captioning: external endpoint configured in `.env`
