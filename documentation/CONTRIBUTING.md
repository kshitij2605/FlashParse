# Contributing

## Development Setup

### Prerequisites

- Python 3.10+
- Access to a GPU with ~48 GB VRAM (for GLM-OCR vLLM + layout detector)
- Access to any OpenAI-compatible VLM server for image captioning
- LibreOffice (optional, only needed for non-PDF visual formats like DOCX, PPTX)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd glm_hybrid_ocr

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Optional: install spreadsheet support (XLSX)
uv pip install -e ".[spreadsheet]"

# Optional: install LibreOffice for DOCX/PPTX conversion
sudo apt install libreoffice-core
```

### Starting Services

```bash
# 1. Start vLLM server for GLM-OCR
./scripts/start_vllm.sh

# 2. (Optional) Configure VLM endpoint if not using default
export VLM_ENDPOINT="http://your-vlm-server:8008/v1/chat/completions"

# 3. Start the API server
python -m glm_hybrid_ocr.api.main
```

### Running Tests

```bash
# File format routing and extraction tests (no GPU needed)
pytest tests/test_file_formats.py -v

# All unit tests
pytest

# Manual pipeline test (requires running services)
python tests/test_pipeline.py /path/to/test.pdf ./test_output

# Concurrency verification
python tests/test_concurrency.py

# Performance timing
python tests/test_timing.py
```

## Project Structure

```
src/glm_hybrid_ocr/
  api/              # FastAPI app and routes
  clients/          # HTTP clients (VLM)
  config/           # Settings, prompts, constants
  models/           # Dataclasses and type definitions
  pipeline/         # Main orchestrator (routes visual vs text-based formats)
  vlm/              # VLM classification and captioning
  markdown/         # Markdown assembly
  utils/            # Image/text utilities, format conversion, direct extraction
    convert.py      #   Visual format → PDF (LibreOffice headless)
    extract.py      #   Direct text extraction (TXT, CSV, XLSX, HTML → markdown)
tests/
  test_file_formats.py  # 80 tests for format routing and extraction
```

## Code Style

- **Formatter/Linter**: [Ruff](https://docs.astral.sh/ruff/)
- **Type hints**: Use throughout, especially for public APIs
- **Async**: All VLM interactions are async. Use `asyncio.create_task` for concurrent work, `run_in_executor` for blocking calls.

```bash
# Format
ruff format src/

# Lint
ruff check src/
```

## Key Patterns

### Async/Sync Bridge

GLM-OCR is threaded (sync). VLM captioning is async. When bridging:

```python
# Run sync code in executor
result = await loop.run_in_executor(None, sync_function, args)

# Bridge thread callback to async
def callback_from_thread(data):
    loop.call_soon_threadsafe(async_queue.put_nowait, data)
```

### VLM Response Parsing

VLM responses may or may not contain structured tags. Always implement fallback parsing:

```python
# Primary: parse structured tags
match = re.search(r'\[category\](.*?)\[/category\]', response)

# Fallback: keyword matching
if not match:
    category = infer_category_from_text(response)
```

### Configuration

All settings use `pydantic-settings` with environment variable overrides:

```python
class VLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLM_")
    endpoint: str = "http://default:8008/v1/chat/completions"
```

Override in environment: `VLM_ENDPOINT=http://other-server:8008/v1/chat/completions`

## Adding a New Optimization Strategy

1. Implement the change
2. Test on the 12-page reference PDF (`examples/output/2022統合報告書-1-12/`)
3. Record before/after timings from `parsing_metrics.json`
4. Document in `documentation/inferencing_strategies.md` with:
   - Date and commit hash
   - Problem statement
   - Solution description
   - Files changed
   - Results table with timings
5. If the change involves an architectural decision, add an entry to `documentation/ADR.md`

## Modifying the GLM-OCR Package

The upstream `glmocr` package has been modified with custom commits. Changes are documented in `documentation/inferencing_strategies.md` under "Changes to GLM-OCR Repository".

When modifying `glmocr`:
- Maintain backward compatibility (new parameters should be optional with `None` defaults)
- Protect shared state with locks when adding thread callbacks
- Always send sentinel signals in both success and error paths
- Reinstall after changes: `pip install -e /path/to/GLM-OCR`

## Output Format

Output must remain compatible with the `deepseek-ocr-api` format. When changing output files:
- `metadata.json` — keep all existing fields, add new ones only
- `parsing_metrics.json` — keep the `timing`, `statistics`, `images[]`, `tables[]` structure
- `.mmd` files — maintain the markdown format with image references and category-specific caption formatting
