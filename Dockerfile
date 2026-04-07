FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    libreoffice-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ src/
COPY glmocr_config.yaml ./

# Install the git dependency first (bypasses hatchling's direct-reference restriction)
RUN pip install --no-cache-dir "glmocr[selfhosted] @ git+https://github.com/kshitij2605/GLM-OCR-FlashParse.git"

# Install the package (non-editable)
RUN pip install --no-cache-dir --no-deps .

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "glm_hybrid_ocr.api.main"]
