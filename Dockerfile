# AgenticRAG v0.4 — Docker image
# Build: docker build -t agenticrag:v0.4 .
# Run:   docker run -p 7860:7860 --env-file .env agenticrag:v0.4

FROM python:3.10-slim

LABEL org.opencontainers.image.title="AgenticRAG"
LABEL org.opencontainers.image.description="Enterprise knowledge base Q&A with Agentic RAG"
LABEL org.opencontainers.image.version="0.4.0"

# System dependencies for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-ita \
    tesseract-ocr-spa \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY main.py app_gradio.py ./
COPY .env.template ./

# Expose Gradio default port
EXPOSE 7860

# Environment defaults (override with --env-file or -e)
ENV RAG_LLM_MODEL=deepseek-v4-pro
ENV RAG_LOG_LEVEL=INFO

# Start the Gradio Web UI
CMD ["python", "app_gradio.py"]
