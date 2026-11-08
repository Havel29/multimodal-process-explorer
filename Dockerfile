FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

LABEL org.opencontainers.image.source="https://github.com/Havel29/multimodal-process-explorer" \
      org.opencontainers.image.description="Multimodal Process Explorer: multimodal predictive process monitoring, case by case" \
      org.opencontainers.image.licenses="CC-BY-NC-SA-4.0"

# Which explorer configuration to serve; see config/.
ARG EXPLORER_CONFIG=config/xray_pathways.toml

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    HF_LOCAL_FILES_ONLY=1 \
    EXPLORER_CONFIG=/app/${EXPLORER_CONFIG}

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config
COPY scripts ./scripts
COPY explainers ./explainers
COPY adapters ./adapters
RUN HF_LOCAL_FILES_ONLY=0 python scripts/download_model_assets.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
