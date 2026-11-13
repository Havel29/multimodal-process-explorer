# Two variants of the same image:
#   CPU (default, ~3 GB, x86-64 and ARM64): docker build .
#   GPU (NVIDIA CUDA, ~10 GB, x86-64):       docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime --build-arg TORCH_INDEX= .
# The CUDA base already ships PyTorch; on the slim base it comes from the CPU wheel index.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/Havel29/multimodal-process-explorer" \
      org.opencontainers.image.description="Multimodal Process Explorer: multimodal predictive process monitoring, case by case" \
      org.opencontainers.image.licenses="CC-BY-NC-SA-4.0"

# Which explorer configuration to serve; see config/.
ARG EXPLORER_CONFIG=config/xray_pathways.toml
# PyTorch wheel index; empty when the base image already has PyTorch
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    HF_LOCAL_FILES_ONLY=1 \
    EXPLORER_CONFIG=/app/${EXPLORER_CONFIG}

WORKDIR /app
COPY requirements.txt .
RUN if [ -n "$TORCH_INDEX" ]; then \
        pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url "$TORCH_INDEX"; \
    fi \
    && pip install --no-cache-dir -r requirements.txt

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
