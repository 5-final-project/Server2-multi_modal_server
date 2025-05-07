##############################################
# Dockerfile for running Qwen3-4B with vllm
# Container port: 8000
# Map to host port: 8776 (docker run -p 8776:8000)
##############################################

# Base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install vllm with all extras (includes torch, transformers, accelerate, bitsandbytes, etc.)
RUN pip install --no-cache-dir "vllm[all]"

# Create model directory
RUN mkdir -p /models/qwen3-4b

# Download the model shards
RUN cd /models/qwen3-4b \
    && wget --progress=dot:giga -O model-00001-of-00003.safetensors "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/model-00001-of-00003.safetensors?download=true" \
    && wget --progress=dot:giga -O model-00002-of-00003.safetensors "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/model-00002-of-00003.safetensors?download=true" \
    && wget --progress=dot:giga -O model-00003-of-00003.safetensors "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/model-00003-of-00003.safetensors?download=true"

# Expose the default vllm serving port
EXPOSE 8000

# Default command to start the vllm server
CMD ["vllm", "serve", \
     "--model-path", "/models/qwen3-4b", \
     "--host", "0.0.0.0", \
     "--port", "8000"]

# Usage:
#   docker build -t vllm-qwen3-4b .
#   docker run -d -p 8000:8000 --gpus all --name vllm-qwen3-4b vllm-qwen3-4b