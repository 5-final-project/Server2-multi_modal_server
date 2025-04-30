# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Step 1: Clean apt state, update, install https transport and certificates
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        apt-transport-https \
        curl \
        software-properties-common && \
    sed -i 's|http://|https://|g' /etc/apt/sources.list && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Step 2: Add deadsnakes PPA and install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        tzdata \
        python3.10 \
        python3.10-distutils \
        python3.10-dev \
        git \
        libgl1-mesa-glx && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Step 3: Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy application code
COPY ./llama_api ./llama_api

# Step 5: Expose port and run the app
EXPOSE 8776
CMD ["uvicorn", "llama_api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# docker run -it -d --gpus all \
#   -v /mnt/d/team5/multi_modal_server:/app \
#   -v /mnt/d/team5/multi_modal_server/hf_cache:/app/hf_cache \
#   llama4
