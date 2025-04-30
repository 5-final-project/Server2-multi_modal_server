# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Step 1: Clean apt state, update, install https transport and certificates
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates apt-transport-https && \
    # Change sources to HTTPS
    sed -i 's/http:/https:/g' /etc/apt/sources.list && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Step 2: Update again (now over HTTPS) and install software-properties-common
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Step 3: Add PPA, update, install Python & build tools
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    tzdata \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    python3.10-dev \
    git \
    && \
    # Link python3 to python3.10
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    # Ensure pip is up to date
    python3 -m pip install --upgrade pip && \
    apt-get install -y libgl1-mesa-glx \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./llama_api ./llama_api

# Expose port
EXPOSE 8776

# Define the command to run the application
CMD ["uvicorn", "llama_api.main:app", "--host", "0.0.0.0", "--port", "8776"]

# docker run -it -d --gpus all \
#   -v /mnt/d/team5/multi_modal_server:/app \
#   -v /mnt/d/team5/multi_modal_server/hf_cache:/app/hf_cache \
#   llama4
