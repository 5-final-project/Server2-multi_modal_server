# Use an official NVIDIA CUDA runtime as a parent image
# This includes CUDA Toolkit and cuDNN, essential for GPU acceleration with PyTorch
# Using CUDA 12.1.1 and cuDNN 8 on Ubuntu 22.04. Adjust versions if needed based on your specific hardware/torch version.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
# Prevents Python from writing pyc files to disc (equivalent to python -B)
ENV PYTHONDONTWRITEBYTECODE 1
# Ensures Python output is sent straight to terminal without being buffered
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install Python 3.10 and pip, plus essential build tools
# Using deadsnakes PPA for modern Python versions on Ubuntu
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    python3.10-dev \
    git \
    && \
    # Link python3 to python3.10 and pip3 to pip
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    # Ensure pip is up to date for python3
    python3 -m pip install --upgrade pip && \
    # Clean up apt lists to reduce image size
    rm -rf /var/lib/apt/lists/*

# Install system dependencies that might be needed by torch or other libraries
# Add any specific system dependencies required by your model or libraries here if necessary.
# Example: RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*


# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install Python packages specified in requirements.txt
# Using --no-cache-dir reduces image size
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code into the container at /app
COPY ./llama_api ./llama_api

# Expose port 8776 to the outside world
EXPOSE 8776

# Define the command to run the application
# Use uvicorn to run the FastAPI app found in llama_api/main.py
# Bind to 0.0.0.0 to allow external connections
# Use the specified port 8776
# --workers 1 is a common starting point, adjust based on performance needs
CMD ["uvicorn", "llama_api.main:app", "--host", "0.0.0.0", "--port", "8776"]
