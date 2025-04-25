# Stage 1: Build environment with dependencies
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies required by some Python packages (e.g., soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    # Add build-essential if any packages require C compilation and it's not included
    # build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir packaging

RUN pip install --no-cache-dir packaging setuptools wheel

# Install Python dependencies
# Using --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime environment
FROM python:3.10-slim

WORKDIR /app

# Install required system libraries copied from builder stage implicitly by python base,
# but explicitly install runtime deps like libsndfile1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*


# Set environment variables (can be overridden at runtime)
# Defaults match the updated config.py
ENV PYTHONUNBUFFERED=1 \
    PORT=8872 \
    # IMPORTANT: Large models should NOT be baked into the image. Mount them instead.
    # Default MODEL_ID from config.py
    MODEL_ID=Qwen/Qwen2.5-Omni-7B \
    # Default Redis URL (assumes Redis runs on 'redis' hostname in Docker network)
    REDIS_URL=redis://redis:6379/0 \
    # Default upload/output directories inside the container
    UPLOAD_DIR=/tmp/multimodal_uploads \
    AUDIO_OUTPUT_DIR=/tmp/multimodal_audio_outputs \
    # Set other ENV VARS from config.py as needed (e.g., TORCH_DTYPE, USE_FLASH_ATTENTION_2)
    # These often make more sense to set at runtime (e.g., via docker-compose)
    TORCH_DTYPE=auto \
    USE_FLASH_ATTENTION_2=True \
    DISABLE_TALKER=False

# Create non-root user for security
RUN addgroup --system app && adduser --system --group app
# Create directories and set permissions BEFORE copying code
RUN mkdir -p $UPLOAD_DIR && chown -R app:app $UPLOAD_DIR && \
    mkdir -p $AUDIO_OUTPUT_DIR && chown -R app:app $AUDIO_OUTPUT_DIR

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code AFTER setting up user/dirs
COPY --chown=app:app . /app

# No longer copying qwen_omni_utils.py explicitly

# Switch to non-root user
USER app

# Expose the application port defined by ENV
EXPOSE ${PORT}

# Default command to run the FastAPI server using Uvicorn with the configured port
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "${PORT}"]

# --- Notes ---
# 1. Model Files: Mount model files into the container (e.g., at /models or $MODEL_PATH if set differently).
# 2. qwen_omni_utils.py: Assumed handled by transformers with trust_remote_code=True.
# 3. Redis: Requires a separate Redis container (use docker-compose).
# 4. Celery Worker: Run worker in a separate container using this image, overriding CMD.
#    CMD ["celery", "-A", "app.core.queue_manager.celery_app", "worker", "--loglevel=info"]
