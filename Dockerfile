# Stage 1: Build environment with dependencies
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies if needed (e.g., for packages with C extensions)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Install poetry or just use pip with requirements.txt
# Using pip for simplicity based on requirements.txt
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir reduces image size
# Consider using a virtual environment within the image if preferred
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime environment
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1 \
    # Set default port for FastAPI/Uvicorn inside the container
    PORT=8872 \
    # Set default model path (can be mounted as volume or baked in if small)
    # IMPORTANT: Large models should NOT be baked into the image. Mount them instead.
    MODEL_PATH=/models/Qwen/Qwen2.5-Omni-7B-GPTQ-4bit \
    # Default Redis URL (assumes Redis runs on 'redis' hostname in Docker network)
    REDIS_URL=redis://redis:6379/0 \
    # Default upload directory inside the container
    UPLOAD_DIR=/tmp/multimodal_uploads

# Create non-root user for security
RUN addgroup --system app && adduser --system --group app
RUN chown -R app:app /app
# Create upload directory and set permissions
RUN mkdir -p $UPLOAD_DIR && chown -R app:app $UPLOAD_DIR

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=app:app . /app

# Copy the qwen_omni_utils.py script (assuming it's in the project root)
# Adjust the source path if it's located elsewhere (e.g., app/utils/)
COPY --chown=app:app qwen_omni_utils.py /app/
# If it's in app/utils: COPY --chown=app:app app/utils/qwen_omni_utils.py /app/app/utils/

# Switch to non-root user
USER app

# Expose the application port
EXPOSE ${PORT}

# Default command to run the FastAPI server using Uvicorn
# The worker command needs to be run separately (e.g., via docker-compose or another container)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8872"]

# --- Notes ---
# 1. Model Files: This Dockerfile assumes the model files are mounted into the container
#    at /models. Baking large models into the image is inefficient.
# 2. qwen_omni_utils.py: Ensure this script is present in the build context (project root)
#    or adjust the COPY command if it's located elsewhere.
# 3. Redis: This setup requires a separate Redis container, typically managed with Docker Compose.
# 4. Celery Worker: Run the worker in a separate container using the same image but
#    overriding the CMD, e.g., CMD ["celery", "-A", "app.core.queue_manager.celery_app", "worker", "--loglevel=info"]
