##############################################
# Dockerfile for FastAPI proxy to vLLM server
# Listens on port 8776 and forwards to vLLM (localhost:8000)
##############################################

FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn httpx

# Create app directory
WORKDIR /app

# Copy app code
COPY app /app

# Expose FastAPI port
EXPOSE 8776

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8776"]
