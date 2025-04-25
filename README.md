# Multimodal LLM Inference API 🚀

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Queue](https://img.shields.io/badge/Queue-Celery%20%26%20Redis-red.svg)](https://docs.celeryq.dev/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Qwen%2FQwen2.5--Omni--7B-yellow)](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

<!-- Add License Badge if applicable: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

---

## 🌟 Introduction

This project provides a robust and scalable API server for performing inference with the **Qwen 2.5 Omni** multimodal Large Language Model (LLM) directly from the Hugging Face Hub (`Qwen/Qwen2.5-Omni-7B`). It leverages FastAPI for high performance, Celery for asynchronous task queueing, and supports text, image, audio, and video inputs, along with optional audio output generation.

The application is designed with a modular structure, includes Docker support, and offers configuration options for performance tuning (like Flash Attention 2) and resource management.

---

## ✨ Features

- **High-Performance API:** Built with FastAPI for asynchronous request handling.
- **Multimodal Input:** Accepts text prompts along with image, audio, and video files.
- **Qwen 2.5 Omni Model:** Uses the `Qwen/Qwen2.5-Omni-7B` model from Hugging Face Hub.
- **Optional Audio Output:** Can generate speech output (`.wav`) corresponding to the text response, with selectable voices (Chelsie/Ethan).
- **Asynchronous Task Queueing:** Uses Celery and Redis to handle potentially long-running inference tasks without blocking the API.
- **Scalability:** Workers can be scaled independently.
- **Configuration Management:** Centralized settings using Pydantic (`app/core/config.py`).
- **Performance Options:** Supports Flash Attention 2 (if available and enabled) for potential speedups. Option to disable audio generation (`DISABLE_TALKER`) to save memory.
- **Docker Support:** Includes an updated `Dockerfile` and `docker-compose.yml` example for containerized deployment.
- **Modular Design:** Organized into API, Core Logic, and Utilities.
- **Automatic Documentation:** Interactive API documentation via Swagger UI (`/docs`) and ReDoc (`/redoc`).

---

## 📂 Project Structure

```
.
├── app/                    # Main application source code
│   ├── api/                # API related modules
│   │   ├── endpoints/      # API route definitions (inference.py)
│   │   └── schemas/        # Pydantic models (inference.py)
│   ├── core/               # Core application logic
│   │   ├── config.py       # Configuration settings
│   │   ├── inference_worker.py # Celery task definition
│   │   ├── model_loader.py # Model/processor loading
│   │   ├── processing.py   # Core inference execution
│   │   └── queue_manager.py # Celery application setup
│   ├── utils/              # Utility functions
│   │   └── helpers.py
│   ├── __init__.py
│   └── main.py             # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

_(Note: `qwen_omni_utils.py` is no longer explicitly required in the root, assuming `transformers` handles necessary utilities when `trust_remote_code=True`)_

---

## 📋 Prerequisites

- **Python:** 3.10+
- **pip:** Python package installer
- **Docker & Docker Compose:** (Recommended)
- **Redis:** Running instance (can be run via Docker)
- **Hardware:** Sufficient CPU, RAM, and **GPU memory** (especially for the 7B model). Check Qwen 2.5 Omni documentation for specific requirements. Flash Attention 2 requires compatible hardware (NVIDIA Ampere or newer recommended) and `torch.float16` or `torch.bfloat16` dtype.

---

## ⚙️ Setup

1.  **Clone Repository:**

    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create & Activate Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # .\venv\Scripts\activate # Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    # Ensure flash-attn is installed correctly if using:
    # pip install -U flash-attn --no-build-isolation
    ```

4.  **Configure Environment Variables (Optional):**
    - Create a `.env` file in the project root or set environment variables directly.
    - Key options (see `app/core/config.py` for defaults):
      - `MODEL_ID`: Hugging Face model ID (default: `Qwen/Qwen2.5-Omni-7B`).
      - `DEVICE_MAP`: (default: `auto`).
      - `TORCH_DTYPE`: (default: `auto`). Use `bfloat16` or `float16` for Flash Attention 2.
      - `USE_FLASH_ATTENTION_2`: (default: `True`). Set to `False` to disable.
      - `DISABLE_TALKER`: (default: `False`). Set to `True` to disable audio output generation and save ~2GB GPU memory.
      - `REDIS_URL`: (default: `redis://localhost:6379/0`).
      - `UPLOAD_DIR`: (default: `/tmp/multimodal_uploads`).
      - `AUDIO_OUTPUT_DIR`: (default: `/tmp/multimodal_audio_outputs`).
      - `SERVER_PORT`: (default: `8872`).
    - **Example `.env` file:**
      ```dotenv
      # MODEL_ID=Qwen/Qwen2.5-Omni-7B # Optional: Override default
      TORCH_DTYPE=bfloat16 # Example: Use bfloat16 for Flash Attention
      USE_FLASH_ATTENTION_2=True
      # DISABLE_TALKER=True # Example: Disable audio output
      REDIS_URL=redis://localhost:6379/0
      SERVER_PORT=8872
      ```

---

## ▶️ Running Locally

1.  **Start Redis:**

    - Docker: `docker run -d -p 6379:6379 --name multimodal-redis redis`
    - Or ensure local Redis is running.

2.  **Start Celery Worker:**

    - (Activate venv)

    ```bash
    celery -A app.core.queue_manager.celery_app worker --loglevel=info -P solo
    ```

    - (Use `-P gevent/eventlet -c <num>` for concurrency)

3.  **Start FastAPI Server:**
    - (Activate venv)
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port ${SERVER_PORT:-8872} --reload
    ```
    - Access API at `http://localhost:8872` (or configured port).
    - Docs: `http://localhost:8872/docs`.

---

## 🐳 Running with Docker (Recommended)

1.  **Create/Update `docker-compose.yml`:**
    Use the example below in the project root. **Crucially, update model mounting if not using Hugging Face Hub caching.**

    ```yaml
    version: "3.8"

    services:
      redis:
        image: redis:alpine
        container_name: multimodal-redis
        ports:
          - "6379:6379"
        volumes:
          - redis_data:/data
        restart: always

      api:
        build: .
        container_name: multimodal-api
        ports:
          - "${SERVER_PORT:-8872}:${SERVER_PORT:-8872}" # Use env var for host port mapping
        # volumes: # Optional: Mount local cache or specific model dir if needed
        # - ~/.cache/huggingface:/root/.cache/huggingface # Mount HF cache
        # - /path/to/local/models:/models # Example if model isn't downloaded automatically
        environment:
          # Pass necessary runtime environment variables
          - REDIS_URL=redis://redis:6379/0
          - SERVER_PORT=${SERVER_PORT:-8872}
          - MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-Omni-7B}
          - TORCH_DTYPE=${TORCH_DTYPE:-auto}
          - USE_FLASH_ATTENTION_2=${USE_FLASH_ATTENTION_2:-True}
          - DISABLE_TALKER=${DISABLE_TALKER:-False}
          - DEVICE_MAP=${DEVICE_MAP:-auto}
          # HF_HOME: /root/.cache/huggingface # Set HF cache location inside container if needed
          # TRANSFORMERS_CACHE: /root/.cache/huggingface/hub # More specific cache
        depends_on:
          - redis
        restart: unless-stopped
        # --- GPU Allocation (Requires NVIDIA Container Toolkit) ---
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: all # Or specify count: 1
                  capabilities: [gpu]

      worker:
        build: .
        container_name: multimodal-worker
        command: [
            "celery",
            "-A",
            "app.core.queue_manager.celery_app",
            "worker",
            "--loglevel=info",
            "-P",
            "solo",
          ] # Or gevent/eventlet
        # volumes: # Optional: Mount cache/models same as API service
        # - ~/.cache/huggingface:/root/.cache/huggingface
        # - /path/to/local/models:/models
        environment:
          # Pass necessary runtime environment variables
          - REDIS_URL=redis://redis:6379/0
          - MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-Omni-7B}
          - TORCH_DTYPE=${TORCH_DTYPE:-auto}
          - USE_FLASH_ATTENTION_2=${USE_FLASH_ATTENTION_2:-True}
          - DISABLE_TALKER=${DISABLE_TALKER:-False}
          - DEVICE_MAP=${DEVICE_MAP:-auto}
          # HF_HOME: /root/.cache/huggingface
          # TRANSFORMERS_CACHE: /root/.cache/huggingface/hub
        depends_on:
          - redis
          # - api # Optional dependency
        restart: unless-stopped
        # --- GPU Allocation ---
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: all # Or specify count: 1
                  capabilities: [gpu]

    volumes:
      redis_data:
    ```

    **Notes:**

    - This compose file assumes the model will be downloaded automatically by `transformers` into the container's cache. You might want to mount your host's Hugging Face cache (`~/.cache/huggingface`) to avoid re-downloading.
    - GPU allocation (`deploy.resources`) requires the NVIDIA Container Toolkit to be installed on the host. Adjust `count` as needed.

2.  **Build and Run:**
    ```bash
    # Optional: Export env vars if not using .env file for compose
    # export SERVER_PORT=8872
    # export TORCH_DTYPE=bfloat16
    docker-compose up --build -d
    ```
    - View logs: `docker-compose logs -f`
    - Stop: `docker-compose down` (use `-v` to remove Redis volume)

---

## 🚀 API Usage

Access interactive documentation (Swagger UI) at `http://localhost:8872/docs` (or your configured port).

**Endpoints:**

- `POST /api/v1/predict/async`: Submits an inference task.
  - **Input:** `multipart/form-data`
    - `prompt` (string, required)
    - `images` (file, optional)
    - `audios` (file, optional)
    - `videos` (file, optional)
    - `system_prompt` (string, optional): Overrides default. Use Qwen audio prompt if `return_audio=True`.
    - `return_audio` (boolean, optional, default: `False`): Request audio output.
    - `use_audio_in_video` (boolean, optional, default: `True`): Process audio in videos.
    - `speaker` (string, optional, default: `Chelsie` if audio requested): `Chelsie` or `Ethan`.
    - `max_new_tokens` (integer, optional, default: `512`).
  - **Output:** JSON with `task_id`.
- `GET /api/v1/predict/status/{task_id}`: Checks task status.
  - **Output:** JSON with `task_id`, `status`, and `result` (containing `generated_text` and `generated_audio_path` on success, or `error` details).
- `GET /api/v1/audio/{filename}`: Serves the generated audio file.
  - Use the `filename` returned in `generated_audio_path` from a successful status check.

**Example `curl` Requests:**

1.  **Submit Task Requesting Audio:**

    ```bash
    curl -X POST "http://localhost:8872/api/v1/predict/async" \
         -H "accept: application/json" \
         -F "prompt=Tell me a short story about a robot learning to sing." \
         -F "return_audio=true" \
         -F "speaker=Ethan" \
         # Ensure system prompt for audio is used if needed by model version
         -F "system_prompt=You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
    ```

2.  **Check Status:** (Replace `{task_id}`)

    ```bash
    curl -X GET "http://localhost:8872/api/v1/predict/status/{task_id}"
    ```

    _(If successful and audio was requested, the response `result` will contain `generated_audio_path`: "some-uuid.wav")_

3.  **Download Audio:** (Replace `{filename}` with the actual path from status)
    ```bash
    curl -X GET "http://localhost:8872/api/v1/audio/{filename}" -o output_story.wav
    ```

---

## 🔧 Configuration Summary

Key settings (via `.env` or environment variables):

- `MODEL_ID`: Which Hugging Face model to use.
- `DEVICE_MAP`: How to distribute model across devices.
- `TORCH_DTYPE`: Precision (`auto`, `bfloat16`, `float16`).
- `USE_FLASH_ATTENTION_2`: Enable/disable Flash Attention 2.
- `DISABLE_TALKER`: Disable audio generation capability.
- `REDIS_URL`: Redis connection string.
- `SERVER_PORT`: API server port.
- `UPLOAD_DIR`, `AUDIO_OUTPUT_DIR`: Temporary file locations.

---
