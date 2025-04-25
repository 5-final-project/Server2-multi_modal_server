# Multimodal LLM Inference API 🚀

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Queue](https://img.shields.io/badge/Queue-Celery%20%26%20Redis-red.svg)](https://docs.celeryq.dev/)

<!-- Add License Badge if applicable: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

---

## 🌟 Introduction

This project provides a robust and scalable API server for performing inference with multimodal Large Language Models (LLMs), specifically tailored for the **Qwen 2.5 Omni** model (using GPTQ quantization). It leverages FastAPI for high performance, Celery for asynchronous task queueing, and supports text, image, audio, and video inputs.

The application is designed with a modular structure for maintainability and includes Docker support for easy deployment.

---

## ✨ Features

- **High-Performance API:** Built with FastAPI for asynchronous request handling.
- **Multimodal Input:** Accepts text prompts along with image, audio, and video files.
- **Qwen 2.5 Omni GPTQ:** Optimized for the quantized Qwen 2.5 Omni model.
- **Asynchronous Task Queueing:** Uses Celery and Redis to handle potentially long-running inference tasks without blocking the API, ensuring responsiveness.
- **Scalability:** Workers can be scaled independently to handle varying loads.
- **Configuration Management:** Centralized settings using Pydantic.
- **Docker Support:** Includes a `Dockerfile` for containerized deployment (Docker Compose recommended for managing services).
- **Modular Design:** Code is organized into logical components (API, Core Logic, Utilities).
- **Automatic Documentation:** Interactive API documentation via Swagger UI (`/docs`) and ReDoc (`/redoc`).

---

## 📂 Project Structure

```
.
├── app/                    # Main application source code
│   ├── api/                # API related modules
│   │   ├── endpoints/      # API route definitions (e.g., inference.py)
│   │   └── schemas/        # Pydantic models for request/response validation
│   ├── core/               # Core application logic
│   │   ├── config.py       # Configuration settings
│   │   ├── inference_worker.py # Celery task definition
│   │   ├── model_loader.py # Model and processor loading logic
│   │   ├── processing.py   # Core inference execution logic
│   │   └── queue_manager.py # Celery application setup
│   ├── utils/              # Utility functions (e.g., file handling)
│   │   └── helpers.py
│   ├── __init__.py
│   └── main.py             # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration for the application/worker
├── .gitignore              # Git ignore rules
├── qwen_omni_utils.py      # IMPORTANT: Required utility script (needs to be added)
└── README.md               # This file
```

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python:** Version 3.10 or higher.
- **pip:** Python package installer.
- **Docker & Docker Compose:** (Recommended for running services like Redis and deploying).
- **Redis:** A running Redis instance (can be run via Docker).
- **Qwen 2.5 Omni Model Files:** Download the specific GPTQ model version you intend to use.
- **`qwen_omni_utils.py`:** This utility script, likely from the Qwen-Omni repository or example source, is **required** for processing multimodal inputs. Place it in the project's root directory or adjust paths in the code/Dockerfile accordingly.

---

## ⚙️ Setup

1.  **Clone the Repository:**

    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and Activate Virtual Environment:**

    ```bash
    python -m venv venv
    # Linux/macOS
    source venv/bin/activate
    # Windows
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Place `qwen_omni_utils.py`:**

    - Download or copy the `qwen_omni_utils.py` script.
    - Place it in the root directory of this project.

5.  **Configure Environment Variables:**
    - The application uses environment variables for configuration (see `app/core/config.py`). Key variables include:
      - `MODEL_PATH`: **Required.** Path to your downloaded Qwen model directory.
      - `REDIS_URL`: URL for your Redis instance (defaults to `redis://localhost:6379/0`).
      - `UPLOAD_DIR`: Temporary directory for file uploads (defaults to `/tmp/multimodal_uploads`).
      - `PORT`: Port for the FastAPI server (defaults to `8872`).
    - You can set these variables directly in your shell or create a `.env` file (uncomment `env_file = ".env"` in `app/core/config.py` if you do).
    - **Example `.env` file:**
      ```dotenv
      MODEL_PATH=/path/to/your/Qwen/Qwen2.5-Omni-7B-GPTQ-4bit
      REDIS_URL=redis://localhost:6379/0
      # PORT=8872 # Optional, defaults work
      ```

---

## ▶️ Running Locally

1.  **Start Redis:**

    - If using Docker: `docker run -d -p 6379:6379 --name multimodal-redis redis`
    - Or ensure your local Redis server is running.

2.  **Start Celery Worker:**

    - Open a terminal, activate the virtual environment, and run:

    ```bash
    celery -A app.core.queue_manager.celery_app worker --loglevel=info -P solo
    ```

    - (`-P solo` is good for debugging; use `-P gevent` or `-P eventlet` with `-c <num_workers>` for concurrency).

3.  **Start FastAPI Server:**
    - Open another terminal, activate the virtual environment, and run:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8872 --reload
    ```
    - The API will be available at `http://localhost:8872`.
    - Swagger UI documentation: `http://localhost:8872/docs`.

---

## 🐳 Running with Docker

Using Docker Compose is the recommended way to manage the application, worker, and Redis services together.

1.  **Create `docker-compose.yml`:**
    Create a file named `docker-compose.yml` in the project root with the following content (adjust paths and resource limits as needed):

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
          - "8872:8872" # Map host port 8872 to container port 8872
        volumes:
          # IMPORTANT: Mount your model directory here
          - /path/to/your/models:/models:ro # Mount models read-only
          # Optional: Mount upload directory if you want persistence outside container
          # - ./uploads:/tmp/multimodal_uploads
        environment:
          # Ensure these match your setup if different from Dockerfile defaults
          - REDIS_URL=redis://redis:6379/0
          - MODEL_PATH=/models/Qwen/Qwen2.5-Omni-7B-GPTQ-4bit # Path inside container
          - PORT=8872
          # Add any other necessary environment variables
        depends_on:
          - redis
        restart: unless-stopped
        # Add resource limits if needed (e.g., for GPU access)
        # deploy:
        #   resources:
        #     reservations:
        #       devices:
        #         - driver: nvidia
        #           count: 1 # Request 1 GPU
        #           capabilities: [gpu]

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
        volumes:
          # IMPORTANT: Mount your model directory here too
          - /path/to/your/models:/models:ro # Mount models read-only
          # Optional: Mount upload directory
          # - ./uploads:/tmp/multimodal_uploads
        environment:
          # Ensure these match your setup
          - REDIS_URL=redis://redis:6379/0
          - MODEL_PATH=/models/Qwen/Qwen2.5-Omni-7B-GPTQ-4bit # Path inside container
          # Add any other necessary environment variables
        depends_on:
          - redis
          - api # Optional: wait for api to be healthy if needed
        restart: unless-stopped
        # Add resource limits if needed (e.g., for GPU access)
        # deploy:
        #   resources:
        #     reservations:
        #       devices:
        #         - driver: nvidia
        #           count: 1 # Request 1 GPU
        #           capabilities: [gpu]

    volumes:
      redis_data:
    ```

    **Important:**

    - Replace `/path/to/your/models` with the actual path to the directory containing your Qwen model files on your host machine.
    - Ensure `qwen_omni_utils.py` is in the project root before building.
    - Adjust GPU resource allocation (`deploy.resources`) if you are using GPUs and have the NVIDIA container toolkit installed.

2.  **Build and Run:**
    ```bash
    docker-compose up --build -d
    ```
    - `-d` runs the services in detached mode.
    - View logs: `docker-compose logs -f`
    - Stop services: `docker-compose down`

---

## 🚀 API Usage

Access the interactive documentation via Swagger UI at `http://localhost:8872/docs`.

**Endpoints:**

- `POST /api/v1/predict/async`: Submits an inference task.
  - **Input:** `multipart/form-data` containing:
    - `prompt` (string, required): The text prompt.
    - `system_prompt` (string, optional): System message for the model.
    - `images` (file, optional): One or more image files.
    - `audios` (file, optional): One or more audio files.
    - `videos` (file, optional): One or more video files.
  - **Output:** JSON with `task_id`.
- `GET /api/v1/predict/status/{task_id}`: Checks the status of a task.
  - **Input:** `task_id` from the async prediction response.
  - **Output:** JSON with `task_id`, `status` (`PENDING`, `STARTED`, `SUCCESS`, `FAILURE`), and `result` (containing `generated_text` on success or error details on failure).

**Example `curl` Requests:**

1.  **Submit Text-Only Task:**

    ```bash
    curl -X POST "http://localhost:8872/api/v1/predict/async" \
         -H "accept: application/json" \
         -F "prompt=Translate the following English text to French: 'Hello, world!'"
    ```

    _(Response will contain a `task_id`)_

2.  **Submit Task with Image:**

    ```bash
    curl -X POST "http://localhost:8872/api/v1/predict/async" \
         -H "accept: application/json" \
         -F "prompt=Describe this image." \
         -F "images=@/path/to/your/image.jpg"
    ```

3.  **Check Task Status:** (Replace `{task_id}` with the actual ID received)
    ```bash
    curl -X GET "http://localhost:8872/api/v1/predict/status/{task_id}" \
         -H "accept: application/json"
    ```

---

## 🔧 Configuration

Key configuration options are managed in `app/core/config.py` and can be overridden using environment variables:

- `MODEL_PATH`: Path to the LLM model directory.
- `DEVICE_MAP`: Device placement for the model (e.g., "cuda", "auto").
- `TORCH_DTYPE_STR`: Data type for model tensors (e.g., "float16", "bfloat16").
- `ATTN_IMPLEMENTATION`: Attention mechanism (e.g., "flash_attention_2").
- `REDIS_URL`: Connection URL for Redis (used by Celery).
- `UPLOAD_DIR`: Directory for temporary file uploads.
- `DEFAULT_SYSTEM_PROMPT`: Default system message if none is provided in the request.
- `PORT`: Port the FastAPI server listens on.

---

<!-- Optional Sections:
## 🤝 Contributing

Contributions are welcome! Please follow standard fork-and-pull-request workflows.

## 📜 License

This project is licensed under the [MIT License](LICENSE). -->
