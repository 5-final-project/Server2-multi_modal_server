import logging
import sys
from fastapi import FastAPI, status
from contextlib import asynccontextmanager

from app.api.endpoints import inference as inference_router
from app.core.config import settings
from app.core.model_loader import load_model_and_processor # To trigger loading if needed

# --- Logging Configuration ---
# Configure logging basic settings
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)
# You might want to use more advanced logging configuration (e.g., Loguru, structlog)
# or configure file logging based on settings.
logger = logging.getLogger(__name__)


# --- Lifespan Management (Optional but Recommended) ---
# Use lifespan events for setup/teardown, like loading the model once
# Note: The current setup loads the model lazily on first request or in the worker.
# Loading here ensures it's ready before the first request hits the *main* process,
# but workers still need to load their own copy unless using a shared model service.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Application startup...")
    logger.info(f"Using model path: {settings.MODEL_PATH}")
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"Redis URL for Celery: {settings.REDIS_URL}")
    try:
        # Pre-load model in the main process if desired (optional)
        # load_model_and_processor()
        # logger.info("Model and processor pre-loaded in main process.")
        pass # Model loading is handled lazily or by workers currently
    except Exception as e:
        logger.exception(f"Failed to load model during startup: {e}")
        # Depending on policy, you might want to prevent startup
        # raise RuntimeError("Model loading failed, cannot start application.") from e
    yield
    # --- Shutdown ---
    logger.info("Application shutdown...")
    # Add any cleanup logic here if needed


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Multimodal LLM API",
    description="API server for Qwen 2.5 Omni model inference with task queueing.",
    version="0.1.0",
    lifespan=lifespan # Use the lifespan context manager
)

# --- Include Routers ---
# Include the router for inference endpoints
app.include_router(inference_router.router, prefix="/api/v1", tags=["Inference"])


# --- Root Endpoint ---
@app.get("/", status_code=status.HTTP_200_OK, tags=["Health Check"])
async def read_root():
    """
    Root endpoint for basic health check.
    """
    return {"status": "ok", "message": "Multimodal LLM API is running."}

# --- Uvicorn Entry Point (for running directly) ---
# This allows running `python app/main.py` but `uvicorn app.main:app --reload` is preferred
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly...")
    # Note: Reloading might not work well with model loading in this direct run mode.
    # Port is now read from settings
    uvicorn.run(app, host="0.0.0.0", port=settings.SERVER_PORT)
