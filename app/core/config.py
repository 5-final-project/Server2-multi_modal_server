import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/home/chentianqi/model/Qwen/Qwen2.5-Omni-7B-GPTQ-4bit") # Default path from example
    DEVICE_MAP: str = os.getenv("DEVICE_MAP", "cuda") # Or "auto" or specific device like "cuda:0"
    TORCH_DTYPE_STR: str = os.getenv("TORCH_DTYPE", "float16") # e.g., "float16", "bfloat16"
    ATTN_IMPLEMENTATION: str = os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2")

    # Queue settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = REDIS_URL
    CELERY_RESULT_BACKEND: str = REDIS_URL

    # Temporary storage for uploaded files
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/multimodal_uploads")

    # System prompt
    DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."

    class Config:
        # If you have a .env file, uncomment the line below
        # env_file = ".env"
        case_sensitive = True

settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
