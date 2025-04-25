import os
import torch
from pydantic_settings import BaseSettings
from typing import Literal

# Determine default dtype based on availability
default_dtype = "auto"
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        # default_dtype = "bfloat16" # Prefer bfloat16 if available for better stability/performance
        pass # Keep auto for now, let transformers decide based on hardware
    else:
        # default_dtype = "float16"
        pass

class Settings(BaseSettings):
    # Model settings
    MODEL_ID: str = os.getenv("MODEL_ID", "Qwen/Qwen2.5-Omni-7B")
    DEVICE_MAP: str = os.getenv("DEVICE_MAP", "auto") # "auto" lets transformers handle placement
    # TORCH_DTYPE: Literal["auto", "bfloat16", "float16", "float32"] = os.getenv("TORCH_DTYPE", default_dtype)
    TORCH_DTYPE: str = os.getenv("TORCH_DTYPE", "auto") # Use "auto" as default
    USE_FLASH_ATTENTION_2: bool = os.getenv("USE_FLASH_ATTENTION_2", True) # Default to True if flash-attn installed and supported
    DISABLE_TALKER: bool = os.getenv("DISABLE_TALKER", False) # Set to True to disable audio output and save memory

    # Queue settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = REDIS_URL
    CELERY_RESULT_BACKEND: str = REDIS_URL

    # Temporary storage for uploads and outputs
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/multimodal_uploads")
    AUDIO_OUTPUT_DIR: str = os.getenv("AUDIO_OUTPUT_DIR", "/tmp/multimodal_audio_outputs") # Directory for generated audio

    # Default inference parameters
    DEFAULT_SYSTEM_PROMPT_TEXT: str = "You are a helpful assistant."
    # Specific system prompt required for audio generation
    AUDIO_SYSTEM_PROMPT_TEXT: str = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
    DEFAULT_SPEAKER: Literal["Chelsie", "Ethan"] = "Chelsie"
    AUDIO_SAMPLE_RATE: int = 24000 # Qwen Omni default sample rate

    # Server settings
    SERVER_PORT: int = os.getenv("SERVER_PORT", 8872)

    class Config:
        # If you have a .env file, uncomment the line below
        # env_file = ".env"
        case_sensitive = True

settings = Settings()

# Ensure temporary directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.AUDIO_OUTPUT_DIR, exist_ok=True)

# Determine attn_implementation based on setting and availability
attn_implementation = "flash_attention_2" if settings.USE_FLASH_ATTENTION_2 else "sdpa" # or None/eager

# Log effective settings
import logging
logger = logging.getLogger(__name__)
logger.info(f"Model ID: {settings.MODEL_ID}")
logger.info(f"Device Map: {settings.DEVICE_MAP}")
logger.info(f"Torch Dtype: {settings.TORCH_DTYPE}")
logger.info(f"Attention Implementation: {attn_implementation}")
logger.info(f"Disable Talker (Audio Output): {settings.DISABLE_TALKER}")
logger.info(f"Redis URL: {settings.REDIS_URL}")
logger.info(f"Upload Dir: {settings.UPLOAD_DIR}")
logger.info(f"Audio Output Dir: {settings.AUDIO_OUTPUT_DIR}")
logger.info(f"Server Port: {settings.SERVER_PORT}")
