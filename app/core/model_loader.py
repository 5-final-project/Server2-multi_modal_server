import os
import torch
from typing import Tuple
import logging

from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    BitsAndBytesConfig # For potential future quantization (e.g., 4-bit)
)
# Try importing flash_attn; proceed if available and enabled
try:
    import flash_attn
    _flash_attn_available = True
except ImportError:
    _flash_attn_available = False
    logging.warning("flash-attn library not found. Flash Attention 2 will be disabled.")


from .config import settings, attn_implementation # Import effective attn_implementation

logger = logging.getLogger(__name__)

# --- Global variables to hold the loaded model and processor ---
_model = None
_processor = None
_torch_dtype_loaded = None


def get_torch_dtype(dtype_str: str):
    """Converts string representation to torch dtype object or returns 'auto'."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto" # Keep 'auto' as string for from_pretrained
    }
    selected_dtype = dtype_map.get(dtype_str.lower(), "auto")
    if selected_dtype != "auto" and not isinstance(selected_dtype, str):
         logger.info(f"Resolved torch_dtype '{dtype_str}' to {selected_dtype}")
    elif selected_dtype == "auto":
         logger.info(f"Using torch_dtype='auto'")
    else:
         logger.warning(f"Unsupported torch_dtype string '{dtype_str}'. Defaulting to 'auto'.")
         selected_dtype = "auto"
    return selected_dtype


def load_model_and_processor() -> Tuple[Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor]:
    """Loads the Hugging Face model and processor based on settings."""
    global _model, _processor, _torch_dtype_loaded

    if _model is not None and _processor is not None:
        logger.info("Model and processor already loaded.")
        return _model, _processor

    logger.info(f"Loading model: {settings.MODEL_ID}")
    logger.info(f"Using device_map: {settings.DEVICE_MAP}")
    logger.info(f"Requested torch_dtype: {settings.TORCH_DTYPE}")
    logger.info(f"Using attn_implementation: {attn_implementation}")
    logger.info(f"Disable Talker (Audio Output): {settings.DISABLE_TALKER}")

    _torch_dtype_loaded = get_torch_dtype(settings.TORCH_DTYPE)
    effective_attn_impl = attn_implementation if _flash_attn_available and settings.USE_FLASH_ATTENTION_2 else None

    if effective_attn_impl:
        logger.info(f"Attempting to load with attn_implementation='{effective_attn_impl}'")
        # FlashAttention requires float16 or bfloat16
        if _torch_dtype_loaded not in [torch.float16, torch.bfloat16, "auto"]:
             logger.warning(f"Flash Attention 2 requires float16 or bfloat16 dtype, but got {_torch_dtype_loaded}. Disabling Flash Attention.")
             effective_attn_impl = None # Disable if dtype is incompatible
    else:
         logger.info("Flash Attention 2 is disabled or unavailable.")


    try:
        # Load Model from Hugging Face Hub
        _model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            settings.MODEL_ID,
            torch_dtype=_torch_dtype_loaded,
            device_map=settings.DEVICE_MAP,
            attn_implementation=effective_attn_impl,
            trust_remote_code=True # Often required for complex models
            # Add quantization config here if needed in the future
            # quantization_config=BitsAndBytesConfig(...)
        )
        logger.info("Model loaded successfully.")

        # Disable talker if configured
        if settings.DISABLE_TALKER:
            try:
                _model.disable_talker()
                logger.info("Talker disabled successfully (audio output unavailable).")
            except AttributeError:
                logger.warning("Model does not have a 'disable_talker' method. Audio might still be generated if supported.")


        # Load Processor
        _processor = Qwen2_5OmniProcessor.from_pretrained(
            settings.MODEL_ID,
            trust_remote_code=True # Often required
        )
        logger.info("Processor loaded successfully.")

        # Store the actual dtype the model was loaded with if 'auto' was used
        if _torch_dtype_loaded == "auto":
             _torch_dtype_loaded = _model.dtype # Get the actual dtype
             logger.info(f"Model loaded with dtype: {_torch_dtype_loaded} (resolved from 'auto')")


        return _model, _processor

    except Exception as e:
        logger.exception(f"Error loading model or processor from Hugging Face Hub ({settings.MODEL_ID}): {e}")
        raise RuntimeError(f"Failed to load model/processor {settings.MODEL_ID}") from e


def get_model_and_processor() -> Tuple[Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor]:
    """Returns the loaded model and processor."""
    if _model is None or _processor is None:
        load_model_and_processor() # Ensure they are loaded
    # We don't return dtype anymore as it's handled internally or by the model object itself
    return _model, _processor

# Example of how to trigger loading on module import (optional, usually done lazily)
# load_model_and_processor()
