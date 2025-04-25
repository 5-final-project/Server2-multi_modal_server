import os
import torch
from types import MethodType
from typing import Tuple

from transformers import (
    Qwen2_5OmniModel,
    Qwen2_5OmniProcessor,
    AutoTokenizer # Keep AutoTokenizer for potential future use or flexibility
)
from transformers.utils.hub import cached_file
from gptqmodel import GPTQModel, QuantizeConfig, BACKEND
from gptqmodel.models.base import BaseGPTQModel, move_to
from gptqmodel.models.auto import MODEL_MAP, SUPPORTED_MODELS
from gptqmodel.models._const import CPU

from .config import settings
import logging

logger = logging.getLogger(__name__)

# --- GPTQ Model Definition (Adapted from example) ---
class Qwen25OmniThiknerGPTQ(BaseGPTQModel):
    loader = Qwen2_5OmniModel
    base_modules = [
        "thinker.model.embed_tokens",
        "thinker.model.norm",
        "token2wav",
        "thinker.audio_tower",
        "thinker.model.rotary_emb",
        "thinker.visual",
        "talker"
    ]
    pre_lm_head_norm_module = "thinker.model.norm"
    require_monkeypatch = False
    layers_node = "thinker.model.layers"
    layer_type = "Qwen2_5OmniDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    def pre_quantize_generate_hook_start(self):
        self.thinker.visual = move_to(self.thinker.visual, device=self.quantize_config.device)
        self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        self.thinker.visual = move_to(self.thinker.visual, device=CPU)
        self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=CPU)

    def preprocess_dataset(self, sample: dict) -> dict:
        # Basic preprocessing, might need adjustment based on actual dataset if used for quantization
        return sample

# Register the custom model with GPTQModel
MODEL_MAP["qwen2_5_omni"] = Qwen25OmniThiknerGPTQ
SUPPORTED_MODELS.append("qwen2_5_omni")

# --- Monkey Patch for Speaker Loading (Adapted from example) ---
def _patched_from_config(cls, config, *args, **kwargs):
    # Ensure trust_remote_code is handled appropriately if present
    kwargs.pop("trust_remote_code", None) # Pop it as GPTQModel might handle it differently

    # Use the original _from_config method
    model = cls._from_config(config, **kwargs)

    # Load speakers - Use model_path from settings
    spk_path = cached_file(
        settings.MODEL_PATH, # Use configured model path
        "spk_dict.pt",
        subfolder=kwargs.pop("subfolder", None),
        cache_dir=kwargs.pop("cache_dir", None),
        force_download=kwargs.pop("force_download", False),
        proxies=kwargs.pop("proxies", None),
        resume_download=kwargs.pop("resume_download", None),
        local_files_only=kwargs.pop("local_files_only", False),
        token=kwargs.pop("use_auth_token", None),
        revision=kwargs.pop("revision", None),
    )
    if spk_path is None:
        logger.warning(f"Speaker dictionary (spk_dict.pt) not found in {settings.MODEL_PATH}. Speaker loading skipped.")
        # raise ValueError(f"Speaker dictionary not found at {spk_path}") # Optional: make it non-fatal
    else:
        try:
            model.load_speakers(spk_path)
            logger.info(f"Successfully loaded speakers from {spk_path}")
        except Exception as e:
            logger.error(f"Failed to load speakers from {spk_path}: {e}")
            # Decide if this should be a fatal error

    return model

# Apply the patch only if the class exists and hasn't been patched
if hasattr(Qwen2_5OmniModel, 'from_config') and not hasattr(Qwen2_5OmniModel.from_config, '_patched'):
    # Store original method if needed, though GPTQModel might handle this internally
    # Qwen2_5OmniModel._original_from_config = Qwen2_5OmniModel.from_config
    Qwen2_5OmniModel.from_config = classmethod(_patched_from_config)
    Qwen2_5OmniModel.from_config._patched = True # Mark as patched
    logger.info("Patched Qwen2_5OmniModel.from_config for speaker loading.")


# --- Model and Processor Loading Function ---
_model = None
_processor = None
_torch_dtype = None

def get_torch_dtype(dtype_str: str):
    """Converts string representation to torch dtype."""
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float32":
        return torch.float32
    else:
        logger.warning(f"Unsupported torch_dtype '{dtype_str}'. Defaulting to float16.")
        return torch.float16

def load_model_and_processor() -> Tuple[GPTQModel, Qwen2_5OmniProcessor]:
    """Loads the GPTQ model and processor based on settings."""
    global _model, _processor, _torch_dtype

    if _model is not None and _processor is not None:
        logger.info("Model and processor already loaded.")
        return _model, _processor

    logger.info(f"Loading model from: {settings.MODEL_PATH}")
    logger.info(f"Using device_map: {settings.DEVICE_MAP}")
    logger.info(f"Using torch_dtype: {settings.TORCH_DTYPE_STR}")
    logger.info(f"Using attn_implementation: {settings.ATTN_IMPLEMENTATION}")

    _torch_dtype = get_torch_dtype(settings.TORCH_DTYPE_STR)

    try:
        # Load GPTQ Model
        _model = GPTQModel.load(
            settings.MODEL_PATH,
            device_map=settings.DEVICE_MAP,
            torch_dtype=_torch_dtype,
            attn_implementation=settings.ATTN_IMPLEMENTATION,
            # trust_remote_code=True # Often needed for custom models
        )
        logger.info("GPTQ Model loaded successfully.")

        # Load Processor
        _processor = Qwen2_5OmniProcessor.from_pretrained(
            settings.MODEL_PATH,
            # trust_remote_code=True # Often needed
        )
        logger.info("Processor loaded successfully.")

        return _model, _processor

    except Exception as e:
        logger.exception(f"Error loading model or processor: {e}")
        # Depending on the application, you might want to exit or handle this differently
        raise RuntimeError(f"Failed to load model/processor from {settings.MODEL_PATH}") from e

def get_model_processor_and_dtype() -> Tuple[GPTQModel, Qwen2_5OmniProcessor, torch.dtype]:
    """Returns the loaded model, processor, and torch dtype."""
    if _model is None or _processor is None or _torch_dtype is None:
        load_model_and_processor() # Ensure they are loaded
    return _model, _processor, _torch_dtype

# Example of how to trigger loading on module import (optional)
# load_model_and_processor()
