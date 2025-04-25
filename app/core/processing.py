import logging
from typing import Optional, List, Dict, Any
import torch
import os

# Assuming qwen_omni_utils provides process_mm_info
# If it's a local file, ensure it's importable (e.g., in app/utils)
# If it's part of a package, ensure the package is installed
try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    logging.warning("qwen_omni_utils not found. Multimodal processing might fail. "
                    "Ensure the utility script is available and importable.")
    # Define a dummy function if it's missing, to avoid crashing, but log heavily.
    def process_mm_info(messages, **kwargs):
        logging.error("Dummy process_mm_info called. Real implementation missing!")
        # Return empty structures matching expected output types
        return None, None, None # audios, images, videos

from .model_loader import get_model_processor_and_dtype
from .config import settings
from app.utils.helpers import cleanup_file # Import cleanup helper

logger = logging.getLogger(__name__)

def run_inference(
    prompt: str,
    system_prompt: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    audio_paths: Optional[List[str]] = None,
    video_paths: Optional[List[str]] = None,
) -> str:
    """
    Performs inference using the loaded multimodal model.

    Args:
        prompt: The user's text prompt.
        system_prompt: An optional system prompt.
        image_paths: List of paths to temporary image files.
        audio_paths: List of paths to temporary audio files.
        video_paths: List of paths to temporary video files.

    Returns:
        The generated text response from the model.
    """
    model, processor, torch_dtype = get_model_processor_and_dtype()
    generated_text = "Error during inference." # Default error message
    all_temp_files = (image_paths or []) + (audio_paths or []) + (video_paths or [])

    try:
        # 1. Construct the messages payload for the processor
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
             # Use default system prompt if none provided
             messages.append({"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT})


        user_content = [{"type": "text", "text": prompt}]
        if image_paths:
            for path in image_paths:
                user_content.append({"type": "image", "image": path})
        if audio_paths:
             for path in audio_paths:
                 # Assuming 'audio' type is correct for the processor
                 user_content.append({"type": "audio", "audio": path})
        if video_paths:
            for path in video_paths:
                user_content.append({"type": "video", "video": path})

        messages.append({"role": "user", "content": user_content})

        logger.debug(f"Constructed messages: {messages}")

        # 2. Apply chat template and process multimedia info
        # Note: process_mm_info might need adjustments based on its exact implementation
        #       and how it expects file paths vs. loaded data. The example used paths.
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logger.debug(f"Text after applying template: {text}")

        # Assuming process_mm_info works correctly with the message format
        # use_audio_in_video=False is from the example, adjust if needed
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        logger.debug(f"Processed MM info - Audios: {audios}, Images: {images}, Videos: {videos}")


        # 3. Prepare inputs for the model
        inputs = processor(
            text=text,
            audios=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True # Ensure padding is handled correctly
        )

        # Move inputs to the correct device and dtype
        inputs = inputs.to(model.device).to(torch_dtype) # Use loaded dtype
        logger.debug(f"Inputs prepared for model on device: {model.device}, dtype: {torch_dtype}")

        # 4. Generate response
        # Generation parameters might need tuning (max_new_tokens, etc.)
        # return_audio=False is from the example
        with torch.no_grad(): # Ensure inference mode
            output = model.generate(
                **inputs,
                use_audio_in_video=False,
                return_audio=False,
                max_new_tokens=512 # Example: set a reasonable limit
            )
        logger.debug("Model generation complete.")

        # 5. Decode the output
        response_text = processor.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False # As per example
        )
        logger.debug(f"Decoded response: {response_text}")

        # Assuming batch size is 1 for API requests
        generated_text = response_text[0] if response_text else "Failed to decode response."

    except ImportError as e:
         logger.error(f"ImportError during inference, likely qwen_omni_utils: {e}")
         generated_text = f"Error: Missing required utility ({e}). Cannot process multimodal input."
    except Exception as e:
        logger.exception(f"An error occurred during inference: {e}")
        # Provide a more specific error message if possible
        generated_text = f"Error during inference: {str(e)}"
    finally:
        # 6. Cleanup temporary files
        for file_path in all_temp_files:
            cleanup_file(file_path)

    return generated_text
