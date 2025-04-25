import logging
import torch
import os
import uuid
import soundfile as sf
from typing import Optional, List, Dict, Any, Tuple

# Attempt to import process_mm_info, assuming it's made available by transformers
# when loading with trust_remote_code=True, or placed manually in the path.
try:
    # This might be dynamically loaded or require the utils script in the path
    from qwen_omni_utils import process_mm_info
    _process_mm_info_available = True
except ImportError:
    logging.error("Failed to import 'process_mm_info' from 'qwen_omni_utils'. "
                  "Ensure the necessary utility script is available (potentially via trust_remote_code=True during model loading) "
                  "or multimodal processing will fail.")
    _process_mm_info_available = False
    # Define a dummy function to prevent crashes but highlight the issue
    def process_mm_info(*args, **kwargs):
        raise ImportError("process_mm_info function is not available.")


from .model_loader import get_model_and_processor
from .config import settings
from app.utils.helpers import cleanup_file # Import cleanup helper for input files

logger = logging.getLogger(__name__)

def run_inference(
    prompt: str,
    system_prompt: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    audio_paths: Optional[List[str]] = None,
    video_paths: Optional[List[str]] = None,
    # New parameters
    return_audio: bool = False,
    use_audio_in_video: bool = True,
    speaker: Optional[str] = None,
    max_new_tokens: Optional[int] = 512,
) -> Dict[str, Any]:
    """
    Performs inference using the loaded multimodal model, potentially generating audio.

    Args:
        prompt: The user's text prompt.
        system_prompt: An optional system prompt.
        image_paths: List of paths to temporary image files.
        audio_paths: List of paths to temporary audio files.
        video_paths: List of paths to temporary video files.
        return_audio: Whether to generate audio output.
        use_audio_in_video: Whether to process audio within video inputs.
        speaker: The desired speaker voice for audio output ('Chelsie' or 'Ethan').
        max_new_tokens: Maximum number of new tokens for text generation.

    Returns:
        A dictionary containing:
        - 'generated_text': The generated text response.
        - 'generated_audio_path': The relative path to the saved audio file (if generated),
                                   relative to AUDIO_OUTPUT_DIR. None otherwise.
        - 'error': An error message string if inference failed.
    """
    model, processor = get_model_and_processor()
    result = {"generated_text": None, "generated_audio_path": None, "error": None}
    all_temp_input_files = (image_paths or []) + (audio_paths or []) + (video_paths or [])
    generated_audio_full_path = None

    try:
        # --- 1. Determine System Prompt ---
        # Use specific audio prompt if audio is requested, otherwise use provided or default
        if return_audio:
            effective_system_prompt = settings.AUDIO_SYSTEM_PROMPT_TEXT
            logger.info("Using audio-specific system prompt as audio output is requested.")
        else:
            effective_system_prompt = system_prompt or settings.DEFAULT_SYSTEM_PROMPT_TEXT

        # --- 2. Construct Messages Payload ---
        conversation = [{"role": "system", "content": [{"type": "text", "text": effective_system_prompt}]}]
        user_content = [{"type": "text", "text": prompt}]
        if image_paths:
            for path in image_paths: user_content.append({"type": "image", "image": path})
        if audio_paths:
            for path in audio_paths: user_content.append({"type": "audio", "audio": path})
        if video_paths:
            for path in video_paths: user_content.append({"type": "video", "video": path})
        conversation.append({"role": "user", "content": user_content})

        logger.debug(f"Constructed conversation: {conversation}")
        logger.info(f"Parameters - return_audio: {return_audio}, use_audio_in_video: {use_audio_in_video}, speaker: {speaker or settings.DEFAULT_SPEAKER}, max_new_tokens: {max_new_tokens}")

        # --- 3. Preprocess Inputs ---
        # Apply chat template
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        logger.debug(f"Text after applying template: {text}")

        # Process multimedia info using the utility function
        if not _process_mm_info_available:
             raise RuntimeError("Cannot process multimedia inputs: process_mm_info is unavailable.")
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        logger.debug(f"Processed MM info - Audios: {audios}, Images: {images}, Videos: {videos}")

        # Prepare inputs for the model using the processor
        inputs = processor(
            text=text,
            audio=audios, # Note: parameter name is 'audio' not 'audios' for processor
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video # Pass this flag to processor as well
        )
        inputs = inputs.to(model.device).to(model.dtype)
        logger.debug(f"Inputs prepared for model on device: {model.device}, dtype: {model.dtype}")

        # --- 4. Generate Response ---
        effective_speaker = speaker or settings.DEFAULT_SPEAKER if return_audio else None
        generate_kwargs = {
            "use_audio_in_video": use_audio_in_video,
            "return_audio": return_audio and not settings.DISABLE_TALKER, # Only return audio if requested AND not disabled
            "speaker": effective_speaker,
            "max_new_tokens": max_new_tokens,
        }
        logger.info(f"Calling model.generate with kwargs: {generate_kwargs}")

        with torch.no_grad():
            # The model might return (text_ids, audio_tensor) or just text_ids
            output = model.generate(**inputs, **generate_kwargs)

        logger.debug("Model generation complete.")

        # --- 5. Process Output ---
        generated_audio_tensor = None
        if isinstance(output, tuple) and len(output) == 2:
            text_ids, generated_audio_tensor = output
            logger.info("Received text and audio output from model.")
        elif isinstance(output, torch.Tensor):
             text_ids = output
             logger.info("Received only text output from model.")
        else:
             raise TypeError(f"Unexpected output type from model.generate: {type(output)}")

        # Decode text
        decoded_text = processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        result["generated_text"] = decoded_text[0] if decoded_text else "Failed to decode text response."
        logger.info(f"Decoded text: {result['generated_text'][:100]}...")

        # Save audio if generated
        if generated_audio_tensor is not None and generated_audio_tensor.numel() > 0:
            audio_data = generated_audio_tensor.reshape(-1).detach().cpu().numpy()
            # Create a unique filename for the audio output
            audio_filename = f"{uuid.uuid4()}.wav"
            generated_audio_full_path = os.path.join(settings.AUDIO_OUTPUT_DIR, audio_filename)
            try:
                sf.write(generated_audio_full_path, audio_data, settings.AUDIO_SAMPLE_RATE)
                result["generated_audio_path"] = audio_filename # Return relative path
                logger.info(f"Generated audio saved to: {generated_audio_full_path}")
            except Exception as audio_err:
                logger.exception(f"Failed to save generated audio to {generated_audio_full_path}: {audio_err}")
                result["error"] = f"Inference succeeded but failed to save audio: {audio_err}"
                # Don't delete the text result just because audio saving failed
        elif return_audio and (settings.DISABLE_TALKER or generated_audio_tensor is None or generated_audio_tensor.numel() == 0):
             logger.warning("Audio output was requested but not generated (talker might be disabled or model didn't produce audio).")


    except ImportError as e:
         logger.error(f"ImportError during inference, likely qwen_omni_utils: {e}")
         result["error"] = f"Error: Missing required utility ({e}). Cannot process multimodal input."
    except Exception as e:
        logger.exception(f"An error occurred during inference: {e}")
        result["error"] = f"Error during inference: {str(e)}"
        # Ensure audio file isn't referenced if saving failed before completion
        result["generated_audio_path"] = None
        if generated_audio_full_path and os.path.exists(generated_audio_full_path):
             cleanup_file(generated_audio_full_path) # Clean up partially saved audio on error

    finally:
        # Cleanup temporary INPUT files
        for file_path in all_temp_input_files:
            cleanup_file(file_path)

    # Return the result dictionary
    if result["error"]:
         # Ensure text is None if there was a critical error preventing generation
         if result["generated_text"] is None:
              result["generated_text"] = "Inference failed."
    elif result["generated_text"] is None: # Should not happen if no error, but safeguard
         result["generated_text"] = "Inference completed but no text was generated."


    return result
