import logging
from typing import Optional, List, Dict, Any

from .queue_manager import celery_app
from .processing import run_inference
from .model_loader import load_model_and_processor # Ensure model is loaded in worker process
from app.utils.helpers import cleanup_file # Import cleanup helper for input files

logger = logging.getLogger(__name__)

# Ensure the model is loaded when the worker starts
# This consumes memory per worker process. Consider alternatives for large scale.
try:
    logger.info("Worker process started. Loading model and processor...")
    # Pass trust_remote_code=True if needed by the specific model version
    load_model_and_processor()
    logger.info("Model and processor loaded successfully in worker.")
except Exception as e:
    logger.exception(f"Failed to load model in worker process: {e}")
    # Worker might fail tasks if model loading fails.


@celery_app.task(bind=True, name="app.core.inference_worker.inference_task")
def inference_task(
    self, # 'bind=True' provides access to the task instance ('self')
    prompt: str,
    system_prompt: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    audio_paths: Optional[List[str]] = None,
    video_paths: Optional[List[str]] = None,
    # New parameters passed from the API endpoint
    return_audio: bool = False,
    use_audio_in_video: bool = True,
    speaker: Optional[str] = None,
    max_new_tokens: Optional[int] = 512,
) -> Dict[str, Any]:
    """
    Celery task to perform multimodal inference in the background.

    Args:
        prompt: The user's text prompt.
        system_prompt: An optional system prompt.
        image_paths: List of paths to temporary image files.
        audio_paths: List of paths to temporary audio files.
        video_paths: List of paths to temporary video files.
        return_audio: Whether to generate audio output.
        use_audio_in_video: Whether to process audio within video inputs.
        speaker: The desired speaker voice for audio output.
        max_new_tokens: Maximum number of new tokens for text generation.

    Returns:
        A dictionary containing the generated text, optional audio path, or error info.
        Matches the structure expected by InferenceResponse schema.
    """
    task_id = self.request.id
    logger.info(f"Task {task_id}: Received inference request.")
    # Input files are handled by run_inference's finally block now

    try:
        # Update task state to STARTED
        self.update_state(state='STARTED', meta={'status': 'Processing...'})

        logger.info(f"Task {task_id}: Running inference for prompt: '{prompt[:50]}...'")
        # Call run_inference with all parameters
        result_dict = run_inference(
            prompt=prompt,
            system_prompt=system_prompt,
            image_paths=image_paths,
            audio_paths=audio_paths,
            video_paths=video_paths,
            return_audio=return_audio,
            use_audio_in_video=use_audio_in_video,
            speaker=speaker,
            max_new_tokens=max_new_tokens,
        )
        logger.info(f"Task {task_id}: Inference completed.")

        # Check if inference itself reported an error
        if result_dict.get("error"):
             logger.error(f"Task {task_id}: Inference function reported an error: {result_dict['error']}")
             # Update state to FAILURE with the error from inference
             self.update_state(
                 state='FAILURE',
                 meta={
                     'exc_type': 'InferenceError',
                     'exc_message': result_dict['error'],
                     'status': 'Task failed during inference'
                 }
             )
             # Return the error dict so the status endpoint can report it
             return {"error": result_dict['error']}


        # Return the successful result dictionary directly
        # Celery will mark the task as SUCCESS automatically
        return result_dict

    except Exception as e:
        logger.exception(f"Task {task_id}: Unhandled error during inference task execution: {e}")
        # Update state to FAILURE for unhandled exceptions
        self.update_state(
            state='FAILURE',
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'status': 'Task failed unexpectedly'
            }
        )
        # Reraise the exception so Celery marks the task as failed correctly
        raise

    # Note: Input file cleanup is handled within run_inference's finally block.
    # Output audio file cleanup needs a separate strategy (e.g., TTL, manual cleanup endpoint).
