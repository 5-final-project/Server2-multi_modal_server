import logging
from typing import Optional, List, Dict, Any

from .queue_manager import celery_app
from .processing import run_inference
from .model_loader import load_model_and_processor # Ensure model is loaded in worker process
from app.utils.helpers import cleanup_file # Import cleanup helper

logger = logging.getLogger(__name__)

# Ensure the model is loaded when the worker starts
# This might consume significant memory per worker process.
# Consider alternative strategies if memory is a constraint (e.g., shared model server).
try:
    logger.info("Worker process started. Loading model and processor...")
    load_model_and_processor()
    logger.info("Model and processor loaded successfully in worker.")
except Exception as e:
    logger.exception(f"Failed to load model in worker process: {e}")
    # Depending on requirements, you might want the worker to exit or retry.
    # For now, it will log the error and might fail tasks.


@celery_app.task(bind=True, name="app.core.inference_worker.inference_task")
def inference_task(
    self, # 'bind=True' provides access to the task instance ('self')
    prompt: str,
    system_prompt: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    audio_paths: Optional[List[str]] = None,
    video_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Celery task to perform multimodal inference in the background.

    Args:
        prompt: The user's text prompt.
        system_prompt: An optional system prompt.
        image_paths: List of paths to temporary image files.
        audio_paths: List of paths to temporary audio files.
        video_paths: List of paths to temporary video files.

    Returns:
        A dictionary containing the generated text or error information.
    """
    task_id = self.request.id
    logger.info(f"Task {task_id}: Received inference request.")
    all_temp_files = (image_paths or []) + (audio_paths or []) + (video_paths or [])

    try:
        # Update task state to STARTED (optional, requires task_track_started=True)
        self.update_state(state='STARTED', meta={'status': 'Processing...'})

        logger.info(f"Task {task_id}: Running inference for prompt: '{prompt[:50]}...'")
        generated_text = run_inference(
            prompt=prompt,
            system_prompt=system_prompt,
            image_paths=image_paths,
            audio_paths=audio_paths,
            video_paths=video_paths,
            # Note: run_inference now handles its own file cleanup
        )
        logger.info(f"Task {task_id}: Inference completed.")

        # Return result in a structured format matching InferenceResponse schema
        result = {"generated_text": generated_text}
        # Optionally update state to SUCCESS before returning
        # self.update_state(state='SUCCESS', meta=result)
        return result

    except Exception as e:
        logger.exception(f"Task {task_id}: Error during inference task execution: {e}")
        # Update state to FAILURE
        # Celery automatically sets state to FAILURE on unhandled exceptions,
        # but you can customize the metadata.
        self.update_state(
            state='FAILURE',
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'status': 'Task failed'
            }
        )
        # Ensure cleanup happens even if run_inference failed before its finally block
        for file_path in all_temp_files:
            cleanup_file(file_path)
        # Reraise the exception so Celery marks the task as failed
        raise

    # Note: The 'finally' block in run_inference should handle cleanup
    # in most cases, but the extra cleanup in the except block here is a safeguard.
