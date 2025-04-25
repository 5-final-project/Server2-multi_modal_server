import logging
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional
from celery.result import AsyncResult

from app.api.schemas.inference import (
    InferenceInput, # We'll use Form fields instead
    InferenceResponse,
    AsyncTaskResponse,
    TaskStatusResponse
)
from app.core.queue_manager import celery_app
from app.core.inference_worker import inference_task # Import the task
from app.utils.helpers import save_upload_file, cleanup_file
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Helper Function to Get Task Result ---
def get_task_status(task_id: str) -> TaskStatusResponse:
    """Checks the status and result of a Celery task."""
    task_result = AsyncResult(task_id, app=celery_app)

    status_response = TaskStatusResponse(task_id=task_id, status=task_result.status)

    if task_result.successful():
        result_data = task_result.result
        # Ensure result_data is in the expected format (dict with 'generated_text')
        if isinstance(result_data, dict) and 'generated_text' in result_data:
             status_response.result = InferenceResponse(**result_data)
        else:
             # Handle unexpected result format
             logger.error(f"Task {task_id} succeeded but returned unexpected result format: {result_data}")
             status_response.status = "ERROR_RESULT_FORMAT"
             status_response.result = {"error": "Invalid result format from worker."}

    elif task_result.failed():
        # Access traceback or error info if stored in backend
        error_info = task_result.info if isinstance(task_result.info, dict) else {'error': str(task_result.info)}
        status_response.result = error_info
        logger.error(f"Task {task_id} failed. Info: {task_result.info}")
        # Status is already FAILURE

    # Other statuses: PENDING, STARTED, RETRY, REVOKED
    # You might want to return metadata if available for STARTED etc.
    elif task_result.state == 'STARTED' and isinstance(task_result.info, dict):
         status_response.result = task_result.info # e.g., {'status': 'Processing...'}


    return status_response


# --- API Endpoints ---

@router.post("/predict/async", response_model=AsyncTaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_inference_task(
    prompt: str = Form(...),
    system_prompt: Optional[str] = Form(None),
    images: Optional[List[UploadFile]] = File(None, description="Image files for input"),
    audios: Optional[List[UploadFile]] = File(None, description="Audio files for input"),
    videos: Optional[List[UploadFile]] = File(None, description="Video files for input"),
):
    """
    Accepts multimodal input (text, images, audio, video) via form data,
    saves files temporarily, and queues an inference task for background processing.
    """
    image_paths: List[str] = []
    audio_paths: List[str] = []
    video_paths: List[str] = []
    saved_files: List[str] = [] # Keep track of successfully saved files for potential cleanup on error

    try:
        # Save uploaded files asynchronously
        if images:
            for file in images:
                path = await save_upload_file(file)
                if path:
                    image_paths.append(path)
                    saved_files.append(path)
                else:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save image file: {file.filename}")
        if audios:
            for file in audios:
                path = await save_upload_file(file)
                if path:
                    audio_paths.append(path)
                    saved_files.append(path)
                else:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save audio file: {file.filename}")
        if videos:
            for file in videos:
                path = await save_upload_file(file)
                if path:
                    video_paths.append(path)
                    saved_files.append(path)
                else:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save video file: {file.filename}")

        logger.info(f"Dispatching inference task with prompt: '{prompt[:50]}...', {len(image_paths)} images, {len(audio_paths)} audios, {len(video_paths)} videos.")

        # Send task to Celery queue
        # Pass file paths, the worker task will handle reading them
        task = inference_task.delay(
            prompt=prompt,
            system_prompt=system_prompt or settings.DEFAULT_SYSTEM_PROMPT,
            image_paths=image_paths,
            audio_paths=audio_paths,
            video_paths=video_paths
        )

        logger.info(f"Task {task.id} queued successfully.")
        return AsyncTaskResponse(task_id=task.id)

    except HTTPException as http_exc:
         # If file saving failed, clean up any files already saved for this request
         for file_path in saved_files:
             cleanup_file(file_path)
         raise http_exc # Re-raise the HTTP exception
    except Exception as e:
        logger.exception(f"Error submitting inference task: {e}")
        # Clean up any saved files before raising internal server error
        for file_path in saved_files:
            cleanup_file(file_path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error queuing task: {str(e)}")


@router.get("/predict/status/{task_id}", response_model=TaskStatusResponse)
async def get_inference_status(task_id: str):
    """
    Retrieves the status and result (if available) of a previously submitted inference task.
    """
    logger.info(f"Checking status for task: {task_id}")
    status_response = get_task_status(task_id)
    return status_response

# Optional: Synchronous endpoint (might block server, use with caution or separate worker)
# @router.post("/predict/sync", response_model=InferenceResponse)
# async def run_inference_sync(...):
#     # Similar logic to async, but call run_inference directly
#     # Be very careful about blocking the event loop
#     pass
