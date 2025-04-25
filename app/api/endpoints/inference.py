import logging
import os
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional, Literal
from celery.result import AsyncResult

from app.api.schemas.inference import (
    # InferenceInputParameters, # Using Form fields directly
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
        # Check if the result from inference contains an error key
        if isinstance(result_data, dict) and result_data.get("error"):
             logger.error(f"Task {task_id} succeeded but inference reported error: {result_data['error']}")
             # Report as failure even if Celery task succeeded technically
             status_response.status = "FAILURE" # Override Celery's SUCCESS
             status_response.result = {"error": result_data['error'], "details": "Inference process failed."}
        # Check if the result is in the expected format for InferenceResponse
        elif isinstance(result_data, dict) and 'generated_text' in result_data:
             try:
                 # Validate and structure the successful result
                 status_response.result = InferenceResponse(**result_data)
                 status_response.status = "SUCCESS" # Ensure status is SUCCESS
             except Exception as pydantic_err:
                 logger.error(f"Task {task_id} succeeded but result format is invalid: {result_data}. Error: {pydantic_err}")
                 status_response.status = "FAILURE" # Treat format error as failure
                 status_response.result = {"error": "Invalid result format from worker.", "details": str(pydantic_err)}
        else:
             # Handle unexpected successful result format
             logger.error(f"Task {task_id} succeeded but returned unexpected result format: {result_data}")
             status_response.status = "FAILURE" # Treat format error as failure
             status_response.result = {"error": "Invalid result format from worker."}

    elif task_result.failed():
        # Access traceback or error info if stored in backend
        try:
            # Celery stores failure info in result.info or as the result itself
            error_info = task_result.info if task_result.info else str(task_result.result)
            if isinstance(error_info, dict):
                 status_response.result = error_info # Use the dict if available
            else:
                 status_response.result = {'error': str(error_info)} # Wrap string error
        except Exception as e:
             logger.error(f"Error retrieving failure info for task {task_id}: {e}")
             status_response.result = {'error': 'Failed to retrieve error details.'}
        logger.error(f"Task {task_id} failed. Info: {status_response.result}")
        # Status is already FAILURE

    # Handle other states like STARTED, RETRY, etc.
    elif task_result.state == 'STARTED' and isinstance(task_result.info, dict):
         status_response.result = task_result.info # e.g., {'status': 'Processing...'}
    elif task_result.state == 'PENDING':
         status_response.result = {'status': 'Task is waiting in queue.'}
    # Add other states if needed

    return status_response


# --- API Endpoints ---

@router.post("/predict/async", response_model=AsyncTaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_inference_task(
    # --- Input Files ---
    images: Optional[List[UploadFile]] = File(None, description="Image files for input"),
    audios: Optional[List[UploadFile]] = File(None, description="Audio files for input"),
    videos: Optional[List[UploadFile]] = File(None, description="Video files for input"),
    # --- Text and Parameters (as Form data) ---
    prompt: str = Form(..., description="The main text prompt for the model."),
    system_prompt: Optional[str] = Form(None, description="Optional system prompt override."),
    return_audio: bool = Form(False, description="Generate audio output?"),
    use_audio_in_video: bool = Form(True, description="Process audio within video files?"),
    speaker: Optional[Literal["Chelsie", "Ethan"]] = Form(None, description="Voice for audio output ('Chelsie' or 'Ethan')."),
    max_new_tokens: Optional[int] = Form(512, description="Max new tokens for text generation.")
):
    """
    Accepts multimodal input (text, images, audio, video) and parameters via form data,
    saves files temporarily, and queues an inference task for background processing.
    """
    image_paths: List[str] = []
    audio_paths: List[str] = []
    video_paths: List[str] = []
    saved_files: List[str] = [] # Track successfully saved files for cleanup on error

    try:
        # --- Save Uploaded Files ---
        async def save_files(files, path_list):
            if files:
                for file in files:
                    path = await save_upload_file(file)
                    if path:
                        path_list.append(path)
                        saved_files.append(path)
                    else:
                        # Clean up already saved files for this request before raising
                        for p in saved_files: cleanup_file(p)
                        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save file: {file.filename}")

        await save_files(images, image_paths)
        await save_files(audios, audio_paths)
        await save_files(videos, video_paths)

        logger.info(f"Dispatching inference task with prompt: '{prompt[:50]}...', {len(image_paths)} images, {len(audio_paths)} audios, {len(video_paths)} videos.")
        logger.info(f"Task params: return_audio={return_audio}, use_audio_in_video={use_audio_in_video}, speaker={speaker}, max_new_tokens={max_new_tokens}")

        # --- Send Task to Celery Queue ---
        task = inference_task.delay(
            prompt=prompt,
            system_prompt=system_prompt, # Pass None if not provided, processing fn handles default
            image_paths=image_paths,
            audio_paths=audio_paths,
            video_paths=video_paths,
            # Pass new parameters
            return_audio=return_audio,
            use_audio_in_video=use_audio_in_video,
            speaker=speaker,
            max_new_tokens=max_new_tokens
        )

        logger.info(f"Task {task.id} queued successfully.")
        return AsyncTaskResponse(task_id=task.id)

    except HTTPException as http_exc:
         # Re-raise HTTP exceptions (e.g., from file saving failure)
         # Cleanup was handled within save_files
         raise http_exc
    except Exception as e:
        logger.exception(f"Error submitting inference task: {e}")
        # General error during task submission (not file saving)
        # Clean up any saved files before raising internal server error
        for file_path in saved_files:
            cleanup_file(file_path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error queuing task: {str(e)}")


@router.get("/predict/status/{task_id}", response_model=TaskStatusResponse)
async def get_inference_status(task_id: str):
    """
    Retrieves the status and result (if available) of a previously submitted inference task.
    Result includes generated text and potentially a path to the generated audio file.
    """
    logger.info(f"Checking status for task: {task_id}")
    status_response = get_task_status(task_id)
    # Return appropriate status code based on task state
    if status_response.status == "SUCCESS":
         return JSONResponse(content=status_response.dict(), status_code=status.HTTP_200_OK)
    elif status_response.status == "FAILURE":
         return JSONResponse(content=status_response.dict(), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else: # PENDING, STARTED, RETRY etc.
         return JSONResponse(content=status_response.dict(), status_code=status.HTTP_202_ACCEPTED)


@router.get("/audio/{filename}", response_class=FileResponse)
async def get_generated_audio(filename: str):
    """
    Serves a generated audio file. The filename should match the one returned
    in the 'generated_audio_path' field of a successful task status response.
    """
    # Basic security check: prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename.")

    file_path = os.path.join(settings.AUDIO_OUTPUT_DIR, filename)
    logger.info(f"Attempting to serve audio file: {file_path}")

    if os.path.exists(file_path) and os.path.isfile(file_path):
        # Ensure the path is within the intended directory (redundant but safe)
        if os.path.commonpath([settings.AUDIO_OUTPUT_DIR]) == os.path.commonpath([settings.AUDIO_OUTPUT_DIR, file_path]):
             return FileResponse(path=file_path, media_type="audio/wav", filename=filename)
        else:
             logger.warning(f"Attempt to access file outside designated audio directory: {filename}")
             raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
    else:
        logger.error(f"Audio file not found: {file_path}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not found.")
