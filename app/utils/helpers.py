import os
import uuid
import aiofiles
from fastapi import UploadFile
from typing import Optional

from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

async def save_upload_file(upload_file: UploadFile) -> Optional[str]:
    """
    Saves an uploaded file to the temporary directory specified in settings.

    Args:
        upload_file: The file uploaded via FastAPI.

    Returns:
        The full path to the saved file, or None if saving failed.
    """
    if not upload_file or not upload_file.filename:
        logger.warning("Received empty or invalid upload file object.")
        return None

    # Create a unique filename to avoid collisions
    _, ext = os.path.splitext(upload_file.filename)
    unique_filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

    try:
        # Ensure the upload directory exists (config should handle this, but double-check)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        # Asynchronously write the file content
        async with aiofiles.open(file_path, 'wb') as out_file:
            while content := await upload_file.read(1024 * 1024):  # Read in 1MB chunks
                await out_file.write(content)

        logger.info(f"Successfully saved uploaded file to: {file_path}")
        return file_path
    except Exception as e:
        logger.exception(f"Failed to save upload file '{upload_file.filename}' to {file_path}: {e}")
        # Clean up partially written file if it exists
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as rm_err:
                logger.error(f"Error removing partially saved file {file_path}: {rm_err}")
        return None
    finally:
        # Ensure the file pointer is closed, although aiofiles context manager should handle this
        await upload_file.close()

def cleanup_file(file_path: Optional[str]):
    """Removes a file if it exists."""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        except OSError as e:
            logger.error(f"Error removing temporary file {file_path}: {e}")
