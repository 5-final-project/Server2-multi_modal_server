from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

# Import schema and model processors
from .schemas.request import LlamaRequest, QwenRequest
from .model.clova_processor import HyperCLOVAXModelProcessor
from .model.qwen_processor import QwenProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the model processor instance
# Using a dictionary to store context accessible within lifespan
context: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Loads the model on startup.
    """
    logger.info("Application startup: Loading models...")
    try:
        # Instantiate and store the model processors in the context dictionary
        context["clova_processor"] = HyperCLOVAXModelProcessor()
        context["qwen_processor"] = QwenProcessor()
        logger.info("Models loaded successfully.")
        yield # Application runs here
    except Exception as e:
        logger.error(f"Fatal error during model loading: {e}", exc_info=True)
        # Optionally, prevent the app from starting or handle differently
        # For now, we let it yield, but the processors might be None or raise errors later
        yield # Allow app to start but it will likely fail on requests
    finally:
        # Clean up resources on shutdown (if necessary)
        logger.info("Application shutdown.")
        # Clear the context or perform other cleanup
        context.clear()


# Create FastAPI app instance with lifespan management
app = FastAPI(
    title="Llama4 Multi-Modal API",
    description="API to interact with the Llama4 model for text generation based on text and image inputs.",
    version="0.1.0",
    lifespan=lifespan
)

@app.post("/generate", summary="Generate text based on messages", response_description="The generated text response")
async def generate_text(
    request_body: LlamaRequest = Body(...) # Use Body for explicit request body definition
):
    """
    Accepts a list of messages (including text and image URLs) and returns
    a generated text response from the Clova model.
    """
    processor = context.get("clova_processor")
    if not processor:
        logger.error("Clova processor not available in context. Was there an error during startup?")
        raise HTTPException(status_code=500, detail="Model processor is not available. Check server logs.")

    try:
        # Convert Pydantic models to dictionaries expected by the processor
        # FastAPI might do this automatically, but being explicit can prevent issues
        messages_dict_list = [message.model_dump() for message in request_body.messages]

        logger.info(f"Received Clova generation request with {len(messages_dict_list)} messages.")
        # Call the model processor
        response_text = processor.process_messages(messages_dict_list)
        logger.info("Successfully generated Clova response.")
        return {"response": response_text}
    except RuntimeError as e: # Catch errors specifically from the processor
        logger.error(f"Runtime error during Clova processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model processing error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Clova processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.post("/qwen/generate", summary="Generate text using the Qwen model", response_description="The generated text response from Qwen")
async def generate_text_qwen(
    request_body: QwenRequest = Body(...)
):
    """
    Accepts a prompt and returns a generated text response from the Qwen model.
    """
    processor = context.get("qwen_processor")
    if not processor:
        logger.error("Qwen processor not available in context. Was there an error during startup?")
        raise HTTPException(status_code=500, detail="Model processor is not available. Check server logs.")

    try:
        logger.info(f"Received Qwen generation request with prompt: {request_body.prompt[:100]}...") # Log first 100 chars
        # Call the Qwen processor
        response_data = processor.generate(
            prompt=request_body.prompt,
            max_new_tokens=request_body.max_new_tokens,
            enable_thinking=request_body.enable_thinking
        )
        logger.info("Successfully generated Qwen response.")
        return response_data # Qwen processor returns a dict with thinking_content and content
    except Exception as e:
        logger.error(f"An unexpected error occurred during Qwen processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# Example of how to run the server (for development)
# Use: uvicorn llama_api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    # Note: Running directly like this might not be ideal for production.
    # Use a process manager like Gunicorn with Uvicorn workers.
    # Also, model loading might take time, impacting the first request if not pre-loaded.
    uvicorn.run(app, host="0.0.0.0", port=8000)
