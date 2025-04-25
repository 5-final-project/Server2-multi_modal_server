from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

# Import schema and model processor
from .schemas.request import LlamaRequest
from .model.llama_processor import LlamaModelProcessor

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
    logger.info("Application startup: Loading Llama model...")
    try:
        # Instantiate and store the model processor in the context dictionary
        context["llama_processor"] = LlamaModelProcessor()
        logger.info("Llama model loaded successfully.")
        yield # Application runs here
    except Exception as e:
        logger.error(f"Fatal error during model loading: {e}", exc_info=True)
        # Optionally, prevent the app from starting or handle differently
        # For now, we let it yield, but the processor might be None or raise errors later
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
    a generated text response from the Llama4 model.
    """
    llama_processor = context.get("llama_processor")
    if not llama_processor:
        logger.error("Llama processor not available in context. Was there an error during startup?")
        raise HTTPException(status_code=500, detail="Model processor is not available. Check server logs.")

    try:
        # Convert Pydantic models to dictionaries expected by the processor
        # FastAPI might do this automatically, but being explicit can prevent issues
        messages_dict_list = [message.model_dump() for message in request_body.messages]

        logger.info(f"Received request with {len(messages_dict_list)} messages.")
        # Call the model processor
        response_text = llama_processor.process_messages(messages_dict_list)
        logger.info("Successfully generated response.")
        return {"response": response_text}
    except RuntimeError as e: # Catch errors specifically from the processor
        logger.error(f"Runtime error during processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model processing error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# Example of how to run the server (for development)
# Use: uvicorn llama_api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    # Note: Running directly like this might not be ideal for production.
    # Use a process manager like Gunicorn with Uvicorn workers.
    # Also, model loading might take time, impacting the first request if not pre-loaded.
    uvicorn.run(app, host="0.0.0.0", port=8000)
