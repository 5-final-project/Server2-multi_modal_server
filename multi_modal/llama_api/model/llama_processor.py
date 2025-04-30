from transformers import AutoTokenizer, Llama4ForConditionalGeneration
import torch
from typing import List, Dict, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaModelProcessor:
    """Handles loading and interacting with the Llama4 model."""

    def __init__(self, model_id: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.token = ""
        self._load_model()

    def _load_model(self):
        """Loads the processor and model."""
        try:
            logger.info(f"Loading processor for model: {self.model_id}")
            # Check if CUDA is available and set device accordingly
            if torch.cuda.is_available():
                # device_map = "auto"
                # torch_dtype = torch.bfloat16
                logger.info("CUDA available. Using 'auto' device map and bfloat16.")
            else:
                # device_map = None # Let transformers decide, likely CPU
                # torch_dtype = torch.float32 # Use float32 for CPU
                logger.info("CUDA not available. Using CPU and float32.")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.token, cache_dir="/app/hf_cache")
            logger.info(f"Loading model: {self.model_id}")
            
            # Load the model with FP8 quantization settings
            self.model = Llama4ForConditionalGeneration.from_pretrained(
                self.model_id,
                token=self.token,
                cache_dir="/app/hf_cache",
                # tp_plan="auto",
                torch_dtype="auto",
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model {self.model_id}: {e}", exc_info=True)
            raise  # Re-raise the exception to handle it at a higher level

    def process_messages(self, messages: List[Dict[str, Any]], max_new_tokens: int = 256) -> str:
        """
        Processes a list of messages using the Llama4 model.

        Args:
            messages: A list of message dictionaries, conforming to the expected format.
            max_new_tokens: The maximum number of new tokens to generate.

        Returns:
            The generated response string.
        """
        if not self.tokenizer or not self.model:
            logger.error("Model or processor not loaded. Cannot process messages.")
            # Handle this case appropriately, maybe raise an error or return a specific message
            raise RuntimeError("Model or processor failed to load.")

        try:
            logger.info("Applying chat template and tokenizing messages.")
            # Convert Pydantic models back to dicts if necessary, though FastAPI often handles this
            # Ensure messages format matches what processor expects
            formatted_messages = []
            for msg in messages:
                content_list = []
                for item in msg['content']:
                    if item['type'] == 'image':
                        content_list.append({"type": "image", "url": str(item['url'])}) # Ensure URL is string
                    elif item['type'] == 'text':
                        content_list.append({"type": "text", "text": item['text']})
                formatted_messages.append({"role": msg['role'], "content": content_list})


            inputs = self.tokenizer.apply_chat_template(
                formatted_messages, # Use the formatted list
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            logger.info(f"Generating response with max_new_tokens={max_new_tokens}.")
            outputs = self.model.generate(
                **inputs.to(self.model.device),
                max_new_tokens=max_new_tokens,
            )

            logger.info("Decoding generated response.")
            # Decode only the newly generated tokens
            response = self.tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
            logger.info("Response generated successfully.")
            return response.strip() # Strip leading/trailing whitespace

        except Exception as e:
            logger.error(f"Error during message processing: {e}", exc_info=True)
            # Re-raise or return an error message
            raise

# Optional: Pre-load the model when the module is imported
# This can speed up the first request but increases startup time and memory usage.
# Consider using FastAPI's lifespan events for more controlled loading/unloading.
# llama_processor_instance = LlamaModelProcessor()