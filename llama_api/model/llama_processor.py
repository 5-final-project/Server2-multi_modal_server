from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch
from typing import List, Dict, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaModelProcessor:
    """Handles loading and interacting with the Llama4 model."""

    def __init__(self, model_id: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"):
        self.model_id = model_id
        self.processor = None
        self.model = None
        self.token = None
        self._load_model()

    def _load_model(self):
        """Loads the processor and model."""
        try:
            logger.info(f"Loading processor for model: {self.model_id}")
            # Check if CUDA is available and set device accordingly
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.bfloat16
                logger.info("CUDA available. Using 'auto' device map and bfloat16.")
            else:
                device_map = None # Let transformers decide, likely CPU
                torch_dtype = torch.float32 # Use float32 for CPU
                logger.info("CUDA not available. Using CPU and float32.")

            self.processor = AutoProcessor.from_pretrained(self.model_id,token=self.token)
            logger.info(f"Loading model: {self.model_id}")
            self.model = Llama4ForConditionalGeneration.from_pretrained(
                self.model_id,
                token=self.token,
                attn_implementation="flex_attention", # Consider making this conditional if not always supported
                device_map=device_map,
                torch_dtype=torch_dtype,
                # Add low_cpu_mem_usage for potentially large models if needed
                # low_cpu_mem_usage=True if device_map == "auto" else False
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model {self.model_id}: {e}", exc_info=True)
            # Depending on the desired behavior, you might want to raise the exception
            # or handle it gracefully (e.g., set model/processor to None and check later)
            raise

    def process_messages(self, messages: List[Dict[str, Any]], max_new_tokens: int = 256) -> str:
        """
        Processes a list of messages using the Llama4 model.

        Args:
            messages: A list of message dictionaries, conforming to the expected format.
            max_new_tokens: The maximum number of new tokens to generate.

        Returns:
            The generated response string.
        """
        if not self.processor or not self.model:
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


            inputs = self.processor.apply_chat_template(
                formatted_messages, # Use the formatted list
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            logger.info(f"Generating response with max_new_tokens={max_new_tokens}.")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

            logger.info("Decoding generated response.")
            # Decode only the newly generated tokens
            response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
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