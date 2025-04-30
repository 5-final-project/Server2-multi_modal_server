from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch
from typing import List, Dict, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperCLOVAXModelProcessor:
    """Handles loading and interacting with the HyperCLOVAX model."""

    def __init__(self, model_id: str = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"):
        self.model_id = model_id
        self.tokenizer = None
        self.preprocessor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the processor and model."""
        try:
            logger.info(f"Loading components for model: {self.model_id}")
            # Check if CUDA is available and set device accordingly
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                logger.info("CUDA available. Using GPU.")
            else:
                logger.info("CUDA not available. Using CPU.")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir="/app/hf_cache")
            self.preprocessor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, cache_dir="/app/hf_cache")
            
            logger.info(f"Loading model: {self.model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir="/app/hf_cache"
            ).to(device=device)
            
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model {self.model_id}: {e}", exc_info=True)
            raise  # Re-raise the exception to handle it at a higher level

    def process_messages(self, messages: List[Dict[str, Any]], max_new_tokens: int = 256) -> str:
        """
        Processes a list of messages using the HyperCLOVAX model.

        Args:
            messages: A list of message dictionaries, conforming to the expected format.
            max_new_tokens: The maximum number of new tokens to generate.

        Returns:
            The generated response string.
        """
        if not self.tokenizer or not self.model or not self.preprocessor:
            logger.error("Model, tokenizer, or preprocessor not loaded. Cannot process messages.")
            raise RuntimeError("Model or processor failed to load.")

        try:
            logger.info("Processing messages and handling multimedia content.")
            
            # Format the messages in the format expected by HyperCLOVAX
            formatted_messages = []
            has_images = False
            
            for msg in messages:
                role = msg['role']
                
                # Handle different content formats
                if isinstance(msg.get('content'), list):
                    # For handling mixed content (text and images)
                    content_items = []
                    for item in msg['content']:
                        if item.get('type') == 'text':
                            content_items.append({
                                "type": "text", 
                                "text": item['text']
                            })
                        elif item.get('type') == 'image':
                            has_images = True
                            content_items.append({
                                "type": "image",
                                "filename": f"image_{len(content_items)}.jpg",
                                "image": item['url']
                            })
                    
                    if content_items:
                        # If multiple types, use the content items directly
                        if len(content_items) > 1 or any(item["type"] != "text" for item in content_items):
                            formatted_messages.append({"role": role, "content": content_items})
                        # If only text, combine into a single text content
                        else:
                            formatted_messages.append({
                                "role": role, 
                                "content": {"type": "text", "text": content_items[0]["text"]}
                            })
                
                elif isinstance(msg.get('content'), str):
                    # For simple string content
                    formatted_messages.append({
                        "role": role, 
                        "content": {"type": "text", "text": msg['content']}
                    })
                
                elif isinstance(msg.get('content'), dict):
                    # Already formatted content
                    formatted_messages.append({"role": role, "content": msg['content']})
            
            # Get the device from the model
            device = next(self.model.parameters()).device
            
            # Process differently based on whether there are images
            if has_images:
                logger.info("Pre-processing images or videos.")
                new_chat, all_images, is_video_list = self.preprocessor.load_images_videos(formatted_messages)
                
                logger.info("Applying chat template and tokenizing messages.")
                input_ids = self.tokenizer.apply_chat_template(
                    new_chat, 
                    return_tensors="pt", 
                    tokenize=True,
                    add_generation_prompt=True
                )
                
                input_ids = input_ids.to(device)
                
                # Only process images if there are any
                if all_images and len(all_images) > 0:
                    preprocessed = self.preprocessor(all_images, is_video_list=is_video_list)
                    generation_kwargs = preprocessed
                else:
                    generation_kwargs = {}
            else:
                # Text-only processing
                logger.info("Processing text-only messages.")
                input_ids = self.tokenizer.apply_chat_template(
                    formatted_messages, 
                    return_tensors="pt", 
                    tokenize=True,
                    add_generation_prompt=True
                )
                
                input_ids = input_ids.to(device)
                generation_kwargs = {}
            
            logger.info(f"Generating response with max_new_tokens={max_new_tokens}.")
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.6,
                temperature=0.5,
                repetition_penalty=1.0,
                **generation_kwargs
            )
            
            logger.info("Decoding generated response.")
            # Decode the entire output
            full_response = self.tokenizer.batch_decode(output_ids)[0]
            
            # Extract only the assistant's response
            # This is a simplified approach - might need adjustment based on exact output format
            assistant_parts = full_response.split("<assistant>")
            if len(assistant_parts) > 1:
                assistant_response = assistant_parts[-1].split("</s>")[0].strip()
            else:
                # If <assistant> tag not found, return the generated text after the input
                input_text = self.tokenizer.batch_decode(input_ids)[0]
                assistant_response = full_response[len(input_text):].split("</s>")[0].strip()
            
            logger.info("Response generated successfully.")
            return assistant_response

        except Exception as e:
            logger.error(f"Error during message processing: {e}", exc_info=True)
            raise

# Optional: Pre-load the model when the module is imported
# hyperclovax_processor_instance = HyperCLOVAXModelProcessor()