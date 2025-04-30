from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QwenProcessor:
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        """
        Initializes the QwenProcessor with the specified model.
        """
        print(f"Loading Qwen model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir="/app/hf_cache")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/app/hf_cache"
        )
        print("Qwen model loaded successfully.")

    def generate(self, messages: list[dict], max_new_tokens: int = 32768, enable_thinking: bool = True):
        """
        Generates text using the Qwen model based on a list of messages.
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate text
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Parse thinking content if enabled and present
        thinking_content = ""
        content = ""
        try:
            # rindex finding 151668 (</think>) - Specific token ID for Qwen's thinking tag end
            think_token_id = 151668 # Assuming this is the correct ID for Qwen
            index = len(output_ids) - output_ids[::-1].index(think_token_id)
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        except ValueError:
            # If the thinking end token is not found, decode the whole output as content
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            thinking_content = "" # No thinking content found

        return {
            "thinking_content": thinking_content,
            "content": content
        }

# Example usage (optional, for testing)
if __name__ == "__main__":
    processor = QwenProcessor()
    result = processor.generate("Give me a short introduction to large language models.")
    print("Thinking Content:", result["thinking_content"])
    print("Content:", result["content"])
