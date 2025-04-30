from pydantic import BaseModel, Field, HttpUrl
from typing import List, Literal, Union

class ImageContent(BaseModel):
    """Schema for image content within a message."""
    type: Literal["image"]
    url: HttpUrl # Use HttpUrl for basic URL validation

class TextContent(BaseModel):
    """Schema for text content within a message."""
    type: Literal["text"]
    text: str

class Message(BaseModel):
    """Schema for a single message in the conversation."""
    role: str # e.g., "user", "assistant"
    content: List[Union[ImageContent, TextContent]]

class LlamaRequest(BaseModel):
    """Schema for the main API request body."""
    messages: List[Message] = Field(..., description="A list of messages forming the conversation history.")

class QwenRequest(BaseModel):
    """Schema for the Qwen API request body."""
    messages: List[Message] = Field(..., description="A list of messages forming the conversation history for the Qwen model.")
    max_new_tokens: int = Field(32768, description="The maximum number of new tokens to generate.")
    enable_thinking: bool = Field(True, description="Whether to enable thinking mode for Qwen.")
