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
