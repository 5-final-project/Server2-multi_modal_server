from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Literal
import httpx

app = FastAPI()

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9

@app.post("/chat")
async def chat(request: ChatRequest):
    history = request.messages
    prompt = ""
    
    for msg in history:
        if msg.role == "system":
            prompt += f"<|system|>\n{msg.content.strip()}\n"
        elif msg.role == "user":
            prompt += f"<|user|>\n{msg.content.strip()}\n"
        elif msg.role == "assistant":
            prompt += f"<|assistant|>\n{msg.content.strip()}\n"

    prompt += "<|user|>\n"  # for next user message
    
    async with httpx.AsyncClient() as client:
        res = await client.post(
            "http://host.docker.internal:8000/generate",
            json={
                "prompt": prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": False
            }
        )
        output = res.json()
        return {
            "response": output.get("text", ""),
            "raw": output
        }
