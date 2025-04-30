from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from llama_cpp import Llama

class Message(BaseModel):
    user: Optional[str] = Field(None, description="사용자 발화")
    assistant: Optional[str] = Field(None, description="어시스턴트 발화")

class GenerateRequest(BaseModel):
    messages: List[Message] = Field(..., description="대화 히스토리 목록")
    max_tokens: int = Field(128, description="생성할 최대 토큰 수")
    temperature: float = Field(0.7, description="샘플링 온도")
    top_p: float = Field(0.9, description="top-p 샘플링 비율")

app = FastAPI(title="GGUF Qwen3-8B Inference")

# 모델 로드 (GPU 사용, 레이어 수·스레드 수는 환경에 맞게 조절)
llm = Llama(
    model_path="/app/Qwen3-8B-Q4_K_M.gguf",
    n_ctx=32768,
    n_gpu_layers=30,   # 3080 기준, 필요시 조정
    n_threads=8
)

@app.post("/generate")
def generate(req: GenerateRequest):
    # 대화 히스토리에서 prompt 문자열 생성
    prompt_lines: List[str] = []
    for msg in req.messages:
        if msg.user is not None:
            prompt_lines.append(f"User: {msg.user}")
        if msg.assistant is not None:
            prompt_lines.append(f"Assistant: {msg.assistant}")
    # 다음 어시스턴트 응답을 예측하기 위한 마커
    prompt_lines.append("Assistant:")
    prompt_str = "\n".join(prompt_lines)

    try:
        resp = llm(
            prompt=prompt_str,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p
        )
        text = resp.get("choices", [{}])[0].get("text", "")
        return {"text": text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 추론 중 오류 발생: {e}")