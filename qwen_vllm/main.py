# main.py
from collections.abc import AsyncGenerator
import os
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
#import tiktoken
from transformers import AutoTokenizer # For Qwen chat template
import uuid

# FastAPI 앱 초기화
app = FastAPI(
    title="vLLM Document Processing API",  # API 제목
    description="API for document summarization and LLM requests using vLLM",  # API 설명
    version="1.0.0"  # API 버전
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용
    allow_credentials=True,  # 자격 증명 허용
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 모델 설정
MODEL_ID = "colli98/qwen3-1.7B-ko-summary-finetuned"  # 선호하는 모델로 변경 가능
MAX_TOKENS = 2048  # 최대 토큰 수
TEMPERATURE = 0.7  # 샘플링 온도
TOP_P = 0.9  # Top-p 샘플링

# 텍스트 청킹을 위한 토크나이저 초기화
#encoding = tiktoken.get_encoding("cl100k_base")  # OpenAI의 토크나이저 사용, 선호하는 다른 토크나이저 사용 가능

# Qwen tokenizer for chat template
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

class ChatMessage(BaseModel):
    role: str = Field(..., description="메시지 역할 (system, user, assistant)")
    content: str = Field(..., description="메시지 내용")

class LLMRequest(BaseModel):
    prompt: List[ChatMessage]  # LLM에 전달할 메시지 목록 (OpenChatML 형식)
    max_tokens: int = Field(default=1024, description="생성할 최대 토큰 수")
    temperature: float = Field(default=0.7, description="샘플링 온도")
    top_p: float = Field(default=0.9, description="Top-p 핵 샘플링")

class LLMResponse(BaseModel):
    response: str  # LLM 응답
    processing_time: float  # 처리 시간

# 엔진 상태 관리
engine = None  # LLM 엔진 인스턴스
engine_semaphore = asyncio.Semaphore(1)  # 엔진 초기화를 위한 세마포어

async def get_engine():
    """
    LLM 엔진을 비동기적으로 가져오거나 초기화합니다.
    """
    global engine
    if engine is None:
        async with engine_semaphore:
            if engine is None:  # 경쟁 상태를 피하기 위해 다시 확인
                engine_args = AsyncEngineArgs(
                    model=MODEL_ID,  # 사용할 모델 ID
                    tensor_parallel_size=1,  # GPU 설정에 따라 조정
                    gpu_memory_utilization=0.95,  # GPU 메모리 사용률
                    max_num_batched_tokens=1024,  # 메모리에 따라 조정
                    max_model_len=2048,
                )
                engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine


# LLM 추론 실행 함수
async def run_inference(messages: List[ChatMessage], max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) ->  AsyncGenerator[str, None]:
    """
    LLM 추론을 실행합니다.
    :param messages: LLM에 전달할 메시지 목록 (OpenChatML 형식)
    :param max_tokens: 생성할 최대 토큰 수
    :param temperature: 샘플링 온도
    :param top_p: Top-p 샘플링
    :return: LLM 응답 텍스트
    """
    engine = await get_engine()  # LLM 엔진 가져오기

    # Convert List[ChatMessage] to the format expected by tokenizer.apply_chat_template
    # which is a list of dictionaries, e.g., [{"role": "system", "content": "You are a helpful assistant."}, ...]
    chat_input = [{"role": msg.role, "content": msg.content} for msg in messages]

    # Apply the chat template to get the formatted string prompt
    # For Qwen, add_generation_prompt=True is important for the model to generate a response.
    formatted_prompt = tokenizer.apply_chat_template(
        chat_input,
        tokenize=False,
        add_generation_prompt=True
    )
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    request_id = str(uuid.uuid4())
    # Use the formatted_prompt string with the vLLM engine
    async for result in engine.generate(formatted_prompt, request_id, sampling_params):
        for output in result.outputs:
            text = output.text
            if text:
                yield text

# 직접 LLM 요청을 위한 API 엔드포인트
@app.post("/api/generate", response_model=LLMResponse)
async def llm_request(request: LLMRequest) -> LLMResponse:
    """
    직접 LLM 요청을 처리합니다.
    :param request: LLM 요청 객체
    :return: LLM 응답 객체
    """
    start_time = asyncio.get_event_loop().time()  # 시작 시간 기록
    
    # response = await run_inference(
    #     request.prompt, 
    #     max_tokens=request.max_tokens,
    #     temperature=request.temperature,
    #     top_p=request.top_p
    # )  # LLM 추론 실행

    async def stream_response():
        result = ""
        async for result in run_inference(
            request.prompt,
            max_tokens=request.max_tokens,
        temperature=request.temperature
        ):
            for response in result:
                # 클라이언트로 텍스트를 점진적으로 전송
                result += response
        return result
    
    response = await stream_response()
    end_time = asyncio.get_event_loop().time()  # 종료 시간 기록
    processing_time = end_time - start_time  # 처리 시간 계산
    
    return LLMResponse(
        response=response,
        processing_time=processing_time
    )

@app.post("/api/generate-stream", response_model=LLMResponse)
async def generate_stream(request: LLMRequest) -> LLMResponse:

    # 실제 엔진 호출
    async def stream_response():
        result = ""
        async for result in run_inference(
            request.prompt,
            max_tokens=request.max_tokens,
        temperature=request.temperature
        ):
            for response in result:
                # 클라이언트로 텍스트를 점진적으로 전송
                yield response

    return StreamingResponse(stream_response(), media_type="text/plain")

# 상태 확인 엔드포인트
@app.get("/health")
async def health_check():
    """
    API 상태를 확인합니다.
    """
    return {"status": "healthy", "model": MODEL_ID}  # 건강 상태 및 모델 ID 반환

# 서버 시작
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)  # Uvicorn 서버 실행