# main.py
import os
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
#import tiktoken
from transformers import AutoTokenizer # For Qwen chat template

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

# 요청 및 응답을 위한 Pydantic 모델 정의
# class SummarizeRequest(BaseModel):
#     text: str  # 요약할 텍스트
#     max_chunk_size: int = Field(default=1000, description="청크당 최대 토큰 수")  # 청크당 최대 토큰 수
#     summarization_template: str = Field(
#         default="Summarize the following text concisely while preserving key information:\n\n{text}",  # 요약 프롬프트 템플릿
#         description="Template for summarization prompt"
#     )
#     final_template: str = Field(
#         default="Create a comprehensive summary of these summaries, maintaining the key points and overall flow:\n\n{summaries}",  # 최종 요약 프롬프트 템플릿
#         description="Template for final summarization prompt"
#     )

# class SummarizeResponse(BaseModel):
#     chunk_summaries: List[str]  # 각 청크의 요약 목록
#     final_summary: str  # 최종 요약
#     chunk_count: int  # 청크 수
#     processing_time: float  # 처리 시간

class ChatMessage(BaseModel):
    role: str = Field(..., description="메시지 역할 (system, user, assistant)")
    content: str = Field(..., description="메시지 내용")

class LLMRequest(BaseModel):
    prompt: List[ChatMessage]  # LLM에 전달할 메시지 목록 (OpenChatML 형식)
    max_tokens: int = Field(default=512, description="생성할 최대 토큰 수")
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
                    max_num_batched_tokens=4096,  # 메모리에 따라 조정
                )
                engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine

# 의미론적 텍스트 청킹 함수
# def semantic_chunking(text: str, max_chunk_size: int) -> List[str]:
#     """
#     텍스트를 의미론적으로 청크로 나눕니다.
#     :param text: 나눌 텍스트
#     :param max_chunk_size: 청크당 최대 토큰 수
#     :return: 텍스트 청크 목록
#     """
#     # 텍스트를 토큰으로 변환
#     tokens = encoding.encode(text)
    
#     # 텍스트가 충분히 짧으면 단일 청크로 반환
#     if len(tokens) <= max_chunk_size:
#         return [text]
    
#     chunks = []  # 청크 목록
#     current_chunk_tokens = []  # 현재 청크의 토큰
#     current_chunk_size = 0  # 현재 청크의 크기
    
#     # 텍스트를 단락으로 분할
#     paragraphs = text.split("\n\n")
    
#     for paragraph in paragraphs:
#         # 빈 단락 건너뛰기
#         if not paragraph.strip():
#             continue
            
#         paragraph_tokens = encoding.encode(paragraph)  # 단락을 토큰으로 변환
#         paragraph_token_count = len(paragraph_tokens)  # 단락의 토큰 수
        
#         # 단일 단락이 너무 크면 문장으로 분할
#         if paragraph_token_count > max_chunk_size:
#             sentences = paragraph.replace(". ", ".\n").split("\n")  # 문장으로 분할
#             for sentence in sentences:
#                 if not sentence.strip():
#                     continue
                    
#                 sentence_tokens = encoding.encode(sentence)  # 문장을 토큰으로 변환
#                 sentence_token_count = len(sentence_tokens)  # 문장의 토큰 수
                
#                 # 이 문장을 추가하면 청크 크기를 초과하는 경우 새 청크 시작
#                 if current_chunk_size + sentence_token_count > max_chunk_size and current_chunk_size > 0:
#                     chunk_text = encoding.decode(current_chunk_tokens)
#                     chunks.append(chunk_text)
#                     current_chunk_tokens = []
#                     current_chunk_size = 0
                
#                 # 단일 문장이 너무 큰 경우 (드문 경우) 토큰 수로 분할
#                 if sentence_token_count > max_chunk_size:
#                     for i in range(0, sentence_token_count, max_chunk_size):
#                         sub_tokens = sentence_tokens[i:i + max_chunk_size]
#                         sub_text = encoding.decode(sub_tokens)
#                         chunks.append(sub_text)
#                 else:
#                     current_chunk_tokens.extend(sentence_tokens)
#                     current_chunk_size += sentence_token_count
#         else:
#             # 이 단락을 추가하면 청크 크기를 초과하는 경우 새 청크 시작
#             if current_chunk_size + paragraph_token_count > max_chunk_size and current_chunk_size > 0:
#                 chunk_text = encoding.decode(current_chunk_tokens)
#                 chunks.append(chunk_text)
#                 current_chunk_tokens = []
#                 current_chunk_size = 0
            
#             current_chunk_tokens.extend(paragraph_tokens)
#             current_chunk_size += paragraph_token_count
    
#     # 마지막 청크가 비어 있지 않으면 추가
#     if current_chunk_size > 0:
#         chunk_text = encoding.decode(current_chunk_tokens)
#         chunks.append(chunk_text)
    
#     return chunks

# LLM 추론 실행 함수
async def run_inference(messages: List[ChatMessage], max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
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
    
    # Use the formatted_prompt string with the vLLM engine
    result = await engine.generate(formatted_prompt, sampling_params)  # LLM 생성 실행
    response = result[0].outputs[0].text.strip()  # 결과에서 텍스트 추출
    return response

# # 문서 요약을 위한 API 엔드포인트
# @app.post("/api/summarize", response_model=SummarizeResponse)
# async def summarize_document(request: SummarizeRequest) -> SummarizeResponse:
#     """
#     문서를 요약합니다.
#     :param request: 요약 요청 객체
#     :return: 요약 응답 객체
#     """
#     start_time = asyncio.get_event_loop().time()  # 시작 시간 기록
    
#     # 1단계: 의미론적 청킹
#     chunks = semantic_chunking(request.text, request.max_chunk_size)
    
#     # 2단계: 각 청크를 병렬로 요약
#     summarization_tasks = []
#     for chunk in chunks:
#         prompt = request.summarization_template.format(text=chunk)  # 요약 프롬프트 생성
#         # 청크 길이의 1/3 또는 512 중 작은 값을 max_tokens로 설정
#         task = run_inference(prompt, max_tokens=min(len(encoding.encode(chunk)) // 3, 512), temperature=0.3, top_p=0.9)
#         summarization_tasks.append(task)
    
#     chunk_summaries = await asyncio.gather(*summarization_tasks)  # 모든 요약 작업 병렬 실행
    
#     # 3단계: 모든 청크 요약으로부터 최종 요약 생성
#     if len(chunk_summaries) == 1:
#         final_summary = chunk_summaries[0]  # 청크가 하나면 해당 요약이 최종 요약
#     else:
#         all_summaries = "\n\n".join([f"Summary {i+1}:\n{summary}" for i, summary in enumerate(chunk_summaries)])  # 모든 청크 요약 결합
#         final_prompt = request.final_template.format(summaries=all_summaries)  # 최종 요약 프롬프트 생성
#         final_summary = await run_inference(final_prompt, max_tokens=1024, temperature=0.3, top_p=0.9)  # 최종 요약 생성
    
#     end_time = asyncio.get_event_loop().time()  # 종료 시간 기록
#     processing_time = end_time - start_time  # 처리 시간 계산
    
#     return SummarizeResponse(
#         chunk_summaries=chunk_summaries,
#         final_summary=final_summary,
#         chunk_count=len(chunks),
#         processing_time=processing_time
#     )

# 직접 LLM 요청을 위한 API 엔드포인트
@app.post("/api/llm", response_model=LLMResponse)
async def llm_request(request: LLMRequest) -> LLMResponse:
    """
    직접 LLM 요청을 처리합니다.
    :param request: LLM 요청 객체
    :return: LLM 응답 객체
    """
    start_time = asyncio.get_event_loop().time()  # 시작 시간 기록
    
    response = await run_inference(
        request.prompt, 
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )  # LLM 추론 실행
    
    end_time = asyncio.get_event_loop().time()  # 종료 시간 기록
    processing_time = end_time - start_time  # 처리 시간 계산
    
    return LLMResponse(
        response=response,
        processing_time=processing_time
    )

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
