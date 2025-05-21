from collections.abc import AsyncGenerator
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import uuid
import logging
from contextlib import asynccontextmanager
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 설정
MODEL_ID = "colli98/qwen3-1.7B-ko-summary-finetuned"

# Pydantic 모델 정의
class ChatMessage(BaseModel):
    role: str
    content: str

class LLMRequest(BaseModel):
    prompt: List[ChatMessage]
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

class LLMResponse(BaseModel):
    response: str
    processing_time: float

# 전역 변수
engine: Optional[AsyncLLMEngine] = None
tokenizer = None
engine_lock = asyncio.Lock()
MAX_CONCURRENT_REQUESTS = 10
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def initialize_engine():
    """엔진 초기화 함수"""
    global engine, tokenizer
    
    try:
        logger.info(f"토크나이저 로딩 중: {MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        logger.info(f"vLLM 엔진 초기화 중: {MODEL_ID}")
        args = AsyncEngineArgs(
            model=MODEL_ID,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,  # 0.95에서 0.8로 낮춤
            max_num_batched_tokens=2048,  # 1024에서 2048로 증가
            max_model_len=4096,  # 2048에서 4096으로 증가
            enforce_eager=True,  # 메모리 안정성을 위해 추가
            disable_log_stats=False,
        )
        engine = AsyncLLMEngine.from_engine_args(args)
        logger.info("엔진 초기화 완료")
        
    except Exception as e:
        logger.error(f"엔진 초기화 실패: {e}")
        raise

async def get_engine():
    """엔진 인스턴스 반환"""
    global engine
    if engine is None:
        async with engine_lock:
            if engine is None:
                await initialize_engine()
    return engine

async def run_inference(
    messages: List[ChatMessage],
    max_tokens: int,
    temperature: float,
    top_p: float
) -> AsyncGenerator[str, None]:
    """추론 실행 함수"""
    engine_instance = None
    req_id = None
    
    try:
        # 동시 요청 수 제한
        async with request_semaphore:
            engine_instance = await get_engine()
            
            # 채팅 템플릿 적용
            chat_input = [{"role": m.role, "content": m.content} for m in messages]
            prompt = tokenizer.apply_chat_template(
                chat_input,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 샘플링 파라미터 설정
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                skip_special_tokens=True,
                stop_token_ids=None,
            )
            
            req_id = str(uuid.uuid4())
            logger.info(f"추론 시작 - 요청 ID: {req_id}")
            
            # 스트리밍 생성
            accumulated_text = ""
            async for request_output in engine_instance.generate(prompt, params, req_id):
                if request_output.outputs:
                    for output in request_output.outputs:
                        # 새로 생성된 텍스트만 추출
                        current_text = output.text
                        if len(current_text) > len(accumulated_text):
                            new_text = current_text[len(accumulated_text):]
                            accumulated_text = current_text
                            if new_text.strip():  # 빈 문자열이 아닌 경우만 yield
                                yield new_text
                                
                        # 완료 상태 확인
                        if output.finish_reason is not None:
                            logger.info(f"생성 완료 - 요청 ID: {req_id}, 이유: {output.finish_reason}")
                            break
                            
    except asyncio.CancelledError:
        logger.warning(f"요청 취소됨 - 요청 ID: {req_id}")
        # 엔진에서 요청 중단 시도
        if engine_instance and req_id:
            try:
                await engine_instance.abort(req_id)
            except Exception as abort_error:
                logger.error(f"요청 중단 실패: {abort_error}")
        raise
        
    except Exception as e:
        logger.error(f"추론 중 오류 발생 - 요청 ID: {req_id}, 오류: {e}")
        # 엔진에서 요청 중단 시도
        if engine_instance and req_id:
            try:
                await engine_instance.abort(req_id)
            except Exception as abort_error:
                logger.error(f"요청 중단 실패: {abort_error}")
        raise HTTPException(status_code=500, detail=f"추론 실행 중 오류: {str(e)}")

# 생명주기 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시
    logger.info("애플리케이션 시작")
    await initialize_engine()
    yield
    # 종료 시
    logger.info("애플리케이션 종료")

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="vLLM Document Processing API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 스트리밍 응답 엔드포인트
@app.post("/api/generate-stream")
async def generate_stream(request: LLMRequest):
    """스트리밍 생성 엔드포인트"""
    try:
        async def streamer():
            try:
                async for chunk in run_inference(
                    request.prompt,
                    request.max_tokens,
                    request.temperature,
                    request.top_p
                ):
                    # JSON 형태로 chunk 전송
                    yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
                
                # 완료 신호
                yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
                
            except asyncio.CancelledError:
                logger.warning("스트리밍 클라이언트 연결 끊김")
                yield f"data: {json.dumps({'error': 'Connection cancelled', 'done': True})}\n\n"
            except Exception as e:
                logger.error(f"스트리밍 중 오류: {e}")
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
        
        return StreamingResponse(
            streamer(), 
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # nginx 버퍼링 비활성화
            }
        )
        
    except Exception as e:
        logger.error(f"스트리밍 엔드포인트 오류: {e}")
        raise HTTPException(status_code=500, detail=f"스트리밍 시작 실패: {str(e)}")

# 동기식 응답 엔드포인트
@app.post("/api/generate", response_model=LLMResponse)
async def llm_request(request: LLMRequest) -> LLMResponse:
    """동기식 생성 엔드포인트"""
    try:
        start_time = asyncio.get_event_loop().time()
        full_response = ""
        
        async for chunk in run_inference(
            request.prompt,
            request.max_tokens,
            request.temperature,
            request.top_p
        ):
            full_response += chunk
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return LLMResponse(
            response=full_response.strip(),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"동기식 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"텍스트 생성 실패: {str(e)}")

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    try:
        engine_status = "ready" if engine is not None else "not_ready"
        return {
            "status": "healthy",
            "model": MODEL_ID,
            "engine_status": engine_status,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS
        }
    except Exception as e:
        logger.error(f"헬스체크 오류: {e}")
        raise HTTPException(status_code=500, detail="서비스 상태 확인 실패")

# 엔진 상태 확인 엔드포인트
@app.get("/status")
async def get_status():
    """엔진 상태 확인"""
    try:
        if engine is not None:
            return {
                "engine_ready": True,
                "model": MODEL_ID,
                "active_requests": MAX_CONCURRENT_REQUESTS - request_semaphore._value
            }
        else:
            return {
                "engine_ready": False,
                "model": MODEL_ID,
                "active_requests": 0
            }
    except Exception as e:
        logger.error(f"상태 확인 오류: {e}")
        raise HTTPException(status_code=500, detail="상태 확인 실패")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # 프로덕션에서는 reload=False
        workers=1,     # vLLM은 단일 워커 권장
        loop="asyncio",
        log_level="info"
    )