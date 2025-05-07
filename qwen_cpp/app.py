import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from llama_cpp import Llama

# 메시지 모델 정의 (System 필드 포함)
class Message(BaseModel):
    system: Optional[str] = Field(None, description="시스템 발화")
    user: Optional[str] = Field(None, description="사용자 발화")
    assistant: Optional[str] = Field(None, description="어시스턴트 발화")

# 요청 모델 정의
class GenerateRequest(BaseModel):
    messages: List[Message] = Field(..., description="대화 히스토리 목록")
    max_tokens: int = Field(128, description="생성할 최대 토큰 수")
    temperature: float = Field(0.7, description="샘플링 온도")
    top_p: float = Field(0.9, description="top-p 샘플링 비율")

# 응답 모델 정의
class GenerateResponse(BaseModel):
    text: str = Field(..., description="생성된 텍스트")

app = FastAPI(title="GGUF Qwen3-8B Inference (Sequential Processing)")

# 모델 로드 (GPU 사용, 레이어 수·스레드 수는 환경에 맞게 조정)
try:
    llm = Llama(
        model_path="/app/models/Qwen3-8B-Q4_0.gguf", # 모델 경로 확인 필요
        n_ctx=32768,
        n_gpu_layers=30,   # 3080 기준, 필요시 조정
        n_threads=8,
    )
except Exception as e:
    # 모델 로드 실패 시 에러 처리 (예: 로깅 후 종료 또는 llm=None 설정)
    print(f"FATAL: LLM 모델 로드 실패: {e}") # 기본 print 사용
    llm = None

# 비동기 큐를 사용하여 요청을 순차적으로 관리
request_queue = asyncio.Queue(maxsize=100) # 큐 크기 제한 (선택 사항)

# 백그라운드에서 큐의 요청을 순차적으로 처리하는 함수
async def process_request_from_queue():
    if llm is None:
        print("ERROR: LLM 모델이 로드되지 않아 요청 처리를 시작할 수 없습니다.")
        return # 모델 로드 실패 시 처리 중단

    print("INFO: 백그라운드 요청 처리 작업 시작.")
    while True:
        try:
            # 큐에서 요청 데이터와 완료 신호용 Event 객체를 가져옴
            req_dict, completion_event = await request_queue.get()
            print(f"INFO: 큐에서 요청 처리 시작: {id(req_dict)}")

            try:
                # 대화 히스토리에서 prompt 문자열 생성
                prompt_lines: List[str] = []
                for msg_data in req_dict['messages']: # Pydantic 모델 대신 dict 사용
                    if msg_data.get('system') is not None:
                        prompt_lines.append(f"System: {msg_data['system']}")
                    if msg_data.get('user') is not None:
                        prompt_lines.append(f"User: {msg_data['user']}")
                    if msg_data.get('assistant') is not None:
                        prompt_lines.append(f"Assistant: {msg_data['assistant']}")
                prompt_lines.append("Assistant:")
                prompt_str = "\n".join(prompt_lines)

                # 모델 추론 실행 (동기 함수이므로 직접 호출)
                print(f"INFO: LLM 추론 시작: {id(req_dict)}")
                resp = llm(
                    prompt=prompt_str,
                    max_tokens=req_dict['max_tokens'],
                    temperature=req_dict['temperature'],
                    top_p=req_dict['top_p']
                )
                print(f"INFO: LLM 추론 완료: {id(req_dict)}")

                text = resp.get("choices", [{}])[0].get("text", "")
                req_dict['response'] = text.strip() # 결과를 req_dict에 저장
                req_dict['error'] = None

            except Exception as e:
                print(f"ERROR: 모델 추론 중 오류 발생 ({id(req_dict)}): {e}")
                req_dict['response'] = None
                req_dict['error'] = f"모델 추론 중 오류 발생: {e}"

            finally:
                # 처리 완료 신호 전송
                completion_event.set()
                request_queue.task_done()
                print(f"INFO: 요청 처리 완료 및 신호 전송: {id(req_dict)}")

        except asyncio.CancelledError:
            print("INFO: 백그라운드 요청 처리 작업 취소됨.")
            break
        except Exception as e:
            # 큐 처리 루프 자체의 예외 처리
            print(f"ERROR: 큐 처리 루프에서 예외 발생: {e}")
            # 잠시 대기 후 계속 시도 (선택적)
            await asyncio.sleep(1)


# 애플리케이션 시작 시 백그라운드 작업 시작
@app.on_event("startup")
async def startup_event():
    # 백그라운드 큐 처리 작업 시작
    asyncio.create_task(process_request_from_queue())
    print("INFO: FastAPI 애플리케이션 시작 및 백그라운드 작업 준비 완료.")

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM 모델이 로드되지 않았습니다. 서비스를 사용할 수 없습니다.")

    # 요청 데이터를 딕셔너리로 변환
    req_dict = req.model_dump()

    # 이 요청의 완료를 기다릴 Event 객체 생성
    completion_event = asyncio.Event()

    try:
        # 요청 데이터와 Event 객체를 큐에 넣음
        await request_queue.put((req_dict, completion_event))
        print(f"INFO: 요청을 큐에 추가: {id(req_dict)}")

        # 백그라운드 작업이 완료되어 Event가 설정될 때까지 대기
        print(f"INFO: 요청 처리 대기 시작: {id(req_dict)}")
        await completion_event.wait()
        print(f"INFO: 요청 처리 대기 완료: {id(req_dict)}")

        # 결과 확인 및 반환
        if req_dict.get('error'):
            raise HTTPException(status_code=500, detail=req_dict['error'])
        elif req_dict.get('response') is not None:
            return GenerateResponse(text=req_dict['response'])
        else:
            # 예상치 못한 상황
            print(f"ERROR: 처리 완료 후 응답 데이터 누락: {id(req_dict)}")
            raise HTTPException(status_code=500, detail="모델 처리 완료 후 응답을 가져오는 데 실패했습니다.")

    except asyncio.QueueFull:
        print("WARNING: 요청 큐가 가득 찼습니다.")
        raise HTTPException(status_code=429, detail="현재 서버가 처리할 수 있는 요청 수를 초과했습니다. 잠시 후 다시 시도해주세요.")
    except Exception as e:
        # /generate 핸들러 자체의 예외 처리
        print(f"ERROR: /generate 엔드포인트 처리 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"요청 처리 중 서버 내부 오류 발생: {e}")
