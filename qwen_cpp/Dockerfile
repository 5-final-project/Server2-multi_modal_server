# 1) 베이스 이미지: CUDA 지원 Ubuntu
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 2) 시스템 의존성 + llama.cpp 빌드
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git build-essential cmake libopenblas-dev libcurl4-openssl-dev wget && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    make clean && \
    make LLAMA_CUBLAS=1 -j$(nproc)

# 3) Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) 모델 파일 다운로드
RUN wget --progress=dot:giga \
    "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf?download=true" \
    -O /app/Qwen3-8B-Q4_K_M.gguf

# 5) API 코드 복사
COPY api.py /app/api.py

# 6) 8000 포트 노출 및 실행
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
