# 1) 베이스 이미지: 사전 빌드된 llama-cpp-python (GPU/CUDA 지원 포함)
FROM ghcr.io/abetlen/llama-cpp-python:latest

# 2) 작업 디렉토리 설정
WORKDIR /app

# 3) 시스템 유틸리티 및 CUDA 툴킷 설치 (nvcc 포함)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# 3-1) CUDA 툴킷 설치
#  - CUDA 12.1 버전 설치
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \ 
    dpkg -i cuda-keyring_1.1-1_all.deb \ 
    apt-get install -y cuda-toolkit-12-1

RUN export CUDA_HOME=/usr/local/cuda \
    export PATH=$CUDA_HOME/bin:$PATH \
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3-2) CUDA 툴킷 설치 확인
RUN nvcc --version

# # 4) CUDA 환경 변수 설정 (nvcc 경로 및 CMake용)
# ENV CUDAToolkit_ROOT=/usr \
#     CUDACXX=/usr/bin/nvcc

# 5) Python 의존성 설치
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# 6) 애플리케이션 코드 복사
COPY app.py .

# 7) 모델 경로 환경변수 설정 (실행 시 모델 마운트)
ENV MODEL_PATH=/app/models/Qwen3-8B-Q4_K_M.gguf

# 9) llama-cpp-python 사전 제작 휠 설치
#  - GPU 지원 (CUDA 12.1) 휠 사용
RUN git clone --recursive https://github.com/abetlen/llama-cpp-python.git\
    cd llama-cpp-python \
    git submodule update --remote vendor/llama.cpp \
    FORCE_CMAKE=1 \
    CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_CUDA_F16=on -DLLAMA_CURL=OFF" \
    pip install . --upgrade --force-reinstall --no-cache-dir

RUN cd ..

# 8) 포트 노출 및 실행 커맨드
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# 사용 예시:
# docker build -t my-llama-app .
# docker run --gpus all \
#   -v /mnt/d/team5/multi_modal_server/models:/app/models \
#   -p 8000:8000 my-llama-app
