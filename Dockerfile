# Use official Python image with CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ========================================
# CRITICAL: Set cuDNN library path for CTranslate2
# ========================================
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda

# Copy application code first
COPY . .

# ==============================
# Install dependencies in EXACT ORDER (like your Colab)
# ==============================

# 1. PyTorch with CUDA
RUN pip install --no-cache-dir --upgrade \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Backend + API
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    bcrypt \
    httpx \
    sseclient-py \
    email-validator \
    alembic

# 3. Database
RUN pip install --no-cache-dir \
    psycopg2-binary \
    "psycopg[binary,pool]>=3.1.0" \
    sqlalchemy \
    pgvector \
    langgraph-checkpoint-postgres

# 4. AI / ML (with specific versions)
RUN pip install --no-cache-dir \
    faster-whisper \
    sentence-transformers

# 5. Audio Processing
RUN pip install --no-cache-dir \
    librosa \
    opensmile \
    soundfile \
    praat-parselmouth \
    pydub

# 6. LangChain / LangGraph
RUN pip install --no-cache-dir \
    langgraph \
    langchain \
    langchain-google-genai

# 7. Utilities
RUN pip install --no-cache-dir \
    python-dotenv \
    pydantic-settings

# 8. Specific versions (like your Colab)
RUN pip uninstall -y transformers && \
    pip install --no-cache-dir transformers==4.46.1

RUN pip install --no-cache-dir git+https://github.com/huggingface/parler-tts.git

RUN pip install --no-cache-dir "peft==0.17.1"

RUN pip install --no-cache-dir --upgrade CTranslate2

# Verify cuDNN setup
RUN python -c "import ctranslate2; print(f'CTranslate2: {ctranslate2.__version__}')" || echo "CTranslate2 check failed"

# Cloud Run uses PORT environment variable
ENV PORT=8080

# Start FastAPI
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
