<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<div align="center">

![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge)
![Version](https://img.shields.io/badge/version-v69-blue?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)

</div>

<!-- PROJECT HEADER -->
<br />
<div align="center">
  <!-- Uncomment and add logo path when ready
  <img src="images/logo.png" alt="Logo" width="200">
  -->
  
  <h1 align="center">🎙️ Indic-Voice-AI</h1>

  <p align="center">
    <strong>Production-Ready Voice AI Pipeline for Indian Languages</strong>
    <br />
    Complete end-to-end workflow with ASR, LLM, TTS, emotion detection & semantic memory
    <br />
    <br />
    <a href="#-demo"><strong>View Demo »</strong></a>
    ·
    <a href="#-features">Features</a>
    ·
    <a href="https://github.com/GothiProCoder/Voice-Companion/issues">Report Bug</a>
    ·
    <a href="https://github.com/GothiProCoder/Voice-Companion/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>📖 Table of Contents</summary>
  <ol>
    <li><a href="#-about-the-project">About The Project</a></li>
    <li><a href="#-demo">Demo</a></li>
    <li><a href="#-features">Features</a></li>
    <li><a href="#-tech-stack">Tech Stack</a></li>
    <li><a href="#-architecture">Architecture</a></li>
    <li><a href="#-getting-started">Getting Started</a></li>
    <li><a href="#-usage">Usage</a></li>
    <li><a href="#-api-documentation">API Documentation</a></li>
    <li><a href="#-deployment">Deployment</a></li>
    <li><a href="#-contributing">Contributing</a></li>
    <li><a href="#-license">License</a></li>
    <li><a href="#-contact">Contact</a></li>
    <li><a href="#-acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---

## 📖 About The Project

<!-- Uncomment when screenshot is ready
![Product Screenshot](images/screenshot.png)
-->

**Indic-Voice-AI** is a comprehensive, production-grade voice AI pipeline specifically designed for Indian languages. Built with state-of-the-art AI models and modern software architecture, it provides a complete end-to-end workflow for building emotionally-aware voice applications.

### 🎯 Why This Project Exists

Voice AI solutions for Indian languages are often fragmented, requiring developers to piece together multiple services and models. **Indic-Voice-AI** solves this by providing:

- **Complete Pipeline**: From speech input to emotional voice output, all in one place
- **Production-Ready**: Battle-tested architecture with proper error handling, retries, and fallbacks
- **Indian Language Focus**: Optimized for 18+ Indic languages with native speaker support
- **Emotional Intelligence**: Real-time emotion detection and emotionally-appropriate responses
- **Semantic Memory**: Long-term memory system that remembers user context across conversations
- **Open Source**: Fully open, modular, and extendable for your specific use case

### 🌟 Perfect For

- 🏗️ **Developers** building voice AI applications for Indian markets
- 🔬 **Researchers** working on multilingual conversational AI
- 🏢 **Startups** needing a production-ready voice AI foundation
- 🎓 **Students** learning about modern AI pipelines and LangGraph workflows
- 🌍 **Open Source Contributors** passionate about Indic language technology

<p align="right">(<a href="#readme-top">↑ top</a>)</p>

---

## 🎬 Demo

<!-- Uncomment when demo assets are ready
### Live Demo

![Demo GIF](images/demo.gif)

**Try it yourself:** [Live Demo Link](#) *(Coming Soon)*

### Video Walkthrough

[![YouTube Demo](images/youtube-thumbnail.png)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

### Screenshots

<div align="center">
  <img src="images/streamlit-ui.png" alt="Streamlit UI" width="800"/>
  <p><em>Modern Streamlit interface with glassmorphism design</em></p>
</div>
-->

**Note:** Demo assets coming soon! This is a production-ready pipeline currently deployed on Colab + ngrok for testing.

<p align="right">(<a href="#readme-top">↑ top</a>)</p>

---

## ✨ Features

### 🎙️ Speech Processing
- ✅ **Whisper ASR** - Faster-whisper large-v3-turbo for 11 Indic languages
- ✅ **Auto Language Detection** - Automatic language identification
- ✅ **Code-Mixing Support** - Handles Hinglish and other mixed languages
- ✅ **Urdu → Hindi Re-transcription** - Devanagari script output for Urdu input

### 🧠 AI Intelligence
- ✅ **Gemini 2.5-flash LLM** - Context-aware response generation
- ✅ **Emotion Detection** - 25+ acoustic features (LibROSA + OpenSMILE)
- ✅ **Sentiment Analysis** - Real-time emotional state tracking
- ✅ **Intent Classification** - Understanding user goals

### 💾 Memory System
- ✅ **IndicBERT Embeddings** - 768-dim semantic vectors
- ✅ **pgvector Search** - Fast similarity search with HNSW indexes
- ✅ **Multi-Layer Memory** - Long-term, episodic, and short-term memory
- ✅ **Memory Decay** - Time-based forgetting with importance scoring

### 🔊 Text-to-Speech
- ✅ **Parler TTS (Indic)** - 18 languages, 60+ speakers
- ✅ **Emotional Styling** - Dynamic TTS based on detected emotion
- ✅ **Speaker Selection** - Gender and voice preference tracking
- ✅ **High Quality Audio** - 44.1kHz with FP16 optimization

### 🏗️ Architecture
- ✅ **LangGraph Workflow** - 5-phase orchestrated pipeline
- ✅ **FastAPI Backend** - Async, production-ready REST API
- ✅ **SSE Streaming** - Real-time progress updates
- ✅ **PostgreSQL + pgvector** - Scalable database with vector search
- ✅ **Streamlit Frontend** - Modern, responsive UI

### 🔒 Safety & Reliability
- ✅ **Crisis Detection** - Self-harm, abuse, medical concern flagging
- ✅ **Content Moderation** - Multi-layer safety checks
- ✅ **Retry Logic** - Automatic retries with exponential backoff
- ✅ **JSON Repair** - Robust parsing with fallback responses

<p align="right">(<a href="#readme-top">↑ top</a>)</p>

---

## 🛠️ Tech Stack

### Backend
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge)

### AI/ML Models
![Whisper](https://img.shields.io/badge/Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

### Frontend
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

### DevOps & Tools
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Alembic](https://img.shields.io/badge/Alembic-6BA539?style=for-the-badge)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge)

<details>
<summary><strong>📦 Complete Dependencies</strong></summary>

### Core Backend
- **FastAPI** - Modern async web framework
- **Uvicorn** - ASGI server
- **SQLAlchemy** - ORM with pgvector extension
- **Alembic** - Database migrations
- **Pydantic** - Data validation

### AI/ML Stack
- **faster-whisper** - Optimized Whisper inference
- **google-generativeai** - Gemini API client
- **sentence-transformers** - IndicBERT embeddings
- **parler-tts** - Indic multilingual TTS
- **librosa** - Audio analysis
- **opensmile** - Prosody features
- **torch** - Deep learning framework

### Workflow & Orchestration
- **langgraph** - State-driven workflow engine
- **langchain** - LLM orchestration utilities

### Frontend
- **streamlit** - Web UI framework

</details>

<p align="right">(<a href="#readme-top">↑ top</a>)</p>

---

## 🏛️ Architecture

### 5-Phase LangGraph Pipeline

```mermaid
flowchart TD
    A[Audio Input] --> B[Phase 1: Audio Analysis]
    B --> C[Whisper ASR]
    B --> D[Emotion Detection]
    C --> E[Phase 2: Context Preparation]
    D --> E
    E --> F[Memory Retrieval]
    E --> G[Session Context]
    F --> H[Phase 3: LLM Generation]
    G --> H
    H --> I[Gemini 2.5-flash]
    I --> J[Phase 4: TTS Generation]
    J --> K[Parler TTS]
    K --> L[Phase 5: DB Persistence]
    L --> M[PostgreSQL + pgvector]
    M --> N[Audio Output]
```

### System Architecture

```mermaid
graph TB
    subgraph "Frontend"
        A[Streamlit UI]
    end
    
    subgraph "Backend API"
        B[FastAPI]
        C[SSE Streaming]
    end
    
    subgraph "AI Services"
        D[Whisper ASR]
        E[Gemini LLM]
        F[IndicBERT Memory]
        G[Parler TTS]
        H[Emotion Detection]
    end
    
    subgraph "Data Layer"
        I[PostgreSQL]
        J[pgvector]
        K[Audio Storage]
    end
    
    A -->|HTTP/SSE| B
    B --> C
    C --> D
    C --> E
    C --> F
    C --> G
    C --> H
    E --> I
    F --> J
    G --> K
```

### Database Schema

```mermaid
erDiagram
    USER ||--o{ SESSION : creates
    USER ||--o{ CONVERSATION : has
    USER ||--o{ MEMORY : owns
    SESSION ||--o{ CONVERSATION : contains
    CONVERSATION ||--o{ SAFETY_LOG : triggers
    
    USER {
        uuid id PK
        string username
        string email
        string password_hash
        jsonb preferences
    }
    
    CONVERSATION {
        uuid id PK
        uuid user_id FK
        uuid session_id FK
        text user_input
        text ai_response
        string emotion
        jsonb prosody_features
        vector embedding
    }
    
    MEMORY {
        uuid id PK
        uuid user_id FK
        text memory_text
        string memory_type
        float importance_score
        float decay_factor
        vector embedding
    }
```

<p align="right">(<a href="#readme-top">↑ top</a>)</p>

---

## 🚀 Getting Started

### ⚠️ Hardware Requirements

This is a **GPU-intensive** pipeline. Recommended specifications:

- **GPU**: NVIDIA GPU with 8GB+ VRAM (T4, V100, A100)
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ for models and dependencies

**Cloud Options:**
- ✅ Google Colab with GPU runtime (recommended for testing)
- ✅ AWS EC2 with GPU instances (g4dn, p3)
- ✅ Runpod, Lambda Labs, or similar GPU cloud providers

### Prerequisites

**Required:**
- Python 3.10 or higher
- PostgreSQL 14+ with pgvector extension
- CUDA-enabled GPU (for optimal performance)

**Optional:**
- Docker (for containerized deployment)
- ngrok (for exposing local server)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/GothiProCoder/Voice-Companion.git
cd Voice-Companion
```

#### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt
```

#### 3. Setup Database

**Install PostgreSQL with pgvector:**

```bash
# On Ubuntu/Debian
sudo apt-get install postgresql-14 postgresql-14-pgvector

# On macOS with Homebrew
brew install postgresql@14
brew install pgvector
```

**Create Database:**

```sql
CREATE DATABASE voice_ai;
\c voice_ai
CREATE EXTENSION vector;
```

#### 4. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
nano .env
```

**Required Environment Variables:**

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/voice_ai

# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
HUGGINGFACE_TOKEN=your_hf_token_here  # Optional, for private models

# Model Configuration
WHISPER_MODEL_SIZE=large-v3-turbo
WHISPER_DEVICE=cuda  # or "cpu"
PARLER_TTS_MODEL=ai4bharat/indic-parler-tts

# Application
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
FRONTEND_URL=http://localhost:8501
```

<details>
<summary><strong>📝 Complete .env Configuration</strong></summary>

See `.env.example` for all available configuration options including:
- Model parameters (temperature, max tokens)
- Audio settings (sample rate, encoding)
- Memory settings (context size, decay rates)
- Safety settings (content moderation thresholds)

</details>

#### 5. Run Database Migrations

```bash
# Initialize database schema
alembic upgrade head
```

#### 6. Start the Backend

```bash
# Start FastAPI server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`

Access API docs at `http://localhost:8000/docs`

#### 7. Start the Frontend (Optional)

```bash
# In a new terminal
streamlit run frontend/app.py
```

The frontend will open automatically at `http://localhost:8501`

<p align="right">(<a href="#readme-top">↑ top</a>)</p>

---

## 💻 Usage

### Quick Start Example

```python
import requests
import base64

# Encode your audio file
with open("audio.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Make API request
response = requests.post(
    "http://localhost:8000/api/v1/conversations/chat",
    json={
        "session_id": "your-session-id",
        "audio_base64": audio_base64,
        "audio_format": "wav"
    },
    headers={
        "Authorization": f"Bearer {session_token}"
    },
    stream=True  # Enable SSE streaming
)

# Process SSE events
for line in response.iter_lines():
    if line:
        event_data = json.loads(line.decode().replace("data: ", ""))
        print(f"Phase: {event_data.get('phase')}")
        
        if event_data.get('type') == 'complete':
            ai_response = event_data['response']
            print(f"AI: {ai_response['response_text']}")
            print(f"Emotion: {ai_response['detected_emotion']}")
```

### API Endpoints

<details>
<summary><strong>🔐 Authentication</strong></summary>

**POST /api/v1/auth/signup**
```json
{
  "username": "user123",
  "email": "user@example.com",
  "password": "securepass123",
  "display_name": "My Name"
}
```

**POST /api/v1/auth/login**
```json
{
  "username": "user123",
  "password": "securepass123"
}
```

Returns session token for authenticated requests.

</details>

<details>
<summary><strong>💬 Conversations</strong></summary>

**POST /api/v1/conversations/chat** (SSE Streaming)

Process voice input and get AI response:
```json
{
  "session_id": "uuid-here",
  "audio_base64": "base64-encoded-audio",
  "audio_format": "wav"
}
```

**SSE Events:**
- `heartbeat` - Keep-alive (every 10s)
- `progress` - Phase updates
- `complete` - Final response with audio
- `error` - Error details

**GET /api/v1/conversations/history**

Get conversation history:
```
?session_id=uuid&limit=50&include_audio=false
```

</details>

### Integration Example

```python
from backend.services.langgraph_workflow import run_voice_workflow
from backend.database.database import get_db_session

async def process_voice():
    with get_db_session() as db:
        result = await run_voice_workflow(
            db=db,
            user_id="user-uuid",
            session_id="session-uuid",
            audio_path="/path/to/audio.wav",
            session_context={}
        )
        
        print(f"Transcript: {result['transcript']}")
        print(f"Response: {result['llm_response']['response_text']}")
        print(f"TTS Audio: {result['tts_output']['audio_path']}")
```

<p align="right">(<a href="#readme-top">↑ top</a>)</p>

---

## 📚 API Documentation

**Interactive API Docs:**
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

**Key Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/auth/signup` | Create new user account |
| `POST` | `/api/v1/auth/login` | Authenticate user |
| `POST` | `/api/v1/auth/logout` | Invalidate session |
| `GET` | `/api/v1/auth/me` | Get current user profile |
| `POST` | `/api/v1/conversations/chat` | Process voice input (SSE) |
| `GET` | `/api/v1/conversations/history` | Get chat history |
| `POST` | `/api/v1/conversations/sessions` | Create new session |
| `GET` | `/api/v1/conversations/sessions` | List user sessions |

**Response Schema:**

```typescript
interface ChatResponse {
  response_text: string;
  response_language: string;
  detected_emotion: string;
  sentiment: string;
  detected_intent: string;
  tts_speaker: string;
  tts_style_prompt: string;
  audio_base64_wav?: string;
  audio_path?: string;
  memory_updates: MemoryUpdate[];
  safety_flags: SafetyFlags;
}
```

<p align="right">(<a href="#readme-top">↑ top</a>)</p>

---

## 🚢 Deployment

### 🎯 Recommended: Google Colab + ngrok

**This is the tested and recommended deployment method for GPU-accelerated inference.**

We provide a ready-to-use Colab notebook that handles all the setup automatically: **[backend-colab-notebook.ipynb](backend-colab-notebook.ipynb)**

#### Prerequisites

1. **Google Account** - For Google Colab access
2. **ngrok Account** - Free at [ngrok.com](https://ngrok.com) (for exposing backend API)
3. **Database Choice** - Pick one:
   - ✅ **Neon DB** (Recommended): Free PostgreSQL with pgvector - [neon.tech](https://neon.tech)
   - ⚠️ **Local PostgreSQL**: Requires Pinggy tunneling (expires hourly)

#### Step-by-Step Setup

##### 1. Open the Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `backend-colab-notebook.ipynb` from this repository
3. **IMPORTANT**: Enable GPU runtime
   - Runtime → Change runtime type → GPU → T4 GPU (or better)

##### 2. Database Setup

**Option A: Neon DB (Recommended)**

1. Create free account at [neon.tech](https://neon.tech)
2. Create new project
3. Copy connection string (looks like `postgresql://user:pass@host.neon.tech/db?sslmode=require`)
4. Enable pgvector extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

**Option B: Local PostgreSQL (Advanced)**

If you must use local PostgreSQL, you'll need to tunnel it:

```bash
# On your local machine (runs for 1 hour)
ssh -p 443 -R0:localhost:5432 tcp@a.pinggy.io
```

**⚠️ Warning**: Pinggy tunnels expire every hour. You'll need to:
1. Get new tunnel URL every hour
2. Update `.env` file with new `DATABASE_URL`
3. Re-run the uvicorn cell

**💡 Tip**: Just use Neon DB to avoid this hassle!

##### 3. Execute Notebook Cells

**⚠️ CRITICAL: Execute cells in exact order. DO NOT restart kernel until instructed!**

**Cell 1-2**: Clone repository
```python
!git clone -b master https://github.com/GothiProCoder/voice-companion.git
%cd voice-companion
```

**Cell 3-9**: Install dependencies
- PyTorch with CUDA support
- FastAPI + backend packages
- Database drivers (psycopg2, pgvector)
- AI/ML libraries (faster-whisper, sentence-transformers)
- Audio processing (librosa, opensmile)
- LangChain/LangGraph
- Utilities (pyngrok, nest-asyncio)

**Cell 10-13**: Install specific versions
```python
# Transformers 4.46.1 (required for Parler TTS compatibility)
!pip uninstall transformers -y
!pip install transformers==4.46.1

# Parler TTS from source
!pip install git+https://github.com/huggingface/parler-tts.git soundfile -q

# PEFT for model efficiency
!pip install "peft==0.17.1" --no-cache-dir

# CTranslate2 for Whisper optimization
!pip install --upgrade CTranslate2
```

**Cell 14**: ⚠️ **RESTART KERNEL** ⚠️
```python
# ================ RESTART THE KERNEL ====================
```
**Action**: Runtime → Restart runtime (or Ctrl+M then .)

##### 4. Post-Restart Setup

**Cell 15**: Navigate back to directory
```python
%cd voice-companion
```

**Cell 16-17**: Upload `.env` file
```python
from google.colab import files
uploaded = files.upload()  # Select your .env file
```

**Prepare your `.env` file** with:
```env
# Database (Neon DB or Pinggy tunnel URL)
DATABASE_URL=postgresql://user:pass@host/db

# Gemini API
GEMINI_API_KEY=your_gemini_key

# Optional: HuggingFace token for private models
HUGGINGFACE_TOKEN=your_hf_token

# Model configs
WHISPER_MODEL_SIZE=large-v3-turbo
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

# Application URLs (will be overridden by ngrok)
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
```

**Cell 18**: Load environment variables
```python
from dotenv import load_dotenv
load_dotenv('.env')
```

**Cell 19**: Setup ngrok
```python
from pyngrok import ngrok
import nest_asyncio

nest_asyncio.apply()
ngrok.kill()  # Clear old tunnels

# Get your ngrok auth token from https://dashboard.ngrok.com/get-started/your-authtoken
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
```

**Cell 20**: Colab environment setup (cuDNN paths)
```python
import os
import pathlib

# Set cuDNN library path for CTranslate2
cudnn_lib_path = "/usr/lib/x86_64-linux-gnu"
os.environ["LD_LIBRARY_PATH"] = cudnn_lib_path

# Verify installation
import ctranslate2, faster_whisper, torch
print(f"✅ CTranslate2: {ctranslate2.__version__}")
print(f"✅ faster-whisper: {faster_whisper.__version__}")
print(f"✅ PyTorch cuDNN: {torch.backends.cudnn.version()}")
```

**Cell 21**: Start backend with ngrok
```python
# Create ngrok tunnel
backend_tunnel = ngrok.connect(8000)
backend_url = backend_tunnel.public_url

print(f"🚀 BACKEND URL: {backend_url}")
print(f"📋 Copy this URL for your frontend!")

# Start FastAPI backend (models load into GPU here)
!uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

##### 5. Use the Backend

**The backend is now running!** Copy the ngrok URL from the output.

**Connect from local Streamlit:**
```bash
# On your local machine
cd voice-companion
streamlit run frontend/app.py
```

Update `frontend/config.py` or `.env`:
```python
API_BASE_URL = "https://your-ngrok-url.ngrok-free.app"
```

#### Troubleshooting

<details>
<summary><strong>⚠️ Kernel Restart Required Error</strong></summary>

If you see import errors after installing packages:
- You likely skipped the kernel restart step
- Go back to Cell 14 and restart the kernel
- Resume from Cell 15

</details>

<details>
<summary><strong>🔌 Database Connection Failed</strong></summary>

**For Neon DB:**
- Verify pgvector extension is enabled
- Check connection string includes `?sslmode=require`
- Test connection:
  ```python
  import psycopg2
  conn = psycopg2.connect(os.getenv("DATABASE_URL"))
  print("✅ Connected!")
  ```

**For Local PostgreSQL:**
- Pinggy tunnel may have expired (1 hour limit)
- Run the SSH command again to get new URL
- Update `.env` with new `DATABASE_URL`
- Re-run the uvicorn cell (Cell 21)

</details>

<details>
<summary><strong>🎮 GPU Out of Memory</strong></summary>

Reduce model memory usage in `.env`:
```env
WHISPER_COMPUTE_TYPE=int8  # Instead of float16
PARLER_TTS_MODEL=ai4bharat/indic-parler-tts-pretrained  # Smaller model
```

Or use Colab Pro for more VRAM (A100 GPU).

</details>

<details>
<summary><strong>🕐 ngrok Tunnel Stopped</strong></summary>

Free ngrok tunnels are stable but can disconnect. If it stops:
```python
# Re-run Cell 21 to create new tunnel
backend_tunnel = ngrok.connect(8000)
print(backend_tunnel.public_url)
```

**Pro Tip**: Upgrade to ngrok Pro for reserved domains (URL stays same).

</details>

#### Performance Notes

- **First request**: ~30-60s (models loading into GPU)
- **Subsequent requests**: ~7-12s per voice turn
- **Model VRAM usage**: ~6-8GB total
  - Whisper: ~3-4GB
  - Parler TTS: ~2-3GB
  - IndicBERT: ~1GB

---

### Docker Deployment (Advanced)

For self-hosted GPU servers:

```bash
# Build image
docker build -t indic-voice-ai .

# Run with GPU support
docker run -d \
  -p 8000:8000 \
  --gpus all \
  --env-file .env \
  -v $(pwd)/audio_storage:/app/audio_storage \
  indic-voice-ai
```

**Requirements:**
- NVIDIA GPU with CUDA 11.8+
- `nvidia-docker2` installed
- 16GB+ RAM
- PostgreSQL with pgvector (separate container or cloud)

---

### Production Deployment

**Recommended Stack:**
- **Compute**: AWS EC2 g4dn.xlarge (NVIDIA T4 GPU)
- **Database**: Neon DB or AWS RDS PostgreSQL with pgvector
- **Storage**: S3 for audio files
- **Proxy**: nginx with SSL (Let's Encrypt)
- **Monitoring**: CloudWatch, Sentry

**Scaling Considerations:**
- AI models require GPU, scale vertically (bigger GPU)
- API layer can scale horizontally (multiple instances)
- Use Redis for distributed session management
- Consider model serving frameworks (TorchServe, NVIDIA Triton) for production



<p align="right">(<a href="#readme-top">↑ top</a>)</p>

---

## 🤝 Contributing

Contributions are what make the open source community amazing! Any contributions you make are **greatly appreciated**.

### How to Contribute

1. **Fork** the repository
2. **Create** your feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit** your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open** a Pull Request

### Contribution Ideas

We welcome contributions in these areas:

- 🌍 **Language Support** - Add more Indian languages
- 🎨 **UI/UX** - Improve Streamlit frontend
- 📊 **Performance** - Optimize model inference
- 🔒 **Safety** - Enhance content moderation
- 📚 **Documentation** - Improve guides and examples
- 🧪 **Testing** - Add unit and integration tests
- 🐛 **Bug Fixes** - Fix issues from the issue tracker

### Code Style

- Follow PEP 8 for Python code
- Use type hints wherever possible
- Add docstrings to all functions/classes
- Write tests for new features

<p align="right">(<a href="#readme-top">↑ top</a>)</p>

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

This means you can:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Sublicense

**Attribution appreciated but not required!**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="right">(<a href="#readme-top">↑ top</a>)</p>

---

## 📧 Contact

**GothiProCoder**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gotham-chand/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:gotham123283@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GothiProCoder)

**Project Link:** [https://github.com/GothiProCoder/Voice-Companion](https://github.com/GothiProCoder/Voice-Companion)

<p align="right">(<a href="#readme-top">↑ top</a>)</p>

---

## 🙏 Acknowledgments

Special thanks to the amazing open-source projects that made this possible:

- [Whisper (OpenAI)](https://github.com/openai/whisper) - Speech recognition
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper inference
- [Google Gemini](https://ai.google.dev/) - Large language model
- [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) - Indic TTS
- [IndicBERT](https://github.com/ai4bharat/IndicBERT) - Indic language embeddings
- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow orchestration
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Streamlit](https://streamlit.io/) - Data app framework

**Inspired by:**
- Research in emotional AI and conversational systems
- The open-source AI/ML community
- The need for better voice AI tools for Indian languages

<p align="right">(<a href="#readme-top">↑ top</a>)</p>

---

<div align="center">

**⭐ If you find this project useful, please star it on GitHub! ⭐**

Made with ❤️ for the Indic language community

</div>
