"""
GuppShupp Streamlit Application
===============================

Main entry point for the GuppShupp voice AI companion frontend.
Premium, modern UI with dark theme and glassmorphism effects.

Features:
    - Authentication (login/signup)
    - Voice chat with audio recording
    - SSE streaming with progress updates
    - Conversation history
    - Session management

Run:
    streamlit run frontend/app.py

Author: GuppShupp Team
"""

import streamlit as st
import uuid
import time
from datetime import datetime

import base64  # ✅ ADD THIS
import io      # ✅ ADD THIS
import logging

# ✅ ADD: Configure logging for terminal output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure page - MUST be first Streamlit command
st.set_page_config(
    page_title="GuppShupp - Voice Companion",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/guppshupp",
        "Report a bug": "https://github.com/guppshupp/issues",
        "About": "GuppShupp - Your Emotional Voice Companion"
    }
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

def inject_custom_css():
    """Inject premium custom CSS for modern UI."""
    st.markdown("""
    <style>
    /* === ROOT VARIABLES === */
    :root {
        --primary: #6366F1;
        --primary-dark: #4F46E5;
        --primary-light: #818CF8;
        --secondary: #10B981;
        --accent: #F59E0B;
        --background: #0F0F23;
        --surface: #1A1A2E;
        --surface-light: #252540;
        --text-primary: #FFFFFF;
        --text-secondary: #94A3B8;
        --border: rgba(255, 255, 255, 0.1);
        --gradient-primary: linear-gradient(135deg, #6366F1, #8B5CF6);
        --gradient-secondary: linear-gradient(135deg, #10B981, #3B82F6);
        --shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        --glass: rgba(255, 255, 255, 0.05);
    }
    
    /* === GLOBAL STYLES === */
    .stApp {
        background: var(--background);
    }
    
    /* === HEADER === */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.2rem;
        color: var(--text-secondary);
        -webkit-text-fill-color: var(--text-secondary);
    }
    
    /* === GLASS CARDS === */
    .glass-card {
        background: var(--glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    
    /* === BUTTONS === */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* === TEXT INPUTS === */
    .stTextInput > div > div > input {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* === SIDEBAR === */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    
    /* === CHAT MESSAGES === */
    .chat-message {
        display: flex;
        margin: 1rem 0;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .chat-message.user {
        justify-content: flex-end;
    }
    
    .chat-message.assistant {
        justify-content: flex-start;
    }
    
    .message-bubble {
        max-width: 70%;
        padding: 1rem 1.5rem;
        border-radius: 20px;
        line-height: 1.5;
    }
    
    .message-bubble.user {
        background: var(--gradient-primary);
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .message-bubble.assistant {
        background: var(--surface-light);
        color: var(--text-primary);
        border-bottom-left-radius: 4px;
        border: 1px solid var(--border);
    }
    
    /* === EMOTION BADGE === */
    .emotion-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .emotion-joy { background: rgba(16, 185, 129, 0.2); color: #10B981; }
    .emotion-sadness { background: rgba(59, 130, 246, 0.2); color: #3B82F6; }
    .emotion-anger { background: rgba(239, 68, 68, 0.2); color: #EF4444; }
    .emotion-fear { background: rgba(168, 85, 247, 0.2); color: #A855F7; }
    .emotion-neutral { background: rgba(148, 163, 184, 0.2); color: #94A3B8; }
    
    /* === RECORDING INDICATOR === */
    .recording-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #EF4444;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .recording-dot {
        width: 12px;
        height: 12px;
        background: #EF4444;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    
    /* === PROGRESS BAR === */
    .progress-container {
        background: var(--surface);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 4px;
        background: var(--border);
        border-radius: 2px;
        overflow: hidden;
    }
    
    .progress-bar-fill {
        height: 100%;
        background: var(--gradient-primary);
        border-radius: 2px;
        transition: width 0.3s ease;
    }
    
    /* === AUDIO PLAYER === */
    audio {
        width: 100%;
        border-radius: 12px;
        outline: none;
    }
    
    audio::-webkit-media-controls-panel {
        background: var(--surface);
    }
    
    /* === SESSION CARDS === */
    .session-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .session-card:hover {
        background: var(--surface-light);
        border-color: var(--primary);
    }
    
    .session-card.active {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    
    /* === HIDE STREAMLIT BRANDING === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    # header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        # Authentication
        "authenticated": False,
        "user": None,
        "session_token": None,
        
        # Chat
        "current_session_id": None,
        "messages": [],
        "sessions": [],
        
        # UI state
        "is_recording": False,
        "is_processing": False,
        "processing_phase": None,
        "error_message": None,
        
        # API client
        "api_client": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_api_client():
    """Get or create API client."""
    if st.session_state.api_client is None:
        from frontend.utils.api_client import GuppShuppClient
        st.session_state.api_client = GuppShuppClient()
    
    # Sync token if available
    if st.session_state.session_token:
        st.session_state.api_client.session_token = st.session_state.session_token
    
    return st.session_state.api_client


# =============================================================================
# AUTHENTICATION UI
# =============================================================================

def render_auth_page():
    """Render login/signup page."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="main-header">
            <h1>🎙️ GuppShupp</h1>
            <p>Your Emotional Voice Companion</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            render_login_form()
        
        with tab2:
            render_signup_form()


def render_login_form():
    """Render login form."""
    with st.form("login_form"):
        username = st.text_input("Username or Email", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        submitted = st.form_submit_button("Login", use_container_width=True)
        
        if submitted:
            if not username or not password:
                st.error("Please fill in all fields")
                return
            
            try:
                client = get_api_client()
                user = client.login(username, password)
                
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.session_token = client.session_token
                
                st.success(f"Welcome back, {user.display_name or user.username}!")
                time.sleep(0.5)
                st.rerun()
                
            except Exception as e:
                st.error(f"Login failed: {str(e)}")


def render_signup_form():
    """Render signup form."""
    with st.form("signup_form"):
        username = st.text_input("Username", placeholder="Choose a username")
        email = st.text_input("Email", placeholder="your@email.com")
        display_name = st.text_input("Display Name (optional)", placeholder="How should we call you?")
        password = st.text_input("Password", type="password", placeholder="Min 6 characters")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
        
        submitted = st.form_submit_button("Create Account", use_container_width=True)
        
        if submitted:
            if not username or not email or not password:
                st.error("Please fill in all required fields")
                return
            
            if password != confirm_password:
                st.error("Passwords don't match")
                return
            
            if len(password) < 6:
                st.error("Password must be at least 6 characters")
                return
            
            try:
                client = get_api_client()
                user = client.signup(username, email, password, display_name or None)
                
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.session_token = client.session_token
                
                st.success(f"Welcome to GuppShupp, {user.display_name or user.username}!")
                time.sleep(0.5)
                st.rerun()
                
            except Exception as e:
                st.error(f"Signup failed: {str(e)}")


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render sidebar with user info and sessions."""
    with st.sidebar:
        # User info
        user = st.session_state.user
        if user:
            st.markdown(f"""
            <div class="glass-card">
                <h3>👤 {user.display_name or user.username}</h3>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">
                    {user.email}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True):
            create_new_session()
        
        st.divider()
        
        # Sessions list
        st.subheader("💬 Chat Sessions")
        
        try:
            client = get_api_client()
            sessions = client.list_sessions(limit=10)
            
            for session in sessions:
                session_id = session.get("session_id", "")
                title = session.get("title", "New Chat")[:30]
                is_active = session_id == st.session_state.current_session_id
                
                if st.button(
                    f"{'🔵 ' if is_active else ''}{title or 'New Chat'}",
                    key=f"session_{session_id}",
                    use_container_width=True
                ):
                    st.session_state.current_session_id = session_id
                    load_session_history(session_id)
                    st.rerun()
                    
        except Exception as e:
            st.warning("Could not load sessions")
        
        st.divider()
        
        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            logout()


def create_new_session():
    """Create a new chat session."""
    try:
        client = get_api_client()
        session_id = client.create_session()
        st.session_state.current_session_id = session_id
        st.session_state.messages = []
        st.rerun()
    except Exception as e:
        st.error(f"Failed to create session: {e}")


def load_session_history(session_id: str):
    """Load conversation history for session."""
    try:
        client = get_api_client()
        history = client.get_history(session_id, limit=50)
        
        messages = []
        for conv in reversed(history):  # Reverse to get chronological order
            messages.append({
                "role": "user",
                "text": conv.get("user_input_text", ""),
                "emotion": None,
                "timestamp": conv.get("created_at")
            })
            messages.append({
                "role": "assistant",
                "text": conv.get("ai_response_text", ""),
                "emotion": conv.get("detected_emotion"),
                "timestamp": conv.get("created_at")
            })
        
        st.session_state.messages = messages
        
    except Exception as e:
        st.warning(f"Could not load history: {e}")


def logout():
    """Logout user."""
    try:
        client = get_api_client()
        client.logout()
    except Exception:
        pass
    
    # Clear session state
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.session_token = None
    st.session_state.messages = []
    st.session_state.current_session_id = None
    st.session_state.api_client = None
    
    st.rerun()


# =============================================================================
# CHAT UI
# =============================================================================

def render_chat_page():
    """Render main chat interface."""
    # Ensure we have a session
    if not st.session_state.current_session_id:
        create_new_session()
        return
    
    # Chat container
    st.markdown("""
    <div class="main-header" style="padding: 1rem 0;">
        <h1 style="font-size: 2rem;">🎙️ GuppShupp</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Messages container
    chat_container = st.container()
    
    with chat_container:
        render_messages()
    
    # Input area
    st.divider()
    render_input_area()


def render_messages():
    """Render chat messages."""
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        text = msg.get("text", "")
        emotion = msg.get("emotion")
        audio_b64 = msg.get("audio_base64")
        
        # Choose alignment
        if role == "user":
            col1, col2 = st.columns([1, 3])
            with col2:
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="message-bubble user">
                        {text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                emotion_class = f"emotion-{emotion}" if emotion else "emotion-neutral"
                emotion_display = emotion.capitalize() if emotion else ""
                
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="message-bubble assistant">
                        {text}
                        {f'<div class="emotion-badge {emotion_class}">{emotion_display}</div>' if emotion else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Audio player
                if audio_b64:
                    logger.info(f"🔊 Rendering audio player: {len(audio_b64)} chars")
                    st.audio(f"data:audio/wav;base64,{audio_b64}")
                else:
                    logger.warning(f"⚠️ No audio_base64 for assistant message: {text[:50]}...")



def render_input_area():
    """Render audio input area."""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Audio input
        audio_file = st.file_uploader(
            "Upload audio or record",
            type=["wav", "mp3", "webm", "ogg"],
            key="audio_upload",
            label_visibility="collapsed"
        )
        
        # Built-in audio recorder
        audio_bytes = st.audio_input(
            "Record your message",
            key="audio_recorder"
        )
    
    with col2:
        send_enabled = (audio_file is not None or audio_bytes is not None) and not st.session_state.is_processing
        
    if st.button("🎤 Send", use_container_width=True, disabled=not send_enabled):
        # ✅ ENHANCED AUDIO PROCESSING WITH LOGGING
        logger.info("="*80)
        logger.info("🎤 AUDIO INPUT RECEIVED")
        logger.info("="*80)
        
        try:
            audio_data = None
            audio_format = "wav"
            source_name = ""
            
            if audio_bytes:
                audio_data = audio_bytes.getvalue()
                audio_format = "wav"
                source_name = "st.audio_input()"
                logger.info(f"📊 Source: {source_name}")
                logger.info(f"📊 Raw bytes length: {len(audio_data)}")
                
            elif audio_file:
                audio_data = audio_file.getvalue()
                audio_format = audio_file.name.split(".")[-1].lower()
                source_name = f"st.file_uploader({audio_file.name})"
                logger.info(f"📊 Source: {source_name}")
                logger.info(f"📊 Raw bytes length: {len(audio_data)}")
            
            if audio_data:
                # ✅ DETECT AUDIO FORMAT
                logger.info(f"🔍 First 20 bytes (hex): {audio_data[:20].hex()}")
                
                if audio_data[:4] == b'RIFF':
                    detected_format = "WAV"
                    logger.info("✅ Detected format: WAV (RIFF header)")
                elif audio_data[:3] == b'ID3':
                    detected_format = "MP3 with ID3 tags"
                    logger.warning("⚠️ Detected: MP3 with ID3 tags")
                elif audio_data[:2] in [b'\xff\xfb', b'\xff\xfa', b'\xff\xf3', b'\xff\xf2']:
                    detected_format = "MP3"
                    logger.info("✅ Detected format: MP3")
                else:
                    detected_format = "Unknown"
                    logger.warning(f"⚠️ Unknown format. First 4 bytes: {audio_data[:4].hex()}")
                
                # ✅ TRY TO CLEAN WITH PYDUB
                try:
                    from pydub import AudioSegment
                    
                    logger.info("🔧 Attempting to clean audio with pydub...")
                    
                    # Load and normalize
                    audio = AudioSegment.from_file(io.BytesIO(audio_data))
                    logger.info(f"📊 Original: {len(audio)}ms, {audio.frame_rate}Hz, {audio.channels}ch")
                    
                    # Convert to optimal format
                    audio = audio.set_channels(1)
                    audio = audio.set_frame_rate(16000)
                    
                    # Export clean WAV
                    clean_buffer = io.BytesIO()
                    audio.export(clean_buffer, format="wav")
                    clean_buffer.seek(0)
                    audio_data = clean_buffer.read()
                    audio_format = "wav"
                    
                    logger.info(f"✅ Clean WAV: {len(audio_data)} bytes")
                    
                except ImportError:
                    logger.warning("⚠️ pydub not installed - using raw audio")
                    st.warning("⚠️ For better audio quality, install pydub: pip install pydub")
                except Exception as e:
                    logger.warning(f"⚠️ Audio cleaning failed: {e} - using raw audio")
                
                # ✅ ENCODE TO BASE64
                audio_b64 = base64.b64encode(audio_data).decode("ascii")
                logger.info(f"✅ Base64 encoded: {len(audio_b64)} characters")
                logger.info(f"🔍 Base64 first 50 chars: {audio_b64[:50]}")
                
                # ✅ SEND TO BACKEND
                logger.info("📤 Calling process_audio()...")
                process_audio(audio_b64, audio_format)
                
        except Exception as e:
            logger.error(f"❌ AUDIO PROCESSING ERROR: {e}", exc_info=True)
            st.session_state.error_message = f"Error encoding audio: {str(e)}"
            st.rerun()


    
    # Processing indicator
    if st.session_state.is_processing:
        with st.container():
            phase = st.session_state.processing_phase or "Processing"
            st.markdown(f"""
            <div class="progress-container">
                <div class="recording-indicator">
                    <div class="recording-dot"></div>
                    <span>{phase}...</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.spinner(phase)
    
    # Error display
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        st.session_state.error_message = None


def process_audio(audio_base64: str, audio_format: str):
    """Process audio and get AI response with detailed logging."""
    logger.info("="*80)
    logger.info("🚀 PROCESSING AUDIO WITH BACKEND")
    logger.info("="*80)
    
    st.session_state.is_processing = True
    st.session_state.error_message = None
    
    try:
        client = get_api_client()
        
        logger.info(f"🔧 Session ID: {st.session_state.current_session_id}")
        logger.info(f"🔧 Audio format: {audio_format}")
        logger.info(f"🔧 Base64 length: {len(audio_base64)}")
        logger.info(f"🔧 Session token: {'Yes' if client.session_token else 'No'}")
        
        # Add user message placeholder
        st.session_state.messages.append({
            "role": "user",
            "text": "🎤 Audio message...",
            "timestamp": datetime.now()
        })
        
        # Process chat with SSE
        response = None
        event_count = 0
        
        logger.info("📡 Starting SSE stream...")
        
        for event in client.chat_stream(
            session_id=st.session_state.current_session_id,
            audio_base64=audio_base64,
            audio_format=audio_format
        ):
            event_count += 1
            
            if event.event_type == "heartbeat":
                phase = event.data.get("phase", "Processing")
                st.session_state.processing_phase = phase.replace("_", " ").title()
                logger.debug(f"💓 Heartbeat #{event_count}: {phase}")
                
            elif event.event_type == "progress":
                phase = event.data.get("message", "Processing")
                st.session_state.processing_phase = phase
                logger.info(f"📊 Progress #{event_count}: {phase}")
                
            elif event.event_type == "complete":
                response = event.data.get("response", event.data)
                logger.info(f"✅ Complete event received!")
                logger.info(f"   User text: {response.get('user_input_text', '')[:50]}...")
                logger.info(f"   AI text: {response.get('ai_response_text', '')[:50]}...")
                logger.info(f"   Audio base64 length: {len(response.get('response_audio_base64', ''))}")
                logger.info(f"   Total time: {response.get('total_processing_time_ms', 0)}ms")
                break
                
            elif event.event_type == "error":
                error_msg = event.data.get("message", "Unknown error")
                logger.error(f"❌ Error event #{event_count}: {error_msg}")
                raise Exception(error_msg)
        
        logger.info(f"📡 SSE stream ended. Total events: {event_count}")
        
        if response:
            # Update user message with transcription
            st.session_state.messages[-1]["text"] = response.get("user_input_text", "Audio message")
            
            # Add AI response
            st.session_state.messages.append({
                "role": "assistant",
                "text": response.get("ai_response_text", ""),
                "emotion": response.get("detected_emotion"),
                "audio_base64": response.get("response_audio_base64"),
                "timestamp": datetime.now()
            })
            
            logger.info("✅ Messages updated successfully")
        else:
            logger.warning("⚠️ No response received from backend")
        
    except Exception as e:
        logger.error(f"❌ PROCESSING ERROR: {e}", exc_info=True)
        st.session_state.error_message = f"Error: {str(e)}"
        # Remove placeholder message
        if st.session_state.messages and st.session_state.messages[-1]["text"] == "🎤 Audio message...":
            st.session_state.messages.pop()
    
    finally:
        st.session_state.is_processing = False
        st.session_state.processing_phase = None
        logger.info("="*80)
        logger.info("🏁 AUDIO PROCESSING COMPLETE")
        logger.info("="*80)
        st.rerun()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize
    init_session_state()
    inject_custom_css()
    
    # Route based on auth state
    if st.session_state.authenticated:
        render_sidebar()
        render_chat_page()
    else:
        render_auth_page()


if __name__ == "__main__":
    main()
