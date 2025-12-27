"""
GuppShupp Conversation Endpoints
================================

Core chat endpoints with Server-Sent Events (SSE) for heartbeat-based
communication. Handles audio input, workflow orchestration, and response delivery.

Features:
    - SSE heartbeat every 10 seconds (dynamic timeout)
    - Phase progress updates during processing
    - Complete ChatResponse on completion
    - Error handling with retry guidance
    - Conversation history retrieval
    - Session management

Endpoints:
    POST /conversations/chat - Process audio with SSE response
    GET /conversations/history - Get conversation history for session
    GET /conversations/sessions - List user's chat sessions
    POST /conversations/sessions - Create new chat session

Author: GuppShupp Team
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional, Dict, Any, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Request, Query, Path
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.api.deps import (
    get_db,
    get_current_user,
    get_request_context,
    check_chat_rate_limit,
    RequestContext,
)
from backend.database.models import User, Conversation, Session as DBSession
from backend.utils.audio import TempAudioFile, encode_audio_to_base64, cleanup_temp_audio
from backend.schemas.conversation import (
    ChatRequest,
    ChatResponse,
    SSEHeartbeat,
    SSEProgress,
    SSEComplete,
    SSEError,
    ConversationHistoryItem,
    ConversationHistoryResponse,
    SessionInfo,
    CreateSessionRequest,
    CreateSessionResponse,
    ListSessionsResponse,
    PhaseTimings,
    SafetyFlagsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


# =============================================================================
# CONSTANTS
# =============================================================================

HEARTBEAT_INTERVAL_SECONDS = 10  # Send heartbeat every 10 seconds
SSE_CONTENT_TYPE = "text/event-stream"


# =============================================================================
# SSE UTILITIES
# =============================================================================


def format_sse_event(event_type: str, data: dict) -> str:
    """
    Format data as SSE event string.
    
    Args:
        event_type: Event name (heartbeat, progress, complete, error)
        data: Event data dict
        
    Returns:
        Formatted SSE string
    """
    json_data = json.dumps(data, default=str)
    return f"event: {event_type}\ndata: {json_data}\n\n"


async def heartbeat_generator(
    stop_event: asyncio.Event,
    current_phase: Dict[str, str],
    start_time: float
) -> AsyncGenerator[str, None]:
    """
    Generate SSE heartbeat events at regular intervals.
    
    Args:
        stop_event: Event to signal when to stop
        current_phase: Dict tracking current processing phase
        start_time: Request start time
        
    Yields:
        SSE formatted heartbeat strings
    """
    while not stop_event.is_set():
        try:
            # Wait for heartbeat interval or stop event
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=HEARTBEAT_INTERVAL_SECONDS
                )
                # Stop event was set
                break
            except asyncio.TimeoutError:
                # Send heartbeat
                pass
            
            heartbeat = SSEHeartbeat(
                type="heartbeat",
                timestamp=datetime.now(timezone.utc),
                phase=current_phase.get("phase"),
                elapsed_ms=int((time.time() - start_time) * 1000)
            )
            yield format_sse_event("heartbeat", heartbeat.model_dump())
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            break


# =============================================================================
# WORKFLOW EXECUTION
# =============================================================================


async def execute_chat_workflow(
    audio_path: str,
    user_id: str,
    session_id: str,
    conversation_id: str,
    session_context: Optional[Dict[str, Any]],
    request_id: str,
    current_phase: Dict[str, str],
) -> Dict[str, Any]:
    """
    Execute the LangGraph workflow for chat processing.
    
    This is a wrapper around the existing workflow that updates
    the current_phase dict for progress tracking.
    
    Args:
        audio_path: Path to temporary audio file
        user_id: User UUID
        session_id: Session UUID
        conversation_id: Conversation UUID
        session_context: Optional session context
        request_id: Request ID for tracing
        current_phase: Dict to update with current phase
        
    Returns:
        Workflow result dict
    """
    try:
        # Import workflow execute function
        from backend.services.langgraph_workflow import execute_workflow
        
        # Build workflow input
        workflow_input = {
            "audio_path": audio_path,
            "user_id": user_id,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "session_context": session_context or {},
            "request_id": request_id,
        }
        
        # Update phase tracking during execution
        current_phase["phase"] = "initializing"
        
        # Execute workflow - do NOT pass db_session here
        # The workflow creates its own session internally
        result = await execute_workflow(workflow_input)
        
        current_phase["phase"] = "complete"
        return result
        
    except Exception as e:
        current_phase["phase"] = "error"
        logger.error(f"[{request_id}] Workflow execution failed: {e}", exc_info=True)
        raise


def build_chat_response(
    workflow_result: Dict[str, Any],
    request_id: str,
    session_id: UUID,
    conversation_id: UUID,
) -> ChatResponse:
    """
    Build ChatResponse from workflow result.
    
    Args:
        workflow_result: Result from execute_workflow
        request_id: Request ID
        session_id: Session UUID
        conversation_id: Conversation UUID
        
    Returns:
        ChatResponse object
    """
    llm_response = workflow_result.get("llm_response")
    tts_response = workflow_result.get("tts_response")
    transcription = workflow_result.get("transcription")
    prosody = workflow_result.get("prosody_features")
    
    # Get transcription text
    user_input_text = ""
    detected_language = "hi"
    if transcription:
        user_input_text = getattr(transcription, "text", "") or ""
        detected_language = getattr(transcription, "language", "hi") or "hi"
    
    # Get LLM response data
    ai_response_text = ""
    response_language = detected_language
    detected_emotion = "neutral"
    emotion_confidence = 0.5
    sentiment = "neutral"
    detected_intent = "unknown"
    intent_confidence = 0.5
    safety_passed = True
    safety_action = "continue"
    tts_speaker = "Rohit"
    memories_extracted = 0
    
    if llm_response:
        ai_response_text = getattr(llm_response, "response_text", "") or ""
        response_language = getattr(llm_response, "response_language", detected_language) or detected_language
        detected_emotion = getattr(llm_response, "detected_emotion", "neutral") or "neutral"
        emotion_confidence = getattr(llm_response, "emotion_confidence", 0.5) or 0.5
        sentiment = getattr(llm_response, "sentiment", "neutral") or "neutral"
        detected_intent = getattr(llm_response, "detected_intent", "unknown") or "unknown"
        intent_confidence = getattr(llm_response, "intent_confidence", 0.5) or 0.5
        tts_speaker = getattr(llm_response, "tts_speaker", "Rohit") or "Rohit"
        
        # Safety flags
        safety_flags = getattr(llm_response, "safety_flags", None)
        if safety_flags:
            safety_passed = workflow_result.get("safety_passed", True)
            safety_action = workflow_result.get("safety_action", "continue")
        
        # Memory updates
        memory_updates = getattr(llm_response, "memory_updates", [])
        memories_extracted = len(memory_updates) if memory_updates else 0
    
    # Get TTS audio
    # ⚠️ CRITICAL: tts_response is now a DICT after sanitize_for_state()
    # It was converted from TTSResponse dataclass to avoid MsgPack serialization issues
    # ✅ ENHANCED: Get TTS audio with strict validation
    response_audio_base64 = ""
    response_duration_seconds = 0.0

    if tts_response:
        # tts_response is now a dict (from Phase 4 return)
        response_audio_base64 = tts_response.get("audio_base64_wav", "") or ""
        response_duration_seconds = tts_response.get("duration_seconds", 0.0) or 0.0
        
        # ✅ CRITICAL VALIDATION
        if response_audio_base64:
            audio_len = len(response_audio_base64)
            logger.info(
                f"[{request_id}] ✅ TTS audio received: "
                f"{audio_len} chars, duration={response_duration_seconds:.2f}s"
            )
            
            # Sanity check: base64 should be at least 1000 chars for 1+ second audio
            if audio_len < 1000:
                logger.error(
                    f"[{request_id}] ⚠️ Audio suspiciously short: {audio_len} chars. "
                    "Possible truncation in workflow!"
                )
        else:
            logger.error(
                f"[{request_id}] ❌ TTS response missing audio_base64_wav! "
                f"Keys present: {list(tts_response.keys())}"
            )
    else:
        logger.error(f"[{request_id}] ❌ No tts_response in workflow_result!")


    
    # Build phase timings
    # ✅ FIXED: Match workflow field names exactly
    phase_timings = PhaseTimings(
        audio_analysis_ms=workflow_result.get("audio_analysis_time_ms", 0),
        context_preparation_ms=workflow_result.get("context_prep_time_ms", 0),
        llm_generation_ms=workflow_result.get("llm_time_ms", 0),
        tts_generation_ms=workflow_result.get("tts_time_ms", 0),
        database_persistence_ms=workflow_result.get("db_time_ms", 0),
    )

    # ✅ ADD: Debug logging to verify timings
    logger.debug(
        f"[{request_id}] Phase timings: "
        f"audio={phase_timings.audio_analysis_ms}ms, "
        f"context={phase_timings.context_preparation_ms}ms, "
        f"llm={phase_timings.llm_generation_ms}ms, "
        f"tts={phase_timings.tts_generation_ms}ms, "
        f"db={phase_timings.database_persistence_ms}ms"
    )
    
    return ChatResponse(
        request_id=request_id,
        session_id=session_id,
        conversation_id=conversation_id,
        user_input_text=user_input_text,
        detected_language=detected_language,
        ai_response_text=ai_response_text,
        response_language=response_language,
        detected_emotion=detected_emotion,
        emotion_confidence=emotion_confidence,
        sentiment=sentiment,
        detected_intent=detected_intent,
        intent_confidence=intent_confidence,
        response_audio_base64=response_audio_base64,
        response_duration_seconds=response_duration_seconds,
        tts_speaker=tts_speaker,
        total_processing_time_ms=workflow_result.get("total_time_ms", 0),
        phase_timings=phase_timings,
        safety_passed=safety_passed,
        safety_action=safety_action,
        memories_extracted=memories_extracted,
    )


# =============================================================================
# CHAT ENDPOINT WITH SSE
# =============================================================================


@router.post(
    "/chat",
    summary="Process audio and get AI response (SSE)",
    description="""
    Process audio input and receive AI response via Server-Sent Events.
    
    **SSE Event Types:**
    - `heartbeat`: Sent every 10 seconds to indicate processing is ongoing
    - `progress`: Sent when entering a new processing phase
    - `complete`: Final response with full ChatResponse
    - `error`: Error occurred during processing
    
    **Timeout Behavior:**
    Client should reset timeout counter on each heartbeat.
    If no heartbeat received for 30 seconds, connection may be lost.
    """,
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE stream with processing events",
            "content": {"text/event-stream": {}}
        },
        400: {"description": "Invalid audio format or request"},
        401: {"description": "Authentication required"},
        429: {"description": "Rate limit exceeded"},
    }
)
async def chat(
    request: ChatRequest,
    req: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    ctx: RequestContext = Depends(get_request_context),
    _: None = Depends(check_chat_rate_limit),
):
    """
    Process audio input and stream AI response via SSE.
    
    **Request Body:**
    - `session_id`: Chat session UUID (create new or use existing)
    - `audio_base64`: Base64-encoded audio (WAV, MP3, or WebM)
    - `audio_format`: Audio format hint (wav, mp3, webm)
    - `session_context`: Optional TTS preferences
    
    **SSE Response:**
    The response is a stream of Server-Sent Events. Example:
    
    ```
    event: heartbeat
    data: {"type":"heartbeat","timestamp":"...","phase":"audio_analysis"}
    
    event: progress
    data: {"type":"progress","phase":"llm_generation","message":"Generating response..."}
    
    event: complete
    data: {"type":"complete","response":{...ChatResponse...}}
    ```
    """
    logger.info(f"[{ctx.request_id}] Chat request from user: {user.username}")
    
    # Generate conversation ID
    conversation_id = uuid.uuid4()
    start_time = time.time()
    
    # Track current phase for heartbeat
    current_phase: Dict[str, str] = {"phase": "initializing"}
    
    # Event to signal workflow completion
    stop_event = asyncio.Event()
    
    async def event_stream() -> AsyncGenerator[str, None]:
        """
        Generate SSE event stream with concurrent heartbeats.
        
        Uses asyncio.Queue to allow heartbeat task to push events
        while workflow executes, preventing the stream from blocking.
        """
        # Queue for collecting events from multiple sources
        event_queue: asyncio.Queue = asyncio.Queue()
        heartbeat_task = None
        workflow_task = None
        
        async def heartbeat_producer():
            """Push heartbeat events to queue every 10 seconds."""
            try:
                while not stop_event.is_set():
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=HEARTBEAT_INTERVAL_SECONDS)
                        break  # Stop event was set
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        heartbeat = SSEHeartbeat(
                            type="heartbeat",
                            timestamp=datetime.now(timezone.utc),
                            phase=current_phase.get("phase"),
                            elapsed_ms=int((time.time() - start_time) * 1000)
                        )
                        event_str = format_sse_event("heartbeat", heartbeat.model_dump())
                        await event_queue.put(("heartbeat", event_str))
                        logger.debug(f"[{ctx.request_id}] Heartbeat sent: phase={current_phase.get('phase')}")
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[{ctx.request_id}] Heartbeat producer error: {e}")
        
        async def workflow_producer():
            """Execute workflow and push result to queue."""
            try:
                # Progress: Audio processing
                current_phase["phase"] = "audio_processing"
                await event_queue.put(("progress", format_sse_event("progress", {
                    "type": "progress",
                    "phase": "audio_processing",
                    "message": "Decoding audio...",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })))
                
                # Decode audio using context manager
                with TempAudioFile(
                    request.audio_base64,
                    request.audio_format,
                    prefix="chat_"
                ) as temp_audio:
                
                    logger.info(
                        f"[{ctx.request_id}] Audio decoded: {temp_audio.size_bytes} bytes, "
                        f"format={temp_audio.format}"
                    )
                    
                    # Progress: Audio analysis
                    current_phase["phase"] = "audio_analysis"
                    await event_queue.put(("progress", format_sse_event("progress", {
                        "type": "progress",
                        "phase": "audio_analysis",
                        "message": "Analyzing audio with Whisper ASR...",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })))
                    
                    # Execute workflow - now heartbeats continue in parallel!
                    workflow_result = await execute_chat_workflow(
                        audio_path=str(temp_audio.path),
                        user_id=str(user.id),
                        session_id=str(request.session_id),
                        conversation_id=str(conversation_id),
                        session_context=request.session_context or {},
                        request_id=ctx.request_id,
                        current_phase=current_phase,
                    )
                    
                    # Build response
                    chat_response = build_chat_response(
                        workflow_result=workflow_result,
                        request_id=ctx.request_id,
                        session_id=request.session_id,
                        conversation_id=conversation_id,
                    )
                    
                    # Log audio presence
                    audio_len = len(chat_response.response_audio_base64) if chat_response.response_audio_base64 else 0
                    logger.info(
                        f"[{ctx.request_id}] Workflow complete: "
                        f"audio_base64_length={audio_len}, "
                        f"duration={chat_response.response_duration_seconds:.2f}s"
                    )
                    
                    # Send complete event
                    complete_event = SSEComplete(
                        type="complete",
                        response=chat_response,
                        timestamp=datetime.now(timezone.utc)
                    )
                    await event_queue.put(("complete", format_sse_event("complete", complete_event.model_dump())))
                    
            except Exception as e:
                logger.error(f"[{ctx.request_id}] Workflow error: {e}", exc_info=True)
                retryable = _is_retryable_error(e)
                error_event = SSEError(
                    type="error",
                    error=type(e).__name__,
                    message=str(e)[:200],
                    retryable=retryable,
                    request_id=ctx.request_id,
                    timestamp=datetime.now(timezone.utc)
                )
                await event_queue.put(("error", format_sse_event("error", error_event.model_dump())))
            finally:
                # Signal completion
                stop_event.set()
                await event_queue.put(("done", None))
        
        try:
            # Start both producers concurrently
            heartbeat_task = asyncio.create_task(heartbeat_producer())
            workflow_task = asyncio.create_task(workflow_producer())
            
            # Consume events from queue and yield to client
            while True:
                event_type, event_str = await event_queue.get()
                
                if event_type == "done":
                    break
                
                if event_str:
                    yield event_str
                
                # Stop after complete or error
                if event_type in ["complete", "error"]:
                    stop_event.set()
                    break

        except Exception as e:
            logger.error(f"[{ctx.request_id}] SSE stream error: {e}", exc_info=True)
            stop_event.set()
            
            error_event = SSEError(
                type="error",
                error="stream_error",
                message=f"Processing failed: {str(e)[:100]}",
                retryable=True,
                request_id=ctx.request_id,
                timestamp=datetime.now(timezone.utc)
            )
            yield format_sse_event("error", error_event.model_dump())
            
        finally:
            # Cleanup: Cancel all tasks
            stop_event.set()
            
            if heartbeat_task and not heartbeat_task.done():
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            if workflow_task and not workflow_task.done():
                workflow_task.cancel()
                try:
                    await workflow_task
                except asyncio.CancelledError:
                    pass
            
            logger.info(f"[{ctx.request_id}] SSE stream closed")
    
    return StreamingResponse(
        event_stream(),
        media_type=SSE_CONTENT_TYPE,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": ctx.request_id,
        }
    )




async def _heartbeat_loop(
    stop_event: asyncio.Event,
    current_phase: Dict[str, str],
    start_time: float
) -> None:
    """Background task for sending heartbeats."""
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=HEARTBEAT_INTERVAL_SECONDS
            )
            break
        except asyncio.TimeoutError:
            # Heartbeat interval passed, but we can't yield from here
            # The heartbeat is handled in the main stream
            pass


def _is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable."""
    error_type = type(error).__name__
    error_msg = str(error).lower()
    
    # Non-retryable errors
    non_retryable = [
        "valueerror",
        "typeerror",
        "keyerror",
        "attributeerror",
        "invalidaudioformat",
        "audiotoolarge",
    ]
    
    if error_type.lower() in non_retryable:
        return False
    
    # Check for specific retryable patterns
    retryable_patterns = [
        "timeout",
        "rate limit",
        "quota",
        "temporarily",
        "connection",
        "network",
        "503",
        "429",
    ]
    
    for pattern in retryable_patterns:
        if pattern in error_msg:
            return True
    
    # Default to retryable for server errors
    return True


# =============================================================================
# HISTORY ENDPOINTS
# =============================================================================


@router.get(
    "/history",
    response_model=ConversationHistoryResponse,
    summary="Get conversation history for session",
    description="Retrieve past conversations for a specific chat session."
)
async def get_history(
    session_id: UUID = Query(..., description="Session UUID"),
    limit: int = Query(20, ge=1, le=100, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    include_audio: bool = Query(False, description="Include audio in response"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get conversation history for a session.
    
    Returns conversations in reverse chronological order (newest first).
    """
    logger.info(f"[{ctx.request_id}] History request: session={session_id}")
    
    # Verify session belongs to user
    session = db.query(DBSession).filter(
        DBSession.id == session_id,
        DBSession.user_id == user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Get total count
    total_count = db.query(Conversation).filter(
        Conversation.session_id == session_id
    ).count()
    
    # Get conversations
    conversations = db.query(Conversation).filter(
        Conversation.session_id == session_id
    ).order_by(desc(Conversation.created_at)).offset(offset).limit(limit).all()
    
    # Build response items
    items = []
    for conv in conversations:
        item = ConversationHistoryItem(
            id=conv.id,
            user_input_text=conv.user_input_text,
            ai_response_text=conv.ai_response_text,
            detected_emotion=conv.detected_emotion or "neutral",
            sentiment=conv.sentiment or "neutral",
            created_at=conv.created_at,
            has_audio=conv.audio_path is not None if hasattr(conv, 'audio_path') else True,
            response_duration_seconds=conv.response_duration_seconds if hasattr(conv, 'response_duration_seconds') else None,
            tts_speaker=conv.tts_speaker if hasattr(conv, 'tts_speaker') else None,
        )
        
        # Include audio if requested (expensive)
        if include_audio and hasattr(conv, 'audio_path') and conv.audio_path:
            try:
                item.response_audio_base64 = encode_audio_to_base64(conv.audio_path)
            except Exception:
                pass
        
        items.append(item)
    
    return ConversationHistoryResponse(
        items=items,
        total_count=total_count,
        session_id=session_id,
        has_more=(offset + limit) < total_count
    )


# =============================================================================
# CONVERSATION HISTORY (WITH OPTIONAL AUDIO)
# =============================================================================


@router.get(
    "/sessions/{session_id}/conversations",
    response_model=ConversationHistoryResponse,
    summary="Get session conversation history",
    description="Retrieve conversation history for a session. Use include_audio=true to get audio data."
)
async def get_session_conversations(
    session_id: UUID = Path(..., description="Session UUID"),
    include_audio: bool = Query(False, description="Include audio data (increases response size)"),
    limit: int = Query(50, ge=1, le=100, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get conversation history for a session with optional audio.
    
    **Performance Notes:**
    - Without audio: Fast query (~10ms)
    - With audio: Larger response, includes decompressed base64 audio
    
    **Pagination:**
    Use limit/offset for large sessions.
    """
    from backend.utils.audio import decompress_audio_base64
    from backend.schemas.conversation import ConversationHistoryItem, ConversationHistoryResponse
    
    # Verify session belongs to user
    session = db.query(DBSession).filter(
        DBSession.id == session_id,
        DBSession.user_id == user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Get total count
    total_count = db.query(Conversation).filter(
        Conversation.session_id == session_id
    ).count()
    
    # Query conversations - different columns based on include_audio
    if include_audio:
        # Full query with audio (slower, larger)
        conversations = db.query(Conversation).filter(
            Conversation.session_id == session_id
        ).order_by(Conversation.created_at.asc()).offset(offset).limit(limit).all()
    else:
        # Lightweight query without audio (fast)
        conversations = db.query(
            Conversation.id,
            Conversation.user_input_text,
            Conversation.ai_response_text,
            Conversation.detected_emotion,
            Conversation.sentiment,
            Conversation.created_at,
            Conversation.response_audio_duration_seconds,
            Conversation.tts_prompt,
            Conversation.response_audio_base64.isnot(None).label("has_audio")
        ).filter(
            Conversation.session_id == session_id
        ).order_by(Conversation.created_at.asc()).offset(offset).limit(limit).all()
    
    # Build response items
    items = []
    for conv in conversations:
        # NEW: Check for audio file path instead of base64
        has_audio_file = bool(conv.response_audio_path)
        
        items.append(ConversationHistoryItem(
            id=conv.id,
            user_input_text=conv.user_input_text,
            ai_response_text=conv.ai_response_text,
            detected_emotion=conv.detected_emotion or "neutral",
            sentiment=conv.sentiment or "neutral",
            created_at=conv.created_at,
            has_audio=has_audio_file,  # Based on file path existence
            response_duration_seconds=conv.response_audio_duration_seconds,
            tts_speaker=None,  # Not stored separately
            response_audio_base64=None  # DEPRECATED - audio served via /audio/{id} endpoint
        ))
    
    logger.info(
        f"[{ctx.request_id}] Returned {len(items)} conversations "
        f"(include_audio param ignored - audio served via streaming endpoint)"
    )
    
    return ConversationHistoryResponse(
        items=items,
        total_count=total_count,
        session_id=session_id,
        has_more=(offset + limit) < total_count
    )


@router.get(
    "/conversations/{conversation_id}/audio",
    summary="Get audio for a single conversation",
    description="Retrieve decompressed audio for a specific conversation. Use for on-demand audio loading."
)
async def get_conversation_audio(
    conversation_id: UUID = Path(..., description="Conversation UUID"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get audio for a single conversation (on-demand loading).
    
    This endpoint is optimized for lazy loading - fetch audio only when
    the user clicks play, not when loading the full conversation list.
    
    Returns:
        Opus audio file stream (audio/ogg MIME type)
    """
    from backend.utils.audio import get_audio_file_path
    from fastapi.responses import FileResponse
    
    # Get conversation with ownership check via session
    conversation = db.query(Conversation).join(
        DBSession, Conversation.session_id == DBSession.id
    ).filter(
        Conversation.id == conversation_id,
        DBSession.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Check for audio file path (NEW: file-based storage)
    if not conversation.response_audio_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No audio available for this conversation"
        )
    
    # Get absolute path to audio file
    audio_path = get_audio_file_path(conversation.response_audio_path)
    
    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found on disk"
        )
    
    logger.info(f"[{ctx.request_id}] Streaming audio for conversation {conversation_id}")
    
    # Return Opus audio file with proper MIME type
    return FileResponse(
        path=str(audio_path),
        media_type="audio/ogg",
        filename=f"{conversation_id}.opus",
        headers={
            "Content-Disposition": f"inline; filename={conversation_id}.opus",
            "Cache-Control": "public, max-age=31536000",  # Cache for 1 year
        }
    )


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================


@router.get(
    "/sessions",
    response_model=ListSessionsResponse,
    summary="List user's chat sessions",
    description="Get list of all chat sessions for current user."
)
async def list_sessions(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    ctx: RequestContext = Depends(get_request_context),
):
    """List all chat sessions for current user."""
    # Use subquery to get first conversation per session in ONE query
    # This fixes the N+1 query problem
    from sqlalchemy import func
    from sqlalchemy.orm import aliased
    
    # Get sessions
    sessions = db.query(DBSession).filter(
        DBSession.user_id == user.id
    ).order_by(desc(DBSession.session_start)).limit(limit).all()
    
    if not sessions:
        return ListSessionsResponse(sessions=[], total_count=0)
    
    # Get all session IDs
    session_ids = [s.id for s in sessions]
    
    # Subquery to find earliest conversation per session
    earliest_conv_subq = db.query(
        Conversation.session_id,
        func.min(Conversation.created_at).label('earliest_created')
    ).filter(
        Conversation.session_id.in_(session_ids)
    ).group_by(Conversation.session_id).subquery()
    
    # Join to get actual first conversations in ONE query
    first_convs = db.query(Conversation).join(
        earliest_conv_subq,
        (Conversation.session_id == earliest_conv_subq.c.session_id) &
        (Conversation.created_at == earliest_conv_subq.c.earliest_created)
    ).all()
    
    # Build lookup dict
    first_conv_map = {conv.session_id: conv for conv in first_convs}
    
    # Build response
    session_infos = []
    for s in sessions:
        first_conv = first_conv_map.get(s.id)
        title = None
        if first_conv and first_conv.user_input_text:
            text = first_conv.user_input_text
            title = text[:50] + "..." if len(text) > 50 else text
        
        session_infos.append(SessionInfo(
            session_id=s.id,
            user_id=s.user_id,
            created_at=s.session_start,
            last_activity=s.session_end or s.session_start,
            message_count=s.message_count,
            title=title
        ))
    
    return ListSessionsResponse(
        sessions=session_infos,
        total_count=len(session_infos)
    )


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new chat session",
    description="Create a new chat session for the current user."
)
async def create_session(
    request: Optional[CreateSessionRequest] = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    ctx: RequestContext = Depends(get_request_context),
):
    """Create new chat session."""
    new_session = DBSession(
        user_id=user.id,
        is_active=True,
        message_count=0,
        session_context=request.initial_context if request else {},
    )
    
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    
    logger.info(f"[{ctx.request_id}] Created session: {new_session.id}")
    
    return CreateSessionResponse(
        session_id=new_session.id,
        created_at=new_session.session_start,
        message="Session created successfully"
    )
