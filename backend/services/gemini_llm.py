"""
GuppShupp Gemini LLM Service Module
backend/services/gemini_llm.py

PRODUCTION-GRADE LLM SERVICE FOR GUPPSHUPP
- Strictly adheres to Google Gen AI Python SDK (google.genai, google.genai.types)
- Structured JSON output for database integration
- Rich context aggregation: transcript + acoustic + memory + personality
- No restrictions on LLM value generation (emotion/sentiment/intent are free-form)
- Safe by default, enforced by design

References:
- Google Gen AI SDK: https://googleapis.github.io/python-genai
- GuppShupp Spec: GUPPSHUPP_COMPREHENSIVE_SPECIFICATION.docx (authority)
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging

from google import genai
from google.genai import types
from google.genai import errors as genai_errors


# ============================================================================
# DATA MODELS FOR STRUCTURED OUTPUT
# ============================================================================

@dataclass
class MemoryUpdate:
    """Proposed memory to store from this conversation turn."""
    type: str  # "long_term" | "episodic" | "semantic"
    text: str
    category: str  # "work_study", "relationships", "health", "emotional", etc.
    importance: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyFlags:
    """Safety context detected by LLM (pre-guardrail info)."""
    crisis_risk: str  # "low" | "medium" | "high"
    self_harm_mentioned: bool
    abuse_mentioned: bool
    medical_concern: bool
    flagged_keywords: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeminiLLMResponse:
    """
    Structured response from Gemini LLM.
    Maps directly to CONVERSATIONS table columns.
    """
    # Core response
    response_text: str
    response_language: str  # e.g., "hi-en" (Hinglish), "en", "hi"
    
    # Emotion/sentiment/intent (detected by LLM, free-form values)
    detected_emotion: str  # e.g., "sadness", "joy", "anxious", "frustrated"
    emotion_confidence: float  # 0.0 to 1.0
    sentiment: str  # e.g., "negative", "positive", "neutral", "mixed"
    detected_intent: str  # e.g., "expressing_emotion", "greeting", "seeking_help"
    intent_confidence: float  # 0.0 to 1.0
    
    # TTS integration
    tts_style_prompt: str  # MUST start with "speaks" (no speaker name)
    tts_speaker: str  # e.g., "Rohit" for Hindi, "Thoma" for English
    
    # Voice preference tracking (NEW)
    voice_change_requested: bool  # True if user wants voice change
    preferred_speaker_gender: str  # "male", "female", or "any"
    
    # Memory operations (proposed by LLM)
    memory_updates: List[MemoryUpdate]
    
    # Safety pre-check (before dedicated guardrail layer)
    safety_flags: SafetyFlags
    
    # Latency tracking
    generation_time_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for database storage."""
        return {
            "response_text": self.response_text,
            "response_language": self.response_language,
            "detected_emotion": self.detected_emotion,
            "emotion_confidence": self.emotion_confidence,
            "sentiment": self.sentiment,
            "detected_intent": self.detected_intent,
            "intent_confidence": self.intent_confidence,
            "tts_prompt": self.tts_style_prompt,
            "tts_speaker": self.tts_speaker,
            "voice_change_requested": self.voice_change_requested,
            "preferred_speaker_gender": self.preferred_speaker_gender,
            "memory_updates": [m.to_dict() for m in self.memory_updates],
            "safety_flags": self.safety_flags.to_dict(),
            "response_generation_time_ms": self.generation_time_ms,
        }


# ============================================================================
# RESPONSE JSON SCHEMA (GROUND TRUTH)
# ============================================================================

GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "response_text": {
            "type": "string",
            "description": "Main conversational response to user, emotionally attuned"
        },
        "response_language": {
            "type": "string",
            "description": "Best response language code (validate from transcript, don't blindly trust ASR). Examples: 'hi', 'en', 'ta', 'bn', 'hi-en' for code-mixing"
        },
        "detected_emotion": {
            "type": "string",
            "description": "Primary emotion (free-form label, e.g., 'sadness', 'joy', 'anxious')"
        },
        "emotion_confidence": {
            "type": "number",
            "description": "Confidence in emotion detection (0.0 to 1.0)",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "sentiment": {
            "type": "string",
            "description": "Overall sentiment (free-form, e.g., 'positive', 'negative', 'neutral')"
        },
        "detected_intent": {
            "type": "string",
            "description": "User intent (free-form label, e.g., 'expressing_emotion', 'greeting')"
        },
        "intent_confidence": {
            "type": "number",
            "description": "Confidence in intent detection (0.0 to 1.0)",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "tts_style_prompt": {
            "type": "string",
            "description": "TTS description MUST start with 'speaks' (lowercase, no speaker name). Describe pace, pitch, tone, emotion, environment. Example: 'speaks at a moderate pace with a warm, empathetic tone in a close-sounding environment with clear audio quality.'"
        },
        "tts_speaker": {
            "type": "string",
            "description": "Speaker name from available speakers for response_language. Consider: 1) Current session speaker if same language, 2) User's gender preference, 3) Voice change request. Examples: 'Rohit', 'Divya', 'Thoma', 'Mary'"
        },
        "voice_change_requested": {
            "type": "boolean",
            "description": "True if user expressed dislike for current voice or requested change (e.g., 'change your voice', 'I don't like this voice', 'talk differently')"
        },
        "preferred_speaker_gender": {
            "type": "string",
            "enum": ["male", "female", "any"],
            "description": "User's speaker gender preference if mentioned (e.g., 'I want to talk to a girl' -> 'female'), otherwise 'any'"
        },
        "memory_updates": {
            "type": "array",
            "description": "Proposed memories to extract and store. IMPORTANT: Extract facts, preferences, emotional patterns, voice preferences",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["long_term", "episodic", "semantic"]
                    },
                    "text": {
                        "type": "string",
                        "description": "Memory fact or observation"
                    },
                    "category": {
                        "type": "string",
                        "description": "Memory category: personal_info, preferences, relationships, work_study, health, emotional, voice_preference, etc."
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["type", "text", "category", "importance"]
            }
        },
        "safety_flags": {
            "type": "object",
            "description": "Safety context detected (pre-guardrail)",
            "properties": {
                "crisis_risk": {
                    "type": "string",
                    "enum": ["low", "medium", "high"]
                },
                "self_harm_mentioned": {"type": "boolean"},
                "abuse_mentioned": {"type": "boolean"},
                "medical_concern": {"type": "boolean"},
                "flagged_keywords": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["crisis_risk", "self_harm_mentioned", "abuse_mentioned", "medical_concern", "flagged_keywords"]
        }
    },
    "required": [
        "response_text", "response_language", "detected_emotion", "emotion_confidence",
        "sentiment", "detected_intent", "intent_confidence", "tts_style_prompt",
        "tts_speaker", "voice_change_requested", "preferred_speaker_gender",
        "memory_updates", "safety_flags"
    ]
}


# ============================================================================
# INDIC PARLER TTS SPEAKER MAPPING (COMPLETE - ALL 18 LANGUAGES)
# Source: https://huggingface.co/ai4bharat/indic-parler-tts
# ============================================================================

TTS_SPEAKER_MAP = {
    # ===== ASSAMESE (as) =====
    "as": {
        "all": ["Amit", "Sita", "Poonam", "Rakesh"],
        "male": ["Amit", "Rakesh"],
        "female": ["Sita", "Poonam"],
        "recommended": ["Amit", "Sita"],
    },
    # ===== BENGALI (bn) =====
    "bn": {
        "all": ["Arjun", "Aditi", "Tapan", "Rashmi", "Arnav", "Riya"],
        "male": ["Arjun", "Tapan", "Arnav"],
        "female": ["Aditi", "Rashmi", "Riya"],
        "recommended": ["Arjun", "Aditi"],
    },
    # ===== BODO (brx) =====
    "brx": {
        "all": ["Bikram", "Maya", "Kalpana"],
        "male": ["Bikram"],
        "female": ["Maya", "Kalpana"],
        "recommended": ["Bikram", "Maya"],
    },
    # ===== CHHATTISGARHI (hne) =====
    "hne": {
        "all": ["Bhanu", "Champa"],
        "male": ["Bhanu"],
        "female": ["Champa"],
        "recommended": ["Bhanu", "Champa"],
    },
    # ===== DOGRI (doi) =====
    "doi": {
        "all": ["Karan"],
        "male": ["Karan"],
        "female": [],
        "recommended": ["Karan"],
    },
    # ===== ENGLISH (en) =====
    "en": {
        "all": ["Thoma", "Mary", "Swapna", "Dinesh", "Meera", "Jatin", "Aakash", 
                "Sneha", "Kabir", "Tisha", "Chingkhei", "Thoiba", "Priya", "Tarun", 
                "Gauri", "Nisha", "Raghav", "Kavya", "Ravi", "Vikas", "Riya"],
        "male": ["Thoma", "Dinesh", "Jatin", "Aakash", "Kabir", "Thoiba", 
                 "Tarun", "Raghav", "Ravi", "Vikas"],
        "female": ["Mary", "Swapna", "Meera", "Sneha", "Tisha", "Chingkhei", 
                   "Priya", "Gauri", "Nisha", "Kavya", "Riya"],
        "recommended": ["Thoma", "Mary"],
    },
    # ===== GUJARATI (gu) =====
    "gu": {
        "all": ["Yash", "Neha"],
        "male": ["Yash"],
        "female": ["Neha"],
        "recommended": ["Yash", "Neha"],
    },
    # ===== HINDI (hi) =====
    "hi": {
        "all": ["Rohit", "Divya", "Aman", "Rani"],
        "male": ["Rohit", "Aman"],
        "female": ["Divya", "Rani"],
        "recommended": ["Rohit", "Divya"],
    },
    # ===== KANNADA (kn) =====
    "kn": {
        "all": ["Suresh", "Anu", "Chetan", "Vidya"],
        "male": ["Suresh", "Chetan"],
        "female": ["Anu", "Vidya"],
        "recommended": ["Suresh", "Anu"],
    },
    # ===== MALAYALAM (ml) =====
    "ml": {
        "all": ["Anjali", "Anju", "Harish"],
        "male": ["Harish"],
        "female": ["Anjali", "Anju"],
        "recommended": ["Anjali", "Harish"],
    },
    # ===== MANIPURI / MEITEI (mni) =====
    "mni": {
        "all": ["Laishram", "Ranjit"],
        "male": ["Laishram", "Ranjit"],
        "female": [],
        "recommended": ["Laishram", "Ranjit"],
    },
    # ===== MARATHI (mr) =====
    "mr": {
        "all": ["Sanjay", "Sunita", "Nikhil", "Radha", "Varun", "Isha"],
        "male": ["Sanjay", "Nikhil", "Varun"],
        "female": ["Sunita", "Radha", "Isha"],
        "recommended": ["Sanjay", "Sunita"],
    },
    # ===== NEPALI (ne) =====
    "ne": {
        "all": ["Amrita"],
        "male": [],
        "female": ["Amrita"],
        "recommended": ["Amrita"],
    },
    # ===== ODIA (or) =====
    "or": {
        "all": ["Manas", "Debjani"],
        "male": ["Manas"],
        "female": ["Debjani"],
        "recommended": ["Manas", "Debjani"],
    },
    # ===== PUNJABI (pa) =====
    "pa": {
        "all": ["Divjot", "Gurpreet"],
        "male": ["Divjot"],
        "female": ["Gurpreet"],
        "recommended": ["Divjot", "Gurpreet"],
    },
    # ===== SANSKRIT (sa) =====
    "sa": {
        "all": ["Aryan"],
        "male": ["Aryan"],
        "female": [],
        "recommended": ["Aryan"],
    },
    # ===== TAMIL (ta) =====
    "ta": {
        "all": ["Kavitha", "Jaya"],
        "male": [],
        "female": ["Kavitha", "Jaya"],
        "recommended": ["Jaya"],
    },
    # ===== TELUGU (te) =====
    "te": {
        "all": ["Prakash", "Lalitha", "Kiran"],
        "male": ["Prakash", "Kiran"],
        "female": ["Lalitha"],
        "recommended": ["Prakash", "Lalitha"],
    },
}


def get_speaker_for_language(
    language: str,
    gender_preference: str = "any",
    current_speaker: str = None,
    avoid_speakers: list = None,
) -> str:
    """
    Get appropriate TTS speaker based on language and preferences.
    
    Args:
        language: ISO language code (hi, en, ta, etc.)
        gender_preference: "male", "female", or "any"
        current_speaker: Current session speaker (to maintain consistency)
        avoid_speakers: List of speakers user doesn't want
        
    Returns:
        Speaker name string
    """
    import random
    
    # Normalize language code
    lang = language.lower().split("-")[0]  # "hi-en" -> "hi"
    
    # Get language speakers or fallback to Hindi
    lang_speakers = TTS_SPEAKER_MAP.get(lang, TTS_SPEAKER_MAP.get("hi"))
    
    # If current speaker is valid for this language, keep it
    if current_speaker and current_speaker in lang_speakers["all"]:
        if not avoid_speakers or current_speaker not in avoid_speakers:
            return current_speaker
    
    # Get candidates based on gender preference
    if gender_preference == "male" and lang_speakers["male"]:
        candidates = lang_speakers["male"]
    elif gender_preference == "female" and lang_speakers["female"]:
        candidates = lang_speakers["female"]
    else:
        candidates = lang_speakers["recommended"] or lang_speakers["all"]
    
    # Remove avoided speakers
    if avoid_speakers:
        candidates = [s for s in candidates if s not in avoid_speakers]
    
    # Fallback if no candidates left
    if not candidates:
        candidates = lang_speakers["all"]
    
    return random.choice(candidates) if candidates else "Rohit"


# ============================================================================
# GEMINI LLM SERVICE
# ============================================================================

logger = logging.getLogger(__name__)


class GeminiLLMService:
    """
    Core LLM service for GuppShupp.
    
    Responsibilities:
    1. Aggregate rich context (transcript + acoustic + memory + personality)
    2. Generate structured response via Gemini 2.0 Flash
    3. Extract emotion, sentiment, intent (LLM-determined, unconstrained)
    4. Propose memories for storage
    5. Provide TTS styling guidance
    6. Pre-check safety (crisis, self-harm, abuse)
    
    Uses Google Gen AI SDK (official, production-ready).
    No restrictions on LLM value generation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = "gemini-2.5-flash",
        vertexai: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
        http_options: Optional[types.HttpOptions] = None,
    ):
        """
        Initialize Gemini LLM client.
        
        Args:
            api_key: GOOGLE_API_KEY (env var if None)
            model: Model name (gemini-2.0-flash-exp or gemini-2.5-flash)
            vertexai: Use Vertex AI backend
            project: GCP project (for Vertex AI)
            location: GCP region (for Vertex AI)
            http_options: HTTP configuration (timeout, API version, etc.)
        
        SDK Reference:
            - Client init: https://googleapis.github.io/python-genai#client-initialization
            - API versions: v1, v1alpha (Gemini API); v1, v1alpha (Vertex)
        """
        self._client = genai.Client(
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            http_options=http_options,
        )
        self._model = model
        logger.info(f"GeminiLLMService initialized with model={model}")

    def _attempt_json_repair(self, broken_json: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair broken JSON by completing truncated strings/objects.
        """
        import re
        
        try:
            # Try to find where the JSON breaks
            # If it's an unterminated string, try to close it
            if broken_json.count('"') % 2 != 0:
                # Odd number of quotes - unterminated string
                broken_json = broken_json + '"'
            
            # Count braces
            open_braces = broken_json.count('{')
            close_braces = broken_json.count('}')
            if open_braces > close_braces:
                broken_json = broken_json + ('}' * (open_braces - close_braces))
            
            # Try parsing again
            return json.loads(broken_json)
        except:
            return None


    def _call_gemini_with_retry(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Call Gemini with automatic retry on JSON parsing failures.
        Guarantees valid response or fallback.
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Gemini API call attempt {attempt + 1}/{max_retries}")
                
                # Make API call
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                
                # Extract response text (try multiple methods)
                full_text = None
                try:
                    full_text = resp.candidates[0].content.parts[0].text
                except (IndexError, AttributeError):
                    try:
                        full_text = resp.text
                    except:
                        pass
                
                if not full_text:
                    logger.warning(f"Attempt {attempt + 1}: Empty response")
                    continue
                
                logger.debug(f"Raw response length: {len(full_text)} chars")
                
                # Clean response
                cleaned_text = self._clean_json_response(full_text)
                
                # Try parsing
                try:
                    response_dict = json.loads(cleaned_text)
                    
                    # Validate structure
                    if self._validate_response_structure(response_dict):
                        logger.info(f"Valid response received on attempt {attempt + 1}")
                        return response_dict
                    else:
                        logger.warning(f"Attempt {attempt + 1}: Incomplete structure")
                        continue
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Attempt {attempt + 1}: JSON parse failed - {e}")
                    
                    # Try to repair the JSON
                    repaired = self._attempt_json_repair(cleaned_text)
                    if repaired and self._validate_response_structure(repaired):
                        logger.info(f"JSON repaired successfully on attempt {attempt + 1}")
                        return repaired
                    
                    # Log the problematic response
                    logger.debug(f"Problematic JSON (first 1000 chars): {cleaned_text[:1000]}")
                    last_error = e
                    continue
            
            except genai_errors.APIError as e:
                logger.error(f"Attempt {attempt + 1}: API error [{e.code}]: {e.message}")
                last_error = e
                
                # Don't retry on quota/auth errors
                if e.code in [429, 401, 403]:
                    break
                
                # Wait before retry
                import time
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Unexpected error - {e}")
                last_error = e
                continue
        
        # All retries failed - use fallback
        logger.error(f"All {max_retries} attempts failed. Using fallback response.")
        if last_error:
            logger.error(f"Last error: {last_error}")
        
        return self._fallback_response()


    async def _call_gemini_with_retry_async(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Async version with retry logic."""
        import asyncio
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Gemini async call attempt {attempt + 1}/{max_retries}")
                
                resp = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                
                # Extract response
                full_text = None
                try:
                    full_text = resp.candidates[0].content.parts[0].text
                except (IndexError, AttributeError):
                    try:
                        full_text = resp.text
                    except:
                        pass
                
                if not full_text:
                    logger.warning(f"Attempt {attempt + 1}: Empty response")
                    continue
                
                cleaned_text = self._clean_json_response(full_text)
                
                try:
                    response_dict = json.loads(cleaned_text)
                    
                    if self._validate_response_structure(response_dict):
                        logger.info(f"Valid response on attempt {attempt + 1}")
                        return response_dict
                    else:
                        logger.warning(f"Attempt {attempt + 1}: Incomplete structure")
                        continue
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Attempt {attempt + 1}: JSON parse failed - {e}")
                    
                    repaired = self._attempt_json_repair(cleaned_text)
                    if repaired and self._validate_response_structure(repaired):
                        logger.info(f"JSON repaired on attempt {attempt + 1}")
                        return repaired
                    
                    logger.debug(f"Problematic JSON: {cleaned_text[:1000]}")
                    last_error = e
                    continue
            
            except genai_errors.APIError as e:
                logger.error(f"Attempt {attempt + 1}: API error - {e.message}")
                last_error = e
                
                if e.code in [429, 401, 403]:
                    break
                
                await asyncio.sleep(1 * (attempt + 1))
                continue
            
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Unexpected error - {e}")
                last_error = e
                continue
        
        logger.error(f"All {max_retries} attempts failed. Using fallback.")
        return self._fallback_response()
    
    def analyze_and_respond(
        self,
        transcript: str,
        language: str,
        acoustic_features: Dict[str, Any],
        short_term_context: List[Dict[str, Any]],
        long_term_memories: List[Dict[str, Any]],
        episodic_memories: List[Dict[str, Any]],
        session_context: Dict[str, Any],  # Contains: current_tts_speaker, session_language, voice_preferences
        safety_context: Dict[str, Any],
        *,
        temperature: float = 0.7,
        max_output_tokens: int = 2000,
    ) -> GeminiLLMResponse:
        """
        Main entry point: aggregate context, call Gemini, parse structured response.
        
        GUARANTEED TO RETURN A VALID RESPONSE - never crashes.
        Uses retry logic with automatic JSON repair and fallback.
        
        Args:
            transcript: User's transcribed speech text
            language: Detected language code (e.g., "hi", "en", "hi-en")
            acoustic_features: JSON from emotion_detection.py (prosody, pitch, energy, etc.)
            short_term_context: Last N conversations from current session
            long_term_memories: Retrieved facts/preferences (IndicBERT semantic search)
            episodic_memories: Past emotional summaries with context
            character_profile: Aarav personality (name, background, traits, speech style)
            safety_context: Flags from previous turns (e.g., recent crisis mention)
            temperature: 0.0 to 1.0 (lower = more deterministic)
            max_output_tokens: Token limit for response
            
        Returns:
            GeminiLLMResponse with structured output (guaranteed valid)
        """
        import time
        start_time = time.time()

        # Build comprehensive prompt
        prompt = self._build_prompt(
            transcript=transcript,
            language=language,
            acoustic_features=acoustic_features,
            short_term_context=short_term_context,
            long_term_memories=long_term_memories,
            episodic_memories=episodic_memories,
            session_context=session_context,
            safety_context=safety_context,
        )

        # Build Gemini config with structured JSON output
        config = types.GenerateContentConfig(
            system_instruction=self._build_system_instruction(session_context),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=GEMINI_RESPONSE_SCHEMA,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
            ],
        )

        try:
            logger.info(f"Calling Gemini {self._model} for analysis")
            
            # Use retry logic - GUARANTEED to return valid response
            response_dict = self._call_gemini_with_retry(
                prompt=prompt,
                config=config,
                max_retries=3,
            )
            
            # Convert to typed response
            generation_time = int((time.time() - start_time) * 1000)
            structured_response = self._parse_gemini_response(response_dict, generation_time)
            
            logger.info(f"Response ready ({generation_time}ms)")
            return structured_response

        except Exception as e:
            # Ultimate fallback - should NEVER happen with retry logic
            logger.critical(f"CRITICAL: Fallback triggered after retry logic: {e}", exc_info=True)
            generation_time = int((time.time() - start_time) * 1000)
            return self._parse_gemini_response(self._fallback_response(), generation_time)

    def _validate_response_structure(self, response_dict: Dict[str, Any]) -> bool:
        """Check if response has all required keys."""
        required_keys = [
            "response_text", "response_language", "detected_emotion", 
            "emotion_confidence", "sentiment", "detected_intent",
            "intent_confidence", "tts_style_prompt", "tts_speaker",
            "voice_change_requested", "preferred_speaker_gender",
            "memory_updates", "safety_flags"
        ]
        return all(key in response_dict for key in required_keys)


    async def analyze_and_respond_async(
        self,
        transcript: str,
        language: str,
        acoustic_features: Dict[str, Any],
        short_term_context: List[Dict[str, Any]],
        long_term_memories: List[Dict[str, Any]],
        episodic_memories: List[Dict[str, Any]],
        session_context: Dict[str, Any],  # Contains: current_tts_speaker, session_language, voice_preferences
        safety_context: Dict[str, Any],
        *,
        temperature: float = 0.7,
        max_output_tokens: int = 2000,
    ) -> GeminiLLMResponse:
        """
        Async variant using client.aio.
        
        GUARANTEED TO RETURN A VALID RESPONSE - never crashes.
        Uses retry logic with automatic JSON repair and fallback.
        """
        import time
        start_time = time.time()

        prompt = self._build_prompt(
            transcript=transcript,
            language=language,
            acoustic_features=acoustic_features,
            short_term_context=short_term_context,
            long_term_memories=long_term_memories,
            episodic_memories=episodic_memories,
            session_context=session_context,
            safety_context=safety_context,
        )

        config = types.GenerateContentConfig(
            system_instruction=self._build_system_instruction(session_context),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=GEMINI_RESPONSE_SCHEMA,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
            ],
        )

        try:
            logger.info(f"Calling Gemini {self._model} (async)")
            
            # Use async retry logic - GUARANTEED to return valid response
            response_dict = await self._call_gemini_with_retry_async(
                prompt=prompt,
                config=config,
                max_retries=3,
            )
            
            generation_time = int((time.time() - start_time) * 1000)
            structured_response = self._parse_gemini_response(response_dict, generation_time)
            
            logger.info(f"Async response ready ({generation_time}ms)")
            return structured_response

        except Exception as e:
            logger.critical(f"CRITICAL: Fallback after async retry: {e}", exc_info=True)
            generation_time = int((time.time() - start_time) * 1000)
            return self._parse_gemini_response(self._fallback_response(), generation_time)


    @staticmethod
    def _clean_json_response(text: str) -> str:
        """
        Clean Gemini response text to extract valid JSON.
        Handles markdown code blocks and other formatting issues.
        """
        text = text.strip()
        
        # Remove markdown code blocks (`````` or ``````)
        if text.startswith("```"):
            # Find the first newline after ```
            first_newline = text.find('\n')
            if first_newline != -1:
                text = text[first_newline + 1:]
            # Remove trailing ```
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        
        return text

    # ========================================================================
    # PROMPT ENGINEERING & SYSTEM INSTRUCTIONS
    # ========================================================================

    def _build_system_instruction(self, session_context: Dict[str, Any]) -> str:
        """
        Build system instruction for Guppu AI companion.
        Includes dynamic TTS speaker selection rules.
        
        Args:
            session_context: Contains current_tts_speaker, available_speakers, user preferences
        """
        current_speaker = session_context.get("current_tts_speaker", None)
        session_lang = session_context.get("session_language", "auto")
        voice_prefs = session_context.get("voice_preferences", {})
        
        # Build available speakers info
        speakers_info = json.dumps(TTS_SPEAKER_MAP, indent=2, ensure_ascii=False)
        
        return f"""You are GUPPU (गप्पू), an emotionally intelligent AI voice companion for Indian youth.

═══════════════════════════════════════════════════════════════════════════════
CORE IDENTITY
═══════════════════════════════════════════════════════════════════════════════
• Name: Guppu (गप्पू) - meaning "chatterbox", a friendly companion
• Role: Empathetic voice assistant for emotional support and conversation
• Style: Warm, conversational, validates emotions, culturally aware
• Languages: All major Indian languages + English (18 languages supported)

═══════════════════════════════════════════════════════════════════════════════
TTS SPEAKER SELECTION - CRITICAL RULES
═══════════════════════════════════════════════════════════════════════════════

1. RESPONSE LANGUAGE VALIDATION:
   - Analyze the transcript to determine the BEST response language
   - Do NOT blindly trust the detected language from ASR
   - Consider: transcript content, user's previous language, code-mixing patterns
   - Set response_language accurately (hi, en, ta, bn, hi-en, etc.)

2. SPEAKER SELECTION based on response_language:
   - Available speakers per language are in TTS_SPEAKER_MAP
   - Current session speaker: {current_speaker or "None (first turn)"}
   - Session language: {session_lang}

3. SPEAKER CONSISTENCY RULES:
   - If SAME language as previous turn → KEEP the same speaker
   - Only change speaker if:
     a) Language changed (new language → pick from that language's speakers)
     b) User requested voice change
     c) User specified gender preference

4. VOICE CHANGE DETECTION:
   Set voice_change_requested = true if user says:
   - "I don't like your voice", "change your voice", "talk differently"
   - "Your voice is irritating", "speak in a different way"
   - "Can I talk to someone else?", "switch the voice"
   
   Set preferred_speaker_gender accordingly:
   - "I want to talk to a girl/woman/female" → "female"
   - "I want to talk to a boy/man/male" → "male"
   - "Can I speak with a female voice?" → "female"
   - No preference mentioned → "any"

5. TTS DESCRIPTION FORMAT - MUST FOLLOW:
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ tts_style_prompt MUST start with "speaks" (lowercase)                    │
   │ NO speaker name in the description                                       │
   │ Keep it CONCISE (1-2 sentences)                                          │
   │ Describe: pace, pitch, tone, emotion, environment, quality               │
   └──────────────────────────────────────────────────────────────────────────┘
   
   ✅ GOOD EXAMPLES:
   • "speaks at a moderate pace with a warm, empathetic tone in a close-sounding environment with clear audio quality."
   • "speaks slowly with a calm, soothing voice, captured clearly with minimal background noise."
   • "speaks at a fast pace with enthusiasm and energy in a close recording environment."
   • "speaks with a slightly higher pitch, conveying gentle concern in a clear, intimate recording."
   • "speaks at a normal pace with a cheerful, positive tone in high-quality audio."
   
   ❌ BAD EXAMPLES (NEVER DO THIS):
   • "Rohit speaks at a..." (includes speaker name)
   • "The speaker talks..." (doesn't start with "speaks")
   • "A warm voice..." (doesn't start with "speaks")

═══════════════════════════════════════════════════════════════════════════════
MEMORY EXTRACTION - IMPORTANT
═══════════════════════════════════════════════════════════════════════════════
Extract and store valuable information as memories:
- Personal info: name, age, location, occupation
- Preferences: voice preference, language preference, topics of interest
- Emotional patterns: recurring moods, triggers, coping mechanisms
- Relationships: family, friends, significant others
- Work/Study: job, education, stress factors
- Health: mentioned health concerns (without diagnosing)

Categories: personal_info, preferences, relationships, work_study, health, emotional, voice_preference

═══════════════════════════════════════════════════════════════════════════════
SAFETY GUARDRAILS
═══════════════════════════════════════════════════════════════════════════════
- Recognize crisis signals (self-harm, abuse, severe distress)
- For crisis: Set crisis_risk="high", provide helpline numbers
- Avoid medical diagnosis; suggest professional consultation
- Do NOT validate destructive behaviors
- Offer supportive alternatives and resources

═══════════════════════════════════════════════════════════════════════════════
LANGUAGE & CULTURAL SENSITIVITY
═══════════════════════════════════════════════════════════════════════════════
- Respond in user's preferred language or code-mix (Hinglish, Tanglish, etc.)
- Understand Indian cultural context: family dynamics, exam pressure, career expectations
- Use natural, conversational tone appropriate to age and background
- Validate feelings first, then provide perspective if appropriate

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════
Always respond with the EXACT JSON structure specified.
Never deviate from the schema.
Values are FREE-FORM (not restricted to predefined lists).

NOW ANALYZE THE USER'S INPUT AND GENERATE YOUR STRUCTURED RESPONSE."""

    def _build_prompt(
        self,
        transcript: str,
        language: str,
        acoustic_features: Dict[str, Any],
        short_term_context: List[Dict[str, Any]],
        long_term_memories: List[Dict[str, Any]],
        episodic_memories: List[Dict[str, Any]],
        session_context: Dict[str, Any],  # Renamed from character_profile
        safety_context: Dict[str, Any],
    ) -> str:
        """
        Build rich context prompt for Gemini.
        Aggregates all available context in a structured format.
        """
        
        # Format acoustic features
        acoustic_json = json.dumps(acoustic_features, indent=2, ensure_ascii=False)
        
        # Format short-term context (last N turns)
        short_term_str = ""
        if short_term_context:
            short_term_str = "\nSHORT-TERM CONTEXT (Recent conversation):\n"
            for turn in short_term_context[-5:]:  # Last 5 turns
                user_text = turn.get("user_input_text", "")
                ai_text = turn.get("ai_response_text", "")
                emotion = turn.get("detected_emotion", "unknown")
                tts_speaker = turn.get("tts_speaker", "unknown")
                short_term_str += f"- User: {user_text} [{emotion}]\n  AI ({tts_speaker}): {ai_text}\n"
        
        # Format long-term memories
        long_term_str = ""
        if long_term_memories:
            long_term_str = "\nLONG-TERM MEMORIES (Facts, preferences, triggers):\n"
            for mem in long_term_memories:
                text = mem.get("memory_text", "")
                importance = mem.get("importance_score", 0)
                category = mem.get("category", "general")
                long_term_str += f"- [{category}] {text} (importance: {importance})\n"
        
        # Format episodic memories
        episodic_str = ""
        if episodic_memories:
            episodic_str = "\nEPISODIC MEMORIES (Past emotional arcs, summaries):\n"
            for mem in episodic_memories:
                text = mem.get("memory_text", "")
                emotion_tone = mem.get("emotional_tone", "neutral")
                episodic_str += f"- {text} (tone: {emotion_tone})\n"
        
        # Format safety context
        safety_str = ""
        if safety_context:
            crisis = safety_context.get("crisis_risk", "low")
            recent_flags = safety_context.get("recent_flags", [])
            safety_str = f"\nSAFETY CONTEXT:\n- Crisis risk level: {crisis}\n"
            if recent_flags:
                safety_str += f"- Recent flags: {', '.join(recent_flags)}\n"
        
        # Format session context for TTS speaker selection
        current_speaker = session_context.get("current_tts_speaker", "None")
        session_lang = session_context.get("session_language", "auto")
        voice_prefs = session_context.get("voice_preferences", {})
        
        session_str = f"""
SESSION CONTEXT:
- Current TTS Speaker: {current_speaker}
- Session Language: {session_lang}
- User Voice Preferences: {json.dumps(voice_prefs) if voice_prefs else "None specified"}"""
        
        return f"""USER INPUT ANALYSIS
================

TRANSCRIPT:
{transcript}

DETECTED LANGUAGE (from ASR): {language}
(Note: Validate this - choose the BEST response language based on transcript content)

ACOUSTIC FEATURES (librosa/OpenSMILE):
{acoustic_json}{short_term_str}{long_term_str}{episodic_str}{safety_str}{session_str}

TASK:
Analyze the user's input using ALL available context.
Generate a response that is:
1. Emotionally intelligent (validate, empathize, understand nuance)
2. Contextually aware (reference past conversations, memories)
3. In Guppu's warm, friendly style
4. Safe by default (pre-check for crisis, abuse, medical risks)
5. Memory-extractive (propose new facts to remember, especially voice preferences)
6. TTS-optimized (description starting with "speaks", select appropriate speaker)

SPEAKER SELECTION:
- If same language as before and no voice change requested → keep current speaker: {current_speaker}
- If language changed or voice change requested → select new speaker from TTS_SPEAKER_MAP
- Honor gender preference if user mentioned it

EMOTION/SENTIMENT/INTENT GUIDANCE (use these or free-form alternatives):
- Emotions: joy, sadness, anger, fear, surprise, disgust, neutral, anxious, frustrated, calm, overwhelmed
- Intents: greeting, question, complaint, request, expressing_emotion, crisis_signal, venting, seeking_advice
- Sentiments: positive, negative, neutral, mixed

Respond with ONLY the JSON object."""

    # ========================================================================
    # RESPONSE PARSING & FALLBACK
    # ========================================================================

    def _parse_gemini_response(
        self,
        response_dict: Dict[str, Any],
        generation_time: int,
    ) -> GeminiLLMResponse:
        """
        Parse Gemini JSON response into typed GeminiLLMResponse.
        
        The response_dict has already been validated against GEMINI_RESPONSE_SCHEMA,
        but we perform additional type coercion and safety checks.
        """
        
        # Parse memory updates
        memory_updates = []
        for mem_dict in response_dict.get("memory_updates", []):
            try:
                memory_updates.append(MemoryUpdate(
                    type=mem_dict.get("type", "long_term"),
                    text=mem_dict.get("text", ""),
                    category=mem_dict.get("category", "other"),
                    importance=float(mem_dict.get("importance", 0.5)),
                ))
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse memory: {e}")
        
        # Parse safety flags
        safety_dict = response_dict.get("safety_flags", {})
        safety_flags = SafetyFlags(
            crisis_risk=safety_dict.get("crisis_risk", "low"),
            self_harm_mentioned=safety_dict.get("self_harm_mentioned", False),
            abuse_mentioned=safety_dict.get("abuse_mentioned", False),
            medical_concern=safety_dict.get("medical_concern", False),
            flagged_keywords=safety_dict.get("flagged_keywords", []),
        )
        
        # Construct response
        return GeminiLLMResponse(
            response_text=response_dict.get("response_text", ""),
            response_language=response_dict.get("response_language", "en"),
            detected_emotion=response_dict.get("detected_emotion", "neutral"),
            emotion_confidence=float(response_dict.get("emotion_confidence", 0.0)),
            sentiment=response_dict.get("sentiment", "neutral"),
            detected_intent=response_dict.get("detected_intent", "unknown"),
            intent_confidence=float(response_dict.get("intent_confidence", 0.0)),
            tts_style_prompt=response_dict.get("tts_style_prompt", "speaks at a moderate pace with a calm tone in a close-sounding environment with clear audio quality."),
            tts_speaker=response_dict.get("tts_speaker", "Rohit"),
            voice_change_requested=bool(response_dict.get("voice_change_requested", False)),
            preferred_speaker_gender=response_dict.get("preferred_speaker_gender", "any"),
            memory_updates=memory_updates,
            safety_flags=safety_flags,
            generation_time_ms=generation_time,
        )

    @staticmethod
    def _fallback_response() -> Dict[str, Any]:
        """
        Fallback response when Gemini fails (network, quota, etc.).
        Ensures the system never crashes.
        """
        return {
            "response_text": "I'm here for you. Please try again in a moment, I couldn't process that properly.",
            "response_language": "en",
            "detected_emotion": "neutral",
            "emotion_confidence": 0.0,
            "sentiment": "neutral",
            "detected_intent": "unknown",
            "intent_confidence": 0.0,
            "tts_style_prompt": "speaks at a moderate pace with a calm, supportive tone in a close-sounding environment with clear audio quality.",
            "tts_speaker": "Rohit",
            "voice_change_requested": False,
            "preferred_speaker_gender": "any",
            "memory_updates": [],
            "safety_flags": {
                "crisis_risk": "low",
                "self_harm_mentioned": False,
                "abuse_mentioned": False,
                "medical_concern": False,
                "flagged_keywords": [],
            }
        }

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_tts_speaker(self, language: str, emotion: str = None) -> str:
        """
        Get recommended TTS speaker for language + emotion.
        
        Args:
            language: Language code (e.g., "hi", "en", "ta")
            emotion: Optional emotion to influence speaker choice
            
        Returns:
            Speaker name (e.g., "Rohit", "Thoma")
        """
        lang_speakers = TTS_SPEAKER_MAP.get(language, {"default": "Thoma", "female": "Mary"})
        
        # Choose female speaker for certain emotions (sadness, vulnerability)
        if emotion and emotion.lower() in ["sadness", "vulnerability", "anxiety"]:
            return lang_speakers.get("female", lang_speakers.get("default"))
        
        return lang_speakers.get("default", "Thoma")

    def close(self):
        """Close synchronous client."""
        self._client.close()
        logger.info("GeminiLLMService client closed")

    async def aclose(self):
        """Close asynchronous client."""
        await self._client.aio.aclose()
        logger.info("GeminiLLMService async client closed")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize service
    llm_service = GeminiLLMService(api_key="GEMINI_API_KEY")
    
    # Example inputs
    transcript = "मैं परीक्षा में fail हो गया। बहुत depressed हूँ।"  # Hinglish
    language = "hi-en"
    acoustic_features = {
        "pitch": {"f0_mean": 165.2, "f0_variance": 35.1, "f0_range": [110, 220]},
        "energy": {"rms_mean": 0.035, "rms_variance": 0.018},
        "speech_rate": {"onset_rate": 2.1, "syllable_rate": 2.8},
        "pauses": {"count": 12, "total_duration": 3.2, "speech_to_silence_ratio": 0.55},
        "voice_quality": {"jitter": 0.035, "shimmer": 0.22, "breathiness": 0.42},
    }
    
    short_term_context = []
    long_term_memories = [
        {
            "memory_text": "User has high academic pressure from parents",
            "importance_score": 0.9,
        }
    ]
    episodic_memories = []
    
    character_profile = {
        "name": "Aarav",
        "background": "Empathetic AI companion for Indian youth",
        "traits": ["empathetic", "culturally aware", "patient", "thoughtful"],
        "speech_style": "conversational, validates emotions, offers practical support",
    }
    
    safety_context = {"crisis_risk": "low", "recent_flags": []}
    
    # Call LLM service
    try:
        response = llm_service.analyze_and_respond(
            transcript=transcript,
            language=language,
            acoustic_features=acoustic_features,
            short_term_context=short_term_context,
            long_term_memories=long_term_memories,
            episodic_memories=episodic_memories,
            character_profile=character_profile,
            safety_context=safety_context,
            temperature=0.7,
            max_output_tokens=2000,
        )
        
        print("\n" + "="*80)
        print("GEMINI LLM RESPONSE")
        print("="*80)
        print(f"Text: {response.response_text}")
        print(f"Language: {response.response_language}")
        print(f"Emotion: {response.detected_emotion} (conf: {response.emotion_confidence})")
        print(f"Sentiment: {response.sentiment}")
        print(f"Intent: {response.detected_intent} (conf: {response.intent_confidence})")
        print(f"TTS Prompt: {response.tts_style_prompt}")
        print(f"TTS Speaker: {response.tts_speaker}")
        print(f"Memory Updates: {len(response.memory_updates)}")
        print(f"Safety Flags: Crisis={response.safety_flags.crisis_risk}, "
              f"SelfHarm={response.safety_flags.self_harm_mentioned}")
        print(f"Generation Time: {response.generation_time_ms}ms")
        print("="*80)
        
        # Convert to dict for DB storage
        db_dict = response.to_dict()
        print("\nDatabase-ready dict:")
        print(json.dumps(db_dict, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        llm_service.close()
