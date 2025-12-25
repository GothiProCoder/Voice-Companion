"""
GuppShupp Audio Utilities
=========================

Production-grade audio handling utilities for base64 encoding/decoding,
temporary file management, format validation, and duration calculation.

Features:
- Thread-safe temporary file management
- Context manager for automatic cleanup
- Audio format detection via magic bytes
- Duration calculation using multiple backends
- Comprehensive error handling

Author: GuppShupp Team
"""

import base64
import hashlib
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Generator, Union
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Supported audio formats with their magic bytes
AUDIO_MAGIC_BYTES = {
    "wav": [
        (b"RIFF", 0, 4),  # RIFF header
    ],
    "mp3": [
        (b"\xff\xfb", 0, 2),  # MP3 frame sync
        (b"\xff\xfa", 0, 2),  # MP3 frame sync
        (b"\xff\xf3", 0, 2),  # MP3 frame sync
        (b"\xff\xf2", 0, 2),  # MP3 frame sync
        (b"ID3", 0, 3),       # ID3 tag
    ],
    "ogg": [
        (b"OggS", 0, 4),      # Ogg container
    ],
    "webm": [
        (b"\x1a\x45\xdf\xa3", 0, 4),  # WebM/Matroska
    ],
    "m4a": [
        (b"ftyp", 4, 8),      # MP4/M4A container
    ],
    "flac": [
        (b"fLaC", 0, 4),      # FLAC header
    ],
}

# File extension to MIME type mapping
MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "webm": "audio/webm",
    "m4a": "audio/mp4",
    "flac": "audio/flac",
}

# Maximum audio size (10 MB)
MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024

# Maximum audio duration (2 minutes)
MAX_AUDIO_DURATION_SECONDS = 120

# Temporary file directory
TEMP_AUDIO_DIR = Path(tempfile.gettempdir()) / "guppshupp_audio"


# =============================================================================
# EXCEPTIONS
# =============================================================================


class AudioProcessingError(Exception):
    """Base exception for audio processing errors."""
    pass


class InvalidAudioFormatError(AudioProcessingError):
    """Raised when audio format is invalid or unsupported."""
    pass


class AudioTooLargeError(AudioProcessingError):
    """Raised when audio exceeds size limit."""
    pass


class AudioDecodeError(AudioProcessingError):
    """Raised when base64 decoding fails."""
    pass


# =============================================================================
# FORMAT DETECTION
# =============================================================================


def validate_audio_format(data: bytes, expected_format: Optional[str] = None) -> str:
    """
    Detect and validate audio format from raw bytes.
    
    Args:
        data: Raw audio bytes
        expected_format: Optional expected format to validate against
        
    Returns:
        Detected format string (wav, mp3, ogg, webm, m4a, flac)
        
    Raises:
        InvalidAudioFormatError: If format is unsupported or mismatched
        
    Example:
        >>> with open("audio.wav", "rb") as f:
        ...     format = validate_audio_format(f.read())
        >>> print(format)
        "wav"
    """
    if len(data) < 12:
        raise InvalidAudioFormatError("Audio data too short to detect format")
    
    detected_format = None
    
    for format_name, signatures in AUDIO_MAGIC_BYTES.items():
        for magic, offset, end in signatures:
            if data[offset:end] == magic:
                detected_format = format_name
                break
        if detected_format:
            break
    
    if detected_format is None:
        # Log first 20 bytes for debugging
        logger.warning(f"Unknown audio format. First 20 bytes: {data[:20].hex()}")
        raise InvalidAudioFormatError(
            f"Unsupported audio format. Supported: {', '.join(AUDIO_MAGIC_BYTES.keys())}"
        )
    
    if expected_format and detected_format != expected_format:
        logger.warning(
            f"Audio format mismatch: expected {expected_format}, detected {detected_format}"
        )
        # Be lenient - use detected format
    
    return detected_format


def get_mime_type(format: str) -> str:
    """Get MIME type for audio format."""
    return MIME_TYPES.get(format, "application/octet-stream")


# =============================================================================
# BASE64 ENCODING/DECODING
# =============================================================================


def decode_audio_base64(
    data: str,
    expected_format: str = "wav",
    validate: bool = True
) -> Tuple[bytes, str]:
    """
    Decode base64 audio data to raw bytes.
    
    Args:
        data: Base64-encoded audio string
        expected_format: Expected audio format (for validation)
        validate: Whether to validate format after decoding
        
    Returns:
        Tuple of (raw_bytes, detected_format)
        
    Raises:
        AudioDecodeError: If base64 decoding fails
        AudioTooLargeError: If decoded audio exceeds size limit
        InvalidAudioFormatError: If format validation fails
        
    Example:
        >>> audio_bytes, format = decode_audio_base64(base64_string)
        >>> with open(f"output.{format}", "wb") as f:
        ...     f.write(audio_bytes)
    """
    # Clean the base64 string
    cleaned = data.strip().replace("\n", "").replace("\r", "").replace(" ", "")
    
    # Handle data URL format (data:audio/wav;base64,...)
    if cleaned.startswith("data:"):
        try:
            # Extract the base64 part
            _, base64_part = cleaned.split(",", 1)
            cleaned = base64_part
        except ValueError:
            raise AudioDecodeError("Invalid data URL format")
    
    # Decode base64
    try:
        raw_bytes = base64.b64decode(cleaned, validate=True)
    except Exception as e:
        raise AudioDecodeError(f"Invalid base64 encoding: {str(e)}")
    
    # Check size
    if len(raw_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise AudioTooLargeError(
            f"Audio size ({len(raw_bytes) / (1024*1024):.2f} MB) "
            f"exceeds limit ({MAX_AUDIO_SIZE_BYTES / (1024*1024):.0f} MB)"
        )
    
    if len(raw_bytes) < 100:
        raise AudioDecodeError("Audio data too small - possibly corrupted")
    
    # Validate and detect format
    if validate:
        detected_format = validate_audio_format(raw_bytes, expected_format)
    else:
        detected_format = expected_format
    
    logger.info(
        f"Decoded audio: {len(raw_bytes)} bytes, format={detected_format}"
    )
    
    return raw_bytes, detected_format


def encode_audio_to_base64(
    path: Union[str, Path],
    include_data_url: bool = False
) -> str:
    """
    Encode audio file to base64 string.
    
    Args:
        path: Path to audio file
        include_data_url: Whether to include data URL prefix
        
    Returns:
        Base64-encoded string (optionally with data URL prefix)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        AudioTooLargeError: If file exceeds size limit
        
    Example:
        >>> base64_str = encode_audio_to_base64("response.wav")
        >>> base64_url = encode_audio_to_base64("response.wav", include_data_url=True)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    file_size = path.stat().st_size
    if file_size > MAX_AUDIO_SIZE_BYTES:
        raise AudioTooLargeError(
            f"Audio file ({file_size / (1024*1024):.2f} MB) "
            f"exceeds limit ({MAX_AUDIO_SIZE_BYTES / (1024*1024):.0f} MB)"
        )
    
    with open(path, "rb") as f:
        raw_bytes = f.read()
    
    base64_str = base64.b64encode(raw_bytes).decode("ascii")
    
    if include_data_url:
        # Detect format and construct data URL
        format = validate_audio_format(raw_bytes)
        mime_type = get_mime_type(format)
        return f"data:{mime_type};base64,{base64_str}"
    
    return base64_str


# =============================================================================
# TEMPORARY FILE MANAGEMENT
# =============================================================================


def _ensure_temp_dir():
    """Ensure temporary audio directory exists."""
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _generate_temp_filename(
    format: str,
    prefix: str = "guppshupp_"
) -> Path:
    """Generate unique temporary filename."""
    _ensure_temp_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid4().hex[:8]
    filename = f"{prefix}{timestamp}_{unique_id}.{format}"
    return TEMP_AUDIO_DIR / filename


def save_temp_audio(
    data: bytes,
    format: str,
    prefix: str = "input_"
) -> Path:
    """
    Save audio bytes to temporary file.
    
    Args:
        data: Raw audio bytes
        format: Audio format (extension)
        prefix: Filename prefix
        
    Returns:
        Path to saved temporary file
        
    Example:
        >>> temp_path = save_temp_audio(audio_bytes, "wav")
        >>> # Use the file...
        >>> cleanup_temp_audio(temp_path)
    """
    temp_path = _generate_temp_filename(format, prefix)
    
    with open(temp_path, "wb") as f:
        f.write(data)
    
    logger.debug(f"Saved temp audio: {temp_path} ({len(data)} bytes)")
    return temp_path


def cleanup_temp_audio(path: Union[str, Path], ignore_errors: bool = True) -> bool:
    """
    Safely delete temporary audio file.
    
    Args:
        path: Path to file to delete
        ignore_errors: Whether to suppress deletion errors
        
    Returns:
        True if file was deleted, False otherwise
    """
    path = Path(path)
    
    try:
        if path.exists():
            path.unlink()
            logger.debug(f"Cleaned up temp audio: {path}")
            return True
        return False
    except Exception as e:
        if not ignore_errors:
            raise
        logger.warning(f"Failed to cleanup temp audio {path}: {e}")
        return False


def cleanup_old_temp_files(max_age_seconds: int = 3600) -> int:
    """
    Clean up old temporary audio files.
    
    Args:
        max_age_seconds: Maximum file age before deletion
        
    Returns:
        Number of files deleted
    """
    if not TEMP_AUDIO_DIR.exists():
        return 0
    
    deleted_count = 0
    current_time = time.time()
    
    for file_path in TEMP_AUDIO_DIR.glob("guppshupp_*"):
        try:
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    deleted_count += 1
        except Exception as e:
            logger.warning(f"Failed to cleanup old file {file_path}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} old temp audio files")
    
    return deleted_count


# =============================================================================
# CONTEXT MANAGER FOR TEMP FILES
# =============================================================================


class TempAudioFile:
    """
    Context manager for temporary audio file lifecycle.
    
    Automatically creates temp file from base64 data and
    cleans up on exit (even if exceptions occur).
    
    Example:
        >>> with TempAudioFile(base64_data, "wav") as temp_audio:
        ...     print(f"Audio saved to: {temp_audio.path}")
        ...     result = process_audio(temp_audio.path)
        >>> # File is automatically cleaned up here
        
    Attributes:
        path: Path to temporary file
        format: Detected audio format
        size_bytes: Size of audio data
        duration_seconds: Audio duration (if calculated)
    """
    
    _lock = threading.Lock()
    _active_files: set = set()
    
    def __init__(
        self,
        base64_data: str,
        expected_format: str = "wav",
        prefix: str = "input_",
        keep_on_error: bool = False
    ):
        """
        Initialize temp audio file manager.
        
        Args:
            base64_data: Base64-encoded audio data
            expected_format: Expected audio format
            prefix: Filename prefix
            keep_on_error: If True, don't delete file on error (for debugging)
        """
        self.base64_data = base64_data
        self.expected_format = expected_format
        self.prefix = prefix
        self.keep_on_error = keep_on_error
        
        self.path: Optional[Path] = None
        self.format: Optional[str] = None
        self.size_bytes: int = 0
        self.duration_seconds: Optional[float] = None
        self._error_occurred = False
    
    def __enter__(self) -> "TempAudioFile":
        """Decode and save audio to temp file."""
        # Decode base64
        raw_bytes, detected_format = decode_audio_base64(
            self.base64_data,
            self.expected_format
        )
        
        self.format = detected_format
        self.size_bytes = len(raw_bytes)
        
        # Save to temp file
        self.path = save_temp_audio(raw_bytes, detected_format, self.prefix)
        
        # Track active files for cleanup
        with self._lock:
            self._active_files.add(str(self.path))
        
        logger.info(
            f"TempAudioFile created: {self.path} "
            f"({self.size_bytes} bytes, {self.format})"
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Clean up temp file."""
        self._error_occurred = exc_type is not None
        
        # Remove from tracking
        with self._lock:
            self._active_files.discard(str(self.path))
        
        # Cleanup unless keeping on error
        if self.path and not (self._error_occurred and self.keep_on_error):
            cleanup_temp_audio(self.path)
            logger.debug(f"TempAudioFile cleaned up: {self.path}")
        elif self._error_occurred and self.keep_on_error:
            logger.warning(f"TempAudioFile kept for debugging: {self.path}")
        
        # Don't suppress exceptions
        return False
    
    def get_duration(self) -> Optional[float]:
        """
        Calculate and cache audio duration.
        
        Returns:
            Duration in seconds, or None if calculation fails
        """
        if self.duration_seconds is not None:
            return self.duration_seconds
        
        if self.path:
            self.duration_seconds = get_audio_duration(self.path)
        
        return self.duration_seconds
    
    @classmethod
    def cleanup_all_active(cls) -> int:
        """Clean up all active temp files (for shutdown)."""
        with cls._lock:
            files_to_cleanup = list(cls._active_files)
        
        count = 0
        for file_path in files_to_cleanup:
            if cleanup_temp_audio(file_path):
                count += 1
        
        logger.info(f"Cleaned up {count} active temp audio files")
        return count


# =============================================================================
# AUDIO DURATION CALCULATION
# =============================================================================


def get_audio_duration(path: Union[str, Path]) -> Optional[float]:
    """
    Calculate audio duration in seconds.
    
    Tries multiple backends in order:
    1. pydub (if available)
    2. soundfile (if available)
    3. librosa (if available)
    4. WAV header parsing (for WAV files only)
    
    Args:
        path: Path to audio file
        
    Returns:
        Duration in seconds, or None if calculation fails
    """
    path = Path(path)
    
    if not path.exists():
        logger.warning(f"Audio file not found for duration: {path}")
        return None
    
    # Try pydub (handles many formats)
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(path))
        duration = len(audio) / 1000.0  # pydub uses milliseconds
        logger.debug(f"Duration (pydub): {duration:.2f}s")
        return duration
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"pydub duration failed: {e}")
    
    # Try soundfile
    try:
        import soundfile as sf
        info = sf.info(str(path))
        duration = info.duration
        logger.debug(f"Duration (soundfile): {duration:.2f}s")
        return duration
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"soundfile duration failed: {e}")
    
    # Try librosa
    try:
        import librosa
        duration = librosa.get_duration(path=str(path))
        logger.debug(f"Duration (librosa): {duration:.2f}s")
        return duration
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"librosa duration failed: {e}")
    
    # Try manual WAV header parsing
    try:
        if path.suffix.lower() == ".wav":
            duration = _parse_wav_duration(path)
            if duration:
                logger.debug(f"Duration (WAV header): {duration:.2f}s")
                return duration
    except Exception as e:
        logger.debug(f"WAV header parsing failed: {e}")
    
    logger.warning(f"Could not determine audio duration for {path}")
    return None


def _parse_wav_duration(path: Path) -> Optional[float]:
    """Parse WAV header to get duration."""
    with open(path, "rb") as f:
        # Read RIFF header
        riff = f.read(4)
        if riff != b"RIFF":
            return None
        
        # Skip file size
        f.read(4)
        
        # Read WAVE
        wave = f.read(4)
        if wave != b"WAVE":
            return None
        
        # Find fmt chunk
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                return None
            
            chunk_size = int.from_bytes(f.read(4), "little")
            
            if chunk_id == b"fmt ":
                # Parse fmt chunk
                audio_format = int.from_bytes(f.read(2), "little")
                num_channels = int.from_bytes(f.read(2), "little")
                sample_rate = int.from_bytes(f.read(4), "little")
                byte_rate = int.from_bytes(f.read(4), "little")
                block_align = int.from_bytes(f.read(2), "little")
                bits_per_sample = int.from_bytes(f.read(2), "little")
                
                # Skip remaining fmt data
                remaining = chunk_size - 16
                if remaining > 0:
                    f.read(remaining)
                
                break
            else:
                f.seek(chunk_size, 1)
        
        # Find data chunk
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                return None
            
            chunk_size = int.from_bytes(f.read(4), "little")
            
            if chunk_id == b"data":
                # Calculate duration
                duration = chunk_size / byte_rate
                return duration
            else:
                f.seek(chunk_size, 1)
    
    return None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_audio_hash(data: bytes) -> str:
    """Generate SHA-256 hash of audio data for deduplication."""
    return hashlib.sha256(data).hexdigest()


def convert_to_wav(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 16000
) -> Path:
    """
    Convert audio to WAV format (required for Whisper).
    
    Args:
        input_path: Path to input audio
        output_path: Optional output path (auto-generated if not provided)
        sample_rate: Target sample rate (16kHz for Whisper)
        
    Returns:
        Path to converted WAV file
        
    Raises:
        AudioProcessingError: If conversion fails
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)
    
    try:
        from pydub import AudioSegment
        
        audio = AudioSegment.from_file(str(input_path))
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        audio.export(str(output_path), format="wav")
        
        logger.info(f"Converted {input_path} to WAV: {output_path}")
        return output_path
        
    except ImportError:
        raise AudioProcessingError(
            "pydub is required for audio conversion. "
            "Install with: pip install pydub"
        )
    except Exception as e:
        raise AudioProcessingError(f"Failed to convert audio: {e}")


@contextmanager
def temp_audio_context(
    base64_data: str,
    expected_format: str = "wav"
) -> Generator[Path, None, None]:
    """
    Simple context manager for temp audio file.
    
    Args:
        base64_data: Base64-encoded audio
        expected_format: Expected format
        
    Yields:
        Path to temporary audio file
        
    Example:
        >>> with temp_audio_context(audio_base64) as audio_path:
        ...     transcription = whisper.transcribe(audio_path)
    """
    with TempAudioFile(base64_data, expected_format) as temp:
        yield temp.path
