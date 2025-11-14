"""
Autonomous YouTube Studio - Minimal Production Implementation
Hybrid approach: Blueprint 2 foundation + Blueprint 1 algorithmic strengths
Uses CR-CA for causal reasoning and ReAct agents (max_loops="auto") for execution
"""

import os
import json
import time
import hashlib
import subprocess
import re
import argparse
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from enum import Enum
import heapq
from datetime import datetime, timedelta

from swarms import Agent
from swarms.agents.cr_ca_agent import CRCAAgent
from swarms.structs.sequential_workflow import SequentialWorkflow
from loguru import logger

# Optional advanced agents
try:
    from swarms.agents.GoTAgent import GoTAgent, _GoTConfig
    GOT_AVAILABLE = True
except ImportError:
    GOT_AVAILABLE = False
    logger.warning("GoTAgent not available. Script generation will use standard agent.")

try:
    from swarms.agents.AERASIGMA import AERASigmaAgent
    AERA_AVAILABLE = True
except ImportError:
    AERA_AVAILABLE = False
    logger.warning("AERASIGMA not available. Analytics learning will be limited.")

# Retry logic
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    logger.warning("tenacity not available. Retry logic will be basic.")

# Optional imports for production features
try:
    from google.cloud import texttospeech
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("Google Cloud TTS not available. Install: pip install google-cloud-texttospeech")

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    logger.warning("Replicate not available. Install: pip install replicate")

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import pickle
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    logger.warning("YouTube API not available. Install: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")


class VideoState(Enum):
    """State machine for video production pipeline"""
    DISCOVERED = "discovered"
    RANKED = "ranked"
    SCRIPTED = "scripted"
    ASSETS_GENERATED = "assets_generated"
    COMPOSED = "composed"
    VALIDATED = "validated"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    FAILED = "failed"
    OPTIMIZED = "optimized"


class ProductionStatus(Enum):
    """Production status for progress tracking"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class Topic:
    """Topic candidate with causal features"""
    id: str
    keywords: List[str]
    freshness: float
    prior_score: float
    causal_score: float = 0.0
    confidence: float = 0.0
    
    def __lt__(self, other):
        """For priority queue: higher score = higher priority"""
        return (self.causal_score + self.confidence) > (other.causal_score + other.confidence)


@dataclass
class Video:
    """Video production state"""
    video_id: str
    topic: Topic
    state: VideoState = VideoState.DISCOVERED
    script: Optional[str] = None
    assets: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    youtube_id: Optional[str] = None
    idempotency_key: Optional[str] = None


@dataclass
class ProductionProgress:
    """Progress tracking for video production"""
    video_id: str
    current_step: str
    progress_pct: float
    status: ProductionStatus
    errors: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    estimated_completion: Optional[float] = None
    last_updated: float = field(default_factory=time.time)


@dataclass
class TextOverlay:
    """Text overlay specification for video composition"""
    start_time: float
    end_time: float
    text: str
    position: str = "bottom"  # "top", "bottom", "center"
    font_size: int = 48
    font_color: str = "white"
    background: bool = True
    animation: str = "fade"  # "fade", "slide"


# Retry decorator wrapper
def retry_on_failure(max_attempts=3, backoff_base=1):
    """Create retry decorator with exponential backoff"""
    if TENACITY_AVAILABLE:
        return retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=backoff_base, min=backoff_base, max=backoff_base * 4),
            reraise=True
        )
    else:
        # Basic retry without tenacity
        def decorator(func):
            def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            wait_time = backoff_base * (2 ** attempt)
                            time.sleep(wait_time)
                        else:
                            raise last_exception
                return None
            return wrapper
        return decorator


@retry_on_failure(max_attempts=3, backoff_base=1)
def generate_tts_audio(text: str, output_path: str, language_code: str = "en-US", voice_name: str = "en-US-Neural2-F") -> str:
    """
    Generate TTS audio from text using Google Cloud TTS.
    
    Args:
        text: Text to convert to speech
        output_path: Path to save audio file
        language_code: Language code (e.g., "en-US")
        voice_name: Voice name (e.g., "en-US-Neural2-F")
        
    Returns:
        Path to generated audio file
    """
    if not TTS_AVAILABLE:
        logger.warning("TTS not available, creating placeholder")
        # Create silent audio as fallback
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", "10", "-y", output_path
        ], capture_output=True, check=False)
        return output_path
    
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0,
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        logger.info(f"Generated TTS audio: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"TTS generation failed: {e}, using fallback")
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", "10", "-y", output_path
        ], capture_output=True, check=False)
        return output_path


@retry_on_failure(max_attempts=3, backoff_base=1)
def generate_thumbnail_image(prompt: str, output_path: str) -> str:
    """
    Generate thumbnail image using Stable Diffusion via Replicate.
    
    Args:
        prompt: Image generation prompt
        output_path: Path to save image
        
    Returns:
        Path to generated image
    """
    if not REPLICATE_AVAILABLE:
        logger.warning("Image generation not available, creating placeholder")
        # Create placeholder image
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "color=c=blue:s=1920x1080",
            "-frames:v", "1", "-y", output_path
        ], capture_output=True, check=False)
        return output_path
    
    try:
        # Use Stable Diffusion via Replicate
        model = "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478"
        output = replicate.run(
            model,
            input={"prompt": prompt, "width": 1920, "height": 1080}
        )
        
        # Download image
        import urllib.request
        urllib.request.urlretrieve(output[0], output_path)
        logger.info(f"Generated thumbnail: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Image generation failed: {e}, using fallback")
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "color=c=blue:s=1920x1080",
            "-frames:v", "1", "-y", output_path
        ], capture_output=True, check=False)
        return output_path


def parse_script_for_text_overlays(script: str, audio_duration: float, num_overlays: int = 5) -> List[TextOverlay]:
    """
    Parse script to extract key phrases for text overlays.
    
    Args:
        script: Video script text
        audio_duration: Total audio duration in seconds
        num_overlays: Number of text overlays to generate
        
    Returns:
        List of TextOverlay objects
    """
    overlays = []
    
    # Extract sentences (simple split)
    sentences = re.split(r'[.!?]+', script)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        return overlays
    
    # Calculate timing distribution
    overlay_duration = audio_duration / (num_overlays + 1)  # Space them out
    overlay_interval = audio_duration / (num_overlays + 1)
    
    # Select key sentences
    selected_indices = []
    if len(sentences) >= num_overlays:
        step = len(sentences) // num_overlays
        selected_indices = [i * step for i in range(num_overlays)]
    else:
        selected_indices = list(range(len(sentences)))
    
    # Create overlays
    for i, idx in enumerate(selected_indices[:num_overlays]):
        if idx < len(sentences):
            text = sentences[idx]
            # Limit text length
            if len(text) > 60:
                text = text[:57] + "..."
            
            start_time = i * overlay_interval + 2.0  # Start after 2s, space them
            end_time = start_time + min(overlay_duration, 5.0)  # Show for 3-5 seconds
            
            # Position: alternate between bottom and top
            position = "bottom" if i % 2 == 0 else "top"
            
            overlays.append(TextOverlay(
                start_time=start_time,
                end_time=end_time,
                text=text,
                position=position,
                font_size=48,
                font_color="white",
                background=True,
                animation="fade"
            ))
    
    return overlays


def generate_text_overlay_filter(text_overlay: TextOverlay, input_label: str, output_label: str) -> str:
    """
    Generate FFmpeg drawtext filter for a text overlay.
    
    Args:
        text_overlay: TextOverlay specification
        input_label: Input video label (e.g., "[v]")
        output_label: Output video label (e.g., "[vtext]")
        
    Returns:
        FFmpeg filter string
    """
    # Escape text for FFmpeg (escape single quotes and colons)
    escaped_text = text_overlay.text.replace("'", "\\'").replace(":", "\\:")
    
    # Position calculation
    if text_overlay.position == "bottom":
        y_pos = "y=h-th-60"
    elif text_overlay.position == "top":
        y_pos = "y=60"
    else:  # center
        y_pos = "y=(h-text_h)/2"
    
    # Build filter parts
    filter_parts = [
        f"text='{escaped_text}'",
        f"enable='between(t,{text_overlay.start_time},{text_overlay.end_time})'",
        f"fontsize={text_overlay.font_size}",
        f"fontcolor={text_overlay.font_color}",
        "x=(w-text_w)/2",  # Center horizontally
        y_pos,
    ]
    
    # Background box
    if text_overlay.background:
        filter_parts.append("box=1")
        filter_parts.append("boxcolor=black@0.7")
        filter_parts.append("boxborderw=5")
    
    # Animation (fade in/out)
    if text_overlay.animation == "fade":
        fade_duration = 0.5
        fade_in_end = text_overlay.start_time + fade_duration
        fade_out_start = text_overlay.end_time - fade_duration
        alpha_expr = (
            f"if(between(t,{text_overlay.start_time},{fade_in_end}),"
            f"(t-{text_overlay.start_time})/{fade_duration},"
            f"if(between(t,{fade_out_start},{text_overlay.end_time}),"
            f"({text_overlay.end_time}-t)/{fade_duration},1))"
        )
        filter_parts.append(f"alpha='{alpha_expr}'")
    
    # Join filter parts with colons
    filter_str = f"{input_label}drawtext={':'.join(filter_parts)}{output_label}"
    return filter_str


def get_background_music(topic: Topic, duration: float, workspace_dir: str) -> Optional[str]:
    """
    Get background music for video (royalty-free or fallback).
    
    Args:
        topic: Video topic
        duration: Required music duration in seconds
        workspace_dir: Workspace directory for caching
        
    Returns:
        Path to music file, or None if unavailable
    """
    music_dir = f"{workspace_dir}/music"
    os.makedirs(music_dir, exist_ok=True)
    
    # Try to use cached music
    cached_music = f"{music_dir}/background.mp3"
    if os.path.exists(cached_music):
        # Check if duration is sufficient
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", cached_music],
                capture_output=True, text=True, check=True
            )
            music_duration = float(result.stdout.strip())
            if music_duration >= duration:
                logger.info(f"Using cached background music: {cached_music}")
                return cached_music
        except:
            pass
    
    # Fallback: generate simple background tone (very basic)
    # In production, would use Pexels Audio API or similar
    try:
        # Generate a simple sine wave as placeholder
        # This is a very basic fallback - in production use royalty-free music
        logger.info("Generating placeholder background music")
        subprocess.run([
            "ffmpeg", "-f", "lavfi",
            "-i", f"sine=frequency=220:duration={duration}",
            "-af", "volume=0.1",  # Very quiet
            "-y", cached_music
        ], capture_output=True, check=True, timeout=30)
        return cached_music
    except Exception as e:
        logger.warning(f"Could not generate background music: {e}")
        return None


@retry_on_failure(max_attempts=3, backoff_base=2)
def compose_video_with_ffmpeg(
    audio_path: str,
    images: List[str],
    output_path: str,
    duration_per_image: Optional[float] = None,
    transition_type: str = "fade",
    transition_duration: float = 0.5,
    text_overlays: Optional[List[TextOverlay]] = None,
    background_music_path: Optional[str] = None,
    music_volume: float = 0.3,
) -> str:
    """
    Compose video from audio and images using FFmpeg with enhanced features.
    
    Args:
        audio_path: Path to audio file
        images: List of image paths
        output_path: Path to save final video
        duration_per_image: Duration per image (auto-calculated if None)
        transition_type: Transition type ("fade", "crossfade", "slide", "zoom")
        transition_duration: Duration of transitions in seconds
        text_overlays: List of TextOverlay objects
        background_music_path: Optional path to background music
        music_volume: Background music volume (0.0-1.0)
        
    Returns:
        Path to generated video
    """
    try:
        # Get audio duration
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, check=True
        )
        audio_duration = float(result.stdout.strip())
        
        # Calculate image durations
        if duration_per_image is None:
            # Auto-calculate: distribute audio duration across images
            # Account for transitions
            total_transition_time = transition_duration * max(0, len(images) - 1)
            available_time = audio_duration - total_transition_time
            duration_per_image = max(2.0, available_time / len(images)) if images else 5.0
        
        # Build FFmpeg filter complex
        filter_parts = []
        inputs = ["-i", audio_path]
        
        # Add background music if provided
        music_input_index = None
        if background_music_path and os.path.exists(background_music_path):
            inputs.extend(["-i", background_music_path])
            music_input_index = len(images) + 1  # After all image inputs
        
        # Process images: scale and pad
        for i, img_path in enumerate(images):
            inputs.extend(["-loop", "1", "-t", str(duration_per_image + transition_duration), "-i", img_path])
            filter_parts.append(
                f"[{i+1}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]"
            )
        
        # Apply transitions
        if len(images) > 1:
            # Start with first image
            current_video = "[v0]"
            
            # Apply transitions between images
            for i in range(1, len(images)):
                offset = i * duration_per_image - transition_duration
                transition_name = f"vxfade{i}"
                
                # Determine transition type
                if transition_type == "fade":
                    xfade_transition = "fade"
                elif transition_type == "crossfade":
                    xfade_transition = "fade"  # Crossfade uses fade
                elif transition_type == "slide":
                    xfade_transition = "slideleft"
                elif transition_type == "zoom":
                    xfade_transition = "zoom"
                else:
                    xfade_transition = "fade"
                
                # Build xfade filter: [input1][input2]xfade=...:duration=...:offset=...[output]
                transition_filter = (
                    f"{current_video}[v{i}]"
                    f"xfade=transition={xfade_transition}:duration={transition_duration}:offset={offset}"
                    f"[{transition_name}]"
                )
                filter_parts.append(transition_filter)
                current_video = f"[{transition_name}]"
            
            video_output = current_video
        else:
            # Single image, no transitions needed
            video_output = "[v0]"
        
        # Add text overlays
        if text_overlays:
            current_video_label = video_output.strip("[]")
            for i, overlay in enumerate(text_overlays):
                input_label = f"[{current_video_label}]"
                output_label = f"[vtext{i}]"
                text_filter = generate_text_overlay_filter(overlay, input_label, output_label)
                filter_parts.append(text_filter)
                current_video_label = f"vtext{i}"
            
            video_output = f"[{current_video_label}]"
        
        # Audio processing
        audio_output = "[0:a]"
        
        # Mix with background music if available
        if music_input_index is not None:
            # Normalize narration volume
            audio_output = "[0:a]volume=1.0[narr]"
            
            # Process background music: volume control and fade
            music_vol = max(0.0, min(1.0, music_volume))
            music_fade_duration = 2.0
            music_output = (
                f"[{music_input_index}:a]volume={music_vol},"
                f"afade=t=in:st=0:d={music_fade_duration},"
                f"afade=t=out:st={audio_duration - music_fade_duration}:d={music_fade_duration}[music]"
            )
            filter_parts.append(music_output)
            
            # Mix narration and music
            filter_parts.append("[narr][music]amix=inputs=2:duration=first:dropout_transition=2[outa]")
            audio_output = "[outa]"
        
        # Normalize final audio
        filter_parts.append(f"{audio_output}loudnorm=I=-16:TP=-1.5:LRA=11[finala]")
        audio_output = "[finala]"
        
        # Combine all filters
        filter_complex = ";".join(filter_parts)
        
        # Execute FFmpeg
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", video_output.strip("[]"),
            "-map", audio_output.strip("[]"),
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-r", "30",  # 30 fps
            "-shortest",
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Composed video with enhancements: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Video composition failed: {e}")
        raise


def validate_video_quality(video_path: str, audio_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate video quality before finalizing.
    
    Args:
        video_path: Path to video file
        audio_path: Optional path to audio file for duration comparison
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "file_exists": False,
        "file_size_mb": 0.0,
        "duration_seconds": 0.0,
        "video_codec": None,
        "audio_codec": None,
        "resolution": None,
    }
    
    try:
        # Check file exists
        if not os.path.exists(video_path):
            results["errors"].append("Video file does not exist")
            return results
        
        results["file_exists"] = True
        file_size = os.path.getsize(video_path)
        results["file_size_mb"] = file_size / (1024 * 1024)
        
        # Check file size > 1MB
        if file_size < 1024 * 1024:
            results["errors"].append(f"Video file too small: {results['file_size_mb']:.2f} MB")
        
        # Get video properties using ffprobe
        try:
            probe_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height",
                "-of", "json", video_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            video_info = json.loads(probe_result.stdout)
            
            if video_info.get("streams"):
                stream = video_info["streams"][0]
                results["video_codec"] = stream.get("codec_name", "unknown")
                width = stream.get("width", 0)
                height = stream.get("height", 0)
                results["resolution"] = f"{width}x{height}"
                
                # Check codec
                if results["video_codec"] != "h264":
                    results["warnings"].append(f"Video codec is {results['video_codec']}, expected h264")
                
                # Check resolution
                if width != 1920 or height != 1080:
                    results["warnings"].append(f"Resolution is {results['resolution']}, expected 1920x1080")
        except Exception as e:
            results["warnings"].append(f"Could not probe video properties: {e}")
        
        # Get audio properties
        try:
            audio_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-of", "json", video_path
            ]
            audio_result = subprocess.run(audio_cmd, capture_output=True, text=True, check=True)
            audio_info = json.loads(audio_result.stdout)
            
            if audio_info.get("streams"):
                results["audio_codec"] = audio_info["streams"][0].get("codec_name", "unknown")
                if results["audio_codec"] != "aac":
                    results["warnings"].append(f"Audio codec is {results['audio_codec']}, expected aac")
        except Exception as e:
            results["warnings"].append(f"Could not probe audio properties: {e}")
        
        # Get duration
        try:
            duration_cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path
            ]
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
            results["duration_seconds"] = float(duration_result.stdout.strip())
        except Exception as e:
            results["warnings"].append(f"Could not get video duration: {e}")
        
        # Compare with audio duration if provided
        if audio_path and os.path.exists(audio_path):
            try:
                audio_duration_cmd = [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", audio_path
                ]
                audio_duration_result = subprocess.run(audio_duration_cmd, capture_output=True, text=True, check=True)
                audio_duration = float(audio_duration_result.stdout.strip())
                
                duration_diff = abs(results["duration_seconds"] - audio_duration)
                duration_tolerance = audio_duration * 0.05  # 5% tolerance
                
                if duration_diff > duration_tolerance:
                    results["warnings"].append(
                        f"Video duration ({results['duration_seconds']:.2f}s) differs from audio "
                        f"({audio_duration:.2f}s) by {duration_diff:.2f}s"
                    )
            except Exception as e:
                results["warnings"].append(f"Could not compare with audio duration: {e}")
        
        # Overall validation
        results["valid"] = len(results["errors"]) == 0
        
    except Exception as e:
        results["errors"].append(f"Validation error: {e}")
        results["valid"] = False
    
    return results


class QuotaManager:
    """Manages YouTube API quota usage"""
    
    def __init__(self, quota_file: str = "quota_state.json"):
        self.quota_file = quota_file
        self.quota_limit = 10000  # Default daily quota
        self.quota_used = 0
        self.quota_reset_time = time.time() + 86400  # Reset in 24 hours
        self.operation_costs = {
            "upload": 1600,
            "thumbnail": 50,
            "captions": 50,
            "analytics": 1,
        }
        self.load_quota_state()
    
    def load_quota_state(self):
        """Load quota state from file"""
        try:
            if os.path.exists(self.quota_file):
                with open(self.quota_file, "r") as f:
                    data = json.load(f)
                    self.quota_used = data.get("quota_used", 0)
                    self.quota_reset_time = data.get("quota_reset_time", time.time() + 86400)
                    
                    # Reset if past reset time
                    if time.time() > self.quota_reset_time:
                        self.quota_used = 0
                        self.quota_reset_time = time.time() + 86400
        except Exception as e:
            logger.warning(f"Could not load quota state: {e}")
    
    def save_quota_state(self):
        """Save quota state to file"""
        try:
            with open(self.quota_file, "w") as f:
                json.dump({
                    "quota_used": self.quota_used,
                    "quota_reset_time": self.quota_reset_time,
                    "quota_limit": self.quota_limit,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save quota state: {e}")
    
    def check_quota(self, operation: str, required: Optional[int] = None) -> bool:
        """
        Check if quota is available for operation.
        
        Args:
            operation: Operation name (upload, thumbnail, captions, analytics)
            required: Required quota units (uses operation_costs if None)
            
        Returns:
            True if quota available, False otherwise
        """
        # Reset if past reset time
        if time.time() > self.quota_reset_time:
            self.quota_used = 0
            self.quota_reset_time = time.time() + 86400
        
        cost = required if required is not None else self.operation_costs.get(operation, 0)
        available = self.quota_limit - self.quota_used
        
        if available < cost:
            logger.warning(f"Insufficient quota: {available}/{self.quota_limit} available, need {cost}")
            return False
        
        return True
    
    def use_quota(self, operation: str, amount: Optional[int] = None):
        """Record quota usage"""
        cost = amount if amount is not None else self.operation_costs.get(operation, 0)
        self.quota_used += cost
        self.save_quota_state()
        logger.info(f"Used {cost} quota units ({self.quota_used}/{self.quota_limit} used)")
    
    def wait_for_quota_reset(self) -> float:
        """Wait until quota resets. Returns seconds until reset."""
        wait_time = max(0, self.quota_reset_time - time.time())
        if wait_time > 0:
            logger.info(f"Waiting {wait_time/3600:.2f} hours for quota reset")
        return wait_time


class YouTubeStudio:
    """
    Autonomous YouTube Studio Orchestrator
    
    Architecture:
    - CR-CA Agent: Causal reasoning for topic ranking and optimization decisions
    - ReAct Agents: Execution workers with max_loops="auto" for autonomous task completion
    - State Machine: Tracks video production pipeline
    - Priority Queue: Efficient topic selection
    - Production Tools: TTS, image generation, FFmpeg composition
    
    Actually produces video files, not just metadata.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        workspace_dir: str = "yt_workspace",
        max_concurrent: int = 3,
        google_tts_credentials: Optional[str] = None,
        replicate_api_token: Optional[str] = None,
        youtube_credentials: Optional[str] = None,
    ):
        """
        Initialize YouTube Studio.
        
        Args:
            model_name: LLM model for agents
            workspace_dir: Directory for artifacts
            max_concurrent: Maximum concurrent video productions
            google_tts_credentials: Path to Google Cloud credentials JSON
            replicate_api_token: Replicate API token
            youtube_credentials: Path to YouTube API credentials JSON
        """
        self.model_name = model_name
        self.workspace_dir = workspace_dir
        self.max_concurrent = max_concurrent
        
        # Set up API credentials
        if google_tts_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_tts_credentials
        if replicate_api_token:
            os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
        
        # Create workspace
        os.makedirs(workspace_dir, exist_ok=True)
        os.makedirs(f"{workspace_dir}/scripts", exist_ok=True)
        os.makedirs(f"{workspace_dir}/assets", exist_ok=True)
        os.makedirs(f"{workspace_dir}/audio", exist_ok=True)
        os.makedirs(f"{workspace_dir}/images", exist_ok=True)
        os.makedirs(f"{workspace_dir}/videos", exist_ok=True)
        
        # State management
        self.topic_queue: List[Topic] = []  # Priority queue
        self.videos: Dict[str, Video] = {}
        self.state_cache: Dict[str, Any] = {}
        self.production_progress: Dict[str, ProductionProgress] = {}
        self.production_queue: List[Tuple[float, Topic]] = []  # Priority queue for production
        self.production_paused = False
        
        # Quota management
        quota_file = f"{workspace_dir}/quota_state.json"
        self.quota_manager = QuotaManager(quota_file)
        
        # YouTube API
        self.youtube_service = None
        self.youtube_analytics_service = None
        self.youtube_credentials_path = youtube_credentials
        if youtube_credentials and YOUTUBE_API_AVAILABLE:
            self._init_youtube_service()
        
        # Initialize CR-CA agent for causal reasoning
        self.cr_ca = CRCAAgent(
            name="topic-ranker",
            description="Causal reasoning agent for programming tutorial topic ranking and optimization",
            model_name=model_name,
            max_loops=3,
            variables=[
                "programming_trend_relevance",  # How relevant is the programming trend
                "business_trend_relevance",     # How relevant is the business trend
                "swarms_applicability",        # How well Swarms applies to this topic
                "educational_value",           # Learning value for Python developers
                "code_complexity",             # Beginner/intermediate/advanced
                "expected_watch_time",
                "expected_ctr"
            ],
            causal_edges=[
                ("programming_trend_relevance", "expected_ctr"),
                ("business_trend_relevance", "expected_ctr"),
                ("swarms_applicability", "expected_watch_time"),
                ("educational_value", "expected_watch_time"),
                ("code_complexity", "expected_watch_time"),
                ("expected_ctr", "expected_watch_time"),
            ]
        )
        
        # Initialize ReAct agents for execution
        self.discovery_agent = Agent(
            agent_name="discovery",
            system_prompt="""You discover trending programming and business topics that can be taught as Python tutorials using the Swarms framework.

Focus on:
- Current programming trends (AI automation, web development, APIs, data processing, automation)
- Business trends (passive income websites, SaaS applications, automated workflows, AI-powered tools)
- Topics that demonstrate practical Swarms integration (multi-agent systems, automation, workflows)
- Educational value for Python developers learning Swarms

For each topic, identify:
- The programming/business trend
- How Swarms can be applied (e.g., "Flask + Swarms for automated website management")
- Python concepts to teach (Flask, APIs, data processing, etc.)
- Swarms features to demonstrate (agents, workflows, tools, etc.)

Output JSON with: topic_id, keywords[], trend_type (programming/business), swarms_application, python_concepts[], freshness_score, prior_score

Example topics:
- "Building a Passive Income Website with Flask and Swarms Agents"
- "Automating Social Media with Python and Swarms Multi-Agent System"
- "Creating an AI-Powered SaaS with FastAPI and Swarms Workflows"

Reference Swarms documentation: https://docs.swarms.world/en/latest/""",
            model_name=model_name,
            max_loops="auto",
            output_type="json",
        )
        
        self.script_agent = Agent(
            agent_name="script-writer",
            system_prompt="""You write engaging Python tutorial scripts for YouTube that teach the Swarms framework while applying programming/business trends.

Script Structure:
1. Hook (0-15s): Grab attention with the trend/problem
2. Introduction (15-30s): Explain the trend and why Swarms is perfect for it
3. Setup (30-60s): Environment setup and Swarms installation
4. Core Tutorial (60-90%): Step-by-step code walkthrough
   - Explain Python concepts
   - Show Swarms integration
   - Demonstrate multi-agent workflows
   - Include code examples
5. Application (90-95%): Show real-world application
6. Call-to-Action (95-100%): Subscribe, like, next steps

Requirements:
- Reference Swarms documentation: https://docs.swarms.world/en/latest/
- Include actual code snippets with explanations
- Explain both Python concepts AND Swarms features
- Show practical application of the trend
- Use clear, educational language
- Include timestamps for each section

Output markdown format with:
- Code blocks for Python/Swarms code
- Timestamps for each section
- Clear explanations of concepts
- Links to Swarms documentation where relevant""",
            model_name=model_name,
            max_loops="auto",
            output_type="str",
        )
        
        self.asset_agent = Agent(
            agent_name="asset-generator",
            system_prompt="""You generate video assets for Python programming tutorials teaching Swarms.

Thumbnail:
- Include code/terminal elements
- Show Python + Swarms branding
- Highlight the trend/application
- High CTR design with tech aesthetic

B-roll:
- Code screenshots (Python code, Swarms agents)
- Terminal/console outputs
- Architecture diagrams (multi-agent systems)
- Application demos (web interfaces, automation in action)
- Swarms framework visualizations

Text Overlays:
- Code snippets (Python, Swarms API calls)
- Key concepts (agent names, workflow steps)
- Important notes (installation commands, configuration)
- Architecture labels (agent roles, data flow)

Background Music:
- Upbeat, tech-focused
- Not distracting from code explanations

Output JSON with:
- thumbnail_prompt: description with code/tech elements
- broll_descriptions: code screenshots, terminal outputs, diagrams
- text_overlay_phrases: code snippets, key concepts, commands
- music_style: tech/upbeat
- transition_type: smooth transitions for code walkthroughs""",
            model_name=model_name,
            max_loops="auto",
            output_type="json",
            tools=[generate_thumbnail_image],  # Agent can call image generation
        )
        
        self.composer_agent = Agent(
            agent_name="video-composer",
            system_prompt="""You compose final videos by:
- Synchronizing audio with visuals
- Adding transitions
- Inserting captions
- Optimizing for engagement
Output FFmpeg command sequence.""",
            model_name=model_name,
            max_loops="auto",
            output_type="str",
        )
        
        self.optimizer_agent = Agent(
            agent_name="optimizer",
            system_prompt="""You optimize video performance by:
- Analyzing metrics (CTR, watch time, retention)
- Testing title/thumbnail variants
- Adjusting metadata
- Learning from feedback
Use causal reasoning to identify what works.""",
            model_name=model_name,
            max_loops="auto",
            output_type="json",
        )
        
        # Initialize GoTAgent for enhanced script generation (optional)
        self.got_script_agent = None
        if GOT_AVAILABLE:
            try:
                from swarms.agents.GoTAgent import _GoTConfig
                self.got_script_agent = GoTAgent(
                    agent_name="got-script-writer",
                    model_name=model_name,
                    system_prompt="""You write engaging Python tutorial scripts for YouTube that teach the Swarms framework while applying programming/business trends. Use graph-of-thought reasoning to explore multiple narrative structures and merge the best ideas.""",
                    config=_GoTConfig(
                        max_nodes=30,
                        max_iterations=15,
                        expansion_branch_factor=3,
                        enable_merging=True,
                        enable_refinement=True,
                        answer_prefix="Final script:"
                    )
                )
                logger.info("GoTAgent initialized for script generation")
            except Exception as e:
                logger.warning(f"Failed to initialize GoTAgent: {e}. Falling back to standard agent.")
                self.got_script_agent = None
        
        # Initialize AERASIGMA for analytics learning (optional)
        self.aera_learner = None
        if AERA_AVAILABLE:
            try:
                self.aera_learner = AERASigmaAgent(
                    agent_name="aera-analytics-learner",
                    model_name=model_name,
                    learning_enabled=True,
                    analogy_enabled=True,
                    max_iterations=50
                )
                logger.info("AERASIGMA initialized for analytics learning")
            except Exception as e:
                logger.warning(f"Failed to initialize AERASIGMA: {e}. Analytics learning will be limited.")
                self.aera_learner = None
        
        # Optional: Initialize GoTAgent for topic refinement
        self.got_topic_refiner = None
        if GOT_AVAILABLE:
            try:
                from swarms.agents.GoTAgent import _GoTConfig
                self.got_topic_refiner = GoTAgent(
                    agent_name="got-topic-refiner",
                    model_name=model_name,
                    system_prompt="Explore different angles and perspectives for programming tutorial topics that can be taught with Swarms framework.",
                    config=_GoTConfig(max_nodes=20, max_iterations=10)
                )
                logger.info("GoTAgent initialized for topic refinement")
            except Exception as e:
                logger.warning(f"Failed to initialize GoTAgent for topic refinement: {e}")
                self.got_topic_refiner = None
        
        # Workflow orchestration
        self.production_workflow = SequentialWorkflow(
            agents=[
                self.script_agent,
                self.asset_agent,
                self.composer_agent,
            ]
        )
        
        # Load production state
        self._load_production_state()
        
        logger.info("YouTube Studio initialized")
    
    def _init_youtube_service(self):
        """Initialize YouTube Data API and Analytics API services"""
        if not YOUTUBE_API_AVAILABLE:
            logger.warning("YouTube API not available")
            return
        
        try:
            SCOPES = [
                "https://www.googleapis.com/auth/youtube.upload",
                "https://www.googleapis.com/auth/youtube",
                "https://www.googleapis.com/auth/youtube.readonly",
                "https://www.googleapis.com/auth/youtube.force-ssl",  # For channel updates
                "https://www.googleapis.com/auth/yt-analytics.readonly",
            ]
            
            creds = None
            token_file = f"{self.workspace_dir}/youtube_token.pickle"
            
            # Load existing credentials
            if os.path.exists(token_file):
                with open(token_file, "rb") as token:
                    creds = pickle.load(token)
            
            # Refresh or get new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not self.youtube_credentials_path or not os.path.exists(self.youtube_credentials_path):
                        logger.warning("YouTube credentials file not found. YouTube features disabled.")
                        return
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.youtube_credentials_path, SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                
                # Save credentials
                with open(token_file, "wb") as token:
                    pickle.dump(creds, token)
            
            # Build services
            self.youtube_service = build("youtube", "v3", credentials=creds)
            try:
                self.youtube_analytics_service = build("youtubeAnalytics", "v2", credentials=creds)
            except Exception as e:
                logger.warning(f"Could not initialize Analytics API: {e}")
            
            logger.info("YouTube API services initialized")
        except Exception as e:
            logger.error(f"Failed to initialize YouTube services: {e}")
    
    def _load_production_state(self):
        """Load production state from file"""
        state_file = f"{self.workspace_dir}/production_state.json"
        try:
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    data = json.load(f)
                    # Restore video states
                    for video_id, video_data in data.get("videos", {}).items():
                        if video_id in self.videos:
                            # Update state
                            try:
                                self.videos[video_id].state = VideoState(video_data.get("state", "discovered"))
                            except:
                                pass
        except Exception as e:
            logger.warning(f"Could not load production state: {e}")
    
    def _save_production_state(self):
        """Save production state to file"""
        state_file = f"{self.workspace_dir}/production_state.json"
        try:
            data = {
                "videos": {
                    vid: {
                        "state": video.state.value,
                        "youtube_id": video.youtube_id,
                        "idempotency_key": video.idempotency_key,
                    }
                    for vid, video in self.videos.items()
                },
                "timestamp": time.time(),
            }
            with open(state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save production state: {e}")
    
    def track_progress(self, video_id: str, step: str, progress_pct: float, status: ProductionStatus, error: Optional[str] = None):
        """Track production progress"""
        if video_id not in self.production_progress:
            self.production_progress[video_id] = ProductionProgress(
                video_id=video_id,
                current_step=step,
                progress_pct=progress_pct,
                status=status,
            )
        else:
            progress = self.production_progress[video_id]
            progress.current_step = step
            progress.progress_pct = progress_pct
            progress.status = status
            progress.last_updated = time.time()
            if error:
                progress.errors.append(f"{step}: {error}")
        
        # Save progress to file
        progress_file = f"{self.workspace_dir}/production_progress.json"
        try:
            with open(progress_file, "w") as f:
                progress_data = {
                    vid: asdict(prog) for vid, prog in self.production_progress.items()
                }
                json.dump(progress_data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")
    
    def get_production_status(self, video_id: Optional[str] = None) -> Dict[str, Any]:
        """Get production status for video(s)"""
        if video_id:
            if video_id in self.production_progress:
                return asdict(self.production_progress[video_id])
            return {}
        else:
            return {vid: asdict(prog) for vid, prog in self.production_progress.items()}
    
    def discover_topics(self, num_topics: int = 10) -> List[Topic]:
        """
        Discover candidate programming/business topics for Python + Swarms tutorials.
        
        Args:
            num_topics: Number of topics to discover
            
        Returns:
            List of discovered topics
        """
        task = f"""Discover {num_topics} trending programming and business topics that can be taught as Python tutorials using the Swarms framework.

Focus on:
- Current programming trends (AI automation, web development, APIs, data processing)
- Business trends (passive income websites, SaaS applications, automated workflows)
- Topics that demonstrate practical Swarms integration
- Educational value for Python developers

For each topic, identify:
- The programming/business trend
- How Swarms can be applied (e.g., "Flask + Swarms for automated website management")
- Python concepts to teach
- Swarms features to demonstrate

Reference Swarms documentation: https://docs.swarms.world/en/latest/

Output JSON with: topic_id, keywords[], trend_type (programming/business), swarms_application, python_concepts[], freshness_score, prior_score"""
        
        result = self.discovery_agent.run(task)
        
        # Parse results
        topics = []
        try:
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result
            
            if isinstance(data, list):
                topics_data = data
            elif isinstance(data, dict) and "topics" in data:
                topics_data = data["topics"]
            else:
                topics_data = [data]
            
            for i, topic_data in enumerate(topics_data[:num_topics]):
                topic_id = topic_data.get("topic_id", f"topic_{i}_{int(time.time())}")
                keywords = topic_data.get("keywords", [])
                freshness = float(topic_data.get("freshness_score", 0.5))
                prior_score = float(topic_data.get("prior_score", 0.5))
                
                topics.append(Topic(
                    id=topic_id,
                    keywords=keywords,
                    freshness=freshness,
                    prior_score=prior_score
                ))
        except Exception as e:
            logger.error(f"Error parsing discovery results: {e}")
            # Fallback: create dummy topics
            for i in range(num_topics):
                topics.append(Topic(
                    id=f"topic_{i}_{int(time.time())}",
                    keywords=[f"keyword_{i}"],
                    freshness=0.5,
                    prior_score=0.5
                ))
        
        logger.info(f"Discovered {len(topics)} topics")
        return topics
    
    def rank_topics(self, topics: List[Topic]) -> List[Topic]:
        """
        Rank topics using CR-CA causal reasoning.
        
        Uses causal inference to estimate expected watch time uplift.
        
        Args:
            topics: List of topics to rank
            
        Returns:
            Ranked topics with causal scores
        """
        ranked = []
        
        for topic in topics:
            # Build causal analysis prompt
            causal_task = f"""
            Analyze the causal impact of creating a Python tutorial video on topic: {', '.join(topic.keywords)}
            
            This is a programming tutorial channel teaching Python with Swarms framework.
            
            Features:
            - Freshness: {topic.freshness}
            - Prior performance: {topic.prior_score}
            - Keywords: {topic.keywords}
            - Topic type: Programming/Business trend applicable to Swarms
            
            Estimate:
            1. Programming trend relevance (how current/trending is the programming concept)
            2. Business trend relevance (how relevant is the business application)
            3. Swarms applicability (how well Swarms framework applies to this topic)
            4. Educational value (learning value for Python developers)
            5. Code complexity (beginner/intermediate/advanced)
            6. Expected CTR (click-through rate)
            7. Expected watch time
            
            Use causal reasoning to account for confounders like:
            - Programming trend popularity
            - Swarms framework fit
            - Educational value for developers
            - Code complexity vs audience level
            - Competition level in programming tutorial space
            - Channel fit for Python/Swarms content
            """
            
            analysis = self.cr_ca.run(causal_task)
            
            # Extract scores from analysis
            try:
                causal_analysis = analysis.get("causal_analysis", {})
                if isinstance(causal_analysis, str):
                    # Try to extract numeric scores
                    import re
                    scores = re.findall(r'\d+\.?\d*', causal_analysis)
                    topic.causal_score = float(scores[0]) if scores else topic.prior_score
                    topic.confidence = float(scores[1]) if len(scores) > 1 else 0.7
                else:
                    topic.causal_score = float(causal_analysis.get("expected_watch_time", topic.prior_score))
                    topic.confidence = float(causal_analysis.get("confidence", 0.7))
            except Exception as e:
                logger.warning(f"Error extracting causal scores: {e}, using prior score")
                topic.causal_score = topic.prior_score
                topic.confidence = 0.5
            
            ranked.append(topic)
        
        # Sort by causal score + confidence (UCB-like)
        ranked.sort(key=lambda t: t.causal_score + 0.5 * t.confidence, reverse=True)
        
        logger.info(f"Ranked {len(ranked)} topics")
        return ranked
    
    def produce_video(self, topic: Topic, resume_from: Optional[str] = None) -> Video:
        """
        Produce a complete video from topic to final output with retry, validation, and progress tracking.
        
        Args:
            topic: Selected topic
            resume_from: Optional step to resume from (script, assets, compose)
            
        Returns:
            Video object with production state
        """
        # Generate idempotency key
        idempotency_key = hashlib.md5(f"{topic.id}_{topic.keywords}".encode()).hexdigest()
        
        # Check if video already exists with this key
        existing_video = None
        for vid in self.videos.values():
            if vid.idempotency_key == idempotency_key and vid.state in [VideoState.COMPOSED, VideoState.VALIDATED, VideoState.UPLOADED]:
                logger.info(f"Video with idempotency key {idempotency_key} already exists: {vid.video_id}")
                return vid
        
        video_id = hashlib.md5(f"{topic.id}_{time.time()}".encode()).hexdigest()[:12]
        video = Video(video_id=video_id, topic=topic, idempotency_key=idempotency_key)
        self.videos[video_id] = video
        
        logger.info(f"Starting production for video {video_id}: {topic.keywords[0] if topic.keywords else 'unknown'}")
        self.track_progress(video_id, "initialization", 0.0, ProductionStatus.IN_PROGRESS)
        
        try:
            # Step 1: Generate script
            if resume_from != "assets" and resume_from != "compose":
                self.track_progress(video_id, "script_generation", 10.0, ProductionStatus.IN_PROGRESS)
                video.state = VideoState.SCRIPTED
                
                # Use GoTAgent if available, otherwise fall back to standard agent
                script_problem = f"""Write a complete YouTube script for topic: {', '.join(topic.keywords)}

Requirements:
- Hook in first 15 seconds
- Introduction explaining trend and Swarms fit
- Setup section (environment, installation)
- Core tutorial with code walkthrough
- Real-world application demonstration
- Call-to-action

Use graph reasoning to explore different narrative structures and merge the best approach."""
                
                try:
                    if self.got_script_agent:
                        logger.info(f"Using GoTAgent for script generation for video {video_id}")
                        script_result = self.got_script_agent.run(script_problem)
                        # GoTAgent may return dict with answer key or just string
                        if isinstance(script_result, dict):
                            video.script = script_result.get("answer", str(script_result))
                        else:
                            video.script = str(script_result)
                    else:
                        logger.info(f"Using standard agent for script generation for video {video_id}")
                        script_task = f"Write a complete YouTube script for topic: {', '.join(topic.keywords)}. Include hook, structure, timestamps, and CTA."
                        video.script = self.script_agent.run(script_task)
                except Exception as e:
                    self.track_progress(video_id, "script_generation", 10.0, ProductionStatus.FAILED, str(e))
                    # Fallback to standard agent if GoTAgent fails
                    if self.got_script_agent:
                        logger.warning(f"GoTAgent failed, falling back to standard agent: {e}")
                        try:
                            script_task = f"Write a complete YouTube script for topic: {', '.join(topic.keywords)}. Include hook, structure, timestamps, and CTA."
                            video.script = self.script_agent.run(script_task)
                        except Exception as e2:
                            raise e2
                    else:
                        raise
                
                # Save script
                script_path = f"{self.workspace_dir}/scripts/{video_id}.md"
                with open(script_path, "w") as f:
                    f.write(video.script)
                self._save_production_state()
            else:
                # Load existing script
                script_path = f"{self.workspace_dir}/scripts/{video_id}.md"
                if os.path.exists(script_path):
                    with open(script_path, "r") as f:
                        video.script = f.read()
                else:
                    raise FileNotFoundError(f"Script file not found: {script_path}")
            
            # Step 2: Generate assets (actually create them)
            if resume_from != "compose":
                self.track_progress(video_id, "asset_generation", 30.0, ProductionStatus.IN_PROGRESS)
                video.state = VideoState.ASSETS_GENERATED
                asset_task = f"""Generate video assets for script:
{video.script[:500]}...

Create:
1. Thumbnail description (high CTR design) - then generate the actual image
2. B-roll image suggestions with timestamps
3. On-screen text prompts
4. Caption file structure

Use the generate_thumbnail_image tool to create actual thumbnail images."""
                
                try:
                    assets_result = self.asset_agent.run(asset_task)
                    try:
                        if isinstance(assets_result, str):
                            video.assets = json.loads(assets_result)
                        else:
                            video.assets = assets_result
                    except:
                        video.assets = {"raw": str(assets_result)}
                except Exception as e:
                    self.track_progress(video_id, "asset_generation", 30.0, ProductionStatus.FAILED, str(e))
                    raise
                
                # Generate actual thumbnail
                thumbnail_prompt = video.assets.get("thumbnail_prompt", f"YouTube thumbnail for: {', '.join(topic.keywords)}")
                thumbnail_path = f"{self.workspace_dir}/images/{video_id}_thumbnail.png"
                try:
                    generate_thumbnail_image(thumbnail_prompt, thumbnail_path)
                    video.assets["thumbnail_path"] = thumbnail_path
                except Exception as e:
                    self.track_progress(video_id, "thumbnail_generation", 35.0, ProductionStatus.FAILED, str(e))
                    raise
                
                # Generate TTS audio from script
                self.track_progress(video_id, "tts_generation", 50.0, ProductionStatus.IN_PROGRESS)
                clean_text = re.sub(r'[#*\[\]()]', '', video.script)
                clean_text = re.sub(r'\n+', ' ', clean_text)[:5000]  # Limit length
                
                audio_path = f"{self.workspace_dir}/audio/{video_id}.mp3"
                try:
                    generate_tts_audio(clean_text, audio_path)
                    video.assets["audio_path"] = audio_path
                except Exception as e:
                    self.track_progress(video_id, "tts_generation", 50.0, ProductionStatus.FAILED, str(e))
                    raise
                
                # Generate additional images for B-roll
                self.track_progress(video_id, "broll_generation", 60.0, ProductionStatus.IN_PROGRESS)
                broll_images = []
                for i, broll_desc in enumerate(video.assets.get("broll_descriptions", [])[:5]):
                    img_path = f"{self.workspace_dir}/images/{video_id}_broll_{i}.png"
                    try:
                        generate_thumbnail_image(broll_desc, img_path)
                        broll_images.append(img_path)
                    except Exception as e:
                        logger.warning(f"Failed to generate B-roll image {i}: {e}")
                video.assets["broll_images"] = broll_images
                self._save_production_state()
            else:
                # Load existing assets
                metadata_path = f"{self.workspace_dir}/videos/{video_id}_metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        video.assets = metadata.get("assets", {})
            
            # Step 3: Compose video (actually create video file)
            self.track_progress(video_id, "video_composition", 70.0, ProductionStatus.IN_PROGRESS)
            video.state = VideoState.COMPOSED
            video_path = f"{self.workspace_dir}/videos/{video_id}.mp4"
            
            # Use actual assets to compose video
            images = [video.assets.get("thumbnail_path")] + video.assets.get("broll_images", [])
            images = [img for img in images if img and os.path.exists(img)]
            
            if not images:
                # Fallback: use thumbnail if available
                if video.assets.get("thumbnail_path") and os.path.exists(video.assets["thumbnail_path"]):
                    images = [video.assets["thumbnail_path"]]
                else:
                    # Create placeholder
                    placeholder = f"{self.workspace_dir}/images/{video_id}_placeholder.png"
                    subprocess.run([
                        "ffmpeg", "-f", "lavfi", "-i", "color=c=blue:s=1920x1080",
                        "-frames:v", "1", "-y", placeholder
                    ], capture_output=True, check=False)
                    images = [placeholder]
            
            # Get audio duration for text overlay timing
            try:
                audio_duration_result = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", video.assets["audio_path"]],
                    capture_output=True, text=True, check=True
                )
                audio_duration = float(audio_duration_result.stdout.strip())
            except:
                audio_duration = 60.0  # Default fallback
            
            # Generate text overlays from script
            text_overlays = None
            if video.script:
                try:
                    text_overlays = parse_script_for_text_overlays(
                        video.script,
                        audio_duration,
                        num_overlays=min(5, len(images))
                    )
                    # Override with asset agent suggestions if available
                    if video.assets.get("text_overlay_phrases"):
                        # Use agent-suggested phrases with timing
                        text_overlays = []
                        phrases = video.assets.get("text_overlay_phrases", [])[:5]
                        interval = audio_duration / (len(phrases) + 1)
                        for i, phrase in enumerate(phrases):
                            start = (i + 1) * interval
                            text_overlays.append(TextOverlay(
                                start_time=start,
                                end_time=start + 4.0,
                                text=phrase[:60],
                                position="bottom" if i % 2 == 0 else "top"
                            ))
                except Exception as e:
                    logger.warning(f"Could not generate text overlays: {e}")
            
            # Get background music
            background_music = None
            music_volume = 0.3
            if video.assets.get("music_style"):
                try:
                    background_music = get_background_music(
                        topic,
                        audio_duration,
                        self.workspace_dir
                    )
                    music_volume = 0.3  # Default, could be configurable
                except Exception as e:
                    logger.warning(f"Could not get background music: {e}")
            
            # Get transition type from assets
            transition_type = video.assets.get("transition_type", "fade")
            transition_duration = 0.5
            
            try:
                compose_video_with_ffmpeg(
                    audio_path=video.assets["audio_path"],
                    images=images,
                    output_path=video_path,
                    duration_per_image=None,  # Auto-calculate
                    transition_type=transition_type,
                    transition_duration=transition_duration,
                    text_overlays=text_overlays,
                    background_music_path=background_music,
                    music_volume=music_volume,
                )
            except Exception as e:
                self.track_progress(video_id, "video_composition", 70.0, ProductionStatus.FAILED, str(e))
                raise
            
            video.metadata["video_path"] = video_path
            video.metadata["video_file"] = os.path.exists(video_path)
            
            logger.info(f"Video file created: {video_path} ({os.path.getsize(video_path) / 1024 / 1024:.2f} MB)")
            
            # Step 4: Validate video quality
            self.track_progress(video_id, "quality_validation", 85.0, ProductionStatus.IN_PROGRESS)
            validation_results = validate_video_quality(video_path, video.assets.get("audio_path"))
            video.metadata["validation"] = validation_results
            
            if not validation_results["valid"]:
                errors = "; ".join(validation_results["errors"])
                self.track_progress(video_id, "quality_validation", 85.0, ProductionStatus.FAILED, errors)
                video.state = VideoState.FAILED
                logger.error(f"Video validation failed: {errors}")
                return video
            
            video.state = VideoState.VALIDATED
            self.track_progress(video_id, "completed", 100.0, ProductionStatus.COMPLETED)
            
            # Save metadata
            metadata_path = f"{self.workspace_dir}/videos/{video_id}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump({
                    "video_id": video_id,
                    "topic": {
                        "id": topic.id,
                        "keywords": topic.keywords,
                        "causal_score": topic.causal_score
                    },
                    "script_path": script_path,
                    "assets": video.assets,
                    "video_path": video.metadata.get("video_path"),
                    "video_file_exists": video.metadata.get("video_file", False),
                    "state": video.state.value,
                    "created_at": video.created_at
                }, f, indent=2)
            
            logger.info(f"Video {video_id} production completed")
            
        except Exception as e:
            logger.error(f"Error producing video {video_id}: {e}")
            video.state = VideoState.FAILED
            self.track_progress(video_id, "failed", 0.0, ProductionStatus.FAILED, str(e))
            self._save_production_state()
        
        return video
    
    def generate_seo_metadata(self, video: Video) -> Dict[str, Any]:
        """
        Generate SEO-optimized metadata for Python tutorial video.
        
        Args:
            video: Video object
            
        Returns:
            Dictionary with title, description, tags, category
        """
        metadata_task = f"""Generate SEO-optimized YouTube metadata for a Python programming tutorial video teaching Swarms framework.

Topic: {', '.join(video.topic.keywords)}
Script excerpt: {video.script[:300] if video.script else 'N/A'}...

This is a programming tutorial channel focused on Python tutorials with Swarms framework.

Create:
1. Title: Max 60 characters, include: Python, Swarms, tutorial keywords, trend/application
   Example: "Build Passive Income Site with Flask + Swarms | Python Tutorial"
2. Description: Max 5000 characters
   - First 125 chars: keyword-rich with Python, Swarms, tutorial keywords
   - Include: What you'll learn, code overview, Swarms features demonstrated
   - Add timestamps if available
   - Include link to Swarms docs: https://docs.swarms.world/en/latest/
   - Call-to-action for subscription
3. Tags: Include Python, Swarms, tutorial, programming, automation, multi-agent, and trend-specific tags (10-15 tags)
   Examples: python, swarms, python tutorial, swarms framework, multi-agent systems, python automation, flask, fastapi
4. Category ID: 28 (Science & Technology) for programming tutorials

Output JSON format:
{{
    "title": "...",
    "description": "...",
    "tags": ["python", "swarms", "tutorial", ...],
    "category_id": "28"
}}"""
        
        try:
            result = self.script_agent.run(metadata_task)
            if isinstance(result, str):
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    metadata = json.loads(json_match.group())
                else:
                    # Fallback: create basic metadata
                    metadata = {
                        "title": f"{video.topic.keywords[0] if video.topic.keywords else 'Video'} - Complete Guide",
                        "description": video.script[:500] if video.script else "",
                        "tags": video.topic.keywords[:10],
                        "category_id": "22"
                    }
            else:
                metadata = result
            
            # Ensure title is <= 60 chars
            if "title" in metadata and len(metadata["title"]) > 60:
                metadata["title"] = metadata["title"][:57] + "..."
            
            # Ensure description is <= 5000 chars
            if "description" in metadata and len(metadata["description"]) > 5000:
                metadata["description"] = metadata["description"][:4997] + "..."
            
            return metadata
        except Exception as e:
            logger.error(f"Error generating SEO metadata: {e}")
            # Fallback metadata
            return {
                "title": f"{video.topic.keywords[0] if video.topic.keywords else 'Video'} - Complete Guide",
                "description": video.script[:500] if video.script else "",
                "tags": video.topic.keywords[:10] if video.topic.keywords else [],
                "category_id": "22"
            }
    
    @retry_on_failure(max_attempts=3, backoff_base=2)
    def upload_to_youtube(self, video: Video, metadata: Optional[Dict[str, Any]] = None, schedule_time: Optional[datetime] = None) -> Optional[str]:
        """
        Upload video to YouTube with resumable upload.
        
        Args:
            video: Video object
            metadata: Optional metadata dict (generated if not provided)
            schedule_time: Optional datetime to schedule publish
            
        Returns:
            YouTube video ID if successful, None otherwise
        """
        if not self.youtube_service:
            logger.warning("YouTube service not initialized")
            return None
        
        if not self.quota_manager.check_quota("upload"):
            logger.warning("Insufficient quota for upload")
            return None
        
        video_path = video.metadata.get("video_path")
        if not video_path or not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        try:
            video.state = VideoState.UPLOADING
            self.track_progress(video.video_id, "uploading", 90.0, ProductionStatus.IN_PROGRESS)
            
            # Generate metadata if not provided
            if not metadata:
                metadata = self.generate_seo_metadata(video)
            
            # Prepare video metadata
            body = {
                "snippet": {
                    "title": metadata.get("title", "Untitled Video"),
                    "description": metadata.get("description", ""),
                    "tags": metadata.get("tags", []),
                    "categoryId": metadata.get("category_id", "22"),
                },
                "status": {
                    "privacyStatus": "private",  # Start as private, can be changed
                }
            }
            
            # Add scheduled publish time if provided
            if schedule_time:
                body["status"]["publishAt"] = schedule_time.isoformat() + "Z"
            
            # Create media upload
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
            
            # Insert video
            insert_request = self.youtube_service.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media
            )
            
            # Resumable upload
            response = None
            while response is None:
                status, response = insert_request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    self.track_progress(video.video_id, "uploading", 90.0 + (progress * 0.1), ProductionStatus.IN_PROGRESS)
                    logger.info(f"Upload progress: {progress}%")
            
            youtube_id = response["id"]
            video.youtube_id = youtube_id
            video.state = VideoState.UPLOADED
            self.quota_manager.use_quota("upload")
            
            # Upload thumbnail
            thumbnail_path = video.assets.get("thumbnail_path")
            if thumbnail_path and os.path.exists(thumbnail_path) and self.quota_manager.check_quota("thumbnail"):
                try:
                    self.youtube_service.thumbnails().set(
                        videoId=youtube_id,
                        media_body=MediaFileUpload(thumbnail_path)
                    ).execute()
                    self.quota_manager.use_quota("thumbnail")
                    logger.info(f"Thumbnail uploaded for {youtube_id}")
                except Exception as e:
                    logger.warning(f"Failed to upload thumbnail: {e}")
            
            self.track_progress(video.video_id, "uploaded", 100.0, ProductionStatus.COMPLETED)
            self._save_production_state()
            logger.info(f"Video uploaded to YouTube: {youtube_id}")
            return youtube_id
            
        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
            video.state = VideoState.FAILED
            self.track_progress(video.video_id, "upload_failed", 90.0, ProductionStatus.FAILED, str(e))
            return None
    
    def fetch_video_analytics(self, video: Video, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, float]:
        """
        Fetch analytics data for uploaded video.
        
        Args:
            video: Video object with youtube_id
            start_date: Start date (YYYY-MM-DD), defaults to 7 days ago
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            Dictionary with analytics metrics
        """
        if not video.youtube_id:
            logger.warning("Video has no YouTube ID")
            return {}
        
        if not self.youtube_analytics_service:
            logger.warning("Analytics service not available")
            return {}
        
        if not self.quota_manager.check_quota("analytics"):
            logger.warning("Insufficient quota for analytics")
            return {}
        
        try:
            # Default date range: last 7 days
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            # Get channel ID first
            channel_response = self.youtube_service.channels().list(part="id", mine=True).execute()
            if not channel_response.get("items"):
                logger.warning("Could not get channel ID")
                return {}
            
            channel_id = channel_response["items"][0]["id"]
            
            # Query analytics
            analytics_response = self.youtube_analytics_service.reports().query(
                ids=f"channel=={channel_id}",
                startDate=start_date,
                endDate=end_date,
                metrics="views,estimatedMinutesWatched,averageViewDuration,subscribersGained",
                dimensions="video",
                filters=f"video=={video.youtube_id}"
            ).execute()
            
            self.quota_manager.use_quota("analytics")
            
            # Parse results
            metrics = {}
            if analytics_response.get("rows"):
                row = analytics_response["rows"][0]
                metrics = {
                    "views": float(row[0]) if len(row) > 0 else 0,
                    "watch_time_minutes": float(row[1]) if len(row) > 1 else 0,
                    "avg_view_duration_seconds": float(row[2]) if len(row) > 2 else 0,
                    "subscribers_gained": float(row[3]) if len(row) > 3 else 0,
                }
                
                # Calculate derived metrics
                if metrics["views"] > 0:
                    metrics["retention"] = metrics["avg_view_duration_seconds"] / (video.metadata.get("validation", {}).get("duration_seconds", 60) or 60)
                else:
                    metrics["retention"] = 0
                
                video.metrics.update(metrics)
                logger.info(f"Analytics fetched for {video.youtube_id}: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to fetch analytics: {e}")
            return {}
    
    def schedule_upload(self, video: Video, metadata: Optional[Dict[str, Any]] = None) -> Optional[datetime]:
        """
        Calculate optimal publish time and schedule upload.
        
        Args:
            video: Video object
            metadata: Optional metadata
            
        Returns:
            Scheduled datetime if successful
        """
        # Use CR-CA to determine optimal publish time
        scheduling_task = f"""Determine optimal YouTube publish time for video:
Topic: {', '.join(video.topic.keywords)}
Current time: {datetime.now().isoformat()}

Consider:
- Audience timezone patterns (assume US/EU if unknown)
- Day of week effects (Tuesday-Thursday typically best)
- Time of day (2-4 PM or 6-8 PM typically best)
- Competition level
- Topic seasonality

Output optimal publish datetime in ISO format (YYYY-MM-DDTHH:MM:SS)."""
        
        try:
            result = self.cr_ca.run(scheduling_task)
            # Extract datetime from result (simplified - in production would parse better)
            # For now, default to tomorrow at 2 PM
            schedule_time = datetime.now() + timedelta(days=1)
            schedule_time = schedule_time.replace(hour=14, minute=0, second=0, microsecond=0)
            
            # Upload with scheduled time
            youtube_id = self.upload_to_youtube(video, metadata, schedule_time)
            if youtube_id:
                logger.info(f"Video scheduled for {schedule_time}")
                return schedule_time
        except Exception as e:
            logger.error(f"Failed to schedule upload: {e}")
        
        return None
    
    def add_to_playlist(self, video: Video, playlist_name: Optional[str] = None) -> Optional[str]:
        """
        Add video to playlist (create if doesn't exist).
        
        Args:
            video: Video object with youtube_id
            playlist_name: Playlist name (defaults to topic-based)
            
        Returns:
            Playlist ID if successful
        """
        if not video.youtube_id or not self.youtube_service:
            return None
        
        if not self.quota_manager.check_quota("analytics"):  # Playlist operations use minimal quota
            return None
        
        try:
            # Generate playlist name from topic
            if not playlist_name:
                playlist_name = f"{video.topic.keywords[0] if video.topic.keywords else 'Videos'} Series"
            
            # Search for existing playlist
            playlists_response = self.youtube_service.playlists().list(
                part="snippet",
                mine=True,
                maxResults=50
            ).execute()
            
            playlist_id = None
            for item in playlists_response.get("items", []):
                if item["snippet"]["title"] == playlist_name:
                    playlist_id = item["id"]
                    break
            
            # Create playlist if doesn't exist
            if not playlist_id:
                playlist_body = {
                    "snippet": {
                        "title": playlist_name,
                        "description": f"Playlist for {', '.join(video.topic.keywords)}"
                    },
                    "status": {
                        "privacyStatus": "public"
                    }
                }
                playlist_response = self.youtube_service.playlists().insert(
                    part="snippet,status",
                    body=playlist_body
                ).execute()
                playlist_id = playlist_response["id"]
                logger.info(f"Created playlist: {playlist_name}")
            
            # Add video to playlist
            playlist_item_body = {
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video.youtube_id
                    }
                }
            }
            self.youtube_service.playlistItems().insert(
                part="snippet",
                body=playlist_item_body
            ).execute()
            
            logger.info(f"Added video {video.youtube_id} to playlist {playlist_name}")
            return playlist_id
            
        except Exception as e:
            logger.error(f"Failed to add to playlist: {e}")
            return None
    
    def optimize_video(self, video: Video, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize video using CR-CA for causal analysis of what works.
        Uses AERASIGMA to find similar successful videos and suggest optimizations.
        
        Args:
            video: Video to optimize
            metrics: Current performance metrics
            
        Returns:
            Optimization recommendations
        """
        video.metrics = metrics
        
        # Use AERASIGMA to find similar successful videos if available
        aera_suggestions = {}
        if self.aera_learner:
            try:
                # Query AERASIGMA for similar successful videos
                query_state = {
                    "topic_keywords": video.topic.keywords,
                    "metrics": metrics
                }
                
                # Access experience graph if available
                if hasattr(self.aera_learner, 'kb') and hasattr(self.aera_learner.kb, 'experience_graph'):
                    similar_experiences = self.aera_learner.kb.experience_graph.retrieve_similar(
                        query_state=query_state,
                        k=5
                    )
                    
                    if similar_experiences:
                        aera_suggestions["similar_successful_videos"] = len(similar_experiences)
                        logger.info(f"AERASIGMA found {len(similar_experiences)} similar successful videos")
                
                # Use AERASIGMA's causal models to suggest optimizations
                optimization_task_aera = f"""Based on learned patterns, suggest optimizations for video with metrics: {metrics}"""
                try:
                    optimization_result = self.aera_learner.run(optimization_task_aera)
                    if isinstance(optimization_result, dict):
                        aera_suggestions.update(optimization_result)
                    else:
                        aera_suggestions["aera_recommendations"] = str(optimization_result)
                except Exception as e:
                    logger.warning(f"AERASIGMA optimization query failed: {e}")
                    
            except Exception as e:
                logger.warning(f"AERASIGMA optimization failed: {e}")
        
        optimization_task = f"""
        Analyze video performance and optimize:
        
        Video: {video.topic.keywords[0] if video.topic.keywords else 'unknown'}
        Metrics:
        - CTR: {metrics.get('ctr', 0):.2%}
        - Watch time: {metrics.get('watch_time', 0):.2f}s
        - Retention: {metrics.get('retention', 0):.2%}
        - Views: {metrics.get('views', 0)}
        
        Use causal reasoning to:
        1. Identify which factors (title, thumbnail, content) drive performance
        2. Estimate uplift from changing each factor
        3. Recommend optimal title/thumbnail variants
        4. Suggest content improvements
        
        Account for confounders like timing, topic, and audience.
        """
        
        # Use CR-CA for causal analysis
        analysis = self.cr_ca.run(optimization_task)
        
        # Use optimizer agent for actionable recommendations
        recommendations = self.optimizer_agent.run(optimization_task)
        
        try:
            if isinstance(recommendations, str):
                recs = json.loads(recommendations)
            else:
                recs = recommendations
        except:
            recs = {"raw": str(recommendations)}
        
        # Merge AERASIGMA suggestions
        if aera_suggestions:
            recs["aera_suggestions"] = aera_suggestions
        
        video.state = VideoState.OPTIMIZED
        video.metadata["optimization"] = recs
        
        logger.info(f"Optimized video {video.video_id}")
        return recs
    
    def pause_production(self):
        """Pause production loop"""
        self.production_paused = True
        logger.info("Production paused")
    
    def resume_production_loop(self):
        """Resume production loop"""
        self.production_paused = False
        logger.info("Production resumed")
    
    def analytics_feedback_loop(self):
        """
        Periodically fetch analytics and update CR-CA priors.
        Runs as background task to improve topic ranking.
        Uses AERASIGMA to learn causal models from video performance.
        """
        logger.info("Starting analytics feedback loop")
        
        uploaded_videos = [v for v in self.videos.values() if v.youtube_id and v.state == VideoState.UPLOADED]
        
        for video in uploaded_videos:
            try:
                # Fetch analytics
                metrics = self.fetch_video_analytics(video)
                
                if metrics:
                    # Update video metrics
                    video.metrics.update(metrics)
                    
                    # Use AERASIGMA to learn from metrics if available
                    if self.aera_learner:
                        try:
                            # Convert video state to AERASIGMA observation
                            observation = {
                                "topic_keywords": video.topic.keywords,
                                "script_length": len(video.script) if video.script else 0,
                                "has_thumbnails": bool(video.assets.get("thumbnail_path")),
                                "video_duration": video.metadata.get("validation", {}).get("duration_seconds", 0),
                                "metrics": metrics
                            }
                            
                            # Determine success based on retention threshold
                            success_threshold = 0.4
                            is_successful = metrics.get("retention", 0) > success_threshold
                            
                            # Learn causal relationships
                            self.aera_learner.learn_from_experience(
                                x_t=observation,
                                u_t="video_production",
                                x_tp1={"success": is_successful, "retention": metrics.get("retention", 0)}
                            )
                            
                            logger.info(f"AERASIGMA learned from video {video.video_id} (success: {is_successful})")
                        except Exception as e:
                            logger.warning(f"AERASIGMA learning failed for {video.video_id}: {e}")
                    
                    # Use CR-CA to analyze what worked
                    feedback_task = f"""Analyze video performance and update topic ranking:
                    
Video: {video.topic.keywords[0] if video.topic.keywords else 'unknown'}
Metrics:
- Views: {metrics.get('views', 0)}
- Watch time: {metrics.get('watch_time_minutes', 0)} minutes
- Retention: {metrics.get('retention', 0):.2%}
- Subscribers gained: {metrics.get('subscribers_gained', 0)}

Update causal model priors for:
- Topic keywords: {video.topic.keywords}
- Expected watch time (adjust based on actual: {metrics.get('watch_time_minutes', 0)} min)
- Expected CTR (adjust based on views/impressions if available)

Provide updated causal scores for similar topics."""
                    
                    # This would update the CR-CA agent's priors in a full implementation
                    # For now, we log the feedback
                    logger.info(f"Analytics feedback for {video.video_id}: {metrics}")
                    
            except Exception as e:
                logger.warning(f"Error in analytics feedback for {video.video_id}: {e}")
    
    def cleanup_failed_productions(self, max_age_hours: int = 24):
        """
        Clean up failed productions older than max_age_hours.
        
        Args:
            max_age_hours: Maximum age in hours for failed productions
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        failed_videos = [
            v for v in self.videos.values()
            if v.state == VideoState.FAILED and (current_time - v.created_at) > max_age_seconds
        ]
        
        for video in failed_videos:
            logger.info(f"Cleaning up failed video: {video.video_id}")
            # Could delete files, remove from state, etc.
            # For now, just log
    
    # ============================================================================
    # Channel Management Methods
    # ============================================================================
    
    def get_channel_info(self) -> Dict[str, Any]:
        """
        Get complete channel information including branding, statistics, and settings.
        
        Returns:
            Dictionary with channel information
        """
        if not self.youtube_service:
            logger.warning("YouTube service not initialized")
            return {}
        
        if not self.quota_manager.check_quota("analytics"):  # Use analytics quota for channel info
            logger.warning("Insufficient quota for channel info")
            return {}
        
        try:
            response = self.youtube_service.channels().list(
                part="snippet,brandingSettings,contentDetails,statistics,status,contentOwnerDetails",
                mine=True
            ).execute()
            
            self.quota_manager.use_quota("analytics")
            
            if not response.get("items"):
                logger.warning("No channel found")
                return {}
            
            channel = response["items"][0]
            snippet = channel.get("snippet", {})
            branding = channel.get("brandingSettings", {})
            statistics = channel.get("statistics", {})
            status = channel.get("status", {})
            
            channel_info = {
                "id": channel.get("id"),
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "custom_url": snippet.get("customUrl"),
                "published_at": snippet.get("publishedAt"),
                "country": snippet.get("country"),
                "default_language": snippet.get("defaultLanguage"),
                "subscriber_count": int(statistics.get("subscriberCount", 0)),
                "video_count": int(statistics.get("videoCount", 0)),
                "view_count": int(statistics.get("viewCount", 0)),
                "is_verified": status.get("isLinked"),
                "privacy_status": status.get("privacyStatus"),
                "banner_external_url": branding.get("image", {}).get("bannerExternalUrl"),
                "profile_picture_url": snippet.get("thumbnails", {}).get("default", {}).get("url"),
            }
            
            logger.info(f"Retrieved channel info: {channel_info.get('title')}")
            return channel_info
            
        except Exception as e:
            logger.error(f"Failed to get channel info: {e}")
            return {}
    
    @retry_on_failure(max_attempts=3, backoff_base=2)
    def update_channel_description(self, description: str) -> bool:
        """
        Update channel description.
        
        Args:
            description: New channel description (max 1000 characters)
            
        Returns:
            True if successful
        """
        if not self.youtube_service:
            logger.warning("YouTube service not initialized")
            return False
        
        if len(description) > 1000:
            logger.warning(f"Description too long ({len(description)} chars), truncating to 1000")
            description = description[:1000]
        
        if not self.quota_manager.check_quota("upload"):  # Channel updates use upload quota
            logger.warning("Insufficient quota for channel update")
            return False
        
        try:
            # Get current channel info
            response = self.youtube_service.channels().list(
                part="snippet",
                mine=True
            ).execute()
            
            if not response.get("items"):
                logger.error("Could not get channel")
                return False
            
            channel = response["items"][0]
            channel["snippet"]["description"] = description
            
            # Update channel
            update_response = self.youtube_service.channels().update(
                part="snippet",
                body=channel
            ).execute()
            
            self.quota_manager.use_quota("upload")
            logger.info("Channel description updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update channel description: {e}")
            return False
    
    @retry_on_failure(max_attempts=3, backoff_base=2)
    def update_channel_name(self, name: str) -> bool:
        """
        Update channel name.
        
        Args:
            name: New channel name
            
        Returns:
            True if successful
        """
        if not self.youtube_service:
            logger.warning("YouTube service not initialized")
            return False
        
        if not self.quota_manager.check_quota("upload"):
            logger.warning("Insufficient quota for channel update")
            return False
        
        try:
            # Get current channel info
            response = self.youtube_service.channels().list(
                part="snippet",
                mine=True
            ).execute()
            
            if not response.get("items"):
                logger.error("Could not get channel")
                return False
            
            channel = response["items"][0]
            channel["snippet"]["title"] = name
            
            # Update channel
            update_response = self.youtube_service.channels().update(
                part="snippet",
                body=channel
            ).execute()
            
            self.quota_manager.use_quota("upload")
            logger.info(f"Channel name updated to: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update channel name: {e}")
            return False
    
    @retry_on_failure(max_attempts=3, backoff_base=2)
    def update_channel_banner(self, banner_image_path: str) -> bool:
        """
        Upload/update channel banner image.
        
        Args:
            banner_image_path: Path to banner image (recommended: 2560x1440)
            
        Returns:
            True if successful
        """
        if not self.youtube_service:
            logger.warning("YouTube service not initialized")
            return False
        
        if not os.path.exists(banner_image_path):
            logger.error(f"Banner image not found: {banner_image_path}")
            return False
        
        if not self.quota_manager.check_quota("upload"):
            logger.warning("Insufficient quota for banner upload")
            return False
        
        try:
            # Validate image format
            valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
            if not any(banner_image_path.lower().endswith(ext) for ext in valid_extensions):
                logger.error(f"Invalid image format. Supported: {valid_extensions}")
                return False
            
            # Upload banner
            with open(banner_image_path, 'rb') as banner_file:
                media = MediaFileUpload(banner_image_path, mimetype='image/jpeg', resumable=True)
                response = self.youtube_service.channelBanners().insert(
                    media_body=media
                ).execute()
            
            # Get banner URL from response
            banner_url = response.get("url")
            
            # Update channel branding settings
            channel_response = self.youtube_service.channels().list(
                part="brandingSettings",
                mine=True
            ).execute()
            
            if channel_response.get("items"):
                channel = channel_response["items"][0]
                if "brandingSettings" not in channel:
                    channel["brandingSettings"] = {}
                if "image" not in channel["brandingSettings"]:
                    channel["brandingSettings"]["image"] = {}
                channel["brandingSettings"]["image"]["bannerExternalUrl"] = banner_url
                
                self.youtube_service.channels().update(
                    part="brandingSettings",
                    body=channel
                ).execute()
            
            self.quota_manager.use_quota("upload")
            logger.info(f"Channel banner updated: {banner_image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update channel banner: {e}")
            return False
    
    def update_channel_profile_picture(self, profile_image_path: str) -> bool:
        """
        Upload/update channel profile picture.
        
        Note: YouTube API doesn't directly support profile picture upload.
        This would need to be done via Google+ API (deprecated) or manually.
        This method logs a warning and returns False.
        
        Args:
            profile_image_path: Path to profile image (recommended: 800x800)
            
        Returns:
            False (not supported via API)
        """
        logger.warning("Profile picture upload is not supported via YouTube Data API v3. "
                      "Please update manually via YouTube Studio.")
        return False
    
    def get_channel_settings(self) -> Dict[str, Any]:
        """
        Get channel settings including privacy, upload defaults, etc.
        
        Returns:
            Dictionary with channel settings
        """
        if not self.youtube_service:
            logger.warning("YouTube service not initialized")
            return {}
        
        try:
            response = self.youtube_service.channels().list(
                part="brandingSettings,contentDetails",
                mine=True
            ).execute()
            
            if not response.get("items"):
                return {}
            
            channel = response["items"][0]
            branding = channel.get("brandingSettings", {})
            
            settings = {
                "channel_keywords": branding.get("channel", {}).get("keywords", ""),
                "default_tab": branding.get("channel", {}).get("defaultTab", ""),
                "moderate_comments": branding.get("channel", {}).get("moderateComments", False),
                "show_related_channels": branding.get("channel", {}).get("showRelatedChannels", True),
                "show_browse_view": branding.get("channel", {}).get("showBrowseView", True),
            }
            
            return settings
            
        except Exception as e:
            logger.error(f"Failed to get channel settings: {e}")
            return {}
    
    @retry_on_failure(max_attempts=3, backoff_base=2)
    def update_privacy_settings(self, privacy_level: str) -> bool:
        """
        Update channel privacy settings.
        
        Args:
            privacy_level: "public", "unlisted", or "private"
            
        Returns:
            True if successful
        """
        if privacy_level not in ["public", "unlisted", "private"]:
            logger.error(f"Invalid privacy level: {privacy_level}")
            return False
        
        logger.warning("Privacy settings are managed per-video, not per-channel in YouTube API v3")
        return False
    
    def set_comment_moderation(self, enabled: bool) -> bool:
        """
        Configure comment moderation settings.
        
        Args:
            enabled: Enable comment moderation
            
        Returns:
            True if successful
        """
        if not self.youtube_service:
            logger.warning("YouTube service not initialized")
            return False
        
        if not self.quota_manager.check_quota("upload"):
            logger.warning("Insufficient quota for channel update")
            return False
        
        try:
            response = self.youtube_service.channels().list(
                part="brandingSettings",
                mine=True
            ).execute()
            
            if not response.get("items"):
                logger.error("Could not get channel")
                return False
            
            channel = response["items"][0]
            if "brandingSettings" not in channel:
                channel["brandingSettings"] = {}
            if "channel" not in channel["brandingSettings"]:
                channel["brandingSettings"]["channel"] = {}
            
            channel["brandingSettings"]["channel"]["moderateComments"] = enabled
            
            self.youtube_service.channels().update(
                part="brandingSettings",
                body=channel
            ).execute()
            
            self.quota_manager.use_quota("upload")
            logger.info(f"Comment moderation set to: {enabled}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set comment moderation: {e}")
            return False
    
    def check_verification_status(self) -> bool:
        """
        Check if channel is verified.
        
        Returns:
            True if verified, False otherwise
        """
        if not self.youtube_service:
            logger.warning("YouTube service not initialized")
            return False
        
        try:
            response = self.youtube_service.channels().list(
                part="status",
                mine=True
            ).execute()
            
            if not response.get("items"):
                return False
            
            status = response["items"][0].get("status", {})
            # isLinked indicates verification in some contexts
            # Actual verification badge is not directly queryable via API
            return status.get("isLinked", False)
            
        except Exception as e:
            logger.error(f"Failed to check verification status: {e}")
            return False
    
    def check_channel_eligibility(self) -> Dict[str, Any]:
        """
        Check channel eligibility for various features.
        
        Returns:
            Dictionary with eligibility status for different features
        """
        if not self.youtube_service:
            logger.warning("YouTube service not initialized")
            return {}
        
        try:
            channel_info = self.get_channel_info()
            statistics = {}
            
            # Get channel statistics for eligibility checks
            response = self.youtube_service.channels().list(
                part="statistics,contentDetails",
                mine=True
            ).execute()
            
            if response.get("items"):
                statistics = response["items"][0].get("statistics", {})
            
            subscriber_count = int(statistics.get("subscriberCount", 0))
            video_count = int(statistics.get("videoCount", 0))
            
            eligibility = {
                "custom_url_eligible": subscriber_count >= 100,  # Approximate threshold
                "monetization_eligible": subscriber_count >= 1000 and video_count >= 4,  # Approximate
                "live_streaming_eligible": subscriber_count >= 0,  # Generally available
                "community_features_eligible": subscriber_count >= 1000,  # Approximate
                "subscriber_count": subscriber_count,
                "video_count": video_count,
            }
            
            logger.info(f"Channel eligibility: {eligibility}")
            return eligibility
            
        except Exception as e:
            logger.error(f"Failed to check channel eligibility: {e}")
            return {}
    
    def get_account_status(self) -> Dict[str, Any]:
        """
        Get comprehensive account status information.
        
        Returns:
            Dictionary with account status details
        """
        if not self.youtube_service:
            logger.warning("YouTube service not initialized")
            return {}
        
        try:
            channel_info = self.get_channel_info()
            eligibility = self.check_channel_eligibility()
            verification = self.check_verification_status()
            
            status = {
                "channel_info": channel_info,
                "eligibility": eligibility,
                "is_verified": verification,
                "account_active": bool(channel_info),
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get account status: {e}")
            return {}
    
    def get_channel_statistics(self) -> Dict[str, Any]:
        """
        Get channel-level statistics and performance metrics.
        
        Returns:
            Dictionary with channel statistics
        """
        if not self.youtube_service or not self.youtube_analytics_service:
            logger.warning("YouTube services not initialized")
            return {}
        
        try:
            # Get channel ID
            channel_response = self.youtube_service.channels().list(part="id,statistics", mine=True).execute()
            if not channel_response.get("items"):
                return {}
            
            channel_id = channel_response["items"][0]["id"]
            statistics = channel_response["items"][0].get("statistics", {})
            
            # Get analytics data
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            analytics_response = self.youtube_analytics_service.reports().query(
                ids=f"channel=={channel_id}",
                startDate=start_date,
                endDate=end_date,
                metrics="views,estimatedMinutesWatched,subscribersGained,likes,comments,shares"
            ).execute()
            
            self.quota_manager.use_quota("analytics")
            
            analytics_data = {}
            if analytics_response.get("rows"):
                row = analytics_response["rows"][0]
                analytics_data = {
                    "views_30d": float(row[0]) if len(row) > 0 else 0,
                    "watch_time_minutes_30d": float(row[1]) if len(row) > 1 else 0,
                    "subscribers_gained_30d": float(row[2]) if len(row) > 2 else 0,
                    "likes_30d": float(row[3]) if len(row) > 3 else 0,
                    "comments_30d": float(row[4]) if len(row) > 4 else 0,
                    "shares_30d": float(row[5]) if len(row) > 5 else 0,
                }
            
            channel_stats = {
                "subscriber_count": int(statistics.get("subscriberCount", 0)),
                "video_count": int(statistics.get("videoCount", 0)),
                "total_views": int(statistics.get("viewCount", 0)),
                "analytics_30d": analytics_data,
            }
            
            logger.info(f"Retrieved channel statistics")
            return channel_stats
            
        except Exception as e:
            logger.error(f"Failed to get channel statistics: {e}")
            return {}
    
    def apply_channel_branding(self, brand_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply automated branding to channel.
        Uses agents to generate descriptions and organize channel sections.
        
        Args:
            brand_config: Optional branding configuration. If None, generates from video topics.
            
        Returns:
            Dictionary with branding application results
        """
        if not self.youtube_service:
            logger.warning("YouTube service not initialized")
            return {}
        
        results = {}
        
        try:
            # Generate channel description from video topics if not provided
            if not brand_config or "description" not in brand_config:
                # Collect topics from recent videos
                topics = []
                for video in list(self.videos.values())[:10]:  # Last 10 videos
                    if video.topic and video.topic.keywords:
                        topics.extend(video.topic.keywords)
                
                if topics:
                    # Use script agent to generate channel description
                    description_task = f"""Generate a compelling YouTube channel description for a programming tutorial channel focused on Python and Swarms framework.

Topics covered: {', '.join(set(topics[:20]))}

Requirements:
- Max 1000 characters
- Highlight Python + Swarms focus
- Include key topics/trends covered
- Call-to-action for subscription
- Link to Swarms docs: https://docs.swarms.world/en/latest/

Channel description:"""
                    
                    try:
                        generated_description = self.script_agent.run(description_task)
                        if isinstance(generated_description, str):
                            # Extract description if it's in a structured format
                            if len(generated_description) > 1000:
                                generated_description = generated_description[:1000]
                            brand_config = brand_config or {}
                            brand_config["description"] = generated_description
                    except Exception as e:
                        logger.warning(f"Failed to generate description: {e}")
            
            # Update channel description
            if brand_config and "description" in brand_config:
                success = self.update_channel_description(brand_config["description"])
                results["description_updated"] = success
            
            # Update channel name if provided
            if brand_config and "name" in brand_config:
                success = self.update_channel_name(brand_config["name"])
                results["name_updated"] = success
            
            # Update banner if provided
            if brand_config and "banner_path" in brand_config:
                success = self.update_channel_banner(brand_config["banner_path"])
                results["banner_updated"] = success
            
            # Set comment moderation if provided
            if brand_config and "moderate_comments" in brand_config:
                success = self.set_comment_moderation(brand_config["moderate_comments"])
                results["comment_moderation_updated"] = success
            
            logger.info(f"Channel branding applied: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to apply channel branding: {e}")
            return results
    
    def resume_video_production(self, video_id: str) -> Optional[Video]:
        """
        Resume production from last successful step.
        
        Args:
            video_id: Video ID to resume
            
        Returns:
            Video object if successful
        """
        if video_id not in self.videos:
            logger.error(f"Video {video_id} not found")
            return None
        
        video = self.videos[video_id]
        
        # Determine resume point
        if video.state == VideoState.SCRIPTED:
            return self.produce_video(video.topic, resume_from="assets")
        elif video.state == VideoState.ASSETS_GENERATED:
            return self.produce_video(video.topic, resume_from="compose")
        else:
            logger.warning(f"Cannot resume from state: {video.state}")
            return video
    
    def run_autonomous_loop(
        self,
        num_videos: int = 5,
        discover_interval: int = 3600,
        auto_upload: bool = False,
        auto_schedule: bool = False,
        analytics_interval: int = 86400,  # 24 hours
    ):
        """
        Run autonomous production loop with enhanced features.
        
        Args:
            num_videos: Number of videos to produce
            discover_interval: Seconds between topic discovery cycles
            auto_upload: Automatically upload videos to YouTube
            auto_schedule: Automatically schedule uploads
            analytics_interval: Seconds between analytics updates
        """
        logger.info(f"Starting autonomous loop: {num_videos} videos")
        
        produced = 0
        last_discovery = 0
        last_analytics = 0
        active_productions = []  # Track concurrent productions
        
        while produced < num_videos or len(active_productions) > 0:
            # Check if paused
            if self.production_paused:
                time.sleep(60)
                continue
            
            current_time = time.time()
            
            # Discover new topics periodically
            if current_time - last_discovery > discover_interval or len(self.topic_queue) < 3:
                logger.info("Discovering new topics...")
                topics = self.discover_topics(num_topics=10)
                ranked = self.rank_topics(topics)
                
                # Add to priority queue
                for topic in ranked:
                    heapq.heappush(self.topic_queue, topic)
                
                last_discovery = current_time
            
            # Check quota before starting new production
            if auto_upload and not self.quota_manager.check_quota("upload", 1600):
                wait_time = self.quota_manager.wait_for_quota_reset()
                if wait_time > 0:
                    logger.info(f"Waiting for quota reset: {wait_time/3600:.2f} hours")
                    time.sleep(min(wait_time, 3600))  # Wait max 1 hour at a time
                    continue
            
            # Start new productions (respect max_concurrent)
            while len(active_productions) < self.max_concurrent and self.topic_queue and produced < num_videos:
                topic = heapq.heappop(self.topic_queue)
                # Start production (in real implementation would be async)
                video = self.produce_video(topic)
                active_productions.append(video)
                produced += 1
                
                logger.info(f"Started production {produced}/{num_videos}: {video.video_id}")
            
            # Process active productions
            completed = []
            for video in active_productions:
                if video.state == VideoState.VALIDATED:
                    # Production complete, handle upload if enabled
                    if auto_upload:
                        if auto_schedule:
                            self.schedule_upload(video)
                        else:
                            self.upload_to_youtube(video)
                        
                        # Add to playlist
                        self.add_to_playlist(video)
                    
                    completed.append(video)
                    logger.info(f"Completed video: {video.video_id}")
                elif video.state == VideoState.FAILED:
                    completed.append(video)
                    logger.warning(f"Failed video: {video.video_id}")
            
            # Remove completed
            for video in completed:
                active_productions.remove(video)
            
            # Analytics feedback loop
            if current_time - last_analytics > analytics_interval:
                self.analytics_feedback_loop()
                last_analytics = current_time
            
            # Cleanup old failures
            self.cleanup_failed_productions()
            
            # Wait if nothing to do
            if not active_productions and not self.topic_queue:
                time.sleep(60)
        
        logger.info("Autonomous loop completed")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        Dictionary with configuration values
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}


def interactive_setup() -> Dict[str, Any]:
    """
    Interactive setup mode for configuration.
    
    Returns:
        Dictionary with configuration values
    """
    print("\n=== YouTube Studio Interactive Setup ===\n")
    config = {}
    
    # Model name
    default_model = "gpt-4o-mini"
    model_input = input(f"LLM Model Name [{default_model}]: ").strip()
    config["model_name"] = model_input if model_input else default_model
    
    # Workspace directory
    default_workspace = "yt_workspace"
    workspace_input = input(f"Workspace Directory [{default_workspace}]: ").strip()
    config["workspace_dir"] = workspace_input if workspace_input else default_workspace
    
    # Max concurrent
    default_concurrent = "3"
    concurrent_input = input(f"Max Concurrent Productions [{default_concurrent}]: ").strip()
    try:
        config["max_concurrent"] = int(concurrent_input) if concurrent_input else int(default_concurrent)
    except ValueError:
        config["max_concurrent"] = int(default_concurrent)
    
    # Google TTS credentials
    tts_input = input("Google Cloud TTS Credentials Path (optional, press Enter to skip): ").strip()
    if tts_input and os.path.exists(tts_input):
        config["google_tts_credentials"] = tts_input
    elif tts_input:
        print(f"Warning: File not found: {tts_input}")
    
    # Replicate API token
    replicate_input = input("Replicate API Token (optional, press Enter to skip): ").strip()
    if replicate_input:
        config["replicate_api_token"] = replicate_input
    
    # YouTube credentials
    youtube_input = input("YouTube API Credentials Path (optional, press Enter to skip): ").strip()
    if youtube_input and os.path.exists(youtube_input):
        config["youtube_credentials"] = youtube_input
    elif youtube_input:
        print(f"Warning: File not found: {youtube_input}")
    
    # Production settings
    print("\n--- Production Settings ---")
    num_videos_input = input("Number of Videos to Produce [3]: ").strip()
    try:
        config["num_videos"] = int(num_videos_input) if num_videos_input else 3
    except ValueError:
        config["num_videos"] = 3
    
    discover_input = input("Topic Discovery Interval (seconds) [3600]: ").strip()
    try:
        config["discover_interval"] = int(discover_input) if discover_input else 3600
    except ValueError:
        config["discover_interval"] = 3600
    
    auto_upload_input = input("Auto Upload to YouTube? (y/n) [n]: ").strip().lower()
    config["auto_upload"] = auto_upload_input in ["y", "yes", "true", "1"]
    
    auto_schedule_input = input("Auto Schedule Uploads? (y/n) [n]: ").strip().lower()
    config["auto_schedule"] = auto_schedule_input in ["y", "yes", "true", "1"]
    
    analytics_input = input("Analytics Update Interval (seconds) [86400]: ").strip()
    try:
        config["analytics_interval"] = int(analytics_input) if analytics_input else 86400
    except ValueError:
        config["analytics_interval"] = 86400
    
    # Save config option
    save_input = input("\nSave configuration to file? (y/n) [n]: ").strip().lower()
    if save_input in ["y", "yes"]:
        config_path = input("Config file path [config.json]: ").strip() or "config.json"
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    return config


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Autonomous YouTube Studio - Python Tutorial Channel with Swarms Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic production (3 videos)
  python main.py --num-videos 3
  
  # Production with auto-upload
  python main.py --num-videos 5 --auto-upload --youtube-credentials creds.json
  
  # Using configuration file
  python main.py --config config.json
  
  # Interactive setup
  python main.py --interactive
  
  # Channel management
  python main.py --channel-info --apply-branding
  
  # Update channel description
  python main.py --update-description "New channel description"
  
  # Check channel eligibility
  python main.py --check-eligibility

Configuration File Format:
  {
    "model_name": "gpt-4o-mini",
    "workspace_dir": "yt_workspace",
    "max_concurrent": 3,
    "google_tts_credentials": "path/to/credentials.json",
    "replicate_api_token": "token",
    "youtube_credentials": "path/to/youtube-credentials.json",
    "production": {
      "num_videos": 5,
      "discover_interval": 3600,
      "auto_upload": true,
      "auto_schedule": true,
      "analytics_interval": 86400
    }
  }
        """
    )
    
    # Studio configuration
    parser.add_argument("--model-name", type=str, default=None,
                       help="LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--workspace-dir", type=str, default=None,
                       help="Workspace directory (default: yt_workspace)")
    parser.add_argument("--max-concurrent", type=int, default=None,
                       help="Max concurrent video productions (default: 3)")
    parser.add_argument("--google-tts-credentials", type=str, default=None,
                       help="Path to Google Cloud TTS credentials JSON")
    parser.add_argument("--replicate-api-token", type=str, default=None,
                       help="Replicate API token for image generation")
    parser.add_argument("--youtube-credentials", type=str, default=None,
                       help="Path to YouTube API credentials JSON")
    
    # Production settings
    parser.add_argument("--num-videos", type=int, default=None,
                       help="Number of videos to produce (default: 3)")
    parser.add_argument("--discover-interval", type=int, default=None,
                       help="Topic discovery interval in seconds (default: 3600)")
    parser.add_argument("--auto-upload", action="store_true",
                       help="Enable automatic YouTube upload")
    parser.add_argument("--auto-schedule", action="store_true",
                       help="Enable automatic scheduling of uploads")
    parser.add_argument("--analytics-interval", type=int, default=None,
                       help="Analytics update interval in seconds (default: 86400)")
    
    # Configuration and modes
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration JSON file")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive setup mode")
    
    # Channel management
    parser.add_argument("--channel-info", action="store_true",
                       help="Get and display channel information")
    parser.add_argument("--update-description", type=str, default=None,
                       help="Update channel description")
    parser.add_argument("--update-name", type=str, default=None,
                       help="Update channel name")
    parser.add_argument("--update-banner", type=str, default=None,
                       help="Upload channel banner (path to image file)")
    parser.add_argument("--check-eligibility", action="store_true",
                       help="Check channel eligibility for features")
    parser.add_argument("--apply-branding", action="store_true",
                       help="Apply automated channel branding")
    parser.add_argument("--channel-statistics", action="store_true",
                       help="Get channel statistics and analytics")
    
    return parser.parse_args()


def main():
    """
    Main entry point with CLI support.
    
    Supports:
    - Command-line arguments
    - Configuration file loading
    - Interactive setup mode
    - Channel management operations
    - Autonomous video production
    """
    args = parse_arguments()
    
    # Load configuration file if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        if not config:
            logger.error("Failed to load configuration file")
            sys.exit(1)
    
    # Run interactive setup if requested
    if args.interactive:
        interactive_config = interactive_setup()
        # Merge with file config (interactive takes precedence)
        config = {**config, **interactive_config}
    
    # Merge CLI arguments with config (CLI takes precedence)
    final_config = {}
    
    # Studio initialization parameters
    final_config["model_name"] = args.model_name or config.get("model_name", "gpt-4o-mini")
    final_config["workspace_dir"] = args.workspace_dir or config.get("workspace_dir", "yt_workspace")
    final_config["max_concurrent"] = args.max_concurrent or config.get("max_concurrent", 3)
    final_config["google_tts_credentials"] = args.google_tts_credentials or config.get("google_tts_credentials")
    final_config["replicate_api_token"] = args.replicate_api_token or config.get("replicate_api_token")
    final_config["youtube_credentials"] = args.youtube_credentials or config.get("youtube_credentials")
    
    # Production parameters
    production_config = config.get("production", {})
    final_config["num_videos"] = args.num_videos or production_config.get("num_videos") or config.get("num_videos", 3)
    final_config["discover_interval"] = args.discover_interval or production_config.get("discover_interval") or config.get("discover_interval", 3600)
    final_config["auto_upload"] = args.auto_upload or production_config.get("auto_upload", False) or config.get("auto_upload", False)
    final_config["auto_schedule"] = args.auto_schedule or production_config.get("auto_schedule", False) or config.get("auto_schedule", False)
    final_config["analytics_interval"] = args.analytics_interval or production_config.get("analytics_interval") or config.get("analytics_interval", 86400)
    
    # Initialize studio
    try:
        studio = YouTubeStudio(
            model_name=final_config["model_name"],
            workspace_dir=final_config["workspace_dir"],
            max_concurrent=final_config["max_concurrent"],
            google_tts_credentials=final_config.get("google_tts_credentials"),
            replicate_api_token=final_config.get("replicate_api_token"),
            youtube_credentials=final_config.get("youtube_credentials"),
        )
    except Exception as e:
        logger.error(f"Failed to initialize YouTube Studio: {e}")
        sys.exit(1)
    
    # Handle channel management commands
    channel_ops_performed = False
    
    if args.channel_info:
        channel_ops_performed = True
        print("\n=== Channel Information ===")
        channel_info = studio.get_channel_info()
        if channel_info:
            print(f"Channel: {channel_info.get('title', 'N/A')}")
            print(f"Description: {channel_info.get('description', 'N/A')[:100]}...")
            print(f"Subscribers: {channel_info.get('subscriber_count', 0):,}")
            print(f"Videos: {channel_info.get('video_count', 0):,}")
            print(f"Total Views: {channel_info.get('view_count', 0):,}")
            print(f"Custom URL: {channel_info.get('custom_url', 'N/A')}")
            print(f"Verified: {channel_info.get('is_verified', False)}")
        else:
            print("Failed to retrieve channel information")
    
    if args.update_description:
        channel_ops_performed = True
        print(f"\n=== Updating Channel Description ===")
        success = studio.update_channel_description(args.update_description)
        if success:
            print("Channel description updated successfully")
        else:
            print("Failed to update channel description")
    
    if args.update_name:
        channel_ops_performed = True
        print(f"\n=== Updating Channel Name ===")
        success = studio.update_channel_name(args.update_name)
        if success:
            print(f"Channel name updated to: {args.update_name}")
        else:
            print("Failed to update channel name")
    
    if args.update_banner:
        channel_ops_performed = True
        print(f"\n=== Updating Channel Banner ===")
        if not os.path.exists(args.update_banner):
            print(f"Error: Banner file not found: {args.update_banner}")
        else:
            success = studio.update_channel_banner(args.update_banner)
            if success:
                print("Channel banner updated successfully")
            else:
                print("Failed to update channel banner")
    
    if args.check_eligibility:
        channel_ops_performed = True
        print("\n=== Channel Eligibility ===")
        eligibility = studio.check_channel_eligibility()
        if eligibility:
            print(f"Custom URL Eligible: {eligibility.get('custom_url_eligible', False)}")
            print(f"Monetization Eligible: {eligibility.get('monetization_eligible', False)}")
            print(f"Live Streaming Eligible: {eligibility.get('live_streaming_eligible', False)}")
            print(f"Community Features Eligible: {eligibility.get('community_features_eligible', False)}")
            print(f"Subscribers: {eligibility.get('subscriber_count', 0):,}")
            print(f"Videos: {eligibility.get('video_count', 0):,}")
        else:
            print("Failed to check channel eligibility")
    
    if args.apply_branding:
        channel_ops_performed = True
        print("\n=== Applying Channel Branding ===")
        results = studio.apply_channel_branding()
        if results:
            print("Branding application results:")
            for key, value in results.items():
                print(f"  {key}: {value}")
        else:
            print("Failed to apply channel branding")
    
    if args.channel_statistics:
        channel_ops_performed = True
        print("\n=== Channel Statistics ===")
        stats = studio.get_channel_statistics()
        if stats:
            print(f"Subscribers: {stats.get('subscriber_count', 0):,}")
            print(f"Total Videos: {stats.get('video_count', 0):,}")
            print(f"Total Views: {stats.get('total_views', 0):,}")
            analytics = stats.get("analytics_30d", {})
            if analytics:
                print(f"\n30-Day Analytics:")
                print(f"  Views: {analytics.get('views_30d', 0):,.0f}")
                print(f"  Watch Time: {analytics.get('watch_time_minutes_30d', 0):,.0f} minutes")
                print(f"  Subscribers Gained: {analytics.get('subscribers_gained_30d', 0):,.0f}")
                print(f"  Likes: {analytics.get('likes_30d', 0):,.0f}")
                print(f"  Comments: {analytics.get('comments_30d', 0):,.0f}")
                print(f"  Shares: {analytics.get('shares_30d', 0):,.0f}")
        else:
            print("Failed to retrieve channel statistics")
    
    # Run autonomous production loop
    # Skip production only if channel operations were performed AND no videos were explicitly requested
    if channel_ops_performed and args.num_videos is None:
        # Only channel operations, no production
        logger.info("Channel operations completed. No video production requested.")
    else:
        print(f"\n=== Starting Autonomous Production ===")
        print(f"Videos to produce: {final_config['num_videos']}")
        print(f"Auto-upload: {final_config['auto_upload']}")
        print(f"Auto-schedule: {final_config['auto_schedule']}")
        print()
        
        try:
            studio.run_autonomous_loop(
                num_videos=final_config["num_videos"],
                discover_interval=final_config["discover_interval"],
                auto_upload=final_config["auto_upload"],
                auto_schedule=final_config["auto_schedule"],
                analytics_interval=final_config["analytics_interval"]
            )
        except KeyboardInterrupt:
            print("\n\nProduction interrupted by user")
            studio.pause_production()
        except Exception as e:
            logger.error(f"Error in production loop: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()

