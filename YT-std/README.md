# Autonomous YouTube Studio (YT-std)

Minimal production-grade autonomous YouTube video production system using Swarms framework.

## Architecture

Hybrid approach combining:
- **Blueprint 2 foundation**: Production-ready infrastructure, state management, observability
- **Blueprint 1 strengths**: CR-CA causal reasoning algorithms, UCB exploration-exploitation

## Key Features

1. **CR-CA Agent**: Causal reasoning for topic ranking and optimization decisions
2. **ReAct Agents**: Autonomous execution workers with `max_loops="auto"` for task completion
3. **State Machine**: Tracks video production pipeline (discovered → ranked → scripted → assets → composed → validated → uploaded → optimized)
4. **Priority Queue**: Efficient topic selection using causal scores
5. **Actual Video Production**: 
   - Generates TTS audio (Google Cloud TTS)
   - Creates thumbnail images (Stable Diffusion via Replicate)
   - Generates B-roll images
   - Composes final MP4 videos using FFmpeg
   - Produces ready-to-upload video files
6. **Production Pipeline & Reliability**:
   - Retry logic with exponential backoff for all operations
   - Quality validation before finalizing videos
   - Progress tracking with JSON persistence
   - Idempotency keys to prevent duplicate productions
   - State persistence for recovery from failures
   - Resume production from last successful step
7. **YouTube Integration & Automation**:
   - Direct YouTube upload via Data API (resumable uploads)
   - SEO-optimized metadata generation (titles, descriptions, tags)
   - Analytics integration for performance feedback
   - Automated scheduling using CR-CA for optimal publish times
   - Playlist management (auto-create and organize)
   - Quota management with automatic tracking and reset
8. **Account & Channel Management**:
   - Channel information retrieval (name, description, statistics, branding)
   - Channel customization (update name, description, banner)
   - Channel settings management (comment moderation, keywords)
   - Account verification and eligibility checks
   - Automated branding with agent-generated descriptions
   - Channel statistics and performance metrics
9. **Analytics Feedback Loop**:
   - Periodic analytics fetching
   - CR-CA priors update based on actual performance
   - Topic ranking improvement from real data

## Components

### Agents

- **Discovery Agent**: Discovers trending topics using ReAct workflow
- **CR-CA Agent**: Ranks topics using causal inference (estimates expected watch time uplift)
- **Script Agent**: Writes engaging YouTube scripts
- **Asset Agent**: Generates thumbnails, B-roll, captions
- **Composer Agent**: Creates FFmpeg commands for video composition
- **Optimizer Agent**: Optimizes performance using causal analysis

### Data Structures

- **Topic**: Priority queue item with causal features
- **Video**: State machine for production pipeline
- **State Cache**: LRU cache for expensive operations

## Usage

```python
from main import YouTubeStudio

# Initialize studio with API credentials
studio = YouTubeStudio(
    model_name="gpt-4o-mini",
    workspace_dir="yt_workspace",
    max_concurrent=3,
    google_tts_credentials="path/to/google-credentials.json",  # Optional
    replicate_api_token="your-replicate-token",  # Optional
)

# Run autonomous production loop with auto-upload
studio.run_autonomous_loop(
    num_videos=5,
    discover_interval=3600,
    auto_upload=True,  # Automatically upload to YouTube
    auto_schedule=True,  # Schedule optimal publish times
    analytics_interval=86400  # Update analytics every 24 hours
)

# Access produced videos
for video_id, video in studio.videos.items():
    if video.metadata.get("video_file"):
        print(f"Video ready: {video.metadata['video_path']}")
        if video.youtube_id:
            print(f"YouTube ID: {video.youtube_id}")

# Check production status
status = studio.get_production_status()
print(f"Production status: {status}")

# Manually upload a video
from main import VideoState

video = studio.videos["video_id"]
if video.state == VideoState.VALIDATED:
    youtube_id = studio.upload_to_youtube(video)
    if youtube_id:
        # Add to playlist
        studio.add_to_playlist(video, "My Series")
        # Fetch analytics
        metrics = studio.fetch_video_analytics(video)

# Optimize existing video
metrics = {"ctr": 0.05, "watch_time": 120.5, "retention": 0.45, "views": 1000}
optimization = studio.optimize_video(video, metrics)

# Resume failed production
failed_video = studio.resume_video_production("video_id")

# Pause/resume production loop
studio.pause_production()
studio.resume_production_loop()

# Channel Management
# Get channel information
channel_info = studio.get_channel_info()
print(f"Channel: {channel_info.get('title')}")
print(f"Subscribers: {channel_info.get('subscriber_count')}")

# Update channel description
studio.update_channel_description("Python tutorials with Swarms framework. Learn automation, multi-agent systems, and more!")

# Update channel name
studio.update_channel_name("Python Swarms Tutorials")

# Upload channel banner (2560x1440 recommended)
studio.update_channel_banner("path/to/banner.jpg")

# Check verification and eligibility
is_verified = studio.check_verification_status()
eligibility = studio.check_channel_eligibility()
print(f"Verified: {is_verified}")
print(f"Monetization eligible: {eligibility.get('monetization_eligible')}")

# Get account status
account_status = studio.get_account_status()
print(f"Account status: {account_status}")

# Get channel statistics
stats = studio.get_channel_statistics()
print(f"Total subscribers: {stats.get('subscriber_count')}")
print(f"30-day views: {stats.get('analytics_30d', {}).get('views_30d')}")

# Apply automated branding (generates description from video topics)
branding_results = studio.apply_channel_branding()
print(f"Branding applied: {branding_results}")

# Or provide custom branding config
custom_branding = {
    "description": "Your custom channel description",
    "name": "Your Channel Name",
    "banner_path": "path/to/banner.jpg",
    "moderate_comments": True
}
studio.apply_channel_branding(custom_branding)
```

## Environment Variables

```bash
export OPENAI_API_KEY="your-key-here"
export REPLICATE_API_TOKEN="your-replicate-token"  # For image generation
export GOOGLE_APPLICATION_CREDENTIALS="path/to/google-credentials.json"  # For TTS
# Or set in .env file
```

## Setup

1. **Install FFmpeg** (required for video composition):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows: Download from https://ffmpeg.org/download.html
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Cloud TTS** (optional but recommended):
   - Create a Google Cloud project
   - Enable Text-to-Speech API
   - Download credentials JSON
   - Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

4. **Set up Replicate** (optional but recommended):
   - Sign up at https://replicate.com
   - Get API token
   - Set `REPLICATE_API_TOKEN` environment variable

5. **Set up YouTube API** (for upload and analytics):
   - Go to https://console.cloud.google.com
   - Create a new project or select existing
   - Enable YouTube Data API v3 and YouTube Analytics API
   - Create OAuth 2.0 credentials (Desktop app type)
   - Download credentials JSON file
   - Pass path to credentials in `youtube_credentials` parameter
   - First run will open browser for OAuth authorization

## File Structure

```
YT-std/
├── main.py              # Single-file orchestrator
├── requirements.txt     # Dependencies
├── README.md           # This file
└── yt_workspace/       # Generated workspace
    ├── scripts/        # Generated markdown scripts
    ├── assets/         # Asset metadata JSON
    ├── audio/          # Generated TTS audio files (.mp3)
    ├── images/         # Generated images (.png)
    └── videos/         # Final video files (.mp4) + metadata
```

## CS Techniques Used

1. **State Machine**: VideoState enum tracks production pipeline
2. **Priority Queue**: heapq for efficient topic selection
3. **Caching**: Dictionary-based cache for expensive operations
4. **Hash-based IDs**: MD5 for unique video identifiers
5. **Data Structures**: Deque, defaultdict for efficient lookups
6. **Algorithms**: UCB-like exploration-exploitation for topic ranking

## Extending

To add new features:

1. **New Agent**: Add Agent with `max_loops="auto"` to enable ReAct
2. **New State**: Add to VideoState enum and update state machine
3. **New Tool**: Add function and pass to agent's `tools` parameter
4. **New Causal Variable**: Add to CR-CA agent's `variables` and `causal_edges`

## Production Quality

**Yes, it actually produces video files and uploads to YouTube!**

The system:
- ✅ Generates real TTS audio using Google Cloud TTS (or fallback)
- ✅ Creates actual thumbnail images using Stable Diffusion (or fallback)
- ✅ Generates B-roll images for video composition
- ✅ Composes final MP4 video files using FFmpeg
- ✅ Validates video quality (codec, resolution, duration, file size)
- ✅ Uploads to YouTube with resumable uploads
- ✅ Generates SEO-optimized metadata
- ✅ Schedules optimal publish times using CR-CA
- ✅ Fetches analytics and updates ranking priors
- ✅ Manages playlists automatically

**Quality Features:**
- Professional TTS with neural voices
- High-quality image generation (1920x1080)
- Proper video encoding (H.264, AAC)
- Synchronized audio-visual composition
- Quality validation before upload
- Retry logic with exponential backoff
- Progress tracking and state persistence
- Quota management for YouTube API
- Analytics feedback loop for continuous improvement

**Production Pipeline:**
- Error handling with automatic retries (3 attempts, exponential backoff)
- Quality validation (file size, codec, resolution, duration matching)
- Progress tracking saved to JSON files
- Idempotency keys prevent duplicate productions
- State persistence allows recovery from failures
- Resume from last successful step

**YouTube Integration:**
- OAuth2 authentication with token refresh
- Resumable uploads for large files
- SEO metadata optimization (titles, descriptions, tags)
- Automated scheduling using causal reasoning
- Analytics integration (views, watch time, retention)
- Playlist management (create and organize)
- Quota tracking and management (10,000 units/day default)

**Account & Channel Management:**
- Complete channel information retrieval (branding, statistics, settings)
- Channel customization (name, description, banner upload)
- Settings management (comment moderation, channel keywords)
- Verification status and eligibility checks (monetization, custom URL, etc.)
- Automated branding with AI-generated descriptions from video topics
- Channel-level statistics and 30-day analytics
- Account status dashboard (comprehensive account health check)

**Notes:**
- Uses `max_loops="auto"` to enable ReAct workflow for autonomous task completion
- CR-CA agent handles causal reasoning for ranking, optimization, and scheduling
- Minimal file count: single main.py orchestrator
- Production-ready: error handling, logging, state management, retry logic
- Graceful degradation: works even without API credentials (uses fallbacks)

