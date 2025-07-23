# Quick Start Guide

## 1. First Time Setup

```bash
# Clone/navigate to project directory
cd transcript_ru

# Run the setup script
./setup_venv.sh

# Activate virtual environment
source venv/bin/activate
```

## 2. Test Installation

```bash
# Download a small test model first
python download_models.py --model-size tiny

# Test with an audio file
python transcribe_russian.py test_audio.mp3 --model tiny
```

## 3. Production Use

```bash
# Download full model (one-time, ~2GB)
python download_models.py --model-size large

# Process your media files
python transcribe_russian.py your_video.mp4

# For offline use (no internet)
python transcribe_russian.py your_video.mp4 --offline
```

## Virtual Environment Commands

**Activate:**
- macOS/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

**Deactivate:**
- All platforms: `deactivate`

**Check if activated:**
- Your prompt should show `(venv)`

## Troubleshooting

**FFmpeg not found:**
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

**Out of memory:**
- Use smaller model: `--model tiny` or `--model base`
- Process shorter files
- Close other applications

**Slow processing:**
- Use GPU if available: `--device cuda`
- Use smaller model for testing