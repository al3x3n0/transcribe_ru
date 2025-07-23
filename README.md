# Russian Media Transcription and Summarization

A Python script for transcribing and summarizing Russian audio and video files using **LOCAL models only** - no cloud APIs or internet connection required after initial model download.

## Features

- **100% Local Processing**: All models run on your machine
- **Multi-format support**: MP3, WAV, M4A, FLAC, MP4, AVI, MKV, MOV, WebM
- **Accurate Russian transcription** using OpenAI Whisper (local)
- **Automatic summarization** using Russian-optimized transformer models (local)
- **Video support** with automatic audio extraction
- **Progress tracking** and detailed logging
- **Multiple output formats**: TXT transcripts, summaries, and JSON with timestamps
- **Offline mode**: Run without internet after models are cached
- **GPU acceleration**: Optional CUDA support for faster processing
- **Lightweight mode**: Use smaller models for low-memory systems (--lightweight)

## Installation

1. Install system dependencies:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

2. Setup virtual environment:
```bash
# Quick setup (creates venv and installs dependencies)
./setup_venv.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. (Optional) Pre-download models for offline use:
```bash
# Download Whisper only (default - transcript mode)
python download_models.py --model-size large

# Download both Whisper and summarization models
python download_models.py --enable-summary --model-size large

# If you have SSL certificate issues:
python download_models.py --skip-ssl-verify
```

## Usage

Basic usage:
```bash
python transcribe_russian.py input_file.mp4
```

With options:
```bash
python transcribe_russian.py input_file.mp4 \
    --output ./results \
    --model large \
    --device cuda \
    --max-summary-length 500
```

Run in offline mode (no internet):
```bash
python transcribe_russian.py input_file.mp4 --offline
```

### Command-line Arguments

- `input`: Path to audio/video file (required)
- `-o, --output`: Output directory (default: same as input)
- `-m, --model`: Whisper model size: tiny, base, small, medium, large (default: small)
- `--model-dir`: Directory to store/load models (default: ~/.cache/russian_transcriber)
- `--device`: Device to use: cpu, cuda, auto (default: auto)
- `--max-summary-length`: Maximum summary length in tokens (default: 300)
- `--min-summary-length`: Minimum summary length in tokens (default: 50)
- `--offline`: Use only locally cached models (no downloads)
- `--lightweight`: Use smaller models to reduce memory usage
- `--enable-summary`: Enable summarization (requires additional model downloads)

## Output Files

The script generates three files for each input:
- `{filename}_{timestamp}_transcript.txt` - Full transcription
- `{filename}_{timestamp}_summary.txt` - Summary of the content
- `{filename}_{timestamp}_results.json` - Complete results with metadata and timestamps

## Models Used

All models run locally on your machine:

- **Transcription**: OpenAI Whisper (multilingual speech recognition) - stored in `~/.cache/russian_transcriber/whisper/`
- **Summarization**: IlyaGusev/rut5_base_sum_gazeta (Russian text summarization) - stored in `~/.cache/russian_transcriber/summarizer/`

First run will download models (~1.5GB for Whisper large, ~500MB for summarizer). After that, all processing is local.

## System Requirements

- Python 3.8+
- FFmpeg
- 8GB+ RAM (16GB recommended for large model)
- ~3GB disk space for model storage
- GPU optional but recommended for faster processing (CUDA-compatible)

## Download Issues & Alternatives

If you encounter download issues (SSL, network, corporate firewall):

### Option 1: SSL Bypass
```bash
python download_models.py --skip-ssl-verify
```

### Option 2: Use Fallback Summarizer (No Downloads)
```bash
python alternative_download.py
# This creates a simple rule-based Russian summarizer
```

### Option 3: Manual Download
1. Visit: https://huggingface.co/cointegrated/rut5-small
2. Download files to `~/.cache/russian_transcriber/summarizer/`
3. Or run: `python alternative_download.py --help`

### Option 4: Skip Summarization
```bash
# Just transcription, no summary
python transcribe_russian.py file.mp4 --no-summary
```

The script automatically falls back to rule-based summarization if model downloads fail.

## Performance Tips

1. Use smaller Whisper models (tiny, base) for faster processing
2. GPU acceleration significantly improves speed  
3. For long videos, consider splitting into segments
4. Use --lightweight flag on systems with <4GB RAM
5. See MODEL_SIZES.md for detailed performance guide

## Examples

```bash
# Default - transcript only (fastest, no extra downloads)
python transcribe_russian.py podcast.mp3

# With summarization enabled
python transcribe_russian.py podcast.mp3 --enable-summary

# Quick transcription with minimal resources
python transcribe_russian.py podcast.mp3 --model tiny

# High quality transcription
python transcribe_russian.py important_lecture.mp4 --model large

# Process with GPU acceleration
python transcribe_russian.py video.mp4 --device cuda

# Full featured with summarization
python transcribe_russian.py video.mp4 --enable-summary --model large --device cuda
```