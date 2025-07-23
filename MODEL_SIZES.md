# Model Size Guide

## Whisper Models

| Model | Size | Parameters | Speed | Quality | RAM Usage |
|-------|------|------------|-------|---------|-----------|
| tiny | 39 MB | 39M | 10x faster | Good for drafts | ~1 GB |
| base | 74 MB | 74M | 7x faster | Better accuracy | ~1 GB |
| small | 244 MB | 244M | 4x faster | Good balance | ~2 GB |
| medium | 769 MB | 769M | 2x faster | High quality | ~5 GB |
| large | 1.5 GB | 1550M | 1x baseline | Best quality | ~10 GB |

## Summarization Models

| Model | Size | Use Case | RAM Usage |
|-------|------|----------|-----------|
| rut5-small | 85 MB | Lightweight, fast | ~500 MB |
| rut5-base | 223 MB | Better quality | ~1 GB |

## Recommended Configurations

### Minimal Setup (Fast, Low Memory)
```bash
python transcribe_russian.py file.mp4 --model tiny --lightweight
```
- Total download: ~125 MB
- RAM usage: ~1.5 GB
- Speed: Very fast
- Quality: Acceptable for most use cases

### Balanced Setup (Default)
```bash
python transcribe_russian.py file.mp4 --model small
```
- Total download: ~470 MB  
- RAM usage: ~3 GB
- Speed: Good
- Quality: Good for production

### High Quality Setup
```bash
python transcribe_russian.py file.mp4 --model large
```
- Total download: ~1.7 GB
- RAM usage: ~11 GB
- Speed: Slower
- Quality: Best possible

### For Limited Resources
```bash
# Use tiny model with lightweight summarizer
python transcribe_russian.py file.mp4 --model tiny --lightweight

# Process in chunks for very long files
python transcribe_russian.py file.mp4 --model base --lightweight
```

## Performance Tips

1. **Start with tiny/small models** - Often sufficient for clear audio
2. **Use --lightweight flag** - Reduces memory by ~50% for summarization
3. **GPU acceleration** - Use --device cuda for 5-10x speedup
4. **Batch processing** - Process multiple files sequentially to avoid reloading models

## Model Quality Comparison

### Transcription Accuracy (Russian)
- tiny: ~85% accuracy on clear speech
- base: ~90% accuracy
- small: ~92% accuracy
- medium: ~94% accuracy
- large: ~95% accuracy

### Summary Quality
- rut5-small: Good for key points extraction
- rut5-base: Better context understanding and fluency