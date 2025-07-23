#!/usr/bin/env python3
"""
Pre-download models for offline use
"""

import os
import sys
from pathlib import Path
import logging
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def download_models(model_size="small", model_dir=None, lightweight=False):
    """Download all required models for offline use"""
    
    if model_dir is None:
        model_dir = Path.home() / ".cache" / "russian_transcriber"
    else:
        model_dir = Path(model_dir)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Model sizes
    whisper_sizes = {
        'tiny': '39 MB',
        'base': '74 MB', 
        'small': '244 MB',
        'medium': '769 MB',
        'large': '1.5 GB'
    }
    
    # Download Whisper model
    logger.info(f"Downloading Whisper {model_size} model ({whisper_sizes.get(model_size, 'unknown size')})...")
    whisper_cache = model_dir / "whisper"
    whisper_cache.mkdir(exist_ok=True)
    os.environ['WHISPER_CACHE'] = str(whisper_cache)
    
    try:
        model = whisper.load_model(model_size, download_root=str(whisper_cache))
        logger.info(f"✓ Whisper {model_size} model downloaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to download Whisper model: {e}")
        return False
    
    # Download summarization model
    if lightweight or model_size in ['tiny', 'small']:
        model_name = "cointegrated/rut5-small"
        logger.info("Downloading lightweight Russian summarization model (85 MB)...")
    else:
        model_name = "IlyaGusev/rut5_base_sum_gazeta"
        logger.info("Downloading standard Russian summarization model (223 MB)...")
        
    summarizer_cache = model_dir / "summarizer"
    summarizer_cache.mkdir(exist_ok=True)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=summarizer_cache)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=summarizer_cache)
        logger.info("✓ Russian summarization model downloaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to download summarization model: {e}")
        return False
    
    logger.info(f"\n✓ All models downloaded to: {model_dir}")
    logger.info("You can now run the transcription script in offline mode with --offline flag")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for offline use")
    parser.add_argument("--model-size", choices=['tiny', 'base', 'small', 'medium', 'large'], 
                       default='small', help="Whisper model size to download (default: small)")
    parser.add_argument("--model-dir", type=Path, help="Directory to store models")
    parser.add_argument("--lightweight", action="store_true", 
                       help="Download lightweight summarization model")
    
    args = parser.parse_args()
    
    success = download_models(args.model_size, args.model_dir, args.lightweight)
    sys.exit(0 if success else 1)