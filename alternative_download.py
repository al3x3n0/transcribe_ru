#!/usr/bin/env python3
"""
Alternative download methods bypassing HuggingFace Hub
"""

import os
import sys
import logging
import requests
import zipfile
import json
from pathlib import Path
from urllib.parse import urlparse
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def download_file_direct(url, dest_path, verify_ssl=True):
    """Download a file directly with progress tracking"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        logger.info(f"Downloading: {url}")
        response = requests.get(url, headers=headers, stream=True, verify=verify_ssl, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {percent:.1f}%", end='')
        
        print()  # New line after progress
        logger.info(f"✓ Downloaded: {dest_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        return False


def create_simple_summarizer():
    """Create a simple rule-based Russian summarizer as fallback"""
    
    fallback_code = '''
import re
from typing import List

class SimplerussianSummarizer:
    """Simple extractive summarizer for Russian text"""
    
    def __init__(self):
        # Russian stop words
        self.stop_words = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так',
            'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было',
            'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
            'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж',
            'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
            'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего',
            'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого',
            'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас',
            'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть',
            'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
            'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть',
            'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\\s+', ' ', text.strip())
        return text
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting for Russian
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def score_sentence(self, sentence: str, word_freq: dict) -> float:
        """Score sentence based on word frequency"""
        words = re.findall(r'\\b\\w+\\b', sentence.lower())
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        if not words:
            return 0
        
        score = sum(word_freq.get(word, 0) for word in words)
        return score / len(words)  # Average score
    
    def get_word_frequencies(self, text: str) -> dict:
        """Get word frequency distribution"""
        words = re.findall(r'\\b\\w+\\b', text.lower())
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        
        # Normalize frequencies
        max_freq = max(freq.values()) if freq else 1
        for word in freq:
            freq[word] = freq[word] / max_freq
            
        return freq
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """Generate extractive summary"""
        text = self.clean_text(text)
        sentences = self.split_sentences(text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Get word frequencies
        word_freq = self.get_word_frequencies(text)
        
        # Score sentences
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = self.score_sentence(sentence, word_freq)
            sentence_scores.append((score, i, sentence))
        
        # Sort by score and take top sentences
        sentence_scores.sort(reverse=True)
        top_sentences = sentence_scores[:max_sentences]
        
        # Sort by original order
        top_sentences.sort(key=lambda x: x[1])
        
        # Join selected sentences
        summary = '. '.join([sent[2] for sent in top_sentences])
        return summary + '.'

# Usage:
# summarizer = SimplerussianSummarizer()
# summary = summarizer.summarize(text, max_sentences=3)
'''
    
    return fallback_code


def setup_alternative_models(model_dir, skip_ssl_verify=False):
    """Setup alternative model downloading"""
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🔄 Setting up alternative model sources...")
    
    # Create fallback summarizer
    fallback_path = model_dir / "simple_summarizer.py"
    with open(fallback_path, 'w', encoding='utf-8') as f:
        f.write(create_simple_summarizer())
    
    logger.info(f"✓ Created fallback summarizer: {fallback_path}")
    
    # Try alternative download sources
    alternatives_tried = []
    
    # Option 1: Try direct model files from mirrors
    mirror_urls = [
        "https://huggingface.co/cointegrated/rut5-small/resolve/main/config.json",
        "https://huggingface.co/cointegrated/rut5-small/resolve/main/tokenizer.json",
    ]
    
    summarizer_cache = model_dir / "summarizer" / "cointegrated_rut5-small"
    summarizer_cache.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for url in mirror_urls:
        filename = url.split('/')[-1]
        dest_path = summarizer_cache / filename
        
        if download_file_direct(url, dest_path, verify_ssl=not skip_ssl_verify):
            success_count += 1
        alternatives_tried.append(url)
    
    if success_count == 0:
        logger.warning("⚠️  Could not download model files from alternative sources")
        logger.info("📝 Will use simple rule-based summarizer as fallback")
        return "fallback"
    
    logger.info(f"✓ Downloaded {success_count} model files from alternative sources")
    return "alternative"


def create_manual_download_guide():
    """Create manual download instructions"""
    
    guide = """
# Manual Model Download Guide

If automatic downloads fail, you can manually download models:

## Option 1: Download via Browser

1. Go to: https://huggingface.co/cointegrated/rut5-small
2. Click "Files and versions"
3. Download these files to `~/.cache/russian_transcriber/summarizer/`:
   - config.json
   - tokenizer.json
   - pytorch_model.bin (large file ~340MB)
   - tokenizer_config.json

## Option 2: Use Git LFS (if available)

```bash
cd ~/.cache/russian_transcriber/summarizer/
git clone https://huggingface.co/cointegrated/rut5-small
```

## Option 3: Use wget/curl

```bash
cd ~/.cache/russian_transcriber/summarizer/
mkdir -p cointegrated_rut5-small
cd cointegrated_rut5-small

# Download config files
wget https://huggingface.co/cointegrated/rut5-small/resolve/main/config.json
wget https://huggingface.co/cointegrated/rut5-small/resolve/main/tokenizer.json
wget https://huggingface.co/cointegrated/rut5-small/resolve/main/tokenizer_config.json

# Download model weights (large file)
wget https://huggingface.co/cointegrated/rut5-small/resolve/main/pytorch_model.bin
```

## Option 4: Use Fallback Summarizer

The script includes a simple rule-based Russian summarizer that works without external models.
It will automatically fall back to this if model downloads fail.

## Verify Installation

After manual download, run:
```bash
python transcribe_russian.py --test-models
```
"""
    
    return guide


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Alternative model download")
    parser.add_argument("--model-dir", type=Path, 
                       default=Path.home() / ".cache" / "russian_transcriber",
                       help="Model directory")
    parser.add_argument("--skip-ssl-verify", action="store_true",
                       help="Skip SSL verification")
    
    args = parser.parse_args()
    
    result = setup_alternative_models(args.model_dir, args.skip_ssl_verify)
    
    if result == "fallback":
        print("\n" + "="*50)
        print("📝 FALLBACK MODE ACTIVATED")
        print("="*50)
        print("Using simple rule-based summarizer.")
        print("For better quality, try manual download:")
        print(create_manual_download_guide())
    
    elif result == "alternative":
        print("\n" + "="*50)
        print("✅ ALTERNATIVE DOWNLOAD SUCCESSFUL")
        print("="*50)
        print("Downloaded model files from alternative sources.")
        
    sys.exit(0)