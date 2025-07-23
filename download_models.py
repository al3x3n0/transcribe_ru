#!/usr/bin/env python3
"""
Pre-download models for offline use
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for SSL bypass flag early
def should_skip_ssl():
    """Check if SSL should be skipped based on args or env"""
    skip_ssl = os.environ.get('SKIP_SSL_VERIFY', '').lower() in ['1', 'true', 'yes']
    if '--skip-ssl-verify' in sys.argv:
        skip_ssl = True
    return skip_ssl

# Configure SSL BEFORE importing any network libraries
if should_skip_ssl():
    logger.warning("⚠️  SSL verification will be disabled")
    
    # Set all possible SSL bypass environment variables
    os.environ['SKIP_SSL_VERIFY'] = '1'
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_VERIFY'] = 'false'
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow downloads but skip SSL
    
    # Configure SSL context globally
    import ssl
    import urllib.request
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    ssl._create_default_https_context = lambda: ssl_context
    
    # Patch urllib
    original_urlopen = urllib.request.urlopen
    def patched_urlopen(url, *args, **kwargs):
        kwargs['context'] = ssl_context
        return original_urlopen(url, *args, **kwargs)
    urllib.request.urlopen = patched_urlopen

# Now import network libraries after SSL configuration
import whisper

# Import transformers with SSL already configured
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError as e:
    logger.error(f"Failed to import transformers: {e}")
    sys.exit(1)


def setup_ssl_context():
    """Setup SSL context to handle self-signed certificates"""
    # Create unverified context for self-signed certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Monkey patch urllib to use our context
    urllib.request.urlopen = lambda url, *args, **kwargs: urllib.request.urlopen(
        url, *args, context=ssl_context, **kwargs
    ) if 'context' not in kwargs else urllib.request.urlopen(url, *args, **kwargs)
    
    # Set environment variables for requests library
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    
    # Disable SSL warnings if verify=False is used
    import warnings
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    
    return ssl_context


def download_models(model_size="small", model_dir=None, lightweight=False, skip_ssl_verify=False):
    """Download all required models for offline use"""
    
    if model_dir is None:
        model_dir = Path.home() / ".cache" / "russian_transcriber"
    else:
        model_dir = Path(model_dir)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # SSL should already be configured at import time if needed
    ssl_disabled = should_skip_ssl() or skip_ssl_verify
    if ssl_disabled:
        logger.info("🔓 SSL verification is disabled")
        
        # Additional runtime SSL bypass for requests
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            import urllib3
            
            # Disable warnings
            urllib3.disable_warnings()
            
            # Create session with SSL disabled
            session = requests.Session()
            session.verify = False
            
            # Monkey patch requests
            original_request = requests.request
            def no_ssl_request(method, url, **kwargs):
                kwargs['verify'] = False
                kwargs['timeout'] = kwargs.get('timeout', 120)
                return original_request(method, url, **kwargs)
            
            requests.request = no_ssl_request
            requests.get = lambda url, **kwargs: no_ssl_request('GET', url, **kwargs)
            requests.post = lambda url, **kwargs: no_ssl_request('POST', url, **kwargs)
            
            logger.info("✓ Patched requests library for SSL bypass")
            
        except Exception as e:
            logger.warning(f"Could not patch requests: {e}")
    else:
        logger.info("🔒 SSL verification is enabled")
    
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
        # Download with SSL configuration already applied
        download_kwargs = {
            'cache_dir': summarizer_cache,
            'force_download': False,
            'resume_download': True
        }
        
        logger.info(f"📥 Downloading tokenizer for {model_name}...")
        logger.info(f"    Cache directory: {summarizer_cache}")
        if ssl_disabled:
            logger.info("    🔓 Using SSL bypass")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, **download_kwargs)
        logger.info("✓ Tokenizer downloaded successfully")
        
        logger.info(f"📥 Downloading model weights for {model_name}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **download_kwargs)
        logger.info("✓ Model weights downloaded successfully")
        
        logger.info("✅ Russian summarization model downloaded successfully")
            
    except Exception as e:
        logger.error(f"❌ Failed to download summarization model: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        
        if not ssl_disabled:
            logger.info("💡 If this is an SSL error, try using --skip-ssl-verify flag:")
            logger.info("   python download_models.py --skip-ssl-verify")
        else:
            logger.info("💡 SSL bypass is already enabled. This might be a network connectivity issue.")
            logger.info("   Check your internet connection and proxy settings.")
            
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
    parser.add_argument("--skip-ssl-verify", action="store_true",
                       help="Skip SSL certificate verification (use for self-signed certificates)")
    
    args = parser.parse_args()
    
    success = download_models(args.model_size, args.model_dir, args.lightweight, args.skip_ssl_verify)
    sys.exit(0 if success else 1)