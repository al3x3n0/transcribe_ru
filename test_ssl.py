#!/usr/bin/env python3
"""
Test script to verify SSL configuration works for model downloads
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def test_ssl_config():
    """Test SSL configuration"""
    logger.info("Testing SSL configuration...")
    
    # Set SSL bypass
    os.environ['SKIP_SSL_VERIFY'] = '1'
    
    # Import and configure SSL
    from ssl_config import configure_ssl_for_self_signed
    configure_ssl_for_self_signed()
    
    # Test basic requests
    try:
        import requests
        logger.info("Testing requests with SSL bypass...")
        response = requests.get('https://httpbin.org/get', timeout=10)
        logger.info(f"✓ requests test successful: {response.status_code}")
    except Exception as e:
        logger.warning(f"requests test failed: {e}")
    
    # Test HuggingFace Hub access
    try:
        logger.info("Testing HuggingFace Hub access...")
        from transformers import AutoTokenizer
        
        # Try to access model info (should work with SSL bypass)
        tokenizer = AutoTokenizer.from_pretrained(
            "cointegrated/rut5-small", 
            cache_dir=Path.home() / ".cache" / "test_ssl"
        )
        logger.info("✓ HuggingFace Hub access successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ HuggingFace Hub test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_ssl_config()
    if success:
        logger.info("✅ SSL configuration test passed")
        sys.exit(0)
    else:
        logger.error("❌ SSL configuration test failed")
        sys.exit(1)