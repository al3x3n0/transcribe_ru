#!/usr/bin/env python3
"""
SSL configuration utilities for handling self-signed certificates
"""

import os
import ssl
import urllib.request
import warnings
import logging

logger = logging.getLogger(__name__)


def configure_ssl_for_self_signed():
    """Configure SSL to accept self-signed certificates"""
    
    # Check if SSL verification should be skipped
    skip_ssl = os.environ.get('SKIP_SSL_VERIFY', '').lower() in ['1', 'true', 'yes']
    
    if skip_ssl:
        logger.warning("SSL verification disabled via SKIP_SSL_VERIFY environment variable")
        
        # Create unverified SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Monkey patch ssl default context
        ssl._create_default_https_context = lambda: ssl_context
        
        # Set environment variables for various libraries
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
        os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'
        
        # Disable SSL warnings
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')
        
        # Try to disable urllib3 warnings if installed
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except ImportError:
            pass
            
        return True
    
    return False


def get_ssl_instructions():
    """Get instructions for handling SSL certificate issues"""
    return """
If you're experiencing SSL certificate errors, you have several options:

1. **Temporary workaround (not recommended for production):**
   Set environment variable before running:
   ```bash
   export SKIP_SSL_VERIFY=1
   python download_models.py
   ```

2. **Command line flag:**
   ```bash
   python download_models.py --skip-ssl-verify
   ```

3. **Permanent environment variable:**
   Add to your shell profile (~/.bashrc or ~/.zshrc):
   ```bash
   export SKIP_SSL_VERIFY=1
   ```

4. **For specific proxy/firewall issues:**
   ```bash
   export HTTP_PROXY=http://your-proxy:port
   export HTTPS_PROXY=http://your-proxy:port
   ```

⚠️  WARNING: Disabling SSL verification can expose you to security risks.
Only use in trusted environments or when downloading from known sources.
"""