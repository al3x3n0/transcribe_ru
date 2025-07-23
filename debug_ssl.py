#!/usr/bin/env python3
"""
Debug script to test SSL configuration step by step
"""

import os
import sys
import logging

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ssl_step_by_step():
    """Test SSL configuration step by step"""
    
    print("🔧 SSL Debug Test")
    print("=" * 50)
    
    # Step 1: Check environment
    print("1. Environment Variables:")
    ssl_vars = ['SKIP_SSL_VERIFY', 'CURL_CA_BUNDLE', 'REQUESTS_CA_BUNDLE', 
                'SSL_VERIFY', 'PYTHONHTTPSVERIFY', 'HF_HUB_DISABLE_SSL_VERIFY']
    
    for var in ssl_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"   {var}: {value}")
    
    # Step 2: Configure SSL
    print("\n2. Configuring SSL bypass...")
    os.environ['SKIP_SSL_VERIFY'] = '1'
    
    # Import SSL modules
    import ssl
    import urllib.request
    
    # Create unverified context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    print(f"   SSL context created: verify_mode={ssl_context.verify_mode}")
    print(f"   Check hostname: {ssl_context.check_hostname}")
    
    # Step 3: Test basic HTTPS
    print("\n3. Testing basic HTTPS connection...")
    try:
        import requests
        
        # Patch requests
        original_request = requests.request
        def debug_request(method, url, **kwargs):
            kwargs['verify'] = False
            kwargs['timeout'] = 30
            print(f"   Making {method} request to {url}")
            print(f"   SSL verify: {kwargs['verify']}")
            return original_request(method, url, **kwargs)
        
        requests.request = debug_request
        
        response = requests.get('https://httpbin.org/get')
        print(f"   ✓ Basic HTTPS test: {response.status_code}")
        
    except Exception as e:
        print(f"   ❌ Basic HTTPS test failed: {e}")
        return False
    
    # Step 4: Test HuggingFace Hub
    print("\n4. Testing HuggingFace Hub access...")
    try:
        # Set HF environment variables
        os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
        
        from transformers import AutoTokenizer
        
        print("   Attempting to load tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "cointegrated/rut5-small",
            cache_dir="/tmp/ssl_test",
        )
        print("   ✓ HuggingFace Hub test successful")
        return True
        
    except Exception as e:
        print(f"   ❌ HuggingFace Hub test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Print more details for SSL errors
        if 'SSL' in str(e) or 'certificate' in str(e).lower():
            print("   This appears to be an SSL certificate error.")
            print("   Your network might be using a corporate proxy or firewall.")
        
        return False

if __name__ == "__main__":
    success = test_ssl_step_by_step()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ SSL bypass test completed successfully")
        print("You should be able to download models with --skip-ssl-verify")
    else:
        print("❌ SSL bypass test failed")
        print("You may need to configure your proxy settings or network access")
    
    sys.exit(0 if success else 1)