#!/usr/bin/env python3
"""
Test script to verify all fixes are working correctly
"""
import os
import sys

def test_ssl_certificates():
    """Check if SSL certificates exist"""
    print("\n" + "="*60)
    print("1. Testing SSL Certificates")
    print("="*60)
    
    cert_exists = os.path.exists("cert.pem")
    key_exists = os.path.exists("key.pem")
    
    if cert_exists and key_exists:
        print("✓ SSL certificate found: cert.pem")
        print("✓ Private key found: key.pem")
        print("  Status: HTTPS support is configured")
        return True
    else:
        print("✗ SSL certificates not found")
        print("  Run this command to generate certificates:")
        print('  openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"')
        return False

def test_camera_configuration():
    """Check if camera is configured to use correct index"""
    print("\n" + "="*60)
    print("2. Testing Camera Configuration")
    print("="*60)
    
    files_to_check = ['app.py', 'face_verifier.py', 'face_recog_tutorial/main.py']
    all_correct = True
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            print(f"⚠ File not found: {filename}")
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'src=1' in content:
            print(f"✗ {filename} still uses camera index 1 (src=1)")
            print(f"  This should be changed to src=0")
            all_correct = False
        elif 'src=0' in content:
            print(f"✓ {filename} correctly uses camera index 0")
        else:
            print(f"⚠ {filename} - camera configuration not found")
    
    if all_correct:
        print("\n  Status: Camera is configured to use default camera (index 0)")
        return True
    else:
        print("\n  Status: Camera configuration needs fixing")
        return False

def test_email_configuration():
    """Check email error handling"""
    print("\n" + "="*60)
    print("3. Testing Email Configuration")
    print("="*60)
    
    files_to_check = ['app.py', 'testemail.py']
    all_correct = True
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            print(f"⚠ File not found: {filename}")
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_timeout = 'timeout=' in content
        has_debug = 'set_debuglevel' in content or 'traceback' in content
        has_smtp_auth_error = 'SMTPAuthenticationError' in content
        
        print(f"\n{filename}:")
        if has_timeout:
            print(f"  ✓ Has connection timeout")
        else:
            print(f"  ✗ Missing connection timeout")
            all_correct = False
            
        if has_debug:
            print(f"  ✓ Has debug/error logging")
        else:
            print(f"  ✗ Missing debug/error logging")
            all_correct = False
            
        if has_smtp_auth_error:
            print(f"  ✓ Has proper SMTP error handling")
        else:
            print(f"  ✗ Missing SMTP error handling")
            all_correct = False
    
    if all_correct:
        print("\n  Status: Email configuration has proper error handling")
    else:
        print("\n  Status: Email configuration needs improvement")
    
    print("\n  Note: You may need to update the Gmail App Password:")
    print("  1. Go to https://myaccount.google.com/apppasswords")
    print("  2. Enable 2-Factor Authentication")
    print("  3. Generate a new App Password")
    print("  4. Update password in app.py (line ~420) and testemail.py (line ~21)")
    
    return all_correct

def test_imports():
    """Test if required packages are available"""
    print("\n" + "="*60)
    print("4. Testing Required Packages")
    print("="*60)
    
    required_packages = {
        'flask': 'Flask',
        'cv2': 'opencv-python',
        'qrcode': 'qrcode',
        'cryptography': 'cryptography',
        'face_recognition': 'face-recognition (optional for face verification)',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'pyzbar': 'pyzbar',
    }
    
    all_available = True
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Install with: pip install {package.split()[0]}")
            if module != 'face_recognition':  # face_recognition is optional
                all_available = False
    
    return all_available

def main():
    print("\n" + "="*60)
    print("Chiper-Share - Debugging Test Suite")
    print("="*60)
    
    results = {
        'SSL Certificates': test_ssl_certificates(),
        'Camera Configuration': test_camera_configuration(),
        'Email Configuration': test_email_configuration(),
        'Required Packages': test_imports(),
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nYou can now start the application with:")
        print("  python app.py")
        print("\nThen open your browser to:")
        print("  https://localhost:5000")
        print("\nNote: You'll need to accept the self-signed certificate warning.")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nRefer to DEBUGGING_GUIDE.md for detailed instructions.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())





