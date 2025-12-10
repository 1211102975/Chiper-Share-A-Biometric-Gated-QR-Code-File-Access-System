import os
import qrcode
import json
from cryptography.fernet import Fernet
import face_recognition
import pickle

def test_file_write():
    try:
        # Print current working directory
        print(f"Current working directory: {os.getcwd()}")
        
        # Try to write a simple test file
        test_path = "test_write.txt"
        with open(test_path, "w") as f:
            f.write("Test write")
        print(f"✓ Successfully wrote test file: {test_path}")
        
        # Verify the file exists and show its size
        if os.path.exists(test_path):
            print(f"✓ Test file exists, size: {os.path.getsize(test_path)} bytes")
        else:
            print("✗ Test file was not created!")
            
    except Exception as e:
        print(f"Error testing file write: {str(e)}")

def create_test_files():
    try:
        # Create test document
        with open("example.txt", "w") as f:
            f.write("This is a test document for encryption.")
        print("✓ Created example.txt")
        
        # Verify xinyi.jpg exists in known_faces folder
        xinyi2_path = os.path.join("known_faces", "xinyi2.jpg")
        if not os.path.exists(xinyi2_path):
            raise FileNotFoundError(f"xinyi2.jpg not found in {xinyi2_path}")
        print(f"✓ Found xinyi2.jpg at {xinyi2_path}")
        
        # Create test face encoding
        test_encoding = face_recognition.face_encodings(face_recognition.load_image_file(xinyi2_path))[0]
        pkl_path = os.path.join("known_faces", "xinyi2.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(test_encoding, f)
        print(f"✓ Created xinyi2.pkl at {pkl_path}")
        
    except Exception as e:
        print(f"Error in create_test_files: {str(e)}")
        raise

def encrypt_and_create_qr():
    try:
        # Verify example.txt exists
        if not os.path.exists("example.txt"):
            raise FileNotFoundError("example.txt not found")
        
        # Read the test document
        with open("example.txt", "rb") as f:
            file_data = f.read()
        print("✓ Read example.txt")
        
        # Generate encryption key
        key = Fernet.generate_key()
        f = Fernet(key)
        
        # Encrypt the document
        encrypted_data = f.encrypt(file_data)
        print("✓ Encrypted document")
        
        # Create QR code data
        qr_data = {
            'key': key.decode(),
            'encrypted_data': encrypted_data.decode(),
            'email': 'cindyloixy521@gmail.com'
        }
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(json.dumps(qr_data))
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
        # Save QR code
        qr_path = "test_qr.png"
        qr_image.save(qr_path)
        
        # Verify QR code was created
        if os.path.exists(qr_path):
            print(f"✓ QR code generated and saved as {qr_path}")
            print(f"QR code size: {os.path.getsize(qr_path)} bytes")
        else:
            raise FileNotFoundError(f"Failed to create {qr_path}")
            
    except Exception as e:
        print(f"Error in encrypt_and_create_qr: {str(e)}")
        raise

def main():
    try:
        print("Testing file system access...")
        test_file_write()
        
        print("\nStarting test file creation...")
        # Create test files
        create_test_files()
        
        print("\nStarting QR code generation...")
        # Encrypt and create QR
        encrypt_and_create_qr()
        
        print("\nTest files created successfully!")
        print("1. example.txt - Test document")
        print("2. xinyi.pkl - Face encoding (in known_faces folder)")
        print("3. test_qr.png - Generated QR code")
        print("\nTo test the system:")
        print("1. Run the Flask app: python app.py")
        print("2. Visit http://localhost:5000")
        print("3. Use the 'Scan QR Code' tab")
        print("4. Upload test_qr.png")
        print("5. Enter the OTP sent to cyloixy2610@gmail.com")
        print("6. Complete face verification")
        print("7. Download the decrypted document")
        
    except Exception as e:
        print(f"\nError during test setup: {str(e)}")
        print("Please make sure:")
        print("1. You have xinyi.jpg in the known_faces folder")
        print("2. You have write permissions in the current directory")
        print("3. All required packages are installed (qrcode, cryptography, face_recognition)")

if __name__ == "__main__":
    main() 