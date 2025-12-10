import os
import cv2
import qrcode
import json
from cryptography.fernet import Fernet
from face_verifier import face_recognition_pipeline

def generate_key():
    """Generate a new encryption key"""
    return Fernet.generate_key()

def encrypt_file(input_file, output_file, key):
    """Encrypt a file using Fernet symmetric encryption"""
    fernet = Fernet(key)
    
    with open(input_file, 'rb') as f:
        original_data = f.read()
    
    encrypted_data = fernet.encrypt(original_data)
    
    with open(output_file, 'wb') as f:
        f.write(encrypted_data)
    
    return encrypted_data

def create_qr_code(data, output_file="encrypted_qr.png"):
    """Create a QR code containing the encrypted data"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_file)
    return output_file

def encrypt_and_qr(input_file_path):
    """Main function to encrypt file and generate QR code"""
    try:
        # Generate encryption key
        key = generate_key()
        print("Encryption key generated")
        
        # Encrypt the file
        encrypted_file_path = f"{input_file_path}.encrypted"
        encrypted_data = encrypt_file(input_file_path, encrypted_file_path, key)
        print(f"File encrypted and saved to {encrypted_file_path}")
        
        # Combine key and encrypted data for QR code
        qr_data = {
            'key': key.decode(),
            'encrypted_data': encrypted_data.decode()
        }
        
        # Create QR code
        qr_file = create_qr_code(json.dumps(qr_data))  # Use json instead of str for better parsing
        print(f"QR code generated and saved to {qr_file}")
        
        return qr_file, encrypted_file_path, key
    except Exception as e:
        print(f"Error during encryption: {e}")
        return None, None, None

def decrypt_file(encrypted_file_path, output_file_path, key):
    """Decrypt a file using the provided key"""
    try:
        fernet = Fernet(key)
        
        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = fernet.decrypt(encrypted_data)
        
        with open(output_file_path, 'wb') as f:
            f.write(decrypted_data)
        
        return output_file_path
    except Exception as e:
        print(f"Error during decryption: {e}")
        return None

def qr_decryption_flow(qr_image_path, known_faces_dir="known_faces"):
    """Process QR code and decrypt if face recognition passes"""
    try:
        # Read QR code
        detector = cv2.QRCodeDetector()
        img = cv2.imread(qr_image_path)
        
        if img is None:
            print("Error: Could not read QR code image")
            return False
            
        data, _, _ = detector.detectAndDecode(img)
        
        if not data:
            print("No QR code data found")
            return False
        
        # Parse QR data using json
        try:
            qr_data = json.loads(data)
            key = qr_data['key'].encode()
            encrypted_data = qr_data['encrypted_data'].encode()
        except json.JSONDecodeError:
            print("Error: Invalid QR code data format")
            return False
        except KeyError:
            print("Error: Missing required data in QR code")
            return False
        
        print("Starting face verification...")
        name, confidence = face_recognition_pipeline(known_faces_dir)
        
        if confidence > 50:
            print(f"Face recognized as {name} with {confidence}% confidence. Proceeding with decryption.")
            
            # Save encrypted data to temp file
            temp_encrypted = "temp_encrypted.bin"
            with open(temp_encrypted, 'wb') as f:
                f.write(encrypted_data)
            
            # Decrypt and save
            decrypted_file = decrypt_file(temp_encrypted, "decrypted_file", key)
            
            # Clean up temp file
            try:
                os.remove(temp_encrypted)
            except:
                pass
                
            if decrypted_file:
                print(f"File successfully decrypted and saved to {decrypted_file}")
                return decrypted_file
            else:
                print("Decryption failed")
                return False
        else:
            print(f"Face recognition confidence too low ({confidence}%). Access denied.")
            return False
            
    except Exception as e:
        print(f"Error during QR code processing: {e}")
        return False

if __name__ == "__main__":
    # Example usage for encryption
    input_file = "example.txt"  # Change to your file
    qr_file, encrypted_file, key = encrypt_and_qr(input_file)
    
    if qr_file:
        print(f"QR code created: {qr_file}")
        print(f"Encrypted file: {encrypted_file}")
        print(f"Encryption key (keep secure!): {key}")
        
        # Example usage for decryption
        print("\nTo decrypt the file, scan the QR code with the decryption program.")
        print("The program will verify your face before decrypting the file.")