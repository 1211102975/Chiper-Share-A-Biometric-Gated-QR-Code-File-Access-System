import cv2
import base64
import json
import numpy as np
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
import face_verifier  # <-- This will be your modified module
from pyzbar.pyzbar import decode

def decrypt_file(ciphertext, key, nonce):
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    plaintext = cipher.decrypt(ciphertext)
    return plaintext

def scan_qr_code(image_path):
    image = cv2.imread(image_path)
    decoded_objs = decode(image)
    for obj in decoded_objs:
        try:
            # Decode QR to JSON
            data = base64.b64decode(obj.data)
            json_data = json.loads(data)
            return json_data
        except Exception as e:
            print(f"QR decode error: {e}")
            return None
    return None

def main():
    qr_data = scan_qr_code("file_qrcode.png")
    if not qr_data:
        print("No valid QR code found.")
        return

    print("QR scanned. Starting face recognition...")

    # Step 1: Perform face verification
    is_verified, identity, confidence = face_verifier.verify_user(min_confidence=50.0)
    if not is_verified:
        print("Face verification failed. Access denied.")
        return

    print(f"Face verified: {identity} ({confidence:.1f}%)")

    # Step 2: Decrypt and save the file
    ciphertext = base64.b64decode(qr_data["ciphertext"])
    key = base64.b64decode(qr_data["key"])
    nonce = base64.b64decode(qr_data["nonce"])
    filename = qr_data["filename"]

    try:
        plaintext = decrypt_file(ciphertext, key, nonce)
        with open(filename, "wb") as f:
            f.write(plaintext)
        print(f"File decrypted and saved as {filename}")
    except Exception as e:
        print(f"Decryption failed: {e}")

if __name__ == "__main__":
    main()
