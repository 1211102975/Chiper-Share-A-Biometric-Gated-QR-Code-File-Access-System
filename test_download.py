import unittest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
from datetime import datetime, timedelta
from app import app
from db import get_db_connection

def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

        conn = get_db_connection()
        cursor = conn.cursor()

        # Create or reuse user
        cursor.execute("SELECT user_id FROM Users WHERE email=?", ("download@test.com",))
        row = cursor.fetchone()
        if row:
            self.user_id = row[0]
        else:
            cursor.execute("""
                INSERT INTO Users (name, email)
                OUTPUT INSERTED.user_id
                VALUES ('Download Test', 'download@test.com')
            """)
            self.user_id = cursor.fetchone()[0]

        # Clean old data
        cursor.execute("""
            DELETE FROM FileKey
            WHERE file_id IN (SELECT file_id FROM Files WHERE uploaded_by=?)
        """, (self.user_id,))
        cursor.execute("DELETE FROM Files WHERE uploaded_by=?", (self.user_id,))
        conn.commit()

        # Load real PDF
        with open("tests/sample.pdf", "rb") as f:
            self.original_data = f.read()

        # AES-256-GCM encryption
        key = AESGCM.generate_key(bit_length=256)
        iv = os.urandom(12)
        aes = AESGCM(key)

        encrypted_with_tag = aes.encrypt(iv, self.original_data, None)

        ciphertext = encrypted_with_tag[:-16]
        tag = encrypted_with_tag[-16:]

        # Write ciphertext to disk
        os.makedirs("static/uploads", exist_ok=True)
        self.enc_path = "static/uploads/test_enc.bin"
        with open(self.enc_path, "wb") as f:
            f.write(ciphertext)

        # Insert file
        cursor.execute("""
            INSERT INTO Files (uploaded_by, file_path, file_name, expiration_timestamp)
            OUTPUT INSERTED.file_id
            VALUES (?, ?, ?, ?)
        """, (
            self.user_id,
            "/static/uploads/test_enc.bin",
            "sample.pdf",
            datetime.now() + timedelta(hours=1)
        ))
        self.file_id = cursor.fetchone()[0]

        # Insert AES keys
        cursor.execute("""
            INSERT INTO FileKey (file_id, aes_key, iv, tag)
            VALUES (?, ?, ?, ?)
        """, (self.file_id, key, iv, tag))

        conn.commit()
        conn.close()

        # Simulate OTP verified session
        with self.client.session_transaction() as sess:
            sess["qr_data"] = {"file_id": self.file_id}
            sess["otp_verified"] = True


def test_T14_secure_download(self):
        """
        T14 – OTP-verified user should receive decrypted file
        """
        response = self.client.get("/download")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, self.original)

def tearDown(self):
        if os.path.exists(self.enc_path):
            os.remove(self.enc_path)


if __name__ == "__main__":
    unittest.main()
