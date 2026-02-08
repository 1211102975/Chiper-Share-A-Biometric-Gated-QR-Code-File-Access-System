import unittest
import os
from app import app
from db import get_db_connection
from werkzeug.security import generate_password_hash

class TestEncryption(unittest.TestCase):
    """
    T05 – Verify files are encrypted using AES-256-GCM before being saved
    """

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

        conn = get_db_connection()
        cursor = conn.cursor()

        # Clean existing test user
        cursor.execute("SELECT user_id FROM Users WHERE email = ?", ("cryptotest@gmail.com",))
        row = cursor.fetchone()
        if row:
            uid = row[0]
            cursor.execute("DELETE FROM FileKey WHERE file_id IN (SELECT file_id FROM Files WHERE uploaded_by=?)", uid)
            cursor.execute("DELETE FROM Files WHERE uploaded_by=?", uid)
            cursor.execute("DELETE FROM Users WHERE user_id=?", uid)

        # Create test user
        cursor.execute("""
            INSERT INTO Users (name, email, password_hash, profile_pic_path)
            VALUES (?, ?, ?, ?)
        """, ("Crypto User", "cryptotest@gmail.com",
              generate_password_hash("Crypto123"),
              "profile_pics/default.jpg"))
        conn.commit()

        cursor.execute("SELECT user_id FROM Users WHERE email=?", ("cryptotest@gmail.com",))
        self.user_id = cursor.fetchone()[0]
        conn.close()

        # Log user in
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_id
            sess["user_email"] = "cryptotest@gmail.com"

    def test_T05_aes_gcm_encryption(self):
        """
        Verify AES-256-GCM encryption is applied before saving to disk
        """

        pdf = open("tests/sample.pdf", "rb")
        face = open("tests/face.jpg", "rb")

        response = self.client.post("/upload", data={
            "document": (pdf, "report.pdf"),
            "photo_0": (face, "face.jpg"),
            "email_0": "receiver@test.com",
            "expiration_hours": "24"
        }, content_type="multipart/form-data")

        pdf.close()
        face.close()

        self.assertEqual(response.status_code, 200)

        # Fetch encryption metadata
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT F.file_path, K.aes_key, K.iv, K.tag
            FROM Files F
            JOIN FileKey K ON F.file_id = K.file_id
            WHERE F.uploaded_by = ?
            ORDER BY F.file_id DESC
        """, self.user_id)

        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)

        file_path, aes_key, iv, tag = row

        # Validate AES-256-GCM parameters
        self.assertEqual(len(aes_key), 32)   # 256-bit key
        self.assertEqual(len(iv), 12)        # GCM standard nonce
        self.assertIsNotNone(tag)

        # Verify file is not plaintext
        encrypted_path = os.path.join(app.root_path, file_path)

        with open(encrypted_path, "rb") as f:
            encrypted_data = f.read()

        with open("tests/sample.pdf", "rb") as f:
            original = f.read()

        self.assertNotIn(original[:100], encrypted_data)
