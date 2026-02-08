import unittest
import io
import json
from app import app

class TestQRScan(unittest.TestCase):
    """
    T08 – Validate QR code can be scanned and metadata decoded correctly
    """

    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

        # Fake login
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["user_email"] = "qrtest@gmail.com"

        # Generate a real QR
        with open("tests/sample.pdf", "rb") as pdf, open("tests/face.jpg", "rb") as face:
            upload = self.client.post("/upload", data={
                "document": (pdf, "report.pdf"),
                "photo_0": (face, "face.jpg"),
                "email_0": "receiver@test.com",
                "expiration_hours": "24"
            }, content_type="multipart/form-data")

        data = json.loads(upload.data)
        self.qr_id = data["qr_codes"][0]["qr_id"]

        qr = self.client.get(f"/qr/{self.qr_id}")
        self.qr_bytes = bytes(qr.data)

    def test_T08_qr_scan_and_decode(self):
        qr_file = io.BytesIO(self.qr_bytes)
        qr_file.name = "qr.png"

        response = self.client.post("/scan", data={
            "file": (qr_file, "qr.png")
        }, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)

        # Verify session has QR metadata
        with self.client.session_transaction() as sess:
            self.assertIn("qr_data", sess)
            meta = sess["qr_data"]

            self.assertIn("file_id", meta)
            self.assertIn("receiver_email", meta)
            self.assertIn("expiry", meta)

