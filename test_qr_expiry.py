import unittest
import io
import json
from datetime import datetime, timedelta
from app import app
from pyzbar.pyzbar import decode
from PIL import Image
import qrcode

class TestQRExpiry(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

        # Fake login
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["user_email"] = "expiry@gmail.com"

        # Generate QR
        with open("tests/sample.pdf", "rb") as pdf, open("tests/face.jpg", "rb") as face:
            upload = self.client.post("/upload", data={
                "document": (pdf, "report.pdf"),
                "photo_0": (face, "face.jpg"),
                "email_0": "receiver@test.com",
                "expiration_hours": "1"
            }, content_type="multipart/form-data")

        data = json.loads(upload.data)
        qr_id = data["qr_codes"][0]["qr_id"]

        qr = self.client.get(f"/qr/{qr_id}")
        qr_bytes = bytes(qr.data)

        # Decode QR
        img = Image.open(io.BytesIO(qr_bytes))
        decoded = decode(img)[0].data.decode()
        payload = json.loads(decoded)

        # Expire it
        payload["expiry"] = (datetime.now() - timedelta(hours=2)).isoformat()

        # Re-encode expired QR
        qr_img = qrcode.make(json.dumps(payload))
        buf = io.BytesIO()
        qr_img.save(buf, format="PNG")
        self.expired_qr = buf.getvalue()

    def test_T09_expired_qr_rejected(self):
        qr_file = io.BytesIO(self.expired_qr)
        qr_file.name = "expired.png"

        scan = self.client.post("/scan", data={
            "file": (qr_file, "expired.png")
        }, content_type="multipart/form-data")

        self.assertEqual(scan.status_code, 400)

        result = json.loads(scan.data)
        self.assertTrue(result["expired"])

