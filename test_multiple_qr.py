import unittest
import json
from unittest.mock import patch
from app import app

class TestMultipleQR(unittest.TestCase):
    """
    T07 – Verify unique QR code generated for each receiver
    """

    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

        # Fake login session (no DB needed)
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["user_email"] = "multitest@gmail.com"

    @patch("face_recognition.face_encodings")
    def test_T07_multiple_receivers_qr(self, mock_face):
        # Mock face recognition so it does not allocate memory
        mock_face.return_value = [[0.1] * 128]

        with open("tests/sample.pdf", "rb") as pdf, \
             open("tests/face1.jpg", "rb") as face1, \
             open("tests/face2.jpg", "rb") as face2:

            response = self.client.post("/upload", data={
                "document": (pdf, "report.pdf"),

                "photo_0": (face1, "face1.jpg"),
                "email_0": "alice@test.com",

                "photo_1": (face2, "face2.jpg"),
                "email_1": "bob@test.com",

                "expiration_hours": "24"
            }, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)

        # Must return two QR codes
        self.assertEqual(len(data["qr_codes"]), 2)

        # Each QR must be unique
        qr_ids = [qr["qr_id"] for qr in data["qr_codes"]]
        self.assertNotEqual(qr_ids[0], qr_ids[1])

