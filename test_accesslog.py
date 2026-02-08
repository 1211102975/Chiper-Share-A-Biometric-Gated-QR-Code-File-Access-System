import unittest
import pickle
import numpy as np
import io
import cv2
from unittest.mock import patch
from app import app
from db import get_db_connection


class TestAccessLog(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

        with self.client.session_transaction() as sess:
            sess["qr_data"] = {
                "file_id": 1,
                "receiver_email": "receiver@test.com",
                "expiry": "2099-01-01T00:00:00"
            }
            sess["otp_sent"] = False
            sess["log_id"] = None

        self.fake_encoding = np.zeros(128)

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM AccessLog WHERE receiver_email=?", ("receiver@test.com",))
        cursor.execute("DELETE FROM ReceiverFace WHERE receiver_email=?", ("receiver@test.com",))

        cursor.execute("""
            INSERT INTO ReceiverFace (file_id, receiver_email, face_encoding, photo_path)
            VALUES (?, ?, ?, ?)
        """, (
            1,
            "receiver@test.com",
            pickle.dumps(self.fake_encoding),
            "static/receiver_faces/test.jpg"
        ))

        conn.commit()
        conn.close()

    @patch("face_recognition.face_encodings")
    def test_T13_accesslog_created(self, mock_face_encodings):
        """
        T13 – Every successful face verification must create AccessLog
        """

        # Make face_recognition return SAME face as stored → perfect match
        mock_face_encodings.return_value = [self.fake_encoding]

        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", fake_img)

        response = self.client.post(
            "/verify_face_stream",
            data={"frame": (io.BytesIO(buf.tobytes()), "frame.jpg")},
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 200)

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT file_id, receiver_email, confidence_score, otp_status
            FROM AccessLog
            WHERE receiver_email = ?
        """, ("receiver@test.com",))

        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], "receiver@test.com")
        self.assertGreater(row[2], 0)      # confidence_score
        self.assertEqual(row[3], "Sent")   # OTP issued


if __name__ == "__main__":
    unittest.main()
