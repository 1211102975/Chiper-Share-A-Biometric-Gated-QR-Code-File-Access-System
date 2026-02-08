import unittest
from app import app
from db import get_db_connection

class TestOTPVerification(unittest.TestCase):
    """
    T12 – Validate OTP verification accepts valid input and rejects invalid OTP
    """

    def setUp(self):
        app.testing = True
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

        # Insert a fake OTP record into AccessLog
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO AccessLog (
                file_id, receiver_email,
                face_match_result, confidence_score,
                otp_code, otp_status, otp_created_at
            )
            OUTPUT INSERTED.log_id
            VALUES (?, ?, ?, ?, ?, ?, GETDATE())
        """, (
            1, "receiver@test.com",
            1, 0.92,
            "555888", "Sent"
        ))

        self.log_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()

        # Simulate user session
        with self.client.session_transaction() as sess:
            sess["log_id"] = self.log_id
            sess["otp_sent"] = True
            sess["qr_data"] = {
                "file_id": 1,
                "receiver_email": "receiver@test.com"
            }

    def test_T12_verify_otp_valid_and_invalid(self):

        # -----------------------
        # 1️⃣ Wrong OTP
        # -----------------------
        bad = self.client.post("/verify_otp", data={
            "otp": "123111"
        })

        self.assertEqual(bad.status_code, 200)
        self.assertIn("error", bad.json)

        # OTP must still be unverified
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT otp_status FROM AccessLog WHERE log_id=?", self.log_id)
        status = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(status, "Sent")

        # -----------------------
        # 2️⃣ Correct OTP
        # -----------------------
        good = self.client.post("/verify_otp", data={
            "otp": "555888"
        })

        self.assertEqual(good.status_code, 200)
        self.assertTrue(good.json.get("success"))

        # OTP must now be Verified
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT otp_status FROM AccessLog WHERE log_id=?", self.log_id)
        status = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(status, "Verified")
