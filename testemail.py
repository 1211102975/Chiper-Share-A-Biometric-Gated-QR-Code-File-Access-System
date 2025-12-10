import random
import smtplib
from email.message import EmailMessage
import traceback

try:
    otp = ""
    for i in range(6):
        otp += str(random.randint(0, 9))

    print(f"Generated OTP: {otp}")
    print("\nConnecting to Gmail SMTP server...")
    
    server = smtplib.SMTP('smtp.gmail.com', 587, timeout=10)
    server.set_debuglevel(1)  # Enable debug output
    
    print("Starting TLS encryption...")
    server.starttls()

    from_mail = 'cyloixy2610@gmail.com'
    print(f"\nLogging in with: {from_mail}")
    server.login(from_mail, 'jnzb bvae knot evss')
    
    to_mail = input("\nEnter your email: ")

    msg = EmailMessage()
    msg['Subject'] = "OTP Verification"
    msg['From'] = from_mail
    msg['To'] = to_mail
    msg.set_content(f"Your OTP is: {otp}")

    print(f"\nSending email to: {to_mail}")
    server.send_message(msg)
    print("\n✓ Email sent successfully!")
    server.quit()

except smtplib.SMTPAuthenticationError as e:
    print(f"\n✗ SMTP Authentication Error: {e}")
    print("\nPossible solutions:")
    print("1. Enable 2-Factor Authentication on your Gmail account")
    print("2. Generate a new App Password at: https://myaccount.google.com/apppasswords")
    print("3. Use the App Password instead of your regular password")
    print("4. Make sure 'Less secure app access' is enabled (if not using App Password)")

except smtplib.SMTPException as e:
    print(f"\n✗ SMTP Error: {e}")
    
except Exception as e:
    print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")
    traceback.print_exc()