import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender_email="cyloixy2610@gmail.com"
sender_password="gprb gqku lqdf iemg"
reciever_email="cyloixy2610@gmail.com"

subject = "Test Email for OTP"
body = "your otp is : 123456"

msg= MIMEMultipart()
msg['From']= sender_email
msg['To']=reciever_email
msg['Subject']= subject
msg.attach(MIMEText(body,'plain'))

try:
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    text = msg.as_string()
    server.sendmail(sender_email, reciever_email, text)
    print("Email sent successfully!")
except Exception as e:
    print(f"Failed to send email: {e}")