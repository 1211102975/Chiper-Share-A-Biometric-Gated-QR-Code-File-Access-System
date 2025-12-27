------------------------------------------------------------
-- 1. USERS TABLE
------------------------------------------------------------
CREATE TABLE Users (
    user_id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    profile_pic_path VARCHAR(255) NULL,
    created_at DATETIME DEFAULT GETDATE()
);

------------------------------------------------------------
-- 2. FILES TABLE
-- Stores actual uploaded file metadata + server file path
------------------------------------------------------------
CREATE TABLE Files (
    file_id INT IDENTITY(1,1) PRIMARY KEY,
    uploaded_by INT NOT NULL,
    file_path VARCHAR(500) NOT NULL,     -- saved in /static/uploads/
    file_name VARCHAR(255) NULL,
    file_mime VARCHAR(100) NULL,
    upload_timestamp DATETIME DEFAULT GETDATE(),
    expiration_timestamp DATETIME NOT NULL,

    CONSTRAINT fk_file_user FOREIGN KEY (uploaded_by)
        REFERENCES Users(user_id)
);
------------------------------------------------------------
-- 3. FileKey TABLE
-- Stores AES key, IV, and Tag for each file
------------------------------------------------------------
CREATE TABLE FileKey (
    key_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    file_id INT NOT NULL,
    aes_key VARBINARY(MAX) NOT NULL,
    iv VARBINARY(50) NOT NULL,
    tag VARBINARY(50) NULL,   -- only for AES-GCM
    created_at DATETIME DEFAULT GETDATE(),

    FOREIGN KEY (file_id) REFERENCES Files(file_id)
);
------------------------------------------------------------
-- 4. ReceiverFace TABLE
-- Stores receiver's uploaded face image path + face encoding
------------------------------------------------------------
CREATE TABLE ReceiverFace (
    receiver_face_id INT IDENTITY(1,1) PRIMARY KEY,
    file_id INT NOT NULL,
    receiver_email VARCHAR(255) NOT NULL,
    photo_path VARCHAR(500) NOT NULL,
    face_encoding VARBINARY(MAX) NOT NULL,
    created_at DATETIME DEFAULT GETDATE(),

    FOREIGN KEY (file_id) REFERENCES Files(file_id)
);

------------------------------------------------------------
-- 5. QRCODE TABLE
-- each receiver has their own QR code image + encrypted metadata
------------------------------------------------------------
CREATE TABLE QRCode (
    qr_id INT IDENTITY(1,1) PRIMARY KEY,
    file_id INT NOT NULL,
    receiver_email VARCHAR(255) NOT NULL,
    qr_image_path VARCHAR(500) NOT NULL,
    qr_metadata NVARCHAR(MAX) NOT NULL,
    qr_timestamp DATETIME DEFAULT GETDATE(),

    FOREIGN KEY (file_id) REFERENCES Files(file_id)
);


------------------------------------------------------------
-- 6. ACCESS LOG TABLE
-- Stores face verification results + OTP verification + download status
------------------------------------------------------------
CREATE TABLE AccessLog (
    log_id INT IDENTITY(1,1) PRIMARY KEY,
    file_id INT NOT NULL,

    receiver_email VARCHAR(255) NOT NULL,
    face_match_result BIT NOT NULL,        -- 1=match, 0=no
    confidence_score FLOAT NOT NULL,

    otp_code VARCHAR(10) NULL,
    otp_created_at DATETIME NULL,
    otp_status VARCHAR(30) NULL,           -- Sent / Verified / Invalid

    access_result VARCHAR(30) NULL,        -- Success / Failed Face / Failed OTP
    access_time DATETIME DEFAULT GETDATE(),

    FOREIGN KEY (file_id) REFERENCES Files(file_id)
);

------------------------------------------------------------