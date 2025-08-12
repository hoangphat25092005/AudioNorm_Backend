from fastapi_mail import FastMail, ConnectionConfig
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Get environment variables for email configuration
mail_username = os.getenv("MAIL_USERNAME", "")
mail_password = os.getenv("MAIL_PASSWORD", "")
mail_server = os.getenv("MAIL_SERVER", "smtp.gmail.com")
mail_port = int(os.getenv("MAIL_PORT", "587"))
mail_from_name = os.getenv("MAIL_FROM_NAME", "AudioNorm Platform")

# Initialize email configuration only if credentials are provided
fast_mail = None

def is_email_configured() -> bool:
    """Check if email is properly configured"""
    return bool(mail_username and mail_password and "@" in mail_username)

def get_fast_mail():
    """Get FastMail instance, initialize if not already done"""
    global fast_mail
    
    if not is_email_configured():
        logger.warning("Email not configured. Set MAIL_USERNAME and MAIL_PASSWORD environment variables.")
        return None
    
    if fast_mail is None:
        try:
            conf = ConnectionConfig(
                MAIL_USERNAME=mail_username,
                MAIL_PASSWORD=mail_password,
                MAIL_FROM=mail_username,
                MAIL_PORT=mail_port,
                MAIL_SERVER=mail_server,
                MAIL_FROM_NAME=mail_from_name,
                MAIL_STARTTLS=True,
                MAIL_SSL_TLS=False,
                USE_CREDENTIALS=True,
                VALIDATE_CERTS=True,
                TEMPLATE_FOLDER=Path(__file__).parent.parent / 'templates'
            )
            fast_mail = FastMail(conf)
            logger.info("Email configuration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize email configuration: {str(e)}")
            return None
    
    return fast_mail
