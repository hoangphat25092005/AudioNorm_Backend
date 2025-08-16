from fastapi_mail import MessageSchema, MessageType
from app.config.email import get_fast_mail, is_email_configured
from app.config.database import get_db
from pydantic import EmailStr
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EmailService:
    @staticmethod
    async def send_reset_password_email(user_email: str, username: str, reset_url: str, is_verified: bool = False):
        """Send password reset email to user, only if user is verified"""
        from app.config.email import get_fast_mail, is_email_configured
        import os
        logger = logging.getLogger(__name__)

        if not is_verified:
            logger.warning(f"User {user_email} attempted password reset but is not verified.")
            return {"status": "forbidden", "message": "User must verify email before using forgot password."}

        if not is_email_configured():
            logger.warning("Email not configured. Skipping reset password email.")
            return {"status": "skipped", "message": "Email not configured"}

        fast_mail = get_fast_mail()
        if not fast_mail:
            logger.error("Failed to get email configuration")
            return {"status": "error", "message": "Email service unavailable"}

        subject = "Reset Your AudioNorm Password"
        html_body = f"""
        <html>
        <body>
            <h2>Hello, {username}!</h2>
            <p>We received a request to reset your AudioNorm password.</p>
            <p>Click the link below to set a new password. This link will expire in 1 hour.</p>
            <a href='{reset_url}' style='padding:10px 20px;background:#4CAF50;color:white;text-decoration:none;border-radius:5px;'>Reset Password</a>
            <p>If you did not request this, you can ignore this email.</p>
        </body>
        </html>
        """
        text_body = f"Hello, {username}!\nReset your password using this link (valid for 1 hour): {reset_url}"

        message = MessageSchema(
            subject=subject,
            recipients=[user_email],
            body=text_body,
            html=html_body,
            subtype=MessageType.html
        )
        try:
            await fast_mail.send_message(message)
            logger.info(f"Reset password email sent to {user_email}")
            return {"status": "success", "message": "Reset password email sent"}
        except Exception as e:
            logger.error(f"Error sending reset password email to {user_email}: {str(e)}")
            return {"status": "error", "message": f"Failed to send reset password email: {str(e)}"}

    @staticmethod
    async def send_verification_email(user_email: str, username: str, token: str):
        """Send email verification link to user"""
        from app.config.email import get_fast_mail, is_email_configured
        import os
        logger = logging.getLogger(__name__)

        if not is_email_configured():
            logger.warning("Email not configured. Skipping verification email.")
            return {"status": "skipped", "message": "Email not configured"}

        fast_mail = get_fast_mail()
        if not fast_mail:
            logger.error("Failed to get email configuration")
            return {"status": "error", "message": "Email service unavailable"}

        # You may want to set this to your frontend or backend URL
        base_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        verify_url = f"{base_url}/verify-email?token={token}"

        subject = "Verify your AudioNorm account"
        html_body = f"""
        <html>
        <body>
            <h2>Welcome, {username}!</h2>
            <p>Thank you for registering at AudioNorm.</p>
            <p><b>Verify your email to receive notify email when others comment on your feedback and to enable forgot password feature.</b></p>
            <p>Please verify your email address by clicking the link below:</p>
            <a href='{verify_url}' style='padding:10px 20px;background:#4CAF50;color:white;text-decoration:none;border-radius:5px;'>Verify Email</a>
            <p>If you did not create this account, you can ignore this email.</p>
        </body>
        </html>
        """
        text_body = f"Welcome, {username}!\nVerify your email to receive notify email when others comment on your feedback and to enable forgot password feature.\nPlease verify your email: {verify_url}"

        message = MessageSchema(
            subject=subject,
            recipients=[user_email],
            body=text_body,
            html=html_body,
            subtype=MessageType.html
        )
        try:
            await fast_mail.send_message(message)
            logger.info(f"Verification email sent to {user_email}")
            return {"status": "success", "message": "Verification email sent"}
        except Exception as e:
            logger.error(f"Error sending verification email to {user_email}: {str(e)}")
            return {"status": "error", "message": f"Failed to send verification email: {str(e)}"}
    
    @staticmethod
    async def send_feedback_notification(
        feedback_author_email: str,
        feedback_author_name: str,
        responder_name: str,
        responder_email: str,
        feedback_title: str,
        response_content: str,
        feedback_id: str
    ):
        """Send email notification when someone responds to user's feedback"""
        
        # Check if email is configured
        if not is_email_configured():
            logger.warning("Email not configured. Skipping feedback notification email.")
            return {"status": "skipped", "message": "Email not configured"}
        
        fast_mail = get_fast_mail()
        if not fast_mail:
            logger.error("Failed to get email configuration")
            return {"status": "error", "message": "Email service unavailable"}
        
        subject = f"New Response to Your Feedback: {feedback_title}"

        # Create HTML email body (no commenter email, id, or web url)
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .response-box {{ background-color: white; padding: 15px; border-left: 4px solid #4CAF50; margin: 15px 0; }}
                .footer {{ text-align: center; padding: 20px; color: #666; }}
                .feedback-info {{ background-color: #e8f5e8; padding: 15px; margin: 15px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AudioNorm Platform</h1>
                    <p>📧 New Response to Your Feedback</p>
                </div>
                <div class="content">
                    <h2>Hello {feedback_author_name}!</h2>
                    <p><strong>{responder_name}</strong> has responded to your feedback:</p>
                    <div class="feedback-info">
                        <h3>Your Feedback: "{feedback_title}"</h3>
                    </div>
                    <div class="response-box">
                        <h4>💬 Response:</h4>
                        <p>{response_content}</p>
                    </div>
                </div>
                <div class="footer">
                    <p>This is an automated message from AudioNorm Platform.</p>
                    <p>If you don't want to receive these notifications, you can update your preferences in your account settings.</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Create text version for email clients that don't support HTML (no commenter email, id, or web url)
        text_body = f"""
        Hello {feedback_author_name}!

        {responder_name} has responded to your feedback: "{feedback_title}"

        Response: {response_content}

        Best regards,
        AudioNorm Platform Team
        """
        
        message = MessageSchema(
            subject=subject,
            recipients=[feedback_author_email],
            body=text_body,
            html=html_body,
            subtype=MessageType.html
        )
        
        try:
            await fast_mail.send_message(message)
            logger.info(f"Email notification sent to {feedback_author_email} for feedback {feedback_id}")
            return {"status": "success", "message": "Email sent successfully"}
        except Exception as e:
            logger.error(f"Error sending email to {feedback_author_email}: {str(e)}")
            return {"status": "error", "message": f"Failed to send email: {str(e)}"}
    
    @staticmethod
    async def send_feedback_confirmation(
        user_email: str,
        username: str,
        feedback_content: str,
        feedback_id: str
    ):
        """Send confirmation email when user submits feedback"""
        
        # Check if email is configured
        if not is_email_configured():
            logger.warning("Email not configured. Skipping feedback confirmation email.")
            return {"status": "skipped", "message": "Email not configured"}
        
        fast_mail = get_fast_mail()
        if not fast_mail:
            logger.error("Failed to get email configuration")
            return {"status": "error", "message": "Email service unavailable"}
        
        subject = "Feedback Submitted Successfully - AudioNorm Platform"
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #2196F3; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .feedback-box {{ background-color: white; padding: 15px; border-left: 4px solid #2196F3; margin: 15px 0; }}
                .footer {{ text-align: center; padding: 20px; color: #666; }}
                .button {{ background-color: #2196F3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AudioNorm Platform</h1>
                    <p>✅ Feedback Submitted Successfully</p>
                </div>
                <div class="content">
                    <h2>Thank you, {username}!</h2>
                    <p>We've received your feedback and appreciate you taking the time to help us improve.</p>
                    
                    <div class="feedback-box">
                        <h4>Your Feedback:</h4>
                        <p>{feedback_content}</p>
                        <p><small>Feedback ID: {feedback_id}</small></p>
                    </div>
                    
                    <h3>What happens next?</h3>
                    <ul>
                        <li>📧 You'll receive email notifications for any responses</li>
                        <li>👥 Other users may respond to your feedback</li>
                        <li>🔄 You can track responses and continue the conversation</li>
                    </ul>
                    
                    <p><a href="http://localhost:8000/docs" class="button">View Your Feedback</a></p>
                </div>
                <div class="footer">
                    <p>Thank you for helping us improve AudioNorm Platform!</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_body = f"""
        Thank you, {username}!
        
        We've received your feedback and appreciate you taking the time to help us improve.
        
        Your Feedback: {feedback_content}
        Feedback ID: {feedback_id}
        
        What happens next?
        - You'll receive email notifications for any responses
        - Other users may respond to your feedback
        - You can track responses and continue the conversation
        
        View your feedback at: http://localhost:8000/docs
        
        Thank you for helping us improve AudioNorm Platform!
        """
        
        message = MessageSchema(
            subject=subject,
            recipients=[user_email],
            body=text_body,
            html=html_body,
            subtype=MessageType.html
        )
        
        try:
            await fast_mail.send_message(message)
            logger.info(f"Feedback confirmation email sent to {user_email}")
            return {"status": "success", "message": "Confirmation email sent successfully"}
        except Exception as e:
            logger.error(f"Error sending confirmation email to {user_email}: {str(e)}")
            return {"status": "error", "message": f"Failed to send confirmation email: {str(e)}"}
    
    @staticmethod
    async def send_bulk_notification(
        subject: str,
        recipients: List[str],
        content: str,
        notification_type: str = "general"
    ):
        """Send bulk notifications to multiple users"""
        
        # Check if email is configured
        if not is_email_configured():
            logger.warning("Email not configured. Skipping bulk notification.")
            return {"status": "skipped", "message": "Email not configured"}
        
        fast_mail = get_fast_mail()
        if not fast_mail:
            logger.error("Failed to get email configuration")
            return {"status": "error", "message": "Email service unavailable"}
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #FF9800; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .footer {{ text-align: center; padding: 20px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AudioNorm Platform</h1>
                    <p>📢 Platform Notification</p>
                </div>
                <div class="content">
                    <h2>{subject}</h2>
                    <div style="background-color: white; padding: 15px; border-radius: 5px;">
                        {content}
                    </div>
                </div>
                <div class="footer">
                    <p>AudioNorm Platform Team</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        message = MessageSchema(
            subject=subject,
            recipients=recipients,
            body=content,
            html=html_body,
            subtype=MessageType.html
        )
        
        try:
            await fast_mail.send_message(message)
            logger.info(f"Bulk notification sent to {len(recipients)} recipients")
            return {"status": "success", "message": f"Bulk notification sent to {len(recipients)} recipients"}
        except Exception as e:
            logger.error(f"Error sending bulk notification: {str(e)}")
            return {"status": "error", "message": f"Failed to send bulk notification: {str(e)}"}
