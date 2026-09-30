import logging

from django.conf import settings
from django.template import Context, Template
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Email, Mail

from apps.users.models import EmailTemplate

logger = logging.getLogger(__name__)


class EmailUtils:
    restriction_enabled = settings.EMAIL_RESTRICTION

    @staticmethod
    def send_email_via_sendgrid_template(
        to_emails,
        template_name,
        context_data,
        content_type="html",
        from_email=None,
    ):
        if EmailUtils.restriction_enabled:
            logger.info("[EmailUtils] Email sending restricted, skipping")
            return True
        if from_email is None:
            from_email = settings.DEFAULT_FROM_EMAIL
        if isinstance(to_emails, str):
            to_emails = [to_emails]

        try:
            email_template = EmailTemplate.objects.get(name=template_name)
        except EmailTemplate.DoesNotExist:
            logger.warning("Email template '%s' not found.", template_name)
            return True

        try:
            subject = Template(email_template.subject).render(Context(context_data))
            content = Template(email_template.content).render(Context(context_data))
        except Exception as exc:
            raise ValueError(f"Error rendering template: {exc}") from exc

        message = Mail(
            from_email=Email(from_email, name=settings.EMAIL_TITLE_CARD_NAME),
            to_emails=to_emails,
            subject=subject,
            html_content=content if content_type == "html" else None,
            plain_text_content=content if content_type == "text" else None,
        )

        try:
            sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
            response = sg.send(message)
            logger.info(
                "[EmailUtils.send_email_via_sendgrid_template] SendGrid status: %s",
                response.status_code,
            )
            return response
        except Exception as exc:
            logger.exception("[EmailUtils.send_email_via_sendgrid_template] Error")
            raise RuntimeError(f"Error sending email using SendGrid: {exc}") from exc
