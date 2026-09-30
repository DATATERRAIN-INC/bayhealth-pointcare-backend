from django.core.management.base import BaseCommand

from apps.users.constants import PASSWORD_RESET_TEMPLATE
from apps.users.models import EmailTemplate

PASSWORD_RESET_SUBJECT = "Your BayHealth PointCare password reset code"
PASSWORD_RESET_CONTENT = """
<p>Hello {{ user_name }},</p>
<p>Your password reset code is <strong>{{ reset_code }}</strong>.</p>
<p>This code is valid for 10 minutes.</p>
<p>If you did not request a password reset, you can ignore this email.</p>
"""


class Command(BaseCommand):
    help = "Seed EmailTemplate rows used by SendGrid (e.g. PASSWORD_RESET)."

    def handle(self, *args, **options):
        obj, created = EmailTemplate.objects.update_or_create(
            name=PASSWORD_RESET_TEMPLATE,
            defaults={
                "subject": PASSWORD_RESET_SUBJECT,
                "content": PASSWORD_RESET_CONTENT.strip(),
            },
        )
        action = "Created" if created else "Updated"
        self.stdout.write(self.style.SUCCESS(f"{action} email template: {obj.name}"))
