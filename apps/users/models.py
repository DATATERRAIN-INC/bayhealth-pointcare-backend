import secrets
import uuid
from datetime import timedelta

from django.contrib.auth.hashers import check_password, make_password
from django.db import models
from django.utils.timezone import now


class User(models.Model):
    cognito_id = models.CharField(max_length=128, unique=True)
    cognito_username = models.CharField(max_length=128, unique=True, null=True, blank=True)
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=150, blank=True)
    last_name = models.CharField(max_length=150, blank=True)
    phone_number = models.CharField(max_length=20, blank=True)
    is_active = models.BooleanField(default=True)
    is_confirmed = models.BooleanField(default=False)
    password_changed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.email


class EmailTemplate(models.Model):
    name = models.CharField(max_length=255, unique=True)
    subject = models.CharField(max_length=255)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "email_templates"
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class PasswordResetRequest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    reset_code = models.CharField(max_length=255)
    token = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    is_used = models.BooleanField(default=False)
    used_at = models.DateTimeField(null=True, blank=True)

    @staticmethod
    def generate_reset_code():
        return f"{secrets.randbelow(900000) + 100000}"

    @staticmethod
    def create_reset_request(user):
        raw_reset_code = PasswordResetRequest.generate_reset_code()
        hashed_reset_code = make_password(raw_reset_code)
        token = str(uuid.uuid4())
        expiration_time = now() + timedelta(minutes=10)

        reset_request, _created = PasswordResetRequest.objects.update_or_create(
            user=user,
            defaults={
                "reset_code": hashed_reset_code,
                "token": token,
                "expires_at": expiration_time,
                "is_used": False,
                "used_at": None,
            },
        )
        return reset_request, raw_reset_code

    def verify_reset_code(self, input_code):
        return check_password(input_code, self.reset_code)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["user"], name="uniq_one_reset_request_per_user"),
        ]
