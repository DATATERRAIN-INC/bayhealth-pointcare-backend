from django.urls import path

from apps.users.views import (
    ForgotPasswordView,
    LoginView,
    LogoutView,
    RegisterView,
    ResetPasswordView,
    ValidatePasswordTokenView,
    VerifyResetCodeView,
)

urlpatterns = [
    path("register/", RegisterView.as_view(), name="register"),
    path("login/", LoginView.as_view(), name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("forgot-password/", ForgotPasswordView.as_view(), name="forgot_password"),
    path("verify-reset-code/", VerifyResetCodeView.as_view(), name="verify_reset_code"),
    path(
        "validate-password-token/",
        ValidatePasswordTokenView.as_view(),
        name="validate_password_token",
    ),
    path("reset-password/", ResetPasswordView.as_view(), name="reset_password"),
]
