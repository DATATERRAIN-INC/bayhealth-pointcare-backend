import json
import sys
from pathlib import Path

from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
# Fast Collect Mode semantic resolver (Haiku 5.5 + low effort).
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5").strip()
ANTHROPIC_EFFORT = os.environ.get("ANTHROPIC_EFFORT", "low").strip() or "low"
ANTHROPIC_TIMEOUT_S = float(os.environ.get("ANTHROPIC_TIMEOUT_S", "1.5"))
ANTHROPIC_MIN_CONFIDENCE = float(os.environ.get("ANTHROPIC_MIN_CONFIDENCE", "0.72"))
CHROME_CHANNEL = os.environ.get("CHROME_CHANNEL", "chrome").strip()
# Optional absolute path to chrome/google-chrome (preferred on Linux EC2).
CHROME_EXECUTABLE = os.environ.get("CHROME_EXECUTABLE", "").strip()
# Opt-in: some locked-down Linux hosts need --no-sandbox. Keep off by default.
CHROME_NO_SANDBOX = os.environ.get("CHROME_NO_SANDBOX", "false").lower() in (
    "true",
    "1",
    "yes",
)
HEADLESS = os.environ.get("HEADLESS", "false").lower() in ("true", "1", "yes")


def application_test_mode_from_env(env: dict | None = None) -> bool:
    """True only when the application-only test flag is explicitly enabled.

    Disabled by default. A production APP_ENV or POINTCARE_ENV stays off unless
    POINTCARE_APPLICATION_TEST_MODE_ALLOW_PROD is also explicitly true, so a
    copied local flag cannot turn this on for unrestricted production calls.
    """
    src = os.environ if env is None else env

    def _flag(name: str) -> bool:
        return str(src.get(name, "")).strip().lower() in ("true", "1", "yes")

    if not _flag("POINTCARE_APPLICATION_TEST_MODE"):
        return False
    pointcare_env = (
        str(src.get("APP_ENV") or src.get("POINTCARE_ENV") or "development")
        .strip()
        .lower()
    )
    if pointcare_env in ("production", "prod") and not _flag(
        "POINTCARE_APPLICATION_TEST_MODE_ALLOW_PROD"
    ):
        return False
    return True


# Temporary Stage E application-only testing. Read once at import; tests patch this.
# When on and the Chrome profile is already signed in: skip greeting, spoken
# sign-in, logout, and Stage D; land on the saved program summary / application.
# Keep POINTCARE_APPLICATION_AFTER_SIGNIN false while using this flag.
# Default remains off so production runs the full intro → Stage D → Stage E flow.
APPLICATION_TEST_MODE = application_test_mode_from_env()


def application_after_signin_from_env(env: dict | None = None) -> bool:
    """True only when intro and sign-in should stay, and Stage D should not start.

    Disabled by default. Production stays off unless
    POINTCARE_APPLICATION_AFTER_SIGNIN_ALLOW_PROD is also explicitly true.
    Separate from APPLICATION_TEST_MODE (which also skips the intro). Prefer
    only one of these flags on at a time; TEST_MODE takes precedence in startup.
    """
    src = os.environ if env is None else env

    def _flag(name: str) -> bool:
        return str(src.get(name, "")).strip().lower() in ("true", "1", "yes")

    if not _flag("POINTCARE_APPLICATION_AFTER_SIGNIN"):
        return False
    pointcare_env = (
        str(src.get("APP_ENV") or src.get("POINTCARE_ENV") or "development")
        .strip()
        .lower()
    )
    if pointcare_env in ("production", "prod") and not _flag(
        "POINTCARE_APPLICATION_AFTER_SIGNIN_ALLOW_PROD"
    ):
        return False
    return True


APPLICATION_AFTER_SIGNIN = application_after_signin_from_env()

# Visible at process start so operators can confirm shortcuts are off for full calls.
import logging as _logging

_logging.getLogger(__name__).info(
    "pointcare_flow_flags APPLICATION_TEST_MODE=%s APPLICATION_AFTER_SIGNIN=%s "
    "(full call requires both false; restart after .env changes)",
    APPLICATION_TEST_MODE,
    APPLICATION_AFTER_SIGNIN,
)

MAX_STEPS = int(os.environ.get("MAX_STEPS", "25"))
PORT = int(os.environ.get("PORT", "8001"))
# Public HTTPS origin used in chrome_url / open_url (preferred over request Host).
PUBLIC_BASE_URL = (
    os.environ.get("PUBLIC_BASE_URL", "").strip()
    or os.environ.get("FORM_FILLER_PUBLIC_BASE_URL", "").strip()
).rstrip("/")
USE_SYSTEM_CHROME_PROFILE = os.environ.get("USE_SYSTEM_CHROME_PROFILE", "true").lower() in (
    "true",
    "1",
    "yes",
)


def _default_system_chrome_user_data() -> Path:
    if os.environ.get("CHROME_USER_DATA"):
        return Path(os.environ["CHROME_USER_DATA"]).expanduser()
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
    # Linux / macOS typical Chromium user-data roots.
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    return Path.home() / ".config" / "google-chrome"


def _default_agent_data_root() -> Path:
    if os.environ.get("CHROME_AGENT_DATA"):
        return Path(os.environ["CHROME_AGENT_DATA"]).expanduser()
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", str(ROOT))) / "PointcareAgentChrome"
    return Path.home() / ".cache" / "PointcareAgentChrome"


SYSTEM_CHROME_USER_DATA = _default_system_chrome_user_data()


def _last_chrome_profile() -> str:
    local_state = SYSTEM_CHROME_USER_DATA / "Local State"
    try:
        data = json.loads(local_state.read_text(encoding="utf-8"))
        last = (data.get("profile") or {}).get("last_used") or ""
        if last:
            return str(last)
    except Exception:
        pass
    return "Default"


def _split_chrome_profile_path(raw: str) -> tuple[Path, str] | None:
    """Accept a full profile folder path pasted in .env.

    Examples:
      C:\\Users\\...\\Google\\Chrome\\User Data\\Profile 2
      C:\\Users\\...\\Google\\Chrome\\User Data\\Default
    """
    text = (raw or "").strip().strip('"').strip("'")
    if not text:
        return None
    path = Path(text).expanduser()
    # If they pasted "...\\User Data", treat profile as Default / last used.
    if path.name.lower() == "user data":
        return path, (
            os.environ.get("CHROME_PROFILE_DIRECTORY", "").strip()
            or _last_chrome_profile()
        )
    # Normal case: "...\\User Data\\Profile 2"
    if path.parent.name.lower() == "user data":
        return path.parent, path.name
    # Already a user-data dir with an explicit profile name env.
    if (path / "Local State").exists():
        return path, (
            os.environ.get("CHROME_PROFILE_DIRECTORY", "").strip() or "Default"
        )
    return path.parent, path.name


_profile_from_path = _split_chrome_profile_path(os.environ.get("CHROME_PROFILE_PATH", ""))

if _profile_from_path:
    CHROME_PROFILE_DIR, CHROME_PROFILE_DIRECTORY = _profile_from_path
else:
    CHROME_PROFILE_DIR = Path(
        os.environ.get("CHROME_PROFILE_DIR")
        or (SYSTEM_CHROME_USER_DATA if USE_SYSTEM_CHROME_PROFILE else (ROOT / "chrome_profile"))
    )
    CHROME_PROFILE_DIRECTORY = (
        os.environ.get("CHROME_PROFILE_DIRECTORY", "").strip()
        or (_last_chrome_profile() if USE_SYSTEM_CHROME_PROFILE else "Default")
    )

CHROME_CLONE_DIR = Path(os.environ.get("CHROME_CLONE_DIR") or (ROOT / "chrome_profile_session"))

# Chrome 136+ refuses remote debugging on its default "User Data" folder, so the
# agent runs the profile from its own folder (seeded once from CHROME_PROFILE_PATH).
CHROME_AGENT_DATA_ROOT = _default_agent_data_root()
# One agent clone per source profile so changing CHROME_PROFILE_PATH actually switches.
_agent_profile_key = "".join(
    ch if ch.isalnum() or ch in ("-", "_") else "_"
    for ch in (CHROME_PROFILE_DIRECTORY or "Default")
).strip("_") or "Default"
CHROME_AGENT_DATA = CHROME_AGENT_DATA_ROOT / _agent_profile_key
CHROME_SOURCE_PROFILE_DIR = CHROME_PROFILE_DIR / CHROME_PROFILE_DIRECTORY


def chrome_profile_label(profile_directory: str = "") -> str:
    name = (profile_directory or CHROME_PROFILE_DIRECTORY).strip() or "Default"
    local_state = CHROME_PROFILE_DIR / "Local State"
    try:
        data = json.loads(local_state.read_text(encoding="utf-8"))
        info = ((data.get("profile") or {}).get("info_cache") or {}).get(name) or {}
        display = str(info.get("gaia_name") or info.get("user_name") or info.get("name") or "").strip()
        if display:
            return f"{name} ({display})"
    except Exception:
        pass
    return name
FORM_FILLER_WEBHOOK_URL = os.environ.get("FORM_FILLER_WEBHOOK_URL", "").strip()
RETELL_API_KEY = os.environ.get("RETELL_API_KEY", "").strip()
RETELL_FROM_NUMBER = (
    os.environ.get("RETELL_FROM_NUMBER", "").strip()
    or os.environ.get("FORM_FILLER_FROM_NUMBER", "").strip()
)
FORM_FILLER_AGENT_ID = os.environ.get("FORM_FILLER_AGENT_ID", "").strip()
RETELL_VOICE = os.environ.get("RETELL_VOICE", "").strip() or "retell-Cimo"
RETELL_MODEL = os.environ.get("RETELL_MODEL", "").strip()

# Custom LLM WebSocket (primary Collect Mode path). When enabled, the Pointcare
# Eligibility agent uses response_engine type custom-llm instead of retell-llm.
USE_CUSTOM_LLM = os.environ.get("USE_CUSTOM_LLM", "true").lower() in ("true", "1", "yes")
# Optional explicit URL. Otherwise derived from FORM_FILLER_WEBHOOK_URL host.
RETELL_LLM_WEBSOCKET_URL = os.environ.get("RETELL_LLM_WEBSOCKET_URL", "").strip()
# Transcript webhook may still log; only apply answers if this is true.
TRANSCRIPT_FALLBACK_APPLY = os.environ.get("TRANSCRIPT_FALLBACK_APPLY", "false").lower() in (
    "true",
    "1",
    "yes",
)


def retell_llm_websocket_url() -> str:
    if RETELL_LLM_WEBSOCKET_URL:
        return RETELL_LLM_WEBSOCKET_URL.rstrip("/")
    raw = FORM_FILLER_WEBHOOK_URL.strip()
    if not raw:
        return ""
    # https://host/.../webhooks/retell -> wss://host/llm-websocket
    from urllib.parse import urlparse

    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return ""
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/llm-websocket"


# Pointcare onboarding entry (welcome screens → Create My Profile / sign-in).
_DEFAULT_POINTCARE_URL = "https://my.pointcare.com/onboarding?code=YoTOXOvK"
POINTCARE_URL = os.environ.get("POINTCARE_URL", "").strip() or _DEFAULT_POINTCARE_URL
