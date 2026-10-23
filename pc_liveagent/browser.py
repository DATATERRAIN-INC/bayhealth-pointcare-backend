from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.async_api import Browser, Page, async_playwright

from .config import (
    CHROME_AGENT_DATA,
    CHROME_CHANNEL,
    CHROME_EXECUTABLE,
    CHROME_NO_SANDBOX,
    CHROME_PROFILE_DIR,
    CHROME_PROFILE_DIRECTORY,
    CHROME_SOURCE_PROFILE_DIR,
    HEADLESS,
    SYSTEM_CHROME_USER_DATA,
    chrome_profile_label,
)

logger = logging.getLogger(__name__)

CDP_PORT = int(os.environ.get("CHROME_DEBUG_PORT", "9222"))
CDP_URL = f"http://127.0.0.1:{CDP_PORT}"

# If Chrome is already running without debugging, quit it and reopen with your profile.
CHROME_RESTART = os.environ.get("CHROME_RESTART", "true").lower() in ("true", "1", "yes")

_playwright = None
_browser: Browser | None = None
_lock = asyncio.Lock()


def _is_executable_file(path: Path) -> bool:
    """True when path is an existing executable file (not a directory)."""
    try:
        if not path.is_file():
            return False
        if sys.platform == "win32":
            # os.access(X_OK) is unreliable on Windows; accept common binaries.
            return path.suffix.lower() in {".exe", ".bat", ".cmd"} or os.access(
                path, os.X_OK
            )
        return os.access(path, os.X_OK)
    except OSError:
        return False


def _sessions_holding_pages() -> list[str]:
    """Call IDs that still hold a live Page on the shared CDP connection."""
    try:
        from . import live_form

        return live_form.session_ids_with_pages()
    except Exception:
        return []


def _chrome_candidates_windows() -> list[Path]:
    roots = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")),
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("LOCALAPPDATA", "")),
    ]
    out: list[Path] = []
    for root in roots:
        if not str(root):
            continue
        out.append(root / "Google" / "Chrome" / "Application" / "chrome.exe")
    return out


def _chrome_candidates_linux() -> list[Path]:
    """Ordered Google Chrome candidates for Ubuntu/Debian-style hosts."""
    out: list[Path] = []
    for name in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium"):
        found = shutil.which(name)
        if found:
            out.append(Path(found))
    out.extend(
        [
            Path("/usr/bin/google-chrome"),
            Path("/usr/bin/google-chrome-stable"),
            Path("/usr/bin/chromium-browser"),
            Path("/usr/bin/chromium"),
            Path("/snap/bin/chromium"),
        ]
    )
    return out


def _chrome_executable() -> str:
    """Resolve the Chrome binary for the current OS.

    Prefers ``CHROME_EXECUTABLE`` when set. For channel chrome/google-chrome,
    searches platform-specific install locations. Other ``CHROME_CHANNEL`` values
    (msedge, chromium, chrome-beta, …) remain Playwright channel names and are
    not resolved here — callers that need a path must set ``CHROME_EXECUTABLE``.
    """
    explicit = (
        (CHROME_EXECUTABLE or os.environ.get("CHROME_EXECUTABLE") or "")
        .strip()
        .strip('"')
        .strip("'")
    )
    if explicit:
        path = Path(explicit).expanduser()
        if _is_executable_file(path):
            logger.info("Using CHROME_EXECUTABLE=%s", path)
            return str(path)
        raise RuntimeError(
            f"CHROME_EXECUTABLE is set to {explicit!r} but that path is not an "
            "executable file. Install Chrome or correct CHROME_EXECUTABLE."
        )

    channel = (CHROME_CHANNEL or "chrome").strip().lower()
    # Backward compatible: empty / chrome / google-chrome → system Google Chrome.
    if channel not in ("chrome", "google-chrome", ""):
        # Non-path channels (e.g. msedge) are not resolved as filesystem paths.
        raise RuntimeError(
            f"CHROME_CHANNEL={CHROME_CHANNEL!r} is not a path-based Chrome install. "
            "Install Google Chrome, set CHROME_CHANNEL=chrome, or set "
            "CHROME_EXECUTABLE to the full browser binary path."
        )

    candidates = (
        _chrome_candidates_windows()
        if sys.platform == "win32"
        else _chrome_candidates_linux()
    )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _is_executable_file(candidate):
            logger.info("Resolved Chrome executable=%s platform=%s", candidate, sys.platform)
            return key

    searched = ", ".join(str(p) for p in candidates[:8]) or "(none)"
    logger.error(
        "Google Chrome executable not found platform=%s channel=%s searched=%s",
        sys.platform,
        CHROME_CHANNEL,
        searched,
    )
    raise RuntimeError(
        "Google Chrome was not found. On Linux install google-chrome-stable and "
        "ensure /usr/bin/google-chrome is on PATH, or set CHROME_EXECUTABLE to the "
        "full binary path. On Windows install Chrome or set CHROME_EXECUTABLE / "
        "CHROME_CHANNEL."
    )


def _cdp_up(url: str | None = None) -> bool:
    target = url or CDP_URL
    try:
        with urllib.request.urlopen(f"{target}/json/version", timeout=1) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _singleton_lock_path(user_data_dir: Path) -> Path:
    return user_data_dir / "SingletonLock"


def _profile_really_locked(user_data_dir: Path) -> bool:
    """True only when Chrome holds a live SingletonLock (ignore stale empty lockfile)."""
    return _singleton_lock_path(user_data_dir).exists()


def _chrome_processes_running() -> bool:
    if sys.platform != "win32":
        try:
            # Process name is usually "chrome" even when launched as google-chrome.
            out = subprocess.check_output(
                ["pgrep", "-f", "chrome"],
                stderr=subprocess.DEVNULL,
            )
            return bool(out.strip())
        except Exception:
            return False
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq chrome.exe", "/NH"],
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return "chrome.exe" in out.lower()
    except Exception:
        return False


def _runtime_user_data_dir() -> Path:
    """Folder Chrome is launched with. Never Chrome's default folder (debugging is blocked there)."""
    try:
        is_default = CHROME_PROFILE_DIR.resolve() == SYSTEM_CHROME_USER_DATA.resolve()
    except Exception:
        is_default = False
    return CHROME_AGENT_DATA if is_default else CHROME_PROFILE_DIR


def _uses_agent_clone(user_data_dir: Path) -> bool:
    try:
        return user_data_dir.resolve() == CHROME_AGENT_DATA.resolve()
    except Exception:
        return False


def _runtime_profile_directory(_user_data_dir: Path) -> str:
    """Folder name inside the agent Chrome directory.

    Keep the source name (Profile 2). Copying that profile into a folder named
    Default makes Chrome open a second empty profile, so a sign-in does not stick.
    """
    name = (CHROME_PROFILE_DIRECTORY or "").strip()
    return name or "Default"


def _seed_marker(user_data_dir: Path) -> Path:
    return user_data_dir / ".seeded_from"


def _seed_token() -> str:
    # Path only: a live Chrome profile's mtime changes constantly and would re-seed every call.
    return f"{CHROME_SOURCE_PROFILE_DIR}|{CHROME_PROFILE_DIRECTORY}"


def _read_os_crypt(local_state: Path) -> dict | None:
    """Chrome cookie key. A copied profile stays signed out if this key differs."""
    try:
        data = json.loads(local_state.read_text(encoding="utf-8"))
    except Exception:
        return None
    crypt = data.get("os_crypt") if isinstance(data, dict) else None
    if isinstance(crypt, dict) and crypt.get("encrypted_key"):
        return crypt
    return None


def _cookie_bytes(profile: Path) -> int:
    for rel in ("Network/Cookies", "Cookies"):
        path = profile / rel
        if path.is_file():
            try:
                return path.stat().st_size
            except OSError:
                return 0
    return 0


def _sync_os_crypt(user_data_dir: Path) -> None:
    """Keep the source profile's cookie key in the agent Chrome folder."""
    crypt = _read_os_crypt(CHROME_PROFILE_DIR / "Local State")
    if not crypt:
        return
    dest_state = user_data_dir / "Local State"
    try:
        data = json.loads(dest_state.read_text(encoding="utf-8")) if dest_state.exists() else {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if data.get("os_crypt") == crypt:
        return
    data["os_crypt"] = crypt
    user_data_dir.mkdir(parents=True, exist_ok=True)
    dest_state.write_text(json.dumps(data), encoding="utf-8")
    logger.info("Synced Chrome login key into %s", user_data_dir)


def _seed_agent_profile(user_data_dir: Path) -> None:
    """Copy the chosen Chrome profile into the agent folder once.

    A later call reuses that folder, including a sign-in done in the agent
    window. It does not delete the folder and it does not read the caller's
    normal Chrome while that window has the login files open.
    """
    if not _uses_agent_clone(user_data_dir):
        return
    source = CHROME_SOURCE_PROFILE_DIR
    if not source.is_dir():
        raise RuntimeError(
            f"Chrome profile folder not found: {source}. "
            "Check CHROME_PROFILE_PATH in .env."
        )
    dest = user_data_dir / _runtime_profile_directory(user_data_dir)
    marker = _seed_marker(user_data_dir)
    token = _seed_token()
    if dest.is_dir():
        logger.info("Reusing agent Chrome profile at %s", dest)
        try:
            if not marker.exists() or marker.read_text(encoding="utf-8").strip() != token:
                marker.write_text(token, encoding="utf-8")
        except OSError:
            logger.warning("Could not write Chrome seed marker at %s", marker)
        return
    user_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Seeding agent Chrome from %s into %s",
        source,
        dest,
    )
    if sys.platform == "win32":
        # Exclude bulky caches; keep logins, cookies, and Local Storage.
        cmd = [
            "robocopy",
            str(source),
            str(dest),
            "/E",
            "/NFL",
            "/NDL",
            "/NJH",
            "/NJS",
            "/NC",
            "/NS",
            "/R:0",
            "/W:0",
            "/XD",
            "Cache",
            "Code Cache",
            "GPUCache",
            "GrShaderCache",
            "ShaderCache",
            "Service Worker",
            "Storage",
            "/XF",
            "LOCK",
            "lockfile",
            "*.tmp",
        ]
        completed = subprocess.run(cmd, check=False)
        # robocopy codes 0-7 are success/partial copy; 8+ includes failed files.
        # Locked files (Login Data while Chrome is open) are common and non-fatal.
        if completed.returncode >= 16:
            raise RuntimeError(
                f"Failed to seed Chrome profile from {source} "
                f"(robocopy exit {completed.returncode})"
            )
        if completed.returncode >= 8:
            logger.warning(
                "Profile files were locked while copying from %s. "
                "The normal Chrome window was left open. Sign in once in the "
                "agent window; later calls reuse %s.",
                source,
                dest,
            )
            marker.write_text(token, encoding="utf-8")
            return
    else:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(
            source,
            dest,
            ignore=shutil.ignore_patterns(
                "Cache",
                "Code Cache",
                "GPUCache",
                "GrShaderCache",
                "ShaderCache",
                "Service Worker",
            ),
        )
    # Copied cookies only decrypt with the source key. A sign-in made later in
    # this window uses the agent key and must not be overwritten on the next call.
    if _cookie_bytes(dest) >= 50_000:
        _sync_os_crypt(user_data_dir)
    marker.write_text(token, encoding="utf-8")
    logger.info(
        "Agent Chrome ready with source profile %s (%s)",
        CHROME_PROFILE_DIRECTORY,
        chrome_profile_label(),
    )


def _cdp_belongs_to(user_data_dir: Path) -> bool:
    port = _devtools_port(user_data_dir)
    if not port:
        return False
    return _cdp_up(f"http://127.0.0.1:{port}")


def _kill_debug_chrome() -> None:
    """Stop any Chrome holding the agent debug port so a new profile can launch."""
    if sys.platform != "win32":
        subprocess.run(
            ["pkill", "-9", "-f", f"remote-debugging-port={CDP_PORT}"],
            check=False,
            timeout=8,
        )
        return
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
        f"Where-Object {{ $_.CommandLine -like '*remote-debugging-port={CDP_PORT}*' "
        "-or $_.CommandLine -like '*PointcareAgentChrome*' }} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        logger.warning("Timed out stopping debug Chrome; continuing launch")


def _kill_chrome_using(user_data_dir: Path) -> None:
    """Kill only the Chrome processes started with this --user-data-dir."""
    if sys.platform != "win32":
        subprocess.run(["pkill", "-9", "-f", str(user_data_dir)], check=False)
        return
    needle = str(user_data_dir).replace("'", "''")
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
        f"Where-Object {{ $_.CommandLine -like '*{needle}*' }} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        logger.warning("Timed out stopping Chrome for %s; continuing launch", user_data_dir)


def _chrome_using(user_data_dir: Path) -> bool:
    """True when a Chrome process was started with this --user-data-dir."""
    needle = str(user_data_dir)
    if sys.platform != "win32":
        try:
            out = subprocess.check_output(
                ["ps", "-eo", "args="],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in out.splitlines():
                low = line.lower()
                if "chrome" in low and needle in line:
                    return True
            return False
        except Exception:
            return False
    needle_ps = needle.replace("'", "''")
    script = (
        "@(Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
        f"Where-Object {{ $_.CommandLine -like '*{needle_ps}*' }}).Count"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", script],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return int((out or "0").strip() or 0) > 0
    except Exception:
        return False


def _devtools_port(user_data_dir: Path) -> int | None:
    path = user_data_dir / "DevToolsActivePort"
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").splitlines()[0].strip())
    except Exception:
        return None


def _launch_chrome(user_data_dir: Path, profile_directory: str, start_url: str = "") -> None:
    user_data_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale lock leftovers that block a fresh start after a normal quit.
    for name in ("SingletonLock", "SingletonSocket", "SingletonCookie", "lockfile"):
        path = user_data_dir / name
        try:
            if path.exists() or path.is_symlink():
                path.unlink(missing_ok=True)
        except Exception:
            pass

    exe = _chrome_executable()
    # Pass each flag as its own argv item (no PowerShell). Spaces in
    # "User Data" / "Profile 4" must stay inside one argument or Chrome
    # opens a fresh empty profile.
    # Bind CDP to localhost only — never expose the debugging port publicly.
    args = [
        exe,
        f"--remote-debugging-port={CDP_PORT}",
        "--remote-debugging-address=127.0.0.1",
        f"--user-data-dir={str(user_data_dir)}",
        f"--profile-directory={profile_directory}",
        "--no-first-run",
        "--no-default-browser-check",
        "--start-maximized",
        "--disable-session-crashed-bubble",
        "--hide-crash-restore-bubble",
        "--disable-blink-features=AutomationControlled",
        "--disable-features=FedCm,AutofillServerCommunication",
    ]
    if sys.platform != "win32":
        # Common on EC2/containers where /dev/shm is small.
        args.append("--disable-dev-shm-usage")
    if CHROME_NO_SANDBOX:
        # Opt-in only (some locked-down Linux hosts require it).
        args.append("--no-sandbox")
        logger.warning("Launching Chrome with --no-sandbox (CHROME_NO_SANDBOX=true)")
    if HEADLESS:
        args.append("--headless=new")
    args.append(start_url or "about:blank")
    logger.info(
        "Launching agent Chrome exe=%s folder=%s start_url=%s debug_port=%s",
        exe,
        user_data_dir,
        start_url or "about:blank",
        CDP_PORT,
    )
    kwargs: dict = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
        kwargs["close_fds"] = True
    subprocess.Popen(args, **kwargs)


async def _wait_for_cdp(user_data_dir: Path, timeout_s: float = 40) -> None:
    global CDP_URL
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if await asyncio.to_thread(_cdp_up, CDP_URL):
            return
        port = await asyncio.to_thread(_devtools_port, user_data_dir)
        if port:
            candidate = f"http://127.0.0.1:{port}"
            if await asyncio.to_thread(_cdp_up, candidate):
                CDP_URL = candidate
                return
        await asyncio.sleep(0.25)
    hint = (
        "Close the agent Chrome window (or run pc_liveagent/open_agent_chrome.bat), then retry."
        if sys.platform == "win32"
        else "Stop any stale agent Chrome on this host, confirm CHROME_EXECUTABLE, then retry."
    )
    raise RuntimeError(
        f"Chrome did not open remote debugging on port {CDP_PORT}. "
        f"Agent Chrome folder: {user_data_dir}. {hint}"
    )


async def _ensure_chrome_process(start_url: str = "") -> None:
    """Attach to the agent Chrome window, or open it. Your normal Chrome is never touched."""
    global CDP_URL

    user_data_dir = _runtime_user_data_dir()
    profile_directory = _runtime_profile_directory(user_data_dir)
    logger.info(
        "Requested Chrome source=%s agent_folder=%s profile=%s",
        CHROME_SOURCE_PROFILE_DIR,
        user_data_dir,
        chrome_profile_label(),
    )

    # If an old agent window is still on 9222, drop it when the env profile changed.
    if await asyncio.to_thread(_cdp_up):
        if await asyncio.to_thread(_cdp_belongs_to, user_data_dir):
            # User closed the window but a half-dead debug process can still answer
            # HTTP with no page targets — relaunch so Get Started → Sign in can run.
            if not await asyncio.to_thread(_cdp_has_page_targets):
                await _restart_agent_chrome(
                    user_data_dir,
                    profile_directory,
                    start_url,
                    reason="cdp_up_but_no_page_targets",
                )
                logger.info(
                    "Chrome ready (relaunch after closed window) cdp=%s", CDP_URL
                )
                return
            CDP_URL = f"http://127.0.0.1:{CDP_PORT}"
            port = await asyncio.to_thread(_devtools_port, user_data_dir)
            if port:
                CDP_URL = f"http://127.0.0.1:{port}"
            logger.info("Connected to existing Chrome CDP at %s", CDP_URL)
            return
        logger.warning(
            "Debug Chrome is open for a different profile; restarting for %s",
            chrome_profile_label(),
        )
        await _restart_agent_chrome(
            user_data_dir,
            profile_directory,
            start_url,
            reason="chrome_profile_switch",
        )
        return

    await asyncio.to_thread(_seed_agent_profile, user_data_dir)

    port = await asyncio.to_thread(_devtools_port, user_data_dir)
    if port:
        candidate = f"http://127.0.0.1:{port}"
        if await asyncio.to_thread(_cdp_up, candidate):
            CDP_URL = candidate
            logger.info("Attached to running Chrome via DevTools port %s", port)
            return

    if await asyncio.to_thread(_chrome_using, user_data_dir):
        if not CHROME_RESTART:
            raise RuntimeError(
                "The agent Chrome window is open without remote debugging. "
                "Close it or set CHROME_RESTART=true in .env."
            )
        logger.warning("Closing agent Chrome window that has no debugging port")
        await asyncio.to_thread(_kill_chrome_using, user_data_dir)
        await asyncio.sleep(1.5)

    _launch_chrome(user_data_dir, profile_directory, start_url)
    CDP_URL = f"http://127.0.0.1:{CDP_PORT}"
    await _wait_for_cdp(user_data_dir)
    logger.info(
        "Chrome ready source_profile=%s agent_profile=%s cdp=%s",
        chrome_profile_label(),
        profile_directory,
        CDP_URL,
    )


def _cdp_has_page_targets() -> bool:
    """True when the debug port lists at least one page target."""
    try:
        return any(
            isinstance(t, dict) and str(t.get("type") or "") == "page"
            for t in cdp_list_targets()
        )
    except Exception:
        return False


def _browser_looks_alive(browser: Browser) -> bool:
    """False after the user closes Chrome or the CDP socket dies."""
    try:
        if hasattr(browser, "is_connected") and not browser.is_connected():
            return False
        _ = browser.contexts
        return True
    except Exception:
        return False


async def _restart_agent_chrome(
    user_data_dir: Path,
    profile_directory: str,
    start_url: str,
    *,
    reason: str,
) -> None:
    """Kill the agent debug Chrome and launch a fresh maximized window."""
    global CDP_URL
    logger.warning("Restarting agent Chrome reason=%s folder=%s", reason, user_data_dir)
    try:
        from . import live_form

        await live_form.detach_all_pages_for_browser_restart(reason=reason)
    except Exception:
        logger.exception("detach_all_pages_for_browser_restart failed reason=%s", reason)
    await asyncio.to_thread(_kill_debug_chrome)
    await asyncio.to_thread(_kill_chrome_using, user_data_dir)
    await _disconnect_playwright(force=True, reason=reason)
    await asyncio.sleep(1.0)
    await asyncio.to_thread(_seed_agent_profile, user_data_dir)
    _launch_chrome(user_data_dir, profile_directory, start_url)
    CDP_URL = f"http://127.0.0.1:{CDP_PORT}"
    await _wait_for_cdp(user_data_dir)


async def _connect() -> Browser:
    global _playwright, _browser
    if _browser is not None:
        if _browser_looks_alive(_browser):
            return _browser
        logger.warning("Existing Playwright CDP browser unusable; reconnecting")
        await _disconnect_playwright(force=True, reason="stale_browser")
    if not await asyncio.to_thread(_cdp_up, CDP_URL):
        raise RuntimeError(f"Chrome CDP is not reachable at {CDP_URL}")
    logger.info("Starting Playwright CDP connection to %s", CDP_URL)
    _playwright = await async_playwright().start()
    try:
        _browser = await _playwright.chromium.connect_over_cdp(CDP_URL)
    except Exception:
        await _disconnect_playwright(force=True, reason="cdp_connect_failed")
        raise
    if not _browser_looks_alive(_browser):
        await _disconnect_playwright(force=True, reason="cdp_connected_dead")
        raise RuntimeError("Connected to Chrome CDP but the browser is already closed")
    logger.info(
        "Playwright connected over CDP %s contexts=%s",
        CDP_URL,
        len(_browser.contexts),
    )
    return _browser


async def _page_from_browser(browser: Browser) -> Page:
    """Reuse the default CDP context/tab. Never call new_context over CDP.

    ``connect_over_cdp`` rejects ``new_context()`` once the window was closed
    (TargetClosedError). Prefer an existing page, else open a tab.
    """
    contexts = list(browser.contexts or [])
    if not contexts:
        raise RuntimeError(
            "CDP browser has no contexts (Chrome window was closed). Restart required."
        )
    context = contexts[0]
    live = [p for p in context.pages if not p.is_closed()]
    if live:
        return live[0]
    return await context.new_page()


async def _disconnect_playwright(*, force: bool = False, reason: str = "") -> None:
    """Stop the shared Playwright CDP client.

    Does not kill the Chrome process. Refuses to disconnect while sessions still
    hold Page objects unless ``force=True`` (shutdown / profile restart after detach).
    """
    global _playwright, _browser
    owners = _sessions_holding_pages()
    if owners and not force:
        logger.warning(
            "Skip Playwright disconnect; sessions still hold pages owners=%s reason=%s",
            owners[:8],
            reason or "-",
        )
        return
    logger.info(
        "Disconnecting Playwright CDP force=%s reason=%s owners=%s",
        force,
        reason or "-",
        owners[:8] if owners else [],
    )
    _browser = None
    if _playwright is not None:
        try:
            await _playwright.stop()
        except Exception:
            logger.exception("Playwright stop failed reason=%s", reason or "-")
        _playwright = None


def is_playwright_closed_error(exc: BaseException) -> bool:
    """True when Playwright reports a closed page/browser/connection handler."""
    msg = str(exc).lower()
    return any(
        needle in msg
        for needle in (
            "handler is closed",
            "target closed",
            "target page, context or browser has been closed",
            "browser has been closed",
            "connection closed",
            "protocol error",
        )
    )


async def _maximize(page: Page) -> None:
    try:
        session = await page.context.new_cdp_session(page)
        info = await session.send("Browser.getWindowForTarget")
        await session.send(
            "Browser.setWindowBounds",
            {"windowId": info["windowId"], "bounds": {"windowState": "maximized"}},
        )
    except Exception:
        try:
            await page.evaluate(
                "() => { window.moveTo(0, 0); window.resizeTo(screen.availWidth, screen.availHeight); }"
            )
        except Exception:
            pass


def _already_on(page_url: str, target: str) -> bool:
    have = (page_url or "").split("#")[0].rstrip("/")
    want = (target or "").split("#")[0].rstrip("/")
    if not want:
        return True
    return have == want or have.startswith(want) or want in have


async def _open_page_once(url: str | None) -> Page:
    """Attach to agent Chrome, open/navigate one tab, maximize the window."""
    await _ensure_chrome_process(url or "")
    browser = await _connect()
    page = await _page_from_browser(browser)
    await page.bring_to_front()
    if not HEADLESS:
        await _maximize(page)
    if url and not _already_on(page.url, url):
        logger.info("Navigating existing tab from %s to %s", page.url, url)
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                logger.warning("goto attempt %s failed: %s", attempt + 1, exc)
                if is_playwright_closed_error(exc):
                    break
                try:
                    await page.wait_for_timeout(800)
                except Exception:
                    break
        if last_exc:
            if is_playwright_closed_error(last_exc):
                raise last_exc
            try:
                live = [item for item in page.context.pages if not item.is_closed()]
            except Exception:
                live = []
            if not live:
                raise last_exc
            page = live[-1]
            logger.warning(
                "Page navigation failed (%s); using open tab %s", last_exc, page.url
            )
            try:
                await page.wait_for_timeout(800)
            except Exception:
                pass
    logger.info("Chrome tab is on %s", page.url)
    return page


async def open_page(url: str | None = None) -> Page:
    """Use the visible tab in your Chrome profile and open the target URL there.

    If the user closed the agent Chrome window, CDP may still answer with no
    usable context. Detect that, relaunch a maximized window, and retry once.
    """
    async with _lock:
        try:
            return await _open_page_once(url)
        except Exception as exc:
            if not (
                is_playwright_closed_error(exc)
                or "no contexts" in str(exc).lower()
                or "already closed" in str(exc).lower()
            ):
                raise
            user_data_dir = _runtime_user_data_dir()
            profile_directory = _runtime_profile_directory(user_data_dir)
            logger.warning(
                "open_page recovered from closed Chrome err=%s; relaunching",
                exc,
            )
            await _restart_agent_chrome(
                user_data_dir,
                profile_directory,
                url or "",
                reason="open_page_closed_browser",
            )
            return await _open_page_once(url)


def cdp_list_targets() -> list[dict]:
    """Return Chrome CDP page targets from the agent debugging port."""
    try:
        with urllib.request.urlopen(f"{CDP_URL}/json/list", timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def cdp_target_for_url(page_url: str = "") -> dict:
    """Pick the best CDP page target for the live Pointcare tab."""
    targets = [
        t
        for t in cdp_list_targets()
        if isinstance(t, dict) and str(t.get("type") or "") == "page"
    ]
    if not targets:
        return {}
    want = (page_url or "").split("#")[0].rstrip("/").lower()
    if want:
        for t in targets:
            have = str(t.get("url") or "").split("#")[0].rstrip("/").lower()
            if have == want or want in have or have in want:
                return t
        for t in targets:
            if "pointcare" in str(t.get("url") or "").lower():
                return t
    return targets[0]


def local_devtools_url(target: dict) -> str:
    """Clickable DevTools URL for the agent Chrome tab (same machine)."""
    if not target:
        return ""
    target_id = str(target.get("id") or "").strip()
    ws = str(target.get("webSocketDebuggerUrl") or "").strip()
    if target_id:
        return (
            f"{CDP_URL}/devtools/inspector.html"
            f"?ws=127.0.0.1:{CDP_PORT}/devtools/page/{target_id}"
        )
    if ws.startswith("ws://"):
        return f"{CDP_URL}/devtools/inspector.html?ws={ws[5:]}"
    frontend = str(target.get("devtoolsFrontendUrl") or "").strip()
    if frontend.startswith("http"):
        return frontend
    if frontend.startswith("/"):
        return f"{CDP_URL}{frontend}"
    return frontend


async def close_browser() -> None:
    """App shutdown: detach session pages, then drop the shared Playwright client."""
    try:
        from . import live_form

        await live_form.detach_all_pages_for_browser_restart(reason="app_shutdown")
    except Exception:
        logger.exception("detach on close_browser failed")
    # Disconnect automation only. Do not kill the Chrome window.
    await _disconnect_playwright(force=True, reason="app_shutdown")
