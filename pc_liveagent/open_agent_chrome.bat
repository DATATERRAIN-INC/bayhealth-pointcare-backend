@echo off
rem Opens the Chrome window the voice agent controls (separate from your normal Chrome).
rem Uses CHROME_PROFILE_PATH from .env via a short Python helper.
cd /d "%~dp0.."
for /f "usebackq delims=" %%I in (`venv313\Scripts\python.exe -c "from pc_liveagent.config import CHROME_AGENT_DATA, POINTCARE_URL; print(CHROME_AGENT_DATA); print(POINTCARE_URL)"`) do (
  if not defined AGENT_DIR (
    set "AGENT_DIR=%%I"
  ) else (
    set "START_URL=%%I"
  )
)
if not defined AGENT_DIR (
  echo Could not resolve agent Chrome folder from .env
  exit /b 1
)
echo Opening agent Chrome at %AGENT_DIR%
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" ^
  --remote-debugging-port=9222 ^
  --user-data-dir="%AGENT_DIR%" ^
  --profile-directory=Default ^
  --no-first-run --no-default-browser-check --start-maximized ^
  --disable-blink-features=AutomationControlled ^
  --disable-features=FedCm,AutofillServerCommunication ^
  "%START_URL%"
