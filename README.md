# bayhealth-pointcare-backend

## Pointcare Live Agent (`pc_liveagent`)

Collect Mode API: Retell Custom LLM + Playwright fills Pointcare during the live call.

Dependencies live in the root `requirements.txt` (same file as the rest of the backend).

### Install

```bat
pip install -r requirements.txt
playwright install chromium
```

### Run

From the repo root (with `.env` present):

```bat
python run.py
```

Listens on `http://127.0.0.1:8001` (override with `PORT` in `.env`).

Optional: open the agent Chrome window manually via `pc_liveagent\open_agent_chrome.bat`.

### Tests

```bat
python -m unittest pc_liveagent.test_collect_mode pc_liveagent.test_live_form pc_liveagent.test_browser_launch
```

### Layout

- `pc_liveagent/` — FastAPI Collect Mode (questions, Playwright, Retell WebSocket)
- `run.py` — uvicorn entrypoint (repo root)
- `deploy/nginx-pointcare-liveagent.conf.snippet` — Nginx locations for FastAPI (8001)

### Production (Ubuntu / EC2)

Required `.env` for the live-agent service:

```env
PORT=8001
PUBLIC_BASE_URL=https://bach.gapincare-backend.socialroots-dev.net
CHROME_CHANNEL=chrome
CHROME_EXECUTABLE=/usr/bin/google-chrome-stable
# Optional on locked-down hosts:
# CHROME_NO_SANDBOX=true
# HEADLESS=true
CHROME_AGENT_DATA=/home/ubuntu/.cache/PointcareAgentChrome
```

Nginx **must** proxy `/form-filler/`, `/webhooks/retell`, and `/llm-websocket/` to
`127.0.0.1:8001` (see `deploy/nginx-pointcare-liveagent.conf.snippet`). If those
paths hit Django/Gunicorn on `:8000`, `/form-filler/sessions/.../chrome` returns a
Django 404 even though FastAPI has the route.

Deploy / restart:

```bash
cd /home/ubuntu/bayhealth-pointcare-backend
git pull
source /home/ubuntu/venvs/pointcare-liveagent/bin/activate
pip install -r requirements.txt
sudo systemctl restart pointcare-liveagent.service
sudo nginx -t && sudo systemctl reload nginx   # after adding the snippet
curl -sS http://127.0.0.1:8001/ | head
curl -sS http://127.0.0.1:8001/openapi.json | head
```

Rollback: redeploy the previous git revision and `sudo systemctl restart pointcare-liveagent.service`.
