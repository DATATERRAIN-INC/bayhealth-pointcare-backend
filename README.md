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
python -m unittest pc_liveagent.test_collect_mode pc_liveagent.test_live_form
```

### Layout

- `pc_liveagent/` — FastAPI Collect Mode (questions, Playwright, Retell WebSocket)
- `run.py` — uvicorn entrypoint (repo root)
