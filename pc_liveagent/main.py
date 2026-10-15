import asyncio
import logging
import re
import sys
import time

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config
from .browser import close_browser
from .form_bridge import (
    enrich_webhook_call,
    event_kind,
    handle_submit_field_tool,
    latest_question_and_answer,
    on_caller_transcript,
    on_call_ended,
    on_call_started,
    parse_tool_args,
    target_url_from_call,
    transcript_from_call,
)
from .live_form import ELIG_QUESTIONS, describe_live_browser, get_session
from .retell_call import place_form_filler_call
from .retell_llm_ws import handle_retell_llm_websocket

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logger = logging.getLogger(__name__)

app = FastAPI(title="Pointcare Live Agent")


@app.on_event("shutdown")
async def _shutdown_browser() -> None:
    await close_browser()


class FormFillerCallIn(BaseModel):
    phone_number: str = Field(..., min_length=10)
    name: str = ""
    form_name: str = ""
    agent_id: str = ""
    form_url: str | None = None


class WelcomeTestIn(BaseModel):
    form_url: str | None = None


INDEX = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pointcare Live Agent</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 44rem; margin: 2rem auto; padding: 0 1rem; }
    code, pre { background: #f7f7f7; }
    pre { padding: 0.75rem; overflow: auto; }
  </style>
</head>
<body>
  <h1>Pointcare Live Agent</h1>
  <p>Collect Mode: Retell Custom LLM + Playwright fills Pointcare in real time.</p>
  <p>Start with <code>venv313\\Scripts\\python.exe run.py</code> (port 8001).</p>
  <ul>
    <li><code>POST /form-filler/outbound/</code> — place a live call</li>
    <li><code>GET /form-filler/sessions/{{call_id}}</code> — live session + Chrome URL</li>
    <li><code>GET /form-filler/sessions/{{call_id}}/chrome</code> — open/focus agent Chrome</li>
    <li><code>WS /llm-websocket/{call_id}</code> — Retell Custom LLM</li>
    <li><code>POST /webhooks/retell</code> and <code>/webhooks/retell/tool</code></li>
  </ul>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX


def _request_base_url(request: Request) -> str:
    """Public base for clickable open_url / chrome_url links.

    Prefer ``PUBLIC_BASE_URL`` (production HTTPS origin) so links stay on the
    FastAPI/nginx path even when Host headers vary. Fall back to forwarded
    host when running behind a tunnel or local proxy.
    """
    configured = (getattr(config, "PUBLIC_BASE_URL", None) or "").strip().rstrip("/")
    if configured:
        return configured
    # Prefer forwarded host when running behind a tunnel / proxy.
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme or "http").split(",")[0].strip()
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").split(",")[0].strip()
    if host:
        return f"{proto}://{host}"
    return str(request.base_url).rstrip("/")


def _chrome_session_url(base: str, call_id: str) -> str:
    cid = (call_id or "").strip()
    if not cid:
        return ""
    return f"{base.rstrip('/')}/form-filler/sessions/{cid}/chrome"


@app.post("/form-filler/outbound/")
async def form_filler_outbound(
    payload: FormFillerCallIn,
    background_tasks: BackgroundTasks,
    request: Request,
):
    result = await asyncio.to_thread(
        place_form_filler_call,
        phone_number=payload.phone_number,
        name=payload.name,
        form_name=payload.form_name,
        agent_id=payload.agent_id,
    )
    if not result.get("ok"):
        raise HTTPException(
            status_code=int(result.get("status_code") or 502),
            detail=result.get("error") or "Failed to place call.",
        )
    call_id = str(result.get("call_id") or "")
    form_url = (payload.form_url or "").strip() or config.POINTCARE_URL
    base = _request_base_url(request)
    chrome_link = _chrome_session_url(base, call_id)
    # Open Chrome and auto-click the welcome pages before/while the call connects.
    if call_id and form_url:
        try:
            started = await on_call_started(call_id, form_url)
            # Rebuild open_url against the public base so Postman/ops can click it.
            live = await get_session(call_id)
            browser = await describe_live_browser(
                live, call_id=call_id, open_base_url=base
            )
            # Never claim success when the browser snapshot says otherwise.
            if not browser.get("opened"):
                browser["opened"] = False
                browser["error"] = (
                    browser.get("error")
                    or (started.get("error") if isinstance(started, dict) else "")
                    or "Browser session did not open."
                )
            result["browser"] = browser
            result["chrome_url"] = browser.get("open_url") or chrome_link
        except Exception as exc:
            logger.exception("Failed to open Pointcare for call_id=%s", call_id)
            # Safe public error — no filesystem paths or secrets.
            safe = "Browser startup failed. Check server logs for details."
            err_text = str(exc)
            if "Chrome was not found" in err_text or "CHROME_EXECUTABLE" in err_text:
                safe = (
                    "Google Chrome was not found on this host. "
                    "Install google-chrome-stable or set CHROME_EXECUTABLE."
                )
            elif "remote debugging" in err_text.lower():
                safe = "Chrome started but remote debugging did not become ready."
            result["browser"] = {
                "opened": False,
                "error": safe,
                "open_url": chrome_link,
            }
            result["chrome_url"] = chrome_link
            background_tasks.add_task(on_call_started, call_id, form_url)
    return result


@app.post("/form-filler/welcome/")
async def form_filler_welcome(
    request: Request, payload: WelcomeTestIn | None = None
):
    """Dev helper: open Pointcare and auto-click welcome pages to the sign-in screen."""
    call_id = f"welcome_test_{int(time.time())}"
    form_url = config.POINTCARE_URL
    if payload and (payload.form_url or "").strip():
        form_url = payload.form_url.strip()
    try:
        session = await on_call_started(call_id, form_url)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    live = await get_session(call_id)
    browser = await describe_live_browser(
        live, call_id=call_id, open_base_url=_request_base_url(request)
    )
    return {
        "ok": True,
        **session,
        "call_id": call_id,
        "chrome_profile": config.CHROME_PROFILE_DIRECTORY,
        "pointcare_url": form_url,
        "browser": browser,
        "chrome_url": browser.get("open_url") or browser.get("url") or "",
    }


@app.get("/form-filler/sessions/{call_id}")
async def form_filler_session(call_id: str, request: Request):
    session = await get_session(call_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "session_not_found",
                "message": "No live Pointcare Live Agent session for that call_id.",
                "service": "pointcare-liveagent",
                "call_id": call_id,
            },
        )
    base = _request_base_url(request)
    browser = await describe_live_browser(
        session, call_id=call_id, open_base_url=base
    )
    # Strip internal filesystem paths from the public API payload.
    public_browser = {
        k: v
        for k, v in browser.items()
        if k not in ("chrome_user_data", "cdp_url")
    }
    return {
        "call_id": session.call_id,
        "url": browser.get("url") or session.url,
        "status": session.status,
        "welcome_done": session.welcome_done,
        "filled": session.filled,
        "history": session.history[-40:],
        "error": session.error,
        "ready": session.ready.is_set(),
        "signed_in": session.signed_in,
        "eligibility_step": session.elig_step,
        "browser": public_browser,
        "chrome_url": browser.get("open_url") or _chrome_session_url(base, call_id),
        "service": "pointcare-liveagent",
    }


@app.get("/form-filler/sessions/{call_id}/chrome", response_class=HTMLResponse)
async def form_filler_open_chrome(call_id: str, request: Request):
    """Clickable page: focus agent Chrome and show live URL + DevTools link.

    Distinctive HTML so operators can tell FastAPI handled the request
    (vs a Django/nginx catch-all 404).
    """
    session = await get_session(call_id)
    if session is None:
        # Distinctive HTML 404 — if the public URL shows Django's page instead,
        # Nginx is routing /form-filler/ to Django (8000), not FastAPI (8001).
        return HTMLResponse(
            status_code=404,
            content=f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Session not found</title></head>
<body style="font-family:system-ui;max-width:40rem;margin:2rem auto;padding:0 1rem">
  <h1>Pointcare Live Agent — session not found</h1>
  <p>No active browser session for call <code>{call_id}</code>.</p>
  <p>This response is from <strong>pointcare-liveagent</strong> (FastAPI).
     If you instead see a Django 404 page at this URL, Nginx is not proxying
     <code>/form-filler/</code> to the live-agent service on port 8001.</p>
</body>
</html>""",
        )
    opened = False
    if session.page is not None:
        try:
            await session.page.bring_to_front()
            opened = True
        except Exception:
            logger.exception("bring_to_front failed call_id=%s", call_id)
    browser = await describe_live_browser(
        session, call_id=call_id, open_base_url=_request_base_url(request)
    )
    live = browser.get("url") or session.url or ""
    # DevTools is localhost-only; useful on the EC2 host, not via public HTTPS.
    devtools = browser.get("devtools_url") or ""
    profile = browser.get("chrome_profile") or ""
    step = session.elig_step or ""
    status = "running" if (opened or browser.get("opened")) else "session present (browser not attached)"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pointcare Live Agent — {call_id}</title>
  <meta http-equiv="refresh" content="5">
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 42rem; margin: 2rem auto; padding: 0 1rem; }}
    a.btn {{ display: inline-block; margin: 0.4rem 0.4rem 0.4rem 0; padding: 0.65rem 1rem;
             background: #0b57d0; color: #fff; text-decoration: none; border-radius: 8px; }}
    a.secondary {{ background: #444; }}
    code, pre {{ background: #f4f4f4; padding: 0.15rem 0.35rem; border-radius: 4px; }}
    .meta {{ color: #444; font-size: 0.95rem; }}
  </style>
</head>
<body>
  <h1>Pointcare Live Agent — Chrome session</h1>
  <p class="meta">service <code>pointcare-liveagent</code> · status <code>{status}</code></p>
  <p class="meta">Call <code>{call_id}</code> · profile <code>{profile}</code>
     · step <code>{step or "-"}</code></p>
  <p>Live tab URL:</p>
  <p><a href="{live}">{live or "(unavailable)"}</a></p>
  <p>
    <a class="btn" href="{live or "#"}" target="_blank" rel="noreferrer">Open live URL</a>
    <a class="btn secondary" href="{devtools or "#"}" target="_blank" rel="noreferrer">Open Chrome DevTools (localhost)</a>
  </p>
  <p class="meta">On the application host, the automation window was brought to the front when possible.
     DevTools only works from that host (CDP is bound to 127.0.0.1).</p>
  <p class="meta">This page refreshes every 5s so the URL stays current.</p>
</body>
</html>
"""


@app.websocket("/llm-websocket/{call_id}")
async def retell_llm_websocket(websocket: WebSocket, call_id: str):
    """Retell Custom LLM primary path: server owns every spoken turn."""
    await handle_retell_llm_websocket(websocket, call_id)


@app.post("/webhooks/retell/tool")
async def retell_form_tool(request: Request):
    """Retell custom function: submit_field → Playwright fill on Pointcare."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object")

    call = body.get("call") if isinstance(body.get("call"), dict) else {}
    args = parse_tool_args(body)
    call_id = str(call.get("call_id") or call.get("id") or body.get("call_id") or "").strip()
    tool_call = body.get("tool_call") if isinstance(body.get("tool_call"), dict) else {}
    request_id = str(
        body.get("tool_call_id")
        or tool_call.get("tool_call_id")
        or tool_call.get("id")
        or ""
    ).strip()
    name = str(body.get("name") or body.get("function_name") or "submit_field").strip()
    form_url = target_url_from_call(call) if call else config.POINTCARE_URL
    logger.info(
        "Retell tool name=%s call_id=%s args_keys=%s field=%r value=%r",
        name,
        call_id,
        list(args.keys()),
        str(args.get("field_id") or args.get("field") or "")[:40],
        str(args.get("value") or args.get("answer") or "")[:80],
    )

    if name and name not in ("submit_field", "custom"):
        # Still attempt fill if field_id is present.
        if "field_id" not in args and "field" not in args:
            return {"result": f"Unknown tool {name}. Use submit_field."}

    if not call_id:
        return {"result": "Missing call_id; cannot fill Pointcare."}

    field_id = str(args.get("field_id") or args.get("field") or "").strip()
    if field_id == "eligibility_answer":
        # Live tool webhooks often have an empty transcript. Trust short clear
        # answers (Yes/No/marital options) from the tool args; only reject when
        # the transcript is present and clearly disagrees.
        submitted = str(args.get("value") or args.get("answer") or "").strip()
        transcript = transcript_from_call(call)
        _question, latest_caller = latest_question_and_answer(call, transcript)
        norm_latest = " ".join(re.findall(r"[a-z0-9]+", latest_caller.lower()))
        norm_submitted = " ".join(re.findall(r"[a-z0-9]+", submitted.lower()))
        clear_short = bool(
            re.fullmatch(
                r"(yes|no|yeah|yep|yup|nope|nah|never married|married|separated|"
                r"divorced|widowed|single)",
                norm_submitted,
            )
        )
        matches_caller = bool(
            norm_latest
            and norm_submitted
            and (
                norm_latest == norm_submitted
                or norm_latest in norm_submitted
                or norm_submitted in norm_latest
            )
        )
        if norm_latest and not matches_caller and not clear_short:
            session = await get_session(call_id)
            question = (
                ELIG_QUESTIONS.get(session.elig_step, "Please answer the current question.")
                if session
                else "Please answer the current question."
            )
            logger.warning(
                "Rejected inferred eligibility answer call_id=%s submitted=%r latest_caller=%r",
                call_id,
                submitted[:80],
                latest_caller[:80],
            )
            return {"result": question}

    spoken = await handle_submit_field_tool(
        call_id, args, form_url, request_id=request_id
    )
    logger.info(
        "Retell tool done call_id=%s request_id=%s field=%s result=%r",
        call_id,
        request_id or "-",
        args.get("field_id") or args.get("field") or "-",
        spoken[:160],
    )
    return {"result": spoken}


@app.post("/webhooks/retell")
async def retell_webhook(request: Request, background_tasks: BackgroundTasks):
    """Open/close the Pointcare browser session around the Retell call."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object")

    event = str(body.get("event") or body.get("name") or body.get("type") or "").strip()
    call = body.get("call") if isinstance(body.get("call"), dict) else body
    if isinstance(body.get("data"), dict) and not str((call or {}).get("call_id") or ""):
        call = body["data"]
    if not isinstance(call, dict):
        call = body

    call = enrich_webhook_call(body, call if isinstance(call, dict) else {})
    call_id = str(call.get("call_id") or call.get("id") or "").strip()
    transcript = transcript_from_call(call)
    form_url = target_url_from_call(call)
    kind = event_kind(event)
    twtc = call.get("transcript_with_tool_calls")
    tobj = call.get("transcript_object")
    logger.info(
        "Retell webhook event=%s kind=%s call_id=%s transcript_chars=%s twtc=%s tobj=%s body_keys=%s",
        event,
        kind,
        call_id,
        len(transcript),
        len(twtc) if isinstance(twtc, list) else 0,
        len(tobj) if isinstance(tobj, list) else 0,
        sorted(str(k) for k in body.keys())[:20],
    )
    if not call_id:
        return {"ok": True, "queued": False, "event": event}

    if kind == "start":
        background_tasks.add_task(on_call_started, call_id, form_url)
        return {
            "ok": True,
            "queued": True,
            "event": event,
            "call_id": call_id,
            "action": "open_pointcare",
            "url": form_url,
        }

    if kind == "transcript":
        background_tasks.add_task(on_caller_transcript, call_id, call, form_url)
        return {
            "ok": True,
            "queued": True,
            "event": event,
            "call_id": call_id,
            "action": "log_caller_words",
            "transcript_chars": len(transcript),
        }

    if kind == "ended":
        background_tasks.add_task(on_call_ended, call_id, call, form_url)
        return {
            "ok": True,
            "queued": True,
            "event": event,
            "call_id": call_id,
            "action": "end_session",
            "transcript_chars": len(transcript),
        }

    return {
        "ok": True,
        "queued": False,
        "event": event,
        "call_id": call_id,
        "action": "pointcare_live_fill",
        "transcript_chars": len(transcript),
    }
