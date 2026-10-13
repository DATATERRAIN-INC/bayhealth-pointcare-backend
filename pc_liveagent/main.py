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
from .live_form import ELIG_QUESTIONS, get_session
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
    <li><code>WS /llm-websocket/{call_id}</code> — Retell Custom LLM</li>
    <li><code>POST /webhooks/retell</code> and <code>/webhooks/retell/tool</code></li>
  </ul>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX


@app.post("/form-filler/outbound/")
async def form_filler_outbound(payload: FormFillerCallIn, background_tasks: BackgroundTasks):
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
    # Open Chrome and auto-click the welcome pages before/while the call connects.
    if call_id and form_url:
        try:
            session = await on_call_started(call_id, form_url)
            result["browser"] = {
                "opened": True,
                "welcome_done": bool(session.get("welcome_done")),
                "url": session.get("url"),
                "error": session.get("error") or "",
            }
        except Exception as exc:
            logger.exception("Failed to open Pointcare for call_id=%s", call_id)
            result["browser"] = {"opened": False, "error": str(exc)}
            background_tasks.add_task(on_call_started, call_id, form_url)
    return result


@app.post("/form-filler/welcome/")
async def form_filler_welcome(payload: WelcomeTestIn | None = None):
    """Dev helper: open Pointcare and auto-click welcome pages to the sign-in screen."""
    call_id = f"welcome_test_{int(time.time())}"
    form_url = config.POINTCARE_URL
    if payload and (payload.form_url or "").strip():
        form_url = payload.form_url.strip()
    try:
        session = await on_call_started(call_id, form_url)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        **session,
        "call_id": call_id,
        "chrome_profile": config.CHROME_PROFILE_DIRECTORY,
        "pointcare_url": form_url,
    }


@app.get("/form-filler/sessions/{call_id}")
async def form_filler_session(call_id: str):
    session = await get_session(call_id)
    if session is None:
        raise HTTPException(status_code=404, detail="No live session for that call_id")
    return {
        "call_id": session.call_id,
        "url": session.url,
        "status": session.status,
        "welcome_done": session.welcome_done,
        "filled": session.filled,
        "history": session.history[-40:],
        "error": session.error,
        "ready": session.ready.is_set(),
        "signed_in": session.signed_in,
        "eligibility_step": session.elig_step,
    }


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
