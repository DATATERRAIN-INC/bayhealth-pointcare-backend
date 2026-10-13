# -*- coding: utf-8 -*-
"""
Medi-Cal Renewal & Retention call flow (BACH).

Business logic lives in Retell LLM states + edges.
Backend custom functions persist structured variables.
"""
from __future__ import annotations

from typing import Any, Dict, List

DEFAULT_AGENT_NAME = "Lauren"
ORG_NAME = "Bay Area Community Health"

NODE_OPENING = "opening_identity"
NODE_LANGUAGE = "language_preference"
NODE_PURPOSE = "purpose"
NODE_CONTACT = "contact_information"
NODE_RENEWAL = "renewal_status"
NODE_RENEWAL_OFFER = "renewal_followup_offer"
NODE_POINTCARE = "pointcare_registration"
NODE_CHANGES = "changes_check"
NODE_HOUSEHOLD_INCOME = "household_income"
NODE_WORK_EXEMPTION = "work_exemption"
NODE_SUPPORT = "additional_support"
NODE_DOCS = "document_readiness"
NODE_BENEFITS = "additional_benefits"
NODE_APPOINTMENT = "healthcare_appointment"
NODE_ASSISTANCE = "offer_renewal_assistance"
NODE_OUTCOME = "final_outcome"
NODE_CLOSING = "closing"
NODE_TRANSFER = "agent_transfer"
NODE_CALLBACK = "callback_end"

STARTING_STATE = NODE_OPENING

# Path flags passed through Retell dynamic variables via edge parameters.
FLOW_PATH_YES_NO_CHANGES = "renewal_yes_no_changes"
FLOW_PATH_YES_WITH_CHANGES = "renewal_yes_with_changes"
FLOW_PATH_NO = "renewal_no"

RENEWAL_STATUS_VALUES = (
    "COMPLETED",
    "SUBMITTED_PENDING",
    "NOT_COMPLETED",
    "NOT_RECEIVED",
)

FINAL_OUTCOMES = (
    "RENEWAL_COMPLETED",
    "RENEWAL_SUBMITTED",
    "ASSISTANCE_PROVIDED",
    "ENROLLMENT_TEAM_NEEDED",
    "APPOINTMENT_SCHEDULED",
    "FOLLOW_UP_REQUIRED",
    "PATIENT_DECLINED",
    "RENEWAL_VIDEO_SENT",
    "POINTCARE_GUIDED",
)


def begin_message(patient_name: str) -> str:
    who = (patient_name or "").strip() or "{{patient_name}}"
    return f"Hello, may I speak with {who}?"


def general_prompt(*, agent_name: str = DEFAULT_AGENT_NAME) -> str:
    agent = (agent_name or "").strip() or DEFAULT_AGENT_NAME
    return f"""
You are {agent}, a friendly enrollment specialist from {ORG_NAME} (BACH).
You are on an outbound Medi-Cal renewal & retention call with {{{{patient_name}}}}.

BRANCHING FLOW (follow edges exactly — do not invent skips):
01 opening_identity → 01b language_preference → 02 purpose → 03 contact_information → 04 renewal_status
04 YES → 04a changes_check
  - no changes → 09 benefits (skip household/work/support/docs)
  - changes → 05 household_income → 09 benefits (skip work/support/docs)
04 NO → 04b renewal_followup_offer (THREE options — ask exactly once):
  - video texted → send_renewal_video_sms → briefly confirm → 05 household_income → …
  - Pointcare registration on this call → pointcare_registration (guide them; do NOT transfer yet)
  - live enrollment agent → agent_transfer
09 benefits → 10 appointment
10 appointment:
  - IF flow_path is renewal_no AND benefits=yes AND appointment scheduled → 14 transfer
  - NORMAL: otherwise (including appointment help = No) → immediately transition_to_offer_renewal_assistance. Never invent a goodbye.

GLOBAL RULES:
- Stay in the current state. Ask ONLY what that state allows.
- When a state lists REQUIRED spoken lines, say those lines FIRST, word-for-word when quoted.
- After identity is confirmed, ALWAYS ask the language question before any Medi-Cal intro.
- Ask ONE short question, then STOP and WAIT.
- After one clear answer: immediately call the required transition_to_* tool (do not invent extra chatter).
- Pass renewal_status and flow_path on transitions when the state says so.
- FORBIDDEN until Assistance/Closing/Transfer/Pointcare end: inventing goodbye / end_call early.
- Never continue talking over the patient.
- NEVER make the patient repeat a yes/no already given.
- Do NOT call save_progress or schedule_appointment — both disabled (cause silent waits). Use transition_to_* with parameters only.
- Never leave silence after an answer — transition in the same turn.
- Understand natural language. No "Unsure" path.
- Never disclose Medi-Cal details before identity is verified.
- Callback request → callback_end. DNC → record_dnc then end (only in allowed states).

TRANSFER RULES:
- Transfer when the patient chooses a live enrollment agent in renewal_followup_offer,
  when Assistance chooses phone/Enrollment help, OR when Appointment early-transfer rule matches.
- Then transition_to_agent_transfer and transfer_to_bach_agent to {{{{agent_number}}}}.
- Never say goodbye instead of transferring.
""".strip()


def _edge(dest: str, description: str, parameters: Dict[str, Any] | None = None) -> Dict[str, Any]:
    edge: Dict[str, Any] = {
        "destination_state_name": dest,
        "description": description,
    }
    if parameters:
        edge["parameters"] = parameters
    return edge


def _json_props(props: Dict[str, Any], required: List[str] | None = None) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def build_tools(function_url: str) -> Dict[str, Any]:
    """Shared tool definitions keyed by name."""
    save_params = _json_props(
        {
            "note": {"type": "string", "description": "Short note of what was captured"},
            "address": {"type": "string"},
            "phone": {"type": "string"},
            "email": {"type": "string"},
            "renewal_status": {
                "type": "string",
                "enum": list(RENEWAL_STATUS_VALUES),
            },
            "flow_path": {
                "type": "string",
                "enum": [
                    FLOW_PATH_YES_NO_CHANGES,
                    FLOW_PATH_YES_WITH_CHANGES,
                    FLOW_PATH_NO,
                ],
            },
            "household_changed": {"type": "boolean"},
            "household_change_details": {"type": "string"},
            "income_changed": {"type": "boolean"},
            "income_change_details": {"type": "string"},
            "support_status": {"type": "string"},
            "document_ready": {"type": "boolean"},
            "assistance_type": {
                "type": "string",
                "enum": [
                    "enrollment_team",
                    "phone_help",
                    "appointment",
                    "renewal_video",
                    "pointcare",
                    "live_agent",
                    "declined",
                    "",
                ],
            },
            "preferred_language": {"type": "string"},
            "additional_benefits_interest": {"type": "boolean"},
            "appointment_needed": {"type": "boolean"},
            "appointment_details": {"type": "string"},
            "final_outcome": {"type": "string", "enum": list(FINAL_OUTCOMES)},
            "callback_requested": {"type": "boolean"},
            "callback_details": {"type": "string"},
            "dnc_requested": {"type": "boolean"},
            "patient_verified": {"type": "boolean"},
        }
    )

    return {
        "end_call": {
            "type": "end_call",
            "name": "end_call",
            "description": "End the call after goodbye, wrong person, decline, voicemail, DNC, or callback confirmation.",
        },
        "save_progress": {
            "type": "custom",
            "name": "save_progress",
            "url": function_url,
            "method": "POST",
            "description": (
                "Optional: persist richer Medi-Cal fields. "
                "Do NOT use for simple yes/no — use state transitions instead. "
                "Never call this alone and go silent; always transition in the same turn."
            ),
            "speak_during_execution": False,
            "speak_after_execution": False,
            "parameters": save_params,
            "timeout_ms": 5000,
        },
        "request_callback": {
            "type": "custom",
            "name": "request_callback",
            "url": function_url,
            "method": "POST",
            "description": "Create a callback / follow-up request when the patient is busy or transfer fails.",
            "speak_during_execution": True,
            "speak_after_execution": True,
            "execution_message_description": "One moment while I note that callback.",
            "parameters": _json_props(
                {
                    "callback_details": {
                        "type": "string",
                        "description": "Preferred time or reason for callback",
                    }
                }
            ),
            "timeout_ms": 5000,
        },
        "record_dnc": {
            "type": "custom",
            "name": "record_dnc",
            "url": function_url,
            "method": "POST",
            "description": "Record Do Not Call / opt-out. Call before ending when patient asks not to be called.",
            "speak_during_execution": False,
            "speak_after_execution": True,
            "parameters": _json_props(
                {"reason": {"type": "string"}},
            ),
            "timeout_ms": 5000,
        },
        "schedule_appointment": {
            "type": "custom",
            "name": "schedule_appointment",
            "url": function_url,
            "method": "POST",
            "description": (
                "Record a healthcare appointment request (preferred day/time). "
                "Call once, then immediately continue/transition. Do not go silent."
            ),
            "speak_during_execution": False,
            "speak_after_execution": False,
            "parameters": _json_props(
                {
                    "preferred_time": {"type": "string"},
                    "notes": {"type": "string"},
                },
                required=["preferred_time"],
            ),
            "timeout_ms": 5000,
        },
        "send_renewal_video_sms": {
            "type": "custom",
            "name": "send_renewal_video_sms",
            "url": function_url,
            "method": "POST",
            "description": (
                "Text the Medi-Cal renewal follow-up video link to the patient's phone. "
                "Use ONLY after they choose the video option in renewal_followup_offer "
                "(not Pointcare, not live agent). "
                "After success: briefly confirm the video was texted, then immediately "
                "transition_to_household_income and continue the remaining questions. "
                "Do NOT say goodbye. Do NOT call end_call."
            ),
            "speak_during_execution": True,
            "speak_after_execution": True,
            "execution_message_description": "One moment while I text you that video link.",
            "parameters": _json_props({}),
            "timeout_ms": 8000,
        },
        "transfer_to_bach_agent": {
            "type": "transfer_call",
            "name": "transfer_to_bach_agent",
            "description": (
                "Transfer the patient to a live BACH enrollment agent at {{agent_number}}. "
                "Use after phone/Enrollment help is chosen, or after the Appointment early-transfer rule. "
                "If transfer fails, offer one retry or a callback — do not keep talking about saving."
            ),
            "transfer_destination": {
                "type": "predefined",
                "number": "{{agent_number}}",
            },
            "transfer_option": {"type": "cold_transfer"},
            "speak_during_execution": True,
            "execution_message_type": "static_text",
            "execution_message_description": "Please hold while I connect you with our enrollment team.",
        },
    }


def build_states(function_url: str) -> List[Dict[str, Any]]:
    tools = build_tools(function_url)
    transfer = tools["transfer_to_bach_agent"]
    end = tools["end_call"]

    return [
        {
            "name": NODE_OPENING,
            "state_prompt": f"""
## Node 01 Opening & Identity
The begin message already asked: "Hello, may I speak with {{{{patient_name}}}}?"
After they confirm (yeah / yes / speaking / that's me / hello):
IMMEDIATELY call transition_to_language_preference with patient_verified=true.
Do NOT speak the Lauren / Medi-Cal intro here.
Do NOT ask about renewal, address, phone, or email here.
FORBIDDEN here: language question is the NEXT state — do not skip it.
If wrong person/refuse/voicemail: callback_end or end_call as appropriate.
""".strip(),
            "edges": [
                _edge(
                    NODE_LANGUAGE,
                    "REQUIRED next right after identity confirm. Do not leave silence.",
                    _json_props({"patient_verified": {"type": "boolean"}}),
                ),
                _edge(
                    NODE_CALLBACK,
                    "Wrong person, unable to verify, refused, or busy → callback/end path.",
                ),
            ],
            "tools": [end],
        },
        {
            "name": NODE_LANGUAGE,
            "state_prompt": """
## Node 01b Language preference
When you ENTER this state, speak promptly and ask ONLY:
"Is it okay to continue in English, or would you prefer another language?"
STOP and WAIT for their answer.
- English / fine / okay / yes → immediately transition_to_purpose
  Pass preferred_language="English".
- Another language named (e.g. Spanish) → briefly acknowledge, continue in that language when you can, then immediately transition_to_purpose
  Pass preferred_language to what they asked for.
FORBIDDEN: Medi-Cal intro, renewal, contact questions, goodbye.
""".strip(),
            "edges": [
                _edge(
                    NODE_PURPOSE,
                    "REQUIRED next after language is settled.",
                    _json_props({"preferred_language": {"type": "string"}}),
                ),
            ],
        },
        {
            "name": NODE_PURPOSE,
            "state_prompt": f"""
## Node 02 Purpose + Lauren intro
When you ENTER this state, speak promptly (no silence). Say EXACTLY this (use the patient's first name when known):
"Hi {{{{patient_name}}}}, I'm {DEFAULT_AGENT_NAME}, an enrollment specialist with {ORG_NAME}. I see that your Medi-Cal coverage is coming up for renewal soon, and we want to make sure everything is updated so your coverage continues without any gap. Is now a good time to go through a few quick questions?"
STOP and WAIT for their answer.
FORBIDDEN here:
- Do NOT ask the language question again.
- Do NOT ask if renewal is already completed.
- Do NOT ask address / phone / email / household / income.
- Do NOT say that's all / goodbye / end the call.
If they can talk (yes / okay / continue): immediately transition_to_contact_information.
If busy: transition_to_callback_end.
""".strip(),
            "edges": [
                _edge(
                    NODE_CONTACT,
                    "REQUIRED next after purpose + good-time yes.",
                ),
                _edge(NODE_CALLBACK, "Patient is busy and wants a callback."),
            ],
        },
        {
            "name": NODE_CONTACT,
            "state_prompt": """
## Node 03 Contact Information
When you ENTER this state, FIRST say:
"Let me quickly confirm your contact information so you don't miss any important notices."
Then ask about ALL THREE, one at a time, WAIT each time. For each, offer an update:

1) Address:
- "Is your current address still {{address}}? Do you need to update it?"
- If address unknown: "What is your current address?"
- IMPORTANT meaning of answers:
  - "yes" / "still correct" / "same" / "no update" / "No" to updating → KEEP current address. Do NOT ask for address again.
  - "need to update" / gives a new address → capture new address once, then move on.

2) Phone:
- "Is this the best phone number to reach you, or do you need to update your phone number?"
- "same" / "yes" / "no update" / "No" to updating → KEEP current number. Do NOT re-ask.
- Only if they want to update → ask for the new number once.

3) Email:
- "Do you need to add or update an email address, or are you all set with email?"
- "no" / "same" / "all set" / skip → move on. Do NOT interrogate.
- If they want to add/update → ask once, accept even partial, move on.

FORBIDDEN: renewal, household, goodbye, end call, treating "No I don't need to update" as missing info.
When all three are done: immediately transition_to_renewal_status.
""".strip(),
            "edges": [
                _edge(
                    NODE_RENEWAL,
                    "REQUIRED next after contact checks.",
                ),
            ],
        },
        {
            "name": NODE_RENEWAL,
            "state_prompt": f"""
## Node 04 Renewal Status (BRANCH)
Ask ONCE, say: "Have you already completed your Medi-Cal renewal this year or received and completed the form?"

On first answer, map status and BRANCH (never end the call here):
- YES / already completed / submitted → renewal_status=COMPLETED or SUBMITTED_PENDING
  → immediately transition_to_changes_check
  Set flow_path hint toward completed path.
- NO / not yet / not received → renewal_status=NOT_COMPLETED or NOT_RECEIVED
  → immediately transition_to_renewal_followup_offer
  Set flow_path="{FLOW_PATH_NO}"

FORBIDDEN: goodbye, ending after Yes/No, skipping straight to household after No.
""".strip(),
            "edges": [
                _edge(
                    NODE_CHANGES,
                    "Renewal YES / completed / submitted → changes check next.",
                    _json_props(
                        {
                            "renewal_status": {
                                "type": "string",
                                "enum": ["COMPLETED", "SUBMITTED_PENDING"],
                            },
                            "flow_path": {
                                "type": "string",
                                "enum": [
                                    FLOW_PATH_YES_NO_CHANGES,
                                    FLOW_PATH_YES_WITH_CHANGES,
                                ],
                            },
                        }
                    ),
                ),
                _edge(
                    NODE_RENEWAL_OFFER,
                    (
                        f"Renewal NO / not completed → offer video, Pointcare, or live agent. "
                        f"Set flow_path={FLOW_PATH_NO}."
                    ),
                    _json_props(
                        {
                            "renewal_status": {
                                "type": "string",
                                "enum": ["NOT_COMPLETED", "NOT_RECEIVED"],
                            },
                            "flow_path": {
                                "type": "string",
                                "enum": [FLOW_PATH_NO],
                            },
                        },
                        required=["renewal_status", "flow_path"],
                    ),
                ),
            ],
        },
        {
            "name": NODE_RENEWAL_OFFER,
            "state_prompt": f"""
## Node 04b Renewal follow-up offer (renewal NO path only)
When you ENTER this state, ask ONCE and WAIT — say EXACTLY:
"I can text you a short Medi-Cal renewal video, or guide you through Pointcare registration on this call, or connect you with a live enrollment agent. Which would you prefer?"

On first clear answer:
- VIDEO / text me / send the video / SMS / link →
  1) Immediately call send_renewal_video_sms
  2) After it succeeds, say briefly: "I've just texted you the video. Let's keep going with a few quick questions."
  3) Immediately transition_to_household_income with flow_path="{FLOW_PATH_NO}" and assistance_type="renewal_video"
  Do NOT say goodbye. Do NOT call end_call.
- POINTCARE / registration / guide me / website / set up account →
  immediately transition_to_pointcare_registration with flow_path="{FLOW_PATH_NO}" and assistance_type="pointcare"
- LIVE AGENT / enrollment agent / transfer / talk to a person / connect me →
  immediately transition_to_agent_transfer with flow_path="{FLOW_PATH_NO}" and assistance_type="live_agent"

FORBIDDEN: two-option wording ("continue with questions"); ending after video; inventing goodbye here.
""".strip(),
            "edges": [
                _edge(
                    NODE_HOUSEHOLD_INCOME,
                    (
                        f"Video SMS sent — continue remaining questions. "
                        f"Set flow_path={FLOW_PATH_NO}."
                    ),
                    _json_props(
                        {
                            "flow_path": {
                                "type": "string",
                                "enum": [FLOW_PATH_NO],
                            },
                            "assistance_type": {
                                "type": "string",
                                "enum": ["renewal_video"],
                            },
                        },
                        required=["flow_path"],
                    ),
                ),
                _edge(
                    NODE_POINTCARE,
                    (
                        f"Patient chose Pointcare registration guidance on this call. "
                        f"Set flow_path={FLOW_PATH_NO}."
                    ),
                    _json_props(
                        {
                            "flow_path": {
                                "type": "string",
                                "enum": [FLOW_PATH_NO],
                            },
                            "assistance_type": {
                                "type": "string",
                                "enum": ["pointcare"],
                            },
                        },
                        required=["flow_path", "assistance_type"],
                    ),
                ),
                _edge(
                    NODE_TRANSFER,
                    (
                        f"Patient chose a live enrollment agent. Transfer next. "
                        f"Set flow_path={FLOW_PATH_NO}."
                    ),
                    _json_props(
                        {
                            "flow_path": {
                                "type": "string",
                                "enum": [FLOW_PATH_NO],
                            },
                            "assistance_type": {
                                "type": "string",
                                "enum": ["live_agent", "enrollment_team", "phone_help"],
                            },
                        },
                        required=["flow_path", "assistance_type"],
                    ),
                ),
            ],
            "tools": [tools["send_renewal_video_sms"]],
        },
        {
            "name": NODE_POINTCARE,
            "state_prompt": f"""
## Node 04c Pointcare registration guidance
You are still {DEFAULT_AGENT_NAME} from {ORG_NAME}. Guide {{{{patient_name}}}} through creating their Pointcare account on THEIR screen. You do NOT collect or store their data — you only coach them while they type.

RULES:
- One instruction at a time. Wait for them to confirm before the next step.
- Never ask them to say passwords, SSN, or verification codes aloud.
- Never invent goodbye until the flow is done.

STEPS (in order):
1) Confirm they have the Pointcare page open and are ready.
2) Email or Google/Apple sign-in — guide whichever they choose.
3) Email confirmation if needed.
4) Create password (they type privately).
5) Phone number + SMS verification code (they type the code themselves).
6) Profile fields one at a time (name, DOB, address, phone, email).
7) Consent form — briefly explain what Pointcare may do (monitor coverage, respond on their behalf, submit renewals, and marketing contact). Then guide signature.
8) Optional SSN / Medicaid ID — they may skip.
9) Confirm profile looks right.

When account setup is complete:
- If they still need household/income help for Medi-Cal, transition_to_household_income with flow_path="{FLOW_PATH_NO}".
- If they are done / all set, transition_to_closing with final_outcome="POINTCARE_GUIDED".
- If they ask for a live person, transition_to_agent_transfer.
""".strip(),
            "edges": [
                _edge(
                    NODE_HOUSEHOLD_INCOME,
                    f"Pointcare setup done; continue Medi-Cal questions. flow_path={FLOW_PATH_NO}.",
                    _json_props(
                        {
                            "flow_path": {
                                "type": "string",
                                "enum": [FLOW_PATH_NO],
                            },
                            "assistance_type": {
                                "type": "string",
                                "enum": ["pointcare"],
                            },
                        },
                        required=["flow_path"],
                    ),
                ),
                _edge(
                    NODE_CLOSING,
                    "Pointcare setup done and patient is all set — close the call.",
                    _json_props(
                        {
                            "final_outcome": {
                                "type": "string",
                                "enum": ["POINTCARE_GUIDED", "ASSISTANCE_PROVIDED"],
                            },
                            "assistance_type": {
                                "type": "string",
                                "enum": ["pointcare"],
                            },
                        }
                    ),
                ),
                _edge(
                    NODE_TRANSFER,
                    "Patient wants a live enrollment agent during/after Pointcare guidance.",
                    _json_props(
                        {
                            "assistance_type": {
                                "type": "string",
                                "enum": ["live_agent", "enrollment_team", "phone_help"],
                            }
                        }
                    ),
                ),
            ],
            "tools": [end],
        },
        {
            "name": NODE_CHANGES,
            "state_prompt": f"""
## Node 04a Changes Check (renewal YES path only)
Ask ONCE whether anything changed that could affect Medi-Cal:
address, household members, and/or employment/income.

- No changes → immediately transition_to_additional_benefits
  Set flow_path="{FLOW_PATH_YES_NO_CHANGES}"
- Any changes reported → immediately transition_to_household_income
  Set flow_path="{FLOW_PATH_YES_WITH_CHANGES}"

FORBIDDEN: docs, work exemption, goodbye, end call.
""".strip(),
            "edges": [
                _edge(
                    NODE_BENEFITS,
                    f"No address/household/income changes. Skip to benefits. flow_path={FLOW_PATH_YES_NO_CHANGES}.",
                    _json_props(
                        {
                            "flow_path": {
                                "type": "string",
                                "enum": [FLOW_PATH_YES_NO_CHANGES],
                            },
                            "household_changed": {"type": "boolean"},
                            "income_changed": {"type": "boolean"},
                        },
                        required=["flow_path"],
                    ),
                ),
                _edge(
                    NODE_HOUSEHOLD_INCOME,
                    f"Changes reported. Collect household/income details. flow_path={FLOW_PATH_YES_WITH_CHANGES}.",
                    _json_props(
                        {
                            "flow_path": {
                                "type": "string",
                                "enum": [FLOW_PATH_YES_WITH_CHANGES],
                            }
                        },
                        required=["flow_path"],
                    ),
                ),
            ],
        },
        {
            "name": NODE_HOUSEHOLD_INCOME,
            "state_prompt": f"""
## Node 05 Household + Income
Ask one at a time, WAIT each time:
1) Has anyone moved in or out of your household recently? (brief details if yes)
2) Has your employment or income changed recently? (brief details if yes)

Then BRANCH using flow_path / renewal_status (do not ask the patient which path):
- If flow_path is "{FLOW_PATH_NO}" OR renewal was not completed → immediately transition_to_work_exemption
- If flow_path is "{FLOW_PATH_YES_WITH_CHANGES}" (renewal YES + changes) → immediately transition_to_additional_benefits
  (skip work exemption, support, and documents)

FORBIDDEN: goodbye, end call, inventing other skips.
""".strip(),
            "edges": [
                _edge(
                    NODE_WORK_EXEMPTION,
                    f"Renewal NO path only (flow_path={FLOW_PATH_NO}). Continue to work exemption.",
                    _json_props(
                        {
                            "household_changed": {"type": "boolean"},
                            "household_change_details": {"type": "string"},
                            "income_changed": {"type": "boolean"},
                            "income_change_details": {"type": "string"},
                            "flow_path": {
                                "type": "string",
                                "enum": [FLOW_PATH_NO],
                            },
                        }
                    ),
                ),
                _edge(
                    NODE_BENEFITS,
                    f"Renewal YES + changes path (flow_path={FLOW_PATH_YES_WITH_CHANGES}). Skip to benefits.",
                    _json_props(
                        {
                            "household_changed": {"type": "boolean"},
                            "household_change_details": {"type": "string"},
                            "income_changed": {"type": "boolean"},
                            "income_change_details": {"type": "string"},
                            "flow_path": {
                                "type": "string",
                                "enum": [FLOW_PATH_YES_WITH_CHANGES],
                            },
                        }
                    ),
                ),
            ],
        },
        {
            "name": NODE_WORK_EXEMPTION,
            "state_prompt": """
## Node 06 Work Exemption pre-screen (renewal NO path only)
Ask once: Are you currently working, in school, or taking care of someone at home?
On first answer: immediately transition_to_additional_support.
FORBIDDEN: goodbye, end call, skip to benefits/transfer.
""".strip(),
            "edges": [
                _edge(
                    NODE_SUPPORT,
                    "REQUIRED next after work-exemption answer.",
                    _json_props({"support_status": {"type": "string"}}),
                ),
            ],
        },
        {
            "name": NODE_SUPPORT,
            "state_prompt": """
## Node 07 Additional Support (renewal NO path only)
Ask once: Do you need any additional support completing your Medi-Cal renewal?
On first answer: immediately transition_to_document_readiness.
FORBIDDEN: goodbye, end call, transfer.
""".strip(),
            "edges": [
                _edge(
                    NODE_DOCS,
                    "REQUIRED next after support answer.",
                    _json_props({"support_status": {"type": "string"}}),
                ),
            ],
        },
        {
            "name": NODE_DOCS,
            "state_prompt": """
## Node 08 Document Readiness (renewal NO path only)
Ask once about ID / proof of address / income documents ready.
On first yes/no: immediately transition_to_additional_benefits.
FORBIDDEN: goodbye, end call, transfer.
""".strip(),
            "edges": [
                _edge(
                    NODE_BENEFITS,
                    "REQUIRED next after documents answer.",
                    _json_props({"document_ready": {"type": "boolean"}}),
                ),
            ],
        },
        {
            "name": NODE_BENEFITS,
            "state_prompt": """
## Node 09 Additional Benefits
Ask once about interest in food assistance or other community care programs.
On first Yes or No: immediately transition_to_healthcare_appointment.
Pass additional_benefits_interest=true/false.
FORBIDDEN: goodbye, end call, transfer here.
""".strip(),
            "edges": [
                _edge(
                    NODE_APPOINTMENT,
                    "REQUIRED next after benefits answer.",
                    _json_props(
                        {"additional_benefits_interest": {"type": "boolean"}},
                        required=["additional_benefits_interest"],
                    ),
                ),
            ],
        },
        {
            "name": NODE_APPOINTMENT,
            "state_prompt": f"""
## Node 10 Healthcare Appointment (BRANCH)
Ask one at a time, WAIT:
1) When was your last healthcare visit?
2) Do you need help scheduling a healthcare appointment?
If yes: ask preferred day/time ONCE, then treat that as "appointment scheduled".
Do NOT call schedule_appointment (disabled — causes silent waits).
Pass preferred time in appointment_details on the transition.

Then BRANCH immediately (same turn — no extra questions):
- EARLY TRANSFER (skip Assistance/Outcome/Closing):
  ONLY if ALL are true:
  • flow_path is "{FLOW_PATH_NO}" (renewal was NOT completed), AND
  • additional_benefits_interest is true (they said Yes to benefits), AND
  • they asked for scheduling help and gave a preferred time
  → immediately transition_to_agent_transfer
  Say NOTHING about errors / "couldn't schedule" / asking again if they want transfer.
- NORMAL (including when they say NO to scheduling help):
  → immediately transition_to_offer_renewal_assistance

FORBIDDEN:
- Do NOT apologize about scheduling failures.
- Do NOT ask "is there anything else I can help with?"
- Do NOT say goodbye / have a great day here.
- Do NOT ask "would you like me to connect you" when early-transfer rules already match — just transfer.
- Do NOT leave silence after their answer.
""".strip(),
            "edges": [
                _edge(
                    NODE_TRANSFER,
                    (
                        f"EARLY TRANSFER: flow_path={FLOW_PATH_NO} AND benefits interest=yes "
                        "AND appointment preferred time captured. Skip assistance. "
                        "Do not ask again — transfer immediately."
                    ),
                    _json_props(
                        {
                            "appointment_needed": {"type": "boolean"},
                            "appointment_details": {"type": "string"},
                            "assistance_type": {
                                "type": "string",
                                "enum": ["appointment"],
                            },
                            "flow_path": {
                                "type": "string",
                                "enum": [FLOW_PATH_NO],
                            },
                        },
                        required=["appointment_details", "flow_path"],
                    ),
                ),
                _edge(
                    NODE_ASSISTANCE,
                    "Normal path — go to Assistance preference next.",
                    _json_props(
                        {
                            "appointment_needed": {"type": "boolean"},
                            "appointment_details": {"type": "string"},
                        }
                    ),
                ),
            ],
        },
        {
            "name": NODE_ASSISTANCE,
            "state_prompt": """
## Node 11 Final help preference
Ask ONCE and WAIT (short wording only):
"Would you like phone help from our Enrollment Team now, or are you all set for now?"

On first clear answer — call a transition tool in the SAME turn:
- phone help / Enrollment Team / connect me / transfer → immediately transition_to_agent_transfer
- no / don't need help / all set / decline → immediately transition_to_final_outcome

FORBIDDEN: thank-and-end yourself; "have a great day" here; silence; asking them to repeat.
""".strip(),
            "edges": [
                _edge(
                    NODE_TRANSFER,
                    "Phone help / Enrollment Team chosen — transfer next.",
                    _json_props(
                        {
                            "assistance_type": {
                                "type": "string",
                                "enum": ["enrollment_team", "phone_help"],
                            }
                        }
                    ),
                ),
                _edge(
                    NODE_OUTCOME,
                    "Declined / all set — closing path.",
                    _json_props(
                        {
                            "assistance_type": {
                                "type": "string",
                                "enum": ["declined", ""],
                            }
                        }
                    ),
                ),
            ],
        },
        {
            "name": NODE_OUTCOME,
            "state_prompt": f"""
## Node 12 Final Outcome (no-transfer path only)
No extra questions. Immediately transition_to_closing with final_outcome
(prefer PATIENT_DECLINED or RENEWAL_COMPLETED from: {', '.join(FINAL_OUTCOMES)}).
Do not speak a long goodbye here — Closing handles that.
""".strip(),
            "edges": [
                _edge(
                    NODE_CLOSING,
                    "REQUIRED next. Pass final_outcome; Closing says goodbye and ends.",
                    _json_props(
                        {
                            "final_outcome": {
                                "type": "string",
                                "enum": list(FINAL_OUTCOMES),
                            }
                        }
                    ),
                ),
            ],
        },
        {
            "name": NODE_CLOSING,
            "state_prompt": """
## Node 13 Closing (no-transfer path only)
Say one short thank-you, then immediately call end_call.
Example: "Thanks for your time today. Bay Area Community Health is here if you need us. Have a great day!"
Do not offer transfer here.
""".strip(),
            "edges": [],
            "tools": [end],
        },
        {
            "name": NODE_TRANSFER,
            "state_prompt": """
## Node 14 Agent Transfer (FINAL STEP)
As soon as you ENTER this state:
Say ONLY: "I'm connecting you to our enrollment team now. Please hold."
Then immediately call transfer_to_bach_agent ONCE to {{agent_number}}.

FORBIDDEN:
- Do NOT apologize about scheduling.
- Do NOT ask if they want to be connected — they are already in transfer.
- Do NOT say goodbye or that the call is done.
If transfer fails: offer ONE retry OR request_callback.
""".strip(),
            "edges": [
                _edge(NODE_CALLBACK, "Transfer failed or agent unavailable — create callback."),
            ],
            "tools": [transfer],
        },
        {
            "name": NODE_CALLBACK,
            "state_prompt": """
## Callback / End path
Confirm callback or that the call will end (wrong person / decline / DNC).
Call request_callback or record_dnc as appropriate, then end_call.
Do not discuss patient Medi-Cal details if identity was not verified.
""".strip(),
            "edges": [],
            "tools": [end],
        },
    ]


def build_llm_payload(
    *,
    function_url: str,
    patient_name: str,
    agent_display_name: str = DEFAULT_AGENT_NAME,
) -> Dict[str, Any]:
    tools = build_tools(function_url)
    return {
        "general_prompt": general_prompt(agent_name=agent_display_name),
        "begin_message": begin_message(patient_name),
        "start_speaker": "agent",
        "starting_state": STARTING_STATE,
        "states": build_states(function_url),
        "general_tools": [
            tools["request_callback"],
            tools["record_dnc"],
        ],
    }
