"""Domain schemas — all structured data the LLM must produce via Instructor."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat state node names — used by LangGraph edges
# ---------------------------------------------------------------------------

class Node(str, Enum):
    INTENT         = "intent"
    SUGGEST_DOCTOR = "suggest_doctor"
    DOCTOR_INFO = "doctor_info"
    LOCATION    = "location"
    DATETIME    = "datetime"
    FETCH_SLOTS = "fetch_slots"
    COLLECT     = "collect"
    CONFIRM     = "confirm"
    DONE        = "done"


# ---------------------------------------------------------------------------
# Structured outputs extracted by Instructor
# ---------------------------------------------------------------------------

class ConsentResult(BaseModel):
    """Whether the patient explicitly accepted the data-processing consent."""
    accepted: bool = Field(description="True only if the patient clearly and explicitly accepts")
    reason: str = Field(description="One-sentence explanation of the classification")


class RedFlagResult(BaseModel):
    """LLM verification of a deterministic red-flag pattern match."""
    is_emergency: bool = Field(description="True if the message describes a medical emergency")
    category: str = Field(default="", description="Red-flag category (e.g. dolor_toracico)")
    reason: str = Field(default="", description="One-sentence justification")


class IntentResult(BaseModel):
    """Classifier output: what does the patient want?"""
    intent: Literal["book_appointment", "doctor_info", "find_doctor", "greeting", "other"]
    reason: str = Field(description="One-sentence explanation of the classification")


class SymptomExtraction(BaseModel):
    """Extracted from the patient's description of their health problem."""
    symptom_text: str = Field(description="The symptoms in the patient's own words")
    suspected_specialty: str = Field(
        description="Medical specialty most likely to treat these symptoms, in Spanish (e.g. Cardiología)"
    )
    reasoning: str = Field(description="One-sentence justification of the specialty choice")


class DoctorChoice(BaseModel):
    """The doctor the patient picked from the suggestion list."""
    chosen: bool = Field(description="True only if the patient clearly picked one doctor from the list")
    doctor_id: str = Field(default="", description="The doctor_id of the chosen doctor, empty if none chosen")
    doctor_name: str = Field(default="", description="Name of the chosen doctor, empty if none chosen")


class LocationChoice(BaseModel):
    """Extracted when the patient selects a clinic location."""
    localidad_id: str = Field(description="Numeric ID of the chosen localidad")
    localidad_name: str = Field(description="Human-readable name of the chosen localidad")


class DateRangeResult(BaseModel):
    """Extracted date or date range from a natural-language patient message."""
    date_start: str = Field(description="Start date in yyyy-MM-dd format")
    date_end: str = Field(
        description="End date in yyyy-MM-dd format — same as date_start for a single day"
    )
    reasoning: str = Field(description="Step-by-step date arithmetic shown to the user")


class SlotChoice(BaseModel):
    """The specific appointment slot the patient chose from the list."""
    slot_datetime: str = Field(description="Chosen slot in yyyy-MM-dd HH:mm format")


class YesNoResult(BaseModel):
    """Whether the patient accepted a proposed option."""
    yes: bool = Field(description="True only if the patient clearly accepts the proposal")


class PatientData(BaseModel):
    """All patient details needed to finalise the appointment."""
    full_name: str
    symptoms: str
    email: str


class AppointmentConfirmation(BaseModel):
    """Final confirmation before submitting the booking."""
    confirmed: bool
    patient: PatientData
    slot_datetime: str
    localidad_id: str


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class ChatState(BaseModel):
    """Full mutable state passed between LangGraph nodes."""
    session_id: str
    # Empty until the patient selects (or arrives with) a doctor.
    doctor_id: str = ""

    # Conversation history as list of {"role": ..., "content": ...} dicts
    messages: list[dict] = Field(default_factory=list)

    # Current node / phase
    current_node: Node = Node.INTENT

    # Doctor suggestion flow (patient describes symptoms first)
    suggested_doctors: list[dict] = Field(default_factory=list)
    symptoms_hint: str | None = None

    # Populated as the conversation progresses
    doctor_data: dict | None = None
    localidades: list[dict] = Field(default_factory=list)
    localidad_id: str | None = None
    localidad_name: str | None = None
    locations_presented: bool = False
    date_start: str | None = None
    date_end: str | None = None
    available_slots: list[dict] = Field(default_factory=list)
    slot_datetime: str | None = None
    # Proactive slot offer: the single slot proposed with a SÍ/NO control
    offered_slot: str | None = None
    proactive_offer_done: bool = False
    patient: PatientData | None = None
    booking_url: str | None = None

    # Response to send back to the patient after each node
    reply: str = ""
    # Optional text the UI renders AFTER structured widgets (e.g. the doctor list)
    reply_footnote: str = ""
    # Per-turn: location options for the UI to render as clickable buttons
    location_options: list[dict] = Field(default_factory=list)
    # Per-turn: short reply options (e.g. SÍ/NO, dates) rendered as buttons
    quick_replies: list[str] = Field(default_factory=list)

    # Explicit consent (first turn) — recorded with UTC timestamp
    consent_requested: bool = False
    consent_given: bool = False
    consent_timestamp: str | None = None

    # Human-in-the-loop: True while the graph is paused awaiting explicit
    # patient confirmation before the irreversible booking action.
    awaiting_human: bool = False

    completed: bool = False
