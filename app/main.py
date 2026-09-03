"""FastAPI application entry point."""

from __future__ import annotations

from pathlib import Path

import structlog
import logfire
from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.chat import router as chat_router
from app.config import get_settings

log = structlog.get_logger()

settings = get_settings()

# Logfire is optional: a run with no token executes every span locally but
# exports nothing, so observability never breaks startup (estimator pattern).
try:
    logfire.configure(service_name="cliniai_v2", send_to_logfire="if-token-present")
    _logfire_enabled = True
except Exception:  # noqa: BLE001
    log.warning("logfire_disabled", reason="configuration_failed")
    _logfire_enabled = False

app = FastAPI(
    title="CliniAI v2 — Medical Appointment Chatbot",
    version="0.1.0",
)

if _logfire_enabled:
    logfire.instrument_fastapi(app)

app.include_router(chat_router)

_WEB_DIR = Path(__file__).parent / "web"


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the chat web UI."""
    return FileResponse(_WEB_DIR / "index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
