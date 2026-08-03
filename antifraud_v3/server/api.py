"""REST API for the History and Settings screens (REWRITE_PLAN.md §9) — separate from ws.py's
WebSocket endpoint since these are plain request/response, not the live-call stream.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..llm import reset_provider
from ..storage import history
from ..storage.settings_store import load_settings, save_settings

router = APIRouter(prefix="/api")


@router.get("/calls")
def api_list_calls():
    return history.list_calls()


@router.get("/calls/{call_id}")
def api_get_call(call_id: int):
    detail = history.get_call_detail(call_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="call not found")
    return detail


class SettingsUpdate(BaseModel):
    llm_model: str | None = None
    llm_effort: str | None = None
    debounce_chunks: int | None = None
    hard_triggers: list[str] | None = None


@router.get("/settings")
def api_get_settings():
    return load_settings()


@router.post("/settings")
def api_save_settings(update: SettingsUpdate):
    updates = {k: v for k, v in update.model_dump().items() if v is not None}
    saved = save_settings(updates)
    if "llm_model" in updates or "llm_effort" in updates:
        # Otherwise the next call would still be served by the stale cached ClaudeProvider —
        # see llm/__init__.py's reset_provider() docstring.
        reset_provider()
    return saved
