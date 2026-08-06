"""REST API for the History and Settings screens (docs/DESIGN.md §9) — separate from ws.py's
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


@router.delete("/calls/{call_id}", status_code=204)
def api_delete_call(call_id: int):
    if not history.delete_call(call_id):
        raise HTTPException(status_code=404, detail="call not found")


@router.delete("/calls")
def api_delete_all_calls():
    return {"deleted": history.delete_all_calls()}


class SettingsUpdate(BaseModel):
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_effort: str | None = None
    hard_triggers: list[str] | None = None
    line2_backend: str | None = None
    asr_backend: str | None = None
    llm_final_summary_enabled: bool | None = None
    emotion_backend: str | None = None


@router.get("/settings")
def api_get_settings():
    return load_settings()


@router.post("/settings")
def api_save_settings(update: SettingsUpdate):
    updates = {k: v for k, v in update.model_dump().items() if v is not None}
    saved = save_settings(updates)
    if {"llm_provider", "llm_model", "llm_effort"} & updates.keys():
        # Otherwise the next call would still be served by the stale cached provider — see
        # llm/__init__.py's reset_provider() docstring.
        reset_provider()
    return saved
