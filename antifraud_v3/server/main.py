"""FastAPI app entry point — serves the WebSocket API and the static frontend from one
process (REWRITE_PLAN.md §9: "one local web app", no separate hosting).

Run with: uvicorn antifraud_v3.server.main:app --reload
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

load_dotenv()

from .ws import router as ws_router  # noqa: E402 (must follow load_dotenv())

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="AFG 守話 — antifraud_v3")
app.include_router(ws_router)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
