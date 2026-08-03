"""FastAPI app entry point — serves the REST API, the WebSocket API, and the static frontend
from one process (REWRITE_PLAN.md §9: "one local web app", no separate hosting).

Run with: uvicorn antifraud_v3.server.main:app --reload
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

load_dotenv()

from ..storage.history import init_db  # noqa: E402 (must follow load_dotenv())
from .api import router as api_router  # noqa: E402
from .ws import router as ws_router  # noqa: E402

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

init_db()

app = FastAPI(title="AFG 守話 — antifraud_v3")
app.include_router(ws_router)
app.include_router(api_router)
# StaticFiles mounted at "/" is a catch-all prefix match — it MUST be registered last, or it
# would intercept /api/* and /ws/* requests before they ever reach the routers above.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
