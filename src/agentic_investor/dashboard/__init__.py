"""Live operating dashboard: FastAPI + WebSocket over the running paper loop.

Runs in the same process as the loop. `paper-loop --serve-dashboard --port 8000`
starts a uvicorn worker on a background thread; the loop pushes every event
through the in-memory bus (`events.py`); the WebSocket handler fans out to
browser clients. REST endpoints hydrate initial state on page load.
"""
