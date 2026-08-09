"""Rank jobs microservice — FastAPI app that runs the ranking worker loop.

The main API backend creates ``execution_job_items`` rows in the shared
database (queue).  This service claims those items and evaluates each job
posting with the LLM, writing the results back to the shared database.

Run as a single process (HTTP + worker):
    uvicorn app.main:app --port 8002

Or as a pure worker (no HTTP):
    python -m app.worker
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.logging import setup_logging
from app.worker import _init_worker_db, _shutdown_event, _worker_main

WORKER_SHUTDOWN_TIMEOUT = 10.0  # seconds to let the worker clean up


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    _init_worker_db()

    worker_task = asyncio.create_task(_worker_main(), name="rank-worker")

    yield

    # The worker loop can sleep up to POLL_FALLBACK seconds waiting for new
    # items, so cancel it explicitly for a fast shutdown.  An in-flight item
    # keeps its lease in the DB and is recovered by the next worker start.
    _shutdown_event.set()
    worker_task.cancel()
    try:
        await asyncio.wait_for(worker_task, timeout=WORKER_SHUTDOWN_TIMEOUT)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="Rank Jobs Microservice",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "service": "rank-jobs"}

    return app


app = create_app()
