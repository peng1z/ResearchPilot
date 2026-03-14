from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import apply_runtime_overrides, get_settings
from app.models import PublicRuntimeConfig, ReportSearchHit, ReportSummary, ResearchQuestionRequest, ResearchReport, StatusEvent
from app.services.pipeline import ResearchPipeline
from app.store import SQLiteReportStore

settings = get_settings()
report_store = SQLiteReportStore(settings.report_db_path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.pipeline_factory = lambda runtime_settings=None: ResearchPipeline(runtime_settings or settings)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_sse(event: StatusEvent) -> str:
    return f"event: {event.event}\ndata: {json.dumps(event.model_dump(mode='json'))}\n\n"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config", response_model=PublicRuntimeConfig)
async def config() -> PublicRuntimeConfig:
    return PublicRuntimeConfig(
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
        llm_temperature=settings.llm_temperature,
        embedding_backend=settings.embedding_backend,
        embedding_model=settings.embedding_model,
        local_embedding_model=settings.local_embedding_model,
        semantic_scholar_api_key_configured=bool(settings.semantic_scholar_api_key),
    )


@app.post("/research")
async def run_research(request: ResearchQuestionRequest) -> StreamingResponse:
    report_id = str(uuid4())
    queue: asyncio.Queue[Optional[StatusEvent]] = asyncio.Queue()

    async def push_status(event: StatusEvent) -> None:
        await queue.put(event)

    async def background() -> None:
        try:
            runtime_settings = (
                apply_runtime_overrides(settings, request.runtime.model_dump())
                if request.runtime
                else settings
            )
            pipeline = app.state.pipeline_factory(runtime_settings)
            report = await pipeline.run(report_id, request.question, push_status)
            report_store.save(report)
        except Exception as exc:
            await queue.put(
                StatusEvent(
                    event="error",
                    message=str(exc),
                    report_id=report_id,
                )
            )
        finally:
            await queue.put(None)

    async def stream() -> AsyncIterator[str]:
        task = asyncio.create_task(background())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield format_sse(event)
        finally:
            await task

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/report/{report_id}", response_model=ResearchReport)
async def get_report(report_id: str) -> ResearchReport:
    report = report_store.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@app.get("/reports", response_model=list[ReportSummary])
async def list_reports(limit: int = Query(default=10, ge=1, le=100)) -> list[ReportSummary]:
    return report_store.list_reports(limit=limit)


@app.get("/reports/search", response_model=list[ReportSearchHit])
async def search_reports(query: str = Query(min_length=3), limit: int = Query(default=5, ge=1, le=20)) -> list[ReportSearchHit]:
    try:
        pipeline = app.state.pipeline_factory()
        return await pipeline.artifact_store.search_reports(query=query, limit=limit)
    except Exception:
        history = report_store.list_reports(limit=limit)
        filtered = [report for report in history if query.lower() in report.question.lower()]
        return [ReportSearchHit(report=report, score=1.0) for report in filtered[:limit]]
