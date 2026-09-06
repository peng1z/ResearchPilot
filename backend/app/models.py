from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Paper(BaseModel):
    id: str
    title: str
    abstract: str
    source: Literal["semantic_scholar", "arxiv", "openalex"]
    url: Optional[str] = None
    year: Optional[int] = None
    authors: list[str] = Field(default_factory=list)
    doi: Optional[str] = None


class PaperExtraction(BaseModel):
    paper_id: str
    title: str
    claims: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class SynthesisOutput(BaseModel):
    consensus: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    open_gaps: list[str] = Field(default_factory=list)


class RelatedWorkDraft(BaseModel):
    markdown: str


class ReportReference(BaseModel):
    label: str
    paper_id: str
    title: str
    source: Literal["semantic_scholar", "arxiv", "openalex"]
    year: Optional[int] = None
    url: Optional[str] = None


class ResearchQuestionRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    runtime: Optional["RuntimeSettings"] = None


class RuntimeSettings(BaseModel):
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_temperature: Optional[float] = None
    embedding_backend: Optional[str] = None
    embedding_model: Optional[str] = None
    local_embedding_model: Optional[str] = None
    semantic_scholar_api_key: Optional[str] = None
    openalex_mailto: Optional[str] = None


class ToolProvenance(BaseModel):
    """What produced this report, and what it is not.

    A report naming neither the build nor the model leaves a reader unable to
    tell what ran. The pipeline has since gained a third retrieval source and
    concurrent extraction, both of which change its output, so "ResearchPilot
    produced this" is not enough to place a report.

    Never carries credentials. Provider and model names are configuration; the
    key is not, and does not belong in a file a user will share.
    """

    name: str = "ResearchPilot"
    version: str
    commit: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    repository: str = "https://github.com/peng1z/ResearchPilot"
    method_paper: str = "https://arxiv.org/abs/2603.14629"
    method_paper_note: str = (
        "Describes version 1 of the method, which retrieves from Semantic "
        "Scholar and arXiv. This report was produced by the build identified "
        "above, which may differ. The method paper describes how this report "
        "was produced; it is not a source for the topic and does not belong "
        "in the reference list below."
    )


class ResearchReport(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    papers: list[Paper]
    extractions: list[PaperExtraction]
    synthesis: SynthesisOutput
    related_work_markdown: str
    warnings: list[str] = Field(default_factory=list)
    references: list[ReportReference] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Optional so an older stored report still validates; new runs set it.
    tool: Optional[ToolProvenance] = None


class ReportSummary(BaseModel):
    id: str
    question: str
    created_at: datetime
    paper_count: int
    warning_count: int


class ReportSearchHit(BaseModel):
    report: ReportSummary
    score: float


class PublicRuntimeConfig(BaseModel):
    llm_provider: str
    llm_model: str
    llm_base_url: Optional[str] = None
    llm_temperature: float
    embedding_backend: str
    embedding_model: Optional[str] = None
    local_embedding_model: str
    semantic_scholar_api_key_configured: bool


class StatusEvent(BaseModel):
    event: Literal[
        "queued",
        "agent_started",
        "agent_progress",
        "agent_completed",
        "done",
        "error",
    ]
    message: str
    report_id: Optional[str] = None
    agent: Optional[str] = None
    data: dict = Field(default_factory=dict)


class SearchResults(BaseModel):
    papers: list[Paper]
    warnings: list[str] = Field(default_factory=list)
