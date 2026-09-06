from __future__ import annotations

import asyncio

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.models import (
    Paper,
    PaperExtraction,
    RelatedWorkDraft,
    SearchResults,
    StatusEvent,
    SynthesisOutput,
)
from app.services.pipeline import ResearchPipeline


def _settings() -> Settings:
    return Settings(llm_provider="openai", llm_model="gpt-4.1-mini", llm_api_key="sk-test")


def _paper(pid: str) -> Paper:
    return Paper(id=pid, title=f"Title {pid}", abstract="Abstract", source="arxiv")


class FakeSearch:
    def __init__(self, papers, warnings=None):
        self.result = SearchResults(papers=papers, warnings=list(warnings or []))

    async def run(self, question, limit=10):
        return self.result


class FakeStore:
    def __init__(self, fail: Exception | None = None):
        self.fail = fail
        self.stored_papers: list = []
        self.stored_reports: list = []

    async def store_papers(self, report_id, papers, extractions):
        if self.fail:
            raise self.fail
        self.stored_papers.append((report_id, papers, extractions))

    async def store_report(self, report):
        if self.fail:
            raise self.fail
        self.stored_reports.append(report)


class FakeExtraction:
    """Stands in for ExtractionAgent, which the pipeline drives via aextract."""

    async def aextract(self, question, paper) -> PaperExtraction:
        return PaperExtraction(paper_id=paper.id, title=paper.title, claims=["c"])


def _pipeline(papers, *, warnings=None, store=None, monkeypatch=None) -> ResearchPipeline:
    import app.services.pipeline as module
    import contextlib

    # dspy_context builds a live LM; the agents are stubbed, so neutralise it.
    monkeypatch.setattr(module, "dspy_context", lambda settings: contextlib.nullcontext())

    return ResearchPipeline(
        _settings(),
        search_agent=FakeSearch(papers, warnings),
        extraction_agent=FakeExtraction(),
        synthesis_agent=lambda q, extractions: SynthesisOutput(
            consensus=["agreed"], contradictions=[], open_gaps=["gap"]
        ),
        writer_agent=lambda q, synthesis, papers: RelatedWorkDraft(markdown="# Related Work"),
        artifact_store=store or FakeStore(),
    )


async def _run(pipeline, report_id="r1", question="q"):
    events: list[StatusEvent] = []

    async def collect(event: StatusEvent) -> None:
        events.append(event)

    report = await pipeline.run(report_id, question, collect)
    return report, events


@pytest.mark.anyio
async def test_a_full_run_produces_a_report_and_the_expected_event_sequence(monkeypatch) -> None:
    pipeline = _pipeline([_paper("p1"), _paper("p2")], monkeypatch=monkeypatch)
    report, events = await _run(pipeline)

    assert report.id == "r1"
    assert len(report.papers) == 2
    assert len(report.extractions) == 2
    assert report.related_work_markdown == "# Related Work"

    assert events[0].event == "queued"
    assert events[-1].event == "done"
    agents_started = [e.agent for e in events if e.event == "agent_started"]
    assert agents_started == ["SearchAgent", "ExtractionAgent", "SynthesisAgent", "WriterAgent"]


@pytest.mark.anyio
async def test_references_are_numbered_in_paper_order(monkeypatch) -> None:
    pipeline = _pipeline([_paper("a"), _paper("b"), _paper("c")], monkeypatch=monkeypatch)
    report, _ = await _run(pipeline)
    assert [r.label for r in report.references] == ["R1", "R2", "R3"]
    assert [r.paper_id for r in report.references] == ["a", "b", "c"]


@pytest.mark.anyio
async def test_search_warnings_reach_the_report_and_the_stream(monkeypatch) -> None:
    pipeline = _pipeline([_paper("p1")], warnings=["arxiv timed out"], monkeypatch=monkeypatch)
    report, events = await _run(pipeline)
    assert report.warnings == ["arxiv timed out"]
    assert any(e.data and e.data.get("warning") for e in events)


@pytest.mark.anyio
async def test_no_papers_still_finishes(monkeypatch) -> None:
    pipeline = _pipeline([], monkeypatch=monkeypatch)
    report, events = await _run(pipeline)
    assert report.papers == []
    assert events[-1].event == "done"


@pytest.mark.anyio
async def test_a_persistence_failure_is_recorded_in_the_report(monkeypatch) -> None:
    """Regression: the warning was appended to a list pydantic had already
    copied, so it reached the event stream and never the report."""
    store = FakeStore(fail=RuntimeError("qdrant unreachable"))
    pipeline = _pipeline([_paper("p1")], store=store, monkeypatch=monkeypatch)
    report, events = await _run(pipeline)

    assert any("qdrant unreachable" in w for w in report.warnings), report.warnings
    assert any(e.agent == "QdrantArtifactStore" for e in events)


@pytest.mark.anyio
async def test_a_persistence_failure_does_not_abort_the_run(monkeypatch) -> None:
    store = FakeStore(fail=RuntimeError("boom"))
    pipeline = _pipeline([_paper("p1")], store=store, monkeypatch=monkeypatch)
    report, events = await _run(pipeline)
    assert events[-1].event == "done"
    assert report.related_work_markdown == "# Related Work"


@pytest.mark.anyio
async def test_artifacts_are_stored_when_persistence_works(monkeypatch) -> None:
    store = FakeStore()
    pipeline = _pipeline([_paper("p1")], store=store, monkeypatch=monkeypatch)
    report, _ = await _run(pipeline)
    assert store.stored_reports == [report]
    assert store.stored_papers[0][0] == "r1"


@pytest.mark.anyio
async def test_extraction_progress_is_reported_per_paper(monkeypatch) -> None:
    pipeline = _pipeline([_paper("p1"), _paper("p2")], monkeypatch=monkeypatch)
    _, events = await _run(pipeline)
    progress = [e for e in events if e.event == "agent_progress" and e.agent == "ExtractionAgent"]
    assert len(progress) == 2
    assert progress[0].data["total"] == 2


class SlowExtraction:
    """Records how many extractions are in flight at once."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.in_flight = 0
        self.peak_in_flight = 0

    async def aextract(self, question, paper) -> PaperExtraction:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        return PaperExtraction(paper_id=paper.id, title=paper.title, claims=["c"])


@pytest.mark.anyio
async def test_extractions_run_concurrently(monkeypatch) -> None:
    extraction = SlowExtraction()
    pipeline = _pipeline([_paper(f"p{index}") for index in range(6)], monkeypatch=monkeypatch)
    pipeline.extraction_agent = extraction

    report, _ = await _run(pipeline)

    assert extraction.peak_in_flight > 1
    assert [item.paper_id for item in report.extractions] == [f"p{index}" for index in range(6)]


@pytest.mark.anyio
async def test_extraction_concurrency_is_bounded_by_the_setting(monkeypatch) -> None:
    extraction = SlowExtraction()
    pipeline = _pipeline([_paper(f"p{index}") for index in range(8)], monkeypatch=monkeypatch)
    pipeline.extraction_agent = extraction
    pipeline.settings = pipeline.settings.model_copy(update={"extraction_concurrency": 3})

    await _run(pipeline)

    assert extraction.peak_in_flight == 3


@pytest.mark.anyio
async def test_report_records_what_produced_it(monkeypatch) -> None:
    # A report naming neither the build nor the model leaves a reader unable to
    # tell what ran, and the pipeline has since gained a third retrieval source
    # and concurrent extraction, both of which change its output.
    pipeline = _pipeline([_paper("p1")], monkeypatch=monkeypatch)
    report, _ = await _run(pipeline)

    assert report.tool is not None
    assert report.tool.name == "ResearchPilot"
    assert report.tool.version
    assert report.tool.model
    assert report.tool.method_paper == "https://arxiv.org/abs/2603.14629"
    assert "not a source for the topic" in report.tool.method_paper_note


@pytest.mark.anyio
async def test_report_carries_no_credentials(monkeypatch) -> None:
    # A report is a file a user shares. A key that reaches it is a key leaked.
    pipeline = _pipeline([_paper("p1")], monkeypatch=monkeypatch)
    pipeline.settings = pipeline.settings.model_copy(
        update={"llm_api_key": "sk-test-should-never-appear"}
    )
    report, _ = await _run(pipeline)

    serialised = report.model_dump_json()
    assert "sk-test-should-never-appear" not in serialised
    assert "api_key" not in serialised.lower()


class HalfBrokenExtraction:
    """Fails on one paper, succeeds on the rest."""

    def __init__(self, failing_id: str) -> None:
        self.failing_id = failing_id

    async def aextract(self, question, paper) -> PaperExtraction:
        if paper.id == self.failing_id:
            raise ValueError("Model returned no text for this field: None")
        return PaperExtraction(paper_id=paper.id, title=paper.title, claims=["c"])


@pytest.mark.anyio
async def test_one_unusable_extraction_costs_one_paper_not_the_run(monkeypatch) -> None:
    papers = [_paper("p1"), _paper("p2"), _paper("p3")]
    pipeline = _pipeline(papers, monkeypatch=monkeypatch)
    pipeline.extraction_agent = HalfBrokenExtraction("p2")

    report, events = await _run(pipeline)

    assert [item.paper_id for item in report.extractions] == ["p1", "p3"]
    assert any("Extraction failed for p2" in warning for warning in report.warnings)
    assert any("Skipped" in event.message for event in events)
