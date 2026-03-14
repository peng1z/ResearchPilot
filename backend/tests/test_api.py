from __future__ import annotations

import tempfile

from fastapi.testclient import TestClient

import app.main as main_module
from app.models import Paper, PaperExtraction, ResearchReport, StatusEvent, SynthesisOutput
from app.store import SQLiteReportStore


class FakePipeline:
    async def run(self, report_id: str, question: str, status) -> ResearchReport:
        await status(
            StatusEvent(
                event="queued",
                message="Research request accepted.",
                report_id=report_id,
            )
        )
        report = ResearchReport(
            id=report_id,
            question=question,
            papers=[
                Paper(
                    id="paper-1",
                    title="Test Paper",
                    abstract="A useful abstract.",
                    source="semantic_scholar",
                )
            ],
            extractions=[
                PaperExtraction(
                    paper_id="paper-1",
                    title="Test Paper",
                    claims=["Claim"],
                    methods=["Method"],
                    datasets=["Dataset"],
                    results=["Result"],
                    limitations=["Limitation"],
                )
            ],
            synthesis=SynthesisOutput(
                consensus=["Consensus"],
                contradictions=["Contradiction"],
                open_gaps=["Gap"],
            ),
            related_work_markdown="## Related Work\n\nTest output.",
            warnings=[],
            references=[],
        )
        await status(
            StatusEvent(
                event="done",
                message="Research pipeline finished.",
                report_id=report_id,
                data={"report": report.model_dump(mode="json")},
            )
        )
        return report


class FailingFactory:
    def __call__(self, *_args, **_kwargs):
        raise RuntimeError("Qdrant is unavailable.")


def make_report(report_id: str, question: str) -> ResearchReport:
    return ResearchReport(
        id=report_id,
        question=question,
        papers=[
            Paper(
                id="paper-1",
                title="Test Paper",
                abstract="A useful abstract.",
                source="semantic_scholar",
            )
        ],
        extractions=[
            PaperExtraction(
                paper_id="paper-1",
                title="Test Paper",
                claims=["Claim"],
                methods=["Method"],
                datasets=["Dataset"],
                results=["Result"],
                limitations=["Limitation"],
            )
        ],
        synthesis=SynthesisOutput(
            consensus=["Consensus"],
            contradictions=["Contradiction"],
            open_gaps=["Gap"],
        ),
        related_work_markdown="## Related Work\n\nTest output.",
        warnings=[],
        references=[],
    )


def test_research_stream_and_report_lookup() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        main_module.report_store = SQLiteReportStore(f"{tmpdir}/reports.db")
        main_module.app.state.pipeline_factory = lambda *_args, **_kwargs: FakePipeline()
        client = TestClient(main_module.app)

        response = client.post("/research", json={"question": "What is retrieval-augmented generation?"})

        assert response.status_code == 200
        assert "event: queued" in response.text
        assert "event: done" in response.text

        report_id = response.text.split('"report_id": "')[1].split('"')[0]
        report_response = client.get(f"/report/{report_id}")
        assert report_response.status_code == 200
        payload = report_response.json()
        assert payload["question"] == "What is retrieval-augmented generation?"
        assert payload["related_work_markdown"].startswith("## Related Work")


def test_missing_report_returns_404() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        main_module.report_store = SQLiteReportStore(f"{tmpdir}/reports.db")
        client = TestClient(main_module.app)
        response = client.get("/report/missing")
        assert response.status_code == 404


def test_research_streams_setup_failures_as_error_events() -> None:
    main_module.app.state.pipeline_factory = FailingFactory()
    client = TestClient(main_module.app)

    response = client.post("/research", json={"question": "What is retrieval-augmented generation?"})

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "Qdrant is unavailable." in response.text


def test_list_reports_returns_recent_history() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        main_module.report_store = SQLiteReportStore(f"{tmpdir}/reports.db")
        main_module.report_store.save(make_report("r1", "Question one"))
        main_module.report_store.save(make_report("r2", "Question two"))
        client = TestClient(main_module.app)

        response = client.get("/reports")

        assert response.status_code == 200
        payload = response.json()
        assert len(payload) == 2
        assert payload[0]["question"] in {"Question one", "Question two"}


def test_search_reports_falls_back_to_question_matching() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        main_module.report_store = SQLiteReportStore(f"{tmpdir}/reports.db")
        main_module.report_store.save(make_report("r1", "Graph neural networks for molecules"))
        main_module.report_store.save(make_report("r2", "Retrieval augmented generation methods"))
        main_module.app.state.pipeline_factory = FailingFactory()
        client = TestClient(main_module.app)

        response = client.get("/reports/search", params={"query": "retrieval"})

        assert response.status_code == 200
        payload = response.json()
        assert len(payload) == 1
        assert payload[0]["report"]["id"] == "r2"


def test_config_exposes_safe_runtime_defaults() -> None:
    client = TestClient(main_module.app)

    response = client.get("/config")

    assert response.status_code == 200
    payload = response.json()
    assert "llm_provider" in payload
    assert "llm_model" in payload
    assert "semantic_scholar_api_key_configured" in payload
    assert "llm_api_key" not in payload
