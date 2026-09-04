from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.models import ResearchReport, SynthesisOutput
from app.store import InMemoryReportStore, SQLiteReportStore


def _report(rid: str, *, question: str = "q", minutes_ago: int = 0, papers=None, warnings=None):
    return ResearchReport(
        id=rid,
        question=question,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        papers=papers or [],
        extractions=[],
        synthesis=SynthesisOutput(consensus=[], contradictions=[], open_gaps=[]),
        related_work_markdown="",
        warnings=warnings or [],
        references=[],
    )


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    """Both stores implement the same protocol and must behave alike."""
    if request.param == "memory":
        return InMemoryReportStore()
    return SQLiteReportStore(str(tmp_path / "reports.db"))


def test_a_saved_report_can_be_read_back(store) -> None:
    store.save(_report("r1", question="how deep is the ocean"))
    assert store.get("r1").question == "how deep is the ocean"


def test_an_unknown_id_returns_none(store) -> None:
    assert store.get("missing") is None


def test_saving_the_same_id_replaces_the_report(store) -> None:
    store.save(_report("r1", question="first"))
    store.save(_report("r1", question="second"))
    assert store.get("r1").question == "second"
    assert len(store.list_reports()) == 1


def test_listing_is_newest_first(store) -> None:
    store.save(_report("old", minutes_ago=30))
    store.save(_report("new", minutes_ago=1))
    store.save(_report("middle", minutes_ago=10))
    assert [r.id for r in store.list_reports()] == ["new", "middle", "old"]


def test_listing_respects_the_limit(store) -> None:
    for index in range(5):
        store.save(_report(f"r{index}", minutes_ago=index))
    assert len(store.list_reports(limit=2)) == 2


def test_listing_an_empty_store_is_not_an_error(store) -> None:
    assert store.list_reports() == []


def test_clear_removes_everything(store) -> None:
    store.save(_report("r1"))
    store.clear()
    assert store.list_reports() == []
    assert store.get("r1") is None


def test_summaries_count_papers_and_warnings(store) -> None:
    from app.models import Paper

    papers = [Paper(id=f"p{i}", title=f"t{i}", abstract="a", source="arxiv") for i in range(3)]
    store.save(_report("r1", papers=papers, warnings=["w1", "w2"]))
    summary = store.list_reports()[0]
    assert summary.paper_count == 3
    assert summary.warning_count == 2


def test_sqlite_data_survives_a_new_instance(tmp_path) -> None:
    """The point of the SQLite store: history outlives the process."""
    path = str(tmp_path / "reports.db")
    SQLiteReportStore(path).save(_report("r1", question="persisted"))
    assert SQLiteReportStore(path).get("r1").question == "persisted"


def test_sqlite_connections_are_closed(tmp_path, recwarn) -> None:
    """`with sqlite3.connect(...)` commits but does not close; every call leaked one."""
    import gc
    import warnings

    path = str(tmp_path / "reports.db")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        store = SQLiteReportStore(path)
        for index in range(5):
            store.save(_report(f"r{index}"))
            store.get(f"r{index}")
            store.list_reports()
        del store
        gc.collect()

    leaks = [w for w in caught if issubclass(w.category, ResourceWarning)
             and "unclosed database" in str(w.message)]
    assert leaks == [], f"{len(leaks)} sqlite connections were left open"
