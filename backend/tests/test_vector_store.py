from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.models import Paper, PaperExtraction, ResearchReport, SynthesisOutput
from app.vector_store import QdrantArtifactStore


class FakeClient:
    """Records calls instead of talking to Qdrant."""

    def __init__(self, existing: list[str] | None = None) -> None:
        self.existing = list(existing or [])
        self.created: list[tuple[str, int]] = []
        self.upserts: list[tuple[str, list]] = []
        self.searches: list[dict] = []
        self.search_results: list = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.existing])

    def create_collection(self, collection_name, vectors_config):
        self.existing.append(collection_name)
        self.created.append((collection_name, vectors_config.size))

    def upsert(self, collection_name, points):
        self.upserts.append((collection_name, points))

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return self.search_results


class FakeEmbedder:
    def __init__(self, enabled: bool = True, dim: int = 4) -> None:
        self.enabled = enabled
        self.dim = dim
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [[0.1] * self.dim for _ in texts]


def _store(embedder=None, existing=None) -> QdrantArtifactStore:
    store = QdrantArtifactStore("http://127.0.0.1:1", embedder=embedder)
    store.client = FakeClient(existing)
    return store


def _paper(pid: str = "p1") -> Paper:
    return Paper(id=pid, title=f"Title {pid}", abstract="Abstract", source="arxiv")


def _report(rid: str = "r1") -> ResearchReport:
    return ResearchReport(
        id=rid, question="q", papers=[_paper()], extractions=[],
        synthesis=SynthesisOutput(consensus=["c"], contradictions=[], open_gaps=["g"]),
        related_work_markdown="# draft", warnings=[], references=[],
    )


# --- collection naming ----------------------------------------------------


def test_collections_are_named_for_the_embedding_backend() -> None:
    """Switching backends must not mix vector spaces in one collection."""
    dense = QdrantArtifactStore("http://127.0.0.1:1", embedder=FakeEmbedder(enabled=True))
    plain = QdrantArtifactStore("http://127.0.0.1:1", embedder=FakeEmbedder(enabled=False))
    assert dense.papers_collection != plain.papers_collection
    assert dense.papers_collection.endswith("_dense")
    assert plain.papers_collection.endswith("_artifacts")


def test_no_embedder_behaves_like_a_disabled_one() -> None:
    assert QdrantArtifactStore("http://127.0.0.1:1").reports_collection.endswith("_artifacts")


# --- point identity -------------------------------------------------------


def test_point_ids_are_stable_across_calls() -> None:
    assert QdrantArtifactStore._stable_uuid("r1", "p1") == QdrantArtifactStore._stable_uuid("r1", "p1")


@pytest.mark.parametrize(
    ("a", "b"),
    [(("r1", "p1"), ("r1", "p2")), (("r1", "p1"), ("r2", "p1")), (("report", "r1"), ("r1", "report"))],
)
def test_different_inputs_get_different_point_ids(a, b) -> None:
    """A collision would silently overwrite another report's stored paper."""
    assert QdrantArtifactStore._stable_uuid(*a) != QdrantArtifactStore._stable_uuid(*b)


# --- storing papers -------------------------------------------------------


@pytest.mark.anyio
async def test_storing_papers_creates_the_collection_once() -> None:
    store = _store()
    await store.store_papers("r1", [_paper("p1")], [])
    await store.store_papers("r1", [_paper("p2")], [])
    assert len(store.client.created) == 1


@pytest.mark.anyio
async def test_an_existing_collection_is_not_recreated() -> None:
    store = _store(existing=["researchpilot_papers_artifacts"])
    await store.store_papers("r1", [_paper()], [])
    assert store.client.created == []


@pytest.mark.anyio
async def test_no_papers_means_no_upsert() -> None:
    store = _store()
    await store.store_papers("r1", [], [])
    assert store.client.upserts == []


@pytest.mark.anyio
async def test_extractions_are_attached_to_their_own_paper() -> None:
    store = _store()
    extraction = PaperExtraction(paper_id="p2", title="Title p2", claims=["c"])
    await store.store_papers("r1", [_paper("p1"), _paper("p2")], [extraction])
    payloads = [point.payload for point in store.client.upserts[0][1]]
    by_paper = {p["paper"]["id"]: p["extraction"] for p in payloads}
    assert by_paper["p2"]["paper_id"] == "p2"
    assert by_paper["p1"] is None


@pytest.mark.anyio
async def test_the_embedder_sees_title_and_abstract() -> None:
    embedder = FakeEmbedder()
    store = _store(embedder=embedder)
    await store.store_papers("r1", [_paper()], [])
    assert "Title p1" in embedder.calls[0][0]
    assert "Abstract" in embedder.calls[0][0]


@pytest.mark.anyio
async def test_collection_dimension_follows_the_embedder() -> None:
    store = _store(embedder=FakeEmbedder(dim=7))
    await store.store_papers("r1", [_paper()], [])
    assert store.client.created[0][1] == 7


@pytest.mark.anyio
async def test_without_an_embedder_placeholder_vectors_are_distinct() -> None:
    """Identical vectors would make every paper equally similar to any query."""
    store = _store()
    await store.store_papers("r1", [_paper("p1"), _paper("p2"), _paper("p3")], [])
    vectors = [point.vector for point in store.client.upserts[0][1]]
    assert len({tuple(v) for v in vectors}) == 3


# --- storing and searching reports ---------------------------------------


@pytest.mark.anyio
async def test_report_vector_covers_the_question_and_the_synthesis() -> None:
    embedder = FakeEmbedder()
    store = _store(embedder=embedder)
    await store.store_report(_report())
    text = embedder.calls[0][0]
    for fragment in ("q", "c", "g", "# draft"):
        assert fragment in text


@pytest.mark.anyio
async def test_search_returns_summaries_with_scores() -> None:
    store = _store(embedder=FakeEmbedder())
    store.client.search_results = [
        SimpleNamespace(payload=_report("r1").model_dump(mode="json"), score=0.87)
    ]
    hits = await store.search_reports("q", limit=3)
    assert len(hits) == 1
    assert hits[0].report.id == "r1"
    assert hits[0].report.paper_count == 1
    assert hits[0].score == pytest.approx(0.87)
    assert store.client.searches[0]["limit"] == 3


@pytest.mark.anyio
async def test_a_missing_score_becomes_zero() -> None:
    store = _store(embedder=FakeEmbedder())
    store.client.search_results = [
        SimpleNamespace(payload=_report().model_dump(mode="json"), score=None)
    ]
    assert (await store.search_reports("q"))[0].score == 0.0
