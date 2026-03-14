from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.models import Paper
from app.services.search import SearchAgent


def test_dedupe_prefers_unique_title_or_doi() -> None:
    papers = [
        Paper(id="1", title="Paper A", abstract="A", source="semantic_scholar", doi="10.1000/xyz"),
        Paper(id="2", title="Paper A Duplicate", abstract="A2", source="arxiv", doi="10.1000/xyz"),
        Paper(id="3", title="Paper B", abstract="B", source="arxiv"),
        Paper(id="4", title="Paper B", abstract="B2", source="semantic_scholar"),
    ]

    deduped = SearchAgent._dedupe_and_limit(papers, limit=10)

    assert [paper.id for paper in deduped] == ["1", "3"]


@pytest.mark.anyio
async def test_search_agent_returns_partial_results_when_one_source_fails() -> None:
    agent = SearchAgent()
    semantic_papers = [
        Paper(id="1", title="Paper A", abstract="A", source="semantic_scholar"),
        Paper(id="2", title="Paper B", abstract="B", source="semantic_scholar"),
    ]

    agent._search_semantic_scholar = AsyncMock(return_value=semantic_papers)
    agent._search_arxiv = AsyncMock(side_effect=RuntimeError("rate limited"))

    results = await agent.run("test question", limit=10)

    assert [paper.id for paper in results.papers] == ["1", "2"]
    assert results.warnings == ["arXiv search failed: rate limited"]
