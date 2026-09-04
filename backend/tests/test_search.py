from __future__ import annotations

import httpx
import pytest
import respx

from app.services.search import SearchAgent

SS_URL = SearchAgent.SEMANTIC_SCHOLAR_URL
ARXIV_URL = SearchAgent.ARXIV_URL


def _arxiv_feed(entries: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>"""


ARXIV_ENTRY = """
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <published>2024-01-22T18:00:00Z</published>
    <title>Attention
      Is All You Need</title>
    <summary>  A transformer
      architecture.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
  </entry>
"""

SS_PAYLOAD = {
    "data": [
        {
            "paperId": "ss-1",
            "title": "Deep Learning",
            "abstract": "A survey of deep learning.",
            "url": "https://example.org/ss-1",
            "year": 2015,
            "authors": [{"name": "Yann"}, {"name": ""}, {}],
            "externalIds": {"DOI": "10.1000/deep"},
        }
    ]
}




@pytest.mark.anyio
@respx.mock
async def test_semantic_scholar_parses_fields_and_sends_expected_query() -> None:
    route = respx.get(SS_URL).mock(return_value=httpx.Response(200, json=SS_PAYLOAD))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed("")))

    results = await SearchAgent().run("deep learning", limit=5)

    request = route.calls.last.request
    assert request.url.params["query"] == "deep learning"
    assert request.url.params["limit"] == "5"
    assert "externalIds" in request.url.params["fields"]
    assert "x-api-key" not in request.headers

    paper = results.papers[0]
    assert paper.id == "ss-1"
    assert paper.doi == "10.1000/deep"
    assert paper.year == 2015
    assert paper.source == "semantic_scholar"
    assert paper.authors == ["Yann"]


@pytest.mark.anyio
@respx.mock
async def test_semantic_scholar_sends_api_key_header_when_configured() -> None:
    route = respx.get(SS_URL).mock(return_value=httpx.Response(200, json=SS_PAYLOAD))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed("")))

    await SearchAgent(semantic_scholar_api_key="secret").run("q")

    assert route.calls.last.request.headers["x-api-key"] == "secret"


@pytest.mark.anyio
@respx.mock
async def test_semantic_scholar_skips_items_without_abstract_and_without_doi() -> None:
    payload = {
        "data": [
            {"paperId": "a", "title": "No abstract", "abstract": None},
            {"paperId": "b", "title": "Blank abstract", "abstract": "   "},
            {"paperId": "c", "title": "Kept", "abstract": "Real abstract."},
        ]
    }
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json=payload))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed("")))

    results = await SearchAgent().run("q")

    assert [paper.id for paper in results.papers] == ["c"]
    assert results.papers[0].doi is None
    assert results.papers[0].url is None


@pytest.mark.anyio
@respx.mock
async def test_semantic_scholar_skips_malformed_item_without_dropping_the_batch() -> None:
    payload = {
        "data": [
            {"title": "Missing paperId", "abstract": "Still an abstract."},
            {"paperId": "good", "title": "Fine", "abstract": "Fine abstract."},
        ]
    }
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json=payload))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed("")))

    results = await SearchAgent().run("q")

    assert [paper.id for paper in results.papers] == ["good"]


@pytest.mark.anyio
@respx.mock
async def test_arxiv_parses_entry_and_collapses_whitespace() -> None:
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    route = respx.get(ARXIV_URL).mock(
        return_value=httpx.Response(200, text=_arxiv_feed(ARXIV_ENTRY))
    )

    results = await SearchAgent().run("transformers", limit=3)

    request = route.calls.last.request
    assert request.url.params["search_query"] == "all:transformers"
    assert request.url.params["max_results"] == "3"

    paper = results.papers[0]
    assert paper.id == "2401.12345v1"
    assert paper.title == "Attention Is All You Need"
    assert paper.abstract == "A transformer architecture."
    assert paper.year == 2024
    assert paper.url == "http://arxiv.org/abs/2401.12345v1"
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.doi is None


@pytest.mark.anyio
@respx.mock
async def test_arxiv_skips_entries_without_summary() -> None:
    entry = """
  <entry>
    <id>http://arxiv.org/abs/1</id>
    <title>No summary</title>
  </entry>
"""
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get(ARXIV_URL).mock(
        return_value=httpx.Response(200, text=_arxiv_feed(entry + ARXIV_ENTRY))
    )

    results = await SearchAgent().run("q")

    assert [paper.id for paper in results.papers] == ["2401.12345v1"]


@pytest.mark.anyio
@respx.mock
async def test_arxiv_falls_back_to_slugged_title_when_id_is_missing() -> None:
    entry = """
  <entry>
    <title>A Curious Paper</title>
    <summary>Body.</summary>
  </entry>
"""
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed(entry)))

    results = await SearchAgent().run("q")

    assert results.papers[0].id == "a-curious-paper"
    assert results.papers[0].year is None


@pytest.mark.anyio
@respx.mock
async def test_arxiv_tolerates_unparseable_published_date() -> None:
    entry = """
  <entry>
    <id>http://arxiv.org/abs/2</id>
    <published>never</published>
    <title>Odd Date</title>
    <summary>Body.</summary>
  </entry>
"""
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed(entry)))

    results = await SearchAgent().run("q")

    assert results.papers[0].year is None


@pytest.mark.anyio
@respx.mock
async def test_http_error_from_one_source_becomes_a_warning() -> None:
    respx.get(SS_URL).mock(return_value=httpx.Response(429, text="slow down"))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed(ARXIV_ENTRY)))

    results = await SearchAgent().run("q")

    assert [paper.id for paper in results.papers] == ["2401.12345v1"]
    assert len(results.warnings) == 1
    assert results.warnings[0].startswith("Semantic Scholar search failed:")


@pytest.mark.anyio
@respx.mock
async def test_both_sources_failing_raises_runtime_error_carrying_both_warnings() -> None:
    respx.get(SS_URL).mock(return_value=httpx.Response(500))
    respx.get(ARXIV_URL).mock(side_effect=httpx.ConnectError("no route"))

    with pytest.raises(RuntimeError) as excinfo:
        await SearchAgent().run("q")

    message = str(excinfo.value)
    assert "Semantic Scholar search failed" in message
    assert "arXiv search failed" in message


@pytest.mark.anyio
@respx.mock
async def test_empty_results_from_both_sources_raises_runtime_error() -> None:
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed("")))

    with pytest.raises(RuntimeError, match="No papers with abstracts were found"):
        await SearchAgent().run("q")


@pytest.mark.anyio
@respx.mock
async def test_same_paper_from_both_sources_is_deduped_when_only_one_carries_a_doi() -> None:
    title = "Attention Is All You Need"
    payload = {
        "data": [
            {
                "paperId": "ss-dup",
                "title": title,
                "abstract": "A transformer architecture.",
                "externalIds": {"DOI": "10.1000/attn"},
            }
        ]
    }
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json=payload))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed(ARXIV_ENTRY)))

    results = await SearchAgent().run("q")

    assert [paper.id for paper in results.papers] == ["ss-dup"]


@pytest.mark.anyio
@respx.mock
async def test_limit_is_applied_across_the_merged_result_set() -> None:
    payload = {
        "data": [
            {"paperId": f"ss-{index}", "title": f"Paper {index}", "abstract": "Body."}
            for index in range(5)
        ]
    }
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json=payload))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed(ARXIV_ENTRY)))

    results = await SearchAgent().run("q", limit=2)

    assert len(results.papers) == 2
