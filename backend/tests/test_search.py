from __future__ import annotations

import httpx
import pytest
import respx

from app.models import Paper
from app.services.search import (
    SearchAgent,
    _decode_inverted_abstract,
    _normalize_doi,
    _strip_openalex_wildcards,
)

SS_URL = SearchAgent.SEMANTIC_SCHOLAR_URL
ARXIV_URL = SearchAgent.ARXIV_URL
OPENALEX_URL = SearchAgent.OPENALEX_URL

OPENALEX_PAYLOAD = {
    "results": [
        {
            "id": "https://openalex.org/W3168867926",
            "doi": "https://doi.org/10.1000/lora",
            "display_name": "LoRA: Low-Rank Adaptation",
            "publication_year": 2021,
            "authorships": [
                {"author": {"display_name": "Edward Hu"}},
                {"author": {}},
                {},
            ],
            "abstract_inverted_index": {"We": [0], "propose": [1], "LoRA": [2, 4], "and": [3]},
        }
    ]
}


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
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))

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
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))

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
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))

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
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))

    results = await SearchAgent().run("q")

    assert [paper.id for paper in results.papers] == ["good"]


@pytest.mark.anyio
@respx.mock
async def test_arxiv_parses_entry_and_collapses_whitespace() -> None:
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    route = respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))
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
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))
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
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))
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
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed(entry)))

    results = await SearchAgent().run("q")

    assert results.papers[0].year is None


@pytest.mark.anyio
@respx.mock
async def test_http_error_from_one_source_becomes_a_warning() -> None:
    respx.get(SS_URL).mock(return_value=httpx.Response(429, text="slow down"))
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed(ARXIV_ENTRY)))

    results = await SearchAgent().run("q")

    assert [paper.id for paper in results.papers] == ["2401.12345v1"]
    assert len(results.warnings) == 1
    assert results.warnings[0].startswith("Semantic Scholar search failed:")


@pytest.mark.anyio
@respx.mock
async def test_both_sources_failing_raises_runtime_error_carrying_both_warnings() -> None:
    respx.get(SS_URL).mock(return_value=httpx.Response(500))
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))
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
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))

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
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))
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
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed(ARXIV_ENTRY)))

    results = await SearchAgent().run("q", limit=2)

    assert len(results.papers) == 2


def test_normalize_doi_collapses_the_spellings_the_sources_use() -> None:
    assert _normalize_doi("https://doi.org/10.1000/xyz") == "10.1000/xyz"
    assert _normalize_doi("http://doi.org/10.1000/xyz") == "10.1000/xyz"
    assert _normalize_doi("doi:10.1000/xyz") == "10.1000/xyz"
    assert _normalize_doi("10.1000/xyz") == "10.1000/xyz"
    assert _normalize_doi("  10.1000/xyz  ") == "10.1000/xyz"
    assert _normalize_doi(None) is None
    assert _normalize_doi("   ") is None


def test_decode_inverted_abstract_orders_by_position_not_by_token() -> None:
    # "and" appears twice, and the tokens are given out of order -- both are
    # routine in OpenAlex payloads.
    index = {"world": [1, 4], "hello": [0, 3], "again": [5], "and": [2]}

    assert _decode_inverted_abstract(index) == "hello world and hello world again"
    assert _decode_inverted_abstract({}) == ""
    assert _decode_inverted_abstract(None) == ""


@pytest.mark.anyio
@respx.mock
async def test_openalex_parses_a_work_and_decodes_its_abstract() -> None:
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed("")))
    route = respx.get(OPENALEX_URL).mock(
        return_value=httpx.Response(200, json=OPENALEX_PAYLOAD)
    )

    results = await SearchAgent().run("lora", limit=7)

    request = route.calls.last.request
    assert request.url.params["search"] == "lora"
    assert request.url.params["per-page"] == "7"
    assert "mailto" not in request.url.params

    paper = results.papers[0]
    assert paper.id == "W3168867926"
    assert paper.source == "openalex"
    assert paper.title == "LoRA: Low-Rank Adaptation"
    assert paper.abstract == "We propose LoRA and LoRA"
    assert paper.year == 2021
    assert paper.authors == ["Edward Hu"]
    assert paper.doi == "10.1000/lora"
    assert paper.url == "https://doi.org/10.1000/lora"


@pytest.mark.anyio
@respx.mock
async def test_openalex_sends_mailto_for_the_polite_pool_when_configured() -> None:
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed("")))
    route = respx.get(OPENALEX_URL).mock(
        return_value=httpx.Response(200, json=OPENALEX_PAYLOAD)
    )

    await SearchAgent(openalex_mailto="me@example.org").run("lora")

    assert route.calls.last.request.url.params["mailto"] == "me@example.org"


@pytest.mark.anyio
@respx.mock
async def test_openalex_skips_works_without_an_abstract_or_an_id() -> None:
    payload = {
        "results": [
            {"id": "https://openalex.org/W1", "display_name": "No abstract"},
            {"id": "", "display_name": "No id", "abstract_inverted_index": {"a": [0]}},
            {
                "id": "https://openalex.org/W3",
                "display_name": "Kept",
                "abstract_inverted_index": {"body": [0]},
            },
        ]
    }
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed("")))
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json=payload))

    results = await SearchAgent().run("q")

    assert [paper.id for paper in results.papers] == ["W3"]
    assert results.papers[0].doi is None
    assert results.papers[0].year is None


@pytest.mark.anyio
@respx.mock
async def test_openalex_failure_is_reported_without_losing_the_other_sources() -> None:
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json=SS_PAYLOAD))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed("")))
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(503))

    results = await SearchAgent().run("q")

    assert [paper.id for paper in results.papers] == ["ss-1"]
    assert results.warnings == [
        "OpenAlex search failed: Server error '503 Service Unavailable' for url "
        "'https://api.openalex.org/works?search=q&per-page=10'\n"
        "For more information check: "
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/503"
    ]


@pytest.mark.anyio
@respx.mock
async def test_a_doi_spelled_differently_by_two_sources_is_still_one_paper() -> None:
    # The case that motivated adding OpenAlex: Semantic Scholar reports a bare
    # DOI, OpenAlex reports the same DOI as a doi.org URL, and the titles are
    # not byte-identical either. Only DOI normalisation collapses these.
    ss_payload = {
        "data": [
            {
                "paperId": "ss-lora",
                "title": "LoRA: Low-Rank Adaptation of Large Language Models",
                "abstract": "We propose LoRA.",
                "externalIds": {"DOI": "10.1000/lora"},
            }
        ]
    }
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json=ss_payload))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed("")))
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json=OPENALEX_PAYLOAD))

    results = await SearchAgent().run("lora")

    assert [paper.id for paper in results.papers] == ["ss-lora"]


def test_strip_openalex_wildcards_removes_the_characters_that_400() -> None:
    assert _strip_openalex_wildcards("Does CoT help?") == "Does CoT help"
    assert _strip_openalex_wildcards("a * b") == "a b"
    assert _strip_openalex_wildcards("  spaced   out  ") == "spaced out"
    assert _strip_openalex_wildcards("no punctuation") == "no punctuation"


@pytest.mark.anyio
@respx.mock
async def test_openalex_query_drops_the_question_mark_that_would_be_rejected() -> None:
    # OpenAlex reads ? as wildcard syntax and answers 400 for a stemmed search,
    # and research questions almost always end in one.
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_arxiv_feed("")))
    route = respx.get(OPENALEX_URL).mock(
        return_value=httpx.Response(200, json=OPENALEX_PAYLOAD)
    )

    await SearchAgent().run("How does LoRA compare to full fine-tuning?")

    assert route.calls.last.request.url.params["search"] == (
        "How does LoRA compare to full fine-tuning"
    )


def test_interleave_takes_one_paper_from_each_source_in_turn() -> None:
    def paper(identifier: str) -> Paper:
        return Paper(id=identifier, title=identifier, abstract="x", source="arxiv")

    merged = SearchAgent._interleave(
        [
            [paper("a1"), paper("a2"), paper("a3")],
            [paper("b1")],
            [paper("c1"), paper("c2")],
        ]
    )

    assert [item.id for item in merged] == ["a1", "b1", "c1", "a2", "c2", "a3"]
    assert SearchAgent._interleave([]) == []
    assert SearchAgent._interleave([[], []]) == []


@pytest.mark.anyio
@respx.mock
async def test_a_source_returning_a_full_page_does_not_crowd_out_the_others() -> None:
    # arXiv routinely returns a full page. Before interleaving it consumed the
    # whole limit and OpenAlex never appeared in a result set.
    arxiv_entries = "".join(
        f"""
  <entry>
    <id>http://arxiv.org/abs/{index}</id>
    <title>arXiv paper {index}</title>
    <summary>Body {index}.</summary>
  </entry>"""
        for index in range(10)
    )
    openalex = {
        "results": [
            {
                "id": f"https://openalex.org/W{index}",
                "display_name": f"OpenAlex work {index}",
                "abstract_inverted_index": {"body": [0]},
            }
            for index in range(10)
        ]
    }
    respx.get(SS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get(ARXIV_URL).mock(
        return_value=httpx.Response(200, text=_arxiv_feed(arxiv_entries))
    )
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json=openalex))

    results = await SearchAgent().run("q", limit=10)

    sources = {paper.source for paper in results.papers}
    assert sources == {"arxiv", "openalex"}
    assert len(results.papers) == 10
