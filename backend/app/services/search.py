from __future__ import annotations

import asyncio
from typing import Optional
import xml.etree.ElementTree as ET

import httpx

from app.models import Paper, SearchResults


def _normalize_doi(raw: Optional[str]) -> Optional[str]:
    """Reduce a DOI to its bare form.

    Sources disagree on presentation: Semantic Scholar reports `10.1000/xyz`
    while OpenAlex reports `https://doi.org/10.1000/xyz`. Dedupe keys on the
    DOI, so the two spellings have to collapse to one.
    """
    if not raw:
        return None
    doi = raw.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return doi or None


def _decode_inverted_abstract(index: Optional[dict[str, list[int]]]) -> str:
    """Rebuild plain text from OpenAlex's token -> positions mapping.

    OpenAlex ships abstracts as an inverted index rather than a string. A
    token may occur at several positions, and the positions of the tokens
    that are present need not be contiguous, so the text is assembled by
    sorting on position rather than by filling a fixed-size list.
    """
    if not index:
        return ""
    placed: list[tuple[int, str]] = [
        (position, token)
        for token, positions in index.items()
        for position in positions
    ]
    placed.sort()
    return " ".join(token for _, token in placed)


def _strip_openalex_wildcards(question: str) -> str:
    """Remove the characters OpenAlex reads as wildcards.

    OpenAlex rejects `?` and `*` in a stemmed `search=` with a 400: they are
    wildcard syntax there, and wildcards are only legal in an exact search.
    Research questions end in a question mark far more often than not, so the
    punctuation is dropped rather than the query being switched to exact.
    """
    return " ".join(question.replace("?", " ").replace("*", " ").split())


def _parse_year(published: str) -> Optional[int]:
    """Read the year out of an Atom <published> timestamp, tolerating junk."""
    try:
        return int(published[:4])
    except ValueError:
        return None


class SearchAgent:
    SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    ARXIV_URL = "https://export.arxiv.org/api/query"
    OPENALEX_URL = "https://api.openalex.org/works"

    def __init__(
        self,
        semantic_scholar_api_key: Optional[str] = None,
        openalex_mailto: Optional[str] = None,
    ) -> None:
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.openalex_mailto = openalex_mailto

    async def run(self, question: str, limit: int = 10) -> SearchResults:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "ResearchPilot/0.1"},
        ) as client:
            results = await asyncio.gather(
                self._search_semantic_scholar(client, question, limit),
                self._search_arxiv(client, question, limit),
                self._search_openalex(client, question, limit),
                return_exceptions=True,
            )
        warnings: list[str] = []
        paper_batches: list[list[Paper]] = []
        source_names = ["Semantic Scholar", "arXiv", "OpenAlex"]

        for source_name, result in zip(source_names, results):
            if isinstance(result, Exception):
                warnings.append(f"{source_name} search failed: {result}")
            else:
                paper_batches.append(result)

        papers = self._dedupe_and_limit(self._interleave(paper_batches), limit)
        if not papers:
            warning_text = "; ".join(warnings) if warnings else "No papers with abstracts were found."
            raise RuntimeError(f"SearchAgent could not retrieve papers. {warning_text}")
        return SearchResults(papers=papers, warnings=warnings)

    async def _search_semantic_scholar(
        self,
        client: httpx.AsyncClient,
        question: str,
        limit: int,
    ) -> list[Paper]:
        headers = {}
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key
        response = await client.get(
            self.SEMANTIC_SCHOLAR_URL,
            params={
                "query": question,
                "limit": limit,
                "fields": "paperId,title,abstract,url,year,authors,externalIds",
            },
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
        results: list[Paper] = []
        for item in payload.get("data", []):
            abstract = (item.get("abstract") or "").strip()
            paper_id = item.get("paperId")
            # A single malformed record must not discard every other result
            # from this source: run() turns any exception here into a warning
            # and drops the entire batch.
            if not abstract or not paper_id:
                continue
            results.append(
                Paper(
                    id=paper_id,
                    title=item.get("title", "Untitled"),
                    abstract=abstract,
                    source="semantic_scholar",
                    url=item.get("url"),
                    year=item.get("year"),
                    authors=[author.get("name", "") for author in item.get("authors", []) if author.get("name")],
                    doi=_normalize_doi((item.get("externalIds") or {}).get("DOI")),
                )
            )
        return results

    async def _search_arxiv(
        self,
        client: httpx.AsyncClient,
        question: str,
        limit: int,
    ) -> list[Paper]:
        response = await client.get(
            self.ARXIV_URL,
            params={
                "search_query": f"all:{question}",
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        results: list[Paper] = []
        for entry in root.findall("atom:entry", ns):
            paper_id = entry.findtext("atom:id", default="", namespaces=ns)
            title = " ".join((entry.findtext("atom:title", default="", namespaces=ns)).split())
            abstract = " ".join((entry.findtext("atom:summary", default="", namespaces=ns)).split())
            if not abstract:
                continue
            authors = [author.findtext("atom:name", default="", namespaces=ns) for author in entry.findall("atom:author", ns)]
            year = _parse_year(entry.findtext("atom:published", default="", namespaces=ns))
            results.append(
                Paper(
                    id=paper_id.rsplit("/", maxsplit=1)[-1] or title.lower().replace(" ", "-"),
                    title=title,
                    abstract=abstract,
                    source="arxiv",
                    url=paper_id,
                    year=year,
                    authors=[author for author in authors if author],
                )
            )
        return results

    async def _search_openalex(
        self,
        client: httpx.AsyncClient,
        question: str,
        limit: int,
    ) -> list[Paper]:
        params: dict[str, str | int] = {
            "search": _strip_openalex_wildcards(question),
            "per-page": limit,
        }
        if self.openalex_mailto:
            params["mailto"] = self.openalex_mailto
        response = await client.get(self.OPENALEX_URL, params=params)
        response.raise_for_status()
        payload = response.json()

        results: list[Paper] = []
        for item in payload.get("results", []):
            abstract = _decode_inverted_abstract(item.get("abstract_inverted_index")).strip()
            work_id = (item.get("id") or "").rsplit("/", maxsplit=1)[-1]
            if not abstract or not work_id:
                continue
            doi = _normalize_doi(item.get("doi"))
            authors = [
                (authorship.get("author") or {}).get("display_name", "")
                for authorship in item.get("authorships", [])
            ]
            results.append(
                Paper(
                    id=work_id,
                    title=item.get("display_name") or item.get("title") or "Untitled",
                    abstract=abstract,
                    source="openalex",
                    url=item.get("doi") or item.get("id"),
                    year=item.get("publication_year"),
                    authors=[author for author in authors if author],
                    doi=doi,
                )
            )
        return results

    @staticmethod
    def _interleave(batches: list[list[Paper]]) -> list[Paper]:
        """Take one paper from each source in turn.

        Concatenating the batches in source order and then cutting at `limit`
        gives every slot to the earliest sources: a source that returns a full
        page leaves nothing for the ones behind it, so the later sources never
        appear in a result set at all. Round-robin keeps the merge honest and
        still preserves each source's own relevance ordering.
        """
        merged: list[Paper] = []
        for index in range(max((len(batch) for batch in batches), default=0)):
            for batch in batches:
                if index < len(batch):
                    merged.append(batch[index])
        return merged

    @staticmethod
    def _dedupe_and_limit(papers: list[Paper], limit: int) -> list[Paper]:
        # The same paper routinely comes back from both sources with a DOI on
        # one record and none on the other, so a single "doi or title" key
        # misses the duplicate. Track both identifiers independently.
        seen_dois: set[str] = set()
        seen_titles: set[str] = set()
        deduped: list[Paper] = []
        for paper in papers:
            doi = (paper.doi or "").strip().lower()
            title = " ".join(paper.title.split()).lower()
            if (doi and doi in seen_dois) or (title and title in seen_titles):
                continue
            if doi:
                seen_dois.add(doi)
            if title:
                seen_titles.add(title)
            deduped.append(paper)
            if len(deduped) >= limit:
                break
        return deduped
