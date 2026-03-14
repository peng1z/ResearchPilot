from __future__ import annotations

import asyncio
from typing import Optional
import xml.etree.ElementTree as ET

import httpx

from app.models import Paper, SearchResults


class SearchAgent:
    SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    ARXIV_URL = "https://export.arxiv.org/api/query"

    def __init__(self, semantic_scholar_api_key: Optional[str] = None) -> None:
        self.semantic_scholar_api_key = semantic_scholar_api_key

    async def run(self, question: str, limit: int = 10) -> SearchResults:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "ResearchPilot/0.1"},
        ) as client:
            results = await asyncio.gather(
                self._search_semantic_scholar(client, question, limit),
                self._search_arxiv(client, question, limit),
                return_exceptions=True,
            )
        warnings: list[str] = []
        paper_batches: list[list[Paper]] = []
        source_names = ["Semantic Scholar", "arXiv"]

        for source_name, result in zip(source_names, results):
            if isinstance(result, Exception):
                warnings.append(f"{source_name} search failed: {result}")
            else:
                paper_batches.append(result)

        papers = self._dedupe_and_limit([paper for batch in paper_batches for paper in batch], limit)
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
            if not abstract:
                continue
            results.append(
                Paper(
                    id=item["paperId"],
                    title=item.get("title", "Untitled"),
                    abstract=abstract,
                    source="semantic_scholar",
                    url=item.get("url"),
                    year=item.get("year"),
                    authors=[author.get("name", "") for author in item.get("authors", []) if author.get("name")],
                    doi=(item.get("externalIds") or {}).get("DOI"),
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
            year_text = entry.findtext("atom:published", default="", namespaces=ns)
            results.append(
                Paper(
                    id=paper_id.rsplit("/", maxsplit=1)[-1] or title.lower().replace(" ", "-"),
                    title=title,
                    abstract=abstract,
                    source="arxiv",
                    url=paper_id,
                    year=int(year_text[:4]) if len(year_text) >= 4 else None,
                    authors=[author for author in authors if author],
                )
            )
        return results

    @staticmethod
    def _dedupe_and_limit(papers: list[Paper], limit: int) -> list[Paper]:
        seen: set[str] = set()
        deduped: list[Paper] = []
        for paper in papers:
            key = (paper.doi or paper.title).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(paper)
            if len(deduped) >= limit:
                break
        return deduped
