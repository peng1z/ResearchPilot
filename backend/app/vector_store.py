from __future__ import annotations

import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.embeddings import EmbeddingService
from app.models import Paper, PaperExtraction, ReportSearchHit, ReportSummary, ResearchReport


class QdrantArtifactStore:
    def __init__(self, url: str, embedder: Optional[EmbeddingService] = None) -> None:
        self.client = QdrantClient(url=url)
        self.embedder = embedder
        suffix = "dense" if embedder and embedder.enabled else "artifacts"
        self.papers_collection = f"researchpilot_papers_{suffix}"
        self.reports_collection = f"researchpilot_reports_{suffix}"

    def _ensure_collection(self, name: str, size: int) -> None:
        collections = {item.name for item in self.client.get_collections().collections}
        if name not in collections:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )

    async def _paper_vectors(self, papers: list[Paper]) -> list[list[float]]:
        if self.embedder and self.embedder.enabled:
            texts = [f"{paper.title}\n\n{paper.abstract}" for paper in papers]
            return await self.embedder.embed_texts(texts)
        return [[float(index)] for index, _ in enumerate(papers, start=1)]

    async def _report_vector(self, report: ResearchReport) -> list[float]:
        if self.embedder and self.embedder.enabled:
            text = "\n\n".join(
                [
                    report.question,
                    "\n".join(report.synthesis.consensus),
                    "\n".join(report.synthesis.contradictions),
                    "\n".join(report.synthesis.open_gaps),
                    report.related_work_markdown,
                ]
            )
            return (await self.embedder.embed_texts([text]))[0]
        return [1.0]

    async def store_papers(
        self,
        report_id: str,
        papers: list[Paper],
        extractions: list[PaperExtraction],
    ) -> None:
        vectors = await self._paper_vectors(papers)
        vector_size = len(vectors[0]) if vectors else 1
        self._ensure_collection(self.papers_collection, vector_size)
        extraction_lookup = {item.paper_id: item for item in extractions}
        points = []
        for paper, vector in zip(papers, vectors):
            extraction = extraction_lookup.get(paper.id)
            points.append(
                PointStruct(
                    id=self._stable_uuid(report_id, paper.id),
                    vector=vector,
                    payload={
                        "report_id": report_id,
                        "paper": paper.model_dump(mode="json"),
                        "extraction": extraction.model_dump(mode="json") if extraction else None,
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=self.papers_collection, points=points)

    async def store_report(self, report: ResearchReport) -> None:
        vector = await self._report_vector(report)
        self._ensure_collection(self.reports_collection, len(vector))
        point = PointStruct(
            id=self._stable_uuid("report", report.id),
            vector=vector,
            payload=report.model_dump(mode="json"),
        )
        self.client.upsert(collection_name=self.reports_collection, points=[point])

    async def search_reports(self, query: str, limit: int = 5) -> list[ReportSearchHit]:
        vector = await self.embed_query(query)
        self._ensure_collection(self.reports_collection, len(vector))
        results = self.client.search(
            collection_name=self.reports_collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
        hits: list[ReportSearchHit] = []
        for result in results:
            payload = result.payload or {}
            report = ResearchReport.model_validate(payload)
            hits.append(
                ReportSearchHit(
                    report=ReportSummary(
                        id=report.id,
                        question=report.question,
                        created_at=report.created_at,
                        paper_count=len(report.papers),
                        warning_count=len(report.warnings),
                    ),
                    score=float(result.score or 0.0),
                )
            )
        return hits

    async def embed_query(self, query: str) -> list[float]:
        if self.embedder and self.embedder.enabled:
            return (await self.embedder.embed_texts([query]))[0]
        return [1.0]

    @staticmethod
    def _stable_uuid(*parts: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "::".join(parts)))
