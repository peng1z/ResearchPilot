from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import Awaitable, Callable, Optional

os.environ.setdefault("DSPY_CACHEDIR", os.path.join(tempfile.gettempdir(), "researchpilot-dspy-cache"))

import dspy

from app.config import Settings
from app.embeddings import EmbeddingService
from app.llm import dspy_context, parse_json_payload
from app.models import Paper, PaperExtraction, RelatedWorkDraft, ReportReference, ResearchReport, SearchResults, StatusEvent, SynthesisOutput
from app.services.search import SearchAgent
from app.vector_store import QdrantArtifactStore

StatusCallback = Callable[[StatusEvent], Awaitable[None]]


class ExtractionSchema(PaperExtraction):
    paper_id: str = ""
    title: str = ""


class SynthesisSchema(SynthesisOutput):
    pass


class RelatedWorkSchema(RelatedWorkDraft):
    pass


class ExtractionSignature(dspy.Signature):
    """Extract structured findings from a research paper abstract. Return JSON only."""

    question = dspy.InputField()
    title = dspy.InputField()
    abstract = dspy.InputField()
    extraction_json = dspy.OutputField(
        desc="JSON object with arrays: claims, methods, datasets, results, limitations."
    )


class SynthesisSignature(dspy.Signature):
    """Synthesize multiple structured paper summaries. Return JSON only."""

    question = dspy.InputField()
    extractions_json = dspy.InputField()
    synthesis_json = dspy.OutputField(
        desc="JSON object with arrays: consensus, contradictions, open_gaps."
    )


class WriterSignature(dspy.Signature):
    """Write an academic-style related work section in markdown with inline citation labels."""

    question = dspy.InputField()
    synthesis_json = dspy.InputField()
    bibliography_json = dspy.InputField()
    markdown = dspy.OutputField(
        desc="Markdown only. Use inline citation labels like [R1], [R2]. End with a '## References' section listing the labels."
    )


class ExtractionAgent(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(ExtractionSignature)

    async def aextract(self, question: str, paper: Paper) -> PaperExtraction:
        prediction = await self.predictor.acall(
            question=question, title=paper.title, abstract=paper.abstract
        )
        return self._to_extraction(prediction, paper)

    def forward(self, question: str, paper: Paper) -> PaperExtraction:
        prediction = self.predictor(question=question, title=paper.title, abstract=paper.abstract)
        return self._to_extraction(prediction, paper)

    @staticmethod
    def _to_extraction(prediction, paper: Paper) -> PaperExtraction:
        parsed = parse_json_payload(
            prediction.extraction_json,
            ExtractionSchema,
        )
        return PaperExtraction(
            paper_id=paper.id,
            title=paper.title,
            claims=parsed.claims,
            methods=parsed.methods,
            datasets=parsed.datasets,
            results=parsed.results,
            limitations=parsed.limitations,
        )


class SynthesisAgent(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(SynthesisSignature)

    def forward(self, question: str, extractions: list[PaperExtraction]) -> SynthesisOutput:
        prediction = self.predictor(
            question=question,
            extractions_json=json.dumps([item.model_dump(mode="json") for item in extractions], indent=2),
        )
        return parse_json_payload(prediction.synthesis_json, SynthesisSchema)


class WriterAgent(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(WriterSignature)

    def forward(self, question: str, synthesis: SynthesisOutput, papers: list[Paper]) -> RelatedWorkDraft:
        bibliography = [
            {
                "label": f"R{index}",
                "paper_id": paper.id,
                "title": paper.title,
                "year": paper.year,
                "source": paper.source,
                "url": paper.url,
            }
            for index, paper in enumerate(papers, start=1)
        ]
        prediction = self.predictor(
            question=question,
            synthesis_json=json.dumps(synthesis.model_dump(mode="json"), indent=2),
            bibliography_json=json.dumps(bibliography, indent=2),
        )
        return RelatedWorkDraft(markdown=prediction.markdown.strip())


class ResearchPipeline:
    def __init__(
        self,
        settings: Settings,
        search_agent: Optional[SearchAgent] = None,
        extraction_agent: Optional[ExtractionAgent] = None,
        synthesis_agent: Optional[SynthesisAgent] = None,
        writer_agent: Optional[WriterAgent] = None,
        artifact_store: Optional[QdrantArtifactStore] = None,
    ) -> None:
        self.settings = settings
        self.embedder = EmbeddingService(settings)
        self.search_agent = search_agent or SearchAgent(settings.semantic_scholar_api_key, settings.openalex_mailto)
        self.extraction_agent = extraction_agent or ExtractionAgent()
        self.synthesis_agent = synthesis_agent or SynthesisAgent()
        self.writer_agent = writer_agent or WriterAgent()
        self.artifact_store = artifact_store or QdrantArtifactStore(settings.qdrant_url, embedder=self.embedder)

    async def run(self, report_id: str, question: str, status: StatusCallback) -> ResearchReport:
        with dspy_context(self.settings):
            await status(StatusEvent(event="queued", message="Research request accepted.", report_id=report_id))

            await status(
                StatusEvent(
                    event="agent_started",
                    agent="SearchAgent",
                    message="Searching Semantic Scholar, arXiv and OpenAlex.",
                    report_id=report_id,
                )
            )
            search_results: SearchResults = await self.search_agent.run(question, limit=10)
            papers = search_results.papers
            warnings = list(search_results.warnings)
            await status(
                StatusEvent(
                    event="agent_completed",
                    agent="SearchAgent",
                    message=f"Collected {len(papers)} unique papers.",
                    report_id=report_id,
                    data={"paper_count": len(papers), "warnings": search_results.warnings},
                )
            )
            for warning in search_results.warnings:
                await status(
                    StatusEvent(
                        event="agent_progress",
                        agent="SearchAgent",
                        message=warning,
                        report_id=report_id,
                        data={"warning": True},
                    )
                )

            await status(
                StatusEvent(
                    event="agent_started",
                    agent="ExtractionAgent",
                    message="Extracting structured findings from abstracts.",
                    report_id=report_id,
                )
            )
            # One LLM call per paper, and they do not depend on each other.
            # Run them together, bounded so a large result set cannot burst
            # past the provider's rate limit. Progress is reported on
            # completion, so the order of the events is not the order of the
            # papers; `index` says which paper each event is about.
            gate = asyncio.Semaphore(self.settings.extraction_concurrency)
            completed = 0

            async def extract(index: int, paper: Paper) -> PaperExtraction:
                nonlocal completed
                async with gate:
                    extraction = await self.extraction_agent.aextract(question, paper)
                completed += 1
                await status(
                    StatusEvent(
                        event="agent_progress",
                        agent="ExtractionAgent",
                        message=f"Processed {completed}/{len(papers)} abstracts.",
                        report_id=report_id,
                        data={
                            "paper_title": paper.title,
                            "index": index,
                            "total": len(papers),
                            "completed": completed,
                        },
                    )
                )
                return extraction

            extractions: list[PaperExtraction] = list(
                await asyncio.gather(
                    *(extract(index, paper) for index, paper in enumerate(papers, start=1))
                )
            )
            await status(
                StatusEvent(
                    event="agent_completed",
                    agent="ExtractionAgent",
                    message=f"Extracted structured summaries for {len(extractions)} papers.",
                    report_id=report_id,
                )
            )

            await status(
                StatusEvent(
                    event="agent_started",
                    agent="SynthesisAgent",
                    message="Synthesizing consensus, contradictions, and gaps.",
                    report_id=report_id,
                )
            )
            synthesis = self.synthesis_agent(question, extractions)
            await status(
                StatusEvent(
                    event="agent_completed",
                    agent="SynthesisAgent",
                    message="Synthesis completed.",
                    report_id=report_id,
                )
            )

            await status(
                StatusEvent(
                    event="agent_started",
                    agent="WriterAgent",
                    message="Drafting related work section.",
                    report_id=report_id,
                )
            )
            draft = self.writer_agent(question, synthesis, papers)
            await status(
                StatusEvent(
                    event="agent_completed",
                    agent="WriterAgent",
                    message="Related work draft completed.",
                    report_id=report_id,
                )
            )

            report = ResearchReport(
                id=report_id,
                question=question,
                papers=papers,
                extractions=extractions,
                synthesis=synthesis,
                related_work_markdown=draft.markdown,
                warnings=warnings,
                references=[
                    ReportReference(
                        label=f"R{index}",
                        paper_id=paper.id,
                        title=paper.title,
                        source=paper.source,
                        year=paper.year,
                        url=paper.url,
                    )
                    for index, paper in enumerate(papers, start=1)
                ],
            )
            try:
                await self.artifact_store.store_papers(report.id, papers, extractions)
                await self.artifact_store.store_report(report)
            except Exception as exc:
                # Append to the report's own list. `warnings` was copied by
                # pydantic during construction above, so appending there left
                # the stored and returned report with no record of the failure
                # -- only a transient status event nobody could go back to.
                report.warnings.append(f"Artifact persistence warning: {exc}")
                await status(
                    StatusEvent(
                        event="agent_progress",
                        agent="QdrantArtifactStore",
                        message=f"Artifact persistence warning: {exc}",
                        report_id=report_id,
                        data={"warning": True},
                    )
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
