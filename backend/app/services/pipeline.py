from __future__ import annotations

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

    def forward(self, question: str, paper: Paper) -> PaperExtraction:
        prediction = self.predictor(question=question, title=paper.title, abstract=paper.abstract)
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
        self.search_agent = search_agent or SearchAgent(settings.semantic_scholar_api_key)
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
                    message="Searching Semantic Scholar and arXiv.",
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
            extractions: list[PaperExtraction] = []
            for index, paper in enumerate(papers, start=1):
                await status(
                    StatusEvent(
                        event="agent_progress",
                        agent="ExtractionAgent",
                        message=f"Processing abstract {index}/{len(papers)}.",
                        report_id=report_id,
                        data={"paper_title": paper.title, "index": index, "total": len(papers)},
                    )
                )
                extractions.append(self.extraction_agent(question, paper))
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
                warnings.append(f"Artifact persistence warning: {exc}")
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
