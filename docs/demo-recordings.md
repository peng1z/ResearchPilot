# Demo recordings

`frontend/demo/*.json` are real pipeline outputs, captured end to end and
shipped with the page. The hosted demo renders one on first paint, so it
needs no backend, no API key and no running cost, and a visitor sees a
finished report immediately rather than a two-minute progress bar.

They are fixtures, not fakes: every claim, contradiction and reference in
them came out of an actual run against the live Semantic Scholar, arXiv and
OpenAlex APIs and a live model.

## Recapturing them

Recordings should be refreshed whenever the report schema changes, or when
a pipeline change makes the old output unrepresentative.

1. Configure `backend/.env` with a working `LLM_API_KEY`.
2. Run the pipeline for each question and write the raw result to
   `demo-recordings/<slug>.json`.
3. Convert to fixtures under `frontend/demo/`, dropping warnings that
   describe the recording machine rather than the system.

Step 3 currently drops `Artifact persistence warning: ...`, which only
appears because Qdrant runs under docker-compose and was not up on the
recording machine. Source failures are deliberately **kept**: the page
presents them as evidence that the sources are queried in parallel and
allowed to fail independently, which is the behaviour the pipeline is
built for.

## Typing

Fixtures are imported through `frontend/demo/index.ts` and typed as
`DemoRun`, whose `report` is the same `ResearchReport` the live code path
uses. A backend schema change therefore fails `tsc` at build time instead
of producing a broken page.
