"use client";

import React, { FormEvent, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

import { citedInDraft } from "./cited";
import { checksFor } from "../demo/checks";
import type {
  PublicRuntimeConfig,
  ReportSearchHit,
  ReportSummary,
  ResearchReport,
  RuntimeSettings,
  StatusEvent,
} from "./types";
import { demoRuns } from "../demo";
import type { DemoRun } from "../demo/types";

// Resolved at build time, and deliberately left empty when unset rather than
// defaulting to localhost. The hosted demo is served over https, where a
// request to http://localhost:8000 is blocked as mixed content -- silently,
// because the passive loads swallow their errors. An empty base means the
// page makes no backend requests at all and runs purely on the recorded
// runs; docker-compose and `npm run dev` supply the value through
// NEXT_PUBLIC_API_BASE.
const buildTimeApiBase = process.env.NEXT_PUBLIC_API_BASE ?? "";

/** Reject a base the browser will refuse to call from the current page. */
function mixedContentWarning(base: string): string | null {
  if (typeof window === "undefined" || !base.startsWith("http://")) {
    return null;
  }
  if (window.location.protocol !== "https:") {
    return null;
  }
  const host = (() => {
    try {
      return new URL(base).hostname;
    } catch {
      return "";
    }
  })();
  if (host === "localhost" || host === "127.0.0.1") {
    return null;
  }
  return "This page is served over https, so the browser will block a plain http backend. Use an https address.";
}

const SOURCE_LABELS: Record<string, string> = {
  arxiv: "arXiv",
  openalex: "OpenAlex",
  semantic_scholar: "Semantic Scholar",
};

const ALL_SOURCES = ["semantic_scholar", "arxiv", "openalex"];

/* What opening a drawer costs, so the label is a decision and not a dare. */
function readingMinutes(markdown: string): number {
  return Math.max(1, Math.round(markdown.trim().split(/\s+/).length / 220));
}

/* The three claims a synthesis makes, in the order the argument moves. */
const SYNTHESIS_PARTS = [
  { key: "consensus", label: "Consensus" },
  { key: "contradictions", label: "Contradictions" },
  { key: "open_gaps", label: "Open gaps" },
] as const;

/** Papers per source, plus the sources that returned nothing and why. */
function describeSources(report: ResearchReport) {
  const counts = new Map<string, number>();
  for (const paper of report.papers) {
    counts.set(paper.source, (counts.get(paper.source) ?? 0) + 1);
  }
  const contributed = ALL_SOURCES.filter((source) => counts.has(source)).map((source) => ({
    source,
    label: SOURCE_LABELS[source] ?? source,
    count: counts.get(source) ?? 0,
  }));
  const failed = report.warnings
    .map((warning) => /^(.+?) search failed: (.+)$/s.exec(warning))
    .filter((match): match is RegExpExecArray => match !== null)
    .map((match) => ({ label: match[1], reason: match[2] }));
  return { contributed, failed };
}

function splitSseChunks(buffer: string): { events: string[]; remainder: string } {
  const chunks = buffer.split("\n\n");
  const remainder = chunks.pop() ?? "";
  return { events: chunks, remainder };
}

function parseSseEvent(raw: string): StatusEvent | null {
  const lines = raw.split("\n");
  let dataLine = "";
  for (const line of lines) {
    if (line.startsWith("data:")) {
      dataLine = line.slice(5).trim();
    }
  }
  if (!dataLine) {
    return null;
  }
  return JSON.parse(dataLine) as StatusEvent;
}

// Verbatim from https://peng1z.github.io/publications/researchpilot/citation.bib,
// which is the authority. A hand-written second copy drifts: mine had cs.CL
// where the canonical entry says cs.IR, and no DOI, until it was compared.
const BIBTEX = `@misc{zhang2026researchpilot,
  title = {{ResearchPilot: A Local-First Multi-Agent System for Literature Synthesis and Related Work Drafting}},
  author = {Peng Zhang},
  year = {2026},
  eprint = {2603.14629},
  archivePrefix = {arXiv},
  primaryClass = {cs.IR},
  doi = {10.48550/arXiv.2603.14629},
  url = {https://arxiv.org/abs/2603.14629},
  note = {Version 1, preprint}
}`;

/**
 * The paper this artifact accompanies, and how to cite it.
 *
 * A footer strip rather than a header: the recorded report above is what a
 * visitor came for. But the site carried no link to the paper at all, so a
 * reader who wanted to cite the work had nowhere to go.
 */
function Citation() {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(BIBTEX);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard access can be refused; the entry is on screen to select.
      setCopied(false);
    }
  }

  return (
    <footer id="cite" className="border-t border-[var(--rule)] pt-8 mt-4">
      <h2 className="text-xl font-semibold">Cite this work</h2>
      <p className="mt-3 max-w-3xl leading-7 text-[var(--ink-2)]">
        This page is the artifact for{" "}
        <a className="underline" href="https://arxiv.org/abs/2603.14629">
          ResearchPilot: A Local-First Multi-Agent System for Literature Synthesis and Related
          Work Drafting
        </a>
        , Peng Zhang. arXiv:2603.14629, version 1 preprint, 15 March 2026. DOI{" "}
        <a className="underline" href="https://doi.org/10.48550/arXiv.2603.14629">
          10.48550/arXiv.2603.14629
        </a>
        .
      </p>
      <details className="drawer">
        <summary>Where the paper and this build differ</summary>
        <div className="drawer-body">
          <p className="max-w-3xl leading-7 text-[var(--ink-2)]">
            Version 1 of the paper describes retrieval from Semantic Scholar and arXiv. The
            recordings here were produced by a later build that also queries OpenAlex, and the
            OpenAlex results in them are not part of what the paper reports. Where the two
            differ, the code and the recorded runs describe this build; the paper describes
            version 1.
          </p>
          <p className="mt-3 max-w-3xl leading-7 text-[var(--ink-2)]">
            The method paper belongs in a description of how a synthesis was produced. It is not
            a source for any topic above and does not belong in the reference list of a review
            drafted with it.
          </p>
        </div>
      </details>
      <div className="mt-4 flex flex-wrap gap-3">
        <a
          href="https://arxiv.org/abs/2603.14629"
          className="btn"
        >
          Abstract
        </a>
        <a
          href="https://arxiv.org/pdf/2603.14629"
          className="btn"
        >
          PDF
        </a>
        <a
          href="https://peng1z.github.io/publications/researchpilot/"
          className="btn"
        >
          Paper page
        </a>
        <a
          href="https://peng1z.github.io/publications/researchpilot/citation.bib"
          className="btn"
        >
          BibTeX file
        </a>
        <a
          href="https://peng1z.github.io/publications/researchpilot/citation.ris"
          className="btn"
        >
          RIS
        </a>
        <button
          type="button"
          onClick={copy}
          className="btn"
        >
          {copied ? "BibTeX copied" : "Copy BibTeX"}
        </button>
      </div>
      <details className="drawer">
        <summary>BibTeX</summary>
        <pre className="drawer-body overflow-x-auto text-xs leading-5">{BIBTEX}</pre>
      </details>
    </footer>
  );
}

function SourcePanel({ report }: { report: ResearchReport }) {
  const { contributed, failed } = describeSources(report);
  const other = report.warnings.filter((warning) => !/ search failed: /.test(warning));

  return (
    <div className="group">
      <h3>Retrieval</h3>
      <p className="mt-2">
        {contributed.map((entry) => (
          <span key={entry.source} className="tally">
            <span>{entry.label}</span> <span className="font-semibold">{entry.count}</span>
          </span>
        ))}
        {failed.map((entry) => (
          <span key={entry.label} className="tally text-[var(--ink-3)]" title={entry.reason}>
            {entry.label} unavailable
          </span>
        ))}
      </p>
      {failed.length > 0 ? (
        <p className="mt-3 text-sm leading-6 text-[var(--ink-2)]">
          The sources are queried in parallel and each one is allowed to fail on its own. {" "}
          {failed.map((entry) => entry.label).join(" and ")} returned an error on this run, so the
          report was built from the {contributed.length} that answered, and the failure is recorded
          on the report rather than discarded.
        </p>
      ) : null}
      {other.length > 0 ? (
        <ul className="mt-3 space-y-2 text-sm text-[var(--ink-2)]">
          {other.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

export default function Home() {
  const [question, setQuestion] = useState(demoRuns[0].question);
  const [events, setEvents] = useState<StatusEvent[]>([]);
  const [report, setReport] = useState<ResearchReport | null>(demoRuns[0].report);
  const [demo, setDemo] = useState<DemoRun | null>(demoRuns[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<ReportSummary[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<ReportSearchHit[]>([]);
  const [config, setConfig] = useState<PublicRuntimeConfig | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [runtime, setRuntime] = useState<RuntimeSettings>({});
  const [apiBase, setApiBase] = useState(buildTimeApiBase);

  useEffect(() => {
    // The same guard the live run uses: a base the browser will refuse to
    // call must not be probed passively either, or typing one into the
    // field fires two requests that are blocked before they leave.
    if (!apiBase || mixedContentWarning(apiBase)) {
      return;
    }
    void loadHistory();
    void loadConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase]);

  async function loadHistory() {
    try {
      const response = await fetch(`${apiBase}/reports`);
      if (!response.ok) {
        return;
      }
      const payload = (await response.json()) as ReportSummary[];
      setHistory(payload);
    } catch {
      // Ignore passive history-load failures in the UI.
    }
  }

  async function loadConfig() {
    try {
      const response = await fetch(`${apiBase}/config`);
      if (!response.ok) {
        return;
      }
      const payload = (await response.json()) as PublicRuntimeConfig;
      setConfig(payload);
      setRuntime({
        llm_provider: payload.llm_provider,
        llm_model: payload.llm_model,
        llm_base_url: payload.llm_base_url ?? "",
        llm_temperature: payload.llm_temperature,
        embedding_backend: payload.embedding_backend,
        embedding_model: payload.embedding_model ?? "",
        local_embedding_model: payload.local_embedding_model,
      });
    } catch {
      // Ignore passive config-load failures in the UI.
    }
  }

  async function loadReport(reportId: string) {
    try {
      const response = await fetch(`${apiBase}/report/${reportId}`);
      if (!response.ok) {
        throw new Error("Failed to load saved report.");
      }
      const payload = (await response.json()) as ResearchReport;
      setReport(payload);
      setDemo(null);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    }
  }

  function showDemo(run: DemoRun) {
    setDemo(run);
    setReport(run.report);
    setQuestion(run.question);
    setEvents([]);
    setError(null);
  }

  async function runResearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!apiBase) {
      setError(
        "No backend configured. This is the hosted demo, which ships recorded runs and calls nothing. To run a live question, open Run Settings and point API Base at a ResearchPilot backend you control.",
      );
      return;
    }
    const mixedContent = mixedContentWarning(apiBase);
    if (mixedContent) {
      setError(mixedContent);
      return;
    }
    setLoading(true);
    setEvents([]);
    setReport(null);
    setDemo(null);
    setError(null);

    try {
      const response = await fetch(`${apiBase}/research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, runtime }),
      });
      if (!response.ok || !response.body) {
        throw new Error("Failed to start research pipeline.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const { events: rawEvents, remainder } = splitSseChunks(buffer);
        buffer = remainder;

        for (const rawEvent of rawEvents) {
          const parsed = parseSseEvent(rawEvent);
          if (!parsed) {
            continue;
          }
          setEvents((current) => [...current, parsed]);
          if (parsed.event === "done" && parsed.data?.report) {
            setReport(parsed.data.report as ResearchReport);
            void loadHistory();
          }
          if (parsed.event === "error") {
            throw new Error(parsed.message);
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  function updateRuntime<K extends keyof RuntimeSettings>(key: K, value: RuntimeSettings[K]) {
    setRuntime((current) => ({ ...current, [key]: value }));
  }

  async function searchHistory(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!apiBase || !searchQuery.trim()) {
      setSearchResults([]);
      return;
    }
    try {
      const response = await fetch(`${apiBase}/reports/search?query=${encodeURIComponent(searchQuery)}&limit=5`);
      if (!response.ok) {
        throw new Error("Failed to search report history.");
      }
      const payload = (await response.json()) as ReportSearchHit[];
      setSearchResults(payload);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    }
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-[46rem] flex-col gap-8 px-6 py-10 md:px-8">
      <section className="border-b border-[var(--rule)] pb-10">
        {/* A running head, the way an offprint carries one. The review tool
            opens with a dark title bar; a document does not have chrome. */}
        <div className="runninghead mb-10">
          <span>ResearchPilot · arXiv:2603.14629</span>
          <nav aria-label="Primary">
            <a href="https://arxiv.org/abs/2603.14629">Paper</a>
            <a href="https://github.com/peng1z/ResearchPilot">Code</a>
            <a href="#recorded-runs">Examples</a>
            <a href="#cite">Cite</a>
          </nav>
        </div>
        <div>
          <div>
            <h1 className="max-w-3xl text-4xl font-semibold leading-tight md:text-6xl">
              A multi-agent research co-pilot for fast literature synthesis.
            </h1>
            <p className="mt-4 max-w-2xl text-lg leading-8 text-[var(--ink-2)]">
              Give it one research question. It searches Semantic Scholar, arXiv and OpenAlex in
              parallel, extracts structured findings from each abstract, synthesises consensus,
              contradictions and open gaps across them, and drafts a citation-aware related work
              section.
            </p>
            <p className="mt-4 max-w-2xl text-lg leading-8 text-[var(--ink-2)]">
              <strong className="text-[var(--ink)]">
                You are reading recorded runs, not live ones.
              </strong>{" "}
              {apiBase
                ? "A backend is configured, so Start Research will run a real one."
                : "Browsing them needs no API key and no backend, and the page requests nothing. To run your own question, open Run Settings and point it at a backend you host."}
            </p>
            <div id="recorded-runs" className="mt-6">
              <h2 className="text-base">Recorded runs</h2>
              <div className="mt-3 border-b border-[var(--rule)]">
                {demoRuns.map((run) => (
                  // The badge sits beside the button, not inside it: in the
                  // button it became part of the accessible name, so the
                  // control announced itself as the question plus the word
                  // "checked". Outside, the name stays the question and the
                  // status is still read.
                  <div
                    key={run.slug}
                    className="entry"
                    style={{ display: "flex", alignItems: "baseline", gap: 10, padding: 0 }}
                  >
                    <button
                      type="button"
                      onClick={() => showDemo(run)}
                      aria-pressed={demo?.slug === run.slug}
                      className={demo?.slug === run.slug ? "entry-selected" : "entry-muted"}
                      style={{
                        flex: 1,
                        textAlign: "left",
                        background: "none",
                        border: 0,
                        padding: "12px 0",
                        cursor: "pointer",
                        fontFamily: "var(--serif)",
                      }}
                    >
                      {run.question}
                    </button>
                    {checksFor(run.slug) ? (
                      <span className="tier" title="Claims checked against their sources">
                        checked
                      </span>
                    ) : null}
                  </div>
                ))}
              </div>
              <p className="mt-4 max-w-2xl text-sm text-[var(--ink-2)]">
                These are real pipeline outputs, captured end to end and shipped with the page, so
                the demo costs nothing to run and needs no server. To run your own question live,
                add an API key under <span className="font-semibold">Run Settings</span>.
              </p>
              <p className="mt-2 max-w-2xl text-sm text-[var(--ink-2)]">
                Each run also has its own page, with the retrieval, the papers, the synthesis and
                the limits of that particular run:{" "}
                {demoRuns.map((run, index) => (
                  <span key={run.slug}>
                    {index > 0 ? ", " : ""}
                    <a className="underline" href={`/runs/${run.slug}/`}>
                      {run.slug}
                    </a>
                  </span>
                ))}
                .
              </p>
            </div>
          </div>
          <details className="mt-10 border-t border-[var(--rule)] pt-6">
            <summary className="cursor-pointer font-semibold">
              Run your own question {apiBase ? "" : "(needs a backend you run)"}
            </summary>
            <p className="mt-3 text-sm leading-6 text-[var(--ink-2)]">
              {apiBase
                ? "A backend is configured. Starting a run will search live sources and call your model."
                : "This deployment hosts no backend and starts nothing. Point API Base at a ResearchPilot backend you run, and supply a key it can use."}
            </p>
          <form onSubmit={runResearch} className="mt-4">
            <label htmlFor="question" className="label">
              Research Question
            </label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={6}
              className="control"
            />
            <button
              type="button"
              onClick={() => setShowSettings((current) => !current)}
              className="btn mt-4"
            >
              {showSettings ? "Hide Settings" : "Run Settings"}
            </button>
            {showSettings ? (
              <div className="mt-5 space-y-3 border-t border-[var(--rule)] pt-5">
                <div className="grid gap-3 md:grid-cols-2">
                  <label className="text-sm">
                    <span className="label">Provider</span>
                    <select
                      value={runtime.llm_provider ?? ""}
                      onChange={(event) => updateRuntime("llm_provider", event.target.value)}
                      className="control"
                    >
                      <option value="openai">OpenAI</option>
                      <option value="anthropic">Anthropic</option>
                      <option value="groq">Groq</option>
                      <option value="openrouter">OpenRouter</option>
                    </select>
                  </label>
                  <label className="text-sm">
                    <span className="label">Model</span>
                    <input
                      value={runtime.llm_model ?? ""}
                      onChange={(event) => updateRuntime("llm_model", event.target.value)}
                      className="control"
                    />
                  </label>
                  <label className="text-sm md:col-span-2">
                    <span className="label">
                      ResearchPilot API Base
                    </span>
                    <input
                      value={apiBase}
                      onChange={(event) => setApiBase(event.target.value.trim())}
                      placeholder="https://your-researchpilot-backend.example.com"
                      className="control"
                    />
                    <span className="mt-1 block text-xs leading-5 text-[var(--ink-2)]">
                      {apiBase
                        ? (mixedContentWarning(apiBase) ??
                          "Live runs, saved history and report search will use this backend.")
                        : "Empty: the page runs entirely on the recorded runs above and makes no requests. Point this at a backend you control to run a live question."}
                    </span>
                  </label>
                  <label className="text-sm">
                    <span className="label">API Key</span>
                    <input
                      type="password"
                      value={runtime.llm_api_key ?? ""}
                      onChange={(event) => updateRuntime("llm_api_key", event.target.value)}
                      placeholder="Optional per-run override"
                      className="control"
                    />
                  </label>
                  <label className="text-sm">
                    <span className="label">Base URL</span>
                    <input
                      value={runtime.llm_base_url ?? ""}
                      onChange={(event) => updateRuntime("llm_base_url", event.target.value)}
                      className="control"
                    />
                  </label>
                  <label className="text-sm">
                    <span className="label">Embedding Backend</span>
                    <select
                      value={runtime.embedding_backend ?? "auto"}
                      onChange={(event) => updateRuntime("embedding_backend", event.target.value)}
                      className="control"
                    >
                      <option value="auto">auto</option>
                      <option value="remote">remote</option>
                      <option value="local">local</option>
                    </select>
                  </label>
                  <label className="text-sm">
                    <span className="label">Embedding Model</span>
                    <input
                      value={runtime.embedding_model ?? ""}
                      onChange={(event) => updateRuntime("embedding_model", event.target.value)}
                      className="control"
                    />
                  </label>
                  <label className="text-sm md:col-span-2">
                    <span className="label">Local Embedding Model</span>
                    <input
                      value={runtime.local_embedding_model ?? ""}
                      onChange={(event) => updateRuntime("local_embedding_model", event.target.value)}
                      className="control"
                    />
                  </label>
                </div>
                <p className="text-xs leading-5 text-[var(--ink-2)]">
                  Settings apply to this run only. Current server default: {config ? `${config.llm_provider}/${config.llm_model}` : "loading..."}.
                </p>
              </div>
            ) : null}
            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary mt-5 w-full"
            >
              {loading ? "Running Pipeline" : "Start Research"}
            </button>
            <p className="mt-4 text-sm leading-6 text-[var(--ink-2)]">
              Backend streams agent lifecycle events over SSE and returns the final report as structured JSON plus markdown.
            </p>
          </form>
          <div className="group">
          <div className="mb-4 flex items-baseline gap-3">
            <h2>Agent feed</h2>
            <span className="text-sm text-[var(--ink-3)]">{loading ? "live" : "idle"}</span>
          </div>

          <div>
            {events.length === 0 ? (
              <p className="text-sm text-[var(--ink-2)]">No events yet.</p>
            ) : (
              events.map((item, index) => (
                <div key={`${item.event}-${index}`} className="entry">
                  <div className="flex items-center justify-between gap-3">
                    <strong className="text-sm">{item.agent ?? item.event}</strong>
                    <span className="text-xs text-[var(--ink-3)]">{item.event}</span>
                  </div>
                  <p className="mt-2 text-sm leading-6 text-[var(--ink-2)]">{item.message}</p>
                </div>
              ))
            )}
          </div>

          {error ? <p className="mt-4 text-sm text-red-700">{error}</p> : null}

          <div className="group">
            <h3>Report History</h3>
            <form onSubmit={searchHistory} className="mt-3 flex gap-2">
              <input
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Search past reports"
                className="control min-w-0 flex-1"
              />
              <button
                type="submit"
                className="btn"
              >
                Search
              </button>
            </form>

            {searchResults.length > 0 ? (
              <div className="mt-4">
                {searchResults.map(({ report: item, score }) => (
                  <button
                    key={`search-${item.id}`}
                    type="button"
                    onClick={() => void loadReport(item.id)}
                    className="entry"
                  >
                    <p className="text-sm font-semibold text-[var(--ink)]">{item.question}</p>
                    <p className="mt-1 text-xs text-[var(--ink-3)]">
                      score {score.toFixed(2)} • {item.paper_count} papers
                    </p>
                  </button>
                ))}
              </div>
            ) : null}

            <div className="mt-4">
              {history.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => void loadReport(item.id)}
                  className="entry"
                >
                  <p className="text-sm font-semibold text-[var(--ink)]">{item.question}</p>
                  <p className="mt-1 text-xs text-[var(--ink-3)]">
                    {new Date(item.created_at).toLocaleString()} • {item.paper_count} papers
                    {item.warning_count > 0 ? ` • ${item.warning_count} warnings` : ""}
                  </p>
                </button>
              ))}
              {history.length === 0 ? <p className="text-sm text-[var(--ink-2)]">No saved reports yet.</p> : null}
            </div>
          </div>
          </div>
          </details>
        </div>
      </section>

      <section className="numbered">
        <h2 className="text-2xl font-semibold" style={{ marginTop: 0 }}>
          This run
        </h2>
        {report ? (
          <div className="mt-4">
            <p className="text-[var(--ink-2)]" style={{ marginTop: 0 }}>
              {report.question}
            </p>

            {/* What the run cost and covered, before any of its prose. A
                reader deciding whether to trust the synthesis is asking how
                many papers it rests on and which sources answered. */}
            <dl className="particulars">
              <div className="particular">
                <dt>Papers</dt>
                <dd>{report.papers.length}</dd>
              </div>
              {citedInDraft(report.related_work_markdown) !== null ? (
                <div className="particular">
                  <dt>Cited</dt>
                  <dd>{citedInDraft(report.related_work_markdown)}</dd>
                </div>
              ) : null}
              <div className="particular">
                <dt>Sources</dt>
                <dd>{describeSources(report).contributed.length} of 3</dd>
              </div>
              <div className="particular">
                <dt>Model</dt>
                <dd>{report.tool?.model ?? "not recorded"}</dd>
              </div>
            </dl>

            {/* The synthesis is what a visitor came to judge, so it is the one
                thing that stays open. Everything it rests on is a drawer. */}
            <div className="group">
              <h3>Synthesis</h3>
              {demo && checksFor(demo.slug) ? (
                <p className="mt-2 text-sm text-[var(--ink-3)]">
                  Every claim below was checked against the papers it rests on, by an AI agent and
                  not by a human expert. The verdicts, the evidence and what could not be verified
                  are on{" "}
                  <a href={`/runs/${demo.slug}/`}>the page for this run</a>.
                </p>
              ) : (
                <p className="mt-2 text-sm text-[var(--ink-3)]">
                  Generated from the abstracts below and shipped unedited. No claim here was
                  checked against its source.
                </p>
              )}
              {SYNTHESIS_PARTS.map(({ key, label }) => {
                const items = report.synthesis[key];
                if (items.length === 0) {
                  return null;
                }
                return (
                  <div key={key} className="claim" data-kind={key}>
                    <p className="label">{label}</p>
                    <ul className="mt-1 list-disc space-y-2 pl-5 text-[var(--ink-2)]">
                      {items.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                );
              })}
            </div>

            <details className="drawer numbered">
              <summary>
                Related work draft
                <span className="drawer-count">
                  {readingMinutes(report.related_work_markdown)} min read
                </span>
              </summary>
              <div className="drawer-body markdown prose prose-neutral max-w-none">
                <ReactMarkdown>{report.related_work_markdown}</ReactMarkdown>
              </div>
            </details>

            <details className="drawer numbered">
              <summary>
                Retrieved papers
                <span className="drawer-count">{report.papers.length} papers</span>
              </summary>
              <div className="drawer-body">
                {report.references.map((reference) => (
                  <div key={reference.paper_id} className="entry">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="text-sm font-semibold text-[var(--ink)]">
                          [{reference.label}] {reference.title}
                        </p>
                        <p className="mt-1 text-xs text-[var(--ink-3)]">
                          {reference.source} {reference.year ? `· ${reference.year}` : ""}
                        </p>
                      </div>
                      {reference.url ? (
                        <a href={reference.url} target="_blank" rel="noreferrer" className="text-sm">
                          Open
                        </a>
                      ) : null}
                    </div>
                  </div>
                ))}
              </div>
            </details>

            <SourcePanel report={report} />
          </div>
        ) : (
          <p className="mt-4 text-sm leading-6 text-[var(--ink-2)]">
            Submit a question to render the final markdown report here.
          </p>
        )}
      </section>

      <Citation />
    </main>
  );
}
