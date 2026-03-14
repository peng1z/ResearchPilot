"use client";

import React, { FormEvent, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

type StatusEvent = {
  event: string;
  message: string;
  report_id?: string | null;
  agent?: string | null;
  data?: Record<string, unknown>;
};

type RuntimeSettings = {
  llm_provider?: string;
  llm_model?: string;
  llm_api_key?: string;
  llm_base_url?: string;
  llm_temperature?: number;
  embedding_backend?: string;
  embedding_model?: string;
  local_embedding_model?: string;
  semantic_scholar_api_key?: string;
};

type PublicRuntimeConfig = {
  llm_provider: string;
  llm_model: string;
  llm_base_url?: string | null;
  llm_temperature: number;
  embedding_backend: string;
  embedding_model?: string | null;
  local_embedding_model: string;
  semantic_scholar_api_key_configured: boolean;
};

type ResearchReport = {
  id: string;
  question: string;
  related_work_markdown: string;
  warnings: string[];
  synthesis: {
    consensus: string[];
    contradictions: string[];
    open_gaps: string[];
  };
  references: Array<{
    label: string;
    paper_id: string;
    title: string;
    source: string;
    year?: number | null;
    url?: string | null;
  }>;
  papers: Array<{
    id: string;
    title: string;
    source: string;
    year?: number | null;
  }>;
};

type ReportSummary = {
  id: string;
  question: string;
  created_at: string;
  paper_count: number;
  warning_count: number;
};

type ReportSearchHit = {
  report: ReportSummary;
  score: number;
};

const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

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

export default function Home() {
  const [question, setQuestion] = useState("What are the recent trends in retrieval-augmented generation for question answering?");
  const [events, setEvents] = useState<StatusEvent[]>([]);
  const [report, setReport] = useState<ResearchReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<ReportSummary[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<ReportSearchHit[]>([]);
  const [config, setConfig] = useState<PublicRuntimeConfig | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [runtime, setRuntime] = useState<RuntimeSettings>({});

  useEffect(() => {
    void loadHistory();
    void loadConfig();
  }, []);

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
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    }
  }

  async function runResearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setEvents([]);
    setReport(null);
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
    if (!searchQuery.trim()) {
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
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-8 px-6 py-10 md:px-10">
      <section className="rounded-[2rem] border border-[var(--border)] bg-[var(--surface)] p-8 shadow-[0_24px_80px_rgba(24,38,31,0.08)] backdrop-blur">
        <p className="mb-3 text-sm uppercase tracking-[0.3em] text-[var(--accent)]">ResearchPilot</p>
        <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <div>
            <h1 className="max-w-3xl text-4xl font-semibold leading-tight md:text-6xl">
              A multi-agent research co-pilot for fast literature synthesis.
            </h1>
            <p className="mt-4 max-w-2xl text-lg leading-8 text-[var(--muted)]">
              Search papers, extract structured claims, synthesize the field, and draft a related work section from one research question.
            </p>
          </div>
          <form onSubmit={runResearch} className="rounded-[1.5rem] border border-[var(--border)] bg-[var(--panel)] p-5">
            <label htmlFor="question" className="mb-2 block text-sm font-semibold uppercase tracking-[0.2em] text-[var(--accent)]">
              Research Question
            </label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={6}
              className="w-full rounded-2xl border border-[var(--border)] bg-white px-4 py-3 text-base text-[var(--text)] outline-none"
            />
            <button
              type="button"
              onClick={() => setShowSettings((current) => !current)}
              className="mt-4 rounded-full border border-[var(--border)] px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em]"
            >
              {showSettings ? "Hide Settings" : "Run Settings"}
            </button>
            {showSettings ? (
              <div className="mt-4 space-y-3 rounded-2xl border border-[var(--border)] bg-white px-4 py-4">
                <div className="grid gap-3 md:grid-cols-2">
                  <label className="text-sm">
                    <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">Provider</span>
                    <select
                      value={runtime.llm_provider ?? ""}
                      onChange={(event) => updateRuntime("llm_provider", event.target.value)}
                      className="w-full rounded-xl border border-[var(--border)] px-3 py-2"
                    >
                      <option value="openai">OpenAI</option>
                      <option value="anthropic">Anthropic</option>
                      <option value="groq">Groq</option>
                      <option value="openrouter">OpenRouter</option>
                    </select>
                  </label>
                  <label className="text-sm">
                    <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">Model</span>
                    <input
                      value={runtime.llm_model ?? ""}
                      onChange={(event) => updateRuntime("llm_model", event.target.value)}
                      className="w-full rounded-xl border border-[var(--border)] px-3 py-2"
                    />
                  </label>
                  <label className="text-sm">
                    <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">API Key</span>
                    <input
                      type="password"
                      value={runtime.llm_api_key ?? ""}
                      onChange={(event) => updateRuntime("llm_api_key", event.target.value)}
                      placeholder="Optional per-run override"
                      className="w-full rounded-xl border border-[var(--border)] px-3 py-2"
                    />
                  </label>
                  <label className="text-sm">
                    <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">Base URL</span>
                    <input
                      value={runtime.llm_base_url ?? ""}
                      onChange={(event) => updateRuntime("llm_base_url", event.target.value)}
                      className="w-full rounded-xl border border-[var(--border)] px-3 py-2"
                    />
                  </label>
                  <label className="text-sm">
                    <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">Embedding Backend</span>
                    <select
                      value={runtime.embedding_backend ?? "auto"}
                      onChange={(event) => updateRuntime("embedding_backend", event.target.value)}
                      className="w-full rounded-xl border border-[var(--border)] px-3 py-2"
                    >
                      <option value="auto">auto</option>
                      <option value="remote">remote</option>
                      <option value="local">local</option>
                    </select>
                  </label>
                  <label className="text-sm">
                    <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">Embedding Model</span>
                    <input
                      value={runtime.embedding_model ?? ""}
                      onChange={(event) => updateRuntime("embedding_model", event.target.value)}
                      className="w-full rounded-xl border border-[var(--border)] px-3 py-2"
                    />
                  </label>
                  <label className="text-sm md:col-span-2">
                    <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">Local Embedding Model</span>
                    <input
                      value={runtime.local_embedding_model ?? ""}
                      onChange={(event) => updateRuntime("local_embedding_model", event.target.value)}
                      className="w-full rounded-xl border border-[var(--border)] px-3 py-2"
                    />
                  </label>
                </div>
                <p className="text-xs leading-5 text-[var(--muted)]">
                  Settings apply to this run only. Current server default: {config ? `${config.llm_provider}/${config.llm_model}` : "loading..."}.
                </p>
              </div>
            ) : null}
            <button
              type="submit"
              disabled={loading}
              className="mt-4 inline-flex w-full items-center justify-center rounded-full bg-[var(--accent)] px-5 py-3 text-sm font-semibold uppercase tracking-[0.2em] text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Running Pipeline" : "Start Research"}
            </button>
            <p className="mt-4 text-sm leading-6 text-[var(--muted)]">
              Backend streams agent lifecycle events over SSE and returns the final report as structured JSON plus markdown.
            </p>
          </form>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-[0.85fr_1.15fr]">
        <div className="rounded-[1.5rem] border border-[var(--border)] bg-[var(--panel)] p-6">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-2xl font-semibold">Agent Feed</h2>
            <span className="text-sm uppercase tracking-[0.2em] text-[var(--accent)]">
              {loading ? "Live" : "Idle"}
            </span>
          </div>

          <div className="space-y-3">
            {events.length === 0 ? (
              <p className="text-sm text-[var(--muted)]">No events yet.</p>
            ) : (
              events.map((item, index) => (
                <div key={`${item.event}-${index}`} className="rounded-2xl border border-[var(--border)] px-4 py-3">
                  <div className="flex items-center justify-between gap-3">
                    <strong className="text-sm">{item.agent ?? item.event}</strong>
                    <span className="text-xs uppercase tracking-[0.2em] text-[var(--accent)]">{item.event}</span>
                  </div>
                  <p className="mt-2 text-sm leading-6 text-[var(--muted)]">{item.message}</p>
                </div>
              ))
            )}
          </div>

          {error ? <p className="mt-4 text-sm text-red-700">{error}</p> : null}

          <div className="mt-8 rounded-2xl border border-[var(--border)] px-4 py-4">
            <h3 className="text-lg font-semibold">Report History</h3>
            <form onSubmit={searchHistory} className="mt-4 flex gap-2">
              <input
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Search past reports"
                className="min-w-0 flex-1 rounded-full border border-[var(--border)] bg-white px-4 py-2 text-sm outline-none"
              />
              <button
                type="submit"
                className="rounded-full border border-[var(--border)] px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em]"
              >
                Search
              </button>
            </form>

            {searchResults.length > 0 ? (
              <div className="mt-4 space-y-2">
                {searchResults.map(({ report: item, score }) => (
                  <button
                    key={`search-${item.id}`}
                    type="button"
                    onClick={() => void loadReport(item.id)}
                    className="block w-full rounded-2xl border border-[var(--border)] bg-white px-4 py-3 text-left"
                  >
                    <p className="text-sm font-semibold text-[var(--text)]">{item.question}</p>
                    <p className="mt-1 text-xs uppercase tracking-[0.18em] text-[var(--accent)]">
                      score {score.toFixed(2)} • {item.paper_count} papers
                    </p>
                  </button>
                ))}
              </div>
            ) : null}

            <div className="mt-4 space-y-2">
              {history.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => void loadReport(item.id)}
                  className="block w-full rounded-2xl border border-[var(--border)] bg-white px-4 py-3 text-left"
                >
                  <p className="text-sm font-semibold text-[var(--text)]">{item.question}</p>
                  <p className="mt-1 text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                    {new Date(item.created_at).toLocaleString()} • {item.paper_count} papers
                    {item.warning_count > 0 ? ` • ${item.warning_count} warnings` : ""}
                  </p>
                </button>
              ))}
              {history.length === 0 ? <p className="text-sm text-[var(--muted)]">No saved reports yet.</p> : null}
            </div>
          </div>
        </div>

        <div className="rounded-[1.5rem] border border-[var(--border)] bg-[var(--panel)] p-6">
          <h2 className="text-2xl font-semibold">Related Work Draft</h2>
          {report ? (
            <div className="mt-5 space-y-6">
              <div className="rounded-2xl border border-[var(--border)] bg-white px-4 py-3 text-sm text-[var(--muted)]">
                <p className="font-semibold text-[var(--text)]">Question</p>
                <p className="mt-2">{report.question}</p>
              </div>

              <div className="markdown prose prose-neutral max-w-none">
                <ReactMarkdown>{report.related_work_markdown}</ReactMarkdown>
              </div>

              {report.warnings.length > 0 ? (
                <div className="rounded-2xl border border-amber-300 bg-amber-50 px-4 py-3">
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-amber-800">Warnings</p>
                  <ul className="mt-3 space-y-2 text-sm text-amber-900">
                    {report.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </div>
              ) : null}

              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-2xl border border-[var(--border)] px-4 py-3">
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-[var(--accent)]">Consensus</p>
                  <ul className="mt-3 space-y-2 text-sm text-[var(--muted)]">
                    {report.synthesis.consensus.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
                <div className="rounded-2xl border border-[var(--border)] px-4 py-3">
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-[var(--accent)]">Contradictions</p>
                  <ul className="mt-3 space-y-2 text-sm text-[var(--muted)]">
                    {report.synthesis.contradictions.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
                <div className="rounded-2xl border border-[var(--border)] px-4 py-3">
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-[var(--accent)]">Open Gaps</p>
                  <ul className="mt-3 space-y-2 text-sm text-[var(--muted)]">
                    {report.synthesis.open_gaps.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="rounded-2xl border border-[var(--border)] px-4 py-4">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-[var(--accent)]">Retrieved Papers</p>
                  <span className="text-xs uppercase tracking-[0.2em] text-[var(--muted)]">{report.papers.length} papers</span>
                </div>
                <div className="mt-4 space-y-3">
                  {report.references.map((reference) => (
                    <div key={reference.paper_id} className="rounded-2xl border border-[var(--border)] bg-white px-4 py-3">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <p className="text-sm font-semibold text-[var(--text)]">
                            [{reference.label}] {reference.title}
                          </p>
                          <p className="mt-1 text-xs uppercase tracking-[0.18em] text-[var(--accent)]">
                            {reference.source} {reference.year ? `• ${reference.year}` : ""}
                          </p>
                        </div>
                        {reference.url ? (
                          <a
                            href={reference.url}
                            target="_blank"
                            rel="noreferrer"
                            className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]"
                          >
                            Open
                          </a>
                        ) : null}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <p className="mt-4 text-sm leading-6 text-[var(--muted)]">
              Submit a question to render the final markdown report here.
            </p>
          )}
        </div>
      </section>
    </main>
  );
}
