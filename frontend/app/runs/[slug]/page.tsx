import { Fragment } from "react";

import type { Metadata } from "next";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";

import { demoRuns } from "../../../demo";

const SITE = "https://researchpilot.peng1z.workers.dev";

export function generateStaticParams() {
  return demoRuns.map((run) => ({ slug: run.slug }));
}

function find(slug: string) {
  return demoRuns.find((run) => run.slug === slug);
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const run = find(slug);
  if (!run) {
    return {};
  }
  const url = `${SITE}/runs/${run.slug}/`;
  const description =
    `A recorded ResearchPilot run for "${run.question}": ` +
    `${run.report.papers.length} papers retrieved, structured findings extracted from each ` +
    `abstract, and consensus, contradictions and open gaps synthesised across them.`;
  return {
    title: `${run.question} — a recorded ResearchPilot run`,
    description,
    // Each case owns its own canonical. Pointing these at the paper page would
    // ask a crawler to treat three different runs as one document.
    alternates: { canonical: url },
    openGraph: { type: "article", url, title: run.question, description },
    twitter: { card: "summary_large_image", title: run.question, description },
  };
}

export default async function RunPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const run = find(slug);
  if (!run) {
    notFound();
  }

  const report = run.report;
  const bySource = report.papers.reduce<Record<string, number>>((acc, paper) => {
    acc[paper.source] = (acc[paper.source] ?? 0) + 1;
    return acc;
  }, {});
  const failed = report.warnings
    .map((warning) => /^(.+?) search failed: /.exec(warning))
    .filter((match): match is RegExpExecArray => match !== null)
    .map((match) => match[1]);

  // Describes what the page actually shows: one recorded run, its question,
  // the papers it retrieved, and the article whose method produced it. Not a
  // ScholarlyArticle -- this is a run of a tool, not a paper, and saying
  // otherwise to a parser that cannot check is the whole failure mode.
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "Dataset",
    name: `Recorded ResearchPilot run: ${run.question}`,
    description: `A single end-to-end run retrieving ${report.papers.length} papers, extracting structured findings from their abstracts, and synthesising consensus, contradictions and open gaps.`,
    url: `${SITE}/runs/${run.slug}/`,
    license: "https://opensource.org/licenses/MIT",
    creator: { "@type": "Person", name: "Peng Zhang", url: "https://github.com/peng1z" },
    isBasedOn: "https://github.com/peng1z/ResearchPilot",
    variableMeasured: ["consensus", "contradictions", "open gaps"],
    citation: {
      "@type": "ScholarlyArticle",
      name: "ResearchPilot: A Local-First Multi-Agent System for Literature Synthesis and Related Work Drafting",
      author: { "@type": "Person", name: "Peng Zhang" },
      identifier: "arXiv:2603.14629",
      url: "https://arxiv.org/abs/2603.14629",
    },
  };

  return (
    <main className="mx-auto flex min-h-screen max-w-4xl flex-col gap-8 px-6 py-10">
      <script
        type="application/ld+json"
        // A literal built from the fixture in this file, not user input.
        dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
      />
      <nav aria-label="Primary" className="flex flex-wrap gap-5 text-sm">
        <a className="underline" href="/">
          All runs
        </a>
        <a className="underline" href="https://arxiv.org/abs/2603.14629">
          Paper
        </a>
        <a className="underline" href="https://github.com/peng1z/ResearchPilot">
          Code
        </a>
      </nav>

      <header>
        <p className="text-xs uppercase tracking-[0.2em] text-[var(--accent)]">Recorded run</p>
        <h1 className="mt-3 text-3xl font-semibold leading-tight md:text-4xl">{run.question}</h1>
        <p className="mt-4 leading-7 text-[var(--muted)]">
          Nothing in the output was edited. This is one run of a non-deterministic system: it shows
          what the pipeline produced on that occasion, not what it produces in general.
        </p>
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm text-[var(--muted)]">
          {(
            [
              ["Recorded", report.created_at ? report.created_at.slice(0, 10) : null],
              ["Duration", `${run.elapsedSeconds}s`],
              ["Model", report.tool?.model ?? null],
              [
                "Software",
                report.tool
                  ? `${report.tool.name} ${report.tool.version}` +
                    (report.tool.commit ? ` (${report.tool.commit})` : "")
                  : null,
              ],
            ] as const
          ).map(([label, value]) => (
            <Fragment key={label}>
              <dt>{label}</dt>
              {/* An absent field says so. Filling it in from a guess would make
                  the record less trustworthy than leaving the gap visible. */}
              <dd className={value ? "font-medium text-[var(--text)]" : "italic"}>
                {value ?? "not recorded"}
              </dd>
            </Fragment>
          ))}
        </dl>
      </header>

      <section>
        <h2 className="text-xl font-semibold">Retrieval</h2>
        <ul className="mt-3 leading-7 text-[var(--muted)]">
          {Object.entries(bySource).map(([source, count]) => (
            <li key={source}>
              {source}: {count} papers
            </li>
          ))}
          {failed.map((name) => (
            <li key={name}>{name}: returned an error, so this run has none of its results</li>
          ))}
        </ul>
        <p className="mt-3 leading-7 text-[var(--muted)]">
          Sources are queried in parallel and each is allowed to fail on its own. Version 1 of the
          paper describes Semantic Scholar and arXiv; OpenAlex results here come from a later build
          and are not part of what the paper reports.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold">Papers</h2>
        <ol className="mt-3 space-y-2 leading-7">
          {report.papers.map((paper) => (
            <li key={paper.id}>
              {paper.url ? (
                <a className="underline" href={paper.url}>
                  {paper.title}
                </a>
              ) : (
                paper.title
              )}{" "}
              <span className="text-[var(--muted)]">
                ({paper.source}
                {paper.year ? `, ${paper.year}` : ""})
              </span>
            </li>
          ))}
        </ol>
      </section>

      <section>
        <h2 className="text-xl font-semibold">Synthesis</h2>
        {(
          [
            ["Consensus", report.synthesis.consensus],
            ["Contradictions", report.synthesis.contradictions],
            ["Open gaps", report.synthesis.open_gaps],
          ] as const
        ).map(([label, items]) => (
          <div key={label} className="mt-4">
            <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
              {label}
            </h3>
            {items.length === 0 ? (
              <p className="mt-2 text-[var(--muted)]">None reported for this question.</p>
            ) : (
              <ul className="mt-2 space-y-2 leading-7 text-[var(--muted)]">
                {items.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </section>

      <section>
        <h2 className="text-xl font-semibold">Related work draft</h2>
        <div className="markdown prose prose-neutral mt-3 max-w-none">
          <ReactMarkdown>{report.related_work_markdown}</ReactMarkdown>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold">Limits of this run</h2>
        <ul className="mt-3 space-y-2 leading-7 text-[var(--muted)]">
          <li>
            The synthesis is generated. It has not been checked against the papers it cites, and
            nothing here should be read as a verified account of the literature.
          </li>
          <li>
            Findings are extracted from abstracts, not full texts, so a claim qualified in a
            paper&apos;s body can arrive here unqualified.
          </li>
          {failed.length > 0 ? (
            <li>
              {failed.join(" and ")} returned an error, so this run is drawn from the sources that
              answered rather than from all of them.
            </li>
          ) : null}
          <li>One run of one question. It measures nothing.</li>
        </ul>
      </section>

      <footer>
        <h2 className="text-xl font-semibold">Cite the method</h2>
        <p className="mt-3 leading-7 text-[var(--muted)]">
          Produced with ResearchPilot:{" "}
          <a className="underline" href="https://arxiv.org/abs/2603.14629">
            ResearchPilot: A Local-First Multi-Agent System for Literature Synthesis and Related
            Work Drafting
          </a>
          , Peng Zhang, arXiv:2603.14629, version 1 preprint. The method paper describes how this
          was produced; it is not a source for the topic above.
        </p>
      </footer>
    </main>
  );
}
