import "./globals.css";
import type { Metadata } from "next";
import { ReactNode } from "react";

const SITE = "https://researchpilot.peng1z.workers.dev";
const PAPER = "https://arxiv.org/abs/2603.14629";
const REPO = "https://github.com/peng1z/ResearchPilot";

const DESCRIPTION =
  "Three recorded multi-agent literature runs, captured end to end: papers " +
  "retrieved from arXiv and OpenAlex, structured findings extracted from " +
  "abstracts, consensus and contradictions synthesised, and a related-work " +
  "section drafted. Artifact for arXiv:2603.14629.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE),
  title: "ResearchPilot — recorded multi-agent literature synthesis runs",
  description: DESCRIPTION,
  authors: [{ name: "Peng Zhang" }],
  keywords: [
    "literature review",
    "related work generation",
    "multi-agent",
    "large language models",
    "scientific information retrieval",
    "research synthesis",
  ],
  alternates: { canonical: SITE },
  openGraph: {
    type: "website",
    url: SITE,
    siteName: "ResearchPilot",
    title: "ResearchPilot — recorded multi-agent literature synthesis runs",
    description: DESCRIPTION,
  },
  twitter: {
    card: "summary_large_image",
    title: "ResearchPilot — recorded multi-agent literature synthesis runs",
    description: DESCRIPTION,
  },
};

/**
 * Structured data describing the *software*, and naming the article it
 * accompanies.
 *
 * Deliberately not Google Scholar's `citation_*` meta tags. Those assert that
 * the page they sit on is the article, and Scholar's guidelines say it "does
 * not index pages that merely describe or link to papers". This page is the
 * artifact, not the paper; arXiv:2603.14629 is the article and already carries
 * those tags. Claiming otherwise here would be a false statement to a parser
 * that has no way to check it.
 */
const STRUCTURED_DATA = {
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  name: "ResearchPilot",
  description:
    "A local-first multi-agent system for literature review. Given a research " +
    "question it searches Semantic Scholar, arXiv and OpenAlex in parallel, " +
    "extracts structured findings from abstracts, synthesises consensus, " +
    "contradictions and open gaps, and drafts a citation-aware related work " +
    "section.",
  url: SITE,
  codeRepository: REPO,
  programmingLanguage: ["Python", "TypeScript"],
  runtimePlatform: ["Python 3.11+", "Node.js 20+"],
  license: "https://opensource.org/licenses/MIT",
  author: {
    "@type": "Person",
    name: "Peng Zhang",
    url: "https://github.com/peng1z",
  },
  citation: {
    "@type": "ScholarlyArticle",
    name:
      "ResearchPilot: A Local-First Multi-Agent System for Literature " +
      "Synthesis and Related Work Drafting",
    author: { "@type": "Person", name: "Peng Zhang" },
    datePublished: "2026-03-15",
    identifier: "arXiv:2603.14629",
    url: PAPER,
    sameAs: PAPER,
  },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <script
          type="application/ld+json"
          // The payload is a literal in this file, not user or model input.
          dangerouslySetInnerHTML={{ __html: JSON.stringify(STRUCTURED_DATA) }}
        />
        {children}
      </body>
    </html>
  );
}
