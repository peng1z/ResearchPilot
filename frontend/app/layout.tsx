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

/* The share card. Relative, so metadataBase makes it absolute. */
const OG_IMAGE = "/opengraph-image.png";
const OG_ALT =
  "A paper-coloured card headed A multi-agent research co-pilot for fast literature " +
  "synthesis, summarising the recorded runs: 3 runs, 30 papers, from Semantic Scholar, " +
  "arXiv and OpenAlex.";

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
    // The card lives in public/ rather than as app/opengraph-image.png.
    // The file convention wins over an explicit openGraph.images and drops
    // its alt with it, so the convention costs the alt text; declaring the
    // whole thing here keeps both. (opengraph-image.alt.txt is accepted as
    // a file on Next 15.5 and emits nothing at all.)
    images: [{ url: OG_IMAGE, width: 1200, height: 630, alt: OG_ALT }],
  },
  twitter: {
    card: "summary_large_image",
    title: "ResearchPilot — recorded multi-agent literature synthesis runs",
    description: DESCRIPTION,
    images: [{ url: OG_IMAGE, alt: OG_ALT }],
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

const DIRECTION_CONTRACT = "<!-- THESIS: A reading page for one recorded synthesis, refusing the product-landing arrangement this category ships -- the artifact leads and the paper rides in the footer. OWN-WORLD: White ground, Charter-class serif body on a 34rem measure, system sans for headings because headings here are navigation not voice, one citation-blue accent, rules instead of cards. STORY: A researcher checks whether the system did what the paper claims, reads one run end to end, and leaves able to cite it. FIRST VIEWPORT: Nav rule, heading, what it does in four lines, the recorded-not-live notice, then the three runs as the first thing clickable; running is a disclosure below, never the opening act. FORM: Category standard executed straight; candidate 4 of 7 on the grounded list, taken as the standing exit. Seed da9de08a. FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance. -->";

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
                {/* The direction this page commits to, in the emitted markup so it can
            be audited against what shipped. A JSX comment never reaches the
            output; this does. */}
        <div
          suppressHydrationWarning
          dangerouslySetInnerHTML={{ __html: DIRECTION_CONTRACT }}
        />
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
