import React from "react";
import { render, screen, within } from "@testing-library/react";

import { checksFor, CHECKED_SLUGS, VERDICTS, EVIDENCE_TIERS } from "../demo/checks";
import { demoRuns } from "../demo";

import { readFileSync } from "node:fs";

const LORA = "lora-vs-finetuning";

describe("the checking record", () => {
  it("covers exactly the runs that were checked, and claims no others", () => {
    expect(CHECKED_SLUGS).toEqual([LORA]);
    for (const run of demoRuns) {
      if (run.slug !== LORA) {
        expect(checksFor(run.slug)).toBeNull();
      }
    }
  });

  // The file is hand-written, so the shape is enforced here rather than by a
  // structural cast that TypeScript cannot apply to the literal JSON.
  it("uses only the declared verdict and evidence vocabularies", () => {
    const checks = checksFor(LORA)!;
    for (const claim of checks.claims) {
      expect(VERDICTS).toContain(claim.verdict);
      for (const item of claim.evidence) {
        expect(EVIDENCE_TIERS).toContain(item.evidence_tier);
      }
    }
  });

  it("checks every claim the run actually makes, and invents none", () => {
    const checks = checksFor(LORA)!;
    const run = demoRuns.find((r) => r.slug === LORA)!;
    const stated = [
      ...run.report.synthesis.consensus,
      ...run.report.synthesis.contradictions,
      ...run.report.synthesis.open_gaps,
    ];
    expect(checks.claims.map((c) => c.claim).sort()).toEqual([...stated].sort());
  });

  it("checks every source the run cites", () => {
    const checks = checksFor(LORA)!;
    const run = demoRuns.find((r) => r.slug === LORA)!;
    expect(checks.sources.map((s) => s.ref).sort()).toEqual(
      run.report.references.map((r) => r.label).sort(),
    );
  });

  // A verdict that asserts something must show what it rests on; one that
  // asserts nothing must show where it was looked for. Neither may be bare.
  it("never states a verdict without evidence or a search record", () => {
    const checks = checksFor(LORA)!;
    for (const claim of checks.claims) {
      if (claim.verdict === "unverified") {
        expect(claim.evidence, `${claim.id} should cite nothing`).toHaveLength(0);
        expect(claim.searched?.length ?? 0, `${claim.id} needs a search record`).toBeGreaterThan(0);
      } else {
        expect(claim.evidence.length, `${claim.id} needs evidence`).toBeGreaterThan(0);
        for (const item of claim.evidence) {
          expect(item.quote.length, `${claim.id}/${item.ref} needs a quote`).toBeGreaterThan(20);
          expect(item.url).toMatch(/^https:\/\//);
          expect(item.where.length).toBeGreaterThan(0);
        }
      }
    }
  });

  // The record is imported at build time and also served for download. Two
  // copies of one file is exactly how a downloadable artifact stops matching
  // the page that describes it.
  it("serves the same record it renders", () => {
    const imported = JSON.stringify(checksFor(LORA));
    const served = JSON.stringify(
      JSON.parse(readFileSync("public/runs/lora-vs-finetuning/checks.json", "utf8")),
    );
    expect(served).toBe(imported);
  });

  it("keeps existence and relevance as separate judgements", () => {
    const checks = checksFor(LORA)!;
    // Every source is real; not every real source belongs in the list.
    expect(checks.sources.every((s) => s.exists)).toBe(true);
    expect(checks.sources.some((s) => s.relevant_to_question === false)).toBe(true);
  });
});

describe("what the checked run page says", () => {
  it("says who checked it and that it was not a human expert", async () => {
    const { default: RunPage } = await import("../app/runs/[slug]/page");
    render(await RunPage({ params: Promise.resolve({ slug: LORA }) }));

    expect(screen.getByText(/not a human expert review/i)).toBeInTheDocument();
    expect(screen.getAllByText(/AI agent/i).length).toBeGreaterThan(0);
  });

  it("labels evidence tier per item, not once for the whole case", async () => {
    const { default: RunPage } = await import("../app/runs/[slug]/page");
    render(await RunPage({ params: Promise.resolve({ slug: LORA }) }));

    const checks = checksFor(LORA)!;
    const fromAbstract = checks.claims
      .flatMap((c) => c.evidence)
      .filter((e) => e.evidence_tier === "abstract").length;
    const fromFullText = checks.claims
      .flatMap((c) => c.evidence)
      .filter((e) => e.evidence_tier === "full-text").length;

    // Both tiers are present, so neither label may be applied to everything.
    expect(fromAbstract).toBeGreaterThan(0);
    expect(fromFullText).toBeGreaterThan(0);
    expect(screen.getAllByText("abstract only")).toHaveLength(fromAbstract);
    expect(screen.getAllByText("full text")).toHaveLength(fromFullText);
  });

  it("shows an unverified claim as unverified, not as refuted or as passing", async () => {
    const { default: RunPage } = await import("../app/runs/[slug]/page");
    render(await RunPage({ params: Promise.resolve({ slug: LORA }) }));

    const checks = checksFor(LORA)!;
    const unverified = checks.claims.filter((c) => c.verdict === "unverified");
    expect(unverified.length).toBeGreaterThan(0);
    expect(screen.getAllByText("not verified")).toHaveLength(unverified.length);

    // Scoped to the verdict tags. A page-wide regex would trip on the page's
    // own sentence that an unverified claim is "not a finding that the claim
    // is false" -- which is the distinction being protected, not a breach.
    const tags = [...document.querySelectorAll(".verdict-tag")].map((n) => n.textContent ?? "");
    expect(tags.length).toBeGreaterThan(0);
    for (const tag of tags) {
      expect(tag).not.toMatch(/false|refuted|wrong|incorrect/i);
    }
  });

  it("still says an unchecked run is unchecked", async () => {
    const { default: RunPage } = await import("../app/runs/[slug]/page");
    const other = demoRuns.find((r) => r.slug !== LORA)!;
    render(await RunPage({ params: Promise.resolve({ slug: other.slug }) }));

    expect(screen.getByText(/No claim below was checked against its source/i)).toBeInTheDocument();
    expect(screen.queryByText(/not a human expert review/i)).toBeNull();
  });
});

describe("the public summary", () => {
  // The summary is the loudest thing on the page and the easiest place to
  // assert more than the evidence carries. It may only name conclusions that
  // survived, and it must keep the qualifications attached.
  it("claims nothing the per-claim verdicts do not support", () => {
    const checks = checksFor(LORA)!;
    const supported = checks.claims.filter((c) => c.verdict === "supported").length;
    const narrower = checks.claims.filter(
      (c) => c.verdict === "supported-with-narrower-scope",
    ).length;
    const unverified = checks.claims.filter((c) => c.verdict === "unverified").length;

    expect(checks.public_summary.established.length).toBeLessThanOrEqual(supported);
    expect(checks.public_summary.established_with_limits).toHaveLength(narrower);
    expect(checks.public_summary.not_established).toHaveLength(unverified);
  });

  it("says an unverified claim was not established, never that it is false", () => {
    const checks = checksFor(LORA)!;
    expect(checks.public_summary.not_established_note).toMatch(/not a finding that they are wrong/i);
    for (const item of checks.public_summary.not_established) {
      expect(item).not.toMatch(/\b(false|wrong|incorrect|refuted)\b/i);
    }
  });

  it("keeps the scope limit on the one narrowed conclusion", async () => {
    const checks = checksFor(LORA)!;
    const narrowed = checks.public_summary.established_with_limits.join(" ");
    // The 4.5x figure must never appear without the domain it came from.
    expect(narrowed).toMatch(/4\.5/);
    expect(narrowed).toMatch(/protein language models/i);

    const { default: RunPage } = await import("../app/runs/[slug]/page");
    render(await RunPage({ params: Promise.resolve({ slug: LORA }) }));
    const shown = document.body.textContent ?? "";
    const idx = shown.indexOf("4.5-fold");
    expect(idx).toBeGreaterThan(-1);
    expect(shown).toMatch(/ESM2 3B/);
  });
});
