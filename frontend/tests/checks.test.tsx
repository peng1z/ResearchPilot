import React from "react";
import { render, screen, within } from "@testing-library/react";

import { checksFor, CHECKED_SLUGS, VERDICTS, EVIDENCE_TIERS } from "../demo/checks";
import { demoRuns } from "../demo";

import { readFileSync } from "node:fs";

const LORA = "lora-vs-finetuning";

describe("the checking record", () => {
  it("returns nothing for a run that has no record", () => {
    // The guard that matters is that absence never renders as a pass. All
    // three recorded runs are checked today; a fourth added tomorrow must
    // come back null until someone writes its record.
    expect(checksFor("a-run-nobody-checked")).toBeNull();
    expect(CHECKED_SLUGS.every((slug) => demoRuns.some((r) => r.slug === slug))).toBe(true);
  });

  // The file is hand-written, so the shape is enforced here rather than by a
  // structural cast that TypeScript cannot apply to the literal JSON.
  it.each(CHECKED_SLUGS)("%s uses only the declared vocabularies", (slug) => {
    const checks = checksFor(slug)!;
    for (const claim of checks.claims) {
      expect(VERDICTS).toContain(claim.verdict);
      for (const item of claim.evidence) {
        expect(EVIDENCE_TIERS).toContain(item.evidence_tier);
      }
    }
  });

  it.each(CHECKED_SLUGS)("%s checks every claim the run makes, and invents none", (slug) => {
    const checks = checksFor(slug)!;
    const run = demoRuns.find((r) => r.slug === slug)!;
    const stated = [
      ...run.report.synthesis.consensus,
      ...run.report.synthesis.contradictions,
      ...run.report.synthesis.open_gaps,
    ];
    expect(checks.claims.map((c) => c.claim).sort()).toEqual([...stated].sort());
  });

  it.each(CHECKED_SLUGS)("%s checks every source the run cites", (slug) => {
    const checks = checksFor(slug)!;
    const run = demoRuns.find((r) => r.slug === slug)!;
    expect(checks.sources.map((s) => s.ref).sort()).toEqual(
      run.report.references.map((r) => r.label).sort(),
    );
  });

  // A verdict that asserts something must show what it rests on; one that
  // asserts nothing must show where it was looked for. Neither may be bare.
  it.each(CHECKED_SLUGS)("%s never states a verdict without evidence or a search record", (slug) => {
    const checks = checksFor(slug)!;
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
  it.each(CHECKED_SLUGS)("%s serves the same record it renders", (slug) => {
    const imported = JSON.stringify(checksFor(slug));
    const served = JSON.stringify(
      JSON.parse(readFileSync(`public/runs/${slug}/checks.json`, "utf8")),
    );
    expect(served).toBe(imported);
  });

  it.each(CHECKED_SLUGS)("%s keeps existence and relevance separate", (slug) => {
    const checks = checksFor(slug)!;
    // Every source is real; not every real source belongs in the list.
    expect(checks.sources.every((s) => s.exists)).toBe(true);
    expect(checks.sources.some((s) => s.relevant_to_question === false)).toBe(true);
  });
});

describe("what the checked run page says", () => {
  it.each(CHECKED_SLUGS)("%s says who checked it and that it was not a human expert", async (slug) => {
    const { default: RunPage } = await import("../app/runs/[slug]/page");
    render(await RunPage({ params: Promise.resolve({ slug }) }));

    expect(screen.getByText(/not a human expert review/i)).toBeInTheDocument();
    expect(screen.getAllByText(/AI agent/i).length).toBeGreaterThan(0);
  });

  it.each(CHECKED_SLUGS)("%s labels evidence tier per item, not once for the case", async (slug) => {
    const { default: RunPage } = await import("../app/runs/[slug]/page");
    render(await RunPage({ params: Promise.resolve({ slug }) }));

    const checks = checksFor(slug)!;
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

  it.each(CHECKED_SLUGS)("%s shows unverified as unverified, not refuted", async (slug) => {
    const { default: RunPage } = await import("../app/runs/[slug]/page");
    render(await RunPage({ params: Promise.resolve({ slug }) }));

    const checks = checksFor(slug)!;
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

  // The invariant, stated per run rather than for one hand-picked run: a page
  // says it was checked exactly when a record exists for it, and says it was
  // not when none does.
  it.each(demoRuns.map((r) => [r.slug] as const))(
    "%s says whether it was checked, matching whether a record exists",
    async (slug) => {
      const { default: RunPage } = await import("../app/runs/[slug]/page");
      render(await RunPage({ params: Promise.resolve({ slug }) }));

      if (checksFor(slug)) {
        expect(screen.getByText(/not a human expert review/i)).toBeInTheDocument();
        expect(screen.queryByText(/No claim below was checked against its source/i)).toBeNull();
      } else {
        expect(
          screen.getByText(/No claim below was checked against its source/i),
        ).toBeInTheDocument();
        expect(screen.queryByText(/not a human expert review/i)).toBeNull();
      }
    },
  );
});

describe("a contradiction is not the same as an absence", () => {
  // The CoT run has the case that forced this distinction: a cited paper
  // tested the thing and reported the opposite. Recording that as merely
  // "unverified" would have lost the strongest finding in the set.
  it("uses the contradicted verdict only where a source says the opposite", () => {
    for (const slug of CHECKED_SLUGS) {
      const checks = checksFor(slug)!;
      for (const claim of checks.claims) {
        if (claim.verdict !== "contradicted-by-a-cited-paper") continue;
        expect(claim.evidence.length, `${slug}/${claim.id}`).toBeGreaterThan(0);
        // A contradiction must quote the paper that contradicts it.
        expect(claim.limits.join(" ")).toMatch(/contradict|opposite|does not|no improvement/i);
      }
    }
  });

  it("puts every contradicted claim in the public summary's own bucket", () => {
    for (const slug of CHECKED_SLUGS) {
      const checks = checksFor(slug)!;
      const contradicted = checks.claims.filter(
        (c) => c.verdict === "contradicted-by-a-cited-paper",
      ).length;
      expect(checks.public_summary.contradicted, slug).toHaveLength(contradicted);
    }
  });
});

describe("the public summary", () => {
  // The summary is the loudest thing on the page and the easiest place to
  // assert more than the evidence carries. It may only name conclusions that
  // survived, and it must keep the qualifications attached.
  it.each(CHECKED_SLUGS)("%s claims nothing the per-claim verdicts do not support", (slug) => {
    const checks = checksFor(slug)!;
    const supported = checks.claims.filter((c) => c.verdict === "supported").length;
    const narrower = checks.claims.filter(
      (c) => c.verdict === "supported-with-narrower-scope",
    ).length;
    const unverified = checks.claims.filter((c) => c.verdict === "unverified").length;

    expect(checks.public_summary.established.length).toBeLessThanOrEqual(supported);
    expect(checks.public_summary.established_with_limits).toHaveLength(narrower);
    expect(checks.public_summary.not_established).toHaveLength(unverified);
  });

  it.each(CHECKED_SLUGS)("%s says unverified, never false", (slug) => {
    const checks = checksFor(slug)!;
    expect(checks.public_summary.not_established_note).toMatch(/not a finding that they are wrong/i);
    for (const item of checks.public_summary.not_established) {
      expect(item).not.toMatch(/\b(false|wrong|incorrect|refuted)\b/i);
    }
  });

  // LoRA-specific: the 4.5x figure is the sharpest scope error found.
  it("keeps the scope limit on the LoRA run's narrowed conclusion", async () => {
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

describe("the README's summary table", () => {
  // These figures were hand-typed once and one of them was already wrong.
  // A number in prose that restates data is a number that goes stale, so it
  // is pinned to the records here rather than to anyone's memory.
  it("matches the records it summarises", () => {
    const readme = readFileSync("../README.md", "utf8");
    const cell = (label: string) => {
      const m = readme.match(new RegExp(`\\| ${label} \\| ([^|]+) \\|`));
      expect(m, `README row not found: ${label}`).not.toBeNull();
      return (m as RegExpMatchArray)[1].trim();
    };

    const all = CHECKED_SLUGS.map((slug) => checksFor(slug)!);
    const sources = all.flatMap((c) => c.sources);
    const claims = all.flatMap((c) => c.claims);
    const count = (v: string) => claims.filter((c) => c.verdict === v).length;

    expect(cell("Sources that exist, with matching titles")).toBe(
      `${sources.filter((s) => s.exists && s.title_matches).length} of ${sources.length}`,
    );
    expect(cell("Sources that do not bear on their question")).toBe(
      String(sources.filter((s) => s.relevant_to_question === false).length),
    );
    expect(cell("Claims supported")).toBe(String(count("supported")));
    expect(cell("Claims that hold only in a narrower form")).toBe(
      String(count("supported-with-narrower-scope")),
    );
    expect(cell("Claims partly supported")).toBe(String(count("partly-supported")));
    expect(cell("Claims contradicted by a paper the run cites")).toBe(
      String(count("contradicted-by-a-cited-paper")),
    );
    expect(cell("Claims that could not be traced to any retrieved paper")).toBe(
      String(count("unverified")),
    );
    expect(readme).toContain(`${sources.length} sources and ${claims.length} claims`);
  });
});
