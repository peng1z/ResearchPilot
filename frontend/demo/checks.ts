import loraChecks from "./lora-vs-finetuning.checks.json";

/** One reference, resolved against the record that registered it. */
export type SourceCheck = {
  ref: string;
  cited_as: { title: string; year: number; id: string; url: string };
  exists: boolean;
  primary_record: Record<string, string | number>;
  corroboration?: Record<string, string | boolean>;
  title_matches: boolean;
  authors_verified: string;
  version_note?: string;
  venue_not_recorded?: string;
  data_quality_note?: string;
  open_access_full_text?: string;
  /** Judged separately from existence: a paper can be real and still off-topic. */
  relevant_to_question: boolean | "partly";
  relevance_note: string;
};

export type Evidence = {
  ref: string;
  /** Which tier this particular quote came from, per item, not per case. */
  evidence_tier: "full-text" | "abstract";
  url: string;
  where: string;
  quote: string;
  note?: string;
};

export type ClaimCheck = {
  id: string;
  kind: "consensus" | "contradiction" | "open_gap";
  claim: string;
  verdict: "supported" | "supported-with-narrower-scope" | "partly-supported" | "unverified";
  evidence: Evidence[];
  /** Where an unverified claim was looked for, so the search is auditable. */
  searched?: { ref: string; tier: string; url: string; result: string }[];
  limits: string[];
};

/** Only what the evidence carries, with the qualifications kept on. */
export type PublicSummary = {
  established: string[];
  established_with_limits: string[];
  not_established: string[];
  not_established_note: string;
  retrieval: string;
};

export type RunChecks = {
  run_slug: string;
  question: string;
  checked_on: string;
  checked_by: { who: string; not: string; method: string; limits: string[] };
  public_summary: PublicSummary;
  sources: SourceCheck[];
  claims: ClaimCheck[];
  summary: Record<string, number | Record<string, number>>;
};

/**
 * Checks exist for one run so far. A run with no entry here is unchecked and
 * says so; absence must never render as a pass.
 */
const CHECKS: Record<string, RunChecks> = {
  // Cast through unknown: the JSON is written by hand and TypeScript infers
  // each entry's own literal shape, which never structurally matches the
  // union above. The shape is enforced at test time instead, by validating
  // the file against the verdict and tier vocabularies -- a stronger check
  // than the structural one, since it also catches a typo'd verdict.
  "lora-vs-finetuning": loraChecks as unknown as RunChecks,
};

export const VERDICTS = [
  "supported",
  "supported-with-narrower-scope",
  "partly-supported",
  "unverified",
] as const;

export const EVIDENCE_TIERS = ["full-text", "abstract"] as const;

/** Every run that has a checks file, for tests and for listing. */
export const CHECKED_SLUGS = Object.keys(CHECKS);

export function checksFor(slug: string): RunChecks | null {
  return CHECKS[slug] ?? null;
}
