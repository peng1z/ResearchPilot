import type { ResearchReport } from "../app/types";

export type DemoRun = {
  slug: string;
  question: string;
  /** Wall-clock seconds the recorded pipeline run took. */
  elapsedSeconds: number;
  report: ResearchReport;
};
