// Generated from real pipeline runs; see docs/demo-recordings.md.
import type { DemoRun } from "./types";

import chain_of_thought from "./chain-of-thought.json";
import lora_vs_finetuning from "./lora-vs-finetuning.json";
import rag_hallucination from "./rag-hallucination.json";

export const demoRuns: DemoRun[] = [
  chain_of_thought as DemoRun,
  lora_vs_finetuning as DemoRun,
  rag_hallucination as DemoRun,
];
