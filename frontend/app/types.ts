// Shared with the recorded demo fixtures in demo/, which are typed against
// ResearchReport so a backend schema change breaks the build rather than
// the deployed page.
export type StatusEvent = {
  event: string;
  message: string;
  report_id?: string | null;
  agent?: string | null;
  data?: Record<string, unknown>;
};

export type RuntimeSettings = {
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

export type PublicRuntimeConfig = {
  llm_provider: string;
  llm_model: string;
  llm_base_url?: string | null;
  llm_temperature: number;
  embedding_backend: string;
  embedding_model?: string | null;
  local_embedding_model: string;
  semantic_scholar_api_key_configured: boolean;
};

export type ResearchReport = {
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

export type ReportSummary = {
  id: string;
  question: string;
  created_at: string;
  paper_count: number;
  warning_count: number;
};

export type ReportSearchHit = {
  report: ReportSummary;
  score: number;
};
