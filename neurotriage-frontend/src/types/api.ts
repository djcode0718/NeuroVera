/**
 * API response type definitions for NeuroTriage backend communication
 */

export interface AgentTraceEntry {
  agent: string;
  summary: string;
  key_evidence: string;
  status?: "pending" | "running" | "done" | "error";
}

export interface AnalysisResult {
  run_id: string;
  status: "completed" | "failed";
  classification: string;
  confidence: number;
  predictions: Record<string, number>;
  gradcam_image: string; // base64 PNG
  draft_report: string;
  routing: "auto-clear" | "needs-review" | "urgent";
  justification: string;
  reasoning_trace: AgentTraceEntry[];
  critic_revision_count: number;
  models_used: Record<string, string>;
  error_message?: string;
}

export interface HistoryItem {
  run_id: string;
  top_class: string;
  top_confidence: number;
  routing: "auto-clear" | "needs-review" | "urgent";
  created_at: string;
}

export interface AnalysisSummary {
  run_id: string;
  top_class: string;
  top_confidence: number;
  routing: string;
  created_at: string;
}
