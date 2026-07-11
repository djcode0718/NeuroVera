// NeuroTriage API types — single source of truth
// DO NOT add fields beyond what is in the backend contract

export interface AnalysisResult {
  run_id: string;
  status: string; // "completed"
  classification: string; // "glioma" | "meningioma" | "notumor" | "pituitary"
  confidence: number; // 0.0–1.0
  predictions: {
    glioma: number;
    meningioma: number;
    notumor: number;
    pituitary: number;
  };
  gradcam_image: string; // RAW base64 PNG — prepend "data:image/png;base64," before use
  draft_report: string; // markdown-formatted text — render with react-markdown
  routing: 'urgent' | 'needs-review' | 'auto-clear';
  justification: string; // plain text, NOT markdown
  reasoning_trace: Array<{
    agent: string;
    summary: string;
    key_evidence: string;
  }>;
  critic_revision_count: number; // 0, 1, or 2
  models_used: Record<string, string>; // keys: vision, drafting, orchestrator, critic (critic optional)
}

export interface AnalysisSummary {
  run_id: string;
  top_class: string;
  top_confidence: number;
  routing: 'urgent' | 'needs-review' | 'auto-clear';
  created_at: string; // ISO 8601
}

export interface HealthCheck {
  status: string;
  environment: {
    GROQ_API_KEY: 'configured' | 'missing';
    GEMINI_API_KEY: 'configured' | 'missing';
    OLLAMA_MODEL: string;
  };
  database: string;
}
