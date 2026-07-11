# NeuroTriage

**A multi-agent AI system for brain MRI analysis** — combining computer vision, retrieval-augmented reporting, independent self-verification, and deterministic clinical triage routing, orchestrated end-to-end with LangGraph.

> ⚠️ **Research prototype. Not a diagnostic tool.** Built on a pretrained, non-clinically-validated classifier. Not for use in real patient care. See [Scope & Limitations](#scope--limitations).

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features & Invariants](#key-features--invariants)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Scope & Limitations](#scope--limitations)

---

## Overview

NeuroTriage analyzes brain MRI scans and classifies them into one of four categories — **glioma, meningioma, pituitary tumor, or no tumor** — then produces a structured clinical report, complete with heatmap-based explainability, similar-case retrieval, and an automated triage routing decision.

What makes this more than a "classify → explain" tool is the **agent architecture**: rather than a single linear pipeline, a dedicated Critic Agent independently cross-checks the drafted report against the raw model evidence, and can send it back for revision if it finds the report overstates certainty or ignores a plausible alternative diagnosis. This verification step is a first-class part of the control flow, not a logging afterthought — implemented as a genuine conditional edge in the LangGraph orchestration graph.

---

## System Architecture

The system runs a **5-agent pipeline** orchestrated with LangGraph:

```
User Upload → FastAPI /analyze endpoint
    │
    ▼
[Vision Agent] ──────> Classifies scan into 4 classes (glioma, meningioma, pituitary, notumor)
    │                  Extracts a 512-dim embedding and generates a Grad-CAM heatmap overlay
    ▼
[Retrieval Agent] ───> Queries the Case Bank for the top-3 similar historical cases (cosine similarity)
    │                  Retrieves clinical reference notes for the predicted tumor type
    ▼
[Drafting Agent] ────> Synthesizes Vision + Retrieval data into a structured medical report
    │                  (Groq → Gemini → Ollama fallback chain, with low-confidence hedging)
    ▼
[Critic Agent] ──────> Independently verifies the draft against raw Vision probabilities only
    │                  Flags overstated certainty or ignored alternative findings
    ▼  Verdict?
    ├─ "revise" (max 2 loops) ──> back to Drafting Agent, with issue notes attached
    └─ "approved" (or retry cap reached)
         │
         ▼
[Orchestrator] ──────> Deterministically routes the case (urgent / needs-review / auto-clear)
    │                  Generates the final clinical justification and reasoning trace
    ▼
API Response ────────> Rendered as interactive results, heatmap, and report in the React frontend
```

**Why this counts as genuinely multi-agent, not just a pipeline:** the Critic → Drafting Agent edge is a *conditional* edge in the LangGraph state machine — the graph's actual execution path is decided at runtime by an agent's own judgment (whether the draft holds up against the evidence), not fixed in advance by application code.

---

## Key Features & Invariants

### 1. Deterministic clinical triage routing
- **Urgent** — classified as `glioma` or `pituitary` with confidence > 75%
- **Auto-clear** — classified as `notumor` with confidence > 80% and no unresolved Critic issues
- **Needs-review** — confidence between 40–75%, or any unresolved Critic issues

### 2. Independent Critic verification
- The Critic Agent receives **only** `vision_output` — never the retrieval context — so its check is grounded in raw model evidence, not influenced by the same text the Drafting Agent already saw
- If confidence is below 60%, the report must use cautious hedging language ("suggests," "may indicate")
- Near-ties (any alternative class within 0.15 of the top class) must be explicitly acknowledged in the draft

### 3. Resilient LLM fallback chain
- **Groq** (`llama-3.1-8b-instant`) tried first, for fast responses
- Falls back to **Gemini** (`gemini-2.5-flash`) on quota/429 errors
- Falls back to **Ollama** (`mistral`, local) as a fully offline-capable last resort
- Gracefully degrades to a vision-only justification if every LLM provider fails — the system never hard-crashes on an LLM outage

### 4. Database resilience
- If SQLite is unavailable or the case bank is empty, the Retrieval Agent returns empty results and the pipeline continues rather than failing the whole run

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python), Conda-managed environment |
| Orchestration | LangGraph (conditional-edge state machine) |
| Vision model | VGG16-based CNN, 4-class brain tumor classifier |
| Explainability | Custom Grad-CAM implementation |
| Retrieval | Cosine similarity over a local case bank (SQLite) |
| LLM providers | Groq → Gemini → Ollama (fallback chain) |
| Frontend | React 19 + TypeScript + Vite |
| Frontend styling | Tailwind CSS 4 |
| Frontend data viz | Recharts |
| Frontend routing | React Router 7 |
| HTTP client | Axios |

---

## Project Structure

```
NeuroTriage/
├── app/
│   ├── main.py                  # FastAPI entry point, CORS, startup model load
│   ├── graph/
│   │   └── graph.py              # LangGraph pipeline definition (nodes + conditional edges)
│   ├── agents/
│   │   ├── vision.py              # Vision Agent — classification, embedding, Grad-CAM
│   │   ├── retrieval.py           # Retrieval Agent — case bank similarity search
│   │   ├── drafting.py            # Drafting Agent — LLM report synthesis
│   │   ├── critic.py              # Critic Agent — independent verification
│   │   └── orchestrator.py        # Orchestrator — deterministic routing + justification
│   ├── models/
│   │   └── vision_loader.py       # VGG16 model loading (with mock fallback for dev)
│   ├── utils/
│   │   └── gradcam.py             # Grad-CAM heatmap generation
│   ├── db/
│   │   ├── models.py               # SQLAlchemy models (AnalysisRun, case bank)
│   │   └── seed_case_bank.py       # Seeds the case bank with reference cases + notes
│   └── routes/
│       └── analyze.py              # /analyze, /history, /results/{run_id} endpoints
├── test_cases/                    # Proof-of-concept test cases (e.g. critic-loop demo)
└── neurotriage-frontend/
    └── src/
        ├── pages/
        │   ├── UploadPage.tsx        # Upload flow
        │   ├── ResultsPage.tsx       # Report + heatmap + reasoning trace display
        │   └── HistoryPage.tsx       # Past analysis runs
        ├── components/
        │   ├── UploadZone.tsx
        │   ├── ResultsDisplay.tsx
        │   ├── HistoryTable.tsx
        │   └── Disclaimer.tsx        # Persistent research-prototype notice
        └── types/
            └── api.ts                 # Shared TypeScript API types
```

---

## Quick Start

### Prerequisites
- Anaconda or Miniforge installed
- Node.js (v18+) and npm
- Ollama running locally (optional — only needed for the offline LLM fallback)

### 1. Set up the Conda environment (backend)

```bash
conda activate neurotriage-env

# Initialize the database and seed the case bank (17 cases + 12 reference notes)
conda run -n neurotriage-env python -c "from app.db.models import init_db; from app.db.seed_case_bank import seed_case_bank; init_db(); seed_case_bank(verbose=True)"
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
OLLAMA_MODEL=mistral
NEUROTRIAGE_DEV_MODE=true  # set true for offline mock-model loading during development
```

### 3. Run the FastAPI backend

```bash
conda run -n neurotriage-env uvicorn app.main:app --reload
# Runs on http://localhost:8000
```

### 4. Run the React frontend

```bash
cd neurotriage-frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

Visit `http://localhost:5173` to use the app.

---

## API Reference

### `POST /analyze`
Accepts a multipart/form-data image upload (JPEG/PNG, max 20MB), runs the full 5-agent LangGraph pipeline synchronously, and returns the complete analysis result — classification, confidence scores, Grad-CAM heatmap, draft report, routing decision, justification, and full reasoning trace.

```bash
curl -X POST -F "file=@scan.jpg" http://localhost:8000/analyze
```

### `GET /history`
Returns a summary list of all past analysis runs (run ID, top class, confidence, routing, timestamp).

### `GET /results/{run_id}`
Returns the full stored result for a specific past analysis run.

### `GET /health`
Returns backend health status, including which LLM API keys are configured and database initialization state.

---

## Scope & Limitations

- **Not a diagnostic tool.** This system has no regulatory clearance and has not been clinically validated on real patient populations or equipment.
- **No pixel-level tumor segmentation.** Classification + Grad-CAM localization only — the model identifies tumor *type* and highlights *where* it focused, not precise tumor boundaries.
- **No real patient data used anywhere**, including in the seed case bank — all data is sourced from a public, anonymized research dataset.
- **Single-user, local-first design** — no authentication or multi-user data isolation in this version.
- **Model preprocessing is non-standard:** the vision model expects raw (unnormalized) pixel values — this is documented in code but is an easy detail to miss if extending or replacing the model.

This project demonstrates system design patterns — explainable classification, retrieval-grounded reporting, independent agent verification, deterministic routing — rather than being a deployable clinical product. Any real-world adoption path would require clinical validation, regulatory approval, and hospital-specific data infrastructure.
