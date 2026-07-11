# NeuroTriage MVP

NeuroTriage is a multi-agent AI system that analyzes brain MRI scans to assist radiologists in tumor classification and clinical triage. The system combines computer vision (VGG16 neural network), retrieval-augmented generation (cosine similarity against a historical case bank), independent validation (Critic Agent), and LLM-synthesized clinical reports with LangGraph orchestration.

---

## System Architecture

The NeuroTriage system follows a **5-agent orchestration architecture** built with LangGraph:

```
User Upload → FastAPI /analyze endpoint
    ↓
[Vision Agent] ──────> Classifies scan into 4 classes (glioma, meningioma, pituitary, notumor)
    │                  Extracts 512-dim embedding and generates Grad-CAM heatmap overlay
    ↓
[Retrieval Agent] ───> Queries Case Bank for top-3 similar historical cases (cosine similarity)
    │                  Retrieves clinical reference notes for the predicted tumor type
    ↓
[Drafting Agent] ────> Synthesizes Vision + Retrieval data into a 6-section medical report
    │                  (using Groq -> Gemini -> Ollama fallback chain with 低置信度 hedging)
    ↓
[Critic Agent] ──────> Independently verifies draft report against raw Vision probabilities
    │                  Checks for overstated certainty or ignored alternative findings
    ↓  [Revise?]
    ├─ Verdict: "revise" (max 2 loops) ─> Sends back to Drafting Agent with issue notes
    └─ Verdict: "approved" (or cap reached)
         ↓
[Orchestrator] ──────> Deterministically routes case ("urgent", "needs-review", "auto-clear")
    │                  Generates final clinical justification and maps reasoning trace
    ↓
API Response ────────> Displays interactive visualizations and reports on React Frontend
```

---

## Key Features & Invariants

1. **Deterministic Clinical Triage Routing**:
   - **Urgent**: Classified as `glioma` or `pituitary` with confidence > 75%.
   - **Auto-Clear**: Classified as `notumor` with confidence > 80% and no unresolved Critic issues.
   - **Needs-Review**: Confidence between 40% and 75%, OR any unresolved Critic issues.
2. **Independent Critic Verification**:
   - The Critic Agent receives *only* `vision_output` (no retrieval context) to ensure unbiased verification.
   - If confidence is < 60%, the report must use cautious hedging language (e.g., *"suggests"*, *"may indicate"*).
   - Near-ties (any alternative class within 0.15 of the top class) must be acknowledged in the draft report.
3. **Resilient LLM Fallback Chain**:
   - Tries **Groq** (`llama-3.1-8b-instant`) first for rapid responses.
   - Falls back to **Gemini** (`gemini-2.5-flash`) on quota/429 limits.
   - Falls back to **Ollama** (`mistral` locally) as a robust, offline-capable option.
   - Gracefully downgrades to **Vision-Only** justification and fallback reasoning if all LLM providers fail.
4. **Database Resilience**:
   - If SQLite is unavailable or the case bank is empty, the Retrieval Agent gracefully returns empty lists and the analysis proceeds seamlessly.

---

## Quick Start

### Prerequisites
- Anaconda or Miniforge installed.
- Node.js (v18+) and npm installed.
- Ollama running locally (optional, for local model execution).

### 1. Setup Conda Environment (Backend)
Navigate to the root directory and activate the environment:
```bash
# Activate environment
conda activate neurotriage-env

# Initialize database and seed case bank (17 cases + 12 reference notes)
conda run -n neurotriage-env python -c "from app.db.models import init_db; from app.db.seed_case_bank import seed_case_bank; init_db(); seed_case_bank(verbose=True)"
```

### 2. Configure Environment Variables
Copy `.env.example` or update your `.env` file in the root with valid API keys:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
OLLAMA_MODEL=mistral
NEUROTRIAGE_DEV_MODE=true  # Set to true for offline mock-model loading
```

### 3. Run FastAPI Backend
Start the server:
```bash
conda run -n neurotriage-env uvicorn app.main:app --reload
# Starts on http://localhost:8000
```

### 4. Run React Frontend
Navigate to the frontend folder, install dependencies, and run:
```bash
cd neurotriage-frontend
npm install
npm run dev
# Starts on http://localhost:5173
```

---

## API Documentation

### `POST /analyze`
Uploads an MRI scan for analysis.
- **Request**: `multipart/form-data` with field `file` containing JPEG or PNG up to 20MB.
- **Response** (HTTP 200):
```json
{
  "run_id": "uuid",
  "status": "completed",
  "classification": "glioma",
  "confidence": 0.85,
  "predictions": {
    "glioma": 0.85,
    "meningioma": 0.10,
    "notumor": 0.03,
    "pituitary": 0.02
  },
  "gradcam_image": "data:image/png;base64,...",
  "draft_report": "markdown text...",
  "routing": "urgent",
  "justification": "justification text...",
  "reasoning_trace": [
    {
      "agent": "vision",
      "summary": "Classified as glioma with 85.0% confidence",
      "key_evidence": "glioma: 85.0%"
    }
  ],
  "critic_revision_count": 0,
  "models_used": {
    "vision": "vgg16",
    "drafting": "groq/llama-3.1-8b-instant",
    "orchestrator": "groq/llama-3.1-8b-instant"
  }
}
```

### `GET /history`
Retrieves past completed analyses, sorted newest first.
- **Response** (HTTP 200):
```json
[
  {
    "run_id": "uuid",
    "top_class": "glioma",
    "top_confidence": 0.85,
    "routing": "urgent",
    "created_at": "2026-07-11T12:00:00"
  }
]
```

### `GET /results/{run_id}`
Retrieves full result details for a specific run.
- **Response** (HTTP 200): Same structure as `POST /analyze` success response.

---

## Testing

To run the full backend test suites:
```bash
# Run unit, integration and API tests
conda run -n neurotriage-env pytest
```

Individual test targets:
- `pytest test_api_endpoints.py` (API validation, health check, endpoints)
- `pytest test_orchestrator.py` (Orchestrator routing, justifications, traces)
- `pytest test_vision_loader.py` (Singleton loader, retry backoffs, mocks)
- `pytest app/agents/test_retrieval.py` (Cosine similarity, case retrieval)
- `pytest test_drafting_agent.py` (LLM prompt synthesis, section completeness)

---

## Troubleshooting Guide

- **Protobuf Version Errors**:
  If you encounter `Detected incompatible Protobuf Gencode/Runtime versions` errors when running TensorFlow/Keras, ensure `protobuf` is upgraded to a compatible version in your active conda environment:
  ```bash
  conda run -n neurotriage-env pip install -U protobuf
  ```
- **HuggingFace Hub Download Timeout / Offline Mode**:
  If you are running in an offline sandbox or have proxy blocks, set `NEUROTRIAGE_DEV_MODE=true` in your `.env`. This triggers the immediate mock-model bypass for Keras/VGG16 without hanging on downloads.
- **Frontend 404 on API Routes**:
  Verify your Vite proxy is running on `/api`. If you request `/api/history`, Vite will rewrite this to `http://localhost:8000/history` which is handled by the backend's `/history` router. Ensure your backend is running on port `8000`.
