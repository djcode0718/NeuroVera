# Requirements Document

## Introduction

NeuroTriage is a full-stack web application that analyzes brain MRI scans using a coordinated team
of five AI agents orchestrated through LangGraph. The system classifies scans into four categories
(glioma, meningioma, no tumor, pituitary tumor), generates a Grad-CAM visual explanation, produces
a plain-language clinical report via an LLM-powered Drafting Agent, and routes the case to an urgency
tier via an Orchestrator Agent. A Critic Agent provides first-class quality verification with the
ability to send the draft back for revision — implementing a feedback loop capped at two retries.

The system is a proof-of-concept research tool. It makes no claim of clinical validity and handles
no real patient data.

---

## Glossary

- **Vision_Agent**: The agent responsible for running the pretrained VGG16 classifier and generating
  Grad-CAM heatmaps.
- **Retrieval_Agent**: The agent responsible for finding similar historical cases from the local
  case bank using cosine similarity on 512-dim embeddings.
- **Drafting_Agent**: The LLM-powered agent that writes the structured plain-language report from
  Vision and Retrieval outputs.
- **Critic_Agent**: The LLM-powered agent that independently cross-checks the draft report against
  raw Vision output and returns a verdict of "approved" or "revise".
- **Orchestrator_Agent**: The final LLM-powered agent that decides the routing tier and writes a
  justification referencing all agent evidence.
- **Graph**: The LangGraph-compiled stateful graph connecting all five agents.
- **GraphState**: The shared in-memory state object passed between agents within a single LangGraph
  run.
- **Backend**: The FastAPI Python 3.11 server that accepts uploads, invokes the Graph, and persists
  results.
- **Frontend**: The React + Vite TypeScript single-page application.
- **System**: The complete NeuroTriage system comprising Backend, Frontend, and all agents.
- **Case_Bank**: The SQLite table of historical case entries with 512-dim feature vectors.
- **LLM_Client**: The component that implements the Groq → Gemini → Ollama fallback chain.
- **DB**: The SQLite database storing analysis runs and the Case_Bank.
- **AnalysisResult**: The JSON object returned to the Frontend containing all agent outputs.
- **CriticRouter**: The LangGraph conditional edge function deciding whether to retry drafting or
  proceed to orchestration.
- **Routing**: One of three urgency tiers: "auto-clear", "needs-review", or "urgent".

---

## Requirements

### Requirement 1: MRI Image Upload

**User Story:** As a researcher, I want to upload a brain MRI scan through a web interface, so that
I can receive an automated analysis without running code manually.

#### Acceptance Criteria

1. WHEN a user submits a JPEG or PNG file through the Frontend upload interface, THE Backend SHALL
   accept the file via a `POST /analyze` endpoint using `multipart/form-data`.
2. WHEN an uploaded file is not a valid JPEG or PNG, THE Backend SHALL reject the request with an
   HTTP 422 response and a descriptive error message before invoking the Graph.
3. WHEN an uploaded file exceeds 20 MB, THE Backend SHALL reject the request with an HTTP 422
   response indicating the size limit.
4. THE Frontend SHALL provide a drag-and-drop upload area that accepts JPEG and PNG files.
5. WHEN a file is selected for upload, THE Frontend SHALL display a loading indicator while the
   Backend processes the request.

---

### Requirement 2: Vision Agent — Classification

**User Story:** As a researcher, I want the system to classify the MRI scan into one of four tumor
categories, so that I have an objective baseline from the pretrained model.

#### Acceptance Criteria

1. THE Vision_Agent SHALL load the `AyanKantiDas/BrainTumorVGG16` Keras model from the HuggingFace
   Hub on Backend startup and cache it for all subsequent requests.
2. WHEN the Vision_Agent processes an image, THE Vision_Agent SHALL resize the image to 150×150×3
   pixels and cast pixel values to float32 WITHOUT dividing by 255.
3. THE Vision_Agent SHALL map classifier output indices to class names using the fixed mapping:
   index 0 = "glioma", index 1 = "meningioma", index 2 = "notumor", index 3 = "pituitary".
4. THE Vision_Agent SHALL return a `predictions` dictionary containing per-class probabilities for
   all four classes, where the values sum to 1.0 within a tolerance of 1e-4.
5. THE Vision_Agent SHALL identify the `top_class` as the class with the highest probability and
   the `top_confidence` as that class's probability.
6. IF the model file cannot be loaded after three attempts with exponential backoff, THEN THE
   Backend SHALL return HTTP 503 to the client with a message indicating the model is unavailable.

---

### Requirement 3: Vision Agent — Grad-CAM

**User Story:** As a researcher, I want a visual heatmap showing which regions of the scan influenced
the classification, so that I can interpret the model's reasoning spatially.

#### Acceptance Criteria

1. THE Vision_Agent SHALL compute a Grad-CAM heatmap using `tf.GradientTape` targeting the
   `dropout_4` layer of the VGG16 model for the predicted top class.
2. THE Vision_Agent SHALL blend the normalized Grad-CAM heatmap onto the original resized image and
   return the result as a base64-encoded PNG string.
3. THE Vision_Agent SHALL extract a 512-dimensional feature embedding from the `dropout_5` layer
   output for use by the Retrieval_Agent.
4. THE Frontend SHALL overlay the Grad-CAM heatmap image on the uploaded scan in the results view.

---

### Requirement 4: Retrieval Agent — Similar Case Lookup

**User Story:** As a researcher, I want the system to surface similar historical cases from a local
case bank, so that the report has grounding in precedent rather than pure model output.

#### Acceptance Criteria

1. THE Retrieval_Agent SHALL query the Case_Bank for entries matching the `top_class` from Vision
   output and compute cosine similarity between the query embedding and each stored 512-dim
   `feature_vector`.
2. THE Retrieval_Agent SHALL return the top-3 most similar cases sorted by cosine similarity in
   non-increasing order.
3. THE Retrieval_Agent SHALL retrieve reference notes from the `reference_notes` table for the
   predicted `top_class` and include them in its output.
4. IF no matching cases are found in the Case_Bank, THEN THE Retrieval_Agent SHALL return an empty
   `similar_cases` list without raising an error.
5. THE Case_Bank SHALL be seeded with at least 10 entries spanning all four tumor classes before
   the system is deployed.

---

### Requirement 5: Drafting Agent — Report Generation

**User Story:** As a researcher, I want a structured, plain-language report of the MRI findings, so
that I can understand the model's conclusions without reading raw probability arrays.

#### Acceptance Criteria

1. WHEN the Drafting_Agent receives Vision and Retrieval outputs on its first pass, THE Drafting_Agent
   SHALL generate a structured report containing all of the following sections: Classification,
   Confidence, Visual Evidence, Similar Cases, Clinical Notes, and Caveats.
2. WHEN `top_confidence` is less than 0.6, THE Drafting_Agent SHALL include explicit hedging
   language in the Confidence section acknowledging the ambiguity.
3. WHEN the Drafting_Agent is called on a revision pass, THE Drafting_Agent SHALL explicitly address
   each issue in the Critic's `issues` list in the revised report.
4. THE Drafting_Agent SHALL record which LLM model was used to generate the report in the
   `model_used` field of its output.
5. THE Drafting_Agent SHALL implement the LLM fallback chain: attempt Groq first; on HTTP 429 or
   quota error fall back to Gemini; on any Gemini error fall back to Ollama.

---

### Requirement 6: Critic Agent — Report Verification

**User Story:** As a researcher, I want an independent agent to verify that the report faithfully
represents the model's raw evidence, so that overconfident or inaccurate claims are caught before
the final result is delivered.

#### Acceptance Criteria

1. THE Critic_Agent SHALL evaluate the draft report using only the raw VisionOutput (predictions,
   top_class, top_confidence, gradcam description) and SHALL NOT have access to the Retrieval
   output during evaluation.
2. WHEN the Critic_Agent detects that the draft overstates certainty relative to `top_confidence`,
   THE Critic_Agent SHALL return `verdict = "revise"` with a specific issue string describing the
   overstatement.
3. WHEN the Critic_Agent detects that the draft ignores non-top-class predictions that are within
   0.15 of the top confidence, THE Critic_Agent SHALL return `verdict = "revise"` with a specific
   issue string.
4. WHEN the Critic_Agent finds no material inaccuracies or unsupported claims in the draft, THE
   Critic_Agent SHALL return `verdict = "approved"` with an empty `issues` list.
5. THE Critic_Agent SHALL implement the same LLM fallback chain as the Drafting_Agent.
6. IF all three LLM providers fail during Critic evaluation, THEN THE Critic_Agent SHALL return
   `verdict = "approved"` with `issues = []` rather than blocking the pipeline.

---

### Requirement 7: Critic Revision Loop

**User Story:** As a researcher, I want the system to automatically revise the report when the
Critic finds problems, so that quality issues are corrected without manual intervention.

#### Acceptance Criteria

1. WHEN THE Critic_Agent returns `verdict = "revise"` AND the current `critic_retry_count` is less
   than 2, THE Graph SHALL route back to the Drafting_Agent with the issues list attached to
   GraphState.
2. WHEN THE Critic_Agent returns `verdict = "revise"` AND `critic_retry_count` equals 2, THE Graph
   SHALL route to the Orchestrator_Agent with the full `critic_issues_history` attached.
3. WHEN THE Critic_Agent returns `verdict = "approved"`, THE Graph SHALL route to the
   Orchestrator_Agent regardless of the current `critic_retry_count`.
4. THE CriticRouter SHALL be implemented as a LangGraph conditional edge, not as application-level
   if/else logic outside the graph.
5. THE Graph SHALL increment `critic_retry_count` by 1 each time the Drafting_Agent is invoked on a
   revision pass.
6. FOR ALL analysis runs, `critic_retry_count` in the final GraphState SHALL be less than or equal
   to 2.

---

### Requirement 8: Orchestrator Agent — Routing Decision

**User Story:** As a researcher, I want a final routing decision that categorizes the case by
urgency, so that I can prioritize follow-up actions.

#### Acceptance Criteria

1. THE Orchestrator_Agent SHALL assign `routing = "urgent"` when `top_class` is "glioma" or
   "pituitary" AND `top_confidence` is greater than 0.75.
2. THE Orchestrator_Agent SHALL assign `routing = "needs-review"` when `top_confidence` is between
   0.4 and 0.75 inclusive, or when `critic_issues_history` is non-empty.
3. THE Orchestrator_Agent SHALL assign `routing = "auto-clear"` only when `top_class` is "notumor"
   AND `top_confidence` is greater than 0.80.
4. THE Orchestrator_Agent SHALL produce a written `justification` string that explicitly references
   specific evidence from Vision output (top class, confidence) and any Critic issues.
5. THE Orchestrator_Agent SHALL produce a `reasoning_trace` list with at least one entry per agent
   that completed successfully, where each entry contains non-empty `agent`, `summary`, and
   `key_evidence` fields.
6. WHEN `critic_issues_history` is non-empty, THE Orchestrator_Agent SHALL note the unresolved
   disagreement in the `justification`.
7. THE Orchestrator_Agent SHALL implement the same LLM fallback chain as the Drafting_Agent.
8. IF all LLM providers fail, THEN THE Orchestrator_Agent SHALL produce a routing decision based
   solely on the Vision output confidence rules (Requirements 8.1, 8.2, 8.3) with a justification
   noting that LLM reasoning was unavailable.

---

### Requirement 9: LLM Fallback Chain

**User Story:** As a researcher, I want the system to remain functional even when a cloud LLM
provider is unavailable, so that I can continue using the tool despite API quotas or outages.

#### Acceptance Criteria

1. THE LLM_Client SHALL attempt the Groq API first using the model configured in `GROQ_FAST_MODEL`
   environment variable.
2. WHEN the Groq API returns an HTTP 429 or quota-exceeded error, THE LLM_Client SHALL fall back
   to the Gemini API using the `gemini-2.5-flash` model without retrying Groq.
3. WHEN the Gemini API returns any error, THE LLM_Client SHALL fall back to the local Ollama
   instance using the model in the `OLLAMA_MODEL` environment variable.
4. WHEN a response is successfully obtained, THE LLM_Client SHALL record the provider name and
   model identifier in a `model_used` string in the format `"provider/model-name"`.
5. IF Ollama is running and the configured model is pulled, THEN THE LLM_Client SHALL always return
   a response (the fallback chain SHALL NOT raise an exception).

---

### Requirement 10: SQLite Persistence

**User Story:** As a researcher, I want analysis results to be stored persistently, so that I can
review past runs without re-uploading images.

#### Acceptance Criteria

1. THE Backend SHALL create and maintain a SQLite database with an `analysis_runs` table containing:
   run ID, created_at timestamp, status, original filename, top_class, top_confidence, predictions
   JSON, gradcam base64 image, feature embedding JSON, draft report, routing, justification,
   reasoning trace JSON, critic revision count, models used JSON, and error message.
2. WHEN an analysis run begins, THE Backend SHALL insert a record with `status = "pending"` before
   invoking the Graph.
3. WHEN an analysis run completes successfully, THE Backend SHALL update the record with
   `status = "completed"` and all result fields.
4. IF an analysis run fails, THEN THE Backend SHALL update the record with `status = "failed"` and
   the error message.
5. THE Backend SHALL expose a `GET /history` endpoint that returns a list of past analysis
   summaries from the `analysis_runs` table.
6. THE Case_Bank SHALL be stored in the same SQLite database as a `case_bank` table with: entry ID,
   tumor type, confidence at insertion, summary text, 512-dim feature vector JSON, source file
   reference, and created_at timestamp.

---

### Requirement 11: Frontend Report Display

**User Story:** As a researcher, I want to see the full analysis results in a clear web interface,
so that I can review findings, Grad-CAM visuals, and the agent trace in one place.

#### Acceptance Criteria

1. WHEN an analysis completes, THE Frontend SHALL display the top predicted class and confidence
   percentage prominently.
2. THE Frontend SHALL display the Grad-CAM heatmap overlaid on the uploaded scan image.
3. THE Frontend SHALL display a confidence bar chart showing the probability for all four classes.
4. THE Frontend SHALL display the full draft report text in a readable format.
5. THE Frontend SHALL display the routing decision using a color-coded badge: green for "auto-clear",
   yellow for "needs-review", red for "urgent".
6. THE Frontend SHALL display the Orchestrator's justification text.
7. THE Frontend SHALL display the agent reasoning trace showing each agent's name, summary, and
   key evidence.
8. THE Frontend SHALL indicate how many Critic revision cycles occurred for the run.
9. THE Frontend SHALL display which LLM model was used by each agent.

---

### Requirement 12: Agent Trace Panel

**User Story:** As a researcher, I want to observe each agent's progress as the pipeline runs, so
that I can understand the multi-agent workflow and identify where time is being spent.

#### Acceptance Criteria

1. WHILE the Backend is processing a request, THE Frontend SHALL display an agent trace panel
   showing each of the five agents with a status indicator (pending, running, done).
2. THE Frontend SHALL update the status of each agent as the pipeline progresses, using Server-Sent
   Events or polling the Backend for status updates.
3. WHEN the full pipeline completes, THE Frontend SHALL transition the trace panel from live status
   to a completed-run summary view.

---

### Requirement 13: Proof-of-Concept Critic Loop Demo

**User Story:** As a developer, I want a documented test case showing the Critic loop in action on
an ambiguous scan, so that I can demonstrate the system's verification capability.

#### Acceptance Criteria

1. THE System SHALL include a test script that processes `Testing/glioma/Te-gl_110.jpg` through
   both a standalone Drafting_Agent (no Critic) and the full pipeline.
2. WHEN the test script runs with `Te-gl_110.jpg` (expected confidence ≈0.50), THE Critic_Agent
   SHALL flag overstated certainty in the initial draft and return `verdict = "revise"`.
3. THE test script SHALL save both outputs (draft without Critic and final revised draft) side by
   side in `test_cases/critic_loop_demo.md` along with all Critic issues raised during the run.
4. THE `test_cases/critic_loop_demo.md` file SHALL include: section headers distinguishing the
   two drafts, the list of issues raised per revision cycle, and which LLM model was used.

---

### Requirement 14: Non-Goals and Disclaimers

**User Story:** As a researcher, I want the system to clearly communicate its limitations, so that
it is not mistaken for a clinical diagnostic tool.

#### Acceptance Criteria

1. THE Frontend SHALL display a visible disclaimer on every page stating that NeuroTriage is a
   research proof-of-concept and makes no claim of clinical validity.
2. THE System SHALL not implement user authentication, multi-user sessions, or any storage of real
   patient data.
3. THE System SHALL not perform pixel-level tumor segmentation.

