# Implementation Plan: NeuroTriage

## Overview

This task list implements NeuroTriage in eight sequential phases:
1. FastAPI skeleton with VGG16 classifier integration
2. Grad-CAM visual explanations
3. LangGraph orchestration framework
4. Critic Agent with revision loop
5. Retrieval Agent with case bank seeding
6. React frontend with upload and report display
7. Agent trace UI for live pipeline observation
8. Fallback handling, error polish, and documentation

Each phase builds on the previous, with all tasks using `conda run -n neurotriage-env` for environment consistency.

---

## Tasks

- [x] 1. Phase 1: FastAPI skeleton and VGG16 integration
  - [x] 1.1 Set up FastAPI project structure
    - Initialize project with `conda run -n neurotriage-env pip install fastapi uvicorn`
    - Create directory structure: `app/`, `app/agents/`, `app/models/`, `app/db/`, `test_cases/`
    - Create `app/main.py` with FastAPI app definition
    - Set up environment variable loading (`.env` file with `GROQ_API_KEY`, `GEMINI_API_KEY`, `OLLAMA_MODEL`)
    - _Requirements: 1.1_

  - [x] 1.2 Create SQLite database models and schema
    - Define `AnalysisRun` SQLAlchemy model with all fields from design
    - Define `CaseBankEntry` and `ReferenceNote` models
    - Create database initialization function with `sqlite:///neurotriage.db`
    - _Requirements: 10.1, 10.6_

  - [x] 1.3 Load VGG16 model from HuggingFace on startup
    - Install required packages: `conda run -n neurotriage-env pip install tensorflow keras huggingface_hub pillow numpy`
    - Implement model singleton loader that fetches `AyanKantiDas/BrainTumorVGG16` once at app startup
    - Create `app/models/vision_loader.py` with `load_vgg16_model()` function
    - Add exponential backoff retry logic (max 3 attempts) with HTTP 503 fallback on failure
    - Verify model loads with cache by running app once and checking file exists
    - _Requirements: 2.1, 2.6_

  - [x] 1.4 Implement Vision Agent inference function
    - Create `app/agents/vision.py` with `VisionOutput` TypedDict and `run_vision_agent(state)` function
    - Resize input image to 150×150×3, cast to float32 WITHOUT dividing by 255
    - Map classifier output indices to class names: 0="glioma", 1="meningioma", 2="notumor", 3="pituitary"
    - Verify output probabilities sum to 1.0 within 1e-4 tolerance
    - Identify `top_class` and `top_confidence` from predictions dict
    - _Requirements: 2.2, 2.3, 2.4, 2.5_

  - [x] 1.5 Create `/analyze` endpoint skeleton
    - Create `app/routes/analyze.py` with `POST /analyze` endpoint
    - Accept multipart/form-data file upload with validation (JPEG/PNG only, max 20MB)
    - Insert `AnalysisRun` record with `status="pending"` before invoking graph
    - Return HTTP 422 for invalid files before invoking graph
    - Return HTTP 422 for files exceeding 20MB size limit
    - _Requirements: 1.1, 1.2, 1.3, 10.2_

  - [ ]* 1.6 Write unit tests for Vision Agent
    - Test image preprocessing without normalization (pixel range [0, 255])
    - Test probability sum constraint (1.0 ± 1e-4)
    - Test class name mapping (indices 0-3 → correct strings)
    - Test `top_class` equals argmax of predictions
    - _Requirements: 2.2, 2.3, 2.4, 2.5_

- [x] 2. Phase 1b: Checkpoint - Vision pipeline functional
  - Ensure Vision Agent loads model successfully, accepts image input, and returns valid predictions
  - Test with a sample image from `Testing/` directory
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Phase 2: Grad-CAM integration
  - [x] 3.1 Implement Grad-CAM computation in Vision Agent
    - Use `tf.GradientTape` to compute gradients targeting `dropout_4` layer
    - Compute heatmap by multiplying conv output by pooled gradients
    - Normalize heatmap to range [0, 255], blend onto original resized image
    - Return as base64-encoded PNG string in `gradcam_image` field
    - _Requirements: 3.1, 3.2_

  - [x] 3.2 Extract 512-dim feature embedding from dropout_5
    - Create sub-model extracting output from `dropout_5` layer
    - Return as `feature_embedding: list[float]` with exactly 512 elements
    - Verify embedding vector is not all-zeros
    - _Requirements: 3.3_

  - [x] 3.3 Wire Grad-CAM and embedding extraction into Vision output
    - Update `run_vision_agent()` to populate both `gradcam_image` and `feature_embedding` in output
    - Add to `VisionOutput` TypedDict fields
    - Verify embedding dimension in unit tests
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ]* 3.4 Write property test for Vision output normalization
    - **Property 1: Vision output probability normalization**
    - **Validates: Requirements 2.4, 2.5**
    - Use `hypothesis` to generate random valid MRI images and verify probabilities sum to 1.0 ± 1e-4

  - [ ]* 3.5 Write property test for raw pixel input invariant
    - **Property 2: Raw pixel input invariant**
    - **Validates: Requirements 2.2**
    - Verify pixel array passed to model contains values in [0.0, 255.0], never divided by 255

- [x] 4. Phase 2b: Checkpoint - Grad-CAM visible
  - Ensure Grad-CAM heatmap is computed and returned as valid base64 PNG
  - Verify embedding has exactly 512 floats
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Phase 3: LangGraph skeleton with Vision, Retrieval, Drafting, Orchestrator
  - [x] 5.1 Set up LangGraph graph structure
    - Install `conda run -n neurotriage-env pip install langgraph langchain langchain-groq google-generativeai ollama`
    - Create `app/graph/graph.py` with `GraphState` TypedDict containing all fields from design
    - Define `build_graph()` function returning compiled `CompiledGraph`
    - Add nodes for: vision, retrieval, drafting, orchestrator (no critic yet)
    - _Requirements: 7.4_

  - [x] 5.2 Implement Retrieval Agent with case bank query
    - Create `app/agents/retrieval.py` with `RetrievalOutput` TypedDict and `run_retrieval_agent(state)` function
    - Query SQLite `case_bank` table for entries matching `top_class`
    - Implement cosine similarity computation: `dot(a,b) / (norm(a) * norm(b))`
    - Return top-3 cases sorted by similarity descending
    - Fetch reference notes from `reference_notes` table for predicted class
    - Return empty lists if case bank has no entries (graceful degradation)
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 5.3 Implement Drafting Agent with LLM fallback chain
    - Create `app/agents/drafting.py` with `DraftingOutput` TypedDict and `run_drafting_agent(state)` function
    - Implement LLM fallback chain: Groq → Gemini → Ollama
    - First pass: synthesize Vision + Retrieval into structured report with 6 sections (Classification, Confidence, Visual Evidence, Similar Cases, Clinical Notes, Caveats)
    - When `top_confidence < 0.6`, include hedging language in Confidence section
    - Track which LLM model was used in `model_used` field
    - _Requirements: 5.1, 5.2, 5.5, 9.1, 9.2, 9.3_

  - [x] 5.4 Implement Orchestrator Agent with routing logic
    - Create `app/agents/orchestrator.py` with `OrchestratorOutput` TypedDict and `run_orchestrator_agent(state)` function
    - Implement routing rules: urgent if (glioma or pituitary) AND confidence > 0.75
    - Route to needs-review if 0.4 <= confidence <= 0.75
    - Route to auto-clear only if notumor AND confidence > 0.80
    - Write justification referencing Vision evidence explicitly
    - Generate reasoning_trace with one entry per agent (agent name, summary, key_evidence)
    - Use same LLM fallback chain as Drafting Agent
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.7_

  - [~] 5.5 Wire agents into straight-line graph
    - Add sequential edges: vision → retrieval → drafting → orchestrator
    - No conditional edges yet (no critic)
    - Compile graph with `graph.compile()`
    - Test end-to-end invoke on test image
    - _Requirements: 7.4_

  - [ ]* 5.6 Write property test for LLM fallback
    - **Property 6: LLM fallback always returns**
    - **Validates: Requirements 9.4, 9.5**
    - Mock Ollama as only available provider, verify fallback chain returns response

  - [ ]* 5.7 Write property test for routing validity
    - **Property 7: Routing validity and correctness**
    - **Validates: Requirements 8.1, 8.2, 8.3**
    - Use `hypothesis` to generate confidence/class combinations, verify routing matches rules

- [~] 6. Phase 3b: Checkpoint - Full pipeline without Critic
  - Test with sample image from `Testing/glioma/` directory
  - Verify routing decision is one of three valid tiers
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Phase 4: Critic Agent and revision loop with proof-of-concept
  - [~] 7.1 Implement Critic Agent
    - Create `app/agents/critic.py` with `CriticOutput` TypedDict and `run_critic_agent(state)` function
    - Receive draft report + raw VisionOutput (DELIBERATELY NOT retrieval context)
    - Check if draft confidence matches `top_confidence` (flag overstated certainty)
    - Check if non-top classes within 0.15 of top confidence are ignored (flag missed findings)
    - Check if Grad-CAM regions described accurately
    - Return `verdict="revise"` with specific issue strings when failures detected
    - Return `verdict="approved"` when draft faithfully represents Vision output
    - Use same LLM fallback chain; on all-providers-fail, return `verdict="approved"` rather than raising
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 9.1, 9.2, 9.3_

  - [~] 7.2 Implement critic_router conditional edge
    - Create `critic_router(state: GraphState) -> str` function
    - Return `"drafting"` if `verdict=="revise"` AND `retry_count < 2`
    - Return `"orchestrator"` if `verdict=="approved"` OR `retry_count >= 2`
    - Never raises exception
    - _Requirements: 7.1, 7.2, 7.3_

  - [~] 7.3 Wire Critic into graph with conditional edge
    - Add drafting → critic sequential edge
    - Add conditional edge after critic using `critic_router`
    - Increment `critic_retry_count` on each revision routing to drafting
    - Accumulate `critic_issues_history` list
    - Compile updated graph
    - _Requirements: 7.1, 7.4, 7.5, 7.6_

  - [~] 7.4 Update Drafting Agent to handle revision pass
    - When `critic_retry_count > 0`, accept `critic_issues` from state
    - On revision pass, re-draft report explicitly addressing each issue
    - Include issue acknowledgment language in revised report
    - Increment `revision_number` field in DraftingOutput
    - _Requirements: 5.3, 7.1_

  - [~] 7.5 Create test case: critic_loop_demo.md
    - Process `Testing/glioma/Te-gl_110.jpg` through Drafting Agent alone (no Critic)
    - Process same image through full pipeline with Critic
    - Save both drafts side-by-side in `test_cases/critic_loop_demo.md` with clear section headers
    - Include list of all Critic issues raised per revision cycle
    - Document which LLM models were used by each agent
    - Verify Critic triggers at least one revision (confidence ≈0.50 should flag overstated certainty)
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [ ]* 7.6 Write property test for Critic router
    - **Property 5: Critic router completeness and revision cap**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.5, 7.6**
    - Test all combinations of (verdict, retry_count) pairs

  - [ ]* 7.7 Write property test for reasoning trace completeness
    - **Property 8: Reasoning trace completeness**
    - **Validates: Requirements 8.4, 8.5, 8.6**
    - Verify trace has entry per agent, with non-empty fields

- [~] 8. Phase 4b: Checkpoint - Critic revision loop functional
  - Run critic_loop_demo.md test case and verify Critic flags issues and Drafting revises report
  - Verify revision count never exceeds 2
  - Verify both drafts appear in test_cases/critic_loop_demo.md
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Phase 5: Retrieval Agent case bank seeding
  - [~] 9.1 Create 10–20 seed case bank entries
    - Create at least 10 diverse entries spanning all 4 tumor classes
    - Each entry must include: tumor_type, confidence_at_insertion, summary text, 512-dim feature_vector
    - Use pre-computed embeddings (can be synthetic random vectors for now) seeded with consistent numpy random state
    - Source: can be from existing dataset or manually created summaries
    - Insert into SQLite `case_bank` table using SQLAlchemy
    - _Requirements: 4.5_

  - [~] 9.2 Populate reference notes table
    - Add reference notes for each tumor type (e.g., WHO classification snippets, clinical definitions)
    - Insert at least 2 notes per tumor type into `reference_notes` table
    - Reference notes appear in Retrieval output and quoted in draft reports
    - _Requirements: 4.3_

  - [ ]* 9.3 Write property test for retrieval ordering
    - **Property 4: Retrieval ordering monotonicity**
    - **Validates: Requirements 4.1, 4.2**
    - Generate random embeddings, verify similar_cases sorted descending by similarity_score
    - Verify cosine_similarity(v, v) ≈ 1.0 for any non-zero vector

- [~] 10. Phase 5b: Checkpoint - Retrieval functional with seed data
  - Run Retrieval Agent on test image and verify top-3 cases returned
  - Verify cases sorted by similarity descending
  - Verify reference notes appear in Retrieval output
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Phase 6: React frontend upload and results display
  - [~] 11.1 Set up Vite + React + TypeScript project
    - Initialize: `npm create vite@latest neurotriage-frontend -- --template react-ts`
    - Install dependencies: `npm install`
    - Set up TypeScript configuration
    - Create directory structure: `src/components/`, `src/pages/`, `src/hooks/`, `src/types/`
    - _Requirements: 11.1_

  - [~] 11.2 Create upload component with drag-and-drop
    - Implement drag-and-drop upload area using `react-dropzone`
    - Accept JPEG and PNG files
    - Display loading indicator during upload
    - POST to Backend `/analyze` endpoint with multipart/form-data
    - Handle HTTP 422 errors with user-friendly messages
    - _Requirements: 1.4, 1.5_

  - [~] 11.3 Display Grad-CAM heatmap overlay
    - Render uploaded image with Grad-CAM heatmap overlaid
    - Use base64 `gradcam_image` from response
    - Show on top of original image with transparency/blending
    - _Requirements: 3.4, 11.2_

  - [~] 11.4 Display classification results prominently
    - Show top predicted class in large readable format
    - Display confidence percentage
    - Show confidence bar chart with all four classes using `recharts`
    - _Requirements: 11.1, 11.3_

  - [~] 11.5 Display full draft report and routing decision
    - Render draft report text in readable section
    - Show routing decision with color-coded badge (green=auto-clear, yellow=needs-review, red=urgent)
    - Display Orchestrator's justification text
    - _Requirements: 11.4, 11.5, 11.6_

  - [~] 11.6 Display agent reasoning trace
    - Render reasoning_trace list showing agent name, summary, key_evidence
    - Display in table or card format
    - Show which LLM model was used by each agent
    - Show Critic revision count
    - _Requirements: 11.7, 11.8, 11.9_

  - [~] 11.7 Create history page with past analyses
    - Implement `GET /history` Backend endpoint returning list of `AnalysisSummary` from SQLite
    - Display history as table with run ID, class, confidence, routing, created_at
    - Allow clicking row to re-display full results
    - _Requirements: 10.5_

  - [~] 11.8 Add disclaimer on every page
    - Display visible disclaimer stating NeuroTriage is research PoC with no clinical validity
    - Must appear on upload page and results page
    - _Requirements: 14.1_

  - [ ]* 11.9 Write integration tests for upload flow
    - Test file validation (reject non-image files)
    - Test display of results after successful upload
    - Test history page retrieval

- [~] 12. Phase 6b: Checkpoint - Frontend displays full analysis
  - Upload test image and verify all results display correctly
  - Verify Grad-CAM visible on original image
  - Verify all agent outputs displayed
  - Verify routing badge visible with correct color
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Phase 7: Agent trace UI with live progress
  - [~] 13.1 Implement Server-Sent Events for pipeline status streaming
    - Add `/analyze-stream` endpoint to Backend that streams agent status updates
    - Yield JSON events with format: `{agent: str, status: "pending"|"running"|"done", summary?: str}`
    - Emit status update when each agent completes
    - Close stream when full pipeline completes
    - _Requirements: 12.1, 12.2_

  - [~] 13.2 Create agent trace panel component
    - React component displaying live agent status
    - Show all five agents (Vision, Retrieval, Drafting, Critic, Orchestrator) as rows
    - Status indicator (pending → running → done) per agent
    - Update in real-time as Backend streams status updates
    - _Requirements: 12.1, 12.2_

  - [~] 13.3 Transition trace panel to completed summary
    - When pipeline completes, convert trace panel from live view to completed summary
    - Show final status of each agent
    - Show reasonin_trace with each agent's contribution
    - _Requirements: 12.3_

  - [~] 13.4 Integrate trace panel into upload flow
    - Show trace panel during Backend processing (instead of generic loading spinner)
    - Hide trace panel on error and show error message
    - _Requirements: 12.1, 12.2, 12.3_

  - [ ]* 13.5 Write integration tests for streaming API
    - Test `/analyze-stream` emits correct status events in correct order
    - Test streaming terminates on completion

- [~] 14. Phase 7b: Checkpoint - Agent trace visible during execution
  - Upload test image and watch agent trace update in real-time
  - Verify all five agents show correct status progression
  - Verify trace panel transitions to summary on completion
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. Phase 8: Fallback handling and error polish
  - [~] 15.1 Implement Orchestrator fallback when all LLM providers fail
    - If LLM unavailable during Orchestrator Agent execution, use Vision-only routing rules (Req 8.1-8.3)
    - Generate justification noting LLM unavailable
    - Route to "needs-review" as default safe tier
    - _Requirements: 8.8_

  - [~] 15.2 Add comprehensive error handling
    - Wrap graph invocation in try/except to catch all agent failures
    - Log errors with context (which agent failed, error message)
    - Return HTTP 500 with user-friendly error message
    - Update `analysis_run` record with `status="failed"` and error_message
    - _Requirements: 1.2, 1.3_

  - [~] 15.3 Handle database write failures gracefully
    - If SQLite INSERT fails, log error but still return analysis result to user
    - Do not block user from seeing results due to persistence failure
    - _Requirements: 10.2, 10.3_

  - [~] 15.4 Add request size validation
    - Validate file size before multipart parsing (max 20MB)
    - Return HTTP 413 Payload Too Large if exceeded
    - _Requirements: 1.3_

  - [~] 15.5 Implement Ollama availability check on startup
    - At FastAPI startup, verify Ollama is running with configured model pulled
    - If Ollama unavailable, warn in logs but continue (allows Groq/Gemini to work)
    - _Requirements: 9.5_

  - [~] 15.6 Write comprehensive README
    - Installation instructions (conda env setup, model cache)
    - Environment variables documentation (.env template)
    - Running the Backend and Frontend
    - Test image location and how to run proof-of-concept
    - Disclaimer about research-only nature
    - Architecture overview and component descriptions
    - Citation and references
    - _Requirements: 14.1_

  - [ ]* 15.7 Write property test for draft structure
    - **Property 9: Draft report structure and hedging**
    - **Validates: Requirements 5.1, 5.2**
    - Verify report contains all 6 section headers
    - Verify hedging language appears when confidence < 0.6

  - [ ]* 15.8 Write property test for invalid input rejection
    - **Property 10: Invalid input rejection**
    - **Validates: Requirements 1.2**
    - Test that non-image bytes rejected with HTTP 422

- [~] 16. Phase 8b: Final checkpoint
  - Run full end-to-end test with `Testing/glioma/Te-gl_110.jpg`
  - Verify all error handling paths work
  - Verify README is comprehensive and runnable
  - Ensure all tests pass, ask the user if questions arise.

- [~] 17. Deployment verification
  - Verify all dependencies installable via `conda run -n neurotriage-env`
  - Verify both Backend and Frontend start without errors
  - Verify `/analyze` endpoint accepts valid image and returns complete result
  - Verify `/history` endpoint returns past runs
  - Verify Frontend loads and displays disclaimer
  - Document any manual setup steps in README

---

## Notes

- All tasks use `conda run -n neurotriage-env` to run commands in the isolated environment
- Test tasks marked with `*` are optional and can be skipped for faster MVP, but strongly recommended for quality assurance
- Property-based tests validate universal correctness properties from design document
- Unit tests validate specific examples and edge cases
- Core implementation tasks (unmarked) MUST be completed in order
- Checkpoints ensure incremental validation at reasonable breaks
- Critic_loop_demo.md is included in Phase 4 as a key deliverable demonstrating system capability
- Optional test sub-tasks include unit tests, property tests, and integration tests
- Each task references specific requirements for traceability to feature spec

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "1.4", "1.5"] },
    { "id": 2, "tasks": ["1.6", "3.1", "3.2"] },
    { "id": 3, "tasks": ["3.3", "3.4", "3.5"] },
    { "id": 4, "tasks": ["5.1", "5.2"] },
    { "id": 5, "tasks": ["5.3", "5.4", "5.5"] },
    { "id": 6, "tasks": ["5.6", "5.7"] },
    { "id": 7, "tasks": ["7.1", "7.2", "7.3"] },
    { "id": 8, "tasks": ["7.4", "7.5", "7.6", "7.7"] },
    { "id": 9, "tasks": ["9.1", "9.2", "9.3"] },
    { "id": 10, "tasks": ["11.1", "11.2"] },
    { "id": 11, "tasks": ["11.3", "11.4", "11.5"] },
    { "id": 12, "tasks": ["11.6", "11.7", "11.8", "11.9"] },
    { "id": 13, "tasks": ["13.1", "13.2", "13.3", "13.4", "13.5"] },
    { "id": 14, "tasks": ["15.1", "15.2", "15.3", "15.4", "15.5", "15.6", "15.7", "15.8"] }
  ]
}
```
