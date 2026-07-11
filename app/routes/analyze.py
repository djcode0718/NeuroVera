"""
MRI Analysis endpoint for NeuroTriage backend.

POST /analyze:
  - Accepts multipart/form-data file upload (JPEG/PNG only)
  - Validates file format and size (max 20MB)
  - Creates AnalysisRun record with status="pending"
  - Invokes LangGraph pipeline synchronously
  - Returns complete AnalysisResult with all outputs
  - On error: sets status="failed" and returns HTTP 500

GET /history:
  - Returns list of all past AnalysisRuns

GET /results/{run_id}:
  - Returns a specific AnalysisRun result
"""

import json
import logging
from uuid import uuid4
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.db.models import AnalysisRun, get_db_session
from app.graph.graph import build_graph, build_initial_state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analysis"])

# Maximum file size: 20 MB
MAX_FILE_SIZE = 20 * 1024 * 1024

# Allowed MIME types for image files
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}


# Pydantic models for API responses
class AnalysisSummary(BaseModel):
    run_id: str
    top_class: str
    top_confidence: float
    routing: str
    created_at: str


class AnalysisResult(BaseModel):
    run_id: str
    status: str
    classification: str
    confidence: float
    predictions: dict[str, float]
    gradcam_image: str
    draft_report: str
    routing: str
    justification: str
    reasoning_trace: list[dict]
    critic_revision_count: int
    models_used: dict[str, str]


@router.post("")
async def analyze_mri(file: UploadFile = File(...)):
    """
    Accept MRI scan upload, run full analysis pipeline, and return results.
    
    This endpoint:
    1. Validates that the uploaded file is JPEG or PNG
    2. Validates that the file size does not exceed 20MB
    3. Creates an AnalysisRun record with status="pending" in the database
    4. Invokes the LangGraph pipeline synchronously
    5. Extracts results from all agents
    6. Updates the AnalysisRun record with status="completed" and all results
    7. Returns the complete AnalysisResult
    
    On any error during pipeline execution:
    - Sets status="failed" and error_message in the database record
    - Returns HTTP 500 with user-friendly error message
    
    Args:
        file: Uploaded JPEG or PNG image file via multipart/form-data
        
    Returns:
        AnalysisResult with complete analysis outputs
        
    Raises:
        HTTPException 422: Invalid file format or exceeds size limit
        HTTPException 500: Analysis failed during pipeline execution
        
    Example:
        curl -X POST -F "file=@test_image.jpg" http://localhost:8000/analyze
    """
    
    # Validate content-type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file format. Accepted: JPEG, PNG. Received: {file.content_type}"
        )
    
    # Read file bytes to check size and validate
    file_bytes = await file.read()
    
    # Validate file size
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"File size exceeds maximum of 20MB. Received: {len(file_bytes) / (1024*1024):.1f}MB"
        )
    
    # Ensure file is not empty
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is empty"
        )
    
    # Generate UUID for this analysis run
    run_id = str(uuid4())
    
    # Create AnalysisRun record with status="pending"
    session = get_db_session()
    analysis_run = None
    
    try:
        analysis_run = AnalysisRun(
            id=run_id,
            status="pending",
            original_filename=file.filename
        )
        session.add(analysis_run)
        session.commit()
        logger.info(f"Created pending analysis run: {run_id}")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to create analysis record: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create analysis record: {str(e)}"
        )
    
    # Invoke LangGraph pipeline
    try:
        logger.info(f"Invoking graph for run {run_id}")
        
        graph = build_graph()
        initial_state = build_initial_state(file_bytes)
        final_state = graph.invoke(initial_state)
        
        logger.info(f"Graph execution complete for run {run_id}")
        
        # Extract outputs from final state
        vision_output = final_state.get("vision_output")
        drafting_output = final_state.get("drafting_output")
        orchestrator_output = final_state.get("orchestrator_output")
        critic_retry_count = final_state.get("critic_retry_count", 0)
        
        # Validate that we have all required outputs
        if not vision_output or not drafting_output or not orchestrator_output:
            raise RuntimeError("Pipeline incomplete: missing required agent outputs")
        
        # Extract model names
        models_used = {
            "vision": vision_output.get("model_used", "vgg16"),
            "drafting": drafting_output.get("model_used", "unknown"),
            "orchestrator": orchestrator_output.get("model_used", "unknown")
        }
        
        # If critic ran, include its model
        if final_state.get("critic_output"):
            models_used["critic"] = final_state["critic_output"].get("model_used", "unknown")
        
        # Update analysis_run record with results
        analysis_run.status = "completed"
        analysis_run.top_class = vision_output["top_class"]
        analysis_run.top_confidence = vision_output["top_confidence"]
        analysis_run.predictions_json = json.dumps(vision_output["predictions"])
        analysis_run.gradcam_image_b64 = vision_output.get("gradcam_image", "")
        analysis_run.feature_embedding = json.dumps(vision_output.get("feature_embedding", []))
        analysis_run.draft_report = drafting_output["draft_report"]
        analysis_run.routing = orchestrator_output["routing"]
        analysis_run.justification = orchestrator_output["justification"]
        analysis_run.reasoning_trace_json = json.dumps(orchestrator_output.get("reasoning_trace", []))
        analysis_run.critic_revision_count = critic_retry_count
        analysis_run.models_used_json = json.dumps(models_used)
        
        session.commit()
        logger.info(f"Updated analysis run {run_id} with completed status")
        
        # Build response
        result = AnalysisResult(
            run_id=run_id,
            status="completed",
            classification=vision_output["top_class"],
            confidence=vision_output["top_confidence"],
            predictions=vision_output["predictions"],
            gradcam_image=vision_output.get("gradcam_image", ""),
            draft_report=drafting_output["draft_report"],
            routing=orchestrator_output["routing"],
            justification=orchestrator_output["justification"],
            reasoning_trace=orchestrator_output.get("reasoning_trace", []),
            critic_revision_count=critic_retry_count,
            models_used=models_used
        )
        
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Analysis pipeline failed for run {run_id}: {e}", exc_info=True)
        
        # Update record with error status
        try:
            analysis_run.status = "failed"
            analysis_run.error_message = str(e)
            session.commit()
        except Exception as db_error:
            logger.error(f"Failed to update error status in database: {db_error}")
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
        
    finally:
        session.close()


@router.get("/history")
async def get_history():
    """
    Get history of all past analysis runs.
    
    Returns:
        List of AnalysisSummary objects sorted by created_at descending
    """
    session = get_db_session()
    try:
        runs = session.query(AnalysisRun).filter_by(status="completed").order_by(
            AnalysisRun.created_at.desc()
        ).all()
        
        result = []
        for run in runs:
            result.append({
                "run_id": run.id,
                "top_class": run.top_class,
                "top_confidence": run.top_confidence,
                "routing": run.routing,
                "created_at": run.created_at.isoformat()
            })
        
        return result
        
    finally:
        session.close()


@router.get("/results/{run_id}")
async def get_results(run_id: str):
    """
    Get detailed results for a specific analysis run.
    
    Args:
        run_id: UUID of the analysis run
        
    Returns:
        AnalysisResult with full outputs
        
    Raises:
        HTTPException 404: Run not found or not completed
    """
    session = get_db_session()
    try:
        run = session.query(AnalysisRun).filter_by(id=run_id).first()
        
        if not run:
            raise HTTPException(status_code=404, detail="Analysis run not found")
        
        if run.status != "completed":
            raise HTTPException(status_code=404, detail="Analysis not yet completed")
        
        # Reconstruct AnalysisResult from stored data
        result = {
            "run_id": run.id,
            "status": run.status,
            "classification": run.top_class,
            "confidence": run.top_confidence,
            "predictions": json.loads(run.predictions_json or "{}"),
            "gradcam_image": run.gradcam_image_b64 or "",
            "draft_report": run.draft_report or "",
            "routing": run.routing,
            "justification": run.justification or "",
            "reasoning_trace": json.loads(run.reasoning_trace_json or "[]"),
            "critic_revision_count": run.critic_revision_count,
            "models_used": json.loads(run.models_used_json or "{}")
        }
        
        return result
        
    finally:
        session.close()
