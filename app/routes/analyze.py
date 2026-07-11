"""
MRI Analysis endpoint for NeuroTriage backend.

POST /analyze:
  - Accepts multipart/form-data file upload (JPEG/PNG only)
  - Validates file format and size (max 20MB)
  - Creates AnalysisRun record with status="pending"
  - Returns run_id and status for frontend polling
  - Graph invocation happens in subsequent tasks (Vision Agent, etc.)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from uuid import uuid4
import json

from app.db.models import AnalysisRun, get_db_session

router = APIRouter(prefix="/analyze", tags=["analysis"])

# Maximum file size: 20 MB
MAX_FILE_SIZE = 20 * 1024 * 1024

# Allowed MIME types for image files
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}


@router.post("")
async def analyze_mri(file: UploadFile = File(...)):
    """
    Accept MRI scan upload and create pending analysis record.
    
    This endpoint:
    1. Validates that the uploaded file is JPEG or PNG
    2. Validates that the file size does not exceed 20MB
    3. Creates an AnalysisRun record with status="pending" in the database
    4. Returns the run_id for the frontend to poll for results
    
    The Vision Agent (and subsequent agents) will be invoked in later tasks.
    
    Args:
        file: Uploaded JPEG or PNG image file via multipart/form-data
        
    Returns:
        {
            "run_id": "uuid-string",
            "status": "pending"
        }
        
    Raises:
        HTTPException 422: Invalid file format or exceeds size limit
        
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
    try:
        analysis_run = AnalysisRun(
            id=run_id,
            status="pending",
            original_filename=file.filename
        )
        session.add(analysis_run)
        session.commit()
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create analysis record: {str(e)}"
        )
    finally:
        session.close()
    
    # Return run_id and status to frontend
    return {
        "run_id": run_id,
        "status": "pending"
    }
