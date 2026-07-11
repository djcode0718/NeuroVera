"""
Retrieval Agent for NeuroTriage

The Retrieval Agent is responsible for:
1. Querying the SQLite case_bank table for historical cases matching the top_class
2. Computing cosine similarity between the query embedding (from Vision Agent)
   and each stored feature_vector (512-dimensional)
3. Returning the top-3 most similar cases sorted by similarity descending
4. Fetching clinical reference notes for the predicted tumor type
5. Gracefully degrading if case_bank is empty

The agent uses dot product / (norm_a * norm_b) for cosine similarity computation.

Example:
    from app.agents.retrieval import run_retrieval_agent
    
    state = {
        "vision_output": {
            "top_class": "glioma",
            "top_confidence": 0.85,
            "feature_embedding": [0.1, 0.2, ...],  # 512 floats
            ...
        },
        "retrieval_output": None
    }
    result = run_retrieval_agent(state)
    print(result["retrieval_output"]["similar_cases"])  # top-3 cases
    print(result["retrieval_output"]["reference_notes"])  # reference texts
"""

import logging
import json
from typing import TypedDict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class SimilarCase(TypedDict):
    """
    Represents a single similar case retrieved from the case bank.
    
    Fields:
        case_id: UUID of the case bank entry
        tumor_type: Classification of the tumor (glioma, meningioma, notumor, pituitary)
        confidence: Confidence score at insertion time
        summary: Plain-language description of the case
        similarity_score: Cosine similarity to query embedding (0.0 to 1.0)
    """
    case_id: str
    tumor_type: str
    confidence: float
    summary: str
    similarity_score: float


class RetrievalOutput(TypedDict):
    """
    Output of the Retrieval Agent.
    
    Fields:
        similar_cases: List of top-3 similar cases from case_bank, sorted by similarity descending
        reference_notes: List of clinical reference note texts for the predicted tumor type
    """
    similar_cases: list[SimilarCase]
    reference_notes: list[str]


class RetrievalError(Exception):
    """Raised when Retrieval Agent encounters an error."""
    pass


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Implements: dot(a, b) / (norm(a) * norm(b))
    
    Args:
        a: First vector (list or numpy array)
        b: Second vector (list or numpy array)
        
    Returns:
        Cosine similarity score in range [0.0, 1.0]
        Returns 0.0 if either vector has zero norm (all zeros)
        
    Example:
        >>> a = [1.0, 0.0, 0.0]
        >>> b = [1.0, 0.0, 0.0]
        >>> cosine_similarity(a, b)
        1.0
        
        >>> a = [1.0, 0.0, 0.0]
        >>> b = [0.0, 1.0, 0.0]
        >>> cosine_similarity(a, b)
        0.0
    """
    try:
        # Convert to numpy arrays for computation
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        
        # Compute dot product
        dot_product = np.dot(a_arr, b_arr)
        
        # Compute norms
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        
        # Handle zero-norm case (all-zeros vector)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        
        # Compute cosine similarity
        similarity = float(dot_product / (norm_a * norm_b))
        
        # Clamp to [0, 1] to handle floating point errors
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
        
    except Exception as e:
        logger.error(f"Error computing cosine similarity: {str(e)}", exc_info=True)
        return 0.0


def run_retrieval_agent(state: dict) -> dict:
    """
    Execute the Retrieval Agent: query case bank and retrieve similar cases + reference notes.
    
    This function implements the Retrieval Agent algorithm from the design document:
    1. Extract vision_output from state
    2. Get top_class and feature_embedding from vision_output
    3. Query case_bank table for entries where tumor_type == top_class
    4. For each case:
       a. Parse feature_vector JSON string to list
       b. Compute cosine similarity between query embedding and stored vector
       c. Store case with computed similarity score
    5. Sort by similarity descending, take top-3 results
    6. Query reference_notes table for top_class
    7. Return RetrievalOutput with similar_cases and reference_notes
    8. Handle empty case_bank gracefully: return empty lists without error
    
    Args:
        state: GraphState dict containing at minimum:
            - "vision_output": VisionOutput dict with:
              - "top_class": string tumor type
              - "feature_embedding": list[float] of length 512
              
    Returns:
        state: Updated GraphState dict with "retrieval_output" populated
        
    Raises:
        RetrievalError: If database access fails or embedding parsing fails
        
    Preconditions:
        - state["vision_output"] is populated by Vision Agent
        - state["vision_output"]["feature_embedding"] is a list of 512 floats
        - Database tables (case_bank, reference_notes) exist and are queryable
        
    Postconditions:
        - state["retrieval_output"]["similar_cases"] contains 0-3 cases sorted by similarity
        - state["retrieval_output"]["reference_notes"] contains reference notes for top_class
        - If case_bank is empty, similar_cases is empty list (no error)
        
    Example:
        from app.agents.retrieval import run_retrieval_agent
        
        state = {
            "vision_output": {
                "top_class": "glioma",
                "top_confidence": 0.85,
                "feature_embedding": [0.1, 0.2, ..., 0.3],  # 512 floats
                "predictions": {"glioma": 0.85, ...},
                "gradcam_image": "data:image/png;base64,...",
            },
            "retrieval_output": None
        }
        result = run_retrieval_agent(state)
        
        print(f"Found {len(result['retrieval_output']['similar_cases'])} similar cases")
        for case in result['retrieval_output']['similar_cases']:
            print(f"  {case['tumor_type']}: similarity={case['similarity_score']:.3f}")
    """
    try:
        logger.info("Starting Retrieval Agent")
        
        # Step 1: Extract vision output from state
        vision_output = state.get("vision_output")
        if vision_output is None:
            raise RetrievalError("vision_output not found in state")
        
        logger.debug("Vision output extracted from state")
        
        # Step 2: Get top_class and feature_embedding
        top_class = vision_output.get("top_class")
        feature_embedding = vision_output.get("feature_embedding")
        
        if top_class is None:
            raise RetrievalError("top_class not found in vision_output")
        if feature_embedding is None:
            raise RetrievalError("feature_embedding not found in vision_output")
        
        logger.info(f"Retrieval for top_class={top_class}, embedding_dim={len(feature_embedding)}")
        
        # Verify embedding is 512-dimensional
        if len(feature_embedding) != 512:
            raise RetrievalError(
                f"feature_embedding has {len(feature_embedding)} dimensions, expected 512"
            )
        
        # Step 3: Query case_bank for matching entries
        try:
            from app.db.models import get_db_session
            session = get_db_session()
            logger.debug(f"Database session created")
        except Exception as e:
            logger.error(f"Failed to create database session: {str(e)}", exc_info=True)
            # Graceful degradation: return empty lists if DB is unavailable
            retrieval_output: RetrievalOutput = {
                "similar_cases": [],
                "reference_notes": []
            }
            state["retrieval_output"] = retrieval_output
            logger.info("Database unavailable; returning empty retrieval results")
            return state
        
        try:
            # Import models inside try block to avoid circular imports
            from app.db.models import CaseBankEntry, ReferenceNote
            
            # Query for all case bank entries matching top_class
            matching_cases = session.query(CaseBankEntry).filter(
                CaseBankEntry.tumor_type == top_class
            ).all()
            
            logger.info(f"Found {len(matching_cases)} case bank entries for {top_class}")
            
        except Exception as e:
            logger.error(f"Failed to query case_bank: {str(e)}", exc_info=True)
            session.close()
            # Graceful degradation: return empty lists if query fails
            retrieval_output: RetrievalOutput = {
                "similar_cases": [],
                "reference_notes": []
            }
            state["retrieval_output"] = retrieval_output
            logger.info("Case bank query failed; returning empty results")
            return state
        
        # Step 4: Compute similarity for each case
        cases_with_similarity = []
        
        for case in matching_cases:
            try:
                # Parse feature_vector JSON
                stored_vector = json.loads(case.feature_vector)
                logger.debug(f"Parsed feature_vector for case {case.id}: {len(stored_vector)} dims")
                
                # Verify stored vector is 512-dimensional
                if len(stored_vector) != 512:
                    logger.warning(
                        f"Case {case.id} has {len(stored_vector)} feature dims, skipping"
                    )
                    continue
                
                # Compute cosine similarity
                similarity = cosine_similarity(feature_embedding, stored_vector)
                
                # Create case dict with similarity
                case_with_sim = {
                    "case": case,
                    "similarity": similarity
                }
                cases_with_similarity.append(case_with_sim)
                
                logger.debug(f"Case {case.id}: similarity={similarity:.4f}")
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse feature_vector for case {case.id}: {str(e)}")
                continue
            except Exception as e:
                logger.warning(f"Error processing case {case.id}: {str(e)}")
                continue
        
        # Step 5: Sort by similarity descending and take top-3
        sorted_cases = sorted(
            cases_with_similarity,
            key=lambda x: x["similarity"],
            reverse=True
        )
        top_3_cases = sorted_cases[:3]
        
        logger.info(f"Retrieved top {len(top_3_cases)} similar cases")
        
        # Convert to SimilarCase objects
        similar_cases: list[SimilarCase] = [
            {
                "case_id": case_item["case"].id,
                "tumor_type": case_item["case"].tumor_type,
                "confidence": float(case_item["case"].confidence_at_insertion),
                "summary": case_item["case"].summary,
                "similarity_score": case_item["similarity"]
            }
            for case_item in top_3_cases
        ]
        
        logger.debug(f"Converted {len(similar_cases)} cases to SimilarCase format")
        
        # Step 6: Query reference_notes for top_class
        try:
            reference_note_objects = session.query(ReferenceNote).filter(
                ReferenceNote.tumor_type == top_class
            ).all()
            
            logger.info(f"Found {len(reference_note_objects)} reference notes for {top_class}")
            
            # Extract note text from objects
            reference_notes = [note.note_text for note in reference_note_objects]
            
        except Exception as e:
            logger.error(f"Failed to query reference_notes: {str(e)}", exc_info=True)
            reference_notes = []
            logger.info("Reference notes query failed; continuing with empty list")
        
        # Close database session
        try:
            session.close()
            logger.debug("Database session closed")
        except Exception as e:
            logger.warning(f"Error closing database session: {str(e)}")
        
        # Step 7: Create and return RetrievalOutput
        retrieval_output: RetrievalOutput = {
            "similar_cases": similar_cases,
            "reference_notes": reference_notes
        }
        
        state["retrieval_output"] = retrieval_output
        logger.info(
            f"Retrieval Agent completed: {len(similar_cases)} similar cases, "
            f"{len(reference_notes)} reference notes"
        )
        
        return state
        
    except RetrievalError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in Retrieval Agent: {str(e)}", exc_info=True)
        raise RetrievalError(f"Retrieval Agent failed: {str(e)}") from e
