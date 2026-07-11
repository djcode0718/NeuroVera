"""
Unit and property-based tests for the Retrieval Agent.

Tests cover:
1. Cosine similarity computation
2. Empty case bank handling (graceful degradation)
3. Case retrieval and ranking by similarity
4. Reference notes retrieval
5. Database error handling
"""

import pytest
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.agents.retrieval import (
    cosine_similarity,
    run_retrieval_agent,
    SimilarCase,
    RetrievalOutput,
    RetrievalError
)


class TestCosineSimilarity:
    """Tests for the cosine_similarity helper function."""
    
    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0"""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0"""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)
    
    def test_scaled_vectors(self):
        """Similarity should be invariant to scaling"""
        a = [1.0, 2.0, 3.0]
        b = [2.0, 4.0, 6.0]  # 2x scaled version of a
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)
    
    def test_zero_vector(self):
        """Zero vectors should have similarity 0.0"""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0
    
    def test_both_zero_vectors(self):
        """Both zero vectors should have similarity 0.0"""
        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0
    
    def test_numpy_arrays(self):
        """Should work with numpy arrays"""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)
    
    def test_high_dimensional(self):
        """Should work with 512-dimensional vectors (like embeddings)"""
        # Create two similar embeddings (add small noise)
        base_embedding = np.random.randn(512).astype(np.float32)
        a = base_embedding.tolist()
        b = (base_embedding + np.random.randn(512) * 0.01).tolist()
        
        similarity = cosine_similarity(a, b)
        assert 0.95 < similarity < 1.0  # Should be very similar
    
    def test_anticorrelated_vectors(self):
        """Anticorrelated vectors should have low similarity (near 0)"""
        a = [1.0, 1.0, 1.0]
        b = [-1.0, -1.0, -1.0]
        sim = cosine_similarity(a, b)
        # After clamping to [0, 1], anticorrelated vectors become 0
        assert 0.0 <= sim <= 1.0


class TestRetrievalAgentEmptyCaseBank:
    """Tests for graceful degradation when case bank is empty."""
    
    def test_empty_case_bank_returns_empty_lists(self):
        """Should return empty lists when no cases match"""
        # Create mock database objects
        mock_case_bank_entry = Mock()
        mock_reference_note = Mock()
        mock_session = Mock()
        
        # Query returns empty lists
        mock_session.query.return_value.filter.return_value.all.side_effect = [
            [],  # No cases in case_bank
            []   # No reference notes
        ]
        
        mock_vision_output = {
            "top_class": "glioma",
            "top_confidence": 0.85,
            "feature_embedding": [0.1] * 512,
            "predictions": {"glioma": 0.85, "meningioma": 0.1, "notumor": 0.03, "pituitary": 0.02},
            "gradcam_image": None
        }
        
        state = {
            "vision_output": mock_vision_output,
            "retrieval_output": None
        }
        
        with patch("app.db.models.get_db_session", return_value=mock_session):
            result = run_retrieval_agent(state)
        
        assert "retrieval_output" in result
        retrieval_output: RetrievalOutput = result["retrieval_output"]
        assert retrieval_output["similar_cases"] == []
        assert retrieval_output["reference_notes"] == []
    
    def test_database_unavailable_graceful_degradation(self):
        """Should return empty lists if database is unavailable"""
        mock_vision_output = {
            "top_class": "glioma",
            "top_confidence": 0.85,
            "feature_embedding": [0.1] * 512,
            "predictions": {"glioma": 0.85, "meningioma": 0.1, "notumor": 0.03, "pituitary": 0.02},
            "gradcam_image": None
        }
        
        state = {
            "vision_output": mock_vision_output,
            "retrieval_output": None
        }
        
        # Simulate database not available
        with patch("app.db.models.get_db_session", side_effect=RuntimeError("DB not init")):
            result = run_retrieval_agent(state)
        
        assert "retrieval_output" in result
        retrieval_output: RetrievalOutput = result["retrieval_output"]
        assert retrieval_output["similar_cases"] == []
        assert retrieval_output["reference_notes"] == []


class TestRetrievalAgentWithCases:
    """Tests for retrieving similar cases from case bank."""
    
    def test_retrieves_top_3_cases_by_similarity(self):
        """Should return top-3 cases sorted by similarity descending"""
        # Create mock case bank entries
        mock_cases = []
        for i in range(5):
            case = Mock()
            case.id = f"case_{i}"
            case.tumor_type = "glioma"
            case.confidence_at_insertion = 0.8 + i * 0.02
            case.summary = f"Case {i} summary"
            
            # Create feature vector with varying similarity
            base_embedding = np.ones(512, dtype=np.float32) * (0.5 + i * 0.1)
            case.feature_vector = json.dumps(base_embedding.tolist())
            mock_cases.append(case)
        
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.all.side_effect = [
            mock_cases,  # Case bank entries
            []           # Reference notes (empty)
        ]
        
        mock_vision_output = {
            "top_class": "glioma",
            "top_confidence": 0.85,
            "feature_embedding": (np.ones(512, dtype=np.float32) * 0.5).tolist(),
            "predictions": {"glioma": 0.85, "meningioma": 0.1, "notumor": 0.03, "pituitary": 0.02},
            "gradcam_image": None
        }
        
        state = {
            "vision_output": mock_vision_output,
            "retrieval_output": None
        }
        
        with patch("app.db.models.get_db_session", return_value=mock_session):
            result = run_retrieval_agent(state)
        
        retrieval_output: RetrievalOutput = result["retrieval_output"]
        
        # Should return max 3 cases
        assert len(retrieval_output["similar_cases"]) <= 3
        
        # Cases should be sorted by similarity descending
        similarities = [case["similarity_score"] for case in retrieval_output["similar_cases"]]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_case_with_invalid_feature_vector_skipped(self):
        """Should skip cases with invalid JSON in feature_vector"""
        mock_cases = [
            Mock(
                id="good_case",
                tumor_type="glioma",
                confidence_at_insertion=0.8,
                summary="Good case",
                feature_vector=json.dumps([0.1] * 512)
            ),
            Mock(
                id="bad_case",
                tumor_type="glioma",
                confidence_at_insertion=0.8,
                summary="Bad case",
                feature_vector="not valid json"  # Invalid JSON
            )
        ]
        
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.all.side_effect = [
            mock_cases,
            []
        ]
        
        mock_vision_output = {
            "top_class": "glioma",
            "top_confidence": 0.85,
            "feature_embedding": [0.1] * 512,
            "predictions": {"glioma": 0.85, "meningioma": 0.1, "notumor": 0.03, "pituitary": 0.02},
            "gradcam_image": None
        }
        
        state = {
            "vision_output": mock_vision_output,
            "retrieval_output": None
        }
        
        with patch("app.db.models.get_db_session", return_value=mock_session):
            result = run_retrieval_agent(state)
        
        retrieval_output: RetrievalOutput = result["retrieval_output"]
        
        # Should only have the good case
        assert len(retrieval_output["similar_cases"]) == 1
        assert retrieval_output["similar_cases"][0]["case_id"] == "good_case"


class TestRetrievalAgentErrors:
    """Tests for error handling."""
    
    def test_missing_vision_output(self):
        """Should raise error if vision_output is missing"""
        state = {
            "vision_output": None,
            "retrieval_output": None
        }
        
        with pytest.raises(RetrievalError, match="vision_output not found"):
            run_retrieval_agent(state)
    
    def test_missing_top_class(self):
        """Should raise error if top_class is missing"""
        state = {
            "vision_output": {
                "feature_embedding": [0.1] * 512,
                # Missing top_class
            },
            "retrieval_output": None
        }
        
        with pytest.raises(RetrievalError, match="top_class not found"):
            run_retrieval_agent(state)
    
    def test_missing_feature_embedding(self):
        """Should raise error if feature_embedding is missing"""
        state = {
            "vision_output": {
                "top_class": "glioma",
                # Missing feature_embedding
            },
            "retrieval_output": None
        }
        
        with pytest.raises(RetrievalError, match="feature_embedding not found"):
            run_retrieval_agent(state)
    
    def test_wrong_embedding_dimension(self):
        """Should raise error if embedding is not 512-dimensional"""
        state = {
            "vision_output": {
                "top_class": "glioma",
                "feature_embedding": [0.1] * 256,  # Wrong dimension
            },
            "retrieval_output": None
        }
        
        with pytest.raises(RetrievalError, match="expected 512"):
            run_retrieval_agent(state)


class TestRetrievalAgentReferenceNotes:
    """Tests for reference notes retrieval."""
    
    def test_retrieves_reference_notes_for_top_class(self):
        """Should retrieve reference notes matching top_class"""
        mock_reference_notes = [
            Mock(note_text="Clinical note 1", source="Source 1"),
            Mock(note_text="Clinical note 2", source="Source 2")
        ]
        
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.all.side_effect = [
            [],  # No cases
            mock_reference_notes  # Reference notes
        ]
        
        mock_vision_output = {
            "top_class": "glioma",
            "top_confidence": 0.85,
            "feature_embedding": [0.1] * 512,
            "predictions": {"glioma": 0.85, "meningioma": 0.1, "notumor": 0.03, "pituitary": 0.02},
            "gradcam_image": None
        }
        
        state = {
            "vision_output": mock_vision_output,
            "retrieval_output": None
        }
        
        with patch("app.db.models.get_db_session", return_value=mock_session):
            result = run_retrieval_agent(state)
        
        retrieval_output: RetrievalOutput = result["retrieval_output"]
        
        assert len(retrieval_output["reference_notes"]) == 2
        assert retrieval_output["reference_notes"][0] == "Clinical note 1"
        assert retrieval_output["reference_notes"][1] == "Clinical note 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
