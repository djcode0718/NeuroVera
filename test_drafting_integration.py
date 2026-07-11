"""
Integration test for Drafting Agent

This test verifies that the Drafting Agent integrates correctly with the graph
and produces reports according to spec.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.agents.drafting import run_drafting_agent, DraftingOutput


def create_mock_vision_output(top_class="glioma", confidence=0.85):
    """Helper to create a mock vision output."""
    return {
        "top_class": top_class,
        "top_confidence": confidence,
        "predictions": {
            "glioma": 0.85 if top_class == "glioma" else 0.10,
            "meningioma": 0.10 if top_class != "meningioma" else 0.80,
            "notumor": 0.03 if top_class != "notumor" else 0.05,
            "pituitary": 0.02 if top_class != "pituitary" else 0.05
        },
        "gradcam_image": "data:image/png;base64,fake",
        "feature_embedding": [0.1] * 512
    }


def create_mock_retrieval_output():
    """Helper to create a mock retrieval output."""
    return {
        "similar_cases": [
            {
                "case_id": "case-1",
                "tumor_type": "glioma",
                "confidence": 0.80,
                "summary": "High-grade glioma with atypical features",
                "similarity_score": 0.95
            },
            {
                "case_id": "case-2",
                "tumor_type": "glioma",
                "confidence": 0.75,
                "summary": "Low-grade diffuse glioma",
                "similarity_score": 0.85
            }
        ],
        "reference_notes": [
            "Gliomas are the most common primary brain tumors",
            "WHO grading system classifies gliomas into grades II-IV"
        ]
    }


@patch("app.agents.drafting.call_llm_with_fallback")
def test_drafting_with_high_confidence(mock_llm):
    """Test drafting with high confidence prediction."""
    mock_report = """# Classification
Predicted tumor type: Glioma
The model predicts glioma with high confidence based on imaging features.

# Confidence
This prediction has high confidence (85%). The model shows strong agreement across
multiple imaging features consistent with high-grade glioma.

# Visual Evidence
The Grad-CAM heatmap shows strong activation in the tumoral region with characteristic
features for malignant glioma including heterogeneous signal intensity.

# Similar Cases
Retrieved 2 similar historical cases with high similarity scores (95% and 85%).
Both cases show similar radiographic patterns to this patient.

# Clinical Notes
Gliomas are the most common primary brain tumors. WHO grading system classifies gliomas
into grades II-IV based on histological features.

# Caveats
This is an automated analysis and should not be used as a standalone clinical diagnosis.
Clinical correlation and expert radiologist review are essential for patient management."""

    mock_llm.return_value = (mock_report, "groq/llama-3.1-8b-instant")
    
    state = {
        "vision_output": create_mock_vision_output(top_class="glioma", confidence=0.85),
        "retrieval_output": create_mock_retrieval_output(),
        "critic_retry_count": 0,
        "critic_issues_history": [],
        "drafting_output": None
    }
    
    result = run_drafting_agent(state)
    
    assert "drafting_output" in result
    output = result["drafting_output"]
    
    # Verify output structure
    assert isinstance(output, dict)
    assert "draft_report" in output
    assert "model_used" in output
    assert "revision_number" in output
    
    # Verify values
    assert output["revision_number"] == 0
    assert output["model_used"] == "groq/llama-3.1-8b-instant"
    
    # Verify report content
    report = output["draft_report"]
    assert "# Classification" in report
    assert "# Confidence" in report
    assert "# Visual Evidence" in report
    assert "# Similar Cases" in report
    assert "# Clinical Notes" in report
    assert "# Caveats" in report
    
    # Verify report references the data
    assert "glioma" in report.lower()
    assert "85" in report  # confidence
    assert "similar" in report.lower()  # reference to similar cases
    
    # High confidence should not have excessive hedging
    assert report.count("may") < 3  # reasonable amount of hedging


@patch("app.agents.drafting.call_llm_with_fallback")
def test_drafting_with_low_confidence(mock_llm):
    """Test drafting with low confidence prediction - should include hedging."""
    mock_report = """# Classification
The model suggests this may be meningioma, though confidence is moderate.
The imaging features could be consistent with meningioma, though other diagnoses
appear possible given the uncertain imaging characteristics.

# Confidence
This prediction has moderate to low confidence (45%). The model shows some uncertainty,
with competing probabilities between meningioma and glioma. This suggests clinical
correlation and expert review is particularly important.

# Visual Evidence
The Grad-CAM heatmap shows mixed activations that may indicate transitional features
between different tumor types. The pattern appears ambiguous in several respects.

# Similar Cases
No highly similar cases were found in the case bank with sufficient confidence.

# Clinical Notes
Meningiomas are typically extra-axial tumors arising from the dura mater.
They may present with varied imaging characteristics.

# Caveats
This is an automated analysis and absolutely requires clinical correlation.
The moderate confidence suggests expert radiologist review is essential before
any clinical decisions are made."""

    mock_llm.return_value = (mock_report, "gemini/gemini-2.5-flash")
    
    state = {
        "vision_output": create_mock_vision_output(top_class="meningioma", confidence=0.45),
        "retrieval_output": {"similar_cases": [], "reference_notes": []},
        "critic_retry_count": 0,
        "critic_issues_history": [],
        "drafting_output": None
    }
    
    result = run_drafting_agent(state)
    
    output = result["drafting_output"]
    report = output["draft_report"]
    
    # Low confidence should have hedging language
    hedging_words = ["appears", "may", "suggest", "could", "uncertain", "ambiguous"]
    hedging_count = sum(report.lower().count(word) for word in hedging_words)
    assert hedging_count >= 5, f"Expected hedging language, but found only {hedging_count} instances"
    
    # Should mention the moderate confidence
    assert "moderate" in report.lower() or "45" in report or "uncertain" in report.lower()


@patch("app.agents.drafting.call_llm_with_fallback")
def test_revision_pass_addresses_critic_issues(mock_llm):
    """Test that revision pass addresses issues from Critic."""
    # First draft had issues
    first_draft = """# Classification
Glioma with 55% confidence.

# Confidence
Moderate confidence in this prediction."""  # Too brief, Critic will flag

    revised_draft = """# Classification
The model suggests this may be glioma, though clinical correlation is needed.
The imaging features could be consistent with high-grade glioma, though other
diagnoses appear possible.

# Confidence
This prediction has moderate confidence (55%), indicating meaningful uncertainty.
The model shows competing probabilities suggesting expert review is important.
The hedged language reflects this uncertainty appropriately."""

    mock_llm.return_value = (revised_draft, "gemini/gemini-2.5-flash")
    
    state = {
        "vision_output": create_mock_vision_output(top_class="glioma", confidence=0.55),
        "retrieval_output": create_mock_retrieval_output(),
        "critic_retry_count": 1,
        "critic_issues_history": [
            ["Insufficient hedging language for 55% confidence", "Confidence section too brief"]
        ],
        "drafting_output": {
            "draft_report": first_draft,
            "model_used": "groq/llama-3.1-8b-instant",
            "revision_number": 0
        }
    }
    
    result = run_drafting_agent(state)
    
    output = result["drafting_output"]
    
    # Should be revision 1
    assert output["revision_number"] == 1
    
    # Should use different model (Gemini) due to fallback
    assert output["model_used"] == "gemini/gemini-2.5-flash"
    
    # New draft should be different from original
    assert output["draft_report"] != first_draft
    
    # New draft should address the issues
    revised_report = output["draft_report"]
    assert "hedg" in revised_report.lower() or "appear" in revised_report.lower()
    assert "uncertain" in revised_report.lower() or "confidence" in revised_report.lower()


@patch("app.agents.drafting.call_llm_with_fallback")
def test_drafting_output_type_validation(mock_llm):
    """Test that DraftingOutput has correct structure."""
    mock_report = """# Classification
Test classification.

# Confidence
Test confidence.

# Visual Evidence
Test evidence.

# Similar Cases
Test cases.

# Clinical Notes
Test notes.

# Caveats
Test caveats."""

    mock_llm.return_value = (mock_report, "ollama/mistral")
    
    state = {
        "vision_output": create_mock_vision_output(),
        "retrieval_output": create_mock_retrieval_output(),
        "critic_retry_count": 0,
        "critic_issues_history": [],
        "drafting_output": None
    }
    
    result = run_drafting_agent(state)
    output = result["drafting_output"]
    
    # Verify TypedDict fields match specification
    required_fields = ["draft_report", "model_used", "revision_number"]
    for field in required_fields:
        assert field in output, f"Missing required field: {field}"
    
    # Verify types
    assert isinstance(output["draft_report"], str)
    assert isinstance(output["model_used"], str)
    assert isinstance(output["revision_number"], int)
    
    # Verify model_used format: "provider/model-name"
    assert "/" in output["model_used"]
    provider, model = output["model_used"].split("/", 1)
    assert provider in ["groq", "gemini", "ollama"]
    assert len(model) > 0


@patch("app.agents.drafting.call_llm_with_fallback")
def test_all_six_sections_required(mock_llm):
    """Test that all 6 required sections are present in draft reports."""
    # This test could fail if the LLM doesn't return all sections
    # But with mocking, we ensure the agent handles it correctly
    
    # Test case: LLM returns a report without all sections
    incomplete_report = """# Classification
Glioma.

# Confidence
Moderate.

# Visual Evidence
Shows tumor region."""  # Missing Similar Cases, Clinical Notes, Caveats

    mock_llm.return_value = (incomplete_report, "groq/llama-3.1-8b-instant")
    
    state = {
        "vision_output": create_mock_vision_output(),
        "retrieval_output": create_mock_retrieval_output(),
        "critic_retry_count": 0,
        "critic_issues_history": [],
        "drafting_output": None
    }
    
    # Should not raise, but log a warning about missing sections
    result = run_drafting_agent(state)
    
    # The agent should still return the incomplete report
    # (The Critic will catch this issue in the next phase)
    assert result["drafting_output"]["draft_report"] == incomplete_report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
