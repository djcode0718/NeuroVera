"""
Unit tests for Drafting Agent

Tests the following:
- LLM fallback chain behavior with mocked providers
- First-pass drafting (synthesis of Vision + Retrieval)
- Revision-pass drafting (addressing Critic issues)
- Required sections present in report
- Hedging language when confidence < 0.6
- Model tracking (model_used field)
"""

import pytest
from unittest.mock import patch, MagicMock
from app.agents.drafting import (
    run_drafting_agent,
    call_llm_with_fallback,
    build_first_pass_user_prompt,
    build_revision_pass_user_prompt,
    DraftingError,
    DraftingOutput
)


class TestCallLLMWithFallback:
    """Test the LLM fallback chain implementation."""
    
    @patch("langchain_ollama.ChatOllama")
    def test_fallback_to_ollama_only(self, mock_ollama):
        """Test that when only Ollama is available, it's used."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.content = "Test report with all required sections. # Classification\n# Confidence\n# Visual Evidence\n# Similar Cases\n# Clinical Notes\n# Caveats"
        
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.invoke.return_value = mock_response
        mock_ollama.return_value = mock_ollama_instance
        
        with patch.dict("os.environ", {
            "OLLAMA_MODEL": "mistral",
            "GROQ_API_KEY": "",
            "GEMINI_API_KEY": ""
        }):
            response_text, model_used = call_llm_with_fallback(
                system_prompt="Test system",
                user_prompt="Test user"
            )
        
        assert model_used == "ollama/mistral"
        assert "Test report" in response_text
    
    @patch("langchain_groq.ChatGroq")
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    @patch("langchain_ollama.ChatOllama")
    def test_groq_success(self, mock_ollama, mock_gemini, mock_groq):
        """Test that Groq is tried first and used if successful."""
        mock_response = MagicMock()
        mock_response.content = "Groq response. # Classification\n# Confidence\n# Visual Evidence\n# Similar Cases\n# Clinical Notes\n# Caveats"
        
        mock_groq_instance = MagicMock()
        mock_groq_instance.invoke.return_value = mock_response
        mock_groq.return_value = mock_groq_instance
        
        with patch.dict("os.environ", {
            "GROQ_API_KEY": "test-key",
            "GROQ_FAST_MODEL": "llama-3.1-8b-instant",
            "OLLAMA_MODEL": "mistral"
        }):
            response_text, model_used = call_llm_with_fallback(
                system_prompt="Test system",
                user_prompt="Test user"
            )
        
        assert model_used == "groq/llama-3.1-8b-instant"
        assert "Groq response" in response_text
        # Verify Groq was called (not Gemini or Ollama)
        mock_groq.assert_called_once()


class TestBuildFirstPassPrompt:
    """Test first-pass user prompt generation."""
    
    def test_first_pass_prompt_includes_all_data(self):
        """Verify first-pass prompt includes Vision and Retrieval data."""
        vision_output = {
            "top_class": "glioma",
            "top_confidence": 0.85,
            "predictions": {
                "glioma": 0.85,
                "meningioma": 0.10,
                "notumor": 0.03,
                "pituitary": 0.02
            }
        }
        
        retrieval_output = {
            "similar_cases": [
                {
                    "case_id": "case1",
                    "tumor_type": "glioma",
                    "confidence": 0.80,
                    "summary": "High-grade glioma",
                    "similarity_score": 0.95
                }
            ],
            "reference_notes": [
                "Gliomas are the most common primary brain tumors.",
                "High-grade gliomas typically have poor prognosis."
            ]
        }
        
        prompt = build_first_pass_user_prompt(vision_output, retrieval_output)
        
        # Check that key information is in the prompt
        assert "glioma" in prompt.lower()
        assert "85" in prompt  # confidence percentage
        assert "High-grade glioma" in prompt  # similar case summary
        assert "most common primary brain tumors" in prompt  # reference note


class TestBuildRevisionPassPrompt:
    """Test revision-pass user prompt generation."""
    
    def test_revision_pass_prompt_addresses_issues(self):
        """Verify revision-pass prompt includes previous draft and Critic issues."""
        vision_output = {
            "top_class": "glioma",
            "top_confidence": 0.85
        }
        
        retrieval_output = {
            "similar_cases": [],
            "reference_notes": []
        }
        
        previous_draft = "Previous draft text with issues."
        critic_issues = [
            "Overstated certainty in Classification section",
            "Missing evidence from Grad-CAM interpretation"
        ]
        
        prompt = build_revision_pass_user_prompt(
            vision_output,
            retrieval_output,
            previous_draft,
            critic_issues
        )
        
        # Check that key information is in the prompt
        assert "Previous draft text" in prompt
        assert "Overstated certainty" in prompt
        assert "Grad-CAM interpretation" in prompt
        assert "revise" in prompt.lower()


class TestRunDraftingAgent:
    """Test the main Drafting Agent function."""
    
    @patch("app.agents.drafting.call_llm_with_fallback")
    def test_first_pass_drafting(self, mock_llm_call):
        """Test first-pass drafting (synthesis of Vision + Retrieval)."""
        # Mock LLM response with all required sections
        mock_report = """# Classification
Predicted tumor type: Glioma with 85% confidence.

# Confidence
High confidence in this prediction based on model output.

# Visual Evidence
Grad-CAM highlights show activation in tumor-characteristic regions.

# Similar Cases
Retrieved 1 similar historical case with 95% similarity score.

# Clinical Notes
Gliomas are the most common primary brain tumors.

# Caveats
This is an automated analysis, not a clinical diagnosis."""
        
        mock_llm_call.return_value = (mock_report, "groq/llama-3.1-8b-instant")
        
        state = {
            "vision_output": {
                "top_class": "glioma",
                "top_confidence": 0.85,
                "predictions": {
                    "glioma": 0.85,
                    "meningioma": 0.10,
                    "notumor": 0.03,
                    "pituitary": 0.02
                },
                "gradcam_image": None,
                "feature_embedding": [0.1] * 512
            },
            "retrieval_output": {
                "similar_cases": [],
                "reference_notes": []
            },
            "critic_retry_count": 0,
            "critic_issues_history": [],
            "drafting_output": None
        }
        
        result = run_drafting_agent(state)
        
        # Verify output structure
        assert "drafting_output" in result
        drafting_output = result["drafting_output"]
        
        assert "draft_report" in drafting_output
        assert "model_used" in drafting_output
        assert "revision_number" in drafting_output
        
        assert drafting_output["revision_number"] == 0
        assert drafting_output["model_used"] == "groq/llama-3.1-8b-instant"
        assert "# Classification" in drafting_output["draft_report"]
        assert "# Confidence" in drafting_output["draft_report"]
        assert "# Visual Evidence" in drafting_output["draft_report"]
    
    @patch("app.agents.drafting.call_llm_with_fallback")
    def test_revision_pass_drafting(self, mock_llm_call):
        """Test revision-pass drafting (addressing Critic issues)."""
        mock_revised_report = """# Classification
Predicted tumor type: Glioma (may indicate high-grade variant).

# Confidence
Moderate confidence with some uncertainty given ambiguous features.

# Visual Evidence
Grad-CAM shows mixed activations in tumor regions and non-tumor areas.

# Similar Cases
No clear similar cases in the case bank.

# Clinical Notes
Gliomas require careful clinical correlation for accurate grading.

# Caveats
This is an automated analysis requiring clinical validation."""
        
        mock_llm_call.return_value = (mock_revised_report, "gemini/gemini-2.5-flash")
        
        state = {
            "vision_output": {
                "top_class": "glioma",
                "top_confidence": 0.55,  # Lower confidence triggers hedging check
                "predictions": {
                    "glioma": 0.55,
                    "meningioma": 0.35,
                    "notumor": 0.07,
                    "pituitary": 0.03
                },
                "gradcam_image": None,
                "feature_embedding": [0.1] * 512
            },
            "retrieval_output": {
                "similar_cases": [],
                "reference_notes": []
            },
            "critic_retry_count": 1,
            "critic_issues_history": [
                ["Overstated certainty in confidence section"]
            ],
            "drafting_output": {
                "draft_report": "Original draft",
                "model_used": "groq/llama-3.1-8b-instant",
                "revision_number": 0
            }
        }
        
        result = run_drafting_agent(state)
        
        drafting_output = result["drafting_output"]
        
        assert drafting_output["revision_number"] == 1
        assert drafting_output["model_used"] == "gemini/gemini-2.5-flash"
        # Check that hedging language appears
        assert any(word in drafting_output["draft_report"].lower() 
                  for word in ["may", "appears", "suggest", "uncertainty"])
    
    def test_missing_vision_output_raises_error(self):
        """Test that missing vision_output raises DraftingError."""
        state = {
            "vision_output": None,
            "retrieval_output": {"similar_cases": [], "reference_notes": []},
            "critic_retry_count": 0,
            "critic_issues_history": [],
            "drafting_output": None
        }
        
        with pytest.raises(DraftingError):
            run_drafting_agent(state)
    
    def test_missing_retrieval_output_raises_error(self):
        """Test that missing retrieval_output raises DraftingError."""
        state = {
            "vision_output": {
                "top_class": "glioma",
                "top_confidence": 0.85,
                "predictions": {"glioma": 0.85},
                "gradcam_image": None,
                "feature_embedding": [0.1] * 512
            },
            "retrieval_output": None,
            "critic_retry_count": 0,
            "critic_issues_history": [],
            "drafting_output": None
        }
        
        with pytest.raises(DraftingError):
            run_drafting_agent(state)


class TestReportStructure:
    """Test that generated reports have required structure."""
    
    @patch("app.agents.drafting.call_llm_with_fallback")
    def test_report_contains_all_sections(self, mock_llm_call):
        """Verify that reports contain all 6 required sections."""
        mock_report = """# Classification
This is the classification section.

# Confidence
This is the confidence section.

# Visual Evidence
This is the visual evidence section.

# Similar Cases
This is the similar cases section.

# Clinical Notes
This is the clinical notes section.

# Caveats
This is the caveats section."""
        
        mock_llm_call.return_value = (mock_report, "ollama/mistral")
        
        state = {
            "vision_output": {
                "top_class": "glioma",
                "top_confidence": 0.85,
                "predictions": {"glioma": 0.85},
                "gradcam_image": None,
                "feature_embedding": [0.1] * 512
            },
            "retrieval_output": {
                "similar_cases": [],
                "reference_notes": []
            },
            "critic_retry_count": 0,
            "critic_issues_history": [],
            "drafting_output": None
        }
        
        result = run_drafting_agent(state)
        draft_report = result["drafting_output"]["draft_report"]
        
        required_sections = [
            "# Classification",
            "# Confidence",
            "# Visual Evidence",
            "# Similar Cases",
            "# Clinical Notes",
            "# Caveats"
        ]
        
        for section in required_sections:
            assert section in draft_report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
