"""
Test suite for Orchestrator Agent

Tests cover:
1. Routing rules (pure function)
2. Reasoning trace generation
3. Vision-only justification fallback
4. End-to-end orchestrator execution
"""

import pytest
from app.agents.orchestrator import (
    determine_routing,
    build_reasoning_trace,
    generate_vision_only_justification,
    run_orchestrator_agent,
    OrchestratorOutput
)


class TestDetermineRouting:
    """Test the routing rules pure function"""
    
    def test_urgent_glioma_high_confidence(self):
        """Rule 1: glioma > 0.75 → urgent"""
        result = determine_routing("glioma", 0.85, False)
        assert result == "urgent"
    
    def test_urgent_pituitary_high_confidence(self):
        """Rule 1: pituitary > 0.75 → urgent"""
        result = determine_routing("pituitary", 0.80, False)
        assert result == "urgent"
    
    def test_urgent_threshold_exact(self):
        """Rule 1: confidence must be GREATER than 0.75 (not equal)"""
        result = determine_routing("glioma", 0.75, False)
        # 0.75 is NOT > 0.75, so should not be urgent
        assert result == "needs-review"
    
    def test_urgent_threshold_just_above(self):
        """Rule 1: confidence just above 0.75 is urgent"""
        result = determine_routing("glioma", 0.751, False)
        assert result == "urgent"
    
    def test_auto_clear_notumor_high_confidence(self):
        """Rule 2: notumor > 0.80 → auto-clear"""
        result = determine_routing("notumor", 0.85, False)
        assert result == "auto-clear"
    
    def test_auto_clear_threshold_exact(self):
        """Rule 2: confidence must be GREATER than 0.80 (not equal)"""
        result = determine_routing("notumor", 0.80, False)
        # 0.80 is NOT > 0.80, so should not be auto-clear
        assert result == "needs-review"
    
    def test_auto_clear_threshold_just_above(self):
        """Rule 2: confidence just above 0.80 is auto-clear"""
        result = determine_routing("notumor", 0.801, False)
        assert result == "auto-clear"
    
    def test_needs_review_medium_confidence_low(self):
        """Rule 3: 0.40 <= confidence <= 0.75 → needs-review"""
        result = determine_routing("glioma", 0.40, False)
        assert result == "needs-review"
    
    def test_needs_review_medium_confidence_mid(self):
        """Rule 3: medium confidence → needs-review"""
        result = determine_routing("meningioma", 0.50, False)
        assert result == "needs-review"
    
    def test_needs_review_medium_confidence_high(self):
        """Rule 3: 0.40 <= confidence <= 0.75 → needs-review"""
        result = determine_routing("pituitary", 0.75, False)
        assert result == "needs-review"
    
    def test_needs_review_with_unresolved_issues(self):
        """Rule 3: unresolved issues → needs-review even if high confidence"""
        result = determine_routing("notumor", 0.85, has_unresolved_issues=True)
        # Even though notumor > 0.80 would normally be auto-clear,
        # unresolved issues force needs-review
        assert result == "needs-review"
    
    def test_default_fallback_low_confidence(self):
        """Rule 4: very low confidence → needs-review (default)"""
        result = determine_routing("glioma", 0.25, False)
        # Below 0.40, not auto-clear, not urgent → needs-review
        assert result == "needs-review"
    
    def test_default_fallback_meningioma(self):
        """Rule 4: meningioma (not urgent class) → needs-review"""
        result = determine_routing("meningioma", 0.85, False)
        # meningioma doesn't trigger urgent even at high confidence
        assert result == "needs-review"
    
    def test_meningioma_medium_confidence(self):
        """meningioma at medium confidence → needs-review"""
        result = determine_routing("meningioma", 0.50, False)
        assert result == "needs-review"
    
    def test_confidence_range_boundaries(self):
        """Test confidence range boundaries precisely"""
        # Test 0.40 boundary (inclusive)
        assert determine_routing("glioma", 0.40, False) == "needs-review"
        
        # Test 0.75 boundary (inclusive)
        assert determine_routing("meningioma", 0.75, False) == "needs-review"
        
        # Test 0.80 boundary (exclusive for auto-clear)
        assert determine_routing("notumor", 0.80, False) == "needs-review"
        assert determine_routing("notumor", 0.801, False) == "auto-clear"
        
        # Test 0.75 boundary (exclusive for urgent)
        assert determine_routing("glioma", 0.75, False) == "needs-review"
        assert determine_routing("glioma", 0.751, False) == "urgent"


class TestReasoningTrace:
    """Test reasoning trace generation from agent outputs"""
    
    def test_vision_only(self):
        """Trace with only Vision Agent output"""
        state = {
            "vision_output": {
                "top_class": "glioma",
                "top_confidence": 0.85,
                "predictions": {
                    "glioma": 0.85,
                    "meningioma": 0.10,
                    "notumor": 0.03,
                    "pituitary": 0.02
                }
            }
        }
        
        trace = build_reasoning_trace(state)
        
        assert len(trace) >= 1
        assert trace[0]["agent"] == "Vision Agent"
        assert "glioma" in trace[0]["summary"]
        assert "85.0%" in trace[0]["summary"]
        assert "glioma: 85.0%" in trace[0]["key_evidence"]
    
    def test_vision_and_retrieval(self):
        """Trace with Vision and Retrieval agents"""
        state = {
            "vision_output": {
                "top_class": "glioma",
                "top_confidence": 0.85,
                "predictions": {
                    "glioma": 0.85,
                    "meningioma": 0.10,
                    "notumor": 0.03,
                    "pituitary": 0.02
                }
            },
            "retrieval_output": {
                "similar_cases": [
                    {
                        "case_id": "case_1",
                        "tumor_type": "glioma",
                        "confidence": 0.80,
                        "summary": "Similar glioma case",
                        "similarity_score": 0.92
                    }
                ],
                "reference_notes": ["Note 1", "Note 2"]
            }
        }
        
        trace = build_reasoning_trace(state)
        
        assert len(trace) >= 2
        assert trace[0]["agent"] == "Vision Agent"
        assert trace[1]["agent"] == "Retrieval Agent"
        assert "1 similar" in trace[1]["summary"]
        assert "0.92" in trace[1]["key_evidence"]
    
    def test_empty_retrieval(self):
        """Trace when no similar cases found"""
        state = {
            "vision_output": {
                "top_class": "notumor",
                "top_confidence": 0.90,
                "predictions": {
                    "glioma": 0.02,
                    "meningioma": 0.03,
                    "notumor": 0.90,
                    "pituitary": 0.05
                }
            },
            "retrieval_output": {
                "similar_cases": [],
                "reference_notes": []
            }
        }
        
        trace = build_reasoning_trace(state)
        
        assert len(trace) >= 2
        assert trace[1]["agent"] == "Retrieval Agent"
        assert "0 similar" in trace[1]["summary"]
        assert "No similar cases" in trace[1]["key_evidence"]
    
    def test_full_pipeline_trace(self):
        """Trace with all agents"""
        state = {
            "vision_output": {
                "top_class": "glioma",
                "top_confidence": 0.50,
                "predictions": {
                    "glioma": 0.50,
                    "meningioma": 0.30,
                    "notumor": 0.15,
                    "pituitary": 0.05
                }
            },
            "retrieval_output": {
                "similar_cases": [
                    {
                        "case_id": "case_1",
                        "tumor_type": "glioma",
                        "confidence": 0.75,
                        "summary": "Similar glioma",
                        "similarity_score": 0.88
                    }
                ],
                "reference_notes": ["Reference note"]
            },
            "drafting_output": {
                "draft_report": "Report text...",
                "model_used": "groq/llama-3.1-8b-instant",
                "revision_number": 0
            },
            "critic_output": {
                "verdict": "revise",
                "issues": ["Overstated certainty", "Missing qualifier"],
                "model_used": "gemini/gemini-2.5-flash"
            }
        }
        
        trace = build_reasoning_trace(state)
        
        # Should have: Vision, Retrieval, Drafting, Critic
        assert len(trace) >= 4
        assert trace[0]["agent"] == "Vision Agent"
        assert trace[1]["agent"] == "Retrieval Agent"
        assert trace[2]["agent"] == "Drafting Agent"
        assert trace[3]["agent"] == "Critic Agent"
        
        # Check Critic trace details
        assert "revise" in trace[3]["summary"]
        assert "2 issues" in trace[3]["key_evidence"]


class TestVisionOnlyJustification:
    """Test vision-only justification fallback"""
    
    def test_urgent_high_confidence(self):
        """Justification for urgent routing at high confidence"""
        justification = generate_vision_only_justification(
            top_class="glioma",
            top_confidence=0.85,
            routing="urgent",
            has_unresolved_issues=False
        )
        
        assert "URGENT" in justification
        assert "glioma" in justification
        assert "85.0%" in justification
        assert "vision-only" in justification or "Vision" in justification
        assert "Immediate specialist review" in justification
    
    def test_needs_review_medium_confidence(self):
        """Justification for needs-review at medium confidence"""
        justification = generate_vision_only_justification(
            top_class="meningioma",
            top_confidence=0.60,
            routing="needs-review",
            has_unresolved_issues=False
        )
        
        assert "NEEDS-REVIEW" in justification
        assert "meningioma" in justification
        assert "60.0%" in justification
        assert "Manual review" in justification
    
    def test_auto_clear_high_confidence(self):
        """Justification for auto-clear at high confidence"""
        justification = generate_vision_only_justification(
            top_class="notumor",
            top_confidence=0.90,
            routing="auto-clear",
            has_unresolved_issues=False
        )
        
        assert "AUTO-CLEAR" in justification
        assert "notumor" in justification
        assert "90.0%" in justification
        assert "routine documentation" in justification or "No immediate action" in justification
    
    def test_with_unresolved_issues(self):
        """Justification mentions unresolved critic issues"""
        justification = generate_vision_only_justification(
            top_class="glioma",
            top_confidence=0.85,
            routing="needs-review",
            has_unresolved_issues=True
        )
        
        assert "unresolved" in justification.lower()
        assert "critic" in justification.lower()
    
    def test_medical_disclaimer(self):
        """All justifications include medical disclaimer"""
        for routing in ["urgent", "needs-review", "auto-clear"]:
            justification = generate_vision_only_justification(
                top_class="glioma",
                top_confidence=0.75,
                routing=routing,
                has_unresolved_issues=False
            )
            
            assert "qualified medical professional" in justification
            assert "research" in justification or "automated" in justification


class TestEndToEndOrchestrator:
    """Test end-to-end orchestrator execution"""
    
    def test_orchestrator_high_confidence_glioma(self):
        """Full orchestrator run with high-confidence glioma"""
        state = {
            "image_bytes": b"fake_image_bytes",
            "vision_output": {
                "top_class": "glioma",
                "top_confidence": 0.85,
                "predictions": {
                    "glioma": 0.85,
                    "meningioma": 0.10,
                    "notumor": 0.03,
                    "pituitary": 0.02
                },
                "gradcam_image": "base64data",
                "feature_embedding": [0.1] * 512
            },
            "retrieval_output": {
                "similar_cases": [
                    {
                        "case_id": "case_1",
                        "tumor_type": "glioma",
                        "confidence": 0.80,
                        "summary": "Similar case",
                        "similarity_score": 0.92
                    }
                ],
                "reference_notes": ["Note"]
            },
            "drafting_output": {
                "draft_report": "Report...",
                "model_used": "groq/llama-3.1-8b-instant",
                "revision_number": 0
            },
            "critic_output": {
                "verdict": "approved",
                "issues": [],
                "model_used": "groq/llama-3.1-8b-instant"
            },
            "critic_retry_count": 0,
            "critic_issues_history": []
        }
        
        # Run orchestrator (will fail LLM, but that's ok - test the fallback)
        state = run_orchestrator_agent(state)
        
        # Verify output structure
        assert "orchestrator_output" in state
        output = state["orchestrator_output"]
        
        assert output["routing"] == "urgent"
        assert isinstance(output["justification"], str)
        assert len(output["justification"]) > 0
        assert isinstance(output["reasoning_trace"], list)
        assert len(output["reasoning_trace"]) >= 1
        assert output["model_used"] in ["vision-only", "no-llm", "groq/llama-3.1-8b-instant"]
    
    def test_orchestrator_with_unresolved_issues(self):
        """Orchestrator routes to needs-review when issues unresolved"""
        state = {
            "image_bytes": b"fake_image_bytes",
            "vision_output": {
                "top_class": "notumor",
                "top_confidence": 0.85,
                "predictions": {
                    "glioma": 0.05,
                    "meningioma": 0.05,
                    "notumor": 0.85,
                    "pituitary": 0.05
                },
                "gradcam_image": "base64data",
                "feature_embedding": [0.1] * 512
            },
            "retrieval_output": {
                "similar_cases": [],
                "reference_notes": []
            },
            "drafting_output": {
                "draft_report": "Report...",
                "model_used": "groq/llama-3.1-8b-instant",
                "revision_number": 1
            },
            "critic_output": {
                "verdict": "revise",
                "issues": ["Some issue"],
                "model_used": "groq/llama-3.1-8b-instant"
            },
            "critic_retry_count": 2,
            "critic_issues_history": [
                ["Issue 1"],
                ["Issue 2"]
            ]
        }
        
        state = run_orchestrator_agent(state)
        output = state["orchestrator_output"]
        
        # Despite high confidence, unresolved issues force needs-review
        assert output["routing"] == "needs-review"
        assert "unresolved" in output["justification"].lower() or "issue" in output["justification"].lower()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
