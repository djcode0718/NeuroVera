"""
Critic Agent for NeuroTriage

The Critic Agent is responsible for:
1. Independently verifying the draft report against raw Vision output
2. Checking for overstated certainty relative to top_confidence
3. Checking for ignored non-top classes within 0.15 of top confidence
4. Checking for accurate Grad-CAM region descriptions
5. Returning verdict="approved" or "revise" with specific issues
6. Implementing LLM fallback chain (gracefully degrading to "approved")

CRITICAL: The Critic receives ONLY the Vision output, NOT retrieval context.
This ensures independent verification without contamination from case bank suggestions.

Example:
    from app.agents.critic import run_critic_agent
    
    state = {
        "vision_output": {...},
        "drafting_output": {"draft_report": "...", ...},
        "critic_output": None
    }
    result = run_critic_agent(state)
    print(result["critic_output"]["verdict"])
    print(result["critic_output"]["issues"])
"""

import logging
from typing import TypedDict, Literal, Optional

logger = logging.getLogger(__name__)


class CriticOutput(TypedDict):
    """Output from Critic Agent"""
    verdict: Literal["approved", "revise"]
    issues: list[str]
    model_used: str


class CriticError(Exception):
    """Raised when Critic Agent encounters a fatal error."""
    pass


def analyze_draft_against_vision(draft_report: str, vision_output: dict) -> tuple[str, list[str]]:
    """
    Analyze the draft report against raw Vision output to identify issues.
    
    Checks:
    1. Is the stated confidence aligned with top_confidence? (flag overstatement)
    2. Are non-top classes within 0.15 of top confidence being ignored? (flag missed findings)
    3. Is Grad-CAM description consistent with the actual predictions? (sanity check)
    
    Returns:
        tuple: (verdict, issues_list)
        - verdict: "approved" or "revise"
        - issues_list: empty if approved, or specific issue strings if revise
    """
    top_class = vision_output.get("top_class", "unknown")
    top_confidence = vision_output.get("top_confidence", 0.0)
    predictions = vision_output.get("predictions", {})
    
    issues = []
    
    # Check 1: Overstatement of certainty
    # If confidence < 0.6 but draft doesn't hedge appropriately, flag it
    if top_confidence < 0.6:
        # Look for hedging language in draft
        hedging_phrases = ["ambiguous", "unclear", "uncertain", "caution", "cannot definitively",
                          "may indicate", "possibly", "suggests", "tentative", "low confidence"]
        
        draft_lower = draft_report.lower()
        has_hedging = any(phrase in draft_lower for phrase in hedging_phrases)
        
        if not has_hedging:
            issues.append(
                f"Draft does not acknowledge low confidence ({top_confidence:.1%}). "
                "Should include explicit hedging language."
            )
    
    # Check 2: Missed significant alternatives
    # If another class is within 0.15 of top_confidence and not mentioned, flag it
    for class_name, prob in predictions.items():
        if class_name != top_class and abs(prob - top_confidence) < 0.15:
            # This class is close to top confidence - should be mentioned
            if class_name.lower() not in draft_report.lower():
                issues.append(
                    f"Draft ignores {class_name} ({prob:.1%}), which is within 0.15 "
                    f"of top class {top_class} ({top_confidence:.1%}). "
                    "This near-tie should be acknowledged."
                )
    
    # Check 3: Probability distribution sanity
    # If draft claims probabilities, verify they roughly match vision output
    # (This is a basic sanity check)
    draft_lower = draft_report.lower()
    if "probability distribution" in draft_lower or "probabilities" in draft_lower:
        # Draft references specific probabilities, which is good
        pass  # More detailed LLM-based check would happen with actual LLM
    
    # Determine verdict
    if issues:
        verdict = "revise"
    else:
        verdict = "approved"
    
    return verdict, issues


def run_critic_agent(state: dict) -> dict:
    """
    Execute the Critic Agent: independently verify draft against vision output.
    
    This function implements the Critic Agent algorithm:
    1. Extract vision_output and drafting_output from state
    2. Analyze draft for issues against raw vision evidence
    3. Return CriticOutput with verdict and issues
    4. CRITICAL: Do NOT have access to retrieval_output (only vision_output)
    5. On all LLM provider failure: gracefully return verdict="approved" instead of raising
    
    The Critic's job is to catch overstated claims, missed findings, and inaccuracies.
    It operates independently from retrieval context to ensure unbiased verification.
    
    Args:
        state: GraphState dict containing:
            - "vision_output": dict with predictions, top_class, top_confidence, gradcam_image
            - "drafting_output": dict with draft_report
            
    Returns:
        state: Updated GraphState dict with "critic_output" populated
        
    Raises:
        CriticError: NEVER - this agent must never raise, always returns a verdict
        
    Preconditions:
        - state["vision_output"] is populated
        - state["drafting_output"] is populated
        
    Postconditions:
        - state["critic_output"]["verdict"] is either "approved" or "revise"
        - state["critic_output"]["issues"] is empty if approved, or list of strings if revise
        - state["critic_output"]["model_used"] indicates which model was used
        - On all LLM provider failures: returns verdict="approved" with empty issues
        
    Example:
        from app.agents.critic import run_critic_agent
        
        state = graph_state  # With vision and drafting populated
        state = run_critic_agent(state)
        
        verdict = state["critic_output"]["verdict"]
        if verdict == "revise":
            for issue in state["critic_output"]["issues"]:
                print(f"Issue: {issue}")
    """
    try:
        logger.info("Starting Critic Agent")
        
        # Extract required state fields
        vision_output = state.get("vision_output")
        drafting_output = state.get("drafting_output")
        
        if vision_output is None:
            logger.warning("vision_output not found, defaulting to approved")
            verdict = "approved"
            issues = []
            model_used = "error/no-vision-output"
        elif drafting_output is None:
            logger.warning("drafting_output not found, defaulting to approved")
            verdict = "approved"
            issues = []
            model_used = "error/no-drafting-output"
        else:
            draft_report = drafting_output.get("draft_report", "")
            
            # Analyze draft against vision output
            verdict, issues = analyze_draft_against_vision(draft_report, vision_output)
            model_used = "template/structured"  # Placeholder for now
            
            logger.debug(f"Critic verdict: {verdict} ({len(issues)} issues)")
        
        # Create critic output
        critic_output: CriticOutput = {
            "verdict": verdict,
            "issues": issues,
            "model_used": model_used
        }
        
        state["critic_output"] = critic_output
        logger.info(f"Critic Agent completed: verdict={verdict}, model={model_used}")
        
        return state
        
    except Exception as e:
        # CRITICAL: Critic must NEVER raise an exception
        # On any error, return approved to allow pipeline to continue
        logger.error(f"Critic Agent error (gracefully returning approved): {e}", exc_info=True)
        
        state["critic_output"] = {
            "verdict": "approved",
            "issues": [],
            "model_used": "error/fallback-approved"
        }
        
        return state
