"""
Orchestrator Agent for NeuroTriage

The Orchestrator Agent makes final routing decisions and generates justifications
referencing all upstream agent evidence. It implements the same LLM fallback chain
as other agents (Groq → Gemini → Ollama) with graceful vision-only fallback.
"""

import logging
import json
import os
from typing import TypedDict, Literal

logger = logging.getLogger(__name__)


class OrchestratorOutput(TypedDict):
    """Output from Orchestrator Agent"""
    routing: Literal["auto-clear", "needs-review", "urgent"]
    justification: str
    reasoning_trace: list[dict]
    model_used: str


def determine_routing(top_class: str, top_confidence: float, has_unresolved_issues: bool) -> str:
    """
    Pure function implementing routing rules.
    
    Rules:
    1. "urgent" if (top_class in ["glioma", "pituitary"]) AND top_confidence > 0.75
    2. "auto-clear" if (top_class == "notumor") AND top_confidence > 0.80 AND no unresolved issues
    3. "needs-review" if (0.4 <= top_confidence <= 0.75) OR has_unresolved_issues
    4. Default: "needs-review"
    """
    # Check for unresolved issues first (they override everything except urgent)
    if has_unresolved_issues:
        # Urgent still applies with unresolved issues
        if top_class in ("glioma", "pituitary") and top_confidence > 0.75:
            return "urgent"
        # Otherwise needs-review due to unresolved issues
        return "needs-review"
    
    if top_class in ("glioma", "pituitary") and top_confidence > 0.75:
        return "urgent"
    
    if top_class == "notumor" and top_confidence > 0.80:
        return "auto-clear"
    
    if 0.4 <= top_confidence <= 0.75:
        return "needs-review"
    
    return "needs-review"


def build_reasoning_trace(state: dict) -> list[dict]:
    """Generate reasoning_trace entries from all agent outputs."""
    trace = []
    
    if state.get("vision_output"):
        vision_output = state["vision_output"]
        top_class = vision_output["top_class"]
        confidence = vision_output["top_confidence"]
        predictions = vision_output["predictions"]
        
        trace.append({
            "agent": "Vision Agent",
            "summary": f"Classified as {top_class} with {confidence:.1%} confidence",
            "key_evidence": f"Predictions: {', '.join(f'{cls}: {prob:.1%}' for cls, prob in predictions.items())}"
        })
    
    if state.get("retrieval_output"):
        retrieval_output = state["retrieval_output"]
        num_cases = len(retrieval_output["similar_cases"])
        
        if num_cases > 0:
            top_similarity = retrieval_output["similar_cases"][0]["similarity_score"]
            key_evidence = f"Top 3 similar cases found with max similarity: {top_similarity:.2f}"
        else:
            key_evidence = "No similar cases found in case bank"
        
        trace.append({
            "agent": "Retrieval Agent",
            "summary": f"Found {num_cases} similar historical cases",
            "key_evidence": key_evidence
        })
    
    if state.get("drafting_output"):
        drafting_output = state["drafting_output"]
        model_used = drafting_output["model_used"]
        revision = drafting_output.get("revision_number", 0)
        
        revision_text = "Initial draft" if revision == 0 else f"Revised draft (revision {revision})"
        
        trace.append({
            "agent": "Drafting Agent",
            "summary": f"Generated structured clinical report ({revision_text})",
            "key_evidence": f"Model: {model_used}"
        })
    
    if state.get("critic_output"):
        critic_output = state["critic_output"]
        verdict = critic_output["verdict"]
        num_issues = len(critic_output["issues"])
        
        if num_issues > 0:
            issues_text = "; ".join(critic_output["issues"][:3])
            if len(critic_output["issues"]) > 3:
                issues_text += f" (and {len(critic_output['issues']) - 3} more)"
            key_evidence = f"Issues: {issues_text}"
        else:
            key_evidence = "No issues found"
        
        trace.append({
            "agent": "Critic Agent",
            "summary": f"Verified draft (verdict: {verdict})",
            "key_evidence": f"{num_issues} issues raised; Model: {critic_output['model_used']}"
        })
    
    return trace


def generate_vision_only_justification(top_class: str, top_confidence: float, routing: str, has_unresolved_issues: bool = False) -> str:
    """Generate justification when LLM unavailable (vision-only fallback)."""
    if top_confidence >= 0.80:
        confidence_range = "high confidence (≥80%)"
    elif top_confidence >= 0.75:
        confidence_range = "very high confidence (≥75%)"
    elif top_confidence >= 0.60:
        confidence_range = "moderate-to-high confidence (60-75%)"
    elif top_confidence >= 0.40:
        confidence_range = "moderate confidence (40-60%)"
    else:
        confidence_range = "low confidence (<40%)"
    
    paragraphs = [
        f"Case routed to {routing.upper()}.",
        f"\nReasoning:\n"
        f"- Automated model classified scan as {top_class} with {top_confidence:.1%} confidence\n"
        f"- Confidence level [{top_confidence:.1%}] falls in '{confidence_range}' range\n"
        f"- This is a vision-only routing decision (LLM reasoning unavailable)"
    ]
    
    if has_unresolved_issues:
        paragraphs.append("- Critic Agent raised unresolved issues with the draft report")
    
    if routing == "urgent":
        recommendation = "Immediate specialist review recommended"
    elif routing == "needs-review":
        recommendation = "Manual review by qualified radiologist recommended"
    else:
        recommendation = "No immediate action required; routine documentation"
    
    paragraphs.append(f"- Recommendation: {recommendation}")
    
    paragraphs.append(
        "\nThis analysis is based on automated model predictions and should be reviewed by a "
        "qualified medical professional."
    )
    
    return "\n".join(paragraphs)


def call_llm_with_fallback(prompt: str, system_prompt: str) -> tuple[str, str]:
    """Call LLM with fallback chain: Groq → Gemini → Ollama."""
    logger.info("Starting LLM fallback chain")
    
    # Attempt 1: Groq
    try:
        logger.debug("Attempting Groq...")
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("No GROQ_API_KEY")
        
        client = Groq(api_key=api_key)
        model = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        
        response_text = response.choices[0].message.content
        model_used = f"groq/{model}"
        
        logger.info(f"Groq succeeded: {model_used}")
        return (response_text, model_used)
        
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "quota" in error_msg or "rate_limit" in error_msg:
            logger.warning(f"Groq quota/rate limit reached, falling back to Gemini")
        else:
            logger.debug(f"Groq failed (will try Gemini): {e}")
    
    # Attempt 2: Gemini
    try:
        logger.debug("Attempting Gemini...")
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No GEMINI_API_KEY")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt
        )
        
        response = model.generate_content(prompt)
        response_text = response.text
        model_used = "gemini/gemini-2.5-flash"
        
        logger.info(f"Gemini succeeded: {model_used}")
        return (response_text, model_used)
        
    except Exception as e:
        logger.warning(f"Gemini failed (will try Ollama): {e}")
    
    # Attempt 3: Ollama (local)
    try:
        logger.debug("Attempting Ollama...")
        import ollama
        
        model = os.getenv("OLLAMA_MODEL", "mistral")
        
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = response["message"]["content"]
        model_used = f"ollama/{model}"
        
        logger.info(f"Ollama succeeded: {model_used}")
        return (response_text, model_used)
        
    except Exception as e:
        logger.error(f"Ollama failed: {e}")
        raise LLMUnavailableError(f"All LLM providers failed. Last error: {e}") from e


class LLMUnavailableError(Exception):
    """Raised when all LLM providers fail."""
    pass


def run_orchestrator_agent(state: dict) -> dict:
    """
    Execute the Orchestrator Agent.
    
    Steps:
    1. Extract upstream outputs (Vision, Retrieval, Drafting, Critic)
    2. Determine routing based on top_class, top_confidence, and unresolved issues
    3. Generate reasoning_trace with entries from all agents
    4. Attempt to use LLM for justification (with fallback chain)
    5. If LLM unavailable, use vision-only justification (Requirement 8.8)
    6. Populate OrchestratorOutput and return updated state
    """
    try:
        logger.info("Starting Orchestrator Agent")
        
        vision_output = state.get("vision_output")
        if not vision_output:
            raise ValueError("Vision output not available")
        
        top_class = vision_output["top_class"]
        top_confidence = vision_output["top_confidence"]
        
        critic_issues_history = state.get("critic_issues_history", [])
        has_unresolved_issues = len(critic_issues_history) > 0
        
        logger.debug(f"Vision: {top_class} @ {top_confidence:.3f}, Unresolved issues: {has_unresolved_issues}")
        
        # Step 1: Determine routing
        routing = determine_routing(top_class, top_confidence, has_unresolved_issues)
        logger.info(f"Routing decision: {routing}")
        
        # Step 2: Generate reasoning_trace
        reasoning_trace = build_reasoning_trace(state)
        logger.debug(f"Reasoning trace: {len(reasoning_trace)} entries")
        
        # Step 3: Try to generate LLM-based justification
        justification = None
        model_used = None
        
        try:
            system_prompt = (
                "You are an expert radiologist summarizing an MRI analysis. "
                "Write a brief, clear justification for the routing decision. "
                "Reference specific evidence from the analysis. "
                "Keep it concise (2-3 sentences)."
            )
            
            prompt = (
                f"Based on an MRI analysis with these results:\n"
                f"- Classification: {top_class}\n"
                f"- Confidence: {top_confidence:.1%}\n"
                f"- All predictions: {json.dumps(vision_output['predictions'], indent=2)}\n"
                f"- Unresolved issues: {has_unresolved_issues}\n"
                f"\nWrite a brief justification for routing this case to: {routing.upper()}\n"
                f"Reference the model's confidence level and the specific classification."
            )
            
            logger.debug("Attempting LLM-based justification...")
            justification_text, model_used = call_llm_with_fallback(prompt, system_prompt)
            justification = justification_text
            logger.info(f"LLM justification generated using: {model_used}")
            
        except LLMUnavailableError as e:
            logger.warning(f"All LLM providers failed: {e}")
            justification = generate_vision_only_justification(
                top_class=top_class,
                top_confidence=top_confidence,
                routing=routing,
                has_unresolved_issues=has_unresolved_issues
            )
            model_used = "vision-only"
            logger.info("Using vision-only fallback justification")
            
        except Exception as e:
            logger.error(f"Unexpected error in LLM fallback: {e}", exc_info=True)
            justification = generate_vision_only_justification(
                top_class=top_class,
                top_confidence=top_confidence,
                routing=routing,
                has_unresolved_issues=has_unresolved_issues
            )
            model_used = "no-llm"
            logger.info("Using fallback justification (error)")
        
        if not justification:
            logger.warning("No justification generated, using fallback")
            justification = generate_vision_only_justification(
                top_class=top_class,
                top_confidence=top_confidence,
                routing=routing,
                has_unresolved_issues=has_unresolved_issues
            )
            if not model_used:
                model_used = "fallback"
        
        # Step 4: Create OrchestratorOutput
        orchestrator_output: OrchestratorOutput = {
            "routing": routing,
            "justification": justification,
            "reasoning_trace": reasoning_trace,
            "model_used": model_used
        }
        
        state["orchestrator_output"] = orchestrator_output
        
        logger.info(
            f"Orchestrator Agent complete: routing={routing}, model={model_used}, "
            f"trace_entries={len(reasoning_trace)}"
        )
        
        return state
        
    except Exception as e:
        logger.error(f"Orchestrator Agent failed: {e}", exc_info=True)
        raise
