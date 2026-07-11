"""
Drafting Agent for NeuroTriage

The Drafting Agent is responsible for:
1. Synthesizing Vision + Retrieval data into a structured clinical report
2. Implementing an LLM fallback chain: Groq → Gemini → Ollama
3. Handling first-pass drafting (initial synthesis)
4. Handling revision passes (addressing Critic issues)
5. Including hedging language when confidence < 0.6
6. Tracking which LLM model was used

Report Structure (6 sections):
    1. Classification - Predicted tumor type with confidence
    2. Confidence - Assessment of confidence level (hedging if < 0.6)
    3. Visual Evidence - Interpretation of Grad-CAM heatmap
    4. Similar Cases - Reference to retrieved similar cases
    5. Clinical Notes - General information about predicted tumor type
    6. Caveats - Disclaimer that this is automated analysis, not clinical diagnosis

First Pass: Synthesize Vision + Retrieval data into structured report
Revision Pass: Re-draft addressing each issue from critic_issues

Example:
    from app.agents.drafting import run_drafting_agent
    
    state = {
        "vision_output": {...},
        "retrieval_output": {...},
        "critic_retry_count": 0,
        "critic_issues_history": [],
        "drafting_output": None
    }
    result = run_drafting_agent(state)
    print(result["drafting_output"]["draft_report"])
    print(result["drafting_output"]["model_used"])
"""

import logging
import os
from typing import TypedDict, Optional, Literal, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
import json

logger = logging.getLogger(__name__)


class DraftingOutput(TypedDict):
    """
    Output from the Drafting Agent.
    
    Fields:
        draft_report: str - Complete structured clinical report with 6 sections
        model_used: str - Format: "provider/model-name" (e.g., "groq/llama-3.1-8b-instant")
        revision_number: int - 0 for first pass, increments on each revision
    """
    draft_report: str
    model_used: str
    revision_number: int


class DraftingError(Exception):
    """Raised when Drafting Agent encounters an error."""
    pass


def call_llm_with_fallback(
    system_prompt: str,
    user_prompt: str
) -> Tuple[str, str]:
    """
    Try LLM providers in order: Groq → Gemini → Ollama.
    
    This implements the LLM fallback chain from the design spec:
    1. Try Groq with model from GROQ_FAST_MODEL env var (default: llama-3.1-8b-instant)
    2. On 429/quota error, fall back to Gemini (gemini-2.5-flash)
    3. On any error, fall back to Ollama (model from OLLAMA_MODEL env var)
    4. Return response from first successful provider
    5. Record which provider answered in model_used field
    
    Args:
        system_prompt: System message for the LLM
        user_prompt: User prompt (main query) for the LLM
        
    Returns:
        Tuple of (response_text, model_used_str)
        Where model_used_str is formatted as "provider/model-name"
        
    Raises:
        DraftingError: If all three providers fail (rare if Ollama is running)
    """
    
    # Try Groq first
    try:
        logger.debug("Attempting Groq LLM call...")
        from langchain_groq import ChatGroq
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.warning("GROQ_API_KEY not set, skipping Groq")
        else:
            groq_model = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
            
            llm = ChatGroq(
                model_name=groq_model,
                temperature=0.3,
                api_key=groq_api_key
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            logger.info(f"Groq LLM call successful with model {groq_model}")
            return response.content, f"groq/{groq_model}"
            
    except Exception as e:
        # Check if it's a quota/rate limit error
        error_str = str(e).lower()
        if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
            logger.warning(f"Groq quota/rate limit exceeded: {str(e)}, falling back to Gemini")
        else:
            logger.warning(f"Groq failed: {str(e)}, falling back to Gemini")
    
    # Try Gemini as fallback
    try:
        logger.debug("Attempting Gemini LLM call...")
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY not set, skipping Gemini")
        else:
            gemini_model = "gemini-2.5-flash"
            
            llm = ChatGoogleGenerativeAI(
                model=gemini_model,
                temperature=0.3,
                api_key=gemini_api_key
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            logger.info(f"Gemini LLM call successful with model {gemini_model}")
            return response.content, f"gemini/{gemini_model}"
            
    except Exception as e:
        logger.warning(f"Gemini failed: {str(e)}, falling back to Ollama")
    
    # Try Ollama as final fallback (should always work if running locally)
    try:
        logger.debug("Attempting Ollama LLM call...")
        from langchain_ollama import ChatOllama
        
        ollama_model = os.getenv("OLLAMA_MODEL", "mistral")
        
        llm = ChatOllama(
            model=ollama_model,
            temperature=0.3
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        logger.info(f"Ollama LLM call successful with model {ollama_model}")
        return response.content, f"ollama/{ollama_model}"
        
    except Exception as e:
        logger.error(f"Ollama failed (all providers exhausted): {str(e)}", exc_info=True)
        raise DraftingError(f"All LLM providers failed: {str(e)}") from e


def build_first_pass_user_prompt(
    vision_output: dict,
    retrieval_output: dict
) -> str:
    """
    Build user prompt for first-pass drafting (synthesis of Vision + Retrieval data).
    
    Args:
        vision_output: VisionOutput TypedDict with predictions, top_class, top_confidence, gradcam_image
        retrieval_output: RetrievalOutput TypedDict with similar_cases, reference_notes
        
    Returns:
        Formatted user prompt string
    """
    
    top_class = vision_output.get("top_class", "unknown")
    top_confidence = vision_output.get("top_confidence", 0.0)
    predictions = vision_output.get("predictions", {})
    
    similar_cases = retrieval_output.get("similar_cases", [])
    reference_notes = retrieval_output.get("reference_notes", [])
    
    # Format predictions for display
    predictions_str = "\n".join(
        f"  - {cls}: {conf:.1%}" for cls, conf in sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )
    
    # Format similar cases
    if similar_cases:
        similar_cases_str = "\n".join(
            f"  - Case {i+1}: {case.get('tumor_type', 'unknown')} (similarity: {case.get('similarity_score', 0):.2f})\n"
            f"    Summary: {case.get('summary', 'N/A')}"
            for i, case in enumerate(similar_cases[:3])
        )
    else:
        similar_cases_str = "  No similar cases found in case bank"
    
    # Format reference notes
    if reference_notes:
        reference_notes_str = "\n".join(f"  - {note}" for note in reference_notes)
    else:
        reference_notes_str = "  No reference notes available"
    
    # Build the prompt
    prompt = f"""Please write a structured clinical report based on the following MRI analysis results.

MODEL PREDICTIONS:
{predictions_str}

PREDICTION SUMMARY:
- Predicted Class: {top_class}
- Confidence: {top_confidence:.1%}

SIMILAR HISTORICAL CASES:
{similar_cases_str}

CLINICAL REFERENCE NOTES:
{reference_notes_str}

Please generate a report with the following sections:
1. Classification
2. Confidence
3. Visual Evidence
4. Similar Cases
5. Clinical Notes
6. Caveats

Make the report clear, professional, and appropriate for medical review. Be precise about confidence levels."""

    return prompt


def build_revision_pass_user_prompt(
    vision_output: dict,
    retrieval_output: dict,
    previous_draft: str,
    critic_issues: list[str]
) -> str:
    """
    Build user prompt for revision pass (addressing Critic issues).
    
    Args:
        vision_output: VisionOutput TypedDict
        retrieval_output: RetrievalOutput TypedDict
        previous_draft: The previous draft report
        critic_issues: List of issue strings from Critic
        
    Returns:
        Formatted user prompt string for revision
    """
    
    top_class = vision_output.get("top_class", "unknown")
    top_confidence = vision_output.get("top_confidence", 0.0)
    
    # Format issues
    issues_str = "\n".join(f"  - {issue}" for issue in critic_issues)
    
    prompt = f"""You previously drafted the following clinical report:

{previous_draft}

A clinical reviewer has identified the following issues that need to be addressed:

{issues_str}

Please revise the report to address each of these concerns. The predicted class is {top_class} with {top_confidence:.1%} confidence.

Please provide a revised report with the same structure:
1. Classification
2. Confidence
3. Visual Evidence
4. Similar Cases
5. Clinical Notes
6. Caveats

Be sure to explicitly acknowledge and address each issue raised by the reviewer."""

    return prompt


def run_drafting_agent(state: dict) -> dict:
    """
    Execute the Drafting Agent: synthesize Vision + Retrieval data into a structured report.
    
    This agent uses an LLM fallback chain (Groq → Gemini → Ollama) to generate a clinical report.
    
    On first pass (critic_retry_count == 0):
        - Synthesize Vision output + Retrieval output into structured report
        - Include all 6 required sections
        - Apply hedging language if top_confidence < 0.6
    
    On revision pass (critic_retry_count > 0):
        - Re-draft the report, explicitly addressing each issue from critic_issues
        - Include acknowledgment of feedback
    
    Args:
        state: GraphState dict containing:
            - "vision_output": VisionOutput with predictions, top_class, top_confidence
            - "retrieval_output": RetrievalOutput with similar_cases, reference_notes
            - "critic_retry_count": int (0 for first pass, >0 for revision)
            - "critic_issues_history": list[list[str]] (accumulated issues from each revision)
            - "drafting_output": Optional (from previous revision if exists)
            
    Returns:
        state: Updated GraphState dict with "drafting_output" populated
        
    Raises:
        DraftingError: If LLM call fails after exhausting all providers
        
    Preconditions:
        - state["vision_output"] is populated (Vision Agent has run)
        - state["retrieval_output"] is populated (Retrieval Agent has run)
        - GROQ_API_KEY, GEMINI_API_KEY, OLLAMA_MODEL configured in environment
        - At least Ollama is running locally
        
    Postconditions:
        - state["drafting_output"]["draft_report"] contains 6-section structured report
        - state["drafting_output"]["model_used"] is "provider/model-name"
        - state["drafting_output"]["revision_number"] matches current retry count
    """
    
    try:
        logger.info("Starting Drafting Agent")
        
        # Extract inputs from state
        vision_output = state.get("vision_output")
        retrieval_output = state.get("retrieval_output")
        critic_retry_count = state.get("critic_retry_count", 0)
        critic_issues_history = state.get("critic_issues_history", [])
        previous_drafting_output = state.get("drafting_output")
        
        # Validate inputs
        if not vision_output:
            raise DraftingError("vision_output not found in state")
        if not retrieval_output:
            raise DraftingError("retrieval_output not found in state")
        
        logger.debug(f"Drafting Agent: retry_count={critic_retry_count}")
        
        # System prompt (same for first pass and revision)
        system_prompt = """You are a medical AI assistant analyzing brain MRI scans. Your job is to write clear, 
structured clinical reports based on automated model predictions. Be precise and honest 
about confidence levels. Include appropriate hedging language when confidence is low.

Write reports with these sections:
1. Classification - the predicted diagnosis
2. Confidence - assessment of the prediction confidence
3. Visual Evidence - interpretation of the Grad-CAM heatmap
4. Similar Cases - reference to historical similar cases if available
5. Clinical Notes - background on the predicted condition
6. Caveats - disclaimer that this is automated, not a clinical diagnosis

When confidence < 0.6, use hedging language like "appears to", "may indicate", "suggests".

IMPORTANT: Each section should be clearly labeled with a header (e.g., "# Classification").
Format the report as markdown with clear section headers and bullet points where appropriate."""
        
        # Build user prompt based on pass type
        if critic_retry_count == 0:
            # First pass: synthesize vision + retrieval
            logger.info("Drafting Agent: First pass - synthesizing Vision + Retrieval data")
            user_prompt = build_first_pass_user_prompt(vision_output, retrieval_output)
        else:
            # Revision pass: address critic issues
            logger.info(f"Drafting Agent: Revision pass {critic_retry_count} - addressing critic issues")
            
            if not previous_drafting_output:
                raise DraftingError(
                    f"Revision pass requested but no previous drafting_output in state"
                )
            
            previous_draft = previous_drafting_output.get("draft_report", "")
            
            # Get issues from most recent critic feedback
            if critic_issues_history and len(critic_issues_history) > 0:
                current_issues = critic_issues_history[-1]
            else:
                current_issues = []
            
            user_prompt = build_revision_pass_user_prompt(
                vision_output,
                retrieval_output,
                previous_draft,
                current_issues
            )
        
        logger.debug(f"User prompt ({len(user_prompt)} chars) ready for LLM")
        
        # Call LLM with fallback chain
        logger.info("Calling LLM with fallback chain (Groq → Gemini → Ollama)")
        report_text, model_used = call_llm_with_fallback(system_prompt, user_prompt)
        
        logger.info(f"LLM response received from {model_used} ({len(report_text)} chars)")
        
        # Extract the report text
        draft_report = report_text.strip()
        
        # Verify report contains all 6 required sections (case-insensitive check)
        required_sections = [
            "classification",
            "confidence",
            "visual evidence",
            "similar cases",
            "clinical notes",
            "caveats"
        ]
        
        report_lower = draft_report.lower()
        missing_sections = []
        for section in required_sections:
            if section not in report_lower:
                logger.warning(f"Draft report missing section: {section}")
                missing_sections.append(section)
        
        if missing_sections:
            logger.warning(
                f"Draft report incomplete: missing sections: {', '.join(missing_sections)}"
            )
        
        # Check if hedging is present when confidence < 0.6
        top_confidence = vision_output.get("top_confidence", 1.0)
        if top_confidence < 0.6:
            hedging_words = ["appears", "may", "suggest", "could", "likely", "possible", "unclear"]
            has_hedging = any(word in report_lower for word in hedging_words)
            
            if not has_hedging:
                logger.warning(
                    f"Draft confidence < 0.6 ({top_confidence:.1%}) but no hedging language detected"
                )
        
        # Create DraftingOutput
        drafting_output: DraftingOutput = {
            "draft_report": draft_report,
            "model_used": model_used,
            "revision_number": critic_retry_count
        }
        
        # Update state with drafting output
        state["drafting_output"] = drafting_output
        
        logger.info(
            f"Drafting Agent complete: revision={critic_retry_count}, "
            f"model={model_used}, report_length={len(draft_report)}"
        )
        
        return state
        
    except DraftingError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in Drafting Agent: {str(e)}", exc_info=True)
        raise DraftingError(f"Drafting Agent failed: {str(e)}") from e
