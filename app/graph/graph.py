"""
LangGraph Graph Definition for NeuroTriage

This module implements the LangGraph computation graph that orchestrates the five agents:
1. Vision Agent - Classifies MRI scan and generates Grad-CAM
2. Retrieval Agent - Finds similar historical cases from case bank
3. Drafting Agent - Generates plain-language report
4. Critic Agent - Verifies draft against raw Vision evidence
5. Orchestrator Agent - Routes case to urgency tier

Graph topology:
    - Sequential edges: vision → retrieval → drafting → critic
    - Conditional edge after critic: if verdict="revise" and retry < 2 → drafting else → orchestrator
    - Final node: orchestrator (always runs, produces routing)

StateGraph pattern:
    - Define all nodes as functions that take state → updated state
    - Use StateGraph to wire nodes with edges
    - Conditional edges implement the critic revision loop
    - Compile graph for execution

Execution model:
    graph.invoke(initial_state) runs the full pipeline and returns final_state
    Each agent updates specific fields in the shared GraphState
"""

import logging
from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ============================================================================
# TypeDicts for all agent outputs (from design spec)
# ============================================================================

class VisionOutput(TypedDict):
    """Output from Vision Agent"""
    predictions: dict[str, float]
    top_class: str
    top_confidence: float
    gradcam_image: Optional[str]
    feature_embedding: Optional[list[float]]


class SimilarCase(TypedDict):
    """A historical case entry returned by Retrieval Agent"""
    case_id: str
    tumor_type: str
    confidence: float
    summary: str
    similarity_score: float


class RetrievalOutput(TypedDict):
    """Output from Retrieval Agent"""
    similar_cases: list[SimilarCase]
    reference_notes: list[str]


class DraftingOutput(TypedDict):
    """Output from Drafting Agent"""
    draft_report: str
    model_used: str
    revision_number: int


class CriticOutput(TypedDict):
    """Output from Critic Agent"""
    verdict: Literal["approved", "revise"]
    issues: list[str]
    model_used: str


class OrchestratorOutput(TypedDict):
    """Output from Orchestrator Agent"""
    routing: Literal["auto-clear", "needs-review", "urgent"]
    justification: str
    reasoning_trace: list[dict]
    model_used: str


# ============================================================================
# GraphState: Shared state passed between all agents
# ============================================================================

class GraphState(TypedDict):
    """
    Shared state dict passed through the LangGraph computation.
    
    Each agent receives this state, updates specific fields, and passes it forward.
    
    Fields:
        image_bytes: Raw bytes of uploaded MRI image (JPEG/PNG)
        vision_output: Output from Vision Agent (populated after vision node)
        retrieval_output: Output from Retrieval Agent (populated after retrieval node)
        drafting_output: Output from Drafting Agent (populated after drafting node)
        critic_output: Output from Critic Agent (populated after critic node)
        orchestrator_output: Output from Orchestrator Agent (final output)
        critic_retry_count: Number of times Critic has sent back to Drafting (0-2)
        critic_issues_history: List of issue lists from each Critic verdict
    """
    image_bytes: bytes
    vision_output: Optional[VisionOutput]
    retrieval_output: Optional[RetrievalOutput]
    drafting_output: Optional[DraftingOutput]
    critic_output: Optional[CriticOutput]
    orchestrator_output: Optional[OrchestratorOutput]
    critic_retry_count: int
    critic_issues_history: list[list[str]]


# ============================================================================
# Node implementations (wrappers around agent functions)
# ============================================================================

def vision_node(state: GraphState) -> GraphState:
    """
    Execute the Vision Agent node.
    
    This node:
    1. Extracts image_bytes from state
    2. Calls run_vision_agent to perform classification and Grad-CAM
    3. Populates state["vision_output"]
    4. Returns updated state
    
    Raises:
        VisionError if inference fails
    """
    logger.info("Vision node starting")
    
    from app.agents.vision import run_vision_agent
    
    try:
        state = run_vision_agent(state)
        logger.info(f"Vision node complete: {state['vision_output']['top_class']} "
                   f"({state['vision_output']['top_confidence']:.2%})")
        return state
    except Exception as e:
        logger.error(f"Vision node failed: {e}", exc_info=True)
        raise


def retrieval_node(state: GraphState) -> GraphState:
    """
    Execute the Retrieval Agent node.
    
    This node:
    1. Extracts vision_output with feature_embedding and top_class
    2. Calls run_retrieval_agent to query case bank
    3. Populates state["retrieval_output"]
    4. Returns updated state
    
    Gracefully handles missing case bank (returns empty lists)
    """
    logger.info("Retrieval node starting")
    
    from app.agents.retrieval import run_retrieval_agent
    
    try:
        state = run_retrieval_agent(state)
        num_cases = len(state['retrieval_output']['similar_cases'])
        num_notes = len(state['retrieval_output']['reference_notes'])
        logger.info(f"Retrieval node complete: {num_cases} similar cases, {num_notes} reference notes")
        return state
    except Exception as e:
        logger.error(f"Retrieval node failed: {e}", exc_info=True)
        raise


def drafting_node(state: GraphState) -> GraphState:
    """
    Execute the Drafting Agent node.
    
    This node:
    1. Extracts vision_output and retrieval_output
    2. On first pass (retry_count=0): generates initial draft
    3. On revision pass (retry_count>0): re-drafts addressing critic_issues
    4. Calls run_drafting_agent with all context
    5. Populates state["drafting_output"] with revision_number tracking
    6. Returns updated state
    """
    logger.info(f"Drafting node starting (retry_count={state['critic_retry_count']})")
    
    from app.agents.drafting import run_drafting_agent
    
    try:
        state = run_drafting_agent(state)
        revision = state['drafting_output']['revision_number']
        model = state['drafting_output']['model_used']
        logger.info(f"Drafting node complete: revision {revision} using {model}")
        return state
    except Exception as e:
        logger.error(f"Drafting node failed: {e}", exc_info=True)
        raise


def critic_node(state: GraphState) -> GraphState:
    """
    Execute the Critic Agent node.
    
    This node:
    1. Extracts draft_report from drafting_output and raw vision_output
    2. Calls run_critic_agent to verify draft
    3. Populates state["critic_output"] with verdict and issues
    4. On verdict="revise": appends issues to critic_issues_history
    5. Returns updated state
    
    If all LLM providers fail: returns verdict="approved" (graceful degradation)
    """
    logger.info("Critic node starting")
    
    from app.agents.critic import run_critic_agent
    
    try:
        state = run_critic_agent(state)
        verdict = state['critic_output']['verdict']
        num_issues = len(state['critic_output']['issues'])
        logger.info(f"Critic node complete: verdict={verdict} ({num_issues} issues)")
        
        # Track issues history for orchestrator context
        if verdict == "revise":
            state['critic_issues_history'].append(state['critic_output']['issues'])
        
        return state
    except Exception as e:
        logger.error(f"Critic node failed: {e}", exc_info=True)
        raise


def orchestrator_node(state: GraphState) -> GraphState:
    """
    Execute the Orchestrator Agent node.
    
    This node:
    1. Extracts all upstream outputs (vision, retrieval, drafting, critic)
    2. Calls run_orchestrator_agent to determine routing
    3. Populates state["orchestrator_output"] with routing, justification, reasoning_trace
    4. Returns updated state
    
    This is the final node - always executes regardless of critic verdict
    """
    logger.info("Orchestrator node starting")
    
    from app.agents.orchestrator import run_orchestrator_agent
    
    try:
        state = run_orchestrator_agent(state)
        routing = state['orchestrator_output']['routing']
        model = state['orchestrator_output']['model_used']
        logger.info(f"Orchestrator node complete: routing={routing} using {model}")
        return state
    except Exception as e:
        logger.error(f"Orchestrator node failed: {e}", exc_info=True)
        raise


# ============================================================================
# Conditional Edge Router
# ============================================================================

def critic_router(state: GraphState) -> str:
    """
    Conditional edge router after the Critic Agent node.
    
    Logic:
    - IF critic_output.verdict == "revise" AND critic_retry_count < 2:
        RETURN "drafting" (route back to Drafting Agent for revision)
    - ELSE:
        RETURN "orchestrator" (proceed to Orchestrator Agent)
    
    This function never raises - it must always return a valid node name.
    
    NOTE: The drafting node is responsible for incrementing critic_retry_count
    when called on a revision pass. This router does NOT modify state.
    
    Preconditions:
        - state["critic_output"] is populated (Critic node has run)
        - state["critic_retry_count"] is a non-negative integer
    
    Postconditions:
        - Returns exactly one of: "drafting" or "orchestrator"
        - The returned value corresponds to the next node to execute
        - state is not modified
    
    Args:
        state: Current GraphState
        
    Returns:
        str: Either "drafting" or "orchestrator"
    """
    try:
        critic_verdict = state['critic_output']['verdict']
        retry_count = state['critic_retry_count']
        
        logger.debug(f"critic_router: verdict={critic_verdict}, retry_count={retry_count}")
        
        # Check if we should retry drafting
        if critic_verdict == "revise" and retry_count < 2:
            logger.info(f"Routing back to drafting for revision {retry_count + 1}")
            return "drafting"
        else:
            # Either approved or retry cap reached
            if critic_verdict == "revise":
                logger.info(f"Routing to orchestrator (retry cap reached at {retry_count})")
            else:
                logger.info("Routing to orchestrator (critic approved)")
            return "orchestrator"
            
    except Exception as e:
        logger.error(f"critic_router failed: {e}", exc_info=True)
        # Safe default: proceed to orchestrator on any error
        return "orchestrator"


# ============================================================================
# Graph Construction
# ============================================================================

def build_graph():
    """
    Construct and compile the LangGraph computation graph.
    
    Graph structure:
        START
          ↓
        vision (node)
          ↓
        retrieval (node)
          ↓
        drafting (node)
          ↓
        critic (node)
          ↓
        critic_router (conditional edge)
        /              \
    drafting          orchestrator (node)
      ↓                    ↓
    critic ←───────────────→ END
      ↓ (if retry < 2)
    critic_router
    
    The conditional edge after critic implements the revision loop:
    - If verdict="revise" and retry_count < 2: route back to drafting
    - Otherwise: route to orchestrator (final node)
    
    Returns:
        CompiledGraph: A compiled LangGraph ready for .invoke(initial_state)
        
    Raises:
        Exception: If graph construction fails (e.g., missing agent modules)
    """
    logger.info("Building LangGraph computation graph")
    
    try:
        # Create StateGraph with GraphState schema
        graph = StateGraph(GraphState)
        
        # Add all five agent nodes
        graph.add_node("vision", vision_node)
        graph.add_node("retrieval", retrieval_node)
        graph.add_node("drafting", drafting_node)
        graph.add_node("critic", critic_node)
        graph.add_node("orchestrator", orchestrator_node)
        
        logger.debug("Added 5 agent nodes")
        
        # Add sequential edges: vision → retrieval → drafting → critic
        graph.add_edge("vision", "retrieval")
        graph.add_edge("retrieval", "drafting")
        graph.add_edge("drafting", "critic")
        
        logger.debug("Added sequential edges")
        
        # Add conditional edge after critic
        # This implements the revision loop logic
        graph.add_conditional_edges(
            "critic",
            critic_router,
            {
                "drafting": "drafting",      # Revise: loop back to drafting
                "orchestrator": "orchestrator"  # Approve or cap: proceed to orchestrator
            }
        )
        
        logger.debug("Added conditional edge from critic")
        
        # Add edge from orchestrator to END (final node)
        graph.add_edge("orchestrator", END)
        
        logger.debug("Added edge to END")
        
        # Set entry point to vision node
        graph.set_entry_point("vision")
        
        logger.debug("Set entry point to vision")
        
        # Compile the graph
        compiled_graph = graph.compile()
        
        logger.info("Graph compiled successfully")
        
        return compiled_graph
        
    except Exception as e:
        logger.error(f"Failed to build graph: {e}", exc_info=True)
        raise


# ============================================================================
# Utility: Build initial state
# ============================================================================

def build_initial_state(image_bytes: bytes) -> GraphState:
    """
    Construct an initial GraphState for a new analysis run.
    
    Args:
        image_bytes: Raw bytes of MRI image (JPEG/PNG)
        
    Returns:
        GraphState: Initial state with all agent outputs as None and counters as 0
    """
    return {
        "image_bytes": image_bytes,
        "vision_output": None,
        "retrieval_output": None,
        "drafting_output": None,
        "critic_output": None,
        "orchestrator_output": None,
        "critic_retry_count": 0,
        "critic_issues_history": []
    }
