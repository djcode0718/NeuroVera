#!/usr/bin/env python
"""Quick verification that drafting module imports correctly."""

from app.agents.drafting import run_drafting_agent, DraftingOutput, call_llm_with_fallback
import inspect

# Verify function signatures
sig = inspect.signature(run_drafting_agent)
print('✓ run_drafting_agent signature:', sig)

sig = inspect.signature(call_llm_with_fallback)
print('✓ call_llm_with_fallback signature:', sig)

# Verify DraftingOutput TypedDict
print('✓ DraftingOutput fields:', DraftingOutput.__annotations__)

print('✓ All imports successful!')
