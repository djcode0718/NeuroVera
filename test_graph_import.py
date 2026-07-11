#!/usr/bin/env python
"""Quick test to verify LangGraph structure compiles."""

import sys
sys.path.insert(0, '.')

# Try importing the graph module
from app.graph import build_graph, GraphState, build_initial_state
print('✓ Graph module imports successfully')
print(f'✓ GraphState type: {GraphState}')
print('✓ Graph building...')

try:
    graph = build_graph()
    print('✓ Graph compiled successfully')
    print(f'✓ Graph type: {type(graph)}')
except Exception as e:
    print(f'✗ Graph compilation failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\nAll imports successful!')
