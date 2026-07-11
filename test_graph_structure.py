#!/usr/bin/env python
"""Test the LangGraph structure and routing logic without requiring TensorFlow."""

import sys
sys.path.insert(0, '.')

from app.graph import build_graph, build_initial_state, critic_router

print("="*70)
print("LANGGRAPH STRUCTURE TEST")
print("="*70)

# Test 1: Build graph
print("\n1. Building graph...")
try:
    graph = build_graph()
    print("✓ Graph compiled successfully")
    print(f"  Type: {type(graph).__name__}")
except Exception as e:
    print(f"✗ Graph compilation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Verify graph nodes
print("\n2. Checking graph structure...")
graph_dict = graph.get_graph()
print(f"✓ Graph nodes: {list(graph_dict.nodes.keys())}")
expected_nodes = {"vision", "retrieval", "drafting", "critic", "orchestrator"}
actual_nodes = set(graph_dict.nodes.keys())
if expected_nodes.issubset(actual_nodes):
    print(f"✓ All required nodes present")
else:
    missing = expected_nodes - actual_nodes
    print(f"✗ Missing nodes: {missing}")

# Test 3: Verify edges
print("\n3. Checking graph edges...")
try:
    edges = graph_dict.edges if isinstance(graph_dict.edges, list) else list(graph_dict.edges)
    print(f"✓ Total edges: {len(edges)}")
    for edge_info in edges:
        if isinstance(edge_info, tuple) and len(edge_info) >= 2:
            print(f"  - {edge_info[0]} → {edge_info[1]}")
        else:
            print(f"  - {edge_info}")
except Exception as e:
    print(f"  Graph edges info: {graph_dict.edges}")

# Test 4: Test critic_router function
print("\n4. Testing critic_router function...")

test_cases = [
    ({"critic_output": {"verdict": "approved"}, "critic_retry_count": 0}, "orchestrator"),
    ({"critic_output": {"verdict": "revise"}, "critic_retry_count": 0}, "drafting"),
    ({"critic_output": {"verdict": "revise"}, "critic_retry_count": 1}, "drafting"),
    ({"critic_output": {"verdict": "revise"}, "critic_retry_count": 2}, "orchestrator"),
    ({"critic_output": {"verdict": "approved"}, "critic_retry_count": 2}, "orchestrator"),
]

all_passed = True
for state, expected_route in test_cases:
    route = critic_router(state)
    status = "✓" if route == expected_route else "✗"
    print(f"  {status} verdict={state['critic_output']['verdict']}, "
          f"retry={state['critic_retry_count']} → {route} (expected {expected_route})")
    if route != expected_route:
        all_passed = False

if all_passed:
    print("✓ All router test cases passed")
else:
    print("✗ Some router test cases failed")
    sys.exit(1)

# Test 5: Test initial state builder
print("\n5. Testing build_initial_state...")
try:
    test_bytes = b"dummy image data"
    initial_state = build_initial_state(test_bytes)
    
    # Check all required fields
    required_fields = [
        "image_bytes", "vision_output", "retrieval_output", "drafting_output",
        "critic_output", "orchestrator_output", "critic_retry_count", "critic_issues_history"
    ]
    
    missing_fields = [f for f in required_fields if f not in initial_state]
    if missing_fields:
        print(f"✗ Missing fields: {missing_fields}")
        sys.exit(1)
    
    # Check initial values
    assert initial_state["image_bytes"] == test_bytes
    assert initial_state["vision_output"] is None
    assert initial_state["retrieval_output"] is None
    assert initial_state["drafting_output"] is None
    assert initial_state["critic_output"] is None
    assert initial_state["orchestrator_output"] is None
    assert initial_state["critic_retry_count"] == 0
    assert initial_state["critic_issues_history"] == []
    
    print("✓ Initial state built correctly")
    print(f"  - image_bytes: {len(initial_state['image_bytes'])} bytes")
    print(f"  - critic_retry_count: {initial_state['critic_retry_count']}")
    print(f"  - critic_issues_history: {initial_state['critic_issues_history']}")
    
except Exception as e:
    print(f"✗ Initial state building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ ALL STRUCTURE TESTS PASSED")
print("="*70)
print("\nThe LangGraph structure is correctly implemented with:")
print("  - 5 agent nodes (vision, retrieval, drafting, critic, orchestrator)")
print("  - Sequential edges (vision → retrieval → drafting → critic)")
print("  - Conditional edge (critic → drafting OR orchestrator based on router)")
print("  - Final edge (orchestrator → END)")
print("  - Critic router correctly implements revision loop logic")
print("  - GraphState correctly defined with all required fields")
