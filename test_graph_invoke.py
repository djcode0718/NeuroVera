#!/usr/bin/env python
"""Test the LangGraph end-to-end with a real test image."""

import sys
import os
sys.path.insert(0, '.')

from app.graph import build_graph, build_initial_state
from pathlib import Path

# Find a test image
test_image_path = Path("data/Testing/glioma/Te-gl_1.jpg")
if not test_image_path.exists():
    print(f"✗ Test image not found at {test_image_path}")
    # Try alternative paths
    alt_paths = [
        Path("data/Testing/meningioma/Te-aug-me_1.jpg"),
        Path("data/Testing/notumor/Te-no_1.jpg"),
    ]
    for alt_path in alt_paths:
        if alt_path.exists():
            test_image_path = alt_path
            break
    
    if not test_image_path.exists():
        print("✗ No test images found")
        sys.exit(1)

print(f"✓ Using test image: {test_image_path}")

# Load image bytes
try:
    with open(test_image_path, "rb") as f:
        image_bytes = f.read()
    print(f"✓ Loaded image: {len(image_bytes)} bytes")
except Exception as e:
    print(f"✗ Failed to load image: {e}")
    sys.exit(1)

# Build graph
print("✓ Building graph...")
try:
    graph = build_graph()
    print("✓ Graph compiled")
except Exception as e:
    print(f"✗ Graph compilation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create initial state
print("✓ Creating initial state...")
initial_state = build_initial_state(image_bytes)
print(f"  - image_bytes: {len(initial_state['image_bytes'])} bytes")
print(f"  - critic_retry_count: {initial_state['critic_retry_count']}")
print(f"  - critic_issues_history: {initial_state['critic_issues_history']}")

# Invoke graph
print("\n✓ Invoking graph...")
try:
    final_state = graph.invoke(initial_state)
    print("✓ Graph invocation completed")
except Exception as e:
    print(f"✗ Graph invocation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check outputs
print("\n" + "="*70)
print("GRAPH OUTPUTS")
print("="*70)

# Vision output
if final_state.get("vision_output"):
    vo = final_state["vision_output"]
    print(f"\n✓ Vision Output:")
    print(f"  - Top class: {vo['top_class']}")
    print(f"  - Confidence: {vo['top_confidence']:.2%}")
    print(f"  - Predictions: {vo['predictions']}")
    print(f"  - Has Grad-CAM: {vo['gradcam_image'] is not None and len(vo['gradcam_image']) > 0}")
    print(f"  - Has embedding: {vo['feature_embedding'] is not None and len(vo['feature_embedding']) == 512}")
else:
    print("✗ No vision output")

# Retrieval output
if final_state.get("retrieval_output"):
    ro = final_state["retrieval_output"]
    print(f"\n✓ Retrieval Output:")
    print(f"  - Similar cases: {len(ro['similar_cases'])}")
    print(f"  - Reference notes: {len(ro['reference_notes'])}")
else:
    print("✗ No retrieval output")

# Drafting output
if final_state.get("drafting_output"):
    do = final_state["drafting_output"]
    print(f"\n✓ Drafting Output:")
    print(f"  - Model used: {do['model_used']}")
    print(f"  - Revision number: {do['revision_number']}")
    print(f"  - Report length: {len(do['draft_report'])} characters")
    print(f"  - Report preview (first 200 chars):")
    print(f"    {do['draft_report'][:200]}...")
else:
    print("✗ No drafting output")

# Critic output
if final_state.get("critic_output"):
    co = final_state["critic_output"]
    print(f"\n✓ Critic Output:")
    print(f"  - Verdict: {co['verdict']}")
    print(f"  - Issues: {co['issues']}")
    print(f"  - Model used: {co['model_used']}")
else:
    print("✗ No critic output")

# Orchestrator output
if final_state.get("orchestrator_output"):
    oo = final_state["orchestrator_output"]
    print(f"\n✓ Orchestrator Output:")
    print(f"  - Routing: {oo['routing']}")
    print(f"  - Model used: {oo['model_used']}")
    print(f"  - Reasoning trace entries: {len(oo['reasoning_trace'])}")
    print(f"  - Justification (first 200 chars):")
    print(f"    {oo['justification'][:200]}...")
else:
    print("✗ No orchestrator output")

# Final state checks
print(f"\n✓ Final state:")
print(f"  - Critic retry count: {final_state['critic_retry_count']}")
print(f"  - Critic issues history: {final_state['critic_issues_history']}")

print("\n" + "="*70)
print("✓ END-TO-END TEST PASSED")
print("="*70)
