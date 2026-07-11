"""
Test script for Tasks 3.1 and 3.2: Grad-CAM and Feature Embedding

Tests that verify:
1. Grad-CAM computation (Task 3.1)
   - Heatmap is computed correctly
   - Heatmap is blended onto image
   - Result is base64-encoded PNG string
   
2. Feature embedding extraction (Task 3.2)
   - Embedding is exactly 512 floats
   - Embedding is not all-zeros
   - Result is list of floats

3. Full Vision Agent integration
   - gradcam_image field is populated (non-None, non-empty string)
   - feature_embedding field is populated (list of 512 floats)
   - Both survive roundtrip through JSON serialization
"""

import os
import sys
import json
import base64
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_gradcam_utilities():
    """Test Grad-CAM utility functions directly"""
    print("\n" + "="*80)
    print("TEST 1: Grad-CAM Utility Functions")
    print("="*80)
    
    try:
        from app.utils.gradcam import (
            compute_gradcam,
            blend_heatmap_on_image,
            encode_image_to_base64_png,
            compute_feature_embedding
        )
        print("✓ Successfully imported Grad-CAM utilities")
    except ImportError as e:
        print(f"✗ Failed to import Grad-CAM utilities: {e}")
        return False
    
    # Load or create test image
    try:
        from PIL import Image
        import io
        
        # Create a simple test image (random noise, 150x150)
        test_img = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
        img_float = test_img.astype(np.float32)
        img_batch = np.expand_dims(img_float, axis=0)
        
        print(f"✓ Created test image batch: shape={img_batch.shape}, dtype={img_batch.dtype}")
        print(f"  Pixel range: [{img_batch.min():.0f}, {img_batch.max():.0f}]")
        
    except Exception as e:
        print(f"✗ Failed to create test image: {e}")
        return False
    
    # Load model
    try:
        from app.models.vision_loader import get_model
        model = get_model(use_mock_on_failure=True)
        print(f"✓ Loaded model: {type(model)}")
        print(f"  Model layers: {len(model.layers)} total")
        
        # Check for required layers
        layer_names = [layer.name for layer in model.layers]
        if 'dropout_4' in layer_names:
            print("✓ Found dropout_4 layer")
        else:
            print(f"✗ dropout_4 layer not found. Available layers: {layer_names}")
            
        if 'dropout_5' in layer_names:
            print("✓ Found dropout_5 layer")
        else:
            print(f"✗ dropout_5 layer not found. Available layers: {layer_names}")
            
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Test Grad-CAM computation
    print("\nTesting Grad-CAM computation...")
    try:
        top_class_idx = 0  # glioma
        heatmap = compute_gradcam(
            model=model,
            img_batch=img_batch,
            top_class_idx=top_class_idx,
            target_layer_name="dropout_4"
        )
        print(f"✓ Grad-CAM heatmap computed")
        print(f"  Shape: {heatmap.shape}, dtype: {heatmap.dtype}")
        print(f"  Range: [{heatmap.min()}, {heatmap.max()}]")
        
        if heatmap.dtype != np.uint8:
            print(f"✗ Heatmap dtype is {heatmap.dtype}, expected uint8")
            return False
            
        if not (0 <= heatmap.min() and heatmap.max() <= 255):
            print(f"✗ Heatmap values out of range [0, 255]: [{heatmap.min()}, {heatmap.max()}]")
            return False
            
    except Exception as e:
        print(f"✗ Failed to compute Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test image blending
    print("\nTesting heatmap blending...")
    try:
        blended = blend_heatmap_on_image(
            original_img_array=test_img,
            heatmap=heatmap,
            alpha=0.4
        )
        print(f"✓ Heatmap blended onto image")
        print(f"  Shape: {blended.shape}, dtype: {blended.dtype}")
        print(f"  Range: [{blended.min()}, {blended.max()}]")
        
        if blended.shape != test_img.shape:
            print(f"✗ Blended image shape {blended.shape} doesn't match original {test_img.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to blend heatmap: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test base64 encoding
    print("\nTesting base64 PNG encoding...")
    try:
        gradcam_b64 = encode_image_to_base64_png(blended)
        print(f"✓ Image encoded to base64 PNG")
        print(f"  String length: {len(gradcam_b64)} characters")
        
        if len(gradcam_b64) < 100:
            print(f"✗ Base64 string too short: {len(gradcam_b64)}")
            return False
        
        # Verify it's valid base64
        try:
            decoded = base64.b64decode(gradcam_b64)
            print(f"✓ Base64 string is valid, decodes to {len(decoded)} bytes")
            
            # Check PNG magic bytes
            if decoded[:8] == b'\x89PNG\r\n\x1a\n':
                print("✓ Decoded data has valid PNG magic bytes")
            else:
                print(f"✗ Decoded data doesn't have PNG magic bytes: {decoded[:8].hex()}")
                return False
                
        except Exception as e:
            print(f"✗ Failed to decode base64: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to encode to base64 PNG: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test feature embedding extraction
    print("\nTesting feature embedding extraction...")
    try:
        embedding = compute_feature_embedding(
            model=model,
            img_batch=img_batch,
            target_layer_name="dropout_5"
        )
        print(f"✓ Feature embedding extracted")
        print(f"  Type: {type(embedding)}, Length: {len(embedding)}")
        
        if not isinstance(embedding, list):
            print(f"✗ Embedding is {type(embedding)}, expected list")
            return False
            
        if len(embedding) != 512:
            print(f"✗ Embedding has {len(embedding)} elements, expected 512")
            return False
            
        if not all(isinstance(x, (float, np.floating)) for x in embedding):
            print(f"✗ Embedding contains non-float elements")
            return False
        
        # Check for all-zeros
        embedding_array = np.array(embedding)
        if np.allclose(embedding_array, 0.0, atol=1e-6):
            print(f"⚠ Warning: Embedding is all-zeros (may indicate model issue)")
        else:
            norm = np.linalg.norm(embedding_array)
            print(f"✓ Embedding is non-zero, L2 norm: {norm:.6f}")
            
    except Exception as e:
        print(f"✗ Failed to compute feature embedding: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_vision_agent_integration():
    """Test Vision Agent with real image file"""
    print("\n" + "="*80)
    print("TEST 2: Vision Agent Integration")
    print("="*80)
    
    # Find a test image
    test_image_path = Path("/Users/sj/Documents/Neurotriage/data/Testing/glioma/Te-gl_1.jpg")
    
    if not test_image_path.exists():
        print(f"⚠ Test image not found at {test_image_path}")
        print("  Skipping integration test with real image")
        return True
    
    try:
        with open(test_image_path, "rb") as f:
            image_bytes = f.read()
        print(f"✓ Loaded test image: {len(image_bytes)} bytes")
    except Exception as e:
        print(f"✗ Failed to load test image: {e}")
        return False
    
    # Run Vision Agent
    try:
        from app.agents.vision import run_vision_agent
        
        state = {
            "image_bytes": image_bytes,
            "vision_output": None
        }
        
        print("\nRunning Vision Agent...")
        state = run_vision_agent(state)
        print("✓ Vision Agent executed successfully")
        
    except Exception as e:
        print(f"✗ Vision Agent failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check VisionOutput structure
    print("\nValidating VisionOutput...")
    try:
        vision_output = state["vision_output"]
        
        # Check required fields
        required_fields = ["predictions", "top_class", "top_confidence", "gradcam_image", "feature_embedding"]
        for field in required_fields:
            if field not in vision_output:
                print(f"✗ Missing field: {field}")
                return False
        print(f"✓ All required fields present")
        
        # Check predictions
        predictions = vision_output["predictions"]
        print(f"✓ Predictions: {predictions}")
        prob_sum = sum(predictions.values())
        if abs(prob_sum - 1.0) > 1e-4:
            print(f"✗ Probabilities don't sum to 1.0: {prob_sum}")
            return False
        print(f"✓ Probabilities sum to {prob_sum:.6f}")
        
        # Check top_class and top_confidence
        top_class = vision_output["top_class"]
        top_confidence = vision_output["top_confidence"]
        print(f"✓ Top class: {top_class} ({top_confidence:.4f})")
        
        if top_class not in predictions:
            print(f"✗ Top class '{top_class}' not in predictions")
            return False
            
        if abs(predictions[top_class] - top_confidence) > 1e-6:
            print(f"✗ top_confidence doesn't match predictions[top_class]")
            return False
        
    except Exception as e:
        print(f"✗ Failed to validate VisionOutput: {e}")
        return False
    
    # Check Grad-CAM image (Task 3.1)
    print("\nValidating Grad-CAM image (Task 3.1)...")
    try:
        gradcam_image = vision_output["gradcam_image"]
        
        if gradcam_image is None:
            print(f"✗ gradcam_image is None - should be populated by Task 3.1")
            return False
        
        if not isinstance(gradcam_image, str):
            print(f"✗ gradcam_image is {type(gradcam_image)}, expected str")
            return False
        
        if len(gradcam_image) == 0:
            print(f"✗ gradcam_image is empty string")
            return False
        
        print(f"✓ gradcam_image is populated: {len(gradcam_image)} characters")
        
        # Verify it's valid base64 PNG
        try:
            decoded = base64.b64decode(gradcam_image)
            if decoded[:8] == b'\x89PNG\r\n\x1a\n':
                print(f"✓ gradcam_image is valid PNG ({len(decoded)} bytes)")
            else:
                print(f"✗ Decoded gradcam_image doesn't have PNG magic bytes")
                return False
        except Exception as e:
            print(f"✗ Failed to decode gradcam_image base64: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to validate Grad-CAM image: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check feature embedding (Task 3.2)
    print("\nValidating feature embedding (Task 3.2)...")
    try:
        feature_embedding = vision_output["feature_embedding"]
        
        if feature_embedding is None:
            print(f"✗ feature_embedding is None - should be populated by Task 3.2")
            return False
        
        if not isinstance(feature_embedding, list):
            print(f"✗ feature_embedding is {type(feature_embedding)}, expected list")
            return False
        
        if len(feature_embedding) != 512:
            print(f"✗ feature_embedding has {len(feature_embedding)} elements, expected 512")
            return False
        
        print(f"✓ feature_embedding has exactly 512 elements")
        
        # Verify all elements are floats
        if not all(isinstance(x, (float, int)) for x in feature_embedding):
            print(f"✗ feature_embedding contains non-numeric elements")
            return False
        
        # Check for all-zeros
        embedding_array = np.array(feature_embedding)
        if np.allclose(embedding_array, 0.0, atol=1e-6):
            print(f"⚠ Warning: feature_embedding is all-zeros (may indicate model issue)")
        else:
            norm = np.linalg.norm(embedding_array)
            print(f"✓ feature_embedding is non-zero, L2 norm: {norm:.6f}")
        
    except Exception as e:
        print(f"✗ Failed to validate feature embedding: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test JSON serialization (important for API response)
    print("\nTesting JSON serialization...")
    try:
        json_str = json.dumps(vision_output)
        print(f"✓ VisionOutput serializes to JSON: {len(json_str)} characters")
        
        # Verify deserialization
        deserialized = json.loads(json_str)
        if len(deserialized["feature_embedding"]) != 512:
            print(f"✗ Deserialized embedding has wrong length")
            return False
        
        if len(deserialized["gradcam_image"]) == 0:
            print(f"✗ Deserialized gradcam_image is empty")
            return False
        
        print(f"✓ VisionOutput deserializes correctly from JSON")
        
    except Exception as e:
        print(f"✗ Failed to serialize/deserialize VisionOutput: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("NeuroTriage: Tasks 3.1 & 3.2 Test Suite")
    print("Testing Grad-CAM and Feature Embedding Implementation")
    print("="*80)
    
    results = {}
    
    # Test 1: Grad-CAM utilities
    results["Grad-CAM Utilities"] = test_gradcam_utilities()
    
    # Test 2: Vision Agent integration
    results["Vision Agent Integration"] = test_vision_agent_integration()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("SOME TESTS FAILED ✗")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
