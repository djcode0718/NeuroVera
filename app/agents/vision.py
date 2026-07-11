"""
Vision Agent for NeuroTriage

The Vision Agent is responsible for:
1. Decoding MRI image bytes (JPEG/PNG)
2. Resizing to 150×150×3 pixels
3. Casting to float32 WITHOUT normalization (pixel range [0, 255])
4. Running inference on VGG16 classifier
5. Mapping outputs to class names
6. Verifying probability normalization
7. Returning VisionOutput TypedDict with predictions, top_class, top_confidence

CRITICAL: Pixel values must remain in range [0, 255] and must NOT be divided by 255.
This is required by the model and verified in design specification and property tests.

Example:
    from app.agents.vision import run_vision_agent
    
    state = {
        "image_bytes": open("test.jpg", "rb").read(),
        "vision_output": None
    }
    result = run_vision_agent(state)
    print(result["vision_output"]["top_class"])
    print(result["vision_output"]["top_confidence"])
"""

import logging
from typing import TypedDict, Optional
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Class name mapping from model indices to human-readable labels
CLASS_NAMES = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary"
}

# Target image dimensions for model input
TARGET_SIZE = (150, 150)

# Probability normalization tolerance: probabilities should sum to 1.0 ± this value
PROB_TOLERANCE = 1e-4


class VisionOutput(TypedDict):
    """
    Output of the Vision Agent inference.
    
    Fields:
        predictions: dict mapping class names to probabilities (sum ≈ 1.0)
        top_class: string name of the highest-probability class
        top_confidence: float confidence of the top class (0.0 to 1.0)
        gradcam_image: base64-encoded PNG heatmap (added in Task 3.1)
        feature_embedding: 512-dim feature vector (added in Task 3.2)
    """
    predictions: dict[str, float]
    top_class: str
    top_confidence: float
    gradcam_image: Optional[str]  # None until Task 3.1
    feature_embedding: Optional[list[float]]  # None until Task 3.2


class VisionError(Exception):
    """Raised when Vision Agent encounters an error."""
    pass


def run_vision_agent(state: dict) -> dict:
    """
    Execute the Vision Agent: load image, preprocess, run inference, map outputs.
    
    This function implements the Vision Agent algorithm from the design document:
    1. Extract image_bytes from state
    2. Decode image using PIL (auto-detects JPEG/PNG)
    3. Resize to 150×150×3 using PIL
    4. Convert to numpy array, cast to float32
    5. CRITICAL: Do NOT divide by 255 (pixel range remains [0, 255])
    6. Expand dims for batch axis: shape becomes (1, 150, 150, 3)
    7. Call model.predict() with raw pixel values
    8. Extract probabilities and map indices to class names
    9. Verify sum of probabilities ≈ 1.0 ± 1e-4
    10. Find argmax as top_class and corresponding probability as top_confidence
    11. Return VisionOutput dict (gradcam_image and feature_embedding are None for now)
    
    Args:
        state: GraphState dict containing at minimum:
            - "image_bytes": bytes of JPEG/PNG image
            
    Returns:
        state: Updated GraphState dict with "vision_output" populated
        
    Raises:
        VisionError: If image decoding, preprocessing, or inference fails
        
    Preconditions:
        - state["image_bytes"] is non-empty valid JPEG/PNG data
        - model is loaded and available from vision_loader
        - model expects (1, 150, 150, 3) input with pixel values in [0, 255]
        
    Postconditions:
        - state["vision_output"]["predictions"] contains all 4 classes
        - sum(predictions.values()) ≈ 1.0 ± 1e-4
        - top_class is the key with maximum probability
        - top_confidence equals predictions[top_class]
        - gradcam_image is None (will be added in Task 3.1)
        - feature_embedding is None (will be added in Task 3.2)
        
    Example:
        from app.agents.vision import run_vision_agent
        
        with open("test_image.jpg", "rb") as f:
            image_bytes = f.read()
        
        state = {"image_bytes": image_bytes, "vision_output": None}
        state = run_vision_agent(state)
        
        print(f"Top class: {state['vision_output']['top_class']}")
        print(f"Confidence: {state['vision_output']['top_confidence']:.3f}")
        print(f"All predictions: {state['vision_output']['predictions']}")
    """
    try:
        # Import model (loaded via vision_loader on app startup)
        from app.models.vision_loader import get_model
        
        logger.info("Starting Vision Agent inference")
        
        # Extract image bytes from state
        image_bytes = state.get("image_bytes")
        if image_bytes is None or len(image_bytes) == 0:
            raise VisionError("No image_bytes provided in state")
        
        logger.debug(f"Processing image: {len(image_bytes)} bytes")
        
        # Step 1: Decode image using PIL
        try:
            img_pil = Image.open(io.BytesIO(image_bytes))
            logger.debug(f"Image decoded: format={img_pil.format}, mode={img_pil.mode}, size={img_pil.size}")
        except Exception as e:
            raise VisionError(f"Failed to decode image: {str(e)}") from e
        
        # Convert to RGB if needed (handles grayscale, RGBA, etc.)
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")
            logger.debug(f"Converted image to RGB mode")
        
        # Step 2: Resize to 150×150×3
        try:
            img_resized = img_pil.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
            logger.debug(f"Image resized to {TARGET_SIZE}")
        except Exception as e:
            raise VisionError(f"Failed to resize image: {str(e)}") from e
        
        # Step 3: Convert to numpy array and cast to float32
        # CRITICAL: Do NOT divide by 255 - model expects raw pixel values [0, 255]
        try:
            img_array = np.array(img_resized, dtype=np.uint8)
            img_float = img_array.astype(np.float32)  # Range: [0, 255]
            
            # Verify pixel range
            min_val = float(img_float.min())
            max_val = float(img_float.max())
            logger.debug(f"Pixel range after conversion: [{min_val:.1f}, {max_val:.1f}]")
            
            if min_val < 0.0 or max_val > 255.0:
                raise VisionError(
                    f"Pixel values out of range [0, 255]: [{min_val}, {max_val}]"
                )
        except VisionError:
            raise
        except Exception as e:
            raise VisionError(f"Failed to convert image to array: {str(e)}") from e
        
        # Step 4: Expand dimensions for batch axis: (150, 150, 3) -> (1, 150, 150, 3)
        img_batch = np.expand_dims(img_float, axis=0)
        logger.debug(f"Image batch shape: {img_batch.shape}, dtype: {img_batch.dtype}")
        
        # Verify batch shape and dtype
        if img_batch.shape != (1, 150, 150, 3):
            raise VisionError(
                f"Unexpected batch shape: {img_batch.shape}, expected (1, 150, 150, 3)"
            )
        if img_batch.dtype != np.float32:
            raise VisionError(
                f"Unexpected batch dtype: {img_batch.dtype}, expected float32"
            )
        
        # Step 5: Get model and run inference
        try:
            model = get_model()
            logger.debug("VGG16 model loaded successfully")
        except Exception as e:
            raise VisionError(f"Failed to load model: {str(e)}") from e
        
        try:
            logger.debug("Running model inference...")
            raw_output = model.predict(img_batch, verbose=0)  # Shape: (1, 4)
            logger.debug(f"Raw model output shape: {raw_output.shape}")
        except Exception as e:
            raise VisionError(f"Model inference failed: {str(e)}") from e
        
        # Step 6: Extract probabilities from batch dimension and verify normalization
        try:
            # raw_output shape is (1, 4), we want the first row (index 0)
            raw_probs = raw_output[0]  # Shape: (4,)
            
            # If output is logits (sum != 1.0), apply softmax to convert to probabilities
            prob_sum = float(np.sum(raw_probs))
            logger.debug(f"Raw output sum: {prob_sum:.6f}")
            
            # Check if we need to apply softmax (if sum is far from 1.0)
            if abs(prob_sum - 1.0) > PROB_TOLERANCE:
                logger.debug("Raw output appears to be logits, applying softmax")
                raw_probs = np.exp(raw_probs) / np.sum(np.exp(raw_probs))
                prob_sum = float(np.sum(raw_probs))
                logger.debug(f"After softmax, sum: {prob_sum:.6f}")
            
            # Verify probabilities sum to approximately 1.0
            if abs(prob_sum - 1.0) > PROB_TOLERANCE:
                logger.warning(
                    f"Probabilities sum to {prob_sum:.6f}, outside tolerance ±{PROB_TOLERANCE}"
                )
            
        except Exception as e:
            raise VisionError(f"Failed to process model output: {str(e)}") from e
        
        # Step 7: Map indices to class names
        try:
            predictions = {
                CLASS_NAMES[i]: float(raw_probs[i])
                for i in range(len(raw_probs))
            }
            logger.debug(f"Predictions: {predictions}")
        except Exception as e:
            raise VisionError(f"Failed to map class indices: {str(e)}") from e
        
        # Step 8: Find top class and confidence
        try:
            # argmax of raw_probs gives the index of highest probability
            top_idx = int(np.argmax(raw_probs))
            top_class = CLASS_NAMES[top_idx]
            top_confidence = float(raw_probs[top_idx])
            
            logger.info(f"Inference complete: top_class={top_class}, confidence={top_confidence:.4f}")
        except Exception as e:
            raise VisionError(f"Failed to determine top class: {str(e)}") from e
        
        # Step 9: Compute Grad-CAM heatmap (Task 3.1)
        try:
            from app.utils.gradcam import compute_gradcam, blend_heatmap_on_image, encode_image_to_base64_png
            
            logger.info("Computing Grad-CAM heatmap...")
            
            # Get model output shape to map class name to index
            class_to_idx = {v: k for k, v in CLASS_NAMES.items()}
            top_class_idx = class_to_idx[top_class]
            
            # Compute Grad-CAM targeting dropout_4 layer
            heatmap = compute_gradcam(
                model=model,
                img_batch=img_batch,
                top_class_idx=top_class_idx,
                target_layer_name="dropout_4"
            )
            logger.debug(f"Grad-CAM heatmap computed: shape={heatmap.shape}")
            
            # Blend heatmap onto original resized image
            blended = blend_heatmap_on_image(
                original_img_array=img_float.astype(np.uint8),
                heatmap=heatmap,
                alpha=0.4
            )
            logger.debug(f"Heatmap blended onto image: shape={blended.shape}")
            
            # Encode to base64 PNG string
            gradcam_b64 = encode_image_to_base64_png(blended)
            logger.info(f"Grad-CAM image encoded to base64: {len(gradcam_b64)} characters")
            
        except Exception as e:
            logger.error(f"Failed to compute Grad-CAM: {str(e)}", exc_info=True)
            # Don't fail the entire vision agent for Grad-CAM error - log and continue
            gradcam_b64 = None
        
        # Step 10: Extract feature embedding from dropout_5 (Task 3.2)
        try:
            from app.utils.gradcam import compute_feature_embedding
            
            logger.info("Extracting feature embedding...")
            
            # Extract 512-dim feature embedding
            feature_embedding = compute_feature_embedding(
                model=model,
                img_batch=img_batch,
                target_layer_name="dropout_5"
            )
            
            # Verify it's exactly 512 elements and not all-zeros
            if len(feature_embedding) != 512:
                raise ValueError(f"Feature embedding has {len(feature_embedding)} elements, expected 512")
            
            # Check if all-zeros
            embedding_array = np.array(feature_embedding)
            if np.allclose(embedding_array, 0.0, atol=1e-6):
                logger.warning("Feature embedding is all-zeros - may indicate model issue")
            
            logger.info(f"Feature embedding extracted: 512 floats")
            
        except Exception as e:
            logger.error(f"Failed to compute feature embedding: {str(e)}", exc_info=True)
            # Don't fail the entire vision agent for embedding error - log and continue
            feature_embedding = None
        
        # Step 11: Create VisionOutput with all fields populated
        vision_output: VisionOutput = {
            "predictions": predictions,
            "top_class": top_class,
            "top_confidence": top_confidence,
            "gradcam_image": gradcam_b64,  # Now populated (Task 3.1)
            "feature_embedding": feature_embedding  # Now populated (Task 3.2)
        }
        
        # Update state with vision output
        state["vision_output"] = vision_output
        logger.info("Vision Agent completed successfully")
        
        return state
        
    except VisionError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in Vision Agent: {str(e)}", exc_info=True)
        raise VisionError(f"Vision Agent failed: {str(e)}") from e
