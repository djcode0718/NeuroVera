"""
Grad-CAM Computation Module

Implements Grad-CAM (Gradient-weighted Class Activation Mapping) for producing
visual explanations of CNN model predictions. Also extracts feature embeddings
from intermediate layers.

Design specification (from design.md):
- Compute Grad-CAM using tf.GradientTape targeting dropout_4 layer
- Compute heatmap by multiplying conv output by pooled gradients
- Normalize heatmap to [0, 255] range
- Blend onto original resized image with alpha=0.4
- Return as base64-encoded PNG string
- Extract 512-dim feature embedding from dropout_5 layer

References:
- Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization"
- Implementation pattern from: https://keras.io/examples/vision/grad_cam/
"""

import logging
import base64
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Tuple

logger = logging.getLogger(__name__)


def compute_gradcam(
    model: tf.keras.Model,
    img_batch: np.ndarray,
    top_class_idx: int,
    target_layer_name: str = "dropout_4"
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a given image and target class.
    
    This function implements the Grad-CAM algorithm:
    1. Create a sub-model with target_layer output as intermediate output
    2. Use tf.GradientTape to compute gradients of class logit w.r.t. target layer
    3. Multiply conv output by pooled gradients to get importance map
    4. Normalize to [0, 255] range for visualization
    
    Args:
        model: Keras model with layers including target_layer_name
        img_batch: Input image batch, shape (1, 150, 150, 3), dtype float32, range [0, 255]
        top_class_idx: Index of target class for gradient computation (0-3 for 4 classes)
        target_layer_name: Name of target layer (default "dropout_4")
        
    Returns:
        Heatmap as numpy array, shape (H, W), dtype uint8, range [0, 255]
        
    Raises:
        RuntimeError: If target layer not found or gradients cannot be computed
        ValueError: If input shapes/types invalid
        
    Preconditions:
        - model contains a layer named target_layer_name
        - img_batch has shape (1, 150, 150, 3) and dtype float32
        - img_batch pixel values in range [0, 255]
        - top_class_idx in range [0, 3] for 4-class model
        
    Postconditions:
        - Returned heatmap has shape (H, W) where H, W are spatial dims of target layer
        - Heatmap values in range [0, 255] (uint8)
        - Heatmap is single-channel (grayscale), ready for conversion to RGB for blending
        
    Algorithm:
        1. Build grad_model with target_layer output as intermediate
        2. Inside GradientTape context:
           a. Pass img_batch through grad_model -> (conv_outputs, model_output)
           b. Extract logits for top_class_idx: loss = model_output[:, top_class_idx]
        3. Compute gradients: grads = tape.gradient(loss, conv_outputs)
        4. Pool gradients: pooled_grads = mean(grads, axes=[spatial + batch])
        5. Weight conv outputs: heatmap = mean(conv_outputs * pooled_grads, axis=-1)
        6. Apply ReLU to remove negative values
        7. Normalize to [0, 255]: heatmap = (heatmap / max(heatmap)) * 255
        8. Convert to uint8 and return
    """
    try:
        logger.debug(f"Computing Grad-CAM for class {top_class_idx} using layer '{target_layer_name}'")

        # Rebuild the functional graph explicitly. Sequential models loaded via
        # load_model() in Keras 3 don't retain call-history metadata, so
        # target_layer.output raises "layer has never been called". Re-calling
        # each layer on a fresh symbolic input rebuilds a proper graph.
        inputs = tf.keras.Input(shape=model.input_shape[1:])
        x = inputs
        target_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == target_layer_name:
                target_output = x

        if target_output is None:
            raise RuntimeError(f"Layer '{target_layer_name}' not found in model")

        grad_model = tf.keras.Model(inputs=inputs, outputs=[target_output, x])
        
        # Compute gradients using GradientTape
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch)
            # Loss is the logit for the target class
            loss = predictions[:, top_class_idx]
        
        # Get gradients of loss w.r.t. conv_outputs
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            raise RuntimeError("Failed to compute gradients - target layer may not be differentiable")
        
        logger.debug(f"Gradients shape: {grads.shape}, conv_outputs shape: {conv_outputs.shape}")
        
        # Determine if target layer output is convolutional (4D) or dense (2D)
        # For 4D output (conv layers), reduce across spatial dims (1, 2) and batch (0)
        # For 2D output (dense layers), only reduce across batch (0)
        output_ndims = len(conv_outputs.shape)
        
        if output_ndims == 4:
            # Convolutional layer output: (batch, height, width, channels)
            pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])  # Shape: (channels,)
            logger.debug(f"Target layer is 4D (conv), pooling over batch, height, width")
        elif output_ndims == 2:
            # Dense layer output: (batch, features)
            # For dense layers, we use a simpler approach: just use the gradient as importance
            pooled_grads = tf.reduce_mean(grads, axis=0)  # Shape: (features,)
            logger.debug(f"Target layer is 2D (dense), pooling over batch only")
        else:
            raise RuntimeError(f"Unexpected output dimensionality: {output_ndims}")
        
        logger.debug(f"Pooled gradients shape: {pooled_grads.shape}")
        
        # Weight the conv/dense outputs by pooled gradients
        conv_outputs_first = conv_outputs[0]  # Remove batch dimension
        weighted = conv_outputs_first * pooled_grads  # Broadcast multiply
        
        # Reduce to get importance map
        if output_ndims == 4:
            # For conv outputs, reduce across channels
            heatmap = tf.reduce_mean(weighted, axis=-1)  # Shape: (height, width)
        else:
            # For dense outputs, the weighted output IS the importance (1D)
            # Reshape it to a 2D spatial map for visualization
            # We'll create a simple heatmap by repeating the values
            heatmap = weighted  # Shape: (features,)
            # Reshape to 8x8 spatial map for visualization
            features = heatmap.shape[0]
            spatial_size = int(np.sqrt(features))
            if spatial_size * spatial_size < features:
                spatial_size += 1
            heatmap_reshaped = tf.concat(
                [heatmap, tf.zeros(spatial_size * spatial_size - features)], 
                axis=0
            )
            heatmap = tf.reshape(heatmap_reshaped, (spatial_size, spatial_size))
            logger.debug(f"Dense layer heatmap reshaped to {heatmap.shape}")
        
        # Apply ReLU to remove negative values
        heatmap = tf.nn.relu(heatmap)
        
        # Normalize to [0, 1]
        heatmap_max = tf.reduce_max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        else:
            logger.warning("Heatmap is all zeros - model may not be discriminative for this class")
        
        # Convert to uint8 [0, 255]
        heatmap_uint8 = tf.cast(heatmap * 255, tf.uint8).numpy()
        
        logger.debug(f"Heatmap computed: shape={heatmap_uint8.shape}, min={heatmap_uint8.min()}, max={heatmap_uint8.max()}")
        
        return heatmap_uint8
        
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Failed to compute Grad-CAM: {str(e)}", exc_info=True)
        raise RuntimeError(f"Grad-CAM computation failed: {str(e)}") from e


def blend_heatmap_on_image(
    original_img_array: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Blend a grayscale heatmap onto an RGB image using alpha transparency.
    
    The blending formula is:
        output = original * (1 - alpha) + heatmap_rgb * alpha
    
    This creates a visual overlay where the heatmap highlights important regions
    while preserving visibility of the underlying MRI scan.
    
    Args:
        original_img_array: Original RGB image, shape (H, W, 3), dtype uint8, range [0, 255]
        heatmap: Grayscale heatmap, shape (H_h, W_h), dtype uint8, range [0, 255]
        alpha: Blending factor, range [0, 1]. alpha=0 shows only original, alpha=1 shows only heatmap.
               Default 0.4 emphasizes original image while showing heatmap.
        
    Returns:
        Blended image as RGB numpy array, shape (H, W, 3), dtype uint8, range [0, 255]
        
    Raises:
        ValueError: If input shapes/types invalid
        
    Preconditions:
        - original_img_array has shape (H, W, 3) and dtype uint8
        - heatmap has shape (H', W') and dtype uint8 or float32
        - alpha in range [0, 1]
        - heatmap and original_img_array have compatible spatial dimensions (will be resized)
        
    Postconditions:
        - Returned image has same shape as original_img_array: (H, W, 3)
        - Returned image has dtype uint8 with values in [0, 255]
        - Output preserves original image appearance when alpha is small
        
    Algorithm:
        1. Resize heatmap to match original image size if needed
        2. Convert single-channel heatmap to 3-channel (replicate across RGB)
        3. Convert to float32 for blending calculations (to avoid overflow)
        4. Compute blended = original * (1 - alpha) + heatmap_rgb * alpha
        5. Clip to [0, 255] to handle any rounding issues
        6. Convert back to uint8
        7. Return blended image
    """
    try:
        logger.debug(f"Blending heatmap onto image: original_shape={original_img_array.shape}, heatmap_shape={heatmap.shape}, alpha={alpha}")
        
        # Validate alpha
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        
        # Validate original image
        if original_img_array.shape[2] != 3:
            raise ValueError(f"Original image must be RGB (3 channels), got shape {original_img_array.shape}")
        
        # Resize heatmap if needed to match original spatial dimensions
        original_shape = original_img_array.shape[:2]  # (H, W)
        if heatmap.shape != original_shape:
            logger.debug(f"Resizing heatmap from {heatmap.shape} to {original_shape}")
            # Convert heatmap to PIL, resize, convert back
            heatmap_pil = Image.fromarray(heatmap, mode='L')
            heatmap_pil = heatmap_pil.resize(original_shape[::-1], Image.Resampling.BILINEAR)  # PIL uses (W, H)
            heatmap = np.array(heatmap_pil)
        
        # Convert heatmap to RGB by replicating across channels
        # This makes the grayscale heatmap work with RGB blending
        heatmap_rgb = np.stack([heatmap, heatmap, heatmap], axis=-1)  # Shape: (H, W, 3)
        
        # Convert to float32 for blending (avoid uint8 overflow)
        original_float = original_img_array.astype(np.float32)
        heatmap_float = heatmap_rgb.astype(np.float32)
        
        # Apply alpha blending formula
        blended = original_float * (1 - alpha) + heatmap_float * alpha
        
        # Clip to valid range and convert to uint8
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        logger.debug(f"Blended image shape: {blended.shape}, dtype: {blended.dtype}, value range: [{blended.min()}, {blended.max()}]")
        
        return blended
        
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to blend heatmap: {str(e)}", exc_info=True)
        raise ValueError(f"Heatmap blending failed: {str(e)}") from e


def encode_image_to_base64_png(image_array: np.ndarray) -> str:
    """
    Encode an RGB image array to a base64-encoded PNG string.
    
    This is used to embed the Grad-CAM overlay image in the VisionOutput JSON
    response, allowing the frontend to display it directly without additional
    HTTP requests.
    
    Args:
        image_array: RGB image as numpy array, shape (H, W, 3), dtype uint8
        
    Returns:
        Base64-encoded PNG string, can be used directly in HTML as:
            <img src="data:image/png;base64,{returned_string}" />
        
    Raises:
        ValueError: If input shape/type invalid
        
    Preconditions:
        - image_array has shape (H, W, 3) and dtype uint8
        - image_array pixel values in [0, 255]
        
    Postconditions:
        - Returned string starts with valid PNG magic bytes (encoded)
        - Returned string is non-empty and valid base64
        
    Algorithm:
        1. Convert numpy array to PIL Image (RGB mode)
        2. Encode to PNG format in memory (BytesIO)
        3. Read PNG bytes and encode to base64
        4. Return as string
    """
    try:
        logger.debug(f"Encoding image to base64 PNG: shape={image_array.shape}, dtype={image_array.dtype}")
        
        # Validate input
        if image_array.shape[2] != 3:
            raise ValueError(f"Image must be RGB (3 channels), got shape {image_array.shape}")
        if image_array.dtype != np.uint8:
            raise ValueError(f"Image must be uint8, got {image_array.dtype}")
        
        # Convert to PIL Image
        img_pil = Image.fromarray(image_array, mode='RGB')
        
        # Encode to PNG in memory
        png_buffer = io.BytesIO()
        img_pil.save(png_buffer, format='PNG')
        png_bytes = png_buffer.getvalue()
        
        # Encode to base64
        b64_string = base64.b64encode(png_bytes).decode('utf-8')
        
        logger.debug(f"Image encoded to base64 PNG: {len(b64_string)} characters")
        
        return b64_string
        
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to encode image to base64 PNG: {str(e)}", exc_info=True)
        raise ValueError(f"Image encoding failed: {str(e)}") from e


def compute_feature_embedding(
    model: tf.keras.Model,
    img_batch: np.ndarray,
    target_layer_name: str = "dropout_5"
) -> list[float]:
    """
    Extract a 512-dimensional feature embedding from an intermediate layer.
    
    This embedding is used for:
    1. Storing in the case bank for similarity-based retrieval
    2. Computing cosine similarity with stored embeddings to find similar cases
    
    The embedding captures high-level semantic features learned by the CNN,
    which are more suitable for similarity comparison than raw pixel values.
    
    Args:
        model: Keras model with layers including target_layer_name
        img_batch: Input image batch, shape (1, 150, 150, 3), dtype float32, range [0, 255]
        target_layer_name: Name of target layer (default "dropout_5")
        
    Returns:
        Feature embedding as list of exactly 512 floats
        
    Raises:
        RuntimeError: If target layer not found or embedding extraction fails
        ValueError: If returned embedding has wrong dimension
        
    Preconditions:
        - model contains a layer that outputs 512 dimensions
        - img_batch has shape (1, 150, 150, 3) and dtype float32
        - img_batch pixel values in range [0, 255]
        
    Postconditions:
        - Returned list has exactly 512 elements
        - Each element is a float
        - Embedding is not all-zeros (indicates meaningful feature extraction)
        
    Algorithm:
        1. Try to get target layer from model
        2. If output dim is not 512, search for nearest 512-dim layer
        3. Create sub-model with 512-dim layer output
        4. Call model.predict(img_batch) to get activations
        5. Extract first row of batch (shape: (512,))
        6. Verify not all-zeros
        7. Convert to Python list of floats
        8. Return
    """
    try:
        logger.debug(f"Extracting feature embedding, target layer name: '{target_layer_name}'")
        
        # Try to get the target layer first
        target_layer = model.get_layer(target_layer_name)
        embedding_layer = None
        
        if target_layer is not None:
            try:
                if hasattr(target_layer, 'output_shape'):
                    output_shape = target_layer.output_shape
                elif hasattr(target_layer, 'output') and hasattr(target_layer.output, 'shape'):
                    output_shape = target_layer.output.shape
                else:
                    output_shape = None
                
                if output_shape is not None:
                    if isinstance(output_shape, (list, tuple)):
                        layer_dim = output_shape[-1]
                    else:
                        layer_dim = 1
                    
                    logger.debug(f"Target layer '{target_layer_name}' output dimension: {layer_dim}")
                    
                    if layer_dim == 512:
                        embedding_layer = target_layer
                else:
                    logger.warning(f"Could not determine output shape of target layer '{target_layer_name}'")
            except Exception as e:
                logger.warning(f"Error accessing target layer: {e}")
        
        if target_layer is None or embedding_layer is None:
            logger.warning(f"Target layer '{target_layer_name}' not usable. Searching for 512-dim layer...")
        
        # If we didn't find a 512-dim layer at target, search the model
        if embedding_layer is None:
            for layer in reversed(model.layers):
                try:
                    if hasattr(layer, 'output_shape'):
                        output_shape = layer.output_shape
                    elif hasattr(layer, 'output') and hasattr(layer.output, 'shape'):
                        output_shape = layer.output.shape
                    else:
                        continue
                    
                    if isinstance(output_shape, (list, tuple)) and len(output_shape) > 0:
                        layer_dim = output_shape[-1]
                    else:
                        continue
                    
                    if layer_dim == 512:
                        logger.info(f"Found 512-dim layer: {layer.name}")
                        embedding_layer = layer
                        break
                except Exception as e:
                    logger.debug(f"Skipping layer {layer.name}: {str(e)}")
                    continue
        
        if embedding_layer is None:
            raise RuntimeError("Could not find a layer with 512-dimension output in the model")
        
        logger.debug(f"Using layer '{embedding_layer.name}' for feature embedding")
        
        # Create a model that outputs the embedding layer
        embed_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=embedding_layer.output
        )
        
        # Extract embedding
        embedding_batch = embed_model.predict(img_batch, verbose=0)
        
        # Get the first (and only) row
        embedding = embedding_batch[0]
        
        logger.debug(f"Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
        
        # Extract dimension
        embedding_dim = embedding.shape[0] if len(embedding.shape) > 0 else 1
        
        if embedding_dim != 512:
            raise ValueError(f"Expected embedding dimension 512, got {embedding_dim}")
        
        # Check if all-zeros (indicates potential issue)
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm < 1e-6:
            logger.warning("Embedding is all-zeros or near-zero - model may not be producing meaningful features")
        else:
            logger.debug(f"Embedding norm: {embedding_norm:.6f}")
        
        # Convert to Python list of floats
        embedding_list = embedding.astype(np.float32).tolist()
        
        # Final verification
        if len(embedding_list) != 512:
            raise ValueError(f"Embedding list has {len(embedding_list)} elements, expected 512")
        
        logger.debug(f"Feature embedding extracted: 512 floats, norm={embedding_norm:.6f}")
        
        return embedding_list
        
    except (RuntimeError, ValueError):
        raise
    except Exception as e:
        logger.error(f"Failed to compute feature embedding: {str(e)}", exc_info=True)
        raise RuntimeError(f"Feature embedding extraction failed: {str(e)}") from e
