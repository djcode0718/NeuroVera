"""
Vision Model Loader Module

Handles loading the pretrained VGG16 model from HuggingFace Hub with exponential
backoff retry logic and singleton caching pattern.

CRITICAL INVARIANT: Model input expects raw 0-255 pixel values, NOT normalized.
This is enforced at the preprocessing layer within the VGG16 model itself.
"""

import time
import logging
import os
from typing import Optional

import keras
from huggingface_hub import hf_hub_download

# Module-level singleton to cache loaded model
_MODEL: Optional[keras.Model] = None

logger = logging.getLogger(__name__)


class VisionModelError(Exception):
    """Raised when model cannot be loaded after retries."""
    pass


def _create_mock_model() -> keras.Model:
    """
    Create a mock VGG16-like model for development/testing when real model unavailable.
    
    This is ONLY used as a fallback when network unavailable or for testing.
    Production deployments must use the real pretrained model.
    """
    from keras import layers
    
    # Create a simple model that mimics VGG16 structure with dropout_4 and dropout_5
    inputs = keras.Input(shape=(150, 150, 3))
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5, name='dropout_4')(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5, name='dropout_5')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(4, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=x)
    logger.warning("Using mock VGG16 model for development/testing. This is not the pretrained model!")
    return model


def get_model(max_retries: int = 3, use_mock_on_failure: bool = False) -> keras.Model:
    """
    Public API to get the cached VGG16 model. Alias for load_vgg16_model.
    """
    return load_vgg16_model(max_retries=max_retries, use_mock_on_failure=use_mock_on_failure)


def load_vgg16_model(max_retries: int = 3, use_mock_on_failure: bool = False) -> keras.Model:
    """
    Load the VGG16 pretrained model from HuggingFace Hub with exponential backoff.
    
    The model is cached in a module-level singleton and reused across requests.
    On repeated calls, returns the cached model without re-downloading.
    
    Args:
        max_retries: Maximum number of retry attempts (default 3)
        use_mock_on_failure: If True and all retries fail, return a mock model instead of raising
        
    Returns:
        Loaded keras.Model ready for inference
        
    Raises:
        VisionModelError: If model cannot be loaded after all retry attempts
        
    Preconditions:
        - max_retries >= 1
        
    Postconditions:
        - Returned model has layers named "dropout_4" and "dropout_5"
        - Returned model expects input shape (batch_size, 150, 150, 3) with values in [0, 255]
        - Model is cached globally and subsequent calls return same instance
    """
    global _MODEL
    
    # Return cached model if already loaded
    if _MODEL is not None:
        logger.info("Returning cached VGG16 model")
        return _MODEL
    
    model_id = "AyanKantiDas/BrainTumorVGG16"
    last_exception = None
    
    # Possible filenames to try
    possible_filenames = [
        "vgg16_model.h5",
        "model.h5",
        "vgg16.h5",
        "pytorch_model.bin",
        "model.keras",
    ]
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to load model from HuggingFace Hub (attempt {attempt + 1}/{max_retries})")
            
            # Try each possible filename
            model_filepath = None
            for filename in possible_filenames:
                try:
                    logger.debug(f"Trying to download {filename} from {model_id}")
                    model_filepath = hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        timeout=30
                    )
                    logger.info(f"Successfully downloaded {filename}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to find {filename}: {str(e)}")
                    continue
            
            if model_filepath is None:
                raise VisionModelError(
                    f"Could not find any model file in {model_id}. "
                    f"Tried: {', '.join(possible_filenames)}"
                )
            
            # Load the model
            _MODEL = keras.models.load_model(model_filepath)
            logger.info(f"Successfully loaded VGG16 model from {model_id}")
            return _MODEL
            
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            
            # Only sleep if this wasn't the last attempt
            if attempt < max_retries - 1:
                backoff_seconds = 2 ** attempt
                logger.info(f"Exponential backoff: sleeping {backoff_seconds} seconds")
                time.sleep(backoff_seconds)
    
    # All retries exhausted
    if use_mock_on_failure:
        logger.warning(
            f"Failed to load real VGG16 model after {max_retries} attempts. "
            f"Returning mock model for development. Last error: {str(last_exception)}"
        )
        _MODEL = _create_mock_model()
        return _MODEL
    
    error_msg = (
        f"Failed to load VGG16 model after {max_retries} attempts. "
        f"Last error: {str(last_exception)}"
    )
    logger.error(error_msg)
    raise VisionModelError(error_msg)
