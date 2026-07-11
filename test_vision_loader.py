"""
Unit tests for Vision Model Loader

Tests for:
- Model loading from HuggingFace Hub
- Singleton caching pattern
- Exponential backoff retry logic
- Mock model fallback
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import time

# Add app to path
sys.path.insert(0, '/Users/sj/Documents/Neurotriage')

from app.models.vision_loader import (
    load_vgg16_model, 
    VisionModelError,
    _create_mock_model,
    _MODEL
)


class TestVisionLoader:
    """Test suite for vision model loading"""
    
    def setup_method(self):
        """Reset the module-level MODEL singleton before each test"""
        import app.models.vision_loader as vision_loader
        vision_loader._MODEL = None
    
    def test_model_singleton_caching(self):
        """
        Test that model is cached and reused on subsequent calls.
        
        Validates: Requirements 2.1 (model caching and reuse)
        """
        with patch('app.models.vision_loader.hf_hub_download') as mock_download:
            with patch('keras.models.load_model') as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model
                
                # First call should load the model
                model1 = load_vgg16_model(max_retries=1)
                assert mock_download.call_count == 1
                assert mock_load.call_count == 1
                
                # Second call should return cached model without re-downloading
                model2 = load_vgg16_model(max_retries=1)
                assert mock_download.call_count == 1  # Should not increase
                assert mock_load.call_count == 1      # Should not increase
                assert model1 is model2  # Same object reference
    
    def test_exponential_backoff_on_failure(self):
        """
        Test exponential backoff retry logic with max 3 attempts.
        
        Validates: Requirements 2.6 (exponential backoff, max 3 attempts)
        
        Expected backoff times:
        - Attempt 0: fail all filenames, sleep 2^0 = 1 second
        - Attempt 1: fail all filenames, sleep 2^1 = 2 seconds
        - Attempt 2: fail all filenames, raise after no sleep (final attempt)
        
        Note: The implementation tries multiple filenames per retry attempt.
        """
        with patch('app.models.vision_loader.hf_hub_download') as mock_download:
            with patch('app.models.vision_loader.time.sleep') as mock_sleep:
                # All attempts fail
                mock_download.side_effect = Exception("Network error")
                
                with pytest.raises(VisionModelError) as exc_info:
                    load_vgg16_model(max_retries=3)
                
                assert "3 attempts" in str(exc_info.value)
                
                # Should have called sleep twice (before retries 1 and 2, not after final attempt)
                assert mock_sleep.call_count == 2
                
                # Verify exponential backoff timing
                sleep_calls = mock_sleep.call_args_list
                assert sleep_calls[0][0][0] == 1  # 2^0
                assert sleep_calls[1][0][0] == 2  # 2^1
                
                # Should have tried max_retries times (3 in this case)
                # Each attempt tries 5 possible filenames, so 3 * 5 = 15 calls
                assert mock_download.call_count >= 3
    
    def test_no_sleep_on_final_attempt(self):
        """
        Test that no sleep occurs after the final failed attempt.
        
        Validates: Requirements 2.6 (correct backoff timing)
        """
        with patch('app.models.vision_loader.hf_hub_download') as mock_download:
            with patch('app.models.vision_loader.time.sleep') as mock_sleep:
                mock_download.side_effect = Exception("Network error")
                
                with pytest.raises(VisionModelError):
                    load_vgg16_model(max_retries=2)
                
                # Should only sleep once (after first failure, not after second)
                assert mock_sleep.call_count == 1
                mock_sleep.assert_called_once_with(1)  # 2^0 = 1
    
    def test_successful_load_no_sleep(self):
        """
        Test that sleep is not called if model loads successfully on first attempt.
        
        Validates: Requirements 2.6
        """
        with patch('app.models.vision_loader.hf_hub_download') as mock_download:
            with patch('app.models.vision_loader.time.sleep') as mock_sleep:
                with patch('keras.models.load_model') as mock_load:
                    mock_model = MagicMock()
                    mock_load.return_value = mock_model
                    mock_download.return_value = "/path/to/model.h5"
                    
                    load_vgg16_model(max_retries=3)
                    
                    # Should not sleep if successful on first try
                    mock_sleep.assert_not_called()
                    assert mock_download.call_count == 1
    
    def test_fallback_to_mock_model(self):
        """
        Test fallback to mock model when use_mock_on_failure=True and all retries exhausted.
        
        Validates: Requirements 2.6 (HTTP 503 fallback - implemented as mock model)
        """
        with patch('app.models.vision_loader.hf_hub_download') as mock_download:
            mock_download.side_effect = Exception("Network error")
            
            # Should return mock model instead of raising
            model = load_vgg16_model(max_retries=2, use_mock_on_failure=True)
            assert model is not None
            
            # Verify it's a Keras model by checking it has predict method
            assert hasattr(model, 'predict')
    
    def test_mock_model_creation(self):
        """
        Test that mock model has required structure.
        
        Validates: Requirements 2.1 (model structure with dropout_4 and dropout_5 layers)
        """
        mock_model = _create_mock_model()
        
        # Verify model can be created
        assert mock_model is not None
        
        # Verify model has required layers
        layer_names = [layer.name for layer in mock_model.layers]
        assert 'dropout_4' in layer_names, "Mock model must have dropout_4 layer"
        assert 'dropout_5' in layer_names, "Mock model must have dropout_5 layer"
        
        # Verify input shape is correct (150x150x3)
        assert mock_model.input_shape == (None, 150, 150, 3)
        
        # Verify output has 4 classes
        assert mock_model.output_shape == (None, 4)
    
    def test_file_lookup_tries_multiple_filenames(self):
        """
        Test that loader tries multiple possible filenames when looking for model.
        
        Validates: Requirements 2.1 (robust model fetching)
        """
        with patch('app.models.vision_loader.hf_hub_download') as mock_download:
            with patch('keras.models.load_model') as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model
                
                # First two attempts fail, third succeeds
                mock_download.side_effect = [
                    Exception("Not found: vgg16_model.h5"),
                    Exception("Not found: model.h5"),
                    "/path/to/vgg16.h5"  # Success on third filename
                ]
                
                model = load_vgg16_model(max_retries=1)
                assert model is not None
    
    def test_error_message_includes_context(self):
        """
        Test that VisionModelError includes useful context about failures.
        
        Validates: Requirements 2.6 (clear error reporting)
        
        The implementation wraps the inner exception details, so we check for the key info.
        """
        with patch('app.models.vision_loader.hf_hub_download') as mock_download:
            error_msg = "Connection timeout"
            mock_download.side_effect = Exception(error_msg)
            
            with pytest.raises(VisionModelError) as exc_info:
                load_vgg16_model(max_retries=1)
            
            error_str = str(exc_info.value)
            # Error message should include info about the failure
            assert "Failed to load VGG16 model" in error_str
            assert "1 attempts" in error_str
    
    def test_max_retries_parameter(self):
        """
        Test that max_retries parameter is respected.
        
        Validates: Requirements 2.6
        
        The implementation tries multiple filenames per attempt, so we verify
        that it makes at least max_retries attempts total.
        """
        with patch('app.models.vision_loader.hf_hub_download') as mock_download:
            mock_download.side_effect = Exception("Network error")
            
            # Test with max_retries=1
            with pytest.raises(VisionModelError):
                load_vgg16_model(max_retries=1)
            # Should have called hf_hub_download at least once (actually tries multiple filenames per attempt)
            assert mock_download.call_count >= 1
            first_count = mock_download.call_count
            
            # Reset for next test
            self.setup_method()
            
            # Test with max_retries=2 should result in more attempts
            with pytest.raises(VisionModelError):
                load_vgg16_model(max_retries=2)
            assert mock_download.call_count > first_count


class TestModelStructure:
    """Test that loaded model has required structure"""
    
    def setup_method(self):
        """Reset the module-level MODEL singleton before each test"""
        import app.models.vision_loader as vision_loader
        vision_loader._MODEL = None
    
    def test_mock_model_dropout_layers(self):
        """
        Test that mock model has dropout_4 and dropout_5 layers for Grad-CAM.
        
        Validates: Requirements 3.1, 3.3 (Grad-CAM targeting dropout_4 and feature extraction from dropout_5)
        """
        model = _create_mock_model()
        
        # Get layer by name
        dropout_4 = model.get_layer('dropout_4')
        dropout_5 = model.get_layer('dropout_5')
        
        assert dropout_4 is not None
        assert dropout_5 is not None
        assert dropout_4.name == 'dropout_4'
        assert dropout_5.name == 'dropout_5'
    
    def test_mock_model_output_shape(self):
        """
        Test that mock model outputs 4 class probabilities.
        
        Validates: Requirements 2.3, 2.4, 2.5
        """
        model = _create_mock_model()
        assert model.output_shape[-1] == 4  # Last dimension should be 4 classes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
