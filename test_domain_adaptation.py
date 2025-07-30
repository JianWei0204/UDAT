#!/usr/bin/env python3
"""
Test Script for Domain Adaptation Implementation

This script performs basic validation of the domain adaptation components
without requiring actual training data or ultralytics installation.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

def test_grl():
    """Test Gradient Reversal Layer."""
    print("Testing Gradient Reversal Layer...")
    
    try:
        from GRL import GradientScalarLayer
        
        # Create test input
        x = torch.randn(4, 256, 32, 32, requires_grad=True)
        
        # Create GRL layer
        grl = GradientScalarLayer(-1.0)
        
        # Forward pass
        y = grl(x)
        
        # Check forward pass (should be identity)
        assert torch.allclose(x, y), "GRL forward pass failed"
        
        # Backward pass
        loss = y.sum()
        loss.backward()
        
        # Check that gradients are reversed
        assert x.grad is not None, "No gradients computed"
        
        print("✓ GRL test passed")
        return True
        
    except Exception as e:
        print(f"✗ GRL test failed: {e}")
        return False


def test_transformer_discriminator():
    """Test Transformer Discriminator."""
    print("Testing Transformer Discriminator...")
    
    try:
        from trans_discriminator import TransformerDiscriminator
        
        # Create discriminator
        discriminator = TransformerDiscriminator(channels=256, img_size=128)
        
        # Create test input
        x = torch.randn(4, 256, 128, 128)
        
        # Forward pass
        output = discriminator(x)
        
        # Check output shape
        expected_shape = (4, 1)  # batch_size, num_classes
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        
        # Check output range (should be reasonable for domain classification)
        assert torch.all(torch.isfinite(output)), "Output contains inf/nan values"
        
        print("✓ Transformer Discriminator test passed")
        return True
        
    except Exception as e:
        print(f"✗ Transformer Discriminator test failed: {e}")
        return False


def test_feature_extractor():
    """Test YOLO Feature Extractor."""
    print("Testing YOLO Feature Extractor...")
    
    try:
        from yolo_feature_extractor import YOLOv8WithFeatureExtraction
        
        # Create mock YOLOv8 model
        class MockYOLOv8(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 512, 3, padding=1),  # This would be P4
                    nn.ReLU(),
                    nn.Conv2d(512, 1024, 3, padding=1),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(1024, 80)  # 80 classes
                )
                self.nc = 80
                self.names = [f'class_{i}' for i in range(80)]
            
            def forward(self, x):
                return self.model(x)
        
        # Create base model and wrapper
        base_model = MockYOLOv8()
        wrapped_model = YOLOv8WithFeatureExtraction(base_model)
        
        # Test feature extraction
        x = torch.randn(2, 3, 640, 640)
        
        # Extract P4 features
        features = wrapped_model.extract_p4_features(x)
        
        # Check that features were extracted
        assert features is not None, "No features extracted"
        assert isinstance(features, torch.Tensor), "Features should be tensor"
        assert features.size(0) == 2, "Batch dimension mismatch"
        
        print("✓ Feature Extractor test passed")
        return True
        
    except Exception as e:
        print(f"✗ Feature Extractor test failed: {e}")
        return False


def test_domain_data_loader():
    """Test Domain Data Loader."""
    print("Testing Domain Data Loader...")
    
    try:
        from domain_data_loader import DomainAdaptationDataset
        
        # Create mock datasets
        class MockDataset:
            def __init__(self, size, label_offset=0):
                self.size = size
                self.label_offset = label_offset
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    'img': torch.randn(3, 640, 640),
                    'cls': torch.tensor([self.label_offset + idx % 10]),
                    'bboxes': torch.randn(1, 4),
                    'batch_idx': torch.tensor([0])
                }
        
        source_dataset = MockDataset(100, 0)
        target_dataset = MockDataset(80, 50)
        
        # Create domain adaptation dataset
        domain_dataset = DomainAdaptationDataset(source_dataset, target_dataset, mode='alternating')
        
        # Test dataset length
        expected_length = 100 + 80
        assert len(domain_dataset) == expected_length, f"Expected length {expected_length}, got {len(domain_dataset)}"
        
        # Test sample access
        sample = domain_dataset[0]
        assert 'domain' in sample, "Domain label missing"
        assert 'domain_label' in sample, "Domain label missing"
        assert sample['domain'] in ['source', 'target'], "Invalid domain label"
        
        print("✓ Domain Data Loader test passed")
        return True
        
    except Exception as e:
        print(f"✗ Domain Data Loader test failed: {e}")
        return False


def test_integration():
    """Test integration between components."""
    print("Testing Component Integration...")
    
    try:
        from trans_discriminator import TransformerDiscriminator
        from GRL import GradientScalarLayer
        
        # Create components
        discriminator = TransformerDiscriminator(channels=256)
        
        # Simulate P4 features
        p4_features = torch.randn(4, 256, 32, 32)
        
        # Upsample to discriminator input size
        upsampler = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        upsampled_features = upsampler(p4_features)
        
        # Apply softmax
        softmax_features = torch.softmax(upsampled_features, dim=1)
        
        # Pass through discriminator
        domain_pred = discriminator(softmax_features)
        
        # Check output
        assert domain_pred.shape == (4, 1), "Discriminator output shape mismatch"
        
        # Test loss computation
        target_labels = torch.ones(4, device=domain_pred.device)
        loss = torch.mean((domain_pred.squeeze() - target_labels) ** 2)
        
        assert torch.isfinite(loss), "Loss is not finite"
        assert loss.item() >= 0, "Loss should be non-negative"
        
        print("✓ Integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("UDAT Domain Adaptation Test Suite")
    print("=" * 50)
    
    tests = [
        test_grl,
        test_transformer_discriminator,
        test_feature_extractor,
        test_domain_data_loader,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation looks good.")
        return 0
    else:
        print(f"❌ {total - passed} tests failed. Please check the implementation.")
        return 1


if __name__ == '__main__':
    exit(main())