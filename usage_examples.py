#!/usr/bin/env python3
"""
Usage Example for YOLOv8 Domain Adaptation

This script shows how to use the domain adaptation implementation
for training YOLOv8 with source and target domain data.
"""

import torch
from yolov8_domain_model import DomainAdaptiveYOLOv8
from train_yolov8_simple import train_domain_adaptive_yolo


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===\n")
    
    # Initialize the domain adaptive model
    model = DomainAdaptiveYOLOv8(nc=80, translayer_channels=256)
    
    # Example input
    x = torch.randn(1, 3, 640, 640)
    
    # Forward pass to get both detections and P4 features
    detections, p4_features = model.forward_with_domain_features(x)
    
    print(f"Input shape: {x.shape}")
    print(f"P4 features for domain adaptation: {p4_features.shape}")
    print(f"Number of detection scales: {len(detections)}")
    
    # Test domain discriminator
    upsampler = torch.nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
    p4_upsampled = upsampler(p4_features)
    domain_pred = model.discriminator(torch.nn.functional.softmax(p4_upsampled, dim=1))
    print(f"Domain prediction: {domain_pred.shape} (0=source, 1=target)")


def example_training_setup():
    """Example of how to set up training."""
    print("\n=== Training Setup Example ===\n")
    
    # Training configuration
    config = {
        'epochs': 100,
        'batch_size': 16,
        'lr': 0.001,          # Main network learning rate
        'lr_d': 0.0001,       # Discriminator learning rate (10x lower)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './runs/domain_adapt_training'
    }
    
    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nTo start training:")
    print("1. Prepare your datasets:")
    print("   - Source domain: Labeled detection data (e.g., COCO format)")
    print("   - Target domain: Unlabeled real-world images")
    print("\n2. Modify dataset paths in train_yolov8_simple.py")
    print("\n3. Run training:")
    print("   python train_yolov8_simple.py")


def example_custom_dataset():
    """Example of how to create custom dataset loaders."""
    print("\n=== Custom Dataset Example ===\n")
    
    print("To use your own datasets, modify the DummyDataset class in train_yolov8_simple.py:")
    print()
    print("```python")
    print("class CustomDataset(Dataset):")
    print("    def __init__(self, data_path, domain='source', img_size=640):")
    print("        self.data_path = data_path")
    print("        self.domain = domain")
    print("        self.img_size = img_size")
    print("        # Load your image paths and annotations")
    print("        self.images = self.load_images()")
    print("        self.labels = self.load_labels() if domain == 'source' else None")
    print("    ")
    print("    def __getitem__(self, idx):")
    print("        # Load and preprocess image")
    print("        img = self.load_image(self.images[idx])")
    print("        ")
    print("        # Load labels for source domain only")
    print("        if self.domain == 'source':")
    print("            labels = self.labels[idx]")
    print("        else:")
    print("            labels = torch.empty(0, 6)  # Empty for target domain")
    print("        ")
    print("        return {'img': img, 'labels': labels, 'domain': self.domain}")
    print("```")


def example_loss_monitoring():
    """Example of loss components to monitor during training."""
    print("\n=== Loss Monitoring Example ===\n")
    
    print("Key metrics to monitor during domain adaptation training:")
    print()
    print("1. Main Network Losses:")
    print("   - Supervised Loss: Standard detection loss on source data")
    print("   - Adversarial Loss: Loss for fooling discriminator with target data")
    print("   - Total Main Loss: supervised_loss + adversarial_loss")
    print()
    print("2. Discriminator Loss:")
    print("   - Domain Classification Loss: How well discriminator distinguishes domains")
    print("   - Target: Loss should decrease as discriminator improves")
    print()
    print("3. Domain Predictions:")
    print("   - Source Domain Predictions: Should approach 0.0")
    print("   - Target Domain Predictions: Should approach 1.0")
    print("   - If predictions converge to 0.5, discriminator is confused (good for adaptation)")
    print()
    print("Example monitoring code:")
    print("```python")
    print("# During training loop")
    print("if batch_idx % 50 == 0:")
    print("    print(f'Supervised Loss: {supervised_loss:.4f}')")
    print("    print(f'Adversarial Loss: {adversarial_loss:.4f}')")
    print("    print(f'Discriminator Loss: {discriminator_loss:.4f}')")
    print("    print(f'Source Pred: {source_pred:.3f}, Target Pred: {target_pred:.3f}')")
    print("```")


def example_inference():
    """Example of inference with the trained model."""
    print("\n=== Inference Example ===\n")
    
    print("After training, use the model for standard object detection:")
    print()
    print("```python")
    print("# Load trained model")
    print("model = DomainAdaptiveYOLOv8(nc=80)")
    print("checkpoint = torch.load('runs/domain_adapt/final_model.pt')")
    print("model.model.load_state_dict(checkpoint['model_state_dict'])")
    print("model.model.eval()")
    print()
    print("# Inference (discriminator not used)")
    print("with torch.no_grad():")
    print("    detections, _ = model.forward_with_domain_features(input_image)")
    print("    # Process detections as standard YOLOv8 output")
    print("```")
    print()
    print("Note: During inference, only the main detection network is used.")
    print("The discriminator is only needed during training.")


def example_advanced_configuration():
    """Advanced configuration options."""
    print("\n=== Advanced Configuration ===\n")
    
    print("Advanced settings you can modify:")
    print()
    print("1. Translayer Channels:")
    print("   - Default: 256 channels")
    print("   - Adjust based on your backbone size")
    print("   - model = DomainAdaptiveYOLOv8(translayer_channels=512)")
    print()
    print("2. Domain Loss Weight:")
    print("   - Default: 0.1")
    print("   - Higher values = stronger domain adaptation")
    print("   - model.domain_loss_weight = 0.2")
    print()
    print("3. Learning Rate Schedule:")
    print("   - Main network: Start with 0.001")
    print("   - Discriminator: 10x lower (0.0001)")
    print("   - Reduce both by 0.1 every 60 epochs")
    print()
    print("4. Training Strategy:")
    print("   - Can modify alternating frequency")
    print("   - Can add curriculum learning")
    print("   - Can adjust gradient clipping")


if __name__ == "__main__":
    print("YOLOv8 Domain Adaptation - Usage Examples")
    print("=" * 50)
    
    example_basic_usage()
    example_training_setup()
    example_custom_dataset()
    example_loss_monitoring()
    example_inference()
    example_advanced_configuration()
    
    print("\n" + "=" * 50)
    print("Complete Documentation:")
    print("- See README_domain_adaptation.md for detailed architecture info")
    print("- Run demo_domain_adaptation.py for interactive demonstration")
    print("- Check train_yolov8_simple.py for complete training implementation")
    print("\nFor questions or issues, refer to the original UDAT-car implementation")
    print("in train_new.py which this YOLOv8 version is based on.")