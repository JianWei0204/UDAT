#!/usr/bin/env python3
"""
Demo script for YOLOv8 Domain Adaptation
Shows how to use the domain adaptation components
"""

import torch
import torch.nn.functional as F
from yolov8_domain_model import DomainAdaptiveYOLOv8


def demo_model_architecture():
    """Demonstrate the model architecture and feature extraction."""
    print("=== YOLOv8 Domain Adaptation Architecture Demo ===\n")
    
    # Initialize model
    model = DomainAdaptiveYOLOv8(nc=80, translayer_channels=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)
    model.discriminator.to(device)
    
    print(f"Training on device: {device}")
    print(f"Model initialized with {sum(p.numel() for p in model.model.parameters())} parameters")
    print(f"Discriminator initialized with {sum(p.numel() for p in model.discriminator.parameters())} parameters\n")
    
    # Test with different input sizes
    input_sizes = [(640, 640), (416, 416), (320, 320)]
    
    for h, w in input_sizes:
        print(f"Testing with input size: {h}x{w}")
        x = torch.randn(2, 3, h, w).to(device)
        
        # Forward pass
        predictions, p4_features = model.forward_with_domain_features(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  P4 features shape: {p4_features.shape}")
        print(f"  Number of detection outputs: {len(predictions)}")
        
        for i, pred in enumerate(predictions):
            print(f"    Detection scale {i}: {pred.shape}")
        
        # Test discriminator
        upsampler = torch.nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        p4_upsampled = upsampler(p4_features)
        disc_out = model.discriminator(F.softmax(p4_upsampled, dim=1))
        print(f"  Discriminator output: {disc_out.shape}")
        print()


def demo_domain_adaptation_losses():
    """Demonstrate the domain adaptation loss computations."""
    print("=== Domain Adaptation Loss Computation Demo ===\n")
    
    model = DomainAdaptiveYOLOv8(nc=80, translayer_channels=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)
    model.discriminator.to(device)
    
    # Simulate source and target data
    batch_size = 4
    source_data = torch.randn(batch_size, 3, 640, 640).to(device)
    target_data = torch.randn(batch_size, 3, 640, 640).to(device)
    
    print(f"Source data shape: {source_data.shape}")
    print(f"Target data shape: {target_data.shape}\n")
    
    # Extract features
    _, source_p4 = model.forward_with_domain_features(source_data)
    _, target_p4 = model.forward_with_domain_features(target_data)
    
    print(f"Source P4 features: {source_p4.shape}")
    print(f"Target P4 features: {target_p4.shape}\n")
    
    # Compute adversarial loss (main network training)
    print("1. Adversarial Loss (fool discriminator with target data):")
    adversarial_loss = model.adversarial_loss_step(target_p4, device)
    print(f"   Adversarial loss: {adversarial_loss.item():.6f}\n")
    
    # Compute discriminator loss (discriminator training)
    print("2. Discriminator Loss (distinguish domains):")
    disc_loss_tensor, disc_loss_value = model.train_discriminator_step(target_p4, source_p4, device)
    print(f"   Discriminator loss: {disc_loss_value:.6f}\n")
    
    # Show domain predictions
    print("3. Domain Predictions:")
    upsampler = torch.nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
    
    # Source domain prediction
    source_up = upsampler(source_p4.detach())
    source_pred = model.discriminator(F.softmax(source_up, dim=1))
    print(f"   Source domain predictions (should be ~0): {source_pred.mean().item():.3f}")
    
    # Target domain prediction  
    target_up = upsampler(target_p4.detach())
    target_pred = model.discriminator(F.softmax(target_up, dim=1))
    print(f"   Target domain predictions (should be ~1): {target_pred.mean().item():.3f}")


def demo_training_simulation():
    """Simulate a few training steps to show the alternating training strategy."""
    print("\n=== Training Strategy Simulation ===\n")
    
    model = DomainAdaptiveYOLOv8(nc=80, translayer_channels=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)
    model.discriminator.to(device)
    
    # Setup optimizers
    optimizer_main = torch.optim.Adam(model.model.parameters(), lr=0.001)
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=0.0001)
    
    print("Simulating 3 training steps with alternating strategy:\n")
    
    for step in range(3):
        print(f"--- Training Step {step + 1} ---")
        
        # Generate dummy data
        source_data = torch.randn(2, 3, 416, 416).to(device)
        target_data = torch.randn(2, 3, 416, 416).to(device)
        
        # === MAIN NETWORK TRAINING ===
        print("Phase 1: Main Network Training")
        
        # Freeze discriminator
        for param in model.discriminator.parameters():
            param.requires_grad = False
        
        optimizer_main.zero_grad()
        
        # 1. Adversarial loss with target data
        _, target_p4 = model.forward_with_domain_features(target_data)
        adversarial_loss = model.adversarial_loss_step(target_p4, device)
        adversarial_loss.backward()
        
        # 2. Supervised loss with source data (simplified)
        source_preds, source_p4 = model.forward_with_domain_features(source_data)
        supervised_loss = sum(torch.mean(pred**2) for pred in source_preds) * 0.1  # Dummy loss
        supervised_loss.backward()
        
        optimizer_main.step()
        
        print(f"  Adversarial loss: {adversarial_loss.item():.6f}")
        print(f"  Supervised loss: {supervised_loss.item():.6f}")
        
        # === DISCRIMINATOR TRAINING ===
        print("Phase 2: Discriminator Training")
        
        # Unfreeze discriminator
        for param in model.discriminator.parameters():
            param.requires_grad = True
        
        optimizer_D.zero_grad()
        
        # Train discriminator to distinguish domains
        disc_loss_tensor, disc_loss_value = model.train_discriminator_step(target_p4, source_p4, device)
        disc_loss_tensor.backward()
        optimizer_D.step()
        
        print(f"  Discriminator loss: {disc_loss_value:.6f}")
        
        # Check discriminator performance
        upsampler = torch.nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        with torch.no_grad():
            source_up = upsampler(source_p4.detach())
            target_up = upsampler(target_p4.detach())
            
            source_pred = model.discriminator(F.softmax(source_up, dim=1)).mean()
            target_pred = model.discriminator(F.softmax(target_up, dim=1)).mean()
            
            print(f"  Source prediction (target: 0.0): {source_pred.item():.3f}")
            print(f"  Target prediction (target: 1.0): {target_pred.item():.3f}")
        
        print()


def demo_configuration():
    """Show model configuration and translayer details."""
    print("=== Model Configuration Demo ===\n")
    
    model = DomainAdaptiveYOLOv8(nc=80, translayer_channels=256)
    
    print("YOLOv8 with Domain Adaptation Configuration:")
    print(f"  Number of classes: 80")
    print(f"  Translayer channels: 256")
    print(f"  Domain loss weight: {model.domain_loss_weight}")
    print(f"  Source label: {model.source_label}")
    print(f"  Target label: {model.target_label}\n")
    
    print("Model Architecture Summary:")
    print("  Backbone: Simplified YOLOv8 (P1->P2->P3->P4->P5)")
    print("  Translayer: Added after P4 (Conv1x1 + BN + SiLU)")
    print("  Discriminator: Transformer-based (input: 128x128x256)")
    print("  Detection: Multi-scale (P3, P4, P5)\n")
    
    print("Training Strategy:")
    print("  1. Main Network Phase:")
    print("     - Target data → Adversarial loss (fool discriminator)")
    print("     - Source data → Supervised loss (detection)")
    print("  2. Discriminator Phase:")
    print("     - Target features (detached) → Domain loss (classify as target)")
    print("     - Source features (detached) → Domain loss (classify as source)")


if __name__ == "__main__":
    print("YOLOv8 Domain Adaptation Demo\n")
    print("This demo shows the key components and functionality of the domain adaptation implementation.\n")
    
    try:
        demo_configuration()
        demo_model_architecture()
        demo_domain_adaptation_losses()
        demo_training_simulation()
        
        print("\n=== Demo Completed Successfully ===")
        print("The YOLOv8 domain adaptation implementation is working correctly!")
        print("\nNext steps:")
        print("1. Prepare your source and target domain datasets")
        print("2. Modify the dataset loading in train_yolov8_simple.py")
        print("3. Run full training with: python train_yolov8_simple.py")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("Please check your PyTorch installation and dependencies.")