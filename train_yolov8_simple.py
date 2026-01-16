"""
Simplified YOLOv8 Domain Adaptation Training Script
Based on UDAT approach with alternating training strategy
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path

from yolov8_domain_model import DomainAdaptiveYOLOv8
from trans_discriminator import TransformerDiscriminator
from GRL import GradientScalarLayer


class DummyDataset(Dataset):
    """Dummy dataset for testing domain adaptation training."""
    
    def __init__(self, domain='source', size=1000, img_size=640):
        self.domain = domain
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate dummy data
        img = torch.randn(3, self.img_size, self.img_size)
        
        # Dummy labels (simplified for detection) - fixed size for collation
        # Format: [batch_idx, class, x_center, y_center, width, height]
        num_boxes = 3  # Fixed number for consistency
        labels = torch.rand(num_boxes, 6)
        labels[:, 0] = 0  # batch index
        labels[:, 1] = torch.randint(0, 80, (num_boxes,)).float()  # class
        
        return {
            'img': img,
            'labels': labels,
            'domain': self.domain
        }


def detection_loss(predictions, targets):
    """Simplified detection loss (normally would be more complex)."""
    # This is a placeholder - in real implementation you'd use proper YOLO loss
    total_loss = 0.0
    for pred in predictions:
        # Simple MSE loss for demonstration
        target_shape = pred.shape
        dummy_target = torch.zeros_like(pred)
        total_loss += nn.MSELoss()(pred, dummy_target)
    
    return total_loss / len(predictions)


def adjust_learning_rate(optimizer, epoch, base_lr, schedule=[60, 120, 180], gamma=0.1):
    """Adjust learning rate according to schedule."""
    lr = base_lr
    for milestone in schedule:
        if epoch >= milestone:
            lr *= gamma
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def train_domain_adaptive_yolo(
    source_data_path=None,
    target_data_path=None,
    epochs=100,
    batch_size=8,
    lr=0.001,
    lr_d=0.0001,
    device=None,
    save_dir='./runs/domain_adapt'
):
    """
    Train YOLOv8 with domain adaptation.
    
    Args:
        source_data_path: Path to source domain data
        target_data_path: Path to target domain data  
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate for main network
        lr_d: Learning rate for discriminator
        device: Training device
        save_dir: Directory to save checkpoints
    """
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print("Initializing YOLOv8 with Domain Adaptation...")
    da_yolo = DomainAdaptiveYOLOv8(nc=80, translayer_channels=256)
    da_yolo.model.to(device)
    da_yolo.discriminator.to(device)
    
    # Setup datasets (using dummy data for demonstration)
    print("Setting up datasets...")
    source_dataset = DummyDataset(domain='source', size=1000)
    target_dataset = DummyDataset(domain='target', size=800)
    
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizers
    print("Setting up optimizers...")
    optimizer_main = optim.Adam(da_yolo.model.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(da_yolo.discriminator.parameters(), lr=lr_d, betas=(0.9, 0.99))
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Adjust learning rates
        current_lr = adjust_learning_rate(optimizer_main, epoch, lr)
        current_lr_d = adjust_learning_rate(optimizer_D, epoch, lr_d)
        
        # Set models to training mode
        da_yolo.model.train()
        da_yolo.discriminator.train()
        
        # Initialize metrics
        total_main_loss = 0.0
        total_adversarial_loss = 0.0
        total_discriminator_loss = 0.0
        total_supervised_loss = 0.0
        num_batches = 0
        
        # Create iterators
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        # Training loop - alternate between source and target
        max_batches = min(len(source_loader), len(target_loader))
        
        for batch_idx in range(max_batches):
            
            # === MAIN NETWORK TRAINING ===
            
            # 1. Get target batch and compute adversarial loss
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
            
            target_imgs = target_batch['img'].to(device)
            
            # Freeze discriminator for main network training
            for param in da_yolo.discriminator.parameters():
                param.requires_grad = False
            
            optimizer_main.zero_grad()
            
            # Forward target data through main network
            target_predictions, target_p4_features = da_yolo.forward_with_domain_features(target_imgs)
            
            # Compute adversarial loss (fool discriminator)
            adversarial_loss = da_yolo.adversarial_loss_step(target_p4_features, device)
            adversarial_loss.backward()
            
            # 2. Get source batch and compute supervised loss
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_batch = next(source_iter)
            
            source_imgs = source_batch['img'].to(device)
            source_labels = source_batch['labels']
            
            # Forward source data through main network
            source_predictions, source_p4_features = da_yolo.forward_with_domain_features(source_imgs)
            
            # Compute supervised detection loss
            supervised_loss = detection_loss(source_predictions, source_labels)
            supervised_loss.backward()
            
            # Update main network
            optimizer_main.step()
            
            # === DISCRIMINATOR TRAINING ===
            
            # Unfreeze discriminator
            for param in da_yolo.discriminator.parameters():
                param.requires_grad = True
            
            optimizer_D.zero_grad()
            
            # Train discriminator to distinguish domains
            discriminator_loss_tensor, discriminator_loss_value = da_yolo.train_discriminator_step(
                target_p4_features, source_p4_features, device
            )
            discriminator_loss_tensor.backward()
            optimizer_D.step()
            
            # Update metrics
            total_adversarial_loss += adversarial_loss.item()
            total_supervised_loss += supervised_loss.item()
            total_discriminator_loss += discriminator_loss_value
            total_main_loss += (adversarial_loss.item() + supervised_loss.item())
            num_batches += 1
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{max_batches}")
                print(f"  Supervised Loss: {supervised_loss.item():.4f}")
                print(f"  Adversarial Loss: {adversarial_loss.item():.4f}")
                print(f"  Discriminator Loss: {discriminator_loss_value:.4f}")
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start_time
        avg_main_loss = total_main_loss / num_batches
        avg_adversarial_loss = total_adversarial_loss / num_batches
        avg_supervised_loss = total_supervised_loss / num_batches
        avg_discriminator_loss = total_discriminator_loss / num_batches
        
        print(f"\nEpoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
        print(f"Average Losses:")
        print(f"  Main Network: {avg_main_loss:.4f}")
        print(f"  Supervised: {avg_supervised_loss:.4f}")
        print(f"  Adversarial: {avg_adversarial_loss:.4f}")
        print(f"  Discriminator: {avg_discriminator_loss:.4f}")
        print(f"Learning Rates: Main={current_lr:.6f}, Discriminator={current_lr_d:.6f}")
        print("-" * 60)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': da_yolo.model.state_dict(),
                'discriminator_state_dict': da_yolo.discriminator.state_dict(),
                'optimizer_main_state_dict': optimizer_main.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'avg_main_loss': avg_main_loss,
                'avg_discriminator_loss': avg_discriminator_loss,
            }
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_checkpoint = {
        'epoch': epochs,
        'model_state_dict': da_yolo.model.state_dict(),
        'discriminator_state_dict': da_yolo.discriminator.state_dict(),
        'optimizer_main_state_dict': optimizer_main.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }
    final_path = save_dir / 'final_model.pt'
    torch.save(final_checkpoint, final_path)
    print(f"Final model saved: {final_path}")
    
    print("Training completed!")


def test_training():
    """Test the domain adaptation training with small scale."""
    print("Testing YOLOv8 Domain Adaptation Training...")
    
    train_domain_adaptive_yolo(
        epochs=5,
        batch_size=4,
        lr=0.001,
        lr_d=0.0001,
        save_dir='./test_runs/domain_adapt_test'
    )


if __name__ == "__main__":
    test_training()