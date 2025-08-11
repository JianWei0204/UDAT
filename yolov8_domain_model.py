"""
YOLOv8 Model with Bridge Layer and Domain Adaptation Support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from trans_discriminator import TransformerDiscriminator


class YOLOv8WithTranslayer(nn.Module):
    """
    YOLOv8 model with bridge layer (translayer) for domain adaptation.
    Adds a bridge layer at P4 level and provides access to intermediate features.
    """
    
    def __init__(self, nc=80, ch=3, translayer_channels=256):
        super().__init__()
        self.nc = nc  # number of classes
        self.ch = ch  # input channels
        
        # Simplified YOLOv8 backbone (CSPDarknet)
        self.backbone = nn.ModuleList([
            # P1/2
            nn.Conv2d(ch, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            
            # P2/4  
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            
            # P3/8
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            
            # P4/16 - This is where we add translayer
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
        ])
        
        # Bridge layer (translayer) at P4
        self.translayer = nn.Sequential(
            nn.Conv2d(512, translayer_channels, 1),
            nn.BatchNorm2d(translayer_channels),
            nn.SiLU(inplace=True)
        )
        
        # Continue backbone after translayer
        self.backbone_post = nn.ModuleList([
            # P5/32
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.BatchNorm2d(1024),
            nn.SiLU(inplace=True),
        ])
        
        # Simple detection head
        self.head = nn.ModuleList([
            # Detection layers for P3, P4, P5
            nn.Conv2d(256, nc + 5, 1),  # P3 detection (256 channels)
            nn.Conv2d(translayer_channels, nc + 5, 1),  # P4 detection (translayer channels)
            nn.Conv2d(1024, nc + 5, 1), # P5 detection (1024 channels)
        ])
        
        self.stride = torch.tensor([8., 16., 32.])  # strides for P3, P4, P5
        
    def forward(self, x, return_features=False):
        """Forward pass with optional feature extraction."""
        features = []
        
        # Backbone forward to P4
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [2, 5, 8]:  # After P1, P2, P3 blocks
                features.append(x)
        
        # P4 features before translayer
        p4_raw = x
        
        # Apply translayer (bridge layer)
        p4_trans = self.translayer(x)
        features.append(p4_trans)  # P4 features after translayer
        
        # Continue with backbone (P5)
        x = p4_raw  # Continue from raw P4 for main detection path
        for layer in self.backbone_post:
            x = layer(x)
        p5 = x
        features.append(p5)  # P5 features
        
        if return_features:
            return features
        
        # Simple detection head (normally would be more complex FPN structure)
        detections = []
        # Use P3, P4 (after translayer), P5 features
        detection_features = [features[2], features[3], features[4]]  # P3, P4_trans, P5
        for i, (feat, head_layer) in enumerate(zip(detection_features, self.head)):
            det = head_layer(feat)
            detections.append(det)
        
        return detections
    
    def get_p4_features(self, x):
        """Extract P4 features specifically for domain adaptation."""
        # Forward to P4 layer
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 8:  # After P4 layers (conv + bn + silu)
                break
        
        # Apply translayer
        p4_trans = self.translayer(x)
        return p4_trans


class DomainAdaptiveYOLOv8:
    """
    Wrapper class for YOLOv8 with domain adaptation capabilities.
    Integrates the translayer model with domain discriminator.
    """
    
    def __init__(self, nc=80, translayer_channels=256):
        self.model = YOLOv8WithTranslayer(nc=nc, translayer_channels=translayer_channels)
        self.discriminator = TransformerDiscriminator(channels=translayer_channels)
        
        # Domain adaptation parameters
        self.domain_loss_weight = 0.1
        self.source_label = 0
        self.target_label = 1
        
    def forward_with_domain_features(self, x):
        """Forward pass that returns both predictions and domain-relevant features."""
        # Get model features
        features = self.model(x, return_features=True)
        
        # Get predictions
        predictions = self.model(x, return_features=False)
        
        # P4 features for domain adaptation (features[3] is P4 after translayer)
        p4_features = features[3]
        
        return predictions, p4_features
    
    def domain_loss_function(self, D_out, label, device):
        """Weighted MSE loss for domain classification."""
        target_label = torch.FloatTensor(D_out.data.size()).fill_(label).to(device)
        return torch.mean((D_out - target_label).abs() ** 2)
    
    def train_discriminator_step(self, target_features, source_features, device):
        """Single step of discriminator training."""
        discriminator_loss = 0.0
        
        # Upsample features to discriminator input size (128x128)
        interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        
        # Target domain features (detached)
        target_up = interp(target_features.detach())
        target_out = self.discriminator(F.softmax(target_up, dim=1))
        loss_target = self.domain_loss_weight * self.domain_loss_function(
            target_out, self.target_label, device
        )
        discriminator_loss += loss_target.item()
        
        # Source domain features (detached)  
        source_up = interp(source_features.detach())
        source_out = self.discriminator(F.softmax(source_up, dim=1))
        loss_source = self.domain_loss_weight * self.domain_loss_function(
            source_out, self.source_label, device
        )
        discriminator_loss += loss_source.item()
        
        # Combined discriminator loss
        total_d_loss = loss_target + loss_source
        
        return total_d_loss, discriminator_loss
    
    def adversarial_loss_step(self, target_features, device):
        """Compute adversarial loss for main network."""
        # Upsample target features
        interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        target_up = interp(target_features)
        
        # Get discriminator prediction
        target_out = self.discriminator(F.softmax(target_up, dim=1))
        
        # Adversarial loss - fool discriminator by making target look like source
        adversarial_loss = self.domain_loss_weight * self.domain_loss_function(
            target_out, self.source_label, device
        )
        
        return adversarial_loss


def test_model():
    """Test the domain adaptive YOLOv8 model."""
    print("Testing YOLOv8 with Domain Adaptation...")
    
    # Create model
    da_yolo = DomainAdaptiveYOLOv8(nc=80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    da_yolo.model.to(device)
    da_yolo.discriminator.to(device)
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 640, 640).to(device)
    
    print(f"Input shape: {x.shape}")
    
    # Test forward pass
    predictions, p4_features = da_yolo.forward_with_domain_features(x)
    
    print(f"P4 features shape: {p4_features.shape}")
    print(f"Number of detection outputs: {len(predictions)}")
    for i, pred in enumerate(predictions):
        print(f"Detection {i} shape: {pred.shape}")
    
    # Test discriminator with proper upsampling
    upsampler = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
    p4_upsampled = upsampler(p4_features)
    print(f"P4 upsampled shape: {p4_upsampled.shape}")
    
    disc_out = da_yolo.discriminator(F.softmax(p4_upsampled, dim=1))
    print(f"Discriminator output shape: {disc_out.shape}")
    
    print("Model test completed successfully!")


if __name__ == "__main__":
    test_model()