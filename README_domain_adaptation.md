# YOLOv8 Domain Adaptation Implementation

This repository implements domain adaptation for YOLOv8 object detection based on the UDAT (Unsupervised Domain Adaptive Tracking) approach. The implementation adds a bridge layer (translayer) at the P4 backbone level and integrates a domain discriminator for unsupervised domain adaptation.

## Architecture Overview

### 1. Bridge Layer (Translayer)
- **Location**: Added after P4 layer in YOLOv8 backbone (at 16x downsampling)
- **Purpose**: Provides a bottleneck where domain-specific features can be extracted
- **Configuration**: Defined in `yolov8-translayer.yaml`

### 2. Domain Discriminator
- **Type**: Transformer-based discriminator from `trans_discriminator.py`
- **Input**: P4 features after translayer (upsampled to 128x128)
- **Output**: Binary domain classification (source=0, target=1)
- **Gradient Reversal**: Uses GRL (Gradient Reversal Layer) for adversarial training

### 3. Training Strategy
The training alternates between two phases:

#### Main Network Training (Discriminator Frozen)
1. **Target Domain Adversarial Loss**: 
   - Target data → P4 features → Discriminator
   - Label as "source" (fool discriminator)
   - Backprop only to main network

2. **Source Domain Supervised Loss**:
   - Source data → Detection predictions
   - Standard detection loss (box, class, DFL)
   - Backprop only to main network

#### Discriminator Training (Discriminator Unfrozen)
1. **Target Domain Classification**:
   - Target P4 features (detached) → Discriminator
   - Label as "target"
   - Backprop only to discriminator

2. **Source Domain Classification**:
   - Source P4 features (detached) → Discriminator
   - Label as "source"
   - Backprop only to discriminator

## File Structure

```
├── yolov8-translayer.yaml          # YOLOv8 config with bridge layer
├── yolov8_domain_model.py           # Model implementation with translayer
├── train_yolov8_domain_adapt.py     # Full training implementation (Ultralytics-based)
├── train_yolov8_simple.py           # Simplified training demo
├── GRL.py                          # Gradient Reversal Layer
├── trans_discriminator.py          # Transformer domain discriminator
└── README_domain_adaptation.md     # This file
```

## Key Components

### YOLOv8WithTranslayer
```python
class YOLOv8WithTranslayer(nn.Module):
    def __init__(self, nc=80, ch=3, translayer_channels=256):
        # Backbone up to P4
        # Bridge layer (translayer) 
        # Remaining backbone (P5)
        # Detection heads
```

### DomainAdaptiveYOLOv8
```python
class DomainAdaptiveYOLOv8:
    def __init__(self, nc=80, translayer_channels=256):
        self.model = YOLOv8WithTranslayer(...)
        self.discriminator = TransformerDiscriminator(...)
```

## Usage

### Quick Test
```bash
python yolov8_domain_model.py
```

### Training Demo
```bash
python train_yolov8_simple.py
```

### Full Training (requires Ultralytics)
```bash
python train_yolov8_domain_adapt.py
```

## Training Configuration

### Hyperparameters
- **Main Network LR**: 0.001
- **Discriminator LR**: 0.0001 (10x lower)
- **Domain Loss Weight**: 0.1
- **Batch Size**: Configurable (default: 8)

### Loss Components
1. **Supervised Loss**: Standard YOLOv8 detection loss on source data
2. **Adversarial Loss**: MSE loss to fool discriminator on target data
3. **Discriminator Loss**: MSE loss for domain classification

### Data Requirements
- **Source Domain**: Labeled detection data (e.g., synthetic/simulation)
- **Target Domain**: Unlabeled real-world data
- **Format**: Standard YOLO format (images + bounding box annotations)

## Implementation Details

### P4 Feature Extraction
```python
def get_p4_features(self, x):
    # Forward to P4 layer
    for i, layer in enumerate(self.backbone):
        x = layer(x)
        if i == 8:  # After P4 layers
            break
    # Apply translayer
    p4_trans = self.translayer(x)
    return p4_trans
```

### Domain Loss Function
```python
def domain_loss_function(self, D_out, label, device):
    target_label = torch.FloatTensor(D_out.data.size()).fill_(label).to(device)
    return torch.mean((D_out - target_label).abs() ** 2)
```

### Alternating Training Loop
```python
# Phase 1: Train main network
for param in discriminator.parameters():
    param.requires_grad = False

# Target adversarial loss
adversarial_loss = compute_adversarial_loss(target_features)
adversarial_loss.backward()

# Source supervised loss  
supervised_loss = compute_detection_loss(source_predictions)
supervised_loss.backward()

optimizer_main.step()

# Phase 2: Train discriminator
for param in discriminator.parameters():
    param.requires_grad = True

discriminator_loss = compute_discriminator_loss(target_features, source_features)
discriminator_loss.backward()
optimizer_D.step()
```

## Comparison with UDAT-Car

| Aspect | UDAT-Car | YOLOv8-UDAT |
|--------|----------|-------------|
| Base Network | SiamCar | YOLOv8 |
| Bridge Location | Between backbone & head | After P4 layer |
| Task | Object Tracking | Object Detection |
| Features Used | Template & Search | P4 Feature Maps |
| Discriminator Input | 256-channel features | 256-channel P4 features |

## Benefits

1. **Improved Generalization**: Adapts from labeled source to unlabeled target domain
2. **Minimal Architecture Changes**: Only adds bridge layer and discriminator branch
3. **Flexible Training**: Can switch between standard and domain-adaptive training
4. **Reusable Components**: GRL and discriminator can be applied to other architectures

## Notes

- The discriminator operates only during training (not inference)
- P4 features are chosen as they contain mid-level semantic information
- Gradient reversal ensures adversarial training without explicit loss sign flip
- Implementation supports both GPU and CPU training (GPU recommended)