# UDAT: Domain Adaptation for YOLOv8

This repository implements domain adaptation capabilities for YOLOv8 object detection, following the UDAT (Unsupervised Domain Adaptation for Object Detection) approach. The implementation adds a domain discriminator branch after the P4 layer of the YOLOv8 backbone to enable adversarial domain adaptation training.

## Features

- **Domain Adaptation Training**: Implements alternating training strategy between main detection network and domain discriminator
- **P4 Feature Extraction**: Extracts intermediate features from the P4 layer for domain adaptation
- **Transformer Discriminator**: Uses attention-based discriminator for robust domain classification
- **Gradient Reversal Layer**: Implements adversarial training with gradient reversal
- **Compatible with YOLOv8**: Extends existing YOLOv8 DetectionTrainer with minimal modifications

## Architecture

The implementation follows the UDAT-car approach:

1. **Bridge Layer**: Added after the P4 layer (configured in YAML)
2. **Domain Discriminator**: TransformerDiscriminator that classifies features as source/target domain
3. **Alternating Training**:
   - **Main Network Training** (discriminator frozen): Target domain → adversarial loss, Source domain → detection loss
   - **Discriminator Training** (main network frozen): Both domains → domain classification loss

## Files Overview

### Core Implementation
- `domain_adaptation_trainer.py`: Main trainer class extending DetectionTrainer
- `yolo_feature_extractor.py`: Model wrapper for P4 feature extraction
- `domain_data_loader.py`: Data handling for source/target domains
- `train_domain_adaptation.py`: Training script for domain adaptation

### Existing Modules (from UDAT-car)
- `GRL.py`: Gradient Reversal Layer implementation
- `trans_discriminator.py`: Transformer-based domain discriminator
- `train_new.py`: Reference implementation of UDAT-car approach

### Configuration
- `source_dataset.yaml`: Example source domain dataset configuration
- `target_dataset.yaml`: Example target domain dataset configuration
- `domain_adaptation_config.py`: Configuration examples and parameters

## Installation

1. Install YOLOv8 and dependencies:
```bash
pip install ultralytics torch torchvision
```

2. Clone this repository:
```bash
git clone https://github.com/JianWei0204/UDAT.git
cd UDAT
```

## Usage

### Basic Domain Adaptation Training

```bash
python train_domain_adaptation.py \
    --source_data source_dataset.yaml \
    --target_data target_dataset.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --batch 16 \
    --discriminator_lr 1e-4 \
    --adv_loss_weight 0.1
```

### Advanced Configuration

```bash
python train_domain_adaptation.py \
    --source_data /path/to/source.yaml \
    --target_data /path/to/target.yaml \
    --model yolov8s.pt \
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --device 0,1,2,3 \
    --discriminator_lr 1e-4 \
    --adv_loss_weight 0.1 \
    --discriminator_channels 512 \
    --lr0 0.01 \
    --warmup_epochs 3 \
    --project runs/my_domain_adapt \
    --name experiment_1 \
    --val \
    --plots
```

## Dataset Format

Both source and target domain datasets should follow the YOLOv8 format:

```yaml
# dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 80
names: ['class1', 'class2', ...]
domain: source  # or target
domain_label: 0  # 0 for source, 1 for target
```

Directory structure:
```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       ├── img3.jpg
│       └── img4.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        ├── img3.txt
        └── img4.txt
```

## Key Parameters

### Domain Adaptation Parameters
- `--discriminator_lr`: Learning rate for discriminator (default: 1e-4)
- `--adv_loss_weight`: Weight for adversarial loss (default: 0.1)
- `--discriminator_channels`: Input channels for discriminator, should match P4 feature channels (default: 512)

### Training Parameters
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--lr0`: Initial learning rate for main network
- `--warmup_epochs`: Warmup epochs

## Implementation Details

### Alternating Training Strategy

The training alternates between two phases each iteration:

1. **Main Network Phase**:
   - Freeze discriminator parameters
   - Process target domain data → extract P4 features → discriminator → adversarial loss (fool discriminator)
   - Process source domain data → standard detection loss
   - Update main network parameters

2. **Discriminator Phase**:
   - Freeze main network parameters
   - Process target domain features (detached) → discriminator → classify as target (label=1)
   - Process source domain features (detached) → discriminator → classify as source (label=0)
   - Update discriminator parameters

### Feature Extraction

P4 features are extracted using hooks registered on the appropriate layer of the YOLOv8 backbone. Features are upsampled to 128×128 and passed through softmax before feeding to the discriminator.

### Loss Functions

- **Detection Loss**: Standard YOLOv8 loss (box_loss + cls_loss + dfl_loss)
- **Adversarial Loss**: MSE loss between discriminator output and source label for target domain data
- **Discriminator Loss**: MSE loss for domain classification on both source and target features

## Extending the Implementation

### Custom Discriminator
To use a different discriminator architecture:

```python
from domain_adaptation_trainer import DomainAdaptationTrainer

class CustomDomainTrainer(DomainAdaptationTrainer):
    def setup_discriminator(self):
        self.discriminator = MyCustomDiscriminator(channels=512)
        # ... rest of setup
```

### Custom Feature Extraction
To extract features from different layers:

```python
from yolo_feature_extractor import YOLOv8WithFeatureExtraction

class CustomFeatureModel(YOLOv8WithFeatureExtraction):
    def _register_feature_hooks(self):
        # Register hooks on different layers
        # ... custom implementation
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or use gradient accumulation
2. **Feature Dimension Mismatch**: Adjust `discriminator_channels` to match P4 feature dimensions
3. **Training Instability**: Adjust `adv_loss_weight` or learning rates

### Monitoring Training

Use TensorBoard or similar tools to monitor:
- Detection losses (box, cls, dfl)
- Adversarial loss
- Discriminator loss
- Learning rates for both networks

## Citation

If you use this implementation, please cite:

```bibtex
@article{udat2024,
  title={UDAT: Unsupervised Domain Adaptation for Object Detection},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the AGPL-3.0 License - see the LICENSE file for details.

## Acknowledgments

- Based on the YOLOv8 implementation by Ultralytics
- Inspired by the UDAT-car domain adaptation approach
- Uses Transformer-based discriminator architecture