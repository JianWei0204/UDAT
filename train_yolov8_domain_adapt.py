# Ultralytics YOLO 🚀 with Domain Adaptation, AGPL-3.0 license

import math
import random
import time
import numpy as np
from copy import copy, deepcopy
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Try to import from ultralytics - if not available, create minimal compatible versions
try:
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.models import yolo
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import LOGGER, RANK
    from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
    from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
    from ultralytics.utils import TQDM
except ImportError:
    # Create minimal compatible classes for development
    LOGGER = None
    RANK = -1
    
    class BaseTrainer:
        def __init__(self, cfg=None, overrides=None, _callbacks=None):
            self.args = cfg or {}
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.start_epoch = 0
            self.epochs = 100
            
        def run_callbacks(self, event): pass
        def setup_model(self): pass
        def _setup_train(self, world_size): pass
        def _setup_ddp(self, world_size): pass
        def optimizer_step(self): pass
        def validate(self): return {}, 0.0
        def save_metrics(self, metrics): pass
        def final_eval(self): pass
        def plot_training_samples(self, batch, ni): pass
        def plot_metrics(self): pass
        def save_model(self): pass
        
    def de_parallel(model): return model
    def torch_distributed_zero_first(rank): return nullcontext()
    from contextlib import nullcontext
    TQDM = range

# Import domain adaptation components
from trans_discriminator import TransformerDiscriminator
from GRL import GradientScalarLayer

class DomainAdaptiveDetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for domain adaptive training based on a detection model.
    Implements UDAT (Unsupervised Domain Adaptive Tracking) approach for YOLOv8.
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        
        # Domain adaptation specific parameters
        self.source_loader = None
        self.target_loader = None
        self.discriminator = None
        self.optimizer_D = None
        self.domain_loss_weight = 0.1
        
        # Domain labels
        self.source_label = 0
        self.target_label = 1

    def setup_domain_adaptation(self):
        """Setup domain discriminator and optimizer."""
        # Initialize domain discriminator
        self.discriminator = TransformerDiscriminator(channels=256)  # P4 feature channels
        self.discriminator.train()
        self.discriminator = self.discriminator.to(self.device)
        
        # Setup discriminator optimizer
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.args.lr0 * 0.1,  # Lower learning rate for discriminator
            betas=(0.9, 0.99)
        )
        self.optimizer_D.zero_grad()

    def build_domain_dataset(self, img_path, mode="train", batch=None, domain="source"):
        """
        Build YOLO Dataset for specific domain.
        
        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode.
            batch (int, optional): Size of batches.
            domain (str): "source" or "target" domain.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        # You can customize this to load different datasets for source and target domains
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_domain_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train", domain="source"):
        """Construct and return dataloader for specific domain."""
        assert mode in ["train", "val"]
        with torch_distributed_zero_first(rank):
            dataset = self.build_domain_dataset(dataset_path, mode, batch_size, domain)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def setup_model(self):
        """Setup the model with domain adaptation components."""
        super().setup_model()
        self.setup_domain_adaptation()

    def get_p4_features(self, x):
        """Extract P4 level features from the model backbone."""
        # This method extracts features at P4 level (after layer 6 in backbone)
        # We need to modify the model to provide access to intermediate features
        model = de_parallel(self.model)
        
        # Forward through backbone layers up to P4
        for i, layer in enumerate(model.model[:7]):  # Up to layer 6 (P4)
            x = layer(x)
            if i == 6:  # P4 layer
                p4_features = x
                break
        
        return p4_features

    def forward_with_features(self, batch):
        """Forward pass that returns both predictions and P4 features."""
        model = de_parallel(self.model)
        x = batch["img"]
        
        # Extract P4 features during forward pass
        features = []
        for i, layer in enumerate(model.model):
            x = layer(x)
            if i == 6:  # P4 layer (after backbone layer 6)
                features.append(x)
                
        return x, features

    def domain_loss_function(self, D_out, label):
        """Weighted MSE loss for domain classification."""
        target_label = torch.FloatTensor(D_out.data.size()).fill_(label).to(self.device)
        return torch.mean((D_out - target_label).abs() ** 2)

    def train_discriminator(self, target_features, source_features):
        """Train the domain discriminator."""
        # Enable discriminator gradients
        for param in self.discriminator.parameters():
            param.requires_grad = True
            
        self.optimizer_D.zero_grad()
        
        # Upsample features to 128x128 for discriminator input
        interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        
        # Process target domain features (detached to avoid affecting main network)
        target_features_up = [interp(feat.detach()) for feat in target_features]
        target_domain_output = torch.stack([
            self.discriminator(F.softmax(feat, dim=1)) for feat in target_features_up
        ]).mean(0)
        
        # Loss for target domain (should be classified as target)
        loss_target = self.domain_loss_weight * self.domain_loss_function(
            target_domain_output, self.target_label
        )
        loss_target.backward()
        
        # Process source domain features (detached)
        source_features_up = [interp(feat.detach()) for feat in source_features]
        source_domain_output = torch.stack([
            self.discriminator(F.softmax(feat, dim=1)) for feat in source_features_up
        ]).mean(0)
        
        # Loss for source domain (should be classified as source)
        loss_source = self.domain_loss_weight * self.domain_loss_function(
            source_domain_output, self.source_label
        )
        loss_source.backward()
        
        self.optimizer_D.step()
        
        return loss_target.item() + loss_source.item()

    def train_generator_adversarial(self, target_features):
        """Train the main network with adversarial loss."""
        # Freeze discriminator parameters
        for param in self.discriminator.parameters():
            param.requires_grad = False
            
        # Upsample target features
        interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        target_features_up = [interp(feat) for feat in target_features]
        
        # Get discriminator output for target features
        target_domain_output = torch.stack([
            self.discriminator(F.softmax(feat, dim=1)) for feat in target_features_up
        ]).mean(0)
        
        # Adversarial loss - fool discriminator by making target features look like source
        adversarial_loss = self.domain_loss_weight * self.domain_loss_function(
            target_domain_output, self.source_label
        )
        
        return adversarial_loss

    def _do_train(self, world_size=1):
        """Custom training loop with domain adaptation."""
        if world_size > 1:
            self._setup_ddp(world_size)

        self._setup_train(world_size)

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        
        # Setup domain-specific data loaders
        # Note: In practice, you would set different paths for source and target domains
        self.source_loader = self.get_domain_dataloader(
            self.args.data, self.args.batch, RANK, mode="train", domain="source"
        )
        self.target_loader = self.get_domain_dataloader(
            self.args.data, self.args.batch, RANK, mode="train", domain="target"
        )
        
        # Create iterators for alternating between domains
        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            self.model.train()
            
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            
            pbar = enumerate(self.train_loader)
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(pbar, total=nb)

            self.tloss = None
            self.optimizer.zero_grad()
            
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                ni = i + nb * epoch
                
                # Warmup
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.args.batch]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )

                # Get target domain batch
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_loader)
                    target_batch = next(target_iter)

                # Get source domain batch  
                try:
                    source_batch = next(source_iter)
                except StopIteration:
                    source_iter = iter(self.source_loader)
                    source_batch = next(source_iter)

                # Preprocess batches
                target_batch = self.preprocess_batch(target_batch)
                source_batch = self.preprocess_batch(source_batch)

                # === MAIN NETWORK TRAINING ===
                
                # 1. Train with target data (adversarial loss)
                target_pred, target_features = self.forward_with_features(target_batch)
                adversarial_loss = self.train_generator_adversarial(target_features)
                
                # Scale adversarial loss
                adversarial_loss = adversarial_loss / self.accumulate
                self.scaler.scale(adversarial_loss).backward()

                # 2. Train with source data (supervised loss)
                source_pred = self.model(source_batch)
                supervised_loss, loss_items = self.criterion(source_pred, source_batch)
                
                # Scale supervised loss
                supervised_loss = supervised_loss / self.accumulate
                self.scaler.scale(supervised_loss).backward()

                # === DISCRIMINATOR TRAINING ===
                discriminator_loss = self.train_discriminator(target_features, [source_pred])

                # Optimize
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                if RANK in {-1, 0}:
                    mloss = (mloss * i + loss_items) / (i + 1) if self.tloss is not None else loss_items
                    self.tloss = mloss
                    mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                    pbar.set_description(
                        (f"Epoch {epoch + 1}/{self.epochs} | GPU_mem {mem} | "
                         f"Sup_loss {supervised_loss:.4f} | Adv_loss {adversarial_loss:.4f} | "
                         f"D_loss {discriminator_loss:.4f}")
                    )
                    self.run_callbacks("on_batch_end")

                self.run_callbacks("on_train_batch_end")

            # Validation and other epoch-end operations
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")

            if RANK in {-1, 0}:
                final_epoch = epoch + 1 == self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.plots:
                    self.plot_training_samples(batch, ni)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

        # End training
        if RANK in {-1, 0}:
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")

        torch.cuda.empty_cache()
        self.run_callbacks("on_teardown")

    def save_model(self):
        """Save both main model and discriminator."""
        super().save_model()
        
        # Save discriminator checkpoint
        if hasattr(self, 'discriminator') and self.discriminator is not None:
            discriminator_ckpt = {
                'epoch': self.epoch,
                'model': deepcopy(de_parallel(self.discriminator)).half(),
                'optimizer': self.optimizer_D.state_dict(),
                'date': datetime.now().isoformat()
            }
            torch.save(discriminator_ckpt, self.last.parent / 'discriminator_last.pt')
            if self.best_fitness == self.fitness:
                torch.save(discriminator_ckpt, self.best.parent / 'discriminator_best.pt')

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build YOLO Dataset - wrapper for compatibility."""
        return self.build_domain_dataset(img_path, mode, batch, domain="source")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader - wrapper for compatibility."""
        return self.get_domain_dataloader(dataset_path, batch_size, rank, mode, domain="source")

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride)
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model with translayer configuration."""
        # Use the translayer configuration if available
        if cfg is None:
            cfg = 'yolov8-translayer.yaml'
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )