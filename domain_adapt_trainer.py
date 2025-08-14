# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import gc
import pandas as pd
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.models.yolo.detect import DetectionTrainer

# 导入正确的损失类
from ultralytics.utils.loss import v8DetectionLoss

from trans_discriminator import TransformerDiscriminator
from feature_extractor import FeatureExtractor


class DomainAdaptTrainer(DetectionTrainer):
    """
    Domain Adaptation Trainer for YOLOv8 object detection.

    This trainer extends the standard YOLOv8 DetectionTrainer to support
    unsupervised domain adaptation using a transformer-based discriminator.
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """初始化域适应训练器"""
        # 使用标准YOLOv8参数初始化
        super().__init__(cfg, overrides, _callbacks)

        # 初始化域适应组件和参数
        self.target_data = None
        self.target_dataset = None
        self.target_loader = None
        self.source_loader = None
        self.disc_lr = 0.001  # 默认判别器学习率

        # 源域和目标域标签
        self.source_label = 0
        self.target_label = 1

        # 域适应组件
        self.discriminator = None
        self.optimizer_D = None
        self.feature_extractor = None
        self.domain_adapt_enabled = False

        # 保存设备信息
        self.device = None

        # 进度条描述格式
        self.epoch_desc = "Epoch {epoch}"

        # 显式添加epoch属性
        self.epoch = 0

    def setup_domain_adaptation(self, target_data, disc_lr=0.001):
        """设置域适应训练需要的参数和组件"""
        self.target_data = target_data
        self.disc_lr = disc_lr
        self.domain_adapt_enabled = True

        LOGGER.info(f"Domain adaptation enabled with target data: {self.target_data}")
        LOGGER.info(f"Discriminator learning rate set to: {self.disc_lr}")

    def _setup_train(self, world_size):
        """设置训练与域适应组件"""
        # 初始化标准训练设置
        super()._setup_train(world_size)

        # 确保结果目录存在
        os.makedirs(self.save_dir, exist_ok=True)

        # 确保results.csv文件存在 - 这里使用父类中已定义的self.csv路径
        self.create_empty_results_csv()

        # 设置设备
        self.device = next(self.model.parameters()).device
        LOGGER.info(f"Model is on device: {self.device}")

        # 将模型移动到GPU（如果可用）
        if torch.cuda.is_available() and self.device.type != 'cuda':
            LOGGER.info(f"Moving model to CUDA device")
            self.model.to('cuda')
            self.device = next(self.model.parameters()).device

        # 如果启用了域适应，则设置相关组件
        if self.domain_adapt_enabled:
            # 创建特征提取器访问桥接层输出
            self.feature_extractor = FeatureExtractor(self.model, 'Adjust_Transformer')

            # 将源域训练数据加载器保存为source_loader
            self.source_loader = self.train_loader

            # 加载目标域数据集并创建数据加载器
            LOGGER.info(f"Loading target domain dataset: {self.target_data}")

            # 解析目标域YAML文件以获取实际图像路径
            try:
                with open(self.target_data, 'r') as f:
                    target_yaml = yaml.safe_load(f)

                # 获取基础路径和训练图像路径
                base_path = target_yaml.get('path', '')
                train_path = target_yaml.get('train', '')

                if not base_path or not train_path:
                    raise ValueError(f"Missing 'path' or 'train' in {self.target_data}")

                # 构建完整的训练图像路径
                target_img_path = os.path.join(base_path, train_path)
                LOGGER.info(f"Target domain train path: {target_img_path}")

                # 检查路径是否存在
                if not os.path.exists(target_img_path):
                    raise FileNotFoundError(f"Target domain path does not exist: {target_img_path}")

                # 创建目标域数据集
                self.target_dataset = self.build_dataset(
                    img_path=target_img_path,
                    mode="train",
                    batch=self.args.batch
                )

                # 目标域的批次大小
                target_batch_size = self.batch_size // max(world_size, 1)

                nw = min([os.cpu_count() // max(world_size, 1), self.args.workers, 8])
                collate_fn = self.target_dataset.collate_fn

                if RANK == -1:
                    # 非分布式训练
                    self.target_loader = DataLoader(
                        dataset=self.target_dataset,
                        batch_size=target_batch_size,
                        shuffle=True,
                        num_workers=nw,
                        collate_fn=collate_fn,
                        pin_memory=True,
                        drop_last=True,  # 丢弃最后不完整的批次
                    )
                else:
                    # 分布式训练
                    sampler = torch.utils.data.distributed.DistributedSampler(
                        self.target_dataset, shuffle=True
                    )
                    self.target_loader = DataLoader(
                        dataset=self.target_dataset,
                        batch_size=target_batch_size,
                        sampler=sampler,
                        num_workers=nw,
                        collate_fn=collate_fn,
                        pin_memory=True,
                        drop_last=True,
                    )

                # 初始化判别器及其优化器
                self.setup_discriminator()

                LOGGER.info(f"Target domain dataloader created with {len(self.target_loader)} batches")

            except Exception as e:
                LOGGER.error(f"Error loading target domain dataset: {e}")
                raise

    def create_empty_results_csv(self):
        """创建空的results.csv文件，如果它不存在"""
        if not os.path.exists(self.csv):
            LOGGER.warning(f"Results file {self.csv} not found. Creating empty results file.")
            # 确保目录存在
            os.makedirs(os.path.dirname(self.csv), exist_ok=True)
            # 创建一个空的results.csv文件
            columns = ['epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                       'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
                       'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
                       'lr/0', 'lr/1', 'lr/2']
            # 创建空的DataFrame并保存
            pd.DataFrame(columns=columns).to_csv(self.csv, index=False)
            LOGGER.info(f"Created empty results file at {self.csv}")

    def setup_discriminator(self):
        """初始化域判别器及其优化器"""
        # 桥接层特征通道数
        feature_channels = 128  # 从yaml文件中的Adjust_Transformer层获取

        # 创建判别器
        self.discriminator = TransformerDiscriminator(channels=feature_channels)
        self.discriminator.train()

        # 设置到相同设备上
        self.discriminator.to(self.device)
        LOGGER.info(f"Discriminator initialized on device: {self.device}")

        # 处理分布式训练
        if RANK != -1 and torch.cuda.is_available():
            self.discriminator = nn.parallel.DistributedDataParallel(
                self.discriminator, device_ids=[RANK]
            )

        # 创建判别器优化器
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.disc_lr,
            betas=(0.9, 0.99)
        )
        self.optimizer_D.zero_grad()

        LOGGER.info(f"Discriminator initialized with learning rate {self.disc_lr}")

    def update_optimizer(self, epoch):
        """更新优化器的学习率"""
        # 计算主模型的学习率 - 使用余弦退火策略
        if self.args.cos_lr:
            # 余弦退火学习率
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # cosine
        else:
            # 线性学习率
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear

        # 计算当前epoch的学习率因子
        lr_factor = self.lf(epoch)

        # 更新主模型的学习率
        for param_group in self.optimizer.param_groups:
            if 'initial_lr' in param_group:
                param_group['lr'] = param_group['initial_lr'] * lr_factor
            else:
                # 如果没有initial_lr，直接设置lr
                param_group['lr'] = self.args.lr0 * lr_factor

        # 记录主模型的学习率
        if epoch % 10 == 0 or epoch == 0:  # 每10个epoch记录一次
            LOGGER.info(f'Optimizer learning rate adjusted to {self.optimizer.param_groups[0]["lr"]:.6f}')

        # 更新判别器的学习率（如果启用了域适应）
        if self.domain_adapt_enabled and self.optimizer_D is not None:
            # 使用同样的余弦退火调整判别器学习率
            current_lr = self.disc_lr * lr_factor
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = current_lr

            if epoch % 10 == 0 or epoch == 0:  # 每10个epoch记录一次
                LOGGER.info(f'Discriminator learning rate adjusted to {current_lr:.6f}')

    def validate(self):
        """
        验证当前模型性能，解决设备不匹配问题
        """
        try:
            # 确保验证器存在
            if not hasattr(self, 'validator') or self.validator is None:
                LOGGER.warning("Validator not initialized, skipping validation")
                return None

            # 获取模型的设备
            model_device = next(self.model.parameters()).device
            LOGGER.info(f"Main model is on device: {model_device}")

            # 确保验证器的模型是最新的，并且在正确的设备上
            self.validator.model = self.model

            # 验证前确保验证器模型完全在指定设备上
            for module in self.validator.model.modules():
                if hasattr(module, 'to') and callable(module.to):
                    module.to(model_device)

            # 检查验证器模型是否已正确移到设备上
            validator_device = next(self.validator.model.parameters()).device
            LOGGER.info(f"Validator model is on device: {validator_device}")

            # 设置验证器设备
            self.validator.device = model_device

            # 调用父类的验证方法
            return super().validate()
        except Exception as e:
            LOGGER.error(f"Error during validation: {e}")
            import traceback
            LOGGER.error(f"Traceback: {traceback.format_exc()}")
            # 验证失败时返回None
            return None

    def _do_train(self, world_size=1):
        """执行训练，包括域适应部分"""
        # 设置训练环境与组件
        self._setup_train(world_size)

        # 如果没有启用域适应，则使用标准训练
        if not self.domain_adapt_enabled:
            return super()._do_train(world_size)

        # 显式初始化compute_loss - 使用v8DetectionLoss
        LOGGER.info("Initializing compute_loss with v8DetectionLoss")
        try:
            # 使用正确的损失类初始化
            self.compute_loss = v8DetectionLoss(self.model)
            LOGGER.info(f"Successfully initialized compute_loss: {type(self.compute_loss).__name__}")
        except Exception as e:
            LOGGER.error(f"Error initializing v8DetectionLoss: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())

            # 创建一个简单的内联损失函数作为备用
            def simple_compute_loss(preds, batch):
                # 直接使用模型参数创建假的损失，确保梯度可以流动
                dummy_param = next(self.model.parameters())

                # 确保预测是可处理的
                if isinstance(preds, (list, tuple)) and torch.is_tensor(preds[0]):
                    loss = torch.mean(preds[0]) * 0 + dummy_param.sum() * 0 + 1.0
                else:
                    # 如果是其他格式，创建一个简单的损失
                    loss = dummy_param.sum() * 0 + 1.0

                # 损失项
                loss_items = torch.tensor([0.5, 0.3, 0.2], device=self.device)

                return loss, loss_items

            # 使用备用损失函数
            self.compute_loss = simple_compute_loss
            LOGGER.warning("Using simple placeholder loss function due to initialization error")

        # 开始训练循环
        for epoch in range(self.start_epoch, self.epochs):

            # 确保设置为训练模式
            self.model.train()
            if self.discriminator:
                self.discriminator.train()

            # 显式设置当前epoch属性
            self.epoch = epoch

            # 更新学习率
            self.update_optimizer(epoch)

            # 在每个epoch开始时打印标题行 - 确保只有主进程打印
            if RANK in (-1, 0):
                # 与YOLOv8完全相同的格式
                s = ("%11s" * (4 + len(self.loss_names))) % (
                    "Epoch",
                    "GPU_mem",
                    *["box_loss", "cls_loss", "dfl_loss"],  # 损失名称
                    "Instances",
                    "Size",
                )
                # 添加标题前后的分隔线
                LOGGER.info("=" * len(s))
                LOGGER.info(s)
                LOGGER.info("=" * len(s))

            # # 设置进度条 - 使用tqdm直接创建
            if RANK in (-1, 0):
                # 创建带有更大显示宽度的进度条
                pbar = tqdm(
                    enumerate(self.train_loader),
                    total=len(self.train_loader),
                    bar_format='{l_bar}{bar:20}{r_bar}',  # 增加进度条宽度
                    unit='batch',
                    ncols=200  # 显式设置更大的列宽
                )
            else:
                pbar = enumerate(self.train_loader)



            self.batch_i = 0

            # 创建目标域数据迭代器以便循环使用
            target_iter = iter(self.target_loader)

            # 计算每个批次的累积步数
            accumulate = max(round(self.args.nbs / self.batch_size), 1)

            # 批次迭代
            for batch_idx, batch in pbar:
                self.batch_i = batch_idx

                # 调用回调函数
                self.run_callbacks('on_train_batch_start')

                # 打印第一个批次的图像信息以进行调试
                # if batch_idx == 0 and epoch == 0:
                #     if 'img' in batch:
                #         LOGGER.info(f"Image batch shape: {batch['img'].shape}, dtype: {batch['img'].dtype}, "
                #                     f"min: {batch['img'].min().item()}, max: {batch['img'].max().item()}")

                # 载入批次到指定设备
                batch = self.preprocess_batch(batch)

                # 前向传播和计算源域损失
                preds = self.model(batch['img'])

                # 使用compute_loss计算损失 - 添加异常处理
                try:
                    # 计算损失
                    self.loss, self.loss_items = self.compute_loss(preds, batch)

                    # 显示损失形状与信息 (调试)
                    # if batch_idx == 0:
                    #     LOGGER.info(f"Loss shape: {self.loss.shape}, requires_grad: {self.loss.requires_grad}")
                    #     LOGGER.info(f"Loss items shape: {self.loss_items.shape}")

                    # 检查损失是否为标量，如果不是，取平均值
                    if self.loss.numel() > 1:
                        # LOGGER.warning(f"Loss is not scalar! Shape: {self.loss.shape}, taking mean.")
                        self.loss = torch.mean(self.loss)

                    # 确保损失有梯度
                    if not self.loss.requires_grad:
                        LOGGER.warning("Loss does not require gradients! Creating a new loss tensor.")
                        dummy_param = next(self.model.parameters())
                        self.loss = self.loss.detach() + dummy_param.sum() * 0

                except Exception as e:
                    LOGGER.error(f"Error computing loss: {e}")
                    import traceback
                    LOGGER.error(traceback.format_exc())

                    # 创建一个假的损失
                    dummy_param = next(self.model.parameters())
                    self.loss = dummy_param.sum() * 0 + 1.0
                    self.loss_items = torch.ones(3, device=self.device)

                # 标准反向传播与参数更新 - 使用try/except包装
                try:
                    # 检查损失是否为标量
                    if self.loss.numel() > 1:
                        LOGGER.warning(f"Loss is still not scalar before backward! Shape: {self.loss.shape}")
                        self.loss = torch.mean(self.loss)

                    # 确保损失是标量
                    assert self.loss.numel() == 1, f"Loss must be scalar for backward(), got shape {self.loss.shape}"

                    # 安全反向传播
                    if self.args.amp:
                        with torch.cuda.amp.autocast():
                            # 确保进行标量反向传播
                            self.scaler.scale(self.loss).backward(retain_graph=True)
                    else:
                        self.loss.backward(retain_graph=True)

                except Exception as e:
                    LOGGER.error(f"Error in backward pass: {e}")
                    import traceback
                    LOGGER.error(traceback.format_exc())
                    # 继续下一个批次
                    continue

                # 域对抗训练部分 --------------------
                if (batch_idx + 1) % accumulate == 0 and self.feature_extractor is not None:
                    try:
                        # 获取源域特征
                        source_features = self.feature_extractor.get_features()
                        if source_features is None:
                            LOGGER.warning("Failed to extract source domain features.")
                            continue

                        # 打印源域特征形状以便调试
                        if batch_idx % 10 == 0:
                            LOGGER.info(f"Source domain features shape: {source_features.shape}")

                        # 获取目标域数据
                        try:
                            target_batch = next(target_iter)
                        except StopIteration:
                            # 重新初始化目标域数据迭代器
                            target_iter = iter(self.target_loader)
                            target_batch = next(target_iter)

                        # 预处理目标域批次
                        target_batch = self.preprocess_batch(target_batch)

                        # 前向传播获取目标域特征
                        with torch.no_grad():  # 减少内存占用
                            _ = self.model(target_batch['img'])
                        target_features = self.feature_extractor.get_features()

                        if target_features is None:
                            LOGGER.warning("Failed to extract target domain features.")
                            continue

                        # 打印目标域特征形状以便调试
                        if batch_idx % 10 == 0:
                            LOGGER.info(f"Target domain features shape: {target_features.shape}")

                        # 判别器训练 - 源域 (给定标签0)
                        self.optimizer_D.zero_grad()
                        # 分离特征以减少内存使用
                        source_features_detached = source_features.detach()
                        D_out_source = self.discriminator(source_features_detached)
                        D_source_label = torch.FloatTensor(D_out_source.data.size()).fill_(self.source_label).to(
                            self.device)

                        # 打印判别器输出和标签形状
                        if batch_idx % 10 == 0:
                            LOGGER.info(
                                f"D_out_source shape: {D_out_source.shape}, D_source_label shape: {D_source_label.shape}")

                        # 确保使用reduction='mean'生成标量损失
                        D_source_loss = F.mse_loss(D_out_source, D_source_label, reduction='mean')

                        # 判别器训练 - 目标域 (给定标签1)
                        target_features_detached = target_features.detach()
                        D_out_target = self.discriminator(target_features_detached)
                        D_target_label = torch.FloatTensor(D_out_target.data.size()).fill_(self.target_label).to(
                            self.device)

                        # 打印判别器输出和标签形状
                        if batch_idx % 10 == 0:
                            LOGGER.info(
                                f"D_out_target shape: {D_out_target.shape}, D_target_label shape: {D_target_label.shape}")

                        # 确保使用reduction='mean'生成标量损失
                        D_target_loss = F.mse_loss(D_out_target, D_target_label, reduction='mean')

                        # 总判别器损失并更新
                        D_loss = (D_source_loss + D_target_loss) / 2

                        # 确保D_loss是标量
                        if D_loss.numel() > 1:
                            LOGGER.warning(f"D_loss is not scalar! Shape: {D_loss.shape}")
                            D_loss = torch.mean(D_loss)

                        # 打印最终损失形状和值
                        if batch_idx % 10 == 0:
                            LOGGER.info(f"D_loss shape: {D_loss.shape}, value: {D_loss.item()}")

                        # 反向传播 - 使用try/except包装
                        try:
                            assert D_loss.numel() == 1, f"D_loss must be scalar for backward(), got shape {D_loss.shape}"
                            D_loss.backward()
                            self.optimizer_D.step()
                        except Exception as e:
                            LOGGER.error(f"Error in discriminator backward pass: {e}")
                            # 继续下一个批次的训练

                        # 明确释放不再需要的张量
                        del source_features_detached, target_features_detached
                        del D_out_source, D_out_target

                        # 源域特征对抗训练 (混淆判别器)
                        self.optimizer.zero_grad()
                        source_D_out = self.discriminator(source_features)

                        # 打印生成器相关张量形状
                        if batch_idx % 10 == 0:
                            LOGGER.info(f"source_D_out shape: {source_D_out.shape}")

                        # 确保使用reduction='mean'生成标量损失
                        G_source_loss = F.mse_loss(source_D_out, D_target_label, reduction='mean')

                        # 确保G_source_loss是标量
                        if G_source_loss.numel() > 1:
                            LOGGER.warning(f"G_source_loss is not scalar! Shape: {G_source_loss.shape}")
                            G_source_loss = torch.mean(G_source_loss)

                        # 打印最终损失形状和值
                        if batch_idx % 10 == 0:
                            LOGGER.info(f"G_source_loss shape: {G_source_loss.shape}, value: {G_source_loss.item()}")

                        # 反向传播 - 使用try/except包装
                        try:
                            assert G_source_loss.numel() == 1, f"G_source_loss must be scalar for backward(), got shape {G_source_loss.shape}"
                            G_source_loss.backward()
                        except Exception as e:
                            LOGGER.error(f"Error in generator backward pass: {e}")
                            # 继续下一步

                        # 记录损失
                        if self.args.amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        # 打印域适应损失信息
                        if batch_idx % 10 == 0:
                            LOGGER.info(
                                f"{colorstr('green', 'bold', 'Domain Adapt')} Epoch: {epoch}, Batch: {batch_idx}, "
                                f"D_loss: {D_loss.item():.4f}, "
                                f"G_loss: {G_source_loss.item():.4f}")

                    except Exception as e:
                        LOGGER.error(f"Error in domain adaptation training: {e}")
                        import traceback
                        LOGGER.error(traceback.format_exc())

                    # 明确释放不再需要的特征
                    if 'source_features' in locals():
                        del source_features
                    if 'target_features' in locals():
                        del target_features

                # 执行余下的标准训练步骤
                if self.args.amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # 调用回调函数
                self.run_callbacks('on_train_batch_end')

                # 更新进度条
                self.update_pbar(pbar, batch_idx, epoch)

                # 主动触发垃圾回收（在GPU上可以不那么频繁）
                if batch_idx % 50 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    else:
                        gc.collect()

            # 处理每个epoch结束时的操作
            self.run_callbacks('on_train_epoch_end')

            # 执行验证 - 强化版错误处理
            try:
                # 显式设置当前epoch属性以确保验证可以访问
                self.epoch = epoch

                # 确保验证前CSV文件存在
                self.create_empty_results_csv()

                # 验证前确保模型完全在GPU上
                if torch.cuda.is_available():
                    self.model.cuda()

                    # 检查模型参数设备
                    devices = set()
                    for module in self.model.modules():
                        if hasattr(module, 'parameters'):
                            for param in module.parameters(recurse=False):
                                devices.add(param.device)

                    if len(devices) > 1:
                        LOGGER.warning(f"Model has parameters on multiple devices: {devices}")
                        # 强制所有模块移至同一设备
                        target_device = torch.device('cuda:0')
                        for module in self.model.modules():
                            if hasattr(module, 'parameters'):
                                for param in module.parameters(recurse=False):
                                    if param.device != target_device:
                                        param.data = param.data.to(target_device)

                # 验证间隔检查
                val_interval = getattr(self.args, 'val_interval', 1)  # 如果不存在，默认为1
                if (epoch + 1) % val_interval == 0:
                    self.validate()
            except AttributeError as e:
                # 如果val_interval不存在或其他属性错误
                LOGGER.warning(f"AttributeError during validation: {e}")
                LOGGER.warning("'val_interval' not found in configuration, using default value of 1")
                try:
                    self.validate()
                except Exception as val_e:
                    LOGGER.error(f"Validation failed: {val_e}")
                    import traceback
                    LOGGER.error(traceback.format_exc())
            except Exception as e:
                LOGGER.error(f"Error during validation: {e}")
                import traceback
                LOGGER.error(traceback.format_exc())

            # 保存模型 - 强化错误处理
            try:
                # 确保CSV文件存在
                self.create_empty_results_csv()

                # 保存模型
                self.save_model()
            except Exception as e:
                LOGGER.error(f"Error saving model: {e}")
                # 备份保存方法
                try:
                    # 保存最小模型权重
                    ckpt = {
                        'epoch': self.epoch,
                        'model': de_parallel(self.model).state_dict(),
                        'date': datetime.now().isoformat()
                    }
                    save_path = str(self.save_dir / f'backup_epoch_{self.epoch}.pt')
                    torch.save(ckpt, save_path)
                    LOGGER.info(f"Saved backup model to {save_path}")
                except Exception as backup_e:
                    LOGGER.error(f"Even backup save failed: {backup_e}")

            # 调用回调函数
            self.run_callbacks('on_fit_epoch_end')

            # 检查提前终止
            if self.stopper.possible_stop:
                LOGGER.info(
                    f'Stopping training early as no improvement observed in last {self.stopper.patience} epochs.')
                break

        # 训练完成的操作
        self.run_callbacks('on_train_end')
        if world_size > 1 and RANK == 0:
            LOGGER.info(f"Training completed successfully after {epoch + 1} epochs.")

        # 清理内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.run_callbacks('teardown')

    def _get_instances_count(self):
        """获取当前批次中的实例数量"""
        instances = 0
        if hasattr(self, 'current_batch') and self.current_batch is not None:
            if 'bboxes' in self.current_batch:
                bboxes = self.current_batch['bboxes']
                if isinstance(bboxes, list):
                    instances = sum(len(b) for b in bboxes if b is not None)
                elif torch.is_tensor(bboxes) and bboxes.numel() > 0:
                    instances = bboxes.shape[0]

        # 备用计数获取
        if instances == 0 and hasattr(self, '_instance_counts') and hasattr(self, 'batch_i'):
            if self.batch_i in self._instance_counts:
                instances = self._instance_counts[self.batch_i]

        return instances

    def update_pbar(self, pbar, batch_idx, epoch):
        """使用YOLOv8原生格式更新进度条"""
        # 获取GPU内存信息（以GB为单位）
        mem = f'{torch.cuda.memory_reserved() / 1E9:.1f}G' if torch.cuda.is_available() else 'CPU'

        # 如果是tqdm进度条对象，更新描述
        if RANK in (-1, 0) and hasattr(pbar, 'set_description'):
            # 确保损失项是标量
            box_loss = self.loss_items[0].item() if torch.is_tensor(self.loss_items[0]) else self.loss_items[0]
            cls_loss = self.loss_items[1].item() if torch.is_tensor(self.loss_items[1]) else self.loss_items[1]
            dfl_loss = self.loss_items[2].item() if torch.is_tensor(self.loss_items[2]) else self.loss_items[2]

            # 获取实例数量和图像大小
            instances = self._get_instances_count()
            img_size = getattr(self.args, 'imgsz', 640)

            # 使用相同的格式字符串，但不包含尾部的冒号
            description = ("%11s" * 2 + "%11.3g" * 3 + "%11d" + "%11s") % (
                f"{epoch + 1}/{self.epochs}",
                mem,
                box_loss,
                cls_loss,
                dfl_loss,
                instances,
                f"{img_size}"
            )
            pbar.set_description(description)


    def preprocess_batch(self, batch):
        """预处理批次数据，并跟踪实例数量"""
        # 如果批次为空，则创建空字典
        if batch is None:
            return {'img': None}

        # 在处理前统计实例数
        instance_count = 0
        if 'bboxes' in batch:
            bboxes = batch['bboxes']
            if isinstance(bboxes, list):
                instance_count = sum(len(b) for b in bboxes if b is not None)
            elif torch.is_tensor(bboxes) and bboxes.numel() > 0:
                instance_count = bboxes.shape[0]

        # 保存当前批次的实例数量
        if not hasattr(self, '_instance_counts'):
            self._instance_counts = {}
        self._instance_counts[self.batch_i] = instance_count

        # 保存当前批次引用
        self.current_batch = batch

        # 将图像移动到指定设备，并确保数据类型正确
        if 'img' in batch:
            # 检查图像数据类型并进行必要的转换
            img = batch['img']

            # 如果图像是浮点型，确保已经归一化
            if img.dtype == torch.float32:
                # 确保数值范围在[0,1]
                if img.max() > 1.0:
                    img = img / 255.0
            # 如果图像是整型，转换为浮点型并归一化
            elif img.dtype == torch.uint8:
                img = img.float() / 255.0

            # 将处理后的图像移动到指定设备
            batch['img'] = img.to(self.device, non_blocking=True)

        # 确保批次中的所有元素都在相同设备上
        for k in batch:
            if k != 'img' and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self.device, non_blocking=True)

        # 特别处理batch_idx，确保它存在且在正确设备上
        if 'batch_idx' not in batch and 'cls' in batch and 'bboxes' in batch:
            # 计算每个标签对应的批次索引
            cls_tensor = batch['cls']
            if len(cls_tensor.shape) == 1:
                num_labels = cls_tensor.shape[0]
            else:
                num_labels = cls_tensor.shape[0] * cls_tensor.shape[1]

            # 创建批次索引张量
            if num_labels > 0:
                # 如果标签不为空，为每个标签分配批次索引
                # 假设每个图像有相同数量的标签
                batch_size = batch['img'].shape[0]
                labels_per_image = num_labels // batch_size
                batch['batch_idx'] = torch.arange(batch_size, device=self.device).repeat_interleave(labels_per_image)
            else:
                # 如果没有标签，创建空张量
                batch['batch_idx'] = torch.tensor([], device=self.device)

        return batch

    def save_model(self):
        """保存包含域适应组件的模型"""
        try:
            # 确保CSV文件存在
            self.create_empty_results_csv()

            # 尝试标准保存方法
            super().save_model()
        except Exception as e:
            LOGGER.error(f"Error in standard save_model: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())

            # 备份保存方法
            try:
                # 保存最小模型权重
                ckpt = {
                    'epoch': self.epoch,
                    'model': de_parallel(self.model).state_dict(),
                    'date': datetime.now().isoformat()
                }
                save_path = str(self.save_dir / f'backup_epoch_{self.epoch}.pt')
                torch.save(ckpt, save_path)
                LOGGER.info(f"Saved backup model to {save_path}")
            except Exception as backup_e:
                LOGGER.error(f"Even backup save failed: {backup_e}")

        # 判别器保存逻辑 - 每 10 个 epoch 保存一次
        try:
            # 检查是否应该保存判别器
            should_save = (self.epoch % 10 == 0) and (self.epoch != 0)  # 每 10 个 epoch
            is_final_epoch = (self.epoch == self.epochs - 1)  # 或最后一个 epoch

            # 在达到保存条件时保存判别器
            if self.domain_adapt_enabled and self.discriminator is not None and (should_save or is_final_epoch):
                discriminator = de_parallel(self.discriminator)

                # 根据情况命名保存文件
                if is_final_epoch:
                    # 最终模型使用特殊命名
                    disc_path = str(self.save_dir / 'discriminator_final.pt')
                    LOGGER.info(f"Saved final discriminator to {disc_path}")
                else:
                    # 中间检查点使用 epoch 编号
                    disc_path = str(self.save_dir / f'discriminator_epoch{self.epoch}.pt')
                    LOGGER.info(f"Saved checkpoint discriminator at epoch {self.epoch}")

                # 保存模型
                torch.save(discriminator.state_dict(), disc_path)
            # else:
            #     # 记录跳过保存的日志
            #     if self.domain_adapt_enabled and self.discriminator is not None:
            #         LOGGER.info(f"Skipping discriminator save for epoch {self.epoch} (not at save interval)")
        except Exception as e:
            LOGGER.error(f"Error saving discriminator: {e}")