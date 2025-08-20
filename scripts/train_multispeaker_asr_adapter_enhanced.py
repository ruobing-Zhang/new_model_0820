# train_multispeaker_asr_adapter_enhanced.py
import os
import json
import argparse
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.common.parts import adapter_modules
from nemo.core.config import hydra_runner
from nemo.utils import logging

from model_with_spk_inject_adapter import RNNTWithSpkInjectAdapter, ManifestSpeakerDataset


class LitRNNTWithSpkAdapter(pl.LightningModule):
    """
    PyTorch Lightning 封装的 RNNT + 说话人注入 + Adapter 模型
    """
    
    def __init__(self, base_model_path: str, K_max: int = 4, alpha_init: float = 0.1,
                 lr: float = 1e-4, weight_decay: float = 1e-5,
                 use_adapter: bool = True, adapter_cfg: Optional[Dict] = None):
        super().__init__()
        self.save_hyperparameters()
        
        # 加载基础模型
        logging.info(f"从 {base_model_path} 加载基础模型...")
        base_model = EncDecRNNTBPEModel.restore_from(base_model_path, map_location='cpu')
        
        # 创建带说话人注入和adapter支持的模型
        self.model = RNNTWithSpkInjectAdapter(
            base_model=base_model,
            K_max=K_max,
            alpha_init=alpha_init
        )
        
        # 添加adapter支持
        self.use_adapter = use_adapter
        if use_adapter and hasattr(self.model.encoder, 'add_adapter'):
            logging.info("检测到ConformerEncoderAdapter，开始添加线性adapter...")
            
            # 处理adapter配置
            if adapter_cfg is None:
                # 默认adapter配置
                adapter_cfg = {
                    "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
                    "in_features": self.model.encoder.d_model,
                    "dim": 128,  # adapter隐藏层维度
                    "activation": "swish",
                    "norm_position": "pre",
                    "dropout": 0.1
                }
            else:
                # 确保必要的字段存在
                if "_target_" not in adapter_cfg:
                    adapter_cfg["_target_"] = "nemo.collections.common.parts.adapter_modules.LinearAdapter"
                if "in_features" not in adapter_cfg:
                    adapter_cfg["in_features"] = self.model.encoder.d_model
            
            logging.info(f"使用adapter配置: {adapter_cfg}")
            
            try:
                from omegaconf import DictConfig
                
                # 转换为DictConfig以避免unhashable type错误
                if not isinstance(adapter_cfg, DictConfig):
                    adapter_cfg = DictConfig(adapter_cfg)
                
                # 添加adapter到每个ConformerLayer
                self.model.encoder.add_adapter(
                    name="linear_adapter",
                    cfg=adapter_cfg
                )
                
                # 启用adapter
                self.model.encoder.set_enabled_adapters(["linear_adapter"], enabled=True)
                
                logging.info("成功添加并启用线性adapter")
                
                # 冻结所有模型参数
                for param in self.model.parameters():
                    param.requires_grad = False
                
                # 解冻adapter参数
                adapter_params = 0
                for name, param in self.model.named_parameters():
                    if 'adapter' in name.lower():
                        param.requires_grad = True
                        adapter_params += param.numel()
                        logging.debug(f"解冻adapter参数: {name}, shape: {param.shape}")
                
                # 解冻说话人注入参数
                self.model.alpha.requires_grad = True
                alpha_params = self.model.alpha.numel()
                
                # 解冻decoder和joint参数
                decoder_params = 0
                for param in self.model.decoder.parameters():
                    param.requires_grad = True
                    decoder_params += param.numel()
                
                joint_params = 0
                for param in self.model.joint.parameters():
                    param.requires_grad = True
                    joint_params += param.numel()
                
                logging.info(f"参数统计:")
                logging.info(f"  - Adapter参数: {adapter_params:,}")
                logging.info(f"  - 说话人注入参数(alpha): {alpha_params:,}")
                logging.info(f"  - Decoder参数: {decoder_params:,}")
                logging.info(f"  - Joint参数: {joint_params:,}")
                logging.info(f"  - 总可训练参数: {adapter_params + alpha_params + decoder_params + joint_params:,}")
                
            except Exception as e:
                logging.error(f"添加adapter失败: {e}")
                logging.info("回退到说话人注入模式")
                self.use_adapter = False
                self._setup_fallback_training()
        
        elif use_adapter:
            logging.warning("编码器不支持add_adapter方法，回退到说话人注入模式")
            self.use_adapter = False
            self._setup_fallback_training()
        else:
            logging.info("未启用adapter，使用说话人注入模式")
            self._setup_fallback_training()
        
        # 统计总参数
        self._log_parameter_stats()
    
    def _setup_fallback_training(self):
        """设置fallback训练模式（仅训练说话人注入参数）"""
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 只解冻说话人注入参数
        self.model.alpha.requires_grad = True
        
        # 解冻decoder和joint参数
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        
        for param in self.model.joint.parameters():
            param.requires_grad = True
        
        logging.info("Fallback模式：仅训练说话人注入、decoder和joint参数")
    
    def _log_parameter_stats(self):
        """统计并记录参数信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 按模块分类统计
        encoder_trainable = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)
        decoder_trainable = sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)
        joint_trainable = sum(p.numel() for p in self.model.joint.parameters() if p.requires_grad)
        alpha_trainable = self.model.alpha.numel() if self.model.alpha.requires_grad else 0
        
        logging.info("=" * 60)
        logging.info("模型参数统计:")
        logging.info(f"  总参数: {total_params:,}")
        logging.info(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        logging.info(f"  冻结参数: {total_params-trainable_params:,} ({100*(total_params-trainable_params)/total_params:.2f}%)")
        logging.info("按模块分类:")
        logging.info(f"  - Encoder可训练: {encoder_trainable:,}")
        logging.info(f"  - Decoder可训练: {decoder_trainable:,}")
        logging.info(f"  - Joint可训练: {joint_trainable:,}")
        logging.info(f"  - Alpha可训练: {alpha_trainable:,}")
        logging.info("=" * 60)
    
    def forward(self, input_signal, input_signal_length, targets, target_length, spk_labels=None):
        """前向传播"""
        # 编码并注入说话人信息
        encoded, encoded_len = self.model.encode_with_injection(
            input_signal, input_signal_length, spk_labels
        )
        
        # 解码 - 按照标准RNNT模型的方式调用decoder
        decoder_outputs, target_length_out, states = self.model.decoder(targets=targets, target_length=target_length)
        
        # 调试信息：打印形状
        print(f"DEBUG: encoded shape: {encoded.shape}")
        print(f"DEBUG: decoder_outputs shape: {decoder_outputs.shape}")
        print(f"DEBUG: target_length: {target_length}")
        print(f"DEBUG: target_length_out: {target_length_out}")
        
        # Joint网络 - 只传递encoder和decoder输出
        joint_outputs = self.model.joint(
            encoder_outputs=encoded,
            decoder_outputs=decoder_outputs
        )
        
        return joint_outputs
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        input_signal = batch['input_signal']
        input_signal_length = batch['input_signal_length']
        targets = batch['targets']
        target_length = batch['target_length']
        spk_labels = batch.get('spk_labels', None)
        
        # 前向传播
        joint_outputs = self.forward(
            input_signal, input_signal_length, targets, target_length, spk_labels
        )
        
        # 打印调试信息
        print(f"Joint outputs shape: {joint_outputs.shape}")
        print(f"Encoded length: {batch['encoded_length']}")
        print(f"Target length: {target_length}")
        
        # 打印调试信息
        print(f"Validation - Joint outputs shape: {joint_outputs.shape}")
        print(f"Validation - Encoded length: {batch['encoded_length']}")
        print(f"Validation - Target length: {target_length}")
        
        # 计算损失
        loss = self.model.loss(
            log_probs=joint_outputs,
            targets=targets,
            input_lengths=batch['encoded_length'],
            target_lengths=target_length
        )
        
        # 记录损失
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('alpha', self.model.alpha.item(), on_step=True, on_epoch=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        input_signal = batch['input_signal']
        input_signal_length = batch['input_signal_length']
        targets = batch['targets']
        target_length = batch['target_length']
        spk_labels = batch.get('spk_labels', None)
        
        # 前向传播
        joint_outputs = self.forward(
            input_signal, input_signal_length, targets, target_length, spk_labels
        )
        
        # 计算损失
        loss = self.model.loss(
            log_probs=joint_outputs,
            targets=targets,
            input_lengths=batch['encoded_length'],
            target_lengths=target_length
        )
        
        # 记录损失
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """配置优化器"""
        # 收集可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if not trainable_params:
            logging.warning("没有可训练参数！")
            # 强制至少训练alpha参数
            self.model.alpha.requires_grad = True
            trainable_params = [self.model.alpha]
        
        # 按模块分组参数（用于不同学习率）
        encoder_params = []
        decoder_params = []
        joint_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'encoder' in name:
                    encoder_params.append(param)
                elif 'decoder' in name:
                    decoder_params.append(param)
                elif 'joint' in name:
                    joint_params.append(param)
                else:
                    other_params.append(param)
        
        # 创建参数组
        param_groups = []
        if encoder_params:
            param_groups.append({'params': encoder_params, 'lr': self.hparams.lr, 'name': 'encoder'})
        if decoder_params:
            param_groups.append({'params': decoder_params, 'lr': self.hparams.lr, 'name': 'decoder'})
        if joint_params:
            param_groups.append({'params': joint_params, 'lr': self.hparams.lr, 'name': 'joint'})
        if other_params:
            param_groups.append({'params': other_params, 'lr': self.hparams.lr, 'name': 'other'})
        
        # 如果没有参数组，使用所有可训练参数
        if not param_groups:
            param_groups = trainable_params
        
        logging.info(f"优化器参数组:")
        for i, group in enumerate(param_groups):
            if isinstance(group, dict):
                num_params = sum(p.numel() for p in group['params'])
                logging.info(f"  组 {i} ({group.get('name', 'unnamed')}): {num_params:,} 参数, lr={group['lr']}")
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def create_dummy_batch(batch_size: int = 2, seq_len: int = 320, vocab_size: int = 1000):
    """创建模拟批次数据用于测试"""
    # 确保targets索引在词汇表范围内（0到vocab_size-1）
    # 使用更小的序列长度避免维度问题
    encoded_len = seq_len // 8  # 8倍下采样，更符合实际模型
    target_len = 5  # 固定为5，避免过长的序列
    
    return {
        'input_signal': torch.randn(batch_size, seq_len),
        'input_signal_length': torch.tensor([seq_len] * batch_size),
        'targets': torch.randint(0, min(vocab_size, 999), (batch_size, target_len)),  # 确保不超过词汇表大小
        'target_length': torch.tensor([target_len] * batch_size),
        'encoded_length': torch.tensor([encoded_len] * batch_size),  # 8倍下采样
        'spk_labels': torch.randn(batch_size, 4, encoded_len)  # (B, K=4, T_enc)
    }


def main():
    parser = argparse.ArgumentParser(description='训练多说话人ASR模型（支持Adapter）')
    parser.add_argument('--base_model', type=str, required=True,
                        help='基础模型路径 (.nemo文件)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='最大训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--use_adapter', action='store_true',
                        help='是否使用adapter微调')
    parser.add_argument('--adapter_dim', type=int, default=128,
                        help='Adapter隐藏层维度')
    parser.add_argument('--test_mode', action='store_true',
                        help='测试模式（使用模拟数据）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Adapter配置
    adapter_cfg = None
    if args.use_adapter:
        from omegaconf import DictConfig
        adapter_cfg = DictConfig({
            "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
            "dim": args.adapter_dim,
            "activation": "swish",
            "norm_position": "pre",
            "dropout": 0.1
        })
    
    # 创建模型
    model = LitRNNTWithSpkAdapter(
        base_model_path=args.base_model,
        K_max=4,
        alpha_init=0.1,
        lr=args.lr,
        weight_decay=1e-5,
        use_adapter=args.use_adapter,
        adapter_cfg=adapter_cfg
    )
    
    if args.test_mode:
        logging.info("测试模式：使用模拟数据进行训练")
        
        # 创建模拟数据加载器
        class DummyDataLoader:
            def __init__(self, batch_size, num_batches=10):
                self.batch_size = batch_size
                self.num_batches = num_batches
                self.current = 0
            
            def __iter__(self):
                self.current = 0
                return self
            
            def __next__(self):
                if self.current >= self.num_batches:
                    raise StopIteration
                self.current += 1
                return create_dummy_batch(self.batch_size)
            
            def __len__(self):
                return self.num_batches
        
        train_loader = DummyDataLoader(args.batch_size, 20)
        val_loader = DummyDataLoader(args.batch_size, 5)
        
        # 配置回调
        callbacks = [
            ModelCheckpoint(
                dirpath=args.output_dir,
                filename='best-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min',
                save_top_k=3
            ),
            LearningRateMonitor(logging_interval='step')
        ]
        
        # 配置日志
        logger = TensorBoardLogger(
            save_dir=args.output_dir,
            name='multispeaker_asr_adapter'
        )
        
        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator='auto',
            devices=1,
            precision=16,
            gradient_clip_val=1.0,
            log_every_n_steps=10,
            val_check_interval=0.5
        )
        
        # 开始训练
        logging.info("开始训练...")
        trainer.fit(model, train_loader, val_loader)
        
        logging.info(f"训练完成！模型保存在: {args.output_dir}")
    
    else:
        logging.info("实际训练模式需要提供真实数据加载器")
        logging.info("请实现数据加载逻辑或使用 --test_mode 进行测试")


if __name__ == '__main__':
    main()