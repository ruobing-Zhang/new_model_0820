#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapter微调训练脚本

功能说明：
1. 初始化带说话人注入和Adapter的字符级RNNT模型
2. 加载预训练模型权重
3. 在词表中添加说话人标记 <|spk0|>, <|spk1|>, <|spk2|>, <|spk3|>
4. 冻结encoder及之前的模块权重
5. 训练adapter、说话人注入、解码器和joint网络

使用方法：
python adapter_finetune_train.py --config config.yaml --manifest train_manifest.jsonl
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import librosa
from omegaconf import OmegaConf

# 导入NeMo相关模块
from nemo.collections.asr.models import EncDecRNNTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging

# 导入自定义模型
from rnnt_char_model_with_spk_inject import (
    RNNTCharWithSpkInjectAndAdapter,
    ManifestSpeakerDataset
)


class AdapterFineTuneModel(pl.LightningModule):
    """
    Adapter微调的PyTorch Lightning模块
    
    该模块封装了带说话人注入和Adapter的RNNT模型，
    实现了完整的训练、验证和测试流程。
    """
    
    def __init__(self, 
                 pretrained_model_path: str,
                 K_max: int = 4,
                 alpha_init: float = 0.1,
                 num_adapters: int = 4,
                 adapter_bottleneck: int = 256,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-6,
                 train_manifest: str = None,
                 val_manifest: str = None,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 sample_rate: int = 16000):
        """
        初始化Adapter微调模型
        
        Args:
            pretrained_model_path: 预训练模型路径
            K_max: 最大说话人数量
            alpha_init: 说话人注入强度初始值
            num_adapters: Adapter层数量
            adapter_bottleneck: Adapter瓶颈层维度
            learning_rate: 学习率
            weight_decay: 权重衰减
            train_manifest: 训练数据manifest路径
            val_manifest: 验证数据manifest路径
            batch_size: 批次大小
            num_workers: 数据加载器工作进程数
            sample_rate: 音频采样率
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        # 保存配置参数
        self.pretrained_model_path = pretrained_model_path
        self.K_max = K_max
        self.alpha_init = alpha_init
        self.num_adapters = num_adapters
        self.adapter_bottleneck = adapter_bottleneck
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        
        # 说话人标记
        self.speaker_tokens = ['<|spk0|>', '<|spk1|>', '<|spk2|>', '<|spk3|>']
        
        # 初始化模型
        self._setup_model()
        
        # 设置训练指标
        self.train_loss_history = []
        self.val_loss_history = []
        
        logging.info(f"AdapterFineTuneModel初始化完成")
        logging.info(f"模型参数总数: {sum(p.numel() for p in self.parameters()):,}")
        logging.info(f"可训练参数数: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def _setup_model(self):
        """
        设置模型：加载预训练权重、扩展词表、配置训练策略
        """
        logging.info("开始设置模型...")
        
        # 1. 加载预训练模型
        logging.info(f"加载预训练模型: {self.pretrained_model_path}")
        base_model = EncDecRNNTModel.restore_from(self.pretrained_model_path)
        
        # 2. 获取原始词表
        original_labels = base_model.cfg.labels if hasattr(base_model.cfg, 'labels') else base_model.decoder.vocabulary
        logging.info(f"原始词表大小: {len(original_labels)}")
        
        # 3. 扩展词表，添加说话人标记
        extended_labels = list(original_labels) + self.speaker_tokens
        logging.info(f"扩展后词表大小: {len(extended_labels)}")
        logging.info(f"添加的说话人标记: {self.speaker_tokens}")
        
        # 4. 更新配置中的词表
        base_model.cfg.labels = extended_labels
        # 安全地更新decoder配置
        if hasattr(base_model.cfg.decoder, 'vocabulary'):
            base_model.cfg.decoder.vocabulary = extended_labels
        if hasattr(base_model.cfg.decoder, 'vocab_size'):
            base_model.cfg.decoder.vocab_size = len(extended_labels)
        # 安全地更新joint配置
        if hasattr(base_model.cfg.joint, 'vocab_size'):
            base_model.cfg.joint.vocab_size = len(extended_labels)
        
        # 清除原始的数据配置，避免加载不存在的文件
        if hasattr(base_model.cfg, 'train_ds'):
            base_model.cfg.train_ds = None
        if hasattr(base_model.cfg, 'validation_ds'):
            base_model.cfg.validation_ds = None
        if hasattr(base_model.cfg, 'test_ds'):
            base_model.cfg.test_ds = None
        
        # 5. 创建带说话人注入和Adapter的模型
        self.model = RNNTCharWithSpkInjectAndAdapter(
            cfg=base_model.cfg,
            trainer=None,
            K_max=self.K_max,
            alpha_init=self.alpha_init,
            base_model=base_model,
            num_adapters=self.num_adapters,
            adapter_bottleneck=self.adapter_bottleneck
        )
        
        # 6. 重新初始化解码器和joint网络以适应新词表
        self._reinitialize_decoder_and_joint(base_model, extended_labels)
        
        # 7. 设置权重冻结策略
        self._setup_freezing_strategy()
        
        logging.info("模型设置完成")
    
    def _reinitialize_decoder_and_joint(self, base_model, extended_labels):
        """
        重新初始化解码器和joint网络以适应扩展后的词表
        
        注意：解码器和joint网络将使用随机初始化的权重，不复制原始权重，
        因为这些模块需要重新训练以适应新的词表和说话人标记。
        
        Args:
            base_model: 原始预训练模型
            extended_labels: 扩展后的词表
        """
        logging.info("重新初始化解码器和joint网络（使用随机权重）...")
        
        vocab_size = len(extended_labels)
        
        # 重新初始化解码器（使用随机权重）
        from nemo.collections.asr.modules import RNNTDecoder
        decoder_cfg = base_model.cfg.decoder.copy()
        
        # 过滤掉不被RNNTDecoder接受的参数
        decoder_params = {
            'vocab_size': vocab_size,
            'normalization_mode': getattr(decoder_cfg, 'normalization_mode', 'layer_norm'),
            'random_state_sampling': getattr(decoder_cfg, 'random_state_sampling', False),
            'blank_as_pad': getattr(decoder_cfg, 'blank_as_pad', True)
        }
        
        # 添加其他可能的参数
        for key in ['prednet', 'pred_hidden', 'pred_rnn_layers']:
            if hasattr(decoder_cfg, key):
                decoder_params[key] = getattr(decoder_cfg, key)
        
        self.model.decoder = RNNTDecoder(**decoder_params)
        
        # 重新初始化joint网络（使用随机权重）
        from nemo.collections.asr.modules import RNNTJoint
        joint_cfg = base_model.cfg.joint.copy()
        
        # 创建新的joint网络，明确禁用fuse_loss_wer
        jointnet_cfg = {
            'encoder_hidden': base_model.encoder.d_model,
            'pred_hidden': base_model.decoder.pred_hidden,
            'joint_hidden': getattr(joint_cfg, 'joint_hidden', 640),
            'activation': getattr(joint_cfg, 'activation', 'relu')
        }
        
        new_joint = RNNTJoint(
            jointnet=jointnet_cfg,
            num_classes=vocab_size,
            vocabulary=extended_labels,
            log_softmax=getattr(joint_cfg, 'log_softmax', None),
            preserve_memory=getattr(joint_cfg, 'preserve_memory', False),
            fuse_loss_wer=False,  # 明确禁用
            fused_batch_size=getattr(joint_cfg, 'fused_batch_size', None)
        )
        
        self.model.joint = new_joint
        
        # 不复制任何权重，解码器和joint网络将从随机初始化开始训练
        # 这样可以更好地学习新增的说话人标记和多说话人场景
        
        logging.info(f"解码器和joint网络重新初始化完成，新词表大小: {vocab_size}")
        logging.info("解码器和joint网络使用随机初始化权重，将从头开始训练")
    
    def _setup_freezing_strategy(self):
        """
        设置权重冻结策略：
        - 冻结：preprocessor, encoder
        - 可训练：adapter层, 说话人注入参数, decoder, joint网络
        """
        logging.info("设置权重冻结策略...")
        
        # 冻结预处理器
        for param in self.model.preprocessor.parameters():
            param.requires_grad = False
        
        # 冻结编码器
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        
        # 冻结损失函数（如果有参数的话）
        if hasattr(self.model, 'loss') and hasattr(self.model.loss, 'parameters'):
            for param in self.model.loss.parameters():
                param.requires_grad = False
        
        # 确保以下模块可训练：
        # 1. Adapter层
        for adapter in self.model.adapters:
            for param in adapter.parameters():
                param.requires_grad = True
        
        # 2. 说话人注入参数
        self.model.alpha.requires_grad = True
        
        # 3. 解码器
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        
        # 4. Joint网络
        for param in self.model.joint.parameters():
            param.requires_grad = True
        
        # 统计参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logging.info(f"参数统计:")
        logging.info(f"  总参数数: {total_params:,}")
        logging.info(f"  可训练参数数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logging.info(f"  冻结参数数: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    
    def setup(self, stage: str):
        """
        设置数据集
        
        Args:
            stage: 训练阶段 ('fit', 'validate', 'test')
        """
        if stage == 'fit':
            # 训练数据集
            if self.train_manifest:
                self.train_dataset = ManifestSpeakerDataset(
                    manifest_path=self.train_manifest,
                    sample_rate=self.sample_rate
                )
                logging.info(f"训练数据集大小: {len(self.train_dataset)}")
            
            # 验证数据集
            if self.val_manifest:
                self.val_dataset = ManifestSpeakerDataset(
                    manifest_path=self.val_manifest,
                    sample_rate=self.sample_rate
                )
                logging.info(f"验证数据集大小: {len(self.val_dataset)}")
    
    def train_dataloader(self):
        """
        创建训练数据加载器
        """
        if not hasattr(self, 'train_dataset'):
            return None
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """
        创建验证数据加载器
        """
        if not hasattr(self, 'val_dataset'):
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def _collate_fn(self, batch):
        """
        批次数据整理函数
        
        Args:
            batch: 批次数据列表
            
        Returns:
            整理后的批次数据
        """
        # 加载音频数据
        audio_signals = []
        audio_lengths = []
        texts = []
        spk_labels_list = []
        
        for item in batch:
            # 加载音频
            audio_path = item['audio_filepath']
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            audio_signals.append(torch.tensor(audio, dtype=torch.float32))
            audio_lengths.append(len(audio))
            
            # 文本
            texts.append(item['text'])
            
            # 加载说话人标签
            pgt_path = item['pgt_npy']
            if os.path.exists(pgt_path):
                spk_probs = np.load(pgt_path)  # 形状应该是 [K, T] 或 [T, K]
                if spk_probs.ndim == 2:
                    if spk_probs.shape[0] == self.K_max:
                        spk_probs = spk_probs.T  # 转换为 [T, K]
                    elif spk_probs.shape[1] == self.K_max:
                        pass  # 已经是 [T, K] 格式
                    else:
                        # 如果维度不匹配，创建dummy数据
                        T = len(audio) // 160  # 假设帧移为160
                        spk_probs = np.random.rand(T, self.K_max)
                        spk_probs = spk_probs / spk_probs.sum(axis=1, keepdims=True)
                else:
                    # 创建dummy数据
                    T = len(audio) // 160
                    spk_probs = np.random.rand(T, self.K_max)
                    spk_probs = spk_probs / spk_probs.sum(axis=1, keepdims=True)
            else:
                # 创建dummy数据
                T = len(audio) // 160
                spk_probs = np.random.rand(T, self.K_max)
                spk_probs = spk_probs / spk_probs.sum(axis=1, keepdims=True)
            
            spk_labels_list.append(torch.tensor(spk_probs, dtype=torch.float32))
        
        # 对音频进行padding
        max_audio_len = max(audio_lengths)
        padded_audio = torch.zeros(len(batch), max_audio_len)
        for i, audio in enumerate(audio_signals):
            padded_audio[i, :len(audio)] = audio
        
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)
        
        # 对说话人标签进行padding
        max_spk_len = max(spk.shape[0] for spk in spk_labels_list)
        padded_spk_labels = torch.zeros(len(batch), self.K_max, max_spk_len)
        for i, spk in enumerate(spk_labels_list):
            T, K = spk.shape
            padded_spk_labels[i, :K, :T] = spk.T  # 转换为 [B, K, T] 格式
        
        return {
            'input_signal': padded_audio,
            'input_signal_length': audio_lengths,
            'transcripts': texts,
            'spk_labels': padded_spk_labels
        }
    
    def training_step(self, batch, batch_idx):
        """
        训练步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            损失值
        """
        # 设置说话人标签
        self.model.set_speaker_labels(batch['spk_labels'])
        
        # 前向传播
        loss = self.model(
            input_signal=batch['input_signal'],
            input_signal_length=batch['input_signal_length'],
            transcripts=batch['transcripts']
        )
        
        if isinstance(loss, tuple):
            loss = loss[0]
        
        # 记录损失
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_loss_history.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        验证步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            损失值
        """
        # 设置说话人标签
        self.model.set_speaker_labels(batch['spk_labels'])
        
        # 前向传播
        loss = self.model(
            input_signal=batch['input_signal'],
            input_signal_length=batch['input_signal_length'],
            transcripts=batch['transcripts']
        )
        
        if isinstance(loss, tuple):
            loss = loss[0]
        
        # 记录损失
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss_history.append(loss.item())
        
        return loss
    
    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        
        Returns:
            优化器配置
        """
        # 只优化可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
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
    
    def save_model(self, save_path: str):
        """
        保存模型为.nemo格式
        
        Args:
            save_path: 保存路径（应以.nemo结尾）
        """
        logging.info(f"保存模型到: {save_path}")
        
        # 确保保存路径以.nemo结尾
        if not save_path.endswith('.nemo'):
            save_path = save_path.replace('.pt', '.nemo')
            logging.info(f"修改保存路径为: {save_path}")
        
        # 使用NeMo的标准保存方法
        try:
            # 保存为.nemo格式
            self.model.save_to(save_path)
            logging.info("模型已保存为.nemo格式")
            
            # 额外保存说话人标记和超参数信息
            metadata_path = save_path.replace('.nemo', '_metadata.json')
            metadata = {
                'speaker_tokens': self.speaker_tokens,
                'hyperparameters': self.hparams,
                'model_type': 'RNNTCharWithSpkInjectAndAdapter',
                'adapter_config': {
                    'K_max': getattr(self, 'K_max', 4),
                    'alpha_init': getattr(self, 'alpha_init', 0.1),
                    'num_adapters': getattr(self, 'num_adapters', 4),
                    'adapter_bottleneck': getattr(self, 'adapter_bottleneck', 256)
                }
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logging.info(f"元数据已保存到: {metadata_path}")
            
        except Exception as e:
            logging.error(f"保存.nemo格式失败: {e}")
            # 回退到原始的.pt格式保存
            pt_save_path = save_path.replace('.nemo', '.pt')
            logging.info(f"回退到.pt格式保存: {pt_save_path}")
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.model.cfg,
                'speaker_tokens': self.speaker_tokens,
                'hyperparameters': self.hparams
            }, pt_save_path)
            
            logging.info("模型已保存为.pt格式（回退方案）")
        
        logging.info("模型保存完成")


def main():
    """
    主函数：解析参数并启动训练
    """
    parser = argparse.ArgumentParser(description='Adapter微调训练脚本')
    
    # 模型参数
    parser.add_argument('--pretrained_model_path', type=str, 
                       default='/root/autodl-tmp/joint_sortformer_and_asr_0815/pretrained_models/stt_zh_conformer_transducer_large.nemo',
                       help='预训练模型路径')
    parser.add_argument('--K_max', type=int, default=4, help='最大说话人数量')
    parser.add_argument('--alpha_init', type=float, default=0.1, help='说话人注入强度初始值')
    parser.add_argument('--num_adapters', type=int, default=4, help='Adapter层数量')
    parser.add_argument('--adapter_bottleneck', type=int, default=256, help='Adapter瓶颈层维度')
    
    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='权重衰减')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--max_epochs', type=int, default=50, help='最大训练轮数')
    
    # 数据参数
    parser.add_argument('--train_manifest', type=str, required=True, help='训练数据manifest路径')
    parser.add_argument('--val_manifest', type=str, help='验证数据manifest路径')
    parser.add_argument('--sample_rate', type=int, default=16000, help='音频采样率')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./adapter_finetune_output', help='输出目录')
    parser.add_argument('--experiment_name', type=str, default='adapter_finetune', help='实验名称')
    
    # 其他参数
    parser.add_argument('--gpus', type=int, default=1, help='GPU数量')
    parser.add_argument('--precision', type=int, default=16, help='训练精度')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化模型
    model = AdapterFineTuneModel(
        pretrained_model_path=args.pretrained_model_path,
        K_max=args.K_max,
        alpha_init=args.alpha_init,
        num_adapters=args.num_adapters,
        adapter_bottleneck=args.adapter_bottleneck,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate
    )
    
    # 配置回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, 'checkpoints'),
            filename='{epoch}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=10,
            verbose=True
        )
    ]
    
    # 配置日志记录器
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=args.experiment_name
    )
    
    # 配置训练器
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpus,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        val_check_interval=1.0,
        log_every_n_steps=10
    )
    
    # 开始训练
    logging.info("开始Adapter微调训练...")
    trainer.fit(model)
    
    # 保存最终模型为.nemo格式
    final_model_path = os.path.join(args.output_dir, 'final_model.nemo')
    model.save_model(final_model_path)
    
    logging.info("训练完成！")
    logging.info(f"最终模型保存在: {final_model_path}")
    
    # 检查是否成功保存为.nemo格式
    if os.path.exists(final_model_path):
        logging.info(f"✅ .nemo格式模型保存成功: {final_model_path}")
    else:
        # 检查是否有回退的.pt格式文件
        pt_path = final_model_path.replace('.nemo', '.pt')
        if os.path.exists(pt_path):
            logging.info(f"⚠️  回退到.pt格式: {pt_path}")
        else:
            logging.error("❌ 模型保存失败")


if __name__ == '__main__':
    main()