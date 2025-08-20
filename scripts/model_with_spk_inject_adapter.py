# model_with_spk_inject_adapter.py
"""
多说话人语音识别模型 - 基于说话人正弦核注入的RNNT模型

本模块实现了一个创新的多说话人自动语音识别(ASR)模型，主要特点包括：

1. **说话人信息注入机制**：
   - 在Conformer编码器输出后注入说话人特定的正弦核信息
   - 支持多说话人场景，每个说话人使用不同频率的正弦波
   - 通过可学习的alpha参数控制注入强度

2. **技术架构**：
   - 基于NeMo的EncDecRNNTBPEModel构建
   - 使用ConformerEncoderAdapter支持参数高效微调
   - 保持原有RNNT解码器和联合网络不变

3. **核心创新**：
   - 说话人标签时间轴自适应插值
   - 频率分离的正弦核生成
   - 加权融合机制

4. **应用场景**：
   - 多说话人会议转录
   - 对话系统语音识别
   - 说话人感知的语音识别任务

作者：[Your Name]
创建时间：[Date]
版本：1.0
"""

# 标准库导入
import os
import json
from typing import Optional, Dict, Any, List

# 深度学习框架导入
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# NeMo框架相关导入
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel  # RNNT基础模型
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoderAdapter  # Conformer编码器适配器
from nemo.collections.common.parts import adapter_modules  # 适配器模块
from nemo.utils import logging  # 日志工具


class ManifestSpeakerDataset(torch.utils.data.Dataset):
    """
    多说话人语音数据集类
    
    基于manifest文件的多说话人数据集实现，用于加载包含说话人信息的语音数据。
    
    数据格式要求：
        每行JSON必须包含的字段：
        - audio_filepath: 音频文件路径
        - text: 转录文本
        - pgt_npy: 说话人标签文件路径（.npy格式）
        
        可选字段：
        - utt_id: 语音ID标识
        - duration: 音频时长
    
    设计特点：
        1. 延迟加载：不在__getitem__中读取音频数据，而是返回路径
        2. 高效批处理：将音频读取和padding操作交给collate函数统一处理
        3. 灵活性：支持可选字段，适应不同数据格式
    
    Args:
        manifest_path (str): manifest文件路径
        sample_rate (int): 音频采样率，默认16000Hz
    
    Raises:
        AssertionError: 当manifest缺少必要字段时
        ValueError: 当样本缺少pgt_npy字段时
    """
    
    def __init__(self, manifest_path: str, sample_rate: int = 16000):
        """
        初始化数据集
        
        Args:
            manifest_path: manifest文件的完整路径
            sample_rate: 目标音频采样率
        """
        super().__init__()
        self.sample_rate = sample_rate  # 存储采样率配置
        self.items: List[Dict[str, Any]] = []  # 存储所有数据项的列表
        
        # 逐行读取manifest文件
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for ln in f:
                # 跳过空行
                if not ln.strip():
                    continue
                
                # 解析JSON格式的数据行
                j = json.loads(ln)
                
                # 验证必要字段存在性
                assert 'audio_filepath' in j and 'text' in j, "manifest 缺少 audio_filepath/text"
                
                # 检查说话人标签文件
                if 'pgt_npy' not in j:
                    # 训练阶段必须有说话人标签；推理阶段可以不需要
                    raise ValueError(f"样本缺少 pgt_npy 字段: {j}")
                
                # 将有效数据项添加到列表中
                self.items.append(j)

    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            int: 数据集中样本的总数
        """
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取指定索引的数据项
        
        注意：这里采用延迟加载策略，不直接读取音频文件，而是返回文件路径。
        实际的音频读取和预处理工作交给DataLoader的collate函数统一处理，
        这样可以实现更高效的批处理和内存管理。
        
        Args:
            idx: 数据项索引
            
        Returns:
            Dict[str, Any]: 包含以下键值的字典：
                - audio_filepath: 音频文件路径
                - text: 转录文本
                - pgt_npy: 说话人标签文件路径
                - utt_id: 语音ID（可选）
        """
        j = self.items[idx]  # 获取指定索引的数据项
        
        # 返回数据项字典，包含路径信息而非实际数据
        # 这种设计允许在collate阶段进行批量处理和优化
        return {
            "audio_filepath": j['audio_filepath'],  # 音频文件完整路径
            "text": j['text'],  # 对应的转录文本
            "pgt_npy": j['pgt_npy'],  # 说话人标签numpy文件路径
            "utt_id": j.get("utt_id", None),  # 语音唯一标识符（可选）
        }


class RNNTWithSpkInjectAdapter(EncDecRNNTBPEModel):
    """
    基于说话人正弦核注入的RNNT模型
    
    这是一个创新的多说话人语音识别模型，通过在Conformer编码器输出后注入
    说话人特定的正弦核信息来增强模型对多说话人场景的处理能力。
    
    核心特性：
        1. **说话人正弦核注入**：
           - 为每个说话人生成独特频率的正弦波核
           - 在编码器输出的特征表示中注入说话人信息
           - 通过可学习的alpha参数控制注入强度
        
        2. **时间轴自适应**：
           - 支持任意长度的说话人标签输入 (B, K, P)
           - 自动将标签时间轴插值到编码器输出长度 T_enc
           - 保证时间对齐的准确性
        
        3. **参数高效微调**：
           - 使用ConformerEncoderAdapter替换原始编码器
           - 支持adapter-based的参数高效微调
           - 保持预训练权重的同时添加少量可训练参数
        
        4. **架构兼容性**：
           - 继承自NeMo的EncDecRNNTBPEModel
           - 保持解码器和联合网络不变
           - 确保与现有RNNT流程的兼容性
    
    技术实现：
        - 基于频率分离的正弦核生成机制
        - 加权融合的说话人信息注入
        - 端到端的可微分训练流程
    
    Args:
        cfg: 模型配置对象
        trainer: PyTorch Lightning训练器
        K_max (int): 最大支持的说话人数量，默认4
        alpha_init (float): 说话人注入强度的初始值，默认0.1
        base_model: 预训练的基础RNNT模型
    """

    def __init__(self, cfg=None, trainer=None, K_max: int = 4, alpha_init: float = 0.1, base_model=None):
        """
        初始化说话人注入适配器模型
        
        该初始化过程包括以下关键步骤：
        1. 参数设置和配置获取
        2. 父类初始化
        3. 从基础模型复制组件
        4. 编码器替换为ConformerEncoderAdapter
        5. 说话人注入参数初始化
        
        Args:
            cfg: 模型配置，如果为None则从base_model获取
            trainer: PyTorch Lightning训练器实例
            K_max: 支持的最大说话人数量
            alpha_init: 说话人注入强度的初始化值
            base_model: 预训练的RNNT基础模型
        """
        # ==================== 第一步：参数设置和配置获取 ====================
        self.K_max = int(K_max)  # 存储最大说话人数量，确保为整数类型
        
        # 从基础模型获取配置信息
        # 如果没有提供cfg但提供了base_model，则从base_model中提取配置
        if base_model is not None and cfg is None:
            cfg = base_model._cfg if hasattr(base_model, '_cfg') else None
        
        # ==================== 第二步：父类初始化 ====================
        # 直接调用ModelPT的初始化方法，避免EncDecRNNTBPEModel的复杂初始化
        # 这样可以更好地控制初始化过程，特别是在使用预训练模型时
        from nemo.core.classes.modelPT import ModelPT
        ModelPT.__init__(self, cfg=cfg, trainer=None)  # trainer设为None，后续可以重新设置
        
        # ==================== 第三步：从基础模型复制组件 ====================
        if base_model is not None:
            # 复制预处理器组件
            self.preprocessor = base_model.preprocessor
            
            # ==================== 编码器替换为ConformerEncoderAdapter ====================
            # 检查基础模型是否使用Conformer编码器
            if hasattr(base_model.encoder, '__class__') and 'Conformer' in base_model.encoder.__class__.__name__:
                logging.info(f"检测到Conformer编码器，将替换为ConformerEncoderAdapter")
                
                # 获取或构建编码器配置
                encoder_cfg = base_model.encoder._cfg if hasattr(base_model.encoder, '_cfg') else None
                
                if encoder_cfg is None:
                    # 如果没有现成的配置，从现有encoder的属性中推断配置参数
                    # 使用getattr的双重回退机制：先尝试公共属性，再尝试私有属性，最后使用默认值
                    encoder_cfg = {
                        'feat_in': getattr(base_model.encoder, 'feat_in', getattr(base_model.encoder, '_feat_in', 80)),  # 输入特征维度
                        'feat_out': getattr(base_model.encoder, 'feat_out', getattr(base_model.encoder, '_feat_out', 512)),  # 输出特征维度
                        'n_layers': getattr(base_model.encoder, 'n_layers', getattr(base_model.encoder, '_n_layers', 18)),  # Conformer层数
                        'd_model': getattr(base_model.encoder, 'd_model', getattr(base_model.encoder, '_d_model', 512)),  # 模型维度
                        'subsampling': getattr(base_model.encoder, 'subsampling', 'striding'),  # 下采样方式
                        'subsampling_factor': getattr(base_model.encoder, 'subsampling_factor', 4),  # 下采样倍数
                        'self_attention_model': getattr(base_model.encoder, 'self_attention_model', 'rel_pos'),  # 自注意力类型
                        'n_heads': getattr(base_model.encoder, 'n_heads', 8),  # 注意力头数
                        'conv_kernel_size': getattr(base_model.encoder, 'conv_kernel_size', 31),  # 卷积核大小
                        'dropout': getattr(base_model.encoder, 'dropout', 0.1),  # 通用dropout率
                        'dropout_att': getattr(base_model.encoder, 'dropout_att', 0.1),  # 注意力dropout率
                    }
                    logging.info(f"从现有encoder推断配置: {encoder_cfg}")
                
                # 创建新的ConformerEncoderAdapter实例
                # ConformerEncoderAdapter支持adapter-based的参数高效微调
                self.encoder = ConformerEncoderAdapter(
                    feat_in=encoder_cfg['feat_in'],  # 输入特征维度（通常为80，对应log-mel特征）
                    feat_out=encoder_cfg['feat_out'],  # 输出特征维度
                    n_layers=encoder_cfg['n_layers'],  # Conformer块的层数
                    d_model=encoder_cfg['d_model'],  # 模型的隐藏维度
                    subsampling=encoder_cfg['subsampling'],  # 下采样策略（如striding）
                    subsampling_factor=encoder_cfg['subsampling_factor'],  # 下采样倍数
                    self_attention_model=encoder_cfg['self_attention_model'],  # 自注意力机制类型
                    n_heads=encoder_cfg['n_heads'],  # 多头注意力的头数
                    conv_kernel_size=encoder_cfg['conv_kernel_size'],  # 卷积模块的核大小
                    dropout=encoder_cfg['dropout'],  # 通用dropout概率
                    dropout_att=encoder_cfg['dropout_att'],  # 注意力模块的dropout概率
                )
                
                # ==================== 权重复制和初始化 ====================
                # 尝试将原始Conformer编码器的预训练权重复制到新的ConformerEncoderAdapter
                try:
                    # 使用strict=False允许部分权重不匹配（adapter相关的新参数）
                    missing, unexpected = self.encoder.load_state_dict(base_model.encoder.state_dict(), strict=False)
                    
                    # 记录权重加载情况
                    if missing:
                        logging.info(f"ConformerEncoderAdapter加载权重时缺少的键: {missing}")
                    if unexpected:
                        logging.info(f"ConformerEncoderAdapter加载权重时意外的键: {unexpected}")
                    
                    logging.info("成功将原始Conformer权重复制到ConformerEncoderAdapter")
                    
                except Exception as e:
                    # 如果权重复制失败，记录错误并回退到原始编码器
                    logging.error(f"复制Conformer权重失败: {e}")
                    self.encoder = base_model.encoder
                    logging.warning("回退到原始encoder")
                    
            else:
                # 对于非Conformer编码器，直接使用原始编码器
                # 这确保了模型的通用性，可以处理不同类型的编码器
                self.encoder = base_model.encoder
                logging.info(f"使用原始编码器: {type(base_model.encoder)}")
            
            # ==================== 复制其他核心组件 ====================
            # 直接复制解码器、联合网络和损失函数，保持原有的预训练权重
            self.decoder = base_model.decoder  # RNNT解码器（通常是LSTM-based）
            self.joint = base_model.joint      # 联合网络（用于计算编码器和解码器输出的联合概率）
            self.loss = base_model.loss        # RNNT损失函数
            
            # ==================== 联合网络配置优化 ====================
            # 禁用fuse_loss_wer模式以避免神经类型验证错误
            # 这个模式在某些情况下可能导致类型不匹配的问题
            if hasattr(self.joint, '_fuse_loss_wer'):
                self.joint._fuse_loss_wer = False
                logging.info("已禁用 joint 网络的 fuse_loss_wer 模式")
            
            # ==================== WER评估指标设置 ====================
            # 设置词错误率(Word Error Rate)计算模块
            if hasattr(base_model, 'wer'):
                # 如果基础模型已有WER模块，直接复制
                self.wer = base_model.wer
            else:
                # 如果没有，创建新的WER计算模块
                from nemo.collections.asr.metrics.wer import WER
                self.wer = WER(
                    decoding=getattr(base_model, 'decoding', None),  # 解码配置
                    batch_dim_index=0,      # 批次维度索引
                    use_cer=False,          # 不使用字符错误率
                    log_prediction=True,    # 记录预测结果
                    dist_sync_on_step=True, # 分布式训练时同步
                )
            
            # ==================== Tokenizer设置 ====================
            # tokenizer用于文本的编码和解码，需要谨慎处理其获取方式
            if hasattr(base_model, 'tokenizer'):
                # 优先从base_model直接获取tokenizer
                self.tokenizer = base_model.tokenizer
                logging.info(f"成功复制tokenizer: {type(self.tokenizer)}")
            else:
                # 如果base_model没有tokenizer属性，尝试从其他位置获取
                logging.info(f"base_model没有tokenizer属性，尝试从joint.vocabulary获取")
                
                # 尝试从联合网络的vocabulary属性获取
                if hasattr(base_model.joint, 'vocabulary'):
                    self.tokenizer = base_model.joint.vocabulary
                    logging.info(f"从joint.vocabulary获取tokenizer: {type(self.tokenizer)}")
                else:
                    # 如果都找不到，发出警告
                    logging.warning("无法找到tokenizer，可能会影响文本处理")
        
        # ==================== 第四步：说话人注入参数初始化 ====================
        # 获取编码器输出维度，这是说话人信息注入的关键参数
        self.M = self.encoder.d_model if hasattr(self.encoder, 'd_model') else 512
        logging.info(f"编码器输出维度 M = {self.M}")
        
        # 初始化可学习的说话人注入强度参数
        # alpha控制说话人信息对编码器输出的影响程度
        # 使用nn.Parameter使其成为模型的可训练参数
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        logging.info(f"初始化说话人注入强度 alpha = {alpha_init}")

    def inject_speakers(self, enc_out: torch.Tensor, enc_len: torch.Tensor,
                        spk_labels: Optional[torch.Tensor]) -> torch.Tensor:
        """
        说话人信息注入核心方法
        
        该方法实现了基于正弦核的说话人信息注入机制，是本模型的核心创新点。
        通过为不同说话人生成不同频率的正弦波，并根据说话人标签进行加权融合，
        将说话人特定的信息注入到编码器的输出特征中。
        
        技术原理：
            1. 时间轴对齐：将说话人标签从原始时间轴插值到编码器输出时间轴
            2. 频率分离：为每个说话人分配独特的正弦波频率
            3. 加权融合：根据说话人活跃度对正弦核进行加权
            4. 特征注入：将加权后的说话人信息添加到编码器输出
        
        Args:
            enc_out (torch.Tensor): 编码器输出，形状为 (B, T_enc, M)
                - B: 批次大小
                - T_enc: 编码器输出的时间步数
                - M: 编码器输出的特征维度
            enc_len (torch.Tensor): 每个样本的有效长度，形状为 (B,)
            spk_labels (Optional[torch.Tensor]): 说话人标签，形状为 (B, K, P)
                - K: 说话人数量
                - P: 原始时间步数
                - 值表示每个时间步每个说话人的活跃度 [0, 1]
        
        Returns:
            torch.Tensor: 注入说话人信息后的编码器输出，形状为 (B, T_enc, M)
        """
        # ==================== 输入验证和早期返回 ====================
        if spk_labels is None:
            logging.debug("没有说话人标签，跳过注入")
            return enc_out
        
        # 获取张量维度信息
        B, T_enc, M = enc_out.shape  # 批次大小、编码器时间步、特征维度
        B2, K, P = spk_labels.shape  # 批次大小、说话人数、原始时间步
        
        # 验证批次大小一致性
        assert B == B2, f"批次大小不匹配: {B} vs {B2}"
        
        logging.debug(f"注入说话人信息: enc_out={enc_out.shape}, spk_labels={spk_labels.shape}")
        
        # ==================== 时间轴对齐和插值 ====================
        # 将说话人标签从原始时间轴 P 插值到编码器输出时间轴 T_enc
        # 这确保了说话人信息与编码器输出在时间维度上的精确对齐
        spk_labels_interp = F.interpolate(
            spk_labels.float(),  # 转换为浮点数以支持插值操作
            size=T_enc,          # 目标时间步数
            mode='linear',       # 使用线性插值保持平滑性
            align_corners=False  # 不对齐角点，更适合时间序列数据
        )  # 输出形状: (B, K, T_enc)
        
        # ==================== 正弦核生成 ====================
        device = enc_out.device  # 确保计算在正确的设备上进行
        freq_base = 1000.0       # 基础频率，可以根据需要调整
        
        # 为每个说话人生成独特频率的正弦波核
        # 不同频率确保不同说话人的信息在频域上可分离
        sinusoid_kernels = []
        for k in range(K):
            # 为第k个说话人分配频率：freq_k = freq_base * (k + 1)
            freq_k = freq_base * (k + 1)  # 避免零频率，确保频率分离
            
            # 生成时间索引
            t_indices = torch.arange(T_enc, device=device, dtype=torch.float32)
            
            # 生成正弦波：sin(2π * freq_k * t / T_enc)
            # 归一化时间索引确保正弦波在[0, 2π]范围内完整周期
            sin_k = torch.sin(2 * torch.pi * freq_k * t_indices / T_enc)
            sinusoid_kernels.append(sin_k)
        
        # 将所有说话人的正弦核堆叠成张量
        sinusoid_kernels = torch.stack(sinusoid_kernels, dim=0)  # 形状: (K, T_enc)
        
        # ==================== 加权融合计算 ====================
        # 使用说话人标签对正弦核进行加权
        # 广播机制：(B, K, T_enc) * (1, K, T_enc) -> (B, K, T_enc)
        weighted_kernels = spk_labels_interp * sinusoid_kernels.unsqueeze(0)
        
        # 对所有说话人维度求和，得到最终的说话人注入信号
        speaker_injection = weighted_kernels.sum(dim=1)  # 形状: (B, T_enc)
        
        # ==================== 特征维度扩展 ====================
        # 将说话人注入信号扩展到编码器特征维度
        # 这样可以对编码器输出的每个特征维度都施加相同的说话人影响
        speaker_injection = speaker_injection.unsqueeze(-1).expand(-1, -1, M)  # 形状: (B, T_enc, M)
        
        # ==================== 最终注入和输出 ====================
        # 使用可学习的alpha参数控制注入强度
        # 这允许模型在训练过程中自适应地调整说话人信息的影响程度
        injected_out = enc_out + self.alpha * speaker_injection
        
        logging.debug(f"说话人注入完成，alpha={self.alpha.item():.4f}")
        return injected_out

    def encode_with_injection(self, input_signal: torch.Tensor, input_signal_length: torch.Tensor,
                              spk_labels: Optional[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        """
        完整的音频编码和说话人信息注入流程
        
        该方法整合了音频预处理、编码器处理和说话人信息注入三个关键步骤，
        是模型前向传播的核心组件。它将原始音频信号转换为包含说话人信息的
        高级特征表示，为后续的解码和识别提供基础。
        
        处理流程：
            1. 音频预处理：将原始音频转换为模型可处理的特征（如log-mel谱）
            2. 编码器处理：通过Conformer编码器提取音频的语义特征
            3. 说话人注入：在编码器输出中注入说话人特定的正弦核信息
        
        Args:
            input_signal (torch.Tensor): 原始音频信号，形状为 (B, T_audio)
                - B: 批次大小
                - T_audio: 音频采样点数
            input_signal_length (torch.Tensor): 每个音频样本的有效长度，形状为 (B,)
            spk_labels (Optional[torch.Tensor]): 说话人标签，形状为 (B, K, P)
                - K: 说话人数量
                - P: 原始时间步数
                - 如果为None，则跳过说话人信息注入
        
        Returns:
            tuple: 包含两个元素的元组
                - injected_out (torch.Tensor): 注入说话人信息后的编码器输出，形状为 (B, T_enc, M)
                - enc_len (torch.Tensor): 编码器输出的有效长度，形状为 (B,)
        """
        # ==================== 第一步：音频预处理 ====================
        # 将原始音频信号转换为模型可处理的特征表示
        # 通常包括窗函数、FFT、log-mel滤波器组等操作
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal,      # 原始音频波形
            length=input_signal_length,     # 音频有效长度
        )
        
        # ==================== 第二步：编码器处理 ====================
        # 通过Conformer编码器提取音频的高级语义特征
        # 编码器会进行下采样，因此输出长度通常小于输入长度
        enc_out, enc_len = self.encoder(
            audio_signal=processed_signal,      # 预处理后的音频特征
            length=processed_signal_length      # 预处理后的特征长度
        )
        
        # ==================== 第三步：说话人信息注入 ====================
        # 在编码器输出中注入说话人特定的正弦核信息
        # 这是本模型的核心创新，增强了对多说话人场景的处理能力
        injected_out = self.inject_speakers(enc_out, enc_len, spk_labels)
        
        # 返回注入说话人信息后的编码器输出和对应的长度信息
        return injected_out, enc_len