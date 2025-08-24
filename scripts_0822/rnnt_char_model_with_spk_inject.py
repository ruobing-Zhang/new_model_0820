# Copyright (c) 2024, Custom Implementation
# 基于字符级分词器的RNNT模型，添加说话人信息注入和Adapter层

import os
import json
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 导入基础的字符级RNNT模型
from rnnt_models import EncDecRNNTModel


class ManifestSpeakerDataset(torch.utils.data.Dataset):
    """
    基于 manifest 的多说话人数据集：
      每行 JSON 至少包含: audio_filepath, text, pgt_npy
      可选: utt_id, duration
    """
    def __init__(self, manifest_path: str, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.items: List[Dict[str, Any]] = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for ln in f:
                if not ln.strip():
                    continue
                j = json.loads(ln)
                assert 'audio_filepath' in j and 'text' in j, "manifest 缺少 audio_filepath/text"
                if 'pgt_npy' not in j:
                    # 训练期必须有 pgt_npy；推理期可不需要
                    raise ValueError(f"样本缺少 pgt_npy 字段: {j}")
                self.items.append(j)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        j = self.items[idx]
        # 注意：不在这里读音频，交给 collate 统一读并做 padding（更高效）
        # 这里只传路径和文本、npy路径
        return {
            "audio_filepath": j['audio_filepath'],
            "text": j['text'],
            "pgt_npy": j['pgt_npy'],
            "utt_id": j.get("utt_id", None),
        }


class RNNTCharWithSpkInject(EncDecRNNTModel):
    """
    基于字符级分词器的RNNT模型，在 Conformer 编码器输出后，按论文方式注入"说话人正弦核"。
    - 支持 (B, K, T_enc) 的 spk_labels 输入
    - 自动把 P 的时间轴插值到 T_enc
    - 使用可学习标量 alpha 控制注入强度
    - 基于EncDecRNNTModel（字符级分词器）而非BPE分词器
    """

    def __init__(self, cfg=None, trainer=None, K_max: int = 4, alpha_init: float = 0.1, base_model=None):
        """
        初始化带说话人注入的字符级RNNT模型
        
        Args:
            cfg: 模型配置
            trainer: 训练器
            K_max: 最大说话人数量
            alpha_init: 注入强度初始值
            base_model: 基础模型（用于权重复制）
        """
        # 设置参数
        self.K_max = int(K_max)
        
        if base_model is not None:
            # 从基础模型复制配置并初始化父类
            super().__init__(cfg=base_model.cfg, trainer=trainer)
            
            # 复制基础模型的组件
            self.preprocessor = base_model.preprocessor
            self.encoder = base_model.encoder
            self.decoder = base_model.decoder
            self.joint = base_model.joint
            self.loss = base_model.loss
            
            # 复制其他重要属性
            if hasattr(base_model, 'spec_augmentation'):
                self.spec_augmentation = base_model.spec_augmentation
            if hasattr(base_model, 'decoding'):
                self.decoding = base_model.decoding
            if hasattr(base_model, 'wer'):
                self.wer = base_model.wer
            
            # 推断编码器隐维 M
            M = getattr(self.encoder, "d_model", None)
            if M is None:
                # fallback: 找一个权重矩阵的 in_features
                for _, p in self.encoder.named_parameters():
                    if p.dim() == 2:
                        M = p.shape[1]
                        break
            if M is None:
                raise RuntimeError("无法推断 encoder 隐维 M，请检查模型的 encoder 结构。")

            self.M = int(M)
            print(f"[DEBUG] 推断得到编码器隐维 M = {self.M}")
        else:
            # 如果没有base_model，使用提供的配置初始化
            super().__init__(cfg=cfg, trainer=trainer)
            # 设置默认值，稍后动态调整
            self.M = 512  # 默认值

        # 正弦核 Γ ∈ R^{K×M}（缓冲区，不参与梯度）
        k_ids = torch.arange(self.K_max, dtype=torch.float32).unsqueeze(1)  # (K,1)
        m_ids = torch.arange(self.M, dtype=torch.float32).unsqueeze(0)      # (1,M)
        gamma = torch.sin(2 * torch.pi * k_ids * m_ids / float(self.M))     # (K,M)
        self.register_buffer("Gamma", gamma)  # shape: [K_max, M]
        print(f"[DEBUG] 注册正弦核 Gamma，形状: {gamma.shape}")

        # 注入强度 α（可学习）
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        print(f"[DEBUG] 初始化注入强度 alpha = {alpha_init}")

        # 运行态缓存：训练/验证 step 会把当前 batch 的 P 放进来
        self._current_spk_labels: Optional[torch.Tensor] = None  # [B,K,T*] or None

    def inject_speakers(self, enc_out: torch.Tensor, enc_len: torch.Tensor,
                        spk_labels: Optional[torch.Tensor]) -> torch.Tensor:
        """
        说话人信息注入核心函数
        
        Args:
            enc_out: 编码器输出 [B, T_enc, M] 或 [B, M, T_enc]
            enc_len: 编码器输出长度 [B]
            spk_labels: 说话人标签 [B, K, T_any] 或 None
            
        Returns:
            enc_out_tilde: 注入说话人信息后的编码器输出 [B, T_enc, M]
        """
        if spk_labels is None:
            print("[DEBUG] 没有说话人标签，跳过注入")
            return enc_out

        # 检查编码器输出的实际维度
        print(f"[DEBUG] enc_out.shape = {enc_out.shape}")
        
        # 根据实际维度判断格式
        if len(enc_out.shape) == 3:
            if enc_out.shape[2] == self.M:  # [B, T_enc, M]格式
                B, T_enc, M = enc_out.shape
                print(f"[DEBUG] 检测到格式 [B, T_enc, M]: B={B}, T_enc={T_enc}, M={M}")
            elif enc_out.shape[1] == self.M:  # [B, M, T_enc]格式
                B, M, T_enc = enc_out.shape
                print(f"[DEBUG] 检测到格式 [B, M, T_enc]: B={B}, M={M}, T_enc={T_enc}")
                # 转换为[B, T_enc, M]格式
                enc_out = enc_out.transpose(1, 2)
                print(f"[DEBUG] 转换后 enc_out.shape = {enc_out.shape}")
            else:
                raise ValueError(f"无法识别编码器输出格式: {enc_out.shape}, 期望M={self.M}")
        else:
            raise ValueError(f"Unexpected enc_out shape: {enc_out.shape}")
        
        B, T_enc, M = enc_out.shape
        assert M == self.M, f"M mismatch: enc_out.M={M} vs self.M={self.M}"

        # A: [B,M,T] - 转换编码器输出格式
        A = enc_out.transpose(1, 2).contiguous()
        # 归一化
        A_norm = A / (A.norm(p=2, dim=1, keepdim=True) + 1e-8)
        print(f"[DEBUG] 归一化后 A_norm.shape = {A_norm.shape}")

        # 确保 P 的时间轴与 T_enc 一致；期望 P: [B,K,T_enc]
        P = spk_labels.to(A_norm.dtype).to(A_norm.device)
        if P.dim() != 3 or P.size(0) != B or P.size(1) != self.K_max:
            raise RuntimeError(f"spk_labels 维度不正确，期望 [B,{self.K_max},T*]，但得到 {tuple(P.shape)}")

        if P.size(2) != T_enc:
            # 使用 1D 线性插值到 T_enc
            print(f"[DEBUG] 插值说话人标签从 {P.size(2)} 到 {T_enc}")
            P = F.interpolate(P, size=T_enc, mode='linear', align_corners=False)

        # 计算 Γ^T P → [B,M,T]
        # Γ: [K,M]，先转置 [M,K]
        spk_inject = torch.einsum('mk,bkt->bmt', self.Gamma.t(), P)  # [B,M,T]
        print(f"[DEBUG] 说话人注入项 spk_inject.shape = {spk_inject.shape}")
        
        # 应用注入：A_tilde = A_norm + α * spk_inject
        A_tilde = A_norm + self.alpha * spk_inject
        print(f"[DEBUG] 注入强度 alpha = {self.alpha.item():.4f}")
        
        # 转换回 [B,T,M] 格式以匹配后续网络的期望
        result = A_tilde.transpose(1, 2).contiguous()  # [B,T,M]
        print(f"[DEBUG] 最终输出 result.shape = {result.shape}")
        return result

    def encode_with_injection(self, input_signal: torch.Tensor, input_signal_length: torch.Tensor,
                              spk_labels: Optional[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        """
        带说话人注入的编码过程
        
        Args:
            input_signal: 输入音频信号 [B, T]
            input_signal_length: 音频信号长度 [B]
            spk_labels: 说话人标签 [B, K, T_any] 或 None
            
        Returns:
            tuple: (编码器输出, 编码器输出长度)
        """
        # 预处理
        print(f"[DEBUG] 输入音频信号 input_signal.shape = {input_signal.shape}")
        proc, proc_len = self.preprocessor(input_signal=input_signal, length=input_signal_length)
        print(f"[DEBUG] 预处理后 proc.shape = {proc.shape}, proc_len = {proc_len}")
        
        # 编码
        enc_out, enc_len = self.encoder(audio_signal=proc, length=proc_len)
        print(f"[DEBUG] 原始编码器输出 enc_out.shape = {enc_out.shape}, enc_len = {enc_len}")
        
        # 说话人信息注入
        enc_out = self.inject_speakers(enc_out, enc_len, spk_labels)
        print(f"[DEBUG] 注入后编码器输出 enc_out.shape = {enc_out.shape}")
        
        # 确保enc_len与实际的时间维度匹配
        if len(enc_out.shape) == 3:
            actual_T = enc_out.shape[1]  # 假设格式是[B, T, M]
            # 只有当enc_len超过actual_T时才需要调整
            enc_len = torch.clamp(enc_len, max=actual_T)
            print(f"[DEBUG] actual_T: {actual_T}, enc_len调整后: {enc_len}")
        
        return enc_out, enc_len
        
    def forward(self, input_signal=None, input_signal_length=None, transcripts=None):
        """
        模型前向传播
        
        Args:
            input_signal: 输入音频信号 [B, T]
            input_signal_length: 音频信号长度 [B]
            transcripts: 转录文本列表
            
        Returns:
            损失值或编码器输出
        """
        # 编码（包含说话人注入）
        encoded, encoded_len = self.encode_with_injection(input_signal, input_signal_length, self._current_spk_labels)
        
        # 如果有转录文本，计算损失
        if transcripts is not None:
            # 使用字符级分词器处理文本
            # 注意：EncDecRNNTModel使用字符级标签，通常在cfg.labels中定义
            if hasattr(self, 'cfg') and hasattr(self.cfg, 'labels'):
                # 将文本转换为字符级标签索引
                char_to_idx = {char: idx for idx, char in enumerate(self.cfg.labels)}
                tokens = []
                for text in transcripts:
                    token_ids = []
                    for char in text:
                        if char in char_to_idx:
                            token_ids.append(char_to_idx[char])
                        else:
                            # 使用UNK标记或跳过未知字符
                            if '<unk>' in char_to_idx:
                                token_ids.append(char_to_idx['<unk>'])
                    tokens.append(token_ids)
            else:
                raise ValueError("模型配置中缺少字符标签定义")
            
            target_lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long, device=encoded.device)
            max_target_len = target_lengths.max()
            targets = torch.zeros(len(tokens), max_target_len, dtype=torch.long, device=encoded.device)
            for i, t in enumerate(tokens):
                targets[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=encoded.device)
            
            # 前向解码器
            decoder_outputs = self.decoder(targets=targets, target_length=target_lengths)
            if isinstance(decoder_outputs, tuple):
                decoder_outputs = decoder_outputs[0]  # 取第一个元素作为输出tensor
            print(f"[DEBUG] decoder_outputs.shape = {decoder_outputs.shape}")
            
            # 确保编码器和解码器输出格式匹配joint网络期望
            # joint网络期望 encoder: [B, M, T], decoder: [B, U, M]
            # 当前encoded是[B, T, M]格式，需要转换为[B, M, T]
            if encoded.shape[-1] == 512:  # 如果最后一维是特征维度
                encoded = encoded.transpose(1, 2)  # [B, T, M] -> [B, M, T]
            
            print(f"[DEBUG] 转换后编码器格式: {encoded.shape}")
            print(f"[DEBUG] 解码器格式: {decoder_outputs.shape}")
            
            print(f"[DEBUG] 调用joint前格式 - encoded: {encoded.shape}, decoder: {decoder_outputs.shape}")
            
            # 使用joint网络，不使用fuse_loss_wer模式，只获取原始输出
            joint_outputs = self.joint(
                encoder_outputs=encoded, 
                decoder_outputs=decoder_outputs
            )
            print(f"[DEBUG] joint输出类型: {type(joint_outputs)}")
            if isinstance(joint_outputs, tuple):
                print(f"[DEBUG] joint输出tuple长度: {len(joint_outputs)}")
                for i, output in enumerate(joint_outputs):
                    print(f"[DEBUG] joint输出[{i}]格式: {output.shape if hasattr(output, 'shape') else type(output)}")
                # 通常第一个元素是主要输出
                joint_outputs = joint_outputs[0]
            print(f"[DEBUG] 最终joint输出格式: {joint_outputs.shape}")
            
            # 转换为log_probs
            joint_outputs = F.log_softmax(joint_outputs, dim=-1)
            
            # 损失计算
            loss = self.loss(
                log_probs=joint_outputs,
                targets=targets,
                input_lengths=encoded_len,
                target_lengths=target_lengths
            )
            
            return (loss,)
        
        return encoded, encoded_len

    def set_speaker_labels(self, spk_labels: Optional[torch.Tensor]):
        """
        设置当前批次的说话人标签
        
        Args:
            spk_labels: 说话人标签 [B, K, T] 或 None
        """
        self._current_spk_labels = spk_labels
        if spk_labels is not None:
            print(f"[DEBUG] 设置说话人标签，形状: {spk_labels.shape}")
        else:
            print(f"[DEBUG] 清除说话人标签")


class Adapter(nn.Module):
    """
    标准Adapter层，包含下采样-激活-上采样-残差结构。
    用于在预训练模型基础上进行参数高效的微调。
    """
    def __init__(self, d_model, bottleneck_dim=128):
        """
        初始化Adapter层
        
        Args:
            d_model: 输入特征维度
            bottleneck_dim: 瓶颈层维度（通常远小于d_model）
        """
        super().__init__()
        self.d_model = d_model
        self.down = nn.Linear(d_model, bottleneck_dim)  # 下采样
        self.activation = nn.ReLU()  # 激活函数
        self.up = nn.Linear(bottleneck_dim, d_model)  # 上采样
        self.layernorm = nn.LayerNorm(d_model)  # 层归一化
        
        print(f"[DEBUG] 初始化Adapter: d_model={d_model}, bottleneck_dim={bottleneck_dim}")

    def forward(self, x):
        """
        Adapter前向传播
        
        Args:
            x: 输入特征 [B, T, d_model] 或 [B, d_model, T]
            
        Returns:
            输出特征，格式与输入相同
        """
        # 记录原始格式
        original_shape = x.shape
        transpose_needed = False
        
        # 确保输入格式为 [B, T, d_model]
        if x.shape[1] == self.d_model and len(x.shape) == 3:  # 如果是 [B, d_model, T] 格式
            x = x.transpose(1, 2)
            transpose_needed = True
        
        residual = x
        
        # Adapter变换：下采样 -> 激活 -> 上采样
        x = self.down(x)
        x = self.activation(x)
        x = self.up(x)
        
        # 残差连接和层归一化
        x = self.layernorm(x + residual)
        
        # 如果原来是 [B, d_model, T] 格式，转换回去
        if transpose_needed:
            x = x.transpose(1, 2)
        
        return x


class RNNTCharWithSpkInjectAndAdapter(RNNTCharWithSpkInject):
    """
    在说话人注入后插入若干Adapter层的字符级RNNT模型。
    这种设计允许在保持预训练权重的同时，通过少量参数进行任务特定的微调。
    """

    def __init__(self, cfg=None, trainer=None, K_max=4, alpha_init=0.1, base_model=None, 
                 num_adapters=4, adapter_bottleneck=256):
        """
        初始化带Adapter的说话人注入字符级RNNT模型
        
        Args:
            cfg: 模型配置
            trainer: 训练器
            K_max: 最大说话人数量
            alpha_init: 注入强度初始值
            base_model: 基础模型
            num_adapters: Adapter层数量
            adapter_bottleneck: Adapter瓶颈层维度
        """
        super().__init__(cfg, trainer, K_max, alpha_init, base_model)
        
        self.num_adapters = num_adapters
        self.adapter_bottleneck = adapter_bottleneck
        
        # 创建多个Adapter层
        self.adapters = nn.ModuleList([
            Adapter(self.M, adapter_bottleneck) for _ in range(num_adapters)
        ])
        
        print(f"[DEBUG] 创建了 {num_adapters} 个Adapter层，瓶颈维度: {adapter_bottleneck}")

    def encode_with_injection(self, input_signal, input_signal_length, spk_labels):
        """
        带说话人注入和Adapter的编码过程
        
        Args:
            input_signal: 输入音频信号
            input_signal_length: 音频信号长度
            spk_labels: 说话人标签
            
        Returns:
            tuple: (编码器输出, 编码器输出长度)
        """
        # 先进行说话人注入
        enc_out, enc_len = super().encode_with_injection(input_signal, input_signal_length, spk_labels)
        
        print(f"[DEBUG] 说话人注入后，开始通过 {self.num_adapters} 个Adapter层")
        
        # 依次通过所有Adapter层
        for i, adapter in enumerate(self.adapters):
            print(f"[DEBUG] 通过第 {i+1} 个Adapter层")
            enc_out = adapter(enc_out)
            
        print(f"[DEBUG] 所有Adapter层处理完成，最终输出形状: {enc_out.shape}")
        return enc_out, enc_len

    def get_adapter_parameters(self):
        """
        获取所有Adapter层的参数，用于参数高效微调
        
        Returns:
            generator: Adapter参数生成器
        """
        for adapter in self.adapters:
            for param in adapter.parameters():
                yield param

    def freeze_base_model(self):
        """
        冻结基础模型参数，只训练Adapter和说话人注入相关参数
        """
        # 冻结预处理器
        for param in self.preprocessor.parameters():
            param.requires_grad = False
            
        # 冻结编码器
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # 冻结解码器
        for param in self.decoder.parameters():
            param.requires_grad = False
            
        # 冻结joint网络
        for param in self.joint.parameters():
            param.requires_grad = False
            
        # 保持说话人注入参数可训练
        self.alpha.requires_grad = True
        
        # 保持Adapter参数可训练
        for adapter in self.adapters:
            for param in adapter.parameters():
                param.requires_grad = True
                
        print("[DEBUG] 已冻结基础模型参数，保持Adapter和说话人注入参数可训练")

    def unfreeze_all(self):
        """
        解冻所有参数
        """
        for param in self.parameters():
            param.requires_grad = True
        print("[DEBUG] 已解冻所有模型参数")