# model_with_spk_inject.py
import os
import json
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel


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


class RNNTWithSpkInject(EncDecRNNTBPEModel):
    """
    在 Conformer 编码器输出后，按论文方式注入“说话人正弦核”。
    - 支持 (B, K, T_enc) 的 spk_labels 输入
    - 自动把 P 的时间轴插值到 T_enc
    - 使用可学习标量 alpha 控制注入强度
    """

    def __init__(self, cfg=None, trainer=None, K_max: int = 4, alpha_init: float = 0.1, base_model=None):
        # 先初始化父类
        from nemo.core.classes.modelPT import ModelPT
        ModelPT.__init__(self, cfg=cfg, trainer=None)
        
        # 设置参数
        self.K_max = int(K_max)
        
        # 然后从基础模型复制组件
        if base_model is not None:
            self.preprocessor = base_model.preprocessor
            self.encoder = base_model.encoder
            self.decoder = base_model.decoder
            self.joint = base_model.joint
            self.loss = base_model.loss
            # tokenizer可能不存在，需要检查
            if hasattr(base_model, 'tokenizer'):
                self.tokenizer = base_model.tokenizer
                print(f"[DEBUG] 成功复制tokenizer: {type(self.tokenizer)}")
            else:
                print(f"[DEBUG] base_model没有tokenizer属性，类型: {type(base_model)}")
                print(f"[DEBUG] base_model所有属性: {[attr for attr in dir(base_model) if not attr.startswith('_')][:20]}")
                # 尝试其他可能的tokenizer属性名
                for attr_name in ['_tokenizer', 'vocab', 'vocabulary', 'decoder', 'joint']:
                    if hasattr(base_model, attr_name):
                        attr_obj = getattr(base_model, attr_name)
                        if hasattr(attr_obj, 'tokenizer'):
                            self.tokenizer = attr_obj.tokenizer
                            print(f"[DEBUG] 从{attr_name}.tokenizer获取tokenizer")
                            break
                        elif hasattr(attr_obj, 'vocabulary'):
                             vocab_obj = attr_obj.vocabulary
                             print(f"[DEBUG] 找到vocabulary对象，类型: {type(vocab_obj)}")
                             # 如果vocabulary是配置对象，尝试获取实际的tokenizer
                             if hasattr(vocab_obj, 'tokenizer'):
                                 self.tokenizer = vocab_obj.tokenizer
                                 print(f"[DEBUG] 从{attr_name}.vocabulary.tokenizer获取tokenizer")
                             else:
                                 # 直接使用vocabulary作为tokenizer（可能需要进一步处理）
                                 self.tokenizer = vocab_obj
                                 print(f"[DEBUG] 直接使用{attr_name}.vocabulary作为tokenizer")
                             break
            
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
        else:
            # 如果没有base_model，设置默认值，稍后动态调整
            self.M = 512  # 默认值
        
        # tokenizer将在外部设置

        # 正弦核 Γ ∈ R^{K×M}（缓冲区，不参与梯度）
        k_ids = torch.arange(self.K_max, dtype=torch.float32).unsqueeze(1)  # (K,1)
        m_ids = torch.arange(self.M, dtype=torch.float32).unsqueeze(0)      # (1,M)
        gamma = torch.sin(2 * torch.pi * k_ids * m_ids / float(self.M))     # (K,M)
        self.register_buffer("Gamma", gamma)  # shape: [K_max, M]

        # 注入强度 α（可学习）
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

        # 运行态缓存：训练/验证 step 会把当前 batch 的 P 放进来
        self._current_spk_labels: Optional[torch.Tensor] = None  # [B,K,T*] or None

    # ---- 注入核心：把 enc_out 注入说话人核 ----
    def inject_speakers(self, enc_out: torch.Tensor, enc_len: torch.Tensor,
                        spk_labels: Optional[torch.Tensor]) -> torch.Tensor:
        """
        enc_out: [B, T_enc, M]
        enc_len: [B]
        spk_labels: [B, K, T_any] 或 None
        return: enc_out_tilde [B, T_enc, M]
        """
        if spk_labels is None:
            return enc_out

        # 检查编码器输出的实际维度
        print(f"DEBUG: enc_out.shape = {enc_out.shape}")
        
        # 根据实际维度判断格式
        if len(enc_out.shape) == 3:
            if enc_out.shape[2] == self.M:  # [B, T_enc, M]格式
                B, T_enc, M = enc_out.shape
                print(f"DEBUG: 检测到格式 [B, T_enc, M]: B={B}, T_enc={T_enc}, M={M}")
            elif enc_out.shape[1] == self.M:  # [B, M, T_enc]格式
                B, M, T_enc = enc_out.shape
                print(f"DEBUG: 检测到格式 [B, M, T_enc]: B={B}, M={M}, T_enc={T_enc}")
                # 转换为[B, T_enc, M]格式
                enc_out = enc_out.transpose(1, 2)
                print(f"DEBUG: 转换后 enc_out.shape = {enc_out.shape}")
            else:
                raise ValueError(f"无法识别编码器输出格式: {enc_out.shape}, 期望M={self.M}")
        else:
            raise ValueError(f"Unexpected enc_out shape: {enc_out.shape}")
        
        B, T_enc, M = enc_out.shape
        assert M == self.M, f"M mismatch: enc_out.M={M} vs self.M={self.M}"

        # A: [B,M,T]
        A = enc_out.transpose(1, 2).contiguous()
        # 归一化
        A_norm = A / (A.norm(p=2, dim=1, keepdim=True) + 1e-8)

        # 确保 P 的时间轴与 T_enc 一致；期望 P: [B,K,T_enc]
        P = spk_labels.to(A_norm.dtype).to(A_norm.device)
        if P.dim() != 3 or P.size(0) != B or P.size(1) != self.K_max:
            raise RuntimeError(f"spk_labels 维度不正确，期望 [B,{self.K_max},T*]，但得到 {tuple(P.shape)}")

        if P.size(2) != T_enc:
            # 使用 1D 线性插值到 T_enc
            # interpolate 期望 [N,C,L]，这里 N=B, C=K, L=T
            P = F.interpolate(P, size=T_enc, mode='linear', align_corners=False)

        # 计算 Γ^T P → [B,M,T]
        # Γ: [K,M]，先转置 [M,K]
        spk_inject = torch.einsum('mk,bkt->bmt', self.Gamma.t(), P)  # [B,M,T]
        A_tilde = A_norm + self.alpha * spk_inject
        # 转换回 [B,T,M] 格式以匹配 joint 网络的期望
        return A_tilde.transpose(1, 2).contiguous()  # [B,T,M]

    # ---- 供 wrapper 使用的纯前向：返回 enc_out_tilde, enc_len ----
    def encode_with_injection(self, input_signal: torch.Tensor, input_signal_length: torch.Tensor,
                              spk_labels: Optional[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        # preprocess
        proc, proc_len = self.preprocessor(input_signal=input_signal, length=input_signal_length)
        # encoder
        enc_out, enc_len = self.encoder(audio_signal=proc, length=proc_len)  # [B,T,M], [B]
        print(f"DEBUG: 原始编码器输出 enc_out.shape = {enc_out.shape}, enc_len = {enc_len}")
        # inject
        enc_out = self.inject_speakers(enc_out, enc_len, spk_labels)
        print(f"DEBUG: 注入后编码器输出 enc_out.shape = {enc_out.shape}")
        
        # 确保enc_len与实际的时间维度匹配
        if len(enc_out.shape) == 3:
            actual_T = enc_out.shape[1]  # 假设格式是[B, T, M]
            # 只有当enc_len超过actual_T时才需要调整
            enc_len = torch.clamp(enc_len, max=actual_T)
            print(f"DEBUG: actual_T: {actual_T}, enc_len调整后: {enc_len}")
        
        return enc_out, enc_len
