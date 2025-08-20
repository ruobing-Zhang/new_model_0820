#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说话人注入RNNT模型使用示例

这个脚本展示了如何使用带说话人注入功能的RNNT模型进行语音识别。
"""

import torch
import numpy as np
from model_with_spk_inject import create_model_from_bpe_checkpoint

def main():
    # 1. 创建模型
    print("正在加载模型...")
    checkpoint_path = "../pretrained_models/zh_conformer_transducer_large_bpe_init.nemo"
    pgt_dir = "./output_npy"  # p_gt矩阵文件目录
    
    model = create_model_from_bpe_checkpoint(
        checkpoint_path=checkpoint_path,
        K_max=4,  # 最大说话人数
        alpha_init=0.1,  # 初始注入权重
        pgt_dir=pgt_dir
    )
    
    print(f"模型加载成功！")
    print(f"- 编码器隐藏维度: {model.cfg.encoder.d_model}")
    print(f"- 最大说话人数: {model.K_max}")
    print(f"- 当前注入权重: {model.get_speaker_injection_weight():.3f}")
    
    # 2. 准备测试数据
    batch_size = 2
    audio_length = 16000 * 3  # 3秒音频
    
    # 模拟音频信号
    input_signal = torch.randn(batch_size, audio_length)
    input_signal_length = torch.tensor([audio_length, audio_length])
    
    # 模拟音频文件路径（用于加载p_gt矩阵）
    audio_filepaths = [
        "M8013_segment_049.npy",  # 这个文件存在
        "nonexistent_file.npy"    # 这个文件不存在，会使用零矩阵
    ]
    
    print("\n=== 测试不同的前向传播方式 ===")
    
    # 3. 测试不带说话人信息的前向传播
    print("\n1. 不带说话人信息的前向传播:")
    with torch.no_grad():
        output1 = model(input_signal=input_signal, input_signal_length=input_signal_length)
        print(f"   输出形状: {[x.shape for x in output1]}")
    
    # 4. 测试带说话人标签的前向传播
    print("\n2. 带说话人标签的前向传播:")
    with torch.no_grad():
        # 获取编码器输出长度
        processed_signal, processed_length = model.preprocessor(input_signal=input_signal, length=input_signal_length)
        enc_out, enc_len = model.encoder(audio_signal=processed_signal, length=processed_length)
        T_enc = enc_out.shape[1]
        
        # 创建说话人标签矩阵
        spk_labels = torch.zeros(batch_size, model.K_max, T_enc)
        # 第一个样本：说话人0在前半段，说话人1在后半段
        spk_labels[0, 0, :T_enc//2] = 1.0
        spk_labels[0, 1, T_enc//2:] = 1.0
        # 第二个样本：说话人2全程
        spk_labels[1, 2, :] = 1.0
        
        output2 = model(input_signal=input_signal, input_signal_length=input_signal_length, spk_labels=spk_labels)
        print(f"   输出形状: {[x.shape for x in output2]}")
    
    # 5. 测试带文件路径的前向传播
    print("\n3. 带文件路径的前向传播（从p_gt文件加载）:")
    with torch.no_grad():
        output3 = model(input_signal=input_signal, input_signal_length=input_signal_length, audio_filepaths=audio_filepaths)
        print(f"   输出形状: {[x.shape for x in output3]}")
    
    # 6. 调整说话人注入权重
    print("\n=== 调整说话人注入权重 ===")
    original_weight = model.get_speaker_injection_weight()
    print(f"原始权重: {original_weight:.3f}")
    
    # 设置新权重
    new_weight = 0.5
    model.set_speaker_injection_weight(new_weight)
    print(f"新权重: {model.get_speaker_injection_weight():.3f}")
    
    # 测试新权重下的前向传播
    with torch.no_grad():
        output4 = model(input_signal=input_signal, input_signal_length=input_signal_length, spk_labels=spk_labels)
        print(f"新权重下的输出形状: {[x.shape for x in output4]}")
    
    # 恢复原始权重
    model.set_speaker_injection_weight(original_weight)
    print(f"恢复原始权重: {model.get_speaker_injection_weight():.3f}")
    
    print("\n=== 使用示例完成 ===")
    print("\n使用说明:")
    print("1. 不带说话人信息时，模型行为与原始RNNT模型相同")
    print("2. 带说话人标签时，需要提供形状为(B, K_max, T_enc)的spk_labels")
    print("3. 带文件路径时，模型会自动从指定目录加载对应的p_gt矩阵")
    print("4. 可以通过set_speaker_injection_weight()调整注入强度")
    print("5. 模型输出与原始RNNT模型兼容，返回(encoder_outputs, encoder_lengths)")

if __name__ == "__main__":
    main()