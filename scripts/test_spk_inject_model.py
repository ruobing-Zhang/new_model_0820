#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试带说话人注入的RNNT模型
"""

import torch
import numpy as np
import os
from model_with_spk_inject import RNNTWithSpkInject, create_model_from_bpe_checkpoint

def test_model():
    """
    测试模型的各项功能
    """
    print("=== 测试带说话人注入的RNNT模型 ===")
    
    # 设置路径
    bpe_model_path = "../pretrained_models/zh_conformer_transducer_large_bpe_init.nemo"
    pgt_dir = "./output_npy"
    
    # 检查文件是否存在
    if not os.path.exists(bpe_model_path):
        print(f"错误: BPE模型文件不存在: {bpe_model_path}")
        return
    
    if not os.path.exists(pgt_dir):
        print(f"错误: P_gt目录不存在: {pgt_dir}")
        return
    
    print(f"\n1. 从BPE检查点创建模型...")
    try:
        model = create_model_from_bpe_checkpoint(
            checkpoint_path=bpe_model_path,
            K_max=4,
            alpha_init=0.1,
            pgt_dir=pgt_dir
        )
        print("✓ 模型创建成功")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return
    
    print(f"\n2. 检查模型属性...")
    print(f"  - 模型类型: {type(model).__name__}")
    print(f"  - 编码器隐藏维度: {model.cfg.encoder.d_model}")
    print(f"  - 最大说话人数: {model.K_max}")
    print(f"  - 正弦核矩阵形状: {model.Gamma.shape}")
    print(f"  - 当前注入权重: {model.get_speaker_injection_weight():.3f}")
    print(f"  - 分词器词汇表大小: {model.tokenizer.vocab_size}")
    
    # 测试分词器中的说话人token
    print(f"\n3. 测试分词器中的说话人token...")
    test_text = "<|spk0|>你好世界<|spk1|>这是测试"
    try:
        tokens = model.tokenizer.text_to_tokens(test_text)
        token_ids = model.tokenizer.text_to_ids(test_text)
        recovered_text = model.tokenizer.ids_to_text(token_ids)
        
        print(f"  - 原始文本: {test_text}")
        print(f"  - 分词结果: {tokens}")
        print(f"  - Token IDs: {token_ids}")
        print(f"  - 恢复文本: {recovered_text}")
        
        # 检查说话人token是否正确处理
        spk_tokens = ['<|spk0|>', '<|spk1|>', '<|spk2|>', '<|spk3|>']
        for spk_token in spk_tokens:
            if spk_token in tokens:
                print(f"  ✓ 说话人token {spk_token} 正确识别")
            else:
                print(f"  ✗ 说话人token {spk_token} 未正确识别")
                
    except Exception as e:
        print(f"  ✗ 分词器测试失败: {e}")
    
    print(f"\n4. 测试p_gt矩阵加载...")
    # 获取一个示例文件
    npy_files = [f for f in os.listdir(pgt_dir) if f.endswith('.npy')]
    if npy_files:
        test_file = npy_files[0]
        test_audio_path = f"dummy_path/{test_file.replace('.npy', '.wav')}"
        
        print(f"  - 测试文件: {test_file}")
        
        # 直接加载numpy文件查看格式
        pgt_path = os.path.join(pgt_dir, test_file)
        pgt_data = np.load(pgt_path)
        print(f"  - P_gt矩阵形状: {pgt_data.shape}")
        print(f"  - P_gt矩阵数据类型: {pgt_data.dtype}")
        print(f"  - P_gt矩阵值范围: [{pgt_data.min():.3f}, {pgt_data.max():.3f}]")
        
        # 测试模型的加载函数
        try:
            # 模拟音频文件路径
            audio_filepath = test_file.replace('.npy', '.wav')
            pgt_tensor = model.load_pgt_matrix(audio_filepath)
            if pgt_tensor is not None:
                print(f"  ✓ P_gt矩阵加载成功，形状: {pgt_tensor.shape}")
            else:
                print(f"  ✗ P_gt矩阵加载失败")
        except Exception as e:
            print(f"  ✗ P_gt矩阵加载测试失败: {e}")
    else:
        print(f"  ✗ 未找到.npy文件")
    
    print(f"\n5. 测试模型前向传播...")
    try:
        # 创建模拟输入
        batch_size = 2
        audio_length = 16000 * 3  # 3秒音频
        
        input_signal = torch.randn(batch_size, audio_length)
        input_signal_length = torch.tensor([audio_length, audio_length])
        
        # 测试不带说话人信息的前向传播
        print(f"  - 测试不带说话人信息的前向传播...")
        with torch.no_grad():
            output1 = model(input_signal=input_signal, input_signal_length=input_signal_length)
            print(f"    ✓ 输出形状: {[x.shape if hasattr(x, 'shape') else type(x) for x in output1]}")
        
        # 测试带说话人标签的前向传播
        print(f"  - 测试带说话人标签的前向传播...")
        # 获取编码器输出长度来创建匹配的说话人标签
        with torch.no_grad():
            processed_signal, processed_length = model.preprocessor(input_signal=input_signal, length=input_signal_length)
            enc_out, enc_len = model.encoder(audio_signal=processed_signal, length=processed_length)
            T_enc = enc_out.shape[1]
            
            # 创建模拟说话人标签
            spk_labels = torch.zeros(batch_size, model.K_max, T_enc)
            # 第一个样本：说话人0在前半段，说话人1在后半段
            spk_labels[0, 0, :T_enc//2] = 1.0
            spk_labels[0, 1, T_enc//2:] = 1.0
            # 第二个样本：说话人2全程
            spk_labels[1, 2, :] = 1.0
            
            output2 = model(input_signal=input_signal, input_signal_length=input_signal_length, spk_labels=spk_labels)
            print(f"    ✓ 带说话人标签的输出形状: {[x.shape if hasattr(x, 'shape') else type(x) for x in output2]}")
        
        # 测试带文件路径的前向传播（如果有p_gt文件）
        if npy_files:
            print(f"  - 测试带文件路径的前向传播...")
            audio_filepaths = [npy_files[0].replace('.npy', '.wav'), npy_files[0].replace('.npy', '.wav')]
            with torch.no_grad():
                output3 = model(input_signal=input_signal, input_signal_length=input_signal_length, audio_filepaths=audio_filepaths)
                print(f"    ✓ 带文件路径的输出形状: {[x.shape if hasattr(x, 'shape') else type(x) for x in output3]}")
        
        print(f"  ✓ 所有前向传播测试通过")
        
    except Exception as e:
        print(f"  ✗ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n6. 测试说话人注入权重调整...")
    try:
        original_weight = model.get_speaker_injection_weight()
        model.set_speaker_injection_weight(0.5)
        new_weight = model.get_speaker_injection_weight()
        print(f"  - 原始权重: {original_weight:.3f}")
        print(f"  - 新权重: {new_weight:.3f}")
        if abs(new_weight - 0.5) < 1e-6:
            print(f"  ✓ 权重调整成功")
        else:
            print(f"  ✗ 权重调整失败")
        
        # 恢复原始权重
        model.set_speaker_injection_weight(original_weight)
    except Exception as e:
        print(f"  ✗ 权重调整测试失败: {e}")
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    test_model()