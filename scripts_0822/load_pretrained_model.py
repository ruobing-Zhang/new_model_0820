#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加载预训练模型并查看其结构和参数
"""

import os
import sys
import torch
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

def print_model_structure(model, model_name="模型"):
    """
    打印模型结构和参数信息
    """
    print(f"\n=== {model_name} 详细信息 ===")
    
    # 基本信息
    print(f"\n📋 基本信息:")
    print(f"  模型类型: {type(model).__name__}")
    print(f"  设备: {next(model.parameters()).device}")
    print(f"  数据类型: {next(model.parameters()).dtype}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 参数统计:")
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {total_params - trainable_params:,}")
    
    # 模型配置
    if hasattr(model, 'cfg'):
        print(f"\n⚙️ 模型配置:")
        print(f"  配置类型: {type(model.cfg).__name__}")
        
        # 预处理器信息
        if hasattr(model, 'preprocessor') and model.preprocessor is not None:
            print(f"\n🔧 预处理器 ({type(model.preprocessor).__name__}):")
            preprocessor = model.preprocessor
            
            # 尝试获取各种属性
            attrs_to_check = ['sample_rate', 'n_mels', 'n_fft', 'win_length', 'hop_length', 
                            'window_size', 'window_stride', 'features']
            for attr in attrs_to_check:
                if hasattr(preprocessor, attr):
                    value = getattr(preprocessor, attr)
                    print(f"    {attr}: {value}")
                elif hasattr(preprocessor, '_cfg') and hasattr(preprocessor._cfg, attr):
                    value = getattr(preprocessor._cfg, attr)
                    print(f"    {attr}: {value}")
        
        # 编码器信息
        if hasattr(model, 'encoder') and model.encoder is not None:
            print(f"\n🧠 编码器 ({type(model.encoder).__name__}):")
            encoder = model.encoder
            
            # 尝试获取各种属性
            attrs_to_check = ['d_model', 'n_heads', 'num_layers', 'n_layers', 'feat_in', 
                            'feat_out', 'subsampling_factor', 'conv_kernel_size']
            for attr in attrs_to_check:
                if hasattr(encoder, attr):
                    value = getattr(encoder, attr)
                    print(f"    {attr}: {value}")
                elif hasattr(encoder, '_cfg') and hasattr(encoder._cfg, attr):
                    value = getattr(encoder._cfg, attr)
                    print(f"    {attr}: {value}")
        
        # 解码器信息
        if hasattr(model, 'decoder') and model.decoder is not None:
            print(f"\n📝 解码器 ({type(model.decoder).__name__}):")
            decoder = model.decoder
            
            attrs_to_check = ['pred_hidden', 'pred_rnn_layers', 'vocab_size']
            for attr in attrs_to_check:
                if hasattr(decoder, attr):
                    value = getattr(decoder, attr)
                    print(f"    {attr}: {value}")
                elif hasattr(decoder, '_cfg') and hasattr(decoder._cfg, attr):
                    value = getattr(decoder._cfg, attr)
                    print(f"    {attr}: {value}")
        
        # Joint网络信息
        if hasattr(model, 'joint') and model.joint is not None:
            print(f"\n🔗 Joint网络 ({type(model.joint).__name__}):")
            joint = model.joint
            
            attrs_to_check = ['joint_hidden', 'activation']
            for attr in attrs_to_check:
                if hasattr(joint, attr):
                    value = getattr(joint, attr)
                    print(f"    {attr}: {value}")
                elif hasattr(joint, '_cfg') and hasattr(joint._cfg, attr):
                    value = getattr(joint._cfg, attr)
                    print(f"    {attr}: {value}")
        
        # Tokenizer信息
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            print(f"\n🔤 Tokenizer ({type(model.tokenizer).__name__}):")
            tokenizer = model.tokenizer
            
            if hasattr(tokenizer, 'vocab_size'):
                print(f"    词汇表大小: {tokenizer.vocab_size}")
            
            # 测试tokenizer
            try:
                test_text = "你好世界"
                tokens = tokenizer.text_to_ids(test_text)
                decoded = tokenizer.ids_to_text(tokens)
                print(f"    测试编码 '{test_text}' -> {tokens}")
                print(f"    测试解码 {tokens} -> '{decoded}'")
            except Exception as e:
                print(f"    Tokenizer测试失败: {e}")
    
    # 模块参数分布
    print(f"\n📈 各模块参数分布:")
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            module_params[name] = params
            print(f"    {name}: {params:,} 参数")
    
    # 模型状态
    print(f"\n🔍 模型状态:")
    print(f"    训练模式: {model.training}")
    
    return total_params, trainable_params

def main():
    """
    主函数：加载预训练模型并查看结构
    """
    print("开始加载预训练模型...")
    
    # 预训练模型路径
    model_path = "/root/autodl-tmp/joint_sortformer_and_asr_0815/pretrained_models/stt_zh_conformer_transducer_large.nemo"
    
    try:
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        print(f"📁 模型路径: {model_path}")
        
        # 加载模型
        print("\n🔄 正在加载模型...")
        model = nemo_asr.models.EncDecRNNTModel.restore_from(model_path, map_location='cpu')
        
        print("✅ 模型加载成功！")
        
        # 打印模型结构和参数
        total_params, trainable_params = print_model_structure(model, "预训练RNNT模型")
        
        # 打印完整配置（可选）
        print("\n" + "="*50)
        print("📋 完整模型配置:")
        print("="*50)
        if hasattr(model, 'cfg'):
            print(OmegaConf.to_yaml(model.cfg))
        
        return True
        
    except Exception as e:
        print(f"❌ 加载模型失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 模型加载和分析完成！")
    else:
        print("\n❌ 模型加载失败！")
        sys.exit(1)