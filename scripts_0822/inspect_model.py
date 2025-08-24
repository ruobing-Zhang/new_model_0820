#!/usr/bin/env python3
"""
模型检查脚本：加载并打印NeMo模型的完整配置和参数信息
"""

import torch
import argparse
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.utils import logging
import json
from omegaconf import OmegaConf

def inspect_model(model_path):
    """
    检查模型的配置和参数
    
    Args:
        model_path: 模型文件路径
    """
    
    print("=" * 80)
    print(f"正在加载模型: {model_path}")
    print("=" * 80)
    
    try:
        # 加载模型
        model = EncDecRNNTBPEModel.restore_from(model_path)
        print("✓ 模型加载成功")
        
        print("\n" + "=" * 80)
        print("模型基本信息")
        print("=" * 80)
        
        # 基本信息
        print(f"模型类型: {type(model).__name__}")
        
        # 采样率信息（可能不存在）
        if hasattr(model, 'sample_rate'):
            print(f"采样率: {model.sample_rate} Hz")
        elif hasattr(model, '_cfg') and 'sample_rate' in model._cfg:
            print(f"采样率: {model._cfg.sample_rate} Hz")
        else:
            print("采样率: 未找到采样率信息")
        
        # 模型配置
        if hasattr(model, '_cfg'):
            print(f"\n配置对象类型: {type(model._cfg)}")
            
        print("\n" + "=" * 80)
        print("完整模型配置")
        print("=" * 80)
        
        # 打印完整配置
        if hasattr(model, '_cfg'):
            config_dict = OmegaConf.to_container(model._cfg, resolve=True)
            print(json.dumps(config_dict, indent=2, ensure_ascii=False))
        else:
            print("未找到配置信息")
        
        print("\n" + "=" * 80)
        print("模型架构详情")
        print("=" * 80)
        
        # Preprocessor信息
        if hasattr(model, 'preprocessor'):
            print(f"\n[Preprocessor]")
            print(f"  类型: {type(model.preprocessor).__name__}")
            
            # 安全地获取preprocessor属性
            if hasattr(model.preprocessor, 'features'):
                print(f"  特征维度: {model.preprocessor.features}")
            elif hasattr(model.preprocessor, 'n_mels'):
                print(f"  特征维度: {model.preprocessor.n_mels}")
            
            if hasattr(model.preprocessor, 'window_size'):
                print(f"  窗口大小: {model.preprocessor.window_size}")
            elif hasattr(model.preprocessor, 'win_length'):
                print(f"  窗口大小: {model.preprocessor.win_length}")
            
            if hasattr(model.preprocessor, 'window_stride'):
                print(f"  窗口步长: {model.preprocessor.window_stride}")
            elif hasattr(model.preprocessor, 'hop_length'):
                print(f"  窗口步长: {model.preprocessor.hop_length}")
            
            if hasattr(model.preprocessor, 'n_fft'):
                print(f"  FFT点数: {model.preprocessor.n_fft}")
            
            # 显示所有可用属性（调试用）
            attrs = [attr for attr in dir(model.preprocessor) if not attr.startswith('_') and not callable(getattr(model.preprocessor, attr))]
            if attrs:
                print(f"  可用属性: {', '.join(attrs[:10])}{'...' if len(attrs) > 10 else ''}")
        
        # Encoder信息
        if hasattr(model, 'encoder'):
            print(f"\n[Encoder]")
            print(f"  类型: {type(model.encoder).__name__}")
            
            # 安全地获取encoder属性
            if hasattr(model.encoder, 'num_layers'):
                print(f"  层数: {model.encoder.num_layers}")
            elif hasattr(model.encoder, 'n_layers'):
                print(f"  层数: {model.encoder.n_layers}")
            
            if hasattr(model.encoder, 'd_model'):
                print(f"  模型维度: {model.encoder.d_model}")
            
            if hasattr(model.encoder, 'n_heads'):
                print(f"  注意力头数: {model.encoder.n_heads}")
            
            if hasattr(model.encoder, 'feat_in'):
                print(f"  输入特征维度: {model.encoder.feat_in}")
            
            if hasattr(model.encoder, 'feat_out'):
                print(f"  输出特征维度: {model.encoder.feat_out}")
            
            if hasattr(model.encoder, 'subsampling_factor'):
                print(f"  子采样因子: {model.encoder.subsampling_factor}")
            
            if hasattr(model.encoder, 'conv_kernel_size'):
                print(f"  卷积核大小: {model.encoder.conv_kernel_size}")
            
            if hasattr(model.encoder, 'ff_expansion_factor'):
                print(f"  前馈扩展因子: {model.encoder.ff_expansion_factor}")
            
            # 显示所有可用属性（调试用）
            attrs = [attr for attr in dir(model.encoder) if not attr.startswith('_') and not callable(getattr(model.encoder, attr))]
            if attrs:
                print(f"  可用属性: {', '.join(attrs[:10])}{'...' if len(attrs) > 10 else ''}")
        
        # Decoder信息
        if hasattr(model, 'decoder'):
            print(f"\n[Decoder]")
            print(f"  类型: {type(model.decoder).__name__}")
            print(f"  词汇表大小: {model.decoder.vocab_size}")
            if hasattr(model.decoder, 'prediction_network'):
                pred_net = model.decoder.prediction_network
                print(f"  预测网络隐藏维度: {pred_net.pred_hidden}")
                print(f"  预测网络RNN层数: {pred_net.pred_rnn_layers}")
        
        # Joint Network信息
        if hasattr(model, 'joint'):
            print(f"\n[Joint Network]")
            print(f"  类型: {type(model.joint).__name__}")
            if hasattr(model.joint, 'joint_network'):
                joint_net = model.joint.joint_network
                print(f"  联合网络隐藏维度: {joint_net.joint_hidden}")
                print(f"  激活函数: {joint_net.activation}")
        
        # Tokenizer信息
        if hasattr(model, 'tokenizer'):
            print(f"\n[Tokenizer]")
            print(f"  类型: {type(model.tokenizer).__name__}")
            print(f"  词汇表大小: {model.tokenizer.vocab_size}")
            if hasattr(model.tokenizer, 'model_path'):
                print(f"  模型路径: {model.tokenizer.model_path}")
        
        print("\n" + "=" * 80)
        print("模型参数统计")
        print("=" * 80)
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"不可训练参数数量: {total_params - trainable_params:,}")
        print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")
        
        # 各模块参数统计
        print(f"\n各模块参数分布:")
        
        if hasattr(model, 'preprocessor'):
            preprocessor_params = sum(p.numel() for p in model.preprocessor.parameters())
            print(f"  Preprocessor: {preprocessor_params:,} ({preprocessor_params/total_params*100:.2f}%)")
        
        if hasattr(model, 'encoder'):
            encoder_params = sum(p.numel() for p in model.encoder.parameters())
            print(f"  Encoder: {encoder_params:,} ({encoder_params/total_params*100:.2f}%)")
        
        if hasattr(model, 'decoder'):
            decoder_params = sum(p.numel() for p in model.decoder.parameters())
            print(f"  Decoder: {decoder_params:,} ({decoder_params/total_params*100:.2f}%)")
        
        if hasattr(model, 'joint'):
            joint_params = sum(p.numel() for p in model.joint.parameters())
            print(f"  Joint Network: {joint_params:,} ({joint_params/total_params*100:.2f}%)")
        
        print("\n" + "=" * 80)
        print("模型状态信息")
        print("=" * 80)
        
        print(f"训练模式: {model.training}")
        print(f"设备: {next(model.parameters()).device}")
        print(f"数据类型: {next(model.parameters()).dtype}")
        
        # 测试tokenizer功能
        if hasattr(model, 'tokenizer'):
            print("\n" + "=" * 80)
            print("Tokenizer测试")
            print("=" * 80)
            
            test_texts = ["你好世界", "这是一个测试", "语音识别"]
            for text in test_texts:
                try:
                    tokens = model.tokenizer.text_to_tokens(text)
                    ids = model.tokenizer.text_to_ids(text)
                    decoded = model.tokenizer.ids_to_text(ids)
                    print(f"文本: '{text}'")
                    print(f"  Tokens: {tokens}")
                    print(f"  IDs: {ids}")
                    print(f"  解码: '{decoded}'")
                    print()
                except Exception as e:
                    print(f"文本: '{text}' - 错误: {e}")
        
        print("\n" + "=" * 80)
        print("检查完成")
        print("=" * 80)
        
        return model
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="检查NeMo模型配置和参数")
    parser.add_argument("--model_path", type=str, 
                       default="/root/autodl-tmp/joint_sortformer_and_asr_0815/pretrained_models/zh_conformer_transducer_large_bpe_init.nemo",
                       help="模型文件路径")
    parser.add_argument("--save_config", type=str, default=None,
                       help="保存配置到文件")
    
    args = parser.parse_args()
    
    # 检查模型
    model = inspect_model(args.model_path)
    
    # 保存配置（可选）
    if args.save_config and model is not None:
        try:
            if hasattr(model, '_cfg'):
                config_dict = OmegaConf.to_container(model._cfg, resolve=True)
                with open(args.save_config, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                print(f"\n配置已保存到: {args.save_config}")
            else:
                print("\n无法保存配置：未找到配置信息")
        except Exception as e:
            print(f"\n保存配置失败: {e}")

if __name__ == "__main__":
    main()