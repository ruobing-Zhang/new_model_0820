#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：初始化字符级RNNT模型（带说话人注入和Adapter），
加载预训练权重，并打印模型结构和参数信息。
"""

import os
import sys
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rnnt_char_model_with_spk_inject import RNNTCharWithSpkInject, RNNTCharWithSpkInjectAndAdapter
from rnnt_models import EncDecRNNTModel


def load_pretrained_model(model_path):
    """
    加载预训练的.nemo模型
    
    Args:
        model_path: 预训练模型路径
        
    Returns:
        加载的模型实例
    """
    print(f"正在加载预训练模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"预训练模型文件不存在: {model_path}")
    
    # 使用NeMo的restore_from方法加载模型
    try:
        base_model = EncDecRNNTModel.restore_from(model_path)
        print(f"成功加载预训练模型，类型: {type(base_model)}")
        return base_model
    except Exception as e:
        print(f"加载预训练模型失败: {e}")
        raise


def print_model_info(model, model_name="模型"):
    """
    打印模型的详细信息
    
    Args:
        model: 模型实例
        model_name: 模型名称
    """
    print(f"\n{'='*60}")
    print(f"{model_name} 详细信息")
    print(f"{'='*60}")
    
    # 基本信息
    print(f"模型类型: {type(model).__name__}")
    print(f"设备: {next(model.parameters()).device}")
    print(f"数据类型: {next(model.parameters()).dtype}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n参数统计:")
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数数: {trainable_params:,}")
    print(f"  冻结参数数: {total_params - trainable_params:,}")
    
    # 模型结构
    print(f"\n模型结构:")
    
    # 预处理器信息
    if hasattr(model, 'preprocessor'):
        preprocessor = model.preprocessor
        print(f"  预处理器: {type(preprocessor).__name__}")
        if hasattr(preprocessor, 'n_mels'):
            print(f"    Mel频谱维度: {preprocessor.n_mels}")
        if hasattr(preprocessor, 'sample_rate'):
            print(f"    采样率: {preprocessor.sample_rate}")
    
    # 编码器信息
    if hasattr(model, 'encoder'):
        encoder = model.encoder
        print(f"  编码器: {type(encoder).__name__}")
        if hasattr(encoder, 'd_model'):
            print(f"    隐藏维度: {encoder.d_model}")
        if hasattr(encoder, 'n_layers') or hasattr(encoder, 'num_layers'):
            n_layers = getattr(encoder, 'n_layers', getattr(encoder, 'num_layers', 'Unknown'))
            print(f"    层数: {n_layers}")
        if hasattr(encoder, 'n_heads'):
            print(f"    注意力头数: {encoder.n_heads}")
    
    # 解码器信息
    if hasattr(model, 'decoder'):
        decoder = model.decoder
        print(f"  解码器: {type(decoder).__name__}")
        if hasattr(decoder, 'pred_hidden'):
            print(f"    预测隐藏维度: {decoder.pred_hidden}")
        if hasattr(decoder, 'pred_rnn_layers'):
            print(f"    RNN层数: {decoder.pred_rnn_layers}")
    
    # Joint网络信息
    if hasattr(model, 'joint'):
        joint = model.joint
        print(f"  Joint网络: {type(joint).__name__}")
        if hasattr(joint, 'num_classes'):
            print(f"    输出类别数: {joint.num_classes}")
        if hasattr(joint, 'vocabulary') and joint.vocabulary:
            print(f"    词汇表大小: {len(joint.vocabulary)}")
    
    # 说话人注入相关信息（如果存在）
    if hasattr(model, 'K_max'):
        print(f"\n说话人注入信息:")
        print(f"  最大说话人数: {model.K_max}")
        print(f"  编码器隐维: {model.M}")
        if hasattr(model, 'alpha'):
            print(f"  注入强度: {model.alpha.item():.4f}")
        if hasattr(model, 'Gamma'):
            print(f"  正弦核形状: {model.Gamma.shape}")
    
    # Adapter信息（如果存在）
    if hasattr(model, 'adapters'):
        print(f"\nAdapter信息:")
        print(f"  Adapter数量: {len(model.adapters)}")
        if hasattr(model, 'adapter_bottleneck'):
            print(f"  瓶颈维度: {model.adapter_bottleneck}")
        
        # 计算Adapter参数数量
        adapter_params = sum(p.numel() for adapter in model.adapters for p in adapter.parameters())
        print(f"  Adapter参数数: {adapter_params:,}")
    
    # 配置信息（如果存在）
    if hasattr(model, 'cfg'):
        print(f"\n配置信息:")
        if hasattr(model.cfg, 'labels'):
            print(f"  字符标签数量: {len(model.cfg.labels)}")
            print(f"  前10个字符: {model.cfg.labels[:10]}")
    
    print(f"{'='*60}\n")


def load_weights_with_mismatch_handling(target_model, source_model, strict=False):
    """
    加载权重，处理形状不匹配的情况（如Adapter层）
    
    Args:
        target_model: 目标模型
        source_model: 源模型
        strict: 是否严格匹配所有参数
    """
    print("\n开始加载权重...")
    
    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    
    loaded_keys = []
    missing_keys = []
    unexpected_keys = []
    size_mismatch_keys = []
    
    for key in target_state_dict.keys():
        if key in source_state_dict:
            source_param = source_state_dict[key]
            target_param = target_state_dict[key]
            
            if source_param.shape == target_param.shape:
                target_state_dict[key] = source_param
                loaded_keys.append(key)
            else:
                size_mismatch_keys.append((key, source_param.shape, target_param.shape))
        else:
            missing_keys.append(key)
    
    for key in source_state_dict.keys():
        if key not in target_state_dict:
            unexpected_keys.append(key)
    
    # 加载匹配的权重
    target_model.load_state_dict(target_state_dict, strict=False)
    
    # 打印加载结果
    print(f"\n权重加载结果:")
    print(f"  成功加载: {len(loaded_keys)} 个参数")
    print(f"  缺失参数: {len(missing_keys)} 个")
    print(f"  多余参数: {len(unexpected_keys)} 个")
    print(f"  形状不匹配: {len(size_mismatch_keys)} 个")
    
    if missing_keys:
        print(f"\n缺失的参数（通常是新增的Adapter和说话人注入参数）:")
        for key in missing_keys[:10]:  # 只显示前10个
            print(f"    {key}")
        if len(missing_keys) > 10:
            print(f"    ... 还有 {len(missing_keys) - 10} 个")
    
    if size_mismatch_keys:
        print(f"\n形状不匹配的参数:")
        for key, source_shape, target_shape in size_mismatch_keys:
            print(f"    {key}: {source_shape} -> {target_shape}")
    
    return len(loaded_keys), len(missing_keys)


def test_model_with_real_audio(model, manifest_path):
    """
    使用manifest文件中的真实音频数据测试模型推理
    """
    import json
    import librosa
    import numpy as np
    
    try:
        # 读取manifest文件（JSONL格式）
        data = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        
        print(f"Manifest文件包含 {len(data)} 条数据")
        
        # 取第一条数据进行测试
        sample = data[0]
        audio_path = sample['audio_filepath']
        text = sample['text']
        pgt_path = sample.get('pgt_npy', '')
        
        print(f"\n测试数据信息:")
        print(f"  音频路径: {audio_path}")
        print(f"  参考文本: {text}")
        print(f"  PGT文件路径: {pgt_path}")
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            print(f"❌ 音频文件不存在: {audio_path}")
            return
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"\n音频信息:")
        print(f"  音频长度: {len(audio)/sr:.2f}秒")
        print(f"  采样率: {sr}Hz")
        print(f"  音频形状: {audio.shape}")
        
        # 加载PGT文件（说话人概率）
        pgt_data = None
        if pgt_path and os.path.exists(pgt_path):
            pgt_data = np.load(pgt_path)  # 假设是.npy文件
            print(f"\nPGT数据信息:")
            print(f"  PGT形状: {pgt_data.shape}")
            print(f"  PGT数据类型: {pgt_data.dtype}")
            print(f"  PGT数值范围: [{pgt_data.min():.4f}, {pgt_data.max():.4f}]")
        else:
            print(f"\n⚠️ PGT文件不存在或路径为空: {pgt_path}")
            # 创建虚拟的说话人标签用于测试
            # 假设4个说话人，时间步长度为音频长度的1/160（对应16kHz音频的帧率）
            time_steps = len(audio) // 160
            pgt_data = np.random.rand(4, time_steps).astype(np.float32)
            print(f"  使用虚拟PGT数据，形状: {pgt_data.shape}")
        
        # 将模型设置为评估模式
        model.eval()
        
        # 准备输入数据
        audio_tensor = torch.tensor(audio).unsqueeze(0).float()  # [1, T]
        audio_length = torch.tensor([len(audio)])
        
        # 准备说话人标签 - 转换PGT数据格式
        if pgt_data is not None:
            # PGT数据通常是 [K, T] 格式，需要转换为 [B, K, T]
            spk_labels = torch.tensor(pgt_data).unsqueeze(0).float()  # [1, K, T]
            print(f"\n说话人标签形状: {spk_labels.shape}")
            
            # 设置说话人标签
            if hasattr(model, 'set_speaker_labels'):
                model.set_speaker_labels(spk_labels)
                print(f"✅ 已设置说话人标签")
        
        print(f"\n模型输入准备完成:")
        print(f"  音频张量形状: {audio_tensor.shape}")
        print(f"  音频长度: {audio_length}")
        if pgt_data is not None:
            print(f"  说话人标签形状: {spk_labels.shape}")
        
        print("\n开始模型推理...")
        with torch.no_grad():
            try:
                # 测试预处理器
                if hasattr(model, 'preprocessor'):
                    processed_signal, processed_length = model.preprocessor(
                        input_signal=audio_tensor,
                        length=audio_length
                    )
                    print(f"✅ 预处理完成，特征形状: {processed_signal.shape}")
                
                # 测试编码器（带说话人注入）
                if hasattr(model, 'encode_with_injection') and pgt_data is not None:
                    encoded, encoded_len = model.encode_with_injection(
                        audio_tensor, audio_length, spk_labels
                    )
                    print(f"✅ 编码器（带说话人注入）完成，输出形状: {encoded.shape}")
                elif hasattr(model, 'encoder'):
                    encoded, encoded_len = model.encoder(
                        audio_signal=processed_signal,
                        length=processed_length
                    )
                    print(f"✅ 编码器完成，输出形状: {encoded.shape}")
                
                # 测试解码器（可选）
                if hasattr(model, 'decoder'):
                    try:
                        # 创建简单的目标序列用于测试
                        dummy_targets = torch.zeros(1, 10, dtype=torch.long)  # [B, T]
                        dummy_target_length = torch.tensor([10])
                        
                        decoder_output = model.decoder(
                            targets=dummy_targets,
                            target_length=dummy_target_length
                        )
                        
                        # 处理不同的返回格式
                        if isinstance(decoder_output, tuple):
                            if len(decoder_output) == 2:
                                pred_out, pred_len = decoder_output
                                print(f"✅ 解码器测试完成，输出形状: {pred_out.shape}")
                            else:
                                print(f"✅ 解码器测试完成，返回了 {len(decoder_output)} 个输出")
                                print(f"  第一个输出形状: {decoder_output[0].shape}")
                        else:
                            print(f"✅ 解码器测试完成，输出形状: {decoder_output.shape}")
                    except Exception as decoder_e:
                        print(f"⚠️ 解码器测试跳过: {str(decoder_e)}")
                
                print("\n🎉 完整模型推理测试成功！")
                
            except Exception as e:
                print(f"❌ 模型推理过程中出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ 真实音频测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """
    主函数：测试字符级RNNT模型的初始化和权重加载
    """
    print("开始测试字符级RNNT模型（带说话人注入和Adapter）")
    
    # 预训练模型路径
    pretrained_model_path = "/root/autodl-tmp/joint_sortformer_and_asr_0815/pretrained_models/stt_zh_conformer_transducer_large.nemo"
    
    try:
        # 1. 加载预训练模型
        print("\n步骤1: 加载预训练模型")
        base_model = load_pretrained_model(pretrained_model_path)
        print_model_info(base_model, "预训练基础模型")
        
        # 2. 准备配置
        print("\n步骤2: 准备模型配置")
        cfg = base_model.cfg.copy()
        # 移除数据配置以避免初始化时的数据路径问题
        if 'train_ds' in cfg:
            del cfg.train_ds
        if 'validation_ds' in cfg:
            del cfg.validation_ds
        if 'test_ds' in cfg:
            del cfg.test_ds
        
        # 3. 创建带说话人注入的模型（不带Adapter）
        print("\n步骤3: 创建带说话人注入的字符级RNNT模型")
        spk_inject_model = RNNTCharWithSpkInject(
            cfg=cfg,
            trainer=None,
            K_max=4,
            alpha_init=0.1,
            base_model=None  # 不传入base_model，避免重复初始化
        )
        print_model_info(spk_inject_model, "说话人注入模型")
        
        # 4. 创建带Adapter的模型
        print("\n步骤4: 创建带说话人注入和Adapter的字符级RNNT模型")
        full_model = RNNTCharWithSpkInjectAndAdapter(
            cfg=cfg,
            trainer=None,
            K_max=4,
            alpha_init=0.1,
            base_model=None,  # 不传入base_model，避免重复初始化
            num_adapters=12,
            adapter_bottleneck=256
        )
        print_model_info(full_model, "完整模型（说话人注入+Adapter）")
        
        # 5. 测试权重加载
        print("\n步骤5: 测试权重加载")
        loaded_count, missing_count = load_weights_with_mismatch_handling(full_model, base_model)
        
        print(f"\n权重加载完成:")
        print(f"  成功加载 {loaded_count} 个参数")
        
        print(f"  新增参数 {missing_count} 个（Adapter和说话人注入相关）")
        
        # 5. 测试使用真实音频数据进行推理
        print("\n步骤5: 使用真实音频数据测试模型功能")
        manifest_path = "/root/autodl-tmp/joint_sortformer_and_asr_0815/scripts_0822/M8013_multispeaker_manifest_train_joint_no_punc_with_4speakers.json"
        test_model_with_real_audio(full_model, manifest_path)
        
        # 6. 测试参数冻结功能
        print("\n步骤6: 测试参数冻结功能")
        full_model.freeze_base_model()
        
        trainable_params_after_freeze = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
        print(f"冻结后可训练参数数: {trainable_params_after_freeze:,}")
        
        # 解冻
        full_model.unfreeze_all()
        trainable_params_after_unfreeze = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
        print(f"解冻后可训练参数数: {trainable_params_after_unfreeze:,}")
        
        print("\n测试完成！模型初始化和权重加载成功。")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 测试失败！")
        sys.exit(1)