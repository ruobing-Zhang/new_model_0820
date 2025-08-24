#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多说话人语音识别推理脚本
使用训练后的Adapter微调模型进行推理
"""

import torch
import json
import os
import argparse
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm
import time

# NeMo imports
import nemo
from nemo.core.config import hydra_runner
from omegaconf import DictConfig, OmegaConf

# 自定义模型导入
from rnnt_char_model_with_spk_inject import RNNTCharWithSpkInjectAndAdapter
from adapter_finetune_train import AdapterFineTuneModel

def load_manifest(manifest_path):
    """加载manifest文件"""
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_trained_model(model_path, device='cuda'):
    """加载训练后的模型"""
    print(f"加载模型: {model_path}")
    
    # 加载保存的模型数据
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取配置和说话人标记
    config = checkpoint.get('config', {})
    speaker_tokens = checkpoint.get('speaker_tokens', ['<spk:0>', '<spk:1>', '<spk:2>', '<spk:3>'])
    
    print(f"说话人标记: {speaker_tokens}")
    
    # 直接从checkpoint恢复模型
    # 创建一个简化的推理模型类
    import torch.nn as nn
    
    class InferenceModel(nn.Module):
        def __init__(self, checkpoint_data):
            super().__init__()
            self.config = checkpoint_data.get('config', {})
            self.speaker_tokens = checkpoint_data.get('speaker_tokens', [])
            
            # 从checkpoint中恢复模型组件
            if 'model_state_dict' in checkpoint_data:
                # 这里需要重建模型结构来匹配保存的状态字典
                # 由于我们直接加载状态，我们需要确保模型结构匹配
                pass
        
        def forward(self, *args, **kwargs):
            # 推理时的前向传播
            pass
    
    # 尝试直接加载完整的模型对象（如果保存了的话）
    if 'model' in checkpoint:
        model = checkpoint['model']
    else:
        # 否则创建推理模型并加载状态字典
        model = InferenceModel(checkpoint)
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                print(f"加载状态字典失败: {e}")
                # 如果状态字典不匹配，我们需要重新创建AdapterFineTuneModel
                model_params = {
                    'pretrained_model_path': "dummy",
                    'K_max': config.get('K_max', 4),
                    'alpha_init': config.get('alpha_init', 0.1),
                    'num_adapters': config.get('num_adapters', 4),
                    'adapter_bottleneck': config.get('adapter_bottleneck', 256),
                    'learning_rate': config.get('learning_rate', 1e-4),
                    'weight_decay': config.get('weight_decay', 1e-6),
                    'batch_size': config.get('batch_size', 8),
                    'num_workers': config.get('num_workers', 4),
                    'sample_rate': config.get('sample_rate', 16000)
                }
                model = AdapterFineTuneModel(**model_params)
                model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    print(f"模型加载完成，设备: {device}")
    return model, speaker_tokens

def extract_speaker_from_text(text):
    """从文本中提取说话人标记"""
    # 查找 <spk:X> 格式的说话人标记
    import re
    pattern = r'<spk:(\d+)>'
    matches = re.findall(pattern, text)
    if matches:
        return int(matches[0])
    return 0  # 默认说话人

def preprocess_audio(audio_path, target_sr=16000):
    """预处理音频文件"""
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # 转换为torch tensor
    audio_tensor = torch.FloatTensor(audio)
    
    return audio_tensor

def inference_single_sample(model, audio_tensor, speaker_id, device='cuda'):
    """对单个样本进行推理"""
    with torch.no_grad():
        try:
            # 准备输入数据
            audio_tensor = audio_tensor.unsqueeze(0).to(device)  # [1, T]
            audio_length = torch.tensor([audio_tensor.shape[1]], device=device)
            
            # 准备说话人标签 (4个说话人的one-hot编码)
            speaker_labels = torch.zeros(1, 4, audio_tensor.shape[1], device=device)
            speaker_labels[0, speaker_id, :] = 1.0
            
            # 使用模型进行推理
            # 由于我们的模型是基于NeMo的，我们需要调用其内部的推理方法
            # 这里我们直接调用模型的forward方法进行贪婪解码
            
            # 首先通过预处理器
            processed_signal, processed_signal_length = model.preprocessor(
                input_signal=audio_tensor, length=audio_length
            )
            
            # 通过编码器 (包含说话人注入)
            encoded, encoded_len = model.encoder(
                audio_signal=processed_signal, 
                length=processed_signal_length,
                speaker_labels=speaker_labels
            )
            
            # 贪婪解码
            best_hyp, _ = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_len,
                return_hypotheses=False,
            )
            
            # 转换为文本
            if best_hyp is not None and len(best_hyp) > 0:
                # 获取第一个假设的token ids
                token_ids = best_hyp[0].cpu().numpy()
                # 使用tokenizer解码
                text = model.tokenizer.ids_to_text(token_ids)
                return text
            else:
                return ""
                
        except Exception as e:
            print(f"推理错误: {e}")
            import traceback
            traceback.print_exc()
            return ""

def run_inference(model_path, manifest_path, output_path, device='cuda', max_samples=None):
    """运行推理"""
    print(f"开始多说话人语音识别推理")
    print(f"模型路径: {model_path}")
    print(f"数据路径: {manifest_path}")
    print(f"输出路径: {output_path}")
    print(f"设备: {device}")
    
    # 加载模型
    model, speaker_tokens = load_trained_model(model_path, device)
    
    # 加载数据
    data = load_manifest(manifest_path)
    if max_samples:
        data = data[:max_samples]
    
    print(f"总样本数: {len(data)}")
    
    # 推理结果
    results = []
    
    # 开始推理
    start_time = time.time()
    
    for i, sample in enumerate(tqdm(data, desc="推理进度")):
        try:
            # 获取音频路径和文本
            audio_path = sample['audio_filepath']
            ground_truth = sample['text']
            
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                print(f"音频文件不存在: {audio_path}")
                continue
            
            # 提取说话人ID
            speaker_id = extract_speaker_from_text(ground_truth)
            
            # 预处理音频
            audio_tensor = preprocess_audio(audio_path)
            
            # 推理
            predicted_text = inference_single_sample(
                model, audio_tensor, speaker_id, device
            )
            
            # 保存结果
            result = {
                'audio_filepath': audio_path,
                'ground_truth': ground_truth,
                'predicted': predicted_text,
                'speaker_id': speaker_id,
                'sample_id': i
            }
            results.append(result)
            
            # 打印部分结果
            if i < 5 or i % 50 == 0:
                print(f"\n样本 {i}:")
                print(f"  音频: {os.path.basename(audio_path)}")
                print(f"  说话人: {speaker_id}")
                print(f"  真实: {ground_truth}")
                print(f"  预测: {predicted_text}")
            
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            continue
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n推理完成!")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均每样本: {total_time/len(results):.3f}秒")
    print(f"成功处理: {len(results)}/{len(data)} 样本")
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"结果已保存到: {output_path}")
    
    # 计算简单的准确率统计
    calculate_simple_metrics(results)
    
    return results

def calculate_simple_metrics(results):
    """计算简单的评估指标"""
    if not results:
        return
    
    print("\n=== 简单评估指标 ===")
    
    # 按说话人统计
    speaker_stats = {}
    total_samples = len(results)
    
    for result in results:
        speaker_id = result['speaker_id']
        if speaker_id not in speaker_stats:
            speaker_stats[speaker_id] = {'count': 0, 'non_empty': 0}
        
        speaker_stats[speaker_id]['count'] += 1
        if result['predicted'].strip():
            speaker_stats[speaker_id]['non_empty'] += 1
    
    print(f"总样本数: {total_samples}")
    print("\n按说话人统计:")
    for speaker_id, stats in speaker_stats.items():
        success_rate = stats['non_empty'] / stats['count'] * 100
        print(f"  说话人 {speaker_id}: {stats['count']} 样本, {stats['non_empty']} 成功推理 ({success_rate:.1f}%)")
    
    # 总体成功率
    total_success = sum(stats['non_empty'] for stats in speaker_stats.values())
    overall_success_rate = total_success / total_samples * 100
    print(f"\n总体成功推理率: {total_success}/{total_samples} ({overall_success_rate:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='多说话人语音识别推理')
    parser.add_argument('--model_path', type=str, 
                       default='./adapter_finetune_output/final_model.pt',
                       help='训练后的模型路径')
    parser.add_argument('--manifest_path', type=str,
                       default='/root/autodl-tmp/joint_sortformer_and_asr_0815/data/M8013_multispeaker_manifest_train_joint_no_punc_with_4speakers.json',
                       help='测试数据manifest路径')
    parser.add_argument('--output_path', type=str,
                       default='./inference_results.jsonl',
                       help='推理结果输出路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='推理设备 (cuda/cpu)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大推理样本数 (用于测试)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在 {args.model_path}")
        return
    
    if not os.path.exists(args.manifest_path):
        print(f"错误: 数据文件不存在 {args.manifest_path}")
        return
    
    # 运行推理
    results = run_inference(
        model_path=args.model_path,
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        device=args.device,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()