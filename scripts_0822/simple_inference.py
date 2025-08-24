#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的多说话人语音识别推理脚本
"""

import os
import sys
import json
import torch
import librosa
import numpy as np
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_manifest(manifest_path):
    """加载manifest文件"""
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def simple_inference(model_path, manifest_path, output_path, max_samples=10, device='cuda'):
    """简化的推理函数"""
    print(f"开始推理...")
    print(f"模型路径: {model_path}")
    print(f"数据路径: {manifest_path}")
    print(f"输出路径: {output_path}")
    
    # 加载模型checkpoint
    print("加载模型checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 打印checkpoint信息
    print(f"Checkpoint包含的键: {list(checkpoint.keys())}")
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"模型配置: {config}")
    
    if 'speaker_tokens' in checkpoint:
        speaker_tokens = checkpoint['speaker_tokens']
        print(f"说话人标记: {speaker_tokens}")
    
    if 'hyperparameters' in checkpoint:
        hyperparams = checkpoint['hyperparameters']
        print(f"超参数: {hyperparams}")
    
    # 加载manifest数据
    print("加载manifest数据...")
    manifest_data = load_manifest(manifest_path)
    print(f"总样本数: {len(manifest_data)}")
    
    # 限制样本数量
    if max_samples > 0:
        manifest_data = manifest_data[:max_samples]
        print(f"处理样本数: {len(manifest_data)}")
    
    # 创建输出结果
    results = []
    
    for i, sample in enumerate(manifest_data):
        print(f"\n处理样本 {i+1}/{len(manifest_data)}")
        
        # 获取音频文件路径和标签
        audio_filepath = sample['audio_filepath']
        text = sample.get('text', '')
        
        print(f"音频文件: {audio_filepath}")
        print(f"真实文本: {text}")
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_filepath):
            print(f"警告: 音频文件不存在: {audio_filepath}")
            continue
        
        # 简单的结果记录（暂时不进行实际推理）
        result = {
            'audio_filepath': audio_filepath,
            'ground_truth': text,
            'predicted_text': '[推理功能待实现]',  # 占位符
            'sample_index': i
        }
        
        results.append(result)
    
    # 保存结果
    print(f"\n保存结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"推理完成！处理了 {len(results)} 个样本")
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='简化的多说话人语音识别推理')
    parser.add_argument('--model_path', type=str, required=True, help='训练后的模型路径')
    parser.add_argument('--manifest_path', type=str, required=True, help='manifest文件路径')
    parser.add_argument('--output_path', type=str, default='./simple_inference_results.jsonl', help='输出文件路径')
    parser.add_argument('--max_samples', type=int, default=10, help='最大处理样本数')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    # 运行推理
    results = simple_inference(
        model_path=args.model_path,
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
        device=args.device
    )
    
    print(f"\n=== 推理结果摘要 ===")
    print(f"处理样本数: {len(results)}")
    print(f"结果文件: {args.output_path}")

if __name__ == '__main__':
    main()