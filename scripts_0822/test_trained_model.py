#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练后的Adapter微调模型
"""

import torch
import numpy as np
import argparse
import os

def test_model():
    """测试训练后的模型"""
    
    # 模型路径
    model_path = "./adapter_finetune_output/final_model.pt"
    
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        return
    
    print(f"加载模型: {model_path}")
    
    try:
        # 加载模型
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"模型加载成功！")
        print(f"模型包含的键: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"模型状态字典包含 {len(model_state)} 个参数")
            
            # 显示一些关键参数的形状
            key_params = [
                'encoder.layers.0.self_attn.linear_q.weight',
                'decoder.prediction.joint.joint_net.0.weight',
                'speaker_adapter.speaker_embedding.weight'
            ]
            
            for key in key_params:
                if key in model_state:
                    print(f"  {key}: {model_state[key].shape}")
                else:
                    print(f"  {key}: 未找到")
        
        if 'training_info' in checkpoint:
            info = checkpoint['training_info']
            print(f"\n训练信息:")
            print(f"  最终训练损失: {info.get('final_train_loss', 'N/A')}")
            print(f"  最终验证损失: {info.get('final_val_loss', 'N/A')}")
            print(f"  训练轮数: {info.get('epochs', 'N/A')}")
        
        print("\n模型测试完成！")
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()