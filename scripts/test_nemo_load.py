#!/usr/bin/env python3

import os
import gc
import psutil
import torch
from nemo.collections.asr.models import ASRModel

def print_memory_usage(stage):
    """打印当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"[{stage}] 内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"[{stage}] 虚拟内存: {memory_info.vms / 1024 / 1024:.2f} MB")

def test_nemo_load():
    print("开始测试 NeMo 模型加载...")
    
    # 设置环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    print_memory_usage("初始状态")
    
    # 设置 PyTorch
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    
    print_memory_usage("PyTorch设置后")
    
    nemo_path = "./pretrained_models/stt_zh_conformer_transducer_large.nemo"
    device = "cpu"
    
    print(f"正在加载模型: {nemo_path}")
    print(f"使用设备: {device}")
    
    try:
        print_memory_usage("模型加载前")
        
        # 尝试加载模型
        model = ASRModel.restore_from(restore_path=nemo_path, map_location=device)
        
        print_memory_usage("模型加载后")
        
        model = model.to(device)
        model.eval()
        
        print_memory_usage("模型设置后")
        
        print("模型加载成功！")
        print(f"模型类型: {type(model)}")
        
        # 清理
        del model
        gc.collect()
        
        print_memory_usage("清理后")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        print_memory_usage("错误时")
        raise

if __name__ == "__main__":
    test_nemo_load()