#!/usr/bin/env python3
"""
自动对齐说话人标签工具的使用示例

该脚本展示了如何使用auto_align_speaker_labels.py工具来处理说话人标签与ASR编码器时间维度的对齐问题。
包含了完整的使用流程和示例。

作者: AI Assistant
日期: 2024
"""

import os
import sys
import subprocess
from pathlib import Path

# 配置路径
BASE_DIR = "/root/autodl-tmp/joint_sortformer_and_asr_0815"
ASR_MODEL_PATH = f"{BASE_DIR}/pretrained_models/zh_conformer_transducer_large_bpe_init.nemo"
SCRIPT_PATH = f"{BASE_DIR}/scripts/auto_align_speaker_labels.py"

# 示例数据路径
EXAMPLE_MANIFEST = f"{BASE_DIR}/data/example_manifest.json"
EXAMPLE_RTTM_DIR = f"{BASE_DIR}/data/rttm_files"
OUTPUT_NPY_DIR = f"{BASE_DIR}/scripts/output_npy_aligned"
OUTPUT_MANIFEST = f"{BASE_DIR}/data/example_manifest_with_npy.json"

# 说话人ID列表
SPEAKER_IDS = ["spk1", "spk2", "spk3"]


def run_command(cmd, description):
    """
    运行命令并打印结果
    
    Args:
        cmd: 要运行的命令列表
        description: 命令描述
    """
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("输出:")
        print(result.stdout)
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    return True


def create_example_data():
    """
    创建示例数据文件
    """
    print("创建示例数据文件...")
    
    # 创建目录
    os.makedirs(os.path.dirname(EXAMPLE_MANIFEST), exist_ok=True)
    os.makedirs(EXAMPLE_RTTM_DIR, exist_ok=True)
    os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)
    
    # 创建示例manifest文件
    manifest_content = '''{
    "audio_filepath": "/path/to/audio1.wav",
    "text": "这是第一段音频的转录文本",
    "duration": 10.5
}
{
    "audio_filepath": "/path/to/audio2.wav",
    "text": "这是第二段音频的转录文本",
    "duration": 8.2
}
{
    "audio_filepath": "/path/to/audio3.wav",
    "text": "这是第三段音频的转录文本",
    "duration": 12.1
}'''
    
    with open(EXAMPLE_MANIFEST, 'w') as f:
        f.write(manifest_content)
    
    # 创建示例RTTM文件
    rttm_files = {
        "audio1.rttm": '''SPEAKER audio1 1 0.0 3.5 <NA> <NA> spk1 <NA> <NA>
SPEAKER audio1 1 3.5 4.0 <NA> <NA> spk2 <NA> <NA>
SPEAKER audio1 1 7.5 3.0 <NA> <NA> spk1 <NA> <NA>''',
        
        "audio2.rttm": '''SPEAKER audio2 1 0.0 2.1 <NA> <NA> spk1 <NA> <NA>
SPEAKER audio2 1 2.1 3.1 <NA> <NA> spk3 <NA> <NA>
SPEAKER audio2 1 5.2 3.0 <NA> <NA> spk2 <NA> <NA>''',
        
        "audio3.rttm": '''SPEAKER audio3 1 0.0 4.0 <NA> <NA> spk2 <NA> <NA>
SPEAKER audio3 1 4.0 4.1 <NA> <NA> spk1 <NA> <NA>
SPEAKER audio3 1 8.1 4.0 <NA> <NA> spk3 <NA> <NA>'''
    }
    
    for filename, content in rttm_files.items():
        with open(os.path.join(EXAMPLE_RTTM_DIR, filename), 'w') as f:
            f.write(content)
    
    print(f"示例数据已创建:")
    print(f"  - Manifest文件: {EXAMPLE_MANIFEST}")
    print(f"  - RTTM文件目录: {EXAMPLE_RTTM_DIR}")
    print(f"  - 输出NPY目录: {OUTPUT_NPY_DIR}")


def example_single_file_conversion():
    """
    示例1: 单文件转换
    """
    print("\n" + "="*80)
    print("示例1: 单文件RTTM到NPY转换")
    print("="*80)
    
    # 使用第一个音频文件作为示例
    rttm_path = os.path.join(EXAMPLE_RTTM_DIR, "audio1.rttm")
    audio_path = "/path/to/audio1.wav"  # 注意：这是示例路径，实际使用时需要真实音频文件
    output_npy_path = os.path.join(OUTPUT_NPY_DIR, "audio1.npy")
    
    cmd = [
        "python", SCRIPT_PATH,
        "--mode", "convert",
        "--asr_model_path", ASR_MODEL_PATH,
        "--rttm_path", rttm_path,
        "--audio_path", audio_path,
        "--output_npy_path", output_npy_path,
        "--speaker_ids"] + SPEAKER_IDS
    
    print("注意: 由于示例中使用的是虚拟音频路径，此命令可能会失败。")
    print("在实际使用时，请提供真实的音频文件路径。")
    print(f"命令: {' '.join(cmd)}")
    
    # 不实际运行，因为没有真实音频文件
    # run_command(cmd, "单文件RTTM到NPY转换")


def example_batch_processing():
    """
    示例2: 批量处理（模拟）
    """
    print("\n" + "="*80)
    print("示例2: 批量处理整个数据集")
    print("="*80)
    
    cmd = [
        "python", SCRIPT_PATH,
        "--mode", "batch_process",
        "--asr_model_path", ASR_MODEL_PATH,
        "--manifest_path", EXAMPLE_MANIFEST,
        "--rttm_dir", EXAMPLE_RTTM_DIR,
        "--output_npy_dir", OUTPUT_NPY_DIR,
        "--output_manifest_path", OUTPUT_MANIFEST,
        "--speaker_ids"] + SPEAKER_IDS
    
    print("注意: 由于示例中使用的是虚拟音频路径，此命令可能会失败。")
    print("在实际使用时，请确保manifest文件中的音频路径是真实存在的。")
    print(f"命令: {' '.join(cmd)}")
    
    # 不实际运行，因为没有真实音频文件
    # run_command(cmd, "批量处理整个数据集")


def example_update_manifest():
    """
    示例3: 更新manifest文件
    """
    print("\n" + "="*80)
    print("示例3: 更新manifest文件添加NPY路径")
    print("="*80)
    
    cmd = [
        "python", SCRIPT_PATH,
        "--mode", "update_manifest",
        "--asr_model_path", ASR_MODEL_PATH,
        "--manifest_path", EXAMPLE_MANIFEST,
        "--output_manifest_path", OUTPUT_MANIFEST,
        "--output_npy_dir", OUTPUT_NPY_DIR
    ]
    
    run_command(cmd, "更新manifest文件添加NPY路径")


def example_generate_dataloader():
    """
    示例4: 生成数据加载器代码
    """
    print("\n" + "="*80)
    print("示例4: 生成修改后的数据加载器代码")
    print("="*80)
    
    dataloader_output = f"{BASE_DIR}/scripts/modified_dataloader.py"
    
    cmd = [
        "python", SCRIPT_PATH,
        "--mode", "generate_dataloader",
        "--asr_model_path", ASR_MODEL_PATH,
        "--dataloader_output_path", dataloader_output
    ]
    
    run_command(cmd, "生成修改后的数据加载器代码")


def show_usage_instructions():
    """
    显示详细的使用说明
    """
    print("\n" + "="*80)
    print("详细使用说明")
    print("="*80)
    
    instructions = """
1. 准备数据:
   - 确保你有音频文件和对应的RTTM文件
   - 创建manifest文件，包含音频路径、文本和时长信息
   - 准备说话人ID列表

2. 单文件转换:
   python auto_align_speaker_labels.py \
     --mode convert \
     --asr_model_path /path/to/your/asr_model.nemo \
     --rttm_path /path/to/audio.rttm \
     --audio_path /path/to/audio.wav \
     --output_npy_path /path/to/output.npy \
     --speaker_ids spk1 spk2 spk3

3. 批量处理:
   python auto_align_speaker_labels.py \
     --mode batch_process \
     --asr_model_path /path/to/your/asr_model.nemo \
     --manifest_path /path/to/manifest.json \
     --rttm_dir /path/to/rttm_files/ \
     --output_npy_dir /path/to/output_npy/ \
     --output_manifest_path /path/to/updated_manifest.json \
     --speaker_ids spk1 spk2 spk3

4. 更新manifest文件:
   python auto_align_speaker_labels.py \
     --mode update_manifest \
     --asr_model_path /path/to/your/asr_model.nemo \
     --manifest_path /path/to/original_manifest.json \
     --output_manifest_path /path/to/updated_manifest.json \
     --output_npy_dir /path/to/npy_files/

5. 生成数据加载器代码:
   python auto_align_speaker_labels.py \
     --mode generate_dataloader \
     --asr_model_path /path/to/your/asr_model.nemo \
     --dataloader_output_path modified_dataloader.py

6. 在训练中使用:
   - 使用生成的数据加载器代码
   - 在训练循环中获取speaker_labels
   - 将speaker_labels传递给带有说话人注入的模型

重要说明:
- 确保ASR模型路径正确
- 音频文件路径必须存在且可访问
- RTTM文件格式必须正确
- 说话人ID列表应该包含所有可能的说话人
- 生成的NPY文件包含形状为(T_enc, num_speakers)的标签矩阵
- T_enc是ASR编码器输出的时间维度，会自动计算
"""
    
    print(instructions)


def show_tensor_dimension_explanation():
    """
    解释张量维度的含义
    """
    print("\n" + "="*80)
    print("张量维度说明")
    print("="*80)
    
    explanation = """
在测试中看到的 torch.Size([2, 76, 512]) 各维度含义:

1. 第一维 (2): 批次大小 (Batch Size)
   - 表示同时处理的音频样本数量
   - 在训练时可以调整以优化GPU利用率

2. 第二维 (76): 时间维度 (Time Dimension, T_enc)
   - 这是ASR编码器输出的时间步数
   - 由音频长度和编码器的下采样率决定
   - 例如: 10秒音频 → 编码器下采样 → 76个时间步
   - 这个维度会根据音频长度变化

3. 第三维 (512): 特征维度 (Feature Dimension)
   - ASR编码器输出的特征向量维度
   - 由模型架构决定，通常是固定的
   - 在Conformer模型中通常是512或1024

说话人标签矩阵的形状:
- 输入: (T_enc, num_speakers) = (76, 3)
- 批次处理后: (B, T_enc, num_speakers) = (2, 76, 3)

时间维度对齐的重要性:
- ASR编码器将原始音频下采样到更少的时间步
- 说话人标签必须与编码器输出的时间步对齐
- 自动对齐工具会计算正确的T_enc并生成对应的标签矩阵

计算公式:
T_enc ≈ 音频时长(秒) × 采样率 / 编码器下采样因子

例如:
- 音频时长: 10秒
- 采样率: 16000 Hz
- 编码器下采样因子: ~2105 (具体值取决于模型)
- T_enc ≈ 10 × 16000 / 2105 ≈ 76
"""
    
    print(explanation)


def main():
    """
    主函数
    """
    print("自动对齐说话人标签工具 - 使用示例")
    print("="*80)
    
    # 检查脚本是否存在
    if not os.path.exists(SCRIPT_PATH):
        print(f"错误: 找不到脚本文件 {SCRIPT_PATH}")
        return
    
    # 检查ASR模型是否存在
    if not os.path.exists(ASR_MODEL_PATH):
        print(f"警告: ASR模型文件不存在 {ASR_MODEL_PATH}")
        print("请确保模型路径正确")
    
    # 创建示例数据
    create_example_data()
    
    # 运行各种示例
    example_single_file_conversion()
    example_batch_processing()
    example_update_manifest()
    example_generate_dataloader()
    
    # 显示使用说明
    show_usage_instructions()
    show_tensor_dimension_explanation()
    
    print("\n" + "="*80)
    print("示例运行完成!")
    print("="*80)
    print(f"生成的文件:")
    print(f"  - 示例manifest: {EXAMPLE_MANIFEST}")
    print(f"  - RTTM文件目录: {EXAMPLE_RTTM_DIR}")
    print(f"  - 输出NPY目录: {OUTPUT_NPY_DIR}")
    print(f"  - 更新后的manifest: {OUTPUT_MANIFEST}")
    print(f"  - 数据加载器代码: {BASE_DIR}/scripts/modified_dataloader.py")
    print("\n请根据你的实际数据路径修改配置并运行相应的命令。")


if __name__ == "__main__":
    main()