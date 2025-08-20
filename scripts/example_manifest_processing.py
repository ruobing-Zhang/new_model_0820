#!/usr/bin/env python3
"""
演示如何使用修改后的auto_align_speaker_labels.py脚本
通过实际推理处理包含RTTM路径的manifest文件

该示例展示了新的核心功能：
1. 从manifest文件中读取音频和RTTM文件路径
2. 使用预训练ASR模型对音频进行实际推理
3. 获取真实的编码器时间维度（而非计算得出）
4. 将RTTM转换为与编码器输出精确对齐的说话人标签矩阵
"""

import os
import json
import numpy as np
from pathlib import Path

# 导入我们的自动对齐工具
from auto_align_speaker_labels import RTTMToNPYConverter, BatchProcessor

def create_example_manifest_with_rttm():
    """
    创建一个包含音频和RTTM路径的示例manifest文件
    """
    print("\n" + "="*60)
    print("创建示例manifest文件（包含RTTM路径）")
    print("="*60)
    
    # 创建示例目录结构
    base_dir = Path("/tmp/example_dataset")
    audio_dir = base_dir / "audio"
    rttm_dir = base_dir / "rttm"
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例音频文件（空文件，仅用于演示）
    audio_files = [
        "conversation_001.wav",
        "conversation_002.wav", 
        "conversation_003.wav"
    ]
    
    for audio_file in audio_files:
        audio_path = audio_dir / audio_file
        # 创建一个小的示例音频文件（1秒的静音）
        import torchaudio
        # 创建1秒16kHz的静音音频
        waveform = torch.zeros(1, 16000)  # 1通道，16000采样点
        torchaudio.save(str(audio_path), waveform, 16000)
    
    # 创建对应的RTTM文件
    rttm_contents = [
        # conversation_001.rttm - 两个说话人
        "SPEAKER conversation_001 1 0.0 2.5 <NA> <NA> speaker_0 <NA> <NA>\n" +
        "SPEAKER conversation_001 1 2.5 3.0 <NA> <NA> speaker_1 <NA> <NA>\n" +
        "SPEAKER conversation_001 1 5.5 2.0 <NA> <NA> speaker_0 <NA> <NA>\n",
        
        # conversation_002.rttm - 三个说话人
        "SPEAKER conversation_002 1 0.0 1.8 <NA> <NA> speaker_0 <NA> <NA>\n" +
        "SPEAKER conversation_002 1 1.8 2.2 <NA> <NA> speaker_1 <NA> <NA>\n" +
        "SPEAKER conversation_002 1 4.0 1.5 <NA> <NA> speaker_2 <NA> <NA>\n" +
        "SPEAKER conversation_002 1 5.5 1.0 <NA> <NA> speaker_0 <NA> <NA>\n",
        
        # conversation_003.rttm - 单个说话人
        "SPEAKER conversation_003 1 0.0 4.0 <NA> <NA> speaker_0 <NA> <NA>\n" +
        "SPEAKER conversation_003 1 4.5 2.5 <NA> <NA> speaker_0 <NA> <NA>\n"
    ]
    
    for i, (audio_file, rttm_content) in enumerate(zip(audio_files, rttm_contents)):
        rttm_file = audio_file.replace('.wav', '.rttm')
        rttm_path = rttm_dir / rttm_file
        with open(rttm_path, 'w') as f:
            f.write(rttm_content)
    
    # 创建包含RTTM路径的manifest文件
    manifest_data = []
    for audio_file in audio_files:
        audio_path = str(audio_dir / audio_file)
        rttm_file = audio_file.replace('.wav', '.rttm')
        rttm_path = str(rttm_dir / rttm_file)
        
        manifest_entry = {
            "audio_filepath": audio_path,
            "rttm_filepath": rttm_path,  # 新增：RTTM文件路径
            "duration": 8.0,  # 示例时长
            "text": f"Example conversation {audio_file}"
        }
        manifest_data.append(manifest_entry)
    
    # 保存manifest文件
    manifest_path = base_dir / "train_manifest_with_rttm.json"
    with open(manifest_path, 'w') as f:
        for entry in manifest_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ 创建了示例数据集:")
    print(f"   音频目录: {audio_dir}")
    print(f"   RTTM目录: {rttm_dir}")
    print(f"   Manifest文件: {manifest_path}")
    print(f"   包含 {len(manifest_data)} 个条目")
    
    return str(manifest_path), str(base_dir)

def demonstrate_single_file_inference():
    """
    演示单文件推理模式：通过实际推理获取编码器时间维度
    """
    print("\n" + "="*60)
    print("演示单文件推理模式")
    print("="*60)
    
    # 注意：这里需要一个真实的ASR模型路径
    # 在实际使用中，请替换为您的ASR模型路径
    asr_model_path = "/path/to/your/asr_model.nemo"  # 请替换为实际路径
    
    print("\n⚠️  注意：此演示需要真实的ASR模型")
    print(f"请将 asr_model_path 设置为您的ASR模型路径")
    print(f"当前设置: {asr_model_path}")
    
    if not os.path.exists(asr_model_path):
        print("\n❌ ASR模型文件不存在，跳过实际推理演示")
        print("\n💡 如果您有ASR模型，请按以下步骤操作：")
        print("1. 将 asr_model_path 设置为您的模型路径")
        print("2. 重新运行此脚本")
        return
    
    # 如果模型存在，执行实际推理
    try:
        converter = RTTMToNPYConverter(asr_model_path)
        
        # 使用示例文件
        audio_path = "/tmp/example_dataset/audio/conversation_001.wav"
        rttm_path = "/tmp/example_dataset/rttm/conversation_001.rttm"
        output_path = "/tmp/example_dataset/conversation_001_aligned.npy"
        speaker_ids = ["speaker_0", "speaker_1"]
        
        print(f"\n🔄 正在处理: {Path(audio_path).name}")
        
        # 执行转换（通过实际推理）
        conversion_info = converter.convert_rttm_to_npy(
            rttm_path, audio_path, output_path, speaker_ids
        )
        
        print("\n✅ 转换完成！")
        print(f"📊 转换信息:")
        for key, value in conversion_info.items():
            print(f"   {key}: {value}")
        
        # 加载并检查生成的矩阵
        speaker_matrix = np.load(output_path)
        print(f"\n📈 生成的说话人矩阵:")
        print(f"   形状: {speaker_matrix.shape}")
        print(f"   数据类型: {speaker_matrix.dtype}")
        print(f"   值范围: [{speaker_matrix.min():.3f}, {speaker_matrix.max():.3f}]")
        
    except Exception as e:
        print(f"\n❌ 推理过程中出错: {e}")
        print("请检查ASR模型路径和依赖项")

def demonstrate_batch_processing_from_manifest():
    """
    演示从manifest文件批量处理：新的推荐方法
    """
    print("\n" + "="*60)
    print("演示从manifest文件批量处理（推荐方法）")
    print("="*60)
    
    # 创建示例数据
    manifest_path, base_dir = create_example_manifest_with_rttm()
    
    # 设置输出路径
    output_npy_dir = os.path.join(base_dir, "aligned_speaker_labels")
    output_manifest_path = os.path.join(base_dir, "train_manifest_with_aligned_labels.json")
    
    # 说话人ID列表
    speaker_ids = ["speaker_0", "speaker_1", "speaker_2"]
    
    print(f"\n📋 处理参数:")
    print(f"   输入manifest: {manifest_path}")
    print(f"   输出NPY目录: {output_npy_dir}")
    print(f"   输出manifest: {output_manifest_path}")
    print(f"   说话人IDs: {speaker_ids}")
    
    # 注意：这里需要真实的ASR模型
    asr_model_path = "/path/to/your/asr_model.nemo"  # 请替换为实际路径
    
    print(f"\n⚠️  注意：需要真实的ASR模型进行推理")
    print(f"当前ASR模型路径: {asr_model_path}")
    
    if not os.path.exists(asr_model_path):
        print("\n❌ ASR模型不存在，显示处理流程（不执行实际推理）")
        print("\n💡 实际使用时的命令行调用方式：")
        print(f"python auto_align_speaker_labels.py batch_process_manifest \\")
        print(f"  --asr_model_path {asr_model_path} \\")
        print(f"  --manifest_path {manifest_path} \\")
        print(f"  --output_npy_dir {output_npy_dir} \\")
        print(f"  --output_manifest_path {output_manifest_path} \\")
        print(f"  --speaker_ids {' '.join(speaker_ids)} \\")
        print(f"  --rttm_field rttm_filepath")
        return
    
    # 如果模型存在，执行实际批量处理
    try:
        processor = BatchProcessor(asr_model_path)
        
        print("\n🔄 开始批量处理...")
        stats = processor.process_dataset_from_manifest(
            manifest_path=manifest_path,
            output_npy_dir=output_npy_dir,
            output_manifest_path=output_manifest_path,
            speaker_ids=speaker_ids,
            rttm_field="rttm_filepath"  # manifest中RTTM路径的字段名
        )
        
        print("\n✅ 批量处理完成！")
        print(f"📊 处理统计: {stats}")
        
    except Exception as e:
        print(f"\n❌ 批量处理过程中出错: {e}")

def explain_key_improvements():
    """
    解释关键改进点
    """
    print("\n" + "="*60)
    print("关键改进说明")
    print("="*60)
    
    improvements = [
        {
            "标题": "🎯 实际推理获取时间维度",
            "说明": "不再通过计算估算编码器时间维度，而是通过实际推理获取真实的时间维度，确保完美对齐"
        },
        {
            "标题": "📁 支持manifest中的RTTM路径",
            "说明": "可以直接从manifest文件中读取RTTM文件路径，无需单独指定RTTM目录"
        },
        {
            "标题": "🔄 智能路径推断",
            "说明": "如果manifest中没有RTTM路径字段，会自动从音频路径推断RTTM文件位置"
        },
        {
            "标题": "📊 详细处理统计",
            "说明": "提供完整的处理统计信息，包括编码器时间维度分布、音频时长统计等"
        },
        {
            "标题": "🔧 向后兼容",
            "说明": "保留原有的处理方法，确保现有代码仍然可以正常工作"
        }
    ]
    
    for i, improvement in enumerate(improvements, 1):
        print(f"\n{i}. {improvement['标题']}")
        print(f"   {improvement['说明']}")
    
    print("\n" + "="*60)
    print("技术细节")
    print("="*60)
    
    technical_details = [
        "🧠 ASR编码器推理流程：音频 → 预处理 → 编码器 → 获取时间维度",
        "⏱️  时间对齐公式：T_enc = 编码器实际输出的时间维度",
        "🎭 说话人矩阵：形状为 (T_enc, num_speakers)，每个时间步对应说话人活动",
        "💾 输出格式：NPY文件，包含float32类型的说话人标签矩阵",
        "📋 Manifest更新：自动添加speaker_labels_path字段指向NPY文件"
    ]
    
    for detail in technical_details:
        print(f"   {detail}")

def main():
    """
    主演示函数
    """
    print("\n" + "="*80)
    print("自动说话人标签对齐工具 - 实际推理版本演示")
    print("="*80)
    
    print("\n本演示展示了如何使用修改后的auto_align_speaker_labels.py脚本")
    print("通过实际ASR推理获取编码器时间维度，实现精确的说话人标签对齐")
    
    # 1. 解释关键改进
    explain_key_improvements()
    
    # 2. 创建示例数据并演示manifest处理
    demonstrate_batch_processing_from_manifest()
    
    # 3. 演示单文件推理（如果有模型的话）
    demonstrate_single_file_inference()
    
    print("\n" + "="*80)
    print("演示完成")
    print("="*80)
    
    print("\n💡 下一步操作建议：")
    print("1. 准备您的ASR模型文件（.nemo格式）")
    print("2. 准备包含audio_filepath和rttm_filepath的manifest文件")
    print("3. 使用新的batch_process_manifest模式进行批量处理")
    print("4. 检查生成的对齐说话人标签矩阵")
    
    print("\n🔗 相关文件：")
    print("   - auto_align_speaker_labels.py: 主要处理脚本")
    print("   - run_auto_align.sh: 便捷的shell脚本")
    print("   - README_auto_align.md: 详细文档")

if __name__ == "__main__":
    # 添加必要的导入
    import torch
    import torchaudio
    
    main()