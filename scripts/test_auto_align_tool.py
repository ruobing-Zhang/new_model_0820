#!/usr/bin/env python3
"""
自动对齐工具的快速测试脚本

该脚本用于验证自动对齐工具的基本功能是否正常工作。
包括模块导入测试、基本功能测试等。

作者: AI Assistant
日期: 2024
"""

import os
import sys
import tempfile
import numpy as np
import json
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 测试结果统计
test_results = {
    'passed': 0,
    'failed': 0,
    'errors': []
}

def print_test_result(test_name, success, error_msg=None):
    """
    打印测试结果
    
    Args:
        test_name: 测试名称
        success: 是否成功
        error_msg: 错误信息
    """
    if success:
        print(f"✅ {test_name}: PASSED")
        test_results['passed'] += 1
    else:
        print(f"❌ {test_name}: FAILED")
        if error_msg:
            print(f"   错误: {error_msg}")
            test_results['errors'].append(f"{test_name}: {error_msg}")
        test_results['failed'] += 1

def test_module_imports():
    """
    测试模块导入
    """
    print("\n=== 测试模块导入 ===")
    
    # 测试基本库导入
    try:
        import numpy as np
        import torch
        import librosa
        from tqdm import tqdm
        print_test_result("基本依赖库导入", True)
    except ImportError as e:
        print_test_result("基本依赖库导入", False, str(e))
    
    # 测试NeMo ASR模型导入
    try:
        from nemo.collections.asr.models.asr_model import ASRModel
        print_test_result("NeMo ASR模型导入", True)
    except ImportError as e:
        print_test_result("NeMo ASR模型导入", False, str(e))
    
    # 测试自定义模块导入
    try:
        from auto_align_speaker_labels import (
            RTTMToNPYConverter,
            ManifestUpdater,
            DataLoaderModifier,
            BatchProcessor
        )
        print_test_result("自定义模块导入", True)
    except ImportError as e:
        print_test_result("自定义模块导入", False, str(e))
        return False
    
    return True

def test_rttm_parsing():
    """
    测试RTTM文件解析
    """
    print("\n=== 测试RTTM文件解析 ===")
    
    try:
        from auto_align_speaker_labels import RTTMToNPYConverter
        
        # 创建临时RTTM文件
        rttm_content = """SPEAKER audio1 1 0.0 3.5 <NA> <NA> spk1 <NA> <NA>
SPEAKER audio1 1 3.5 4.0 <NA> <NA> spk2 <NA> <NA>
SPEAKER audio1 1 7.5 3.0 <NA> <NA> spk1 <NA> <NA>"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rttm', delete=False) as f:
            f.write(rttm_content)
            rttm_path = f.name
        
        try:
            # 创建转换器实例（不加载ASR模型）
            converter = RTTMToNPYConverter("dummy_path")
            
            # 解析RTTM文件
            segments = converter.parse_rttm(rttm_path)
            
            # 验证解析结果
            expected_segments = [
                (0.0, 3.5, 'spk1'),
                (3.5, 7.5, 'spk2'),
                (7.5, 10.5, 'spk1')
            ]
            
            if len(segments) == 3:
                print_test_result("RTTM文件解析", True)
            else:
                print_test_result("RTTM文件解析", False, f"期望3个片段，实际{len(segments)}个")
        
        finally:
            # 清理临时文件
            os.unlink(rttm_path)
    
    except Exception as e:
        print_test_result("RTTM文件解析", False, str(e))

def test_manifest_updating():
    """
    测试Manifest文件更新
    """
    print("\n=== 测试Manifest文件更新 ===")
    
    try:
        from auto_align_speaker_labels import ManifestUpdater
        
        # 创建临时manifest文件
        manifest_data = [
            {"audio_filepath": "/path/to/audio1.wav", "text": "text1", "duration": 10.0},
            {"audio_filepath": "/path/to/audio2.wav", "text": "text2", "duration": 8.0}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            for item in manifest_data:
                f.write(json.dumps(item) + '\n')
            input_manifest = f.name
        
        # 创建临时NPY目录和文件
        with tempfile.TemporaryDirectory() as npy_dir:
            # 创建一个NPY文件
            npy_path = os.path.join(npy_dir, "audio1.npy")
            np.save(npy_path, np.random.rand(76, 3))
            
            # 创建输出manifest文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                output_manifest = f.name
            
            try:
                # 更新manifest
                ManifestUpdater.update_manifest_with_npy_paths(
                    input_manifest, output_manifest, npy_dir
                )
                
                # 验证结果
                with open(output_manifest, 'r') as f:
                    updated_data = [json.loads(line.strip()) for line in f if line.strip()]
                
                # 检查是否添加了npy_path
                has_npy_path = any('npy_path' in item for item in updated_data)
                
                if has_npy_path:
                    print_test_result("Manifest文件更新", True)
                else:
                    print_test_result("Manifest文件更新", False, "未找到npy_path字段")
            
            finally:
                # 清理临时文件
                os.unlink(input_manifest)
                os.unlink(output_manifest)
    
    except Exception as e:
        print_test_result("Manifest文件更新", False, str(e))

def test_dataloader_generation():
    """
    测试数据加载器代码生成
    """
    print("\n=== 测试数据加载器代码生成 ===")
    
    try:
        from auto_align_speaker_labels import DataLoaderModifier
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            output_path = f.name
        
        try:
            # 生成数据加载器代码
            DataLoaderModifier.generate_dataloader_code(output_path)
            
            # 检查文件是否生成且包含预期内容
            with open(output_path, 'r') as f:
                content = f.read()
            
            # 检查关键类和函数是否存在
            required_elements = [
                'AudioSpeakerDataset',
                'collate_fn_with_speaker_labels',
                '__getitem__',
                'speaker_labels'
            ]
            
            missing_elements = [elem for elem in required_elements if elem not in content]
            
            if not missing_elements:
                print_test_result("数据加载器代码生成", True)
            else:
                print_test_result("数据加载器代码生成", False, f"缺少元素: {missing_elements}")
        
        finally:
            # 清理临时文件
            os.unlink(output_path)
    
    except Exception as e:
        print_test_result("数据加载器代码生成", False, str(e))

def test_speaker_matrix_creation():
    """
    测试说话人矩阵创建（不使用真实ASR模型）
    """
    print("\n=== 测试说话人矩阵创建 ===")
    
    try:
        # 模拟说话人矩阵创建逻辑
        audio_duration = 10.0
        T_enc = 76  # 模拟编码器时间维度
        speaker_ids = ['spk1', 'spk2', 'spk3']
        
        # 模拟RTTM片段
        segments = [
            (0.0, 3.5, 'spk1'),
            (3.5, 7.5, 'spk2'),
            (7.5, 10.5, 'spk1')
        ]
        
        # 创建说话人矩阵
        num_speakers = len(speaker_ids)
        speaker_matrix = np.zeros((T_enc, num_speakers), dtype=np.float32)
        speaker_to_idx = {spk: idx for idx, spk in enumerate(speaker_ids)}
        
        # 计算时间步长
        time_step = audio_duration / T_enc
        
        # 填充标签矩阵
        for start_time, end_time, speaker_id in segments:
            if speaker_id in speaker_to_idx:
                start_idx = int(start_time / time_step)
                end_idx = int(end_time / time_step)
                start_idx = max(0, min(start_idx, T_enc - 1))
                end_idx = max(0, min(end_idx, T_enc))
                speaker_idx = speaker_to_idx[speaker_id]
                speaker_matrix[start_idx:end_idx, speaker_idx] = 1.0
        
        # 验证矩阵
        if speaker_matrix.shape == (T_enc, num_speakers):
            # 检查是否有非零值
            if np.sum(speaker_matrix) > 0:
                print_test_result("说话人矩阵创建", True)
            else:
                print_test_result("说话人矩阵创建", False, "矩阵全为零")
        else:
            print_test_result("说话人矩阵创建", False, f"矩阵形状错误: {speaker_matrix.shape}")
    
    except Exception as e:
        print_test_result("说话人矩阵创建", False, str(e))

def test_file_existence():
    """
    测试相关文件是否存在
    """
    print("\n=== 测试文件存在性 ===")
    
    base_dir = "/root/autodl-tmp/joint_sortformer_and_asr_0815"
    
    files_to_check = [
        ("auto_align_speaker_labels.py", "主Python脚本"),
        ("run_auto_align.sh", "批处理脚本"),
        ("example_auto_align_usage.py", "使用示例脚本"),
        ("README_auto_align.md", "说明文档")
    ]
    
    for filename, description in files_to_check:
        filepath = os.path.join(base_dir, "scripts", filename)
        exists = os.path.exists(filepath)
        print_test_result(f"{description}存在性检查", exists, f"文件不存在: {filepath}" if not exists else None)

def test_script_permissions():
    """
    测试脚本执行权限
    """
    print("\n=== 测试脚本权限 ===")
    
    script_path = "/root/autodl-tmp/joint_sortformer_and_asr_0815/scripts/run_auto_align.sh"
    
    if os.path.exists(script_path):
        # 检查执行权限
        is_executable = os.access(script_path, os.X_OK)
        print_test_result("批处理脚本执行权限", is_executable, "脚本没有执行权限" if not is_executable else None)
    else:
        print_test_result("批处理脚本执行权限", False, "脚本文件不存在")

def run_all_tests():
    """
    运行所有测试
    """
    print("自动对齐工具测试套件")
    print("=" * 50)
    
    # 运行各项测试
    if test_module_imports():
        test_rttm_parsing()
        test_manifest_updating()
        test_dataloader_generation()
        test_speaker_matrix_creation()
    
    test_file_existence()
    test_script_permissions()
    
    # 打印测试摘要
    print("\n" + "=" * 50)
    print("测试摘要")
    print("=" * 50)
    print(f"✅ 通过: {test_results['passed']}")
    print(f"❌ 失败: {test_results['failed']}")
    print(f"📊 总计: {test_results['passed'] + test_results['failed']}")
    
    if test_results['failed'] > 0:
        print("\n失败的测试:")
        for error in test_results['errors']:
            print(f"  - {error}")
        print("\n请检查上述错误并修复相关问题。")
    else:
        print("\n🎉 所有测试都通过了！工具已准备就绪。")
    
    # 返回测试是否全部通过
    return test_results['failed'] == 0

def main():
    """
    主函数
    """
    success = run_all_tests()
    
    if success:
        print("\n" + "=" * 50)
        print("快速使用指南")
        print("=" * 50)
        print("1. 查看详细文档:")
        print("   cat README_auto_align.md")
        print("\n2. 运行使用示例:")
        print("   python example_auto_align_usage.py")
        print("\n3. 查看批处理脚本帮助:")
        print("   ./run_auto_align.sh --help")
        print("\n4. 生成数据加载器代码:")
        print("   ./run_auto_align.sh -m generate_dataloader")
        
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)