#!/usr/bin/env python3
"""
自动对齐说话人标签与ASR编码器时间维度的完整解决方案

该脚本通过实际推理获取ASR编码器的时间维度，然后将RTTM文件转换为对齐的说话人标签矩阵。
主要功能包括：
1. 从训练集manifest文件中读取音频和RTTM文件路径
2. 使用预训练ASR模型对音频进行推理，获取真实的编码器时间维度
3. 将RTTM文件转换为与编码器输出时间维度精确对齐的说话人标签矩阵
4. 更新manifest文件以包含NPY文件路径
5. 批量处理整个数据集

作者: AI Assistant
日期: 2024
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
import torch
from tqdm import tqdm

# 添加NeMo路径
sys.path.append('/root/autodl-tmp/joint_sortformer_and_asr_0815/NeMo-main')
from nemo.collections.asr.models.asr_model import ASRModel


class RTTMToNPYConverter:
    """
    RTTM文件到NPY文件的转换器，通过实际推理获取ASR编码器的时间维度
    
    核心改进：
    1. 从manifest文件中读取音频和RTTM文件路径
    2. 使用预训练ASR模型对真实音频进行推理
    3. 获取编码器输出的实际时间维度
    4. 将RTTM标注精确对齐到编码器时间维度
    """
    
    def __init__(self, asr_model_path: str, sample_rate: int = 16000):
        """
        初始化转换器
        
        Args:
            asr_model_path: ASR模型路径
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        self.asr_model = None
        self.asr_model_path = asr_model_path
        
        # 检查GPU可用性并设置设备
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU for processing")
        
    def _load_asr_model(self):
        """延迟加载ASR模型以节省内存"""
        if self.asr_model is None:
            print(f"Loading ASR model from {self.asr_model_path}...")
            self.asr_model = ASRModel.restore_from(self.asr_model_path)
            
            # 移动模型到指定设备
            self.asr_model = self.asr_model.to(self.device)
            self.asr_model.eval()
            print(f"ASR model loaded successfully on {self.device}!")
    
    def _get_encoder_time_dimension_from_audio(self, audio_path: str) -> Tuple[int, float]:
        """
        通过实际推理获取ASR编码器输出的时间维度
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            Tuple[编码器时间维度, 音频时长]
        """
        self._load_asr_model()
        
        # 加载真实音频文件
        print(f"Loading audio file: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=self.sample_rate)
        audio_duration = len(audio_signal) / sr
        
        # 转换为torch张量并添加batch维度，移动到指定设备
        audio_tensor = torch.FloatTensor(audio_signal).unsqueeze(0).to(self.device)  # (1, T)
        audio_length = torch.tensor([len(audio_signal)]).to(self.device)
        
        print(f"Audio duration: {audio_duration:.2f}s, samples: {len(audio_signal)}")
        print(f"Audio tensor device: {audio_tensor.device}, Model device: {next(self.asr_model.parameters()).device}")
        
        # 通过ASR模型进行实际推理
        with torch.no_grad():
            # 预处理：将原始音频转换为特征
            processed_signal, processed_length = self.asr_model.preprocessor(
                input_signal=audio_tensor, length=audio_length
            )
            print(f"After preprocessing: {processed_signal.shape}, length: {processed_length}")
            
            # 编码器：获取编码后的表示
            encoded, encoded_len = self.asr_model.encoder(
                audio_signal=processed_signal, length=processed_length
            )
            print(f"Encoder output shape: {encoded.shape}, length: {encoded_len}")
            
            # 返回编码器输出的实际时间维度
            # 编码器输出形状: (batch_size, hidden_dim, time_steps)
            # 对于Conformer模型，通常是 (B, D, T) 格式
            T_enc = encoded.shape[2]  # (B, D, T) -> T，时间维度是最后一维
            print(f"Encoder time dimension (T_enc): {T_enc}")
            print(f"Hidden dimension: {encoded.shape[1]}")
            
            return T_enc, audio_duration
    
    def parse_rttm(self, rttm_path: str) -> List[Tuple[float, float, str]]:
        """
        解析RTTM文件
        
        Args:
            rttm_path: RTTM文件路径
            
        Returns:
            [(start_time, end_time, speaker_id), ...]
        """
        segments = []
        with open(rttm_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 8 and parts[0] == 'SPEAKER':
                    start_time = float(parts[3])
                    duration = float(parts[4])
                    end_time = start_time + duration
                    speaker_id = parts[7]
                    segments.append((start_time, end_time, speaker_id))
        
        return segments
    
    def create_speaker_matrix(self, rttm_path: str, audio_path: str, 
                            speaker_ids: List[str]) -> Tuple[np.ndarray, int, float]:
        """
        创建说话人标签矩阵，通过实际推理获取ASR编码器时间维度并精确对齐
        
        Args:
            rttm_path: RTTM文件路径
            audio_path: 音频文件路径
            speaker_ids: 说话人ID列表（实际使用的说话人，但输出矩阵固定为4个说话人）
            
        Returns:
            Tuple[说话人标签矩阵, 编码器时间维度, 音频时长]
            说话人标签矩阵形状为 (4, T_enc) - 固定4个说话人，转置后的矩阵
        """
        print(f"\n=== Processing: {Path(audio_path).name} ===")
        
        # 通过实际推理获取ASR编码器的时间维度
        T_enc, audio_duration = self._get_encoder_time_dimension_from_audio(audio_path)
        
        # 解析RTTM文件
        print(f"Parsing RTTM file: {rttm_path}")
        segments = self.parse_rttm(rttm_path)
        print(f"Found {len(segments)} speaker segments")
        
        # 创建说话人ID到索引的映射（限制最多4个说话人）
        # 如果实际说话人超过4个，只使用前4个；如果少于4个，剩余位置保持为0
        MAX_SPEAKERS = 4
        speaker_to_idx = {}
        
        # 首先映射传入的speaker_ids（最多4个）
        for idx, spk in enumerate(speaker_ids[:MAX_SPEAKERS]):
            speaker_to_idx[spk] = idx
        
        # 如果RTTM中有新的说话人且还有空位，则添加到映射中
        unique_speakers_in_rttm = set(seg[2] for seg in segments)
        for speaker_id in unique_speakers_in_rttm:
            if speaker_id not in speaker_to_idx and len(speaker_to_idx) < MAX_SPEAKERS:
                speaker_to_idx[speaker_id] = len(speaker_to_idx)
        
        print(f"Speaker mapping (max {MAX_SPEAKERS}): {speaker_to_idx}")
        
        # 初始化标签矩阵：固定为4个说话人 × T_enc时间步
        # 注意：这里先创建 (T_enc, 4) 的矩阵，最后会转置为 (4, T_enc)
        speaker_matrix = np.zeros((T_enc, MAX_SPEAKERS), dtype=np.float32)
        
        # 计算时间步长（每个编码器时间步对应的音频时长）
        time_step = audio_duration / T_enc
        print(f"Time step size: {time_step:.4f}s per encoder frame")
        
        # 填充标签矩阵
        segments_processed = 0
        for start_time, end_time, speaker_id in segments:
            if speaker_id in speaker_to_idx:
                # 将时间转换为编码器时间步索引
                start_idx = int(start_time / time_step)
                end_idx = int(end_time / time_step)
                
                # 确保索引在有效范围内
                start_idx = max(0, min(start_idx, T_enc - 1))
                end_idx = max(0, min(end_idx, T_enc))
                
                # 设置标签
                speaker_idx = speaker_to_idx[speaker_id]
                speaker_matrix[start_idx:end_idx, speaker_idx] = 1.0
                
                print(f"  {speaker_id}: {start_time:.2f}s-{end_time:.2f}s -> frames {start_idx}-{end_idx} (speaker_idx: {speaker_idx})")
                segments_processed += 1
            else:
                print(f"  Warning: Speaker {speaker_id} not mapped (max {MAX_SPEAKERS} speakers), skipping")
        
        # 转置矩阵：从 (T_enc, 4) 转为 (4, T_enc)
        speaker_matrix_transposed = speaker_matrix.T
        
        print(f"Processed {segments_processed}/{len(segments)} segments")
        print(f"Original matrix shape: {speaker_matrix.shape}")
        print(f"Final speaker matrix shape (transposed): {speaker_matrix_transposed.shape}")
        print(f"Active speakers: {len([k for k, v in speaker_to_idx.items()])} out of {MAX_SPEAKERS} max speakers")
        
        return speaker_matrix_transposed, T_enc, audio_duration
    
    def convert_rttm_to_npy(self, rttm_path: str, audio_path: str, 
                          output_path: str, speaker_ids: List[str]) -> Dict[str, any]:
        """
        将RTTM文件转换为NPY文件，通过实际推理获取编码器时间维度
        
        Args:
            rttm_path: RTTM文件路径
            audio_path: 音频文件路径
            output_path: 输出NPY文件路径
            speaker_ids: 说话人ID列表
            
        Returns:
            包含转换信息的字典
        """
        # 创建说话人标签矩阵（通过实际推理获取时间维度）
        speaker_matrix, T_enc, audio_duration = self.create_speaker_matrix(
            rttm_path, audio_path, speaker_ids
        )
        
        # 保存为NPY文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, speaker_matrix)
        
        # 返回转换信息
        conversion_info = {
            'audio_path': audio_path,
            'rttm_path': rttm_path,
            'output_path': output_path,
            'audio_duration': audio_duration,
            'encoder_time_dim': T_enc,
            'speaker_matrix_shape': speaker_matrix.shape,
            'num_speakers': len(speaker_ids),
            'speaker_ids': speaker_ids
        }
        
        print(f"✓ Saved speaker matrix {speaker_matrix.shape} to {output_path}")
        print(f"  Audio: {audio_duration:.2f}s -> Encoder: {T_enc} frames")
        
        return conversion_info


class ManifestUpdater:
    """
    Manifest文件更新器，添加NPY文件路径
    """
    
    @staticmethod
    def update_manifest_with_npy_paths(input_manifest: str, output_manifest: str, 
                                     npy_dir: str):
        """
        更新manifest文件，添加npy_path字段
        
        Args:
            input_manifest: 输入manifest文件路径
            output_manifest: 输出manifest文件路径
            npy_dir: NPY文件目录
        """
        updated_count = 0
        total_count = 0
        
        with open(input_manifest, 'r') as infile, open(output_manifest, 'w') as outfile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                
                total_count += 1
                data = json.loads(line)
                
                # 从音频文件路径推断NPY文件路径
                audio_path = data.get('audio_filepath', '')
                if audio_path:
                    # 获取音频文件名（不含扩展名）
                    audio_name = Path(audio_path).stem
                    npy_path = os.path.join(npy_dir, f"{audio_name}.npy")
                    
                    # 检查NPY文件是否存在
                    if os.path.exists(npy_path):
                        data['npy_path'] = npy_path
                        updated_count += 1
                    else:
                        print(f"Warning: NPY file not found for {audio_path}")
                
                # 写入更新后的数据
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"Updated {updated_count}/{total_count} entries in manifest file")
        print(f"Output manifest saved to: {output_manifest}")


class DataLoaderModifier:
    """
    数据加载器修改器，提供加载音频和说话人标签的示例代码
    """
    
    @staticmethod
    def generate_dataloader_code(output_path: str):
        """
        生成修改后的数据加载器代码
        
        Args:
            output_path: 输出代码文件路径
        """
        code = '''
#!/usr/bin/env python3
"""
修改后的数据加载器，支持音频和说话人标签的联合加载

使用方法：
1. 将此代码集成到你的训练脚本中
2. 确保manifest文件包含npy_path字段
3. 在训练循环中使用speaker_labels
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any


class AudioSpeakerDataset(Dataset):
    """
    音频和说话人标签联合数据集
    """
    
    def __init__(self, manifest_path: str, audio_processor=None):
        """
        初始化数据集
        
        Args:
            manifest_path: manifest文件路径
            audio_processor: 音频预处理器
        """
        self.manifest_path = manifest_path
        self.audio_processor = audio_processor
        self.data = []
        
        # 加载manifest文件
        with open(manifest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取数据项
        
        Returns:
            包含音频、文本、说话人标签等信息的字典
        """
        item = self.data[idx]
        
        # 基本信息
        result = {
            'audio_filepath': item['audio_filepath'],
            'text': item.get('text', ''),
            'duration': item.get('duration', 0.0)
        }
        
        # 加载说话人标签（如果存在）
        if 'npy_path' in item and item['npy_path']:
            try:
                speaker_labels = np.load(item['npy_path'])
                result['speaker_labels'] = torch.from_numpy(speaker_labels)
                result['has_speaker_labels'] = True
            except Exception as e:
                print(f"Warning: Failed to load speaker labels from {item['npy_path']}: {e}")
                result['speaker_labels'] = None
                result['has_speaker_labels'] = False
        else:
            result['speaker_labels'] = None
            result['has_speaker_labels'] = False
        
        return result


def collate_fn_with_speaker_labels(batch):
    """
    自定义collate函数，处理变长的音频和说话人标签
    
    Args:
        batch: 批次数据
        
    Returns:
        整理后的批次数据
    """
    # 分离不同类型的数据
    audio_filepaths = [item['audio_filepath'] for item in batch]
    texts = [item['text'] for item in batch]
    durations = [item['duration'] for item in batch]
    speaker_labels_list = [item['speaker_labels'] for item in batch if item['has_speaker_labels']]
    has_speaker_labels = [item['has_speaker_labels'] for item in batch]
    
    # 处理说话人标签（如果存在）
    speaker_labels_batch = None
    if speaker_labels_list:
        # 找到最大时间维度
        max_time_dim = max(labels.shape[0] for labels in speaker_labels_list)
        num_speakers = speaker_labels_list[0].shape[1]
        
        # 创建填充后的批次张量
        batch_size = len(speaker_labels_list)
        speaker_labels_batch = torch.zeros(batch_size, max_time_dim, num_speakers)
        
        # 填充数据
        for i, labels in enumerate(speaker_labels_list):
            time_dim = labels.shape[0]
            speaker_labels_batch[i, :time_dim, :] = labels
    
    return {
        'audio_filepaths': audio_filepaths,
        'texts': texts,
        'durations': durations,
        'speaker_labels': speaker_labels_batch,
        'has_speaker_labels': has_speaker_labels
    }


# 使用示例
if __name__ == "__main__":
    # 创建数据集
    dataset = AudioSpeakerDataset('../data/M8013_multispeaker_manifest_train_joint_no_punc.json')
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn_with_speaker_labels,
        num_workers=2
    )
    
    # 训练循环示例
    for batch in dataloader:
        audio_filepaths = batch['audio_filepaths']
        texts = batch['texts']
        speaker_labels = batch['speaker_labels']  # 形状: (B, T_enc, num_speakers)
        
        # 在这里进行训练
        # 如果speaker_labels不为None，则可以用于说话人注入
        if speaker_labels is not None:
            print(f"Batch with speaker labels: {speaker_labels.shape}")
        else:
            print("Batch without speaker labels")
'''
        
        with open(output_path, 'w') as f:
            f.write(code)
        
        print(f"Generated dataloader code saved to: {output_path}")


class BatchProcessor:
    """
    批量处理器，从manifest文件中读取音频和RTTM路径，通过实际推理处理整个数据集
    
    核心改进：
    1. 支持从manifest文件中直接读取RTTM文件路径
    2. 通过实际推理获取每个音频的编码器时间维度
    3. 提供详细的处理统计信息
    """
    
    def __init__(self, asr_model_path: str):
        """
        初始化批量处理器
        
        Args:
            asr_model_path: ASR模型路径
        """
        self.converter = RTTMToNPYConverter(asr_model_path)
        self.processing_stats = {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'error_files': 0,
            'conversion_info': []
        }
    
    def process_dataset_from_manifest(self, manifest_path: str, output_npy_dir: str, 
                                    output_manifest_path: str, speaker_ids: List[str],
                                    rttm_field: str = 'rttm_filepath'):
        """
        从manifest文件中读取音频和RTTM路径，批量处理整个数据集
        
        Args:
            manifest_path: 输入manifest文件路径（包含audio_filepath和rttm_filepath字段）
            output_npy_dir: 输出NPY文件目录
            output_manifest_path: 输出manifest文件路径
            speaker_ids: 说话人ID列表
            rttm_field: manifest中RTTM文件路径的字段名（默认：'rttm_filepath'）
        """
        print("\n" + "="*60)
        print("开始批量处理数据集（从manifest文件读取路径）")
        print("="*60)
        
        # 创建输出目录
        os.makedirs(output_npy_dir, exist_ok=True)
        
        # 读取manifest文件
        print(f"Reading manifest file: {manifest_path}")
        with open(manifest_path, 'r') as f:
            manifest_data = [json.loads(line.strip()) for line in f if line.strip()]
        
        self.processing_stats['total_files'] = len(manifest_data)
        print(f"Found {len(manifest_data)} entries in manifest")
        
        # 检查manifest格式
        if manifest_data and rttm_field not in manifest_data[0]:
            print(f"Warning: '{rttm_field}' field not found in manifest. Available fields: {list(manifest_data[0].keys())}")
            print("Will try to infer RTTM paths from audio paths...")
            use_inferred_rttm = True
        else:
            use_inferred_rttm = False
        
        # 处理每个音频文件
        for i, item in enumerate(tqdm(manifest_data, desc="Processing audio files")):
            audio_path = item['audio_filepath']
            audio_name = Path(audio_path).stem
            
            print(f"\n[{i+1}/{len(manifest_data)}] Processing: {audio_name}")
            
            # 获取RTTM文件路径
            if use_inferred_rttm:
                # 从音频路径推断RTTM路径（假设在同一目录或相对目录）
                audio_dir = Path(audio_path).parent
                rttm_path = audio_dir / f"{audio_name}.rttm"
                if not rttm_path.exists():
                    # 尝试在rttm子目录中查找
                    rttm_path = audio_dir / "rttm" / f"{audio_name}.rttm"
                rttm_path = str(rttm_path)
            else:
                rttm_path = item[rttm_field]
            
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                print(f"  ❌ Audio file not found: {audio_path}")
                self.processing_stats['error_files'] += 1
                continue
                
            if not os.path.exists(rttm_path):
                print(f"  ❌ RTTM file not found: {rttm_path}")
                self.processing_stats['error_files'] += 1
                continue
            
            # 构建输出NPY文件路径
            npy_path = os.path.join(output_npy_dir, f"{audio_name}.npy")
            
            # 跳过已存在的文件
            if os.path.exists(npy_path):
                print(f"  ⏭️  Skipping existing file: {npy_path}")
                self.processing_stats['skipped_files'] += 1
                continue
            
            try:
                # 转换RTTM到NPY（通过实际推理）
                conversion_info = self.converter.convert_rttm_to_npy(
                    rttm_path, audio_path, npy_path, speaker_ids
                )
                
                # 记录转换信息
                self.processing_stats['conversion_info'].append(conversion_info)
                self.processing_stats['processed_files'] += 1
                
            except Exception as e:
                print(f"  ❌ Error processing {audio_name}: {e}")
                self.processing_stats['error_files'] += 1
        
        # 打印处理统计
        self._print_processing_stats()
        
        # 更新manifest文件
        print("\nUpdating manifest file...")
        ManifestUpdater.update_manifest_with_npy_paths(
            manifest_path, output_manifest_path, output_npy_dir
        )
        
        print("\n" + "="*60)
        print("批量处理完成！")
        print("="*60)
        
        return self.processing_stats
    
    def _print_processing_stats(self):
        """打印处理统计信息"""
        stats = self.processing_stats
        print("\n" + "="*50)
        print("处理统计信息")
        print("="*50)
        print(f"总文件数: {stats['total_files']}")
        print(f"成功处理: {stats['processed_files']}")
        print(f"跳过文件: {stats['skipped_files']}")
        print(f"错误文件: {stats['error_files']}")
        
        if stats['conversion_info']:
            # 统计编码器时间维度分布
            time_dims = [info['encoder_time_dim'] for info in stats['conversion_info']]
            durations = [info['audio_duration'] for info in stats['conversion_info']]
            
            print(f"\n编码器时间维度统计:")
            print(f"  最小值: {min(time_dims)}")
            print(f"  最大值: {max(time_dims)}")
            print(f"  平均值: {np.mean(time_dims):.1f}")
            
            print(f"\n音频时长统计:")
            print(f"  最短: {min(durations):.2f}s")
            print(f"  最长: {max(durations):.2f}s")
            print(f"  平均: {np.mean(durations):.2f}s")
    
    # 保持向后兼容的旧方法
    def process_dataset(self, manifest_path: str, rttm_dir: str, 
                       output_npy_dir: str, output_manifest_path: str,
                       speaker_ids: List[str]):
        """
        批量处理整个数据集（向后兼容方法）
        
        Args:
            manifest_path: 输入manifest文件路径
            rttm_dir: RTTM文件目录
            output_npy_dir: 输出NPY文件目录
            output_manifest_path: 输出manifest文件路径
            speaker_ids: 说话人ID列表
        """
        print("Warning: 使用向后兼容的process_dataset方法")
        print("建议使用新的process_dataset_from_manifest方法")
        
        # 创建临时manifest，添加推断的RTTM路径
        temp_manifest_data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    audio_path = item['audio_filepath']
                    audio_name = Path(audio_path).stem
                    rttm_path = os.path.join(rttm_dir, f"{audio_name}.rttm")
                    item['rttm_filepath'] = rttm_path
                    temp_manifest_data.append(item)
        
        # 写入临时manifest文件
        temp_manifest_path = manifest_path + ".temp"
        with open(temp_manifest_path, 'w') as f:
            for item in temp_manifest_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        try:
            # 使用新方法处理
            return self.process_dataset_from_manifest(
                temp_manifest_path, output_npy_dir, output_manifest_path, speaker_ids
            )
        finally:
            # 清理临时文件
            if os.path.exists(temp_manifest_path):
                os.remove(temp_manifest_path)


def main():
    """
    主函数，提供命令行接口
    """
    parser = argparse.ArgumentParser(
        description="自动对齐说话人标签与ASR编码器时间维度（通过实际推理获取时间维度）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 单文件转换（通过实际推理）
  python auto_align_speaker_labels.py --mode convert \
    --asr_model_path /path/to/asr_model.nemo \
    --rttm_path /path/to/file.rttm \
    --audio_path /path/to/audio.wav \
    --output_npy_path /path/to/output.npy \
    --speaker_ids speaker_0 speaker_1
  
  # 从manifest批量处理（推荐）
  python auto_align_speaker_labels.py --mode batch_process_manifest \
    --asr_model_path /path/to/asr_model.nemo \
    --manifest_path /path/to/manifest_with_rttm.json \
    --output_npy_dir /path/to/output_npy \
    --output_manifest_path /path/to/output_manifest.json \
    --speaker_ids speaker_0 speaker_1 \
    --rttm_field rttm_filepath
  
  # 传统批量处理（向后兼容）
  python auto_align_speaker_labels.py --mode batch_process \
    --asr_model_path /path/to/asr_model.nemo \
    --manifest_path /path/to/manifest.json \
    --rttm_dir /path/to/rttm_files \
    --output_npy_dir /path/to/output_npy \
    --output_manifest_path /path/to/output_manifest.json \
    --speaker_ids speaker_0 speaker_1
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['convert', 'update_manifest', 'generate_dataloader', 'batch_process', 'batch_process_manifest'],
        required=True,
        help='运行模式'
    )
    
    parser.add_argument(
        '--asr_model_path',
        required=True,
        help='ASR模型路径'
    )
    
    parser.add_argument(
        '--manifest_path',
        help='Manifest文件路径'
    )
    
    parser.add_argument(
        '--rttm_dir',
        help='RTTM文件目录'
    )
    
    parser.add_argument(
        '--output_npy_dir',
        help='输出NPY文件目录'
    )
    
    parser.add_argument(
        '--output_manifest_path',
        help='输出manifest文件路径'
    )
    
    parser.add_argument(
        '--speaker_ids',
        nargs='+',
        default=['spk1', 'spk2', 'spk3'],
        help='说话人ID列表'
    )
    
    parser.add_argument(
        '--rttm_path',
        help='单个RTTM文件路径（convert模式）'
    )
    
    parser.add_argument(
        '--audio_path',
        help='单个音频文件路径（convert模式）'
    )
    
    parser.add_argument(
        '--output_npy_path',
        help='单个输出NPY文件路径（convert模式）'
    )
    
    parser.add_argument(
        '--dataloader_output_path',
        default='modified_dataloader.py',
        help='数据加载器代码输出路径'
    )
    
    parser.add_argument(
        '--rttm_field',
        default='rttm_filepath',
        help='manifest中RTTM文件路径的字段名（默认：rttm_filepath）'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'convert':
        # 单文件转换模式
        if not all([args.rttm_path, args.audio_path, args.output_npy_path]):
            print("Error: convert mode requires --rttm_path, --audio_path, and --output_npy_path")
            return
        
        print("\n" + "="*60)
        print("单文件转换模式（通过实际推理获取编码器时间维度）")
        print("="*60)
        
        converter = RTTMToNPYConverter(args.asr_model_path)
        conversion_info = converter.convert_rttm_to_npy(
            args.rttm_path, args.audio_path, args.output_npy_path, args.speaker_ids
        )
        
        print("\n" + "="*50)
        print("转换完成！")
        print("="*50)
        print(f"输出文件: {args.output_npy_path}")
        print(f"编码器时间维度: {conversion_info['encoder_time_dim']}")
        print(f"音频时长: {conversion_info['audio_duration']:.2f}s")
        print(f"说话人矩阵形状: {conversion_info['speaker_matrix_shape']}")
    
    elif args.mode == 'batch_process_manifest':
        # 从manifest批量处理模式（推荐）
        if not all([args.manifest_path, args.output_npy_dir, args.output_manifest_path]):
            print("Error: batch_process_manifest mode requires --manifest_path, --output_npy_dir, and --output_manifest_path")
            return
        
        processor = BatchProcessor(args.asr_model_path)
        stats = processor.process_dataset_from_manifest(
            args.manifest_path, args.output_npy_dir, args.output_manifest_path,
            args.speaker_ids, args.rttm_field
        )
    
    elif args.mode == 'update_manifest':
        # 更新manifest模式
        if not all([args.manifest_path, args.output_manifest_path, args.output_npy_dir]):
            print("Error: update_manifest mode requires --manifest_path, --output_manifest_path, and --output_npy_dir")
            return
        
        ManifestUpdater.update_manifest_with_npy_paths(
            args.manifest_path, args.output_manifest_path, args.output_npy_dir
        )
    
    elif args.mode == 'generate_dataloader':
        # 生成数据加载器代码模式
        DataLoaderModifier.generate_dataloader_code(args.dataloader_output_path)
    
    elif args.mode == 'batch_process':
        # 批量处理模式（向后兼容）
        if not all([args.manifest_path, args.rttm_dir, args.output_npy_dir, args.output_manifest_path]):
            print("Error: batch_process mode requires --manifest_path, --rttm_dir, --output_npy_dir, and --output_manifest_path")
            return
        
        processor = BatchProcessor(args.asr_model_path)
        stats = processor.process_dataset(
            args.manifest_path, args.rttm_dir, args.output_npy_dir, 
            args.output_manifest_path, args.speaker_ids
        )


if __name__ == "__main__":
    main()