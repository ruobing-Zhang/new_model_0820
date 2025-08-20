
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
    dataset = AudioSpeakerDataset('path/to/your/manifest.json')
    
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
