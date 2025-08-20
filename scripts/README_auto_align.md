# 自动说话人标签对齐工具（实际推理版本）

## 🔥 新版本亮点

**v2.0 核心改进：**
- ✅ **实际推理获取时间维度**：不再依赖计算估算，通过实际ASR推理获取真实的编码器时间维度
- ✅ **支持manifest中的RTTM路径**：可直接从manifest文件中读取RTTM文件路径
- ✅ **智能路径推断**：自动推断RTTM文件位置，提高易用性
- ✅ **详细处理统计**：提供完整的处理统计信息和性能分析
- ✅ **完全向后兼容**：保留所有原有功能，无缝升级

## 概述

这个工具集解决了在训练带有说话人注入功能的ASR模型时，说话人标签与ASR编码器输出时间维度不匹配的问题。它提供了完整的解决方案，从RTTM文件处理到数据加载器修改，确保说话人标签能够正确对齐到ASR编码器的时间步。

## 问题背景

在测试中我们看到的 `torch.Size([2, 76, 512])` 各维度含义：

- **第一维 (2)**: 批次大小 (Batch Size)
- **第二维 (76)**: **时间维度 (Time Dimension, T_enc)** - 这是关键维度
- **第三维 (512)**: 特征维度 (Feature Dimension)

### 时间维度对齐的重要性

1. **ASR编码器下采样**: 原始音频经过编码器后，时间维度会被大幅压缩
2. **说话人标签匹配**: 说话人标签矩阵必须与编码器输出的时间步精确对齐
3. **动态计算**: 不同长度的音频会产生不同的T_enc，需要动态计算

## 工具组件

### 1. 核心Python脚本 (`auto_align_speaker_labels.py`)

主要功能模块：

#### RTTMToNPYConverter（支持实际推理）
- 解析RTTM文件
- **通过实际ASR推理获取编码器时间维度**（v2.0新功能）
- 生成对齐的说话人标签矩阵
- 保存为NPY格式
- `_get_encoder_time_dimension_from_audio()`: 实际推理获取时间维度
- `create_speaker_matrix()`: 创建与编码器输出对齐的说话人矩阵
- `convert_rttm_to_npy()`: 完整的转换流程，返回详细信息

#### ManifestUpdater
- 更新manifest文件
- 添加npy_path字段
- 验证NPY文件存在性
- **支持从manifest中读取RTTM路径**（v2.0新功能）

#### DataLoaderModifier
- 生成修改后的数据加载器代码
- 支持音频和说话人标签联合加载
- 处理变长序列的批处理

#### BatchProcessor（增强版）
- 批量处理整个数据集
- **支持manifest中的RTTM路径**（v2.0新功能）
- **智能路径推断**（v2.0新功能）
- **详细处理统计**（v2.0新功能）
- 进度显示和错误处理
- 自动跳过已存在的文件
- `process_dataset_from_manifest()`: 从manifest文件批量处理（推荐）
- `process_dataset()`: 传统批量处理（向后兼容）
- `_print_processing_stats()`: 详细的处理统计信息

### 2. 批处理脚本 (`run_auto_align.sh`)

提供友好的命令行接口：
- 彩色输出和进度提示
- 参数验证和错误检查
- 交互式确认
- 自动目录创建

### 3. 使用示例

#### `example_manifest_processing.py`（v2.0新增）
包含新版本功能的使用示例：
- **演示实际推理功能**
- **manifest中RTTM路径处理**
- **智能路径推断示例**
- **详细统计信息展示**

#### `example_auto_align_usage.py`（向后兼容）
包含完整的使用示例和说明：
- 创建示例数据
- 演示各种使用场景
- 详细的技术说明
- 张量维度解释

## 安装和配置

### 前置条件

```bash
# 确保Python环境
python --version  # 需要Python 3.7+

# 安装依赖
pip install numpy torch librosa tqdm

# 确保NeMo可用
export PYTHONPATH="/root/autodl-tmp/joint_sortformer_and_asr_0815/NeMo-main:$PYTHONPATH"
```

### 文件权限

```bash
chmod +x /root/autodl-tmp/joint_sortformer_and_asr_0815/scripts/run_auto_align.sh
```

## 使用方法

### 🔥 推荐方法：从manifest批量处理（v2.0新功能）

**准备包含RTTM路径的manifest文件：**
```json
{"audio_filepath": "/path/to/audio1.wav", "rttm_filepath": "/path/to/audio1.rttm", "duration": 10.5}
{"audio_filepath": "/path/to/audio2.wav", "rttm_filepath": "/path/to/audio2.rttm", "duration": 8.3}
```

**执行批量处理：**
```bash
python auto_align_speaker_labels.py batch_process_manifest \
  --asr_model_path /path/to/asr_model.nemo \
  --manifest_path /path/to/manifest_with_rttm.json \
  --output_npy_dir /path/to/output_npy \
  --output_manifest_path /path/to/output_manifest.json \
  --speaker_ids speaker_0 speaker_1 speaker_2 \
  --rttm_field rttm_filepath
```

**使用Shell脚本（更简便）：**
```bash
./run_auto_align.sh -m batch_process_manifest \
  -i data/train_with_rttm.json \
  -o data/aligned_npy/ \
  -u data/train_with_aligned_labels.json \
  -s "speaker_0 speaker_1 speaker_2"
```

### 方法1: 使用批处理脚本（推荐）

#### 1. 批量处理整个数据集

```bash
./run_auto_align.sh -m batch_process \
  -i /path/to/train_manifest.json \
  -r /path/to/rttm_files/ \
  -o /path/to/output_npy/ \
  -u /path/to/train_manifest_with_npy.json \
  -s "spk1 spk2 spk3"
```

#### 2. 单文件转换（通过实际推理）

```bash
./run_auto_align.sh -m convert \
  --rttm-path /path/to/audio1.rttm \
  --audio-path /path/to/audio1.wav \
  --output-npy-path /path/to/audio1.npy
```

#### 3. 更新manifest文件

```bash
./run_auto_align.sh -m update_manifest \
  -i /path/to/original_manifest.json \
  -o /path/to/npy_files/ \
  -u /path/to/updated_manifest.json
```

#### 4. 生成数据加载器代码

```bash
./run_auto_align.sh -m generate_dataloader \
  --dataloader-output modified_dataloader.py
```

### 方法2: Shell脚本使用方法

```bash
# 从manifest批量处理（推荐）
./run_auto_align.sh -m batch_process_manifest \
  -i data/train_with_rttm.json \
  -o data/npy/ \
  -u data/train_with_npy.json

# 传统批量处理（向后兼容）
./run_auto_align.sh -m batch_process \
  -i data/train.json \
  -r data/rttm/ \
  -o data/npy/ \
  -u data/train_with_npy.json

# 单文件转换
./run_auto_align.sh -m convert \
  --rttm-path data/audio1.rttm \
  --audio-path data/audio1.wav \
  --output-npy-path data/audio1.npy
```

### 方法3: 直接使用Python脚本

#### 1. 单文件转换（通过实际推理）

```bash
python auto_align_speaker_labels.py convert \
  --asr_model_path /path/to/asr_model.nemo \
  --rttm_path /path/to/file.rttm \
  --audio_path /path/to/audio.wav \
  --output_npy_path /path/to/output.npy \
  --speaker_ids speaker_0 speaker_1
```

#### 2. 传统批量处理（向后兼容）

```bash
python auto_align_speaker_labels.py batch_process \
  --asr_model_path /path/to/asr_model.nemo \
  --manifest_path /path/to/manifest.json \
  --rttm_dir /path/to/rttm_files \
  --output_npy_dir /path/to/output_npy \
  --output_manifest_path /path/to/output_manifest.json \
  --speaker_ids speaker_0 speaker_1
```

#### 3. 更新Manifest文件

```bash
python auto_align_speaker_labels.py update_manifest \
  --manifest_path /path/to/input_manifest.json \
  --output_manifest_path /path/to/output_manifest.json \
  --npy_dir /path/to/npy_files
```

#### 4. 生成数据加载器代码

```bash
python auto_align_speaker_labels.py generate_dataloader \
  --output_path /path/to/dataloader.py
```

## 数据格式说明

### 输入格式

#### 1. Manifest文件格式
```json
{"audio_filepath": "/path/to/audio1.wav", "text": "转录文本", "duration": 10.5}
{"audio_filepath": "/path/to/audio2.wav", "text": "转录文本", "duration": 8.2}
```

#### 2. RTTM文件格式
```
SPEAKER audio1 1 0.0 3.5 <NA> <NA> spk1 <NA> <NA>
SPEAKER audio1 1 3.5 4.0 <NA> <NA> spk2 <NA> <NA>
SPEAKER audio1 1 7.5 3.0 <NA> <NA> spk1 <NA> <NA>
```

### 输出格式

#### 1. 更新后的Manifest文件
```json
{"audio_filepath": "/path/to/audio1.wav", "text": "转录文本", "duration": 10.5, "npy_path": "/path/to/audio1.npy"}
```

#### 2. NPY文件内容
- 形状: `(T_enc, num_speakers)`
- 数据类型: `float32`
- 值: 0.0 或 1.0（表示该时间步是否有对应说话人）

## 技术细节

### 🔥 v2.0核心改进：实际推理获取时间维度

**传统方法（v1.0）**：通过公式估算时间维度
```python
# 创建虚拟音频信号
T_enc ≈ 音频时长(秒) × 采样率 / 编码器下采样因子
```

**新方法（v2.0）**：通过实际ASR推理获取精确时间维度
```python
# 加载真实音频并预处理
audio_signal, _ = librosa.load(audio_path, sr=16000)
processed_signal, processed_length = asr_model.preprocessor(
    input_signal=audio_signal, length=torch.tensor([len(audio_signal)])
)

# 通过编码器获取实际时间维度
encoded, encoded_len = asr_model.encoder(
    audio_signal=processed_signal, length=processed_length
)
T_enc = encoded.shape[1]  # 实际的编码器时间维度
```

### 时间维度对齐原理

工具通过以下步骤实现精确的时间维度对齐：

1. **加载真实音频**: 读取实际的音频文件
2. **ASR预处理**: 通过ASR模型的预处理器处理音频
3. **编码器推理**: 获取编码器输出的实际时间维度
4. **精确对齐**: 将RTTM时间戳精确映射到编码器时间步

### v2.0 vs v1.0 对比

| 特性 | v1.0（传统方法） | v2.0（实际推理） |
|------|------------------|------------------|
| 时间维度获取 | 公式估算 | 实际推理 |
| 精确度 | 近似值 | 精确值 |
| 音频处理 | 虚拟信号 | 真实音频 |
| 对齐质量 | 可能有偏差 | 完美对齐 |
| manifest支持 | 需要推断RTTM路径 | 直接读取RTTM路径 |
| 处理统计 | 基础信息 | 详细统计 |

### 标签矩阵生成

1. **精确时间步映射**: `time_step = audio_duration / T_enc`（基于实际T_enc）
2. **索引计算**: `start_idx = int(start_time / time_step)`
3. **标签设置**: `speaker_matrix[start_idx:end_idx, speaker_idx] = 1.0`
4. **完美对齐**: 确保与ASR编码器输出维度完全匹配

## 在训练中使用

### 1. 使用生成的数据加载器

```python
from modified_dataloader import AudioSpeakerDataset, collate_fn_with_speaker_labels
from torch.utils.data import DataLoader

# 创建数据集
dataset = AudioSpeakerDataset('path/to/manifest_with_npy.json')

# 创建数据加载器
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn_with_speaker_labels,
    num_workers=2
)
```

### 2. 在训练循环中使用

```python
for batch in dataloader:
    audio_filepaths = batch['audio_filepaths']
    texts = batch['texts']
    speaker_labels = batch['speaker_labels']  # 形状: (B, T_enc, num_speakers)
    
    if speaker_labels is not None:
        # 使用说话人注入模型
        output = model(
            input_signal=audio_signal,
            input_signal_length=audio_length,
            spk_labels=speaker_labels  # 传递给模型
        )
    else:
        # 普通前向传播
        output = model(
            input_signal=audio_signal,
            input_signal_length=audio_length
        )
```

## 故障排除

### 常见问题

1. **维度不匹配错误**
   - 确保NPY文件是通过此工具生成的
   - 检查说话人ID列表是否一致
   - 验证音频文件路径正确

2. **文件不存在错误**
   - 检查ASR模型路径
   - 确认RTTM文件存在
   - 验证音频文件可访问

3. **内存不足**
   - 减少批次大小
   - 使用更少的数据加载器工作进程
   - 分批处理大型数据集

### 调试技巧

1. **检查生成的NPY文件**
```python
import numpy as np
labels = np.load('path/to/audio.npy')
print(f"Shape: {labels.shape}")  # 应该是 (T_enc, num_speakers)
print(f"Non-zero entries: {np.sum(labels)}")
```

2. **验证时间维度**
```python
# 在模型中添加调试输出
print(f"Encoder output shape: {encoded.shape}")  # (B, T_enc, M)
print(f"Speaker labels shape: {speaker_labels.shape}")  # (B, T_enc, num_speakers)
```

## 性能优化

### 批量处理优化

1. **并行处理**: 可以修改脚本支持多进程处理
2. **缓存模型**: 避免重复加载ASR模型
3. **增量处理**: 跳过已存在的NPY文件

### 内存优化

1. **延迟加载**: 只在需要时加载ASR模型
2. **批次处理**: 合理设置批次大小
3. **清理缓存**: 及时释放不需要的张量

## 扩展功能

### 支持更多格式

可以扩展工具支持：
- 其他标注格式（如CTM、TextGrid）
- 不同的音频格式
- 多语言ASR模型

### 自定义说话人映射

```python
# 自定义说话人ID映射
speaker_mapping = {
    'speaker_001': 'spk1',
    'speaker_002': 'spk2',
    'unknown': 'spk3'
}
```

## 🎉 v2.0新功能总结

### 核心改进

1. **🔥 实际推理获取时间维度**
   - 不再依赖公式估算，通过真实音频推理获取精确的编码器时间维度
   - 确保说话人标签矩阵与ASR编码器输出完美对齐

2. **📁 manifest中RTTM路径支持**
   - 支持在manifest文件中直接指定RTTM文件路径
   - 新增`batch_process_manifest`模式，简化数据处理流程

3. **🧠 智能路径推断**
   - 当manifest中没有RTTM路径时，自动推断对应的RTTM文件
   - 支持灵活的文件组织结构

4. **📊 详细处理统计**
   - 提供完整的处理统计信息（总数、成功、跳过、错误）
   - 显示编码器时间维度、音频时长等技术细节

5. **🔄 向后兼容**
   - 保留所有v1.0功能，确保现有代码无需修改
   - 提供多种使用方式，满足不同需求

### 使用建议

- **新项目**: 推荐使用`batch_process_manifest`模式，享受最新功能
- **现有项目**: 可以继续使用原有方法，或逐步迁移到新方法
- **高精度需求**: 必须使用v2.0的实际推理功能，确保完美对齐

### 技术优势

- ✅ **精确对齐**: 基于实际推理，消除估算误差
- ✅ **处理效率**: 智能路径处理，减少手动配置
- ✅ **详细反馈**: 完整的处理统计和错误报告
- ✅ **灵活配置**: 支持多种数据组织方式
- ✅ **易于使用**: Shell脚本封装，简化命令行操作

## 许可证

此工具遵循与NeMo相同的许可证。

## 贡献

欢迎提交问题报告和功能请求。在提交之前，请确保：

1. 详细描述问题或功能需求
2. 提供复现步骤（如果是bug）
3. 包含相关的错误日志
4. 测试你的更改

## 更新日志

### v1.0.0 (2024)
- 初始版本发布
- 支持RTTM到NPY转换
- 自动时间维度对齐
- 批量处理功能
- 数据加载器生成
- 完整的使用示例