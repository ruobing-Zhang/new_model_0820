# Adapter微调训练系统

这是一个基于字符级RNNT模型的Adapter微调训练系统，支持说话人信息注入和多说话人语音识别。

## 功能特性

- ✅ **说话人信息注入**: 在Conformer编码器输出后注入说话人正弦核信息
- ✅ **Adapter层**: 支持参数高效的微调，减少训练参数量
- ✅ **词表扩展**: 自动添加说话人标记 `<|spk0|>`, `<|spk1|>`, `<|spk2|>`, `<|spk3|>`
- ✅ **权重冻结策略**: 只训练必要的模块，保持预训练权重
- ✅ **PyTorch Lightning**: 完整的训练、验证和日志记录系统

## 文件结构

```
scripts_0822/
├── rnnt_char_model_with_spk_inject.py    # 核心模型定义
├── adapter_finetune_train.py             # 主训练脚本
├── adapter_finetune_config.yaml          # 配置文件模板
├── example_train_manifest.jsonl          # 训练数据格式示例
├── run_adapter_finetune.sh               # 快速启动脚本
└── README_adapter_finetune.md             # 本文档
```

## 快速开始

### 1. 准备数据

创建训练数据manifest文件（JSONL格式），每行包含：

```json
{
  "audio_filepath": "/path/to/audio.wav",
  "text": "转录文本",
  "pgt_npy": "/path/to/speaker_probs.npy",
  "utt_id": "utterance_id",
  "duration": 3.5
}
```

**说话人概率文件格式**:
- `pgt_npy`: NumPy数组文件，形状为 `[T, K]` 或 `[K, T]`
- `T`: 时间帧数
- `K`: 说话人数量（默认4）
- 数值范围: [0, 1]，每个时间帧的概率和应为1

**文本格式**:
- 普通文本: `"你好世界"`
- 带说话人标记: `"<|spk0|>你好<|spk1|>世界"`

### 2. 修改配置

编辑 `run_adapter_finetune.sh` 中的路径：

```bash
TRAIN_MANIFEST="/path/to/your/train_manifest.jsonl"
VAL_MANIFEST="/path/to/your/val_manifest.jsonl"
```

### 3. 启动训练

```bash
# 方法1: 使用启动脚本
bash run_adapter_finetune.sh

# 方法2: 直接运行Python脚本
python adapter_finetune_train.py \
    --train_manifest /path/to/train_manifest.jsonl \
    --val_manifest /path/to/val_manifest.jsonl \
    --output_dir ./output \
    --max_epochs 50
```

## 详细配置

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `K_max` | 4 | 最大说话人数量 |
| `alpha_init` | 0.1 | 说话人注入强度初始值 |
| `num_adapters` | 4 | Adapter层数量 |
| `adapter_bottleneck` | 256 | Adapter瓶颈层维度 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 1e-4 | 学习率 |
| `weight_decay` | 1e-6 | 权重衰减 |
| `batch_size` | 8 | 批次大小 |
| `max_epochs` | 50 | 最大训练轮数 |
| `precision` | 16 | 训练精度（混合精度） |

### 权重冻结策略

- **冻结模块**: 
  - Preprocessor（预处理器）
  - Encoder（编码器）
  - Loss function（损失函数）

- **可训练模块**:
  - Adapter层
  - 说话人注入参数（alpha）
  - Decoder（解码器）
  - Joint网络

## 训练过程

### 1. 模型初始化

1. 加载预训练的字符级RNNT模型
2. 扩展词表，添加说话人标记
3. 创建带Adapter和说话人注入的新模型
4. 复制encoder及之前模块的权重
5. 重新初始化decoder和joint网络（随机权重）

### 2. 权重管理

```python
# 参数统计示例
总参数数: 130,140,017
可训练参数数: 26,026 (0.02%)
冻结参数数: 130,113,991 (99.98%)
```

### 3. 训练监控

- **TensorBoard日志**: 在 `output_dir/experiment_name/` 中
- **模型检查点**: 在 `output_dir/checkpoints/` 中
- **最终模型**: `output_dir/final_model.pt`

## 输出文件

训练完成后，输出目录包含：

```
output_dir/
├── checkpoints/
│   ├── epoch=10-val_loss=2.3456.ckpt
│   ├── epoch=20-val_loss=2.1234.ckpt
│   └── last.ckpt
├── experiment_name/
│   └── version_0/
│       ├── events.out.tfevents.*
│       └── hparams.yaml
└── final_model.pt
```

## 模型加载和推理

```python
# 加载训练好的模型
checkpoint = torch.load('output_dir/final_model.pt')
model_state_dict = checkpoint['model_state_dict']
config = checkpoint['config']
speaker_tokens = checkpoint['speaker_tokens']

# 重建模型
model = RNNTCharWithSpkInjectAndAdapter(
    cfg=config,
    K_max=4,
    alpha_init=0.1,
    num_adapters=4,
    adapter_bottleneck=256
)
model.load_state_dict(model_state_dict)
model.eval()

# 推理
with torch.no_grad():
    # 设置说话人标签
    model.set_speaker_labels(spk_labels)  # [B, K, T]
    
    # 编码
    encoded, encoded_len = model.encode_with_injection(
        input_signal, input_signal_length, spk_labels
    )
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小 `batch_size`
   - 使用 `precision=16` 开启混合精度
   - 减少 `num_adapters` 或 `adapter_bottleneck`

2. **数据加载错误**
   - 检查manifest文件格式
   - 确保音频文件和PGT文件路径正确
   - 验证PGT文件的NumPy数组形状

3. **训练不收敛**
   - 调整学习率
   - 检查说话人标签质量
   - 增加训练数据量

### 调试模式

```bash
# 启用详细日志
export NEMO_LOG_LEVEL=DEBUG
python adapter_finetune_train.py --train_manifest ... --batch_size 1
```

## 性能优化

### 训练加速

1. **混合精度训练**: `--precision 16`
2. **梯度累积**: `--accumulate_grad_batches 2`
3. **多GPU训练**: `--gpus 2`
4. **数据并行**: `--num_workers 8`

### 内存优化

1. **减小批次大小**: `--batch_size 4`
2. **梯度检查点**: 在模型中启用
3. **优化数据加载**: 使用 `pin_memory=True`

## 扩展功能

### 自定义Adapter

```python
class CustomAdapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim):
        super().__init__()
        # 自定义Adapter结构
        pass
    
    def forward(self, x):
        # 自定义前向传播
        pass
```

### 自定义说话人注入

```python
def custom_speaker_injection(enc_out, spk_labels):
    # 自定义说话人信息注入逻辑
    pass
```

## 引用

如果您使用了这个系统，请引用相关论文：

```bibtex
@article{your_paper,
  title={Multi-Speaker ASR with Speaker Injection and Adapter Fine-tuning},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 许可证

本项目基于Apache 2.0许可证开源。

## 联系方式

如有问题或建议，请联系：[your-email@example.com]