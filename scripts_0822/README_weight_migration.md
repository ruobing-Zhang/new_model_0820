# ASR模型权重迁移指南

本指南介绍如何将预训练的 `EncDecRNNTModel` (字符级) 的权重迁移到 `EncDecRNNTBPEModel` (BPE子词级)。

## 文件说明

- `conformer_transducer_bpe.yaml`: 修改后的BPE模型配置文件
- `setup_tokenizer.py`: Tokenizer设置脚本
- `migrate_weights.py`: 权重迁移脚本
- `test_migrated_model.py`: 模型测试脚本

## 迁移步骤

### 步骤1: 准备Tokenizer

首先需要创建或准备BPE tokenizer：

#### 选项A: 创建新的tokenizer
```bash
cd /root/autodl-tmp/joint_sortformer_and_asr_0815/scripts_0822
python setup_tokenizer.py --vocab_size 1000
```

#### 选项B: 使用现有tokenizer
```bash
python setup_tokenizer.py --existing_tokenizer /path/to/your/tokenizer.model
```

### 步骤2: 执行权重迁移

运行权重迁移脚本：

```bash
python migrate_weights.py
```

该脚本会：
1. 加载预训练的字符级模型
2. 创建新的BPE模型
3. 迁移encoder和preprocessor权重
4. 保存新的.nemo模型文件

### 步骤3: 测试迁移后的模型

验证模型是否正常工作：

```bash
python test_migrated_model.py
```

## 配置文件修改说明

为了确保权重兼容性，我们对BPE配置文件进行了以下修改：

### Encoder配置匹配
- `n_layers`: 17 → 16
- `d_model`: 512 → 256  
- `n_heads`: 8 → 4
- `dropout`: 0.1 → 0.2
- `dropout_att`: 0.1 → 0.2

### Decoder配置匹配
- `pred_rnn_layers`: 1 → 2

### 移除的配置项
- `causal_downsampling`
- `reduction` 相关参数
- `att_context_size` 和 `att_context_style`
- `conv_norm_type` 和 `conv_context_size`
- `dropout_pre_encoder`
- `stochastic_depth` 相关参数

## 权重迁移详情

### 可迁移的组件
1. **Preprocessor**: 音频预处理模块，完全兼容
2. **Encoder**: Conformer编码器，大部分权重兼容
3. **Spec Augmentation**: 频谱增强模块，完全兼容

### 不可迁移的组件
1. **Decoder**: 由于词汇表不同，需要重新训练
2. **Joint Network**: 输出维度可能不同，需要重新训练

## 注意事项

1. **词汇表大小**: BPE模型的词汇表大小取决于tokenizer配置
2. **模型架构**: 确保BPE配置与预训练模型的encoder架构匹配
3. **训练数据**: 迁移后的模型仍需要在目标数据上进行微调
4. **性能验证**: 建议在验证集上测试迁移后模型的性能

## 自定义配置

如果需要修改默认路径，请编辑相应脚本中的路径配置：

### migrate_weights.py
```python
pretrained_model_path = "你的预训练模型路径.nemo"
bpe_config_path = "你的BPE配置文件路径.yaml"
tokenizer_dir = "你的tokenizer目录路径"
output_path = "输出模型路径.nemo"
```

### setup_tokenizer.py
```bash
python setup_tokenizer.py --output_dir /your/tokenizer/path --vocab_size 2000
```

### test_migrated_model.py
```bash
python test_migrated_model.py --model_path /your/model/path.nemo
```

## 故障排除

### 常见问题

1. **形状不匹配错误**
   - 检查配置文件中的模型参数是否与预训练模型匹配
   - 确认encoder的层数、维度、注意力头数等参数

2. **Tokenizer错误**
   - 确保tokenizer目录存在且包含tokenizer.model文件
   - 检查tokenizer类型配置（bpe vs wpe）

3. **内存不足**
   - 减少batch_size
   - 使用较小的模型配置

4. **CUDA错误**
   - 确保PyTorch和CUDA版本兼容
   - 检查GPU内存使用情况

### 调试技巧

1. 使用 `--skip_inference` 参数跳过推理测试，仅测试模型加载
2. 检查日志输出中的警告信息
3. 验证迁移前后的模型参数数量

## 后续步骤

迁移完成后，建议：

1. **微调训练**: 在目标数据集上进行微调
2. **性能评估**: 在测试集上评估WER/CER
3. **超参数调优**: 根据需要调整学习率、batch size等
4. **部署测试**: 在实际应用场景中测试模型性能

## 联系支持

如果遇到问题，请检查：
1. NeMo版本兼容性
2. 配置文件格式
3. 文件路径正确性
4. 系统环境配置