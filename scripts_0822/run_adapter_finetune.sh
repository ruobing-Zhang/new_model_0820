#!/bin/bash

# Adapter微调训练启动脚本
# 使用方法: bash run_adapter_finetune.sh

set -e  # 遇到错误时退出

# 配置参数
PRETRAINED_MODEL="/root/autodl-tmp/joint_sortformer_and_asr_0815/pretrained_models/stt_zh_conformer_transducer_large.nemo"
TRAIN_MANIFEST="/root/autodl-tmp/joint_sortformer_and_asr_0815/data/M8013_multispeaker_manifest_train_joint_no_punc_with_4speakers.json"
VAL_MANIFEST="/root/autodl-tmp/joint_sortformer_and_asr_0815/data/M8013_multispeaker_manifest_train_joint_no_punc_with_4speakers.json"
OUTPUT_DIR="./adapter_finetune_output"
EXPERIMENT_NAME="adapter_finetune_$(date +%Y%m%d_%H%M%S)"

# 模型参数
K_MAX=4
ALPHA_INIT=0.1
NUM_ADAPTERS=12
ADAPTER_BOTTLENECK=256

# 训练参数
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-6
BATCH_SIZE=8
MAX_EPOCHS=10
NUM_WORKERS=4

# 硬件参数
GPUS=1
PRECISION=16

echo "=== Adapter微调训练启动 ==="
echo "预训练模型: $PRETRAINED_MODEL"
echo "训练数据: $TRAIN_MANIFEST"
echo "验证数据: $VAL_MANIFEST"
echo "输出目录: $OUTPUT_DIR"
echo "实验名称: $EXPERIMENT_NAME"
echo "================================"

# 检查预训练模型是否存在
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "错误: 预训练模型不存在: $PRETRAINED_MODEL"
    echo "请检查路径是否正确"
    exit 1
fi

# 检查训练数据是否存在
if [ ! -f "$TRAIN_MANIFEST" ]; then
    echo "警告: 训练数据不存在: $TRAIN_MANIFEST"
    echo "请修改脚本中的TRAIN_MANIFEST路径"
    echo "或者使用示例数据: example_train_manifest.jsonl"
    # exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 启动训练
python adapter_finetune_train.py \
    --pretrained_model_path "$PRETRAINED_MODEL" \
    --train_manifest "$TRAIN_MANIFEST" \
    --val_manifest "$VAL_MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$EXPERIMENT_NAME" \
    --K_max $K_MAX \
    --alpha_init $ALPHA_INIT \
    --num_adapters $NUM_ADAPTERS \
    --adapter_bottleneck $ADAPTER_BOTTLENECK \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --batch_size 2 \
    --max_epochs $MAX_EPOCHS \
    --num_workers $NUM_WORKERS \
    --gpus $GPUS \
    --precision $PRECISION

echo "=== 训练完成 ==="
echo "输出目录: $OUTPUT_DIR"
echo "检查点: $OUTPUT_DIR/checkpoints/"
echo "日志: $OUTPUT_DIR/$EXPERIMENT_NAME/"
echo "最终模型: $OUTPUT_DIR/final_model.pt"