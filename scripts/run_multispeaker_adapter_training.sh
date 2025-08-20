#!/bin/bash

# 多说话人ASR适配器训练脚本
# 使用说话人信息注入的RNNT模型进行训练

set -e  # 遇到错误立即退出

# 配置参数
NEMO_MODEL="/root/autodl-tmp/joint_sortformer_and_asr_0815/pretrained_models/zh_conformer_transducer_large_bpe_init.nemo"
TRAIN_MANIFEST="/root/autodl-tmp/joint_sortformer_and_asr_0815/data/M8013_multispeaker_manifest_train_joint_no_punc_with_4speakers.json"
VAL_MANIFEST="/root/autodl-tmp/joint_sortformer_and_asr_0815/data/M8013_multispeaker_manifest_train_joint_no_punc_with_4speakers.json"

# 训练超参数
DEVICES=1
PRECISION=16
MAX_EPOCHS=10
BATCH_SIZE=4
NUM_WORKERS=2
LR=1e-4
WEIGHT_DECAY=1e-3
K_MAX=4
ALPHA_INIT=0.1

# 检查必要文件是否存在
if [ ! -f "$NEMO_MODEL" ]; then
    echo "错误: 预训练模型文件不存在: $NEMO_MODEL"
    exit 1
fi

if [ ! -f "$TRAIN_MANIFEST" ]; then
    echo "错误: 训练manifest文件不存在: $TRAIN_MANIFEST"
    exit 1
fi

echo "=== 多说话人ASR适配器训练 ==="
echo "预训练模型: $NEMO_MODEL"
echo "训练manifest: $TRAIN_MANIFEST"
echo "验证manifest: $VAL_MANIFEST"
echo "设备数量: $DEVICES"
echo "精度: $PRECISION"
echo "最大轮数: $MAX_EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LR"
echo "最大说话人数: $K_MAX"
echo "注入强度初值: $ALPHA_INIT"
echo "使用适配器: 是"
echo "================================"

# 启动训练
python train_multispeaker_asr_adapter.py \
    --nemo_model "$NEMO_MODEL" \
    --train_manifest "$TRAIN_MANIFEST" \
    --val_manifest "$VAL_MANIFEST" \
    --devices $DEVICES \
    --precision $PRECISION \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --K_max $K_MAX \
    --alpha_init $ALPHA_INIT \
    --use_adapter

echo "训练完成！"