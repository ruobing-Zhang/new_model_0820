#!/bin/bash
# 多说话人ASR适配器微调训练启动脚本

set -e

echo "=== 多说话人ASR适配器微调训练 ==="
echo "开始时间: $(date)"

# 检查必要文件
echo "检查必要文件..."
if [ ! -f "multispeaker_asr_adaptation.yaml" ]; then
    echo "错误: 配置文件 multispeaker_asr_adaptation.yaml 不存在"
    exit 1
fi

if [ ! -f "model_with_spk_inject.py" ]; then
    echo "错误: 模型文件 model_with_spk_inject.py 不存在"
    exit 1
fi

# 设置默认参数
NEMO_MODEL="../pretrained_models/zh_conformer_transducer_large_bpe_init.nemo"
TRAIN_MANIFEST="../data/M8013_multispeaker_manifest_train_joint_no_punc_with_4speakers.json"
VAL_MANIFEST="../data/M8013_multispeaker_manifest_train_joint_no_punc_with_4speakers.json"
MAX_STEPS=2000
DEVICES=1
BATCH_SIZE=8

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --nemo-model)
            NEMO_MODEL="$2"
            shift 2
            ;;
        --train-manifest)
            TRAIN_MANIFEST="$2"
            shift 2
            ;;
        --val-manifest)
            VAL_MANIFEST="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "使用方法: $0 [选项]"
            echo "选项:"
            echo "  --nemo-model PATH        预训练模型路径 (默认: $NEMO_MODEL)"
            echo "  --train-manifest PATH    训练manifest文件路径 (默认: $TRAIN_MANIFEST)"
            echo "  --val-manifest PATH      验证manifest文件路径 (默认: $VAL_MANIFEST)"
            echo "  --max-steps NUM          最大训练步数 (默认: $MAX_STEPS)"
            echo "  --devices NUM            GPU数量 (默认: $DEVICES)"
            echo "  --batch-size NUM         批次大小 (默认: $BATCH_SIZE)"
            echo "  -h, --help               显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "配置参数:"
echo "  预训练模型: $NEMO_MODEL"
echo "  训练数据: $TRAIN_MANIFEST"
echo "  验证数据: $VAL_MANIFEST"
echo "  最大步数: $MAX_STEPS"
echo "  GPU数量: $DEVICES"
echo "  批次大小: $BATCH_SIZE"
echo ""

# 检查数据文件
if [ ! -f "$TRAIN_MANIFEST" ]; then
    echo "警告: 训练manifest文件不存在: $TRAIN_MANIFEST"
fi

if [ ! -f "$VAL_MANIFEST" ]; then
    echo "警告: 验证manifest文件不存在: $VAL_MANIFEST"
fi

# 创建实验目录
EXP_DIR="./experiments/multispeaker_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXP_DIR"
echo "实验目录: $EXP_DIR"

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

echo "开始训练..."
echo ""

# 运行训练
python train_multispeaker_asr_adapter.py \
    --config-path="." \
    --config-name="multispeaker_asr_adaptation.yaml" \
    model.nemo_model="$NEMO_MODEL" \
    model.train_ds.manifest_filepath="$TRAIN_MANIFEST" \
    model.train_ds.batch_size="$BATCH_SIZE" \
    model.validation_ds.manifest_filepath="$VAL_MANIFEST" \
    model.validation_ds.batch_size="$BATCH_SIZE" \
    trainer.devices="$DEVICES" \
    trainer.max_steps="$MAX_STEPS" \
    trainer.precision=16 \
    exp_manager.exp_dir="$EXP_DIR" \
    exp_manager.name="multispeaker_asr_adapter"

echo ""
echo "训练完成!"
echo "结束时间: $(date)"
echo "实验结果保存在: $EXP_DIR"
echo "适配器文件: $EXP_DIR/checkpoints/multispeaker_adapters.pt"