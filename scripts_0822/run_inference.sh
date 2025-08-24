#!/bin/bash

# 多说话人语音识别推理脚本

echo "=== 开始多说话人语音识别推理 ==="

# 配置参数
MODEL_PATH="./adapter_finetune_output/final_model.pt"
MANIFEST_PATH="/root/autodl-tmp/joint_sortformer_and_asr_0815/data/M8013_multispeaker_manifest_train_joint_no_punc_with_4speakers.json"
OUTPUT_PATH="./inference_results.jsonl"
DEVICE="cuda"
MAX_SAMPLES=10  # 先测试10个样本

echo "模型路径: $MODEL_PATH"
echo "数据路径: $MANIFEST_PATH"
echo "输出路径: $OUTPUT_PATH"
echo "设备: $DEVICE"
echo "最大样本数: $MAX_SAMPLES"
echo "================================"

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在 $MODEL_PATH"
    exit 1
fi

# 检查数据文件是否存在
if [ ! -f "$MANIFEST_PATH" ]; then
    echo "错误: 数据文件不存在 $MANIFEST_PATH"
    exit 1
fi

# 运行推理
python inference_multispeaker.py \
    --model_path "$MODEL_PATH" \
    --manifest_path "$MANIFEST_PATH" \
    --output_path "$OUTPUT_PATH" \
    --device "$DEVICE" \
    --max_samples $MAX_SAMPLES

echo "=== 推理完成 ==="
echo "结果文件: $OUTPUT_PATH"

# 显示结果文件的前几行
if [ -f "$OUTPUT_PATH" ]; then
    echo "\n=== 推理结果预览 ==="
    head -n 3 "$OUTPUT_PATH"
fi