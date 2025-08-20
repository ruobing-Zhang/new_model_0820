#!/bin/bash

# RTTM to NPY alignment script runner
# 用于调用 rttm_to_npy_align.py 的便捷脚本

# 默认参数配置
NEMO_PATH="../pretrained_models/stt_zh_conformer_transducer_large.nemo"
MANIFEST="../data/M8013_multispeaker_manifest_train_joint_no_punc.json"
OUTPUT_DIR="./output_npy"
K_MAX=4
DEVICE="cuda"
SOFT_EDGE=false

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/rttm_to_npy_align.py"

# 帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --nemo_path PATH     .nemo 预训练/微调模型路径 (默认: ${NEMO_PATH})"
    echo "  --manifest PATH      JSON Lines manifest 路径 (必需)"
    echo "  --output_dir PATH    输出 .npy 的目录 (默认: ${OUTPUT_DIR})"
    echo "  --k_max NUM          支持的最大说话人数 (默认: ${K_MAX})"
    echo "  --device DEVICE      设备类型 cuda/cpu (默认: ${DEVICE})"
    echo "  --soft_edge          启用边界线性淡入淡出"
    echo "  -h, --help           显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --manifest /path/to/manifest.jsonl --output_dir /path/to/output"
    echo "  $0 --nemo_path /path/to/model.nemo --manifest /path/to/manifest.jsonl --k_max 6"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --nemo_path)
            NEMO_PATH="$2"
            shift 2
            ;;
        --manifest)
            MANIFEST="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --k_max)
            K_MAX="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --soft_edge)
            SOFT_EDGE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "$MANIFEST" ]]; then
    echo "错误: 必须指定 --manifest 参数"
    show_help
    exit 1
fi

# 检查文件是否存在
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "错误: Python脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

if [[ ! -f "$NEMO_PATH" ]]; then
    echo "错误: NeMo模型文件不存在: $NEMO_PATH"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "错误: Manifest文件不存在: $MANIFEST"
    exit 1
fi

# 构建Python命令
PYTHON_CMD="python \"$PYTHON_SCRIPT\" --nemo_path \"$NEMO_PATH\" --manifest \"$MANIFEST\" --output_dir \"$OUTPUT_DIR\" --k_max $K_MAX --device $DEVICE"

if [[ "$SOFT_EDGE" == "true" ]]; then
    PYTHON_CMD="$PYTHON_CMD --soft_edge"
fi

# 显示执行信息
echo "=========================================="
echo "RTTM to NPY Alignment Script"
echo "=========================================="
echo "NeMo模型路径: $NEMO_PATH"
echo "Manifest路径: $MANIFEST"
echo "输出目录: $OUTPUT_DIR"
echo "最大说话人数: $K_MAX"
echo "设备: $DEVICE"
echo "软边界: $SOFT_EDGE"
echo "=========================================="
echo "执行命令: $PYTHON_CMD"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 执行Python脚本
eval $PYTHON_CMD

# 检查执行结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "脚本执行成功完成！"
    echo "输出文件保存在: $OUTPUT_DIR"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "脚本执行失败！"
    echo "=========================================="
    exit 1
fi